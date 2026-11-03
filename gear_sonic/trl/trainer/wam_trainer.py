"""SuperMimic-style training of the existing multimodal Sonic actor.

The actor keeps Sonic's original split: robot-only proprioceptive observations
drive the decoder and a separate G1/teleop/SMPL goal group is encoded by the
existing FSQ bottleneck.  WAM predicts only the robot-side state.  During
imagined rollouts the replayed goal timeline is advanced outside WAM, so one
shared dynamics model can train all goal interfaces without putting tokenizer
features into its state transition.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from numbers import Real
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch import nn

from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.trl.modules.wam import (
    PHYSICAL_STATE_DIM,
    NUM_G1_JOINTS,
    RunningNorm,
    StateSpaceWorldModel,
    WAMConfig,
    physical_state_parts,
)
from gear_sonic.trl.world_model.distributed import DistributedContext
from gear_sonic.trl.world_model.sequence_replay import SequenceReplayBuffer
from gear_sonic.trl.utils.common import custom_instantiate
from gear_sonic.trl.utils.checkpoint import validate_checkpoint_robot
from gear_sonic.utils.wam_critic import WAMCriticFeatures, future_reference_states
from gear_sonic.utils.wam_context import get_push_versions
from gear_sonic.utils.wam_kinematics import G1_MUJOCO_JOINT_NAMES
from gear_sonic.utils.wam_observations import WAMGoalObservations
from gear_sonic.utils.wam_optimization import policy_gradient_metrics
from gear_sonic.utils.wam_reference import WAMReferenceBank
from gear_sonic.utils.wam_noise import wam_observation_noise
from gear_sonic.utils.wam_reward import WAMTrackingReward, reference_state
from gear_sonic.utils.wam_termination import (
    enable_terminal_capture, get_terminal_state, soft_tracking_termination,
)


# Keep the trainer importable in CPU-only analysis environments that do not
# install IsaacLab. This is the same G1 MuJoCo/IsaacLab ordering used by the
# WAM observation terms.
_G1_MUJOCO_ORDER = G1_MUJOCO_JOINT_NAMES
_G1_ISAAC_ORDER = tuple(joint_utils.G1_ISAACLab_ORDER)
_MUJOCO_TO_ISAAC = tuple(_G1_MUJOCO_ORDER.index(name) for name in _G1_ISAAC_ORDER)
_ISAAC_TO_MUJOCO = tuple(_G1_ISAAC_ORDER.index(name) for name in _G1_MUJOCO_ORDER)
# v3 adds canonical future references and FK tracking features to the shared
# physical critic. Old actor weights remain compatible with policy-only load.
PHYSICAL_SCHEMA_VERSION = 4


@contextmanager
def frozen(module):
    flags = [parameter.requires_grad for parameter in module.parameters()]
    module.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags):
            parameter.requires_grad_(flag)


class ObservationLayout:
    """Pack robot observations and multimodal goals into separate flat tensors."""

    EXOGENOUS_FEATURES = {
        "command_multi_future_nonflat", "command_multi_future_lower_body",
        "command_z", "command_z_multi_future_nonflat", "vr_3point_local_target",
        "vr_3point_local_orn_target", "smpl_joints_multi_future_local_nonflat",
        "smpl_joints_multi_future_nonflat", "joint_pos_multi_future_wrist_for_smpl",
    }

    def __init__(
        self, example, state_keys=("actor_obs",), goal_keys=("tokenizer",),
        tokenizer_specs=(), exogenous_features=None,
    ):
        if isinstance(state_keys, str):
            state_keys = [state_keys]
        if isinstance(goal_keys, str):
            goal_keys = [goal_keys]
        # Accept the pre-refactor positional form
        # ObservationLayout(obs, keys, tokenizer_specs).  This is used by
        # downstream analysis scripts; WAMTrainer itself uses the explicit
        # state/goal form below.
        legacy_combined = bool(goal_keys) and all(
            isinstance(item, (tuple, list)) and len(item) >= 3 for item in goal_keys
        )
        if legacy_combined:
            tokenizer_specs = goal_keys
            goal_keys = ()
        self.legacy_combined = legacy_combined
        self.state_specs = []
        offset = 0
        for key in state_keys:
            if key not in example:
                raise ValueError(f"WAM observation group {key!r} is missing")
            value = example[key]
            if value.ndim != 2:
                raise ValueError(f"WAM expects flat observation groups, got {key}: {value.shape}")
            self.state_specs.append((key, offset, offset + value.shape[-1]))
            offset += value.shape[-1]
        self.state_dim = offset
        self.goal_specs = []
        goal_offset = 0
        for key in goal_keys:
            if key not in example:
                raise ValueError(f"WAM goal group {key!r} is missing")
            value = example[key]
            if value.ndim != 2:
                raise ValueError(f"WAM expects flat goal groups, got {key}: {value.shape}")
            self.goal_specs.append((key, goal_offset, goal_offset + value.shape[-1]))
            goal_offset += value.shape[-1]
        self.goal_dim = goal_offset
        self.tokenizer_specs = list(tokenizer_specs)
        self.fixed_indices = []
        self.exogenous_indices = []
        tokenizer_offset = next(
            (start for key, start, _ in self.state_specs if key == "tokenizer"), None
        )
        if tokenizer_offset is not None:
            exogenous = set(self.EXOGENOUS_FEATURES if exogenous_features is None else exogenous_features)
            for spec in self.tokenizer_specs:
                name, start, end = spec[:3]
                if name in {"encoder_index", "command_num_frames"}:
                    self.fixed_indices.extend(range(tokenizer_offset + start, tokenizer_offset + end))
                elif name in exogenous:
                    self.exogenous_indices.extend(range(tokenizer_offset + start, tokenizer_offset + end))
        # ``specs``/``dim`` remain aliases for callers that only inspect the
        # robot state layout.  Goals have their own schema and are never fed to
        # the WAM dynamics network.
        self.specs = self.state_specs
        self.dim = self.state_dim

    def pack(self, obs):
        return torch.cat([obs[key] for key, _, _ in self.state_specs], dim=-1)

    def pack_goal(self, obs):
        if not self.goal_specs:
            first = next(iter(obs.values()))
            return first.new_zeros(first.shape[0], 0)
        return torch.cat([obs[key] for key, _, _ in self.goal_specs], dim=-1)

    def unpack(self, state):
        return {key: state[..., start:end].unsqueeze(1) for key, start, end in self.state_specs}

    def unpack_goal(self, goal):
        return {key: goal[..., start:end].unsqueeze(1) for key, start, end in self.goal_specs}

    def project(self, state, seed, reference=None):
        """Compatibility projection for old combined observation layouts."""
        if not self.fixed_indices and not self.exogenous_indices:
            return state
        projected = state.clone()
        if self.fixed_indices:
            projected[..., self.fixed_indices] = seed[..., self.fixed_indices]
        if reference is not None and self.exogenous_indices:
            projected[..., self.exogenous_indices] = reference[..., self.exogenous_indices]
        return projected


class WAMTrainer:
    """Simulator collection -> k-step WAM fit -> FSQ BPTT -> bootstrapped critic.

    The constructor matches ``train_agent_trl.py``. Data parallel ranks collect
    independent simulator/replay data and average gradients after each of the
    three backward passes. Manual synchronization supports repeated FSQ
    forwards and alternating frozen world-model/critic phases without DDP's
    per-forward reducer bookkeeping. PPO remains a separate trainer.
    """

    def __init__(
        self, args, config, env, model, *, accelerator=None, log_dir=None,
        checkpoint=None, resume=False, world_model_config=None, settings=None,
        critic_config=None, reward_config=None, **kwargs,
    ):
        del kwargs
        if getattr(env, "action_transform_module", None) is not None:
            raise ValueError("WAMTrainer requires direct joint actions; hierarchical ATM is unsupported")
        if getattr(model, "max_rollout_history", 1) > 1:
            raise ValueError("WAMTrainer expects an FSQ MLP actor with max_rollout_history=1")
        if getattr(model, "running_mean_std", None) is not None:
            raise ValueError("WAMTrainer expects FSQ actors without an actor observation normalizer")
        self.args, self.config, self.env, self.policy = args, config, env, model
        self.accelerator = accelerator
        self.device = next(model.parameters()).device
        self.distributed = DistributedContext(self.device)
        if accelerator is not None and accelerator.num_processes != self.distributed.world_size:
            raise ValueError("Accelerator and initialized torch.distributed world sizes disagree")
        self.settings = dict(settings or {})
        reset_sampling = self.settings.get("reset_sampling", "adaptive_failure")
        if reset_sampling != "adaptive_failure":
            raise ValueError(
                "WAM reset_sampling must be 'adaptive_failure' (Sonic's existing sampler); "
                "the SuperMimic return-based reset sampler is not implemented. "
                "Old configs with reset_sampling=return must explicitly select adaptive_failure."
            )
        if int(self.settings.get("warmup_steps", 0)) < 0:
            raise ValueError("WAM warmup_steps must be nonnegative")
        self.actor_return_reduction = str(self.settings.get(
            "actor_return_reduction",
            "sum" if dict((world_model_config or {}).get("cfg", {})).get("state_type") == "physical" else "mean",
        ))
        if self.actor_return_reduction not in {"sum", "mean"}:
            raise ValueError("actor_return_reduction must be 'sum' or 'mean'")
        if int(self.settings.get("gradient_diagnostics_interval", 0)) < 0:
            raise ValueError("gradient_diagnostics_interval must be nonnegative")
        for name in ["window", "horizon", "batch_size", "world_updates", "critic_epochs",
                     "replay_steps", "replay_envs", "collect_steps", "actor_updates"]:
            if int(self.settings.get(name, 1)) < 1:
                raise ValueError(f"WAM setting {name} must be positive")
        if int(self.settings.get("replay_steps", 256)) < int(self.settings.get("window", 8)):
            raise ValueError("replay_steps must be at least the fitting window")
        # The default Hydra timestamp can be evaluated in different seconds on
        # different ranks. All rank sidecars must use rank zero's actual path.
        shared_path = [str(log_dir or "logs_rl/wam")]
        if self.distributed.world_size > 1:
            dist.broadcast_object_list(shared_path, src=0)
        self.log_dir = Path(shared_path[0])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.iteration = 0
        self.world_updates = 0
        self._last_collection_metrics = {}
        self.obs = env.reset_all()
        backbone = getattr(model, "actor_module", None)
        if "state_observation_keys" in self.settings or "goal_observation_keys" in self.settings:
            state_keys = self.settings.get("state_observation_keys", ["wam"])
            goal_keys = self.settings.get("goal_observation_keys", ["tokenizer"])
            exogenous_features = None
        elif "exogenous_tokenizer_features" in self.settings:
            # Compatibility with the first opt-in WAM configuration. New
            # runs should use the explicit split above.
            state_keys = self.settings.get("observation_keys", ["actor_obs", "tokenizer"])
            goal_keys = []
            exogenous_features = self.settings.get("exogenous_tokenizer_features")
        else:
            state_keys, goal_keys, exogenous_features = ["wam"], ["tokenizer"], None
        self.layout = ObservationLayout(
            self.obs, state_keys, goal_keys, getattr(backbone, "tokenizer_obs_specs", ()),
            exogenous_features,
        )
        # The original opt-in WAM format stored ``actor_obs`` and ``tokenizer``
        # in one state vector. Treat that layout as legacy even when it came
        # through the explicit exogenous-feature compatibility branch.
        self.layout.legacy_combined = self.layout.legacy_combined or any(
            key == "tokenizer" for key, _, _ in self.layout.state_specs
        )
        self.context_keys = list(self.settings.get("context_observation_keys", []))
        self.reference_keys = list(self.settings.get("reference_observation_keys", []))
        self.physical_wam = dict((world_model_config or {}).get("cfg", {})).get("state_type", "flat") == "physical"
        simulator = getattr(env, "env", env)
        scene = getattr(simulator, "scene", None)
        robot = None if scene is None else scene["robot"]
        declared_type = env.config.get("robot", {}).get("type")
        identity = declared_type or robot or "g1"
        self.robot_type = joint_utils.get_robot_type(identity) if self.physical_wam else "g1"
        self.robot_spec = None
        if self.physical_wam:
            if robot is not None and joint_utils.get_robot_type(robot) != self.robot_type:
                raise ValueError("Configured WAM robot type does not match the articulation joint names")
            self.robot_spec = joint_utils.get_robot_spec(robot if robot is not None else identity)
        self.distributed.check_consistent(self.robot_spec, "WAM robot joint ordering")
        native_names = joint_utils.get_robot_joint_names(self.robot_type, order="mujoco")
        policy_names = tuple(self.robot_spec["policy_joint_names"]) if self.robot_spec else tuple(joint_utils.G1_ISAACLab_ORDER)
        self._mujoco_to_policy = tuple(native_names.index(name) for name in policy_names)
        self._policy_to_mujoco = tuple(policy_names.index(name) for name in native_names)
        self.independent_imagination = self.physical_wam and bool(
            self.settings.get("independent_imagination", True)
        )
        if not self.independent_imagination and int(self.settings.get("replay_steps", 256)) < int(
            self.settings.get("horizon", 32)
        ):
            raise ValueError("replay_steps must be at least the imagination horizon")
        self.reference_bank = WAMReferenceBank(self.layout.tokenizer_specs) if self.independent_imagination else None
        self.compact_reference = self.independent_imagination and bool(
            self.settings.get("compact_reference", True)
        )
        if self.compact_reference and (
            self.layout.goal_specs != [("tokenizer", 0, self.reference_bank.goal_dim)]
            or not self.reference_bank.routing_specs
        ):
            raise ValueError("Compact WAM references require one tokenizer group with modality routing")
        # Sonic's privileged observations contain the *real* robot trajectory.
        # They cannot describe an independently imagined robot. Physical mode
        # builds tracking features from its predicted state and conditions on
        # canonical current/future robot references plus physical context.
        self.critic_keys = [] if self.physical_wam else list(self.settings.get("critic_observation_keys", []))
        tokenizer_offset = next((start for key, start, _ in self.layout.goal_specs if key == "tokenizer"), None)
        self.goal_observations = WAMGoalObservations(
            self.layout.tokenizer_specs if tokenizer_offset is not None else (),
            tokenizer_offset=tokenizer_offset or 0,
        ) if self.physical_wam else None
        for key in self.context_keys:
            if key not in self.obs:
                raise ValueError(f"WAM context observation group {key!r} is missing")
        for key in self.reference_keys + self.critic_keys:
            if key not in self.obs:
                raise ValueError(f"WAM auxiliary observation group {key!r} is missing")
        self.context_dim = sum(self.obs[key].shape[-1] for key in self.context_keys)
        self.reference_dim = sum(self.obs[key].shape[-1] for key in self.reference_keys)
        self.critic_feature_dim = sum(self.obs[key].shape[-1] for key in self.critic_keys)
        self._actor_history_specs, self._wam_term_specs = self._resolve_history_specs()
        self.actor_obs = self._seed_actor_history(self.obs["actor_obs"], self.obs.get("wam"))
        self.distributed.check_consistent(
            (self.layout.state_specs, self.layout.goal_specs, self.layout.tokenizer_specs,
             self.context_keys, self.reference_keys, self.critic_keys, self.settings,
             args.num_total_batches, config.get("num_learning_iterations"),
             config.get("gamma", 0.99), config.get("lam", 0.95)),
            "WAM layout/settings",
        )
        cfg = dict(world_model_config or {})
        if cfg and not str(cfg.get("_target_", "")).endswith("StateSpaceWorldModel"):
            raise ValueError("trainer=wam requires world_model=wam_dynamics")
        cfg = dict(cfg.get("cfg", {}))
        cfg.pop("_target_", None)
        cfg["state_dim"] = self.layout.state_dim
        # The WAM transition model is intentionally goal-free. Raw multimodal
        # goals remain in replay for the actor's encoder/FSQ forward pass, but
        # never enter the learned robot dynamics or its shared heads.
        cfg["goal_dim"] = 0
        self.env_action_dim = int(env.config.robot.actions_dim)
        self.physical_wam = cfg.get("state_type", "flat") == "physical"
        cfg["action_dim"] = NUM_G1_JOINTS if self.physical_wam else self.env_action_dim
        if cfg.get("context_dim", 0) not in (0, self.context_dim):
            raise ValueError(
                f"world_model.context_dim={cfg.get('context_dim')} does not match "
                f"context observations ({self.context_dim})"
            )
        cfg["context_dim"] = self.context_dim
        self.distributed.check_consistent(cfg, "WAM dynamics settings")
        self.wam_action_dim = int(cfg["action_dim"])
        self.world_model = StateSpaceWorldModel(WAMConfig(**cfg)).to(self.device)
        self.reward_model = None
        simulator = getattr(env, "env", env)
        self.context_schema = deepcopy(getattr(simulator, "_wam_context_schema", None))
        if self.physical_wam and self.settings.get("complete_context", False):
            if self.context_schema is None or self.context_schema.get("dim") != self.context_dim:
                raise ValueError("Complete WAM context requires matching simulator context metadata")
        self.distributed.check_consistent(self.context_schema, "WAM physical context schema")
        self.numerical_spec = {
            "delta_norm_eps": self.world_model.delta_norm.eps,
            "state_error_floor": self.world_model.state_error_floor.cpu().tolist(),
            "observation_noise": wam_observation_noise(self.settings),
            "actor_return_reduction": self.actor_return_reduction,
            "aux_loss_scale": float(self.settings.get("aux_loss_scale", 1.0)),
            "error_frame": self.world_model.error_frame,
            "loss_reduction": self.world_model.loss_reduction,
        } if self.physical_wam else None
        self.distributed.check_consistent(self.numerical_spec, "WAM numerical settings")
        self.control_dt = float(getattr(simulator, "step_dt", self.settings.get("control_dt", 0.02)))
        if self.physical_wam and self.settings.get("align_real_reward", False):
            reward_cfg = dict(reward_config or {})
            self.distributed.check_consistent((reward_cfg, self.control_dt), "WAM reward settings")
            if reward_cfg:
                self.reward_model = custom_instantiate(reward_cfg, _resolve=False).to(self.device)
            else:
                self.reward_model = WAMTrackingReward(robot_type=self.robot_type).to(self.device)
        self.previous_world_action = torch.zeros(
            env.num_envs, self.wam_action_dim, device=self.device
        )
        if self.physical_wam:
            enable_terminal_capture(simulator)
        self.world_step = self.world_model.forward
        if self.settings.get("compile", False):
            self.world_step = torch.compile(self.world_model.forward, dynamic=False)
        self.critic_features = None
        self.critic_future_frames = int(self.settings.get("critic_future_frames", 10))
        self.critic_future_dt = float(self.settings.get("critic_future_dt", self.control_dt))
        if self.physical_wam:
            if self.reference_dim != PHYSICAL_STATE_DIM:
                raise ValueError("Physical WAM critic requires a 71-D robot reference")
            if self.critic_future_frames < 1 or self.critic_future_dt <= 0:
                raise ValueError("WAM critic future frame count and interval must be positive")
            steps = self.critic_future_dt / self.control_dt
            self.critic_future_stride = round(steps)
            if self.critic_future_stride < 1 or abs(steps - self.critic_future_stride) > 1e-6:
                raise ValueError("WAM critic future interval must be a positive multiple of control dt")
            self.critic_features = WAMCriticFeatures(
                kinematics=self.reward_model.kinematics if self.reward_model is not None else None,
                future_frames=self.critic_future_frames, context_dim=self.context_dim,
                robot_type=self.robot_type,
            ).to(self.device)
            self.critic_input_dim = self.critic_features.input_dim
        else:
            self.critic_input_dim = (
                self.layout.state_dim + self.reference_dim + self.critic_feature_dim + self.context_dim
            )
        self.critic_spec = {
            "input_dim": self.critic_input_dim,
            "future_frames": self.critic_future_frames if self.physical_wam else 0,
            "future_dt": self.critic_future_dt if self.physical_wam else 0.0,
            "independent_imagination": self.independent_imagination,
        }
        self.distributed.check_consistent(self.critic_spec, "WAM critic reference settings")
        critic_cfg = dict(critic_config or {})
        if critic_cfg:
            critic_cfg["input_dim"] = self.critic_input_dim
            self.critic = custom_instantiate(critic_cfg, _resolve=False).to(self.device)
        else:
            # Preserve direct construction for small downstream integrations
            # that instantiate WAMTrainer without a Hydra critic config.
            self.critic = nn.Sequential(
                nn.Linear(self.critic_input_dim, 256), nn.SiLU(),
                nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1),
            ).to(self.device)
        lr = float(self.settings.get("learning_rate", 2e-4))
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=lr)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.actor_optimizer = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.95))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_norm = RunningNorm(self.critic_input_dim).to(self.device)
        self.replay = SequenceReplayBuffer(
            int(self.settings.get("replay_steps", 1000)),
            actor_history_layout=(
                self._actor_history_specs if self.settings.get("compact_replay", True) else None
            ),
        )
        tracked = min(env.num_envs, int(self.settings.get("replay_envs", env.num_envs)))
        self.replay_env_ids = torch.randperm(env.num_envs, device=self.device)[:tracked]
        self._last_runtime_path = None
        self._last_checkpoint_path = None
        self.distributed.check_consistent((str(checkpoint), resume), "checkpoint source")
        if checkpoint is not None:
            error = None
            try:
                self.load_checkpoint(checkpoint, resume=resume)
            except Exception as exc:
                error = exc
            if not self.distributed.all_finite(error is None):
                raise RuntimeError("A rank failed to load the WAM checkpoint") from error
        self.distributed.check_consistent(
            [(name, tuple(value.shape), str(value.dtype))
             for module in (self.policy, self.world_model, self.critic)
             for name, value in module.state_dict().items()],
            "WAM model shapes",
        )
        self.distributed.broadcast_modules(self.policy, self.world_model, self.critic, self.critic_norm)

    def _normalizers(self):
        return [module for module in (
            self.world_model.state_norm, self.world_model.action_norm,
            self.world_model.goal_norm, self.world_model.context_norm, self.world_model.delta_norm,
        ) if module is not None]

    def _action(self, state, goal=None, *, actor_obs=None, auxiliary=False):
        if actor_obs is None:
            # Legacy combined layouts unpack their tokenizer together with the
            # policy observation. The new WAM layout supplies actor_obs
            # separately and therefore reaches this branch only in tests or
            # lightweight adapters.
            obs = self.layout.unpack(state)
            if "actor_obs" not in obs:
                obs["actor_obs"] = state
        else:
            if goal is not None and actor_obs.ndim == 2:
                actor_obs = actor_obs.unsqueeze(1)
            obs = {"actor_obs": actor_obs}
        if goal is not None:
            obs.update(self.layout.unpack_goal(goal))
        result = self.policy.forward(
            obs, is_training=auxiliary, cache_encoded_tensors=False
        )
        # UniversalTokenModule returns [B, T, A]; small test policies and
        # lightweight adapters may return [B, A]. Keep both interfaces valid.
        action = result[:, -1] if result.ndim == 3 else result
        limit = self.env.config.get("action_clip_value", None)
        if limit is not None and limit > 0:
            action = action.clamp(-limit, limit)
        return action

    def _pack_context(self, obs):
        if not self.context_keys:
            return None
        return torch.cat([obs[key] for key in self.context_keys], dim=-1)

    @staticmethod
    def _pack_keys(obs, keys):
        if not keys:
            first = next(iter(obs.values()))
            return first.new_zeros(first.shape[0], 0)
        return torch.cat([obs[key].flatten(start_dim=1) for key in keys], dim=-1)

    def _next_physical_reference(self):
        """Capture the old clip's next reference before an auto-reset changes it."""
        if self.reference_dim != PHYSICAL_STATE_DIM:
            raise ValueError("Aligned physical termination requires a 71-D WAM reference")
        simulator = getattr(self.env, "env", self.env)
        command = simulator.command_manager.get_term("motion")
        return reference_state(command)

    def _physical_future_reference(self, *, next_step=False):
        """Snapshot one robot-reference window before the simulator can reset.

        This privileged critic condition comes from the canonical motion
        library, independently of which actor encoder is selected. Saving it
        in replay keeps old samples valid across motion-library reloads.
        """
        simulator = getattr(self.env, "env", self.env)
        command = simulator.command_manager.get_term("motion")
        future = future_reference_states(
            command, num_frames=self.critic_future_frames,
            frame_stride=self.critic_future_stride, next_step=next_step,
        )
        return future.flatten(start_dim=1)

    def _imagination_references(self, batch):
        references = batch.get("references")
        if references is None or not self.physical_wam:
            return references
        # Keep the final pre-reset target for the terminal state and bootstrap;
        # later replay entries belong to another episode and are masked out.
        lengths = batch["mask"].sum(-1)
        indices = torch.minimum(
            torch.arange(batch["mask"].shape[1] + 1, device=self.device)[None, :],
            lengths[:, None],
        )
        return references.gather(
            1, indices[..., None].expand(-1, -1, references.shape[-1]),
        )

    @staticmethod
    def _hold_sequence_at_end(sequence, lengths, *, terminal_state=False):
        """Keep the last valid transition's auxiliary data after an episode ends."""
        indices = torch.minimum(
            torch.arange(sequence.shape[1], device=sequence.device)[None, :],
            (lengths - (0 if terminal_state else 1)).clamp_min(0)[:, None],
        )
        return sequence.gather(1, indices[..., None].expand(-1, -1, sequence.shape[-1]))

    def _imagination_contexts(self, batch):
        return self._hold_sequence_at_end(batch["contexts"], batch["mask"].sum(-1))

    def _resolve_history_specs(self):
        """Resolve term-major actor history layout from Isaac Lab metadata.

        Isaac Lab concatenates each historical observation term as
        ``[history, term_dim]`` before joining terms. WAM emits one frame in
        term order, so advancing the actor history must shift each term block
        independently instead of slicing one global frame-sized prefix.
        """
        observation_manager = getattr(getattr(self.env, "env", None), "observation_manager", None)
        if observation_manager is None:
            return None, None
        names = getattr(observation_manager, "_group_obs_term_names", {})
        dims = getattr(observation_manager, "_group_obs_term_dim", {})
        if "policy" not in names or "wam" not in names:
            return None, None
        policy_names, policy_dims = names["policy"], dims["policy"]
        wam_names, wam_dims = names["wam"], dims["wam"]
        wam_specs = []
        wam_offset = 0
        for name, shape in zip(wam_names, wam_dims):
            flat = 1
            for value in shape:
                flat *= int(value)
            wam_specs.append((name, wam_offset, wam_offset + flat))
            wam_offset += flat
        frame_dims = {name: end - start for name, start, end in wam_specs}
        actor_specs = []
        offset = 0
        for name, shape in zip(policy_names, policy_dims):
            shape = tuple(shape)
            flat = 1
            for value in shape:
                flat *= int(value)
            if name in frame_dims:
                # Isaac Lab may flatten [history, term_dim] into one axis
                # (e.g. joint_pos: (290,)). Recover the history from the
                # matching single-frame term, not from the physical state.
                term_dim = frame_dims[name]
                if term_dim < 1 or flat % term_dim:
                    raise ValueError(f"actor history for {name!r} does not match its WAM frame")
                history = flat // term_dim
            elif len(shape) < 2:
                history, term_dim = 1, flat
            else:
                history, term_dim = int(shape[-2]), int(shape[-1])
            actor_specs.append((name, offset, offset + flat, history, term_dim))
            offset += flat
        if offset != self.obs["actor_obs"].shape[-1] or wam_offset != self.obs["wam"].shape[-1]:
            return None, None
        return actor_specs, wam_specs

    def _seed_actor_history(self, actor_obs, wam_obs):
        """Fill the actor's history with one clean SuperMimic WAM frame."""
        if wam_obs is None or self._actor_history_specs is None or self._wam_term_specs is None:
            return actor_obs
        wam_by_name = {name: (start, end) for name, start, end in self._wam_term_specs}
        updated = actor_obs.clone()
        for name, start, end, history, term_dim in self._actor_history_specs:
            if name not in wam_by_name:
                continue
            wam_start, wam_end = wam_by_name[name]
            if wam_end - wam_start != term_dim:
                continue
            frame = wam_obs[..., wam_start:wam_end].unsqueeze(1)
            updated[..., start:end] = frame.expand(-1, history, -1).reshape(actor_obs.shape[0], -1)
        return updated

    @staticmethod
    def optimizer_step(loss, optimizer, parameters, *, clip_value=None, distributed=None):
        """Fail before mutating weights if imagination produces non-finite gradients."""
        parameters = list(parameters)
        distributed = distributed or DistributedContext(loss.device)
        optimizer.zero_grad(set_to_none=True)
        if not distributed.all_finite(loss):
            raise FloatingPointError("Non-finite WAM loss; inspect model fit and imagination horizon")
        loss.backward()
        distributed.average_gradients(parameters)
        norm = nn.utils.clip_grad_norm_(parameters, 1.0)
        if not distributed.all_finite(norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite WAM gradient; optimizer update was rejected")
        if clip_value is not None:
            nn.utils.clip_grad_value_(parameters, clip_value)
        optimizer.step()
        return norm

    @torch.no_grad()
    def collect(self, steps):
        self.policy.eval()
        norms = self._normalizers()
        before = self.distributed.snapshot_running_norms(norms)
        total = 0.0
        # Accumulate on-device and transfer the diagnostic scalars once per
        # collection, rather than synchronizing CUDA for every new metric.
        diagnostic_sums = torch.zeros(7, dtype=torch.float64, device=self.device)
        action_abs_max = torch.zeros((), device=self.device)
        episode_logs = {}
        for _ in range(steps):
            state = self.layout.pack(self.obs)
            goal = self.layout.pack_goal(self.obs)
            actor_obs = self.actor_obs
            reference = self._pack_keys(self.obs, self.reference_keys)
            critic_state = self._pack_keys(self.obs, self.critic_keys)
            context = self._pack_context(self.obs)
            reference_index = None
            if self.independent_imagination:
                simulator = getattr(self.env, "env", self.env)
                reference_index = self.reference_bank.capture(
                    simulator.command_manager.get_term("motion"), simulator.observation_manager,
                )
            future_reference = (
                self._physical_future_reference()
                if self.physical_wam and not self.independent_imagination else None
            )
            next_future_reference = (
                self._physical_future_reference(next_step=True)
                if self.physical_wam and not self.independent_imagination else None
            )
            transition_reference = (
                self._next_physical_reference()
                if self.physical_wam and (
                    self.settings.get("align_real_termination", False)
                    or self.settings.get("align_real_reward", False)
                )
                else None
            )
            if self.layout.goal_dim:
                action = self._action(state, goal, actor_obs=actor_obs)
            elif self.layout.legacy_combined:
                action = self._action(state)
            else:
                action = self._action(actor_obs)
            noise = float(self.settings.get("action_noise", 0.05))
            executed = action + noise * torch.randn_like(action)
            limit = self.env.config.get("action_clip_value", None)
            if limit is not None and limit > 0:
                executed = executed.clamp(-limit, limit)
            simulator = getattr(self.env, "env", self.env)
            push_before = get_push_versions(simulator, self.env.num_envs) if self.physical_wam else None
            next_obs, rewards, dones, infos = self.env.step({"actions": executed})
            next_state = self.layout.pack(next_obs)
            next_goal = self.layout.pack_goal(next_obs)
            # Goals are still in the returned observation's robot frame. Do
            # this conversion before substituting terminal states: done rows
            # of next_obs already describe the newly reset robot.
            replay_goal, replay_next_goal = goal, next_goal
            if self.compact_reference:
                replay_goal = self.reference_bank.pack_routing(goal)
                replay_next_goal = self.reference_bank.pack_routing(next_goal)
            elif self.physical_wam and self.layout.goal_dim:
                replay_goal = self.goal_observations.to_world(goal, state)
                replay_next_goal = self.goal_observations.to_world(next_goal, next_state)
            next_state_valid = ~dones.bool()
            if self.physical_wam:
                simulator = getattr(self.env, "env", self.env)
                terminal_state = get_terminal_state(simulator)
                if terminal_state is not None:
                    if terminal_state.shape != next_state.shape:
                        raise ValueError("Captured terminal state does not match physical WAM observations")
                    next_state = torch.where(dones[:, None].bool(), terminal_state, next_state)
                    next_state_valid = torch.ones_like(dones, dtype=torch.bool)
                # Random external impulses are not deterministic robot dynamics.
                # Their resulting states can still seed subsequent imagination.
                pushed = get_push_versions(simulator, self.env.num_envs) != push_before
                next_state_valid &= ~pushed.to(device=next_state_valid.device)
            next_wam = next_obs.get("wam")
            next_actor_obs = (
                self._advance_actor_history(actor_obs, next_wam)
                if next_wam is not None else next_obs["actor_obs"]
            )
            if next_wam is not None and dones.any():
                reset_history = self._seed_actor_history(next_obs["actor_obs"], next_wam)
                next_actor_obs = torch.where(dones[:, None].bool(), reset_history, next_actor_obs)
            next_reference = (
                transition_reference if transition_reference is not None
                else self._pack_keys(next_obs, self.reference_keys)
            )
            next_critic_state = self._pack_keys(next_obs, self.critic_keys)
            world_action = self._to_wam_action(executed)
            previous_world_action = self.previous_world_action
            reward = rewards.reshape(state.shape[0], -1).sum(-1)
            timeout = infos.get("time_outs", torch.zeros_like(dones)).bool()
            terminated = infos.get("terminated", dones.bool() & ~timeout).bool()
            at_limit = (
                (executed.abs() >= limit).float().mean()
                if limit is not None and limit > 0 else executed.new_zeros(())
            )
            diagnostic_sums += torch.stack((
                dones.float().mean(), terminated.float().mean(), timeout.float().mean(),
                executed.square().mean(), at_limit, reward.mean(), reward.square().mean(),
            )).double()
            action_abs_max = torch.maximum(action_abs_max, executed.abs().max())
            to_log = infos.get("to_log", {})
            if isinstance(to_log, Mapping):
                for name, value in to_log.items():
                    if not isinstance(name, str) or not name.startswith((
                        "Episode_Reward/", "Episode_Termination/", "Metrics/motion/",
                    )):
                        continue
                    if isinstance(value, torch.Tensor):
                        if value.numel() != 1:
                            continue
                        value = value.detach().reshape(()).to(self.device, dtype=torch.float64)
                    elif isinstance(value, Real):
                        value = diagnostic_sums.new_tensor(value)
                    else:
                        continue
                    # Empty episode-reset batches may produce NaN telemetry;
                    # do not let missing diagnostic values poison all ranks.
                    finite = torch.isfinite(value)
                    if name not in episode_logs:
                        episode_logs[name] = diagnostic_sums.new_zeros(2)
                    accumulator = episode_logs[name]
                    accumulator[0] += torch.where(finite, value, value.new_zeros(()))
                    accumulator[1] += finite
            indices = self.replay_env_ids
            replay_args = (
                state[indices], world_action[indices], reward[indices], next_state[indices],
                dones[indices], terminated[indices],
            )
            if self.layout.goal_dim:
                self.replay.add(
                    *replay_args,
                    replay_goal[indices], replay_next_goal[indices],
                    actor_obs[indices], next_actor_obs[indices],
                    context=context[indices] if context is not None else None,
                    reference=reference[indices] if self.reference_keys and not self.compact_reference else None,
                    next_reference=next_reference[indices] if self.reference_keys and not self.compact_reference else None,
                    critic_state=critic_state[indices] if self.critic_keys else None,
                    next_critic_state=next_critic_state[indices] if self.critic_keys else None,
                    previous_action=previous_world_action[indices],
                    next_state_valid=next_state_valid[indices],
                    future_reference=future_reference[indices] if future_reference is not None else None,
                    next_future_reference=next_future_reference[indices] if next_future_reference is not None else None,
                    reference_index=reference_index[indices] if reference_index is not None else None,
                )
            else:
                self.replay.add(
                    *replay_args,
                    actor_state=actor_obs[indices], next_actor_state=next_actor_obs[indices],
                    context=context[indices] if context is not None else None,
                    reference=reference[indices] if self.reference_keys else None,
                    next_reference=next_reference[indices] if self.reference_keys else None,
                    critic_state=critic_state[indices] if self.critic_keys else None,
                    next_critic_state=next_critic_state[indices] if self.critic_keys else None,
                    previous_action=previous_world_action[indices],
                    next_state_valid=next_state_valid[indices],
                    future_reference=future_reference[indices] if future_reference is not None else None,
                    next_future_reference=next_future_reference[indices] if next_future_reference is not None else None,
                    reference_index=reference_index[indices] if reference_index is not None else None,
                )
            valid = next_state_valid[indices]
            if valid.any():
                self.world_model.update_normalizers(
                    state[indices][valid], world_action[indices][valid], next_state[indices][valid],
                    context=context[indices][valid] if context is not None else None,
                )
            self.obs = next_obs
            self.actor_obs = next_actor_obs
            self.previous_world_action = torch.where(
                dones[:, None].bool(), torch.zeros_like(world_action), world_action
            )
            total += reward.mean().item()
        self.distributed.sync_running_norms(norms, before)
        means = diagnostic_sums / max(steps, 1)
        reward_std = (means[6] - means[5].square()).clamp_min(0).sqrt()
        values = torch.stack((
            means[0], means[1], means[2], means[3].sqrt(), means[4],
            reward_std, action_abs_max.double(),
        )).cpu().tolist()
        self._last_collection_metrics = dict(zip((
            "real/done_fraction", "real/terminated_fraction", "real/truncated_fraction",
            "real/action_rms", "real/action_at_limit_fraction", "real/reward_std",
            "real/action_abs_max",
        ), values))
        for name, accumulator in episode_logs.items():
            value_sum, count = accumulator.cpu().tolist()
            if count:
                self._last_collection_metrics[f"real/{name}"] = value_sum / count
        return total / max(steps, 1)

    def fit_world_model(self):
        window = int(self.settings.get("window", 8))
        batch_size = int(self.settings.get("batch_size", 256))
        metrics = {}
        for _ in range(int(self.settings.get("world_updates", 10))):
            fields = ["states", "actions", "terminated", "mask", "dynamics_mask", "rewards"]
            if self.context_dim:
                fields.append("contexts")
            batch = self.replay.sample(batch_size, window, device=self.device, fields=fields)
            loss_kwargs = {
                "dones": batch["terminated"],
                "mask": batch["mask"],
                "dynamics_mask": batch["dynamics_mask"],
                "rewards": batch["rewards"],
            }
            if self.context_dim:
                loss_kwargs["context"] = batch["contexts"]
            if self.layout.fixed_indices or self.layout.exogenous_indices:
                loss_kwargs["state_projector"] = lambda state, step: self.layout.project(
                    state, batch["states"][:, 0], self._next_reference(batch, step)
                )
            losses = self.world_model.transition_loss(
                batch["states"], batch["actions"], **loss_kwargs,
            )
            self.optimizer_step(
                losses["loss"], self.world_optimizer, self.world_model.parameters(),
                distributed=self.distributed,
            )
            self.world_updates += 1
            metrics = {f"world/{key}": float(value.detach()) for key, value in losses.items()}
        return metrics

    def _next_reference(self, batch, step):
        """Hold old combined-layout references across simulator resets.

        New split layouts keep goals in ``batch["goals"]`` and never enter this
        path.  The helper remains for checkpoints and scripts that used the
        original combined observation-space WAM format.
        """
        return torch.where(
            batch["dynamics_mask"][:, step, None],
            batch["states"][:, step + 1], batch["states"][:, step],
        )

    def _critic_input(self, state, context=None):
        return self._critic_input_with_aux(state, None, None, context)

    def _critic_input_with_aux(
        self, state, reference=None, critic_state=None, context=None, *, future_reference=None,
    ):
        if self.physical_wam:
            if reference is None or future_reference is None:
                raise ValueError("Physical WAM critic requires current and future robot references")
            if future_reference.ndim == state.ndim:
                if future_reference.shape[-1] != self.critic_future_frames * PHYSICAL_STATE_DIM:
                    raise ValueError("Physical WAM critic future reference has the wrong flattened size")
                future_reference = future_reference.reshape(
                    *future_reference.shape[:-1], self.critic_future_frames, PHYSICAL_STATE_DIM,
                )
            if self.context_dim:
                if context is None:
                    raise ValueError("WAM context is required by the critic")
                context = self.world_model.context_norm(context)
            else:
                context = state.new_zeros(*state.shape[:-1], 0)
            return self.critic_features(state, reference, context, future_reference)
        state_features = self.world_model.state_norm(state)
        features = [state_features]
        if self.reference_dim:
            if reference is None:
                raise ValueError("reference features are required by the WAM critic")
            features.append(reference)
        if self.critic_feature_dim:
            if critic_state is None:
                raise ValueError("tracking critic features are required by the WAM critic")
            features.append(critic_state)
        if self.context_dim:
            if context is None:
                raise ValueError("WAM context is required by the critic")
            features.append(self.world_model.context_norm(context))
        return torch.cat(features, dim=-1)

    def _to_wam_action(self, action):
        """Project a Sonic action to the 29 body actions used by SuperMimic."""
        if not self.physical_wam:
            return action
        if action.shape[-1] == self.wam_action_dim:
            if self.robot_type == "n3":
                order = torch.as_tensor(self._policy_to_mujoco, device=action.device)
                return action[..., order]
            # Preserve the historical G1 29-action convention for checkpoints.
            return action
        scene = getattr(getattr(self.env, "env", None), "scene", None)
        robot = None if scene is None else scene["robot"]
        if robot is None:
            raise ValueError("physical WAM needs the IsaacLab robot asset to select body actions")
        indices = joint_utils.get_body_joint_indices(robot)
        body_action = action[..., indices]
        order = torch.as_tensor(self._policy_to_mujoco, device=action.device)
        return body_action[..., order]

    def _physical_actor_frame(self, state, action, context=None):
        """Convert a physical WAM state into one Sonic proprioceptive frame."""
        position, rotation, _, angular_velocity, joint_position, joint_velocity = physical_state_parts(state)
        del position
        index = torch.as_tensor(self._mujoco_to_policy, device=state.device)
        joint_position = joint_position[..., index]
        joint_velocity = joint_velocity[..., index]
        scene = getattr(getattr(self.env, "env", None), "scene", None)
        robot = None if scene is None else scene["robot"]
        if robot is None:
            raise ValueError("physical WAM actor projection needs the IsaacLab robot asset")
        body_indices = joint_utils.get_body_joint_indices(robot)
        # Replay samples shuffle environments and may come from older episodes.
        # A live environment row is never a sample's joint calibration. ξ stores
        # its 29 offsets in canonical order; add them to immutable nominals.
        nominal = getattr(robot.data, "default_joint_pos_nominal", None)
        if nominal is None:
            nominal = robot.data.default_joint_pos[0].detach().clone()
            if self.context_dim >= 33:
                live_context = self._pack_context(self.obs)
                nominal[body_indices] -= live_context[0, 1:30][index]
        elif nominal.ndim == 2:
            nominal = nominal[0]
        default_position = nominal[body_indices].to(device=state.device, dtype=state.dtype)
        if context is not None:
            if context.shape[-1] < 33:
                raise ValueError("Physical actor calibration requires the WAM ξ calibration prefix")
            default_position = default_position + context[..., 1:30][..., index]
        elif self.context_dim:
            raise ValueError("Physical actor projection requires replayed ξ")
        if default_position.shape[-1] != joint_position.shape[-1]:
            raise ValueError("physical WAM and actor joint layouts do not match")
        gravity = -rotation[..., 2, :]
        local_angular_velocity = (rotation.transpose(-1, -2) @ angular_velocity.unsqueeze(-1)).squeeze(-1)
        values = {
            "gravity_dir": gravity,
            "base_ang_vel": local_angular_velocity,
            "joint_pos": joint_position - default_position,
            "joint_vel": joint_velocity,
            "actions": action,
        }
        if self._wam_term_specs is None:
            return torch.cat(tuple(values.values()), dim=-1)
        frame = torch.cat(
            [values[name] for name, _, _ in self._wam_term_specs], dim=-1
        )
        noise_half_width = wam_observation_noise(self.settings)
        noise = frame.new_zeros(frame.shape)
        for name, start, end in self._wam_term_specs:
            width = noise_half_width.get(name, 0.0)
            if width:
                noise[..., start:end] = (2 * torch.rand_like(frame[..., start:end]) - 1) * width
        return frame + noise

    @staticmethod
    def _quat_error_squared(rotation, target_rotation):
        relative = rotation.transpose(-1, -2) @ target_rotation
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
        return torch.acos(cosine).square()

    def _analytic_reward(self, state, reference, action, action_prev):
        if not self.physical_wam:
            raise RuntimeError("analytic WAM reward requires physical state mode")
        if getattr(self, "reward_model", None) is not None:
            return self.reward_model(state, reference, action, action_prev, dt=self.control_dt)
        position, rotation, linear_velocity, angular_velocity, joint_position, joint_velocity = physical_state_parts(state)
        target_position, target_rotation, target_linear_velocity, target_angular_velocity, target_joint_position, target_joint_velocity = physical_state_parts(reference)
        e_pos = (position - target_position).square().sum(-1)
        e_ori = self._quat_error_squared(rotation, target_rotation)
        e_lin = (linear_velocity - target_linear_velocity).square().sum(-1)
        e_ang = (angular_velocity - target_angular_velocity).square().sum(-1)
        e_q = (joint_position - target_joint_position).square().mean(-1)
        e_qd = (joint_velocity - target_joint_velocity).square().mean(-1)
        track = (
            torch.exp(-e_pos / 0.3**2)
            + torch.exp(-e_ori / 0.4**2)
            + torch.exp(-e_lin / 1.0**2)
            + torch.exp(-e_ang / 3.14**2)
            + torch.exp(-e_q / 0.3**2)
            + torch.exp(-e_qd / 1.0**2)
        )
        action_rate = (action - action_prev).square().mean(-1)
        joint_limit = torch.relu(joint_position.abs() - 2.5).mean(-1)
        dt = float(self.settings.get("control_dt", 0.02))
        return (track - 0.1 * action_rate - 2.0 * joint_limit) * dt

    def _analytic_termination(self, state, reference):
        return soft_tracking_termination(
            state[..., :3], state[..., 3:7], reference[..., :3], reference[..., 3:7],
            height_threshold=float(self.settings.get("termination_height", 0.25)),
            tilt_threshold=float(self.settings.get("termination_tilt", 0.8)),
            height_width=float(self.settings.get("termination_height_width", 0.02)),
            tilt_width=float(self.settings.get("termination_tilt_width", 0.04)),
        )

    def _advance_actor_history(self, actor_obs, wam_obs, action=None, *, physical_state=False, context=None):
        """Append one single-step WAM observation to Sonic's actor history."""
        if self.physical_wam and physical_state:
            wam_obs = self._physical_actor_frame(wam_obs, action, context=context)
        if actor_obs.shape[-1] == wam_obs.shape[-1]:
            return wam_obs
        if self._actor_history_specs is not None and self._wam_term_specs is not None:
            wam_by_name = {name: (start, end) for name, start, end in self._wam_term_specs}
            updated = actor_obs.clone()
            for name, start, end, history, term_dim in self._actor_history_specs:
                if name not in wam_by_name:
                    # Future/reference terms are replayed through the tokenizer
                    # stream. Only the robot proprioceptive terms are generated
                    # from an imagined physical state.
                    continue
                wam_start, wam_end = wam_by_name[name]
                if wam_end - wam_start != term_dim:
                    continue
                term_history = actor_obs[..., start:end].reshape(
                    actor_obs.shape[0], history, term_dim
                )
                latest = wam_obs[..., wam_start:wam_end].unsqueeze(1)
                updated[..., start:end] = torch.cat(
                    (term_history[:, 1:], latest), dim=1
                ).reshape(actor_obs.shape[0], -1)
            return updated
        if actor_obs.shape[-1] % wam_obs.shape[-1] != 0:
            raise ValueError(
                "actor_obs must be a concatenated history of the single-step WAM observation; "
                f"got actor_dim={actor_obs.shape[-1]}, wam_dim={wam_obs.shape[-1]}"
            )
        return torch.cat((actor_obs[..., wam_obs.shape[-1] :], wam_obs), dim=-1)

    def _sample_policy_batch(self, horizon):
        size = int(self.settings.get("policy_batch_size", self.settings.get("batch_size", 256)))
        if not self.independent_imagination:
            return self.replay.sample(size, horizon, device=self.device)
        seed = self.replay.sample_starts(
            size, device=self.device, eligible=self.reference_bank.valid_starts,
        )
        batch = self.reference_bank.trajectory(
            seed["reference_index"], seed["goal"], horizon,
            self.critic_future_frames, self.critic_future_stride, self.device,
            routing_only=self.compact_reference,
        )
        # These are seed metadata, not a logged trajectory. The reference
        # advances independently and ξ remains the seed episode's physics.
        batch["states"] = seed["state"][:, None].expand(-1, horizon + 1, -1)
        batch["actor_states"] = seed["actor_state"][:, None].expand(-1, horizon + 1, -1)
        batch["actions"] = seed["action"][:, None].expand(-1, horizon, -1)
        batch["previous_actions"] = seed["previous_action"][:, None].expand(-1, horizon, -1)
        batch["dones"] = torch.zeros(size, horizon, dtype=torch.bool, device=self.device)
        if self.context_dim:
            batch["contexts"] = seed["context"][:, None].expand(-1, horizon, -1)
        return batch

    def improve_policy(self):
        horizon = int(self.settings.get("horizon", 32))
        batch = self._sample_policy_batch(horizon)
        seed = batch["states"][:, 0].detach()
        actor_obs = (
            batch["states"][:, 0].detach()
            if self.layout.legacy_combined
            else batch.get("actor_states", batch["states"])[:, 0].detach()
        )
        gamma = float(self.config.get("gamma", 0.99))
        lam = float(self.config.get("lam", 0.95))
        # ξ is an environment parameter attached to every logged transition.
        # Keep the whole sequence so imagined step t uses ξ_t, matching the
        # SuperMimic replay and WAM training paths.
        context = self._imagination_contexts(batch) if self.context_dim else None
        references = self._imagination_references(batch)
        lengths = batch["mask"].sum(-1)
        future_references = None
        if self.physical_wam:
            if "future_references" not in batch:
                raise ValueError("Physical WAM replay is missing the critic's future robot references")
            future_references = self._hold_sequence_at_end(
                batch["future_references"], lengths, terminal_state=True,
            )
        # Legacy non-physical integrations may still have auxiliary critic
        # observations. Never let their reset-episode rows reach a bootstrap.
        critic_states = None
        if self.critic_feature_dim:
            ends_in_reset = batch["dones"].gather(1, (lengths - 1)[:, None]).squeeze(1)
            critic_states = self._hold_sequence_at_end(
                batch["critic_states"], lengths - ends_in_reset.long(), terminal_state=True,
            )
        goals = batch.get("goals")
        if self.physical_wam and goals is not None:
            goals = self._hold_sequence_at_end(goals, lengths)
        previous_action = (
            batch["previous_actions"][:, 0]
            if self.physical_wam and "previous_actions" in batch
            else batch["actions"][:, 0]
        )
        self.policy.train()
        state = seed
        states, rewards, dones = [], [], []
        survival = torch.ones(seed.shape[0], device=self.device)
        objective = torch.zeros_like(survival)
        auxiliary_loss = seed.new_zeros(())
        gradient_metrics = {}
        with frozen(self.world_model), frozen(self.critic):
            for step in range(horizon):
                states.append(state)
                valid = batch["mask"][:, step]
                goal = goals[:, step] if self.layout.goal_dim else None
                if self.physical_wam and goal is not None:
                    goal = self.goal_observations.to_robot(goal, state)
                    if self.independent_imagination:
                        goal = self.reference_bank.process_goals(goal)
                if self.layout.goal_dim:
                    action = self._action(
                        state, goal, actor_obs=actor_obs, auxiliary=(step == 0)
                    )
                elif self.layout.legacy_combined:
                    action = self._action(state, auxiliary=(step == 0))
                else:
                    action = self._action(actor_obs, auxiliary=(step == 0))
                if step == 0 and getattr(self.policy, "has_aux_loss", False):
                    coefficients = self.policy.aux_loss_coef or {}
                    for name, loss in (self.policy.aux_losses or {}).items():
                        auxiliary_loss = auxiliary_loss + coefficients.get(name, 0.0) * loss
                kwargs = {}
                if not self.layout.goal_dim and (
                    self.layout.fixed_indices or self.layout.exogenous_indices
                ):
                    kwargs["next_state_projector"] = lambda predicted, step=step: self.layout.project(
                        predicted, batch["states"][:, 0], self._next_reference(batch, step)
                    )
                world_action = self._to_wam_action(action)
                context_t = context[:, step] if context is not None else None
                prediction = self.world_step(state, world_action, context=context_t, **kwargs)
                if self.physical_wam:
                    reference = references[:, step + 1]
                    reward = self._analytic_reward(
                        prediction["next_state"], reference, world_action, previous_action
                    )
                    done = self._analytic_termination(prediction["next_state"], reference)
                else:
                    reward = prediction["reward"]
                    done = prediction["termination_logits"].sigmoid()
                objective = objective + gamma**step * survival * reward * valid
                survival = torch.where(valid, survival * (1.0 - done), survival)
                state = torch.where(valid[:, None], prediction["next_state"], state)
                next_actor_obs = self._advance_actor_history(
                    actor_obs, prediction["next_state"], action, physical_state=True,
                    context=context_t,
                )
                actor_obs = torch.where(valid[:, None], next_actor_obs, actor_obs)
                previous_action = world_action
                rewards.append(reward)
                dones.append(done)
            reference = references[:, horizon] if self.reference_dim else None
            critic_state = critic_states[:, horizon] if self.critic_feature_dim else None
            bootstrap_context = context[:, -1] if context is not None else None
            bootstrap_input = self._critic_input_with_aux(
                state, reference, critic_state, bootstrap_context,
                **({"future_reference": future_references[:, horizon]} if self.physical_wam else {}),
            )
            bootstrap = self.critic(
                self.critic_norm(bootstrap_input)
            ).squeeze(-1)
            objective = objective + gamma**lengths * survival * bootstrap
            return_loss = -objective.mean()
            if self.actor_return_reduction == "mean":
                return_loss = return_loss / horizon
            weighted_auxiliary_loss = float(self.settings.get("aux_loss_scale", 1.0)) * auxiliary_loss
            loss = return_loss + weighted_auxiliary_loss
            interval = int(self.settings.get("gradient_diagnostics_interval", 0))
            if interval and self.iteration % interval == 0:
                self.actor_optimizer.zero_grad(set_to_none=True)
                gradient_metrics = policy_gradient_metrics(
                    return_loss, weighted_auxiliary_loss, self.policy.named_parameters(),
                )
            grad = self.optimizer_step(
                loss, self.actor_optimizer, self.policy.parameters(), clip_value=0.1,
                distributed=self.distributed,
            )

        # GAE/lambda targets are fixed once per imagined trajectory. The critic
        # sees detached states so its regression cannot update WAM or the actor.
        states = torch.stack([item.detach() for item in states] + [state.detach()], dim=1)
        rewards = torch.stack(rewards, dim=1).detach()
        dones = torch.stack(dones, dim=1).detach()
        with torch.no_grad():
            critic_context = None
            if context is not None:
                # The replay stores ξ for each transition (H values), while
                # the value sequence has H+1 states.  Hold the final ξ for the
                # bootstrap state, exactly as the terminal physical context
                # is held in SuperMimic.
                critic_context = torch.cat((context, context[:, -1:]), dim=1).detach()
            references = references[:, :horizon + 1] if self.reference_dim else None
            inputs = self._critic_input_with_aux(
                states, references, critic_states, critic_context,
                **({"future_reference": future_references} if self.physical_wam else {}),
            )
            # A valid trajectory has L transitions and L+1 states. Exclude
            # frozen padding (and reset episode metadata) from normalization.
            valid_states = torch.arange(horizon + 1, device=self.device)[None, :] <= lengths[:, None]
            before = self.distributed.snapshot_running_norms([self.critic_norm])
            self.critic_norm.update(inputs[valid_states])
            self.distributed.sync_running_norms([self.critic_norm], before)
            inputs = self.critic_norm(inputs)
            values = self.critic(inputs).squeeze(-1)
            target = values.gather(1, lengths[:, None]).squeeze(1)
            targets = []
            for step in range(horizon - 1, -1, -1):
                next_target = rewards[:, step] + gamma * (1 - dones[:, step]) * (
                    (1 - lam) * values[:, step + 1] + lam * target
                )
                target = torch.where(batch["mask"][:, step], next_target, target)
                targets.append(target)
            targets = torch.stack(targets[::-1], dim=1)
        n = inputs[:, :-1].reshape(-1, inputs.shape[-1]).shape[0]
        minibatches = int(self.settings.get("critic_minibatches", 4))
        for _ in range(int(self.settings.get("critic_epochs", 5))):
            permutation = torch.randperm(n, device=self.device)
            for indices in torch.tensor_split(permutation, min(minibatches, n)):
                x_batch = inputs[:, :-1].reshape(-1, inputs.shape[-1])[indices]
                y_batch = targets.reshape(-1)[indices]
                weights = batch["mask"].reshape(-1).to(dtype=x_batch.dtype)[indices]
                errors = (self.critic(x_batch).squeeze(-1) - y_batch).square()
                critic_loss = (errors * weights).sum() / weights.sum().clamp_min(1)
                self.optimizer_step(
                    critic_loss, self.critic_optimizer, self.critic.parameters(),
                    distributed=self.distributed,
                )
        return {
            "actor/loss": float(loss.detach()), "actor/grad_norm": float(grad),
            "actor/return_loss": float(return_loss.detach()),
            "actor/imagined_return": float(objective.detach().mean()),
            "actor/survival": float(survival.detach().mean()),
            "actor/effective_horizon": float(batch["mask"].sum(-1).float().mean()),
            "actor/aux_loss": float(auxiliary_loss.detach()),
            "critic/loss": float(critic_loss.detach()),
            **gradient_metrics,
        }

    def train(self):
        iterations = self.args.num_total_batches
        if iterations is None:
            iterations = int(self.config.get("num_learning_iterations", 100000))
        window = int(self.settings.get("window", 8))
        step_accumulator = 0.0
        collected_steps = 0
        warmup_steps = int(self.settings.get("warmup_steps", 0))
        # Seed the real replay before any fitting, as in SuperMimic. Retain
        # warmup_before_updates=false for controlled comparisons with old runs.
        missing_warmup = max(0, warmup_steps - len(self.replay))
        if (
            self.iteration < iterations and self.world_updates == 0 and missing_warmup
            and self.settings.get("warmup_before_updates", True)
        ):
            self.collect(missing_warmup)
            collected_steps = missing_warmup
        for iteration in range(self.iteration, iterations):
            started = time.perf_counter()
            self._maybe_resample_motions(iteration)
            decay_iters = int(self.settings.get("learning_rate_decay_iters", 250000))
            min_ratio = float(self.settings.get("learning_rate_min_ratio", 0.03))
            if decay_iters > 0:
                progress = min(iteration / decay_iters, 1.0)
                ratio = min_ratio + (1 - min_ratio) * 0.5 * (1 + torch.cos(
                    torch.tensor(torch.pi * progress, device=self.device)
                ).item())
            else:
                ratio = 1.0
            learning_rate = float(self.settings.get("learning_rate", 2e-4)) * ratio
            for optimizer in (self.world_optimizer, self.actor_optimizer, self.critic_optimizer):
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            utd = float(self.settings.get("utd", 0.0))
            if utd > 0:
                step_accumulator += 1.0 / utd
                collect_steps = int(step_accumulator)
                step_accumulator -= collect_steps
            else:
                collect_steps = int(self.settings.get("collect_steps", 1))
            if collect_steps:
                collected_steps += collect_steps
            metrics = {"real/reward": self.collect(collect_steps) if collect_steps else 0.0}
            metrics.update(self._last_collection_metrics)
            if self.distributed.all_finite(len(self.replay) >= window):
                metrics.update(self.fit_world_model())
            can_imagine = (
                self.world_updates >= int(self.settings.get("warmup_world_updates", 100))
                and iteration >= int(self.settings.get("wm_only_iters", 0))
                and (
                    collected_steps >= warmup_steps
                    or (self.settings.get("warmup_before_updates", True) and len(self.replay) >= warmup_steps)
                )
                and len(self.replay) >= (1 if self.independent_imagination else int(self.settings.get("horizon", 32)))
            )
            if self.distributed.all_finite(can_imagine):
                for _ in range(int(self.settings.get("actor_updates", 1))):
                    metrics.update(self.improve_policy())
            if hasattr(self.env, "sync_and_compute_adaptive_sampling"):
                self.env.sync_and_compute_adaptive_sampling(
                    self.accelerator, sync_across_gpus=False
                )
            self.iteration = iteration + 1
            metrics.update(
                iteration=self.iteration,
                seconds=time.perf_counter() - started,
                learning_rate=learning_rate,
                collected_steps=collected_steps,
                **{"replay/host_gib": self.replay.storage_nbytes() / 1024**3},
            )
            metrics = self.distributed.reduce_metrics(metrics)
            if self.distributed.is_main:
                print("WAM " + " ".join(f"{key}={value:.5g}" for key, value in metrics.items()))
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(metrics, step=self.iteration)
                except ImportError:
                    pass
            frequency = int(self.settings.get("save_interval", 500))
            if frequency > 0 and self.iteration % frequency == 0:
                self.save_checkpoint(self.log_dir / "last.pt")
        self.save_checkpoint(self.log_dir / "last.pt")

    def _maybe_resample_motions(self, iteration):
        """Refresh the loaded motion subset without splicing replay episodes.

        PPO's ImResampleCallback is not run by this trainer. Follow its default
        250-iteration refresh cadence so long runs can visit the full library.
        A reload may reset the simulator, so seal the old replay segment and
        fetch fresh observations before the next action.
        """
        frequency = int(self.settings.get("motion_resample_frequency", 250))
        if frequency <= 0 or iteration == 0 or iteration % frequency:
            return False
        if not hasattr(self.env, "resample_motion"):
            return False
        self.replay.mark_discontinuity()
        if self.reference_bank is not None:
            self.reference_bank.begin_generation(self.replay.reference_indices())
        self.env.resample_motion()
        self.obs = self.env.reset_all()
        self.actor_obs = self._seed_actor_history(self.obs["actor_obs"], self.obs.get("wam"))
        self.previous_world_action.zero_()
        return True

    def save_checkpoint(self, path):
        path = Path(path)
        self.distributed.check_consistent(str(path.resolve()), "checkpoint destination")
        runtime = {
            "rank": self.distributed.rank,
            "world_size": self.distributed.world_size,
            "iteration": self.iteration,
            "torch_rng": torch.get_rng_state(),
        }
        if self.device.type == "cuda":
            runtime["cuda_rng_local"] = torch.cuda.get_rng_state(self.device)
        if self.settings.get("save_replay", False):
            runtime["wam_replay"] = self.replay.state_dict()
            runtime["wam_replay_env_ids"] = self.replay_env_ids.cpu()
            if self.reference_bank is not None:
                runtime["wam_reference_bank"] = self.reference_bank.state_dict()

        # Shared weights/optimizers have one writer. Replay and RNG are local
        # to each rank and stored in generation-specific sidecars, so a failed
        # save cannot pair an old shared checkpoint with a new rank state.
        previous_runtime = (
            self._last_runtime_path if self._last_checkpoint_path == path.resolve() else None
        )
        runtime_path = None
        runtime_names = None
        if self.distributed.world_size > 1:
            generation = [f"{self.iteration}-{time.time_ns()}" if self.distributed.is_main else None]
            dist.broadcast_object_list(generation, src=0)
            runtime_names = [
                f"{path.stem}.{generation[0]}.rank{rank}{path.suffix}"
                for rank in range(self.distributed.world_size)
            ]
            runtime_path = path.parent / runtime_names[self.distributed.rank]
            error = None
            try:
                self._atomic_save(runtime, runtime_path)
            except Exception as exc:
                error = exc
            if not self.distributed.all_finite(error is None):
                raise RuntimeError("A rank failed to write its WAM checkpoint sidecar") from error

        error = None
        if self.distributed.is_main:
            try:
                payload = self._checkpoint_payload()
                payload["wam_world_size"] = self.distributed.world_size
                if runtime_names is None:
                    payload.update(runtime)
                else:
                    payload["wam_runtime_files"] = runtime_names
                self._atomic_save(payload, path)
            except Exception as exc:
                error = exc
        if not self.distributed.all_finite(error is None):
            raise RuntimeError("Rank zero failed to publish the WAM checkpoint") from error
        self._last_runtime_path = runtime_path
        self._last_checkpoint_path = path.resolve()
        # Keep only the rank's previous successful sidecar; the shared manifest
        # now points at the new complete generation.
        if previous_runtime is not None and previous_runtime != runtime_path:
            try:
                previous_runtime.unlink(missing_ok=True)
            except OSError:
                pass

    def _checkpoint_payload(self):
        payload = {
            "policy_state_dict": self.policy.state_dict(),
            "world_model_state_dict": self.world_model.state_dict(),
            "wam_critic_state_dict": self.critic.state_dict(),
            "wam_actor_optimizer": self.actor_optimizer.state_dict(),
            "wam_world_optimizer": self.world_optimizer.state_dict(),
            "wam_critic_optimizer": self.critic_optimizer.state_dict(),
            "wam_critic_norm": self.critic_norm.state_dict(),
            "wam_iteration": self.iteration, "wam_world_updates": self.world_updates,
            "wam_layout": {
                "state": self.layout.state_specs,
                "goal": self.layout.goal_specs,
                "context": self.context_keys,
                "reference": self.reference_keys,
                "critic": self.critic_keys,
            },
            "wam_state_specs": self.layout.state_specs,
            "wam_goal_specs": self.layout.goal_specs,
            "wam_settings": self.settings,
            "wam_tokenizer_specs": self.layout.tokenizer_specs,
            "wam_physical_schema_version": PHYSICAL_SCHEMA_VERSION if self.physical_wam else None,
            "wam_critic_spec": self.critic_spec if self.physical_wam else None,
            "wam_context_schema": self.context_schema if self.physical_wam else None,
            "wam_numerical_spec": self.numerical_spec,
            "wam_robot_spec": getattr(self, "robot_spec", None),
            "wam_replay_spec": {"compact_reference": self.compact_reference},
        }
        return payload

    @staticmethod
    def _atomic_save(payload, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path, *, resume=False):
        # Replay and immutable motion banks are CPU data, even when inference
        # runs on CUDA. Never stage the full checkpoint in scarce GPU memory.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint_robot(checkpoint, getattr(self, "robot_spec", None))
        policy_state = checkpoint.get("policy_state_dict", checkpoint.get("actor_model_state_dict"))
        if policy_state is None:
            model_state = checkpoint.get("model_state_dict", {})
            policy_state = {
                key.removeprefix("policy."): value
                for key, value in model_state.items() if key.startswith("policy.")
            }
        if not policy_state:
            raise ValueError("Checkpoint contains no compatible FSQ policy weights")
        self.policy.load_state_dict(policy_state, strict=True)
        if resume:
            if self.physical_wam and checkpoint.get("wam_physical_schema_version") != PHYSICAL_SCHEMA_VERSION:
                raise ValueError(
                    "Physical WAM state/critic/replay schema changed (independent imagination and context); "
                    "use resume=false to load policy only and refit WAM/critic"
                )
            if self.physical_wam and checkpoint.get("wam_critic_spec") != self.critic_spec:
                raise ValueError("Cannot resume WAM with different critic future reference settings")
            if self.physical_wam and checkpoint.get("wam_context_schema") != self.context_schema:
                raise ValueError("Cannot resume WAM with a different physical context schema")
            saved_numerical_spec = deepcopy(checkpoint.get("wam_numerical_spec"))
            if saved_numerical_spec is not None:
                # Missing keys identify the pre-fix algorithm. Never silently
                # resume its world model under the new target-frame objective.
                saved_numerical_spec.setdefault("error_frame", "predicted")
                saved_numerical_spec.setdefault("loss_reduction", "mean_valid")
            if self.physical_wam and saved_numerical_spec != self.numerical_spec:
                raise ValueError(
                    "Cannot resume WAM with different normalization, noise or actor objective settings "
                    "(including error_frame/loss_reduction). Pre-fix checkpoints use "
                    "world_model.cfg.error_frame=predicted and loss_reduction=mean_valid; "
                    "otherwise use resume=false to load only the policy."
                )
            if checkpoint.get("wam_world_size", 1) != self.distributed.world_size:
                raise ValueError(
                    "WAM resume requires the saved process count; use resume=false to load policy only"
                )
            expected_layout = {
                "state": self.layout.state_specs,
                "goal": self.layout.goal_specs,
                "context": self.context_keys,
                "reference": self.reference_keys,
                "critic": self.critic_keys,
            }
            if checkpoint.get("wam_layout") != expected_layout:
                raise ValueError("Cannot resume WAM with a different observation layout")
            if checkpoint.get("wam_tokenizer_specs") != self.layout.tokenizer_specs:
                raise ValueError("Cannot resume WAM with a different tokenizer schema or goal layout")
            self.world_model.load_state_dict(checkpoint["world_model_state_dict"])
            self.critic.load_state_dict(checkpoint["wam_critic_state_dict"])
            self.actor_optimizer.load_state_dict(checkpoint["wam_actor_optimizer"])
            self.world_optimizer.load_state_dict(checkpoint["wam_world_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["wam_critic_optimizer"])
            if "wam_critic_norm" in checkpoint:
                self.critic_norm.load_state_dict(checkpoint["wam_critic_norm"])
            self.iteration = checkpoint["wam_iteration"]
            self.world_updates = checkpoint["wam_world_updates"]
            runtime = checkpoint
            if "wam_runtime_files" in checkpoint:
                runtime_files = checkpoint["wam_runtime_files"]
                if len(runtime_files) != self.distributed.world_size:
                    raise ValueError("Incomplete WAM rank-state manifest")
                runtime_path = Path(path).parent / runtime_files[self.distributed.rank]
                runtime = torch.load(runtime_path, map_location="cpu", weights_only=False)
                if runtime["rank"] != self.distributed.rank or (
                    runtime["world_size"] != self.distributed.world_size
                    or runtime["iteration"] != self.iteration
                ):
                    raise ValueError("WAM rank checkpoint does not match the shared checkpoint")
                self._last_runtime_path = runtime_path
                self._last_checkpoint_path = Path(path).resolve()
            if "wam_replay" in runtime:
                self.replay.load_state_dict(runtime["wam_replay"])
                # A legacy full replay remains valid; its stored goal schema
                # determines how seeds are read until the next fresh run.
                self.compact_reference = bool(
                    checkpoint.get("wam_replay_spec", {}).get("compact_reference", False)
                )
                if self.reference_bank is not None:
                    if "wam_reference_bank" not in runtime:
                        raise ValueError("WAM replay checkpoint is missing its immutable motion references")
                    self.reference_bank.load_state_dict(runtime["wam_reference_bank"])
                self.replay.mark_discontinuity()
                ids = runtime["wam_replay_env_ids"].to(self.device)
                if ids.max() >= self.env.num_envs:
                    raise ValueError("Restored replay requires the original simulator environment count")
                self.replay_env_ids = ids
            else:
                # Refill the replay on restart before using imagination again.
                self.world_updates = 0
            torch.set_rng_state(runtime["torch_rng"].cpu())
            if self.device.type == "cuda":
                if "cuda_rng_local" in runtime:
                    torch.cuda.set_rng_state(runtime["cuda_rng_local"].cpu(), self.device)
                elif "cuda_rng" in runtime:
                    # Compatibility with the previous single-process format.
                    torch.cuda.set_rng_state(runtime["cuda_rng"][0].cpu(), self.device)
