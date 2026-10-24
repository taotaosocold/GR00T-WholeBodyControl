"""Opt-in Dyna training of the existing FSQ actor through observation dynamics.

The learned state is the selected *original* observation groups, including
tokenizer goals, so imagined actions run the real multimodal encoder -> FSQ ->
decoder. This is an observation-space baseline, not SuperMimic's robot-specific
71-D rigid-body model. It needs validation in simulation before deployment.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from gear_sonic.trl.modules.wam import StateSpaceWorldModel, WAMConfig
from gear_sonic.trl.world_model.distributed import DistributedContext
from gear_sonic.trl.world_model.sequence_replay import SequenceReplayBuffer


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
    """Reversible packing of the exact observation dictionary used by FSQ."""

    EXOGENOUS_FEATURES = {
        "command_multi_future_nonflat", "command_multi_future_lower_body",
        "command_z", "command_z_multi_future_nonflat", "vr_3point_local_target",
        "vr_3point_local_orn_target", "smpl_joints_multi_future_local_nonflat",
        "smpl_joints_multi_future_nonflat", "joint_pos_multi_future_wrist_for_smpl",
    }

    def __init__(self, example, keys, tokenizer_specs=(), exogenous_features=None):
        self.specs = []
        offset = 0
        for key in keys:
            if key not in example:
                raise ValueError(f"WAM observation group {key!r} is missing")
            value = example[key]
            if value.ndim != 2:
                raise ValueError(f"WAM expects flat observation groups, got {key}: {value.shape}")
            self.specs.append((key, offset, offset + value.shape[-1]))
            offset += value.shape[-1]
        self.dim = offset
        self.fixed_indices = []
        self.exogenous_indices = []
        self.tokenizer_specs = list(tokenizer_specs)
        exogenous_features = set(
            self.EXOGENOUS_FEATURES if exogenous_features is None else exogenous_features
        )
        tokenizer_offset = next((start for key, start, _ in self.specs if key == "tokenizer"), None)
        # Encoder routing and frame counts are categorical/exogenous. Copy
        # them from the rollout seed rather than hallucinating fractional IDs.
        if tokenizer_offset is not None:
            for name, start, end, _ in tokenizer_specs:
                if name in {"encoder_index", "command_num_frames"}:
                    self.fixed_indices.extend(range(tokenizer_offset + start, tokenizer_offset + end))
                elif name in exogenous_features:
                    self.exogenous_indices.extend(
                        range(tokenizer_offset + start, tokenizer_offset + end)
                    )

    def pack(self, obs):
        return torch.cat([obs[key] for key, _, _ in self.specs], dim=-1)

    def unpack(self, state):
        return {key: state[..., start:end].unsqueeze(1) for key, start, end in self.specs}

    def project(self, state, seed, reference=None):
        if not self.fixed_indices and not self.exogenous_indices:
            return state
        state = state.clone()
        state[..., self.fixed_indices] = seed[..., self.fixed_indices]
        if reference is not None:
            # Reference poses are exogenous commands, not consequences of the
            # robot's actions. Replaying their known timeline prevents a policy
            # from exploiting dynamics error to invent an easier target motion.
            state[..., self.exogenous_indices] = reference[..., self.exogenous_indices]
        return state


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
        checkpoint=None, resume=False, world_model_config=None, settings=None, **kwargs,
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
        for name in ["window", "horizon", "batch_size", "world_updates", "critic_epochs",
                     "replay_steps", "replay_envs", "collect_steps", "actor_updates"]:
            if int(self.settings.get(name, 1)) < 1:
                raise ValueError(f"WAM setting {name} must be positive")
        if int(self.settings.get("replay_steps", 256)) < int(self.settings.get("window", 8)):
            raise ValueError("replay_steps must be at least the fitting window")
        if int(self.settings.get("replay_steps", 256)) < int(self.settings.get("horizon", 32)):
            raise ValueError("replay_steps must be at least the imagination horizon")
        # The default Hydra timestamp can be evaluated in different seconds on
        # different ranks. All rank sidecars must use rank zero's actual path.
        shared_path = [str(log_dir or "logs_rl/wam")]
        if self.distributed.world_size > 1:
            dist.broadcast_object_list(shared_path, src=0)
        self.log_dir = Path(shared_path[0])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.iteration = 0
        self.world_updates = 0
        self.obs = env.reset_all()
        backbone = getattr(model, "actor_module", None)
        keys = self.settings.get("observation_keys", ["actor_obs", "tokenizer"])
        self.layout = ObservationLayout(
            self.obs, keys, getattr(backbone, "tokenizer_obs_specs", ()),
            self.settings.get("exogenous_tokenizer_features"),
        )
        self.distributed.check_consistent(
            (self.layout.specs, self.layout.tokenizer_specs, self.settings,
             args.num_total_batches, config.get("num_learning_iterations"),
             config.get("gamma", 0.99), config.get("lam", 0.95)),
            "WAM layout/settings",
        )
        cfg = dict(world_model_config or {})
        if cfg and not str(cfg.get("_target_", "")).endswith("StateSpaceWorldModel"):
            raise ValueError("trainer=wam requires world_model=wam_dynamics")
        cfg = dict(cfg.get("cfg", {}))
        cfg.pop("_target_", None)
        cfg["state_dim"] = self.layout.dim
        cfg["action_dim"] = int(env.config.robot.actions_dim)
        if cfg.get("context_dim", 0):
            raise ValueError("Explicit simulation context is not yet wired in WAMTrainer")
        if not cfg.get("learn_reward", True) or not cfg.get("learn_termination", True):
            raise ValueError("WAMTrainer requires both reward and termination heads")
        self.distributed.check_consistent(cfg, "WAM dynamics settings")
        self.world_model = StateSpaceWorldModel(WAMConfig(**cfg)).to(self.device)
        self.world_step = self.world_model.forward
        if self.settings.get("compile", False):
            self.world_step = torch.compile(self.world_model.forward, dynamic=False)
        self.critic = nn.Sequential(
            nn.Linear(self.layout.dim, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1),
        ).to(self.device)
        lr = float(self.settings.get("learning_rate", 2e-4))
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=lr)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.actor_optimizer = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.95))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.replay = SequenceReplayBuffer(int(self.settings.get("replay_steps", 256)))
        tracked = min(env.num_envs, int(self.settings.get("replay_envs", 256)))
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
        self.distributed.broadcast_modules(self.policy, self.world_model, self.critic)

    def _normalizers(self):
        return [module for module in (
            self.world_model.state_norm, self.world_model.action_norm,
            self.world_model.context_norm, self.world_model.delta_norm,
        ) if module is not None]

    def _action(self, state, *, auxiliary=False):
        obs = self.layout.unpack(state)
        result = self.policy.forward(
            obs, is_training=auxiliary, cache_encoded_tensors=False
        )
        action = result[:, -1]
        limit = self.env.config.get("action_clip_value", None)
        if limit is not None and limit > 0:
            action = action.clamp(-limit, limit)
        return action

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
        for _ in range(steps):
            state = self.layout.pack(self.obs)
            action = self._action(state)
            noise = float(self.settings.get("action_noise", 0.05))
            executed = action + noise * torch.randn_like(action)
            limit = self.env.config.get("action_clip_value", None)
            if limit is not None and limit > 0:
                executed = executed.clamp(-limit, limit)
            next_obs, rewards, dones, infos = self.env.step({"actions": executed})
            next_state = self.layout.pack(next_obs)
            reward = rewards.reshape(state.shape[0], -1).sum(-1)
            timeout = infos.get("time_outs", torch.zeros_like(dones)).bool()
            terminated = infos.get("terminated", dones.bool() & ~timeout).bool()
            indices = self.replay_env_ids
            self.replay.add(
                state[indices], executed[indices], reward[indices], next_state[indices],
                dones[indices], terminated[indices],
            )
            valid = ~dones[indices].bool()
            if valid.any():
                self.world_model.update_normalizers(
                    state[indices][valid], executed[indices][valid], next_state[indices][valid]
                )
            self.obs = next_obs
            total += reward.mean().item()
        self.distributed.sync_running_norms(norms, before)
        return total / max(steps, 1)

    def fit_world_model(self):
        window = int(self.settings.get("window", 8))
        batch_size = int(self.settings.get("batch_size", 256))
        metrics = {}
        for _ in range(int(self.settings.get("world_updates", 10))):
            batch = self.replay.sample(batch_size, window, device=self.device)
            losses = self.world_model.transition_loss(
                batch["states"], batch["actions"], dones=batch["terminated"],
                mask=batch["mask"], dynamics_mask=batch["dynamics_mask"],
                rewards=batch["rewards"],
                state_projector=lambda state, step: self.layout.project(
                    state, batch["states"][:, 0], batch["states"][:, step + 1]
                ),
            )
            self.optimizer_step(
                losses["loss"], self.world_optimizer, self.world_model.parameters(),
                distributed=self.distributed,
            )
            self.world_updates += 1
            metrics = {f"world/{key}": float(value.detach()) for key, value in losses.items()}
        return metrics

    def improve_policy(self):
        horizon = int(self.settings.get("horizon", 32))
        batch = self.replay.sample(
            int(self.settings.get("batch_size", 256)), horizon, device=self.device
        )
        seed = batch["states"][:, 0].detach()
        gamma = float(self.config.get("gamma", 0.99))
        lam = float(self.config.get("lam", 0.95))
        self.policy.train()
        state = seed
        states, rewards, dones = [], [], []
        survival = torch.ones(seed.shape[0], device=self.device)
        objective = torch.zeros_like(survival)
        auxiliary_loss = seed.new_zeros(())
        with frozen(self.world_model), frozen(self.critic):
            for step in range(horizon):
                states.append(state)
                valid = batch["mask"][:, step]
                action = self._action(state, auxiliary=(step == 0))
                if step == 0 and getattr(self.policy, "has_aux_loss", False):
                    coefficients = self.policy.aux_loss_coef or {}
                    for name, loss in (self.policy.aux_losses or {}).items():
                        auxiliary_loss = auxiliary_loss + coefficients.get(name, 0.0) * loss
                prediction = self.world_step(state, action)
                reward = prediction["reward"]
                done = prediction["termination_logits"].sigmoid()
                objective = objective + gamma**step * survival * reward * valid
                survival = torch.where(valid, survival * (1.0 - done), survival)
                # At an auto-reset boundary the following reference belongs to
                # another clip. Hold the last valid reference for bootstrapping.
                next_reference = torch.where(
                    batch["dynamics_mask"][:, step, None],
                    batch["states"][:, step + 1], batch["states"][:, step],
                )
                predicted_state = self.layout.project(prediction["next_state"], seed, next_reference)
                state = torch.where(valid[:, None], predicted_state, state)
                rewards.append(reward)
                dones.append(done)
            bootstrap = self.critic(self.world_model.state_norm(state)).squeeze(-1)
            lengths = batch["mask"].sum(-1)
            objective = objective + gamma**lengths * survival * bootstrap
            loss = -objective.mean() / horizon
            loss = loss + float(self.settings.get("aux_loss_scale", 1.0)) * auxiliary_loss
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
            inputs = self.world_model.state_norm(states)
            values = self.critic(inputs).squeeze(-1)
            target = values[:, -1]
            targets = []
            for step in range(horizon - 1, -1, -1):
                next_target = rewards[:, step] + gamma * (1 - dones[:, step]) * (
                    (1 - lam) * values[:, step + 1] + lam * target
                )
                target = torch.where(batch["mask"][:, step], next_target, target)
                targets.append(target)
            targets = torch.stack(targets[::-1], dim=1)
        for _ in range(int(self.settings.get("critic_epochs", 5))):
            predicted = self.critic(inputs[:, :-1]).squeeze(-1)
            critic_errors = F.mse_loss(predicted, targets, reduction="none")
            critic_loss = (critic_errors * batch["mask"]).sum() / batch["mask"].sum().clamp_min(1)
            self.optimizer_step(
                critic_loss, self.critic_optimizer, self.critic.parameters(),
                distributed=self.distributed,
            )
        return {
            "actor/loss": float(loss.detach()), "actor/grad_norm": float(grad),
            "actor/imagined_return": float(objective.detach().mean()),
            "actor/survival": float(survival.detach().mean()),
            "critic/loss": float(critic_loss.detach()),
        }

    def train(self):
        iterations = self.args.num_total_batches
        if iterations is None:
            iterations = int(self.config.get("num_learning_iterations", 100000))
        window = int(self.settings.get("window", 8))
        for iteration in range(self.iteration, iterations):
            started = time.perf_counter()
            metrics = {"real/reward": self.collect(int(self.settings.get("collect_steps", 32)))}
            if self.distributed.all_finite(len(self.replay) >= window):
                metrics.update(self.fit_world_model())
            can_imagine = (
                self.world_updates >= int(self.settings.get("warmup_world_updates", 100)) and (
                    len(self.replay) >= int(self.settings.get("horizon", 32))
                )
            )
            if self.distributed.all_finite(can_imagine):
                for _ in range(int(self.settings.get("actor_updates", 1))):
                    metrics.update(self.improve_policy())
            if hasattr(self.env, "sync_and_compute_adaptive_sampling"):
                self.env.sync_and_compute_adaptive_sampling(
                    self.accelerator, sync_across_gpus=False
                )
            self.iteration = iteration + 1
            metrics.update(iteration=self.iteration, seconds=time.perf_counter() - started)
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
            "wam_iteration": self.iteration, "wam_world_updates": self.world_updates,
            "wam_layout": self.layout.specs, "wam_settings": self.settings,
            "wam_tokenizer_specs": self.layout.tokenizer_specs,
            "wam_exogenous_indices": self.layout.exogenous_indices,
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
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
            if checkpoint.get("wam_world_size", 1) != self.distributed.world_size:
                raise ValueError(
                    "WAM resume requires the saved process count; use resume=false to load policy only"
                )
            if checkpoint.get("wam_layout") != self.layout.specs:
                raise ValueError("Cannot resume WAM with a different observation layout")
            if checkpoint.get("wam_tokenizer_specs") != self.layout.tokenizer_specs or (
                checkpoint.get("wam_exogenous_indices") != self.layout.exogenous_indices
            ):
                raise ValueError("Cannot resume WAM with a different tokenizer schema or goal layout")
            self.world_model.load_state_dict(checkpoint["world_model_state_dict"])
            self.critic.load_state_dict(checkpoint["wam_critic_state_dict"])
            self.actor_optimizer.load_state_dict(checkpoint["wam_actor_optimizer"])
            self.world_optimizer.load_state_dict(checkpoint["wam_world_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["wam_critic_optimizer"])
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
