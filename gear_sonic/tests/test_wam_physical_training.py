"""Physical WAM collection and imagination regressions on a tiny CPU scene."""

import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from vector_quantize_pytorch import FSQ

from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.trl.modules.wam import physical_state_parts
from gear_sonic.trl.trainer.wam_trainer import WAMTrainer
from gear_sonic.utils.wam_kinematics import G1_MUJOCO_JOINT_NAMES
from gear_sonic.utils.wam_observations import WAMGoalObservations
from gear_sonic.utils.wam_reference import WAMReferenceBank


class Config(dict):
    __getattr__ = dict.__getitem__


ISAAC_TO_MUJOCO = [joint_utils.G1_ISAACLab_ORDER.index(name) for name in G1_MUJOCO_JOINT_NAMES]
MUJOCO_TO_ISAAC = [G1_MUJOCO_JOINT_NAMES.index(name) for name in joint_utils.G1_ISAACLab_ORDER]
TOKENIZER_SPECS = (
    ("encoder_index", 0, 3, (3,)),
    ("motion_anchor_ori_b_mf_nonflat", 3, 15, (2, 6)),
    ("motion_anchor_ori_b", 15, 21, (6,)),
    ("smpl_root_ori_b_multi_future", 21, 33, (2, 6)),
    ("vr_3point_local_target", 33, 42, (3, 3)),
)
PROPRIO_NAMES = ["gravity_dir", "base_ang_vel", "joint_pos", "joint_vel", "actions"]
PROPRIO_DIMS = [(3,), (3,), (29,), (29,), (29,)]


def yaw_matrix(angle):
    c, s = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(-1, 3, 3)


class PhysicalFSQPolicy(nn.Module):
    """Small multimodal actor with the library's actual straight-through FSQ."""

    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList((nn.Linear(12, 2), nn.Linear(6, 2), nn.Linear(12, 2)))
        self.quantizer = FSQ([4, 4])
        self.decoder = nn.Linear(93 + 2, 29)
        self.has_aux_loss = False
        self.max_rollout_history = 1

    @property
    def actor_module(self):
        return SimpleNamespace(tokenizer_obs_specs=TOKENIZER_SPECS)

    def forward(self, obs, is_training=False, **kwargs):
        del is_training, kwargs
        goal = obs["tokenizer"]
        latent = sum(
            encoder(goal[..., start:end]) * goal[..., index:index + 1]
            for index, (encoder, start, end) in enumerate(zip(
                self.encoders, (3, 15, 21), (15, 21, 33)
            ))
        )
        token, _ = self.quantizer(latent)
        return 0.05 * self.decoder(torch.cat((obs["actor_obs"], token), -1))


class PhysicalVectorEnv:
    """Wrapper-shaped scene with per-environment calibration and auto-reset."""

    def __init__(self, *, terminate_first=False, terminal_capture=True):
        self.num_envs = 4
        self.config = Config(robot=Config(actions_dim=29), action_clip_value=20.0)
        self.action_transform_module = None
        self.terminate_first = terminate_first
        self.terminal_capture = terminal_capture
        self.last_terminal = None
        nominal = torch.linspace(-0.2, 0.2, 29)
        offsets = torch.arange(1, self.num_envs + 1)[:, None] * 0.003
        defaults = nominal[None] + offsets
        self.robot = SimpleNamespace(
            joint_names=joint_utils.G1_ISAACLab_ORDER, device=torch.device("cpu"),
            data=SimpleNamespace(default_joint_pos=defaults, default_joint_pos_nominal=nominal),
        )
        self.env = SimpleNamespace(
            scene={"robot": self.robot}, step_dt=0.02, common_step_counter=0,
            observation_manager=SimpleNamespace(
                _group_obs_term_names={"policy": PROPRIO_NAMES, "wam": PROPRIO_NAMES},
                _group_obs_term_dim={"policy": PROPRIO_DIMS, "wam": PROPRIO_DIMS},
            ),
        )
        self.state = torch.zeros(self.num_envs, 71)
        self.state[:, 2] = 0.8
        self.state[:, 13:42] = defaults[:, ISAAC_TO_MUJOCO] + 0.04
        self.yaw = torch.linspace(-0.3, 0.3, self.num_envs)
        self.state[:, 3] = (self.yaw / 2).cos()
        self.state[:, 6] = (self.yaw / 2).sin()
        self.reference_yaw = torch.linspace(0.6, 1.2, self.num_envs)
        self.actions = torch.zeros(self.num_envs, 29)
        self.context = torch.zeros(self.num_envs, 33)
        self.context[:, 0] = 1
        self.context[:, 1:30] = offsets.expand(-1, 29)[:, ISAAC_TO_MUJOCO]

    def reference(self):
        reference = self.state.clone()
        reference[:, 3:7] = 0
        reference[:, 3] = (self.reference_yaw / 2).cos()
        reference[:, 6] = (self.reference_yaw / 2).sin()
        return reference

    def future_reference(self, num_frames, *, frame_stride=1, next_step=False):
        reference = self.reference()[:, None].expand(-1, num_frames, -1).clone()
        steps = torch.arange(num_frames) * frame_stride + int(next_step)
        reference[..., 0] += steps[None] * 0.002
        reference[..., 7] = 0.1
        return reference.flatten(1)

    def goal(self):
        relative = yaw_matrix(self.reference_yaw - self.yaw)[..., :2].flatten(-2)
        routing = torch.eye(3)[torch.arange(self.num_envs) % 3]
        vr = torch.arange(self.num_envs * 9).reshape(self.num_envs, 9).float() / 100
        return torch.cat((routing, relative.repeat(1, 2), relative, relative.repeat(1, 2), vr), -1)

    def _obs(self):
        _, rotation, _, angular, q, qd = physical_state_parts(self.state)
        frame = torch.cat((
            -rotation[:, 2], (rotation.transpose(-1, -2) @ angular[..., None]).squeeze(-1),
            q[:, MUJOCO_TO_ISAAC] - self.robot.data.default_joint_pos,
            qd[:, MUJOCO_TO_ISAAC], self.actions,
        ), -1)
        return {
            "actor_obs": frame.clone(), "wam": frame,
            "wam_state": self.state.clone(), "wam_context": self.context.clone(),
            "wam_reference": self.reference(), "tokenizer": self.goal(),
            # Physical imagination must not consume this simulator-only value.
            "critic_obs": torch.full((self.num_envs, 7), 12345.0),
        }

    def reset_all(self):
        return self._obs()

    def step(self, payload):
        self.actions = payload["actions"].detach().clone()
        self.state[:, 13:42] += 0.01 * self.actions[:, ISAAC_TO_MUJOCO]
        self.state[:, 42:71] = self.actions[:, ISAAC_TO_MUJOCO]
        self.state[:, 0] += 0.01
        self.yaw += 0.03
        self.state[:, 3] = (self.yaw / 2).cos()
        self.state[:, 6] = (self.yaw / 2).sin()
        self.env.common_step_counter += 1
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.terminate_first and self.env.common_step_counter == 1:
            done[0] = True
            self.state[0, 2] = 0.3
        self.last_terminal = self.state.clone()
        if self.terminal_capture and getattr(self.env, "_wam_capture_terminal_state", False):
            self.env._wam_terminal_state = self.last_terminal.clone()
            self.env._wam_terminal_state_step = self.env.common_step_counter
            self.env._wam_terminal_state_command = "motion"
        if done.any():
            self.state[done, 0] = 100.0
            self.state[done, 2] = 1.8
            self.yaw[done] = 2.0
            self.state[done, 3] = math.cos(1.0)
            self.state[done, 6] = math.sin(1.0)
        return self._obs(), torch.ones(self.num_envs), done, {"terminated": done}


class PhysicalTrainingTests(unittest.TestCase):
    def test_compact_references_preserve_sampled_rollouts_and_actor_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            trainers = []
            for compact in (False, True):
                trainer = self.make_trainer(Path(directory) / str(compact), horizon=3,
                                            save_replay=True, compact_replay=compact)
                trainer.independent_imagination = True
                trainer.compact_reference = compact
                bank = WAMReferenceBank(TOKENIZER_SPECS)
                frames = 16
                reference = trainer.env.reference()[:1].expand(frames, -1).clone()
                reference[:, 0] = torch.arange(frames) * 0.01
                goal = trainer.goal_observations.to_world(
                    trainer.obs["tokenizer"][:1], trainer.obs["wam_state"][:1],
                ).expand(frames, -1).clone()
                key = bank.register(reference, goal)
                bank.capture = lambda command, manager, trainer=trainer, key=key: torch.tensor(
                    [[key, trainer.env.env.common_step_counter]] * trainer.env.num_envs,
                    dtype=torch.long,
                )
                trainer.reference_bank = bank
                trainer.env.env.command_manager = SimpleNamespace(get_term=lambda name: None)
                torch.manual_seed(155)
                trainer.collect(5)
                trainers.append(trainer)
            full, compact = trainers
            self.assertEqual(full.replay._storage["goal"].shape[-1], 42)
            self.assertEqual(compact.replay._storage["goal"].shape[-1], 3)
            self.assertNotIn("reference", compact.replay._storage)
            torch.manual_seed(500)
            full_batch = full._sample_policy_batch(3)
            torch.manual_seed(500)
            compact_batch = compact._sample_policy_batch(3)
            for name in full_batch:
                torch.testing.assert_close(full_batch[name], compact_batch[name], rtol=0, atol=0)
            for trainer in trainers:
                torch.manual_seed(500)
                trainer.improve_policy()
            for actual, expected in zip(compact.policy.parameters(), full.policy.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            checkpoint = Path(directory) / "compact.pt"
            compact.save_checkpoint(checkpoint)
            restored = self.make_trainer(Path(directory) / "restored", horizon=3)
            restored.independent_imagination = True
            restored.reference_bank = WAMReferenceBank(TOKENIZER_SPECS)
            restored.load_checkpoint(checkpoint, resume=True)
            self.assertTrue(restored.compact_reference)
            torch.manual_seed(600)
            expected_batch = compact._sample_policy_batch(3)
            torch.manual_seed(600)
            restored_batch = restored._sample_policy_batch(3)
            for name in expected_batch:
                torch.testing.assert_close(restored_batch[name], expected_batch[name], rtol=0, atol=0)

    def make_trainer(self, directory, *, terminate_first=False, terminal_capture=True, **settings):
        torch.manual_seed(91)
        env = PhysicalVectorEnv(terminate_first=terminate_first, terminal_capture=terminal_capture)
        trainer = WAMTrainer(
            args=SimpleNamespace(num_total_batches=1), config=Config(gamma=0.99, lam=0.95),
            env=env, model=PhysicalFSQPolicy(), log_dir=Path(directory),
            settings={
                "state_observation_keys": ["wam_state"], "goal_observation_keys": ["tokenizer"],
                "context_observation_keys": ["wam_context"],
                "reference_observation_keys": ["wam_reference"],
                "critic_observation_keys": ["critic_obs"],
                "align_real_termination": True, "align_real_reward": True,
                "independent_imagination": False,
                "window": 2, "horizon": 2, "replay_steps": 8, "replay_envs": 4,
                "batch_size": 3, "policy_batch_size": 3, "world_updates": 1,
                "critic_epochs": 1, "critic_minibatches": 1, "action_noise": 0.0,
                "critic_future_frames": 3,
                **settings,
            },
            world_model_config={
                "_target_": "gear_sonic.trl.modules.wam.StateSpaceWorldModel",
                "cfg": {
                    "state_type": "physical", "hidden": [16, 16],
                    "learn_reward": False, "learn_termination": False,
                    "reward_mode": "analytic", "termination_mode": "analytic",
                },
            },
        )
        # The small scene has no motion library. Preserve the production
        # trainer's pre-step capture point while supplying this scene's target.
        trainer._next_physical_reference = lambda: env.reference().clone()
        trainer._physical_future_reference = lambda *, next_step=False: env.future_reference(
            trainer.critic_future_frames,
            frame_stride=trainer.critic_future_stride,
            next_step=next_step,
        )
        return trainer

    def test_independent_imagination_continues_past_behaviour_done_with_seed_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, horizon=4, policy_batch_size=2, terminate_first=True)
            trainer.collect(4)
            seed = trainer.replay.sample_starts(2)
            # A behaviour policy failure is irrelevant to the new rollout.
            seed["done"].fill_(True)
            seed["terminated"].fill_(True)
            bank = WAMReferenceBank(TOKENIZER_SPECS)
            reference = trainer.obs["wam_reference"][:1].expand(10, -1).clone()
            reference[:, 0] = torch.arange(10) * 0.01
            goal = trainer.goal_observations.to_world(
                trainer.obs["tokenizer"][:1], trainer.obs["wam_state"][:1],
            ).expand(10, -1).clone()
            # All three input modalities advance on the original reference.
            goal[:, 33] = torch.arange(10).float()
            key = bank.register(reference, goal)
            seed["reference_index"] = torch.tensor([[key, 0], [key, 1]])
            trainer.reference_bank = bank
            trainer.independent_imagination = True
            with patch.object(trainer.replay, "sample_starts", return_value=seed):
                batch = trainer._sample_policy_batch(4)
                self.assertTrue(batch["mask"].all())
                torch.testing.assert_close(batch["goals"][0, :, 33], torch.arange(5).float())
                torch.testing.assert_close(batch["contexts"], seed["context"][:, None].expand(-1, 4, -1))
                # The soft analytical head remains responsible for survival.
                with patch.object(trainer, "_analytic_termination", side_effect=lambda state, ref: state[:, 2] * 0):
                    metrics = trainer.improve_policy()
            self.assertEqual(metrics["actor/effective_horizon"], 4)
            self.assertAlmostEqual(metrics["actor/survival"], 1)
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_collection_saves_world_goals_and_imagination_uses_predicted_robot_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            adapter = WAMGoalObservations(TOKENIZER_SPECS)
            initial_state = trainer.obs["wam_state"].clone()
            initial_goal = trainer.obs["tokenizer"].clone()
            expected = adapter.to_world(initial_goal, initial_state)
            collected_goals = []
            original_action = trainer._action

            def collection_forward(state, goal=None, **kwargs):
                collected_goals.append(goal.detach().clone())
                return original_action(state, goal, **kwargs)

            with patch.object(trainer, "_action", side_effect=collection_forward):
                trainer.collect(4)
            torch.testing.assert_close(collected_goals[0], initial_goal)
            saved = trainer.replay.state_dict()["storage"]
            torch.testing.assert_close(saved["goal"][0], expected[trainer.replay_env_ids])
            self.assertFalse(torch.allclose(saved["goal"][0, :, 3:15], initial_goal[trainer.replay_env_ids, 3:15]))
            self.assertNotIn("critic_state", saved)
            self.assertEqual(saved["future_reference"].shape[-1], 3 * 71)
            self.assertEqual(trainer.world_model.goal_dim, 0)
            self.assertEqual(trainer.layout.goal_dim, 42)
            batch = trainer.replay.sample(3, 2)
            seen = []
            original = trainer._action

            def actor_forward(state, goal=None, **kwargs):
                seen.append((state.detach().clone(), goal.detach().clone()))
                return original(state, goal, **kwargs)

            with patch.object(trainer.replay, "sample", return_value=batch), patch.object(
                trainer, "_action", side_effect=actor_forward
            ):
                metrics = trainer.improve_policy()
            self.assertEqual(len(seen), 2)
            for step, (state, goal) in enumerate(seen):
                torch.testing.assert_close(goal, adapter.to_robot(batch["goals"][:, step], state))
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_physical_critic_uses_imagined_xy_yaw_and_ignores_raw_simulator_features(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            self.assertEqual(trainer.critic_keys, [])
            self.assertEqual(trainer.critic_feature_dim, 0)
            xy = torch.tensor([[0.4, -0.2]], requires_grad=True)
            yaw = torch.tensor([0.7], requires_grad=True)
            state = trainer.obs["wam_state"][:1].clone()
            state[:, :2] = xy
            state[:, 3:7] = torch.stack((
                (yaw / 2).cos(), torch.zeros_like(yaw), torch.zeros_like(yaw), (yaw / 2).sin()
            ), -1)
            reference = trainer.obs["wam_reference"][:1]
            context = trainer.obs["wam_context"][:1]
            features = trainer._critic_input_with_aux(
                state, reference, torch.full((1, 7), float("nan")), context,
                future_reference=trainer._physical_future_reference()[:1],
            )
            self.assertEqual(features.shape[-1], 71 + 29 + 42 + 3 + 6 + 33 + 3 * 73)
            self.assertTrue(torch.isfinite(features).all())
            # A fixed value head isolates input observability from random
            # initialization of the trainable critic network.
            weights = torch.linspace(0.1, 1.0, features.shape[-1])
            value = (features * weights).sum()
            xy_grad, yaw_grad = torch.autograd.grad(value, (xy, yaw))
            self.assertTrue((xy_grad.abs() > 1e-7).all())
            self.assertGreater(yaw_grad.abs().item(), 1e-7)

    def test_critic_distinguishes_future_motion_with_identical_current_state(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            state = trainer.obs["wam_state"][:1]
            reference = trainer.obs["wam_reference"][:1]
            context = trainer.obs["wam_context"][:1]
            future = trainer._physical_future_reference()[:1].reshape(1, 3, 71)
            different_future = future.clone()
            different_future[:, 1:, 13] += 0.3
            first = trainer._critic_input_with_aux(
                state, reference, context=context, future_reference=future,
            )
            second = trainer._critic_input_with_aux(
                state, reference, context=context, future_reference=different_future,
            )
            torch.testing.assert_close(first[:, :-3 * 73], second[:, :-3 * 73])
            self.assertFalse(torch.allclose(first[:, -3 * 73:], second[:, -3 * 73:]))
            # Reference information is a critic condition. It does not become
            # an extra FSQ modality or a learned dynamics input.
            self.assertEqual(trainer.world_model.goal_dim, 0)
            self.assertEqual(trainer.world_model.context_dim, 33)
            self.assertEqual(trainer.layout.goal_dim, 42)

    def test_tracking_features_are_recomputed_from_differentiable_imagined_state(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            state = trainer.obs["wam_state"][:1].clone()
            state[:, 16] += 0.2  # Left knee differs from the current target.
            state.requires_grad_(True)
            features = trainer._critic_input_with_aux(
                state, trainer.obs["wam_reference"][:1],
                context=trainer.obs["wam_context"][:1],
                future_reference=trainer._physical_future_reference()[:1],
            )
            # Invariant state71, joint error29, then14 body-position errors.
            body_error = features[:, 100:142]
            gradient = torch.autograd.grad(body_error.square().sum(), state)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient[:, 16].abs().item(), 1e-7)

    def test_imagined_proprioception_uses_each_replay_samples_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            # Samples are intentionally reordered and do not match env count.
            chosen = torch.tensor([3, 1, 3])
            state = trainer.obs["wam_state"][chosen]
            context = trainer.obs["wam_context"][chosen]
            with patch("torch.rand_like", side_effect=lambda value: torch.full_like(value, 0.5)):
                frame = trainer._physical_actor_frame(state, torch.zeros(3, 29), context=context)
            expected = state[:, 13:42][:, MUJOCO_TO_ISAAC] - (
                trainer.env.robot.data.default_joint_pos_nominal[None]
                + context[:, 1:30][:, MUJOCO_TO_ISAAC]
            )
            torch.testing.assert_close(frame[:, 6:35], expected)

    def test_terminal_pre_reset_state_is_replayed_and_supervises_world_model(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, terminate_first=True, window=1, horizon=1)
            expected_future = trainer._physical_future_reference(next_step=True).clone()
            trainer.collect(1)
            saved = trainer.replay.state_dict()["storage"]
            slot = int((trainer.replay_env_ids == 0).nonzero()[0])
            self.assertTrue(saved["done"][0, slot])
            self.assertTrue(saved["next_state_valid"][0, slot])
            torch.testing.assert_close(saved["next_state"][0, slot], trainer.env.last_terminal[0])
            torch.testing.assert_close(
                saved["next_future_reference"][0], expected_future[trainer.replay_env_ids],
            )
            self.assertLess(saved["next_future_reference"][0, slot, 0].item(), 1.0)
            self.assertEqual(trainer.obs["wam_state"][0, 0].item(), 100.0)
            batch = trainer.replay.sample(64, 1)
            self.assertTrue(batch["dones"].any())
            self.assertTrue(batch["dynamics_mask"][batch["dones"]].all())
            with patch.object(trainer.replay, "sample", return_value=batch), patch.object(
                trainer.world_model, "transition_loss", wraps=trainer.world_model.transition_loss
            ) as loss_call:
                metrics = trainer.fit_world_model()
            self.assertTrue(loss_call.call_args.kwargs["dynamics_mask"][batch["dones"]].all())
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_missing_terminal_cache_does_not_supervise_auto_reset_state(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(
                directory, terminate_first=True, terminal_capture=False, window=1, horizon=1
            )
            trainer.collect(1)
            saved = trainer.replay.state_dict()["storage"]
            slot = int((trainer.replay_env_ids == 0).nonzero()[0])
            self.assertFalse(saved["next_state_valid"][0, slot])
            batch = trainer.replay.sample(64, 1)
            self.assertTrue(batch["dones"].any())
            self.assertFalse(batch["dynamics_mask"][batch["dones"]].any())

    def test_imagination_holds_context_at_each_samples_last_valid_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            mask = torch.tensor([[True, True, False, False], [True, True, True, True], [True, False, False, False]])
            context = torch.arange(3 * 4 * 33).reshape(3, 4, 33).float()
            held = trainer._imagination_contexts({"mask": mask, "contexts": context})
            torch.testing.assert_close(held[0], context[0, torch.tensor([0, 1, 1, 1])])
            torch.testing.assert_close(held[1], context[1])
            torch.testing.assert_close(held[2], context[2, torch.tensor([0, 0, 0, 0])])

    def test_short_trajectories_bootstrap_old_episode_and_normalize_only_valid_states(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, horizon=4, policy_batch_size=2)
            trainer.collect(4)
            batch = trainer.replay.sample(2, 4)
            lengths = torch.tensor([1, 2])
            batch["mask"] = torch.arange(4)[None] < lengths[:, None]
            batch["dones"].zero_()
            batch["terminated"].zero_()
            batch["dones"][torch.arange(2), lengths - 1] = True
            # The next clip uses very different physics and reference poses.
            # Its rows must not leak into a short old-episode bootstrap or RMS.
            for row, length in enumerate(lengths.tolist()):
                batch["contexts"][row, length:] = 1000.0 * (row + 1)
                batch["references"][row, length + 1:, :3] = 10000.0 * (row + 1)
                future = batch["future_references"].reshape(2, 5, 3, 71)
                future[row, length + 1:, :, :3] = 20000.0 * (row + 1)
            expected_context = batch["contexts"][torch.arange(2), lengths - 1].clone()
            expected_reference = batch["references"][torch.arange(2), lengths].clone()
            expected_future = batch["future_references"][torch.arange(2), lengths].clone()
            old_count = trainer.critic_norm.count.item()
            old_mean = trainer.critic_norm.mean.clone()
            old_std = trainer.critic_norm.std().clone()
            with patch.object(trainer.replay, "sample", return_value=batch), patch.object(
                trainer, "_critic_input_with_aux", wraps=trainer._critic_input_with_aux
            ) as feature_call, patch.object(
                trainer.critic, "forward", wraps=trainer.critic.forward
            ) as critic_call, patch.object(
                trainer.critic_norm, "update", wraps=trainer.critic_norm.update
            ) as normalization_call:
                metrics = trainer.improve_policy()

            bootstrap_state, bootstrap_reference, _, bootstrap_context = feature_call.call_args_list[0].args
            torch.testing.assert_close(bootstrap_context, expected_context)
            torch.testing.assert_close(bootstrap_reference, expected_reference)
            bootstrap_future = feature_call.call_args_list[0].kwargs["future_reference"]
            torch.testing.assert_close(bootstrap_future.flatten(1), expected_future)
            expected_bootstrap = trainer._critic_input_with_aux(
                bootstrap_state, expected_reference, context=expected_context,
                future_reference=expected_future,
            )
            torch.testing.assert_close(
                critic_call.call_args_list[0].args[0], (expected_bootstrap - old_mean) / old_std
            )

            sequence_state, reference, critic_state, context = feature_call.call_args_list[1].args
            valid_states = torch.arange(5)[None] <= lengths[:, None]
            expected_valid = trainer._critic_input_with_aux(
                sequence_state, reference, critic_state, context,
                future_reference=feature_call.call_args_list[1].kwargs["future_reference"],
            )[valid_states]
            update_values = normalization_call.call_args.args[0]
            torch.testing.assert_close(update_values, expected_valid)
            self.assertEqual(update_values.shape[0], int((lengths + 1).sum()))
            self.assertEqual(trainer.critic_norm.count.item() - old_count, 5)
            torch.testing.assert_close(trainer.critic_norm.mean, expected_valid.mean(0))
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_physical_checkpoint_restores_training_and_replay_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, save_replay=True)
            trainer.collect(4)
            trainer.fit_world_model()
            trainer.improve_policy()
            trainer.iteration = 7
            trainer.world_updates = 11
            checkpoint = Path(directory) / "physical.pt"
            trainer.save_checkpoint(checkpoint)
            payload = torch.load(checkpoint, weights_only=False)
            self.assertEqual(payload["wam_physical_schema_version"], 4)
            self.assertEqual(payload["wam_layout"]["critic"], [])
            self.assertEqual(payload["wam_critic_spec"], trainer.critic_spec)
            dimension = 71 + 29 + 42 + 3 + 6 + 33 + 3 * 73
            self.assertEqual(payload["wam_critic_norm"]["mean"].numel(), dimension)
            self.assertEqual(next(iter(payload["wam_critic_state_dict"].values())).shape[-1], dimension)
            restored = self.make_trainer(Path(directory) / "restored", save_replay=True)
            restored.load_checkpoint(checkpoint, resume=True)
            for before, after in (
                (trainer.policy, restored.policy), (trainer.world_model, restored.world_model),
                (trainer.critic, restored.critic), (trainer.critic_norm, restored.critic_norm),
            ):
                self.assertEqual(set(before.state_dict()), set(after.state_dict()))
                for key, value in before.state_dict().items():
                    torch.testing.assert_close(after.state_dict()[key], value)
            self.assertEqual(restored.iteration, 7)
            self.assertEqual(restored.world_updates, 11)
            self.assertEqual(len(restored.actor_optimizer.state), len(trainer.actor_optimizer.state))
            self.assertEqual(len(restored.world_optimizer.state), len(trainer.world_optimizer.state))
            self.assertEqual(len(restored.critic_optimizer.state), len(trainer.critic_optimizer.state))
            self.assertEqual(len(restored.replay), len(trainer.replay))
            torch.testing.assert_close(restored.replay_env_ids, trainer.replay_env_ids)
            replay = restored.replay.state_dict()["storage"]
            torch.testing.assert_close(replay["goal"], payload["wam_replay"]["storage"]["goal"])
            torch.testing.assert_close(
                replay["future_reference"], payload["wam_replay"]["storage"]["future_reference"],
            )
            self.assertTrue(replay["done"][-1].all())  # Resumed physics starts a new segment.

    def test_resume_rejects_changed_future_spacing_with_same_critic_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            checkpoint = Path(directory) / "physical.pt"
            trainer.save_checkpoint(checkpoint)
            restored = self.make_trainer(Path(directory) / "changed", critic_future_dt=0.04)
            self.assertEqual(trainer.critic_input_dim, restored.critic_input_dim)
            with self.assertRaisesRegex(ValueError, "critic"):
                restored.load_checkpoint(checkpoint, resume=True)
            # A spacing change does not affect the deployment policy schema.
            restored.load_checkpoint(checkpoint, resume=False)
            for name, value in trainer.policy.state_dict().items():
                torch.testing.assert_close(restored.policy.state_dict()[name], value)

    def test_old_physical_checkpoint_requires_policy_only_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_trainer(directory)
            with torch.no_grad():
                for parameter in source.policy.parameters():
                    parameter.add_(0.25)
            payload = source._checkpoint_payload()
            payload["wam_physical_schema_version"] = 2
            payload["wam_layout"]["critic"] = ["critic_obs"]
            # Simulate incompatible old learned state so accidental full
            # loading also fails instead of silently passing this regression.
            payload["world_model_state_dict"] = {"old_dynamics": torch.ones(1)}
            payload["wam_critic_state_dict"] = {"old_critic": torch.ones(1)}
            checkpoint = Path(directory) / "old_physical.pt"
            torch.save(payload, checkpoint)
            restored = self.make_trainer(Path(directory) / "restored")
            world_before = {key: value.clone() for key, value in restored.world_model.state_dict().items()}
            critic_before = {key: value.clone() for key, value in restored.critic.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "schema changed"):
                restored.load_checkpoint(checkpoint, resume=True)
            restored.load_checkpoint(checkpoint, resume=False)
            for key, value in source.policy.state_dict().items():
                torch.testing.assert_close(restored.policy.state_dict()[key], value)
            for key, value in world_before.items():
                torch.testing.assert_close(restored.world_model.state_dict()[key], value)
            for key, value in critic_before.items():
                torch.testing.assert_close(restored.critic.state_dict()[key], value)
            self.assertEqual(restored.iteration, 0)
            self.assertEqual(restored.world_updates, 0)
            self.assertEqual(len(restored.replay), 0)


if __name__ == "__main__":
    unittest.main()
