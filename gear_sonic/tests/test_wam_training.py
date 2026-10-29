"""Numerical training-loop checks with real FSQ and a small vector environment."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from omegaconf import OmegaConf
import torch
from torch import nn
from vector_quantize_pytorch import FSQ

from gear_sonic.trl.trainer.wam_trainer import ObservationLayout, WAMTrainer


class Config(dict):
    __getattr__ = dict.__getitem__


class FSQPolicy(nn.Module):
    """Three modality encoders and the actual library's straight-through FSQ."""

    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([nn.Linear(3, 2) for _ in range(3)])
        self.quantizer = FSQ([4, 4])
        self.decoder = nn.Linear(6, 2)
        self.has_aux_loss = False
        self.max_rollout_history = 1

    @property
    def actor_module(self):
        return SimpleNamespace(tokenizer_obs_specs=[("encoder_index", 9, 12, (3,))])

    def forward(self, obs, is_training=False, **kwargs):
        del is_training, kwargs
        tokens = obs["tokenizer"]
        latent = sum(
            encoder(tokens[..., index * 3:index * 3 + 3]) * tokens[..., 9 + index:10 + index]
            for index, encoder in enumerate(self.encoders)
        )
        quantized, _ = self.quantizer(latent)
        return self.decoder(torch.cat([obs["actor_obs"], quantized], dim=-1))


class VectorEnv:
    def __init__(self):
        self.num_envs = 6
        self.config = Config(robot=Config(actions_dim=2), action_clip_value=1.0)
        self.action_transform_module = None
        self.t = 0
        self.x = torch.randn(self.num_envs, 4)
        self.goals = torch.randn(self.num_envs, 9)
        self.routing = torch.eye(3)[torch.arange(self.num_envs) % 3]

    def _obs(self):
        return {"actor_obs": self.x.clone(), "tokenizer": torch.cat([self.goals, self.routing], -1)}

    def reset_all(self):
        return self._obs()

    def step(self, payload):
        action = payload["actions"]
        assert (action.abs() <= 1.0).all()
        self.x = 0.8 * self.x + 0.1 * torch.cat([action, action], dim=-1)
        self.goals = 0.99 * self.goals
        rewards = 1 - self.x.square().mean(-1)
        self.t += 1
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.t % 3 == 0:
            dones[0] = True
            self.x[0] = torch.randn(4) * 3  # Auto-reset must not train dynamics.
        return self._obs(), rewards, dones, {"time_outs": dones.clone()}


def make_original_fsq_actor():
    """Build the repository's actual actor/tokenizer from small Hydra configs."""
    from gear_sonic.trl.modules.actor_critic_modules import Actor

    dims = {"g1_goal": [3], "teleop_goal": [3], "smpl_goal": [3], "encoder_index": [3]}
    env_config = OmegaConf.create(
        {
            "robot": {"actions_dim": 2, "algo_obs_dim_dict": {"actor_obs": 4}},
            "obs": {
                "group_obs_dims": {"tokenizer": dims},
                "group_obs_names": {"tokenizer": list(dims)},
            },
        }
    )
    encoder_cfg = {
        name: {
            "inputs": [name + "_goal"],
            "params": {
                "_target_": "gear_sonic.trl.modules.base_module.BaseModule",
                "input_dim": None,
                "output_dim": None,
                "num_output_temporal_dims": 1,
                "module_config_dict": {
                    "layer_config": {"type": "MLP", "hidden_dims": [12], "activation": "SiLU"}
                },
            },
        }
        for name in ["g1", "teleop", "smpl"]
    }
    backbone = OmegaConf.create(
        {
            "_target_": "gear_sonic.trl.modules.universal_token_modules.UniversalTokenModule",
            "proprioception_features": ["actor_obs"],
            "num_fsq_levels": 2,
            "fsq_level_list": [4, 4],
            "max_num_tokens": 1,
            "num_future_frames": 1,
            "quantizer": {"_target_": "vector_quantize_pytorch.FSQ"},
            "encoder_sample_probs": {"g1": 1.0, "teleop": 1.0, "smpl": 1.0},
            "encoders": encoder_cfg,
            "decoders": {
                "g1_dyn": {
                    "inputs": ["token_flattened", "proprioception"],
                    "outputs": ["action"],
                    "has_temporal_dim": False,
                    "params": {
                        "_target_": "gear_sonic.trl.modules.base_module.BaseModule",
                        "input_dim": None,
                        "output_dim": None,
                        "module_config_dict": {
                            "layer_config": {
                                "type": "MLP", "hidden_dims": [12], "activation": "SiLU"
                            }
                        },
                    },
                }
            },
        }
    )
    return Actor(
        env_config,
        OmegaConf.create({"init_noise_std": 0.05}),
        backbone,
        input_obs_dict=True,
        has_aux_loss=True,
    )


class TrainingTests(unittest.TestCase):
    def test_physical_reference_survives_auto_reset_during_collection(self):
        class Library:
            def get_time_step_total(self, ids):
                return torch.full_like(ids, 8)

            def get_body_pos_w(self, ids, frames):
                pose = torch.zeros(len(ids), 1, 3)
                pose[:, 0, 2] = 1 + ids * 100 + frames * 0.1
                return pose

            def get_body_quat_w(self, ids, frames):
                pose = torch.zeros(len(ids), 1, 4)
                pose[..., 0] = 1
                return pose

            def get_body_lin_vel_w(self, ids, frames):
                return torch.zeros(len(ids), 1, 3)

            get_body_ang_vel_w = get_body_lin_vel_w

            def get_dof_pos(self, ids, frames):
                return frames[:, None].float().expand(-1, 29)

            get_dof_vel = get_dof_pos

        class ResetEnv(VectorEnv):
            def __init__(self):
                super().__init__()
                self.config.robot.actions_dim = 29
                self.command = SimpleNamespace(
                    motion_lib=Library(), motion_ids=torch.zeros(self.num_envs, dtype=torch.long),
                    motion_start_time_steps=torch.zeros(self.num_envs, dtype=torch.long),
                    time_steps=torch.zeros(self.num_envs, dtype=torch.long), motion_anchor_body_index=0,
                )
                self.command_manager = SimpleNamespace(get_term=lambda name: self.command)

            def _obs(self):
                obs = super()._obs()
                command, library = self.command, self.command.motion_lib
                frame = command.motion_start_time_steps + command.time_steps
                reference = torch.cat((
                    library.get_body_pos_w(command.motion_ids, frame)[:, 0],
                    library.get_body_quat_w(command.motion_ids, frame)[:, 0],
                    torch.zeros(self.num_envs, 6),
                    library.get_dof_pos(command.motion_ids, frame),
                    library.get_dof_vel(command.motion_ids, frame),
                ), dim=-1)
                state = torch.zeros(self.num_envs, 71)
                state[:, 2:4] = 1  # root height and identity quaternion
                obs.update(wam_state=state, wam_reference=reference)
                return obs

            def step(self, payload):
                self.command.time_steps += 1
                self.command.motion_ids[0] = 1
                self.command.time_steps[0] = 0  # Auto-reset to a completely different clip.
                done = torch.zeros(self.num_envs, dtype=torch.bool)
                done[0] = True
                return self._obs(), torch.ones(self.num_envs), done, {"terminated": done}

        with tempfile.TemporaryDirectory() as directory:
            policy = FSQPolicy()
            policy.decoder = nn.Linear(6, 29)
            trainer = WAMTrainer(
                args=SimpleNamespace(num_total_batches=1), config=Config(gamma=0.99, lam=0.95),
                env=ResetEnv(), model=policy, log_dir=directory,
                settings={
                    "state_observation_keys": ["wam_state"], "goal_observation_keys": ["tokenizer"],
                    "reference_observation_keys": ["wam_reference"], "align_real_termination": True,
                    "window": 1, "horizon": 1, "replay_steps": 2, "replay_envs": 6,
                },
                world_model_config={
                    "_target_": "gear_sonic.trl.modules.wam.StateSpaceWorldModel",
                    "cfg": {
                        "state_type": "physical", "hidden": [8],
                        "reward_mode": "analytic", "termination_mode": "analytic",
                    },
                },
            )
            expected = trainer._next_physical_reference().clone()
            trainer.collect(1)
            saved = trainer.replay.state_dict()["storage"]
            torch.testing.assert_close(saved["next_reference"][0], expected[trainer.replay_env_ids])
            self.assertGreater(trainer.obs["wam_reference"][0, 2].item(), 100)
            self.assertAlmostEqual(expected[0, 2].item(), 1.1, places=6)
            self.assertTrue(saved["terminated"].any())
            batch = trainer.replay.sample(64, 1, generator=torch.Generator().manual_seed(7))
            self.assertTrue(batch["terminated"].any())
            self.assertFalse(batch["dynamics_mask"][batch["terminated"]].any())

    def test_imagined_reference_holds_terminal_frame_after_replay_boundary(self):
        trainer = WAMTrainer.__new__(WAMTrainer)
        trainer.physical_wam = True
        trainer.device = torch.device("cpu")
        batch = {
            "references": torch.tensor([[[1.], [2.], [100.], [200.]], [[3.], [4.], [5.], [6.]]]),
            "mask": torch.tensor([[True, False, False], [True, True, True]]),
        }
        torch.testing.assert_close(
            trainer._imagination_references(batch),
            torch.tensor([[[1.], [2.], [2.], [2.]], [[3.], [4.], [5.], [6.]]]),
        )

    def make_trainer(self, directory, policy=None, checkpoint=None, resume=False):
        return WAMTrainer(
            args=SimpleNamespace(num_total_batches=1), config=Config(gamma=0.99, lam=0.95),
            env=VectorEnv(), model=policy or FSQPolicy(), log_dir=directory,
            settings={
                "window": 3, "horizon": 3, "batch_size": 24,
                "world_updates": 2, "critic_epochs": 2, "replay_steps": 16,
                "collect_steps": 6, "warmup_world_updates": 1, "save_replay": True,
                "exogenous_tokenizer_features": ["g1_goal", "teleop_goal", "smpl_goal"],
            },
            world_model_config={
                "_target_": "gear_sonic.trl.modules.wam.StateSpaceWorldModel",
                "cfg": {"hidden": [16, 16], "term_hidden": [8], "reward_hidden": [8]},
            }, checkpoint=checkpoint, resume=resume,
        )

    def test_goal_free_single_step_wam_keeps_fsq_on_actor_path(self):
        class SingleStepEnv(VectorEnv):
            def _obs(self):
                obs = super()._obs()
                obs["wam"] = self.x.clone()
                return obs

        with tempfile.TemporaryDirectory() as directory:
            trainer = WAMTrainer(
                args=SimpleNamespace(num_total_batches=1),
                config=Config(gamma=0.99, lam=0.95),
                env=SingleStepEnv(),
                model=FSQPolicy(),
                log_dir=directory,
                settings={
                    "state_observation_keys": ["wam"],
                    "goal_observation_keys": ["tokenizer"],
                    "window": 3,
                    "horizon": 3,
                    "batch_size": 4,
                    "world_updates": 1,
                    "critic_epochs": 1,
                    "replay_steps": 8,
                    "collect_steps": 4,
                    "warmup_world_updates": 1,
                },
                world_model_config={
                    "_target_": "gear_sonic.trl.modules.wam.StateSpaceWorldModel",
                    "cfg": {"hidden": [8, 8], "term_hidden": [4], "reward_hidden": [4]},
                },
            )
            self.assertEqual(trainer.world_model.goal_dim, 0)
            trainer.collect(4)
            trainer.fit_world_model()
            metrics = trainer.improve_policy()
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_flat_mode_can_explicitly_use_wrapped_critic_observations(self):
        class WrappedEnv(VectorEnv):
            def _obs(self):
                obs = super()._obs()
                obs.update(
                    wam_state=self.x.clone(),
                    wam=self.x.clone(),
                    wam_reference=self.goals[:, :3].clone(),
                    wam_context=torch.arange(33).float().expand(self.num_envs, -1) / 100,
                    critic_obs=torch.cat((self.x, self.goals[:, :2]), dim=-1),
                )
                return obs

        config_path = Path(__file__).resolve().parents[1] / "config/trainer/wam.yaml"
        settings = OmegaConf.to_container(OmegaConf.load(config_path).settings, resolve=True)
        self.assertEqual(settings["critic_observation_keys"], [])
        settings.update(
            window=3, horizon=3, batch_size=4, policy_batch_size=4,
            world_updates=1, critic_epochs=1, critic_minibatches=2, replay_steps=8,
            critic_observation_keys=["critic_obs"],
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = WAMTrainer(
                args=SimpleNamespace(num_total_batches=1),
                config=Config(gamma=0.99, lam=0.95), env=WrappedEnv(), model=FSQPolicy(),
                log_dir=directory, settings=settings,
                # Tiny flat dynamics isolate the default observation schema
                # from the simulator-specific physical state adapter.
                world_model_config={
                    "_target_": "gear_sonic.trl.modules.wam.StateSpaceWorldModel",
                    "cfg": {"hidden": [8, 8], "term_hidden": [4], "reward_hidden": [4]},
                },
            )
            self.assertNotIn("critic", trainer.obs)
            self.assertEqual(trainer.critic_keys, ["critic_obs"])
            self.assertEqual(trainer.context_dim, 33)
            expected_critic = trainer.obs["critic_obs"][trainer.replay_env_ids].clone()
            trainer.collect(4)
            storage = trainer.replay.state_dict()["storage"]
            torch.testing.assert_close(storage["critic_state"][0], expected_critic)
            batch = trainer.replay.sample(4, 3)
            self.assertEqual(batch["critic_states"].shape, (4, 4, 6))
            critic_input = trainer._critic_input_with_aux(
                batch["states"][:, 0], batch["references"][:, 0],
                batch["critic_states"][:, 0], batch["contexts"][:, 0],
            )
            start = trainer.layout.state_dim + trainer.reference_dim
            torch.testing.assert_close(
                critic_input[:, start:start + trainer.critic_feature_dim],
                batch["critic_states"][:, 0],
            )
            metrics = trainer.fit_world_model() | trainer.improve_policy()
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_physical_state_keeps_named_proprioceptive_history_blocks(self):
        # Physical state and proprioception have distinct dimensions. Policy
        # history is term-major, and its term order can differ from WAM's.
        trainer = WAMTrainer.__new__(WAMTrainer)
        trainer.physical_wam = True
        trainer.layout = SimpleNamespace(state_dim=71)
        trainer.env = SimpleNamespace(env=SimpleNamespace(observation_manager=SimpleNamespace(
            _group_obs_term_names={
                "policy": ["actions", "joint_pos"],
                "wam": ["joint_pos", "actions"],
            },
            _group_obs_term_dim={
                "policy": [(3, 2), (2, 3)],
                "wam": [(3,), (2,)],
            },
        )))
        actor_obs = torch.tensor([[10, 11, 20, 21, 30, 31, 40, 41, 42, 50, 51, 52]]).float()
        wam_obs = torch.tensor([[60, 61, 62, 70, 71]]).float()
        trainer.obs = {"actor_obs": actor_obs, "wam": wam_obs, "wam_state": torch.zeros(1, 71)}
        for policy_dims in ([(3, 2), (2, 3)], [(6,), (6,)]):
            with self.subTest(policy_dims=policy_dims):
                # Isaac Lab versions can report either [history, feature]
                # dimensions or a single flattened history dimension.
                trainer.env.env.observation_manager._group_obs_term_dim["policy"] = policy_dims
                trainer._actor_history_specs, trainer._wam_term_specs = trainer._resolve_history_specs()
                self.assertIsNotNone(trainer._actor_history_specs)
                self.assertIsNotNone(trainer._wam_term_specs)
                torch.testing.assert_close(
                    trainer._advance_actor_history(actor_obs, wam_obs),
                    torch.tensor([[20, 21, 30, 31, 70, 71, 50, 51, 52, 60, 61, 62]]).float(),
                )
                torch.testing.assert_close(
                    trainer._seed_actor_history(actor_obs, wam_obs),
                    torch.tensor([[70, 71, 70, 71, 70, 71, 60, 61, 62, 60, 61, 62]]).float(),
                )

    def test_collection_fit_imagination_and_checkpoint(self):
        torch.manual_seed(13)
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.collect(6)
            fit_metrics = trainer.fit_world_model()
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in fit_metrics.values()))
            before = [encoder.weight.detach().clone() for encoder in trainer.policy.encoders]
            model_before = {key: value.clone() for key, value in trainer.world_model.state_dict().items()}
            metrics = trainer.improve_policy()
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
            # All modalities get BPTT gradients through the real FSQ bottleneck.
            for old, encoder in zip(before, trainer.policy.encoders):
                self.assertFalse(torch.equal(old, encoder.weight))
            for key, value in trainer.world_model.state_dict().items():
                self.assertTrue(torch.equal(value, model_before[key]))
            checkpoint = Path(directory) / "resume.pt"
            trainer.save_checkpoint(checkpoint)
            restored = self.make_trainer(directory, checkpoint=checkpoint, resume=True)
            for key, value in trainer.policy.state_dict().items():
                self.assertTrue(torch.equal(value, restored.policy.state_dict()[key]))
            restored.collect(3)
            restored.fit_world_model()
            restored.improve_policy()

    def test_full_training_entry_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            output = io.StringIO()
            with redirect_stdout(output):
                trainer.train()
            self.assertEqual(trainer.iteration, 1)
            self.assertTrue((Path(directory) / "last.pt").exists())
            line = output.getvalue()
            for name in (
                "real/done_fraction", "real/terminated_fraction", "real/truncated_fraction",
                "real/action_rms", "real/reward_std", "actor/effective_horizon",
                "actor/aux_loss", "world/last_valid_fraction",
            ):
                self.assertIn(name + "=", line)

    def test_collection_diagnostics_measure_real_steps_and_filter_scalar_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.settings["action_noise"] = 0.0
            trainer._action = lambda state: state.new_full((state.shape[0], 2), 2.0)
            rewards_by_step = [
                torch.tensor([-6., -4., -6., -4., -6., -4.]),
                torch.tensor([-2., 0., -2., 0., -2., 0.]),
            ]
            calls = []

            def step(payload):
                index = len(calls)
                calls.append(True)
                self.assertTrue((payload["actions"] == 1.0).all())
                terminated = torch.tensor([True, index == 0, False, False, False, False])
                truncated = torch.tensor([False, index == 1, True, False, False, False])
                logs = {
                    "Episode_Reward/action_rate_l2": torch.tensor(-4. + 2. * index),
                    "Episode_Termination/foot": 0.25 + 0.5 * index,
                    "Metrics/motion/error": torch.tensor(0.5 + index),
                    "Metrics/motion/one_element": torch.tensor([8.]),
                    "Episode_Reward/occasionally_nan": torch.tensor(float("nan") if index == 0 else 2.),
                    "Episode_Reward/always_nan": torch.tensor(float("nan")),
                    "Metrics/motion/vector": torch.tensor([1., 2.]),
                    "Metrics/motion/empty": torch.tensor([]),
                    "Episode_Reward/string": "not a scalar",
                    "unrelated": torch.tensor(99.),
                }
                if index == 0:
                    logs["Episode_Reward/sparse"] = torch.tensor(7.)
                return trainer.env._obs(), rewards_by_step[index], terminated | truncated, {
                    "terminated": terminated, "time_outs": truncated, "to_log": logs,
                }

            trainer.env.step = step
            mean_reward = trainer.collect(2)
            metrics = trainer._last_collection_metrics
            self.assertEqual(mean_reward, -3.0)
            self.assertAlmostEqual(metrics["real/done_fraction"], 0.5)
            self.assertAlmostEqual(metrics["real/terminated_fraction"], 0.25)
            self.assertAlmostEqual(metrics["real/truncated_fraction"], 0.25)
            self.assertAlmostEqual(metrics["real/action_rms"], 1.0)
            self.assertAlmostEqual(metrics["real/action_abs_max"], 1.0)
            self.assertAlmostEqual(metrics["real/action_at_limit_fraction"], 1.0)
            expected_std = torch.cat(rewards_by_step).std(unbiased=False).item()
            self.assertAlmostEqual(metrics["real/reward_std"], expected_std, places=6)
            self.assertEqual(metrics["real/Episode_Reward/action_rate_l2"], -3.0)
            self.assertEqual(metrics["real/Episode_Termination/foot"], 0.5)
            self.assertEqual(metrics["real/Metrics/motion/error"], 1.0)
            self.assertEqual(metrics["real/Metrics/motion/one_element"], 8.0)
            self.assertEqual(metrics["real/Episode_Reward/sparse"], 7.0)
            self.assertEqual(metrics["real/Episode_Reward/occasionally_nan"], 2.0)
            for key in (
                "Episode_Reward/always_nan", "Metrics/motion/vector", "Metrics/motion/empty",
                "Episode_Reward/string", "unrelated",
            ):
                self.assertNotIn("real/" + key, metrics)

    def test_imagination_reports_effective_horizon_and_auxiliary_contribution(self):
        class AuxiliaryPolicy(FSQPolicy):
            def __init__(self):
                super().__init__()
                self.has_aux_loss = True
                self.aux_loss_coef = {"decoder_penalty": 0.25}
                self.aux_losses = {}

            def forward(self, obs, is_training=False, **kwargs):
                action = super().forward(obs, is_training=is_training, **kwargs)
                if is_training:
                    self.aux_losses = {"decoder_penalty": self.decoder.weight.square().mean()}
                return action

        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, policy=AuxiliaryPolicy())
            trainer.collect(6)
            trainer.fit_world_model()
            batch = trainer.replay.sample(24, 3, device=trainer.device)
            lengths = torch.tensor([1, 2, 3], device=trainer.device).repeat(8)
            batch["mask"] = torch.arange(3, device=trainer.device)[None, :] < lengths[:, None]
            batch["dynamics_mask"] = batch["mask"] & ~batch["dones"]
            trainer.replay.sample = lambda *args, **kwargs: batch
            expected_aux = 0.25 * trainer.policy.decoder.weight.detach().square().mean().item()
            metrics = trainer.improve_policy()
            self.assertEqual(metrics["actor/effective_horizon"], 2.0)
            self.assertAlmostEqual(metrics["actor/aux_loss"], expected_aux, places=7)
            self.assertAlmostEqual(
                metrics["actor/loss"], -metrics["actor/imagined_return"] / 3 + expected_aux,
                places=6,
            )

    def test_fit_and_imagination_hold_reference_at_reset_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.settings.update(window=1, horizon=1, world_updates=1, batch_size=3)
            trainer.layout.exogenous_indices = list(range(4, 13))
            seed = trainer.layout.pack(trainer.obs)[:3].detach().clone()
            seed[:, 4:13] = 10.0
            logged_next = seed.clone()
            logged_next[:, 4:13] = torch.tensor([20.0, 100.0, 200.0])[:, None]
            logged_next[:, 13:16] = seed[:, 13:16].roll(1, dims=-1)
            # The first transition advances normally. The next two contain
            # auto-reset observations after real termination and truncation.
            batch = {
                "states": torch.stack((seed, logged_next), dim=1),
                "actions": torch.zeros(3, 1, 2),
                "rewards": torch.zeros(3, 1),
                "dones": torch.tensor([[False], [True], [True]]),
                "terminated": torch.tensor([[False], [True], [False]]),
                "mask": torch.ones(3, 1, dtype=torch.bool),
                "dynamics_mask": torch.tensor([[True], [False], [False]]),
            }
            trainer.replay.sample = lambda *args, **kwargs: batch
            expected_reference = seed.clone()
            expected_reference[0, 4:13] = 20.0
            termination_inputs, predictions = [], []
            original_forward = trainer.world_model.forward
            hook = trainer.world_model.termination_head.register_forward_pre_hook(
                lambda module, inputs: termination_inputs.append(inputs[0].detach().clone())
            )

            def checked_forward(state, action, context=None, **kwargs):
                output = original_forward(state, action, context, **kwargs)
                predictions.append(output["next_state"].detach().clone())
                torch.testing.assert_close(
                    termination_inputs[-1], trainer.world_model.state_norm(output["next_state"]),
                )
                return output

            trainer.world_model.forward = checked_forward
            trainer.world_step = checked_forward
            try:
                trainer.fit_world_model()
                self.assertEqual(len(predictions), 1)
                trainer.improve_policy()
            finally:
                hook.remove()
            self.assertEqual(len(predictions), 2)
            self.assertEqual(len(termination_inputs), 2)
            for prediction, terminal_input in zip(predictions, termination_inputs):
                # Both phases must keep the seed route and hold old goals at
                # resets. Exposing 100/200 only during fitting leaks the label.
                torch.testing.assert_close(prediction[:, 4:16], expected_reference[:, 4:16])
                torch.testing.assert_close(
                    terminal_input[:, 4:16],
                    trainer.world_model.state_norm(expected_reference)[:, 4:16],
                )

    def test_actor_projects_reference_before_predicting_termination(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.collect(6)
            trainer.fit_world_model()
            # The toy tokenizer's first nine coordinates are three goal inputs;
            # its final three coordinates route those inputs to FSQ encoders.
            trainer.layout.exogenous_indices = list(range(4, 13))
            batch = trainer.replay.sample(24, 3, device=trainer.device)
            seed = batch["states"][:, 0].detach().clone()
            for step in range(1, 4):
                batch["states"][:, step, 4:13] = seed[:, 4:13] + 0.25 * step
                # Even if the next logged route differs, this rollout keeps
                # the seed modality rather than predicting a fractional route.
                batch["states"][:, step, 13:16] = seed[:, 13:16].roll(1, dims=-1)
            lengths = torch.tensor([1, 2, 3], device=trainer.device).repeat(8)
            batch["mask"] = torch.arange(3, device=trainer.device)[None, :] < lengths[:, None]
            batch["dones"] = torch.zeros_like(batch["mask"])
            batch["dones"][lengths < 3, lengths[lengths < 3] - 1] = True
            batch["terminated"] = batch["dones"].clone()
            batch["dynamics_mask"] = batch["mask"] & ~batch["dones"]
            trainer.replay.sample = lambda *args, **kwargs: batch
            # Make unprojected reference and routing predictions wrong so the
            # checks cannot accidentally pass with an identity dynamics model.
            with torch.no_grad():
                trainer.world_model.dynamics[-1].bias.fill_(0.37)
            calls = []

            def checked_world_step(state, action, **kwargs):
                self.assertIn("next_state_projector", kwargs)
                self.assertTrue(callable(kwargs["next_state_projector"]))
                step = len(calls)
                prediction = trainer.world_model(state, action, **kwargs)
                next_state = prediction["next_state"]
                next_reference = torch.where(
                    batch["dynamics_mask"][:, step, None],
                    batch["states"][:, step + 1], batch["states"][:, step],
                )
                self.assertTrue(torch.equal(next_state[:, 4:13], next_reference[:, 4:13]))
                self.assertTrue(torch.equal(next_state[:, 13:16], seed[:, 13:16]))
                expected_logits = trainer.world_model.termination_head(
                    trainer.world_model.state_norm(next_state)
                ).squeeze(-1)
                torch.testing.assert_close(prediction["termination_logits"], expected_logits)
                calls.append(step)
                return prediction

            trainer.world_step = checked_world_step
            trainer.improve_policy()
            self.assertEqual(calls, [0, 1, 2])

    def test_layout_copies_categorical_routing(self):
        env = VectorEnv()
        obs = env.reset_all()
        layout = ObservationLayout(obs, ["actor_obs", "tokenizer"], [("encoder_index", 9, 12, (3,))])
        seed = layout.pack(obs)
        prediction = torch.randn_like(seed, requires_grad=True)
        projected = layout.project(prediction, seed)
        self.assertTrue(torch.equal(projected[:, -3:], seed[:, -3:]))
        projected.sum().backward()
        self.assertEqual(prediction.grad[:, -3:].abs().sum(), 0)

    def test_reference_timeline_is_exogenous_to_actions(self):
        env = VectorEnv()
        obs = env.reset_all()
        layout = ObservationLayout(
            obs, ["actor_obs", "tokenizer"],
            [("command_z", 0, 1, (1,)), ("encoder_index", 9, 12, (3,))],
        )
        seed = layout.pack(obs)
        reference = seed.clone()
        reference[:, 4] += 3.0
        prediction = torch.randn_like(seed, requires_grad=True)
        projected = layout.project(prediction, seed, reference)
        self.assertTrue(torch.equal(projected[:, 4], reference[:, 4]))
        projected.sum().backward()
        self.assertEqual(prediction.grad[:, 4].abs().sum(), 0)
        self.assertGreater(prediction.grad[:, :4].abs().sum(), 0)

    def test_nonfinite_update_does_not_mutate_parameters(self):
        model = nn.Linear(2, 1)
        before = model.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters())
        loss = model(torch.ones(1, 2)).sum() * float("nan")
        with self.assertRaises(FloatingPointError):
            WAMTrainer.optimizer_step(loss, optimizer, model.parameters())
        self.assertTrue(torch.equal(before, model.weight))

    def test_motion_reload_refreshes_observations_and_seals_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.collect(2)
            calls = []

            def reload_motion():
                calls.append(True)
                trainer.env.x.fill_(7.0)
                trainer.env.goals.fill_(9.0)

            trainer.env.resample_motion = reload_motion
            trainer.settings["motion_resample_frequency"] = 2
            self.assertFalse(trainer._maybe_resample_motions(0))
            self.assertFalse(trainer._maybe_resample_motions(1))
            self.assertTrue(trainer._maybe_resample_motions(2))
            self.assertEqual(len(calls), 1)
            self.assertTrue((trainer.obs["actor_obs"] == 7.0).all())
            self.assertTrue((trainer.obs["tokenizer"][:, :9] == 9.0).all())
            self.assertTrue(trainer.replay.state_dict()["storage"]["done"][-1].all())
            self.assertFalse(trainer.replay.state_dict()["storage"]["terminated"][-1].any())
            trainer.settings["motion_resample_frequency"] = 0
            self.assertFalse(trainer._maybe_resample_motions(4))

    def test_original_actor_universal_token_fsq_end_to_end(self):
        torch.manual_seed(31)
        policy = make_original_fsq_actor()
        self.assertIsInstance(policy.actor_module.quantizer, FSQ)
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, policy=policy)
            seed = trainer.layout.pack(trainer.obs)
            output = policy.actor_module(trainer.layout.unpack(seed), return_dict=True)
            self.assertEqual(output["action_mean"].shape, (6, 1, 2))
            self.assertEqual(output["full_tokens"].shape, (6, 1, 1, 2))
            self.assertEqual(output["full_latent"].shape, (6, 1, 1, 2))
            # Establish that the tokenizer really routes all three modalities.
            self.assertEqual(set(output["encoded_tokens"]), {"g1", "teleop", "smpl"})
            trainer.collect(6)
            trainer.fit_world_model()
            old_encoder_weights = {
                name: next(encoder.parameters()).detach().clone()
                for name, encoder in policy.actor_module.encoders.items()
            }
            world_before = {
                name: value.clone() for name, value in trainer.world_model.state_dict().items()
            }
            metrics = trainer.improve_policy()
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
            for name, encoder in policy.actor_module.encoders.items():
                self.assertFalse(torch.equal(old_encoder_weights[name], next(encoder.parameters())))
            for name, value in trainer.world_model.state_dict().items():
                self.assertTrue(torch.equal(world_before[name], value))
            path = Path(directory) / "original_actor.pt"
            trainer.save_checkpoint(path)
            restored = self.make_trainer(
                directory, policy=make_original_fsq_actor(), checkpoint=path, resume=True
            )
            for name, value in policy.state_dict().items():
                self.assertTrue(torch.equal(value, restored.policy.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()
