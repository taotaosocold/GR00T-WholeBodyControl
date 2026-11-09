"""Calibration-only context and action contracts against SuperMimic semantics."""

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.tests.test_wam_context import asset_fixture
from gear_sonic.tests import test_wam_physical_training as fixtures
from gear_sonic.trl.modules.wam import WAMConfig, StateSpaceWorldModel
from gear_sonic.trl.trainer import wam_trainer
from gear_sonic.utils.wam_context import (
    normalize_supermimic_context, supermimic_context, supermimic_context_schema,
)
from gear_sonic.utils.wam_noise import WAM_OBSERVATION_NOISE


class SuperMimicContextTests(unittest.TestCase):
    def test_snapshot_refreshes_named_offsets_and_com_without_complete_getters(self):
        for robot_type in ("g1", "n3"):
            with self.subTest(robot_type=robot_type):
                asset, offsets = asset_fixture()
                if robot_type == "n3":
                    asset.joint_names = list(reversed(joint_utils.get_robot_joint_names("n3", order="mujoco")))
                    asset.body_names = ["base_link", "waist_yaw_link"]
                asset.root_physx_view.get_masses = None
                asset.root_physx_view.get_inertias = None
                asset._physics_sim_view = None
                asset.root_physx_view.materials[0, :, :2] = 0.4
                asset.root_physx_view.materials[1, :, :2] = 1.1
                legacy = torch.full((2, 33), 999.0)
                before = supermimic_context(asset, legacy)
                torch.testing.assert_close(before[:, 0], torch.tensor([0.4, 1.1]))
                torch.testing.assert_close(before[:, 1:30], offsets.flip(-1))
                torch.testing.assert_close(before[:, 30:33], asset.root_physx_view.coms[:, 1, :3])
                asset.data.default_joint_pos += 0.002
                asset.root_physx_view.coms[:, 1, :3] += 0.01
                after = supermimic_context(asset, legacy)
                torch.testing.assert_close(after[:, 1:30], before[:, 1:30] + 0.002)
                torch.testing.assert_close(after[:, 30:33], before[:, 30:33] + 0.01)
                schema = supermimic_context_schema(asset)
                self.assertEqual(schema["dim"], 33)
                self.assertEqual(schema["kind"], "supermimic_calibration")
                self.assertEqual(schema["prefix_joint_names"],
                                 list(joint_utils.get_robot_joint_names(robot_type, order="mujoco")))
                self.assertEqual(schema["torso_body_name"], joint_utils.get_robot_torso_name(robot_type))

    def test_observation_publishes_33d_schema_in_supermimic_mode(self):
        # Load this observation term without requiring the IsaacLab runtime.
        path = Path(__file__).resolve().parents[1] / "envs/manager_env/mdp/observations.py"
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "wam_context")
        namespace = {"torch": torch, "ManagerBasedEnv": object, "SceneEntityCfg": object}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        asset, _ = asset_fixture()
        asset.device = torch.device("cpu")

        class Scene(dict):
            num_envs = 2

        env = SimpleNamespace(scene=Scene(robot=asset))
        value = namespace["wam_context"](env, SimpleNamespace(name="robot"), context_mode="supermimic")
        self.assertEqual(value.shape, (2, 33))
        self.assertEqual(env._wam_context_schema, supermimic_context_schema(asset))
        with self.assertRaisesRegex(ValueError, "33-D"):
            namespace["wam_context"](env, SimpleNamespace(name="robot"),
                                     context_mode="supermimic", complete=True)

    def test_fixed_affine_transform_preserves_raw_context(self):
        context = torch.zeros(2, 3, 33)
        context[..., 0] = torch.tensor([0.3, 0.75, 1.2])
        context[..., 1:30] = 0.01
        context[..., 30:] = torch.tensor([0.025, -0.05, 0.05])
        before = context.clone()
        transformed = normalize_supermimic_context(context)
        torch.testing.assert_close(transformed[..., 0], torch.tensor([-1.0, 0.0, 1.0]).expand(2, -1))
        torch.testing.assert_close(transformed[..., 1:30], torch.ones(2, 3, 29))
        torch.testing.assert_close(transformed[..., 30:], torch.tensor([0.5, -1.0, 1.0]).expand(2, 3, -1))
        torch.testing.assert_close(context, before)
        with self.assertRaisesRegex(ValueError, "33"):
            normalize_supermimic_context(torch.zeros(2, 34))

    def test_world_model_updates_and_uses_statistics_after_fixed_affine_transform(self):
        context = torch.randn(4, 33) * 0.003
        context[:, 0] = torch.tensor([0.3, 0.6, 0.9, 1.2])
        state = torch.zeros(4, 71)
        state[:, 2], state[:, 3] = 0.8, 1.0
        action = torch.zeros(4, 29)
        model = StateSpaceWorldModel(WAMConfig(
            state_dim=71, action_dim=29, context_dim=33, context_mode="supermimic",
            state_type="physical", hidden=(8,), learn_reward=False, learn_termination=False,
        ))
        raw = context.clone()
        model.update_normalizers(state, action, state, context=context)
        fixed = normalize_supermimic_context(context)
        torch.testing.assert_close(model.context_norm.mean, fixed.mean(0))
        torch.testing.assert_close(model.context_norm.var, fixed.var(0, unbiased=False))
        expected = (fixed - fixed.mean(0)) / torch.sqrt(fixed.var(0, unbiased=False) + 1e-4)
        torch.testing.assert_close(model._model_input(state, action, context)[..., -33:], expected)
        torch.testing.assert_close(context, raw)
        self.assertEqual(model.context_norm.eps, 1e-2)
        self.assertEqual(model.context_norm.min_std, 0.0)
        with self.assertRaisesRegex(ValueError, "context_dim=33"):
            StateSpaceWorldModel(WAMConfig(state_dim=71, action_dim=29, context_dim=39,
                                         state_type="physical", context_mode="supermimic"))


class SuperMimicContextTrainingTests(unittest.TestCase):
    def make_trainer(self, directory, *, context_mode="supermimic", profile="supermimic", **settings):
        constructor = wam_trainer.WAMConfig

        def configured(**kwargs):
            return constructor(**(kwargs | {"context_mode": context_mode}))

        with patch.object(wam_trainer, "WAMConfig", side_effect=configured):
            return fixtures.PhysicalTrainingTests().make_trainer(
                directory, domain_randomization=profile,
                observation_noise={name: 0.0 for name in WAM_OBSERVATION_NOISE}, **settings,
            )

    def test_critic_uses_fixed_context_while_actor_and_replay_keep_raw_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            context = trainer.obs["wam_context"].clone()
            captured = []
            hook = trainer.critic_features.register_forward_pre_hook(
                lambda _module, args: captured.append(args[2].detach().clone())
            )
            # Deliberately set unrelated WM stats: critic must not use these.
            trainer.world_model.context_norm.mean.fill_(100.0)
            trainer._critic_input_with_aux(
                trainer.obs["wam_state"], trainer.env.reference(), None, context,
                future_reference=trainer.env.future_reference(trainer.critic_future_frames),
            )
            hook.remove()
            torch.testing.assert_close(captured[0], normalize_supermimic_context(context))
            self.assertEqual(trainer.critic_norm.eps, 1e-2)
            self.assertEqual(trainer.critic_norm.min_std, 0.0)
            frame = trainer._physical_actor_frame(
                trainer.obs["wam_state"], trainer.env.actions, context=context,
            )
            torch.testing.assert_close(frame[:, 6:35], torch.full((trainer.env.num_envs, 29), 0.04))
            trainer.collect(1)
            torch.testing.assert_close(trainer.replay._storage["context"][0],
                                       context[trainer.replay_env_ids])

    def test_collection_adds_noise_before_one_clip_and_imagination_returns_raw_action(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, action_noise=0.1)
            raw_action = torch.full((trainer.env.num_envs, 29), 20.05)
            with patch.object(trainer.policy, "forward", return_value=raw_action):
                imagined = trainer._action(
                    trainer.layout.pack(trainer.obs), trainer.layout.pack_goal(trainer.obs),
                    actor_obs=trainer.actor_obs,
                )
                torch.testing.assert_close(imagined, raw_action)
                with patch("torch.randn_like", side_effect=lambda value: -torch.ones_like(value)):
                    trainer.collect(1)
            torch.testing.assert_close(trainer.env.actions, torch.full_like(raw_action, 19.95))
            legacy = self.make_trainer(Path(directory) / "legacy", context_mode="legacy", profile="legacy")
            with patch.object(legacy.policy, "forward", return_value=raw_action):
                result = legacy._action(legacy.layout.pack(legacy.obs),
                                        legacy.layout.pack_goal(legacy.obs), actor_obs=legacy.actor_obs)
            torch.testing.assert_close(result, raw_action.clamp(-20, 20))

    def test_resume_rejects_same_size_context_with_changed_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = self.make_trainer(Path(directory) / "legacy", context_mode="legacy", profile="legacy")
            path = Path(directory) / "last.pt"
            legacy.save_checkpoint(path)
            aligned = self.make_trainer(Path(directory) / "aligned")
            self.assertEqual(legacy.context_dim, aligned.context_dim)
            with self.assertRaisesRegex(ValueError, "normalization, noise or actor objective"):
                aligned.load_checkpoint(path, resume=True)
            aligned.load_checkpoint(path, resume=False)


if __name__ == "__main__":
    unittest.main()
