"""The physical WAM profile matches the same domain for G1 and N3."""

from pathlib import Path
from types import SimpleNamespace
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.utils.config_utils import configure_wam_context, configure_wam_observations
from gear_sonic.utils.wam_domain_randomization import (
    randomize_supermimic_material, reset_supermimic_joint_offsets,
)


def articulation_fixture(robot="g1", count=4):
    names = list(reversed(joint_utils.get_robot_joint_names(robot, order="mujoco"))) + ["extra_finger_joint"]
    nominal = torch.arange(len(names)).float() / 100
    materials = torch.ones(count, 5, 3)
    materials[..., 2] = 0.4
    class View:
        def get_material_properties(self):
            return materials
        def set_material_properties(self, values, ids):
            materials[ids.long()] = values[ids.long()]
    asset = SimpleNamespace(
        joint_names=names, body_names=["waist_yaw_link"] if robot == "n3" else ["torso_link"],
        data=SimpleNamespace(joint_pos=torch.zeros(count, len(names)),
                             default_joint_pos=nominal.expand(count, -1).clone()),
        root_physx_view=View(),
    )
    action_names = names[::2] + names[1::2]
    action = SimpleNamespace(_joint_names=action_names, _offset=torch.zeros(count, len(names)))
    class Scene(dict):
        num_envs = count
    env = SimpleNamespace(scene=Scene(robot=asset), action_manager=SimpleNamespace(get_term=lambda name: action))
    return env, asset, action, nominal, materials


class SuperMimicDomainTests(unittest.TestCase):
    def test_calibration_is_bounded_across_repeated_resets_and_maps_action_names(self):
        for robot in ("g1", "n3"):
            with self.subTest(robot=robot):
                env, asset, action, nominal, _ = articulation_fixture(robot)
                untouched = asset.data.default_joint_pos[1].clone()
                for _ in range(40):
                    reset_supermimic_joint_offsets(env, torch.tensor([0, 2]), SimpleNamespace(name="robot"))
                    delta = asset.data.default_joint_pos - nominal
                    self.assertLessEqual(float(delta[:, :-1].abs().max()), 0.010001)
                    self.assertTrue(torch.equal(asset.data.default_joint_pos[1], untouched))
                    self.assertTrue(torch.equal(delta[:, -1], torch.zeros(4)))
                    body_names = joint_utils.get_robot_joint_names(robot, order="mujoco")
                    for i, name in enumerate(body_names):
                        j = asset.joint_names.index(name)
                        self.assertTrue(torch.allclose(env._wam_xi[[0, 2], i + 1], delta[[0, 2], j], atol=1e-7))
                        a = action._joint_names.index(name)
                        self.assertTrue(torch.equal(action._offset[[0, 2], a], asset.data.default_joint_pos[[0, 2], j]))
                self.assertTrue(torch.equal(asset.data.default_joint_pos_nominal, nominal))

    def test_material_draw_is_continuous_shared_per_environment_and_preserves_other_rows(self):
        env, asset, _, _, materials = articulation_fixture(count=512)
        randomize_supermimic_material(env, torch.arange(511), SimpleNamespace(name="robot"))
        mu = materials[:511, 0, 0]
        self.assertTrue(bool(((mu >= 0.3) & (mu <= 1.2)).all()))
        self.assertGreater(mu.unique().numel(), 64)
        self.assertTrue(torch.equal(materials[:511, :, 0], mu[:, None].expand(-1, 5)))
        self.assertTrue(torch.equal(materials[:511, :, 1], mu[:, None].expand(-1, 5)))
        self.assertTrue(torch.equal(materials[:511, :, 2], torch.zeros(511, 5)))
        self.assertTrue(torch.equal(materials[511, :, 0], torch.ones(5)))
        self.assertTrue(torch.allclose(env._wam_xi[:511, 0], mu))

    def test_every_shipped_physical_wam_embodiment_uses_the_same_domain(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
        for experiment in ("sonic_release", "sonic_n3", "sonic_n3_smpl_vr5"):
            with self.subTest(experiment=experiment), initialize_config_dir(config_dir=config_dir, version_base="1.1"):
                config = compose(config_name="base", overrides=[
                    f"+exp=manager/universal_token/all_modes/{experiment}",
                    "trainer=wam", "world_model=wam_dynamics",
                ])
                configure_wam_context(config)
                configure_wam_observations(config)
                self.assertEqual(set(config.manager_env.events), {"_target_", "physics_material", "base_com", "add_joint_default_pos"})
                self.assertEqual(config.manager_env.events.physics_material.mode, "startup")
                self.assertEqual(config.manager_env.events.base_com.mode, "startup")
                self.assertEqual(config.manager_env.events.add_joint_default_pos.mode, "reset")
                torso = "torso_link" if experiment == "sonic_release" else "waist_yaw_link"
                self.assertEqual(config.manager_env.events.base_com.params.asset_cfg.body_names, torso)
                self.assertEqual(config.manager_env.config.terrain_type, "plane")
                self.assertEqual(config.manager_env.config.wam_domain_randomization, "supermimic")
                self.assertEqual(config.manager_env.commands.motion.reset_distribution, "supermimic")
                self.assertFalse(config.trainer.settings.complete_context)
                self.assertEqual(config.manager_env.observations.wam_context.xi.params.context_mode, "supermimic")
                self.assertEqual(config.world_model.cfg.context_mode, "supermimic")
                before = OmegaConf.to_container(config, resolve=False)
                configure_wam_context(config)
                configure_wam_observations(config)
                self.assertEqual(before, OmegaConf.to_container(config, resolve=False))

    def test_ppo_keeps_sonic_randomization(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            config = compose(config_name="base", overrides=["+exp=manager/universal_token/all_modes/sonic_release"])
        events = OmegaConf.to_container(config.manager_env.events, resolve=False)
        configure_wam_context(config)
        configure_wam_observations(config)
        self.assertEqual(OmegaConf.to_container(config.manager_env.events, resolve=False), events)
        self.assertIn("randomize_rigid_body_mass", config.manager_env.events)
        self.assertIn("push_robot", config.manager_env.events)
        self.assertEqual(config.manager_env.config.terrain_type, "trimesh")


if __name__ == "__main__":
    unittest.main()
