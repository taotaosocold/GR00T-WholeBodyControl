"""Physical WAM observations are opt-in and survive saved-config evaluation."""

from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gear_sonic.utils.config_utils import configure_wam_context, configure_wam_observations


GROUPS = ("wam", "wam_state", "wam_reference", "wam_context")


def release_configuration(*overrides):
    directory = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(config_dir=str(directory), version_base="1.1"):
        return compose(config_name="base", overrides=[
            "+exp=manager/universal_token/all_modes/sonic_release", *overrides,
        ])


class WAMObservationConfigurationTests(unittest.TestCase):
    def test_fresh_ppo_does_not_load_or_compute_physical_wam_groups(self):
        config = release_configuration()
        before = OmegaConf.to_container(config.manager_env.observations, resolve=False)
        for name in GROUPS:
            self.assertNotIn(name, config.manager_env.observations)
        self.assertFalse(configure_wam_context(config))
        self.assertFalse(configure_wam_observations(config))
        self.assertEqual(before, OmegaConf.to_container(config.manager_env.observations, resolve=False))

    def test_legacy_ppo_removes_only_reserved_wam_groups(self):
        config = release_configuration()
        with tempfile.TemporaryDirectory() as directory:
            old = OmegaConf.to_container(config, resolve=False)
            for name in GROUPS:
                old["manager_env"]["observations"][name] = {"bad_term": {"func": "missing.wam"}}
            old["manager_env"]["observations"]["camera_custom"] = {"sensor": {"func": "user.camera"}}
            path = Path(directory) / "config.yaml"
            OmegaConf.save(OmegaConf.create(old), path)
            restored = OmegaConf.load(path)
        before = OmegaConf.to_container(restored, resolve=False)
        for name in GROUPS:
            del before["manager_env"]["observations"][name]
        configure_wam_context(restored)
        configure_wam_observations(restored)
        self.assertEqual(before, OmegaConf.to_container(restored, resolve=False))

    def test_wam_templates_are_resolved_and_partial_overrides_are_preserved(self):
        config = release_configuration(
            "trainer=wam", "world_model=wam_dynamics",
            "++manager_env.observations.wam.joint_pos.scale=0.75",
            "++manager_env.observations.wam_context.xi.params.asset_cfg.name=custom_robot",
        )
        configure_wam_context(config)
        configure_wam_observations(config)
        groups = config.manager_env.observations
        self.assertEqual(groups.wam.joint_pos.scale, 0.75)
        self.assertEqual(groups.wam.joint_pos.func, "gear_sonic.envs.manager_env.mdp:joint_pos_rel")
        self.assertEqual(groups.wam_context.xi.params.asset_cfg.name, "custom_robot")
        self.assertFalse(groups.wam_context.xi.params.complete)
        self.assertEqual(groups.wam_context.xi.params.context_mode, "supermimic")
        self.assertEqual(groups.wam_state.root_lin_vel.func, "gear_sonic.envs.manager_env.mdp:wam_root_lin_vel_w")
        for name in GROUPS:
            self.assertNotIn("defaults", groups[name])
            self.assertIn("_target_", groups[name])
        before = OmegaConf.to_container(config, resolve=False)
        configure_wam_context(config)
        configure_wam_observations(config)
        self.assertEqual(before, OmegaConf.to_container(config, resolve=False))

    def test_explicit_hydra_group_selection_still_works(self):
        config = release_configuration(
            "trainer=wam", "world_model=wam_dynamics",
            "manager_env/observations/wam=proprio",
            "++manager_env.observations.wam.joint_pos.params.custom_flag=true",
        )
        configure_wam_context(config)
        self.assertTrue(config.manager_env.observations.wam.joint_pos.params.custom_flag)

    def test_old_wam_checkpoint_can_recreate_missing_physical_groups(self):
        config = release_configuration("trainer=wam", "world_model=wam_dynamics")
        # A real pre-profile checkpoint did not contain the new default key.
        del config.trainer.settings.domain_randomization
        config.trainer.settings.complete_context = True
        config.trainer.settings.terrain_type = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            OmegaConf.save(config, path)
            restored = OmegaConf.load(path)
        configure_wam_context(restored)
        configure_wam_observations(restored)
        for name in GROUPS:
            self.assertIn(name, restored.manager_env.observations)
        self.assertTrue(restored.manager_env.observations.wam_context.xi.params.complete)

    def test_fresh_flat_wam_supplies_requested_groups_without_physical_reconfiguration(self):
        config = release_configuration(
            "trainer=wam", "world_model=wam_dynamics", "world_model.cfg.state_type=flat",
        )
        policy_before = OmegaConf.to_container(config.manager_env.observations.policy, resolve=False)
        self.assertFalse(configure_wam_context(config))
        self.assertFalse(configure_wam_observations(config))
        for name in GROUPS:
            self.assertIn(name, config.manager_env.observations)
        self.assertNotIn("complete", config.manager_env.observations.wam_context.xi.params)
        self.assertEqual(policy_before, OmegaConf.to_container(config.manager_env.observations.policy, resolve=False))
        before = OmegaConf.to_container(config, resolve=False)
        configure_wam_context(config)
        configure_wam_observations(config)
        self.assertEqual(before, OmegaConf.to_container(config, resolve=False))

    def test_custom_flat_groups_are_preserved_without_unused_physical_groups(self):
        config = OmegaConf.create({
            "trainer": {"_target_": "example.WAMTrainer", "settings": {
                "state_observation_keys": ["wam_state"],
                "context_observation_keys": ["custom_context"],
                "goal_observation_keys": ["custom_goal"],
            }},
            "world_model": {"cfg": {"state_type": "flat"}},
            "manager_env": {"observations": {
                "wam_state": {"user_sensor": {"func": "example.flat_state"}},
                "wam": {"user_sensor": {"func": "example.flat_frame"}},
                "custom_context": {"sensor": {"func": "example.context"}},
                "custom_goal": {"sensor": {"func": "example.goal"}},
            }},
        })
        before = OmegaConf.to_container(config, resolve=False)
        self.assertFalse(configure_wam_context(config))
        self.assertFalse(configure_wam_observations(config))
        self.assertEqual(before, OmegaConf.to_container(config, resolve=False))
        self.assertNotIn("wam_reference", config.manager_env.observations)
        self.assertNotIn("wam_context", config.manager_env.observations)

    def test_default_wam_uses_plane_and_legacy_profile_preserves_sonic(self):
        config = release_configuration("trainer=wam", "world_model=wam_dynamics")
        configure_wam_context(config)
        self.assertEqual(config.manager_env.config.terrain_type, "plane")
        legacy = release_configuration(
            "trainer=wam", "world_model=wam_dynamics",
            "trainer.settings.domain_randomization=sonic", "trainer.settings.terrain_type=null",
        )
        configure_wam_context(legacy)
        self.assertEqual(legacy.manager_env.config.terrain_type, "trimesh")
        legacy.trainer.settings.terrain_type = "plane"
        configure_wam_context(legacy)
        self.assertEqual(legacy.manager_env.config.terrain_type, "plane")
        ppo = release_configuration("++trainer.settings.terrain_type=plane")
        configure_wam_context(ppo)
        self.assertEqual(ppo.manager_env.config.terrain_type, "trimesh")

    def test_unknown_wam_terrain_override_fails_before_mutation(self):
        config = release_configuration(
            "trainer=wam", "world_model=wam_dynamics", "trainer.settings.terrain_type=stairs",
        )
        before = OmegaConf.to_container(config, resolve=False)
        with self.assertRaisesRegex(ValueError, "terrain_type"):
            configure_wam_context(config)
        self.assertEqual(before, OmegaConf.to_container(config, resolve=False))


if __name__ == "__main__":
    unittest.main()
