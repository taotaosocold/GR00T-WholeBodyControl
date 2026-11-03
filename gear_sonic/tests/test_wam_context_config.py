"""Complete-context activation stays isolated from PPO configuration."""

import unittest

from omegaconf import OmegaConf

from gear_sonic.utils.config_utils import configure_wam_context


def configuration(target="WAMTrainer", state_type="physical", complete=True):
    return OmegaConf.create({
        "trainer": {"_target_": f"example.{target}", "settings": {"complete_context": complete}},
        "world_model": {"cfg": {"state_type": state_type}},
        "manager_env": {"observations": {"wam_context": {"xi": {
            "func": "example.wam_context", "params": {"asset_cfg": {"name": "robot"}},
        }}}},
    })


class WAMContextConfigurationTests(unittest.TestCase):
    def test_physical_context_activation_is_idempotent_and_preserves_other_parameters(self):
        config = configuration()
        self.assertTrue(configure_wam_context(config))
        self.assertTrue(config.manager_env.observations.wam_context.xi.params.complete)
        self.assertEqual(config.manager_env.observations.wam_context.xi.params.asset_cfg.name, "robot")
        before = OmegaConf.to_container(config)
        configure_wam_context(config)
        self.assertEqual(OmegaConf.to_container(config), before)

    def test_explicit_legacy_override_updates_a_saved_complete_configuration(self):
        config = configuration()
        configure_wam_context(config)
        config.trainer.settings.complete_context = False
        configure_wam_context(config)
        self.assertFalse(config.manager_env.observations.wam_context.xi.params.complete)

    def test_ppo_removes_legacy_context_without_changing_user_groups(self):
        config = configuration(target="PPOTrainer")
        config.manager_env.observations.user_group = {"sensor": {"func": "example.sensor"}}
        before = OmegaConf.to_container(config.manager_env.observations.user_group)
        self.assertFalse(configure_wam_context(config))
        self.assertNotIn("wam_context", config.manager_env.observations)
        self.assertEqual(OmegaConf.to_container(config.manager_env.observations.user_group), before)

    def test_flat_configuration_is_unchanged(self):
        config = configuration(state_type="flat")
        before = OmegaConf.to_container(config)
        self.assertFalse(configure_wam_context(config))
        self.assertEqual(OmegaConf.to_container(config), before)


if __name__ == "__main__":
    unittest.main()
