"""The WAM actor sees the same configured sensors during training and evaluation."""

from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gear_sonic.utils.config_utils import configure_wam_observations
from gear_sonic.utils.wam_noise import WAM_OBSERVATION_NOISE, wam_observation_noise


class WAMNoiseTests(unittest.TestCase):
    def make_config(self):
        terms = {name: {} for name in WAM_OBSERVATION_NOISE}
        return OmegaConf.create({
            "trainer": {"_target_": "gear_sonic.trl.trainer.wam_trainer.WAMTrainer", "settings": {}},
            "world_model": {"cfg": {"state_type": "physical"}},
            "manager_env": {"observations": {"policy": terms, "wam": terms}},
        })

    def assert_matching_noise(self, config):
        expected = wam_observation_noise(config.trainer.settings)
        for name in ("policy", "wam"):
            group = config.manager_env.observations[name]
            for term, width in expected.items():
                self.assertEqual(group[term].noise.n_min, -width)
                self.assertEqual(group[term].noise.n_max, width)
            self.assertEqual(group.enable_corruption, any(expected.values()))

    def test_physical_training_and_saved_evaluation_configs_share_noise(self):
        config = self.make_config()
        OmegaConf.set_struct(config, True)
        self.assertTrue(configure_wam_observations(config))
        self.assert_matching_noise(config)
        before = OmegaConf.to_container(config, resolve=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            OmegaConf.save(config, path)
            evaluated = OmegaConf.load(path)
            # Native policy observations must be restored to the same sensor
            # contract even if an evaluation preset supplies another noise.
            evaluated.manager_env.observations.policy.base_ang_vel.noise.n_max = 9
            configure_wam_observations(evaluated)
        self.assertEqual(before, OmegaConf.to_container(evaluated, resolve=False))

    def test_wam_overrides_and_zero_noise_apply_to_both_groups(self):
        config = self.make_config()
        config.trainer.settings.observation_noise = {"base_ang_vel": 0.1}
        configure_wam_observations(config)
        self.assert_matching_noise(config)
        config.trainer.settings.observation_noise = {name: 0 for name in WAM_OBSERVATION_NOISE}
        configure_wam_observations(config)
        self.assert_matching_noise(config)

    def test_ppo_and_flat_configurations_are_not_changed(self):
        for case in ("ppo", "flat"):
            with self.subTest(case=case):
                config = self.make_config()
                if case == "ppo":
                    config.trainer._target_ = "gear_sonic.trl.trainer.ppo_trainer.PPOTrainer"
                else:
                    config.world_model.cfg.state_type = "flat"
                before = OmegaConf.to_container(config, resolve=False)
                self.assertFalse(configure_wam_observations(config))
                self.assertEqual(before, OmegaConf.to_container(config, resolve=False))

    def test_noise_definition_rejects_unknown_negative_and_nonfinite_values(self):
        for override in ({"unknown": 0.1}, {"joint_pos": -1}, {"gravity_dir": float("nan")}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                wam_observation_noise({"observation_noise": override})

    def test_release_config_and_passive_wam_yaml_match_supermimic_widths(self):
        config_dir = Path(__file__).resolve().parents[1] / "config"
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            config = compose(config_name="base", overrides=[
                "+exp=manager/universal_token/all_modes/sonic_release",
                "trainer=wam", "world_model=wam_dynamics",
            ])
        before = config.manager_env.observations.wam
        for name, width in WAM_OBSERVATION_NOISE.items():
            self.assertEqual(before[name].noise.n_min, -width)
            self.assertEqual(before[name].noise.n_max, width)
        configure_wam_observations(config)
        self.assert_matching_noise(config)


if __name__ == "__main__":
    unittest.main()
