"""Portable N3 experiment composition and explicit data requirements."""

from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gear_sonic.envs.env_utils.n3_utils import (
    N3_MJCF_PATH, N3_MUJOCO_JOINT_NAMES, N3_TRACKED_BODIES, N3_URDF_PATH,
    N3_WRIST_JOINT_NAMES,
)
from gear_sonic.utils.config_utils import (
    configure_wam_context, configure_wam_observations, configure_wam_rewards,
    configure_wam_terminations, validate_robot_motion_config,
)


class N3ConfigTests(unittest.TestCase):
    def config(self, overrides=()):
        directory = Path(__file__).resolve().parents[1] / "config"
        with initialize_config_dir(config_dir=str(directory), version_base="1.1"):
            return compose(config_name="base", overrides=[
                "+exp=manager/universal_token/all_modes/sonic_n3", *overrides,
            ])

    def test_default_experiment_uses_n3_assets_and_three_fsq_modalities(self):
        cfg = self.config()
        for configure in (configure_wam_context, configure_wam_observations,
                          configure_wam_rewards, configure_wam_terminations):
            configure(cfg)
        motion = cfg.manager_env.commands.motion
        self.assertEqual(cfg.manager_env.config.robot.type, "n3")
        self.assertTrue(cfg.trainer._target_.endswith("WAMTrainer"))
        self.assertEqual(cfg.world_model.cfg.state_type, "physical")
        self.assertEqual(cfg.trainer.reward_config.robot_type, "n3")
        self.assertEqual(tuple(motion.body_names), N3_TRACKED_BODIES)
        self.assertEqual(tuple(cfg.trainer.reward_config.tracked_bodies), N3_TRACKED_BODIES)
        self.assertIn("head_yaw_link", motion.body_names)
        self.assertEqual(tuple(cfg.manager_env.observations.tokenizer
                               .joint_pos_multi_future_wrist_for_smpl.params.joint_names),
                         N3_WRIST_JOINT_NAMES)
        self.assertEqual(set(motion.encoder_sample_probs), {"g1", "teleop", "smpl"})
        self.assertTrue(all(value > 0 for value in motion.encoder_sample_probs.values()))
        self.assertEqual(cfg.trainer.reward_config.mjcf_path, N3_MJCF_PATH)
        repository = Path(__file__).resolve().parents[2]
        for asset in (N3_MJCF_PATH, N3_URDF_PATH):
            self.assertFalse(Path(asset).is_absolute())
            self.assertTrue((repository / asset).is_file())
        self.assertEqual(len(N3_MUJOCO_JOINT_NAMES), 29)

    def test_missing_or_placeholder_data_fails_before_simulation(self):
        cfg = self.config()
        self.assertIsNone(cfg.manager_env.commands.motion.motion_lib_cfg.motion_file)
        self.assertIsNone(cfg.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file)
        with self.assertRaisesRegex(ValueError, "N3.*paired.*SMPL"):
            validate_robot_motion_config(cfg)
        cfg.manager_env.commands.motion.motion_lib_cfg.motion_file = "dummy"
        with self.assertRaises(ValueError):
            validate_robot_motion_config(cfg)

    def test_explicit_paired_data_paths_validate_without_loading_pickle(self):
        cfg = self.config()
        with tempfile.TemporaryDirectory() as directory:
            robot = Path(directory) / "robot"
            smpl = Path(directory) / "smpl"
            robot.mkdir()
            smpl.mkdir()
            cfg.manager_env.commands.motion.motion_lib_cfg.motion_file = str(robot)
            cfg.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file = str(smpl)
            validate_robot_motion_config(cfg)
            cfg.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file += "/absent"
            with self.assertRaises(FileNotFoundError):
                validate_robot_motion_config(cfg)

    def test_data_guard_does_not_change_g1_configuration(self):
        cfg = OmegaConf.create({"manager_env": {"config": {"robot": {"type": "g1_model_12_dex"}}}})
        before = OmegaConf.to_container(cfg)
        validate_robot_motion_config(cfg)
        self.assertEqual(OmegaConf.to_container(cfg), before)


if __name__ == "__main__":
    unittest.main()
