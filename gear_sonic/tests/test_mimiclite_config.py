"""Hydra composition tests for the named sonic_mimiclite experiment."""

from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir

from gear_sonic.utils.config_utils import (
    configure_wam_context, configure_wam_observations,
    configure_wam_rewards, configure_wam_terminations,
)


class MimicLiteConfigTests(unittest.TestCase):
    def test_n3_vr5_profile_selects_mimiclite_terms_and_preserves_wam_context(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            config = compose(config_name="base", overrides=[
                "+exp=manager/universal_token/all_modes/sonic_n3_smpl_vr5_mimiclite",
                "trainer=wam", "world_model=wam_dynamics",
            ])
        configure_wam_context(config)
        configure_wam_observations(config)
        configure_wam_rewards(config)
        configure_wam_terminations(config)
        self.assertEqual(config.termination_profile, "mimiclite")
        self.assertTrue(config.finite_guard)
        self.assertEqual(config.world_model.cfg.context_mode, "supermimic")
        self.assertEqual(config.manager_env.observations.wam_context.xi.params.context_mode, "supermimic")
        self.assertEqual(set(config.manager_env.terminations), {
            "_target_", "root_pos_error", "root_ori_error", "body_pos_error", "body_ori_error",
            "ee_body_pos", "time_out",
        })
        self.assertTrue(config.manager_env.terminations.root_pos_error.time_out)
        self.assertFalse(config.manager_env.terminations.ee_body_pos.get("time_out", False))
        self.assertEqual(
            list(config.manager_env.terminations.ee_body_pos.params.body_names),
            [
                "l_leg_ankle_roll_link", "r_leg_ankle_roll_link",
                "l_arm_hand_pitch_link", "r_arm_hand_pitch_link",
            ],
        )
        self.assertEqual(config.manager_env.terminations.root_pos_error.params.min_steps, 25)
        self.assertEqual(config.manager_env.terminations.body_pos_error.params.min_steps, 5)

    def test_g1_profile_uses_torso_root_and_g1_policy_data(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            config = compose(config_name="base", overrides=[
                "+exp=manager/universal_token/all_modes/sonic_mimiclite",
                "trainer=wam", "world_model=wam_dynamics",
            ])
        configure_wam_context(config)
        configure_wam_observations(config)
        configure_wam_rewards(config)
        configure_wam_terminations(config)
        self.assertEqual(config.manager_env.config.robot.type, "g1_model_12_dex")
        self.assertEqual(config.termination_profile, "mimiclite")
        self.assertEqual(config.manager_env.terminations.root_pos_error.params.body_names, ["torso_link"])
        self.assertEqual(config.manager_env.terminations.root_ori_error.params.body_names, ["torso_link"])
        self.assertEqual(config.manager_env.terminations.body_pos_error.params.root_body_names, ["torso_link"])
        self.assertEqual(
            list(config.manager_env.terminations.ee_body_pos.params.body_names),
            [
                "left_ankle_roll_link", "right_ankle_roll_link",
                "left_wrist_yaw_link", "right_wrist_yaw_link",
            ],
        )
        self.assertFalse(config.manager_env.terminations.ee_body_pos.get("time_out", False))
        self.assertIn("g1_29dof", str(config.trainer.reward_config.mjcf_path))


if __name__ == "__main__":
    unittest.main()
