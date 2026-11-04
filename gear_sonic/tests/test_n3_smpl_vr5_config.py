"""Compose the SMPL VR5 variant without Isaac Lab or motion datasets."""

from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gear_sonic.utils.config_utils import (
    configure_wam_context, configure_wam_observations, configure_wam_rewards,
    configure_wam_terminations,
)


class N3SmplVr5ConfigTests(unittest.TestCase):
    def config(self, experiment="sonic_n3_smpl_vr5", overrides=()):
        directory = Path(__file__).resolve().parents[1] / "config"
        with initialize_config_dir(config_dir=str(directory), version_base="1.1"):
            return compose(config_name="base", overrides=[
                f"+exp=manager/universal_token/all_modes/{experiment}", *overrides,
            ])

    def test_variant_preserves_robot_smpl_and_wam_configuration(self):
        old = self.config("sonic_n3")
        new = self.config()
        for cfg in (old, new):
            for configure in (configure_wam_context, configure_wam_observations,
                              configure_wam_rewards, configure_wam_terminations):
                configure(cfg)
        for name in ("g1", "smpl"):
            self.assertEqual(
                OmegaConf.to_container(new.algo.config.actor.backbone.encoders[name]),
                OmegaConf.to_container(old.algo.config.actor.backbone.encoders[name]),
            )
            for term in new.algo.config.actor.backbone.encoders[name].inputs:
                self.assertEqual(
                    OmegaConf.to_container(new.manager_env.observations.tokenizer[term]),
                    OmegaConf.to_container(old.manager_env.observations.tokenizer[term]),
                )
        for name in ("robot", "terrain_type"):
            self.assertEqual(new.manager_env.config[name], old.manager_env.config[name])
        self.assertEqual(new.trainer, old.trainer)
        self.assertEqual(new.world_model, old.world_model)
        self.assertEqual(new.manager_env.rewards, old.manager_env.rewards)
        self.assertEqual(new.manager_env.terminations, old.manager_env.terminations)
        self.assertEqual(new.manager_env.commands.motion.encoder_sample_probs,
                         old.manager_env.commands.motion.encoder_sample_probs)

    def test_teleop_uses_only_registered_smpl_goals(self):
        cfg = self.config()
        self.assertEqual(list(cfg.algo.config.actor.backbone.encoders.teleop.inputs), [
            "smpl_vr_5point_local_target", "smpl_vr_5point_local_orn_target", "smpl_root_ori_b",
        ])
        tokenizer = cfg.manager_env.observations.tokenizer
        for name in cfg.algo.config.actor.backbone.encoders.teleop.inputs:
            self.assertEqual(tokenizer[name].func,
                             f"gear_sonic.envs.manager_env.mdp:{name}")
            self.assertEqual(tokenizer[name].params.command_name, "motion")
        for name in ("command_multi_future_lower_body", "vr_3point_local_target",
                     "vr_3point_local_orn_target", "motion_anchor_ori_b", "command_z"):
            self.assertIsNone(tokenizer[name])
        self.assertTrue(cfg.manager_env.commands.motion.motion_lib_cfg.require_smpl_vr5)
        self.assertEqual(set(cfg.manager_env.commands.motion.encoder_sample_probs),
                         {"g1", "smpl", "teleop"})

    def test_original_experiment_keeps_robot_teleop(self):
        cfg = self.config("sonic_n3")
        self.assertIn("command_multi_future_lower_body",
                      cfg.algo.config.actor.backbone.encoders.teleop.inputs)
        self.assertIn("vr_3point_local_target",
                      cfg.algo.config.actor.backbone.encoders.teleop.inputs)
        self.assertIsNotNone(cfg.manager_env.observations.tokenizer.vr_3point_local_target)
        self.assertFalse(OmegaConf.select(
            cfg, "manager_env.commands.motion.motion_lib_cfg.require_smpl_vr5", default=False,
        ))

    def test_server_dataset_paths_remain_explicit_overrides(self):
        cfg = self.config()
        self.assertIsNone(cfg.manager_env.commands.motion.motion_lib_cfg.motion_file)
        self.assertIsNone(cfg.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file)
        cfg = self.config(overrides=(
            "manager_env.commands.motion.motion_lib_cfg.motion_file=/srv/data/n3/robot",
            "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/srv/data/n3/smpl",
        ))
        self.assertEqual(cfg.manager_env.commands.motion.motion_lib_cfg.motion_file,
                         "/srv/data/n3/robot")
        self.assertEqual(cfg.manager_env.commands.motion.motion_lib_cfg.smpl_motion_file,
                         "/srv/data/n3/smpl")
        self.assertTrue(cfg.manager_env.commands.motion.motion_lib_cfg.require_smpl_vr5)


if __name__ == "__main__":
    unittest.main()
