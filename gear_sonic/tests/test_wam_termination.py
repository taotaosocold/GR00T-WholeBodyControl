"""Hard/soft failure parity and WAM-only environment selection, without Isaac Lab."""

import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from gear_sonic.utils.config_utils import configure_wam_terminations
from gear_sonic.utils.wam_termination import (
    hard_tracking_termination,
    next_motion_frame,
    reference_pose,
    soft_tracking_termination,
    supermimic_failure,
    supermimic_time_out,
    tracking_errors,
)


def quaternion(angle, axis=0):
    result = torch.zeros(4, dtype=torch.float64)
    result[0] = math.cos(angle / 2)
    result[axis + 1] = math.sin(angle / 2)
    return result


class MotionLibrary:
    """Moving reference with distinct poses at every frame and body."""

    def get_time_step_total(self, motion_ids):
        return torch.tensor([4, 3])[motion_ids]

    def get_body_pos_w(self, motion_ids, frames):
        poses = torch.zeros(len(motion_ids), 2, 3, dtype=torch.float64)
        poses[:, 0, 2] = 99  # Selecting the wrong anchor must fail the test.
        poses[:, 1, 2] = motion_ids * 2 + frames * 0.4
        return poses

    def get_body_quat_w(self, motion_ids, frames):
        poses = torch.zeros(len(motion_ids), 2, 4, dtype=torch.float64)
        poses[..., 0] = 1
        return poses


def make_env():
    command = SimpleNamespace(
        motion_ids=torch.tensor([0, 1, 0]),
        motion_start_time_steps=torch.tensor([1, 0, 3]),
        time_steps=torch.tensor([0, 0, 0]),
        motion_anchor_body_index=1,
        motion_lib=MotionLibrary(),
    )
    origins = torch.tensor([[4., 2., 10.], [-2., 8., -7.], [9., -3., 2.]], dtype=torch.float64)
    position, rotation = reference_pose(command)
    command.robot_anchor_pos_w = position + origins
    command.robot_anchor_quat_w = rotation

    def get_term(name):
        if name != "motion":
            raise KeyError(name)
        return command

    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=get_term),
        scene=SimpleNamespace(env_origins=origins),
        episode_length_buf=torch.tensor([10, 10, 10]),
        max_episode_length=500,
    )
    return env, command


class TerminationTests(unittest.TestCase):
    def setUp(self):
        self.position = torch.zeros(3, 3, dtype=torch.float64)
        self.identity = quaternion(0).expand(3, -1)

    def test_height_boundary_uses_strict_greater_than(self):
        actual = self.position.clone()
        actual[:, 2] = torch.tensor([0.249, 0.25, 0.251], dtype=torch.float64)
        hard = hard_tracking_termination(actual, self.identity, self.position, self.identity)
        self.assertEqual(hard.tolist(), [False, False, True])
        soft = soft_tracking_termination(actual, self.identity, self.position, self.identity)
        self.assertLess(soft[0].item(), 0.5)
        self.assertAlmostEqual(soft[1].item(), 0.5, places=7)
        self.assertGreater(soft[2].item(), 0.5)

    def test_tilt_uses_rotation_zz_instead_of_full_orientation_angle(self):
        rotations = torch.stack([
            quaternion(math.acos(0.3)),  # Rzz error = 0.7
            quaternion(math.acos(0.1)),  # Rzz error = 0.9
            quaternion(math.pi, axis=2),  # 180 degree yaw has no tilt.
        ])
        height, tilt = tracking_errors(self.position, rotations, self.position, self.identity)
        torch.testing.assert_close(height, torch.zeros_like(height))
        torch.testing.assert_close(tilt, torch.tensor([0.7, 0.9, 0.], dtype=torch.float64))
        self.assertEqual(
            hard_tracking_termination(self.position, rotations, self.position, self.identity).tolist(),
            [False, True, False],
        )

    def test_errors_ignore_xy_heading_and_quaternion_sign(self):
        actual = self.position.clone()
        actual[:, :2] = torch.tensor([[500., -20.], [-100., 900.], [0., -3.]])
        rotations = torch.stack([quaternion(2., axis=2), -quaternion(0), quaternion(-1., axis=2)])
        self.assertFalse(hard_tracking_termination(
            actual, rotations, self.position, self.identity,
        ).any())

    def test_soft_rule_has_finite_nonzero_height_and_tilt_gradients(self):
        actual = self.position[:1].clone()
        actual[:, 2] = 0.25
        actual.requires_grad_()
        rotation = quaternion(math.acos(0.2)).unsqueeze(0).requires_grad_()
        soft = soft_tracking_termination(
            actual, rotation, self.position[:1], self.identity[:1],
        )
        self.assertAlmostEqual(soft.item(), 0.75, places=7)  # Both boundaries: 1 - .5*.5.
        soft.sum().backward()
        self.assertTrue(torch.isfinite(actual.grad).all())
        self.assertTrue(torch.isfinite(rotation.grad).all())
        self.assertGreater(actual.grad[:, 2].item(), 0)
        self.assertGreater(rotation.grad.abs().sum().item(), 0)

    def test_soft_boundary_tracks_custom_hard_thresholds(self):
        actual = self.position[:1].clone()
        actual[:, 2] = 0.4
        args = actual, self.identity[:1], self.position[:1], self.identity[:1]
        self.assertTrue(hard_tracking_termination(*args).item())
        self.assertFalse(hard_tracking_termination(*args, height_threshold=0.4).item())
        self.assertAlmostEqual(
            soft_tracking_termination(*args, height_threshold=0.4).item(), 0.5, places=7,
        )
        with self.assertRaises(ValueError):
            soft_tracking_termination(*args, height_width=0)

    def test_simulator_checks_next_frame_with_origin_and_last_frame_clamp(self):
        env, command = make_env()
        self.assertEqual(next_motion_frame(command).tolist(), [2, 1, 3])
        command.robot_anchor_pos_w[:, 2] += torch.tensor([0.1, 0.3, 0.], dtype=torch.float64)
        self.assertEqual(supermimic_failure(env).tolist(), [False, True, False])
        # The first robot is >.25m away from the old frame but close to the
        # next frame. This detects checking f_t rather than f_{t+1}.
        old = command.motion_lib.get_body_pos_w(command.motion_ids, command.motion_start_time_steps)
        old_z = old[:, command.motion_anchor_body_index, 2]
        robot_z = (command.robot_anchor_pos_w - env.scene.env_origins)[:, 2]
        self.assertGreater(abs(robot_z[0] - old_z[0]).item(), 0.25)

    def test_clip_and_episode_limits_are_truncations_independent_of_failure(self):
        env, command = make_env()
        env.episode_length_buf = torch.tensor([499, 500, 0])
        self.assertFalse(supermimic_failure(env).any())
        self.assertEqual(supermimic_time_out(env).tolist(), [False, True, True])
        command.robot_anchor_pos_w[0, 2] += 1
        self.assertTrue(supermimic_failure(env)[0])
        self.assertFalse(supermimic_time_out(env)[0])


class TrainerTerminationIntegrationTests(unittest.TestCase):
    def test_analytic_termination_uses_shared_pose_errors_and_custom_thresholds(self):
        from gear_sonic.trl.trainer.wam_trainer import WAMTrainer

        trainer = WAMTrainer.__new__(WAMTrainer)
        trainer.settings = {
            "termination_height": 0.4,
            "termination_tilt": 0.6,
            "termination_height_width": 0.03,
            "termination_tilt_width": 0.01,
        }

        def yaw_roll(yaw, roll):
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            cr, sr = math.cos(roll / 2), math.sin(roll / 2)
            return torch.tensor([cy * cr, cy * sr, sy * sr, sy * cr], dtype=torch.float64)

        state = torch.zeros(3, 71, dtype=torch.float64)
        reference = torch.zeros_like(state)
        reference[:, 2] = 1
        reference[:, 3:7] = yaw_roll(-0.7, 0.2)
        state[:, 2] = reference[:, 2] + torch.tensor([0.399, 0.401, 0.], dtype=torch.float64)
        state[:, 3:7] = torch.stack([
            yaw_roll(1.4, math.acos(math.cos(0.2) - error))
            for error in (0.5, 0.5, 0.7)
        ])
        pose = state[:, :3], state[:, 3:7], reference[:, :3], reference[:, 3:7]
        actual = trainer._analytic_termination(state, reference)
        expected = soft_tracking_termination(
            *pose, height_threshold=0.4, tilt_threshold=0.6,
            height_width=0.03, tilt_width=0.01,
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(hard_tracking_termination(
            *pose, height_threshold=0.4, tilt_threshold=0.6,
        ).tolist(), [False, True, True])
        self.assertLess(actual[0].item(), 0.5)
        self.assertGreater(actual[1].item(), 0.5)
        self.assertGreater(actual[2].item(), 0.5)

    def test_next_physical_reference_is_canonical_and_survives_command_reset(self):
        from gear_sonic.trl.trainer.wam_trainer import WAMTrainer

        class PhysicalMotionLibrary(MotionLibrary):
            def get_dof_pos(self, motion_ids, frames):
                return (motion_ids[:, None] * 100 + frames[:, None] * 10
                        + torch.arange(29)[None, :]).to(torch.float64)

            def get_dof_vel(self, motion_ids, frames):
                return -2 * self.get_dof_pos(motion_ids, frames)

            def get_body_lin_vel_w(self, motion_ids, frames):
                return self.get_body_pos_w(motion_ids, frames) + 5

            def get_body_ang_vel_w(self, motion_ids, frames):
                return self.get_body_pos_w(motion_ids, frames) - 3

        env, command = make_env()
        command.motion_lib = PhysicalMotionLibrary()
        trainer = WAMTrainer.__new__(WAMTrainer)
        trainer.env = SimpleNamespace(env=env)
        trainer.reference_dim = 71
        captured = trainer._next_physical_reference()
        self.assertEqual(captured.shape, (3, 71))
        frames = torch.tensor([2, 1, 3])
        position = command.motion_lib.get_body_pos_w(command.motion_ids, frames)[:, 1]
        rotation = command.motion_lib.get_body_quat_w(command.motion_ids, frames)[:, 1]
        torch.testing.assert_close(captured[:, :3], position)
        torch.testing.assert_close(captured[:, 3:7], rotation)
        torch.testing.assert_close(captured[:, 7:10], position + 5)
        torch.testing.assert_close(captured[:, 10:13], position - 3)
        # Independent expected mapping: Isaac Lab interleaves left/right
        # limbs; canonical WAM order stores each limb together.
        canonical_order = [
            0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
            11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
        ]
        joints = command.motion_lib.get_dof_pos(command.motion_ids, frames)[:, canonical_order]
        torch.testing.assert_close(captured[:, 13:42], joints)
        torch.testing.assert_close(captured[:, 42:71], -2 * joints)
        snapshot = captured.clone()
        command.motion_ids.copy_(1 - command.motion_ids)
        command.motion_start_time_steps.zero_()
        command.time_steps.zero_()
        torch.testing.assert_close(captured, snapshot)
        self.assertFalse(torch.equal(trainer._next_physical_reference(), captured))


class TerminationConfigTests(unittest.TestCase):
    def make_config(self):
        return OmegaConf.create({
            "trainer": {
                "_target_": "gear_sonic.trl.trainer.wam_trainer.WAMTrainer",
                "settings": {"align_real_termination": True},
            },
            "world_model": {"cfg": {"state_type": "physical"}},
            "manager_env": {"terminations": {
                "_target_": "gear_sonic.envs.manager_env.mdp.terminations.TerminationsCfg",
                "foot_pos_xyz": {"params": {"threshold": 0.2}},
                "ee_body_pos": {"params": {"threshold": 0.15}},
                "anchor_ori_full": {"params": {"threshold": 0.2}},
            }},
        })

    def test_wam_replaces_strict_rules_and_keeps_timeout_separate(self):
        config = self.make_config()
        OmegaConf.set_struct(config, True)
        self.assertTrue(configure_wam_terminations(config))
        terms = config.manager_env.terminations
        self.assertEqual(set(terms), {"_target_", "wam_failure", "time_out"})
        self.assertEqual(terms.wam_failure.params.height_threshold, 0.25)
        self.assertEqual(terms.wam_failure.params.tilt_threshold, 0.8)
        self.assertFalse(terms.wam_failure.get("time_out", False))
        self.assertTrue(terms.time_out.time_out)

    def test_ppo_flat_and_unmarked_old_checkpoint_are_unchanged(self):
        for case in ("ppo", "flat", "old_checkpoint", "disabled"):
            with self.subTest(case=case):
                config = self.make_config()
                if case == "ppo":
                    config.trainer._target_ = "gear_sonic.trl.trainer.ppo_trainer.PPOTrainer"
                elif case == "flat":
                    config.world_model.cfg.state_type = "flat"
                elif case == "old_checkpoint":
                    del config.trainer.settings.align_real_termination
                else:
                    config.trainer.settings.align_real_termination = False
                before = OmegaConf.to_container(config, resolve=False)
                self.assertFalse(configure_wam_terminations(config))
                self.assertEqual(before, OmegaConf.to_container(config, resolve=False))

    def test_custom_thresholds_survive_saved_config_and_repeated_selection(self):
        config = self.make_config()
        config.trainer.settings.termination_height = 0.35
        config.trainer.settings.termination_tilt = 0.6
        configure_wam_terminations(config)
        before = OmegaConf.to_container(config, resolve=False)
        self.assertEqual(config.manager_env.terminations.wam_failure.params.height_threshold, 0.35)
        self.assertEqual(config.manager_env.terminations.wam_failure.params.tilt_threshold, 0.6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            OmegaConf.save(config, path)
            restored = OmegaConf.load(path)
            self.assertTrue(configure_wam_terminations(restored))
        self.assertEqual(before, OmegaConf.to_container(restored, resolve=False))

    def test_real_sonic_release_composition_selects_wam_without_changing_ppo(self):
        config_dir = Path(__file__).resolve().parents[1] / "config"
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            ppo = compose(config_name="base", overrides=[
                "+exp=manager/universal_token/all_modes/sonic_release",
            ])
            wam = compose(config_name="base", overrides=[
                "+exp=manager/universal_token/all_modes/sonic_release",
                "trainer=wam", "world_model=wam_dynamics",
            ])
        ppo_terms = OmegaConf.to_container(ppo.manager_env.terminations, resolve=False)
        self.assertFalse(configure_wam_terminations(ppo))
        self.assertEqual(ppo_terms, OmegaConf.to_container(ppo.manager_env.terminations, resolve=False))
        self.assertIn("foot_pos_xyz", ppo_terms)
        self.assertTrue(configure_wam_terminations(wam))
        self.assertEqual(set(wam.manager_env.terminations), {"_target_", "wam_failure", "time_out"})
        self.assertEqual(wam.manager_env.terminations.wam_failure.params.height_threshold, 0.25)


if __name__ == "__main__":
    unittest.main()
