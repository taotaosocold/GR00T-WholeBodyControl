"""Terminal-state snapshots survive Isaac Lab's reset-before-observation order."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gear_sonic.tests.test_wam_state import make_physical_env
from gear_sonic.utils.wam_state import robot_state
from gear_sonic.utils.wam_termination import (
    capture_terminal_state, enable_terminal_capture, get_terminal_state,
    supermimic_failure, supermimic_time_out,
)


def make_capture_env():
    env, command = make_physical_env(with_hands=True)
    command.motion_ids = torch.tensor([0, 1])
    command.motion_start_time_steps = torch.zeros(2, dtype=torch.long)
    command.time_steps = torch.zeros(2, dtype=torch.long)
    command.motion_anchor_body_index = 0
    position = (command.robot_anchor_pos_w - env.scene.env_origins).clone()
    quaternion = command.robot_anchor_quat_w.clone()
    command.motion_lib = SimpleNamespace(
        get_time_step_total=lambda ids: torch.full_like(ids, 10),
        get_body_pos_w=lambda ids, frames: position[ids, None],
        get_body_quat_w=lambda ids, frames: quaternion[ids, None],
    )
    env.episode_length_buf = torch.tensor([1, 1])
    env.max_episode_length = 500
    return env, command


class TerminalCaptureTests(unittest.TestCase):
    def test_disabled_capture_preserves_other_trainers_and_minimal_envs(self):
        env = SimpleNamespace()
        self.assertIsNone(capture_terminal_state(env))
        self.assertIsNone(get_terminal_state(env))
        self.assertFalse(hasattr(env, "_wam_terminal_state"))

    def test_failure_cache_is_post_physics_and_survives_auto_reset(self):
        env, command = make_capture_env()
        enable_terminal_capture(env)
        # Physics has just made robot 0 fall; robot 1 remains healthy.
        command.robot_anchor_pos_w[0, 2] -= 0.5
        terminal = robot_state(env, command).clone()
        failed = supermimic_failure(env)
        self.assertEqual(failed.tolist(), [True, False])
        self.assertEqual(env._wam_terminal_state_step, env.common_step_counter)
        # Isaac Lab now resets robot 0 before generating its returned next_obs.
        command.robot_anchor_pos_w[0, 2] += 0.5
        command.robot.data.joint_pos[0].zero_()
        command.robot.data.body_link_lin_vel_w[0].zero_()
        reset_state = robot_state(env, command)
        self.assertFalse(torch.equal(reset_state[failed], terminal[failed]))
        cached = get_terminal_state(env)
        self.assertFalse(cached.requires_grad)
        torch.testing.assert_close(cached, terminal)
        # This is the consumer's intended use: only replace the done rows.
        corrected = torch.where(failed[:, None], cached, reset_state)
        torch.testing.assert_close(corrected[failed], terminal[failed])
        torch.testing.assert_close(corrected[~failed], reset_state[~failed])

    def test_timeout_alone_captures_terminal_state_before_reset(self):
        env, command = make_capture_env()
        enable_terminal_capture(env)
        env.episode_length_buf[1] = env.max_episode_length
        expected = robot_state(env, command)
        self.assertEqual(supermimic_time_out(env).tolist(), [False, True])
        command.robot.data.joint_pos.add_(50)
        torch.testing.assert_close(get_terminal_state(env), expected)

    def test_failure_and_timeout_share_one_capture_and_next_step_refreshes(self):
        env, command = make_capture_env()
        enable_terminal_capture(env)
        with patch("gear_sonic.utils.wam_termination.robot_state", wraps=robot_state) as state_fn:
            supermimic_failure(env)
            first = get_terminal_state(env)
            supermimic_time_out(env)
            self.assertIs(get_terminal_state(env), first)
            self.assertEqual(state_fn.call_count, 1)
            env.common_step_counter += 1
            command.robot.data.joint_pos.add_(0.1)
            # A cache from a different simulator step must never be consumed.
            self.assertIsNone(get_terminal_state(env))
            supermimic_time_out(env)
            self.assertEqual(state_fn.call_count, 2)
            self.assertIsNot(get_terminal_state(env), first)
            self.assertFalse(torch.equal(get_terminal_state(env), first))

    def test_missing_hook_and_wrong_step_or_command_return_none(self):
        env, _ = make_capture_env()
        enable_terminal_capture(env)
        # A disabled aligned-termination configuration never executes a hook.
        self.assertIsNone(get_terminal_state(env))
        supermimic_failure(env)
        self.assertIsNone(get_terminal_state(env, expected_step=env.common_step_counter + 1))
        self.assertIsNone(get_terminal_state(env, command_name="other_motion"))
        self.assertIsNone(capture_terminal_state(env, command_name="other_motion"))
        self.assertIsNotNone(get_terminal_state(env))
        enable_terminal_capture(env)
        self.assertIsNone(get_terminal_state(env))

    def test_snapshot_is_detached_and_missing_counter_fails_explicitly(self):
        env, command = make_capture_env()
        command.robot.data.joint_pos.requires_grad_()
        enable_terminal_capture(env)
        supermimic_failure(env)
        self.assertFalse(get_terminal_state(env).requires_grad)
        del env.common_step_counter
        with self.assertRaisesRegex(ValueError, "common_step_counter"):
            capture_terminal_state(env)


if __name__ == "__main__":
    unittest.main()
