"""Physical-state ordering and link-origin velocities without Isaac Lab."""

import math
from types import SimpleNamespace
import unittest

import torch

from gear_sonic.utils.wam_kinematics import G1_MUJOCO_JOINT_NAMES
from gear_sonic.utils.wam_reward import robot_state as reward_robot_state
from gear_sonic.utils.wam_state import link_origin_linear_velocity, robot_state


def make_physical_env(with_hands=False):
    """A nonzero anchor index and reversed joints catch silent ordering bugs."""
    names = list(reversed(G1_MUJOCO_JOINT_NAMES))
    if with_hands:
        names += [f"hand_{i}" for i in range(14)]
    count = len(names)
    q = torch.arange(2 * count, dtype=torch.float64).reshape(2, count) / 100
    qd = q + 0.2
    data = SimpleNamespace(
        joint_pos=q, joint_vel=qd,
        body_link_lin_vel_w=torch.tensor([
            [[99., 99., 99.], [0.1, 0.2, 0.3]],
            [[99., 99., 99.], [-0.1, 0.4, -0.3]],
        ], dtype=torch.float64),
    )
    origins = torch.tensor([[5., -4., 2.], [-8., 3., -5.]], dtype=torch.float64)
    position = torch.tensor([[0.1, 0.2, 0.8], [0.4, -0.3, 0.9]], dtype=torch.float64)
    quaternion = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]], dtype=torch.float64)
    omega = torch.tensor([[0., 0., 0.3], [0.4, -0.2, 0.1]], dtype=torch.float64)
    robot = SimpleNamespace(data=data, joint_names=names)
    command = SimpleNamespace(
        robot=robot, robot_anchor_body_index=1,
        robot_anchor_pos_w=position + origins,
        robot_anchor_quat_w=quaternion,
        # The legacy command property is deliberately wrong: WAM must read
        # the articulation's explicit link-origin velocity instead.
        robot_anchor_lin_vel_w=data.body_link_lin_vel_w[:, 1] + 10,
        robot_anchor_ang_vel_w=omega,
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(env_origins=origins), common_step_counter=1,
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    return env, command


class LinkVelocityTests(unittest.TestCase):
    def test_explicit_link_velocity_is_preferred_over_com(self):
        expected = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]], dtype=torch.float64)
        data = SimpleNamespace(body_link_lin_vel_w=expected, body_com_lin_vel_w=expected + 10)
        torch.testing.assert_close(link_origin_linear_velocity(data, 1), expected[:, 1])

    def test_com_fallback_rotates_local_offset_and_uses_correct_cross_sign(self):
        quaternion = torch.tensor([[[math.sqrt(0.5), 0., 0., math.sqrt(0.5)]]], dtype=torch.float64)
        data = SimpleNamespace(
            body_com_lin_vel_w=torch.tensor([[[0.4, 2., 3.]]], dtype=torch.float64),
            body_com_ang_vel_w=torch.tensor([[[0., 0., 2.]]], dtype=torch.float64),
            body_com_pos_b=torch.tensor([[[0.3, 0., 0.]]], dtype=torch.float64),
            body_link_quat_w=quaternion,
        )
        expected = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
        torch.testing.assert_close(link_origin_linear_velocity(data, 0), expected)
        self.assertFalse(torch.allclose(expected, data.body_com_lin_vel_w[:, 0]))

    def test_legacy_names_support_shared_com_offsets_and_wxyz_quaternions(self):
        # A shared [1,body,3] inertial offset must broadcast across environments.
        data = SimpleNamespace(
            body_lin_vel_w=torch.tensor([[[0.4, 2., 3.]], [[0.1, 5., 6.]]], dtype=torch.float64),
            body_ang_vel_w=torch.tensor([[[0., 0., 2.]], [[0., 0., 3.]]], dtype=torch.float64),
            com_pos_b=torch.tensor([[[0.3, 0., 0.]]], dtype=torch.float64),
            body_quat_w=torch.tensor([
                [[-2 * math.sqrt(0.5), 0., 0., -2 * math.sqrt(0.5)]],
                [[math.sqrt(0.5), 0., 0., math.sqrt(0.5)]],
            ], dtype=torch.float64),
        )
        torch.testing.assert_close(
            link_origin_linear_velocity(data, 0),
            torch.tensor([[1., 2., 3.], [1., 5., 6.]], dtype=torch.float64),
        )

    def test_missing_com_metadata_never_silently_returns_com_velocity(self):
        data = SimpleNamespace(body_lin_vel_w=torch.zeros(2, 1, 3))
        with self.assertRaisesRegex(ValueError, "link-origin velocity"):
            link_origin_linear_velocity(data, 0)

    def test_com_offset_and_angular_velocity_have_finite_gradients(self):
        offset = torch.tensor([[[0.2, -0.1, 0.3]]], dtype=torch.float64, requires_grad=True)
        omega = torch.tensor([[[0.4, 0.2, -0.3]]], dtype=torch.float64, requires_grad=True)
        data = SimpleNamespace(
            body_com_lin_vel_w=torch.zeros(1, 1, 3, dtype=torch.float64),
            body_com_ang_vel_w=omega, body_com_pos_b=offset,
            body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64),
        )
        link_origin_linear_velocity(data, 0).square().sum().backward()
        for value in (offset, omega):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)


class PhysicalStateTests(unittest.TestCase):
    def test_named_body_joints_origins_and_link_velocity_match_71d_contract(self):
        for with_hands in (False, True):
            env, command = make_physical_env(with_hands=with_hands)
            state = robot_state(env)
            self.assertEqual(tuple(state.shape), (2, 71))
            self.assertEqual(state.dtype, torch.float64)
            torch.testing.assert_close(state[:, :3], command.robot_anchor_pos_w - env.scene.env_origins)
            torch.testing.assert_close(state[:, 3:7], command.robot_anchor_quat_w)
            torch.testing.assert_close(state[:, 7:10], command.robot.data.body_link_lin_vel_w[:, 1])
            torch.testing.assert_close(state[:, 10:13], command.robot_anchor_ang_vel_w)
            indices = [command.robot.joint_names.index(name) for name in G1_MUJOCO_JOINT_NAMES]
            torch.testing.assert_close(state[:, 13:42], command.robot.data.joint_pos[:, indices])
            torch.testing.assert_close(state[:, 42:71], command.robot.data.joint_vel[:, indices])
            torch.testing.assert_close(reward_robot_state(env, command), state)

    def test_robot_state_does_not_alias_live_joint_buffers(self):
        env, command = make_physical_env()
        state = robot_state(env, command)
        saved = state.clone()
        command.robot.data.joint_pos.add_(100)
        command.robot.data.body_link_lin_vel_w.add_(100)
        torch.testing.assert_close(state, saved)


if __name__ == "__main__":
    unittest.main()
