"""Physical critic goal conditioning and pre-reset reference capture."""

import math
from types import SimpleNamespace
import unittest

import torch

from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
from gear_sonic.utils.wam_critic import WAMCriticFeatures, future_reference_states
from gear_sonic.utils.wam_kinematics import G1_MUJOCO_JOINT_NAMES


def state(batch=2, dtype=torch.float64):
    result = torch.zeros(batch, 71, dtype=dtype)
    result[:, 2] = 0.8
    result[:, 3] = 1
    return result


class CriticFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.model = WAMCriticFeatures().double()
        self.state = state()
        self.context = torch.arange(66, dtype=torch.float64).reshape(2, 33) / 100
        self.future = self.state[:, None].expand(-1, 10, -1).clone()

    def test_future_changes_value_input_when_current_reference_is_identical(self):
        first = self.model(self.state, self.state, self.context, self.future)
        changed = self.future.clone()
        changed[:, 3:, 13] += 0.4
        second = self.model(self.state, self.state, self.context, changed)
        self.assertEqual(first.shape, (2, 914))
        torch.testing.assert_close(first[:, :184], second[:, :184])
        self.assertFalse(torch.equal(first[:, 184:], second[:, 184:]))

    def test_tracking_errors_are_rebuilt_from_the_supplied_robot_state(self):
        first = self.model(self.state, self.state, self.context, self.future)
        robot = self.state.clone()
        robot[:, 0] += 0.3
        robot[:, 13] += 0.2
        second = self.model(robot, self.state, self.context, self.future)
        torch.testing.assert_close(second[:, 71], torch.full((2,), -0.2, dtype=torch.float64))
        torch.testing.assert_close(second[:, 142], torch.full((2,), -0.3, dtype=torch.float64))
        self.assertFalse(torch.equal(first[:, 100:142], second[:, 100:142]))
        torch.testing.assert_close(first[:, 184:], second[:, 184:])
        torch.testing.assert_close(second[:, 151:184], self.context)

    def test_matching_reference_has_zero_tracking_errors(self):
        result = self.model(self.state, self.state, self.context, self.future)
        torch.testing.assert_close(result[:, 71:145], torch.zeros(2, 74, dtype=torch.float64))
        identity_6d = torch.tensor([1., 0., 0., 1., 0., 0.], dtype=torch.float64)
        torch.testing.assert_close(result[:, 145:151], identity_6d.expand(2, -1))

    def test_future_features_use_reference_heading_and_future_local_velocities(self):
        model = WAMCriticFeatures(future_frames=1, context_dim=0).double()
        reference = state(batch=1)
        reference[:, :3] = torch.tensor([2., 3., 0.8])
        reference[:, 3] = math.cos(math.pi / 4)
        reference[:, 6] = math.sin(math.pi / 4)
        future = reference[:, None].clone()
        future[..., :3] += torch.tensor([0., 1., 2.])
        future[..., 3:7] = torch.tensor([0., 0., 0., 1.])
        future[..., 7:10] = torch.tensor([0., 1., 0.])
        future[..., 10:13] = torch.tensor([2., 0., 0.])
        future[..., 13:42] = torch.arange(29)
        future[..., 42:71] = 2
        result = model(reference, reference, None, future)[..., -73:]
        expected = torch.cat((
            torch.tensor([1., 0., 2., 0., -1., 1., 0., 0., 0., 0., -1., 0., -2., 0., 0.]),
            torch.arange(29), torch.full((29,), 0.2),
        )).double()[None]
        torch.testing.assert_close(result, expected, atol=1e-8, rtol=1e-8)

    def test_critic_features_keep_finite_state_gradients(self):
        robot = self.state.clone()
        robot[:, 0] += 0.2
        robot[:, 13:42] = torch.linspace(-0.1, 0.2, 29)
        robot[:, 42:71] = torch.linspace(-0.2, 0.3, 29)
        robot.requires_grad_()
        result = self.model(robot, self.state, self.context, self.future)
        result.square().sum().backward()
        self.assertTrue(torch.isfinite(robot.grad).all())
        self.assertGreater(robot.grad[:, 13:42].abs().sum().item(), 0.1)
        self.assertGreater(robot.grad[:, :3].abs().sum().item(), 0.1)

    def test_batch_and_time_dimensions_match_flattened_evaluation(self):
        model = WAMCriticFeatures(future_frames=4).double()
        robot = self.state[:, None].expand(-1, 3, -1).clone()
        robot[..., 0] = torch.arange(6).reshape(2, 3)
        reference = self.state[:, None].expand_as(robot)
        context = self.context[:, None].expand(-1, 3, -1)
        future = self.future[:, :4, :][:, None].expand(-1, 3, -1, -1)
        result = model(robot, reference, context, future)
        expected = model(
            robot.reshape(6, 71), reference.reshape(6, 71), context.reshape(6, 33),
            future.reshape(6, 4, 71),
        ).reshape(2, 3, -1)
        torch.testing.assert_close(result, expected)

    def test_input_validation_rejects_missing_or_mismatched_goals(self):
        with self.assertRaisesRegex(ValueError, "future_reference"):
            self.model(self.state, self.state, self.context, self.future[:, :-1])
        with self.assertRaisesRegex(ValueError, "context"):
            self.model(self.state, self.state, None, self.future)
        with self.assertRaisesRegex(ValueError, "matching"):
            self.model(self.state, self.state[:1], self.context, self.future)


class ReferenceLibrary:
    def get_time_step_total(self, ids):
        return torch.tensor([8, 10])[ids]

    def get_body_pos_w(self, ids, frames):
        root = torch.stack((frames, ids, torch.full_like(ids, 2)), -1).double()
        return torch.stack((torch.full_like(root, -100), root), 1)

    def get_body_quat_w(self, ids, frames):
        result = torch.zeros(len(frames), 2, 4, dtype=torch.float64)
        result[..., 0] = 1
        return result

    def get_body_lin_vel_w(self, ids, frames):
        return self.get_body_pos_w(ids, frames) + 100

    def get_body_ang_vel_w(self, ids, frames):
        return self.get_body_pos_w(ids, frames) + 200

    def get_dof_pos(self, ids, frames):
        return frames[:, None].double() * 100 + torch.arange(29)

    def get_dof_vel(self, ids, frames):
        return frames[:, None].double() * 10 + torch.arange(29)


class FutureReferenceTests(unittest.TestCase):
    def setUp(self):
        self.command = SimpleNamespace(
            motion_lib=ReferenceLibrary(), motion_ids=torch.tensor([0, 1]),
            motion_start_time_steps=torch.tensor([1, 6]), time_steps=torch.tensor([0, 1]),
            motion_anchor_body_index=1,
        )

    def test_future_grid_uses_requested_stride_and_stays_inside_each_clip(self):
        result = future_reference_states(self.command, num_frames=4, frame_stride=2)
        frames = torch.tensor([[1., 3., 5., 7.], [7., 9., 9., 9.]], dtype=torch.float64)
        self.assertEqual(result.shape, (2, 4, 71))
        torch.testing.assert_close(result[..., 0], frames)
        order = torch.tensor([G1_ISAACLab_ORDER.index(name) for name in G1_MUJOCO_JOINT_NAMES])
        torch.testing.assert_close(result[..., 13:42], frames[..., None] * 100 + order)
        torch.testing.assert_close(result[..., 42:71], frames[..., None] * 10 + order)
        torch.testing.assert_close(result[..., 7:10], result[..., :3] + 100)
        torch.testing.assert_close(result[..., 10:13], result[..., :3] + 200)

    def test_next_reference_is_captured_before_command_reset(self):
        result = future_reference_states(self.command, num_frames=4, frame_stride=2, next_step=True)
        self.command.motion_start_time_steps.zero_()
        self.command.time_steps.zero_()
        self.command.motion_ids.zero_()
        expected = torch.tensor([[2., 4., 6., 7.], [8., 9., 9., 9.]], dtype=torch.float64)
        torch.testing.assert_close(result[..., 0], expected)

    def test_frame_grid_settings_are_positive_integers(self):
        for kwargs in ({"num_frames": 0}, {"num_frames": 1.5}, {"frame_stride": 0}, {"frame_stride": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                future_reference_states(self.command, **kwargs)


if __name__ == "__main__":
    unittest.main()
