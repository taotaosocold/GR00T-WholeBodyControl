"""Goal-frame and gradient checks without Isaac Lab or a running simulator."""

import math
import unittest

import torch

from gear_sonic.utils.wam_observations import WAMGoalObservations


def yaw_state(angle):
    angle = torch.as_tensor(angle, dtype=torch.float64)
    state = torch.zeros(*angle.shape, 71, dtype=torch.float64)
    state[..., 3] = (angle / 2).cos()
    state[..., 6] = (angle / 2).sin()
    return state


def yaw_matrix(angle):
    angle = torch.as_tensor(angle, dtype=torch.float64)
    c, s = angle.cos(), angle.sin()
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(
        *angle.shape, 3, 3
    )


class GoalObservationTests(unittest.TestCase):
    def setUp(self):
        # The untouched values include two VR orientation matrices and the
        # routing index. Deliberately interleave all three converted terms.
        self.specs = (
            ("encoder_index", 0, 3, (3,)),
            ("motion_anchor_ori_b_mf_nonflat", 3, 15, (2, 6)),
            ("vr_3point_local_orn_target", 15, 27, (2, 6)),
            ("motion_anchor_ori_b", 27, 33, (6,)),
            ("smpl_root_ori_b_multi_future", 33, 51, (3, 6)),
            ("vr_3point_local_target", 51, 60, (3, 3)),
        )
        self.adapter = WAMGoalObservations(self.specs)
        self.goal = torch.randn(2, 60, dtype=torch.float64)

    def test_first_two_columns_use_interleaved_row_order(self):
        adapter = WAMGoalObservations([("motion_anchor_ori_b", 0, 6, (6,))])
        identity_6d = torch.tensor([[1., 0., 0., 1., 0., 0.]], dtype=torch.float64)
        actual = adapter.to_world(identity_6d, yaw_state([math.pi / 2]))
        # R_z(90 degrees)[:, :2].reshape(6), not [column0, column1].
        expected = torch.tensor([[0., -1., 1., 0., 0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, atol=1e-15, rtol=0)

    def test_all_modalities_match_matrix_multiplication_without_changing_vr(self):
        state = yaw_state([0.7, -1.2])
        world = self.adapter.to_world(self.goal, state)
        rotation = yaw_matrix([0.7, -1.2])
        expected = self.goal.clone()
        for _, start, end in self.adapter.orientation_slices:
            vectors = self.goal[:, start:end].reshape(2, -1, 3, 2)
            expected[:, start:end] = (rotation[:, None] @ vectors).flatten(-3)
        torch.testing.assert_close(world, expected)
        torch.testing.assert_close(world[:, :3], self.goal[:, :3], atol=0, rtol=0)
        torch.testing.assert_close(world[:, 15:27], self.goal[:, 15:27], atol=0, rtol=0)
        torch.testing.assert_close(world[:, 51:], self.goal[:, 51:], atol=0, rtol=0)

    def test_noisy_orientations_roundtrip_without_orthogonalization(self):
        original = self.goal.clone()
        state = yaw_state([0.5, -0.3])
        world = self.adapter.to_world(self.goal, state)
        reconstructed = self.adapter.to_robot(world, state)
        torch.testing.assert_close(reconstructed, original, atol=2e-15, rtol=1e-14)
        torch.testing.assert_close(self.goal, original, atol=0, rtol=0)
        self.assertNotEqual(world.data_ptr(), self.goal.data_ptr())
        self.assertNotEqual(reconstructed.data_ptr(), world.data_ptr())

    def test_imagined_orientation_rebuilds_the_reference_goal(self):
        adapter = WAMGoalObservations([("motion_anchor_ori_b", 0, 6, (6,))])
        reference = yaw_matrix([0.4])
        real_rotation = yaw_matrix([-0.2])
        observed = (real_rotation.transpose(-1, -2) @ reference)[..., :2].flatten(-2)
        world = adapter.to_world(observed, yaw_state([-0.2]))
        imagined = adapter.to_robot(world, yaw_state([0.8]))
        expected = (yaw_matrix([0.8]).transpose(-1, -2) @ reference)[..., :2].flatten(-2)
        torch.testing.assert_close(imagined, expected)
        self.assertFalse(torch.allclose(imagined, observed))

    def test_time_batches_and_shared_robot_rotation(self):
        timeline = self.goal[:, None].expand(-1, 4, -1).clone()
        state = yaw_state([[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]])
        world = self.adapter.to_world(timeline, state)
        expected = torch.stack([
            self.adapter.to_world(timeline[:, t], state[:, t]) for t in range(4)
        ], dim=1)
        torch.testing.assert_close(world, expected)
        torch.testing.assert_close(self.adapter.to_robot(world, state), timeline)
        shared_state = state[:, 0]
        shared = self.adapter.to_world(timeline, shared_state)
        torch.testing.assert_close(
            shared, self.adapter.to_world(self.goal, shared_state)[:, None].expand_as(shared)
        )

    def test_tokenizer_offset_preserves_other_goal_groups(self):
        adapter = WAMGoalObservations(self.specs, tokenizer_offset=5)
        prefix, suffix = torch.randn(2, 5), torch.randn(2, 7)
        goals = torch.cat((prefix, self.goal, suffix), -1)
        state = yaw_state([0.3, 0.8])
        actual = adapter.to_world(goals, state)
        expected = torch.cat((prefix, self.adapter.to_world(self.goal, state), suffix), -1)
        torch.testing.assert_close(actual, expected)

    def test_rotation_and_goal_gradients_include_roll_pitch_yaw(self):
        # No detach may appear between imagined orientation and the encoder.
        adapter = WAMGoalObservations([("motion_anchor_ori_b", 0, 6, (6,))])
        goal = torch.randn(1, 6, dtype=torch.float64, requires_grad=True)
        quat = torch.tensor([[1., .2, -.3, .1]], dtype=torch.float64, requires_grad=True)

        def rebuild(values, quaternion):
            state = torch.cat((quaternion.new_zeros(1, 3), quaternion, quaternion.new_zeros(1, 64)), -1)
            return adapter.to_robot(values, state)

        self.assertTrue(torch.autograd.gradcheck(rebuild, (goal, quat)))
        output = rebuild(goal, quat)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(goal.grad).all())
        # Use a non-rotation-invariant objective to verify orientation gradients.
        goal.grad = None
        quat.grad = None
        rebuild(goal, quat)[0, 0].backward()
        self.assertGreater(quat.grad.abs().sum().item(), 0)

    def test_world_conversion_preserves_source_gradients(self):
        goal = self.goal.clone().requires_grad_()
        state = yaw_state([0.2, -0.3]).requires_grad_()
        result = self.adapter.to_world(goal, state)
        result[:, 3].sum().backward()
        self.assertGreater(goal.grad.abs().sum().item(), 0)
        self.assertGreater(state.grad[:, 3:7].abs().sum().item(), 0)

    def test_absent_orientation_is_a_new_unchanged_tensor(self):
        adapter = WAMGoalObservations([("encoder_index", 0, 3, (3,))])
        result = adapter.to_world(self.goal, yaw_state([0.2, 0.7]))
        torch.testing.assert_close(result, self.goal, atol=0, rtol=0)
        self.assertNotEqual(result.data_ptr(), self.goal.data_ptr())

    def test_invalid_orientation_spans_and_batches_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "6D"):
            WAMGoalObservations([("motion_anchor_ori_b", 0, 5, (5,))])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            WAMGoalObservations(self.specs, tokenizer_offset=-1)
        with self.assertRaisesRegex(ValueError, "beyond"):
            self.adapter.to_world(self.goal[:, :40], yaw_state([0., 0.]))
        with self.assertRaisesRegex(ValueError, "batch dimensions"):
            self.adapter.to_robot(self.goal, yaw_state([0., 0., 0.]))


if __name__ == "__main__":
    unittest.main()
