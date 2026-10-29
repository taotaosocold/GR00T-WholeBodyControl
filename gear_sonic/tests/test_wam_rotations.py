"""Numerical rotation regressions for physical WAM fitting and BPTT."""

import math
import unittest

import torch

from gear_sonic.trl.modules.wam import (
    StateSpaceWorldModel,
    WAMConfig,
    _mat_to_quat,
    _quat_to_mat,
    _rotvec_to_quat,
    physical_state_transition_error,
)


def physical_state(quaternion):
    shape = quaternion.shape[:-1]
    return torch.cat((
        quaternion.new_zeros(shape + (3,)), quaternion,
        quaternion.new_zeros(shape + (64,)),
    ), dim=-1)


class WAMRotationTests(unittest.TestCase):
    def test_matrix_conversion_handles_exact_half_turns_about_all_axes(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                quaternions = torch.eye(4, dtype=dtype)
                rotations = _quat_to_mat(quaternions)
                recovered = _mat_to_quat(rotations)
                torch.testing.assert_close(_quat_to_mat(recovered), rotations)
                torch.testing.assert_close(recovered.norm(dim=-1), torch.ones(4, dtype=dtype))

    def test_matrix_conversion_round_trips_random_and_near_half_turns(self):
        generator = torch.Generator().manual_seed(81)
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.float64, 1e-12)):
            with self.subTest(dtype=dtype):
                quaternions = torch.randn(2, 30, 4, generator=generator, dtype=dtype)
                quaternions[:, :3, 0] = torch.tensor([0., 1e-8, -1e-8], dtype=dtype)
                rotations = _quat_to_mat(quaternions)
                recovered = _mat_to_quat(rotations)
                torch.testing.assert_close(
                    _quat_to_mat(recovered), rotations, rtol=tolerance, atol=tolerance,
                )

    def test_matrix_conversion_has_finite_derivatives_at_identity_and_half_turn(self):
        for angle in (0., math.pi - 1e-7, math.pi, math.pi + 1e-7):
            with self.subTest(angle=angle):
                vector = torch.tensor([0., 0., angle], dtype=torch.float64, requires_grad=True)
                rotation = _quat_to_mat(_rotvec_to_quat(vector))
                recovered = _mat_to_quat(rotation)
                recovered.sum().backward()
                self.assertTrue(torch.isfinite(vector.grad).all())

    def test_zero_dynamics_preserves_every_heading_including_half_turn(self):
        model = StateSpaceWorldModel(WAMConfig(
            state_dim=71, action_dim=29, hidden=(), state_type="physical",
            learn_reward=False, learn_termination=False,
        )).double()
        angles = torch.tensor([0., math.pi / 2, math.pi, -math.pi, 1.5 * math.pi], dtype=torch.float64)
        vectors = torch.zeros(5, 3, dtype=torch.float64)
        vectors[:, 2] = angles
        quaternion = _rotvec_to_quat(vectors)
        quaternion[2] = torch.tensor([0., 0., 0., 1.], dtype=torch.float64)
        state = physical_state(quaternion)
        predicted = model(state, torch.zeros(5, 29, dtype=torch.float64))["next_state"]
        torch.testing.assert_close(predicted, state, rtol=1e-12, atol=1e-12)

    def test_rotation_error_matches_small_angles_and_pi(self):
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.float64, 1e-12)):
            with self.subTest(dtype=dtype):
                angles = torch.tensor(
                    [0., 1e-9, 1e-6, 1e-4, 1e-3, 0.1, math.pi - 1e-6, math.pi], dtype=dtype,
                )
                vectors = torch.zeros(len(angles), 3, dtype=dtype)
                vectors[:, 2] = angles
                target = physical_state(_rotvec_to_quat(vectors))
                # This exact representation has a zero matrix skew part,
                # which previously made a pi error disappear entirely.
                target[-1, 3:7] = torch.tensor([0., 0., 0., 1.], dtype=dtype)
                predicted = torch.zeros_like(target)
                predicted[:, 3] = 1
                error = physical_state_transition_error(predicted, target)[:, 3:6]
                torch.testing.assert_close(error.norm(dim=-1), angles, rtol=tolerance, atol=tolerance)
                self.assertEqual(error[0].abs().sum().item(), 0)

    def test_small_angle_and_near_pi_loss_gradients_are_correct(self):
        for angle in (0., 1e-9, 1e-6, 1e-4, 1e-3, 0.1, math.pi - 1e-7):
            with self.subTest(angle=angle):
                theta = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
                vector = torch.stack((theta * 0, theta * 0, theta))
                target = physical_state(_rotvec_to_quat(vector))
                predicted = physical_state(torch.tensor([1., 0., 0., 0.], dtype=torch.float64))
                error = physical_state_transition_error(predicted, target)[3:6]
                error.square().sum().backward()
                self.assertTrue(torch.isfinite(theta.grad))
                torch.testing.assert_close(theta.grad, 2 * theta.detach(), rtol=1e-8, atol=1e-12)

    def test_rotation_error_is_invariant_to_quaternion_sign(self):
        generator = torch.Generator().manual_seed(22)
        predicted = physical_state(torch.randn(8, 4, generator=generator, dtype=torch.float64))
        target = physical_state(torch.randn(8, 4, generator=generator, dtype=torch.float64))
        error = physical_state_transition_error(predicted, target)
        predicted[:, 3:7] *= -1
        target[:, 3:7] *= -1
        torch.testing.assert_close(physical_state_transition_error(predicted, target), error)

    def test_rotation_error_gradcheck_near_zero_and_pi(self):
        identity = physical_state(torch.tensor([1., 0., 0., 0.], dtype=torch.float64))

        def rotation_error(vector):
            return physical_state_transition_error(identity, physical_state(_rotvec_to_quat(vector)))[3:6]

        for vector in ([0., 0., 0.], [1e-7, -2e-7, 3e-7], [0.1, 0.2, 0.3], [math.pi - 1e-3, 0., 0.]):
            with self.subTest(vector=vector):
                value = torch.tensor(vector, dtype=torch.float64, requires_grad=True)
                self.assertTrue(torch.autograd.gradcheck(rotation_error, (value,), eps=1e-6, atol=1e-6))

    def test_bptt_crosses_half_turn_without_discontinuous_orientation(self):
        model = StateSpaceWorldModel(WAMConfig(
            state_dim=71, action_dim=29, hidden=(), state_type="physical",
            learn_reward=False, learn_termination=False,
        )).double().requires_grad_(False)
        with torch.no_grad():
            # The first action produces a 0.01-rad local yaw increment.
            model.dynamics[0].weight[5, 71] = 0.01
        state = physical_state(torch.tensor([[0., 0., 0., 1.]], dtype=torch.float64))
        action = torch.ones(1, 29, dtype=torch.float64, requires_grad=True)
        for _ in range(12):
            state = model(state, action)["next_state"]
        expected = _quat_to_mat(_rotvec_to_quat(torch.tensor(
            [[0., 0., math.pi + 0.12]], dtype=torch.float64,
        )))
        torch.testing.assert_close(_quat_to_mat(state[:, 3:7]), expected, rtol=1e-9, atol=1e-9)
        _quat_to_mat(state[:, 3:7])[..., 1, 0].sum().backward()
        self.assertTrue(torch.isfinite(action.grad).all())
        self.assertGreater(action.grad.abs().sum().item(), 0.1)


if __name__ == "__main__":
    unittest.main()
