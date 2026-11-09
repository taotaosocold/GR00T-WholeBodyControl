"""CPU tests for the MimicLite WAM termination profile."""

import unittest

import torch

from gear_sonic.utils.wam_termination import (
    mimiclite_error_terms,
    mimiclite_soft_termination,
)
from gear_sonic.trl.trainer.wam_trainer import WAMTrainer


class TinyKinematics:
    tracked_bodies = ("waist_yaw_link", "hand")

    def __call__(self, state):
        p = state[..., :3]
        # The waist is deliberately offset from the floating root so the test
        # catches accidental use of state[:3] instead of the mapped body.
        waist = p + state.new_tensor([0.0, 0.0, 0.5])
        hand = waist + state.new_tensor([0.4, 0.0, 0.0])
        hand = hand + torch.cat((state[..., 13:14], torch.zeros_like(state[..., 13:15])), dim=-1)
        positions = torch.stack((waist, hand), dim=-2)
        eye = torch.eye(3, dtype=state.dtype, device=state.device)
        rotations = eye.expand(*state.shape[:-1], 2, 3, 3).clone()
        return positions, rotations, torch.zeros_like(positions), torch.zeros_like(positions)


class MimicLiteTerminationTests(unittest.TestCase):
    def test_finite_guard_isolates_only_bad_rows(self):
        values = [torch.tensor([[1.0, 2.0], [float("nan"), 0.0]])]
        bad = WAMTrainer._nonfinite_rows(values, 2)
        self.assertEqual(bad.tolist(), [False, True])

    def test_counter_requires_consecutive_steps_and_resets(self):
        counters = torch.zeros(1, 4)
        errors = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
        for step in range(24):
            counters, failure, timeout, done = mimiclite_soft_termination(
                errors, counters, min_steps=(25, 25, 5, 5), error_temperature=1e-3,
                temperature=1e-3,
            )
            self.assertLess(float(timeout), 0.5, step)
        counters, failure, timeout, done = mimiclite_soft_termination(
            errors, counters, min_steps=(25, 25, 5, 5), error_temperature=1e-3,
            temperature=1e-3,
        )
        self.assertEqual(failure.shape, torch.Size([1]))
        self.assertEqual(timeout.shape, torch.Size([1]))
        self.assertEqual(done.shape, torch.Size([1]))
        self.assertGreater(float(timeout), 0.9)
        counters, *_ = mimiclite_soft_termination(
            torch.zeros_like(errors), counters, min_steps=(25, 25, 5, 5),
            error_temperature=1e-3, temperature=1e-3,
        )
        self.assertLess(float(counters[0, 0]), 0.1)

    def test_invalid_padding_does_not_advance_counter(self):
        counters = torch.ones(2, 4)
        errors = torch.full((2, 4), 2.0)
        counters2, failure, timeout, done = mimiclite_soft_termination(
            errors, counters, min_steps=(2, 2, 2, 2), error_temperature=1e-3,
            temperature=1e-3, valid=torch.tensor([True, False]),
        )
        torch.testing.assert_close(counters2[1], counters[1])
        torch.testing.assert_close(done[1], torch.zeros(()))

    def test_root_uses_mapped_body_and_local_body_errors(self):
        state = torch.zeros(1, 71)
        reference = state.clone()
        state[:, 2] = 10.0
        errors = mimiclite_error_terms(state, reference, TinyKinematics())
        self.assertGreater(float(errors[0, 0]), 9.0)
        state[:, 2] = 0.0
        state[:, 13] = 1.0
        errors = mimiclite_error_terms(state, reference, TinyKinematics())
        self.assertGreater(float(errors[0, 2]), 0.9)


if __name__ == "__main__":
    unittest.main()
