"""Source-frame fitting parity without changing dynamics output statistics."""

import math
from pathlib import Path
import unittest

from omegaconf import OmegaConf
import torch

from gear_sonic.trl.modules.wam import (
    StateSpaceWorldModel, WAMConfig, physical_state_transition_delta,
    physical_state_transition_error,
)


def model(**overrides):
    settings = dict(
        state_dim=71, action_dim=29, hidden=(), state_type="physical",
        learn_reward=False, learn_termination=False,
    )
    settings.update(overrides)
    return StateSpaceWorldModel(WAMConfig(**settings)).double()


def identity_state(batch=1):
    state = torch.zeros(batch, 71, dtype=torch.float64)
    state[:, 3] = 1
    return state


def source_fixture():
    """Hand-calculated SuperMimic networks.WorldModel.state_error fixture.

    Target R = Rz(pi/2) Rx(pi/2) cycles (x,y,z) to (y,z,x).
    Its heading is Rz(pi/2), while R.T cycles to (z,x,y).
    Prediction R is identity, so log(R_target.T R_pred) has three
    components -2*pi/(3*sqrt(3)). No production rotation helper is used.
    """
    predicted, target = identity_state(), identity_state()
    predicted[:, :3] = torch.tensor([1., 2., 3.])
    target[:, :3] = torch.tensor([4., 6., 8.])
    target[:, 3:7] = 0.5
    predicted[:, 7:10] = torch.tensor([10., 11., 12.])
    target[:, 7:10] = torch.tensor([3., 5., 7.])
    predicted[:, 10:13] = torch.tensor([-2., 10., 5.])
    target[:, 10:13] = torch.tensor([1., 2., 3.])
    predicted[:, 13:42], target[:, 13:42] = 0.4, 0.1
    predicted[:, 42:71], target[:, 42:71] = -0.2, 0.6
    angle = -2 * math.pi / (3 * math.sqrt(3))
    expected = torch.tensor([
        -4., 3., -5., angle, angle, angle, 6., 5., 7., 8., 2., -3.,
        *([0.3] * 29), *([-0.8] * 29),
    ], dtype=torch.float64)[None]
    return predicted, target, expected


class WAMErrorFrameTests(unittest.TestCase):
    def test_default_error_matches_hand_calculated_supermimic_fixture(self):
        predicted, target, expected = source_fixture()
        actual = physical_state_transition_error(predicted, target)
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=1e-8)
        self.assertEqual(model().error_frame, "target")

    def test_predicted_frame_retains_legacy_increment_convention(self):
        predicted, target, _ = source_fixture()
        # Prediction is identity, so all legacy vector errors are world-space
        # target minus prediction; its relative rotation has the opposite axis.
        angle = 2 * math.pi / (3 * math.sqrt(3))
        expected = torch.tensor([
            3., 4., 5., angle, angle, angle, -7., -6., -5., 3., -8., -2.,
            *([-0.3] * 29), *([0.8] * 29),
        ], dtype=torch.float64)[None]
        actual = physical_state_transition_error(predicted, target, error_frame="predicted")
        torch.testing.assert_close(actual, expected, atol=3e-8, rtol=1e-8)

    def test_anisotropic_fitting_loss_uses_target_frame(self):
        initial, target = identity_state(), identity_state()
        initial[:, 3] = initial[:, 6] = math.sqrt(0.5)
        target[:, 0] = 1
        states = torch.stack((initial, target), dim=1)
        losses = {}
        for frame, position_sse in (("target", 1.), ("predicted", 0.01)):
            dynamics = model(error_frame=frame)
            with torch.no_grad():
                std = torch.ones(70, dtype=torch.float64)
                std[1] = 10
                dynamics.delta_norm.var.copy_(std.square() - dynamics.delta_norm.eps**2)
            loss = dynamics.transition_loss(states, torch.zeros(1, 1, 29, dtype=torch.float64))["state_loss"]
            torch.testing.assert_close(
                loss, torch.tensor((position_sse + (math.pi / 2)**2) / 70, dtype=torch.float64),
                atol=1e-10, rtol=1e-9,
            )
            loss.backward()
            for parameter in dynamics.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(dynamics.dynamics[-1].bias.grad.abs().sum().item(), 0)
            losses[frame] = loss.detach()
        torch.testing.assert_close(losses["target"] - losses["predicted"], torch.tensor(0.99 / 70, dtype=torch.float64))

    def test_delta_normalizer_keeps_current_frame_and_forward_is_unchanged(self):
        next_state, current, expected = source_fixture()
        current, next_state = current.repeat(2, 1), next_state.repeat(2, 1)
        next_state[1, 0] += 2
        expected = expected.repeat(2, 1)
        expected[1, 1] -= 2  # Current-heading inverse maps +world-x to -y.
        torch.testing.assert_close(physical_state_transition_delta(current, next_state), expected, atol=3e-8, rtol=1e-8)
        models = [model(error_frame=frame) for frame in ("target", "predicted")]
        for dynamics in models:
            dynamics.update_normalizers(current, torch.zeros(2, 29), next_state)
            torch.testing.assert_close(dynamics.delta_norm.mean, expected.mean(0), atol=2e-7, rtol=1e-7)
            torch.testing.assert_close(dynamics.delta_norm.var, expected.var(0, unbiased=False), atol=1e-7, rtol=1e-7)
            with torch.no_grad():
                dynamics.dynamics[-1].bias.fill_(0.1)
        for name in ("mean", "var", "count"):
            torch.testing.assert_close(getattr(models[0].delta_norm, name), getattr(models[1].delta_norm, name))
        torch.testing.assert_close(
            models[0](current, torch.zeros(2, 29))["next_state"],
            models[1](current, torch.zeros(2, 29))["next_state"],
        )

    def test_error_frame_gradient_is_finite_at_identity_and_half_turn(self):
        for frame in ("target", "predicted"):
            for quaternion in ([1., 0., 0., 0.], [0., 0., 0., 1.]):
                with self.subTest(frame=frame, quaternion=quaternion):
                    predicted = identity_state()
                    predicted[:, 3:7] = torch.tensor(quaternion)
                    predicted.requires_grad_()
                    error = physical_state_transition_error(predicted, identity_state(), error_frame=frame)
                    error.square().sum().backward()
                    self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_unknown_frames_and_reductions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "error_frame"):
            model(error_frame="world")
        with self.assertRaisesRegex(ValueError, "loss_reduction"):
            model(loss_reduction="sum")
        with self.assertRaisesRegex(ValueError, "error_frame"):
            physical_state_transition_error(identity_state(), identity_state(), error_frame="world")

    def test_dynamics_config_explicitly_selects_new_frame_and_existing_scale(self):
        path = Path(__file__).resolve().parents[1] / "config/world_model/wam_dynamics.yaml"
        config = OmegaConf.load(path).cfg
        self.assertEqual(config.error_frame, "target")
        self.assertEqual(config.loss_reduction, "mean_valid")
        config.hidden = []
        settings = {key: value for key, value in OmegaConf.to_container(config).items() if not key.startswith("_")}
        dynamics = StateSpaceWorldModel(WAMConfig(**settings))
        self.assertEqual(dynamics.error_frame, "target")
        self.assertEqual(dynamics.loss_reduction, "mean_valid")


class WAMLossReductionTests(unittest.TestCase):
    def test_masked_window_reductions_have_explicit_source_scale(self):
        states = identity_state(2)[:, None].repeat(1, 4, 1)
        states[:, 1:, 13] = torch.tensor([1., 2., 3.])
        actions = torch.zeros(2, 3, 29, dtype=torch.float64)
        mask = torch.tensor([[True, True, True], [True, False, False]])
        # Valid errors are 1,4,9 in row 0 and 1 in row 1, each /70.
        for reduction, denominator in (("mean_valid", 4), ("sum_per_batch", 2)):
            with self.subTest(reduction=reduction):
                dynamics = model(loss_reduction=reduction)
                with torch.no_grad():
                    dynamics.delta_norm.var.fill_(1 - dynamics.delta_norm.eps**2)
                for masks in ({"mask": mask}, {"dynamics_mask": mask}):
                    actual = dynamics.transition_loss(states, actions, **masks)["state_loss"]
                    torch.testing.assert_close(actual, torch.tensor(15 / (70 * denominator), dtype=torch.float64))
                empty = dynamics.transition_loss(states, actions, mask=torch.zeros_like(mask))["state_loss"]
                torch.testing.assert_close(empty, torch.zeros((), dtype=torch.float64))
                empty.backward()
                self.assertTrue(torch.isfinite(dynamics.dynamics[-1].bias.grad).all())

    def test_optional_heads_use_the_same_masked_reduction(self):
        states = torch.zeros(2, 4, 1)
        actions = torch.zeros(2, 3, 1)
        mask = torch.tensor([[True, True, True], [True, False, False]])
        for reduction, denominator in (("mean_valid", 4), ("sum_per_batch", 2)):
            with self.subTest(reduction=reduction):
                dynamics = StateSpaceWorldModel(WAMConfig(
                    state_dim=1, action_dim=1, hidden=(), reward_hidden=(), term_hidden=(),
                    loss_reduction=reduction,
                ))
                with torch.no_grad():
                    for parameter in dynamics.parameters():
                        parameter.zero_()
                losses = dynamics.transition_loss(
                    states, actions, mask=mask, rewards=torch.full((2, 3), 2.), dones=torch.zeros(2, 3),
                )
                torch.testing.assert_close(losses["reward_loss"], torch.tensor(16 / denominator))
                torch.testing.assert_close(losses["termination_loss"], torch.tensor(4 * math.log(2) / denominator))


if __name__ == "__main__":
    unittest.main()
