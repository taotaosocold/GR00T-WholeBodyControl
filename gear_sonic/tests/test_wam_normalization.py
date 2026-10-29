"""Physical output integration and loss weighting have independent scales."""

import unittest

import torch

from gear_sonic.trl.modules.wam import (
    RunningNorm, StateSpaceWorldModel, WAMConfig, _rotvec_to_quat,
)


def physical_model(**overrides):
    config = dict(
        state_dim=71, action_dim=29, hidden=(), state_type="physical",
        learn_reward=False, learn_termination=False,
    )
    config.update(overrides)
    model = StateSpaceWorldModel(WAMConfig(**config))
    model.delta_norm.var.zero_()
    model.delta_norm.mean.zero_()
    return model


def identity_state():
    state = torch.zeros(1, 71)
    state[:, 3] = 1
    return state


class WAMNormalizationTests(unittest.TestCase):
    def test_physical_output_std_is_not_raised_to_loss_floor(self):
        model = physical_model()
        torch.testing.assert_close(model.delta_norm.std(), torch.full((70,), 1e-4))
        with torch.no_grad():
            model.dynamics[0].weight[12, 71] = 1
        action = torch.ones(1, 29, requires_grad=True)
        first = model(identity_state(), action)["next_state"]
        self.assertAlmostEqual(first[0, 13].item(), 1e-4, places=9)
        with torch.no_grad():
            model.state_error_floor.mul_(100)
        second = model(identity_state(), action)["next_state"]
        torch.testing.assert_close(first, second)
        second[:, 13].sum().backward()
        self.assertAlmostEqual(action.grad[0, 0].item(), 1e-4, places=9)
        self.assertTrue(torch.isfinite(action.grad).all())

    def test_each_physical_error_group_uses_its_own_floor(self):
        model = physical_model()
        initial = identity_state()
        target = initial.clone()
        target[:, :3] = 0.003
        target[:, 3:7] = _rotvec_to_quat(torch.full((1, 3), 0.005))
        target[:, 7:10] = 0.02
        target[:, 10:13] = 0.05
        target[:, 13:42] = 0.003
        target[:, 42:71] = 0.05
        states = torch.stack((initial, target), 1)
        actions = torch.zeros(1, 1, 29)
        loss = model.transition_loss(states, actions)["state_loss"]
        torch.testing.assert_close(loss, torch.tensor(1.0), atol=1e-6, rtol=1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(model.dynamics[0].bias.grad).all())
        self.assertGreater(model.dynamics[0].bias.grad.abs().sum().item(), 0)
        with torch.no_grad():
            model.state_error_floor.mul_(2)
        scaled = model.transition_loss(states, actions)["state_loss"]
        torch.testing.assert_close(scaled, loss.detach() / 4)

    def test_measured_output_variance_can_exceed_loss_floor(self):
        model = physical_model()
        model.delta_norm.var.fill_(0.25)
        initial, target = identity_state(), identity_state()
        target[:, 13] = 0.5
        loss = model.transition_loss(
            torch.stack((initial, target), 1), torch.zeros(1, 1, 29),
        )["state_loss"]
        torch.testing.assert_close(loss, torch.tensor(1 / 70))

    def test_generic_normalizers_and_flat_mode_keep_existing_minimum_std(self):
        norm = RunningNorm(2)
        norm.var.zero_()
        torch.testing.assert_close(norm.std(), torch.full((2,), 0.01))
        flat = StateSpaceWorldModel(WAMConfig(state_dim=2, action_dim=1, hidden=()))
        flat.delta_norm.var.zero_()
        torch.testing.assert_close(flat.delta_norm.std(), torch.full((2,), 0.01))

    def test_physical_floor_configuration_requires_six_positive_finite_values(self):
        for value in ((0.1,), (0.,) * 6, (float("nan"),) * 6):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "six finite positive"):
                physical_model(state_error_floor=value)


if __name__ == "__main__":
    unittest.main()
