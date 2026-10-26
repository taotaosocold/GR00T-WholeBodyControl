"""Regression checks for projected world-model states and masked diagnostics."""

import unittest

import torch
from torch.nn import functional as F

from gear_sonic.trl.modules.wam import StateSpaceWorldModel, WAMConfig


class WorldModelProjectionTests(unittest.TestCase):
    def make_model(self):
        model = StateSpaceWorldModel(
            WAMConfig(
                state_dim=2, action_dim=1, hidden=(), term_hidden=(),
                learn_reward=False,
            )
        )
        with torch.no_grad():
            # One reference feature and one physical feature both depend on
            # the action before the reference is replaced by its known value.
            model.dynamics[0].weight.zero_()
            model.dynamics[0].weight[:, -1] = torch.tensor([4.0, 3.0])
            model.dynamics[0].bias.zero_()
            model.termination_head[0].weight.copy_(torch.tensor([[2.0, 5.0]]))
            model.termination_head[0].bias.zero_()
        return model

    def test_forward_termination_uses_projected_state_and_physical_gradient(self):
        model = self.make_model().requires_grad_(False)
        actions = torch.ones(2, 1, requires_grad=True)

        def project(predicted):
            return torch.cat((torch.full_like(predicted[:, :1], 7.0), predicted[:, 1:]), dim=-1)

        output = model(torch.zeros(2, 2), actions, next_state_projector=project)
        torch.testing.assert_close(output["next_state"], torch.tensor([[7.0, 3.0], [7.0, 3.0]]))
        torch.testing.assert_close(output["termination_logits"], torch.full((2,), 29.0))
        output["termination_logits"].sum().backward()
        # Reference gradients are blocked, while the physical 3*a path remains.
        torch.testing.assert_close(actions.grad, torch.full_like(actions, 15.0))

    def test_fit_projects_once_per_step_and_blocks_reference_shortcut(self):
        model = self.make_model()
        states = torch.tensor([[[0.0, 0.0], [7.0, 3.0], [8.0, 6.0]]])
        actions = torch.ones(1, 2, 1, requires_grad=True)
        calls = []

        def project(predicted, step):
            calls.append(step)
            return torch.cat((states[:, step + 1, :1], predicted[:, 1:]), dim=-1)

        losses = model.transition_loss(
            states, actions, dones=torch.zeros(1, 2), state_projector=project,
        )
        self.assertEqual(calls, [0, 1])
        torch.testing.assert_close(losses["state_loss"], torch.tensor(0.0))
        expected = F.binary_cross_entropy_with_logits(
            torch.tensor([[29.0, 46.0]]), torch.zeros(1, 2),
        )
        torch.testing.assert_close(losses["termination_loss"], expected)
        losses["termination_loss"].backward()
        # The terminal classifier may train physical dynamics, but cannot
        # encode the label through a reference delta that gets projected away.
        torch.testing.assert_close(
            model.dynamics[0].weight.grad[0], torch.zeros_like(model.dynamics[0].weight.grad[0]),
        )
        torch.testing.assert_close(model.dynamics[0].bias.grad[0], torch.tensor(0.0))
        self.assertGreater(model.dynamics[0].weight.grad[1].abs().sum().item(), 0.0)
        self.assertTrue((actions.grad > 0).all())

    def test_reference_only_terminal_head_cannot_supply_policy_gradient(self):
        model = self.make_model().requires_grad_(False)
        with torch.no_grad():
            model.termination_head[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        actions = torch.ones(1, 1, requires_grad=True)
        output = model(
            torch.zeros(1, 2), actions,
            next_state_projector=lambda predicted: torch.cat(
                (torch.zeros_like(predicted[:, :1]), predicted[:, 1:]), dim=-1,
            ),
        )
        output["termination_logits"].sum().backward()
        torch.testing.assert_close(actions.grad, torch.zeros_like(actions))

    def test_no_valid_final_target_is_reported_as_missing_coverage(self):
        model = self.make_model()
        states = torch.zeros(2, 4, 2)
        states[:, 1:, 1] = 2.0
        losses = model.transition_loss(
            states, torch.zeros(2, 3, 1),
            mask=torch.tensor([[True, True, False], [True, False, False]]),
            dynamics_mask=torch.tensor([[True, False, False], [False, False, False]]),
        )
        self.assertGreater(losses["state_loss"].item(), 0.0)
        torch.testing.assert_close(losses["state_error_last"], torch.tensor(0.0))
        torch.testing.assert_close(losses["last_valid_fraction"], torch.tensor(0.0))
        torch.testing.assert_close(losses["dynamics_valid_fraction"], torch.tensor(1.0 / 6.0))
        torch.testing.assert_close(losses["valid_steps_mean"], torch.tensor(1.5))

    def test_partial_final_coverage_keeps_only_valid_error_targets(self):
        model = self.make_model()
        states = torch.zeros(2, 3, 2)
        states[:, 1:, 1] = 2.0
        losses = model.transition_loss(
            states, torch.zeros(2, 2, 1),
            dynamics_mask=torch.tensor([[True, True], [True, False]]),
        )
        torch.testing.assert_close(losses["state_error_last"], torch.tensor(2.0))
        torch.testing.assert_close(losses["last_valid_fraction"], torch.tensor(0.5))
        torch.testing.assert_close(losses["dynamics_valid_fraction"], torch.tensor(0.75))
        torch.testing.assert_close(losses["valid_steps_mean"], torch.tensor(2.0))


if __name__ == "__main__":
    unittest.main()
