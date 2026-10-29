"""Separate gradient diagnostics preserve the joint actor/encoder BPTT update."""

import math
import unittest

import torch
from torch import nn

from gear_sonic.utils.wam_optimization import policy_gradient_metrics


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({"robot": nn.Linear(1, 1, bias=False)})
        self.decoders = nn.ModuleDict({"action": nn.Linear(1, 1, bias=False)})
        with torch.no_grad():
            self.encoders["robot"].weight.fill_(2)
            self.decoders["action"].weight.fill_(3)

    def forward(self, goal):
        latent = self.encoders["robot"](goal)
        return self.decoders["action"](latent), latent


class WAMOptimizationTests(unittest.TestCase):
    def make_graph(self):
        actor = TinyActor()
        world, critic = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            world.weight.fill_(5)
            critic.weight.fill_(7)
        world.requires_grad_(False)
        critic.requires_grad_(False)
        action, latent = actor(torch.tensor([[4.0]]))
        return_loss = -critic(world(action)).sum()
        auxiliary_loss = 0.5 * (latent - 1).square().sum()
        return actor, world, critic, return_loss, auxiliary_loss

    def test_separate_norms_match_analytic_gradients_and_preserve_joint_update(self):
        actor, world, critic, return_loss, auxiliary_loss = self.make_graph()
        original = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
        frozen = [world.weight.detach().clone(), critic.weight.detach().clone()]
        optimizer = torch.optim.SGD(actor.parameters(), lr=1e-5)
        # Existing gradients must also survive diagnostics unchanged.
        for parameter in actor.parameters():
            parameter.grad = torch.full_like(parameter, 123)
        metrics = policy_gradient_metrics(return_loss, auxiliary_loss, actor.named_parameters())
        self.assertAlmostEqual(metrics["actor/encoder_return_grad_norm"], 420)
        self.assertAlmostEqual(metrics["actor/decoder_return_grad_norm"], 280)
        self.assertAlmostEqual(metrics["actor/return_grad_norm"], math.hypot(420, 280), places=4)
        self.assertAlmostEqual(metrics["actor/encoder_aux_grad_norm"], 28)
        self.assertAlmostEqual(metrics["actor/decoder_aux_grad_norm"], 0)
        self.assertAlmostEqual(metrics["actor/aux_grad_norm"], 28)
        self.assertAlmostEqual(metrics["actor/aux_to_return_grad_ratio"], 28 / math.hypot(420, 280))
        for parameter in actor.parameters():
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 123))
            self.assertTrue(parameter.requires_grad)
        optimizer.zero_grad(set_to_none=True)
        (return_loss + auxiliary_loss).backward()
        torch.testing.assert_close(actor.encoders["robot"].weight.grad, torch.tensor([[-392.0]]))
        torch.testing.assert_close(actor.decoders["action"].weight.grad, torch.tensor([[-280.0]]))
        optimizer.step()
        for name, parameter in actor.named_parameters():
            self.assertFalse(torch.equal(parameter, original[name]))
        for module, expected in zip((world, critic), frozen):
            torch.testing.assert_close(module.weight, expected)
            self.assertIsNone(module.weight.grad)
            self.assertFalse(module.weight.requires_grad)

    def test_absent_auxiliary_gradient_is_zero_and_does_not_create_grad_fields(self):
        actor, _, _, return_loss, _ = self.make_graph()
        metrics = policy_gradient_metrics(return_loss, torch.zeros(()), actor.named_parameters())
        self.assertEqual(metrics["actor/aux_grad_norm"], 0)
        self.assertEqual(metrics["actor/encoder_aux_grad_norm"], 0)
        self.assertEqual(metrics["actor/decoder_aux_grad_norm"], 0)
        self.assertEqual(metrics["actor/aux_to_return_grad_ratio"], 0)
        self.assertTrue(all(parameter.grad is None for parameter in actor.parameters()))
        return_loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in actor.parameters()))

    def test_grouping_uses_whole_components_and_ignores_unused_frozen_parameters(self):
        encoder = nn.Parameter(torch.tensor(2.0))
        decoder = nn.Parameter(torch.tensor(3.0))
        unrelated = nn.Parameter(torch.tensor(4.0))
        unused = nn.Parameter(torch.tensor(5.0))
        frozen = nn.Parameter(torch.tensor(6.0), requires_grad=False)
        named = [
            ("actor.enc.weight", encoder), ("actor.dec.weight", decoder),
            ("encoder_scale", unrelated), ("encoder.unused", unused), ("decoder.frozen", frozen),
        ]
        metrics = policy_gradient_metrics(2 * encoder + 3 * decoder + 4 * unrelated, encoder, named)
        self.assertAlmostEqual(metrics["actor/return_grad_norm"], math.sqrt(29), places=6)
        self.assertEqual(metrics["actor/encoder_return_grad_norm"], 2)
        self.assertEqual(metrics["actor/decoder_return_grad_norm"], 3)
        self.assertEqual(metrics["actor/encoder_aux_grad_norm"], 1)
        self.assertEqual(metrics["actor/decoder_aux_grad_norm"], 0)
        self.assertTrue(all(parameter.grad is None for _, parameter in named))

    def test_flat_actor_and_empty_parameter_list_have_valid_zero_groups(self):
        parameter = nn.Parameter(torch.tensor(2.0))
        metrics = policy_gradient_metrics(parameter.square(), 0.0, [("weight", parameter)])
        self.assertEqual(metrics["actor/return_grad_norm"], 4)
        self.assertEqual(metrics["actor/encoder_return_grad_norm"], 0)
        self.assertEqual(metrics["actor/decoder_return_grad_norm"], 0)
        empty = policy_gradient_metrics(torch.tensor(0.0), torch.tensor(0.0), [])
        self.assertTrue(all(value == 0 for value in empty.values()))


if __name__ == "__main__":
    unittest.main()
