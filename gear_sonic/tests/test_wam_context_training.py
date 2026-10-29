"""Context, impulse and optimization contracts through the physical trainer."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from gear_sonic.tests import test_wam_physical_training as fixtures
from gear_sonic.utils.wam_context import get_push_versions


class CompleteContextEnv(fixtures.PhysicalVectorEnv):
    body_names = ["pelvis", "torso_link"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.context = torch.cat((self.context, torch.arange(6).float().expand(self.num_envs, -1)), -1)
        self.env._wam_context_schema = {
            "version": 1, "dim": self.context.shape[-1], "body_names": list(self.body_names),
        }


class ImpulseEnv(fixtures.PhysicalVectorEnv):
    def step(self, inputs):
        result = super().step(inputs)
        if self.env.common_step_counter == 2:
            get_push_versions(self.env, self.num_envs)
            self.env._wam_push_version[1] += 1
        return result


class WAMContextTrainingTests(unittest.TestCase):
    def make_trainer(self, directory, env_class=CompleteContextEnv, **settings):
        with patch.object(fixtures, "PhysicalVectorEnv", env_class):
            return fixtures.PhysicalTrainingTests().make_trainer(directory, **settings)

    def test_complete_context_conditions_both_models_without_changing_actor_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, complete_context=True, gradient_diagnostics_interval=1)
            self.assertEqual(trainer.context_dim, 39)
            self.assertEqual(trainer.critic_features.context_dim, 39)
            context_inputs = []
            original_step = trainer.world_step

            def step(state, action, context=None, **kwargs):
                context_inputs.append(context.detach().clone())
                return original_step(state, action, context=context, **kwargs)

            trainer.collect(3)
            trainer.fit_world_model()
            trainer.world_step = step
            model_before = [value.detach().clone() for value in trainer.world_model.parameters()]
            metrics = trainer.improve_policy()
            self.assertEqual(len(context_inputs), 2)
            for value in context_inputs:
                self.assertEqual(value.shape[-1], 39)
                torch.testing.assert_close(value[:, 33:], torch.arange(6).float().expand(len(value), -1))
            self.assertIn("actor/encoder_return_grad_norm", metrics)
            self.assertIn("actor/decoder_return_grad_norm", metrics)
            self.assertAlmostEqual(metrics["actor/return_loss"], -metrics["actor/imagined_return"], places=6)
            for before, after in zip(model_before, trainer.world_model.parameters()):
                torch.testing.assert_close(before, after)

    def test_impulse_breaks_fitting_window_without_ending_real_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, env_class=ImpulseEnv)
            trainer.collect(3)
            storage = trainer.replay._storage
            self.assertFalse(storage["done"][:3].any())
            replay_env = int((trainer.replay_env_ids == 1).nonzero()[0])
            self.assertEqual(storage["next_state_valid"][:3, replay_env].tolist(), [True, False, True])
            with patch("torch.randint", side_effect=[
                torch.zeros(1, dtype=torch.long), torch.tensor([replay_env]),
            ]):
                batch = trainer.replay.sample(1, 3)
            self.assertEqual(batch["mask"].tolist(), [[True, True, True]])
            self.assertEqual(batch["dynamics_mask"].tolist(), [[True, False, False]])

    def test_resume_rejects_same_dimension_with_different_context_meaning(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, complete_context=True)
            path = Path(directory) / "last.pt"
            trainer.save_checkpoint(path)
            other = self.make_trainer(directory, complete_context=True)
            other.context_schema = deepcopy(other.context_schema)
            other.context_schema["body_names"].reverse()
            with self.assertRaisesRegex(ValueError, "physical context schema"):
                other.load_checkpoint(path, resume=True)
            other.load_checkpoint(path, resume=False)

    def test_resume_rejects_same_dimension_with_changed_numerical_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, complete_context=True)
            path = Path(directory) / "last.pt"
            trainer.save_checkpoint(path)
            other = self.make_trainer(directory, complete_context=True, actor_return_reduction="mean")
            with self.assertRaisesRegex(ValueError, "normalization, noise or actor objective"):
                other.load_checkpoint(path, resume=True)


if __name__ == "__main__":
    unittest.main()
