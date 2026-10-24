"""Numerical training-loop checks with real FSQ and a small vector environment."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from omegaconf import OmegaConf
import torch
from torch import nn
from vector_quantize_pytorch import FSQ

from gear_sonic.trl.trainer.wam_trainer import ObservationLayout, WAMTrainer


class Config(dict):
    __getattr__ = dict.__getitem__


class FSQPolicy(nn.Module):
    """Three modality encoders and the actual library's straight-through FSQ."""

    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([nn.Linear(3, 2) for _ in range(3)])
        self.quantizer = FSQ([4, 4])
        self.decoder = nn.Linear(6, 2)
        self.has_aux_loss = False
        self.max_rollout_history = 1

    @property
    def actor_module(self):
        return SimpleNamespace(tokenizer_obs_specs=[("encoder_index", 9, 12, (3,))])

    def forward(self, obs, is_training=False, **kwargs):
        del is_training, kwargs
        tokens = obs["tokenizer"]
        latent = sum(
            encoder(tokens[..., index * 3:index * 3 + 3]) * tokens[..., 9 + index:10 + index]
            for index, encoder in enumerate(self.encoders)
        )
        quantized, _ = self.quantizer(latent)
        return self.decoder(torch.cat([obs["actor_obs"], quantized], dim=-1))


class VectorEnv:
    def __init__(self):
        self.num_envs = 6
        self.config = Config(robot=Config(actions_dim=2), action_clip_value=1.0)
        self.action_transform_module = None
        self.t = 0
        self.x = torch.randn(self.num_envs, 4)
        self.goals = torch.randn(self.num_envs, 9)
        self.routing = torch.eye(3)[torch.arange(self.num_envs) % 3]

    def _obs(self):
        return {"actor_obs": self.x.clone(), "tokenizer": torch.cat([self.goals, self.routing], -1)}

    def reset_all(self):
        return self._obs()

    def step(self, payload):
        action = payload["actions"]
        assert (action.abs() <= 1.0).all()
        self.x = 0.8 * self.x + 0.1 * torch.cat([action, action], dim=-1)
        self.goals = 0.99 * self.goals
        rewards = 1 - self.x.square().mean(-1)
        self.t += 1
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        if self.t % 3 == 0:
            dones[0] = True
            self.x[0] = torch.randn(4) * 3  # Auto-reset must not train dynamics.
        return self._obs(), rewards, dones, {"time_outs": dones.clone()}


def make_original_fsq_actor():
    """Build the repository's actual actor/tokenizer from small Hydra configs."""
    from gear_sonic.trl.modules.actor_critic_modules import Actor

    dims = {"g1_goal": [3], "teleop_goal": [3], "smpl_goal": [3], "encoder_index": [3]}
    env_config = OmegaConf.create(
        {
            "robot": {"actions_dim": 2, "algo_obs_dim_dict": {"actor_obs": 4}},
            "obs": {
                "group_obs_dims": {"tokenizer": dims},
                "group_obs_names": {"tokenizer": list(dims)},
            },
        }
    )
    encoder_cfg = {
        name: {
            "inputs": [name + "_goal"],
            "params": {
                "_target_": "gear_sonic.trl.modules.base_module.BaseModule",
                "input_dim": None,
                "output_dim": None,
                "num_output_temporal_dims": 1,
                "module_config_dict": {
                    "layer_config": {"type": "MLP", "hidden_dims": [12], "activation": "SiLU"}
                },
            },
        }
        for name in ["g1", "teleop", "smpl"]
    }
    backbone = OmegaConf.create(
        {
            "_target_": "gear_sonic.trl.modules.universal_token_modules.UniversalTokenModule",
            "proprioception_features": ["actor_obs"],
            "num_fsq_levels": 2,
            "fsq_level_list": [4, 4],
            "max_num_tokens": 1,
            "num_future_frames": 1,
            "quantizer": {"_target_": "vector_quantize_pytorch.FSQ"},
            "encoder_sample_probs": {"g1": 1.0, "teleop": 1.0, "smpl": 1.0},
            "encoders": encoder_cfg,
            "decoders": {
                "g1_dyn": {
                    "inputs": ["token_flattened", "proprioception"],
                    "outputs": ["action"],
                    "has_temporal_dim": False,
                    "params": {
                        "_target_": "gear_sonic.trl.modules.base_module.BaseModule",
                        "input_dim": None,
                        "output_dim": None,
                        "module_config_dict": {
                            "layer_config": {
                                "type": "MLP", "hidden_dims": [12], "activation": "SiLU"
                            }
                        },
                    },
                }
            },
        }
    )
    return Actor(
        env_config,
        OmegaConf.create({"init_noise_std": 0.05}),
        backbone,
        input_obs_dict=True,
        has_aux_loss=True,
    )


class TrainingTests(unittest.TestCase):
    def make_trainer(self, directory, policy=None, checkpoint=None, resume=False):
        return WAMTrainer(
            args=SimpleNamespace(num_total_batches=1), config=Config(gamma=0.99, lam=0.95),
            env=VectorEnv(), model=policy or FSQPolicy(), log_dir=directory,
            settings={
                "window": 3, "horizon": 3, "batch_size": 24,
                "world_updates": 2, "critic_epochs": 2, "replay_steps": 16,
                "collect_steps": 6, "warmup_world_updates": 1, "save_replay": True,
                "exogenous_tokenizer_features": ["g1_goal", "teleop_goal", "smpl_goal"],
            },
            world_model_config={
                "_target_": "gear_sonic.trl.modules.wam.StateSpaceWorldModel",
                "cfg": {"hidden": [16, 16], "term_hidden": [8], "reward_hidden": [8]},
            }, checkpoint=checkpoint, resume=resume,
        )

    def test_collection_fit_imagination_and_checkpoint(self):
        torch.manual_seed(13)
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.collect(6)
            fit_metrics = trainer.fit_world_model()
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in fit_metrics.values()))
            before = [encoder.weight.detach().clone() for encoder in trainer.policy.encoders]
            model_before = {key: value.clone() for key, value in trainer.world_model.state_dict().items()}
            metrics = trainer.improve_policy()
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
            # All modalities get BPTT gradients through the real FSQ bottleneck.
            for old, encoder in zip(before, trainer.policy.encoders):
                self.assertFalse(torch.equal(old, encoder.weight))
            for key, value in trainer.world_model.state_dict().items():
                self.assertTrue(torch.equal(value, model_before[key]))
            checkpoint = Path(directory) / "resume.pt"
            trainer.save_checkpoint(checkpoint)
            restored = self.make_trainer(directory, checkpoint=checkpoint, resume=True)
            for key, value in trainer.policy.state_dict().items():
                self.assertTrue(torch.equal(value, restored.policy.state_dict()[key]))
            restored.collect(3)
            restored.fit_world_model()
            restored.improve_policy()

    def test_full_training_entry_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.train()
            self.assertEqual(trainer.iteration, 1)
            self.assertTrue((Path(directory) / "last.pt").exists())

    def test_layout_copies_categorical_routing(self):
        env = VectorEnv()
        obs = env.reset_all()
        layout = ObservationLayout(obs, ["actor_obs", "tokenizer"], [("encoder_index", 9, 12, (3,))])
        seed = layout.pack(obs)
        prediction = torch.randn_like(seed, requires_grad=True)
        projected = layout.project(prediction, seed)
        self.assertTrue(torch.equal(projected[:, -3:], seed[:, -3:]))
        projected.sum().backward()
        self.assertEqual(prediction.grad[:, -3:].abs().sum(), 0)

    def test_reference_timeline_is_exogenous_to_actions(self):
        env = VectorEnv()
        obs = env.reset_all()
        layout = ObservationLayout(
            obs, ["actor_obs", "tokenizer"],
            [("command_z", 0, 1, (1,)), ("encoder_index", 9, 12, (3,))],
        )
        seed = layout.pack(obs)
        reference = seed.clone()
        reference[:, 4] += 3.0
        prediction = torch.randn_like(seed, requires_grad=True)
        projected = layout.project(prediction, seed, reference)
        self.assertTrue(torch.equal(projected[:, 4], reference[:, 4]))
        projected.sum().backward()
        self.assertEqual(prediction.grad[:, 4].abs().sum(), 0)
        self.assertGreater(prediction.grad[:, :4].abs().sum(), 0)

    def test_nonfinite_update_does_not_mutate_parameters(self):
        model = nn.Linear(2, 1)
        before = model.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters())
        loss = model(torch.ones(1, 2)).sum() * float("nan")
        with self.assertRaises(FloatingPointError):
            WAMTrainer.optimizer_step(loss, optimizer, model.parameters())
        self.assertTrue(torch.equal(before, model.weight))

    def test_original_actor_universal_token_fsq_end_to_end(self):
        torch.manual_seed(31)
        policy = make_original_fsq_actor()
        self.assertIsInstance(policy.actor_module.quantizer, FSQ)
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory, policy=policy)
            seed = trainer.layout.pack(trainer.obs)
            output = policy.actor_module(trainer.layout.unpack(seed), return_dict=True)
            self.assertEqual(output["action_mean"].shape, (6, 1, 2))
            self.assertEqual(output["full_tokens"].shape, (6, 1, 1, 2))
            self.assertEqual(output["full_latent"].shape, (6, 1, 1, 2))
            # Establish that the tokenizer really routes all three modalities.
            self.assertEqual(set(output["encoded_tokens"]), {"g1", "teleop", "smpl"})
            trainer.collect(6)
            trainer.fit_world_model()
            old_encoder_weights = {
                name: next(encoder.parameters()).detach().clone()
                for name, encoder in policy.actor_module.encoders.items()
            }
            world_before = {
                name: value.clone() for name, value in trainer.world_model.state_dict().items()
            }
            metrics = trainer.improve_policy()
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
            for name, encoder in policy.actor_module.encoders.items():
                self.assertFalse(torch.equal(old_encoder_weights[name], next(encoder.parameters())))
            for name, value in trainer.world_model.state_dict().items():
                self.assertTrue(torch.equal(world_before[name], value))
            path = Path(directory) / "original_actor.pt"
            trainer.save_checkpoint(path)
            restored = self.make_trainer(
                directory, policy=make_original_fsq_actor(), checkpoint=path, resume=True
            )
            for name, value in policy.state_dict().items():
                self.assertTrue(torch.equal(value, restored.policy.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()
