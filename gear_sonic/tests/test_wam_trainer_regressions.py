"""Checkpoint, warmup and compact reference regressions without a simulator."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from gear_sonic.trl.trainer.wam_trainer import WAMTrainer
from gear_sonic.utils.wam_reference import WAMReferenceBank


class ReplayFixture:
    """Only trainer orchestration is exercised here, independently of replay IO."""

    def __init__(self, frames=0):
        self.frames = frames
        self.loaded = None
        self.discontinuous = False

    def __len__(self):
        return self.frames

    def storage_nbytes(self):
        return 0

    def load_state_dict(self, state):
        self.loaded = state
        self.frames = state["frames"]

    def mark_discontinuity(self):
        self.discontinuous = True


def checkpoint_trainer(error_frame="target"):
    trainer = WAMTrainer.__new__(WAMTrainer)
    trainer.device = torch.device("cpu")
    trainer.policy, trainer.world_model, trainer.critic = [nn.Linear(1, 1) for _ in range(3)]
    trainer.actor_optimizer = torch.optim.Adam(trainer.policy.parameters())
    trainer.world_optimizer = torch.optim.Adam(trainer.world_model.parameters())
    trainer.critic_optimizer = torch.optim.Adam(trainer.critic.parameters())
    trainer.critic_norm = nn.Linear(1, 1)
    trainer.layout = SimpleNamespace(
        state_specs=[("wam_state", 0, 71)], goal_specs=[("tokenizer", 0, 7)],
        tokenizer_specs=(("encoder_index", 0, 3, (3,)), ("command_z", 3, 7, (4,))),
    )
    trainer.context_keys, trainer.reference_keys, trainer.critic_keys = [], [], []
    trainer.settings = {}
    trainer.physical_wam = True
    trainer.critic_spec, trainer.context_schema = {"future_frames": 1}, None
    trainer.numerical_spec = {
        "delta_norm_eps": 0.0001, "state_error_floor": [0.1] * 6,
        "observation_noise": {}, "actor_return_reduction": "sum", "aux_loss_scale": 1.,
        "error_frame": error_frame, "loss_reduction": "mean_valid",
    }
    trainer.distributed = SimpleNamespace(world_size=1, rank=0)
    trainer.iteration, trainer.world_updates = 7, 11
    trainer.env = SimpleNamespace(num_envs=2)
    trainer.replay, trainer.reference_bank = ReplayFixture(), None
    trainer.compact_reference = True
    return trainer


def checkpoint_payload(trainer):
    payload = deepcopy(trainer._checkpoint_payload())
    payload["torch_rng"] = torch.get_rng_state()
    return payload


class WAMCheckpointRegressionTests(unittest.TestCase):
    def test_policy_only_load_maps_checkpoint_to_cpu_even_for_cuda_trainer(self):
        trainer = checkpoint_trainer()
        payload = checkpoint_payload(trainer)
        trainer.device = torch.device("cuda:0")
        with patch("gear_sonic.trl.trainer.wam_trainer.torch.load", return_value=payload) as load:
            trainer.load_checkpoint("policy.pt", resume=False)
        self.assertEqual(load.call_args.kwargs["map_location"], "cpu")

    def test_rank_runtime_and_shared_checkpoint_both_load_on_cpu(self):
        trainer = checkpoint_trainer()
        payload = checkpoint_payload(trainer)
        payload["wam_runtime_files"] = ["rank0.pt"]
        runtime = dict(rank=0, world_size=1, iteration=7, torch_rng=torch.get_rng_state())
        with patch("gear_sonic.trl.trainer.wam_trainer.torch.load", side_effect=[payload, runtime]) as load:
            trainer.load_checkpoint("shared.pt", resume=True)
        self.assertEqual(load.call_count, 2)
        self.assertEqual([call.kwargs["map_location"] for call in load.call_args_list], ["cpu", "cpu"])
        self.assertEqual(trainer.world_updates, 0)  # No replay was saved.

    def test_legacy_numerical_spec_resumes_only_with_explicit_predicted_frame(self):
        original = checkpoint_trainer("predicted")
        payload = checkpoint_payload(original)
        payload["wam_numerical_spec"].pop("error_frame")
        payload["wam_numerical_spec"].pop("loss_reduction")
        saved = deepcopy(payload["wam_numerical_spec"])
        compatible = checkpoint_trainer("predicted")
        with patch("gear_sonic.trl.trainer.wam_trainer.torch.load", return_value=payload):
            compatible.load_checkpoint("legacy.pt", resume=True)
        self.assertEqual(compatible.iteration, original.iteration)
        self.assertEqual(payload["wam_numerical_spec"], saved)  # Validation does not rewrite saved metadata.
        for frame, reduction in (("target", "mean_valid"), ("predicted", "sum_per_batch")):
            incompatible = checkpoint_trainer(frame)
            incompatible.numerical_spec["loss_reduction"] = reduction
            with self.subTest(frame=frame, reduction=reduction), patch(
                "gear_sonic.trl.trainer.wam_trainer.torch.load", return_value=payload,
            ), self.assertRaisesRegex(ValueError, "normalization, noise or actor objective"):
                incompatible.load_checkpoint("legacy.pt", resume=True)
        # Policy-only loading remains available across the objective change.
        with patch("gear_sonic.trl.trainer.wam_trainer.torch.load", return_value=payload):
            checkpoint_trainer("target").load_checkpoint("legacy.pt", resume=False)

    def test_saved_goal_schema_selects_legacy_full_or_compact_reference_on_resume(self):
        for compact in (False, True):
            with self.subTest(compact=compact):
                trainer = checkpoint_trainer()
                payload = checkpoint_payload(trainer)
                payload["wam_replay"] = {"frames": 2, "goal_dim": 3 if compact else 7}
                payload["wam_replay_env_ids"] = torch.tensor([0, 1])
                trainer.reference_bank = WAMReferenceBank(trainer.layout.tokenizer_specs)
                payload["wam_reference_bank"] = trainer.reference_bank.state_dict()
                if compact:
                    payload["wam_replay_spec"] = {"compact_reference": True}
                    trainer.compact_reference = False
                else:
                    payload.pop("wam_replay_spec")  # Checkpoints predating compact reference replay.
                with patch("gear_sonic.trl.trainer.wam_trainer.torch.load", return_value=payload):
                    trainer.load_checkpoint("replay.pt", resume=True)
                self.assertEqual(trainer.compact_reference, compact)
                self.assertTrue(trainer.replay.discontinuous)
                self.assertEqual(trainer.replay.loaded["goal_dim"], 3 if compact else 7)
                self.assertEqual(trainer.world_updates, 11)


class WAMWarmupRegressionTests(unittest.TestCase):
    def orchestration(self, *, frames=0, warmup_before_updates=True, iteration=0, total=1):
        trainer = checkpoint_trainer()
        trainer.device = torch.device("cpu")
        trainer.args = SimpleNamespace(num_total_batches=total)
        trainer.config = {}
        trainer.settings = dict(
            warmup_steps=4, warmup_before_updates=warmup_before_updates,
            collect_steps=1, window=1, horizon=1, warmup_world_updates=0,
            wm_only_iters=0, save_interval=0, learning_rate_decay_iters=0,
        )
        trainer.independent_imagination = True
        trainer.distributed = SimpleNamespace(all_finite=bool, reduce_metrics=lambda metrics: metrics, is_main=False)
        trainer.replay = ReplayFixture(frames)
        trainer.iteration, trainer.world_updates = iteration, 0
        trainer._last_collection_metrics = {}
        trainer.log_dir = Path("unused-test-directory")
        trainer._maybe_resample_motions = Mock()
        trainer.save_checkpoint = Mock()
        events = []

        def collect(steps):
            events.append(("collect", steps, len(trainer.replay)))
            trainer.replay.frames += steps
            return 0.

        def fit():
            events.append(("world", len(trainer.replay)))
            trainer.world_updates += 1
            return {}

        def actor():
            events.append(("actor", len(trainer.replay)))
            return {}

        trainer.collect, trainer.fit_world_model, trainer.improve_policy = collect, fit, actor
        return trainer, events

    def test_empty_replay_collects_full_warmup_before_either_optimizer_phase(self):
        trainer, events = self.orchestration()
        trainer.train()
        self.assertEqual(events, [("collect", 4, 0), ("collect", 1, 4), ("world", 5), ("actor", 5)])

    def test_empty_replay_restart_still_warms_up_before_world_updates(self):
        trainer, events = self.orchestration(iteration=7, total=8)
        trainer.train()
        self.assertEqual(events[0], ("collect", 4, 0))
        self.assertTrue(all(event[1] >= 4 for event in events if event[0] in {"world", "actor"}))

    def test_complete_restored_replay_does_not_repeat_warmup(self):
        trainer, events = self.orchestration(frames=4, iteration=7, total=8)
        trainer.train()
        self.assertEqual(events[0], ("collect", 1, 4))
        self.assertEqual([event[0] for event in events], ["collect", "world", "actor"])

    def test_partial_replay_is_filled_before_first_world_update(self):
        trainer, events = self.orchestration(frames=2)
        trainer.train()
        self.assertEqual(events, [("collect", 2, 2), ("collect", 1, 4), ("world", 5), ("actor", 5)])

    def test_legacy_ablation_and_completed_training_do_not_collect_prewarm(self):
        trainer, events = self.orchestration(warmup_before_updates=False)
        trainer.train()
        self.assertEqual(events, [("collect", 1, 0), ("world", 1)])
        trainer, events = self.orchestration(iteration=1, total=1)
        trainer.train()
        self.assertFalse(events)


class WAMCompactReferenceRegressionTests(unittest.TestCase):
    def test_noncontiguous_routing_reconstructs_full_seed_timeline_after_reload(self):
        specs = (
            ("command_z", 0, 1, (1,)), ("encoder_index", 1, 4, (3,)),
            ("command_multi_future_nonflat", 4, 6, (2,)), ("command_num_frames", 6, 7, (1,)),
        )
        bank = WAMReferenceBank(specs)
        states = torch.zeros(5, 71)
        states[:, 3] = 1
        states[:, 0] = torch.arange(5)
        clip_goals = torch.arange(35).reshape(5, 7).float()
        key = bank.register(states, clip_goals)
        seed = torch.tensor([[100., 0., 1., 0., 200., 300., 4.], [400., 0., 0., 1., 500., 600., 2.]])
        routing = bank.pack_routing(seed)
        torch.testing.assert_close(routing, torch.tensor([[0., 1., 0., 4.], [0., 0., 1., 2.]]))
        indices = torch.tensor([[key, 0], [key, 3]])
        full = bank.trajectory(indices, seed, 3, 2, 2, "cpu")
        restored = WAMReferenceBank(specs)
        restored.load_state_dict(deepcopy(bank.state_dict()))
        compact = restored.trajectory(indices, routing, 3, 2, 2, "cpu", routing_only=True)
        for name, value in full.items():
            torch.testing.assert_close(compact[name], value)
        torch.testing.assert_close(compact["goals"][0, :, 0], clip_goals[:4, 0])
        torch.testing.assert_close(compact["goals"][0, :, 6], torch.full((4,), 4.))
        self.assertEqual(compact["mask"][1].tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
