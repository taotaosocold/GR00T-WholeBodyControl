"""Runtime tests for temporal replay; runnable with Python's unittest module."""

import unittest

import torch

from gear_sonic.trl.world_model.sequence_replay import SequenceReplayBuffer


class SequenceReplayTests(unittest.TestCase):
    def test_start_sampling_ignores_old_done_and_preserves_reference_indices(self):
        replay = SequenceReplayBuffer(2)
        for t in range(4):
            replay.add(torch.full((1, 1), float(t)), torch.zeros(1, 1), torch.zeros(1),
                       torch.full((1, 1), float(t + 1)), torch.ones(1, dtype=torch.bool),
                       reference_index=torch.tensor([[7, t]]))
        seed = replay.sample_starts(32, eligible=lambda indices: indices[..., 1] == 2)
        self.assertTrue(seed["done"].all())
        self.assertTrue((seed["state"] == 2).all())
        self.assertTrue((seed["reference_index"] == torch.tensor([7, 2])).all())
        restored = SequenceReplayBuffer()
        restored.load_state_dict(replay.state_dict())
        torch.testing.assert_close(restored.reference_indices(), replay.state_dict()["storage"]["reference_index"])

    def test_unobserved_impulse_masks_all_later_autoregressive_targets(self):
        replay = SequenceReplayBuffer(4)
        for t in range(4):
            replay.add(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1), torch.ones(1, 1),
                       torch.zeros(1, dtype=torch.bool), next_state_valid=torch.tensor([t != 1]))
        window = replay.sample(1, 4, fields=["mask", "dynamics_mask"])
        self.assertEqual(set(window), {"mask", "dynamics_mask"})
        self.assertTrue(window["mask"].all())
        self.assertEqual(window["dynamics_mask"].tolist(), [[True, False, False, False]])

    @staticmethod
    def add_step(buffer, t, done=None, terminated=None, n=3):
        state = (torch.arange(n) * 100 + t).float().unsqueeze(-1)
        next_state = state + 1
        if done is None:
            done = torch.zeros(n, dtype=torch.bool)
        buffer.add(state, state * 2, state[:, 0] * 3, next_state, done, terminated)

    def test_chronological_ring_windows_keep_environment(self):
        replay = SequenceReplayBuffer(capacity_steps=4)
        for t in range(7):
            self.add_step(replay, t)
        batch = replay.sample(100, 4)
        self.assertEqual(batch["states"].shape, (100, 5, 1))
        # Only the latest four steps 3..6 remain; every sampled state sequence
        # must stay on its environment and advance through the wrapped ring.
        torch.testing.assert_close(
            batch["states"][:, :, 0] % 100, torch.arange(3, 8).float().expand(100, -1)
        )
        torch.testing.assert_close(batch["actions"], batch["states"][:, :-1] * 2)
        torch.testing.assert_close(batch["rewards"], batch["states"][:, :-1, 0] * 3)
        self.assertTrue(batch["mask"].all())
        self.assertTrue(batch["dynamics_mask"].all())

    def test_done_masks_terminal_and_post_reset_transitions(self):
        replay = SequenceReplayBuffer(capacity_steps=4)
        for t in range(4):
            self.add_step(
                replay, t, torch.tensor([t == 1]), torch.tensor([t == 1]), n=1
            )
        batch = replay.sample(1, 4)
        self.assertEqual(batch["mask"].tolist(), [[True, True, False, False]])
        self.assertEqual(batch["dynamics_mask"].tolist(), [[True, False, False, False]])
        self.assertEqual(batch["terminated"].tolist(), [[False, True, False, False]])

    def test_truncation_blocks_dynamics_without_being_terminal(self):
        replay = SequenceReplayBuffer(capacity_steps=2)
        self.add_step(replay, 0, torch.tensor([True]), torch.tensor([False]), n=1)
        self.add_step(replay, 1, n=1)
        batch = replay.sample(1, 2)
        self.assertEqual(batch["mask"].tolist(), [[True, False]])
        self.assertFalse(batch["dynamics_mask"].any())
        self.assertFalse(batch["terminated"].any())

    def test_context_is_replayed_per_transition(self):
        replay = SequenceReplayBuffer(capacity_steps=4)
        for t in range(4):
            state = torch.tensor([[float(t)]])
            replay.add(
                state,
                state,
                state[:, 0],
                state + 1,
                torch.zeros(1, dtype=torch.bool),
                context=torch.tensor([[10.0 + t, 20.0 + t]]),
            )
        batch = replay.sample(1, 4)
        torch.testing.assert_close(
            batch["contexts"], torch.tensor([[[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]]])
        )

    def test_future_references_follow_states_across_ring_wrap_and_restore(self):
        original = SequenceReplayBuffer(capacity_steps=4)
        offsets = torch.arange(6, dtype=torch.float64) / 10

        def add(buffer, t):
            state = (torch.arange(3) * 100 + t).float().unsqueeze(-1)
            future = (state.double() + offsets).requires_grad_()
            buffer.add(
                state, state * 2, state[:, 0], state + 1,
                torch.zeros(3, dtype=torch.bool),
                future_reference=future, next_future_reference=future + 1,
            )

        for t in range(7):
            add(original, t)
        snapshot = original.state_dict()
        restored = SequenceReplayBuffer(capacity_steps=1)
        restored.load_state_dict(snapshot)
        add(original, 7)
        add(restored, 7)
        for key, expected in original.state_dict()["storage"].items():
            torch.testing.assert_close(restored.state_dict()["storage"][key], expected)
        batch = restored.sample(100, 4, device="cpu")
        self.assertEqual(batch["future_references"].shape, (100, 5, 6))
        self.assertEqual(batch["future_references"].dtype, torch.float32)
        self.assertEqual(batch["future_references"].device.type, "cpu")
        self.assertFalse(batch["future_references"].requires_grad)
        torch.testing.assert_close(
            batch["future_references"], batch["states"] + offsets.float(),
        )
        # Stored snapshots retain their own old reference timeline.
        self.assertEqual(snapshot["storage"]["future_reference"][0, 0, 0].item(), 3)

    def test_future_reference_inputs_are_paired_and_keep_one_schema(self):
        state = torch.zeros(2, 1)
        future = torch.zeros(2, 6)
        for current, following in [
            (future, None), (None, future),
            (future.unsqueeze(1), future.unsqueeze(1)),
            (future, torch.zeros(2, 5)),
            (torch.zeros(1, 6), torch.zeros(1, 6)),
            (torch.zeros(2, 0), torch.zeros(2, 0)),
        ]:
            with self.subTest(current=current, following=following), self.assertRaises(ValueError):
                SequenceReplayBuffer().add(
                    state, state, state[:, 0], state, torch.zeros(2, dtype=torch.bool),
                    future_reference=current, next_future_reference=following,
                )
        replay = SequenceReplayBuffer()
        replay.add(
            state, state, state[:, 0], state, torch.zeros(2, dtype=torch.bool),
            future_reference=future, next_future_reference=future,
        )
        with self.assertRaisesRegex(ValueError, "optional replay fields"):
            replay.add(state, state, state[:, 0], state, torch.zeros(2, dtype=torch.bool))

    def test_future_reference_snapshot_validation_and_legacy_compatibility(self):
        replay = SequenceReplayBuffer(capacity_steps=2)
        state = torch.zeros(2, 1)
        replay.add(
            state, state, state[:, 0], state, torch.zeros(2, dtype=torch.bool),
            future_reference=torch.zeros(2, 6), next_future_reference=torch.ones(2, 6),
        )
        snapshot = replay.state_dict()
        for missing in ["future_reference", "next_future_reference"]:
            broken = {**snapshot, "storage": dict(snapshot["storage"])}
            del broken["storage"][missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "both future"):
                SequenceReplayBuffer().load_state_dict(broken)
        for current, following in [
            (torch.zeros(1, 2), torch.zeros(1, 2)),
            (torch.zeros(1, 2, 6), torch.zeros(1, 2, 5)),
            (torch.zeros(1, 1, 6), torch.zeros(1, 1, 6)),
            (torch.zeros(1, 2, 0), torch.zeros(1, 2, 0)),
        ]:
            broken = {**snapshot, "storage": dict(snapshot["storage"])}
            broken["storage"].update(future_reference=current, next_future_reference=following)
            with self.subTest(shape=current.shape), self.assertRaisesRegex(ValueError, "future reference"):
                SequenceReplayBuffer().load_state_dict(broken)
        legacy = {**snapshot, "storage": dict(snapshot["storage"])}
        del legacy["storage"]["future_reference"]
        del legacy["storage"]["next_future_reference"]
        restored = SequenceReplayBuffer()
        restored.load_state_dict(legacy)
        self.assertNotIn("future_references", restored.sample(1, 1))

    def test_pre_reset_terminal_state_supervises_last_transition(self):
        replay = SequenceReplayBuffer(capacity_steps=3)
        for t in range(3):
            state = torch.tensor([[float(t)]])
            replay.add(
                state, state, state[:, 0], state + 1,
                torch.tensor([t == 1]), torch.tensor([t == 1]),
                next_state_valid=torch.tensor([True]),
            )
        batch = replay.sample(1, 3)
        self.assertEqual(batch["mask"].tolist(), [[True, True, False]])
        self.assertEqual(batch["dynamics_mask"].tolist(), [[True, True, False]])
        self.assertEqual(batch["states"][0, 2, 0].item(), 2)

    def test_legacy_snapshot_cannot_supervise_reset_state(self):
        replay = SequenceReplayBuffer(capacity_steps=2)
        self.add_step(replay, 0, torch.tensor([True]), n=1)
        snapshot = replay.state_dict()
        del snapshot["storage"]["next_state_valid"]
        restored = SequenceReplayBuffer()
        restored.load_state_dict(snapshot)
        self.assertFalse(restored.sample(1, 1)["dynamics_mask"].any())

    def test_snapshot_restores_cursor_and_continues_inserting(self):
        original = SequenceReplayBuffer(capacity_steps=3)
        for t in range(5):
            self.add_step(original, t)
        snapshot = original.state_dict()
        restored = SequenceReplayBuffer(capacity_steps=1)
        restored.load_state_dict(snapshot)
        self.assertEqual(len(restored), 3)
        self.assertEqual(restored.capacity_steps, 3)
        self.add_step(original, 5)
        self.add_step(restored, 5)
        expected = original.state_dict()
        actual = restored.state_dict()
        for key in expected["storage"]:
            torch.testing.assert_close(expected["storage"][key], actual["storage"][key])
        # Snapshots own their memory and cannot be altered by future writes.
        self.assertEqual(snapshot["storage"]["state"][0, 0, 0].item(), 2)

    def test_empty_partial_restore_and_input_validation(self):
        original = SequenceReplayBuffer(capacity_steps=8)
        restored = SequenceReplayBuffer()
        restored.load_state_dict(original.state_dict())
        self.assertEqual(len(restored), 0)
        with self.assertRaises(ValueError):
            restored.sample(1, 1)
        self.add_step(original, 0)
        restored.load_state_dict(original.state_dict())
        self.assertEqual(restored.sample(1, 1)["states"].shape, (1, 2, 1))
        with self.assertRaises(ValueError):
            restored.sample(1, 2)
        with self.assertRaises(ValueError):
            self.add_step(restored, 1, n=2)


if __name__ == "__main__":
    unittest.main()
