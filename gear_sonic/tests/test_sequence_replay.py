"""Runtime tests for temporal replay; runnable with Python's unittest module."""

import unittest

import torch

from gear_sonic.trl.world_model.sequence_replay import SequenceReplayBuffer


class SequenceReplayTests(unittest.TestCase):
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
