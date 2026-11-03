"""Runtime tests for temporal replay; runnable with Python's unittest module."""

import unittest

import torch

from gear_sonic.trl.world_model.sequence_replay import SequenceReplayBuffer


class SequenceReplayTests(unittest.TestCase):
    def test_field_selection_never_indexes_unrequested_large_storage(self):
        replay = SequenceReplayBuffer(2)
        replay.add(torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2), torch.ones(2, 1),
                   torch.zeros(2, dtype=torch.bool), goal=torch.zeros(2, 100),
                   next_goal=torch.ones(2, 100), actor_state=torch.zeros(2, 80),
                   next_actor_state=torch.ones(2, 80))

        class ForbiddenRead:
            def __getitem__(self, index):
                raise AssertionError("unrequested replay tensor was indexed")

        for key in ("goal", "next_goal", "actor_state", "next_actor_state"):
            replay._storage[key] = ForbiddenRead()
        fields = ["states", "actions", "rewards", "terminated", "mask", "dynamics_mask"]
        batch = replay.sample(3, 1, fields=fields)
        self.assertEqual(list(batch), fields)
        self.assertEqual(replay.sample(3, 1, fields=[]), {})
        with self.assertRaisesRegex(ValueError, "missing requested"):
            replay.sample(1, 1, fields=["contexts"])

    @staticmethod
    def _compact_pair(capacity, histories=(4, 3, 1)):
        layout, start = [], 0
        for index, history in enumerate(histories):
            dim = index + 1
            layout.append((str(index), start, start + history * dim, history, dim))
            start += history * dim
        return (SequenceReplayBuffer(capacity),
                SequenceReplayBuffer(capacity, actor_history_layout=layout), layout, start)

    def _assert_sample_equivalence(self, full, compact):
        for fields in (None, ["actor_states"], ["states", "dynamics_mask"]):
            a = full.sample(29, len(full), fields=fields, generator=torch.Generator().manual_seed(17))
            b = compact.sample(29, len(compact), fields=fields, generator=torch.Generator().manual_seed(17))
            self.assertEqual(set(a), set(b))
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
        torch.manual_seed(31)
        a = full.sample_starts(73)
        torch.manual_seed(31)
        b = compact.sample_starts(73)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)

    def test_compact_history_preserves_noise_resets_halo_and_discontinuities(self):
        # Capacity below history length exercises the halo rather than merely
        # retrieving all the history from ordinary retained transitions.
        full, compact, layout, width = self._compact_pair(2, histories=(5, 3, 1))
        torch.manual_seed(9)
        actors = torch.randn(3, width)
        for time in range(16):
            if time in (4, 11):
                full.mark_discontinuity()
                compact.mark_discontinuity()
                actors = torch.randn_like(actors)  # preserve this actual prefix
            following = torch.cat([
                torch.cat((actors[:, start + dim:end], torch.randn(3, dim)), dim=-1)
                for _, start, end, _, dim in layout
            ], dim=-1)
            done = torch.tensor([time % 3 == 0, time % 5 == 0, False])
            if done[0]:
                # Production resets repeat one noisy observation per term.
                for _, start, end, history, dim in layout:
                    following[0, start:end] = torch.randn(dim).repeat(history)
            if done[1]:
                # Also accept a caller's non-repeated initial noisy history.
                following[1] = torch.randn(width)
            state = torch.arange(3)[:, None].float() + time * 10
            for replay in (full, compact):
                replay.add(state, state, state[:, 0], state + 1, done,
                           actor_state=actors, next_actor_state=following)
            self._assert_sample_equivalence(full, compact)
            actors = following
        # Sealing an existing transition must not reinterpret its already
        # captured next history as a reset observation.
        full.mark_discontinuity()
        compact.mark_discontinuity()
        self._assert_sample_equivalence(full, compact)

    def test_compact_history_checkpoint_roundtrip_continues_exactly(self):
        full, compact, layout, width = self._compact_pair(3)
        torch.manual_seed(13)
        actors = torch.randn(2, width)
        for time in range(10):
            following = torch.cat([
                torch.cat((actors[:, start + dim:end], torch.randn(2, dim)), dim=-1)
                for _, start, end, _, dim in layout
            ], dim=-1)
            state = torch.full((2, 1), float(time))
            for replay in (full, compact):
                replay.add(state, state, state[:, 0], state + 1, torch.zeros(2, dtype=torch.bool),
                           actor_state=actors, next_actor_state=following)
            if time == 6:
                snapshot = compact.state_dict()
                restored = SequenceReplayBuffer(1)
                restored.load_state_dict(snapshot)
                self._assert_sample_equivalence(full, restored)
                compact = restored
            actors = following
        self._assert_sample_equivalence(full, compact)
        self.assertEqual(snapshot["actor_history"]["time"], 7)
        # A legacy full-history checkpoint overrides a new compact preference.
        compact.load_state_dict(full.state_dict())
        self.assertIsNone(compact.actor_history_layout)
        legacy_restored = SequenceReplayBuffer()
        legacy_restored.load_state_dict(full.state_dict())
        self._assert_sample_equivalence(legacy_restored, compact)

    def test_compact_memory_scales_with_frames_and_keeps_no_noise_resampling(self):
        full, compact, layout, width = self._compact_pair(64, histories=(10, 10, 10))
        torch.manual_seed(4)
        actors = torch.cat([torch.randn(4, dim).repeat(1, history)
                            for _, _, _, history, dim in layout], dim=-1)
        for time in range(70):
            following = torch.cat([
                torch.cat((actors[:, start + dim:end], torch.randn(4, dim)), dim=-1)
                for _, start, end, _, dim in layout
            ], dim=-1)
            state = torch.full((4, 1), float(time))
            for replay in (full, compact):
                replay.add(state, state, state[:, 0], state + 1, torch.zeros(4, dtype=torch.bool),
                           actor_state=actors, next_actor_state=following)
            actors = following
        self.assertLess(compact.storage_nbytes(), full.storage_nbytes() / 3)
        self.assertEqual(compact._storage["actor_state"].shape[-1], 6)
        rng = torch.get_rng_state().clone()
        compact._restore_actor_history(torch.tensor([6, 69]), torch.tensor([0, 3]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self._assert_sample_equivalence(full, compact)

    def test_compact_snapshot_preserves_active_noisy_segment_prefixes(self):
        full, compact, layout, width = self._compact_pair(2, histories=(5, 3, 1))
        torch.manual_seed(27)
        actors = torch.randn(2, width)
        for time in range(4):
            following = torch.randn(2, width)  # deliberately nonstandard prefix
            state = torch.full((2, 1), float(time))
            for replay in (full, compact):
                replay.add(state, state, state[:, 0], state + 1, torch.ones(2, dtype=torch.bool),
                           actor_state=actors, next_actor_state=following)
            actors = following
        full.mark_discontinuity()
        compact.mark_discontinuity()
        snapshot = compact.state_dict()
        restored = SequenceReplayBuffer()
        restored.load_state_dict(snapshot)
        self._assert_sample_equivalence(full, restored)
        self.assertTrue(restored._history_discontinuous)
        # Checkpoints own the seed/halo tensors too, not just the main ring.
        saved_prefix = next(iter(snapshot["actor_history"]["seeds"].values()))[1].clone()
        for _, prefix in compact._history_seeds.values():
            prefix.fill_(999)
        torch.testing.assert_close(next(iter(snapshot["actor_history"]["seeds"].values()))[1], saved_prefix)
        fresh = torch.randn_like(actors)
        following = torch.cat([
            torch.cat((fresh[:, start + dim:end], torch.randn(2, dim)), dim=-1)
            for _, start, end, _, dim in layout
        ], dim=-1)
        for replay in (full, restored):
            replay.add(state + 1, state, state[:, 0], state + 2, torch.zeros(2, dtype=torch.bool),
                       actor_state=fresh, next_actor_state=following)
        self._assert_sample_equivalence(full, restored)

    def test_compact_nonterminal_next_history_reset_starts_a_new_prefix(self):
        # Mixed history lengths, a ring shorter than the longest term, and a
        # nonterminal history reset previously lost the reset's prefix when
        # that next history became a later current history.
        layout = [("a", 0, 3, 1, 3), ("b", 3, 9, 3, 2), ("c", 9, 49, 10, 4)]
        full = SequenceReplayBuffer(7)
        compact = SequenceReplayBuffer(7, actor_history_layout=layout)
        torch.manual_seed(930)
        actors = torch.randn(4, 49)
        for time in range(90):
            if time % 17 == 3:
                actors = torch.randn(4, 49)
            following = torch.cat([
                torch.cat((actors[:, start + dim:end], torch.randn(4, dim)), dim=-1)
                for _, start, end, _, dim in layout
            ], dim=-1)
            done = torch.rand(4) < 0.22
            following[done] = torch.randn(int(done.sum()), 49)
            if time % 11 == 2:
                following[0] = torch.randn(49)  # need not coincide with done
            state, action, reward, next_state = torch.randn(4, 5), torch.randn(4, 2), torch.randn(4), torch.randn(4, 5)
            for replay in (full, compact):
                replay.add(state, action, reward, next_state, done, done,
                           actor_state=actors, next_actor_state=following)
            actors = following
            window = min(len(full), 5)
            a = full.sample(19, window, generator=torch.Generator().manual_seed(133 + time))
            b = compact.sample(19, window, generator=torch.Generator().manual_seed(133 + time))
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=f"{time}: {key}")
            torch.manual_seed(time + 991)
            b = compact.sample_starts(19)
            # Legacy restore normalizes its physical ring cursor, whereas a
            # compact snapshot preserves absolute time for the halo. Compare
            # sampled transitions by their unique state, not physical row RNG.
            stored_states = full._storage["state"][:len(full)].flatten(0, 1)
            matches = (b["state"][:, None] == stored_states[None]).all(-1)
            self.assertTrue((matches.sum(-1) == 1).all())
            selected = matches.long().argmax(-1)
            for key, value in b.items():
                expected = full._storage[key][:len(full)].flatten(0, 1)[selected]
                torch.testing.assert_close(expected, value, rtol=0, atol=0, msg=f"start {time}: {key}")
            if time % 23 == 0:
                restored_full, restored_compact = SequenceReplayBuffer(7), SequenceReplayBuffer(7)
                restored_full.load_state_dict(full.state_dict())
                restored_compact.load_state_dict(compact.state_dict())
                full, compact = restored_full, restored_compact
            if time % 19 == 0:
                full.mark_discontinuity()
                compact.mark_discontinuity()

    def test_compact_history_rejects_invalid_layout_and_preserves_single_frame_terms(self):
        for layout in ([], [("x", 0, 3, 0, 3)], [("x", 1, 4, 1, 3)], [("x", 0, 4, 2, 3)]):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                SequenceReplayBuffer(actor_history_layout=layout)
        full, compact, _, width = self._compact_pair(2, histories=(1, 1, 1))
        for time in range(4):
            actors, following = torch.randn(2, width), torch.randn(2, width)
            state = torch.full((2, 1), float(time))
            for replay in (full, compact):
                replay.add(state, state, state[:, 0], state + 1, torch.ones(2, dtype=torch.bool),
                           actor_state=actors, next_actor_state=following)
        self._assert_sample_equivalence(full, compact)

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
