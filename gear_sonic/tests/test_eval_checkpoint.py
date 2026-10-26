"""PPO/WAM evaluation metadata checks that do not require Isaac Lab."""

from types import SimpleNamespace
import unittest

from gear_sonic.trl.utils.checkpoint import get_checkpoint_step


class EvaluationCheckpointTests(unittest.TestCase):
    def test_original_ppo_state_object(self):
        self.assertEqual(get_checkpoint_step({"state": SimpleNamespace(global_step=1500)}), 1500)

    def test_serialized_ppo_state_mapping(self):
        self.assertEqual(get_checkpoint_step({"state": {"global_step": 2000}}), 2000)

    def test_wam_without_trainer_state(self):
        self.assertEqual(get_checkpoint_step({"wam_iteration": 1780}), 1780)

    def test_ppo_step_takes_precedence_including_zero(self):
        for state in (SimpleNamespace(global_step=0), {"global_step": 0}):
            with self.subTest(state=state):
                self.assertEqual(get_checkpoint_step({"state": state, "wam_iteration": 100}), 0)
        self.assertEqual(get_checkpoint_step({"state": {"global_step": 9}, "wam_iteration": 100}), 9)

    def test_missing_ppo_counter_can_fall_back_to_wam(self):
        for state in (None, {}, SimpleNamespace()):
            with self.subTest(state=state):
                self.assertEqual(get_checkpoint_step({"state": state, "wam_iteration": 0}), 0)

    def test_missing_counter_is_an_error(self):
        for checkpoint in ({}, {"state": {}}, {"state": SimpleNamespace()}):
            with self.subTest(checkpoint=checkpoint):
                with self.assertRaisesRegex(ValueError, "neither state.global_step nor wam_iteration"):
                    get_checkpoint_step(checkpoint)

    def test_invalid_counter_is_an_error(self):
        for invalid in (-1, True, False, 1.5, "7", None, float("nan"), float("inf")):
            for source in ("ppo", "wam"):
                with self.subTest(value=invalid, source=source):
                    checkpoint = (
                        {"state": {"global_step": invalid}, "wam_iteration": 99}
                        if source == "ppo" else {"wam_iteration": invalid}
                    )
                    with self.assertRaisesRegex(ValueError, "must be a nonnegative integer"):
                        get_checkpoint_step(checkpoint)

    def test_checkpoint_must_be_a_mapping(self):
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            get_checkpoint_step(None)


if __name__ == "__main__":
    unittest.main()
