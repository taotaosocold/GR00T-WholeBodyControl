"""CPU checks of SMPL/robot sampling by physical time, including rotations."""

import copy
import math
import unittest

import numpy as np
import torch

from gear_sonic.utils.motion_lib.paired_resampling import resample_smpl_to_robot_window


def _smpl(frames, fps):
    times = np.arange(frames, dtype=np.float32) / fps
    pose = np.zeros((frames, 72), dtype=np.float32)
    pose[:, 0] = times * 0.4
    return {
        "fps": fps,
        "pose_aa": pose,
        "smpl_joints": np.broadcast_to(times[:, None, None], (frames, 24, 3)).copy(),
        "transl": np.broadcast_to(times[:, None], (frames, 3)).copy(),
    }


class PairedResamplingTest(unittest.TestCase):
    def assertTimes(self, result, times):
        expected = torch.as_tensor(times, dtype=torch.float32)
        torch.testing.assert_close(result["transl"][:, 0], expected)
        torch.testing.assert_close(result["smpl_joints"][:, 0, 0], expected)
        torch.testing.assert_close(result["pose_aa"][:, 0], expected * 0.4, atol=2e-6, rtol=1e-5)

    def test_30_to_50_preserves_duration_and_does_not_modify_input(self):
        source = _smpl(31, 30)
        original = copy.deepcopy(source)
        result = resample_smpl_to_robot_window(source, 0, 30, 50, 50)
        self.assertTimes(result, np.arange(50) / 50)
        for key in ("pose_aa", "smpl_joints", "transl"):
            np.testing.assert_array_equal(source[key], original[key])
            self.assertEqual(len(result[key]), 50)

    def test_60_to_50(self):
        result = resample_smpl_to_robot_window(_smpl(61, 60), 0, 60, 50, 50)
        self.assertTimes(result, np.arange(50) / 50)

    def test_mixed_source_fps_uses_seconds_rather_than_robot_frame_indices(self):
        result = resample_smpl_to_robot_window(_smpl(51, 50), 9, 30, 50, 35)
        self.assertTimes(result, 0.3 + np.arange(35) / 50)

    def test_nonzero_crop_samples_the_same_source_time_for_all_fields(self):
        result = resample_smpl_to_robot_window(_smpl(61, 30), 15, 30, 50, 25)
        self.assertTimes(result, 0.5 + np.arange(25) / 50)

    def test_same_rate_includes_last_endpoint_and_retains_axis_angle_values(self):
        source = _smpl(8, 50)
        source["pose_aa"][:, 0] += 2 * math.pi
        result = resample_smpl_to_robot_window(source, 0, 50, 50, 8)
        for key in ("pose_aa", "smpl_joints", "transl"):
            torch.testing.assert_close(result[key], torch.from_numpy(source[key]), rtol=0, atol=0)
            result[key].zero_()
            self.assertTrue(np.any(source[key]))

    def test_resampled_grid_can_include_exact_source_endpoint(self):
        result = resample_smpl_to_robot_window(_smpl(31, 30), 0, 30, 50, 51)
        self.assertTimes(result, np.arange(51) / 50)

    def test_integer_samples_in_a_mixed_grid_keep_original_axis_angles(self):
        source = _smpl(4, 30)
        source["pose_aa"][0, 0] = 2 * math.pi + 0.2
        source["pose_aa"][3, 0] = 2 * math.pi + 0.3
        result = resample_smpl_to_robot_window(source, 0, 30, 50, 6)
        torch.testing.assert_close(result["pose_aa"][[0, 5]], torch.from_numpy(source["pose_aa"][[0, 3]]))

    def test_slerp_takes_shortest_arc_across_179_degrees(self):
        source = _smpl(2, 1)
        source["pose_aa"][:] = 0
        source["pose_aa"][0, 2] = math.radians(179)
        source["pose_aa"][1, 2] = math.radians(-179)
        result = resample_smpl_to_robot_window(source, 0, 1, 2, 3)
        self.assertAlmostEqual(abs(result["pose_aa"][1, 2].item()), math.pi, places=5)
        self.assertEqual(result["pose_aa"].shape, (3, 72))

    def test_small_angle_slerp_retains_requested_fraction(self):
        source = _smpl(2, 1)
        source["pose_aa"][:] = 0
        source["pose_aa"][1, 0] = 0.001
        result = resample_smpl_to_robot_window(source, 0, 1, 4, 2)
        self.assertAlmostEqual(result["pose_aa"][1, 0].item(), 0.00025, places=8)

    def test_shaped_pose_input_and_missing_optional_fields(self):
        source = _smpl(3, 50)
        source["pose_aa"] = source["pose_aa"].reshape(3, 24, 3)
        del source["smpl_joints"]
        result = resample_smpl_to_robot_window(source, 0, 50, 50, 3)
        self.assertEqual(result["pose_aa"].shape, (3, 72))
        self.assertEqual(set(result), {"pose_aa", "transl"})

    def test_insufficient_coverage_raises_instead_of_padding_or_stretching(self):
        for start, count in ((0, 52), (31, 1)):
            with self.subTest(start=start, count=count):
                with self.assertRaisesRegex(ValueError, "does not cover"):
                    resample_smpl_to_robot_window(_smpl(31, 30), start, 30, 50, count)

    def test_inconsistent_fields_raise(self):
        source = _smpl(31, 30)
        source["smpl_joints"] = source["smpl_joints"][:-1]
        with self.assertRaisesRegex(ValueError, "inconsistent lengths"):
            resample_smpl_to_robot_window(source, 0, 30, 50, 50)

    def test_bad_fps_and_window_arguments_raise(self):
        for bad_fps in (None, 0, -1, math.inf, math.nan, True, "invalid"):
            for name in ("source", "robot", "target"):
                with self.subTest(fps=bad_fps, name=name):
                    source = _smpl(31, 30)
                    robot_fps, target_fps = 30, 50
                    if name == "source":
                        source["fps"] = bad_fps
                    elif name == "robot":
                        robot_fps = bad_fps
                    else:
                        target_fps = bad_fps
                    with self.assertRaises(ValueError):
                        resample_smpl_to_robot_window(source, 0, robot_fps, target_fps, 50)
        for start, count in ((-1, 10), (0.5, 10), (False, 10), (0, 0), (0, 0.5)):
            with self.subTest(start=start, count=count):
                with self.assertRaises(ValueError):
                    resample_smpl_to_robot_window(_smpl(31, 30), start, 30, 50, count)


if __name__ == "__main__":
    unittest.main()
