"""CPU integration coverage for paired SMPL sampling on the robot FK timeline."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gear_sonic.tests.test_paired_frame_alignment import (
    _FrameIndexedParser,
    _load,
    _motion_library,
    _robot_payload,
    _smpl_payload,
)
from gear_sonic.utils.motion_lib import motion_lib_base as motion_lib


class _ResamplingParser(_FrameIndexedParser):
    """Keep the old pose-only path executable when reproducing the regression."""

    def interploate_pose(self, trans, pose, fps, target_fps):
        times = torch.arange(0, (pose.shape[1] - 1) / fps, 1 / target_fps)
        positions = times * fps
        lower = positions.floor().long()
        upper = (lower + 1).clamp(max=pose.shape[1] - 1)
        blend = positions - lower
        pose = (
            pose[:, lower] * (1 - blend[None, :, None])
            + pose[:, upper] * blend[None, :, None]
        )
        return trans, pose


def _time_indexed_payloads(robot_frames, robot_fps, smpl_frames, smpl_fps):
    """Represent physical time in translations and a small, single-axis pose."""
    robot = _robot_payload(robot_frames, robot_fps)
    robot["root_trans_offset"] /= robot_fps
    robot["pose_aa"] /= robot_fps
    smpl = _smpl_payload(smpl_frames, smpl_fps)
    smpl["smpl_joints"] /= smpl_fps
    smpl["transl"] /= smpl_fps
    smpl["pose_aa"][:] = 0
    smpl["pose_aa"][:, 2] = 0.25 * np.arange(smpl_frames) / smpl_fps
    return robot, smpl


def _strict_library(smpl):
    lib = _motion_library(smpl, cfg={"mode": "strict"})
    lib.mesh_parsers = _ResamplingParser()
    return lib


class SmplLoaderResamplingTest(unittest.TestCase):
    def assertPayloadUnchanged(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key, value in expected.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            elif isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual[key], value)
            else:
                self.assertEqual(actual[key], value)

    def assertPhysicalTimes(self, loaded, expected_times):
        frame_count = len(expected_times)
        self.assertEqual(loaded.global_translation.shape[0], frame_count)
        self.assertEqual(loaded.smpl_pose.shape, (frame_count, 72))
        self.assertEqual(loaded.smpl_joints.shape, (frame_count, 24, 3))
        self.assertEqual(loaded.smpl_transl.shape, (frame_count, 3))
        for values in (
            loaded.global_translation[:, 0, 0],
            loaded.dof_pos[:, 0],
            loaded.smpl_joints[:, 0, 0],
            loaded.smpl_transl[:, 0],
        ):
            torch.testing.assert_close(values, expected_times)
        expected_pose = torch.zeros(frame_count, 72)
        expected_pose[:, 2] = 0.25 * expected_times
        torch.testing.assert_close(loaded.smpl_pose, expected_pose)
        torch.testing.assert_close(
            loaded.smpl_joints,
            expected_times[:, None, None].expand(-1, 24, 3),
        )
        torch.testing.assert_close(
            loaded.smpl_transl,
            expected_times[:, None].expand(-1, 3),
        )

    def test_same_30_fps_sources_resample_all_smpl_fields_to_50_fps(self):
        robot, smpl = _time_indexed_payloads(10, 30, 10, 30)
        original_robot, original_smpl = copy.deepcopy((robot, smpl))

        loaded = _load(_strict_library(smpl), robot, smpl)

        self.assertPhysicalTimes(loaded, torch.arange(15) / 50)
        self.assertPayloadUnchanged(robot, original_robot)
        self.assertPayloadUnchanged(smpl, original_smpl)

    def test_different_source_rates_sample_the_same_physical_times(self):
        for robot_fps, robot_frames, smpl_fps, smpl_frames in (
            (30, 7, 60, 13),
            (60, 13, 30, 7),
            (30, 7, 50, 11),
        ):
            with self.subTest(robot_fps=robot_fps, smpl_fps=smpl_fps):
                robot, smpl = _time_indexed_payloads(
                    robot_frames, robot_fps, smpl_frames, smpl_fps
                )
                original_robot, original_smpl = copy.deepcopy((robot, smpl))

                loaded = _load(_strict_library(smpl), robot, smpl)

                self.assertPhysicalTimes(loaded, torch.arange(10) / 50)
                self.assertPayloadUnchanged(robot, original_robot)
                self.assertPayloadUnchanged(smpl, original_smpl)

    def test_nonzero_crop_uses_robot_start_time_across_source_rates(self):
        robot, smpl = _time_indexed_payloads(12, 30, 23, 60)
        original_robot, original_smpl = copy.deepcopy((robot, smpl))
        with patch.object(motion_lib.random, "randint", return_value=2) as choose_start:
            loaded = _load(_strict_library(smpl), robot, smpl, max_len=5)

        choose_start.assert_called_once_with(0, 7)
        self.assertPhysicalTimes(loaded, 2 / 30 + torch.arange(7) / 50)
        self.assertPayloadUnchanged(robot, original_robot)
        self.assertPayloadUnchanged(smpl, original_smpl)

    def test_same_rate_nonzero_crop_resamples_only_the_selected_window(self):
        robot, smpl = _time_indexed_payloads(12, 30, 12, 30)
        with patch.object(motion_lib.random, "randint", return_value=3):
            loaded = _load(_strict_library(smpl), robot, smpl, max_len=5)

        self.assertPhysicalTimes(loaded, 3 / 30 + torch.arange(7) / 50)

    def test_short_smpl_coverage_raises_without_padding_or_mutating_sources(self):
        robot, smpl = _time_indexed_payloads(10, 30, 5, 30)
        original_robot, original_smpl = copy.deepcopy((robot, smpl))

        with self.assertRaises(ValueError):
            _load(_strict_library(smpl), robot, smpl)

        self.assertPayloadUnchanged(robot, original_robot)
        self.assertPayloadUnchanged(smpl, original_smpl)


if __name__ == "__main__":
    unittest.main()
