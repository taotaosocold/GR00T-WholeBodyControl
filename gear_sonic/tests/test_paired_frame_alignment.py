"""CPU regression tests for paired robot/SMPL trailing-frame alignment.

The real resolver, motion loader, and adaptive sampler run against synthetic,
frame-indexed payloads. Only robot forward kinematics and ground/foot handling
are replaced, so these tests do not need an Isaac scene or motion datasets.
Run with ``python -m unittest gear_sonic.tests.test_paired_frame_alignment``.
"""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from gear_sonic.utils.motion_lib import motion_lib_base as motion_lib


TRIM_CFG = {"mode": "trim_trailing", "max_frame_delta": 2}


def _robot_payload(num_frames, fps=50):
    frames = torch.arange(num_frames, dtype=torch.float32)
    return {
        "fps": fps,
        "root_trans_offset": frames[:, None].expand(-1, 3).clone(),
        "pose_aa": frames[:, None, None].expand(-1, 2, 3).clone(),
    }


def _smpl_payload(num_frames, fps=50):
    frames = np.arange(num_frames, dtype=np.float32)
    return {
        "fps": fps,
        "pose_aa": np.broadcast_to(frames[:, None], (num_frames, 72)).copy(),
        "smpl_joints": np.broadcast_to(
            frames[:, None, None], (num_frames, 24, 3)
        ).copy(),
        "transl": np.broadcast_to(frames[:, None], (num_frames, 3)).copy(),
    }


class _FrameIndexedParser:
    """Expose the input frame identities through a small FK-shaped result."""

    def fk_batch(
        self,
        pose,
        trans,
        *,
        return_full,
        fps,
        target_fps,
        interpolate_data,
        use_parallel_fk,
    ):
        del return_full, use_parallel_fk
        if interpolate_data and fps != target_fps:
            times = torch.arange(0, (pose.shape[1] - 1) / fps, 1 / target_fps)
            positions = times * fps
            lower = positions.floor().long()
            upper = (lower + 1).clamp(max=pose.shape[1] - 1)
            blend = positions - lower
            trans = (
                trans[:, lower] * (1 - blend[None, :, None])
                + trans[:, upper] * blend[None, :, None]
            )
            pose = (
                pose[:, lower] * (1 - blend[None, :, None, None])
                + pose[:, upper] * blend[None, :, None, None]
            )
        return {
            "global_translation": trans[:, :, None, :].expand(-1, -1, 2, -1).clone(),
            "dof_pos": pose[:, :, 1, :2].clone(),
        }


def _motion_library(smpl=None, cfg=None):
    lib = motion_lib.MotionLibBase.__new__(motion_lib.MotionLibBase)
    lib.m_cfg = {}
    lib.curr_motion_keys = ["clip"]
    lib.paired_frame_alignment_cfg = dict(TRIM_CFG if cfg is None else cfg)
    lib.randomize_upper_body_poses = False
    lib.randomize_wrist_poses = False
    lib.target_fps = 50
    lib.use_parallel_fk = False
    lib.mesh_parsers = _FrameIndexedParser()
    lib.fix_trans_height = lambda pose, trans, **kwargs: (trans, 0)
    lib.foot_detect = lambda trans, *args: (
        torch.zeros(len(trans)),
        torch.zeros(len(trans)),
    )
    lib.smpl_data = [smpl] if smpl is not None else None
    lib.object_data = None
    lib.has_action = False
    lib.vid_smpl_pose = None
    return lib


def _load(lib, robot, smpl=None, max_len=-1, is_evaluation=True):
    result = lib.load_motion_with_skeleton(
        [0],
        [robot],
        [smpl] if smpl is not None else None,
        None,
        None,
        motion_lib.FixHeightMode.no_fix,
        None,
        max_len,
        is_evaluation,
        None,
        1,
    )
    return result[0][1]


class PairedFrameAlignmentTest(unittest.TestCase):
    def assertPayloadEqual(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key, value in expected.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(actual[key], value)
            elif isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual[key], value)
            else:
                self.assertEqual(actual[key], value)

    def assertFrameIdentities(self, loaded, frames):
        expected = torch.tensor(frames, dtype=torch.float32)
        for values in (
            loaded.global_translation[:, 0, 0],
            loaded.dof_pos[:, 0],
            loaded.smpl_pose[:, 0],
            loaded.smpl_joints[:, 0, 0],
            loaded.smpl_transl[:, 0],
        ):
            torch.testing.assert_close(values, expected)

    def test_strict_accepts_equal_lengths(self):
        for cfg in ({}, {"mode": "strict"}):
            with self.subTest(cfg=cfg):
                self.assertEqual(
                    motion_lib.resolve_paired_frame_alignment(
                        8, 50, _smpl_payload(8), 50, cfg
                    ),
                    8,
                )

    def test_robot_only_resolver_preserves_source_length(self):
        for cfg in ({}, TRIM_CFG):
            with self.subTest(cfg=cfg):
                self.assertEqual(
                    motion_lib.resolve_paired_frame_alignment(10, 30, None, 50, cfg),
                    10,
                )

    def test_resolver_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            motion_lib.resolve_paired_frame_alignment(
                8, 50, _smpl_payload(8), 50, {"mode": "unknown"}
            )

    def test_strict_rejects_target_fps_length_mismatch_in_both_directions(self):
        for robot_frames, smpl_frames in ((10, 8), (8, 10)):
            with self.subTest(robot=robot_frames, smpl=smpl_frames):
                with self.assertRaises(ValueError):
                    motion_lib.resolve_paired_frame_alignment(
                        robot_frames, 50, _smpl_payload(smpl_frames), 50, {}
                    )

    def test_strict_preserves_non_target_source_length(self):
        self.assertEqual(
            motion_lib.resolve_paired_frame_alignment(
                10, 30, _smpl_payload(8, fps=30), 50, {"mode": "strict"}
            ),
            10,
        )

    def test_trim_takes_common_prefix_without_changing_smpl_payload(self):
        for robot_frames, smpl_frames in ((10, 8), (8, 10), (8, 8)):
            with self.subTest(robot=robot_frames, smpl=smpl_frames):
                smpl = _smpl_payload(smpl_frames)
                original = copy.deepcopy(smpl)
                self.assertEqual(
                    motion_lib.resolve_paired_frame_alignment(
                        robot_frames, 50, smpl, 50, TRIM_CFG
                    ),
                    8,
                )
                self.assertPayloadEqual(smpl, original)

    def test_trim_accepts_one_frame_difference_and_equal_lengths(self):
        cfg = {"mode": "trim_trailing", "max_frame_delta": 1}
        for robot_frames, smpl_frames in ((9, 8), (8, 9), (8, 8)):
            with self.subTest(robot=robot_frames, smpl=smpl_frames):
                self.assertEqual(
                    motion_lib.resolve_paired_frame_alignment(
                        robot_frames, 50, _smpl_payload(smpl_frames), 50, cfg
                    ),
                    8,
                )

    def test_trim_rejects_difference_beyond_configured_limit(self):
        for robot_frames, smpl_frames, max_delta in ((11, 8, 2), (8, 11, 2), (9, 8, 0)):
            with self.subTest(robot=robot_frames, smpl=smpl_frames, limit=max_delta):
                with self.assertRaises(ValueError):
                    motion_lib.resolve_paired_frame_alignment(
                        robot_frames,
                        50,
                        _smpl_payload(smpl_frames),
                        50,
                        {"mode": "trim_trailing", "max_frame_delta": max_delta},
                    )

    def test_trim_requires_both_sources_at_target_fps(self):
        for robot_fps, smpl_fps in ((30, 30), (30, 50), (50, 30)):
            with self.subTest(robot_fps=robot_fps, smpl_fps=smpl_fps):
                with self.assertRaisesRegex(ValueError, "目标帧率"):
                    motion_lib.resolve_paired_frame_alignment(
                        10, robot_fps, _smpl_payload(10, smpl_fps), 50, TRIM_CFG
                    )

    def test_trim_requires_all_three_smpl_arrays(self):
        for field in ("pose_aa", "smpl_joints", "transl"):
            with self.subTest(missing=field):
                smpl = _smpl_payload(8)
                del smpl[field]
                with self.assertRaises(ValueError):
                    motion_lib.resolve_paired_frame_alignment(10, 50, smpl, 50, TRIM_CFG)

    def test_trim_rejects_inconsistent_internal_smpl_lengths(self):
        for field in ("pose_aa", "smpl_joints", "transl"):
            with self.subTest(shortened=field):
                smpl = _smpl_payload(8)
                smpl[field] = smpl[field][:-1]
                with self.assertRaises(ValueError):
                    motion_lib.resolve_paired_frame_alignment(10, 50, smpl, 50, TRIM_CFG)

    def test_trim_needs_at_least_three_common_frames(self):
        self.assertEqual(
            motion_lib.resolve_paired_frame_alignment(
                3, 50, _smpl_payload(3), 50, TRIM_CFG
            ),
            3,
        )
        for robot_frames, smpl_frames in ((1, 2), (2, 1), (2, 3), (3, 2), (2, 2)):
            with self.subTest(robot=robot_frames, smpl=smpl_frames):
                with self.assertRaises(ValueError):
                    motion_lib.resolve_paired_frame_alignment(
                        robot_frames, 50, _smpl_payload(smpl_frames), 50, TRIM_CFG
                    )

    def test_trim_rejects_invalid_frame_delta_configuration(self):
        for max_delta in (-1, 1.5, "2", True):
            with self.subTest(max_frame_delta=max_delta):
                with self.assertRaises(ValueError):
                    motion_lib.resolve_paired_frame_alignment(
                        10,
                        50,
                        _smpl_payload(8),
                        50,
                        {"mode": "trim_trailing", "max_frame_delta": max_delta},
                    )

    def test_loader_trims_both_directions_and_preserves_source_payloads(self):
        for robot_frames, smpl_frames in ((10, 8), (8, 10)):
            with self.subTest(robot=robot_frames, smpl=smpl_frames):
                robot = _robot_payload(robot_frames)
                smpl = _smpl_payload(smpl_frames)
                original_robot, original_smpl = copy.deepcopy((robot, smpl))
                loaded = _load(_motion_library(smpl), robot, smpl)
                self.assertFrameIdentities(loaded, range(8))
                self.assertPayloadEqual(robot, original_robot)
                self.assertPayloadEqual(smpl, original_smpl)

    def test_loader_applies_one_random_window_to_robot_and_all_smpl_arrays(self):
        for robot_frames, smpl_frames in ((10, 8), (8, 10)):
            with self.subTest(robot=robot_frames, smpl=smpl_frames):
                robot, smpl = _robot_payload(robot_frames), _smpl_payload(smpl_frames)
                original_robot, original_smpl = copy.deepcopy((robot, smpl))
                with patch.object(motion_lib.random, "randint", return_value=2) as choose:
                    loaded = _load(_motion_library(smpl), robot, smpl, max_len=5)
                choose.assert_called_once_with(0, 3)
                self.assertFrameIdentities(loaded, range(2, 7))
                self.assertPayloadEqual(robot, original_robot)
                self.assertPayloadEqual(smpl, original_smpl)

    def test_loader_freezes_the_same_cropped_frame_in_both_sources(self):
        for robot_frames, smpl_frames in ((10, 8), (8, 10)):
            with self.subTest(robot=robot_frames, smpl=smpl_frames):
                robot, smpl = _robot_payload(robot_frames), _smpl_payload(smpl_frames)
                original_robot, original_smpl = copy.deepcopy((robot, smpl))
                lib = _motion_library(smpl)
                lib.m_cfg.update(freeze_frame_aug=True, freeze_frame_prob=1.0)
                with (
                    patch.object(motion_lib.random, "randint", return_value=2),
                    patch.object(motion_lib.np.random, "random", return_value=0.0),
                    patch.object(motion_lib.np.random, "randint", return_value=2),
                ):
                    loaded = _load(lib, robot, smpl, max_len=5, is_evaluation=False)
                self.assertFrameIdentities(loaded, [2, 3, 4, 4, 4])
                self.assertPayloadEqual(robot, original_robot)
                self.assertPayloadEqual(smpl, original_smpl)

    def test_loader_rejects_30_fps_paired_trim(self):
        robot, smpl = _robot_payload(10, 30), _smpl_payload(8, 30)
        with self.assertRaisesRegex(ValueError, "目标帧率"):
            _load(_motion_library(smpl), robot, smpl)

    def test_loader_keeps_robot_only_30_fps_resampling(self):
        loaded = _load(_motion_library(), _robot_payload(10, 30))
        self.assertEqual(loaded.global_translation.shape, (15, 2, 3))
        self.assertEqual(loaded.dof_pos.shape, (15, 2))

    def test_loader_rejects_trim_window_shorter_than_three_frames(self):
        smpl = _smpl_payload(8)
        with self.assertRaises(ValueError):
            _load(_motion_library(smpl), _robot_payload(10), smpl, max_len=2)

    def test_loader_strict_rejects_mismatch(self):
        smpl = _smpl_payload(8)
        with self.assertRaises(ValueError):
            _load(_motion_library(smpl, {"mode": "strict"}), _robot_payload(10), smpl)

    def test_adaptive_sampler_matches_loaded_paired_lengths_in_both_directions(self):
        for robot_frames, smpl_frames in ((10, 8), (8, 10)):
            with self.subTest(robot=robot_frames, smpl=smpl_frames):
                robot, smpl = _robot_payload(robot_frames), _smpl_payload(smpl_frames)
                lib = _motion_library(smpl)
                lib._device = torch.device("cpu")
                lib._motion_data_keys = ["clip"]
                lib._motion_data_load = {"clip": robot}
                lib._num_unique_motions = 1
                lib.adaptive_sampling_cfg = {"bin_size": 5}
                lib.init_adaptive_sampling()
                loaded = _load(lib, robot, smpl)
                self.assertEqual(lib.adp_samp_num_frames.tolist(), [len(loaded.dof_pos)])
                self.assertEqual(lib.adp_samp_bins.tolist(), [[0, 0, 5], [0, 5, 8]])
                self.assertEqual(lib.adp_samp_frame_to_bin.tolist(), [0] * 5 + [1] * 3)

    def test_adaptive_sampler_aligns_375_frame_metadata_to_374_frame_smpl(self):
        robot, smpl = _robot_payload(375), _smpl_payload(374)
        robot_path, smpl_path = "synthetic_robot.pkl", "synthetic_smpl.pkl"
        lib = _motion_library()
        lib.smpl_data = [{"path": smpl_path}]
        lib._device = torch.device("cpu")
        lib._motion_data_keys = ["clip"]
        lib._motion_data_load = {
            "clip": {"length": 375, "fps": 50, "path": robot_path}
        }
        lib._num_unique_motions = 1
        lib.adaptive_sampling_cfg = {"bin_size": 50}
        payloads = {robot_path: {"clip": robot}, smpl_path: smpl}
        with patch.object(motion_lib.joblib, "load", side_effect=payloads.__getitem__):
            lib.init_adaptive_sampling()
        self.assertEqual(lib.adp_samp_num_frames.tolist(), [374])
        self.assertEqual(lib.adp_samp_length_starts.tolist(), [0])
        self.assertEqual(lib.adp_samp_total_frames.item(), 374)
        self.assertEqual(lib.adp_samp_frame_to_bin.numel(), 374)
        self.assertEqual(lib.adp_samp_bins[-1].tolist(), [0, 350, 374])


if __name__ == "__main__":
    unittest.main()
