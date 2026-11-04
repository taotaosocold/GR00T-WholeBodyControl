"""Strict sparse-human data checks use the real paired motion loader."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import torch

from gear_sonic.tests.test_paired_frame_alignment import (
    _load, _motion_library, _robot_payload, _smpl_payload,
)


def human_motion(frames=8):
    data = _smpl_payload(frames)
    # A nondegenerate human skeleton, still with frame-indexed translations.
    data["smpl_joints"] += np.arange(24, dtype=np.float32)[None, :, None] * 0.01
    return data


def library(data):
    lib = _motion_library(data, cfg={"mode": "strict"})
    lib.m_cfg["require_smpl_vr5"] = True
    return lib


class SmplVR5DataTests(unittest.TestCase):
    def test_valid_pair_passes_preflight_and_real_loader_without_mutating_data(self):
        robot, smpl = _robot_payload(8), human_motion()
        original = deepcopy(smpl)
        lib = library(smpl)
        lib._validate_smpl_vr5_sources([robot], [smpl])
        loaded = _load(lib, robot, smpl)
        torch.testing.assert_close(loaded.smpl_joints, torch.from_numpy(smpl["smpl_joints"]))
        for field in original:
            np.testing.assert_array_equal(smpl[field], original[field])

    def test_missing_smpl_fails_preflight_and_direct_loader(self):
        robot, lib = _robot_payload(8), library(None)
        for pairs in (None, [None]):
            with self.subTest(pairs=pairs), self.assertRaisesRegex(ValueError, "paired SMPL"):
                lib._validate_smpl_vr5_sources([robot], pairs)
        with self.assertRaisesRegex(ValueError, "clip.*paired SMPL"):
            _load(lib, robot)

    def test_missing_joint_positions_cannot_silently_become_zero_targets(self):
        robot, smpl = _robot_payload(8), human_motion()
        del smpl["smpl_joints"]
        lib = library(smpl)
        with self.assertRaisesRegex(ValueError, "clip.*smpl_joints"):
            lib._validate_smpl_vr5_sources([robot], [smpl])
        with self.assertRaisesRegex(ValueError, "clip.*smpl_joints"):
            _load(lib, robot, smpl)

    def test_missing_pose_cannot_supply_unrelated_joint_rotations(self):
        smpl = human_motion()
        del smpl["pose_aa"]
        with self.assertRaisesRegex(ValueError, "clip.*pose_aa"):
            library(smpl)._validate_smpl_vr5_sources([_robot_payload(8)], [smpl])

    def test_placeholder_and_nonfinite_data_fail_with_clip_context(self):
        for corrupt in ("zero", "nan", "length", "shape", "fps"):
            with self.subTest(corrupt=corrupt):
                smpl = human_motion()
                if corrupt == "zero":
                    smpl["smpl_joints"][:] = 0
                elif corrupt == "nan":
                    smpl["pose_aa"][3, 5] = np.nan
                elif corrupt == "length":
                    smpl["smpl_joints"] = smpl["smpl_joints"][:-1]
                elif corrupt == "shape":
                    smpl["smpl_joints"] = smpl["smpl_joints"][:, :22]
                else:
                    smpl["fps"] = 0
                with self.assertRaisesRegex(ValueError, "clip.*SMPL VR5"):
                    library(smpl)._validate_smpl_vr5_sources([_robot_payload(8)], [smpl])

    def test_directory_sources_are_validated_before_multiprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.pkl"
            smpl = human_motion()
            del smpl["smpl_joints"]
            joblib.dump(smpl, path)
            with self.assertRaisesRegex(ValueError, "clip.*smpl_joints"):
                library(smpl)._validate_smpl_vr5_sources(
                    [_robot_payload(8)], [{"path": str(path)}],
                )

    def test_pair_frame_mismatch_is_reported_before_workers_start(self):
        smpl = human_motion(7)
        with self.assertRaisesRegex(ValueError, "clip.*time alignment.*equal"):
            library(smpl)._validate_smpl_vr5_sources([_robot_payload(8)], [smpl])

    def test_different_source_rates_reject_insufficient_human_time_coverage(self):
        robot, smpl = _robot_payload(31, fps=30), human_motion(31)
        smpl["fps"] = 60
        with self.assertRaisesRegex(ValueError, "clip.*time alignment.*does not cover"):
            library(smpl)._validate_smpl_vr5_sources([robot], [smpl])
        # Coverage of an early crop does not imply coverage of the final crop.
        with self.assertRaisesRegex(ValueError, "does not cover"):
            library(smpl)._validate_smpl_vr5_sources([robot], [smpl], max_len=8)

    def test_preflight_uses_fk_exclusive_endpoint_for_different_rates(self):
        robot, smpl = _robot_payload(7, fps=30), human_motion(10)
        smpl["fps"] = 50
        # FK produces samples 0 through .18s; demanding .20s would reject
        # this valid pair. The exact existing resampling contract must survive.
        library(smpl)._validate_smpl_vr5_sources([robot], [smpl])

    def test_malformed_optional_translation_is_rejected_before_workers(self):
        robot, smpl = _robot_payload(8), human_motion()
        smpl["transl"] = np.zeros((8, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "clip.*time alignment.*transl.*shape"):
            library(smpl)._validate_smpl_vr5_sources([robot], [smpl])

    def test_original_mode_keeps_existing_optional_human_fields(self):
        robot, smpl = _robot_payload(8), human_motion()
        del smpl["smpl_joints"]
        lib = _motion_library(smpl, cfg={"mode": "strict"})
        lib._validate_smpl_vr5_sources([robot], None)
        loaded = _load(lib, robot, smpl)
        torch.testing.assert_close(loaded.smpl_joints, torch.zeros(8, 24, 3))


if __name__ == "__main__":
    unittest.main()
