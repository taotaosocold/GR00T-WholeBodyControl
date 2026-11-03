"""Named storage order must not change N3 FK or simulator reference values."""

from copy import deepcopy
import json
from pathlib import Path
from queue import Queue
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import joblib
import numpy as np

from gear_sonic.envs.env_utils.n3_utils import (
    N3_ISAACLAB_BODY_NAMES, N3_ISAACLAB_DOF_NAMES, N3_MJCF_PATH,
    N3_MUJOCO_BODY_NAMES, N3_MUJOCO_JOINT_NAMES, N3_TRACKED_BODIES,
)
from gear_sonic.utils.motion_lib.n3_motion_order import normalize_n3_motion_order


REPO = Path(__file__).resolve().parents[2]
TOES = {"l_toe_link", "r_toe_link"}


class N3MotionOrderTests(unittest.TestCase):
    @staticmethod
    def native_entry(scale=1.):
        frames = 5
        dof = scale * (.003 * (np.arange(29)[None] + 1) + .0002 * np.arange(frames)[:, None])
        dof = dof.astype(np.float32)
        pose = np.zeros((frames, 32, 3), dtype=np.float32)
        pose[:, 0, 2] = .3 + np.arange(frames) * .01
        world = ET.parse(REPO / N3_MJCF_PATH).getroot().find("worldbody")
        for body in world.iter("body"):
            for joint in body.findall("joint"):
                if joint.get("type", "hinge") == "hinge":
                    index = N3_MUJOCO_JOINT_NAMES.index(joint.get("name"))
                    axis = np.fromstring(joint.get("axis"), sep=" ").astype(np.float32)
                    pose[:, N3_MUJOCO_BODY_NAMES.index(body.get("name"))] = dof[:, index, None] * axis
        return dict(fps=50., pose_aa=pose, dof=dof,
                    root_trans_offset=np.column_stack((np.arange(frames) * .001, np.zeros(frames), np.full(frames, .724))).astype(np.float32),
                    root_rot=np.column_stack((np.zeros((frames, 2)), np.sin(pose[:, 0, 2] / 2), np.cos(pose[:, 0, 2] / 2))).astype(np.float32),
                    joint_names=list(N3_MUJOCO_JOINT_NAMES), body_names=list(N3_MUJOCO_BODY_NAMES),
                    robot_type="n3", motion_order="mujoco",
                    source_joint_names_original=["source provenance remains unchanged"])

    @staticmethod
    def stored_entry(native, joint_names=N3_ISAACLAB_DOF_NAMES, body_names=N3_ISAACLAB_BODY_NAMES):
        stored = deepcopy(native)
        stored["dof"] = native["dof"][:, [N3_MUJOCO_JOINT_NAMES.index(name) for name in joint_names]]
        stored["pose_aa"] = native["pose_aa"][:, [N3_MUJOCO_BODY_NAMES.index(name) for name in body_names]]
        stored["joint_names"], stored["body_names"] = list(joint_names), list(body_names)
        stored["motion_order"] = "isaaclab"
        return stored

    def assert_native(self, result, expected):
        for key in ("pose_aa", "dof", "root_trans_offset", "root_rot"):
            np.testing.assert_array_equal(result[key], expected[key], err_msg=key)
        self.assertEqual(result["joint_names"], list(N3_MUJOCO_JOINT_NAMES))
        self.assertEqual(result["body_names"], list(N3_MUJOCO_BODY_NAMES))
        self.assertEqual(result["motion_order"], "mujoco")

    def test_named_bfs_and_arbitrary_named_order_normalize_without_mutation(self):
        native = self.native_entry()
        for joints, bodies in (
            (N3_ISAACLAB_DOF_NAMES, N3_ISAACLAB_BODY_NAMES),
            (tuple(reversed(N3_MUJOCO_JOINT_NAMES)), tuple(reversed(N3_MUJOCO_BODY_NAMES))),
        ):
            stored = self.stored_entry(native, joints, bodies)
            before = deepcopy(stored)
            result = normalize_n3_motion_order(stored)
            self.assert_native(result, native)
            self.assert_native(normalize_n3_motion_order(result), native)
            self.assertEqual(result["source_joint_names_original"], before["source_joint_names_original"])
            self.assertEqual(stored["joint_names"], before["joint_names"])
            self.assertEqual(stored["body_names"], before["body_names"])
            self.assertEqual(stored["motion_order"], "isaaclab")
            for key in ("pose_aa", "dof", "root_trans_offset", "root_rot"):
                np.testing.assert_array_equal(stored[key], before[key])
            result["pose_aa"][0] = 999
            result["dof"][0] = 999
            np.testing.assert_array_equal(stored["pose_aa"], before["pose_aa"])
            np.testing.assert_array_equal(stored["dof"], before["dof"])

    def test_named_thirty_body_layout_inserts_fixed_toes_at_native_slots(self):
        native = self.native_entry()
        bodies = tuple(name for name in N3_ISAACLAB_BODY_NAMES if name not in TOES)
        stored = self.stored_entry(native, body_names=bodies)
        result = normalize_n3_motion_order(stored)
        self.assert_native(result, native)
        self.assertEqual(stored["pose_aa"].shape, (5, 30, 3))
        for name in TOES:
            np.testing.assert_array_equal(result["pose_aa"][:, N3_MUJOCO_BODY_NAMES.index(name)], 0)

    def test_unnamed_legacy_native_thirty_and_thirty_two_body_contracts_are_preserved(self):
        native = self.native_entry()
        for bodies in (N3_MUJOCO_BODY_NAMES, tuple(n for n in N3_MUJOCO_BODY_NAMES if n not in TOES)):
            legacy = self.stored_entry(native, N3_MUJOCO_JOINT_NAMES, bodies)
            for key in ("body_names", "joint_names", "motion_order"):
                del legacy[key]
            before = deepcopy(legacy)
            result = normalize_n3_motion_order(legacy)
            self.assertEqual(set(result), set(legacy))
            for key in ("pose_aa", "dof", "root_trans_offset", "root_rot"):
                np.testing.assert_array_equal(result[key], before[key])

    def test_incomplete_duplicate_or_truncated_metadata_is_rejected(self):
        native = self.native_entry()
        for key in ("joint_names", "body_names"):
            broken = deepcopy(native)
            del broken[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                normalize_n3_motion_order(broken)
        variants = [
            ("joint_names", list(N3_MUJOCO_JOINT_NAMES[:-1])),
            ("joint_names", list(N3_MUJOCO_JOINT_NAMES[:-1]) + [N3_MUJOCO_JOINT_NAMES[0]]),
            ("joint_names", list(N3_MUJOCO_JOINT_NAMES[:-1]) + ["left_wrist_yaw_joint"]),
            ("body_names", list(N3_MUJOCO_BODY_NAMES[:-1])),
            ("body_names", list(N3_MUJOCO_BODY_NAMES[:-1]) + [N3_MUJOCO_BODY_NAMES[0]]),
            ("body_names", list(N3_MUJOCO_BODY_NAMES[:-1]) + ["pelvis"]),
            ("body_names", [n for n in N3_MUJOCO_BODY_NAMES if n not in {"head_yaw_link", "l_toe_link"}]),
        ]
        for key, value in variants:
            broken = {**native, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                normalize_n3_motion_order(broken)

    def test_order_tag_cannot_replace_names_or_override_explicit_names(self):
        native = self.native_entry()
        legacy = deepcopy(native)
        del legacy["joint_names"]
        del legacy["body_names"]
        # Native declarations remain compatible with the old unnamed format.
        np.testing.assert_array_equal(normalize_n3_motion_order(legacy)["pose_aa"], native["pose_aa"])
        for order in ("isaaclab", "guess", np.asarray(["mujoco", "isaaclab"])):
            with self.subTest(order=order), self.assertRaises(ValueError):
                normalize_n3_motion_order({**legacy, "motion_order": order})
        # Explicit names are authoritative even if a known annotation was
        # mislabeled: mapping by the label would silently corrupt this file.
        named = self.stored_entry(native)
        named["motion_order"] = "mujoco"
        self.assert_native(normalize_n3_motion_order(named), native)

    def test_shapes_values_and_order_declarations_are_validated(self):
        native = self.native_entry()
        for key, value in (
            ("pose_aa", np.zeros((5, 31, 3))), ("pose_aa", np.zeros((5, 32, 4))),
            ("pose_aa", np.zeros((32, 3))), ("pose_aa", np.zeros((0, 32, 3))),
            ("pose_aa", np.full((5, 32, 3), np.nan)),
            ("dof", np.zeros((5, 28))), ("dof", np.zeros((4, 29))),
            ("dof", np.zeros((5, 29), dtype=bool)), ("dof", np.full((5, 29), np.inf)),
            ("motion_order", "guess"), ("motion_order", ["isaaclab", "mujoco"]),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize_n3_motion_order({**native, key: value})

    @staticmethod
    def fk():
        from easydict import EasyDict
        from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
        path = REPO / N3_MJCF_PATH
        cfg = EasyDict(robot_type="n3", asset=EasyDict(assetRoot=str(path.parent), assetFileName=path.name), extend_config=[])
        with patch.object(Humanoid_Batch, "load_mesh"):
            return Humanoid_Batch(cfg)

    def test_native_and_normalized_named_storage_have_identical_real_fk(self):
        import torch
        native = self.native_entry()
        fk = self.fk()
        expected = fk.fk_batch(torch.from_numpy(native["pose_aa"])[None], torch.from_numpy(native["root_trans_offset"])[None],
                               return_full=True, fps=50, target_fps=50)
        for bodies in (N3_ISAACLAB_BODY_NAMES, tuple(n for n in N3_ISAACLAB_BODY_NAMES if n not in TOES)):
            stored = self.stored_entry(native, body_names=bodies)
            normalized = normalize_n3_motion_order(stored)
            actual = fk.fk_batch(torch.from_numpy(normalized["pose_aa"])[None],
                                 torch.from_numpy(normalized["root_trans_offset"])[None], return_full=True, fps=50, target_fps=50)
            for key in ("global_translation", "global_rotation", "dof_pos", "dof_vels"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_real_loader_applies_named_storage_mapping_once_before_live_order_mapping(self):
        import torch
        from easydict import EasyDict
        from gear_sonic.trl.utils.order_converter import N3Converter
        from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot
        from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

        native = self.native_entry()
        converter = N3Converter()
        mapping = converter.get_isaaclab_to_mujoco_mapping()
        variants = {
            "native": native,
            "bfs": self.stored_entry(native),
            "bfs30": self.stored_entry(native, body_names=tuple(n for n in N3_ISAACLAB_BODY_NAMES if n not in TOES)),
        }
        results = {}
        previous_threads = torch.get_num_threads()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name, entry in variants.items():
                    directory = root / name
                    directory.mkdir()
                    joblib.dump({"clip": entry}, directory / "clip.pkl")
                    cfg = EasyDict(converter.motion_metadata, asset=EasyDict(assetRoot=str((REPO / N3_MJCF_PATH).parent),
                                                                             assetFileName=Path(N3_MJCF_PATH).name), extend_config=[])
                    cfg.update(mapping)
                    cfg.body_indexes = torch.tensor([N3_ISAACLAB_BODY_NAMES.index(n) for n in N3_TRACKED_BODIES])
                    cfg.body_indexes_data = cfg.body_indexes.tolist()
                    cfg.motion_file, cfg.target_fps, cfg.multi_thread = str(directory), 50, False
                    cfg.adaptive_sampling = {"enable": True}
                    with patch.object(Humanoid_Batch, "load_mesh"), patch(
                        "gear_sonic.utils.motion_lib.motion_lib_base.mp.Manager",
                        return_value=SimpleNamespace(Queue=Queue),
                    ):
                        library = MotionLibRobot(cfg, 2, "cpu")
                        library.load_motions_for_training(max_num_seqs=1)
                    self.assertEqual(library.adp_samp_num_frames.tolist(), [5])
                    results[name] = {field: getattr(library, field).clone() for field in
                                     ("dof_pos", "dof_vel", "body_pos_w_full", "body_quat_w_full", "body_pos_w")}
                for name in ("bfs", "bfs30"):
                    for field, expected in results["native"].items():
                        torch.testing.assert_close(results[name][field], expected, rtol=0, atol=0, msg=f"{name}:{field}")
        finally:
            torch.set_num_threads(previous_threads)

    def test_upper_body_augmentation_normalizes_named_cached_motion(self):
        from easydict import EasyDict
        from gear_sonic.trl.utils.order_converter import N3Converter
        from gear_sonic.utils.motion_lib.motion_lib_base import FixHeightMode, MotionLibBase

        base, selected = self.native_entry(), self.native_entry(scale=2.)
        loader = object.__new__(MotionLibBase)
        metadata = N3Converter().motion_metadata
        attributes = dict(m_cfg=EasyDict(metadata), target_fps=50, use_parallel_fk=False, mesh_parsers=self.fk(),
                          randomize_upper_body_poses=True, cat_upper_body_poses_prob=1., upper_body_augment_prefixes=["nav"],
                          curr_motion_keys=["nav"], res_non_nav_dataset={0: (self.stored_entry(selected), None)},
                          randomize_wrist_poses=False, smpl_data=None, object_data=None, has_action=False, vid_smpl_pose=None)
        for key, value in attributes.items():
            setattr(loader, key, value)
        result = loader.load_motion_with_skeleton([0], [self.stored_entry(base)], None, None, None,
                                                 FixHeightMode.no_fix, None, -1, False, None, 1)[0]
        canonical_file, motion = result
        expected = selected["dof"].copy()
        legs = metadata["lower_joint_indices_mujoco"]
        expected[:, legs] = base["dof"][:, legs]
        np.testing.assert_allclose(motion.dof_pos.numpy(), expected, atol=1e-7, rtol=1e-6)
        self.assertEqual(canonical_file["body_names"], list(N3_MUJOCO_BODY_NAMES))
        self.assertEqual(canonical_file["motion_order"], "mujoco")

    def test_normalizer_import_requires_only_numpy_and_portable_metadata(self):
        code = ("import sys,json; import gear_sonic.utils.motion_lib.n3_motion_order; "
                "print(json.dumps([name for name in sys.modules if name == 'torch' or name == 'scipy' or name.startswith('isaac')]))")
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
