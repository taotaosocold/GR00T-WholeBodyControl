"""Validate N3 data conversion independently of an Isaac/CUDA runtime."""

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

from gear_sonic.data_process.convert_n3_npz_to_motion_lib import (
    DEFAULT_STRIP_SUFFIX, N3_0905_ISAACLAB_DOF_NAMES, convert_directory, convert_motion,
    motion_key_from_path, order_motion_for_storage,
)
from gear_sonic.envs.env_utils.n3_utils import (
    N3_ISAACLAB_BODY_NAMES, N3_ISAACLAB_DOF_NAMES, N3_MJCF_PATH,
    N3_MUJOCO_BODY_NAMES, N3_MUJOCO_JOINT_NAMES, N3_TRACKED_BODIES,
    N3_TRAINING_TO_DEPLOYMENT_JOINT_NAMES,
)
from gear_sonic.utils.motion_lib.n3_motion_order import normalize_n3_motion_order


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "gear_sonic/data_process/convert_n3_npz_to_motion_lib.py"


class N3ConversionTests(unittest.TestCase):
    @staticmethod
    def fixture(frames=5):
        # Deliberately differ from both native and expected Isaac joint order;
        # root is not body zero either. A coincidental G1 mapping cannot pass.
        joint_names = tuple(reversed(N3_ISAACLAB_DOF_NAMES))
        body_names = tuple(N3_MUJOCO_BODY_NAMES[1:]) + ("base_link",)
        native = (np.arange(29)[None] * .007 + np.arange(frames)[:, None] * .001).astype(np.float32)
        joint_pos = native[:, [N3_MUJOCO_JOINT_NAMES.index(name) for name in joint_names]]
        positions = np.zeros((frames, 32, 3), dtype=np.float32)
        root = body_names.index("base_link")
        positions[:, root] = np.column_stack((np.arange(frames) * .1, np.arange(frames) * -.03,
                                             .724 + np.arange(frames) * .001))
        quaternions = np.zeros((frames, 32, 4), dtype=np.float32)
        quaternions[..., 0] = 1
        quaternions[:, root] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        return dict(fps=np.asarray(50., dtype=np.float32), joint_pos=joint_pos,
                    body_pos_w=positions, body_quat_w=quaternions,
                    joint_names=np.asarray(joint_names), body_names=np.asarray(body_names)), native

    def test_arbitrary_joint_order_and_root_body_make_native_32_body_pose(self):
        data, native = self.fixture()
        result = convert_motion(data)
        self.assertEqual(result["pose_aa"].shape, (5, 32, 3))
        np.testing.assert_allclose(result["dof"], native, atol=0, rtol=0)
        np.testing.assert_allclose(result["root_trans_offset"], data["body_pos_w"][:, -1], atol=0, rtol=0)
        expected_root = np.tile([0., 0., np.pi / 2], (5, 1))
        np.testing.assert_allclose(result["pose_aa"][:, 0], expected_root, atol=2e-7, rtol=0)
        np.testing.assert_allclose(result["root_rot"], data["body_quat_w"][:, -1, [1, 2, 3, 0]], atol=2e-7, rtol=0)
        tree = ET.parse(REPO / N3_MJCF_PATH).getroot()
        actuated = {joint.get("name"): (body.get("name"), np.fromstring(joint.get("axis"), sep=" "))
                    for body in tree.find("worldbody").iter("body")
                    for joint in body.findall("joint") if joint.get("type", "hinge") == "hinge"}
        for index, name in enumerate(N3_MUJOCO_JOINT_NAMES):
            body_name, axis = actuated[name]
            np.testing.assert_allclose(result["pose_aa"][:, N3_MUJOCO_BODY_NAMES.index(body_name)],
                                       native[:, index, None] * axis[None], atol=2e-8, rtol=1e-6)
        for name in ("l_toe_link", "r_toe_link"):
            np.testing.assert_array_equal(result["pose_aa"][:, N3_MUJOCO_BODY_NAMES.index(name)], 0)

    def test_xyzw_and_near_unit_quaternion_are_normalized(self):
        data, _ = self.fixture()
        data["body_quat_w"] = data["body_quat_w"][:, :, [1, 2, 3, 0]]
        data["body_quat_w"][:, -1] *= 1.0005
        result = convert_motion(data, quat_order="xyzw")
        np.testing.assert_allclose(np.linalg.norm(result["root_rot"], axis=-1), 1, atol=1e-7, rtol=0)
        np.testing.assert_allclose(result["pose_aa"][:, 0, 2], np.pi / 2, atol=2e-7, rtol=0)
        np.testing.assert_allclose(result["pose_aa"][:, 0, :2], 0, atol=1e-7, rtol=0)

    def test_conversion_does_not_mutate_input_or_share_output_buffers(self):
        data, _ = self.fixture()
        data["body_quat_w"][:, -1] *= 1.0005
        original = deepcopy(data)
        result = convert_motion(data)
        for key in data:
            np.testing.assert_array_equal(data[key], original[key], err_msg=key)
        result["root_trans_offset"][0] = 999
        result["dof"][0] = 999
        result["root_rot"][0] = 999
        for key in data:
            np.testing.assert_array_equal(data[key], original[key], err_msg=key)

    def test_explicit_names_and_root_are_required_without_metadata(self):
        data, _ = self.fixture()
        names = data.pop("joint_names")
        root = list(data.pop("body_names")).index("base_link")
        with self.assertRaises(ValueError):
            convert_motion(data)
        with self.assertRaises(ValueError):
            convert_motion(data, names)
        entry = convert_motion(data, names, root_body_index=root)
        self.assertEqual(entry["pose_aa"].shape, (5, 32, 3))

    def test_conflicting_or_incomplete_name_metadata_is_rejected(self):
        data, _ = self.fixture()
        with self.assertRaises(ValueError):
            convert_motion(data, list(reversed(data["joint_names"])))
        with self.assertRaises(ValueError):
            convert_motion(data, root_body_index=0)  # metadata says base_link is last
        for field, names in (
            ("joint_names", list(data["joint_names"][:-1])),
            ("joint_names", list(data["joint_names"][:-1]) + ["unknown_joint"]),
            ("joint_names", list(data["joint_names"][:-1]) + [data["joint_names"][0]]),
            ("body_names", list(data["body_names"][:-1])),
            ("body_names", list(data["body_names"][:-1]) + [data["body_names"][0]]),
        ):
            broken = deepcopy(data)
            broken[field] = np.asarray(names)
            with self.subTest(field=field, names=names), self.assertRaises(ValueError):
                convert_motion(broken)

    def test_optional_robot_and_quaternion_metadata_are_validated(self):
        data, native = self.fixture()
        for robot_type, quat_order in (("n3", "wxyz"), (np.asarray([b"n3"]), np.asarray([b"wxyz"]))):
            named = {**data, "robot_type": robot_type, "quat_order": quat_order}
            with self.subTest(robot_type=robot_type, quat_order=quat_order):
                np.testing.assert_array_equal(convert_motion(named)["dof"], native)
        for field, value in (("robot_type", "g1"), ("quat_order", "xyzw")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                convert_motion({**data, field: value})
        xyzw = deepcopy(data)
        xyzw.update(robot_type="n3", quat_order="xyzw")
        xyzw["body_quat_w"] = xyzw["body_quat_w"][:, :, [1, 2, 3, 0]]
        converted = convert_motion(xyzw, quat_order="xyzw")
        np.testing.assert_allclose(converted["pose_aa"][:, 0, 2], np.pi / 2, atol=2e-7, rtol=0)

    def test_nonfinite_values_invalid_fps_and_invalid_quaternions_are_rejected(self):
        original, _ = self.fixture()
        for field in ("joint_pos", "body_pos_w", "body_quat_w"):
            for bad in (np.nan, np.inf, -np.inf):
                data = deepcopy(original)
                data[field].flat[-1] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    convert_motion(data)
        for fps in (0, -1, np.nan, np.inf, True, np.array([30., 50.])):
            data = {**original, "fps": fps}
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                convert_motion(data)
        for quat in ([0, 0, 0, 0], [2, 0, 0, 0]):
            data = deepcopy(original)
            data["body_quat_w"][:, -1] = quat
            with self.subTest(quat=quat), self.assertRaises(ValueError):
                convert_motion(data)
        with self.assertRaises(ValueError):
            convert_motion(original, quat_order="guess")

    def test_missing_fields_short_clips_and_inconsistent_shapes_are_rejected(self):
        data, _ = self.fixture()
        for field in ("fps", "joint_pos", "body_pos_w", "body_quat_w"):
            broken = dict(data)
            del broken[field]
            with self.subTest(missing=field), self.assertRaises((KeyError, ValueError)):
                convert_motion(broken)
        for field, value in (
            ("joint_pos", np.zeros((5, 28))), ("joint_pos", np.zeros((5, 30))),
            ("joint_pos", np.zeros(29)), ("body_pos_w", np.zeros((4, 32, 3))),
            ("body_pos_w", np.zeros((5, 32, 2))), ("body_pos_w", np.zeros((5, 3))),
            ("body_quat_w", np.zeros((5, 31, 4))), ("body_quat_w", np.zeros((5, 32, 3))),
        ):
            with self.subTest(field=field, shape=value.shape), self.assertRaises(ValueError):
                convert_motion({**data, field: value})
        for frames in (0, 1, 2):
            short, _ = self.fixture(frames=frames)
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                convert_motion(short)
        for index in (-1, 32, .5, 31.0, np.float32(31.0)):
            with self.subTest(index=index), self.assertRaises(ValueError):
                convert_motion(data, root_body_index=index)
        # bool compares equal to int, so check it even when its numerical
        # value happens to be exactly the root index named by metadata.
        for shift, index in ((1, False), (2, True)):
            reordered = deepcopy(data)
            reordered["body_names"] = np.roll(data["body_names"], shift)
            reordered["body_pos_w"] = np.roll(data["body_pos_w"], shift, axis=1)
            reordered["body_quat_w"] = np.roll(data["body_quat_w"], shift, axis=1)
            with self.subTest(index=index), self.assertRaises(ValueError):
                convert_motion(reordered, root_body_index=index)

    def test_suffix_removal_is_terminal_and_reserved_metadata_key_is_rejected(self):
        self.assertEqual(motion_key_from_path(Path("210531") / f"walk{DEFAULT_STRIP_SUFFIX}.npz"), "walk")
        self.assertEqual(
            motion_key_from_path(f"walk{DEFAULT_STRIP_SUFFIX}_extra.npz"), f"walk{DEFAULT_STRIP_SUFFIX}_extra",
        )
        self.assertEqual(motion_key_from_path("walk_custom.npz", strip_suffix="_custom"), "walk")
        self.assertEqual(motion_key_from_path("walk_custom.npz", strip_suffix=""), "walk_custom")
        for filename in ("metadata.npz", "walk_metadata.npz", f"walk_metadata{DEFAULT_STRIP_SUFFIX}.npz"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                motion_key_from_path(filename)

    @staticmethod
    def write_npz(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **data)

    @staticmethod
    def write_smpl(directory, key, frames=5):
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(fps=50., pose_aa=np.zeros((frames, 72), dtype=np.float32),
                         smpl_joints=np.zeros((frames, 24, 3), dtype=np.float32),
                         transl=np.zeros((frames, 3), dtype=np.float32)), directory / f"{key}.pkl")

    def test_recursive_batch_pairs_normalized_stems_and_writes_loader_wrappers(self):
        data, _ = self.fixture()
        expected = convert_motion(data)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "input/sub" / f"walk{DEFAULT_STRIP_SUFFIX}.npz", root / "input/run.npz"]
            for path in paths:
                self.write_npz(path, data)
            before = {p: p.read_bytes() for p in paths}
            for key in ("walk", "run"):
                self.write_smpl(root / "smpl", key)
            summary = convert_directory(root / "input", root / "out", smpl_dir=root / "smpl")
            self.assertEqual(summary["files"], 2)
            self.assertEqual(summary["total_frames"], 10)
            self.assertEqual(summary["output_order"], "mujoco")
            for key in ("walk", "run"):
                payload = joblib.load(root / "out" / f"{key}.pkl")
                self.assertEqual(list(payload), [key])
                self.assertEqual(payload[key]["pose_aa"].shape, (5, 32, 3))
                self.assertEqual(payload[key]["joint_names"], list(N3_MUJOCO_JOINT_NAMES))
                self.assertEqual(payload[key]["body_names"], list(N3_MUJOCO_BODY_NAMES))
                self.assertEqual(payload[key]["motion_order"], "mujoco")
                # Compare the stored arrays directly: normalizing first would
                # conceal a regression that still writes the former BFS order.
                for field in ("dof", "pose_aa", "root_trans_offset", "root_rot"):
                    np.testing.assert_array_equal(payload[key][field], expected[field], err_msg=field)
            for path in paths:
                self.assertEqual(path.read_bytes(), before[path])

    def test_batch_reads_optional_robot_and_quaternion_metadata_instead_of_ignoring_it(self):
        data, native = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for field, value in (("robot_type", "g1"), ("quat_order", "xyzw")):
                source = root / f"{field}.npz"
                self.write_npz(source, {**data, field: value})
                output = root / field
                with self.subTest(field=field), self.assertRaises(ValueError):
                    convert_directory(source, output)
                self.assertEqual(list(output.glob("**/*.pkl")), [])
            compatible = deepcopy(data)
            compatible.update(robot_type=np.asarray([b"n3"]), quat_order=np.asarray([b"xyzw"]))
            compatible["body_quat_w"] = compatible["body_quat_w"][:, :, [1, 2, 3, 0]]
            self.write_npz(root / "compatible.npz", compatible)
            summary = convert_directory(root / "compatible.npz", root / "valid", quat_order="xyzw")
            self.assertEqual(summary["files"], 1)
            converted = joblib.load(root / "valid/compatible.pkl")["compatible"]
            np.testing.assert_array_equal(normalize_n3_motion_order(converted)["dof"], native)
            np.testing.assert_allclose(converted["pose_aa"][:, 0, 2], np.pi / 2, atol=2e-7, rtol=0)

    def test_batch_rejects_duplicate_stems_existing_targets_and_missing_pairs_before_publish(self):
        data, _ = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_npz(root / "duplicates" / f"walk{DEFAULT_STRIP_SUFFIX}.npz", data)
            self.write_npz(root / "duplicates/nested/walk.npz", data)
            with self.assertRaises((ValueError, FileExistsError)):
                convert_directory(root / "duplicates", root / "out-duplicates")
            self.assertEqual(list((root / "out-duplicates").glob("**/*.pkl")), [])
            for key in ("a", "z"):
                self.write_npz(root / "existing" / f"{key}.npz", data)
            out = root / "out-existing"
            out.mkdir()
            (out / "z.pkl").write_bytes(b"existing data must remain")
            with self.assertRaises(FileExistsError):
                convert_directory(root / "existing", out)
            self.assertFalse((out / "a.pkl").exists())
            self.assertEqual((out / "z.pkl").read_bytes(), b"existing data must remain")
            self.write_smpl(root / "smpl", "a")
            with self.assertRaises((ValueError, FileNotFoundError)):
                convert_directory(root / "existing", root / "out-missing", smpl_dir=root / "smpl")
            self.assertEqual(list((root / "out-missing").glob("**/*.pkl")), [])

    def test_invalid_later_input_leaves_no_published_pkl_and_dry_run_is_readonly(self):
        good, _ = self.fixture()
        bad = deepcopy(good)
        bad["joint_pos"][0, 0] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_npz(root / "input/a.npz", good)
            self.write_npz(root / "input/z.npz", bad)
            with self.assertRaises(ValueError):
                convert_directory(root / "input", root / "out")
            self.assertEqual(list((root / "out").glob("**/*.pkl")), [])
            (root / "input/z.npz").unlink()
            summary = convert_directory(root / "input", root / "out-dry", dry_run=True)
            self.assertEqual(summary["files"], 1)
            self.assertEqual(summary["total_frames"], 5)
            self.assertEqual(list((root / "out-dry").glob("**/*.pkl")), [])

    def test_explicit_overwrite_preserves_previous_outputs_until_every_input_validates(self):
        good, native = self.fixture()
        bad = deepcopy(good)
        bad["body_quat_w"][:, -1] = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_npz(root / "input/a.npz", good)
            self.write_npz(root / "input/z.npz", bad)
            out = root / "out"
            out.mkdir()
            previous = b"previous output must survive validation failure"
            (out / "a.pkl").write_bytes(previous)
            with self.assertRaises(ValueError):
                convert_directory(root / "input", out, overwrite=True)
            self.assertEqual((out / "a.pkl").read_bytes(), previous)
            self.assertFalse((out / "z.pkl").exists())
            self.write_npz(root / "input/z.npz", good)
            summary = convert_directory(root / "input", out, overwrite=True)
            self.assertEqual(summary["files"], 2)
            for key in ("a", "z"):
                converted = joblib.load(out / f"{key}.pkl")[key]
                np.testing.assert_array_equal(normalize_n3_motion_order(converted)["dof"], native)

    def test_overwrite_rejects_directory_and_symlink_targets_before_publishing(self):
        data, _ = self.fixture()
        for kind in ("directory", "symlink", "broken_symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for key in ("a", "z"):
                    self.write_npz(root / "input" / f"{key}.npz", data)
                out = root / "out"
                out.mkdir()
                previous = b"earlier regular output must survive invalid later target"
                (out / "a.pkl").write_bytes(previous)
                outside = root / "external.pkl"
                if kind == "directory":
                    (out / "z.pkl").mkdir()
                else:
                    if kind == "symlink":
                        outside.write_bytes(b"external contents must survive")
                    (out / "z.pkl").symlink_to(outside)
                with self.assertRaises((ValueError, FileExistsError, OSError)):
                    convert_directory(root / "input", out, overwrite=True)
                self.assertEqual((out / "a.pkl").read_bytes(), previous)
                if kind == "directory":
                    self.assertTrue((out / "z.pkl").is_dir())
                else:
                    self.assertTrue((out / "z.pkl").is_symlink())
                    if kind == "symlink":
                        self.assertEqual(outside.read_bytes(), b"external contents must survive")
                    else:
                        self.assertFalse(outside.exists())

    def test_explicit_joint_order_permits_missing_metadata_but_auto_cannot_guess(self):
        data, native = self.fixture()
        data["joint_pos"] = native.copy()
        del data["joint_names"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_npz(root / "clip.npz", data)
            with self.assertRaises(ValueError):
                convert_directory(root / "clip.npz", root / "auto")
            for profile, names in (("mujoco", N3_MUJOCO_JOINT_NAMES), ("isaaclab", N3_ISAACLAB_DOF_NAMES)):
                data["joint_pos"] = native[:, [N3_MUJOCO_JOINT_NAMES.index(name) for name in names]]
                self.write_npz(root / "clip.npz", data)
                convert_directory(root / "clip.npz", root / profile, joint_order=profile)
                payload = joblib.load(root / profile / "clip.pkl")["clip"]
                np.testing.assert_array_equal(normalize_n3_motion_order(payload)["dof"], native)

    def test_legacy_0905_actual_bfs_profile_needs_no_external_names_json(self):
        data, native = self.fixture()
        aliases = {alias: name for name, alias in N3_TRAINING_TO_DEPLOYMENT_JOINT_NAMES.items()}
        source_names = [aliases[name] for name in N3_0905_ISAACLAB_DOF_NAMES]
        # Independently checked against real old-export metadata and current
        # GPU14 articulation names: renaming does not change the 29 DOF order.
        self.assertEqual(source_names, list(N3_ISAACLAB_DOF_NAMES))
        data["joint_pos"] = native[:, [N3_MUJOCO_JOINT_NAMES.index(name) for name in source_names]]
        data["body_pos_w"] = data["body_pos_w"][:, -1:]
        data["body_quat_w"] = data["body_quat_w"][:, -1:]
        del data["joint_names"], data["body_names"]
        original = deepcopy(data)
        expected = convert_motion(data, N3_0905_ISAACLAB_DOF_NAMES, root_body_index=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / f"clip{DEFAULT_STRIP_SUFFIX}.npz"
            self.write_npz(source, data)
            source_bytes = source.read_bytes()
            for selection in (None, "isaaclab", "mujoco"):
                output_order = "mujoco" if selection is None else selection
                destination = root / (selection or "default")
                kwargs = {} if selection is None else {"output_order": selection}
                summary = convert_directory(source, destination, joint_order="n3_0905", **kwargs)
                self.assertEqual(summary["output_order"], output_order)
                result = joblib.load(destination / "clip.pkl")["clip"]
                joints = N3_MUJOCO_JOINT_NAMES if output_order == "mujoco" else N3_ISAACLAB_DOF_NAMES
                bodies = N3_MUJOCO_BODY_NAMES if output_order == "mujoco" else N3_ISAACLAB_BODY_NAMES
                self.assertEqual(result["joint_names"], list(joints))
                self.assertEqual(result["body_names"], list(bodies))
                self.assertEqual(result["motion_order"], output_order)
                np.testing.assert_array_equal(result["dof"], native[:, [N3_MUJOCO_JOINT_NAMES.index(n) for n in joints]])
                np.testing.assert_array_equal(result["pose_aa"], expected["pose_aa"][:, [N3_MUJOCO_BODY_NAMES.index(n) for n in bodies]])
                self.assertEqual(result["dof"].shape, (5, 29))
                self.assertEqual(result["pose_aa"].shape, (5, 32, 3))
                self.assertEqual(result["source_joint_names"], list(N3_0905_ISAACLAB_DOF_NAMES))
                self.assertEqual(result["source_root_body_index"], 0)
                self.assertEqual(result["source_joint_order"], "n3_0905")
                for field in ("root_trans_offset", "root_rot"):
                    np.testing.assert_array_equal(result[field], expected[field])
                for toe in ("l_toe_link", "r_toe_link"):
                    np.testing.assert_array_equal(result["pose_aa"][:, bodies.index(toe)], 0)
            self.assertEqual(source.read_bytes(), source_bytes)
        for field in data:
            np.testing.assert_array_equal(data[field], original[field], err_msg=field)

    def test_bfs_and_native_storage_preserve_named_values_and_source_arrays(self):
        data, _ = self.fixture()
        native = convert_motion(data)
        before = deepcopy(native)
        default = order_motion_for_storage(native)
        self.assertEqual(default["motion_order"], "mujoco")
        self.assertEqual(default["joint_names"], list(N3_MUJOCO_JOINT_NAMES))
        self.assertEqual(default["body_names"], list(N3_MUJOCO_BODY_NAMES))
        for key in ("dof", "pose_aa", "root_trans_offset", "root_rot"):
            np.testing.assert_array_equal(default[key], native[key], err_msg=key)
        bfs = order_motion_for_storage(native, "isaaclab")
        np.testing.assert_array_equal(bfs["dof"], native["dof"][:, [N3_MUJOCO_JOINT_NAMES.index(n) for n in N3_ISAACLAB_DOF_NAMES]])
        np.testing.assert_array_equal(bfs["pose_aa"], native["pose_aa"][:, [N3_MUJOCO_BODY_NAMES.index(n) for n in N3_ISAACLAB_BODY_NAMES]])
        recovered = normalize_n3_motion_order(bfs)
        for key in ("dof", "pose_aa", "root_trans_offset", "root_rot"):
            np.testing.assert_array_equal(recovered[key], native[key])
            np.testing.assert_array_equal(native[key], before[key])
        native_again = order_motion_for_storage(bfs)
        self.assertEqual(native_again["motion_order"], "mujoco")
        np.testing.assert_array_equal(native_again["dof"], native["dof"])
        np.testing.assert_array_equal(native_again["pose_aa"], native["pose_aa"])
        with self.assertRaises(ValueError):
            order_motion_for_storage(native, "unknown")

    def test_cli_legacy_bfs_input_defaults_to_native_and_explicit_bfs_remains_supported(self):
        data, native = self.fixture()
        data["joint_pos"] = native[:, [N3_MUJOCO_JOINT_NAMES.index(n) for n in N3_ISAACLAB_DOF_NAMES]]
        data["body_pos_w"] = data["body_pos_w"][:, -1:]
        data["body_quat_w"] = data["body_quat_w"][:, -1:]
        del data["joint_names"], data["body_names"]
        expected = convert_motion(data, N3_0905_ISAACLAB_DOF_NAMES, root_body_index=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / f"clip{DEFAULT_STRIP_SUFFIX}.npz"
            self.write_npz(source, data)
            source_bytes = source.read_bytes()
            for output_order, flags in (("mujoco", []), ("isaaclab", ["--output-order", "isaaclab"])):
                command = [sys.executable, str(SCRIPT), "--input", str(source),
                           "--output", str(root / output_order), "--joint-order", "n3_0905", *flags]
                result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["output_order"], output_order)
                stored = joblib.load(root / output_order / "clip.pkl")["clip"]
                joints = N3_MUJOCO_JOINT_NAMES if output_order == "mujoco" else N3_ISAACLAB_DOF_NAMES
                bodies = N3_MUJOCO_BODY_NAMES if output_order == "mujoco" else N3_ISAACLAB_BODY_NAMES
                self.assertEqual(stored["motion_order"], output_order)
                self.assertEqual(stored["joint_names"], list(joints))
                self.assertEqual(stored["body_names"], list(bodies))
                np.testing.assert_array_equal(stored["dof"], native[:, [N3_MUJOCO_JOINT_NAMES.index(n) for n in joints]])
                np.testing.assert_array_equal(stored["pose_aa"], expected["pose_aa"][:, [N3_MUJOCO_BODY_NAMES.index(n) for n in bodies]])
                self.assertEqual(source.read_bytes(), source_bytes)

    def test_converted_output_loads_real_motion_library_and_adaptive_sampler(self):
        import torch
        from easydict import EasyDict
        from gear_sonic.trl.utils.order_converter import N3Converter
        from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot
        from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

        data, native = self.fixture()
        converter = N3Converter()
        mapping = converter.get_isaaclab_to_mujoco_mapping()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_npz(root / "clip.npz", data)
            convert_directory(root / "clip.npz", root / "robot")
            self.write_smpl(root / "smpl", "clip")
            cfg = EasyDict(
                converter.motion_metadata,
                asset=EasyDict(assetRoot=str((REPO / N3_MJCF_PATH).parent), assetFileName=Path(N3_MJCF_PATH).name),
                extend_config=[],
            )
            cfg.update(mapping)
            cfg.body_indexes = torch.tensor([N3_ISAACLAB_BODY_NAMES.index(name) for name in N3_TRACKED_BODIES])
            cfg.body_indexes_data = cfg.body_indexes.tolist()
            cfg.motion_file, cfg.smpl_motion_file = str(root / "robot"), str(root / "smpl")
            cfg.target_fps, cfg.multi_thread = 50, False
            cfg.adaptive_sampling = {"enable": True}
            # Keep real data decoding, FK, coordinate reordering and adaptive
            # bins. A single clip needs no renderer or multiprocessing server.
            with patch.object(Humanoid_Batch, "load_mesh"), patch(
                "gear_sonic.utils.motion_lib.motion_lib_base.mp.Manager",
                return_value=SimpleNamespace(Queue=Queue),
            ):
                library = MotionLibRobot(cfg, 2, "cpu")
                library.load_motions_for_training(max_num_seqs=1)
            self.assertEqual(library._num_unique_motions, 1)
            self.assertEqual(library.adp_samp_num_frames.tolist(), [5])
            self.assertEqual(library.motion_has_smpl.tolist(), [True])
            self.assertEqual(library.body_pos_w.shape, (5, 15, 3))
            np.testing.assert_allclose(library.dof_pos.numpy(), native[:, mapping["mujoco_to_isaaclab_dof"]],
                                       atol=2e-6, rtol=2e-6)

    def test_cli_help_and_module_import_need_no_torch_or_isaac(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=REPO,
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--joint-order", result.stdout)
        command = ("import sys,json; import gear_sonic.data_process.convert_n3_npz_to_motion_lib; "
                   "print(json.dumps([m for m in sys.modules if m == 'torch' or m.startswith('isaac')]))")
        result = subprocess.run([sys.executable, "-c", command], cwd=REPO,
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
