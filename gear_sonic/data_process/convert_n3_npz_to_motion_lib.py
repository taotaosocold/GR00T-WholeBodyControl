#!/usr/bin/env python3
r"""Convert N3 NPZ motions to the per-clip PKL format used by motion_lib.

Input positions are metres in a Z-up world, joint angles are radians, and
``body_quat_w`` defaults to Isaac Lab's wxyz convention. The source frame rate
and root trajectory are preserved. This changes the storage format and joint
ordering; it does not retarget motions between different robot geometries.

Examples, from the repository root::

    # Named NPZ files use their joint_names/body_names metadata automatically.
    python gear_sonic/data_process/convert_n3_npz_to_motion_lib.py \
        --input data/n3/npz --output data/n3/robot

    # Verified legacy n3_0905 exports have a built-in actual PhysX BFS order.
    python gear_sonic/data_process/convert_n3_npz_to_motion_lib.py \
        --input data/n3/npz --output data/n3/robot \
        --joint-order n3_0905 --output-order isaaclab

Only NumPy, SciPy and joblib are required; Isaac Lab and CUDA are not imported.
"""

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.envs.env_utils.n3_utils import (  # noqa: E402
    N3_ISAACLAB_BODY_NAMES,
    N3_ISAACLAB_DOF_NAMES,
    N3_MJCF_PATH,
    N3_MUJOCO_BODY_NAMES,
    N3_MUJOCO_JOINT_NAMES,
    N3_TRAINING_TO_DEPLOYMENT_JOINT_NAMES,
)

DEFAULT_STRIP_SUFFIX = "_from_g1_noetix_n3_0905"
_FIELDS = ("fps", "joint_pos", "body_pos_w", "body_quat_w")

# Actual joint_names metadata from the old Isaac/PhysX training export:
# tracking_n3_mha_mimic/2026-09-15_01-22-53_wf_v5/exported/policy.onnx
# SHA256 838e402e874e7497eb6604d8e6c497e43f8c8dbc58c978ae09e7b132ea70aa80.
# This BFS sibling order differs from simply walking the URDF XML children.
# Verified against old-URDF FK for 189 legacy NPZ clips (945 sampled frames).
N3_0905_ISAACLAB_DOF_NAMES = (
    "l_hip_pitch_joint", "r_hip_pitch_joint", "waist_yaw_joint",
    "l_hip_roll_joint", "r_hip_roll_joint", "head_pitch_joint",
    "l_arm_pitch_joint", "r_arm_pitch_joint", "l_hip_yaw_joint", "r_hip_yaw_joint",
    "head_yaw_joint", "l_arm_roll_joint", "r_arm_roll_joint",
    "l_knee_pitch_joint", "r_knee_pitch_joint", "l_arm_yaw_joint", "r_arm_yaw_joint",
    "l_ankle_pitch_joint", "r_ankle_pitch_joint", "l_elbow_pitch_joint", "r_elbow_pitch_joint",
    "l_ankle_roll_joint", "r_ankle_roll_joint", "l_hand_yaw_joint", "r_hand_yaw_joint",
    "l_hand_roll_joint", "r_hand_roll_joint", "l_hand_pitch_joint", "r_hand_pitch_joint",
)
_JOINT_ALIASES = {alias: name for name, alias in N3_TRAINING_TO_DEPLOYMENT_JOINT_NAMES.items()}


def _canonical_joint_names(names):
    return tuple(_JOINT_ALIASES.get(name, name) for name in names)


def _names(value, label):
    values = np.asarray(value)
    if values.ndim != 1 or values.dtype.kind not in "USO":
        raise ValueError(f"{label} must be a one-dimensional sequence of names")
    result = []
    for name in values.tolist():
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} must contain nonempty strings")
        result.append(name)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate names")
    return tuple(result)


def _numeric(value, label):
    values = np.asarray(value)
    if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError(f"{label} must contain finite real numbers")
    return values.astype(np.float64, copy=False)


def _float32(value, label):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} cannot be represented as finite float32 values")
    return result


def _text_scalar(value, label):
    values = np.asarray(value)
    if values.size != 1:
        raise ValueError(f"{label} must be one string")
    result = values.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    if not isinstance(result, str):
        raise ValueError(f"{label} must be one string")
    return result


@lru_cache(maxsize=1)
def _n3_layout():
    """Read axes and actuated body slots from the vendored training asset."""
    path = Path(__file__).resolve().parents[2] / N3_MJCF_PATH
    world = ET.parse(path).getroot().find("worldbody")
    if world is None:
        raise ValueError(f"N3 MJCF has no worldbody: {path}")
    bodies = tuple(world.iter("body"))
    if tuple(body.get("name") for body in bodies) != N3_MUJOCO_BODY_NAMES:
        raise ValueError("N3 MJCF body order differs from n3_utils metadata")
    joints, slots, axes = [], [], []
    for index, body in enumerate(bodies):
        for joint in body.findall("joint"):
            if joint.get("type", "hinge") == "free":
                continue
            if joint.get("type", "hinge") != "hinge":
                raise ValueError("N3 conversion requires scalar hinge joints")
            axis = np.asarray([float(v) for v in joint.get("axis", "0 0 1").split()])
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
                raise ValueError(f"Invalid N3 hinge axis: {joint.get('name')}")
            joints.append(joint.get("name"))
            slots.append(index)
            axes.append(axis / np.linalg.norm(axis))
    if tuple(joints) != N3_MUJOCO_JOINT_NAMES:
        raise ValueError("N3 MJCF joint order differs from n3_utils metadata")
    return np.asarray(slots), np.asarray(axes)


def convert_motion(data, source_joint_names=None, *, root_body_index=None, quat_order="wxyz"):
    """Return one motion_lib entry, without modifying the input arrays.

    Named NPZ metadata is authoritative. Legacy unnamed files require explicit
    source joint names and root body index. Quaternions within 1e-3 of unit
    length are normalized; other norms are rejected.
    """
    missing = [name for name in _FIELDS if name not in data]
    if missing:
        raise ValueError(f"Missing NPZ fields: {', '.join(missing)}")
    if "robot_type" in data and _text_scalar(data["robot_type"], "robot_type") != "n3":
        raise ValueError("NPZ robot_type must be n3 for N3 conversion")
    if "quat_order" in data and _text_scalar(data["quat_order"], "quat_order") != quat_order:
        raise ValueError("Requested quat_order conflicts with NPZ quat_order metadata")
    fps = _numeric(data["fps"], "fps")
    if fps.size != 1 or fps.item() <= 0:
        raise ValueError("fps must be one finite positive scalar")
    joints = _numeric(data["joint_pos"], "joint_pos")
    positions = _numeric(data["body_pos_w"], "body_pos_w")
    quaternions = _numeric(data["body_quat_w"], "body_quat_w")
    if joints.ndim != 2 or joints.shape[1] != len(N3_MUJOCO_JOINT_NAMES) or joints.shape[0] < 3:
        raise ValueError("joint_pos must have shape [T,29], with at least 3 frames")
    frames = joints.shape[0]
    if positions.ndim != 3 or positions.shape[0] != frames or positions.shape[1] < 1 or positions.shape[2] != 3:
        raise ValueError("body_pos_w must have shape [T,B,3] with the same T as joint_pos")
    if quaternions.shape != (*positions.shape[:2], 4):
        raise ValueError("body_quat_w must have shape [T,B,4] matching body_pos_w")

    raw_names = _names(source_joint_names, "source_joint_names") if source_joint_names is not None else None
    names = _canonical_joint_names(raw_names) if raw_names is not None else None
    if "joint_names" in data:
        metadata_raw_names = _names(data["joint_names"], "joint_names")
        metadata_names = _canonical_joint_names(metadata_raw_names)
        if names is not None and names != metadata_names:
            raise ValueError("Explicit source joint order conflicts with NPZ joint_names metadata")
        names = metadata_names
        raw_names = metadata_raw_names
    if names is None:
        raise ValueError("NPZ has no joint_names; specify --joint-order or --joint-names-file")
    if len(names) != 29 or set(names) != set(N3_MUJOCO_JOINT_NAMES):
        missing_names = sorted(set(N3_MUJOCO_JOINT_NAMES) - set(names))
        extra = sorted(set(names) - set(N3_MUJOCO_JOINT_NAMES))
        raise ValueError(f"N3 source joint names mismatch: missing={missing_names}, extra={extra}")

    if root_body_index is not None and (
        isinstance(root_body_index, (bool, np.bool_))
        or not isinstance(root_body_index, (int, np.integer))
    ):
        raise ValueError("root_body_index must be an integer")
    if "body_names" in data:
        body_names = _names(data["body_names"], "body_names")
        if len(body_names) != positions.shape[1] or "base_link" not in body_names:
            raise ValueError("body_names must match B and include base_link")
        metadata_root = body_names.index("base_link")
        if root_body_index is not None and root_body_index != metadata_root:
            raise ValueError("root_body_index conflicts with base_link in body_names metadata")
        root_body_index = metadata_root
    if root_body_index is None:
        raise ValueError("NPZ has no body_names; specify --root-body-index for base_link")
    if not 0 <= root_body_index < positions.shape[1]:
        raise ValueError("root_body_index is outside the source body array")
    if quat_order not in {"wxyz", "xyzw"}:
        raise ValueError("quat_order must be wxyz or xyzw")

    root_quat = quaternions[:, root_body_index]
    if quat_order == "wxyz":
        root_quat = root_quat[:, [1, 2, 3, 0]]
    norms = np.linalg.norm(root_quat, axis=-1)
    if not np.isfinite(norms).all() or np.any(np.abs(norms - 1.0) > 1e-3):
        raise ValueError("Root quaternion must have unit norm (tolerance 1e-3)")
    root_quat = root_quat / norms[:, None]
    dof = joints[:, [names.index(name) for name in N3_MUJOCO_JOINT_NAMES]]
    slots, axes = _n3_layout()
    pose = np.zeros((frames, len(N3_MUJOCO_BODY_NAMES), 3), dtype=np.float64)
    pose[:, 0] = Rotation.from_quat(root_quat).as_rotvec()
    pose[:, slots] = dof[:, :, None] * axes[None]
    return {
        "fps": float(fps.item()),
        "root_trans_offset": _float32(positions[:, root_body_index], "root_trans_offset"),
        "pose_aa": _float32(pose, "pose_aa"),
        "dof": _float32(dof, "dof"),
        "root_rot": _float32(root_quat, "root_rot"),  # xyzw, as in the G1 converter
        "joint_names": list(N3_MUJOCO_JOINT_NAMES),
        "body_names": list(N3_MUJOCO_BODY_NAMES),
        "source_joint_names": list(raw_names),
        "source_root_body_index": int(root_body_index),
        "source_quat_order": quat_order,
        "robot_type": "n3",
        "motion_order": "mujoco",
    }


def order_motion_for_storage(entry, output_order="isaaclab"):
    """Save pose and DOFs in named BFS order, or retain MJCF native order.

    The training loader normalizes named N3 motions back to native FK order.
    This permutation changes no joint angles, root trajectory or frame times.
    """
    if output_order == "isaaclab":
        joint_names, body_names = N3_ISAACLAB_DOF_NAMES, N3_ISAACLAB_BODY_NAMES
    elif output_order == "mujoco":
        joint_names, body_names = N3_MUJOCO_JOINT_NAMES, N3_MUJOCO_BODY_NAMES
    else:
        raise ValueError(f"Unknown output_order: {output_order}")
    source_joints, source_bodies = list(entry["joint_names"]), list(entry["body_names"])
    return {
        **entry,
        "dof": entry["dof"][:, [source_joints.index(name) for name in joint_names]],
        "pose_aa": entry["pose_aa"][:, [source_bodies.index(name) for name in body_names]],
        "joint_names": list(joint_names),
        "body_names": list(body_names),
        "motion_order": output_order,
    }


def motion_key_from_path(path, strip_suffix=DEFAULT_STRIP_SUFFIX):
    """Match legacy retargeted filenames to the original SMPL sequence key."""
    key = Path(path).stem
    if strip_suffix and key.endswith(strip_suffix):
        key = key[:-len(strip_suffix)]
    if not key or key.endswith("metadata"):
        raise ValueError(f"Invalid or reserved motion key: {key!r}")
    return key


def _load_and_convert(path, source_joint_names, root_body_index, quat_order):
    with np.load(path, allow_pickle=False) as archive:
        fields = (*_FIELDS, "joint_names", "body_names", "robot_type", "quat_order")
        data = {key: archive[key] for key in fields if key in archive}
    return convert_motion(data, source_joint_names, root_body_index=root_body_index, quat_order=quat_order)


def convert_directory(input_path, output_dir, *, source_joint_names=None, joint_order="auto",
                      root_body_index=None, quat_order="wxyz", strip_suffix=DEFAULT_STRIP_SUFFIX,
                      overwrite=False, dry_run=False, smpl_dir=None, output_order="isaaclab"):
    """Convert a single NPZ or a recursive directory into one PKL per clip.

    All inputs are validated in a temporary staging directory before publishing
    any output. Existing files and duplicate sequence keys are rejected by
    default. Optional SMPL checking verifies filenames, not frame alignment.
    """
    profiles = {"auto": None, "isaaclab": N3_ISAACLAB_DOF_NAMES, "mujoco": N3_MUJOCO_JOINT_NAMES,
                "n3_0905": N3_0905_ISAACLAB_DOF_NAMES}
    if joint_order not in profiles:
        raise ValueError(f"Unknown joint_order: {joint_order}")
    if source_joint_names is not None and joint_order != "auto":
        raise ValueError("Use either source_joint_names or a joint_order profile")
    if output_order not in {"isaaclab", "mujoco"}:
        raise ValueError(f"Unknown output_order: {output_order}")
    if root_body_index is None and joint_order == "n3_0905":
        root_body_index = 0
    names = source_joint_names if source_joint_names is not None else profiles[joint_order]
    source, destination = Path(input_path), Path(output_dir)
    if source.is_file() and source.suffix.lower() == ".npz":
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.rglob("*.npz"))
    else:
        raise ValueError(f"Input must be an NPZ file or directory: {source}")
    if not paths:
        raise ValueError(f"No NPZ motion files found in {source}")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Output must be a directory: {destination}")
    if smpl_dir is not None and not Path(smpl_dir).is_dir():
        raise ValueError(f"SMPL directory does not exist: {smpl_dir}")
    plan = []
    seen = set()
    for path in paths:
        key = motion_key_from_path(path, strip_suffix)
        if key in seen:
            raise ValueError(f"Duplicate motion key after stripping suffix: {key}")
        seen.add(key)
        target = destination / f"{key}.pkl"
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"Output path is not a regular file: {target}")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output exists; use --overwrite explicitly: {target}")
        if smpl_dir is not None and not (Path(smpl_dir) / f"{key}.pkl").is_file():
            raise ValueError(f"Missing paired SMPL file: {key}.pkl")
        plan.append((path, key, target))

    total_frames = 0
    if dry_run:
        for path, _, _ in plan:
            try:
                entry = _load_and_convert(path, names, root_body_index, quat_order)
            except (ValueError, KeyError, OSError) as exc:
                raise ValueError(f"{path}: {exc}") from exc
            total_frames += entry["pose_aa"].shape[0]
    else:
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".n3-convert-", dir=destination) as staging:
            for path, key, _ in plan:
                try:
                    entry = _load_and_convert(path, names, root_body_index, quat_order)
                except (ValueError, KeyError, OSError) as exc:
                    raise ValueError(f"{path}: {exc}") from exc
                total_frames += entry["pose_aa"].shape[0]
                entry = order_motion_for_storage(entry, output_order)
                entry["source_joint_order"] = joint_order if source_joint_names is None else "custom"
                joblib.dump({key: entry}, Path(staging) / f"{key}.pkl", compress=3)
            # Recheck after staging so a late conflict also leaves all outputs untouched.
            for _, _, target in plan:
                if target.is_symlink() or (target.exists() and not target.is_file()):
                    raise ValueError(f"Output path is not a regular file: {target}")
                if target.exists() and not overwrite:
                    raise FileExistsError(f"Output appeared during conversion: {target}")
            for _, key, target in plan:
                staged = Path(staging) / f"{key}.pkl"
                if overwrite:
                    os.replace(staged, target)
                else:
                    os.link(staged, target)  # atomic publication without overwriting another file
    return {"files": len(plan), "total_frames": total_frames,
            "output_dir": str(destination), "output_order": output_order, "dry_run": dry_run}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="One NPZ file or directory (searched recursively)")
    parser.add_argument("--output", required=True, help="Output directory for individual robot PKLs")
    order = parser.add_mutually_exclusive_group()
    order.add_argument(
        "--joint-order", choices=("auto", "isaaclab", "mujoco", "n3_0905"), default="auto",
        help="auto reads metadata; n3_0905 is the verified old PhysX BFS export order",
    )
    parser.add_argument("--output-order", choices=("isaaclab", "mujoco"), default="isaaclab",
                        help="Save named BFS order (default) or MJCF native order")
    order.add_argument("--joint-names-file", help="JSON list of the 29 source column names in order")
    parser.add_argument(
        "--root-body-index", type=int, help="Source base_link body index; required without body_names",
    )
    parser.add_argument("--quat-order", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument("--strip-suffix", default=DEFAULT_STRIP_SUFFIX,
                        help="Remove this trailing filename suffix for SMPL pairing; use '' to keep it")
    parser.add_argument("--smpl-dir", help="Require a same-name PKL in this SMPL directory for every motion")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing output PKLs")
    parser.add_argument("--dry-run", action="store_true", help="Validate all clips without writing files")
    args = parser.parse_args(argv)
    try:
        names = None
        if args.joint_names_file:
            names = json.loads(Path(args.joint_names_file).read_text())
        result = convert_directory(
            args.input, args.output, source_joint_names=names, joint_order=args.joint_order,
            root_body_index=args.root_body_index, quat_order=args.quat_order,
            strip_suffix=args.strip_suffix, overwrite=args.overwrite, dry_run=args.dry_run,
            smpl_dir=args.smpl_dir, output_order=args.output_order,
        )
    except (ValueError, KeyError, OSError, ET.ParseError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
