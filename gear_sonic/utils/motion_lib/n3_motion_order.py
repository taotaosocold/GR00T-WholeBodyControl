"""Normalize named N3 motion storage to the MJCF order expected by FK.

PKL storage may use any explicitly named order. Legacy unnamed motions retain
their existing native order. This boundary never changes the robot's FK tree
or the separate native-to-live-simulator permutation applied after FK.
"""

from collections.abc import Mapping

import numpy as np

from gear_sonic.envs.env_utils.n3_utils import N3_MUJOCO_BODY_NAMES, N3_MUJOCO_JOINT_NAMES


def _names(values, label):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"N3 {label} must be a one-dimensional sequence")
    names = []
    for value in array.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str) or not value:
            raise ValueError(f"N3 {label} must contain nonempty names")
        names.append(value)
    if len(set(names)) != len(names):
        raise ValueError(f"N3 {label} contains duplicate names")
    return tuple(names)


def _array(value, label):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"N3 {label} must contain finite real values")
    return array


def normalize_n3_motion_order(entry):
    """Return a native-order view of a named robot motion without mutating it.

    ``joint_names`` describes the optional ``dof`` columns, and ``body_names``
    describes ``pose_aa`` slots. A named 30-body motion must contain base_link
    and all 29 actuated links; the two absent fixed toes are inserted as zero
    rotations. Named 32-body motions must include every physical body.
    Missing both name fields means the legacy native 30/32-body contract;
    supplying only one field is rejected rather than guessing the other order.
    """
    if not isinstance(entry, Mapping):
        raise ValueError("N3 motion entry must be a mapping")
    declared_order = None
    if "motion_order" in entry:
        order = np.asarray(entry["motion_order"])
        if order.size != 1:
            raise ValueError("N3 motion_order must be isaaclab or mujoco")
        declared_order = order.reshape(-1)[0]
        if isinstance(declared_order, bytes):
            declared_order = declared_order.decode("utf-8")
        if not isinstance(declared_order, str) or declared_order not in {"isaaclab", "mujoco"}:
            raise ValueError("N3 motion_order must be isaaclab or mujoco")
    has_joints, has_bodies = "joint_names" in entry, "body_names" in entry
    if not has_joints and not has_bodies:
        if declared_order == "isaaclab":
            raise ValueError("N3 isaaclab storage requires joint_names and body_names")
        return entry
    if has_joints != has_bodies:
        raise ValueError("Named N3 motion requires both joint_names and body_names")
    joint_names = _names(entry["joint_names"], "joint_names")
    body_names = _names(entry["body_names"], "body_names")
    if len(joint_names) != len(N3_MUJOCO_JOINT_NAMES) or set(joint_names) != set(N3_MUJOCO_JOINT_NAMES):
        raise ValueError("N3 joint_names must contain exactly the 29 native joint names")
    actuated_bodies = {name.removesuffix("_joint") + "_link" for name in N3_MUJOCO_JOINT_NAMES}
    compact_bodies = actuated_bodies | {"base_link"}
    if not (
        len(body_names) == 32 and set(body_names) == set(N3_MUJOCO_BODY_NAMES)
        or len(body_names) == 30 and set(body_names) == compact_bodies
    ):
        raise ValueError("N3 body_names must contain all 32 bodies or base_link plus all 29 actuated links")

    normalized = dict(entry)
    frames = None
    if "pose_aa" in entry:
        pose = _array(entry["pose_aa"], "pose_aa")
        if pose.ndim != 3 or pose.shape[1:] != (len(body_names), 3) or pose.shape[0] < 1:
            raise ValueError("N3 pose_aa must have shape [T,len(body_names),3]")
        frames = pose.shape[0]
        if len(body_names) == 32:
            normalized["pose_aa"] = pose[:, [body_names.index(name) for name in N3_MUJOCO_BODY_NAMES], :]
        else:
            expanded = np.zeros((frames, 32, 3), dtype=pose.dtype)
            indices = [N3_MUJOCO_BODY_NAMES.index(name) for name in body_names]
            expanded[:, indices, :] = pose
            normalized["pose_aa"] = expanded
    if "dof" in entry:
        dof = _array(entry["dof"], "dof")
        if dof.ndim != 2 or dof.shape[1] != 29 or dof.shape[0] < 1:
            raise ValueError("N3 dof must have shape [T,29]")
        if frames is not None and dof.shape[0] != frames:
            raise ValueError("N3 pose_aa and dof frame counts differ")
        normalized["dof"] = dof[:, [joint_names.index(name) for name in N3_MUJOCO_JOINT_NAMES]]
    normalized["joint_names"] = list(N3_MUJOCO_JOINT_NAMES)
    normalized["body_names"] = list(N3_MUJOCO_BODY_NAMES)
    if declared_order is not None:
        normalized["motion_order"] = "mujoco"
    return normalized
