"""Joint utility functions and ordering metadata for supported robots.

This module provides joint ordering constants and helper functions for mapping
between motion library data and robot joints.
"""

import torch

# G1 body joint names in IsaacLab order (29 DOF)
G1_ISAACLab_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# G1 hand joint names (14 DOF) - order from g1_43dof.yaml
G1_HAND_JOINTS = [
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

# Caches for joint indices
_body_joint_indices_cache = {}
_hand_joint_indices_cache = {}

G1_MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint",
    "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint",
    "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def get_robot_type(asset_or_name="g1") -> str:
    """Resolve an embodiment without importing simulator-dependent configs.

    Unlabelled legacy fixtures retain the G1 convention. N3 is recognized by
    its distinct named joints, never merely by a 29-dimensional tensor.
    """
    if isinstance(asset_or_name, str):
        if asset_or_name in {"g1", "g1_model_12_dex"}:
            return "g1"
        if asset_or_name == "n3":
            return "n3"
        raise ValueError(f"Unsupported physical WAM robot type: {asset_or_name!r}")
    explicit = getattr(asset_or_name, "sonic_robot_type", None)
    if explicit is not None:
        return get_robot_type(explicit)
    names = getattr(asset_or_name, "joint_names", ())
    return "n3" if any(name.startswith(("l_leg_", "r_leg_", "l_arm_shoulder_")) for name in names) else "g1"


def get_robot_joint_names(asset_or_name="g1", *, order="mujoco") -> tuple[str, ...]:
    """Return named body joints in motion/FK or policy observation order.

    Live N3 articulations supply their actual joint ordering; standalone tools
    use the URDF breadth-first order. G1 retains its historical ordering.
    """
    if order not in {"mujoco", "isaaclab"}:
        raise ValueError(f"Unknown robot joint order: {order!r}")
    kind = get_robot_type(asset_or_name)
    if kind == "g1":
        return G1_MUJOCO_JOINT_NAMES if order == "mujoco" else tuple(G1_ISAACLab_ORDER)
    from gear_sonic.envs.env_utils.n3_utils import N3_ISAACLAB_DOF_NAMES, N3_MUJOCO_JOINT_NAMES

    canonical = tuple(N3_MUJOCO_JOINT_NAMES)
    if order == "mujoco":
        return canonical
    actual = getattr(asset_or_name, "joint_names", None)
    if actual is None:
        return tuple(N3_ISAACLAB_DOF_NAMES)
    selected = tuple(name for name in actual if name in canonical)
    if len(selected) != len(canonical) or set(selected) != set(canonical):
        raise ValueError("N3 articulation must contain all 29 named joints exactly once")
    return selected


def get_robot_torso_name(asset_or_name="g1") -> str:
    return "waist_yaw_link" if get_robot_type(asset_or_name) == "n3" else "torso_link"


def get_robot_root_name(asset_or_name="g1") -> str:
    return "base_link" if get_robot_type(asset_or_name) == "n3" else "pelvis"


def get_robot_spec(asset_or_name="g1") -> dict:
    """Versioned ordering identity saved with physically trained policies."""
    return {
        "type": get_robot_type(asset_or_name),
        "joint_names": list(get_robot_joint_names(asset_or_name, order="mujoco")),
        "policy_joint_names": list(get_robot_joint_names(asset_or_name, order="isaaclab")),
        "root_body": get_robot_root_name(asset_or_name),
    }


def _get_joint_indices_by_names(asset, joint_names: list, cache: dict) -> torch.Tensor:
    """Get indices of specified joints in the robot's joint list."""
    cache_key = (id(asset), tuple(joint_names))
    if cache_key in cache:
        return cache[cache_key]

    robot_joint_names = asset.joint_names
    indices = [robot_joint_names.index(n) for n in joint_names if n in robot_joint_names]
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=asset.device)
    cache[cache_key] = indices_tensor
    return indices_tensor


def get_body_joint_indices(asset) -> torch.Tensor:
    """Select named policy body joints, including N3's two active head joints."""
    names = get_robot_joint_names(asset, order="isaaclab")
    return _get_joint_indices_by_names(asset, names, _body_joint_indices_cache)


def get_hand_joint_indices(asset) -> torch.Tensor:
    """Get indices of hand joints (14 DOF) using G1_HAND_JOINTS."""
    names = () if get_robot_type(asset) == "n3" else G1_HAND_JOINTS
    return _get_joint_indices_by_names(asset, names, _hand_joint_indices_cache)
