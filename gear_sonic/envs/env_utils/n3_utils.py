"""N3 V3 training-asset metadata; safe to import without Isaac Lab.

Joint/body native order follows the vendored training MJCF. Isaac Lab order
is the expected breadth-first URDF traversal with fixed toe links retained.
Each ``ISAACLAB_TO_MUJOCO`` array selects Isaac entries in MuJoCo order.
"""

import math

N3_MJCF_PATH = "gear_sonic/data/assets/robot_description/mjcf/n3.xml"
N3_URDF_PATH = "gear_sonic/data/assets/robot_description/urdf/n3/n3_v3_isaac.urdf"

N3_MUJOCO_JOINT_NAMES = (
    "waist_yaw_joint",
    "l_arm_shoulder_pitch_joint", "l_arm_shoulder_roll_joint", "l_arm_shoulder_yaw_joint",
    "l_arm_elbow_pitch_joint", "l_arm_elbow_yaw_joint", "l_arm_hand_roll_joint", "l_arm_hand_pitch_joint",
    "r_arm_shoulder_pitch_joint", "r_arm_shoulder_roll_joint", "r_arm_shoulder_yaw_joint",
    "r_arm_elbow_pitch_joint", "r_arm_elbow_yaw_joint", "r_arm_hand_roll_joint", "r_arm_hand_pitch_joint",
    "head_pitch_joint", "head_yaw_joint",
    "l_leg_hip_pitch_joint", "l_leg_hip_roll_joint", "l_leg_hip_yaw_joint", "l_leg_knee_pitch_joint",
    "l_leg_ankle_pitch_joint", "l_leg_ankle_roll_joint",
    "r_leg_hip_pitch_joint", "r_leg_hip_roll_joint", "r_leg_hip_yaw_joint", "r_leg_knee_pitch_joint",
    "r_leg_ankle_pitch_joint", "r_leg_ankle_roll_joint",
)

N3_MUJOCO_BODY_NAMES = (
    "base_link", "waist_yaw_link",
    "l_arm_shoulder_pitch_link", "l_arm_shoulder_roll_link", "l_arm_shoulder_yaw_link",
    "l_arm_elbow_pitch_link", "l_arm_elbow_yaw_link", "l_arm_hand_roll_link", "l_arm_hand_pitch_link",
    "r_arm_shoulder_pitch_link", "r_arm_shoulder_roll_link", "r_arm_shoulder_yaw_link",
    "r_arm_elbow_pitch_link", "r_arm_elbow_yaw_link", "r_arm_hand_roll_link", "r_arm_hand_pitch_link",
    "head_pitch_link", "head_yaw_link",
    "l_leg_hip_pitch_link", "l_leg_hip_roll_link", "l_leg_hip_yaw_link", "l_leg_knee_pitch_link",
    "l_leg_ankle_pitch_link", "l_leg_ankle_roll_link", "l_toe_link",
    "r_leg_hip_pitch_link", "r_leg_hip_roll_link", "r_leg_hip_yaw_link", "r_leg_knee_pitch_link",
    "r_leg_ankle_pitch_link", "r_leg_ankle_roll_link", "r_toe_link",
)

N3_ISAACLAB_DOF_NAMES = (
    "waist_yaw_joint", "l_leg_hip_pitch_joint", "r_leg_hip_pitch_joint",
    "l_arm_shoulder_pitch_joint", "r_arm_shoulder_pitch_joint", "head_pitch_joint",
    "l_leg_hip_roll_joint", "r_leg_hip_roll_joint",
    "l_arm_shoulder_roll_joint", "r_arm_shoulder_roll_joint", "head_yaw_joint",
    "l_leg_hip_yaw_joint", "r_leg_hip_yaw_joint", "l_arm_shoulder_yaw_joint", "r_arm_shoulder_yaw_joint",
    "l_leg_knee_pitch_joint", "r_leg_knee_pitch_joint", "l_arm_elbow_pitch_joint", "r_arm_elbow_pitch_joint",
    "l_leg_ankle_pitch_joint", "r_leg_ankle_pitch_joint", "l_arm_elbow_yaw_joint", "r_arm_elbow_yaw_joint",
    "l_leg_ankle_roll_joint", "r_leg_ankle_roll_joint", "l_arm_hand_roll_joint", "r_arm_hand_roll_joint",
    "l_arm_hand_pitch_joint", "r_arm_hand_pitch_joint",
)

N3_ISAACLAB_BODY_NAMES = (
    "base_link", "waist_yaw_link", "l_leg_hip_pitch_link", "r_leg_hip_pitch_link",
    "l_arm_shoulder_pitch_link", "r_arm_shoulder_pitch_link", "head_pitch_link",
    "l_leg_hip_roll_link", "r_leg_hip_roll_link", "l_arm_shoulder_roll_link", "r_arm_shoulder_roll_link",
    "head_yaw_link", "l_leg_hip_yaw_link", "r_leg_hip_yaw_link",
    "l_arm_shoulder_yaw_link", "r_arm_shoulder_yaw_link", "l_leg_knee_pitch_link", "r_leg_knee_pitch_link",
    "l_arm_elbow_pitch_link", "r_arm_elbow_pitch_link", "l_leg_ankle_pitch_link", "r_leg_ankle_pitch_link",
    "l_arm_elbow_yaw_link", "r_arm_elbow_yaw_link", "l_leg_ankle_roll_link", "r_leg_ankle_roll_link",
    "l_arm_hand_roll_link", "r_arm_hand_roll_link", "l_toe_link", "r_toe_link",
    "l_arm_hand_pitch_link", "r_arm_hand_pitch_link",
)

N3_ISAACLAB_TO_MUJOCO_DOF = tuple(N3_ISAACLAB_DOF_NAMES.index(name) for name in N3_MUJOCO_JOINT_NAMES)
N3_MUJOCO_TO_ISAACLAB_DOF = tuple(N3_MUJOCO_JOINT_NAMES.index(name) for name in N3_ISAACLAB_DOF_NAMES)
N3_ISAACLAB_TO_MUJOCO_BODY = tuple(N3_ISAACLAB_BODY_NAMES.index(name) for name in N3_MUJOCO_BODY_NAMES)
N3_MUJOCO_TO_ISAACLAB_BODY = tuple(N3_MUJOCO_BODY_NAMES.index(name) for name in N3_ISAACLAB_BODY_NAMES)

N3_TRACKED_BODIES = (
    "base_link", "l_leg_hip_roll_link", "l_leg_knee_pitch_link", "l_leg_ankle_roll_link",
    "r_leg_hip_roll_link", "r_leg_knee_pitch_link", "r_leg_ankle_roll_link", "waist_yaw_link",
    "l_arm_shoulder_roll_link", "l_arm_elbow_pitch_link", "l_arm_hand_pitch_link",
    "r_arm_shoulder_roll_link", "r_arm_elbow_pitch_link", "r_arm_hand_pitch_link",
    "head_yaw_link",
)
N3_LOWER_BODY_JOINT_NAMES = tuple(
    name for name in N3_MUJOCO_JOINT_NAMES if name.startswith(("l_leg_", "r_leg_"))
)
N3_WRIST_JOINT_NAMES = (
    "l_arm_elbow_yaw_joint", "l_arm_hand_roll_joint", "l_arm_hand_pitch_joint",
    "r_arm_elbow_yaw_joint", "r_arm_hand_roll_joint", "r_arm_hand_pitch_joint",
)

INIT_POS = (0.0, 0.0, 0.724)
INIT_JOINT_POS = {
    "l_leg_hip_pitch_joint": -0.1495,
    "l_leg_knee_pitch_joint": 0.3215,
    "l_leg_ankle_pitch_joint": -0.1720,
    "r_leg_hip_pitch_joint": -0.1495,
    "r_leg_knee_pitch_joint": 0.3215,
    "r_leg_ankle_pitch_joint": -0.1720,
    "l_arm_shoulder_roll_joint": 0.2,
    "r_arm_shoulder_roll_joint": -0.2,
}
N3_DEFAULT_JOINT_POS = tuple(INIT_JOINT_POS.get(name, 0.0) for name in N3_MUJOCO_JOINT_NAMES)

# Exact make_mjlab_cfg parameters from the 2026-11-03 training snapshot.
N3_ACTUATOR_GROUPS = {
    "legs": {
        "joint_names": tuple(name for name in N3_LOWER_BODY_JOINT_NAMES if "ankle" not in name),
        "kp": 200.0, "kd": 15.0, "effort": 145.0, "armature": 0.06086, "friction": 0.01,
    },
    "ankle_pitch": {
        "joint_names": ("l_leg_ankle_pitch_joint", "r_leg_ankle_pitch_joint"),
        "kp": 80.0, "kd": 4.0, "effort": 50.0, "armature": 0.0029075, "friction": 0.01,
    },
    "ankle_roll": {
        "joint_names": ("l_leg_ankle_roll_joint", "r_leg_ankle_roll_joint"),
        "kp": 50.0, "kd": 2.0, "effort": 50.0, "armature": 0.0029075, "friction": 0.01,
    },
    "waist": {
        "joint_names": ("waist_yaw_joint",),
        "kp": 80.0, "kd": 3.0, "effort": 145.0, "armature": 0.06086, "friction": 0.01,
    },
    "arms": {
        "joint_names": tuple(
            name for name in N3_MUJOCO_JOINT_NAMES
            if "_arm_shoulder_" in name or "_arm_elbow_pitch_" in name
        ),
        "kp": 50.0, "kd": 3.0, "effort": 50.0, "armature": 0.0029075, "friction": 0.01,
    },
    "hands": {
        "joint_names": N3_WRIST_JOINT_NAMES,
        "kp": 20.0, "kd": 1.0, "effort": 20.0, "armature": 0.0019182, "friction": 0.01,
    },
    "head": {
        "joint_names": ("head_pitch_joint", "head_yaw_joint"),
        "kp": 20.0, "kd": 3.0, "effort": 20.0, "armature": 0.0019182, "friction": 0.01,
    },
}

N3_JOINT_KP = {name: group["kp"] for group in N3_ACTUATOR_GROUPS.values() for name in group["joint_names"]}
N3_JOINT_KD = {name: group["kd"] for group in N3_ACTUATOR_GROUPS.values() for name in group["joint_names"]}
N3_JOINT_EFFORT = {name: group["effort"] for group in N3_ACTUATOR_GROUPS.values() for name in group["joint_names"]}
N3_JOINT_ARMATURE = {name: group["armature"] for group in N3_ACTUATOR_GROUPS.values() for name in group["joint_names"]}
N3_JOINT_FRICTION = {name: group["friction"] for group in N3_ACTUATOR_GROUPS.values() for name in group["joint_names"]}
N3_JOINT_ACTION_SCALE = {name: 0.25 * N3_JOINT_EFFORT[name] / N3_JOINT_KP[name] for name in N3_MUJOCO_JOINT_NAMES}

# Name aliases from 2026-11-03_n3_current's
# deployment/noetix_robots/n3/config/robot.yaml (SHA256
# 085f33a58f5d3bf9654c883195016a4d9eefee62a8fa4e34de68a023c2a805af).
# This maps coordinates by name, not by the deployment file's joint_order.
# In particular its hand_pitch/hand_roll list order differs from the MJCF tree.
N3_TRAINING_TO_DEPLOYMENT_JOINT_NAMES = {
    "waist_yaw_joint": "waist_yaw_joint",
    "head_pitch_joint": "head_pitch_joint",
    "head_yaw_joint": "head_yaw_joint",
    **{
        f"{side}_{training}_joint": f"{side}_{deployment}_joint"
        for side in ("l", "r")
        for training, deployment in (
            ("arm_shoulder_pitch", "arm_pitch"),
            ("arm_shoulder_roll", "arm_roll"),
            ("arm_shoulder_yaw", "arm_yaw"),
            ("arm_elbow_pitch", "elbow_pitch"),
            ("arm_elbow_yaw", "hand_yaw"),
            ("arm_hand_roll", "hand_roll"),
            ("arm_hand_pitch", "hand_pitch"),
            ("leg_hip_pitch", "hip_pitch"),
            ("leg_hip_roll", "hip_roll"),
            ("leg_hip_yaw", "hip_yaw"),
            ("leg_knee_pitch", "knee_pitch"),
            ("leg_ankle_pitch", "ankle_pitch"),
            ("leg_ankle_roll", "ankle_roll"),
        )
    },
}


def get_n3_policy_parameters(joint_names, control_dt=0.02):
    """Return portable training parameters in the actual policy joint order.

    ``joint_names`` must contain all 29 training joint names exactly once, in
    the order used by the policy's observations and actions. The deployment
    names are aliases in that same order, not a reorder into hardware order.
    Targets follow ``q_target = q_default + action_scale * a``. Gains and
    armatures describe training; these metadata do not apply motor calibration
    or implement a deployment controller. All returned values are plain YAML
    scalars, lists and dictionaries.
    """
    if isinstance(joint_names, (str, bytes, dict, set, frozenset)):
        raise ValueError("N3 joint_names must be an ordered iterable of 29 training joint names")
    try:
        names = list(joint_names)
    except TypeError as error:
        raise ValueError("N3 joint_names must be an ordered iterable of 29 training joint names") from error
    if any(not isinstance(name, str) for name in names):
        raise ValueError("N3 joint_names must contain only training joint-name strings")
    names = [str(name) for name in names]
    expected, actual = set(N3_MUJOCO_JOINT_NAMES), set(names)
    duplicates = sorted(name for name in actual if names.count(name) > 1)
    if len(names) != 29 or actual != expected or duplicates:
        raise ValueError(
            "N3 joint_names must contain each of the 29 training joints exactly once; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}, "
            f"duplicates={duplicates}"
        )
    try:
        dt = float(control_dt)
    except (TypeError, ValueError) as error:
        raise ValueError("N3 control_dt must be a finite positive number") from error
    if isinstance(control_dt, bool) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("N3 control_dt must be a finite positive number")
    return {
        "joint_names": names,
        "deployment_joint_names": [N3_TRAINING_TO_DEPLOYMENT_JOINT_NAMES[name] for name in names],
        "kp": [float(N3_JOINT_KP[name]) for name in names],
        "kd": [float(N3_JOINT_KD[name]) for name in names],
        "effort": [float(N3_JOINT_EFFORT[name]) for name in names],
        "armature": [float(N3_JOINT_ARMATURE[name]) for name in names],
        "action_scale": [float(N3_JOINT_ACTION_SCALE[name]) for name in names],
        "default_joint_pos": [float(INIT_JOINT_POS.get(name, 0.0)) for name in names],
        "control_dt": dt,
        "num_actions": 29,
        "root_name": "base_link",
        "action_semantics": "q_target = q_default + action_scale * a",
    }


__all__ = [name for name in globals() if name.startswith("N3_")] + [
    "INIT_POS", "INIT_JOINT_POS", "get_n3_policy_parameters",
]
