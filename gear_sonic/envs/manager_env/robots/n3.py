"""N3 V3 Isaac Lab articulation with the MJLab training actuator parameters."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

from gear_sonic.envs.env_utils.n3_utils import (
    INIT_JOINT_POS,
    INIT_POS,
    N3_ACTUATOR_GROUPS,
    N3_ISAACLAB_BODY_NAMES,
    N3_ISAACLAB_TO_MUJOCO_BODY,
    N3_ISAACLAB_TO_MUJOCO_DOF,
    N3_JOINT_ACTION_SCALE,
    N3_MUJOCO_JOINT_NAMES,
    N3_MUJOCO_TO_ISAACLAB_BODY,
    N3_MUJOCO_TO_ISAACLAB_DOF,
    N3_URDF_PATH,
)


N3_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": N3_ISAACLAB_BODY_NAMES,
    "isaaclab_to_mujoco_dof": N3_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": N3_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": N3_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": N3_MUJOCO_TO_ISAACLAB_BODY,
}
N3_ACTION_SCALE = dict(N3_JOINT_ACTION_SCALE)

N3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=N3_URDF_PATH,
        fix_base=False,
        # Keep the fixed toe names used by FK, motion data and body mappings.
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=INIT_POS,
        joint_pos={name: INIT_JOINT_POS.get(name, 0.0) for name in N3_MUJOCO_JOINT_NAMES},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        name: ImplicitActuatorCfg(
            joint_names_expr=list(group["joint_names"]),
            effort_limit_sim=group["effort"],
            # make_mjlab_cfg does not add a velocity clamp. The vendor URDF
            # has zero velocity limits, so explicitly avoid inheriting them.
            velocity_limit_sim=float("inf"),
            stiffness=group["kp"],
            damping=group["kd"],
            armature=group["armature"],
            friction=group["friction"],
        )
        for name, group in N3_ACTUATOR_GROUPS.items()
    },
)

__all__ = ["N3_CFG", "N3_ACTION_SCALE", "N3_ISAACLAB_TO_MUJOCO_MAPPING"]
