"""Canonical physical WAM state and Isaac Lab link-origin velocity adapters.

WAM and its forward kinematics use the pelvis *link origin*. Some Isaac Lab
versions expose ``body_lin_vel_w`` at the center of mass instead. Keep that
conversion in one place rather than mixing the two velocity conventions.
This module can be imported without starting Isaac Lab.
"""

from __future__ import annotations

import torch

from gear_sonic.utils.wam_kinematics import G1_MUJOCO_JOINT_NAMES


def _first_tensor(data, names):
    for name in names:
        value = getattr(data, name, None)
        if value is not None:
            return value
    return None


def _body_value(value, body_index, name):
    if value.ndim != 3:
        raise ValueError(f"WAM {name} must have shape [num_envs, num_bodies, D]")
    return value[:, body_index]


def link_origin_linear_velocity(data, body_index):
    """Return one body's world-frame linear velocity at its link origin.

    Prefer the explicit link-frame API. On older Isaac Lab versions,
    ``body_lin_vel_w`` is a CoM velocity alias; conversion therefore also
    requires the CoM offset, link rotation, and angular velocity. Missing
    metadata raises an error instead of silently treating a CoM as a link.
    """
    link_velocity = getattr(data, "body_link_lin_vel_w", None)
    if link_velocity is not None:
        return _body_value(link_velocity, body_index, "body_link_lin_vel_w")

    com_velocity = _first_tensor(data, ("body_com_lin_vel_w", "body_lin_vel_w"))
    angular_velocity = _first_tensor(
        data, ("body_com_ang_vel_w", "body_link_ang_vel_w", "body_ang_vel_w"),
    )
    com_offset = _first_tensor(data, ("body_com_pos_b", "com_pos_b"))
    link_quaternion = _first_tensor(data, ("body_link_quat_w", "body_quat_w"))
    if any(value is None for value in (com_velocity, angular_velocity, com_offset, link_quaternion)):
        raise ValueError(
            "WAM needs body_link_lin_vel_w, or CoM linear/angular velocities, "
            "body_com_pos_b and a world-frame link quaternion to recover link-origin velocity"
        )
    velocity = _body_value(com_velocity, body_index, "body_com_lin_vel_w")
    omega = _body_value(angular_velocity, body_index, "body_com_ang_vel_w")
    offset = _body_value(com_offset, body_index, "body_com_pos_b")
    quaternion = _body_value(link_quaternion, body_index, "body_link_quat_w")
    if quaternion.shape[-1] != 4:
        raise ValueError("WAM link quaternions must use four wxyz coordinates")
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    xyz = quaternion[..., 1:]
    # q * r * q^-1, in the same world frame as velocity and omega.
    xyz, offset = torch.broadcast_tensors(xyz, offset)
    twice_cross = 2 * torch.linalg.cross(xyz, offset, dim=-1)
    world_offset = offset + quaternion[..., :1] * twice_cross + torch.linalg.cross(
        xyz, twice_cross, dim=-1,
    )
    omega, world_offset = torch.broadcast_tensors(omega, world_offset)
    return velocity - torch.linalg.cross(omega, world_offset, dim=-1)


def robot_state(env, command=None, command_name="motion"):
    """Return ``p, quat(wxyz), v_link, omega, q(29), qd(29)`` in canonical order.

    Position is relative to the environment origin. Linear and angular
    velocities remain world-frame, matching reference motion and G1 FK.
    The robot may also have hand joints; only the named 29 body joints enter
    the physical state.
    """
    if command is None:
        command = env.command_manager.get_term(command_name)
    robot = command.robot
    order = torch.as_tensor(
        [robot.joint_names.index(name) for name in G1_MUJOCO_JOINT_NAMES],
        device=robot.data.joint_pos.device,
    )
    return torch.cat((
        command.robot_anchor_pos_w - env.scene.env_origins,
        command.robot_anchor_quat_w,
        link_origin_linear_velocity(robot.data, command.robot_anchor_body_index),
        command.robot_anchor_ang_vel_w,
        robot.data.joint_pos[:, order], robot.data.joint_vel[:, order],
    ), -1)
