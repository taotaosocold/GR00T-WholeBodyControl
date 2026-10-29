"""Shared differentiable physical rewards for real and imagined WAM rollouts."""

from __future__ import annotations

import torch
from torch import nn

from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER
from gear_sonic.trl.modules.wam import physical_state_parts
from gear_sonic.utils.wam_kinematics import G1Kinematics, G1_MUJOCO_JOINT_NAMES
from gear_sonic.utils.wam_state import robot_state as _robot_state
from gear_sonic.utils.wam_termination import next_motion_frame, reference_pose


DEFAULT_STDS = {
    "body_pos": 0.3, "body_ori": 0.4, "body_linvel": 1.0,
    "body_angvel": 3.14, "root_pos": 0.3, "root_ori": 0.4,
}
DEFAULT_WEIGHTS = {
    "tracking_relative_body_pos": 1.0,
    "tracking_relative_body_ori": 1.0,
    "tracking_body_linvel": 1.0,
    "tracking_body_angvel": 1.0,
    "tracking_anchor_pos": 0.5,
    "tracking_anchor_ori": 0.5,
    "action_rate_l2": -0.1,
    "joint_limit": -2.0,
}
_REFERENCE_ORDER = tuple(G1_ISAACLab_ORDER.index(name) for name in G1_MUJOCO_JOINT_NAMES)


def _heading(rotation):
    x, y = rotation[..., 0, 0], rotation[..., 1, 0]
    norm = torch.sqrt(x.square() + y.square() + 1e-9)
    cosine, sine = x / norm, y / norm
    zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
    return torch.stack((cosine, -sine, zero, sine, cosine, zero, zero, zero, one), -1).reshape(
        rotation.shape[:-2] + (3, 3)
    )


def _rotation_error(rotation, reference):
    cosine = ((rotation * reference).sum((-1, -2)) - 1) / 2
    return torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6)).square()


class WAMTrackingReward(nn.Module):
    """SuperMimic's eight reward terms, computed from the same 71-D state.

    ``components`` returns unweighted rates for Isaac Lab's RewardManager,
    which multiplies by weight and control dt. ``forward`` applies those same
    weights and dt once for BPTT through the imagined physical state.
    """

    def __init__(
        self, mjcf_path=None, tracked_bodies=None, stds=None, weights=None,
        soft_limit_factor=0.9,
    ):
        super().__init__()
        self.kinematics = G1Kinematics(mjcf_path=mjcf_path, tracked_bodies=tracked_bodies)
        self.stds = DEFAULT_STDS | dict(stds or {})
        self.weights = DEFAULT_WEIGHTS | dict(weights or {})
        if set(self.stds) != set(DEFAULT_STDS) or any(float(v) <= 0 for v in self.stds.values()):
            raise ValueError("WAM reward standard deviations must be known positive values")
        if set(self.weights) != set(DEFAULT_WEIGHTS):
            raise ValueError("Unknown WAM reward component")
        if not 0 < float(soft_limit_factor) <= 1:
            raise ValueError("soft_limit_factor must be in (0, 1]")
        midpoint = (self.kinematics.q_lo + self.kinematics.q_hi) / 2
        half_range = (self.kinematics.q_hi - self.kinematics.q_lo) * float(soft_limit_factor) / 2
        self.register_buffer("soft_lo", midpoint - half_range)
        self.register_buffer("soft_hi", midpoint + half_range)

    def components(self, state, reference, action, previous_action):
        if action.shape[-1] != 29 or previous_action.shape != action.shape:
            raise ValueError("Physical WAM rewards require matching 29-D body actions")
        position, rotation, _, _, joints, _ = physical_state_parts(state)
        ref_position, ref_rotation, *_ = physical_state_parts(reference)
        body_pos, body_rot, body_vel, body_angvel = self.kinematics(state)
        ref_body_pos, ref_body_rot, ref_body_vel, ref_body_angvel = self.kinematics(reference)
        # Re-anchor reference bodies to the robot's current XY and yaw, while
        # preserving the reference height. Velocity targets remain world-frame.
        delta_heading = _heading(rotation @ ref_rotation.transpose(-1, -2))
        anchor = torch.cat((position[..., :2], ref_position[..., 2:3]), -1)
        ref_body_pos = anchor.unsqueeze(-2) + (
            delta_heading.unsqueeze(-3) @ (ref_body_pos - ref_position.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        ref_body_rot = delta_heading.unsqueeze(-3) @ ref_body_rot
        std = self.stds
        return {
            "tracking_relative_body_pos": torch.exp(
                -(body_pos - ref_body_pos).square().sum(-1).mean(-1) / std["body_pos"]**2
            ),
            "tracking_relative_body_ori": torch.exp(
                -_rotation_error(body_rot, ref_body_rot).mean(-1) / std["body_ori"]**2
            ),
            "tracking_body_linvel": torch.exp(
                -(body_vel - ref_body_vel).square().sum(-1).mean(-1) / std["body_linvel"]**2
            ),
            "tracking_body_angvel": torch.exp(
                -(body_angvel - ref_body_angvel).square().sum(-1).mean(-1) / std["body_angvel"]**2
            ),
            "tracking_anchor_pos": torch.exp(
                -(position - ref_position).square().sum(-1) / std["root_pos"]**2
            ),
            "tracking_anchor_ori": torch.exp(
                -_rotation_error(rotation, ref_rotation) / std["root_ori"]**2
            ),
            "action_rate_l2": (action - previous_action).square().sum(-1),
            "joint_limit": (torch.relu(joints - self.soft_hi) + torch.relu(self.soft_lo - joints)).sum(-1),
        }

    def forward(self, state, reference, action, previous_action, dt=0.02):
        if float(dt) <= 0:
            raise ValueError("WAM reward control dt must be positive")
        terms = self.components(state, reference, action, previous_action)
        return sum(float(self.weights[name]) * value for name, value in terms.items()) * float(dt)


def reference_state(command):
    """Next reference before reset, in the physical WAM's canonical joint order."""
    frame = next_motion_frame(command)
    position, quaternion = reference_pose(command)
    library = command.motion_lib
    joints = library.get_dof_pos(command.motion_ids, frame)
    velocities = library.get_dof_vel(command.motion_ids, frame)
    if joints.shape[-1] != 29 or velocities.shape[-1] != 29:
        raise ValueError("Physical WAM rewards require 29 reference body joints")
    order = torch.as_tensor(_REFERENCE_ORDER, device=joints.device)
    anchor = command.motion_anchor_body_index
    return torch.cat((
        position, quaternion,
        library.get_body_lin_vel_w(command.motion_ids, frame)[:, anchor],
        library.get_body_ang_vel_w(command.motion_ids, frame)[:, anchor],
        joints[:, order], velocities[:, order],
    ), -1)


def robot_state(env, command):
    """Compatibility entry point for the shared canonical physical state."""
    return _robot_state(env, command)


def _body_action(robot, action):
    # With exactly 29 body actions, the sum-of-squares penalty is invariant to
    # their common permutation. For dexterous robots exclude the hand actions.
    if action.shape[-1] == 29:
        return action
    order = torch.as_tensor(
        [robot.joint_names.index(name) for name in G1_MUJOCO_JOINT_NAMES], device=action.device,
    )
    return action[:, order]


@torch.no_grad()
def tracking_reward_term(env, term_name, reward_config=None, command_name="motion"):
    """Unweighted reward rate; RewardManager supplies weight and dt once.

    All eight terms share one FK evaluation per real physics step. The cache
    is populated before auto-reset, using the executed and previous actions.
    """
    cfg = {key: value for key, value in dict(reward_config or {}).items() if not key.startswith("_")}
    command = env.command_manager.get_term(command_name)
    model = getattr(env, "_wam_reward_model", None)
    if model is None or getattr(env, "_wam_reward_config", None) != cfg:
        model = WAMTrackingReward(**cfg).to(
            device=command.robot.data.joint_pos.device, dtype=command.robot.data.joint_pos.dtype,
        )
        env._wam_reward_model = model
        env._wam_reward_config = cfg
        env._wam_reward_cache_step = None
    step = getattr(env, "common_step_counter", None)
    cache_key = (command_name, step)
    if step is None or cache_key != getattr(env, "_wam_reward_cache_step", None):
        env._wam_reward_components = model.components(
            robot_state(env, command), reference_state(command),
            _body_action(command.robot, env.action_manager.action),
            _body_action(command.robot, env.action_manager.prev_action),
        )
        env._wam_reward_cache_step = cache_key
    return env._wam_reward_components[term_name]
