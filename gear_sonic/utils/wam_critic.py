"""Robot tracking and raw future-reference features for the shared WAM critic.

The actor may select G1, teleop or SMPL encoders. Its critic uses a single
robot reference format, independently of those encoders and their FSQ tokens.
All robot-dependent features are rebuilt from the imagined physical state.
"""

from __future__ import annotations

from numbers import Integral

import torch
from torch import nn

from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.trl.modules.wam import (
    PHYSICAL_STATE_DIM,
    physical_state_invariant_features,
    physical_state_parts,
)
from gear_sonic.utils.wam_kinematics import make_robot_kinematics


FUTURE_REFERENCE_DIM = 73


def reference_joint_order(command, device=None):
    """Map the motion library's robot joint order to its physical WAM order."""
    robot = getattr(command, "robot", "g1")
    source_names = joint_utils.get_robot_joint_names(robot, order="isaaclab")
    target_names = joint_utils.get_robot_joint_names(robot, order="mujoco")
    return torch.tensor([source_names.index(name) for name in target_names], device=device)


def _heading(rotation):
    x, y = rotation[..., 0, 0], rotation[..., 1, 0]
    norm = torch.sqrt(x.square() + y.square() + 1e-9)
    cosine, sine = x / norm, y / norm
    zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
    return torch.stack((cosine, -sine, zero, sine, cosine, zero, zero, zero, one), -1).reshape(
        rotation.shape[:-2] + (3, 3)
    )


class WAMCriticFeatures(nn.Module):
    """SuperMimic tracking features plus unencoded robot future references.

    States use the canonical ``p, quat(wxyz), v, w, q(29), qd(29)`` layout.
    ``future_reference`` adds one frame axis before those 71 coordinates.
    Context is supplied by the caller, which controls its normalization.
    The reference-only future features remain independent of the actor's
    modality choice and learned goal representation.
    """

    def __init__(
        self, kinematics=None, future_frames=10, context_dim=33,
        robot_type="g1", mjcf_path=None, tracked_bodies=None,
    ):
        super().__init__()
        if not isinstance(future_frames, Integral) or future_frames < 1:
            raise ValueError("WAM critic future_frames must be a positive integer")
        if not isinstance(context_dim, Integral) or context_dim < 0:
            raise ValueError("WAM critic context_dim must be a nonnegative integer")
        self.kinematics = (
            make_robot_kinematics(robot_type, mjcf_path, tracked_bodies)
            if kinematics is None else kinematics
        )
        self.future_frames = int(future_frames)
        self.context_dim = int(context_dim)
        body_dim = 3 * len(self.kinematics.tracked_bodies)
        self.tracking_dim = PHYSICAL_STATE_DIM + 29 + body_dim + 3 + 6 + self.context_dim
        self.input_dim = self.tracking_dim + self.future_frames * FUTURE_REFERENCE_DIM

    def forward(self, state, reference, context, future_reference):
        if state.shape[-1] != PHYSICAL_STATE_DIM or reference.shape != state.shape:
            raise ValueError("WAM critic state and reference must have matching [...,71] shapes")
        expected = (*state.shape[:-1], self.future_frames, PHYSICAL_STATE_DIM)
        if tuple(future_reference.shape) != expected:
            raise ValueError(f"WAM critic future_reference must have shape {expected}")
        if context is None:
            if self.context_dim:
                raise ValueError("WAM critic requires its physical context")
            context = state.new_zeros(*state.shape[:-1], 0)
        if tuple(context.shape) != (*state.shape[:-1], self.context_dim):
            raise ValueError("WAM critic context batch and feature dimensions must match")

        position, rotation, _, _, joints, _ = physical_state_parts(state)
        ref_position, ref_rotation, _, _, ref_joints, _ = physical_state_parts(reference)
        heading_t = _heading(rotation).transpose(-1, -2)
        body_position = self.kinematics(state)[0]
        ref_body_position = self.kinematics(reference)[0]
        delta_heading = _heading(rotation @ ref_rotation.transpose(-1, -2))
        anchor = torch.cat((position[..., :2], ref_position[..., 2:3]), -1)
        ref_body_position = anchor.unsqueeze(-2) + (
            delta_heading.unsqueeze(-3)
            @ (ref_body_position - ref_position.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        body_error = (
            heading_t.unsqueeze(-3) @ (ref_body_position - body_position).unsqueeze(-1)
        ).squeeze(-1).flatten(-2)
        root_error = (heading_t @ (ref_position - position).unsqueeze(-1)).squeeze(-1)
        relative_rotation = (rotation.transpose(-1, -2) @ ref_rotation)[..., :2].flatten(-2)

        future_position, future_rotation, future_velocity, future_angular, future_joints, future_qd = (
            physical_state_parts(future_reference)
        )
        reference_heading_t = _heading(ref_rotation).transpose(-1, -2).unsqueeze(-3)
        future_displacement = (
            reference_heading_t @ (future_position - ref_position.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        future_orientation = (reference_heading_t @ future_rotation)[..., :2].flatten(-2)
        future_rotation_t = future_rotation.transpose(-1, -2)
        future_velocity = (future_rotation_t @ future_velocity.unsqueeze(-1)).squeeze(-1)
        future_angular = (future_rotation_t @ future_angular.unsqueeze(-1)).squeeze(-1)
        future = torch.cat((
            future_displacement, future_orientation, future_velocity, future_angular,
            future_joints, future_qd * 0.1,
        ), -1).flatten(-2)
        return torch.cat((
            physical_state_invariant_features(state), ref_joints - joints,
            body_error, root_error, relative_rotation, context, future,
        ), -1)


def future_reference_states(command, num_frames=10, frame_stride=1, next_step=False):
    """Capture one clip's raw future states before simulator auto-reset.

    Frame zero is the command's current frame, matching Sonic's G1 future
    observation grid. ``next_step`` advances that grid by one motion-library
    frame, so a terminal next-state reference can be captured before reset.
    Offsets are expressed in the library's resampled frames, not source FPS.
    At the end of a clip its last frame is repeated; another clip is never read.
    """
    if not isinstance(num_frames, Integral) or num_frames < 1:
        raise ValueError("num_frames must be a positive integer")
    if not isinstance(frame_stride, Integral) or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")
    library = command.motion_lib
    motion_ids = command.motion_ids
    cursor = command.motion_start_time_steps + command.time_steps + int(next_step)
    last = library.get_time_step_total(motion_ids) - 1
    frames = cursor[:, None] + torch.arange(num_frames, device=cursor.device) * frame_stride
    frames = torch.minimum(frames, last[:, None]).clamp_min(0)
    ids = motion_ids.repeat_interleave(num_frames)
    frames = frames.flatten()
    anchor = command.motion_anchor_body_index
    joints = library.get_dof_pos(ids, frames)
    velocities = library.get_dof_vel(ids, frames)
    if joints.shape[-1] != 29 or velocities.shape[-1] != 29:
        raise ValueError("Physical WAM critic references require 29 body joints")
    order = reference_joint_order(command, device=joints.device)
    state = torch.cat((
        library.get_body_pos_w(ids, frames)[:, anchor],
        library.get_body_quat_w(ids, frames)[:, anchor],
        library.get_body_lin_vel_w(ids, frames)[:, anchor],
        library.get_body_ang_vel_w(ids, frames)[:, anchor],
        joints[:, order], velocities[:, order],
    ), -1)
    return state.reshape(motion_ids.shape[0], num_frames, PHYSICAL_STATE_DIM)
