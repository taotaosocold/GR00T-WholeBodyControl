"""SuperMimic reset distributions, independent of Isaac Lab and its joint order."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def _uniform_ranges(ranges, count, like):
    bounds = like.new_tensor(ranges)
    samples = torch.rand(count, len(ranges), device=like.device, dtype=like.dtype)
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * samples


def _left_rotvec(rotation, vector):
    """Apply Exp(vector) on the left of a wxyz reference quaternion."""
    half_angle = 0.5 * torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    delta = torch.cat((half_angle.cos(), 0.5 * torch.sinc(half_angle / torch.pi) * vector), -1)
    left_w, left_v = delta[..., :1], delta[..., 1:]
    right_w, right_v = rotation[..., :1], rotation[..., 1:]
    return torch.cat((
        left_w * right_w - (left_v * right_v).sum(-1, keepdim=True),
        left_w * right_v + right_w * left_v + torch.cross(left_v, right_v, dim=-1),
    ), -1)


def apply_supermimic_reset(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    env_ids: torch.Tensor | Sequence[int],
    body_joint_indices: torch.Tensor | Sequence[int],
    hard_joint_limits: torch.Tensor,
    *,
    pose_range: Mapping[str, Sequence[float]],
    velocity_range: Mapping[str, Sequence[float]],
    joint_position_range: Sequence[float],
    add_noise: bool = True,
) -> None:
    """Update selected reference states in place using SuperMimic's rules.

    Pose keys retain SONIC's roll/pitch/yaw names, but their three sampled
    values form one rotation vector, rather than Euler angles. Only the 29
    named body joints receive position noise; extra hand joints keep their
    nominal reference values. Joint velocities are deliberately untouched.
    Hard joint limits are applied even for noise-free evaluation.
    """
    ids = torch.as_tensor(env_ids, device=joint_pos.device, dtype=torch.long)
    joints = torch.as_tensor(body_joint_indices, device=joint_pos.device, dtype=torch.long)
    if ids.ndim != 1 or joints.ndim != 1 or joints.numel() != 29:
        raise ValueError("SuperMimic reset requires environment rows and 29 named body joints")
    if ids.numel() == 0:
        return
    if hard_joint_limits.ndim == 2:
        limits = hard_joint_limits.unsqueeze(0).expand(len(ids), -1, -1)
    elif hard_joint_limits.ndim == 3 and hard_joint_limits.shape[0] == 1:
        limits = hard_joint_limits.expand(len(ids), -1, -1)
    else:
        limits = hard_joint_limits[ids]
    if tuple(limits.shape) != (len(ids), joint_pos.shape[-1], 2):
        raise ValueError("SuperMimic reset requires hard joint position limits for every joint")

    if add_noise:
        keys = ("x", "y", "z", "roll", "pitch", "yaw")
        pose = _uniform_ranges([pose_range.get(key, (0.0, 0.0)) for key in keys], len(ids), root_pos)
        root_pos[ids] += pose[:, :3]
        root_quat[ids] = _left_rotvec(root_quat[ids], pose[:, 3:])
        velocity = _uniform_ranges(
            [velocity_range.get(key, (0.0, 0.0)) for key in keys], len(ids), root_lin_vel,
        )
        root_lin_vel[ids] += velocity[:, :3]
        root_ang_vel[ids] += velocity[:, 3:]
        joint_noise = _uniform_ranges([joint_position_range] * len(joints), len(ids), joint_pos)
        joint_pos[ids[:, None], joints[None, :]] += joint_noise

    joint_pos[ids] = torch.clamp(joint_pos[ids], limits[..., 0], limits[..., 1])
