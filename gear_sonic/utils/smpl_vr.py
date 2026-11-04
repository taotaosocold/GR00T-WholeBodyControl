"""SMPL-only VR targets, independent of robot FK and body-model files.

Joint positions follow the existing SMPL observation convention: pelvis-relative
(or translated), Z-up, world-oriented positions. ``smpl_y_up`` describes *pose
rotations* only; it must never cause a second up-axis conversion of positions.
All quaternions use wxyz. Point order is left wrist, right wrist, head, left
ankle, right ankle. Runtime functions only check layouts, avoiding GPU sync;
validate source data once per clip with ``validate_smpl_vr5_data``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


SMPL_VR5_JOINT_NAMES = ("left_wrist", "right_wrist", "head", "left_ankle", "right_ankle")
SMPL_VR5_JOINT_INDICES = (20, 21, 15, 7, 8)
# SMPL's 24-joint tree, including the two terminal hand joints.
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)


def _check_tensor(value, name, trailing_shape):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim < len(trailing_shape) + 1 or tuple(value.shape[-len(trailing_shape):]) != trailing_shape:
        raise ValueError(f"{name} must have shape (B, {', '.join(map(str, trailing_shape))})")


def _pose_joints(pose_aa):
    if not isinstance(pose_aa, torch.Tensor) or not pose_aa.is_floating_point():
        raise ValueError("SMPL pose_aa must be a floating-point tensor")
    if pose_aa.ndim >= 2 and pose_aa.shape[-1] == 72:
        return pose_aa.reshape(*pose_aa.shape[:-1], 24, 3)
    if pose_aa.ndim >= 3 and tuple(pose_aa.shape[-2:]) == (24, 3):
        return pose_aa
    raise ValueError("SMPL pose_aa must have shape (B, 72) or (B, 24, 3)")


def _check_root(root_quat_w, source, batch_shape):
    _check_tensor(root_quat_w, "SMPL root_quat_w", (4,))
    if tuple(root_quat_w.shape[:-1]) != tuple(batch_shape):
        raise ValueError("SMPL root_quat_w and source must have the same batch shape")
    if root_quat_w.dtype != source.dtype or root_quat_w.device != source.device:
        raise ValueError("SMPL root_quat_w and source must have the same dtype and device")
    return torch.nn.functional.normalize(root_quat_w, dim=-1)


def _quat_mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(dim=-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv)), dim=-1)


def _quat_inv(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _quat_apply(q, v):
    vector = q[..., 1:]
    twice_cross = 2 * torch.linalg.cross(vector, v)
    return v + q[..., :1] * twice_cross + torch.linalg.cross(vector, twice_cross)


def _axis_angle_quat(axis_angle):
    half_angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True) * 0.5
    scale = 0.5 * torch.sinc(half_angle / math.pi)
    return torch.cat((torch.cos(half_angle), axis_angle * scale), dim=-1)


def validate_smpl_vr5_data(smpl_data):
    """Validate shapes, finite values and noncollapsed points once per clip.

    Input arrays are never mutated. Temporal pose and joint arrays must agree.
    Individual stationary/zero frames are allowed; a whole clip with all five
    points collapsed onto its pelvis is rejected. Coordinate conventions cannot
    be inferred numerically: positions must follow the documented Z-up contract.
    """
    if not isinstance(smpl_data, Mapping):
        raise ValueError("SMPL VR5 data must be a mapping")
    missing = [name for name in ("fps", "pose_aa", "smpl_joints") if name not in smpl_data]
    if missing:
        raise ValueError(f"SMPL VR5 data is missing required fields: {missing}")
    try:
        fps = float(smpl_data["fps"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("SMPL VR5 fps must be positive and finite") from exc
    if isinstance(smpl_data["fps"], bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("SMPL VR5 fps must be positive and finite")
    tensors = {}
    for name in ("pose_aa", "smpl_joints"):
        try:
            value = torch.as_tensor(smpl_data[name]).detach().cpu()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"SMPL VR5 {name} must be a numeric temporal array") from exc
        if value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"SMPL VR5 {name} must contain real numbers")
        if not torch.isfinite(value).all():
            raise ValueError(f"SMPL VR5 {name} must contain only finite values")
        tensors[name] = value
    pose, joints = tensors["pose_aa"], tensors["smpl_joints"]
    if pose.ndim not in (2, 3) or tuple(pose.shape[1:]) not in ((72,), (24, 3)):
        raise ValueError("SMPL VR5 pose_aa must have shape (T, 72) or (T, 24, 3)")
    if joints.ndim != 3 or tuple(joints.shape[1:]) != (24, 3):
        raise ValueError("SMPL VR5 smpl_joints must have shape (T, 24, 3)")
    if len(pose) == 0 or len(pose) != len(joints):
        raise ValueError("SMPL VR5 pose_aa and smpl_joints must have equal nonempty temporal lengths")
    relative_points = joints[:, SMPL_VR5_JOINT_INDICES, :] - joints[:, :1, :]
    if not torch.any(relative_points != 0):
        raise ValueError("SMPL VR5 all five joint positions collapse onto the pelvis throughout the clip")


def smpl_vr5_local_positions(smpl_joints, root_quat_w):
    """Return flattened 15-D points in the canonical SMPL root frame.

    ``smpl_joints[...,24,3]`` is Z-up/world-oriented regardless of the pose's
    ``smpl_y_up`` flag. ``root_quat_w[...,4]`` is the command's SMPL quaternion
    after up-axis conversion and SMPL base-rotation removal. Subtracting pelvis
    translation supports both precentered and translated input positions.
    """
    _check_tensor(smpl_joints, "SMPL smpl_joints", (24, 3))
    root = _check_root(root_quat_w, smpl_joints, smpl_joints.shape[:-2])
    points = smpl_joints[..., SMPL_VR5_JOINT_INDICES, :] - smpl_joints[..., :1, :]
    root_inv = _quat_inv(root).unsqueeze(-2).expand(*points.shape[:-1], 4)
    return _quat_apply(root_inv, points).flatten(start_dim=-2)


def smpl_vr5_local_orientations(pose_aa, root_quat_w, *, smpl_y_up=False):
    """Return flattened 20-D point FK quaternions in the SMPL root frame.

    Axis angles are parent-relative: accumulate the full chain before converting
    to Z-up, and remove the SMPL base rotation from every global joint frame,
    as already done for the supplied root. Applying the offset to the root alone
    would mix joint-axis conventions. No body model, betas or robot is involved.
    """
    pose = _pose_joints(pose_aa)
    root = _check_root(root_quat_w, pose, pose.shape[:-2])
    local = _axis_angle_quat(pose)
    global_quats = [local[..., 0, :]]
    for joint, parent in enumerate(SMPL_PARENTS[1:], start=1):
        global_quats.append(_quat_mul(global_quats[parent], local[..., joint, :]))
    selected = torch.stack([global_quats[index] for index in SMPL_VR5_JOINT_INDICES], dim=-2)
    if smpl_y_up:
        up = selected.new_tensor((math.sqrt(0.5), math.sqrt(0.5), 0., 0.))
        selected = _quat_mul(up.expand_as(selected), selected)
    # Identical to remove_smpl_base_rot(q, w_last=False).
    inverse_base = selected.new_tensor((0.5, -0.5, -0.5, -0.5)).expand_as(selected)
    selected = _quat_mul(selected, inverse_base)
    root_inv = _quat_inv(root).unsqueeze(-2).expand_as(selected)
    relative = torch.nn.functional.normalize(_quat_mul(root_inv, selected), dim=-1)
    relative = torch.where(relative[..., :1] < 0, -relative, relative)
    return relative.flatten(start_dim=-2)


def smpl_vr5_targets(smpl_joints, pose_aa, root_quat_w, *, smpl_y_up=False):
    """Return ``(positions[...,15], orientations[...,20])`` in one point order."""
    return (smpl_vr5_local_positions(smpl_joints, root_quat_w),
            smpl_vr5_local_orientations(pose_aa, root_quat_w, smpl_y_up=smpl_y_up))
