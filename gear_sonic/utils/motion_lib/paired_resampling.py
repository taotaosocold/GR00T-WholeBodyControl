"""Sample paired SMPL observations on the robot FK time grid.

Robot window indices describe robot source frames, not SMPL source frames.
Keeping the sampling grid in seconds avoids changing motion speed when their
frame rates differ. This module does not trim, pad, or modify source motions.
"""

import math
import operator

import torch

from gear_sonic.isaac_utils import rotations


def _positive_fps(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return value


def _integer(value, name, minimum):
    try:
        if isinstance(value, bool):
            raise TypeError
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _interpolate_pose(lower, upper, blend):
    """Interpolate SMPL axis angles over the shortest rotation arc (wxyz)."""
    q0 = rotations.axis_angle_to_quaternion(lower.reshape(-1, 24, 3))
    q1 = rotations.axis_angle_to_quaternion(upper.reshape(-1, 24, 3))
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    weights = blend[:, None, None]
    interpolated = rotations.slerp(q0, q1, weights)
    # rotations.slerp uses a fixed midpoint near identical quaternions. Use
    # normalized lerp in that region so an arbitrary sample keeps its time.
    near_identical = dot.abs() > 1 - 1e-6
    linear = torch.nn.functional.normalize(torch.lerp(q0, q1, weights), dim=-1)
    interpolated = torch.where(near_identical, linear, interpolated)
    interpolated = torch.nn.functional.normalize(interpolated, dim=-1)
    return rotations.quaternion_to_angle_axis(interpolated).reshape(-1, 72)


def resample_smpl_to_robot_window(
    smpl_data, robot_start_frame, robot_fps, target_fps, num_output_frames
):
    """Return present SMPL temporal fields at the actual robot output times.

    Output frame ``i`` samples source time
    ``robot_start_frame / robot_fps + i / target_fps``. All present SMPL fields
    must have one consistent source length and positive finite ``fps``. Source
    coverage must contain the complete requested grid; insufficient coverage is
    an error, never a reason to repeat frames or stretch a clip.

    Positions and translations use linear interpolation. ``pose_aa`` accepts
    ``(T, 72)`` or ``(T, 24, 3)`` and returns ``(N, 72)`` using quaternion SLERP.
    Samples that coincide with source frames retain the original values.
    """
    robot_start_frame = _integer(robot_start_frame, "robot_start_frame", 0)
    num_output_frames = _integer(num_output_frames, "num_output_frames", 1)
    robot_fps = _positive_fps(robot_fps, "robot_fps")
    target_fps = _positive_fps(target_fps, "target_fps")
    smpl_fps = _positive_fps(smpl_data.get("fps"), "smpl fps")

    fields = {}
    for key in ("pose_aa", "smpl_joints", "transl"):
        if key not in smpl_data:
            continue
        try:
            values = torch.as_tensor(smpl_data[key])
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"SMPL {key} must be a temporal array") from exc
        if values.ndim < 2 or values.shape[0] == 0:
            raise ValueError(f"SMPL {key} must be a nonempty temporal array")
        if key == "pose_aa":
            if values.shape[1:] not in ((72,), (24, 3)):
                raise ValueError(f"SMPL pose_aa must have shape (T, 72) or (T, 24, 3), got {values.shape}")
            values = values.reshape(-1, 72)
        elif key == "smpl_joints" and values.shape[1:] != (24, 3):
            raise ValueError(f"SMPL smpl_joints must have shape (T, 24, 3), got {values.shape}")
        elif key == "transl" and values.shape[1:] != (3,):
            raise ValueError(f"SMPL transl must have shape (T, 3), got {values.shape}")
        if not values.is_floating_point():
            values = values.float()
        fields[key] = values
    if not fields:
        raise ValueError("SMPL has no pose_aa, smpl_joints, or transl temporal field")
    lengths = {key: len(values) for key, values in fields.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"SMPL temporal fields have inconsistent lengths: {lengths}")
    source_length = next(iter(lengths.values()))

    times = robot_start_frame / robot_fps + torch.arange(num_output_frames, dtype=torch.float64) / target_fps
    positions = times * smpl_fps
    rounded = positions.round()
    on_source_frame = torch.isclose(positions, rounded, atol=1e-7, rtol=0)
    positions = torch.where(on_source_frame, rounded, positions)
    if positions[-1] > source_length - 1:
        raise ValueError(
            "SMPL does not cover the requested robot time grid: "
            f"requested=[{times[0].item():.9g}, {times[-1].item():.9g}]s, "
            f"SMPL=[0, {(source_length - 1) / smpl_fps:.9g}]s, "
            f"smpl_frames={source_length}, smpl_fps={smpl_fps}, robot_fps={robot_fps}"
        )
    lower_indices = positions.floor().long()
    upper_indices = (lower_indices + 1).clamp(max=source_length - 1)
    blend = positions - lower_indices

    result = {}
    for key, values in fields.items():
        lower = values[lower_indices.to(values.device)]
        if bool(on_source_frame.all()):
            result[key] = lower.clone()
            continue
        upper = values[upper_indices.to(values.device)]
        weight = blend.to(device=values.device, dtype=values.dtype)
        if key == "pose_aa":
            output = _interpolate_pose(lower, upper, weight)
        else:
            output = torch.lerp(lower, upper, weight.reshape(-1, *([1] * (values.ndim - 1))))
        direct = on_source_frame.to(values.device).reshape(-1, *([1] * (values.ndim - 1)))
        result[key] = torch.where(direct, lower, output)
    return result
