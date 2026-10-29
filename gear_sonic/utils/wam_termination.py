"""Shared SuperMimic failure criteria for simulator collection and BPTT.

This module has no Isaac Lab dependency so the hard and soft forms can be
checked against each other without starting a simulator. Quaternions use wxyz.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from gear_sonic.utils.wam_state import robot_state


def enable_terminal_capture(env, command_name="motion"):
    """Enable a physical WAM trainer's post-physics, pre-reset state cache.

    The simulator calls the aligned failure/timeout terms before resetting
    environments. Those terms capture all robots once per control step;
    the trainer can then replace auto-reset next states for its done rows.
    Other trainers do not opt in and incur no state gathering or allocation.
    """
    env._wam_capture_terminal_state = True
    env._wam_terminal_capture_command = command_name
    env._wam_terminal_state = None
    env._wam_terminal_state_step = None
    env._wam_terminal_state_command = None


@torch.no_grad()
def capture_terminal_state(env, command_name="motion"):
    """Cache the current physical state during a pre-reset manager callback.

    Calling this after ``env.step`` cannot recover a terminal state: the
    environment has already reset. Callers must be a failure/timeout/reward
    term evaluated between physics and reset. Without an enabled capture
    hook the consumer gets ``None`` and must keep its conservative done mask.
    """
    if not getattr(env, "_wam_capture_terminal_state", False):
        return None
    if command_name != getattr(env, "_wam_terminal_capture_command", None):
        return None
    step = getattr(env, "common_step_counter", None)
    if step is None:
        raise ValueError("WAM terminal-state capture requires common_step_counter")
    if (
        getattr(env, "_wam_terminal_state_step", None) == step
        and getattr(env, "_wam_terminal_state_command", None) == command_name
    ):
        return env._wam_terminal_state
    state = robot_state(env, command_name=command_name).detach().clone()
    env._wam_terminal_state = state
    env._wam_terminal_state_step = step
    env._wam_terminal_state_command = command_name
    return state


def get_terminal_state(env, command_name="motion", expected_step=None):
    """Return this step's pre-reset state, or ``None`` when no hook captured it.

    A stale previous-step cache is deliberately rejected, including when
    aligned termination was disabled or the simulator did not execute the
    capture callback. This never substitutes an auto-reset state as a target.
    """
    if not getattr(env, "_wam_capture_terminal_state", False):
        return None
    if expected_step is None:
        expected_step = getattr(env, "common_step_counter", None)
    if expected_step is None or expected_step != getattr(env, "_wam_terminal_state_step", None):
        return None
    if command_name != getattr(env, "_wam_terminal_state_command", None):
        return None
    return getattr(env, "_wam_terminal_state", None)


def tracking_errors(position, quaternion, reference_position, reference_quaternion):
    """Return root-height and R[2,2] errors, independent of heading and XY."""
    quaternion = F.normalize(quaternion, dim=-1)
    reference_quaternion = F.normalize(reference_quaternion, dim=-1)
    tilt = 1 - 2 * quaternion[..., 1:3].square().sum(-1)
    reference_tilt = 1 - 2 * reference_quaternion[..., 1:3].square().sum(-1)
    return (
        (reference_position[..., 2] - position[..., 2]).abs(),
        (reference_tilt - tilt).abs(),
    )


def hard_tracking_termination(
    position, quaternion, reference_position, reference_quaternion,
    height_threshold=0.25, tilt_threshold=0.8,
):
    height_error, tilt_error = tracking_errors(
        position, quaternion, reference_position, reference_quaternion,
    )
    return (height_error > height_threshold) | (tilt_error > tilt_threshold)


def soft_tracking_termination(
    position, quaternion, reference_position, reference_quaternion,
    height_threshold=0.25, tilt_threshold=0.8,
    height_width=0.02, tilt_width=0.04,
):
    """Differentiable continuation weights use the same hard-failure boundary."""
    if height_width <= 0 or tilt_width <= 0:
        raise ValueError("WAM soft-termination widths must be positive")
    height_error, tilt_error = tracking_errors(
        position, quaternion, reference_position, reference_quaternion,
    )
    alive = (1 - torch.sigmoid((height_error - height_threshold) / height_width)) * (
        1 - torch.sigmoid((tilt_error - tilt_threshold) / tilt_width)
    )
    return 1 - alive


def next_motion_frame(command):
    """Isaac Lab checks failure after physics, before advancing the command."""
    cursor = command.motion_start_time_steps + command.time_steps + 1
    last_frame = command.motion_lib.get_time_step_total(command.motion_ids) - 1
    return torch.minimum(cursor, last_frame).clamp_min(0)


def reference_pose(command):
    """Return the old clip's next anchor pose in local environment coordinates."""
    frame = next_motion_frame(command)
    anchor = command.motion_anchor_body_index
    library = command.motion_lib
    return (
        library.get_body_pos_w(command.motion_ids, frame)[:, anchor],
        library.get_body_quat_w(command.motion_ids, frame)[:, anchor],
    )


def supermimic_failure(
    env, command_name="motion", height_threshold=0.25, tilt_threshold=0.8,
):
    """Hard simulator failure matching the WAM's imagined soft termination."""
    capture_terminal_state(env, command_name)
    command = env.command_manager.get_term(command_name)
    position, quaternion = reference_pose(command)
    return hard_tracking_termination(
        command.robot_anchor_pos_w - env.scene.env_origins,
        command.robot_anchor_quat_w,
        position, quaternion,
        height_threshold=height_threshold, tilt_threshold=tilt_threshold,
    )


def supermimic_time_out(env, command_name="motion"):
    """Clip completion and episode limits are truncations, not physical failures."""
    capture_terminal_state(env, command_name)
    command = env.command_manager.get_term(command_name)
    cursor = command.motion_start_time_steps + command.time_steps + 1
    clip_end = cursor >= command.motion_lib.get_time_step_total(command.motion_ids)
    return clip_end | (env.episode_length_buf >= env.max_episode_length)
