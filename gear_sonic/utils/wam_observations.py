"""Rebuild Sonic's robot-relative goal orientations during WAM imagination.

The reference joint poses and reference-root-local VR targets are exogenous.
The G1, teleop and SMPL root orientations, however, use the *current robot*
as their frame. Replay stores these orientation slices in world coordinates;
each imagined actor forward converts them to its own predicted robot frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from gear_sonic.trl.modules.wam import physical_state_parts


# These terms all encode R_robot.T @ R_reference. Heading-only orientation
# terms and reference-root-local VR observations deliberately stay unchanged.
ROBOT_RELATIVE_ORIENTATION_TERMS = frozenset({
    "motion_anchor_ori_b",
    "motion_anchor_ori_b_nonflat",
    "motion_anchor_ori_b_mf",
    "motion_anchor_ori_b_mf_nonflat",
    "smpl_root_ori_b_multi_future",
    "smpl_root_ori_b_multi_future_flatten",
})


class WAMGoalObservations:
    """Rotate only the robot-relative 6D orientations in packed actor goals.

    ``tokenizer_specs`` is the actor module's sequence of
    ``(name, start, end, original_shape)`` entries. ``tokenizer_offset`` locates
    that group if the packed goal also contains other observation groups.
    All remaining values, including modality routing, are copied unchanged.

    Goals may have shape ``[B, G]`` or ``[B, H, G]``. Physical robot states
    have shape ``[B, 71]`` or ``[B, H, 71]`` in the canonical WAM layout;
    a ``[B, 71]`` state can also rotate a complete ``[B, H, G]`` timeline.
    Both methods return a new tensor and preserve autograd.
    """

    def __init__(self, tokenizer_specs: Sequence, *, tokenizer_offset: int = 0):
        if not isinstance(tokenizer_offset, int) or tokenizer_offset < 0:
            raise ValueError("tokenizer_offset must be a non-negative integer")
        slices = []
        for spec in tokenizer_specs:
            name, start, end = spec[:3]
            if name not in ROBOT_RELATIVE_ORIENTATION_TERMS:
                continue
            if start < 0 or end <= start or (end - start) % 6:
                raise ValueError(f"{name} must contain one or more 6D orientations")
            slices.append((name, tokenizer_offset + start, tokenizer_offset + end))
        self.orientation_slices = tuple(slices)

    def to_world(self, goals: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        """Prepare replay goals using the robot state from their collection."""
        return self._rotate(goals, robot_state, inverse=False)

    def to_robot(self, goals: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        """Build actor goals from replay using the current imagined robot."""
        return self._rotate(goals, robot_state, inverse=True)

    def _rotate(self, goals, robot_state, *, inverse):
        if goals.ndim < 2:
            raise ValueError("packed goals must have at least one batch dimension")
        result = goals.clone()
        if not self.orientation_slices:
            return result
        _, rotation, *_ = physical_state_parts(robot_state)
        state_dims, goal_dims = robot_state.shape[:-1], goals.shape[:-1]
        if len(state_dims) > len(goal_dims):
            raise ValueError("robot state has more batch dimensions than goals")
        # Missing dimensions are future/timeline axes, following shared batch
        # axes. In particular, [B,71] broadcasts over [B,H,G], not over B.
        rotation = rotation.reshape(
            *state_dims, *((1,) * (len(goal_dims) - len(state_dims))), 3, 3
        )
        try:
            rotation = torch.broadcast_to(rotation, (*goal_dims, 3, 3))
        except RuntimeError as exc:
            raise ValueError("robot state and goal batch dimensions do not match") from exc
        if inverse:
            rotation = rotation.transpose(-1, -2)
        for name, start, end in self.orientation_slices:
            if end > goals.shape[-1]:
                raise ValueError(f"{name} extends beyond the packed goal dimension")
            # Isaac Lab observations flatten R[..., :2], a [3,2] matrix,
            # rather than concatenating two length-three columns. Multiplying
            # the two columns directly preserves the existing sensor noise;
            # no Gram-Schmidt/6D-to-rotation conversion is appropriate here.
            vectors = goals[..., start:end].reshape(*goal_dims, -1, 3, 2)
            result[..., start:end] = (rotation.unsqueeze(-3) @ vectors).flatten(-3)
        return result
