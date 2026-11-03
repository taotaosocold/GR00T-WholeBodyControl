"""Checkpoint metadata shared by evaluation without importing the simulator."""

from collections.abc import Mapping
from numbers import Integral
from typing import Any


def validate_checkpoint_robot(checkpoint: Mapping[str, Any], expected: Mapping | None) -> None:
    """Reject equal-sized policies belonging to different joint semantics.

    Pre-embodiment WAM/PPO files are G1 checkpoints. They remain loadable for
    G1; a 29-D tensor alone never establishes compatibility with N3.
    """
    if expected is None:
        return
    saved = checkpoint.get("wam_robot_spec")
    if saved is None and expected.get("type") == "g1":
        return
    if saved != dict(expected):
        raise ValueError(
            "Checkpoint robot/joint ordering does not match the current embodiment. "
            "N3 requires an N3 checkpoint; G1 weights cannot be reused by dimension alone."
        )


def get_checkpoint_step(checkpoint: Mapping[str, Any]) -> int:
    """Read an evaluation/export step from PPO or WAM checkpoint metadata.

    PPO's original ``state.global_step`` takes precedence, including zero.
    Its state may be a TrainerState-like object or a serialized mapping. WAM
    stores its iteration directly because it does not use TrainerState.
    Reject malformed metadata rather than silently labeling an export step 0.
    """
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint metadata must be a mapping")

    missing = object()
    state = checkpoint.get("state")
    if isinstance(state, Mapping):
        step = state.get("global_step", missing)
    else:
        step = getattr(state, "global_step", missing)
    source = "state.global_step"
    if step is missing:
        source = "wam_iteration"
        step = checkpoint.get(source, missing)
    if step is missing:
        raise ValueError("Checkpoint has neither state.global_step nor wam_iteration")
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
        raise ValueError(f"Checkpoint {source} must be a nonnegative integer, got {step!r}")
    return int(step)
