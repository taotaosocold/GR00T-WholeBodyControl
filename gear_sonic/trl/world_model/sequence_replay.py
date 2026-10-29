"""Vector-environment replay windows with episode and ring-boundary masks.

One insertion stores one timestep for all selected environments. Windows use
the same environment throughout and advance in chronological order, including
when the storage ring wraps. The buffer lives on CPU; sampled batches can be
transferred to the policy device in one operation per tensor.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor


class SequenceReplayBuffer:
    """Fixed-size CPU ring of batched simulator transitions.

    ``len(buffer)`` is the number of available *timesteps*, rather than the
    number of individual environment transitions. The environment subset must
    keep the same ordering on every insertion.
    """

    _BASE_FIELDS = ("state", "action", "reward", "next_state", "done", "terminated")
    _GOAL_FIELDS = ("goal", "next_goal")
    _ACTOR_FIELDS = ("actor_state", "next_actor_state")
    _CONTEXT_FIELDS = ("context",)
    _REFERENCE_FIELDS = ("reference", "next_reference")
    _FUTURE_REFERENCE_FIELDS = ("future_reference", "next_future_reference")
    _CRITIC_FIELDS = ("critic_state", "next_critic_state")
    _PREVIOUS_ACTION_FIELDS = ("previous_action",)
    _NEXT_STATE_FIELDS = ("next_state_valid",)
    _REFERENCE_INDEX_FIELDS = ("reference_index",)
    # Preserve the old introspection hook for callers that only use the base
    # observation replay format.
    _FIELDS = _BASE_FIELDS

    def __init__(self, capacity_steps: int = 256) -> None:
        if capacity_steps < 1:
            raise ValueError("capacity_steps must be positive")
        self.capacity_steps = int(capacity_steps)
        self._storage: dict[str, Tensor] = {}
        self._cursor = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def num_envs(self) -> int:
        return self._storage["state"].shape[1] if self._storage else 0

    @torch.no_grad()
    def add(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
        terminated: Tensor | None = None,
        goal: Tensor | None = None,
        next_goal: Tensor | None = None,
        actor_state: Tensor | None = None,
        next_actor_state: Tensor | None = None,
        context: Tensor | None = None,
        reference: Tensor | None = None,
        next_reference: Tensor | None = None,
        critic_state: Tensor | None = None,
        next_critic_state: Tensor | None = None,
        previous_action: Tensor | None = None,
        next_state_valid: Tensor | None = None,
        future_reference: Tensor | None = None,
        next_future_reference: Tensor | None = None,
        reference_index: Tensor | None = None,
    ) -> None:
        """Store one vectorized step, including actual executed actions.

        ``done`` includes both termination and time-limit truncation. Use
        ``terminated`` for the true terminal label so truncations can bootstrap
        the critic. Auto-reset next observations never supervise dynamics on a
        done transition unless ``next_state_valid`` confirms that a pre-reset
        terminal state was captured. That state supervises the last transition,
        while the episode mask still excludes subsequent reset observations.
        Optional future-reference tensors contain the flattened raw robot
        reference window at each corresponding state, independently of the
        actor's multimodal goal representation.
        """
        if state.ndim != 2 or action.ndim != 2:
            raise ValueError("state and action must have shapes [N,D] and [N,A]")
        if state.shape[0] < 1 or state.shape[1] < 1 or action.shape[1] < 1:
            raise ValueError("environment, state, and action dimensions must be positive")
        if next_state.shape != state.shape or action.shape[0] != state.shape[0]:
            raise ValueError("state, next_state, and action batches must match")
        n = state.shape[0]
        if reward.numel() != n or done.numel() != n:
            raise ValueError("reward and done must contain one value per environment")
        if terminated is None:
            terminated = done
        if terminated.numel() != n:
            raise ValueError("terminated must contain one value per environment")
        if (goal is None) != (next_goal is None):
            raise ValueError("goal and next_goal must be provided together")
        if goal is not None:
            if goal.ndim != 2 or next_goal.ndim != 2:
                raise ValueError("goal and next_goal must have shapes [N,G]")
            if goal.shape != next_goal.shape or goal.shape[0] != n or goal.shape[1] < 1:
                raise ValueError("goal and next_goal batches must match [N,G]")
        if (actor_state is None) != (next_actor_state is None):
            raise ValueError("actor_state and next_actor_state must be provided together")
        if actor_state is not None:
            if actor_state.ndim != 2 or next_actor_state.ndim != 2:
                raise ValueError("actor_state and next_actor_state must have shapes [N,D]")
            if actor_state.shape != next_actor_state.shape or actor_state.shape[0] != n:
                raise ValueError("actor state batches must match [N,D]")
        if context is not None:
            if context.ndim != 2 or context.shape[0] != n or context.shape[1] < 1:
                raise ValueError("context must have shape [N,C]")
        if (reference is None) != (next_reference is None):
            raise ValueError("reference and next_reference must be provided together")
        if reference is not None:
            if reference.ndim != 2 or next_reference.ndim != 2:
                raise ValueError("reference tensors must have shapes [N,D]")
            if reference.shape != next_reference.shape or reference.shape[0] != n:
                raise ValueError("reference batches must match")
        if (future_reference is None) != (next_future_reference is None):
            raise ValueError("future_reference and next_future_reference must be provided together")
        if future_reference is not None:
            if future_reference.ndim != 2 or next_future_reference.ndim != 2:
                raise ValueError("future reference tensors must have shapes [N,D]")
            if (
                future_reference.shape != next_future_reference.shape
                or future_reference.shape[0] != n
                or future_reference.shape[1] < 1
            ):
                raise ValueError("future reference batches must match [N,D] with positive D")
        if (critic_state is None) != (next_critic_state is None):
            raise ValueError("critic_state and next_critic_state must be provided together")
        if critic_state is not None:
            if critic_state.ndim != 2 or next_critic_state.ndim != 2:
                raise ValueError("critic tensors must have shapes [N,D]")
            if critic_state.shape != next_critic_state.shape or critic_state.shape[0] != n:
                raise ValueError("critic batches must match")
        if previous_action is not None:
            if previous_action.ndim != 2 or previous_action.shape[0] != n:
                raise ValueError("previous_action must have shape [N,A]")
        values = {
            "state": state.detach().to(device="cpu", dtype=torch.float32),
            "action": action.detach().to(device="cpu", dtype=torch.float32),
            "reward": reward.detach().reshape(n).to(device="cpu", dtype=torch.float32),
            "next_state": next_state.detach().to(device="cpu", dtype=torch.float32),
            "done": done.detach().reshape(n).to(device="cpu", dtype=torch.bool),
            "terminated": terminated.detach().reshape(n).to(device="cpu", dtype=torch.bool),
        }
        if next_state_valid is None:
            next_state_valid = ~done.bool()
        if next_state_valid.numel() != n:
            raise ValueError("next_state_valid must contain one value per environment")
        values["next_state_valid"] = next_state_valid.detach().reshape(n).to(device="cpu", dtype=torch.bool)
        if reference_index is not None:
            if reference_index.shape != (n, 2) or reference_index.dtype != torch.long:
                raise ValueError("reference_index must contain integer [clip key, frame] pairs")
            values["reference_index"] = reference_index.detach().cpu().clone()
        if goal is not None:
            values["goal"] = goal.detach().to(device="cpu", dtype=torch.float32)
            values["next_goal"] = next_goal.detach().to(device="cpu", dtype=torch.float32)
        if actor_state is not None:
            values["actor_state"] = actor_state.detach().to(device="cpu", dtype=torch.float32)
            values["next_actor_state"] = next_actor_state.detach().to(device="cpu", dtype=torch.float32)
        if context is not None:
            values["context"] = context.detach().to(device="cpu", dtype=torch.float32)
        if reference is not None:
            values["reference"] = reference.detach().to(device="cpu", dtype=torch.float32)
            values["next_reference"] = next_reference.detach().to(device="cpu", dtype=torch.float32)
        if future_reference is not None:
            values["future_reference"] = future_reference.detach().to(device="cpu", dtype=torch.float32)
            values["next_future_reference"] = next_future_reference.detach().to(device="cpu", dtype=torch.float32)
        if critic_state is not None:
            values["critic_state"] = critic_state.detach().to(device="cpu", dtype=torch.float32)
            values["next_critic_state"] = next_critic_state.detach().to(device="cpu", dtype=torch.float32)
        if previous_action is not None:
            values["previous_action"] = previous_action.detach().to(device="cpu", dtype=torch.float32)
        if (values["terminated"] & ~values["done"]).any():
            raise ValueError("terminated=True requires done=True")
        if not self._storage:
            self._storage = {
                key: torch.empty((self.capacity_steps, *value.shape), dtype=value.dtype)
                for key, value in values.items()
            }
        elif set(values) != set(self._storage):
            raise ValueError("optional replay fields must be present on every transition")
        for key, value in values.items():
            if value.shape != self._storage[key].shape[1:]:
                raise ValueError("transition dimensions changed after buffer initialization")
        for key, value in values.items():
            self._storage[key][self._cursor].copy_(value)
        self._cursor = (self._cursor + 1) % self.capacity_steps
        self._size = min(self._size + 1, self.capacity_steps)

    def sample(
        self,
        batch_size: int,
        window_length: int,
        device: torch.device | str | None = None,
        *,
        generator: torch.Generator | None = None,
        fields=None,
    ) -> dict[str, Tensor]:
        """Sample ``[B,K,...]`` windows without crossing the ring's newest end.

        ``mask`` includes the first done transition so its reward and terminal
        label can be learned. Later transitions receive zero weight.
        ``dynamics_mask`` excludes a done transition unless its true pre-reset
        next state was captured. ``states`` contains
        the first state followed by each logged next state, hence ``K+1`` frames.
        """
        if batch_size < 1 or window_length < 1:
            raise ValueError("batch_size and window_length must be positive")
        if window_length > self._size:
            raise ValueError("not enough replay timesteps for the requested window")
        starts = torch.randint(
            self._size - window_length + 1, (batch_size,), generator=generator
        )
        envs = torch.randint(self.num_envs, (batch_size,), generator=generator)
        oldest = (self._cursor - self._size) % self.capacity_steps
        offsets = torch.arange(window_length)
        time_ids = (oldest + starts[:, None] + offsets[None, :]) % self.capacity_steps
        windows = {
            key: value[time_ids, envs[:, None]] for key, value in self._storage.items()
        }
        done = windows["done"]
        # The validity at t depends on done values strictly before t.
        mask = torch.cat(
            [torch.ones((batch_size, 1), dtype=torch.bool), (~done[:, :-1]).cumprod(1).bool()],
            dim=1,
        )
        output = {
            "states": torch.cat([windows["state"][:, :1], windows["next_state"]], dim=1),
            "actions": windows["action"],
            "rewards": windows["reward"],
            "dones": done,
            "terminated": windows["terminated"],
            "mask": mask,
            # An unobserved impulse breaks every later autoregressive target
            # from this seed, even when it does not reset the simulator.
            "dynamics_mask": mask & windows["next_state_valid"].cumprod(1).bool(),
        }
        if "goal" in windows:
            output["goals"] = torch.cat([windows["goal"][:, :1], windows["next_goal"]], dim=1)
        if "actor_state" in windows:
            output["actor_states"] = torch.cat(
                [windows["actor_state"][:, :1], windows["next_actor_state"]], dim=1
            )
        if "context" in windows:
            output["contexts"] = windows["context"]
        if "reference" in windows:
            output["references"] = torch.cat(
                [windows["reference"][:, :1], windows["next_reference"]], dim=1
            )
        if "future_reference" in windows:
            output["future_references"] = torch.cat(
                [windows["future_reference"][:, :1], windows["next_future_reference"]], dim=1
            )
        if "critic_state" in windows:
            output["critic_states"] = torch.cat(
                [windows["critic_state"][:, :1], windows["next_critic_state"]], dim=1
            )
        if "previous_action" in windows:
            output["previous_actions"] = windows["previous_action"]
        if fields is not None:
            missing = set(fields) - set(output)
            if missing:
                raise ValueError(f"Replay sample is missing requested fields: {sorted(missing)}")
            output = {key: output[key] for key in fields}
        if device is not None:
            output = {key: value.to(device=device) for key, value in output.items()}
        return output

    def sample_starts(self, batch_size, device=None, *, eligible=None):
        """Sample individual real states without consulting subsequent done.

        ``eligible`` may reject reference clips that have no next frame. The
        real episode after a sampled start never limits imagined survival.
        """
        if batch_size < 1 or not self._size:
            raise ValueError("policy starts require a nonempty replay and positive batch")
        keep = torch.ones(self._size, self.num_envs, dtype=torch.bool)
        if eligible is not None:
            keep &= eligible(self._storage["reference_index"][:self._size])
        candidates = keep.flatten().nonzero(as_tuple=False).flatten()
        if not candidates.numel():
            raise ValueError("Replay has no motion reference with a valid next frame")
        selected = candidates[torch.randint(len(candidates), (batch_size,))]
        times, envs = selected // self.num_envs, selected % self.num_envs
        result = {key: value[times, envs] for key, value in self._storage.items()}
        return {key: value.to(device) for key, value in result.items()} if device is not None else result

    def reference_indices(self):
        """Keys retained by the ring, for pruning unloaded reference generations."""
        value = self._storage.get("reference_index")
        return None if value is None else value[:self._size]

    def state_dict(self) -> dict[str, Any]:
        """Return filled storage in chronological order, independent of cursor."""
        oldest = (self._cursor - self._size) % self.capacity_steps
        ids = (oldest + torch.arange(self._size)) % self.capacity_steps
        return {
            "capacity_steps": self.capacity_steps,
            "size": self._size,
            "storage": {key: value[ids].clone() for key, value in self._storage.items()},
        }

    def mark_discontinuity(self) -> None:
        """Seal the newest episode segment before resuming a reset simulator.

        A checkpoint restores replay, not simulator physics. Mark the boundary
        as a truncation so windows never connect old states to the new run.
        """
        if self._size:
            latest = (self._cursor - 1) % self.capacity_steps
            self._storage["done"][latest].fill_(True)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a replay snapshot, including its original capacity."""
        capacity = int(state["capacity_steps"])
        size = int(state["size"])
        storage = state["storage"]
        if capacity < 1 or not 0 <= size <= capacity:
            raise ValueError("invalid replay capacity or size")
        fields = set(storage)
        if size and not fields.issubset(
            set(self._BASE_FIELDS)
            | set(self._GOAL_FIELDS)
            | set(self._ACTOR_FIELDS)
            | set(self._CONTEXT_FIELDS)
            | set(self._REFERENCE_FIELDS)
            | set(self._FUTURE_REFERENCE_FIELDS)
            | set(self._CRITIC_FIELDS)
            | set(self._PREVIOUS_ACTION_FIELDS)
            | set(self._NEXT_STATE_FIELDS)
            | set(self._REFERENCE_INDEX_FIELDS)
        ):
            raise ValueError("replay snapshot contains unknown transition tensors")
        if size and not set(self._BASE_FIELDS).issubset(fields):
            raise ValueError("replay snapshot is missing transition tensors")
        if fields.intersection(self._GOAL_FIELDS) and not set(self._GOAL_FIELDS).issubset(fields):
            raise ValueError("replay snapshot must contain both goal tensors")
        if fields.intersection(self._ACTOR_FIELDS) and not set(self._ACTOR_FIELDS).issubset(fields):
            raise ValueError("replay snapshot must contain both actor state tensors")
        if fields.intersection(self._REFERENCE_FIELDS) and not set(self._REFERENCE_FIELDS).issubset(fields):
            raise ValueError("replay snapshot must contain both reference tensors")
        if fields.intersection(self._FUTURE_REFERENCE_FIELDS):
            if not set(self._FUTURE_REFERENCE_FIELDS).issubset(fields):
                raise ValueError("replay snapshot must contain both future reference tensors")
            future = storage["future_reference"]
            next_future = storage["next_future_reference"]
            if (
                not isinstance(future, Tensor) or not isinstance(next_future, Tensor)
                or future.ndim != 3 or next_future.shape != future.shape
                or future.shape[-1] < 1
                or "state" not in storage or not isinstance(storage["state"], Tensor)
                or storage["state"].ndim < 2 or future.shape[:2] != storage["state"].shape[:2]
            ):
                raise ValueError("invalid replay future reference tensor shape")
        if fields.intersection(self._CRITIC_FIELDS) and not set(self._CRITIC_FIELDS).issubset(fields):
            raise ValueError("replay snapshot must contain both critic tensors")
        restored = {}
        # Older snapshots never stored pre-reset terminal states.
        storage = dict(storage)
        if size and "next_state_valid" not in storage:
            storage["next_state_valid"] = ~storage["done"].bool()
        for key, value in storage.items():
            if not isinstance(value, Tensor) or value.ndim < 2 or value.shape[0] != size:
                raise ValueError("invalid replay tensor shape")
            restored[key] = torch.empty((capacity, *value.shape[1:]), dtype=value.dtype)
            restored[key][:size].copy_(value.detach().cpu())
        self.capacity_steps = capacity
        self._storage = restored
        self._size = size
        self._cursor = size % capacity


__all__ = ["SequenceReplayBuffer"]
