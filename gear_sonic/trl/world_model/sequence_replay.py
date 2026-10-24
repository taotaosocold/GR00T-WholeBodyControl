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

    _FIELDS = ("state", "action", "reward", "next_state", "done", "terminated")

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
    ) -> None:
        """Store one vectorized step, including actual executed actions.

        ``done`` includes both termination and time-limit truncation. Use
        ``terminated`` for the true terminal label so truncations can bootstrap
        the critic. Auto-reset next observations never supervise dynamics on a
        done transition, regardless of which kind of done occurred.
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
        values = {
            "state": state.detach().to(device="cpu", dtype=torch.float32),
            "action": action.detach().to(device="cpu", dtype=torch.float32),
            "reward": reward.detach().reshape(n).to(device="cpu", dtype=torch.float32),
            "next_state": next_state.detach().to(device="cpu", dtype=torch.float32),
            "done": done.detach().reshape(n).to(device="cpu", dtype=torch.bool),
            "terminated": terminated.detach().reshape(n).to(device="cpu", dtype=torch.bool),
        }
        if (values["terminated"] & ~values["done"]).any():
            raise ValueError("terminated=True requires done=True")
        if not self._storage:
            self._storage = {
                key: torch.empty((self.capacity_steps, *value.shape), dtype=value.dtype)
                for key, value in values.items()
            }
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
    ) -> dict[str, Tensor]:
        """Sample ``[B,K,...]`` windows without crossing the ring's newest end.

        ``mask`` includes the first done transition so its reward and terminal
        label can be learned. Later transitions receive zero weight.
        ``dynamics_mask`` additionally excludes that done transition, whose next
        observation may already belong to a reset episode. ``states`` contains
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
            "dynamics_mask": mask & ~done,
        }
        if device is not None:
            output = {key: value.to(device=device) for key, value in output.items()}
        return output

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
        if size and set(storage) != set(self._FIELDS):
            raise ValueError("replay snapshot is missing transition tensors")
        restored = {}
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
