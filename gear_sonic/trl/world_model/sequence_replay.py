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

    def __init__(self, capacity_steps: int = 256, *, actor_history_layout=None) -> None:
        if capacity_steps < 1:
            raise ValueError("capacity_steps must be positive")
        self.capacity_steps = int(capacity_steps)
        self._storage: dict[str, Tensor] = {}
        self._cursor = 0
        self._size = 0
        self._configure_actor_history(actor_history_layout)

    def _configure_actor_history(self, layout):
        """Describe term-major history without changing the public replay schema.

        Layout entries are ``(name, start, end, history, term_dim)``. Only
        current/next *frames* are stored in the main ring. An H-1 frame halo
        keeps the oldest retained history intact when the ring overwrites it.
        Unusual segment prefixes are retained verbatim, never re-noised.
        """
        self.actor_history_layout = None if layout is None else tuple(tuple(item) for item in layout)
        self._history_specs = []
        offset = frame_offset = prefix_offset = 0
        max_history = 1
        for item in self.actor_history_layout or ():
            if len(item) != 5:
                raise ValueError("actor history layout entries require name/start/end/history/term_dim")
            _, start, end, history, dim = item
            if any(not isinstance(v, int) or isinstance(v, bool) for v in (start, end, history, dim)):
                raise ValueError("actor history layout dimensions must be integers")
            if start != offset or history < 1 or dim < 1 or end - start != history * dim:
                raise ValueError("actor history layout must cover contiguous positive term histories")
            self._history_specs.append((start, end, history, dim, frame_offset, frame_offset + dim,
                                        prefix_offset, prefix_offset + (history - 1) * dim))
            offset, frame_offset = end, frame_offset + dim
            prefix_offset += (history - 1) * dim
            max_history = max(max_history, history)
        if self.actor_history_layout is not None and not self._history_specs:
            raise ValueError("actor history layout must not be empty")
        self._history_width, self._frame_width = offset, frame_offset
        self._prefix_width, self._halo_steps = prefix_offset, max_history - 1
        self._history_time = 0
        self._actor_segments = self._actor_next_resets = self._actor_halo = None
        self._history_seeds = {}
        self._next_history_seeds = {}
        self._previous_next_actor = None
        self._history_discontinuous = True

    def storage_nbytes(self) -> int:
        """Allocated tensor bytes, including compact-history halo and seeds."""
        values = list(self._storage.values()) + [
            self._actor_segments, self._actor_next_resets, self._actor_halo, self._previous_next_actor,
        ]
        for seeds in (self._history_seeds, self._next_history_seeds):
            for ids, prefix in seeds.values():
                values.extend((ids, prefix))
        return sum(value.numel() * value.element_size() for value in values if value is not None)

    def _actor_frames(self, value):
        return torch.cat([value[..., end - dim:end]
                          for _, end, _, dim, *_ in self._history_specs], dim=-1)

    def _actor_prefix(self, value):
        return torch.cat([value[..., start:end - dim]
                          for start, end, _, dim, *_ in self._history_specs], dim=-1)

    def _remember_seed(self, destination, time, actors, selected):
        # The common reset fills every history slot with the same noisy frame.
        # Such prefixes need only a segment marker, not another history tensor.
        repeated = torch.cat([
            actors[:, end - dim:end].repeat(1, history - 1)
            for _, end, history, dim, *_ in self._history_specs
        ], dim=-1)
        prefix = self._actor_prefix(actors)
        ids = (selected & (prefix != repeated).any(-1)).nonzero(as_tuple=False).flatten()
        if ids.numel():
            destination[time] = (ids, prefix[ids])

    def _store_actor_history(self, actors, following, done):
        n = len(actors)
        if actors.shape[-1] != self._history_width:
            raise ValueError("actor state width does not match compact history layout")
        if self._actor_segments is None:
            self._actor_segments = torch.zeros(self.capacity_steps, n, dtype=torch.long)
            self._actor_next_resets = torch.zeros(self.capacity_steps, n, dtype=torch.bool)
            self._actor_halo = torch.zeros(self._halo_steps, n, self._frame_width, dtype=actors.dtype)
        time = self._history_time
        if time >= self.capacity_steps and self._halo_steps:
            overwritten_time = time - self.capacity_steps
            self._actor_halo[overwritten_time % self._halo_steps].copy_(
                self._storage["actor_state"][self._cursor])
        if self._previous_next_actor is None or self._history_discontinuous:
            fresh = torch.ones(n, dtype=torch.bool)
        else:
            fresh = self._storage["done"][(self._cursor - 1) % self.capacity_steps].clone()
            # A caller may replace next_history without ending the physical
            # episode (e.g. a history reset). That prefix is also the seed of
            # the next current history, even when its values exactly match the
            # previously supplied next_actor_state.
            fresh |= self._actor_next_resets[(self._cursor - 1) % self.capacity_steps]
            fresh |= (actors != self._previous_next_actor).any(-1)
        previous_segment = (
            self._actor_segments[(self._cursor - 1) % self.capacity_steps].clone()
            if time else torch.zeros(n, dtype=torch.long)
        )
        self._actor_segments[self._cursor] = torch.where(fresh, time, previous_segment)
        self._remember_seed(self._history_seeds, time, actors, fresh)
        shifted_prefix = torch.cat([
            actors[:, start + dim:end]
            for start, end, _, dim, *_ in self._history_specs
        ], dim=-1)
        next_reset = (self._actor_prefix(following) != shifted_prefix).any(-1)
        self._actor_next_resets[self._cursor] = next_reset
        self._remember_seed(self._next_history_seeds, time, following, next_reset)
        self._previous_next_actor = following.clone()
        self._history_discontinuous = False
        self._history_time += 1
        # Prefixes older than the oldest possible halo cannot be sampled again.
        cutoff = self._history_time - self.capacity_steps - self._halo_steps
        for seeds in (self._history_seeds, self._next_history_seeds):
            for key in list(seeds):
                if key < cutoff:
                    del seeds[key]
        return self._actor_frames(actors), self._actor_frames(following)

    def _restore_actor_history(self, times, envs, *, following=False):
        """Rebuild exact original term histories for arbitrary sampled rows."""
        shape = times.shape
        times, envs = times.reshape(-1), envs.expand(shape).reshape(-1)
        rows = times % self.capacity_steps
        segments = self._actor_segments[rows, envs]
        oldest = self._history_time - self._size
        parts = []
        for start, end, history, dim, frame_start, frame_end, prefix_start, _ in self._history_specs:
            ages = times[:, None] + torch.arange(1 - history, 1)
            clamped = torch.maximum(ages, segments[:, None])
            frames = self._storage["actor_state"][clamped % self.capacity_steps, envs[:, None], frame_start:frame_end]
            if self._halo_steps:
                in_halo = clamped < oldest
                halo = self._actor_halo[clamped % self._halo_steps, envs[:, None], frame_start:frame_end]
                frames = torch.where(in_halo[..., None], halo, frames)
            for segment in segments[(ages < segments[:, None]).any(-1)].unique().tolist():
                seed = self._history_seeds.get(segment)
                if seed is None:
                    continue
                seed_envs, prefixes = seed
                # Sparse prefixes only occur at true discontinuities or when
                # callers supply a non-repeated initial noisy history.
                seed_lookup = torch.full((self.num_envs,), -1, dtype=torch.long)
                seed_lookup[seed_envs] = torch.arange(len(seed_envs))
                selected = (segments == segment) & (seed_lookup[envs] >= 0)
                indices = selected.nonzero(as_tuple=False).flatten()
                offsets = ages[indices] - segment + history - 1
                prefix = prefixes[seed_lookup[envs[indices]], prefix_start:prefix_start + (history - 1) * dim]
                prefix = prefix.reshape(-1, history - 1, dim)
                replacement = prefix.gather(1, offsets.clamp(0, history - 2)[..., None].expand(-1, -1, dim))
                frames[indices] = torch.where((offsets < history - 1)[..., None], replacement, frames[indices])
            parts.append(frames.flatten(1))
        result = torch.cat(parts, dim=-1)
        if following:
            compact_next = self._storage["next_actor_state"][rows, envs]
            reset = self._actor_next_resets[rows, envs]
            parts = []
            for start, end, history, dim, fs, fe, *_ in self._history_specs:
                appended = torch.cat((result[:, start + dim:end], compact_next[:, fs:fe]), dim=-1)
                repeated = compact_next[:, fs:fe].repeat(1, history)
                parts.append(torch.where(reset[:, None], repeated, appended))
            result = torch.cat(parts, dim=-1)
            for time in times[reset].unique().tolist():
                seed = self._next_history_seeds.get(time)
                if seed is None:
                    continue
                seed_envs, prefix = seed
                lookup = torch.full((self.num_envs,), -1, dtype=torch.long)
                lookup[seed_envs] = torch.arange(len(seed_envs))
                selected = ((times == time) & (lookup[envs] >= 0)).nonzero(as_tuple=False).flatten()
                for start, end, _, dim, _, _, ps, pe in self._history_specs:
                    result[selected, start:end - dim] = prefix[lookup[envs[selected]], ps:pe]
        return result.reshape(*shape, self._history_width)

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
        full_actors = None
        if self.actor_history_layout is not None:
            if "actor_state" not in values:
                raise ValueError("compact actor history requires actor_state and next_actor_state")
            if values["actor_state"].shape[-1] != self._history_width:
                raise ValueError("actor state width does not match compact history layout")
            full_actors = values["actor_state"], values["next_actor_state"]
            values["actor_state"] = self._actor_frames(full_actors[0])
            values["next_actor_state"] = self._actor_frames(full_actors[1])
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
        if full_actors is not None:
            self._store_actor_history(*full_actors, values["done"])
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
        dependencies = {
            "states": ("state", "next_state"), "actions": ("action",),
            "rewards": ("reward",), "dones": ("done",),
            "terminated": ("terminated",), "mask": ("done",),
            "dynamics_mask": ("done", "next_state_valid"),
        }
        pairs = {
            "goals": ("goal", "next_goal"),
            "actor_states": ("actor_state", "next_actor_state"),
            "references": ("reference", "next_reference"),
            "future_references": ("future_reference", "next_future_reference"),
            "critic_states": ("critic_state", "next_critic_state"),
        }
        singles = {"contexts": "context", "previous_actions": "previous_action"}
        dependencies.update({key: pair for key, pair in pairs.items() if pair[0] in self._storage})
        dependencies.update({key: (source,) for key, source in singles.items() if source in self._storage})
        requested = list(dependencies) if fields is None else list(fields)
        missing = set(requested) - set(dependencies)
        if missing:
            raise ValueError(f"Replay sample is missing requested fields: {sorted(missing)}")
        needed = {source for key in requested for source in dependencies[key]}
        windows = {key: self._storage[key][time_ids, envs[:, None]] for key in needed}
        mask = None
        if "mask" in requested or "dynamics_mask" in requested:
            done = windows["done"]
        # The validity at t depends on done values strictly before t.
            mask = torch.cat(
                [torch.ones((batch_size, 1), dtype=torch.bool), (~done[:, :-1]).cumprod(1).bool()], dim=1,
            )
        output = {}
        for key in requested:
            if key == "mask":
                output[key] = mask
            elif key == "dynamics_mask":
                output[key] = mask & windows["next_state_valid"].cumprod(1).bool()
            elif key == "actor_states" and self.actor_history_layout is not None:
                times = self._history_time - self._size + starts[:, None] + offsets[None, :]
                current = self._restore_actor_history(times[:, :1], envs[:, None])
                following = self._restore_actor_history(times, envs[:, None], following=True)
                output[key] = torch.cat((current, following), dim=1)
            elif len(dependencies[key]) == 2:
                current, following = dependencies[key]
                output[key] = torch.cat((windows[current][:, :1], windows[following]), dim=1)
            else:
                output[key] = windows[dependencies[key][0]]
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
        if self.actor_history_layout is not None:
            # ``times`` are physical ring rows; convert to retained absolute time.
            oldest = self._history_time - self._size
            absolute = oldest + (times - oldest) % self.capacity_steps
            result["actor_state"] = self._restore_actor_history(absolute, envs)
            result["next_actor_state"] = self._restore_actor_history(absolute, envs, following=True)
        return {key: value.to(device) for key, value in result.items()} if device is not None else result

    def reference_indices(self):
        """Keys retained by the ring, for pruning unloaded reference generations."""
        value = self._storage.get("reference_index")
        return None if value is None else value[:self._size]

    def state_dict(self) -> dict[str, Any]:
        """Return filled storage in chronological order, independent of cursor."""
        oldest = (self._cursor - self._size) % self.capacity_steps
        ids = (oldest + torch.arange(self._size)) % self.capacity_steps
        result = {
            "capacity_steps": self.capacity_steps,
            "size": self._size,
            # Advanced indexing already owns its storage. A second clone would
            # duplicate a potentially multi-GiB replay during checkpointing.
            "storage": {key: value[ids] for key, value in self._storage.items()},
        }
        if self.actor_history_layout is not None:
            result["actor_history"] = {
                "layout": self.actor_history_layout,
                "time": self._history_time,
                "segments": None if self._actor_segments is None else self._actor_segments[ids],
                "next_resets": None if self._actor_next_resets is None else self._actor_next_resets[ids],
                "halo": None if self._actor_halo is None else self._actor_halo.clone(),
                "seeds": {key: (envs.clone(), prefix.clone()) for key, (envs, prefix) in self._history_seeds.items()},
                "next_seeds": {key: (envs.clone(), prefix.clone()) for key, (envs, prefix) in self._next_history_seeds.items()},
                "previous_next_actor": None if self._previous_next_actor is None else self._previous_next_actor.clone(),
                "discontinuous": self._history_discontinuous,
            }
        return result

    def mark_discontinuity(self) -> None:
        """Seal the newest episode segment before resuming a reset simulator.

        A checkpoint restores replay, not simulator physics. Mark the boundary
        as a truncation so windows never connect old states to the new run.
        """
        if self._size:
            latest = (self._cursor - 1) % self.capacity_steps
            self._storage["done"][latest].fill_(True)
        self._history_discontinuous = True

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
        history = state.get("actor_history")
        # An old checkpoint stores full histories. Its actual schema takes
        # precedence over the constructor's requested compression setting.
        self._configure_actor_history(None if history is None else history["layout"])
        absolute_time = size if history is None else int(history["time"])
        if absolute_time < size:
            raise ValueError("invalid compact history time")
        row_ids = (absolute_time - size + torch.arange(size)) % capacity
        restored = {}
        # Older snapshots never stored pre-reset terminal states.
        storage = dict(storage)
        if size and "next_state_valid" not in storage:
            storage["next_state_valid"] = ~storage["done"].bool()
        for key, value in storage.items():
            if not isinstance(value, Tensor) or value.ndim < 2 or value.shape[0] != size:
                raise ValueError("invalid replay tensor shape")
            restored[key] = torch.empty((capacity, *value.shape[1:]), dtype=value.dtype)
            restored[key][row_ids] = value.detach().cpu()
        self.capacity_steps = capacity
        self._storage = restored
        self._size = size
        self._cursor = absolute_time % capacity
        self._history_time = absolute_time
        if history is not None and size:
            if "actor_state" not in restored or restored["actor_state"].shape[-1] != self._frame_width:
                raise ValueError("compact replay actor frame width does not match history layout")
            n = self.num_envs
            segments, next_resets, halo = history["segments"], history["next_resets"], history["halo"]
            if segments.shape != (size, n) or next_resets.shape != (size, n):
                raise ValueError("invalid compact history segment metadata")
            if halo.shape != (self._halo_steps, n, self._frame_width):
                raise ValueError("invalid compact history halo")
            self._actor_segments = torch.zeros(capacity, n, dtype=torch.long)
            self._actor_next_resets = torch.zeros(capacity, n, dtype=torch.bool)
            self._actor_segments[row_ids] = segments.cpu()
            self._actor_next_resets[row_ids] = next_resets.cpu()
            self._actor_halo = halo.cpu().clone()
            self._history_seeds = {int(key): (ids.cpu().clone(), prefix.cpu().clone())
                                   for key, (ids, prefix) in history["seeds"].items()}
            self._next_history_seeds = {int(key): (ids.cpu().clone(), prefix.cpu().clone())
                                        for key, (ids, prefix) in history["next_seeds"].items()}
            self._previous_next_actor = history["previous_next_actor"].cpu().clone()
            if self._previous_next_actor.shape != (n, self._history_width):
                raise ValueError("invalid compact previous actor history")
            self._history_discontinuous = bool(history["discontinuous"])


__all__ = ["SequenceReplayBuffer"]
