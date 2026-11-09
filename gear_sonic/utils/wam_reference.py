"""Immutable clip references for policy imagination, independent of real resets.

Each loaded clip is rendered once with the simulator's own tokenizer terms.
Replay stores only a clip key and frame cursor. References therefore continue
past a behaviour-policy failure and remain valid after motion subset reloads.
"""
from __future__ import annotations

from copy import copy
from types import SimpleNamespace

import torch

from gear_sonic.utils.wam_critic import future_reference_states
from gear_sonic.utils.wam_noise import apply_reference_noise, reference_noise_spec


# These observations depend only on the reference, except for orientations
# rendered with an identity robot and converted by WAMGoalObservations later.
_REFERENCE_TERMS = frozenset({
    "encoder_index", "command_num_frames", "command_multi_future_nonflat",
    "command_multi_future_lower_body", "command_z", "command_z_multi_future_nonflat",
    "motion_anchor_ori_b", "motion_anchor_ori_b_nonflat", "motion_anchor_ori_b_mf",
    "motion_anchor_ori_b_mf_nonflat", "smpl_root_ori_b_multi_future",
    "smpl_root_ori_b_multi_future_flatten", "smpl_root_ori_b",
    "smpl_vr_5point_local_target", "smpl_vr_5point_local_orn_target", "vr_3point_local_target",
    "vr_3point_local_orn_target", "smpl_joints_multi_future_local_nonflat",
    "smpl_joints_multi_future_nonflat", "joint_pos_multi_future_wrist_for_smpl",
})


class WAMReferenceBank:
    """CPU clip tensors shared by all replay starts from the same generation."""

    def __init__(self, tokenizer_specs=()):
        self.specs = tuple(tokenizer_specs)
        self.goal_dim = max((spec[2] for spec in self.specs), default=0)
        self.routing_specs = tuple(
            spec for spec in self.specs if spec[0] in {"encoder_index", "command_num_frames"}
        )
        self.clips = {}
        self.current = {}
        self.next_key = 0
        self.transforms = ()
        self._lengths = torch.zeros(0, dtype=torch.long)

    def pack_routing(self, goals):
        """Keep only seed-specific routing; clip goals already live in the bank."""
        if goals.shape[-1] != self.goal_dim:
            raise ValueError("WAM routing requires the complete tokenizer layout")
        if not self.routing_specs:
            return goals[..., :0]
        return torch.cat([goals[..., start:end] for _, start, end, *_ in self.routing_specs], -1)

    def register(self, states, goals):
        if states.ndim != 2 or states.shape[-1] != 71 or states.shape[0] < 2:
            raise ValueError("WAM imagination clips need at least two 71-D physical frames")
        if goals.ndim != 2 or goals.shape[0] != states.shape[0]:
            raise ValueError("WAM clip goals must align with physical reference frames")
        key = self.next_key
        self.next_key += 1
        self.clips[key] = (states.detach().cpu().float().clone(), goals.detach().cpu().float().clone())
        if key >= len(self._lengths):
            lengths = torch.zeros(max(key + 1, max(16, 2 * len(self._lengths))), dtype=torch.long)
            lengths[:len(self._lengths)] = self._lengths
            self._lengths = lengths
        self._lengths[key] = len(states)
        return key

    @torch.no_grad()
    def capture(self, command, observation_manager):
        names = observation_manager._group_obs_term_names["tokenizer"]
        configs = observation_manager._group_obs_term_cfgs["tokenizer"]
        if tuple(names) != tuple(spec[0] for spec in self.specs):
            raise ValueError("WAM reference terms do not match the actor tokenizer layout")
        unsupported = set(names) - _REFERENCE_TERMS
        if unsupported:
            raise ValueError(f"WAM independent imagination does not support reference terms: {sorted(unsupported)}")
        transforms = []
        group_cfg = getattr(observation_manager, "_group_obs_cfg", {}).get("tokenizer")
        corrupt = group_cfg is None or group_cfg.enable_corruption
        for spec, cfg in zip(self.specs, configs):
            if getattr(cfg, "modifiers", None):
                raise ValueError("Stateful tokenizer modifiers are unsupported for independent imagination")
            cfg_noise = getattr(cfg, "noise", None)
            noise = None
            if cfg_noise is not None and corrupt:
                noise = reference_noise_spec(cfg_noise)
            clip, scale = getattr(cfg, "clip", None), getattr(cfg, "scale", None)
            if spec[0] in {"encoder_index", "command_num_frames"} and (
                noise is not None or clip is not None or scale is not None
            ):
                raise ValueError("WAM modality/frame routing must use unmodified observation values")
            transforms.append((spec[1], spec[2], noise, clip, scale))
        self.transforms = tuple(transforms)
        indices = torch.empty(command.num_envs, 2, dtype=torch.long, device=command.motion_ids.device)
        for motion_id in command.motion_ids.unique().tolist():
            if motion_id not in self.current:
                state_parts, goal_parts = [], []
                clip_id = command.motion_ids.new_tensor([motion_id])
                total = int(command.motion_lib.get_time_step_total(clip_id).item())
                if total < 2:
                    raise ValueError("WAM independent imagination requires motion clips with at least two frames")
                for start in range(0, total, 2048):
                    frames = torch.arange(start, min(start + 2048, total), device=clip_id.device)
                    proxy, proxy_env = self._command_view(command, motion_id, frames, total)
                    state_parts.append(future_reference_states(proxy, num_frames=1).squeeze(1).cpu())
                    values = []
                    for spec, cfg in zip(self.specs, configs):
                        value = cfg.func(proxy_env, **cfg.params)
                        value = value.reshape(len(frames), -1)
                        if value.shape[-1] != spec[2] - spec[1]:
                            raise ValueError(f"WAM reference term {spec[0]} changed shape")
                        values.append(value)
                    goal_parts.append(torch.cat(values, -1).cpu())
                self.current[motion_id] = self.register(torch.cat(state_parts), torch.cat(goal_parts))
        indices[:, 0] = torch.tensor(
            [self.current[motion_id] for motion_id in command.motion_ids.tolist()],
            dtype=torch.long, device=indices.device,
        )
        indices[:, 1] = command.motion_start_time_steps + command.time_steps
        return indices

    @staticmethod
    def _command_view(command, motion_id, frames, total):
        """Copy command cursors; never advance/reset the simulator or its manager."""
        proxy = copy(command)
        count, dev = len(frames), frames.device
        origins = torch.zeros(count, 3, device=dev)
        env = SimpleNamespace(num_envs=count, device=command.device, scene=SimpleNamespace(env_origins=origins))
        proxy._env = env
        proxy._offline = False
        proxy.use_ref_motion_root_quat_w_as_anchor = False
        proxy.ref_motion_root_rotation_noise = None
        proxy.motion_ids = torch.full_like(frames, motion_id)
        proxy.time_steps = frames
        proxy.motion_start_time_steps = torch.zeros_like(frames)
        proxy.motion_num_steps = torch.full_like(frames, total)
        proxy.future_motion_ids = proxy.motion_ids.repeat_interleave(proxy.num_future_frames)
        proxy.smpl_future_motion_ids = proxy.motion_ids.repeat_interleave(proxy.smpl_num_future_frames)
        for name in ("future_time_steps_init", "smpl_future_time_steps_init", "vr_3point_body_offsets"):
            value = getattr(command, name)
            setattr(proxy, name, value[:1].expand(count, *value.shape[1:]))
        proxy.encoder_index = command.encoder_index[:1].new_zeros(count, command.encoder_index.shape[-1])
        if getattr(command, "per_env_num_frames", None) is not None:
            proxy.per_env_num_frames = torch.full_like(frames, proxy.num_future_frames)
        bodies = max(command.robot_anchor_body_index + 1, 1)
        rotation = origins.new_zeros(count, bodies, 4)
        rotation[..., 0] = 1
        proxy.robot = SimpleNamespace(data=SimpleNamespace(
            body_pos_w=origins[:, None].expand(-1, bodies, -1), body_quat_w=rotation,
        ))
        # Reference capture needs the same robot-specific joint permutation as
        # the live command; the synthetic data above only replaces root poses.
        if hasattr(getattr(command, "robot", None), "joint_names"):
            proxy.robot.joint_names = tuple(command.robot.joint_names)
        env.command_manager = SimpleNamespace(get_term=lambda name: proxy)
        return proxy, env

    def valid_starts(self, indices):
        indices = indices.cpu()
        keys = indices[..., 0]
        if ((keys < 0) | (keys >= len(self._lengths))).any():
            raise ValueError("Replay refers to a missing WAM reference clip")
        lengths = self._lengths[keys]
        if (lengths == 0).any():
            raise ValueError("Replay refers to a missing WAM reference clip")
        return (indices[..., 1] >= 0) & (indices[..., 1] < lengths - 1)

    def trajectory(
        self, indices, seed_goals, horizon, future_frames, future_stride, device,
        *, routing_only=False,
    ):
        """Advance each reference cursor; only the physical clip boundary masks it."""
        indices = indices.cpu()
        batch = len(indices)
        goal_dim = self.goal_dim if routing_only else seed_goals.shape[-1]
        expected_seed_dim = (
            sum(end - start for _, start, end, *_ in self.routing_specs)
            if routing_only else self.goal_dim
        )
        if seed_goals.shape != (batch, expected_seed_dim):
            raise ValueError("WAM reference seed has an incompatible routing/goal layout")
        goals = torch.empty(batch, horizon + 1, goal_dim)
        states = torch.empty(batch, horizon + 1, 71)
        future = torch.empty(batch, horizon + 1, future_frames, 71)
        lengths = torch.empty(batch, dtype=torch.long)
        for key in indices[:, 0].unique().tolist():
            rows = indices[:, 0] == key
            clip_state, clip_goal = self.clips[key]
            cursor = indices[rows, 1]
            last = len(clip_state) - 1
            lengths[rows] = (last - cursor).clamp(0, horizon)
            times = (cursor[:, None] + torch.arange(horizon + 1)).clamp_max(last)
            states[rows], goals[rows] = clip_state[times], clip_goal[times]
            future_times = (times[..., None] + torch.arange(future_frames) * future_stride).clamp_max(last)
            future[rows] = clip_state[future_times]
        # Modality routing / variable frame choice belongs to the sampled seed,
        # not to a random environment used while rendering the clip bank.
        route_offset = 0
        for _, start, end, *_ in self.routing_specs:
            source_start = route_offset if routing_only else start
            source_end = source_start + end - start
            goals[..., start:end] = seed_goals[:, None, source_start:source_end].detach().cpu()
            route_offset += end - start
        mask = torch.arange(horizon)[None] < lengths[:, None]
        return {"goals": goals.to(device), "references": states.to(device),
                "future_references": future.flatten(-2).to(device), "mask": mask.to(device)}

    def process_goals(self, goals):
        """Match ObservationManager's noise -> clip -> scale after robot rotation."""
        if not self.transforms:
            return goals
        goals = goals.clone()
        for start, end, noise, clip, scale in self.transforms:
            value = goals[..., start:end]
            original_shape = next(spec[3] for spec in self.specs if spec[1:3] == (start, end))
            if noise is not None:
                # Custom current-frame noise needs the true reference-frame
                # axis, independent of any leading imagination rollout axis.
                if isinstance(noise, dict):
                    value = value.reshape(*goals.shape[:-1], *original_shape)
                    value = apply_reference_noise(value, noise).reshape(*goals.shape[:-1], end - start)
                else:
                    value = apply_reference_noise(value, noise)
            if clip is not None:
                value = value.clamp(*clip)
            if scale is not None:
                value = value.reshape(*goals.shape[:-1], *original_shape)
                value = (value * torch.as_tensor(scale, device=value.device, dtype=value.dtype)).reshape(
                    *goals.shape[:-1], end - start,
                )
            goals[..., start:end] = value
        return goals

    def begin_generation(self, retained_indices=None):
        self.current.clear()
        retained = set() if retained_indices is None else set(retained_indices[..., 0].unique().tolist())
        self.clips = {key: value for key, value in self.clips.items() if key in retained}
        self._lengths.zero_()
        for key, (state, _) in self.clips.items():
            self._lengths[key] = len(state)

    def state_dict(self):
        return {"clips": self.clips, "next_key": self.next_key, "transforms": self.transforms}

    def load_state_dict(self, state):
        self.clips = {int(key): (s.cpu(), g.cpu()) for key, (s, g) in state["clips"].items()}
        self.next_key = int(state["next_key"])
        self.transforms = tuple(state["transforms"])
        self.current.clear()
        self._lengths = torch.zeros(self.next_key, dtype=torch.long)
        for key, (clip, _) in self.clips.items():
            self._lengths[key] = len(clip)
