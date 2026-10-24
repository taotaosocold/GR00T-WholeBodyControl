"""Differentiable world-action model utilities for model-based WBC training.

The original SONIC/GR00T policy is kept intact: observations still enter the
FSQ token bottleneck and the existing decoders still produce joint actions.
This module is an optional model-based branch.  It learns a residual dynamics
model from transitions collected by Isaac Lab and provides short imagined
rollouts through which an actor can be updated with BPTT.

The implementation deliberately uses flat tensors instead of a robot-specific
state layout.  A caller can therefore use proprioception, privileged state, or
an FSQ token embedding as ``state`` while keeping the same API.  The model is
deterministic, which matches the SuperMimic setting and keeps inference cheap;
stochastic environment effects can be represented by the optional ``context``
vector (for example terrain/friction or motion-command features).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    """Build the small residual MLP used by dynamics and termination heads."""
    layers: list[nn.Module] = []
    dim = in_dim
    for width in hidden:
        layers.extend((nn.Linear(dim, width), nn.LayerNorm(width), nn.SiLU()))
        dim = width
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Streaming normalizer with distributed-safe tensor buffers.

    Statistics are updated explicitly through :meth:`update`, so validation and
    imagined rollouts never mutate the model.  The update is performed in
    ``float64`` and copied back to the module dtype for stable long runs.
    """

    def __init__(self, dim: int, eps: float = 1e-4, max_count: float = 2e8):
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        self.eps = float(eps)
        self.max_count = float(max_count)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1, self.mean.numel()).float()
        if values.numel() == 0 or self.count >= self.max_count:
            return
        # Accumulate in float64 but retain the buffer dtype (normally fp32).
        x = values.double()
        n = torch.tensor(float(x.shape[0]), dtype=torch.float64, device=x.device)
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        old_count = self.count.double()
        total = old_count + n
        delta = batch_mean - self.mean.double()
        new_mean = self.mean.double() + delta * n / total
        new_var = (
            self.var.double() * old_count
            + batch_var * n
            + delta.square() * old_count * n / total
        ) / total
        self.mean.copy_(new_mean.to(self.mean.dtype))
        self.var.copy_(new_var.clamp_min(0).to(self.var.dtype))
        self.count.copy_(total.to(self.count.dtype))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std()

    def std(self) -> torch.Tensor:
        # Static routing bits and short warmup batches can have zero variance.
        # A finite scale keeps their Jacobians bounded during imagination.
        return torch.sqrt(self.var + self.eps**2).clamp_min(1e-2)


@dataclass
class WAMConfig:
    """Configuration for :class:`DifferentiableWorldModel`.

    ``hidden`` is intentionally smaller than the 2048-wide SuperMimic model by
    default.  The dimensions of GR00T observations are usually larger, so the
    caller can scale this tuple without changing any training code.
    """

    state_dim: int
    action_dim: int
    context_dim: int = 0
    hidden: tuple[int, ...] = (512, 512, 512)
    term_hidden: tuple[int, ...] = (128, 128)
    reward_hidden: tuple[int, ...] = (128, 128)
    learn_termination: bool = True
    learn_reward: bool = True
    reward_loss_coef: float = 1.0
    residual_scale: float = 1.0


class StateSpaceWorldModel(nn.Module):
    """Learned residual dynamics, reward, and differentiable termination predictors.

    The model predicts ``s[t+1] = s[t] + delta(s[t], a[t], context[t])``.  A
    residual parameterisation is easier to fit for high-dimensional humanoid
    states and lets the model be initialised as an identity map.  All methods
    support leading batch dimensions, so tensors from vectorised Isaac Lab
    environments can be passed without reshaping by the caller.

    Args:
        cfg: :class:`WAMConfig` or a Hydra-compatible mapping/object exposing
            the same attributes.
    """

    def __init__(self, cfg: WAMConfig | object):
        super().__init__()
        self.cfg = cfg
        self.state_dim = int(cfg.state_dim)
        self.action_dim = int(cfg.action_dim)
        self.context_dim = int(getattr(cfg, "context_dim", 0))
        self.residual_scale = float(getattr(cfg, "residual_scale", 1.0))
        hidden = tuple(int(v) for v in getattr(cfg, "hidden", (512, 512, 512)))
        term_hidden = tuple(int(v) for v in getattr(cfg, "term_hidden", (128, 128)))
        reward_hidden = tuple(int(v) for v in getattr(cfg, "reward_hidden", term_hidden))
        self.reward_loss_coef = float(getattr(cfg, "reward_loss_coef", 1.0))
        if self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("state_dim and action_dim must be positive")
        model_in = self.state_dim + self.action_dim + self.context_dim
        self.dynamics = _mlp(model_in, hidden, self.state_dim)
        # Zero initial residual keeps the model compatible with a warm-started
        # policy and avoids unstable imagined rollouts before fitting starts.
        final = self.dynamics[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.state_norm = RunningNorm(self.state_dim)
        self.action_norm = RunningNorm(self.action_dim)
        self.context_norm = RunningNorm(self.context_dim) if self.context_dim else None
        self.delta_norm = RunningNorm(self.state_dim)
        learn_term = bool(getattr(cfg, "learn_termination", True))
        self.termination_head = (
            _mlp(self.state_dim, term_hidden, 1) if learn_term else None
        )
        learn_reward = bool(getattr(cfg, "learn_reward", True))
        self.reward_head = _mlp(model_in, reward_hidden, 1) if learn_reward else None

    def _context(self, context: Optional[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        if self.context_dim == 0:
            return state.new_zeros(*state.shape[:-1], 0)
        if context is None:
            return state.new_zeros(*state.shape[:-1], self.context_dim)
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context last dimension must be {self.context_dim}, got {context.shape[-1]}"
            )
        return context

    def _model_input(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state last dimension must be {self.state_dim}, got {state.shape[-1]}"
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"action last dimension must be {self.action_dim}, got {action.shape[-1]}"
            )
        context = self._context(context, state)
        state_n = self.state_norm(state)
        action_n = self.action_norm(action)
        if self.context_norm is not None:
            context = self.context_norm(context)
        return torch.cat((state_n, action_n, context), dim=-1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Predict next state, reward, and termination logits for one transition."""
        x = self._model_input(state, action, context)
        delta = self.dynamics(x) * self.residual_scale
        next_state = state + delta * self.delta_norm.std()
        out = {"next_state": next_state, "delta": delta}
        if self.reward_head is not None:
            out["reward"] = self.reward_head(x).squeeze(-1)
        if self.termination_head is not None:
            out["termination_logits"] = self.termination_head(
                self.state_norm(next_state)
            ).squeeze(-1)
        return out

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Differentiable one-step transition used by imagined rollouts."""
        return self.forward(state, action, context)["next_state"]

    def rollout(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        *,
        detach: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Unroll the dynamics for a batch of action sequences.

        Args:
            state: Initial state ``(B, state_dim)``.
            actions: Action sequence ``(B, horizon, action_dim)``.
            context: Optional ``(B, horizon, context_dim)`` or ``(B, context_dim)``.
            detach: Detach each predicted state.  Set this for diagnostics; keep
                it false for policy BPTT through the frozen WAM.

        Returns:
            Dict containing ``states`` with shape ``(B, horizon + 1, state_dim)``
            and optional ``termination_logits`` with shape ``(B, horizon)``.
        """
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape (B,H,A), got {tuple(actions.shape)}")
        if state.ndim != 2 or state.shape[0] != actions.shape[0]:
            raise ValueError("state must have shape (B,state_dim) matching actions batch")
        if context is not None and context.ndim == 2:
            context = context.unsqueeze(1).expand(-1, actions.shape[1], -1)
        if context is not None and context.shape[:2] != actions.shape[:2]:
            raise ValueError("context must have shape (B,H,C) or (B,C)")
        states = [state]
        rewards = []
        terms = []
        current = state
        for t in range(actions.shape[1]):
            ctx_t = None if context is None else context[:, t]
            result = self.forward(current, actions[:, t], ctx_t)
            current = result["next_state"]
            if detach:
                current = current.detach()
            states.append(current)
            if "reward" in result:
                rewards.append(result["reward"])
            if "termination_logits" in result:
                terms.append(result["termination_logits"])
        output = {"states": torch.stack(states, dim=1)}
        if rewards:
            output["rewards"] = torch.stack(rewards, dim=1)
        if terms:
            output["termination_logits"] = torch.stack(terms, dim=1)
        return output

    def transition_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        dynamics_mask: Optional[torch.Tensor] = None,
        state_projector: Optional[Callable] = None,
    ) -> dict[str, torch.Tensor]:
        """Fit a window of logged transitions.

        ``states`` has shape ``(B,H+1,D)`` and ``actions`` has ``(B,H,A)``.
        The autoregressive prediction is intentionally used instead of an
        independent one-step loss, matching SuperMimic's open-loop objective.
        ``mask`` may be ``(B,H)`` to ignore padded transitions. ``rewards``
        optionally supplies logged scalar rewards with shape ``(B,H)``.
        """
        if states.ndim != 3 or actions.ndim != 3:
            raise ValueError("states and actions must be rank-3 tensors")
        if states.shape[1] != actions.shape[1] + 1:
            raise ValueError("states must contain one more timestep than actions")
        if mask is None:
            mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
        if mask.shape != actions.shape[:2]:
            raise ValueError("mask must have shape (B,H)")
        if context is not None and context.ndim == 2:
            context = context.unsqueeze(1).expand(-1, actions.shape[1], -1)
        if context is not None and context.shape[:2] != actions.shape[:2]:
            raise ValueError("context must have shape (B,H,C) or (B,C)")
        current = states[:, 0]
        state_errors, reward_preds, term_logits = [], [], []
        for t in range(actions.shape[1]):
            result = self.forward(
                current, actions[:, t], None if context is None else context[:, t]
            )
            current = result["next_state"]
            if state_projector is not None:
                current = state_projector(current, t)
            state_errors.append(
                ((current - states[:, t + 1]) / self.state_norm.std()).square().mean(-1)
            )
            if "reward" in result:
                reward_preds.append(result["reward"])
            if "termination_logits" in result:
                term_logits.append(result["termination_logits"])
        error = torch.stack(state_errors, dim=1)
        valid = mask.to(error.dtype)
        state_valid = valid if dynamics_mask is None else valid * dynamics_mask.to(error.dtype)
        state_loss = (error * state_valid).sum() / state_valid.sum().clamp_min(1)
        losses = {"state_loss": state_loss, "loss": state_loss}
        if reward_preds and rewards is not None:
            reward_targets = rewards.to(reward_preds[0].dtype)
            if reward_targets.shape != (actions.shape[0], actions.shape[1]):
                reward_targets = reward_targets.reshape(actions.shape[0], actions.shape[1])
            reward_error = (torch.stack(reward_preds, dim=1) - reward_targets).square()
            reward_loss = (reward_error * valid).sum() / valid.sum().clamp_min(1)
            losses["reward_loss"] = reward_loss
            losses["loss"] = losses["loss"] + self.reward_loss_coef * reward_loss
        if term_logits and dones is not None:
            logits = torch.stack(term_logits, dim=1)
            if dones.shape != logits.shape:
                dones = dones.reshape_as(logits)
            term_loss = F.binary_cross_entropy_with_logits(logits, dones.float(), reduction="none")
            term_loss = (term_loss * valid).sum() / valid.sum().clamp_min(1)
            losses["termination_loss"] = term_loss
            losses["loss"] = losses["loss"] + term_loss
        losses["state_error_last"] = (error[:, -1] * state_valid[:, -1]).sum() / (
            state_valid[:, -1].sum().clamp_min(1)
        )
        return losses

    @torch.no_grad()
    def update_normalizers(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> None:
        """Update running statistics from real transitions only."""
        self.state_norm.update(states)
        self.action_norm.update(actions)
        if self.context_norm is not None and context is not None:
            self.context_norm.update(context)
        if next_states is not None:
            self.delta_norm.update(next_states - states)

    def imagine(
        self,
        policy: Callable[[torch.Tensor, Optional[torch.Tensor], int], torch.Tensor],
        state: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        horizon: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Generate a short differentiable rollout from a policy callback.

        WAM parameters are temporarily frozen while autograd remains enabled,
        so gradients flow from imagined rewards through ``state`` into the
        policy but do not update the dynamics model.  This is the key distinction
        from a model-free PPO update and follows the SuperMimic training loop.
        """
        if horizon < 1:
            raise ValueError("horizon must be positive")
        requires_grad = [p.requires_grad for p in self.parameters()]
        self.requires_grad_(False)
        try:
            current = state
            states, actions, rewards, terms = [current], [], [], []
            for t in range(horizon):
                ctx_t = context
                if context is not None and context.ndim == 3:
                    ctx_t = context[:, t]
                action = policy(current, ctx_t, t)
                result = self.forward(current, action, ctx_t)
                current = result["next_state"]
                states.append(current)
                actions.append(action)
                if "reward" in result:
                    rewards.append(result["reward"])
                if "termination_logits" in result:
                    terms.append(result["termination_logits"])
            output = {
                "states": torch.stack(states, dim=1),
                "actions": torch.stack(actions, dim=1),
            }
            if terms:
                output["termination_logits"] = torch.stack(terms, dim=1)
            if rewards:
                output["rewards"] = torch.stack(rewards, dim=1)
            return output
        finally:
            for parameter, flag in zip(self.parameters(), requires_grad):
                parameter.requires_grad_(flag)


# Backward-compatible name for experiments that imported the initial flat-state
# prototype.  The canonical FSQ-aware implementation lives in
# ``gear_sonic.trl.world_model``.
DifferentiableWorldModel = StateSpaceWorldModel

__all__ = ["StateSpaceWorldModel", "DifferentiableWorldModel", "RunningNorm", "WAMConfig"]
