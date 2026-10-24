"""Small replay and model-based actor-critic utilities for WAM.

This module is independent of TRL's PPO trainer. It can run alongside the
existing FSQ actor as an auxiliary update, or by itself for an offline warm
start. The actor interface is intentionally just ``policy(latent)`` (a second
``step`` argument is also accepted), so an MLP or FSQ policy can be plugged in.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .wam import DifferentiableWorldModel


def _clone_item(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_item(item) for key, item in value.items()}
    return value


def _stack_items(values: list[Any], device: torch.device | None = None) -> Any:
    first = values[0]
    if isinstance(first, Mapping):
        return {key: _stack_items([value[key] for value in values], device) for key in first}
    if isinstance(first, Tensor):
        result = torch.stack(values)
        return result.to(device=device) if device is not None else result
    result = torch.as_tensor(values)
    return result.to(device=device) if device is not None else result


class TransitionReplayBuffer:
    """A CPU ring buffer that keeps nested multimodal observations intact."""

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._storage: list[dict[str, Any]] = []
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(
        self,
        obs: Any,
        action: Any,
        reward: float | Tensor,
        next_obs: Any,
        done: bool | Tensor = False,
        **extras: Any,
    ) -> None:
        transition = {
            "obs": _clone_item(obs),
            "action": _clone_item(action),
            "reward": _clone_item(reward),
            "next_obs": _clone_item(next_obs),
            "done": _clone_item(done),
            **{key: _clone_item(value) for key, value in extras.items()},
        }
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._cursor] = transition
        self._cursor = (self._cursor + 1) % self.capacity

    def extend(self, transitions: Iterable[Mapping[str, Any]]) -> None:
        for transition in transitions:
            self.add(
                transition["obs"],
                transition.get("action", transition.get("actions")),
                transition.get("reward", transition.get("rewards", 0.0)),
                transition.get("next_obs", transition.get("next_observation")),
                transition.get("done", transition.get("dones", False)),
                **{
                    key: value
                    for key, value in transition.items()
                    if key not in {
                        "obs",
                        "action",
                        "actions",
                        "reward",
                        "rewards",
                        "next_obs",
                        "next_observation",
                        "done",
                        "dones",
                    }
                },
            )

    def sample(self, batch_size: int, device: torch.device | str | None = None) -> dict[str, Any]:
        if not self._storage:
            raise ValueError("cannot sample from an empty replay buffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        count = min(int(batch_size), len(self._storage))
        selected = random.sample(self._storage, count)
        target_device = torch.device(device) if device is not None else None
        return {
            key: _stack_items([item[key] for item in selected], target_device)
            for key in selected[0]
        }


class _ValueMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: Tensor) -> Tensor:
        return self.net(latent).squeeze(-1)


class ModelBasedActorCriticTrainer:
    """Train WAM, an actor, and a critic from replay and imagined rollouts.

    ``update`` first performs a one-step world-model update (when a replay batch
    is available), then backpropagates a BPTT actor objective through WAM. The
    critic target uses GAE over imagined rewards. No simulator-specific or TRL
    internals are required.
    """

    def __init__(
        self,
        world_model: DifferentiableWorldModel,
        actor: nn.Module | Callable[..., Tensor],
        critic: nn.Module | Callable[[Tensor], Tensor] | None = None,
        *,
        replay_buffer: TransitionReplayBuffer | None = None,
        world_model_optimizer: torch.optim.Optimizer | None = None,
        actor_optimizer: torch.optim.Optimizer | None = None,
        critic_optimizer: torch.optim.Optimizer | None = None,
        world_model_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        horizon: int = 8,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        critic_coef: float = 0.5,
        uncertainty_coef: float = 0.0,
    ) -> None:
        self.world_model = world_model
        self.actor = actor
        if critic is None:
            reference_parameter = next(world_model.parameters())
            self.critic = _ValueMLP(world_model.latent_dim).to(
                device=reference_parameter.device, dtype=reference_parameter.dtype
            )
        else:
            self.critic = critic
        self.replay_buffer = replay_buffer
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.critic_coef = float(critic_coef)
        self.uncertainty_coef = float(uncertainty_coef)
        self.world_model_optimizer = world_model_optimizer or torch.optim.Adam(
            world_model.parameters(), lr=world_model_lr
        )
        self.actor_optimizer = (
            actor_optimizer
            if actor_optimizer is not None or not isinstance(actor, nn.Module)
            else torch.optim.Adam(actor.parameters(), lr=actor_lr)
        )
        self.critic_optimizer = critic_optimizer or (
            torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
            if isinstance(self.critic, nn.Module)
            else None
        )

    @staticmethod
    def _call_actor(actor: Callable[..., Tensor], latent: Tensor, step: int) -> Tensor:
        try:
            action = actor(latent, step)
        except TypeError:
            action = actor(latent)
        if isinstance(action, Mapping):
            action = action.get("action", action.get("actions"))
        if action is None:
            raise ValueError("actor must return an action tensor")
        return action

    @staticmethod
    def _call_critic(critic: Callable[[Tensor], Tensor], states: Tensor) -> Tensor:
        original_shape = states.shape[:-1]
        values = critic(states.reshape(-1, states.shape[-1]))
        if isinstance(values, Mapping):
            values = values.get("value", values.get("values"))
        if values is None:
            raise ValueError("critic must return a value tensor")
        values = values.reshape(*original_shape, -1)
        return values[..., 0]

    def world_model_update(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        self.world_model_optimizer.zero_grad(set_to_none=True)
        actions = batch.get("action", batch.get("actions"))
        losses = (
            self.world_model.window_loss(batch)
            if isinstance(actions, Tensor) and actions.ndim == 3
            else self.world_model.loss(batch)
        )
        losses["loss"].backward()
        self.world_model_optimizer.step()
        return {key: value.detach() for key, value in losses.items()}

    def imagined_loss(self, start: Any) -> dict[str, Tensor]:
        # WAM.imagine freezes all model parameters, including lazy encoders,
        # while retaining its input/action Jacobian for the actor update.
        rollout = self.world_model.imagine(
            start,
            lambda latent, step: self._call_actor(self.actor, latent, step),
            horizon=self.horizon,
            discount=self.gamma,
            detach_state=False,
        )
        states = rollout["states"]
        next_states = rollout["next_states"]
        # Critic regression must not send gradients into the actor/WAM graph.
        values = self._call_critic(self.critic, states.detach())
        next_values = self._call_critic(self.critic, next_states.detach())
        rewards = rollout["rewards"]
        continuations = 1.0 - rollout["terminations"]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[..., 0])
        for idx in range(rewards.shape[-1] - 1, -1, -1):
            delta = (
                rewards[..., idx]
                + self.gamma * continuations[..., idx] * next_values[..., idx]
                - values[..., idx]
            )
            gae = delta + self.gamma * self.gae_lambda * continuations[..., idx] * gae
            advantages[..., idx] = gae
        returns = advantages + values
        critic_loss = F.mse_loss(values, returns.detach())
        # Eq. (2): bootstrap the imagined return with the critic at the final
        # state. Freeze critic weights for this actor objective, but keep the
        # derivative with respect to the final latent state.
        critic_flags = None
        if isinstance(self.critic, nn.Module):
            critic_flags = [parameter.requires_grad for parameter in self.critic.parameters()]
            self.critic.requires_grad_(False)
        try:
            final_value = self._call_critic(self.critic, rollout["final_latent"])
        finally:
            if critic_flags is not None:
                for parameter, flag in zip(self.critic.parameters(), critic_flags):
                    parameter.requires_grad_(flag)
        terminations = rollout["terminations"]
        survival_before = torch.cat(
            [
                torch.ones_like(terminations[..., :1]),
                (1.0 - terminations).cumprod(dim=-1)[..., :-1],
            ],
            dim=-1,
        )
        survival_final = (1.0 - terminations).prod(dim=-1)
        discounts = self.gamma ** torch.arange(
            rewards.shape[-1], device=rewards.device, dtype=rewards.dtype
        )
        actor_return = (survival_before * rewards * discounts).sum(dim=-1)
        actor_return = actor_return + self.gamma**self.horizon * survival_final * final_value
        if self.uncertainty_coef:
            actor_return = actor_return - self.uncertainty_coef * (
                survival_before * rollout["uncertainties"] * discounts
            ).sum(dim=-1)
        actor_loss = -actor_return.mean()
        return {
            "loss": actor_loss + self.critic_coef * critic_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "advantages": advantages.detach(),
            "returns": returns.detach(),
            "rollout": rollout,
        }

    def update(
        self,
        start: Any,
        *,
        replay_batch: Mapping[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """Run one optional replay update and one imagined actor-critic update."""

        metrics: dict[str, Tensor] = {}
        if replay_batch is not None:
            metrics.update(
                {
                    f"world_model/{key}": value
                    for key, value in self.world_model_update(replay_batch).items()
                }
            )
        elif self.replay_buffer is not None and len(self.replay_buffer):
            device = next(self.world_model.parameters()).device
            metrics.update(
                {
                    f"world_model/{key}": value
                    for key, value in self.world_model_update(
                        self.replay_buffer.sample(min(256, len(self.replay_buffer)), device)
                    ).items()
                }
            )

        imagined = self.imagined_loss(start)
        if self.actor_optimizer is not None:
            self.actor_optimizer.zero_grad(set_to_none=True)
            imagined["actor_loss"].backward(retain_graph=self.critic_optimizer is not None)
            self.actor_optimizer.step()
        if self.critic_optimizer is not None:
            self.critic_optimizer.zero_grad(set_to_none=True)
            imagined["critic_loss"].backward()
            self.critic_optimizer.step()
        metrics.update(
            {
                "actor/loss": imagined["actor_loss"].detach(),
                "critic/loss": imagined["critic_loss"].detach(),
                "actor/mean_return": (-imagined["actor_loss"]).detach(),
                "actor/mean_uncertainty": imagined["rollout"]["uncertainties"].detach().mean(),
            }
        )
        return metrics
