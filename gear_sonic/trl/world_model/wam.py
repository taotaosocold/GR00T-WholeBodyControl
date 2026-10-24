"""A small, differentiable world model (WAM) for whole-body control.

The policy in this repository already turns multiple modalities into FSQ
tokens.  WAM deliberately sits *after* that interface: it accepts an FSQ token
tensor (or the complete observation dictionary), predicts latent dynamics and
reward in one batched pass, and can be used for imagined actor updates or
latent MPC.  Keeping the adapter independent from ``UniversalTokenModule``
means old checkpoints and non-FSQ observations continue to work unchanged.

The implementation is intentionally lightweight.  It is useful as a trainable
baseline and as an integration point for a larger SuperMimic-style dynamics
model; callers can replace the MLP blocks without changing the rollout API.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Sequential:
    """Build a compact residual-free MLP used by the dynamics heads."""

    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
    for _ in range(max(0, depth - 1)):
        layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class FSQObservationAdapter(nn.Module):
    """Convert tensor or multimodal observations into a flat feature tensor.

    ``UniversalTokenModule`` emits continuous FSQ codes with shape ``(...,
    num_tokens, token_dim)``.  Passing those codes under ``token_key`` (default
    ``"fsq_tokens"``) preserves the existing quantizer semantics.  A mapping
    can also contain proprioception, command, teleoperation, or vision values;
    all selected values are concatenated in a stable order.

    Spatial image dimensions are average-pooled before flattening.  This keeps
    WAM cheap enough for imagined rollouts while leaving the original image
    encoder untouched in the actor.
    """

    DEFAULT_EXCLUDED_KEYS = frozenset(
        {
            "encoder_index",
            "tokenizer_mask",
            "frame_mask",
            "attention_mask",
            "done",
            "dones",
            "reward",
            "rewards",
            "action",
            "actions",
        }
    )

    def __init__(
        self,
        feature_order: Sequence[str] | None = None,
        token_key: str = "fsq_tokens",
        exclude_keys: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.feature_order = tuple(feature_order or ())
        self.token_key = token_key
        self.exclude_keys = frozenset(exclude_keys or self.DEFAULT_EXCLUDED_KEYS)

    @staticmethod
    def _as_tensor(value: Tensor | object, reference: Tensor | None = None) -> Tensor:
        if isinstance(value, Tensor):
            return value
        return torch.as_tensor(value, device=reference.device if reference is not None else None)

    @staticmethod
    def _flatten(value: Tensor, *, preserve_token_axes: bool = False) -> Tensor:
        # Scalars and vectors are treated as one feature dimension.
        if value.ndim == 0:
            return value.reshape(1, 1)
        if value.ndim == 1:
            return value.unsqueeze(-1)
        # FSQ tokens use [B,T,num_tokens,token_dim]. Keep both token axes and
        # flatten them; averaging would destroy the discrete interface.
        if preserve_token_axes and value.ndim >= 3:
            return value.reshape(*value.shape[:-2], -1)
        # For image-like values, average over spatial dimensions. This also
        # handles [B,C,H,W] and [B,T,C,H,W] without assumptions about T.
        if value.ndim >= 4:
            value = value.mean(dim=(-2, -1))
        return value.reshape(*value.shape[:-1], -1)

    def _mapping_features(self, obs: Mapping[str, object]) -> list[Tensor]:
        keys: list[str] = []
        if self.token_key in obs:
            keys.append(self.token_key)
        if self.feature_order:
            keys.extend(key for key in self.feature_order if key in obs and key not in keys)
        else:
            keys.extend(
                key for key in sorted(obs) if key not in keys and key not in self.exclude_keys
            )
        if not keys:
            raise ValueError("Observation mapping has no numeric features for WAM")

        features: list[Tensor] = []
        reference: Tensor | None = None
        for key in keys:
            value = self._as_tensor(obs[key], reference)
            if not (value.is_floating_point() or value.is_complex()):
                value = value.float()
            # FSQ tokens carry a token axis and a code axis.  Preserve both
            # axes (and any batch/temporal prefix) instead of treating them as
            # image dimensions and averaging them away.
            if key == self.token_key and value.ndim >= 3:
                value = value.reshape(*value.shape[:-2], -1)
            else:
                value = self._flatten(value, preserve_token_axes=key == self.token_key)
            reference = value
            features.append(value)
        return features

    def forward(self, obs: Tensor | Mapping[str, object]) -> Tensor:
        if isinstance(obs, Mapping):
            features = self._mapping_features(obs)
            # Broadcast a single-frame modality to a temporal modality when
            # possible (e.g. proprioception [B,D] + tokens [B,T,K]).
            ndim = max(feature.ndim for feature in features)
            aligned = []
            for feature in features:
                while feature.ndim < ndim:
                    feature = feature.unsqueeze(-2)
                aligned.append(feature)
            # A current proprioceptive vector is commonly paired with a
            # temporal FSQ token window.  Expand singleton temporal axes
            # before concatenating (``[B,1,D]`` -> ``[B,T,D]``).  Without
            # this step the adapter rejects perfectly valid multimodal
            # observations even though the modalities share a batch.
            if ndim >= 3:
                temporal = max(item.shape[-2] for item in aligned)
                aligned = [
                    item.expand(*item.shape[:-2], temporal, item.shape[-1])
                    if item.shape[-2] == 1 and temporal > 1
                    else item
                    for item in aligned
                ]
            # Broadcast frame-invariant modalities (e.g. actor_obs) across the
            # token/history axis while keeping the view differentiable.
            target_dims = [
                max(feature.shape[axis] for feature in aligned)
                for axis in range(1, ndim - 1)
            ]
            expanded = []
            for feature in aligned:
                shape = list(feature.shape)
                for offset, axis in enumerate(range(1, ndim - 1)):
                    if shape[axis] == 1:
                        shape[axis] = target_dims[offset]
                expanded.append(feature.expand(*shape))
            aligned = expanded
            try:
                return torch.cat(aligned, dim=-1)
            except RuntimeError as exc:
                raise ValueError(
                    "Multimodal WAM observations must share batch/temporal dimensions"
                ) from exc
        value = self._as_tensor(obs)
        if not (value.is_floating_point() or value.is_complex()):
            value = value.float()
        return self._flatten(value)


class DifferentiableWorldModel(nn.Module):
    """Predict latent state transitions, rewards, and termination signals.

    Args:
        obs_dim: Flat observation dimension.  ``None`` uses ``LazyLinear`` and
            is convenient when Hydra resolves dimensions at runtime.
        action_dim: Number of action dimensions.
        latent_dim: Size of WAM's compact state representation.
        hidden_dim: Width of the dynamics and prediction heads.
        depth: Number of hidden MLP blocks.
        adapter: Optional ``FSQObservationAdapter``.  The adapter is an
            identity for an already-flat tensor and accepts dictionaries for
            multimodal data.

    ``forward`` returns a dictionary so callers can add heads (contact,
    imitation, uncertainty) without changing downstream code.  The ``loss``
    method is suitable for replay-buffer training and supports sequence-shaped
    tensors ``[batch, time, feature]``.
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 1,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        depth: int = 2,
        adapter: FSQObservationAdapter | None = None,
        latent_loss_weight: float = 1.0,
        reward_loss_weight: float = 1.0,
        termination_loss_weight: float = 0.1,
        uncertainty_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or latent_dim <= 0:
            raise ValueError("action_dim and latent_dim must be positive")
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        if isinstance(adapter, Mapping):
            adapter_cfg = dict(adapter)
            target = adapter_cfg.pop("_target_", None)
            adapter_cfg.pop("_recursive_", None)
            if target is None:
                self.adapter = FSQObservationAdapter(**adapter_cfg)
            else:
                module_name, class_name = target.rsplit(".", 1)
                adapter_type = getattr(importlib.import_module(module_name), class_name)
                self.adapter = adapter_type(**adapter_cfg)
        else:
            self.adapter = adapter or FSQObservationAdapter()
        self.latent_loss_weight = float(latent_loss_weight)
        self.reward_loss_weight = float(reward_loss_weight)
        self.termination_loss_weight = float(termination_loss_weight)
        self.uncertainty_loss_weight = float(uncertainty_loss_weight)
        self._compiled_step = None

        if obs_dim is None:
            self.obs_encoder = nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
        else:
            self.obs_encoder = _mlp(int(obs_dim), hidden_dim, latent_dim, depth)
        self.dynamics = _mlp(latent_dim + action_dim, hidden_dim, latent_dim, depth)
        self.reward_head = _mlp(latent_dim + action_dim, hidden_dim, 1, max(1, depth - 1))
        self.termination_head = _mlp(latent_dim + action_dim, hidden_dim, 1, max(1, depth - 1))
        self.uncertainty_head = _mlp(latent_dim + action_dim, hidden_dim, 1, max(1, depth - 1))

    def encode_observation(self, obs: Tensor | Mapping[str, object]) -> Tensor:
        features = self.adapter(obs)
        return self.obs_encoder(features)

    def enable_compile(self, mode: str = "reduce-overhead"):
        """Compile the one-step transition for high-throughput imagination.

        Compilation is opt-in because the first call pays a graph-capture cost
        and some deployment environments do not ship TorchInductor.  The
        returned module is the same WAM, so callers can write
        ``wam.enable_compile(); trainer.update(...)``.
        """
        compiler = getattr(torch, "compile", None)
        if compiler is None:
            return self
        self._compiled_step = compiler(self.step, mode=mode)
        return self

    def _transition(self, latent: Tensor, action: Tensor) -> dict[str, Tensor]:
        step_fn = self._compiled_step or self.step
        return step_fn(latent, action)

    def step(self, latent: Tensor, action: Tensor) -> dict[str, Tensor]:
        if action.ndim == 0:
            action = action.reshape(1, 1)
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action dimension {self.action_dim}, got {action.shape[-1]}"
            )
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, got {latent.shape[-1]}"
            )
        model_input = torch.cat((latent, action), dim=-1)
        delta = self.dynamics(model_input)
        next_latent = latent + delta
        reward = self.reward_head(model_input).squeeze(-1)
        termination_logit = self.termination_head(model_input).squeeze(-1)
        # A positive prediction-error estimate, trained against detached
        # normalized latent residuals by loss()/window_loss(). It is a model
        # error proxy, rather than a calibrated epistemic uncertainty.
        uncertainty = F.softplus(self.uncertainty_head(model_input).squeeze(-1))
        return {
            "next_latent": next_latent,
            "reward": reward,
            "termination_logit": termination_logit,
            "termination": torch.sigmoid(termination_logit),
            "uncertainty": uncertainty,
        }

    def forward(
        self,
        obs: Tensor | Mapping[str, object],
        action: Tensor,
        *,
        return_latent: bool = False,
    ) -> dict[str, Tensor]:
        latent = self.encode_observation(obs)
        output = self.step(latent, action)
        if return_latent:
            output["latent"] = latent
        return output

    @staticmethod
    def _get(batch: Mapping[str, object], *keys: str) -> object:
        for key in keys:
            if key in batch:
                return batch[key]
        raise KeyError(f"Batch is missing one of {keys}")

    @staticmethod
    def _scalar_target(value: object, prediction: Tensor) -> Tensor:
        """Match scalar labels to predictions without squeezing batch=1."""
        target = torch.as_tensor(value, device=prediction.device, dtype=prediction.dtype)
        if target.numel() != prediction.numel():
            raise ValueError("Scalar labels must match prediction batch/temporal dimensions")
        return target.reshape_as(prediction)

    @staticmethod
    def _prediction_error_target(prediction: Tensor, target: Tensor) -> Tensor:
        # Normalize by target energy so the error head is insensitive to a
        # uniform latent rescaling. Clamp the scale for near-zero targets.
        target = target.detach()
        error = (prediction.detach() - target).square().mean(dim=-1)
        scale = target.square().mean(dim=-1).clamp_min(1e-3)
        return error / scale

    def loss(self, batch: Mapping[str, object]) -> dict[str, Tensor]:
        """Compute one-step WAM losses from replay transitions.

        Required keys are ``obs``, ``action``, ``next_obs`` and ``reward``;
        ``done``/``dones`` is optional and defaults to zero.  ``obs`` and
        ``next_obs`` may each be dictionaries containing FSQ tokens and other
        modalities.
        """

        obs = self._get(batch, "obs", "observation")
        next_obs = self._get(batch, "next_obs", "next_observation")
        action = self._get(batch, "action", "actions")
        reward = self._get(batch, "reward", "rewards")
        action = action if isinstance(action, Tensor) else torch.as_tensor(action)
        reward = reward if isinstance(reward, Tensor) else torch.as_tensor(reward)
        pred = self(obs, action)
        target_latent = self.encode_observation(next_obs)
        if target_latent.shape != pred["next_latent"].shape:
            raise ValueError("obs and next_obs must have matching batch/temporal shapes")
        latent_loss = F.smooth_l1_loss(pred["next_latent"], target_latent.detach())
        reward_loss = F.mse_loss(
            pred["reward"], self._scalar_target(reward, pred["reward"])
        )
        uncertainty_loss = F.mse_loss(
            pred["uncertainty"],
            self._prediction_error_target(pred["next_latent"], target_latent),
        )

        done_obj = batch.get("done", batch.get("dones"))
        if done_obj is None:
            termination_loss = pred["termination_logit"].new_zeros(())
        else:
            termination_loss = F.binary_cross_entropy_with_logits(
                pred["termination_logit"],
                self._scalar_target(done_obj, pred["termination_logit"]),
            )
        total = (
            self.latent_loss_weight * latent_loss
            + self.reward_loss_weight * reward_loss
            + self.termination_loss_weight * termination_loss
            + self.uncertainty_loss_weight * uncertainty_loss
        )
        return {
            "loss": total,
            "latent_loss": latent_loss,
            "reward_loss": reward_loss,
            "termination_loss": termination_loss,
            "uncertainty_loss": uncertainty_loss,
        }

    def window_loss(self, batch: Mapping[str, object]) -> dict[str, Tensor]:
        """Fit an autoregressive replay window, matching SuperMimic Eq. (1).

        Flat tensor observations use ``[B, K+1, D]``.  Mapping observations
        use the same leading dimensions for every value.  The first frame
        seeds the model and subsequent frames are targets of the model's own
        predictions, exposing multi-step drift instead of only one-step error.
        """
        obs = self._get(batch, "obs", "observation")
        actions = self._get(batch, "action", "actions")
        rewards = batch.get("reward", batch.get("rewards"))
        dones = batch.get("done", batch.get("dones"))
        if not isinstance(actions, Tensor) or actions.ndim != 3:
            raise ValueError("window_loss expects actions with shape [B,K,A]")
        if isinstance(obs, Mapping):
            first = {key: value[:, 0] for key, value in obs.items()}
            targets = [
                {key: value[:, t + 1] for key, value in obs.items()}
                for t in range(actions.shape[1])
            ]
        else:
            if not isinstance(obs, Tensor) or obs.ndim != 3:
                raise ValueError("tensor obs must have shape [B,K+1,D]")
            first = obs[:, 0]
            targets = [obs[:, t + 1] for t in range(actions.shape[1])]
        latent = self.encode_observation(first)
        state_errors, reward_errors, term_errors, uncertainty_errors = [], [], [], []
        for t, target_obs in enumerate(targets):
            transition = self._transition(latent, actions[:, t])
            target_latent = self.encode_observation(target_obs).detach()
            state_errors.append(
                F.smooth_l1_loss(
                    transition["next_latent"], target_latent, reduction="none"
                ).mean(-1)
            )
            uncertainty_errors.append(
                F.mse_loss(
                    transition["uncertainty"],
                    self._prediction_error_target(transition["next_latent"], target_latent),
                    reduction="none",
                )
            )
            if rewards is not None:
                reward_t = (
                    rewards[:, t]
                    if isinstance(rewards, Tensor) and rewards.ndim > 1
                    else rewards
                )
                reward_errors.append(
                    F.mse_loss(
                        transition["reward"],
                        self._scalar_target(reward_t, transition["reward"]),
                        reduction="none",
                    )
                )
            if dones is not None:
                done_t = dones[:, t] if isinstance(dones, Tensor) and dones.ndim > 1 else dones
                term_errors.append(
                    F.binary_cross_entropy_with_logits(
                        transition["termination_logit"],
                        self._scalar_target(done_t, transition["termination_logit"]),
                        reduction="none",
                    )
                )
            latent = transition["next_latent"]
        latent_loss = torch.stack(state_errors, dim=-1).mean()
        reward_loss = (
            torch.stack(reward_errors, dim=-1).mean()
            if reward_errors
            else latent_loss.new_zeros(())
        )
        term_loss = (
            torch.stack(term_errors, dim=-1).mean()
            if term_errors
            else latent_loss.new_zeros(())
        )
        uncertainty_loss = torch.stack(uncertainty_errors, dim=-1).mean()
        total = (
            self.latent_loss_weight * latent_loss
            + self.reward_loss_weight * reward_loss
            + self.termination_loss_weight * term_loss
            + self.uncertainty_loss_weight * uncertainty_loss
        )
        return {
            "loss": total,
            "latent_loss": latent_loss,
            "reward_loss": reward_loss,
            "termination_loss": term_loss,
            "uncertainty_loss": uncertainty_loss,
        }

    def imagine(
        self,
        start: Tensor | Mapping[str, object],
        policy: Callable[..., Tensor],
        horizon: int = 8,
        discount: float = 0.99,
        *,
        detach_state: bool = False,
    ) -> dict[str, Tensor]:
        """Roll out a policy entirely in latent space.

        The model and policy are evaluated in parallel over the batch.  The
        function is differentiable with respect to policy parameters, allowing
        actor updates from imagined returns without stepping IsaacLab.  A
        policy may accept either ``(latent)`` or ``(latent, step)``.
        """

        if horizon <= 0:
            raise ValueError("horizon must be positive")
        states, next_states, actions, rewards, terminations, uncertainties = [], [], [], [], [], []
        # Freeze WAM parameters during actor imagination while retaining the
        # autograd path through actions. This prevents the actor update from
        # accidentally training the dynamics model a second time.
        requires_grad = [parameter.requires_grad for parameter in self.parameters()]
        # Attribute assignment also works on UninitializedParameter; Tensor's
        # in-place requires_grad_() rejects a not-yet-materialized LazyLinear.
        # Freeze before encoding the start so its encoder cannot collect
        # gradients from the actor objective either.
        for parameter in self.parameters():
            parameter.requires_grad = False
        try:
            latent = (
                start
                if isinstance(start, Tensor) and start.shape[-1] == self.latent_dim
                else self.encode_observation(start)
            )
            for step_idx in range(horizon):
                try:
                    action = policy(latent, step_idx)
                except TypeError:
                    action = policy(latent)
                if isinstance(action, Mapping):
                    action = action.get("action", action.get("actions"))
                if action is None:
                    raise ValueError("Imagined policy must return a Tensor or action mapping")
                transition = self._transition(latent, action)
                states.append(latent)
                actions.append(action)
                rewards.append(transition["reward"])
                terminations.append(transition["termination"])
                uncertainties.append(transition["uncertainty"])
                latent = transition["next_latent"]
                next_states.append(latent)
                if detach_state:
                    latent = latent.detach()
            return {
                "states": torch.stack(states, dim=-2),
                "next_states": torch.stack(next_states, dim=-2),
                "actions": torch.stack(actions, dim=-2),
                "rewards": torch.stack(rewards, dim=-1),
                "terminations": torch.stack(terminations, dim=-1),
                "uncertainties": torch.stack(uncertainties, dim=-1),
                "discounted_return": self.discounted_return(
                    torch.stack(rewards, dim=-1), torch.stack(terminations, dim=-1), discount
                ),
                "final_latent": latent,
            }
        finally:
            for parameter, flag in zip(self.parameters(), requires_grad):
                parameter.requires_grad = flag

    @staticmethod
    def discounted_return(rewards: Tensor, terminations: Tensor, discount: float = 0.99) -> Tensor:
        """Compute continuation-aware discounted returns over the last axis."""

        value = torch.zeros_like(rewards[..., 0])
        returns: list[Tensor] = []
        for idx in range(rewards.shape[-1] - 1, -1, -1):
            value = rewards[..., idx] + discount * (1.0 - terminations[..., idx]) * value
            returns.append(value)
        return torch.stack(returns[::-1], dim=-1)

    def imagined_actor_loss(
        self,
        start: Tensor | Mapping[str, object],
        policy: Callable[..., Tensor],
        horizon: int = 8,
        discount: float = 0.99,
        uncertainty_coef: float = 0.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return a policy loss from model-generated trajectories."""

        rollout = self.imagine(start, policy, horizon, discount)
        returns = rollout["discounted_return"]
        if uncertainty_coef:
            returns = returns - uncertainty_coef * rollout["uncertainties"]
        loss = -returns[..., 0].mean()
        return loss, rollout


class LatentMPC:
    """Vectorized cross-entropy planner over a trained WAM.

    Planning in latent space avoids expensive simulator calls during action
    selection.  A policy prior may be supplied (for example the FSQ actor's
    mean action); otherwise the planner starts from zero actions.
    """

    def __init__(
        self,
        world_model: DifferentiableWorldModel,
        horizon: int = 8,
        population: int = 256,
        elite_count: int = 32,
        iterations: int = 3,
        action_std: float = 0.3,
        action_low: float | Tensor | None = None,
        action_high: float | Tensor | None = None,
    ) -> None:
        if horizon <= 0 or population <= 0 or elite_count <= 0:
            raise ValueError("horizon, population, and elite_count must be positive")
        if elite_count > population:
            raise ValueError("elite_count cannot exceed population")
        self.world_model = world_model
        self.horizon = horizon
        self.population = population
        self.elite_count = elite_count
        self.iterations = max(1, iterations)
        self.action_std = float(action_std)
        self.action_low = action_low
        self.action_high = action_high

    def _bounds(self, reference: Tensor) -> tuple[Tensor | None, Tensor | None]:
        low = self.action_low
        high = self.action_high
        if low is not None:
            low = torch.as_tensor(low, device=reference.device, dtype=reference.dtype)
        if high is not None:
            high = torch.as_tensor(high, device=reference.device, dtype=reference.dtype)
        return low, high

    @torch.no_grad()
    def __call__(
        self,
        start: Tensor | Mapping[str, object],
        policy_prior: Callable[[Tensor], Tensor] | None = None,
        discount: float = 0.99,
    ) -> Tensor:
        latent = (
            start
            if isinstance(start, Tensor) and start.shape[-1] == self.world_model.latent_dim
            else self.world_model.encode_observation(start)
        )
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        batch = latent.shape[0]
        device, dtype = latent.device, latent.dtype
        action_shape = (batch, self.population, self.horizon, self.world_model.action_dim)
        low, high = self._bounds(latent)
        mean = torch.zeros(
            (batch, self.horizon, self.world_model.action_dim), device=device, dtype=dtype
        )
        if policy_prior is not None:
            prior = policy_prior(latent)
            if isinstance(prior, Mapping):
                prior = prior.get("action", prior.get("actions"))
            if prior is not None:
                mean[:] = prior.unsqueeze(-2) if prior.ndim == 2 else prior
        std = torch.full_like(mean, self.action_std)

        for _ in range(self.iterations):
            samples = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                action_shape, device=device, dtype=dtype
            )
            if low is not None or high is not None:
                samples = torch.maximum(samples, low) if low is not None else samples
                samples = torch.minimum(samples, high) if high is not None else samples
            state = latent.unsqueeze(1).expand(batch, self.population, -1)
            rewards: list[Tensor] = []
            terminations: list[Tensor] = []
            for step_idx in range(self.horizon):
                transition = self.world_model._transition(state, samples[..., step_idx, :])
                rewards.append(transition["reward"])
                terminations.append(transition["termination"])
                state = transition["next_latent"]
            reward_tensor = torch.stack(rewards, dim=-1)
            term_tensor = torch.stack(terminations, dim=-1)
            scores = self.world_model.discounted_return(
                reward_tensor, term_tensor, discount
            )[..., 0]
            elite_idx = scores.topk(self.elite_count, dim=1).indices
            elite = samples.gather(
                1,
                elite_idx[..., None, None].expand(
                    -1, -1, self.horizon, self.world_model.action_dim
                ),
            )
            mean = elite.mean(dim=1)
            std = elite.std(dim=1, unbiased=False).clamp_min(1e-3)
        return mean[:, 0]


# WAM is the short name used in the SuperMimic notes.
WAM = DifferentiableWorldModel

