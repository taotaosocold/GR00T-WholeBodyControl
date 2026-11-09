"""Differentiable world model used by the SuperMimic-style Sonic trainer.

The Sonic policy remains responsible for multimodal goals: G1 retargeted,
teleoperation, and SMPL goal observations are encoded by their original
encoders, quantized by FSQ, and decoded into actions. The WAM dynamics model
is deliberately goal-agnostic. It predicts the robot-side state transition
from ``(state, action, context)``. In physical mode, reward and termination
are computed analytically from predicted and reference robot states. Raw goals
and FSQ tokens stay on the actor/alignment path.

The canonical WAM path uses the 71-dimensional physical state from
SuperMimic. A compatibility ``flat`` mode remains available for existing
unit-test integrations. In either case, goal tensors are never part of the
dynamics input and are supplied separately during fitting and imagined
rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F

from gear_sonic.utils.wam_context import normalize_supermimic_context


NUM_G1_JOINTS = 29
PHYSICAL_STATE_DIM = 3 + 4 + 3 + 3 + NUM_G1_JOINTS + NUM_G1_JOINTS
PHYSICAL_DELTA_DIM = 3 + 3 + 3 + 3 + NUM_G1_JOINTS + NUM_G1_JOINTS


def _quat_to_mat(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to matrices, matching SuperMimic's state layout."""
    quaternion = F.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ), dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices using their largest quaternion component.

    The trace-only formula divides by the scalar quaternion component, which
    vanishes at half-turns. Selecting the largest of four equivalent formulas
    avoids that singularity. Clamp inactive square-root branches as well so
    their derivatives cannot produce NaNs during autograd.
    """
    m00, m01, m02 = matrix[..., 0, :].unbind(-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(-1)
    diagonal = torch.stack((
        1 + m00 + m11 + m22,
        1 + m00 - m11 - m22,
        1 - m00 + m11 - m22,
        1 - m00 - m11 + m22,
    ), dim=-1)
    magnitudes = torch.sqrt(diagonal.clamp_min(torch.finfo(matrix.dtype).eps))
    candidates = torch.stack((
        torch.stack((diagonal[..., 0], m21 - m12, m02 - m20, m10 - m01), -1),
        torch.stack((m21 - m12, diagonal[..., 1], m01 + m10, m02 + m20), -1),
        torch.stack((m02 - m20, m01 + m10, diagonal[..., 2], m12 + m21), -1),
        torch.stack((m10 - m01, m02 + m20, m12 + m21, diagonal[..., 3]), -1),
    ), dim=-2) / (2 * magnitudes[..., :, None])
    largest = diagonal.argmax(-1)
    quaternion = candidates.gather(
        -2, largest[..., None, None].expand(largest.shape + (1, 4)),
    ).squeeze(-2)
    quaternion = F.normalize(quaternion, dim=-1)
    return torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternions, retaining leading dimensions."""
    left_w, left_v = left[..., :1], left[..., 1:]
    right_w, right_v = right[..., :1], right[..., 1:]
    return torch.cat((
        left_w * right_w - (left_v * right_v).sum(-1, keepdim=True),
        left_w * right_v + right_w * left_v + torch.cross(left_v, right_v, dim=-1),
    ), dim=-1)


def _rotvec_to_quat(vector: torch.Tensor) -> torch.Tensor:
    """Quaternion exponential with the analytic sinc limit at zero angle."""
    half_angle = 0.5 * torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    scale = 0.5 * torch.sinc(half_angle / torch.pi)
    return torch.cat((half_angle.cos(), scale * vector), dim=-1)


def _quat_to_rotvec(quaternion: torch.Tensor) -> torch.Tensor:
    """Shortest rotation vector, stable at zero and finite at half-turns.

    The shortest-axis convention necessarily changes branch at exactly pi.
    Both one-sided derivatives remain finite; no zero skew axis is divided by
    its norm at a half-turn, and the small-angle limit retains its derivative.
    """
    quaternion = F.normalize(quaternion, dim=-1)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sine, quaternion[..., :1])
    sine_squared = sine.square()
    scale = torch.where(
        sine < 1e-4,
        2 + sine_squared / 3 + 3 * sine_squared.square() / 20,
        angle / sine.clamp_min(1e-4),
    )
    return vector * scale


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        vector.shape[:-1] + (3, 3)
    )


def _so3_exp(vector: torch.Tensor) -> torch.Tensor:
    theta2 = vector.square().sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    a = torch.where(theta2 < 1e-8, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(
        theta2 < 1e-8,
        0.5 - theta2 / 24,
        (1 - torch.cos(theta)) / (theta2 + 1e-12),
    )
    skew = _skew(vector)
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype).expand(skew.shape)
    return eye + a[..., None] * skew + b[..., None] * (skew @ skew)


def _yaw_mat(matrix: torch.Tensor) -> torch.Tensor:
    forward_x, forward_y = matrix[..., 0, 0], matrix[..., 1, 0]
    norm = torch.sqrt(forward_x.square() + forward_y.square() + 1e-9)
    cosine, sine = forward_x / norm, forward_y / norm
    zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
    return torch.stack(
        (cosine, -sine, zero, sine, cosine, zero, zero, zero, one), dim=-1
    ).reshape(matrix.shape[:-2] + (3, 3))


def physical_state_parts(state: torch.Tensor):
    """Unpack the canonical SuperMimic physical state ``(p,R,v,w,q,qd)``."""
    if state.shape[-1] != PHYSICAL_STATE_DIM:
        raise ValueError(f"physical WAM state must have {PHYSICAL_STATE_DIM} values")
    return (
        state[..., 0:3],
        _quat_to_mat(state[..., 3:7]),
        state[..., 7:10],
        state[..., 10:13],
        state[..., 13:42],
        state[..., 42:71],
    )


def physical_state_invariant_features(state: torch.Tensor) -> torch.Tensor:
    """Heading/position invariant input used by SuperMimic's WAM."""
    position, rotation, linear_velocity, angular_velocity, joint_position, joint_velocity = physical_state_parts(state)
    heading = _yaw_mat(rotation)
    rotation_t = rotation.transpose(-1, -2)
    return torch.cat(
        (
            position[..., 2:3],
            (heading.transpose(-1, -2) @ rotation)[..., :, :2].reshape(state.shape[:-1] + (6,)),
            (rotation_t @ linear_velocity.unsqueeze(-1)).squeeze(-1),
            (rotation_t @ angular_velocity.unsqueeze(-1)).squeeze(-1),
            joint_position,
            joint_velocity,
        ), dim=-1,
    )


def physical_state_transition_delta(current: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    """Current-frame increments used to normalize and integrate WAM outputs.

    These are dynamics targets, rather than the target-frame prediction error
    used by SuperMimic's autoregressive fitting objective.
    """
    position, rotation, linear_velocity, angular_velocity, joint_position, joint_velocity = physical_state_parts(current)
    next_position, _, next_linear_velocity, next_angular_velocity, next_joint_position, next_joint_velocity = physical_state_parts(next_state)
    heading = _yaw_mat(rotation)
    rotation_t = rotation.transpose(-1, -2)
    quaternion = F.normalize(current[..., 3:7], dim=-1)
    next_quaternion = F.normalize(next_state[..., 3:7], dim=-1)
    inverse = torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)
    dtheta = _quat_to_rotvec(_quat_multiply(inverse, next_quaternion))
    return torch.cat(
        (
            (heading.transpose(-1, -2) @ (next_position - position).unsqueeze(-1)).squeeze(-1),
            dtheta,
            (rotation_t @ (next_linear_velocity - linear_velocity).unsqueeze(-1)).squeeze(-1),
            (rotation_t @ (next_angular_velocity - angular_velocity).unsqueeze(-1)).squeeze(-1),
            next_joint_position - joint_position,
            next_joint_velocity - joint_velocity,
        ), dim=-1,
    )


def physical_state_transition_error(
    predicted: torch.Tensor, target: torch.Tensor, *, error_frame: str = "target",
) -> torch.Tensor:
    """Autoregressive fitting error, expressed in the true target pose's frame.

    ``predicted`` reproduces the earlier WAM fitting convention for checkpoint
    continuation and ablations. Its component signs also follow that earlier
    convention; the weighted squared-error objective is insensitive to them.
    Neither choice changes the current-frame dynamics output targets.
    """
    if error_frame == "target":
        return physical_state_transition_delta(target, predicted)
    if error_frame == "predicted":
        return physical_state_transition_delta(predicted, target)
    raise ValueError(f"unknown WAM error_frame: {error_frame}; expected target or predicted")


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

    def __init__(self, dim: int, eps: float = 1e-4, max_count: float = 2e8, min_std: float = 1e-2):
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        self.eps = float(eps)
        self.min_std = float(min_std)
        if self.eps <= 0 or self.min_std < 0:
            raise ValueError("normalizer eps must be positive and min_std nonnegative")
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
        return torch.sqrt(self.var + self.eps**2).clamp_min(self.min_std)


@dataclass
class WAMConfig:
    """Configuration for :class:`StateSpaceWorldModel`.

    ``hidden`` is intentionally smaller than the 2048-wide SuperMimic model by
    default.  The dimensions of GR00T observations are usually larger, so the
    caller can scale this tuple without changing any training code.
    """

    state_dim: int
    action_dim: int
    # Kept for checkpoint/config compatibility. WAMTrainer sets this to zero;
    # goals are never part of the Sonic dynamics input.
    goal_dim: int = 0
    context_dim: int = 0
    hidden: tuple[int, ...] = (512, 512, 512)
    term_hidden: tuple[int, ...] = (128, 128)
    reward_hidden: tuple[int, ...] = (128, 128)
    learn_termination: bool = True
    learn_reward: bool = True
    reward_loss_coef: float = 1.0
    residual_scale: float = 1.0
    # ``flat`` preserves the original GR00T prototype. ``physical`` uses the
    # canonical SuperMimic state and frame-local invariant deltas.
    state_type: str = "flat"
    reward_mode: str = "learned"
    termination_mode: str = "learned"
    # Physical output scaling and rollout-error weighting are independent.
    # The six floors correspond to dp, dtheta, dv, dw, dq and dqd.
    delta_norm_eps: float = 1e-4
    state_error_floor: tuple[float, ...] = (0.003, 0.005, 0.02, 0.05, 0.003, 0.05)
    error_frame: str = "target"
    # Preserve the existing per-valid-transition scale by default. The source
    # SuperMimic objective sums each window's errors and averages over the batch.
    loss_reduction: str = "mean_valid"
    context_mode: str = "legacy"


class StateSpaceWorldModel(nn.Module):
    """Learned dynamics with flat compatibility and physical SuperMimic modes.

    In ``flat`` mode the model predicts ``s[t+1] = s[t] + delta``. In
    ``physical`` mode it consumes SuperMimic invariant features and predicts
    frame-local ``dp, dtheta, dv, dw, dq, dqd`` increments, which are
    integrated with the current root rotation. All methods support leading
    batch dimensions, so tensors from vectorised Isaac Lab environments can be
    passed without reshaping by the caller.

    Args:
        cfg: :class:`WAMConfig` or a Hydra-compatible mapping/object exposing
            the same attributes.
    """

    def __init__(self, cfg: WAMConfig | object):
        super().__init__()
        self.cfg = cfg
        self.state_dim = int(cfg.state_dim)
        self.action_dim = int(cfg.action_dim)
        self.goal_dim = int(getattr(cfg, "goal_dim", 0))
        self.context_dim = int(getattr(cfg, "context_dim", 0))
        self.context_mode = str(getattr(cfg, "context_mode", "legacy"))
        if self.context_mode not in {"legacy", "supermimic"}:
            raise ValueError(f"unknown WAM context_mode: {self.context_mode}")
        if self.context_mode == "supermimic" and self.context_dim != 33:
            raise ValueError("SuperMimic context mode requires context_dim=33")
        self.residual_scale = float(getattr(cfg, "residual_scale", 1.0))
        self.state_type = str(getattr(cfg, "state_type", "flat"))
        self.reward_mode = str(getattr(cfg, "reward_mode", "learned"))
        self.termination_mode = str(getattr(cfg, "termination_mode", "learned"))
        hidden = tuple(int(v) for v in getattr(cfg, "hidden", (512, 512, 512)))
        term_hidden = tuple(int(v) for v in getattr(cfg, "term_hidden", (128, 128)))
        reward_hidden = tuple(int(v) for v in getattr(cfg, "reward_hidden", term_hidden))
        self.reward_loss_coef = float(getattr(cfg, "reward_loss_coef", 1.0))
        self.error_frame = str(getattr(cfg, "error_frame", "target"))
        self.loss_reduction = str(getattr(cfg, "loss_reduction", "mean_valid"))
        if self.error_frame not in {"target", "predicted"}:
            raise ValueError(f"unknown WAM error_frame: {self.error_frame}; expected target or predicted")
        if self.loss_reduction not in {"mean_valid", "sum_per_batch"}:
            raise ValueError(
                f"unknown WAM loss_reduction: {self.loss_reduction}; expected mean_valid or sum_per_batch"
            )
        if self.state_dim < 1 or self.action_dim < 1 or self.goal_dim < 0:
            raise ValueError("state/action dimensions must be positive and goal_dim non-negative")
        model_in = self.state_dim + self.action_dim + self.context_dim
        if self.state_type == "physical":
            if self.state_dim != PHYSICAL_STATE_DIM:
                raise ValueError(
                    f"physical WAM state must have {PHYSICAL_STATE_DIM} values, got {self.state_dim}"
                )
            model_in = PHYSICAL_STATE_DIM + self.action_dim + self.context_dim
            dynamics_out = PHYSICAL_DELTA_DIM
        elif self.state_type == "flat":
            dynamics_out = self.state_dim
        else:
            raise ValueError(f"unknown WAM state_type: {self.state_type}")
        reward_in = model_in + self.goal_dim
        self.dynamics = _mlp(model_in, hidden, dynamics_out)
        # Zero initial residual keeps the model compatible with a warm-started
        # policy and avoids unstable imagined rollouts before fitting starts.
        final = self.dynamics[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        self.state_norm = RunningNorm(self.state_dim)
        self.action_norm = RunningNorm(self.action_dim)
        self.goal_norm = RunningNorm(self.goal_dim) if self.goal_dim else None
        self.context_norm = (
            RunningNorm(self.context_dim, eps=1e-2, min_std=0.0)
            if self.context_mode == "supermimic" else RunningNorm(self.context_dim)
        ) if self.context_dim else None
        self.delta_dim = dynamics_out
        if self.state_type == "physical":
            self.delta_norm = RunningNorm(
                self.delta_dim, eps=float(getattr(cfg, "delta_norm_eps", 1e-4)), min_std=0.0,
            )
            floors = torch.as_tensor(getattr(
                cfg, "state_error_floor", (0.003, 0.005, 0.02, 0.05, 0.003, 0.05),
            ), dtype=torch.float32)
            if floors.shape != (6,) or not torch.isfinite(floors).all() or (floors <= 0).any():
                raise ValueError("physical state_error_floor requires six finite positive values")
            self.register_buffer("state_error_floor", torch.repeat_interleave(
                floors, torch.tensor((3, 3, 3, 3, NUM_G1_JOINTS, NUM_G1_JOINTS)),
            ))
        else:
            self.delta_norm = RunningNorm(self.delta_dim)
        learn_term = bool(getattr(cfg, "learn_termination", True)) and self.termination_mode == "learned"
        self.termination_head = (
            _mlp(self.state_dim + self.goal_dim, term_hidden, 1) if learn_term else None
        )
        learn_reward = bool(getattr(cfg, "learn_reward", True)) and self.reward_mode == "learned"
        self.reward_head = _mlp(model_in, reward_hidden, 1) if learn_reward else None
        if self.reward_head is not None and self.goal_dim:
            self.reward_head = _mlp(reward_in, reward_hidden, 1)

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

    def context_features(self, context: torch.Tensor) -> torch.Tensor:
        """Map raw replay context to the privileged model-input convention."""
        if self.context_mode == "supermimic":
            return normalize_supermimic_context(context)
        return context

    def _goal(self, goal: Optional[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        if self.goal_dim == 0:
            return state.new_zeros(*state.shape[:-1], 0)
        if goal is None:
            raise ValueError("goal is required when WAMConfig.goal_dim is non-zero")
        if goal.shape[-1] != self.goal_dim:
            raise ValueError(
                f"goal last dimension must be {self.goal_dim}, got {goal.shape[-1]}"
            )
        return goal

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
        model_state = (
            physical_state_invariant_features(state)
            if self.state_type == "physical" else state
        )
        state_n = self.state_norm(model_state)
        action_n = self.action_norm(action)
        if self.context_norm is not None:
            context = self.context_norm(self.context_features(context))
        return torch.cat((state_n, action_n, context), dim=-1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        *,
        goal: Optional[torch.Tensor] = None,
        next_state_projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Predict next state, reward, and termination logits for one transition.

        Project known references and routing bits before evaluating termination,
        so its head sees the same next state returned to the rollout. Otherwise
        unconstrained predictions of projected features can provide a shortcut
        to the termination objective and its policy gradients.
        """
        x = self._model_input(state, action, context)
        goal = self._goal(goal, state)
        goal_n = self.goal_norm(goal) if self.goal_norm is not None else goal
        delta = self.dynamics(x) * self.residual_scale
        if self.state_type == "physical":
            position, rotation, linear_velocity, angular_velocity, joint_position, joint_velocity = physical_state_parts(state)
            delta = delta * self.delta_norm.std() + self.delta_norm.mean
            dp, dtheta, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, NUM_G1_JOINTS, NUM_G1_JOINTS), -1)
            heading = _yaw_mat(rotation)
            next_position = position + (heading @ dp.unsqueeze(-1)).squeeze(-1)
            # Integrate on unit quaternions directly. Matrix-to-quaternion
            # conversion is unnecessary here and would introduce chart
            # switches as the robot passes through a 180-degree heading.
            next_quaternion = F.normalize(_quat_multiply(
                F.normalize(state[..., 3:7], dim=-1), _rotvec_to_quat(dtheta),
            ), dim=-1)
            next_linear_velocity = linear_velocity + (rotation @ dv.unsqueeze(-1)).squeeze(-1)
            next_angular_velocity = angular_velocity + (rotation @ dw.unsqueeze(-1)).squeeze(-1)
            next_state = torch.cat(
                (
                    next_position,
                    next_quaternion,
                    next_linear_velocity,
                    next_angular_velocity,
                    joint_position + dq,
                    joint_velocity + dqd,
                ), dim=-1,
            )
        else:
            next_state = state + delta * self.delta_norm.std()
        if next_state_projector is not None:
            next_state = next_state_projector(next_state)
        out = {"next_state": next_state, "delta": delta}
        if self.reward_head is not None:
            reward_input = torch.cat((x, goal_n), dim=-1) if self.goal_dim else x
            out["reward"] = self.reward_head(reward_input).squeeze(-1)
        if self.termination_head is not None:
            term_state = (
                physical_state_invariant_features(next_state)
                if self.state_type == "physical" else next_state
            )
            term_input = torch.cat((self.state_norm(term_state), goal_n), dim=-1)
            out["termination_logits"] = self.termination_head(term_input).squeeze(-1)
        return out

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        *,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Differentiable one-step transition used by imagined rollouts."""
        return self.forward(state, action, context, goal=goal)["next_state"]

    def rollout(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
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
        if self.goal_dim:
            if goal is None or goal.ndim != 3 or goal.shape[:2] != actions.shape[:2]:
                raise ValueError("goal must have shape (B,H,goal_dim) when goal_dim is non-zero")
        states = [state]
        rewards = []
        terms = []
        current = state
        for t in range(actions.shape[1]):
            ctx_t = None if context is None else context[:, t]
            goal_t = None if goal is None else goal[:, t]
            result = self.forward(current, actions[:, t], ctx_t, goal=goal_t)
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
        goals: Optional[torch.Tensor] = None,
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
        if self.goal_dim:
            if goals is None or goals.shape[:2] != actions.shape[:2]:
                raise ValueError("goals must have shape (B,H,goal_dim) when goal_dim is non-zero")
        current = states[:, 0]
        state_errors, reward_preds, term_logits = [], [], []
        for t in range(actions.shape[1]):
            result = self.forward(
                current, actions[:, t], None if context is None else context[:, t],
                goal=None if goals is None else goals[:, t],
                next_state_projector=(
                    (lambda predicted, step=t: state_projector(predicted, step))
                    if state_projector is not None else None
                ),
            )
            current = result["next_state"]
            error_target = states[:, t + 1]
            if self.state_type == "physical":
                transition = physical_state_transition_error(
                    current, error_target, error_frame=self.error_frame,
                )
                error_std = torch.maximum(self.delta_norm.std(), self.state_error_floor)
                state_errors.append((transition / error_std).square().mean(-1))
            else:
                state_errors.append(
                    ((current - error_target) / self.state_norm.std()).square().mean(-1)
                )
            if "reward" in result:
                reward_preds.append(result["reward"])
            if "termination_logits" in result:
                term_logits.append(result["termination_logits"])
        error = torch.stack(state_errors, dim=1)
        valid = mask.to(error.dtype)
        state_valid = valid if dynamics_mask is None else valid * dynamics_mask.to(error.dtype)
        state_loss = self._reduce_transition_loss(error, state_valid)
        losses = {"state_loss": state_loss, "loss": state_loss}
        if reward_preds and rewards is not None:
            reward_targets = rewards.to(reward_preds[0].dtype)
            if reward_targets.shape != (actions.shape[0], actions.shape[1]):
                reward_targets = reward_targets.reshape(actions.shape[0], actions.shape[1])
            reward_error = (torch.stack(reward_preds, dim=1) - reward_targets).square()
            reward_loss = self._reduce_transition_loss(reward_error, valid)
            losses["reward_loss"] = reward_loss
            losses["loss"] = losses["loss"] + self.reward_loss_coef * reward_loss
        if term_logits and dones is not None:
            logits = torch.stack(term_logits, dim=1)
            if dones.shape != logits.shape:
                dones = dones.reshape_as(logits)
            term_loss = F.binary_cross_entropy_with_logits(logits, dones.float(), reduction="none")
            term_loss = self._reduce_transition_loss(term_loss, valid)
            losses["termination_loss"] = term_loss
            losses["loss"] = losses["loss"] + term_loss
        losses["last_valid_fraction"] = state_valid[:, -1].mean()
        losses["dynamics_valid_fraction"] = state_valid.mean()
        losses["valid_steps_mean"] = valid.sum(-1).mean()
        # Preserve the historical zero when the final step has no valid state
        # target. Read it together with last_valid_fraction: zero coverage means
        # the error is unobserved, rather than a perfect multi-step prediction.
        losses["state_error_last"] = (error[:, -1] * state_valid[:, -1]).sum() / (
            state_valid[:, -1].sum().clamp_min(1)
        )
        return losses

    def _reduce_transition_loss(self, error: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        denominator = valid.sum().clamp_min(1) if self.loss_reduction == "mean_valid" else max(len(error), 1)
        return (error * valid).sum() / denominator

    @torch.no_grad()
    def update_normalizers(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        goals: Optional[torch.Tensor] = None,
    ) -> None:
        """Update running statistics from real transitions only."""
        if self.state_type == "physical":
            self.state_norm.update(physical_state_invariant_features(states))
        else:
            self.state_norm.update(states)
        self.action_norm.update(actions)
        if self.goal_norm is not None and goals is not None:
            self.goal_norm.update(goals)
        if self.context_norm is not None and context is not None:
            self.context_norm.update(self.context_features(context))
        if next_states is not None:
            if self.state_type == "physical":
                self.delta_norm.update(physical_state_transition_delta(states, next_states))
            else:
                self.delta_norm.update(next_states - states)

    def imagine(
        self,
        policy: Callable[[torch.Tensor, Optional[torch.Tensor], int], torch.Tensor],
        state: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        horizon: int = 1,
        goal: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Generate a short differentiable rollout from a policy callback.

        WAM parameters are temporarily frozen while autograd remains enabled,
        so gradients flow from imagined rewards through ``state`` into the
        policy but do not update the dynamics model.  This is the key distinction
        from a model-free PPO update and follows the SuperMimic training loop.
        """
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if self.goal_dim:
            if goal is None or goal.ndim != 3 or goal.shape[:2] != (state.shape[0], horizon):
                raise ValueError("goal must have shape (B,H,goal_dim) when goal_dim is non-zero")
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
                goal_t = None if goal is None else goal[:, t]
                result = self.forward(current, action, ctx_t, goal=goal_t)
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


class WAMCritic(nn.Sequential):
    """Value network used by the WAM trainer.

    Keeping this small backbone as a module makes the critic Hydra-configurable
    like the rest of GR00T while preserving the original three-layer MLP state
    layout used by existing WAM checkpoints.
    """

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (256, 128)):
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        layers: list[nn.Module] = []
        dim = int(input_dim)
        for width in hidden:
            layers.extend((nn.Linear(dim, int(width)), nn.SiLU()))
            dim = int(width)
        layers.append(nn.Linear(dim, 1))
        super().__init__(*layers)


# Backward-compatible name for experiments that imported the initial flat-state
# prototype.  The canonical FSQ-aware implementation lives in
# ``gear_sonic.trl.world_model``.
DifferentiableWorldModel = StateSpaceWorldModel

__all__ = [
    "StateSpaceWorldModel", "DifferentiableWorldModel", "RunningNorm", "WAMConfig", "WAMCritic",
    "PHYSICAL_STATE_DIM", "PHYSICAL_DELTA_DIM", "physical_state_parts",
    "physical_state_invariant_features", "physical_state_transition_delta", "physical_state_transition_error",
]
