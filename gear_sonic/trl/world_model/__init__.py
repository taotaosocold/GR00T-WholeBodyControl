"""Differentiable latent world models for model-based whole-body control.

The public API intentionally does not depend on the environment implementation.
``WAM`` can therefore be used with IsaacLab rollouts, replay buffers, or a
standalone simulator while the existing FSQ tokenizer remains unchanged.
"""

from gear_sonic.trl.modules.wam import StateSpaceWorldModel, WAMConfig

from .mb_trainer import ModelBasedActorCriticTrainer, TransitionReplayBuffer
from .sequence_replay import SequenceReplayBuffer
from .wam import (
    DifferentiableWorldModel,
    FSQObservationAdapter,
    LatentMPC,
    WAM,
)

__all__ = [
    "DifferentiableWorldModel",
    "FSQObservationAdapter",
    "LatentMPC",
    "WAM",
    "ModelBasedActorCriticTrainer",
    "TransitionReplayBuffer",
    "SequenceReplayBuffer",
    "StateSpaceWorldModel",
    "WAMConfig",
]
