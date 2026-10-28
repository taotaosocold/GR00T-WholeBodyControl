"""World-model utilities for model-based whole-body control.

``StateSpaceWorldModel`` is the canonical goal-free dynamics used by
``trainer=wam``. ``WAM`` and ``ModelBasedActorCriticTrainer`` remain available
for legacy latent-model experiments and do not participate in that trainer.
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
