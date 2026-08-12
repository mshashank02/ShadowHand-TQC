"""CUDA-native reinforcement-learning components for the ShadowHand migration."""

from .tqc import (
    QuantileCritic,
    SquashedGaussianActor,
    TQCBatch,
    TQCConfig,
    TQCLearner,
    build_target_quantiles,
    quantile_huber_loss,
    truncate_target_quantiles,
)
from .normalization import (
    CudaVecNormalize,
    SB3_OBSERVATION_KEY_ORDER,
    TorchRunningMeanStd,
    flatten_observations,
)
from .replay import (
    CudaHERReplayBuffer,
    HERReplaySample,
    ReplayMemoryPlan,
    plan_replay_memory,
    shadowhand_sparse_reward,
)

__all__ = [
    "QuantileCritic",
    "SquashedGaussianActor",
    "TQCBatch",
    "TQCConfig",
    "TQCLearner",
    "build_target_quantiles",
    "CudaVecNormalize",
    "CudaHERReplayBuffer",
    "flatten_observations",
    "HERReplaySample",
    "plan_replay_memory",
    "quantile_huber_loss",
    "ReplayMemoryPlan",
    "SB3_OBSERVATION_KEY_ORDER",
    "TorchRunningMeanStd",
    "shadowhand_sparse_reward",
    "truncate_target_quantiles",
]
