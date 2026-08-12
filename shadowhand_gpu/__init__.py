"""Direct MuJoCo Warp components for the ShadowHand GPU migration.

The package deliberately keeps MuJoCo Warp and Warp as optional imports so the
existing CPU training environment continues to work unchanged.
"""

from .model_loader import ModelLoadReport, load_project_model
from .sensors import SensorLayout, build_sensor_layout
from .task import ShadowHandTaskConfig, ShadowHandTaskStep, ShadowHandWarpTask
from .trainer import (
    AutoTuneRecommendation,
    CudaTQCTrainer,
    EvaluationResult,
    TrainerConfig,
    reference_gradient_steps,
)

__all__ = [
    "ModelLoadReport",
    "SensorLayout",
    "ShadowHandTaskConfig",
    "ShadowHandTaskStep",
    "ShadowHandWarpTask",
    "CudaTQCTrainer",
    "AutoTuneRecommendation",
    "EvaluationResult",
    "TrainerConfig",
    "reference_gradient_steps",
    "build_sensor_layout",
    "load_project_model",
]
