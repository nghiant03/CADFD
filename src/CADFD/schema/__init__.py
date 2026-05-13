"""Schema module for fault diagnosis configuration.

This module exports the fundamental types and configuration classes
shared across all phases: injection, training, and evaluation.
"""

from CADFD.schema.config import (
    EvaluateConfig,
    InjectionConfig,
    OptimizeConfig,
    TrainConfig,
)
from CADFD.schema.fault import FaultConfig, FaultType, MarkovConfig
from CADFD.schema.manifest import (
    DatasetInfo,
    EnvInfo,
    GitInfo,
    RunManifest,
    Timing,
)
from CADFD.schema.window import WindowConfig

__all__ = [
    "DatasetInfo",
    "EnvInfo",
    "EvaluateConfig",
    "FaultConfig",
    "FaultType",
    "GitInfo",
    "InjectionConfig",
    "MarkovConfig",
    "OptimizeConfig",
    "RunManifest",
    "Timing",
    "TrainConfig",
    "WindowConfig",
]
