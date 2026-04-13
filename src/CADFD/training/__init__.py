"""Training module for fault diagnosis models.

Provides the Trainer class, loss functions, oversampling utilities,
and training callbacks.
"""

from CADFD.training.callbacks import (
    CheckpointCallback,
    ClassMetrics,
    EarlyStoppingCallback,
    LoggingCallback,
    TrainMetrics,
    TrainingCallback,
)
from CADFD.training.loss import FocalLoss
from CADFD.training.oversampling import oversample_minority
from CADFD.training.trainer import TrainResult, Trainer

__all__ = [
    "CheckpointCallback",
    "ClassMetrics",
    "EarlyStoppingCallback",
    "FocalLoss",
    "LoggingCallback",
    "TrainMetrics",
    "TrainResult",
    "Trainer",
    "TrainingCallback",
    "oversample_minority",
]
