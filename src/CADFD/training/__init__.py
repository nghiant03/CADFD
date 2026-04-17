"""Training module for fault diagnosis models.

Provides the Trainer class, loss functions, oversampling utilities,
and training callbacks.
"""

from CADFD.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    HistoryCallback,
    LoggingCallback,
    TrainMetrics,
    TrainingCallback,
)
from CADFD.training.loss import FocalLoss
from CADFD.training.oversampling import oversample_minority
from CADFD.training.trainer import TrainResult, Trainer, build_loss

__all__ = [
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "FocalLoss",
    "HistoryCallback",
    "LoggingCallback",
    "TrainMetrics",
    "TrainResult",
    "Trainer",
    "TrainingCallback",
    "build_loss",
    "oversample_minority",
]
