"""Evaluation module for fault diagnosis models.

Provides the Evaluator class and metric computation utilities.
"""

from CADFD.evaluation.evaluator import EvalResult, Evaluator
from CADFD.evaluation.metrics import (
    ClassMetrics,
    compute_class_metrics,
    confusion_matrix,
    macro_f1,
)

__all__ = [
    "ClassMetrics",
    "EvalResult",
    "Evaluator",
    "compute_class_metrics",
    "confusion_matrix",
    "macro_f1",
]
