"""Evaluation metrics for fault diagnosis models.

Provides functions for computing per-class and aggregate metrics
from model predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass
class ClassMetrics:
    """Per-class precision, recall, and F1 score.

    Attributes:
        precision: Per-class precision array.
        recall: Per-class recall array.
        f1: Per-class F1 score array.
        support: Per-class sample count array.
    """

    precision: list[float]
    recall: list[float]
    f1: list[float]
    support: list[int]


def compute_class_metrics(
    all_preds: list[torch.Tensor],
    all_targets: list[torch.Tensor],
    num_classes: int,
) -> ClassMetrics:
    """Compute per-class precision, recall, and F1 from collected predictions.

    Args:
        all_preds: List of prediction tensors (flattened per batch).
        all_targets: List of target tensors (flattened per batch).
        num_classes: Number of classes.

    Returns:
        :class:`ClassMetrics` with per-class metrics.
    """
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    valid = targets >= 0
    preds = preds[valid]
    targets = targets[valid]

    precision = []
    recall = []
    f1 = []
    support = []

    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision.append(p)
        recall.append(r)
        f1.append(f)
        support.append(int(((targets == c).sum().item())))

    return ClassMetrics(precision=precision, recall=recall, f1=f1, support=support)


def macro_f1(class_metrics: ClassMetrics) -> float:
    """Compute macro-averaged F1 from per-class metrics."""
    if not class_metrics.f1:
        return 0.0
    return sum(class_metrics.f1) / len(class_metrics.f1)


def confusion_matrix(
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32],
    num_classes: int,
) -> NDArray[np.int64]:
    """Compute a dense confusion matrix.

    Rows index true labels; columns index predicted labels. Entries
    outside ``[0, num_classes)`` are ignored.

    Args:
        y_true: Ground-truth label array.
        y_pred: Predicted label array (same shape as ``y_true``).
        num_classes: Number of classes in the output matrix.

    Returns:
        Confusion matrix of shape ``(num_classes, num_classes)``.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    if y_true.size == 0:
        return cm
    mask = (
        (y_true >= 0)
        & (y_true < num_classes)
        & (y_pred >= 0)
        & (y_pred < num_classes)
    )
    idx = y_true[mask].astype(np.int64) * num_classes + y_pred[mask].astype(np.int64)
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)
