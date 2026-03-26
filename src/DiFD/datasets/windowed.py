"""Unified windowed-split container and windowing utilities.

Provides ``WindowedSplits``, a single container that every data-preparation
path produces, plus the low-level sliding-window helper used internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from DiFD.schema.types import WindowConfig


@dataclass
class WindowedSplits:
    """Unified container for windowed train/val/test arrays.

    Holds the numpy arrays produced by any data-preparation path
    (per-group windowing, graph-aligned windowing, etc.) together
    with arbitrary metadata that downstream consumers (e.g. model
    constructors) may need.

    Attributes:
        X_train: Training features ``(N, window_size, features)``.
        y_train: Training labels ``(N, window_size)``.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        metadata: Extra information (e.g. graph topology).
    """

    X_train: NDArray[np.float32]
    y_train: NDArray[np.int32]
    X_val: NDArray[np.float32]
    y_val: NDArray[np.int32]
    X_test: NDArray[np.float32]
    y_test: NDArray[np.int32]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def input_size(self) -> int:
        """Return the feature dimension (last axis of X_train)."""
        return int(self.X_train.shape[-1])

    @property
    def has_val(self) -> bool:
        """Return whether a non-empty validation set exists."""
        return len(self.X_val) > 0

    @property
    def has_test(self) -> bool:
        """Return whether a non-empty test set exists."""
        return len(self.X_test) > 0


def create_windows(
    data: NDArray[np.float32],
    labels: NDArray[np.int32],
    window_size: int,
    stride: int,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Create sliding windows from contiguous data.

    Args:
        data: Feature array of shape ``(timesteps, features)``.
        labels: Label array of shape ``(timesteps,)``.
        window_size: Number of timesteps per window.
        stride: Step size between consecutive windows.

    Returns:
        Tuple of ``(X, y)`` where X has shape ``(num_windows, window_size, features)``
        and y has shape ``(num_windows, window_size)``.
    """
    if len(data) < window_size:
        return (
            np.empty((0, window_size, data.shape[1]), dtype=np.float32),
            np.empty((0, window_size), dtype=np.int32),
        )

    starts = list(range(0, len(data) - window_size + 1, stride))
    X = np.stack([data[i : i + window_size] for i in starts])
    y = np.stack([labels[i : i + window_size] for i in starts])
    return X.astype(np.float32), y.astype(np.int32)


def split_and_window(
    features: NDArray[np.float32],
    labels: NDArray[np.int32],
    wc: WindowConfig,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
]:
    """Chronologically split a single contiguous block and create windows."""
    n = len(features)
    train_end = int(n * wc.train_ratio)

    if wc.val_ratio > 0:
        val_len = int(train_end * wc.val_ratio)
        val_start = train_end - val_len
    else:
        val_start = train_end

    X_tr, y_tr = create_windows(features[:val_start], labels[:val_start], wc.window_size, wc.train_stride)
    X_va, y_va = create_windows(features[val_start:train_end], labels[val_start:train_end], wc.window_size, wc.test_stride)
    X_te, y_te = create_windows(features[train_end:], labels[train_end:], wc.window_size, wc.test_stride)

    return X_tr, y_tr, X_va, y_va, X_te, y_te


def validate_features(
    requested: list[str] | None,
    available: list[str],
) -> list[str]:
    """Validate and resolve the feature list.

    Args:
        requested: Feature names requested by the caller, or ``None`` to use all.
        available: Feature names available in the dataset.

    Returns:
        Resolved list of feature names.

    Raises:
        ValueError: If any requested name is not in *available*.
    """
    if requested is not None:
        unknown = set(requested) - set(available)
        if unknown:
            msg = f"Unknown features: {sorted(unknown)}. Available: {available}"
            raise ValueError(msg)
        return list(requested)
    return list(available)


def collect_splits(
    wc: WindowConfig,
    n_feat: int,
    train_X_parts: list[NDArray[np.float32]],
    train_y_parts: list[NDArray[np.int32]],
    val_X_parts: list[NDArray[np.float32]],
    val_y_parts: list[NDArray[np.int32]],
    test_X_parts: list[NDArray[np.float32]],
    test_y_parts: list[NDArray[np.int32]],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
]:
    """Concatenate per-group window parts into final arrays.

    Returns empty arrays with correct shape when no windows were produced.
    """
    X_train = np.concatenate(train_X_parts) if train_X_parts else np.empty((0, wc.window_size, n_feat), dtype=np.float32)
    y_train = np.concatenate(train_y_parts) if train_y_parts else np.empty((0, wc.window_size), dtype=np.int32)
    X_val = np.concatenate(val_X_parts) if val_X_parts else np.empty((0, wc.window_size, n_feat), dtype=np.float32)
    y_val = np.concatenate(val_y_parts) if val_y_parts else np.empty((0, wc.window_size), dtype=np.int32)
    X_test = np.concatenate(test_X_parts) if test_X_parts else np.empty((0, wc.window_size, n_feat), dtype=np.float32)
    y_test = np.concatenate(test_y_parts) if test_y_parts else np.empty((0, wc.window_size), dtype=np.int32)
    return X_train, y_train, X_val, y_val, X_test, y_test
