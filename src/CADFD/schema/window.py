"""Windowing schema definitions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WindowConfig(BaseModel):
    """Configuration for sliding window dataset creation.

    Attributes:
        window_size: Number of timesteps per window.
        train_stride: Stride for training windows (allows overlap).
        test_stride: Stride for testing windows (typically no overlap).
        train_ratio: Fraction of data for training (chronological split).
        val_ratio: Fraction of data for validation, taken from end of
            training portion (chronological). Set to 0.0 to disable.
    """

    model_config = ConfigDict(frozen=True)

    window_size: int = Field(default=60, ge=1)
    train_stride: int = Field(default=10, ge=1)
    test_stride: int = Field(default=60, ge=1)
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "window_size": self.window_size,
            "train_stride": self.train_stride,
            "test_stride": self.test_stride,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WindowConfig":
        """Reconstruct from dictionary."""
        return cls(
            window_size=data.get("window_size", 60),
            train_stride=data.get("train_stride", 10),
            test_stride=data.get("test_stride", 60),
            train_ratio=data.get("train_ratio", 0.8),
            val_ratio=data.get("val_ratio", 0.1),
        )
