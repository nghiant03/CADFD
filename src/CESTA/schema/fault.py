"""Fault-domain schema definitions."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FaultType(IntEnum):
    """Fault type enumeration.

    Integer values are used directly as labels in the dataset.
    New fault types should be added with sequential integer values.
    """

    NORMAL = 0
    SPIKE = 1
    DRIFT = 2
    STUCK = 3

    @classmethod
    def from_string(cls, name: str) -> "FaultType":
        """Convert string name to FaultType."""
        return cls[name.upper()]

    @classmethod
    def names(cls) -> list[str]:
        """Return list of all fault type names."""
        return [ft.name for ft in cls]

    @classmethod
    def fault_names(cls) -> list[str]:
        """Return list of fault type names excluding NORMAL."""
        return [ft.name for ft in cls if ft != cls.NORMAL]

    @classmethod
    def count(cls) -> int:
        """Return total number of fault types including NORMAL."""
        return len(cls)


class FaultConfig(BaseModel):
    """Configuration for a specific fault type.

    Attributes:
        fault_type: The type of fault.
        transition_prob: Probability of transitioning from NORMAL to this fault.
        average_duration: Expected duration in timesteps before returning to NORMAL.
        params: Fault-specific parameters (e.g., magnitude for spike).
    """

    model_config = ConfigDict(frozen=True, use_enum_values=False)

    fault_type: FaultType
    transition_prob: float = Field(default=0.02, ge=0.0, le=1.0)
    average_duration: int = Field(default=10, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fault_type", mode="before")
    @classmethod
    def _parse_fault_type(cls, value: Any) -> Any:
        if isinstance(value, str):
            return FaultType.from_string(value)
        return value

    def return_prob(self) -> float:
        """Probability of returning to NORMAL at each timestep."""
        return 1.0 / self.average_duration

class MarkovConfig(BaseModel):
    """Configuration for the Markov chain state generator.

    Attributes:
        fault_configs: List of fault configurations (excluding NORMAL).
        seed: Random seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    fault_configs: list[FaultConfig] = Field(default_factory=list)
    seed: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _set_default_configs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("fault_configs"):
                data = dict(data)
                data["fault_configs"] = cls._default_fault_configs()
        return data

    @staticmethod
    def _default_fault_configs() -> list["FaultConfig"]:
        """Return default fault configurations."""
        return [
            FaultConfig(
                fault_type=FaultType.SPIKE,
                transition_prob=0.010,
                average_duration=2,
                params={
                    "magnitude_sigma_range": (0.8, 2.0),
                    "magnitude_range": (1.0, 4.0),
                },
            ),
            FaultConfig(
                fault_type=FaultType.DRIFT,
                transition_prob=0.0015,
                average_duration=20,
                params={
                    "drift_rate_sigma_range": (0.02, 0.08),
                    "drift_rate_range": (0.05, 0.15),
                },
            ),
            FaultConfig(
                fault_type=FaultType.STUCK,
                transition_prob=0.0030,
                average_duration=10,
                params={"jitter_sigma_factor": 0.05},
            ),
        ]

    def get_config(self, fault_type: FaultType) -> FaultConfig | None:
        """Get configuration for a specific fault type."""
        for cfg in self.fault_configs:
            if cfg.fault_type == fault_type:
                return cfg
        return None

