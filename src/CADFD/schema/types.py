"""Backward-compatible schema primitive exports."""

from CADFD.schema.fault import FaultConfig, FaultType, MarkovConfig
from CADFD.schema.window import WindowConfig

__all__ = [
    "FaultConfig",
    "FaultType",
    "MarkovConfig",
    "WindowConfig",
]
