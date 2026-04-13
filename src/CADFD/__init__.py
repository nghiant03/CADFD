"""CADFD - Deep Learning Fault Diagnosis.

A research framework for fault diagnosis using deep learning
on sensor time series data.
"""

from CADFD.datasets import InjectedDataset
from CADFD.schema import (
    FaultConfig,
    FaultType,
    InjectionConfig,
    MarkovConfig,
    WindowConfig,
)

__version__ = "0.1.0"

__all__ = [
    "FaultType",
    "FaultConfig",
    "MarkovConfig",
    "WindowConfig",
    "InjectionConfig",
    "InjectedDataset",
]
