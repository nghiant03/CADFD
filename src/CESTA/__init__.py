"""CESTA - Deep Learning Fault Diagnosis.

A research framework for fault diagnosis using deep learning
on sensor time series data.
"""

from CESTA.datasets import InjectedDataset
from CESTA.schema import (
    DataConfig,
    DataSplitConfig,
    FaultConfig,
    FaultType,
    InjectionConfig,
    MarkovConfig,
    WindowConfig,
)

__version__ = "0.1.0"

__all__ = [
    "DataConfig",
    "DataSplitConfig",
    "FaultType",
    "FaultConfig",
    "MarkovConfig",
    "WindowConfig",
    "InjectionConfig",
    "InjectedDataset",
]
