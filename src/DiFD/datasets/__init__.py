"""Dataset loaders for fault diagnosis.

Provides standardized access to sensor datasets for fault injection.
"""

from DiFD.datasets.base import BaseDataset
from DiFD.datasets.graph import GraphDataset, GraphMetadata, load_adjacency_matrix
from DiFD.datasets.injected import InjectedDataset
from DiFD.datasets.intel_lab import IntelLabDataset
from DiFD.datasets.loading import load_dataset
from DiFD.datasets.registry import get_dataset, list_datasets, register_dataset
from DiFD.datasets.windowed import WindowedSplits

__all__ = [
    "BaseDataset",
    "GraphDataset",
    "GraphMetadata",
    "InjectedDataset",
    "IntelLabDataset",
    "WindowedSplits",
    "load_adjacency_matrix",
    "load_dataset",
    "get_dataset",
    "list_datasets",
    "register_dataset",
]
