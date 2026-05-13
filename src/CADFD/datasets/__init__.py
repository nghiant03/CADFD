"""Dataset loaders for fault diagnosis.

Provides standardized access to sensor datasets for fault injection.

Sub-packages
------------
- ``raw`` — Raw dataset loaders (pre-injection stage).
- ``injected`` — Post-injection containers, graph topology, and windowing.
"""

from CADFD.batch import GraphWindowBatch
from CADFD.datasets.injected.graph import (
    GraphDataset,
    GraphMetadata,
    load_adjacency_matrix,
    load_directed_edges,
    pack_link_mask,
    unpack_link_mask,
)
from CADFD.datasets.injected.loading import load_dataset
from CADFD.datasets.injected.tabular import InjectedDataset
from CADFD.datasets.injected.windowed import WindowedSplits
from CADFD.datasets.raw import get_dataset, list_datasets
from CADFD.datasets.raw.base import BaseDataset
from CADFD.datasets.raw.intel_lab import IntelLabDataset

__all__ = [
    "BaseDataset",
    "GraphDataset",
    "GraphMetadata",
    "GraphWindowBatch",
    "InjectedDataset",
    "IntelLabDataset",
    "WindowedSplits",
    "load_adjacency_matrix",
    "load_dataset",
    "load_directed_edges",
    "pack_link_mask",
    "unpack_link_mask",
    "get_dataset",
    "list_datasets",
]
