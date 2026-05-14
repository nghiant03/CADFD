"""Dataset loaders for fault diagnosis.

Provides standardized access to sensor datasets for fault injection.

Sub-packages
------------
- ``raw`` — Raw dataset loaders (pre-injection stage).
- ``injected`` — Post-injection containers, graph topology, and windowing.
"""

from CESTA.batch import GraphWindowBatch
from CESTA.datasets.injected.graph import (
    GraphDataset,
    GraphMetadata,
    load_adjacency_matrix,
    load_directed_edges,
    pack_link_mask,
    unpack_link_mask,
)
from CESTA.datasets.injected.loading import load_dataset
from CESTA.datasets.injected.tabular import InjectedDataset
from CESTA.datasets.injected.windowed import WindowedSplits
from CESTA.datasets.raw import get_dataset, list_datasets
from CESTA.datasets.raw.base import BaseDataset
from CESTA.datasets.raw.intel_lab import IntelLabDataset

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
