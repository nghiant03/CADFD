"""Injected dataset containers, graph topology, and windowing utilities."""

from CADFD.datasets.injected.graph import (
    GraphDataset,
    GraphMetadata,
    load_adjacency_matrix,
)
from CADFD.datasets.injected.loading import load_dataset
from CADFD.datasets.injected.tabular import InjectedDataset
from CADFD.datasets.injected.windowed import WindowedSplits

__all__ = [
    "GraphDataset",
    "GraphMetadata",
    "InjectedDataset",
    "WindowedSplits",
    "load_adjacency_matrix",
    "load_dataset",
]
