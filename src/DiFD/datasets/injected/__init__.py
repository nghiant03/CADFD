"""Injected dataset containers, graph topology, and windowing utilities."""

from DiFD.datasets.injected.graph import (
    GraphDataset,
    GraphMetadata,
    load_adjacency_matrix,
)
from DiFD.datasets.injected.loading import load_dataset
from DiFD.datasets.injected.tabular import InjectedDataset
from DiFD.datasets.injected.windowed import WindowedSplits

__all__ = [
    "GraphDataset",
    "GraphMetadata",
    "InjectedDataset",
    "WindowedSplits",
    "load_adjacency_matrix",
    "load_dataset",
]
