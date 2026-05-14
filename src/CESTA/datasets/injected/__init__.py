"""Injected dataset containers, graph topology, and windowing utilities."""

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

__all__ = [
    "GraphDataset",
    "GraphMetadata",
    "GraphWindowBatch",
    "InjectedDataset",
    "WindowedSplits",
    "load_adjacency_matrix",
    "load_dataset",
    "load_directed_edges",
    "pack_link_mask",
    "unpack_link_mask",
]
