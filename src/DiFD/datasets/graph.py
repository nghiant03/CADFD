"""Graph topology construction for sensor networks.

Loads adjacency matrices from connectivity data files and provides
a GraphDataset container for graph-structured fault diagnosis data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from DiFD.datasets.injected import InjectedDataset
from DiFD.datasets.windowed import (
    WindowedSplits,
    split_and_window,
    collect_splits,
    validate_features,
)
from DiFD.logging import logger
from DiFD.schema.types import WindowConfig


@dataclass
class GraphMetadata:
    """Typed metadata produced by graph-based data preparation.

    Attributes:
        adjacency: Binary adjacency matrix ``(num_nodes, num_nodes)``.
        node_ids: Sorted sensor/group IDs forming graph nodes.
        num_nodes: Number of graph nodes.
        threshold: Connectivity probability threshold used to build edges.
    """

    adjacency: NDArray[np.float32]
    node_ids: list[int]
    num_nodes: int
    threshold: float


def load_adjacency_matrix(
    connectivity_path: str | Path,
    node_ids: list[int],
    threshold: float = 0.5,
) -> NDArray[np.float32]:
    """Load adjacency matrix from a connectivity data file.

    Reads pairwise connectivity probabilities and thresholds them to
    produce a binary adjacency matrix.  Self-loops are always added.

    The connectivity file is whitespace-separated with three columns::

        source_id  dest_id  connectivity_probability

    Only edges between nodes present in *node_ids* are kept.

    Args:
        connectivity_path: Path to the connectivity data file.
        node_ids: Sorted sensor/group IDs forming graph nodes.
        threshold: Minimum connectivity probability to create an edge (0-1).

    Returns:
        Binary symmetric adjacency matrix ``(num_nodes, num_nodes)``
        with self-loops, dtype float32.
    """
    connectivity_path = Path(connectivity_path)
    num_nodes = len(node_ids)
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    with connectivity_path.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            src, dst = int(parts[0]), int(parts[1])
            prob = float(parts[2])
            if src in id_to_idx and dst in id_to_idx and prob >= threshold:
                i, j = id_to_idx[src], id_to_idx[dst]
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    np.fill_diagonal(adj, 1.0)

    num_edges = int(adj.sum() - num_nodes)
    logger.info(
        "Graph: {} nodes, {} edges (threshold={:.2f}), density={:.2f}%",
        num_nodes,
        num_edges // 2,
        threshold,
        100.0 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
    )

    return adj


@dataclass
class GraphDataset(InjectedDataset):
    """Graph-structured sensor dataset.

    Extends ``InjectedDataset`` with graph topology (adjacency matrix
    and node mapping).  Overrides ``prepare()`` to perform graph-aligned
    windowing where all sensor features are concatenated per timestep.

    Attributes:
        adjacency: Binary adjacency matrix (num_nodes, num_nodes) with self-loops.
        node_ids: Sorted sensor/group IDs forming graph nodes.
        threshold: Connectivity probability threshold used to build edges.
    """

    adjacency: NDArray[np.float32] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    node_ids: list[int] = field(default_factory=list)
    threshold: float = 0.5

    @property
    def num_nodes(self) -> int:
        """Return the number of graph nodes."""
        return len(self.node_ids)

    def save(self, path: str | Path) -> None:
        """Save graph dataset to directory.

        Writes the injected data (CSV + meta JSON) via the parent class,
        then adds graph-specific files (adjacency matrix + graph metadata).
        """
        super().save(path)

        directory = Path(path)
        np.save(directory / "adjacency.npy", self.adjacency)

        meta = {
            "node_ids": self.node_ids,
            "threshold": self.threshold,
        }
        (directory / "graph_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> GraphDataset:
        """Load graph dataset from directory.

        Expects the directory to contain both injected data files and
        graph-specific files (``adjacency.npy``, ``graph_meta.json``).

        Raises:
            FileNotFoundError: If graph files are missing.
        """
        directory = Path(path)

        parent = InjectedDataset.load(directory)
        adjacency: NDArray[np.float32] = np.load(directory / "adjacency.npy")
        meta = json.loads((directory / "graph_meta.json").read_text())

        return cls(
            df=parent.df,
            config=parent.config,
            feature_names=parent.feature_names,
            adjacency=adjacency,
            node_ids=meta["node_ids"],
            threshold=meta["threshold"],
        )

    @classmethod
    def from_connectivity(
        cls,
        path: str | Path,
        connectivity_path: str | Path,
        threshold: float = 0.5,
    ) -> GraphDataset:
        """Build graph dataset from a saved InjectedDataset and connectivity file.

        Args:
            path: Path to directory containing the saved InjectedDataset.
            connectivity_path: Path to whitespace-separated connectivity file
                with columns ``source_id dest_id probability``.
            threshold: Minimum connectivity probability for an edge (default 0.5).

        Returns:
            GraphDataset with adjacency loaded from the connectivity file.
        """
        parent = InjectedDataset.load(path)
        df = parent.df
        group_col = parent.group_column
        node_ids = sorted(int(g) for g in df[group_col].unique())
        adj = load_adjacency_matrix(connectivity_path, node_ids, threshold=threshold)
        return cls(
            df=df,
            config=parent.config,
            feature_names=parent.feature_names,
            adjacency=adj,
            node_ids=node_ids,
            threshold=threshold,
        )

    def prepare(
        self,
        window_config: WindowConfig | None = None,
        features: list[str] | None = None,
    ) -> WindowedSplits:
        """Convert this graph dataset into windowed train/val/test arrays.

        Unlike ``InjectedDataset.prepare`` (which windows each group
        independently), this method aligns ALL sensor groups onto a common
        time axis and concatenates their features at each timestep.  The
        resulting ``input_size = num_nodes * features_per_node`` lets a
        GCN reshape and apply graph convolutions across the sensor topology.

        The label at each timestep is the *maximum* fault label across all
        sensors (i.e., any fault at any sensor is propagated).

        Args:
            window_config: Windowing configuration. Falls back to the
                injection config stored inside the dataset.
            features: Subset of feature names. ``None`` uses all features.

        Returns:
            WindowedSplits with windowed arrays and graph metadata.

        Raises:
            ValueError: If any name in *features* is not in the dataset.
        """
        wc = window_config if window_config is not None else self.config.window
        selected_features = validate_features(features, self.feature_names)

        df = self.df
        group_col = self.group_column
        node_ids = self.node_ids

        pivot_dfs: list[NDArray[np.float32]] = []
        for nid in node_ids:
            group_df = df[df[group_col] == nid].sort_values("timestamp").reset_index(drop=True)
            pivot_dfs.append(group_df[selected_features].to_numpy(dtype=np.float32))

        min_len = min(len(p) for p in pivot_dfs)
        for i in range(len(pivot_dfs)):
            pivot_dfs[i] = pivot_dfs[i][:min_len]

        combined = np.concatenate(pivot_dfs, axis=1)

        label_arrays: list[NDArray[np.int32]] = []
        for nid in node_ids:
            gdf = df[df[group_col] == nid].sort_values("timestamp").reset_index(drop=True)
            label_arrays.append(gdf["fault_state"].to_numpy(dtype=np.int32)[:min_len])

        stacked = np.stack(label_arrays, axis=0)
        labels_all = stacked.max(axis=0).astype(np.int32)

        X_tr, y_tr, X_va, y_va, X_te, y_te = split_and_window(
            combined, labels_all, wc
        )

        n_feat = len(selected_features) * len(node_ids)
        X_train, y_train, X_val, y_val, X_test, y_test = collect_splits(
            wc, n_feat,
            [X_tr] if len(X_tr) > 0 else [],
            [y_tr] if len(y_tr) > 0 else [],
            [X_va] if len(X_va) > 0 else [],
            [y_va] if len(y_va) > 0 else [],
            [X_te] if len(X_te) > 0 else [],
            [y_te] if len(y_te) > 0 else [],
        )

        graph_meta = GraphMetadata(
            adjacency=self.adjacency,
            node_ids=self.node_ids,
            num_nodes=self.num_nodes,
            threshold=self.threshold,
        )

        return WindowedSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            metadata={"graph": graph_meta},
        )
