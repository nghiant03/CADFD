"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

from CADFD.datasets.injected.tabular import InjectedDataset


def load_dataset(path: str | Path) -> InjectedDataset:
    """Load the appropriate dataset variant from a directory.

    If the directory contains graph metadata files (``adjacency.npy``
    and ``graph_meta.json``), a :class:`GraphDataset` is returned.
    Otherwise a plain :class:`InjectedDataset` is returned.

    Since ``GraphDataset`` is a subclass of ``InjectedDataset``, the
    return type is always ``InjectedDataset`` and callers can use
    ``.prepare()`` polymorphically.

    Args:
        path: Path to the dataset directory.

    Returns:
        ``InjectedDataset`` or ``GraphDataset``.
    """
    from CADFD.datasets.injected.graph import GraphDataset

    directory = Path(path)
    has_graph = (
        (directory / "adjacency.npy").exists()
        and (directory / "graph_meta.json").exists()
    )

    if has_graph:
        return GraphDataset.load(directory)
    return InjectedDataset.load(directory)
