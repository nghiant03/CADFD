"""CLI subcommand for data preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from DiFD.logging import logger

app = typer.Typer(no_args_is_help=True)


@app.command("graph")
def prepare_graph(
    data: Annotated[
        Path,
        typer.Argument(help="Path to injected dataset directory"),
    ],
    connectivity: Annotated[
        Path,
        typer.Argument(help="Path to connectivity data file"),
    ],
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Connectivity probability threshold"),
    ] = 0.5,
) -> None:
    """Add graph topology to an injected dataset.

    Reads pairwise connectivity probabilities from the connectivity file,
    builds a binary adjacency matrix, and saves it alongside the existing
    injected dataset files.
    """
    from DiFD.datasets.injected.graph import GraphDataset

    if not data.exists():
        logger.error("Dataset directory not found: {}", data)
        raise typer.Exit(code=1)

    if not connectivity.exists():
        logger.error("Connectivity file not found: {}", connectivity)
        raise typer.Exit(code=1)

    logger.info("Building graph from: {}", connectivity)
    graph_ds = GraphDataset.from_connectivity(data, connectivity, threshold=threshold)
    graph_ds.save(data)
    logger.info(
        "Saved graph data: {} nodes, threshold={:.2f} -> {}",
        graph_ds.num_nodes,
        graph_ds.threshold,
        data,
    )
