"""CLI subcommand for data preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from CESTA.logging import logger

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
    seed: Annotated[
        int,
        typer.Option("--seed", "-s", help="Dynamic link simulation seed"),
    ] = 0,
    rho: Annotated[
        float,
        typer.Option("--rho", help="Shared environment burst intensity fraction"),
    ] = 0.5,
    q_bad_base: Annotated[
        float,
        typer.Option("--q-bad-base", help="Base GOOD-to-BAD transition probability"),
    ] = 0.02,
    q_recover_base: Annotated[
        float,
        typer.Option("--q-recover-base", help="Base BAD-to-GOOD transition probability"),
    ] = 0.20,
    bad_success_floor: Annotated[
        float,
        typer.Option("--bad-success-floor", help="Success probability while link state is BAD"),
    ] = 0.05,
) -> None:
    """Add dynamic directed graph topology to an injected dataset."""
    from CESTA.datasets.injected.graph import GraphDataset

    if not data.exists():
        logger.error("Dataset directory not found: {}", data)
        raise typer.Exit(code=1)

    if not connectivity.exists():
        logger.error("Connectivity file not found: {}", connectivity)
        raise typer.Exit(code=1)

    logger.info("Building dynamic directed graph from: {}", connectivity)
    graph_ds = GraphDataset.from_connectivity(
        data,
        connectivity,
        threshold=threshold,
        seed=seed,
        rho=rho,
        q_bad_base=q_bad_base,
        q_recover_base=q_recover_base,
        bad_success_floor=bad_success_floor,
    )
    graph_ds.save(data)
    logger.info(
        "Saved dynamic graph data: T={}, nodes={}, directed_edges={}, threshold={:.2f} -> {}",
        graph_ds.link_mask.shape[0],
        graph_ds.num_nodes,
        graph_ds.edge_index.shape[1],
        graph_ds.threshold,
        data,
    )
