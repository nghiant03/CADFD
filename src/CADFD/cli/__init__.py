"""CADFD CLI - Centralized command-line interface using Typer."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from CADFD.cli.evaluate import evaluate
from CADFD.cli.inject import inject
from CADFD.cli.optimize import app as optimize_app
from CADFD.cli.prepare import app as prepare_app
from CADFD.cli.report import app as report_app
from CADFD.cli.train import train
from CADFD.logging import configure_logging

app = typer.Typer(
    name="cadfd",
    help="CADFD - Communication-Aware Distributed Fault Diagnosis",
    no_args_is_help=True,
)


@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Configure global options."""
    level = "DEBUG" if debug else "INFO"
    configure_logging(level=level, verbose=verbose)


def _print_name_table(title: str, names: list[str]) -> None:
    console = Console()
    table = Table(title=title, show_header=True)
    table.add_column("Name", style="cyan", min_width=len(title))
    for name in names:
        table.add_row(name)
    console.print(table)


@app.command("list")
def list_items(
    category: Annotated[
        str,
        typer.Argument(help="What to list: datasets, models, metrics, or runs"),
    ],
    runs_dir: Annotated[
        Path,
        typer.Option("--runs-dir", help="Root runs directory for 'runs' listings"),
    ] = Path("runs"),
) -> None:
    """List datasets, models, metrics, or runs."""
    category_key = category.lower()
    if category_key == "datasets":
        from CADFD.datasets import list_datasets

        _print_name_table("Available Datasets", list_datasets())
        return
    if category_key == "models":
        from CADFD.models import list_models

        _print_name_table("Available Models", list_models())
        return
    if category_key == "metrics":
        metrics = ["accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc"]
        _print_name_table("Available Metrics", metrics)
        return
    if category_key == "runs":
        from CADFD.cli.report import list_runs

        list_runs(runs_dir)
        return

    raise typer.BadParameter("Expected one of: datasets, models, metrics, runs")


app.command("inject", help="Inject faults into sensor datasets")(inject)
app.add_typer(prepare_app, name="prepare", help="Prepare dataset variants (e.g. graph topology)")
app.command("train", help="Train deep learning models")(train)
app.command("evaluate", help="Evaluate trained models")(evaluate)
app.add_typer(optimize_app, name="optimize", help="Hyperparameter optimization with Optuna")
app.add_typer(report_app, name="report", help="Aggregate run artifacts into comparison reports")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
