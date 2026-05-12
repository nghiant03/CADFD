"""CLI subcommand for hyperparameter optimization with Optuna."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from CADFD.logging import logger
from CADFD.schema import OptimizeConfig

app = typer.Typer(no_args_is_help=True, invoke_without_command=True)

_defaults = OptimizeConfig()


@app.callback()
def optimize(
    ctx: typer.Context,
    data: Annotated[
        Optional[Path],
        typer.Option("--data", "-d", help="Path to injected dataset directory"),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help=f"Model architecture (default: {_defaults.model})"),
    ] = None,
    n_trials: Annotated[
        Optional[int],
        typer.Option("--n-trials", "-n", help=f"Number of trials (default: {_defaults.n_trials})"),
    ] = None,
    timeout: Annotated[
        Optional[int],
        typer.Option("--timeout", "-t", help="Optimization timeout in seconds"),
    ] = None,
    epochs: Annotated[
        Optional[int],
        typer.Option("--epochs", "-e", help=f"Epochs per trial (default: {_defaults.epochs})"),
    ] = None,
    metric: Annotated[
        Optional[str],
        typer.Option(
            "--metric",
            help=f"Metric to optimize: val_loss|val_macro_f1|val_acc (default: {_defaults.metric})",
        ),
    ] = None,
    sampler: Annotated[
        Optional[str],
        typer.Option("--sampler", help=f"Sampler: tpe|random (default: {_defaults.sampler})"),
    ] = None,
    pruner: Annotated[
        Optional[str],
        typer.Option("--pruner", help=f"Pruner: median|none (default: {_defaults.pruner})"),
    ] = None,
    study_name: Annotated[
        Optional[str],
        typer.Option("--study-name", help="Optuna study name (default: cadfd-<model>)"),
    ] = None,
    storage: Annotated[
        Optional[str],
        typer.Option("--storage", help=f"Optuna storage URL (default: {_defaults.storage})"),
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", "-s", help=f"Random seed (default: {_defaults.seed})"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Path to write best params as JSON"),
    ] = None,
    features: Annotated[
        Optional[list[str]],
        typer.Option("--features", "-f", help="Subset of features to use (default: all)"),
    ] = None,
) -> None:
    """Run hyperparameter optimization with Optuna."""
    if ctx.invoked_subcommand is not None:
        return
    if data is None:
        raise typer.BadParameter("Missing option '--data'.")

    overrides: dict[str, Any] = {}
    if model is not None:
        overrides["model"] = model
    if n_trials is not None:
        overrides["n_trials"] = n_trials
    if timeout is not None:
        overrides["timeout"] = timeout
    if epochs is not None:
        overrides["epochs"] = epochs
    if metric is not None:
        overrides["metric"] = metric
        # Auto-align direction with the metric.
        overrides["direction"] = "minimize" if metric == "val_loss" else "maximize"
    if sampler is not None:
        overrides["sampler"] = sampler
    if pruner is not None:
        overrides["pruner"] = pruner
    if study_name is not None:
        overrides["study_name"] = study_name
    if storage is not None:
        overrides["storage"] = storage
    if seed is not None:
        overrides["seed"] = seed
    if features:
        overrides["features"] = features

    config = OptimizeConfig(**overrides)
    logger.debug("OptimizeConfig: {}", config.to_dict())

    from CADFD.optimization import Optimizer

    optimizer = Optimizer(config=config, data_path=data)
    study = optimizer.run()

    try:
        best = study.best_trial
    except ValueError:
        logger.warning("No completed trials; nothing to save.")
        return

    if output is not None:
        payload = {
            "study_name": config.resolved_study_name(),
            "metric": config.metric,
            "direction": study.direction.name.lower(),
            "best_value": best.value,
            "best_trial_number": best.number,
            "best_params": best.params,
            "optimize_config": config.to_dict(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        logger.info("Best params written to: {}", output)


@app.command("show")
def optimize_show(
    study_name: Annotated[
        str,
        typer.Argument(help="Name of the study to show"),
    ],
    storage: Annotated[
        Optional[str],
        typer.Option("--storage", help=f"Optuna storage URL (default: {_defaults.storage})"),
    ] = None,
    top: Annotated[
        int,
        typer.Option("--top", "-k", help="Number of top trials to display"),
    ] = 10,
) -> None:
    """Show results from an existing Optuna study."""
    import optuna
    from rich.console import Console
    from rich.table import Table

    resolved_storage = storage if storage is not None else _defaults.storage
    study = optuna.load_study(study_name=study_name, storage=resolved_storage)

    console = Console()
    direction = study.direction.name.lower()
    console.print(f"[bold]Study:[/bold] {study_name}  [bold]direction:[/bold] {direction}  [bold]trials:[/bold] {len(study.trials)}")

    completed = [t for t in study.trials if t.value is not None]
    reverse = direction == "maximize"
    completed.sort(key=lambda t: t.value if t.value is not None else 0.0, reverse=reverse)

    table = Table(title=f"Top {min(top, len(completed))} trials", show_header=True)
    table.add_column("#", style="cyan")
    table.add_column("value", style="green")
    table.add_column("state")
    table.add_column("params")
    for trial in completed[:top]:
        table.add_row(
            str(trial.number),
            f"{trial.value:.6f}" if trial.value is not None else "-",
            trial.state.name,
            ", ".join(f"{k}={v}" for k, v in trial.params.items()),
        )
    console.print(table)

    try:
        best = study.best_trial
        console.print(f"\n[bold]Best trial:[/bold] #{best.number}  value={best.value}")
    except ValueError:
        console.print("[yellow]No completed trials yet.[/yellow]")
