"""CLI subcommand for fault injection."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from CADFD.logging import logger
from CADFD.schema import InjectionConfig
from CADFD.seed import seed_everything

app = typer.Typer(no_args_is_help=True)


@app.command("run")
def inject_run(
    dataset: Annotated[
        str,
        typer.Argument(help="Dataset to use"),
    ],
    data_path: Annotated[
        Path,
        typer.Argument(help="Path to raw data file"),
    ],
    output: Annotated[
        Path,
        typer.Argument(help="Output path for injected dataset"),
    ],
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML/JSON injection config file (defaults to InjectionConfig() when omitted)",
        ),
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", "-s", help="Override seed from config"),
    ] = None,
) -> None:
    """Run fault injection on a dataset.

    All injection parameters live in the YAML config file. See
    ``config/injection/*.yaml`` for examples (5/10/15/20% fault ratios with
    DRIFT and STUCK as the dominant challenger faults).
    """
    from CADFD.datasets import get_dataset
    from CADFD.injection import FaultInjector

    if config is None:
        injection_config = InjectionConfig()
    else:
        suffix = config.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            injection_config = InjectionConfig.from_yaml(config)
        elif suffix == ".json":
            import json
            with config.open() as fh:
                injection_config = InjectionConfig.from_dict(json.load(fh))
        else:
            msg = f"Unsupported config extension: {config.suffix}"
            raise typer.BadParameter(msg)

    if seed is not None:
        injection_config = injection_config.model_copy(update={"seed": seed})
        injection_config = injection_config.model_copy(
            update={"markov": injection_config.markov.model_copy(update={"seed": seed})}
        )

    logger.info("Loading dataset: {}", dataset)
    ds = get_dataset(dataset, data_path)

    if injection_config.seed is not None:
        seed_everything(injection_config.seed)

    logger.info("Running fault injection with seed={}", injection_config.seed)
    injector = FaultInjector(injection_config)
    result = injector.run(ds)

    logger.info("Saving to: {}", output)
    result.save(output)
    result.print_summary()


@app.command("list")
def inject_list() -> None:
    """List available datasets."""
    from rich.console import Console
    from rich.table import Table

    from CADFD.datasets import list_datasets

    console = Console()
    datasets = list_datasets()
    table = Table(title="Available Datasets", show_header=True)
    table.add_column("Name", style="cyan", min_width=len(table.title or ""))
    for ds in datasets:
        table.add_row(ds)
    console.print(table)
