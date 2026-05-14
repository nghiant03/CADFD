"""CLI subcommand for fault injection."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from CESTA.logging import logger
from CESTA.schema import InjectionConfig
from CESTA.schema.config import load_config_file
from CESTA.seed import seed_everything


def inject(
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
            help="Path to YAML/JSON injection config file",
        ),
    ] = None,
) -> None:
    """Run fault injection on a dataset.

    All injection parameters live in the YAML config file. See
    ``config/injection/*.yaml`` for examples (5/10/15/20% fault ratios with
    DRIFT and STUCK as the dominant challenger faults).
    """
    from CESTA.datasets import get_dataset
    from CESTA.injection import FaultInjector

    injection_config = InjectionConfig.model_validate(load_config_file(config)) if config is not None else InjectionConfig()

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
