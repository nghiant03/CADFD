"""CLI subcommand for model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from CADFD.datasets import load_dataset
from CADFD.evaluation import Evaluator
from CADFD.logging import logger
from CADFD.models import create_model, get_model_class
from CADFD.schema import EvaluateConfig
from CADFD.schema.types import FaultType

app = typer.Typer(no_args_is_help=True)

_defaults = EvaluateConfig()


@app.command("run")
def evaluate_run(
    model: Annotated[
        Path,
        typer.Option("--model", "-m", help="Path to trained model directory"),
    ],
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to injected dataset directory"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory for evaluation results"),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", "-b", help=f"Evaluation batch size (default: {_defaults.batch_size})"),
    ] = None,
) -> None:
    """Evaluate a trained model on test data."""
    import torch

    from CADFD.models.base import BaseModel

    config = EvaluateConfig(
        batch_size=batch_size if batch_size is not None else _defaults.batch_size,
    )

    logger.info("Loading data from: {}", data)
    dataset = load_dataset(data)
    dataset.print_summary()

    logger.info("Loading model from: {}", model)
    meta = BaseModel.load_metadata(model)
    model_name = str(meta.get("model_name", "lstm"))
    model_config = meta.get("model_config", {})
    assert isinstance(model_config, dict)

    train_cfg = meta.get("train_config")
    saved_features: list[str] | None = None
    if isinstance(train_cfg, dict):
        saved_features = train_cfg.get("features")

    model_cls = get_model_class(model_name)
    prepared = dataset.prepare(
        features=saved_features,
        required_metadata=model_cls.required_metadata,
    )

    if not prepared.has_test:
        logger.error("No test data available in dataset")
        raise typer.Exit(code=1)

    input_size = prepared.input_size
    num_classes = FaultType.count()
    net = create_model(
        model_name,
        input_size=input_size,
        num_classes=num_classes,
        metadata=prepared.metadata,
    )
    assert isinstance(net, BaseModel)
    net.load_state_dict(
        torch.load(model / "weight.pt", weights_only=True)
    )
    logger.info("Model: {} ({:,} parameters)", net.name, net.count_parameters())

    evaluator = Evaluator(config=config)
    logger.info("Evaluating with batch_size={}", config.batch_size)
    result = evaluator.evaluate(net, prepared.X_test, prepared.y_test)
    evaluator.log_results(result)

    save_dir = output if output is not None else model
    result.save(
        save_dir,
        train_config=train_cfg,  # type: ignore[arg-type]
        injection_config=dataset.config.to_dict(),
    )
    logger.info("Results saved to: {}", save_dir)


@app.command("list")
def evaluate_list() -> None:
    """List available evaluation metrics."""
    from rich.console import Console
    from rich.table import Table

    metrics = ["accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc"]
    console = Console()
    table = Table(title="Available Metrics", show_header=True)
    table.add_column("Name", style="cyan", min_width=len(table.title or ""))
    for m in metrics:
        table.add_row(m)
    console.print(table)
