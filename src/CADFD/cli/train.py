"""CLI subcommand for model training."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Optional

import typer

from CADFD.datasets import load_dataset
from CADFD.evaluation import Evaluator
from CADFD.logging import logger
from CADFD.models import create_model, get_model_class
from CADFD.schema import (
    EvaluateConfig,
    RunManifest,
    Timing,
    TrainConfig,
)
from CADFD.schema.fault import FaultType
from CADFD.training import (
    CheckpointCallback,
    EarlyStoppingCallback,
    HistoryCallback,
    LoggingCallback,
    Trainer,
    build_loss,
)
from CADFD.utils import (
    collect_env_info,
    collect_git_info,
    generate_run_id,
    utc_now_iso,
)

_FIELD_DEFAULTS = TrainConfig.model_fields


def _field_default(name: str) -> object:
    """Get the default value for a TrainConfig field."""
    return _FIELD_DEFAULTS[name].default


def train(
    model: Annotated[
        str,
        typer.Argument(help="Model architecture"),
    ],
    data: Annotated[
        Path,
        typer.Argument(help="Path to injected dataset directory"),
    ],
    epochs: Annotated[
        Optional[int],
        typer.Option("--epochs", "-e", help=f"Training epochs (default: {_field_default('epochs')})"),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", "-b", help=f"Batch size (default: {_field_default('batch_size')})"),
    ] = None,
    learning_rate: Annotated[
        Optional[float],
        typer.Option("--lr", help=f"Learning rate (default: {_field_default('learning_rate')})"),
    ] = None,
    use_focal_loss: Annotated[
        Optional[bool],
        typer.Option("--focal-loss/--no-focal-loss", help="Use focal loss instead of cross-entropy"),
    ] = None,
    focal_gamma: Annotated[
        Optional[float],
        typer.Option("--focal-gamma", help=f"Focal loss gamma (default: {_field_default('focal_gamma')})"),
    ] = None,
    oversample: Annotated[
        Optional[bool],
        typer.Option("--oversample/--no-oversample", help="Oversample minority classes"),
    ] = None,
    oversample_ratio: Annotated[
        Optional[float],
        typer.Option(
            "--oversample-ratio",
            help=f"Target minority/majority ratio (default: {_field_default('oversample_ratio')})",
        ),
    ] = None,
    val_ratio: Annotated[
        Optional[float],
        typer.Option(
            "--val-ratio",
            help=f"Fraction of training data for validation (default: {_field_default('val_ratio')})",
        ),
    ] = None,
    early_stopping: Annotated[
        Optional[bool],
        typer.Option("--early-stopping/--no-early-stopping", help="Enable early stopping"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Parent directory for runs (default: runs/<model>). A new run subdirectory is created per invocation.",
        ),
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", "-s", help=f"Random seed (default: {_field_default('seed')})"),
    ] = None,
    features: Annotated[
        Optional[list[str]],
        typer.Option("--features", "-f", help="Feature(s) to train on (default: all)"),
    ] = None,
    config_file: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to YAML config file"),
    ] = None,
) -> None:
    """Train a fault diagnosis model."""
    if config_file is not None:
        config = TrainConfig.from_yaml(config_file)
        config = config.model_copy(update={"model": model})
    else:
        config = TrainConfig(model=model)

    cli_overrides: dict[str, object] = {}
    if epochs is not None:
        cli_overrides["epochs"] = epochs
    if batch_size is not None:
        cli_overrides["batch_size"] = batch_size
    if learning_rate is not None:
        cli_overrides["learning_rate"] = learning_rate
    if use_focal_loss is not None:
        cli_overrides["use_focal_loss"] = use_focal_loss
    if focal_gamma is not None:
        cli_overrides["focal_gamma"] = focal_gamma
    if oversample is not None:
        cli_overrides["oversample"] = oversample
    if oversample_ratio is not None:
        cli_overrides["oversample_ratio"] = oversample_ratio
    if val_ratio is not None:
        cli_overrides["val_ratio"] = val_ratio
    if seed is not None:
        cli_overrides["seed"] = seed
    if features is not None:
        cli_overrides["features"] = features

    if cli_overrides:
        config = config.model_copy(update=cli_overrides)
    logger.debug("TrainConfig: {}", config.to_dict())

    logger.info("Loading data from: {}", data)
    dataset = load_dataset(data)
    dataset.print_summary()

    model_cls = get_model_class(config.model)
    prepared = dataset.prepare(
        features=config.features,
        required_metadata=model_cls.required_metadata,
    )

    logger.debug(
        "Windowed shapes: X_train={}, y_train={}, X_val={}, y_val={}, X_test={}, y_test={}",
        prepared.X_train.shape,
        prepared.y_train.shape,
        prepared.X_val.shape,
        prepared.y_val.shape,
        prepared.X_test.shape,
        prepared.y_test.shape,
    )

    input_size = prepared.input_size
    num_classes = FaultType.count()
    logger.debug(
        "Creating model: arch={}, input_size={}, num_classes={}",
        config.model,
        input_size,
        num_classes,
    )

    net = create_model(
        config.model,
        input_size=input_size,
        num_classes=num_classes,
        metadata=prepared.metadata,
        **config.model_kwargs,
    )
    logger.info(
        "Model: {} ({:,} parameters)", net.name, net.count_parameters()
    )

    output_root = output if output is not None else Path(f"runs/{config.model}")
    git = collect_git_info()
    run_id = generate_run_id(config.model, config.seed, git)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run dir: {}", run_dir)

    callbacks = [
        LoggingCallback(),
        CheckpointCallback(
            save_path=run_dir,
            config_dict=config.to_dict(),
            monitor=config.checkpoint_monitor,
        ),
        HistoryCallback(save_path=run_dir),
    ]

    if early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                patience=10, monitor=config.early_stopping_monitor
            )
        )

    trainer = Trainer(config=config, callbacks=callbacks)

    env = collect_env_info(trainer.device)
    dataset_info = dataset.describe(data)

    logger.info(
        "Training for {} epochs | batch_size={} | lr={} | focal_loss={} | oversample={}",
        config.epochs,
        config.batch_size,
        config.learning_rate,
        config.use_focal_loss,
        config.oversample,
    )

    started_at = utc_now_iso()
    t0 = time.perf_counter()
    result = trainer.fit(
        model=net,
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        X_val=prepared.X_val if prepared.has_val else None,
        y_val=prepared.y_val if prepared.has_val else None,
        metadata=prepared.metadata,
        node_mask_train=prepared.node_mask_train,
        edge_mask_train=prepared.edge_mask_train,
        node_mask_val=prepared.node_mask_val if prepared.has_val else None,
        edge_mask_val=prepared.edge_mask_val if prepared.has_val else None,
    )
    duration = time.perf_counter() - t0
    ended_at = utc_now_iso()

    logger.info(
        "Training complete at epoch {} | best_val_loss={:.4f}",
        result.stopped_epoch,
        result.best_val_loss if result.best_val_loss is not None else float("nan"),
    )
    logger.info("Model saved to: {}", run_dir)

    if prepared.has_test:
        logger.info("--- Final Test Evaluation ---")
        weight_path = run_dir / "weight.pt"
        if weight_path.exists():
            import torch

            net.load_state_dict(torch.load(weight_path, map_location=trainer.device))
            logger.info("Reloaded best checkpoint from {} for test evaluation", weight_path)
        else:
            logger.warning(
                "No checkpoint at {}; evaluating final-epoch weights", weight_path
            )
        evaluator = Evaluator(
            config=EvaluateConfig(batch_size=config.batch_size),
            device=str(trainer.device),
        )
        criterion = build_loss(config, trainer.device)
        eval_result = evaluator.evaluate(
            net,
            prepared.X_test,
            prepared.y_test,
            criterion=criterion,
            metadata=prepared.metadata,
            node_mask=prepared.node_mask_test,
            edge_mask=prepared.edge_mask_test,
        )
        evaluator.log_results(eval_result)

        eval_result.save(
            run_dir,
            train_config=config.to_dict(),
            injection_config=dataset.config.to_dict(),
        )
        logger.info("Results saved to: {}", run_dir)

    manifest = RunManifest(
        run_id=run_id,
        kind="train",
        seed=config.seed,
        model=config.model,
        num_parameters=net.count_parameters(),
        git=git,
        env=env,
        dataset=dataset_info,
        timing=Timing(
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration,
            epochs_run=result.stopped_epoch,
        ),
        train_config=config.to_dict(),
        injection_config=dataset.config.to_dict(),
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))
    logger.info("Manifest written to: {}", run_dir / "manifest.json")
