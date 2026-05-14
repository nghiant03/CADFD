"""CLI subcommand for model evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from CADFD.datasets import load_dataset
from CADFD.evaluation import Evaluator
from CADFD.logging import logger
from CADFD.models import create_model, get_model_class
from CADFD.schema import (
    EvaluateConfig,
    RunManifest,
    Timing,
)
from CADFD.schema.fault import FaultType
from CADFD.utils import (
    collect_env_info,
    collect_git_info,
    generate_run_id,
    utc_now_iso,
)


def evaluate(
    model: Annotated[
        Path,
        typer.Option("--model", "-m", help="Path to trained model directory"),
    ],
    data: Annotated[
        Path,
        typer.Option("--data", "-d", help="Path to injected dataset directory"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory for evaluation results"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Evaluation batch size",
        ),
    ] = 64,
) -> None:
    """Evaluate a trained model on test data."""
    import torch

    from CADFD.models.base import BaseModel

    config = EvaluateConfig(batch_size=batch_size)

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
    model_kwargs = {}
    if isinstance(train_cfg, dict):
        saved_model_kwargs = train_cfg.get("model_kwargs", {})
        if isinstance(saved_model_kwargs, dict):
            model_kwargs = saved_model_kwargs
    net = create_model(
        model_name,
        input_size=input_size,
        num_classes=num_classes,
        metadata=prepared.metadata,
        **model_kwargs,
    )
    assert isinstance(net, BaseModel)
    net.load_state_dict(torch.load(model / "weight.pt", weights_only=True))
    logger.info("Model: {} ({:,} parameters)", net.name, net.count_parameters())

    evaluator = Evaluator(config=config)
    logger.info("Evaluating with batch_size={}", config.batch_size)

    started_at = utc_now_iso()
    t0 = time.perf_counter()
    result = evaluator.evaluate(
        net,
        prepared.X_test,
        prepared.y_test,
        metadata=prepared.metadata,
        node_mask=prepared.node_mask_test,
        edge_mask=prepared.edge_mask_test,
    )
    duration = time.perf_counter() - t0
    ended_at = utc_now_iso()

    evaluator.log_results(result)

    if output is not None:
        git = collect_git_info()
        run_id = generate_run_id(
            model_name,
            int(train_cfg.get("seed", 0)) if isinstance(train_cfg, dict) else 0,
            git,
        )
        save_dir = output / run_id
    else:
        save_dir = model
    save_dir.mkdir(parents=True, exist_ok=True)

    result.save(
        save_dir,
        train_config=train_cfg,  # type: ignore[arg-type]
        injection_config=dataset.config.to_dict(),
    )
    logger.info("Results saved to: {}", save_dir)

    if output is not None:
        env = collect_env_info(evaluator.device)
        dataset_info = dataset.describe(data)
        seed = int(train_cfg.get("seed", 0)) if isinstance(train_cfg, dict) else 0
        manifest = RunManifest(
            run_id=save_dir.name,
            kind="evaluate",
            seed=seed,
            model=model_name,
            num_parameters=net.count_parameters(),
            git=collect_git_info(),
            env=env,
            dataset=dataset_info,
            timing=Timing(
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=duration,
            ),
            train_config=train_cfg if isinstance(train_cfg, dict) else None,
            eval_config=config.to_dict(),
            injection_config=dataset.config.to_dict(),
        )
        (save_dir / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2))
        logger.info("Manifest written to: {}", save_dir / "manifest.json")
