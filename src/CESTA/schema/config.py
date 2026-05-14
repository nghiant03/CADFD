"""Pipeline configuration classes.

This module defines configuration classes for all pipeline phases:
injection, training, evaluation, and optimization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from CESTA.schema.fault import FaultType, MarkovConfig
from CESTA.schema.window import WindowConfig


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file as raw mapping data."""
    resolved = Path(path)
    if not resolved.exists():
        msg = f"Config file not found: {resolved}"
        raise FileNotFoundError(msg)

    suffix = resolved.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(resolved.read_text()) or {}
    elif suffix == ".json":
        raw = json.loads(resolved.read_text())
    else:
        msg = f"Unsupported config extension: {resolved.suffix}"
        raise ValueError(msg)

    if not isinstance(raw, dict):
        msg = f"Config file must contain a mapping: {resolved}"
        raise ValueError(msg)
    return raw


class InjectionConfig(BaseModel):
    """Complete configuration for fault injection pipeline.

    This is the main config object that gets serialized as metadata.

    Attributes:
        markov: Markov chain configuration.
        window: Windowing configuration.
        resample_freq: Resampling frequency string (e.g., "30s").
        target_features: Features to inject faults into.
        all_features: All features to include in the output.
        interpolation_method: Method for interpolating missing values.
        group_column: Column to group by (e.g., "moteid").
        seed: Global random seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    markov: MarkovConfig = Field(default_factory=MarkovConfig)
    window: WindowConfig = Field(default_factory=WindowConfig)
    resample_freq: str = "5min"
    target_features: list[str] = Field(default_factory=lambda: ["temp"])
    all_features: list[str] = Field(default_factory=lambda: ["temp", "humid", "light", "volt"])
    interpolation_method: str = "linear"
    group_column: str = "moteid"
    seed: int | None = None

    @model_validator(mode="after")
    def _propagate_seed(self) -> "InjectionConfig":
        """Propagate seed to markov config if not set."""
        if self.seed is not None and self.markov.seed is None:
            object.__setattr__(
                self,
                "markov",
                self.markov.model_copy(update={"seed": self.seed}),
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "markov": self.markov.to_dict(),
            "window": self.window.to_dict(),
            "resample_freq": self.resample_freq,
            "target_features": self.target_features,
            "all_features": self.all_features,
            "interpolation_method": self.interpolation_method,
            "group_column": self.group_column,
            "seed": self.seed,
            "fault_type_mapping": {ft.name: ft.value for ft in FaultType},
        }


class TrainConfig(BaseModel):
    """Configuration for model training.

    Attributes:
        model: Model architecture name.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Optimizer learning rate.
        use_focal_loss: Whether to use focal loss instead of cross-entropy.
        focal_gamma: Focusing parameter for focal loss (higher = more focus on hard examples).
        focal_alpha: Per-class balancing weights for focal loss. None means uniform.
        oversample: Whether to oversample minority (non-NORMAL) classes.
        oversample_ratio: Target ratio of minority to majority samples (1.0 = balanced).
        communication_penalty_weight: Weight for communication auxiliary loss (0 = disabled).
        communication_penalty_mode: ``"linear"`` for L1 penalty or ``"budget_hinge"`` for
            ``relu(ratio - target)^2``.
        target_request_ratio: Target active request ratio for budget_hinge mode.
        gate_entropy_weight: Weight for gate entropy regularization (0 = disabled).
            Positive weight encourages higher gate entropy to prevent collapse to all-zero.
        gumbel_tau_start: Initial Gumbel-Softmax temperature.
        gumbel_tau_end: Final Gumbel-Softmax temperature after annealing.
        gumbel_tau_anneal_epochs: Number of epochs over which to linearly anneal
            temperature from ``gumbel_tau_start`` to ``gumbel_tau_end``.
        checkpoint_monitor: Metric to monitor for model checkpointing
            (``val_loss``, ``val_macro_f1``, or ``val_acc``).
        early_stopping_monitor: Metric to monitor for early stopping
            (``val_loss``, ``val_macro_f1``, or ``val_acc``).
        features: Subset of feature names to train on. None means all features.
        val_ratio: Fraction of training data to use for validation (0.0 = no split).
        seed: Random seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    use_focal_loss: bool = False
    focal_gamma: float = Field(default=2.0, ge=0.0)
    focal_alpha: list[float] | None = None
    oversample: bool = False
    oversample_ratio: float = Field(default=1.0, gt=0.0, le=1.0)
    communication_penalty_weight: float = Field(default=0.0, ge=0.0)
    communication_penalty_mode: str = Field(default="linear", pattern=r"^(linear|budget_hinge)$")
    target_request_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    gate_entropy_weight: float = Field(default=0.0, ge=0.0)
    gumbel_tau_start: float = Field(default=1.0, gt=0.0)
    gumbel_tau_end: float = Field(default=1.0, gt=0.0)
    gumbel_tau_anneal_epochs: int = Field(default=0, ge=0)
    checkpoint_monitor: str = Field(default="val_loss", pattern=r"^(val_loss|val_macro_f1|val_acc)$")
    early_stopping_monitor: str = Field(default="val_loss", pattern=r"^(val_loss|val_macro_f1|val_acc)$")
    features: list[str] | None = None
    val_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    seed: int = 42
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _flatten_config_file_sections(cls, data: Any) -> Any:
        if isinstance(data, dict) and "train" in data:
            train_section = data.get("train") or {}
            if not isinstance(train_section, dict):
                msg = "Train config 'train' section must be a mapping"
                raise ValueError(msg)
            merged = dict(train_section)
            if "model_kwargs" in data:
                merged["model_kwargs"] = data["model_kwargs"]
            return merged
        return data

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "use_focal_loss": self.use_focal_loss,
            "focal_gamma": self.focal_gamma,
            "focal_alpha": self.focal_alpha,
            "oversample": self.oversample,
            "oversample_ratio": self.oversample_ratio,
            "communication_penalty_weight": self.communication_penalty_weight,
            "communication_penalty_mode": self.communication_penalty_mode,
            "target_request_ratio": self.target_request_ratio,
            "gate_entropy_weight": self.gate_entropy_weight,
            "gumbel_tau_start": self.gumbel_tau_start,
            "gumbel_tau_end": self.gumbel_tau_end,
            "gumbel_tau_anneal_epochs": self.gumbel_tau_anneal_epochs,
            "checkpoint_monitor": self.checkpoint_monitor,
            "early_stopping_monitor": self.early_stopping_monitor,
            "features": self.features,
            "val_ratio": self.val_ratio,
            "seed": self.seed,
            "model_kwargs": self.model_kwargs,
        }


class EvaluateConfig(BaseModel):
    """Configuration for model evaluation.

    Attributes:
        batch_size: Evaluation batch size.
    """

    model_config = ConfigDict(frozen=True)

    batch_size: int = Field(default=64, ge=1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_size": self.batch_size,
        }


class OptimizeConfig(BaseModel):
    """Configuration for hyperparameter optimization with Optuna.

    Attributes:
        model: Model architecture to optimize.
        n_trials: Number of Optuna trials.
        timeout: Optimization timeout in seconds (None = unlimited).
        seed: Random seed for sampler reproducibility.
        storage: Optuna storage URL (e.g. ``sqlite:///optuna.db``).
        study_name: Optuna study name. Defaults to ``cesta-<model>``.
        direction: ``minimize`` or ``maximize``.
        metric: Validation metric to optimize. One of ``val_loss``,
            ``val_macro_f1``, ``val_acc``.
        epochs: Number of training epochs per trial.
        sampler: Optuna sampler to use. One of ``tpe``, ``random``.
        pruner: Optuna pruner. One of ``median``, ``none``.
        startup_trials: Number of random trials before TPE/MedianPruner kicks in.
        load_if_exists: Resume an existing study with the same name.
        features: Subset of feature names to train on. None = all features.
    """

    model_config = ConfigDict(frozen=True)

    model: str = "lstm"
    n_trials: int = Field(default=20, ge=1)
    timeout: int | None = None
    seed: int = 42
    storage: str = "sqlite:///optuna.db"
    study_name: str | None = None
    direction: str = Field(default="minimize", pattern=r"^(minimize|maximize)$")
    metric: str = Field(
        default="val_loss",
        pattern=r"^(val_loss|val_macro_f1|val_acc)$",
    )
    epochs: int = Field(default=20, ge=1)
    sampler: str = Field(default="tpe", pattern=r"^(tpe|random)$")
    pruner: str = Field(default="median", pattern=r"^(median|none)$")
    startup_trials: int = Field(default=5, ge=0)
    load_if_exists: bool = True
    features: list[str] | None = None

    @model_validator(mode="after")
    def _align_direction_with_metric(self) -> "OptimizeConfig":
        direction = "minimize" if self.metric == "val_loss" else "maximize"
        if self.direction != direction:
            object.__setattr__(self, "direction", direction)
        return self

    def resolved_study_name(self) -> str:
        """Return the study name, defaulting to ``cesta-<model>``."""
        return self.study_name if self.study_name is not None else f"cesta-{self.model}"

    def resolved_direction(self) -> str:
        """Return the direction inferred from the metric if not overridden."""
        if self.metric == "val_loss":
            return "minimize"
        return "maximize"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "n_trials": self.n_trials,
            "timeout": self.timeout,
            "seed": self.seed,
            "storage": self.storage,
            "study_name": self.study_name,
            "direction": self.direction,
            "metric": self.metric,
            "epochs": self.epochs,
            "sampler": self.sampler,
            "pruner": self.pruner,
            "startup_trials": self.startup_trials,
            "load_if_exists": self.load_if_exists,
            "features": self.features,
        }
