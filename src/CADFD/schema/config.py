"""Pipeline configuration classes.

This module defines configuration classes for all pipeline phases:
injection, training, evaluation, and optimization.

Each config class owns its default values (Single Source of Truth).
CLI modules should use None defaults and fall back to these schema defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from CADFD.schema.types import FaultType, MarkovConfig, WindowConfig


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
    all_features: list[str] = Field(
        default_factory=lambda: ["temp", "humid", "light", "volt"]
    )
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InjectionConfig":
        """Reconstruct from dictionary."""
        defaults = cls()
        return cls(
            markov=MarkovConfig.from_dict(data.get("markov", {})),
            window=WindowConfig.from_dict(data.get("window", {})),
            resample_freq=data.get("resample_freq", defaults.resample_freq),
            target_features=data.get("target_features", defaults.target_features),
            all_features=data.get("all_features", defaults.all_features),
            interpolation_method=data.get(
                "interpolation_method", defaults.interpolation_method
            ),
            group_column=data.get("group_column", defaults.group_column),
            seed=data.get("seed", defaults.seed),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InjectionConfig":
        """Load injection configuration from a YAML file.

        The YAML file mirrors the structure produced by :meth:`to_dict`.
        Tuple-like values (e.g. ``magnitude_range: [a, b]``) may be written
        as YAML lists.
        """
        import yaml

        resolved = Path(path)
        if not resolved.exists():
            msg = f"Injection config file not found: {resolved}"
            raise FileNotFoundError(msg)

        with resolved.open() as fh:
            raw = yaml.safe_load(fh) or {}

        return cls.from_dict(raw)


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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load configuration from a YAML file.

        The YAML file should contain a ``train`` section for training
        parameters and an optional ``model_kwargs`` section for
        model-specific architecture parameters.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Populated TrainConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        import yaml

        resolved = Path(path)
        if not resolved.exists():
            msg = f"Config file not found: {resolved}"
            raise FileNotFoundError(msg)

        with resolved.open() as fh:
            raw = yaml.safe_load(fh) or {}

        train_section = raw.get("train", {})
        model_kwargs_section = raw.get("model_kwargs", {})

        merged: dict[str, Any] = {**train_section}
        if model_kwargs_section:
            merged["model_kwargs"] = model_kwargs_section

        return cls.from_dict(merged)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        """Reconstruct from dictionary."""
        model = data.get("model")
        if model is None:
            msg = "'model' key is required in TrainConfig data"
            raise ValueError(msg)
        defaults = cls(model=model)
        return cls(
            model=model,
            epochs=data.get("epochs", defaults.epochs),
            batch_size=data.get("batch_size", defaults.batch_size),
            learning_rate=data.get("learning_rate", defaults.learning_rate),
            use_focal_loss=data.get("use_focal_loss", defaults.use_focal_loss),
            focal_gamma=data.get("focal_gamma", defaults.focal_gamma),
            focal_alpha=data.get("focal_alpha", defaults.focal_alpha),
            oversample=data.get("oversample", defaults.oversample),
            oversample_ratio=data.get("oversample_ratio", defaults.oversample_ratio),
            communication_penalty_weight=data.get(
                "communication_penalty_weight", defaults.communication_penalty_weight
            ),
            communication_penalty_mode=data.get(
                "communication_penalty_mode", defaults.communication_penalty_mode
            ),
            target_request_ratio=data.get(
                "target_request_ratio", defaults.target_request_ratio
            ),
            gate_entropy_weight=data.get(
                "gate_entropy_weight", defaults.gate_entropy_weight
            ),
            gumbel_tau_start=data.get(
                "gumbel_tau_start", defaults.gumbel_tau_start
            ),
            gumbel_tau_end=data.get(
                "gumbel_tau_end", defaults.gumbel_tau_end
            ),
            gumbel_tau_anneal_epochs=data.get(
                "gumbel_tau_anneal_epochs", defaults.gumbel_tau_anneal_epochs
            ),
            checkpoint_monitor=data.get(
                "checkpoint_monitor", defaults.checkpoint_monitor
            ),
            early_stopping_monitor=data.get(
                "early_stopping_monitor", defaults.early_stopping_monitor
            ),
            features=data.get("features", defaults.features),
            val_ratio=data.get("val_ratio", defaults.val_ratio),
            seed=data.get("seed", defaults.seed),
            model_kwargs=data.get("model_kwargs", defaults.model_kwargs),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluateConfig":
        """Reconstruct from dictionary."""
        defaults = cls()
        return cls(
            batch_size=data.get("batch_size", defaults.batch_size),
        )


class OptimizeConfig(BaseModel):
    """Configuration for hyperparameter optimization with Optuna.

    Attributes:
        model: Model architecture to optimize.
        n_trials: Number of Optuna trials.
        timeout: Optimization timeout in seconds (None = unlimited).
        seed: Random seed for sampler reproducibility.
        storage: Optuna storage URL (e.g. ``sqlite:///optuna.db``).
        study_name: Optuna study name. Defaults to ``cadfd-<model>``.
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

    def resolved_study_name(self) -> str:
        """Return the study name, defaulting to ``cadfd-<model>``."""
        return self.study_name if self.study_name is not None else f"cadfd-{self.model}"

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizeConfig":
        """Reconstruct from dictionary."""
        defaults = cls()
        return cls(
            model=data.get("model", defaults.model),
            n_trials=data.get("n_trials", defaults.n_trials),
            timeout=data.get("timeout", defaults.timeout),
            seed=data.get("seed", defaults.seed),
            storage=data.get("storage", defaults.storage),
            study_name=data.get("study_name", defaults.study_name),
            direction=data.get("direction", defaults.direction),
            metric=data.get("metric", defaults.metric),
            epochs=data.get("epochs", defaults.epochs),
            sampler=data.get("sampler", defaults.sampler),
            pruner=data.get("pruner", defaults.pruner),
            startup_trials=data.get("startup_trials", defaults.startup_trials),
            load_if_exists=data.get("load_if_exists", defaults.load_if_exists),
            features=data.get("features", defaults.features),
        )
