# AGENTS.md

AGENTS MUST KEEP THIS FILE UP TO DATE AFTER A CODE CHANGE.
This repository is a research project for fault diagnosis analysis.

## Environment Management

- Use **uv** for Python environment management.

## Code Quality

- Use **ruff** for linting and formatting `.py` source files.
- Use **pyright** for type checking `.py` source files.
- Run tests with `uv run pytest`.
- Do **not** use `TYPE_CHECKING` from `typing`. Use `from __future__ import annotations` and lazy imports inside functions instead.
- When changing a function, file, class, or variable, **always reconsider whether its name still accurately describes its purpose**. Rename if the name no longer fits.

## Notebook Conventions

- In every Jupyter notebook, the **import block must be in the top cell**.

## Project Structure

```
src/DiFD/
├── schema/            # Pydantic config models: FaultType, FaultConfig, MarkovConfig, WindowConfig, InjectionConfig
├── cli/               # Typer CLI with subcommands (inject, prepare, train, evaluate, optimize)
├── injection/         # Fault injection: Markov generator, fault injectors, registry
├── datasets/          # Dataset loaders, InjectedDataset, GraphDataset, GraphMetadata, WindowedSplits, windowing, loading
├── models/            # Deep learning model definitions (LSTM, GRU, Autoformer, Transformer, Informer, PatchTST, GCN)
├── training/          # Trainer, focal loss, oversampling, and callbacks
├── evaluation/        # Metrics and evaluator
├── optimization/      # Optuna hyperparameter sweep
├── seed.py            # seed_everything() utility for reproducibility

data/                  # Raw datasets and injected outputs
tests/                 # Unit tests per module
notebooks/             # Jupyter notebooks for analysis
```

## Schema Module (`schema/`)

The `schema/` module contains Pydantic configuration models used by injection, training, and evaluation:

- `FaultType` - Enum: NORMAL=0, SPIKE=1, DRIFT=2, STUCK=3
- `FaultConfig` - Configuration for a single fault type (transition prob, duration, params)
- `MarkovConfig` - Markov chain configuration (list of fault configs, seed)
- `WindowConfig` - Sliding window parameters (size, strides, train ratio, val ratio)
- `InjectionConfig` - Complete injection pipeline config (serializable as metadata)
- `TrainConfig` - Training configuration (model, epochs, batch_size, learning_rate, use_focal_loss, focal_gamma, focal_alpha, oversample, oversample_ratio, features, seed)
- `EvaluateConfig` - Evaluation configuration (batch_size)
- `OptimizeConfig` - Optimization configuration (model, n_trials, seed, storage)

## Configuration Design Pattern

**Single Source of Truth (SSOT)**: All default values live exclusively in Pydantic schema classes (`schema/config.py`, `schema/types.py`). CLI modules use `None` as default and fall back to schema defaults.

**Pattern**:
```python
# CLI: Use None defaults
@app.command()
def run(
    window_size: Optional[int] = None,  # NOT = 60
    ...
):
    defaults = WindowConfig()
    config = WindowConfig(
        window_size=window_size if window_size is not None else defaults.window_size,
    )
```

**Rationale**:
- Prevents value drift between CLI and schema
- Single place to update defaults
- Schema classes document the canonical defaults
- CLI `--help` can reference schema or show "default: from config"

## Datasets Module (`datasets/`)

- `InjectedDataset` (`injected.py`) - Container with injected DataFrame + config + save/load. Has `.prepare(window_config, features) -> WindowedSplits` for per-group chronological windowing.
- `GraphDataset` (`graph.py`) - Subclass of `InjectedDataset` that adds graph topology (adjacency matrix, node IDs, threshold). Overrides `.prepare()` for graph-aligned windowing (concatenates all sensor features per timestep). Returns `GraphMetadata` in `WindowedSplits.metadata["graph"]`. Built via `GraphDataset.from_connectivity(path, connectivity_path, threshold)` or loaded from disk with `GraphDataset.load(path)`.
- `GraphMetadata` (`graph.py`) - Typed dataclass holding `adjacency`, `node_ids`, `num_nodes`, `threshold`. Stored in `WindowedSplits.metadata["graph"]` by `GraphDataset.prepare()`.
- `WindowedSplits` (`windowed.py`) - Unified dataclass holding windowed `X_train/y_train/X_val/y_val/X_test/y_test` arrays + `metadata` dict. Properties: `input_size`, `has_val`, `has_test`.
- `load_adjacency_matrix` (`graph.py`) - Loads binary adjacency matrix from a connectivity data file (whitespace-separated: `source dest probability`), thresholds by connectivity probability.
- `load_dataset` (`loading.py`) - Loads the appropriate dataset variant (`InjectedDataset` or `GraphDataset`) based on which files exist on disk.
- `validate_features` (`windowed.py`) - Shared feature-name validation used by both `InjectedDataset.prepare()` and `GraphDataset.prepare()`.
- `collect_splits` (`windowed.py`) - Shared helper to concatenate per-group window parts into final arrays with correct empty fallbacks.

### Data Preparation Pattern

All dataset types expose a `.prepare()` method returning `WindowedSplits`. The CLI calls `load_dataset(path)` which returns the right variant, then `dataset.prepare(...)` dispatches polymorphically. Graph metadata travels via `WindowedSplits.metadata["graph"]`.

`create_model` accepts `metadata` and automatically validates model requirements and extracts architecture-specific kwargs (e.g. `num_nodes`, `adjacency` for GCN).

```python
dataset = load_dataset(data)
prepared = dataset.prepare(features=config.features)
net = create_model(config.model, input_size=prepared.input_size,
                   num_classes=num_classes, metadata=prepared.metadata)
```

## Training Module (`training/`)

- `FocalLoss` (`loss.py`) - Focal loss for imbalanced multi-class classification. gamma=0 recovers CE.
- `oversample_minority` (`oversampling.py`) - Window-level oversampling: duplicates windows containing any non-NORMAL label until minority count reaches `ratio * majority_count`.
- `Trainer` (`trainer.py`) - Full training loop with Adam optimizer, optional focal loss, optional oversampling, and callback hooks. Returns `TrainResult` with per-epoch history. Expects val data passed explicitly (produced by `dataset.prepare()`).
- `TrainingCallback` (`callbacks.py`) - Abstract base; implementations: `LoggingCallback`, `EarlyStoppingCallback`, `CheckpointCallback`.

## Evaluation Module (`evaluation/`)

- `compute_class_metrics` (`metrics.py`) - Per-class precision, recall, F1, support from prediction tensors.
- `macro_f1` (`metrics.py`) - Macro-averaged F1 from per-class metrics.
- `Evaluator` (`evaluator.py`) - Runs inference on a dataset, computes all metrics, captures predictions (y_true, y_pred, y_prob), returns `EvalResult`. Handles device placement.
- `EvalResult` (`evaluator.py`) - Dataclass holding loss, accuracy, macro_f1, per-class ClassMetrics, y_true, y_pred, y_prob. Has `save(path)` to persist `eval_metrics.json` (metrics + configs) and `predictions.npz` (numpy arrays). Has `load(path)` class method.

## Workflow

1. **Fault Injection**: `uv run difd inject run intel_lab data/raw/Intel/data.txt data/injected/intel_lab`
2. **Graph Preparation** (optional): `uv run difd prepare graph data/injected/intel_lab data/raw/Intel/connectivity.txt`
3. **Training**: `uv run difd train run lstm data/injected/intel_lab`
4. **Evaluation**: `uv run difd evaluate run --model models/lstm --data data/injected/intel_lab`
5. **Optimization**: `uv run difd optimize run --data data/injected/intel_lab --n-trials 100`

## CLI Structure

The CLI uses **Typer** with a centralized command namespace:

```
difd                    # Main entry point
├── inject              # Fault injection subcommands
│   ├── run             # Run fault injection
│   └── list-datasets   # List available datasets
├── prepare             # Data preparation subcommands
│   └── graph           # Add graph topology to injected dataset
├── train               # Training subcommands
│   ├── run             # Train a model
│   └── list-models     # List available models
├── evaluate            # Evaluation subcommands
│   ├── run             # Evaluate a model
│   └── metrics         # List available metrics
└── optimize            # Hyperparameter optimization
    ├── run             # Run Optuna optimization
    └── show            # Show study results
```

Run `difd --help` or `difd <subcommand> --help` for detailed options.

## Adding New Fault Types

1. Add new value to `FaultType` enum in `schema/types.py`.
2. Create injector class in `injection/faults.py` subclassing `BaseFaultInjector`.
3. Register in `injection/registry.py` with `register_fault()`.
4. Add default config in `MarkovConfig._default_fault_configs()`.

## Adding New Datasets

1. Implement a new dataset class in `src/DiFD/datasets/` subclassing `BaseDataset`.
2. Implement: `name`, `feature_columns`, `group_column`, `timestamp_column`, `load()`, `preprocess()`.
3. Register in `datasets/registry.py` with `register_dataset()`.

## Model Metadata Requirements

Models declare required dataset metadata via `required_metadata` (a `ClassVar[set[str]]` on `BaseModel`). The model registry (`create_model`) validates these before construction and extracts architecture-specific kwargs automatically.

```python
class GCNClassifier(BaseModel):
    required_metadata: ClassVar[set[str]] = {"graph"}
```

To add a new model that needs special metadata:
1. Set `required_metadata` on the model class.
2. Add extraction logic to `_extract_metadata_kwargs` in `models/registry.py`.

## CLI Options (inject run)

CLI options use `None` defaults; actual defaults come from schema classes.

```
DATASET                Dataset name (required positional argument)
DATA_PATH              Path to raw data file (required positional argument)
OUTPUT                 Output path for .npz file (required positional argument)
-s, --seed             Random seed for reproducibility
--resample-freq        Resampling frequency (default from InjectionConfig: 30s)
-t, --target-features  Features to inject faults into
-a, --all-features     All features to include in output
-w, --window-size      Window size in timesteps (default from WindowConfig: 60)
--train-stride         Stride for training windows (default from WindowConfig: 10)
--test-stride          Stride for test windows (default from WindowConfig: 60)
--spike-prob           Transition probability to spike (default from MarkovConfig: 0.02)
--drift-prob           Transition probability to drift (default from MarkovConfig: 0.01)
--stuck-prob           Transition probability to stuck (default from MarkovConfig: 0.015)
-c, --config           Path to JSON config file (CLI args override)
```
