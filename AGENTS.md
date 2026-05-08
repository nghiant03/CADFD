# AGENTS.md

AGENTS MUST KEEP THIS FILE UP TO DATE AFTER A CODE CHANGE.
This repository is a research project for fault diagnosis analysis.

## Environment Management

- Use **uv** for Python environment management.

## Code Quality

- Use **ruff** for linting and formatting `.py` source files.
- Use **pyright** for type checking `.py` source files.
- Do **not** use `TYPE_CHECKING` from `typing`. Use `from __future__ import annotations` and lazy imports inside functions instead.
- When changing a function, file, class, or variable, **always reconsider whether its name still accurately describes its purpose**. Rename if the name no longer fits.

## Notebook Conventions

- In every Jupyter notebook, the **import block must be in the top cell**.

## Project Structure

```
src/CADFD/
├── schema/            # Pydantic config models: FaultType, FaultConfig, MarkovConfig, WindowConfig, InjectionConfig
├── cli/               # Typer CLI with subcommands (inject, prepare, train, evaluate)
├── injection/         # Fault injection: Markov generator, fault injectors, registry
├── datasets/          # Dataset loaders and injected containers
│   ├── raw/           # Pre-injection: BaseDataset, IntelLabDataset, ESP32DHT11Dataset, registry
│   └── injected/      # Post-injection: InjectedDataset, GraphDataset, windowing, loading
├── models/            # Deep learning model definitions
│   ├── temporal/      # Temporal models: CNN1D, LSTM, GRU, Transformer, Autoformer, Informer, PatchTST, ModernTCN
│   └── spatial/       # Spatial models: ST-GCN
├── training/          # Trainer, focal loss, oversampling, and callbacks
├── evaluation/        # Metrics, ClassMetrics, evaluator
├── optimization/      # Optuna search spaces and Optimizer for HPO
├── utils.py           # Shared runtime helpers (git/env collectors, run id, sha256)
├── seed.py            # seed_everything() utility for reproducibility

firmware/              # ESP32-S3 Rust firmware (esp-idf-hal, PlatformIO-free)
config/                # YAML config files per model (lstm.yaml, gru.yaml, etc.)
data/                  # Raw datasets and injected outputs
docs/                  # Research plans and experiment documentation, including CESTA
notebooks/             # Jupyter notebooks for analysis
```

## Research Documentation

- `docs/CESTA/PROPOSAL.md` - Research proposal for CESTA, a communication-efficient spatial-temporal method using receiver-side learned request and compression over existing graph edges.
- `docs/CESTA/EXPERIMENT.md` - Experiment plan for CESTA, including baselines, ablations, Pareto selection between Gumbel-Softmax and RL, and TX+RX energy metrics.


## Schema Module (`schema/`)

The `schema/` module contains Pydantic configuration models used by injection, training, and evaluation:

- `FaultType` - Enum: NORMAL=0, SPIKE=1, DRIFT=2, STUCK=3
- `FaultConfig` - Configuration for a single fault type (transition prob, duration, params)
- `MarkovConfig` - Markov chain configuration (list of fault configs, seed)
- `WindowConfig` - Sliding window parameters (size, strides, train ratio, val ratio)
- `InjectionConfig` - Complete injection pipeline config (serializable as metadata)
- `TrainConfig` - Training configuration (model, epochs, batch_size, learning_rate, use_focal_loss, focal_gamma, focal_alpha, oversample, oversample_ratio, features, seed, model_kwargs). Supports `from_yaml(path)` for YAML config files.
- `EvaluateConfig` - Evaluation configuration (batch_size)
- `OptimizeConfig` - Optuna HPO configuration (model, n_trials, timeout, epochs, metric, sampler, pruner, startup_trials, study_name, storage, direction, load_if_exists, features, seed). Provides `resolved_study_name()` and `resolved_direction()` helpers.
- `RunManifest` / `EnvInfo` / `GitInfo` / `DatasetInfo` / `Timing` (`schema/manifest.py`) - Pure pydantic models for the per-run reproducibility manifest written as `manifest.json`. Runtime collectors live in `CADFD.utils`; `DatasetInfo` is produced by `InjectedDataset.describe(path)`.

## Utilities (`utils.py`)

Single-file module with shared runtime helpers, used mainly by the train/evaluate CLIs:

- `collect_git_info(cwd=None)` - commit SHA, branch, dirty flag (via `dulwich`, no `git` binary required)
- `collect_env_info(device)` - python / torch / cuda / host / device name / cadfd version
- `generate_run_id(model, seed, git)` - `<utc_ts>_<model>_seed<seed>_<shortsha>` (non-pure: samples wall clock)
- `utc_now_iso()` - ISO-8601 UTC timestamp
- `sha256_file(path)` - streaming SHA-256 of a file (used by `InjectedDataset.describe`)

## Run Artifacts

Every `cadfd train run` invocation creates a dedicated run directory; runs are never overwritten.

```
runs/<model>/<utc_ts>_<model>_seed<seed>_<shortsha>/
├── weight.pt              # best-val-loss checkpoint
├── config.json            # {model_name, model_config, train_config}
├── history.jsonl          # one JSON TrainMetrics per epoch (HistoryCallback)
├── manifest.json          # RunManifest: git, env, dataset hash, timing, configs
├── eval_metrics.json      # loss, accuracy, macro_f1, per_class, class_names, confusion_matrix, train_config, injection_config
└── predictions.npz        # y_true, y_pred (int32), y_prob (float32)
```

`cadfd evaluate run` writes the same artifacts (plus a `kind="evaluate"` `manifest.json`) into a new run subdirectory when `--output` is provided; otherwise it writes in-place alongside the loaded model.

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

Organized into two sub-packages by pipeline stage:

### Raw Sub-package (`datasets/raw/`)

Pre-injection dataset loaders and registry.

- `BaseDataset` (`raw/base.py`) - Abstract base for raw dataset loaders: `name`, `feature_columns`, `group_column`, `timestamp_column`, `load()`, `preprocess()`.
- `IntelLabDataset` (`raw/intel_lab.py`) - Concrete loader for Intel Berkeley Research Lab sensor data.
- `ESP32DHT11Dataset` (`raw/esp32_dht11.py`) - Loader for ESP32-S3 DHT11 sensor readings (CSV from MQTT subscriber). Features: `temperature`, `humidity`. Groups by `device_id`.
- `register_dataset` / `get_dataset` / `list_datasets` (`raw/registry.py`) - Dynamic dataset registry.

### Injected Sub-package (`datasets/injected/`)

Post-injection containers, graph topology, and windowing.

- `InjectedDataset` (`injected/tabular.py`) - Container with injected DataFrame + config + save/load. Has `.prepare(window_config, features, required_metadata) -> WindowedSplits` for per-group chronological windowing.
- `GraphDataset` (`injected/graph.py`) - Subclass of `InjectedDataset` that adds graph topology (adjacency matrix, node IDs, threshold). Overrides `.prepare()` for graph-aligned windowing (concatenates all sensor features per timestep, keeps **per-node labels** of shape `(num_windows, window_size, num_nodes)`). When `required_metadata` does not include `"graph"`, delegates to `InjectedDataset.prepare()` so non-graph models work on graph datasets without shape mismatch. Returns `GraphMetadata` in `WindowedSplits.metadata["graph"]`. Built via `GraphDataset.from_connectivity(path, connectivity_path, threshold)` or loaded from disk with `GraphDataset.load(path)`.
- `GraphMetadata` (`injected/graph.py`) - Typed dataclass holding `adjacency`, `node_ids`, `num_nodes`, `threshold`. Stored in `WindowedSplits.metadata["graph"]` by `GraphDataset.prepare()`.
- `WindowedSplits` (`injected/windowed.py`) - Unified dataclass holding windowed data partitions + `metadata` dict. Includes input-shape metadata and split-availability flags.
- `load_adjacency_matrix` (`injected/graph.py`) - Loads binary adjacency matrix from a connectivity data file (whitespace-separated: `source dest probability`), thresholds by connectivity probability.
- `load_dataset` (`injected/loading.py`) - Loads the appropriate dataset variant (`InjectedDataset` or `GraphDataset`) based on which files exist on disk.
- `validate_features` (`injected/windowed.py`) - Shared feature-name validation used by both `InjectedDataset.prepare()` and `GraphDataset.prepare()`.
- `collect_splits` (`injected/windowed.py`) - Shared helper to concatenate per-group window parts into final arrays with correct empty fallbacks. Accepts `label_trailing_shape` for per-node label dimensions.

### Data Preparation Pattern

All dataset types expose a `.prepare(required_metadata=...)` method returning `WindowedSplits`. The CLI calls `load_dataset(path)` which returns the right variant, then `dataset.prepare(required_metadata=model_cls.required_metadata)` dispatches polymorphically. When a `GraphDataset` receives `required_metadata` without `"graph"`, it falls back to per-group tabular windowing so non-graph models work seamlessly. Graph metadata travels via `WindowedSplits.metadata["graph"]`.

`create_model` accepts `metadata` and automatically validates model requirements and extracts architecture-specific kwargs (e.g. `num_nodes`, `adjacency` for GCN).

```python
dataset = load_dataset(data)
model_cls = get_model_class(config.model)
prepared = dataset.prepare(features=config.features,
                           required_metadata=model_cls.required_metadata)
net = create_model(config.model, input_size=prepared.input_size,
                   num_classes=num_classes, metadata=prepared.metadata)
```

## Training Module (`training/`)

- `FocalLoss` (`loss.py`) - Focal loss for imbalanced multi-class classification. gamma=0 recovers CE.
- `oversample_minority` (`oversampling.py`) - Window-level oversampling: duplicates windows containing any non-NORMAL label until minority count reaches `ratio * majority_count`.
- `Trainer` (`trainer.py`) - Full training loop with Adam optimizer, optional focal loss, optional oversampling, and callback hooks. Returns `TrainResult` with per-epoch history. Expects val data passed explicitly (produced by `dataset.prepare()`).
- `TrainingCallback` (`callbacks.py`) - Abstract base; implementations: `LoggingCallback`, `EarlyStoppingCallback`, `CheckpointCallback`, `HistoryCallback` (per-epoch JSONL dump of `TrainMetrics`).

## Evaluation Module (`evaluation/`)

- `compute_class_metrics` (`metrics.py`) - Per-class precision, recall, F1, support from prediction tensors.
- `macro_f1` (`metrics.py`) - Macro-averaged F1 from per-class metrics.
- `Evaluator` (`evaluator.py`) - Runs inference on a dataset, computes all metrics, captures predictions (y_true, y_pred, y_prob), returns `EvalResult`. Handles device placement.
- `EvalResult` (`evaluator.py`) - Dataclass holding loss, accuracy, macro_f1, per-class ClassMetrics, y_true, y_pred, y_prob. Has `save(path)` to persist `eval_metrics.json` (metrics + configs) and `predictions.npz` (numpy arrays). Has `load(path)` class method.

## Firmware Module (`firmware/`)

Rust firmware for ESP32-S3 with DHT11 sensor, built with `esp-idf-hal` (std environment).

### Structure

```
firmware/
├── Cargo.toml            # Dependencies: esp-idf-svc, esp-idf-hal, serde_json
├── build.rs              # ESP-IDF build integration via embuild
├── sdkconfig.defaults    # ESP-IDF Kconfig (WiFi, SNTP, MQTT)
├── .cargo/config.toml    # Target: xtensa-esp32s3-espidf
└── src/
    ├── main.rs           # Entry point: init → WiFi → NTP → MQTT loop
    ├── config.rs         # WiFi SSID/password, MQTT broker, device ID, DHT pin
    ├── wifi.rs           # BlockingWifi connection via esp-idf-svc
    ├── mqtt.rs           # EspMqttClient connection and publish
    └── dht.rs            # Bit-banged DHT11 protocol over GPIO
```

### MQTT Payload

Publishes JSON to `cafd/readings/<device_id>` every 30s:
```json
{"device_id": "esp32_01", "timestamp": 1718000000, "temperature": 25.3, "humidity": 60.1}
```

### Build & Flash

Requires `espup` (Rust ESP toolchain) and `espflash`:
```bash
cd firmware
cargo check
cargo build --release
espflash flash target/xtensa-esp32s3-espidf/release/cafd-firmware --monitor
```

### Lab Server Stack

ESP32 devices connect via WiFi to an on-prem MQTT broker (Mosquitto). Recommended stack:
- **Mosquitto** — MQTT broker
- **Telegraf** — MQTT → InfluxDB bridge
- **InfluxDB** — Time-series storage
- **Grafana** — Dashboard
- **Python MQTT subscriber** — Export to `data/raw/esp32_dht11/` CSV for CADFD pipeline

## Workflow

1. **Fault Injection**: `uv run cadfd inject run intel_lab data/raw/Intel/data.txt data/injected/intel_lab`
2. **Graph Preparation** (optional): `uv run cadfd prepare graph data/injected/intel_lab data/raw/Intel/connectivity.txt`
3. **Training**: `uv run cadfd train run lstm data/injected/intel_lab` or with config: `uv run cadfd train run lstm data/injected/intel_lab --config config/lstm.yaml`
4. **Hyperparameter Search** (optional): `uv run cadfd optimize run --data data/injected/intel_lab --model lstm --n-trials 20 --epochs 10`
5. **Evaluation**: `uv run cadfd evaluate run --model runs/lstm/<run_id> --data data/injected/intel_lab`

## Optimization Module (`optimization/`)

Optuna-driven hyperparameter search. Each trial samples both training-loop
hyperparameters (learning rate, batch size, focal loss, oversampling) and
model-architecture hyperparameters from a per-model search space, then trains
a fresh model with `Trainer` for `OptimizeConfig.epochs` epochs and reports
the configured validation metric back to Optuna.

- `Optimizer` (`optimizer.py`) - Builds the study (sampler/pruner/storage),
  loads the dataset once, and runs `n_trials`. Reports per-epoch metric to
  Optuna and supports pruning via `_OptunaPruneCallback` (a `TrainingCallback`
  that returns `False` when `trial.should_prune()` fires, stopping the
  trainer early and raising `optuna.TrialPruned`).
- `search_spaces.py` - Per-model `(trial) -> dict` functions registered for
  `lstm`, `gru`, `cnn1d`, `transformer`, `autoformer`, `informer`, `patchtst`,
  `modern_tcn`, `stgcn`. `suggest_train_hyperparams` covers shared training
  knobs. Use `register_search_space(name, fn)` to add new model spaces.
- Studies persist to `OptimizeConfig.storage` (default `sqlite:///optuna.db`)
  under `OptimizeConfig.resolved_study_name()` (default `cadfd-<model>`),
  so runs can be resumed (`load_if_exists=True`).

### CLI

```
cadfd optimize run --data <dir> [--model lstm] [--n-trials N] [--epochs E]
                   [--metric val_loss|val_macro_f1|val_acc] [--sampler tpe|random]
                   [--pruner median|none] [--study-name NAME] [--storage URL]
                   [--timeout SECONDS] [--seed S] [--output best_params.json]
cadfd optimize show <study_name> [--storage URL] [--top K]
```

The `--metric` option auto-aligns the study direction (`val_loss` →
minimize, others → maximize). The selected metric must be available in
`TrainMetrics`; the dataset must have a non-empty validation split.

## CLI Structure

The CLI uses **Typer** with a centralized command namespace:

```
cadfd                    # Main entry point
├── inject              # Fault injection subcommands
│   ├── run             # Run fault injection
│   └── list            # List available datasets
├── prepare             # Data preparation subcommands
│   └── graph           # Add graph topology to injected dataset
├── train               # Training subcommands
│   ├── run             # Train a model
│   └── list            # List available models
└── evaluate            # Evaluation subcommands
    ├── run             # Evaluate a model
    └── list            # List available metrics
```

Run `cadfd --help` or `cadfd <subcommand> --help` for detailed options.

## Adding New Fault Types

1. Add new value to `FaultType` enum in `schema/types.py`.
2. Create injector class in `injection/faults.py` subclassing `BaseFaultInjector`.
3. Register in `injection/registry.py` with `register_fault()`.
4. Add default config in `MarkovConfig._default_fault_configs()`.

## Fault Injection Parameters

Per-event randomization and per-mote scaling are first-class:

- **Per-event random ranges**: `magnitude_range`, `drift_rate_range` are tuples
  `(min, max)`; the injector samples a fresh value per fault event.
- **Per-mote sigma scaling**: `magnitude_sigma_range` (SPIKE) and
  `drift_rate_sigma_range` (DRIFT) are tuples interpreted as multipliers on
  the mote's local std. They override the absolute ranges when present.
  `FaultInjector` (`injection/injector.py`) computes per-(mote, feature) std
  and median from the NORMAL portion and injects them as `_mote_std` /
  `_mote_median` into `params` before calling each injector's `apply()`.
- **STUCK jitter**: `jitter_sigma_factor` adds Gaussian noise of std
  `factor * _mote_std` around the frozen value to simulate subtle freezes.
- **Defaults** (`MarkovConfig._default_fault_configs`) are tuned for a
  challenging benchmark: ~5-7% combined fault ratio, sigma-relative
  magnitudes, randomized drift rates, jittered stuck.

## Adding New Datasets

1. Implement a new dataset class in `src/CADFD/datasets/raw/` subclassing `BaseDataset`.
2. Implement: `name`, `feature_columns`, `group_column`, `timestamp_column`, `load()`, `preprocess()`.
3. Register in `datasets/raw/registry.py` with `register_dataset()`.

## Model Metadata Requirements

Models declare required dataset metadata via `required_metadata` (a `ClassVar[set[str]]` on `BaseModel`). The model registry (`create_model`) validates these before construction and extracts architecture-specific kwargs automatically.

```python
class STGCNClassifier(BaseModel):
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
--evaluation-stride    Stride for held-out windows (default from WindowConfig: 60)
--spike-prob           Transition probability to spike (default from MarkovConfig)
--spike-duration       Average spike duration (default from MarkovConfig)
--spike-magnitude-min  Minimum absolute spike offset (default from MarkovConfig)
--spike-magnitude-max  Maximum absolute spike offset (default from MarkovConfig)
--spike-sigma-min      Min sigma multiplier on per-mote std (overrides absolute)
--spike-sigma-max      Max sigma multiplier on per-mote std
--drift-prob           Transition probability to drift (default from MarkovConfig)
--drift-duration       Average drift duration (default from MarkovConfig)
--drift-rate-min       Minimum absolute drift rate per timestep
--drift-rate-max       Maximum absolute drift rate per timestep
--drift-sigma-min      Min sigma multiplier on per-mote std (overrides absolute)
--drift-sigma-max      Max sigma multiplier on per-mote std
--stuck-prob           Transition probability to stuck (default from MarkovConfig)
--stuck-duration       Average stuck duration (default from MarkovConfig)
--stuck-jitter-sigma-factor  Sigma fraction for stuck-with-noise jitter
-c, --config           Path to JSON config file (CLI args override)
```
