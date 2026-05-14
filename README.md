# CESTA: Communication-Efficient Spatial-Temporal Aggregation

## Overview

A research framework for **sensor fault diagnosis** using deep learning on time-series data. Supports Markov-chain fault injection, temporal and spatio-temporal models, and end-to-end experiment pipelines — from raw sensor readings to evaluated classifiers.

## Features

- **Markov-chain fault injection** — Simulates realistic sensor faults (spike, drift, stuck) with configurable transition probabilities and durations
- **Temporal models** — CNN1D, LSTM, GRU, Transformer, Autoformer, Informer, PatchTST, ModernTCN
- **Spatio-temporal models** — ST-GCN and CESTA with graph-based sensor topology, dynamic edge masks, and per-node classification
- **Focal loss & oversampling** — Handles class imbalance common in fault diagnosis datasets
- **Optuna hyperparameter optimization** — Automated sweep with configurable trials and storage
- **ESP32-S3 firmware** — Rust firmware for real-world DHT11 sensor data collection via MQTT; see [`firmware/README.md`](firmware/README.md)

## Getting Started

### Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) for environment management

### Installation

```bash
git clone https://github.com/Sinner/CESTA.git
cd CESTA
uv sync
```

### Usage

```bash
# 1. Inject faults into raw sensor data
uv run cesta inject intel_lab data/raw/Intel/data.txt data/injected/intel_lab

# 2. (Optional) Add graph topology for spatio-temporal models
uv run cesta prepare graph data/injected/intel_lab data/raw/Intel/connectivity.txt

# 3. Train a model
uv run cesta train config/model/lstm.yaml data/injected/intel_lab

# 4. Evaluate
uv run cesta evaluate --model runs/lstm/<run_id> --data data/injected/intel_lab

# 5. (Optional) Hyperparameter optimization
uv run cesta optimize --data data/injected/intel_lab --n-trials 100
```

## Project Structure

```
src/CESTA/
├── schema/            # Pydantic config models
├── cli/               # Typer CLI (inject, prepare, train, evaluate, optimize)
├── injection/         # Markov generator, fault injectors, registry
├── datasets/
│   ├── raw/           # Dataset loaders: IntelLab, ESP32-DHT11, registry
│   └── injected/      # InjectedDataset, GraphDataset, windowing
├── models/
│   ├── temporal/      # CNN1D, LSTM, GRU, Transformer, Autoformer, Informer, PatchTST, ModernTCN
│   └── spatial/       # ST-GCN, CESTA
├── training/          # Trainer, focal loss, oversampling, callbacks
├── evaluation/        # Metrics, evaluator
├── optimization/      # Optuna sweep
└── seed.py            # Reproducibility utility

firmware/              # ESP32-S3 Rust firmware (esp-idf-hal)
config/                # YAML config files per model
data/                  # Raw datasets and injected outputs
```

## CLI

```
cesta
├── inject              # Run fault injection on a dataset
├── prepare
│   └── graph           # Add graph topology to injected dataset
├── train               # Train a model
├── evaluate            # Evaluate a trained model
├── optimize            # Run Optuna hyperparameter optimization
│   └── show            # Display study results
├── report              # Aggregate run artifacts into comparison reports
└── list                # List datasets, models, metrics, or runs
```

Run `uv run cesta --help` for full options.

## Fault Types

| Fault   | Description                                      |
|---------|--------------------------------------------------|
| Normal  | No fault — clean sensor reading                  |
| Spike   | Sudden transient deviation from true value       |
| Drift   | Gradual cumulative offset over time              |
| Stuck   | Sensor output frozen at a constant value         |

Fault sequences are generated via a **Markov chain** with configurable transition probabilities and per-fault duration distributions.

## Configuration

Training configs are YAML files in `config/model/`:

```bash
uv run cesta train config/model/lstm.yaml data/injected/intel_lab
```

Large command surfaces use config files validated directly by Pydantic. Small utility commands define direct CLI defaults.

## Firmware

See [`firmware/README.md`](firmware/README.md) for firmware requirements, configuration, usage, MQTT payload format, and deployment notes.

## Development

```bash
uv run ruff check src/          # Lint
uv run ruff format src/         # Format
uv run pyright src/             # Type check
```

## Extension

### Add a new dataset

1. Subclass `BaseDataset` in `src/CESTA/datasets/raw/`
2. Implement `name`, `feature_columns`, `group_column`, `timestamp_column`, `load()`, `preprocess()`
3. Add it to `_DATASET_LOADERS` in `datasets/raw/__init__.py`

### Add a new fault type

1. Add value to `FaultType` enum in `schema/types.py`
2. Create injector in `injection/faults.py` subclassing `BaseFaultInjector`
3. Register in `injection/registry.py`
4. Add default config in `MarkovConfig._default_fault_configs()`

### Add a new model

1. Subclass `BaseModel` in `models/temporal/` or `models/spatial/`
2. Set `required_metadata` class variable if the model needs graph or other metadata
3. Register in `models/registry.py`
