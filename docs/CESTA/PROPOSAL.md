# Research proposal: CESTA

## Name

**CESTA**: Communication-Efficient Spatial-Temporal Aggregation.

The method name is intentionally one uppercase word for paper and implementation consistency.

## Motivation

Fault diagnosis in sensor networks benefits from temporal modeling, but dense spatial models can waste energy by communicating with all neighbors even when local temporal evidence is sufficient. Current project results show strong temporal baselines and weak ST-GCN performance, so a convincing Q1-level contribution should not merely beat ST-GCN. The method should exceed temporal-only macro-F1 while reducing communication energy relative to dense spatial communication.

## Research questions

1. Can a lightweight spatial-temporal model exceed temporal-only macro-F1 by selectively requesting neighbor information?
2. Can learned request and compression decisions reduce measured and theoretical communication energy while preserving or improving accuracy?
3. Does a trainable communication controller outperform rule-based uncertainty/change-triggered communication at the same energy budget?
4. Is Gumbel-Softmax or reinforcement learning better for learning the communication policy under a Pareto criterion?

## Hypotheses

1. CESTA improves average macro-F1 over the best temporal-only model across Intel fault ratios.
2. CESTA reduces communication energy compared with dense spatial message passing by requesting only useful neighbor embeddings and compressing transmitted messages.
3. Receiver-side local uncertainty and temporal embedding state are sufficient to decide when neighbor information is useful without inspecting neighbor embeddings before communication.
4. Gumbel-Softmax will be easier to train and more reproducible than RL, but the final main design will be selected by Pareto dominance rather than preference.

## Success criteria

Primary success criterion:

```text
Average Δ macro-F1 >= +0.01 over the best temporal-only model per fault ratio
and Pareto-superior energy/accuracy behavior against dense spatial communication.
```

Preferred Q1-level target:

```text
Average Δ macro-F1 >= +0.03 to +0.04 over the best temporal-only model per fault ratio
and substantial measured communication-energy reduction against dense spatial communication.
```

Secondary criteria:

- improve over a fixed temporal backbone used inside CESTA;
- outperform or match required spatial baselines including HiFiNet, if HiFiNet targets sensor/graph fault diagnosis;
- reduce theoretical TX+RX radio energy compared with dense spatial message passing;
- show lower or comparable edge inference cost than dense spatial models;
- remain lightweight enough to classify as edge-oriented, with ESP32-S3 as a loose lower target.

## Core design

CESTA is a distributed receiver-side request model. Each node first encodes its own local temporal window, estimates local diagnosis uncertainty, and decides which existing graph neighbors to request from and at what compression ratio.

The graph topology is fixed to the existing connectivity graph. CESTA learns dynamic request and compression decisions over these candidate edges; it does not perform unconstrained graph discovery in the main method.

### Local temporal encoder

Each node processes its own window:

```text
x_i ∈ R^{T × F}
h_i,1:T = TemporalEncoder(x_i)
```

Recommended first encoders:

- lightweight GRU;
- lightweight depthwise-separable TCN as a later alternative.

The encoder should be small enough that gains cannot be explained only by a larger temporal backbone.

### Receiver-side communication decision module

For receiver node `i` and candidate neighbor `j`, the gate uses only local information from receiver `i`:

```text
g_i,j = RequestGate(pool(h_i,1:T), uncertainty_i, edge_features_i,j)
r_i,j = CompressionGate(pool(h_i,1:T), uncertainty_i, edge_features_i,j)
```

where:

- `g_i,j` is request/no-request;
- `r_i,j` selects compression ratio;
- `uncertainty_i` can include entropy or margin from the local classifier;
- `edge_features_i,j` may include existing connectivity weight, hop flag, or normalized node degree.

The gate does not inspect neighbor embeddings before requesting. This avoids hidden pre-communication cost and keeps the energy model honest.

### Message compression

If requested, neighbor `j` sends a compressed temporal embedding:

```text
m_j→i = Compress_r(h_j,1:T)
```

Compression ratios should include staged options such as:

```text
r ∈ {0.25, 0.5, 1.0}
```

The aggregation module projects received messages back to the common hidden dimension before fusion.

### Neighbor aggregation

CESTA aggregates only requested messages:

```text
a_i,1:T = Aggregate({Up(m_j→i) | j ∈ N(i), g_i,j = 1})
```

The first aggregation design should be lightweight:

- normalized sum or single-head additive attention;
- no multi-head attention in the initial implementation;
- optional learned fusion gate between local and neighbor context.

### Classifier

The final classifier combines local and aggregated spatial context:

```text
z_i,1:T = Fusion(h_i,1:T, a_i,1:T)
y_i,1:T = Classifier(z_i,1:T)
```

The output remains many-to-many per node and timestep to match current CADFD graph-model evaluation:

```text
(batch, window_size, num_nodes, num_classes)
```

## Training strategies

Two controller-training options will be explored:

1. **Gumbel-Softmax / straight-through estimators** for differentiable request and compression decisions.
2. **Reinforcement learning** for non-differentiable energy/accuracy reward optimization.

The main paper design will be whichever is Pareto-superior:

```text
higher macro-F1 at equal/lower measured energy
or lower measured energy at equal/higher macro-F1.
```

If tied, prefer Gumbel-Softmax because it is simpler, more reproducible, and easier to integrate into the existing supervised training loop.

## Energy model

The theoretical communication energy model should count both transmission and receiving energy for every active message.

For a message of `k` bits over distance `d`:

```text
E_tx(k, d) = E_elec · k + E_amp · k · d^n
E_rx(k) = E_elec · k
E_msg(k, d) = E_tx(k, d) + E_rx(k)
```

Use the cited first-order radio model from Mahajan et al. when writing the paper:

- free-space mode for short distances;
- multipath mode for long distances;
- threshold distance `d0 = sqrt(E_fs / E_mp)`;
- include TX and RX energy in all CESTA and baseline communication totals.

The paper should also measure on-device energy on ESP32-S3 or the smallest feasible edge-class target available.

## Efficiency metrics

Primary metrics should be based on energy consumption:

- measured on-device energy per inference/window;
- theoretical communication energy per window using TX+RX;
- energy reduction versus dense spatial communication;
- macro-F1 per Joule or macro-F1 per communication-energy unit;
- average requested edges per node/window;
- transmitted bits per node/window;
- compression-ratio distribution;
- parameter count and latency.

## Required baselines

1. Best temporal-only model per fault ratio.
2. Fixed temporal backbone matching CESTA's encoder without communication.
3. ST-GCN.
4. HiFiNet, if it targets sensor/graph fault diagnosis.
5. Dense learned message passing with all existing graph edges and full embeddings.
6. Rule-based uncertainty/change-triggered communication with matched communication budget.
7. Static top-k graph communication using strongest connectivity edges.
8. Random communication at matched average budget.

## Scope

In scope:

- Intel graph datasets across all four fault ratios;
- temp-only input for comparability with existing baselines;
- learned receiver-side request and compression over existing graph edges;
- Gumbel and RL training comparison;
- theoretical TX+RX energy and on-device edge energy evaluation.

Out of scope for the first paper iteration:

- unconstrained latent graph discovery;
- dependence on quantization as the core novelty;
- multi-hop communication protocols beyond existing graph neighbors;
- assuming ESP32-S3 feasibility without measurement.

## Edge deployment position

ESP32-S3 is a loose target, not a strict hard requirement. The model should aim for the smallest feasible configuration and be defensibly classified as edge-oriented. Quantization and pruning are evaluation tools only, not part of the main algorithmic contribution.

## Main risks

1. **Temporal baselines are already strong.** A 3–4 macro-F1 point improvement may require better spatial signal than the current Intel connectivity graph provides.
2. **ST-GCN is too weak to be decisive.** The paper must include stronger dense and rule-based spatial baselines.
3. **Gate collapse.** Learned requests may become all-on or all-off without careful penalties and temperature/reward schedules.
4. **Energy accounting ambiguity.** All communication claims must include TX and RX energy, plus measured edge energy when possible.
5. **HiFiNet availability.** If HiFiNet is the closest spatial method, it must be reproduced or clearly bounded as unavailable/inapplicable.

## Implementation milestones

1. Implement CESTA with a GRU encoder, receiver-side request gate, compression selector, lightweight aggregation, and graph-model output shape.
2. Add auxiliary communication-energy/loss support to training.
3. Add evaluation logging for requested edges, transmitted bits, theoretical TX+RX energy, and model compute metrics.
4. Run a sanity experiment on `Intel_fault15`.
5. Run staged ablations for request-only, compression-only, and full request+compression.
6. Compare Gumbel-Softmax and RL controllers under Pareto dominance.
7. Run full average-performance evaluation across all fault ratios.
