# Experiment plan: CESTA

## Question

Can CESTA exceed temporal-only macro-F1 while reducing communication energy through receiver-side learned request and compression decisions over the existing sensor graph?

## Primary hypothesis

CESTA will improve average macro-F1 over the best temporal-only model per fault ratio by at least +0.01, preferably +0.03 to +0.04, while reducing communication energy relative to dense spatial message passing.

## Data

Use the current harder Intel injected graph datasets:

```text
data/injected/Intel_fault05
data/injected/Intel_fault10
data/injected/Intel_fault15
data/injected/Intel_fault20
```

Use temp-only input for comparability:

```text
features: ["temp"]
```

Graph preparation should use a directed candidate edge list from `connectivity.txt` and a once-sampled bursty link-success mask. Runtime graph availability is dynamic:

```text
active_edge[t,e] = link_success[t,e] & node_observed[t,sender(e)] & node_observed[t,receiver(e)]
```

Missing node labels are stored as `-1` and excluded by masked loss/metrics. Complete-case timestamp filtering is not viable for the current Intel graph data because no timestamp contains all 55 nodes.

The first decisive development dataset is:

```text
data/injected/Intel_fault15
```

## Current temporal targets

Current HPO retrain macro-F1 targets from existing results:

| Fault ratio | Best temporal baseline | Macro-F1 |
|---|---|---:|
| fault05 | GRU | 0.8574 |
| fault10 | LSTM | 0.8764 |
| fault15 | GRU | 0.8999 |
| fault20 | GRU | 0.9042 |

Minimum paper target by fault ratio is approximately:

| Fault ratio | Minimum target |
|---|---:|
| fault05 | 0.8674 |
| fault10 | 0.8864 |
| fault15 | 0.9099 |
| fault20 | 0.9142 |

Preferred Q1-level target is +0.03 to +0.04 average macro-F1 over those best temporal baselines.

## Baselines and controls

### Temporal baselines

1. Best temporal-only model per fault ratio.
2. Fixed CESTA temporal encoder without communication.
3. GRU/LSTM/ModernTCN HPO retrain results as strong temporal references.

### Spatial baselines

1. ST-GCN.
2. HiFiNet, if the supplied paper confirms it targets sensor/graph fault diagnosis.
3. Dense learned message passing: same encoder/aggregator as CESTA, all currently available directed candidate edges active, full embeddings transmitted.
4. Static top-k graph communication using strongest connectivity edges.
5. Random communication at matched average communication budget.

### Rule-based controls

1. Uncertainty-triggered communication using entropy or prediction margin.
2. Change-triggered communication using local change/anomaly magnitude.
3. Combined uncertainty + change trigger.

Rule-based thresholds must be tuned to match CESTA's average communication budget for fair comparison.

## Metrics

### Accuracy metrics

- macro-F1;
- per-class F1;
- accuracy;
- confusion matrix;
- average Δ macro-F1 against the best temporal-only model per fault ratio;
- average Δ macro-F1 against a fixed temporal backbone.

### Communication and energy metrics

Primary energy metrics should be based on energy consumption:

- measured on-device energy per window/inference;
- theoretical TX+RX communication energy per window;
- energy reduction versus dense learned message passing;
- macro-F1 per Joule or macro-F1 per communication-energy unit;
- active request ratio;
- requested edges per node/window;
- transmitted bits per node/window;
- compression-ratio distribution;
- receiver RX energy share;
- sender TX energy share.

### Edge metrics

- parameter count;
- serialized model size;
- inference latency on edge-class target;
- peak memory estimate;
- effect of int8/dynamic quantization as evaluation only.

## Theoretical energy calculation

For every active receiver-side request from sender node `j` to receiver node `i`, count both TX and RX energy only when the directed candidate edge is available at that timestamp/window.

```text
E_tx(k, d) = E_elec · k + E_amp · k · d^n
E_rx(k) = E_elec · k
E_msg(k, d) = E_tx(k, d) + E_rx(k)
```

Use free-space or multipath amplifier constants according to threshold distance:

```text
d0 = sqrt(E_fs / E_mp)
```

For CESTA:

```text
E_CESTA = Σ_windows Σ_t Σ_edges j→i available[t,j→i] · g_i,j,t · E_msg(k_i,j, d_i,j)
```

where `k_i,j` depends on hidden dimension, compression ratio, numeric precision, and protocol overhead if modeled.

For dense learned message passing:

```text
E_dense = Σ_windows Σ_t Σ_edges j→i available[t,j→i] · E_msg(k_full, d_j,i)
```

Report reduction:

```text
reduction = 1 - E_CESTA / E_dense
```

## Staged experiments

### Stage 0: feasibility checks

Goal: verify graph data shape, output shape, and training loop compatibility.

Run on `Intel_fault15` for a very small epoch budget.

Checks:

- graph batch carries `x`, `y`, `node_mask`, directed `edge_index`, and per-window `edge_mask`;
- logits shape is `(batch, window_size, num_nodes, num_classes)`;
- loss computes against per-node labels only where `node_mask` is true;
- communication stats are non-empty;
- requested edge ratio is not NaN;
- model can overfit a tiny batch.

### Stage 1: temporal encoder baseline

Train the CESTA temporal encoder without communication.

Purpose:

- establish the fixed backbone baseline;
- separate temporal encoder strength from spatial communication contribution.

Required outputs:

- macro-F1;
- per-class F1;
- parameter count;
- latency estimate.

### Stage 2: dense learned message passing

Train the same encoder and aggregation module with all currently available directed candidate edges active and full embeddings transmitted.

Purpose:

- establish the upper bound for the CESTA architecture without communication limits;
- provide a stronger spatial baseline than ST-GCN.

Required outputs:

- macro-F1;
- per-class F1;
- theoretical TX+RX energy;
- measured edge energy if available.

### Stage 3: request-only CESTA

Train receiver-side learned request gates with full embedding transmission when active.

Compare:

- Gumbel-Softmax request gate;
- RL request policy.

Purpose:

- isolate the benefit of deciding whether to communicate.

### Stage 4: compression-only CESTA

Keep all currently available directed candidate edges active but learn/select compression ratio.

Compare:

- fixed compression ratios;
- Gumbel-Softmax compression selector;
- RL compression selector if feasible.

Purpose:

- isolate the benefit of reducing payload size.

### Stage 5: full CESTA

Train receiver-side request gate and compression selector together.

Compare:

- Gumbel request + Gumbel compression;
- RL request + RL compression;
- hybrid Gumbel pretraining followed by RL fine-tuning if neither pure method dominates.

The main design is selected by Pareto dominance:

```text
higher macro-F1 at equal/lower measured energy
or lower measured energy at equal/higher macro-F1.
```

If Pareto-tied, choose the simpler Gumbel design.

### Stage 6: rule-based controls

Evaluate rule-based controllers at matched average communication budgets:

1. entropy threshold;
2. prediction-margin threshold;
3. local change magnitude threshold;
4. combined uncertainty + change threshold.

Purpose:

- prove that learned communication is better than simple triggering.

### Stage 7: full benchmark across all fault ratios

Run the selected CESTA variants across all four fault ratios.

Required comparisons:

- best temporal per fault ratio;
- fixed temporal backbone;
- ST-GCN;
- HiFiNet if applicable;
- dense learned message passing;
- static top-k;
- random budget-matched;
- best rule-based budget-matched controller.

## Ablations

Required ablations:

1. no communication;
2. dense full communication;
3. request-only;
4. compression-only;
5. request + compression;
6. uncertainty removed from gate input;
7. local embedding removed from gate input;
8. fusion gate removed;
9. static top-k neighbors;
10. random neighbors at matched budget;
11. Gumbel-Softmax versus RL;
12. per-window versus per-timestep decision if implementation cost permits;
13. compression ratios `{0.25, 0.5, 1.0}` versus smaller/larger sets;
14. different energy penalty weights;
15. quantized versus non-quantized edge evaluation;
16. GAT single-head attention versus degree-normalized mean aggregation;
17. attention over received set only versus softmax over padded full neighbor set;
18. multi-head attention versus single-head if neighbor sets grow large (>4).

## Hyperparameter sweeps

Minimum sweep axes:

- hidden size: `32`, `64`, `128`;
- gate penalty weight;
- bits/energy penalty weight;
- compression-ratio set;
- dropout;
- Gumbel temperature schedule;
- RL reward weights if RL is used.

The selection metric should not be macro-F1 alone. Use Pareto frontier analysis over macro-F1 and measured/theoretical energy.

## Expected outcomes

Best case:

- CESTA improves average macro-F1 by +0.03 to +0.04 over best temporal baselines;
- communication energy falls substantially versus dense spatial communication;
- learned request/compression dominates rule-based controls;
- gate activation is higher for uncertain, DRIFT, and STUCK windows.

Minimum acceptable outcome:

- CESTA improves average macro-F1 by at least +0.01 over best temporal baselines;
- CESTA is Pareto-superior to dense learned message passing or at least to ST-GCN/HiFiNet if dense message passing is too costly.

Negative outcome:

- CESTA cannot exceed best temporal baselines;
- communication is only useful at all-on or near-all-on budgets;
- rule-based triggers match learned gates.

If negative, reposition the contribution as an energy-aware spatial communication study only if energy savings are strong and accuracy remains close to temporal baselines.

## Failure modes

1. Gate collapse to all-off due to energy penalty overpowering classification loss.
2. Gate collapse to all-on because spatial messages are too useful or penalty is too weak.
3. Compression selector always chooses full embeddings.
4. RL policy instability or high variance.
5. The existing Intel connectivity graph lacks useful spatial signal.
6. Energy model overstates savings relative to measured ESP32-S3 behavior because radio wake/sleep overhead dominates.
7. Dense learned message passing beats CESTA by too much, weakening selective-communication claims.
8. HiFiNet outperforms CESTA without much extra cost.

## Reproducibility notes

Record for every run:

- dataset path and fault ratio;
- selected features;
- graph threshold, directed edge count, node count, dynamic-link seed, and burst-simulation parameters;
- random seed;
- model config;
- training controller type: Gumbel, RL, or rule-based;
- energy constants and distance assumptions;
- measured-energy hardware setup;
- communication stats;
- run manifest and git state.

## First implementation checkpoint

Implement a minimal CESTA variant first:

```text
GRU temporal encoder
receiver-side local gate
Gumbel request only
full embedding when active
GAT-inspired single-head attention aggregation
communication stats logging
```

Run this on `Intel_fault15` before adding compression or RL.
