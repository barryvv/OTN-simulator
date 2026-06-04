# OTN Simulator (End-to-End Guide)

This repository simulates topology, failures, timeseries metrics (BBE/BBER/ES), and alarm propagation,
and fine-tunes LLMs to predict alarm root causes from the resulting data.  
Below is a start-to-end workflow that matches the current scripts and file layout.
The simulator/data pipeline comes first (sections 0–9); all LLM **post-training**
(SFT, GRPO, and the from-scratch GRPO implementation) lives in the
[Post-Training](#post-training-llm-fine-tuning) section at the end.

## 0) Prereqs
- Python 3.9+ recommended
- Packages used by the scripts: `networkx`, `pandas`, `pyyaml`, `matplotlib`

## 1) Generate topology (no-regen and regen)
This writes board-level topology preview files (graphml, edges, lightpaths).

```bash
# no regen
ARANGO_SKIP=1 REGEN_HOPS=0 TOPO_PREVIEW_DIR=outputs/topology_preview_noregen python3 run.py

# regen inserted on optical spans (adjust hop interval)
ARANGO_SKIP=1 REGEN_HOPS=2 TOPO_PREVIEW_DIR=outputs/topology_preview_regen python3 run.py
```

Outputs:
- `outputs/topology_preview_noregen/board.graphml`
- `outputs/topology_preview_noregen/lightpaths.txt`
- `outputs/topology_preview_regen/board.graphml`
- `outputs/topology_preview_regen/board_regen.graphml`
- `outputs/topology_preview_regen/lightpaths.txt`

## 2) Convert graphml + lightpaths into topology.yaml
This produces the `topology.yaml` used by the simulator.

```bash
# no-regen topology.yaml
python3 convert_graphml_to_topology.py \
  --graphml outputs/topology_preview_noregen/board.graphml \
  --lightpaths outputs/topology_preview_noregen/lightpaths.txt \
  --out topology_noregen.yaml

# regen topology.yaml
python3 convert_graphml_to_topology.py \
  --graphml outputs/topology_preview_regen/board_regen.graphml \
  --lightpaths outputs/topology_preview_regen/lightpaths.txt \
  --out topology_regen.yaml
```

## 3) (Optional) Build electrical lightpaths with roles
Useful for diagnostics and graph coloring.

```bash
python3 make_electrical_topology.py \
  --input outputs/topology_preview_noregen/lightpaths.txt \
  --output outputs/topology_preview_noregen/electrical_lightpaths_roles.txt

python3 make_electrical_topology.py \
  --input outputs/topology_preview_regen/lightpaths.txt \
  --output outputs/topology_preview_regen/electrical_lightpaths_roles.txt
```

## 4) Create a failure file
Edit `outputs/alarm_flows/failure.csv` using this header:

```
Board,Event,Severity,Time,DurationMs,BoardRole
ROADM0-OA002$0,Fiber_Cut,Critical,2025-10-26 00:00:42,60000,RELAY
```

## 5) Run the simulator (metrics + alarm flow)
This generates timeseries and alarm_flow in a run folder.

```bash
python3 otn_simulator.py \
  --topology topology_noregen.yaml \
  --roles roles.yaml \
  --simulator simulator.yaml \
  --failures outputs/alarm_flows/failure.csv \
  --outdir data_outputs_noregen \
  --alarm-flow \
  --alarm-outdir outputs/alarm_flows/test_runs/noregen \
  --paths-from outputs/topology_preview_noregen/lightpaths.txt

python3 otn_simulator.py \
  --topology topology_regen.yaml \
  --roles roles.yaml \
  --simulator simulator.yaml \
  --failures outputs/alarm_flows/failure.csv \
  --outdir data_outputs_regen \
  --alarm-flow \
  --alarm-outdir outputs/alarm_flows/test_runs/regen \
  --paths-from outputs/topology_preview_regen/lightpaths.txt
```

Outputs:
- `data_outputs_*/*/timeseries_*.csv` (BBE/BBER/ES)
- `outputs/alarm_flows/test_runs/*/alarm_flow.csv`
- `outputs/alarm_flows/test_runs/*/local_alarms_from_data.csv`

## 6) Plot topology
```bash
python3 plot_topology_graph.py \
  --graphml outputs/topology_preview_noregen/board.graphml \
  --out outputs/topology_preview_noregen/topology_graph.png \
  --mode board --labels \
  --roles outputs/topology_preview_noregen/electrical_lightpaths_roles.txt

python3 plot_topology_graph.py \
  --graphml outputs/topology_preview_regen/board_regen.graphml \
  --out outputs/topology_preview_regen/topology_graph.png \
  --mode board --labels \
  --roles outputs/topology_preview_regen/electrical_lightpaths_roles.txt
```

## 7) Plot per-board metrics (BBE/BBER/ES)
```bash
python3 plot_bbe_lat_bber_es.py \
  --failure outputs/alarm_flows/failure.csv \
  --topology topology_noregen.yaml \
  --data-dir data_outputs_noregen \
  --outdir outputs/plots_noregen \
  --boards-from-alarm-flow outputs/alarm_flows/test_runs/noregen/alarm_flow.csv \
  --affected-all

python3 plot_bbe_lat_bber_es.py \
  --failure outputs/alarm_flows/failure.csv \
  --topology topology_regen.yaml \
  --data-dir data_outputs_regen \
  --outdir outputs/plots_regen \
  --boards-from-alarm-flow outputs/alarm_flows/test_runs/regen/alarm_flow.csv \
  --affected-all
```

## 8) Quick sanity checks
- `alarm_flow.csv` should contain RuleIdx >= 0 rows for propagated alarms.
- `regen_boundary` rows should appear in regen runs only.
- Plot windows should align to ArrivalTime/ClearTime if you enabled those in the plotter.

## 9) Generate large-scale benchmark datasets

The `sfc_bfl/` package and the `generate_from_dsfc.py` /
`generate_train_from_dsfc.py` scripts produce structured datasets where
each Observation Window (OW) carries metrics, alarms, and a label
describing the injected failure(s).

### 9.1) DSFC-driven Dataset A (regional failures, k ∈ {1,2,3,5,8,10})

Uses an externally supplied DSFC (Diagnostic Service Function Chain)
cover file. Each line in the file is one lightpath as a comma-separated
list of board names (e.g. `ROADM0-Tributary1,ROADM0-XCON1,...`).

Regional failures are constructed by walking **downstream** along the
lightpath signal direction starting from a root board on that lightpath,
guaranteeing every board in `failed_boards` is observable. **One failure
per region board** is injected (kind-appropriate event chosen by
`event_for_board`); each failure produces independent alarm propagation
with One-to-One/One-to-Many root cause attribution.

```bash
ARANGO_SKIP=1 python3 generate_from_dsfc.py \
  --dsfc path/to/sfc_cover.paths.txt \
  --outdir outputs/sfc_bfl/dsfc_dataset_a \
  [--pilot]            # smaller subset for debug
  [--seed 42]
  [--num-roadms 8] [--eg 5]
```

### 9.2) DSFC-driven training data (per-domain, k=1)

Reads multiple `domain_*/sfc_cover.paths.txt` files and emits one OW per
board on every lightpath in each domain.

```bash
ARANGO_SKIP=1 python3 generate_train_from_dsfc.py \
  --dsfc-dir path/to/DSFC_train \
  --outdir outputs/sfc_bfl/dsfc_dataset_train \
  [--pilot]
```

`DSFC_train/` layout:
```
DSFC_train/
  domain_0/sfc_cover.paths.txt
  domain_1/sfc_cover.paths.txt
  domain_2/sfc_cover.paths.txt
```

### 9.3) SFC-BFL benchmark suite (Datasets A/B/C/D/E + TRAIN)

```bash
# all datasets, full scale
python3 -m sfc_bfl.run_all --dataset all --outdir outputs/sfc_bfl/full

# pilot (small subset for each)
python3 -m sfc_bfl.run_all --dataset all --pilot --outdir outputs/sfc_bfl/pilot

# single dataset
python3 -m sfc_bfl.run_all --dataset A --outdir outputs/sfc_bfl/A
```

| Dataset | Purpose |
|---------|---------|
| TRAIN | 3 domains × every board, k=1 (per-board single-failure training set) |
| A | Regional failures, k ∈ {1,2,3,5,8,10} |
| B | False-positive noise injection at varying ratios |
| C | Coverage masking: drop fraction of metrics |
| D | Board-count scaling (200/400/800/1600 boards) × k-dimension |
| E | Multi-domain simultaneous failures |

### 9.4) Output layout

Each dataset directory follows this layout:

```
<outdir>/
  boards.csv              # per-board metadata (id, kind, parent_node, role)
  edges.csv               # board-graph edges
  lightpaths.json         # service paths
  services.json           # service metadata
  domain_stats.json       # summary statistics
  ow_metrics/             # one parquet shard per N OWs
    _metrics_part_0.parquet
    _metrics_part_1.parquet
    ...
  ow_alarms.jsonl         # one JSON line per OW: {"ow_id": ..., "alarms": [...]}
  ow_labels.jsonl         # one JSON line per OW: ground-truth failure
```

Multi-domain datasets (TRAIN, E) nest by domain:
```
<outdir>/domain_0/...
<outdir>/domain_1/...
<outdir>/domain_2/...
```

### 9.5) Label schema (`ow_labels.jsonl`)

Each line is a JSON object describing one OW's ground truth:

| Field | Meaning |
|-------|---------|
| `ow_id` | Unique OW index within the dataset |
| `domain_id` | Domain index (0 for single-domain datasets) |
| `root_board` | Primary failure board (normalized, zero-padded) |
| `root_event` | Primary failure event |
| `failed_boards` | List of all injected failure boards (length = `region_size_k`) |
| `failed_events` | List of injected events (parallel to `failed_boards`) |
| `region_size_k` | Number of simultaneous failures (1 for single-failure OWs) |
| `severity`, `failure_time`, `duration_ms` | Failure metadata |

For Dataset A v3, every board in `failed_boards` is guaranteed to have at
least one **hop=0** local alarm in `ow_alarms.jsonl`, and every alarm row
attributes back to a root in `failed_boards` (no Many-to-One attribution).

## Notes
- Use `ARANGO_SKIP=1` to avoid ArangoDB.
- Use `REGEN_HOPS` to control regen insertion frequency (0 = no regen).
- Use `ELECTRICAL_GROUPS=N` (or `--eg N`) to control electrical-layer chain count per ROADM (more → larger boards-per-domain).

---

# Post-Training (LLM Fine-Tuning)

Once the simulator has produced alarm/metric data, the second half of the
project fine-tunes LLMs (Qwen 2.5) to predict alarm root causes from the
metrics alone. Everything related to post-training — dataset construction,
SFT, GRPO, and inference — is collected here.

ML dependencies (on top of the simulator deps): `torch`, `transformers`,
`peft`, `trl`, `datasets`, `accelerate`.

## P1) Build training data

```bash
# SFT: prompt + ground-truth completion pairs
python3 generate_sft_data.py \
  --rules outputs/Static\ files/rule_database.csv \
  --run-dirs outputs/alarm_flows/test_runs/noregen outputs/alarm_flows/test_runs/regen \
  --out outputs/llm_data/train.jsonl

# GRPO: prompt + structured ground-truth fields (boards/alarms/severity/...)
python3 generate_grpo_data.py --out outputs/grpo_data/train.jsonl
```

The GRPO JSONL carries the ground-truth fields the reward functions score
against (`ground_truth_boards`, `ground_truth_alarms`,
`ground_truth_root_board`, `ground_truth_root_event`,
`ground_truth_row_count`, `ground_truth_severity_dist`, ...).

## P2) Supervised fine-tuning (SFT)

Standard LoRA SFT to teach the base model the output format and basic
mapping before any RL.

```bash
python3 train_sft_qwen7b.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-file outputs/llm_data/train.jsonl \
  --output-dir adapters/qwen-7b-failure
```

`train_agent_sft.py` is a variant that trains on agent/tool-use traces.

## P3) GRPO — TRL (production)

Reinforcement fine-tuning with Hugging Face TRL's `GRPOTrainer`. This is
the battle-tested path used for actual training. Rewards compare predicted
boards/alarms/severity distributions against ground truth
(`board_reward_func`, `alarm_reward_func`, `severity_reward_func`,
`format_reward_func`).

```bash
python3 train_grpo_gemma3b.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --train-file outputs/grpo_data/train.jsonl \
  --output-dir adapters/qwen-3b-grpo
```

## P4) Inference

```bash
# GRPO adapter, compared against the base model
python3 inference_grpo.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --adapter adapters/qwen-3b-grpo \
  --prompt-file outputs/grpo_data/train.jsonl \
  --sample-index 0 --compare-base

# SFT adapter / generic Qwen inference
python3 inference_qwen.py --adapter adapters/qwen-7b-failure ...

# Batched inference over a dataset
python3 batch_inference.py ...
```

## P5) GRPO from scratch (raw PyTorch)

`train_grpo_scratch.py` + the `grpo_scratch/` package implement the GRPO
algorithm in raw PyTorch — the policy update loop, log-prob computation,
group-relative advantage estimation, and clipped surrogate loss — **without
using `trl.GRPOTrainer`**. It reuses the same dataset and OTN reward
functions as the TRL trainer, so the two can be compared directly. This is
the educational / first-principles counterpart to P3.

Module layout:
```
grpo_scratch/
├── sampling.py            # group sampling (G completions per prompt)
├── log_probs.py           # per-token log-prob computation with masking
├── advantages.py          # group-relative advantage baseline (GRPO / Dr. GRPO)
├── losses.py              # token-level clipped surrogate loss + optional KL
├── rewards.py             # OTN reward functions (discrete tiers)
├── rewards_continuous.py  # continuous similarity-based reward variant
└── trainer.py             # training loop, phase-by-phase
```

Run the math tests (group-relative advantages, token-level clipped
surrogate, KL toggle, log-prob shift+mask):
```bash
python -m pytest tests/test_grpo_scratch.py -v
```

Launch a training run:
```bash
python train_grpo_scratch.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --train-file outputs/grpo_data/train.jsonl \
  --output-dir adapters/qwen-3b-grpo-scratch \
  --group-size 8 --max-steps 300 \
  --lr 1e-4 --lr-scheduler cosine --warmup-steps 10 \
  --reward-fn continuous
```

Key knobs that make it actually learn on a flat reward landscape:
- **`--scale-rewards`** *(off by default)* — when off, advantages use the
  Dr. GRPO mean-only baseline instead of dividing by the within-group std.
  Std-scaling a near-flat reward amplifies rounding noise into
  unit-magnitude advantages; the mean-only baseline avoids it.
- **`--reward-fn continuous`** — smooth similarity slopes instead of
  discrete tiers, so group-relative advantages don't collapse to zero.
- token-level per-token-mean clipped surrogate (built in) removes the
  length bias of summed-sequence log-probs.

**Run on Google Colab (no local GPU needed):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/barryvv/OTN-simulator/blob/main/train_grpo_scratch_colab.ipynb)

The notebook `train_grpo_scratch_colab.ipynb` clones this repo, installs the
deps, runs the unit tests, uploads your `train.jsonl`, and trains Qwen 2.5-3B
with bf16 + LoRA. Watch `reward_mean` drift upward over the run; `loss≈0`
every step is expected for on-policy single-update GRPO (the gradient, not
the loss value, drives the update).
