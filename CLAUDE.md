# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OTN (Optical Transport Network) simulator that generates network topologies, injects failures, produces time-series metrics (BBE/BBER/ES), propagates alarms via rule database, and trains LLMs (Qwen 2.5-3B/7B with LoRA) to predict alarm root causes from metrics alone.

## End-to-End Pipeline

```
1. Topology Generation (run.py)
   → GraphML + lightpaths.txt

2. Topology Conversion (convert_graphml_to_topology.py)
   → topology.yaml + roles.yaml

3. Failure Injection (failure.csv, manual or failure_generator.py)

4. Simulation (otn_simulator.py)
   → timeseries CSVs (BBE/BBER/ES per node) + alarm_flow.csv

5. Training Data (generate_sft_data.py / generate_grpo_data.py)
   → JSONL with prompts + ground truth

6. Model Training (train_sft_qwen7b.py / train_grpo_gemma3b.py)
   → LoRA adapters in adapters/

7. Inference (inference_qwen.py / inference_grpo.py)
```

## Key Commands

### Generate topology
```bash
# No regenerators
ARANGO_SKIP=1 REGEN_HOPS=0 TOPO_PREVIEW_DIR=outputs/topology_preview_noregen python3 run.py

# With regenerators every 2 hops
ARANGO_SKIP=1 REGEN_HOPS=2 TOPO_PREVIEW_DIR=outputs/topology_preview_regen python3 run.py
```

### Convert to topology YAML
```bash
python3 convert_graphml_to_topology.py \
  --graphml outputs/topology_preview_noregen/board.graphml \
  --lightpaths outputs/topology_preview_noregen/lightpaths.txt \
  --out topology_noregen.yaml
```

### Run simulator
```bash
python3 otn_simulator.py \
  --topology topology_noregen.yaml --roles roles.yaml --simulator simulator.yaml \
  --failures outputs/alarm_flows/failure.csv --outdir data_outputs_noregen \
  --alarm-flow --alarm-outdir outputs/alarm_flows/test_runs/noregen \
  --paths-from outputs/topology_preview_noregen/lightpaths.txt
```

### Generate training data
```bash
# SFT
python3 generate_sft_data.py --rules outputs/Static\ files/rule_database.csv \
  --run-dirs outputs/alarm_flows/test_runs/noregen outputs/alarm_flows/test_runs/regen \
  --out outputs/llm_data/train.jsonl

# GRPO
python3 generate_grpo_data.py --out outputs/grpo_data/train.jsonl
```

### Train models
```bash
# SFT (Qwen 7B)
python3 train_sft_qwen7b.py --base-model Qwen/Qwen2.5-7B-Instruct \
  --train-file outputs/llm_data/train.jsonl --output-dir adapters/qwen-7b-failure

# GRPO (Qwen 3B)
python3 train_grpo_gemma3b.py --base-model Qwen/Qwen2.5-3B-Instruct \
  --train-file outputs/grpo_data/train.jsonl --output-dir adapters/qwen-3b-grpo
```

### Inference
```bash
python3 inference_grpo.py --base-model Qwen/Qwen2.5-3B-Instruct \
  --adapter adapters/qwen-3b-grpo --prompt-file outputs/grpo_data/train.jsonl \
  --sample-index 0 --compare-base
```

### Visualization
```bash
python3 plot_topology_graph.py --graphml outputs/topology_preview_noregen/board.graphml \
  --out topology_graph.png --mode board --labels

python3 plot_bbe_lat_bber_es.py --failure outputs/alarm_flows/failure.csv \
  --topology topology_noregen.yaml --data-dir data_outputs_noregen --outdir outputs/plots_noregen
```

## Architecture

### Layer Model
- **Optical layer**: OA (amplifier), OD (demux), OM (mux), FIU, SC2, Line, Regenerator boards
- **Electrical layer**: Tributary, XCON (cross-connect), Line boards
- **VMF roles**: SRC (source,creation), RELAY (intermediate), SNK (termination,sink) — assigned per service per node

### Alarm Propagation
Rule-based engine in `alarm_generator.py` loads a CSV rule database with columns: `FunctionType, EventName, ActionType, ActionTarget, OutputAlarm`. Actions are PROPAGATE or TRIGGER with targets like RELAY or TERMINATION. Each rule has an index for audit trail.

### GRPO Reward Design
`train_grpo_gemma3b.py` extracts CSV blocks from model completions and computes rewards by comparing predicted boards/alarms/severity distributions against ground truth. Key reward functions: `board_reward_func`, `alarm_reward_func`, `severity_reward_func`, `format_reward_func`.

### Naming Conventions
- Boards: `ROADM{id}-{Type}{idx}$0` (e.g., `ROADM0-Tributary001$0`)
- Services: `S{id}` (e.g., `S1`, `S10`)
- Segments: `{node1}-{node2}` (lexicographically ordered)
- VMF IDs: `{node_id}_{role}_{service_id}`

## Configuration

- `simulator.yaml` — simulation params: time steps, anomaly types (spike/shift/burst), role-based priors, noise
- `roles.yaml` — VMF role assignments (SRC/RELAY/SNK per service+node)
- `topology.yaml` — generated network structure with nodes, services, path segments

## Environment

- Python 3.9+ with `networkx`, `pandas`, `pyyaml`, `matplotlib`
- ML: `torch`, `transformers`, `peft`, `trl`, `datasets`
- Use `ARANGO_SKIP=1` to bypass ArangoDB dependency
- Virtual envs: `.venv/` (main), `.venv-rag/` (RAG with sentence-transformers + faiss)
- Generated outputs (`data_outputs*/`, `outputs/`, `adapters/`) are gitignored
