# OTN Simulator (End-to-End Guide)

This repository simulates topology, failures, timeseries metrics (BBE/BBER/ES), and alarm propagation.  
Below is a start-to-end workflow that matches the current scripts and file layout.

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

## Notes
- Use `ARANGO_SKIP=1` to avoid ArangoDB.
- Use `REGEN_HOPS` to control regen insertion frequency (0 = no regen).
