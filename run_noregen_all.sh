#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
SIM_DIR="${SIM_DIR:-$(cd "$(dirname "$0")" && pwd)}"
TOPO_DIR="$SIM_DIR/regen_compare/noregen/topology"
OUT_BASE="$SIM_DIR/outputs/noregen_runs"
RULES="$SIM_DIR/outputs/Static files/rule_database.csv"
TOPO_YAML="$OUT_BASE/topology_noregen.yaml"
ROLES_YAML="$OUT_BASE/roles_noregen.yaml"
SIMULATOR_YAML="$SIM_DIR/simulator.yaml"

mkdir -p "$OUT_BASE"

run_event() {
  local EVENT="$1"
  local BOARD="$2"
  local ROLE="$3"
  local SEV="$4"

  echo ""
  echo "============================================"
  echo "  Running: $EVENT (noregen)"
  echo "============================================"

  local RUN_DIR="$OUT_BASE/$EVENT"
  local EVENT_LOWER
  EVENT_LOWER=$(echo "$EVENT" | tr '[:upper:]' '[:lower:]')
  local DATA_DIR="$OUT_BASE/data_outputs_${EVENT_LOWER}"
  mkdir -p "$RUN_DIR" "$DATA_DIR"

  # Create failure.csv
  printf 'Board,Event,Severity,Time,DurationMs,BoardRole\n%s,%s,%s,2025-10-26 00:00:42,60000,%s\n' \
    "$BOARD" "$EVENT" "$SEV" "$ROLE" > "$RUN_DIR/failure.csv"

  echo "[1/3] Running simulator for $EVENT..."
  $PYTHON "$SIM_DIR/otn_simulator.py" \
    --topology "$TOPO_YAML" \
    --roles "$ROLES_YAML" \
    --simulator "$SIMULATOR_YAML" \
    --outdir "$DATA_DIR" \
    --failures "$RUN_DIR/failure.csv" \
    --alarm-flow \
    --alarm-outdir "$RUN_DIR" \
    --alarm-rules "$RULES" \
    --alarm-spikes-from-flow \
    --paths-from "$TOPO_DIR/lightpaths.txt" \
    --seed 2025

  # Copy lightpaths to run dir for prompt generation
  cp "$TOPO_DIR/lightpaths.txt" "$RUN_DIR/lightpaths.txt"

  echo "[2/3] Generating plots for $EVENT..."
  local PLOT_DIR="$RUN_DIR/plots"
  mkdir -p "$PLOT_DIR"
  $PYTHON "$SIM_DIR/plot_alarm_flow_synthetic.py" \
    --alarm-flow "$RUN_DIR/alarm_flow.csv" \
    --outdir "$PLOT_DIR"

  echo "[3/3] Done: $EVENT -> $RUN_DIR"
  echo "  - failure.csv"
  echo "  - alarm_flow.csv"
  ls "$PLOT_DIR"/*.png 2>/dev/null | wc -l | xargs -I{} echo "  - {} plot files"
}

# Fiber failures: OA relay board
run_event "Fiber_Cut"       'ROADM0-OA002$0'   "RELAY" "Critical"
run_event "Fiber_Crack"     'ROADM0-OA002$0'   "RELAY" "Critical"
run_event "Fiber_Aging"     'ROADM0-OA002$0'   "RELAY" "Critical"

# Line/XCON failures: Line/XCON creation boards
run_event "Line_Disconnect"      'ROADM0-Line025$0'  "SRC" "Critical"
run_event "XCON_Port_Down"       'ROADM0-XCON025$0'  "SRC" "Critical"
run_event "XCON_Fabric_Fault"    'ROADM0-XCON025$0'  "SRC" "Critical"
run_event "XCON_Buffer_Overflow" 'ROADM0-XCON025$0'  "SRC" "Critical"

echo ""
echo "============================================"
echo "  All done! Outputs in: $OUT_BASE"
echo "============================================"
