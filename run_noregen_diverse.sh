#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
SIM_DIR="${SIM_DIR:-$(cd "$(dirname "$0")" && pwd)}"
TOPO_DIR="$SIM_DIR/regen_compare/noregen/topology"
OUT_BASE="$SIM_DIR/outputs/grpo_runs"
RULES="$SIM_DIR/outputs/Static files/rule_database.csv"
TOPO_YAML="$SIM_DIR/outputs/noregen_runs/topology_noregen.yaml"
ROLES_YAML="$SIM_DIR/outputs/noregen_runs/roles_noregen.yaml"
SIMULATOR_YAML="$SIM_DIR/simulator.yaml"

mkdir -p "$OUT_BASE"

COUNT=0

run_event() {
  local EVENT="$1"
  local BOARD="$2"
  local ROLE="$3"
  local SEV="$4"
  local SEED="$5"

  # Build a safe run name: replace $ with _ in board name
  local BOARD_SAFE
  BOARD_SAFE=$(echo "$BOARD" | tr '$' '_')
  local RUN_NAME="${EVENT}_${BOARD_SAFE}_s${SEED}"
  local RUN_DIR="$OUT_BASE/$RUN_NAME"
  local DATA_DIR="$OUT_BASE/data_outputs_${RUN_NAME}"

  if [ -f "$RUN_DIR/alarm_flow.csv" ]; then
    echo "[skip] $RUN_NAME (already exists)"
    COUNT=$((COUNT + 1))
    return
  fi

  COUNT=$((COUNT + 1))
  echo ""
  echo "[$COUNT] Running: $RUN_NAME"

  mkdir -p "$RUN_DIR" "$DATA_DIR"

  # Create failure.csv
  printf 'Board,Event,Severity,Time,DurationMs,BoardRole\n%s,%s,%s,2025-10-26 00:00:42,60000,%s\n' \
    "$BOARD" "$EVENT" "$SEV" "$ROLE" > "$RUN_DIR/failure.csv"

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
    --seed "$SEED"

  # Copy lightpaths for prompt generation
  cp "$TOPO_DIR/lightpaths.txt" "$RUN_DIR/lightpaths.txt"

  echo "  -> $RUN_DIR (alarm_flow=$(wc -l < "$RUN_DIR/alarm_flow.csv") rows)"
}

SEEDS=(2025 2026 2027 2028 2029 2030 2031)

for S in "${SEEDS[@]}"; do
  # ── Fiber failures: OA relay boards on different ROADMs (3 boards each) ──
  run_event "Fiber_Cut"   'ROADM0-OA002$0' "RELAY" "Critical" "$S"
  run_event "Fiber_Cut"   'ROADM1-OA001$0' "RELAY" "Critical" "$S"
  run_event "Fiber_Cut"   'ROADM3-OA004$0' "RELAY" "Critical" "$S"

  run_event "Fiber_Crack" 'ROADM0-OA002$0' "RELAY" "Critical" "$S"
  run_event "Fiber_Crack" 'ROADM3-OA004$0' "RELAY" "Critical" "$S"
  run_event "Fiber_Crack" 'ROADM4-OA001$0' "RELAY" "Critical" "$S"

  run_event "Fiber_Aging" 'ROADM0-OA002$0' "RELAY" "Critical" "$S"
  run_event "Fiber_Aging" 'ROADM4-OA001$0' "RELAY" "Critical" "$S"
  run_event "Fiber_Aging" 'ROADM1-OA001$0' "RELAY" "Critical" "$S"

  # ── Line failures: Line boards on different ROADMs (3 boards) ──
  run_event "Line_Disconnect" 'ROADM0-Line025$0' "SRC" "Critical" "$S"
  run_event "Line_Disconnect" 'ROADM1-Line020$0' "SRC" "Critical" "$S"
  run_event "Line_Disconnect" 'ROADM2-Line025$0' "SRC" "Critical" "$S"

  # ── XCON failures: XCON boards on different ROADMs (3 boards each) ──
  run_event "XCON_Port_Down"       'ROADM0-XCON025$0' "SRC" "Critical" "$S"
  run_event "XCON_Port_Down"       'ROADM3-XCON027$0' "SRC" "Critical" "$S"
  run_event "XCON_Port_Down"       'ROADM1-XCON051$0' "SRC" "Critical" "$S"

  run_event "XCON_Fabric_Fault"    'ROADM0-XCON025$0' "SRC" "Critical" "$S"
  run_event "XCON_Fabric_Fault"    'ROADM2-XCON096$0' "SRC" "Critical" "$S"
  run_event "XCON_Fabric_Fault"    'ROADM3-XCON027$0' "SRC" "Critical" "$S"

  run_event "XCON_Buffer_Overflow" 'ROADM0-XCON025$0' "SRC" "Critical" "$S"
  run_event "XCON_Buffer_Overflow" 'ROADM1-XCON051$0' "SRC" "Critical" "$S"
  run_event "XCON_Buffer_Overflow" 'ROADM2-XCON096$0' "SRC" "Critical" "$S"
done

echo ""
echo "============================================"
echo "  Done! $COUNT runs in: $OUT_BASE"
echo "============================================"
