#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  generate_test_set.sh — Generate held-out test scenarios for evaluation
#
#  Test-IID (seeds 800-806): Same board-event combos as training (21 combos)
#  Test-OOD (seeds 900-906): Different boards for the same events (21 combos)
#
#  Usage:
#    ./generate_test_set.sh              # run both IID and OOD
#    ./generate_test_set.sh --split iid  # run IID only
#    ./generate_test_set.sh --split ood  # run OOD only
#    ./generate_test_set.sh --split all  # run both (default)
# ============================================================================

PYTHON="${PYTHON:-python3}"
SIM_DIR="${SIM_DIR:-$(cd "$(dirname "$0")" && pwd)}"
TOPO_DIR="$SIM_DIR/regen_compare/noregen/topology"
RULES="$SIM_DIR/outputs/Static files/rule_database.csv"
TOPO_YAML="$SIM_DIR/outputs/noregen_runs/topology_noregen.yaml"
ROLES_YAML="$SIM_DIR/outputs/noregen_runs/roles_noregen.yaml"
SIMULATOR_YAML="$SIM_DIR/simulator.yaml"

# ── Parse arguments ──
SPLIT="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      SPLIT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--split iid|ood|all]"
      exit 1
      ;;
  esac
done

if [[ "$SPLIT" != "iid" && "$SPLIT" != "ood" && "$SPLIT" != "all" ]]; then
  echo "Error: --split must be 'iid', 'ood', or 'all' (got '$SPLIT')"
  exit 1
fi

COUNT=0

run_event() {
  local EVENT="$1"
  local BOARD="$2"
  local ROLE="$3"
  local SEV="$4"
  local SEED="$5"
  local OUT_BASE="$6"

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

# ============================================================================
#  Test-IID: Same board-event combos as training, new seeds
# ============================================================================
run_iid() {
  local OUT_BASE="$SIM_DIR/outputs/test_runs_iid"
  mkdir -p "$OUT_BASE"

  local SEEDS_IID=(800 801 802 803 804 805 806)

  echo "============================================"
  echo "  Generating Test-IID split"
  echo "  Seeds: ${SEEDS_IID[*]}"
  echo "  Output: $OUT_BASE"
  echo "============================================"

  for S in "${SEEDS_IID[@]}"; do
    # ── Fiber failures: same OA relay boards as training ──
    run_event "Fiber_Cut"   'ROADM0-OA002$0' "RELAY" "Critical" "$S" "$OUT_BASE"
    run_event "Fiber_Cut"   'ROADM1-OA001$0' "RELAY" "Critical" "$S" "$OUT_BASE"
    run_event "Fiber_Cut"   'ROADM3-OA004$0' "RELAY" "Critical" "$S" "$OUT_BASE"

    run_event "Fiber_Crack" 'ROADM0-OA002$0' "RELAY" "Critical" "$S" "$OUT_BASE"
    run_event "Fiber_Crack" 'ROADM3-OA004$0' "RELAY" "Critical" "$S" "$OUT_BASE"
    run_event "Fiber_Crack" 'ROADM4-OA001$0' "RELAY" "Critical" "$S" "$OUT_BASE"

    run_event "Fiber_Aging" 'ROADM0-OA002$0' "RELAY" "Critical" "$S" "$OUT_BASE"
    run_event "Fiber_Aging" 'ROADM4-OA001$0' "RELAY" "Critical" "$S" "$OUT_BASE"
    run_event "Fiber_Aging" 'ROADM1-OA001$0' "RELAY" "Critical" "$S" "$OUT_BASE"

    # ── Line failures: same Line boards as training ──
    run_event "Line_Disconnect" 'ROADM0-Line025$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "Line_Disconnect" 'ROADM1-Line020$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "Line_Disconnect" 'ROADM2-Line025$0' "SRC" "Critical" "$S" "$OUT_BASE"

    # ── XCON failures: same XCON boards as training ──
    run_event "XCON_Port_Down"       'ROADM0-XCON025$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "XCON_Port_Down"       'ROADM3-XCON027$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "XCON_Port_Down"       'ROADM1-XCON051$0' "SRC" "Critical" "$S" "$OUT_BASE"

    run_event "XCON_Fabric_Fault"    'ROADM0-XCON025$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "XCON_Fabric_Fault"    'ROADM2-XCON096$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "XCON_Fabric_Fault"    'ROADM3-XCON027$0' "SRC" "Critical" "$S" "$OUT_BASE"

    run_event "XCON_Buffer_Overflow" 'ROADM0-XCON025$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "XCON_Buffer_Overflow" 'ROADM1-XCON051$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "XCON_Buffer_Overflow" 'ROADM2-XCON096$0' "SRC" "Critical" "$S" "$OUT_BASE"
  done

  echo ""
  echo "  Test-IID complete: $COUNT runs in $OUT_BASE"
}

# ============================================================================
#  Test-OOD: Different boards (unseen during training), new seeds
#
#  Training OA boards:   ROADM0-OA002, ROADM1-OA001, ROADM3-OA004, ROADM4-OA001
#  OOD OA boards:        ROADM2-OA003, ROADM4-OA002, ROADM1-OA003
#
#  Training Line boards: ROADM0-Line025, ROADM1-Line020, ROADM2-Line025
#  OOD Line boards:      ROADM3-Line030, ROADM4-Line020, ROADM0-Line030
#
#  Training XCON boards: ROADM0-XCON025, ROADM1-XCON051, ROADM2-XCON096, ROADM3-XCON027
#  OOD XCON boards:      ROADM4-XCON030, ROADM2-XCON050, ROADM1-XCON030
# ============================================================================
run_ood() {
  local OUT_BASE="$SIM_DIR/outputs/test_runs_ood"
  mkdir -p "$OUT_BASE"

  local SEEDS_OOD=(900 901 902 903 904 905 906)

  echo "============================================"
  echo "  Generating Test-OOD split"
  echo "  Seeds: ${SEEDS_OOD[*]}"
  echo "  Output: $OUT_BASE"
  echo "============================================"

  for S in "${SEEDS_OOD[@]}"; do
    # ── Fiber failures: OOD OA relay boards (unseen during training) ──
    run_event "Fiber_Cut"   'ROADM2-OA003$0' "RELAY" "Critical" "$S" "$OUT_BASE"
    run_event "Fiber_Cut"   'ROADM4-OA002$0' "RELAY" "Critical" "$S" "$OUT_BASE"
    run_event "Fiber_Cut"   'ROADM1-OA003$0' "RELAY" "Critical" "$S" "$OUT_BASE"

    run_event "Fiber_Crack" 'ROADM2-OA003$0' "RELAY" "Critical" "$S" "$OUT_BASE"
    run_event "Fiber_Crack" 'ROADM4-OA002$0' "RELAY" "Critical" "$S" "$OUT_BASE"
    run_event "Fiber_Crack" 'ROADM1-OA003$0' "RELAY" "Critical" "$S" "$OUT_BASE"

    run_event "Fiber_Aging" 'ROADM2-OA003$0' "RELAY" "Critical" "$S" "$OUT_BASE"
    run_event "Fiber_Aging" 'ROADM4-OA002$0' "RELAY" "Critical" "$S" "$OUT_BASE"
    run_event "Fiber_Aging" 'ROADM1-OA003$0' "RELAY" "Critical" "$S" "$OUT_BASE"

    # ── Line failures: OOD Line boards (unseen during training) ──
    run_event "Line_Disconnect" 'ROADM3-Line030$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "Line_Disconnect" 'ROADM4-Line020$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "Line_Disconnect" 'ROADM0-Line030$0' "SRC" "Critical" "$S" "$OUT_BASE"

    # ── XCON failures: OOD XCON boards (unseen during training) ──
    run_event "XCON_Port_Down"       'ROADM4-XCON030$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "XCON_Port_Down"       'ROADM2-XCON050$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "XCON_Port_Down"       'ROADM1-XCON030$0' "SRC" "Critical" "$S" "$OUT_BASE"

    run_event "XCON_Fabric_Fault"    'ROADM4-XCON030$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "XCON_Fabric_Fault"    'ROADM2-XCON050$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "XCON_Fabric_Fault"    'ROADM1-XCON030$0' "SRC" "Critical" "$S" "$OUT_BASE"

    run_event "XCON_Buffer_Overflow" 'ROADM4-XCON030$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "XCON_Buffer_Overflow" 'ROADM2-XCON050$0' "SRC" "Critical" "$S" "$OUT_BASE"
    run_event "XCON_Buffer_Overflow" 'ROADM1-XCON030$0' "SRC" "Critical" "$S" "$OUT_BASE"
  done

  echo ""
  echo "  Test-OOD complete: $COUNT runs in $OUT_BASE"
}

# ============================================================================
#  Main: run the requested split(s)
# ============================================================================
echo ""
echo "============================================"
echo "  generate_test_set.sh  (split=$SPLIT)"
echo "============================================"

if [[ "$SPLIT" == "iid" || "$SPLIT" == "all" ]]; then
  run_iid
fi

if [[ "$SPLIT" == "ood" || "$SPLIT" == "all" ]]; then
  run_ood
fi

echo ""
echo "============================================"
echo "  All done! Total runs: $COUNT"
echo "  IID output: $SIM_DIR/outputs/test_runs_iid"
echo "  OOD output: $SIM_DIR/outputs/test_runs_ood"
echo "============================================"
