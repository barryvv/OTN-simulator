#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
SIM_DIR="${SIM_DIR:-$(cd "$(dirname "$0")" && pwd)}"
OUT_BASE="$SIM_DIR/outputs/noregen_runs"
PROMPTS_DIR="$OUT_BASE/prompts"
ANALYSIS_DIR="$OUT_BASE/llm_analysis"

# inference_qwen.py settings — adjust as needed
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
ADAPTER="${ADAPTER:-adapters/qwen-failure}"
MAX_TOKENS="${MAX_TOKENS:-600}"
DTYPE="${DTYPE:-float16}"
EXTRA_ARGS="${EXTRA_ARGS:---load-in-4bit}"

mkdir -p "$ANALYSIS_DIR"

EVENTS=(
  Fiber_Cut
  Fiber_Crack
  Fiber_Aging
  Line_Disconnect
  XCON_Port_Down
  XCON_Fabric_Fault
  XCON_Buffer_Overflow
)

for EVENT in "${EVENTS[@]}"; do
  EVENT_LOWER=$(echo "$EVENT" | tr '[:upper:]' '[:lower:]')
  PROMPT_FILE="$PROMPTS_DIR/prompt_${EVENT_LOWER}.txt"
  OUTPUT_FILE="$ANALYSIS_DIR/llm_analysis_${EVENT_LOWER}.txt"

  if [ ! -f "$PROMPT_FILE" ]; then
    echo "[SKIP] Missing prompt: $PROMPT_FILE"
    continue
  fi

  echo ""
  echo "============================================"
  echo "  Inference: $EVENT"
  echo "============================================"

  $PYTHON "$SIM_DIR/inference_qwen.py" \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --prompt-file "$PROMPT_FILE" \
    --max-tokens "$MAX_TOKENS" \
    --dtype "$DTYPE" \
    $EXTRA_ARGS \
    2>&1 | tee "$OUTPUT_FILE"

  echo "  -> Saved to $OUTPUT_FILE"
done

echo ""
echo "============================================"
echo "  All inference done! Outputs in: $ANALYSIS_DIR"
echo "============================================"
