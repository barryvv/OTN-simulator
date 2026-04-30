#!/usr/bin/env python3
"""Rule-based reverse-lookup baseline for RCA comparison.

Takes the same input as the LLM (timeseries metrics embedded in GRPO JSONL
prompts, topology/lightpaths, rule database) and performs backward rule tracing
to predict the root cause WITHOUT any LLM.

Algorithm:
  1. Parse the user prompt to extract pattern_features (peak_board, pattern_kind,
     bbe_max, etc.)
  2. Determine board type from peak_board name
  3. Apply decision-tree mapping:
       Board type + pattern_kind  -->  predicted event
  4. Run alarm_generator.py forward propagation to predict alarm flow
  5. Format output to match the LLM output structure (for evaluate.py)

Usage:
  python baseline_rule_reverse.py \
    --test-file outputs/grpo_data/test.jsonl \
    --rules outputs/Static\ files/rule_database.csv \
    --lightpaths regen_compare/noregen/topology/lightpaths.txt \
    --out outputs/eval/baseline_rule_outputs.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from alarm_generator import generate_alarm_flow, load_rule_database
from generate_sft_data import EVENT_FIXES


# ── bbe_max lookup table (unique values per event type) ─────────────────────
#
# Each event produces a unique bbe_max value (set in otn_simulator._profile_for_alarm).
# v6 adds +-20% noise, so we use midpoint + tolerance for matching.
#
#   Fiber_Aging:          mult=2.8  -> bbe_max ~ 28.0
#   XCON_Port_Down:       mult=3.2  -> bbe_max ~ 32.0
#   Line_Disconnect:      mult=3.0  -> bbe_max ~ 36.0  (step_recovery: 10*3.0*1.2)
#   Fiber_Crack:          mult=3.5  -> bbe_max ~ 42.0
#   XCON_Buffer_Overflow: mult=4.6  -> bbe_max ~ 46.0
#   XCON_Fabric_Fault:    mult=4.2  -> bbe_max ~ 50.4
#   Fiber_Cut:            mult=5.5  -> bbe_max ~ 55.0
#
# Sorted by bbe_max so we can use closest-match when noisy.

BBE_MAX_TABLE: list[tuple[float, str]] = [
    (28.0, "Fiber_Aging"),
    (32.0, "XCON_Port_Down"),
    (36.0, "Line_Disconnect"),
    (42.0, "Fiber_Crack"),
    (46.0, "XCON_Buffer_Overflow"),
    (50.4, "XCON_Fabric_Fault"),
    (55.0, "Fiber_Cut"),
]


def _predict_event_from_bbe_max(bbe_max: float) -> str:
    """Predict event type from bbe_max using closest-match against the lookup table."""
    if bbe_max <= 0:
        return ""
    best_event = ""
    best_dist = float("inf")
    for ref_bbe, event in BBE_MAX_TABLE:
        dist = abs(bbe_max - ref_bbe)
        if dist < best_dist:
            best_dist = dist
            best_event = event
    return best_event


# ── Feature extraction from prompt text ─────────────────────────────────────

def _extract_pattern_features(user_text: str) -> dict[str, Any]:
    """Extract pattern_features values from the user prompt text."""
    features: dict[str, Any] = {}

    pf_match = re.search(r"pattern_features:.*?bbe_max=([\d.]+)", user_text)
    pk_match = re.search(r"pattern_features:.*?pattern_kind=(\w+)", user_text)
    pb_match = re.search(r"pattern_features:.*?peak_board=([\w\-\$]+)", user_text)
    pn_match = re.search(r"pattern_features:.*?peak_node=([\w]+)", user_text)
    br_match = re.search(r"pattern_features:.*?burst_regions=(\d+)", user_text)
    lbf_match = re.search(r"pattern_features:.*?longest_burst_frac=([\d.]+)", user_text)
    bber_match = re.search(r"pattern_features:.*?bber_bbe_ratio=([\d.]+)", user_text)
    es_match = re.search(r"pattern_features:.*?es_frac=([\d.]+)", user_text)

    if pf_match:
        features["bbe_max"] = float(pf_match.group(1))
    if pk_match:
        features["pattern_kind"] = pk_match.group(1)
    if pb_match:
        features["peak_board"] = pb_match.group(1)
    if pn_match:
        features["peak_node"] = pn_match.group(1)
    if br_match:
        features["burst_regions"] = int(br_match.group(1))
    if lbf_match:
        features["longest_burst_frac"] = float(lbf_match.group(1))
    if bber_match:
        features["bber_bbe_ratio"] = float(bber_match.group(1))
    if es_match:
        features["es_frac"] = float(es_match.group(1))

    return features


def _extract_lightpath_lines(user_text: str) -> list[str]:
    """Extract raw lightpath lines from the user prompt.

    The prompt contains a section like:
        topology/lightpaths lines=N sample:
        ROADM0-Tributary041$0-SRC,...,ROADM1-Tributary051$0-TERMINATION
        ...

    We extract those comma-separated lightpath lines.
    """
    lines = user_text.split("\n")
    lightpath_lines: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("topology/lightpaths"):
            in_section = True
            continue
        if in_section:
            # Lightpath lines start with ROADM
            if stripped.startswith("ROADM"):
                lightpath_lines.append(stripped)
            else:
                # End of lightpath section
                if lightpath_lines:
                    break
    return lightpath_lines


# ── Board type classification ───────────────────────────────────────────────

def _classify_board_type(board_name: str) -> str:
    """Classify board into category: Fiber, XCON, or Line.

    Parse the board type from the portion after the dash in the board name.
    E.g. ROADM0-OA002$0 -> "OA" -> Fiber
    """
    upper = board_name.upper()
    dash_idx = upper.find("-")
    if dash_idx < 0:
        return "unknown"
    after_dash = upper[dash_idx + 1:]
    for prefix in ["XCON"]:
        if after_dash.startswith(prefix):
            return "XCON"
    for prefix in ["LINE", "TRIBUTARY"]:
        if after_dash.startswith(prefix):
            return "Line"
    for prefix in ["OA", "OD", "OM", "FIU", "SC2"]:
        if after_dash.startswith(prefix):
            return "Fiber"
    return "unknown"


# ── Decision tree ───────────────────────────────────────────────────────────

DECISION_TREE: dict[tuple[str, str], str] = {
    # (board_category, pattern_kind) -> event_name
    ("Fiber", "ramp"):            "Fiber_Aging",
    ("Fiber", "step"):            "Fiber_Cut",
    ("Fiber", "step_recovery"):   "Fiber_Crack",
    ("Fiber", "burst"):           "Fiber_Aging",       # fallback for Fiber+burst
    ("XCON", "step"):             "XCON_Port_Down",
    ("XCON", "burst"):            "XCON_Buffer_Overflow",
    ("XCON", "step_recovery"):    "XCON_Fabric_Fault",
    ("XCON", "ramp"):             "XCON_Port_Down",    # fallback for XCON+ramp
    ("Line", "step"):             "Line_Disconnect",
    ("Line", "step_recovery"):    "Line_Disconnect",
    ("Line", "ramp"):             "Line_Disconnect",
    ("Line", "burst"):            "Line_Disconnect",
}


def _predict_event(features: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Apply decision tree to predict root cause event.

    Uses a two-tier approach:
      Tier 1 (primary): bbe_max closest-match against the known lookup table.
             Each event type has a unique bbe_max value, making this the
             single most reliable discriminator.
      Tier 2 (secondary): Board type + pattern_kind decision tree.
             Used when pattern_kind is available to confirm or override.

    The board type (from peak_board name) is always used for category
    validation -- if bbe_max predicts a category that disagrees with the
    board type, the board type wins within its category.

    Returns (predicted_event, predicted_board, reasoning_lines).
    """
    reasoning: list[str] = []

    peak_board = features.get("peak_board", "")
    pattern_kind = features.get("pattern_kind", "")
    bbe_max = features.get("bbe_max", 0.0)
    burst_regions = features.get("burst_regions", 0)
    bber_ratio = features.get("bber_bbe_ratio", 0.0)
    es_frac = features.get("es_frac", 0.0)

    if not peak_board:
        reasoning.append("Step 1: No peak_board found in pattern_features. Cannot determine board type.")
        return "unknown", "", reasoning

    # Step 1: Board type -> category
    category = _classify_board_type(peak_board)
    reasoning.append(f"Step 1: peak_board={peak_board} -> board category: {category}")

    # Step 2a: bbe_max-based prediction (primary discriminator)
    bbe_predicted = _predict_event_from_bbe_max(bbe_max) if bbe_max > 0 else ""

    # Step 2b: pattern_kind-based prediction (secondary, if available)
    pk_predicted = ""
    if pattern_kind and pattern_kind != "unknown":
        key = (category, pattern_kind)
        pk_predicted = DECISION_TREE.get(key, "")

    # Step 2c: Resolve prediction
    #   - If pattern_kind gives a prediction, prefer it (it uses board type + temporal shape)
    #   - Otherwise, use bbe_max prediction
    #   - Cross-check: if bbe_max prediction category matches board category, use it
    #   - If categories conflict, fall back to category-default from bbe_max within
    #     the correct board category

    category_map = {
        "Fiber_Aging": "Fiber", "Fiber_Crack": "Fiber", "Fiber_Cut": "Fiber",
        "XCON_Port_Down": "XCON", "XCON_Buffer_Overflow": "XCON", "XCON_Fabric_Fault": "XCON",
        "Line_Disconnect": "Line",
    }

    if pk_predicted:
        predicted_event = pk_predicted
        reasoning.append(
            f"Step 2: pattern_kind={pattern_kind}, burst_regions={burst_regions} "
            f"-> predicted event: {predicted_event}"
        )
        if bbe_predicted and bbe_predicted != predicted_event:
            reasoning.append(
                f"  (bbe_max={bbe_max:.1f} suggests {bbe_predicted}, "
                f"but pattern_kind overrides)"
            )
    elif bbe_predicted:
        bbe_category = category_map.get(bbe_predicted, "unknown")
        if bbe_category == category:
            # bbe_max prediction agrees with board category -> use it
            predicted_event = bbe_predicted
            reasoning.append(
                f"Step 2: bbe_max={bbe_max:.1f} closest match -> {predicted_event} "
                f"(category {bbe_category} matches board category {category})"
            )
        else:
            # Category conflict: bbe_max says different category than the board.
            # Trust the board category and pick the bbe_max-closest event WITHIN
            # the board's category.
            candidates = [
                (ref, evt) for ref, evt in BBE_MAX_TABLE
                if category_map.get(evt) == category
            ]
            if candidates:
                best = min(candidates, key=lambda x: abs(x[0] - bbe_max))
                predicted_event = best[1]
                reasoning.append(
                    f"Step 2: bbe_max={bbe_max:.1f} suggests {bbe_predicted} ({bbe_category}), "
                    f"but board is {category}. Closest {category} event by bbe_max -> {predicted_event}"
                )
            else:
                # Last resort fallback
                fallback_map = {"Fiber": "Fiber_Cut", "XCON": "XCON_Port_Down", "Line": "Line_Disconnect"}
                predicted_event = fallback_map.get(category, "unknown")
                reasoning.append(
                    f"Step 2: bbe_max={bbe_max:.1f} category conflict, "
                    f"fallback -> {predicted_event}"
                )
    else:
        # No bbe_max, no pattern_kind -> pure category fallback
        fallback_map = {"Fiber": "Fiber_Cut", "XCON": "XCON_Port_Down", "Line": "Line_Disconnect"}
        predicted_event = fallback_map.get(category, "unknown")
        reasoning.append(
            f"Step 2: No bbe_max or pattern_kind available, "
            f"category fallback -> {predicted_event}"
        )

    # Step 3: Secondary feature confirmation
    confirm_parts = []
    if bber_ratio > 0:
        confirm_parts.append(f"bber_bbe_ratio={bber_ratio:.2f}")
    if es_frac > 0:
        confirm_parts.append(f"es_frac={es_frac:.4f}")
    if bbe_max > 0:
        confirm_parts.append(f"bbe_max={bbe_max:.1f}")
    if confirm_parts:
        reasoning.append(f"Step 3: Secondary features: {', '.join(confirm_parts)}")

    # Step 4: Conclusion
    reasoning.append(f"Conclusion: Root cause is {predicted_event} on board {peak_board}.")

    return predicted_event, peak_board, reasoning


# ── Role assignment for forward propagation ─────────────────────────────────

# Fiber events occur on optical boards (RELAY role).
# XCON/Line events occur on electrical boards (SRC role, i.e. ODUk-Creation).
EVENT_ROLE_MAP: dict[str, str] = {
    "Fiber_Aging":          "RELAY",
    "Fiber_Crack":          "RELAY",
    "Fiber_Cut":            "RELAY",
    "Line_Disconnect":      "SRC",
    "XCON_Port_Down":       "SRC",
    "XCON_Fabric_Fault":    "SRC",
    "XCON_Buffer_Overflow": "SRC",
}


# ── Forward alarm propagation ───────────────────────────────────────────────

def _run_forward_propagation(
    predicted_board: str,
    predicted_event: str,
    rules_path: Path,
    lightpaths_path: Path,
) -> pd.DataFrame:
    """Run alarm_generator forward propagation for the predicted root cause.

    Creates a temporary failure.csv with the correct BoardRole, calls
    generate_alarm_flow, and returns the resulting alarm_flow DataFrame.
    """
    rules_df = load_rule_database(str(rules_path))
    board_role = EVENT_ROLE_MAP.get(predicted_event, "SRC")

    # Create temporary failure.csv
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as tmp_fail:
        writer = csv.writer(tmp_fail)
        writer.writerow(["Board", "Event", "Time", "Severity", "DurationMs", "BoardRole"])
        writer.writerow([
            predicted_board,
            predicted_event,
            "2025-10-26 00:00:42",
            "Critical",
            "60000",
            board_role,
        ])
        tmp_fail_path = tmp_fail.name

    # Create temporary output path
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        generate_alarm_flow(
            graphml_path=str(lightpaths_path),  # unused when paths_file is set
            df_rules=rules_df,
            out_csv=tmp_out_path,
            failures_csv=tmp_fail_path,
            paths_file=str(lightpaths_path),
            max_hops=4,
            rng_seed=2025,
        )
        result = pd.read_csv(tmp_out_path)
        return result
    except Exception as e:
        print(f"[warn] forward propagation failed: {e}")
        return pd.DataFrame()
    finally:
        Path(tmp_fail_path).unlink(missing_ok=True)
        Path(tmp_out_path).unlink(missing_ok=True)


# ── Output formatting ──────────────────────────────────────────────────────

def _format_alarm_flow_csv(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Format alarm flow DataFrame as CSV string (matching LLM output style)."""
    if df.empty:
        return "alarm_flow: empty"
    if max_rows > 0 and len(df) > max_rows:
        head = df.head(max_rows).to_csv(index=False).strip()
        return head + f"\n... (truncated {len(df) - max_rows} rows)"
    return df.to_csv(index=False).strip()


def _format_model_output(
    predicted_event: str,
    predicted_board: str,
    reasoning: list[str],
    alarm_flow_df: pd.DataFrame,
) -> str:
    """Format output to look like what the LLM would produce.

    Must contain "Root cause (predicted):" and "alarm_flow rows (CSV):"
    sections so that evaluate.py can parse them.
    """
    fixes = EVENT_FIXES.get(predicted_event, [
        "Verify telemetry alignment around the failure time.",
        "Confirm topology/roles mapping matches rule database types.",
        "Inspect affected boards and links for physical or configuration faults.",
    ])

    alarm_csv = _format_alarm_flow_csv(alarm_flow_df)

    parts = [
        "Metric-based diagnosis:",
        "\n".join(reasoning),
        "",
        "Root cause (predicted):",
        f"- Board: {predicted_board}",
        f"- Event: {predicted_event}",
        f"- Time: 2025-10-26 00:00:42",
        f"- Severity: Critical",
        "",
        "Predicted alarm_flow rows (CSV):",
        alarm_csv,
        "",
        "RCA reasoning:",
        "- Rule-based reverse lookup baseline.",
        f"- Board type + pattern_kind decision tree -> {predicted_event}.",
        "- Forward propagation via alarm_generator.py.",
        "Fix ideas:",
    ]
    for i, fix in enumerate(fixes):
        parts.append(f"{i + 1}) {fix}")

    return "\n".join(parts)


# ── Main pipeline ───────────────────────────────────────────────────────────

def process_example(
    example: dict[str, Any],
    sample_index: int,
    rules_path: Path,
    lightpaths_path: Path,
) -> dict[str, Any]:
    """Process one GRPO test example and produce baseline output."""
    # Extract the user prompt text
    prompt = example.get("prompt", [])
    user_text = ""
    for msg in prompt:
        if msg.get("role") == "user":
            user_text = msg["content"]
            break

    # Step 1-2: Extract features from prompt
    features = _extract_pattern_features(user_text)

    # Step 3-4: Apply decision tree
    predicted_event, predicted_board, reasoning = _predict_event(features)

    # Step 5: Forward propagation
    alarm_flow_df = pd.DataFrame()
    if predicted_board and predicted_event != "unknown":
        alarm_flow_df = _run_forward_propagation(
            predicted_board, predicted_event, rules_path, lightpaths_path,
        )

    # Format model output
    model_output = _format_model_output(
        predicted_event, predicted_board, reasoning, alarm_flow_df,
    )

    # Extract alarm flow metadata for evaluation
    if not alarm_flow_df.empty:
        acols = {str(c).strip().lower(): c for c in alarm_flow_df.columns}
        c_prop_board = acols.get("proptoboard")
        c_prop_alarm = acols.get("propalarm")
        c_severity = acols.get("severity")

        pred_boards = sorted(alarm_flow_df[c_prop_board].dropna().unique().tolist()) if c_prop_board else []
        pred_alarms = alarm_flow_df[c_prop_alarm].value_counts().to_dict() if c_prop_alarm else {}
        pred_severity = alarm_flow_df[c_severity].value_counts().to_dict() if c_severity else {}
        pred_row_count = len(alarm_flow_df)
    else:
        pred_boards = [predicted_board] if predicted_board else []
        pred_alarms = {predicted_event: 1} if predicted_event != "unknown" else {}
        pred_severity = {}
        pred_row_count = 0

    # Build output matching evaluate.py expected format
    result = {
        "sample_index": sample_index,
        "model_output": model_output,
        "ground_truth_root_event": example.get("ground_truth_root_event", ""),
        "ground_truth_root_board": example.get("ground_truth_root_board", ""),
        "ground_truth_boards": example.get("ground_truth_boards", "[]"),
        "ground_truth_alarms": example.get("ground_truth_alarms", "{}"),
        "ground_truth_severity_dist": example.get("ground_truth_severity_dist", "{}"),
        "ground_truth_row_count": example.get("ground_truth_row_count", 0),
        "predicted_event": predicted_event,
        "predicted_board": predicted_board,
        "predicted_boards": json.dumps(pred_boards),
        "predicted_alarms": json.dumps(pred_alarms),
        "predicted_severity_dist": json.dumps(pred_severity),
        "predicted_row_count": pred_row_count,
        "method": "rule_reverse",
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rule-based reverse-lookup baseline for RCA comparison."
    )
    ap.add_argument(
        "--test-file", type=Path, required=True,
        help="Path to GRPO JSONL test file (same format as train.jsonl)",
    )
    ap.add_argument(
        "--rules", type=Path,
        default=Path("outputs/Static files/rule_database.csv"),
        help="Path to rule_database.csv",
    )
    ap.add_argument(
        "--lightpaths", type=Path,
        default=Path("regen_compare/noregen/topology/lightpaths.txt"),
        help="Path to lightpaths.txt for forward propagation",
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path("outputs/eval/baseline_rule_outputs.jsonl"),
        help="Output JSONL path",
    )
    ap.add_argument(
        "--max-samples", type=int, default=0,
        help="Max samples to process (0 = all)",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample details",
    )
    args = ap.parse_args()

    if not args.test_file.exists():
        raise SystemExit(f"[error] test file not found: {args.test_file}")
    if not args.rules.exists():
        raise SystemExit(f"[error] rules file not found: {args.rules}")
    if not args.lightpaths.exists():
        raise SystemExit(f"[error] lightpaths file not found: {args.lightpaths}")

    # Load test examples
    examples: list[dict[str, Any]] = []
    with args.test_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if args.max_samples > 0:
        examples = examples[:args.max_samples]

    print(f"[info] loaded {len(examples)} test examples from {args.test_file}")
    print(f"[info] rules: {args.rules}")
    print(f"[info] lightpaths: {args.lightpaths}")

    # Process each example
    results: list[dict[str, Any]] = []
    correct_event = 0
    correct_board = 0

    for i, example in enumerate(examples):
        result = process_example(example, i, args.rules, args.lightpaths)
        results.append(result)

        gt_event = result["ground_truth_root_event"]
        gt_board = result["ground_truth_root_board"]
        pred_event = result["predicted_event"]
        pred_board = result["predicted_board"]

        event_match = pred_event == gt_event
        board_match = pred_board == gt_board
        if event_match:
            correct_event += 1
        if board_match:
            correct_board += 1

        if args.verbose:
            status = "OK" if event_match and board_match else "MISS"
            print(
                f"  [{status}] sample {i}: "
                f"pred={pred_event}/{pred_board} "
                f"gt={gt_event}/{gt_board}"
            )

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Print summary
    n = len(results)
    print(f"\n{'=' * 60}")
    print(f"Rule-Based Baseline Results ({n} samples)")
    print(f"{'=' * 60}")
    print(f"Event Accuracy:  {correct_event}/{n} = {correct_event / n * 100:.1f}%")
    print(f"Board Accuracy:  {correct_board}/{n} = {correct_board / n * 100:.1f}%")

    # Category accuracy (Fiber/XCON/Line match)
    category_map = {
        "Fiber_Aging": "Fiber", "Fiber_Crack": "Fiber", "Fiber_Cut": "Fiber",
        "XCON_Port_Down": "XCON", "XCON_Buffer_Overflow": "XCON", "XCON_Fabric_Fault": "XCON",
        "Line_Disconnect": "Line",
    }
    correct_cat = sum(
        1 for r in results
        if category_map.get(r["predicted_event"]) == category_map.get(r["ground_truth_root_event"])
    )
    print(f"Category Accuracy: {correct_cat}/{n} = {correct_cat / n * 100:.1f}%")

    # Per-event breakdown
    print(f"\nPer-event breakdown:")
    from collections import Counter
    event_counts: Counter[str] = Counter()
    event_correct: Counter[str] = Counter()
    for r in results:
        gt = r["ground_truth_root_event"]
        event_counts[gt] += 1
        if r["predicted_event"] == gt:
            event_correct[gt] += 1
    for evt in sorted(event_counts):
        total = event_counts[evt]
        correct = event_correct[evt]
        print(f"  {evt:30s}  {correct}/{total} = {correct / total * 100:.1f}%")

    print(f"\n[ok] wrote {n} results -> {args.out}")


if __name__ == "__main__":
    main()
