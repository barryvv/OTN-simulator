#!/usr/bin/env python3
"""Generate GRPO training data from diverse simulator runs.

Output: outputs/grpo_data/train.jsonl with columns:
  - prompt              (list of message dicts — system + user, NO assistant)
  - ground_truth_root_board
  - ground_truth_root_event
  - ground_truth_alarm_flow_csv
  - ground_truth_boards  (JSON list of unique PropToBoard values)
  - ground_truth_alarms  (JSON dict alarm_name -> count)
  - ground_truth_max_hop
  - ground_truth_row_count
  - ground_truth_severity_dist (JSON dict severity -> count)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

# Reuse from generate_sft_data.py
from generate_sft_data import (
    SYSTEM_PROMPT,
    METRIC_GUIDE,
    _read_csv,
    _timeseries_summary,
    _lightpaths_summary,
    _board_catalog,
    _rules_text,
)


def _rules_text_compact(path: Path) -> str:
    """Read rules CSV, strip comments/blanks, and drop regen-only rows."""
    if not path.exists():
        return "(missing rule database)"
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    kept = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Skip regen-specific rules (noregen topology doesn't use them)
        if "RegenIn" in stripped or "RegenOut" in stripped:
            continue
        kept.append(stripped)
    return "\n".join(kept)

ALARM_FLOW_SCHEMA = (
    "SourceBoard,SourceAlarm,SourceEvent,SourceStartTime,SourceDurationSec,"
    "SourceRole,PropToBoard,PropAlarm,PropEvent,PropRole,ArrivalTime,ClearTime,"
    "ViaEdgeType,Layer,Direction,Severity,DelayMs,DelaySec,RuleNote,RuleIdx,Hop,PrevBoard"
)


def _find_timeseries_dir(run_name: str, ts_root: Path) -> Path | None:
    """Find the data_outputs directory for a run."""
    cand = ts_root / f"data_outputs_{run_name}"
    if cand.exists():
        return cand
    # Also check by stripping the run name prefix patterns
    for d in ts_root.iterdir():
        if d.is_dir() and d.name.startswith("data_outputs_") and run_name in d.name:
            return d
    return None


def build_grpo_example(
    run_dir: Path,
    rules_path: Path,
    timeseries_root: Path,
    timeseries_max_files: int = 5,
) -> dict[str, Any] | None:
    """Build one GRPO training example from a simulator run directory."""
    failure_path = run_dir / "failure.csv"
    alarm_flow_path = run_dir / "alarm_flow.csv"
    if not failure_path.exists() or not alarm_flow_path.exists():
        return None

    df_fail = _read_csv(failure_path)
    df_alarm = _read_csv(alarm_flow_path)
    if df_fail.empty or df_alarm.empty:
        return None

    # Extract root cause(s) from failure.csv
    fcols = {str(c).strip().lower(): c for c in df_fail.columns}
    c_board = fcols.get("board") or list(df_fail.columns)[0]
    c_event = fcols.get("event") or fcols.get("alarm") or list(df_fail.columns)[1]

    # Primary root cause (first row, for backwards compatibility)
    row = df_fail.iloc[0]
    board = str(row.get(c_board))
    event = str(row.get(c_event))

    # All root causes (for multi-failure support)
    all_root_boards = [str(r) for r in df_fail[c_board].tolist()]
    all_root_events = [str(r) for r in df_fail[c_event].tolist()]
    n_failures = len(df_fail)

    # Build timeseries summary — use run_dir name to find matching data_outputs
    run_name = run_dir.name
    ts_summary = _timeseries_summary(run_name, timeseries_root, timeseries_max_files)
    if "missing" in ts_summary:
        # Fallback: use event name
        ts_summary = _timeseries_summary(event, timeseries_root, timeseries_max_files)

    topo_summary = _lightpaths_summary(run_dir)
    catalog = _board_catalog(run_dir)
    rules_full = _rules_text_compact(rules_path)

    catalog_section = f"\n{catalog}\n" if catalog else ""
    # Build prompt (system + user only — GRPO generates the assistant response)
    user_msg = (
        f"{topo_summary}\n"
        f"{catalog_section}\n"
        f"{ts_summary}\n\n"
        f"{METRIC_GUIDE}\n\n"
        f"Rule database (CSV):\n{rules_full}\n\n"
        "Task:\n"
        "1) Infer the likely root cause (board, event, time, severity) using feature-based reasoning:\n"
        "   a) Identify failure category from peak_board type (OA/OD/OM/FIU→Fiber, XCON→XCON, Line→Line)\n"
        "   b) Use pattern_kind + burst_regions to narrow the specific event\n"
        "   c) Confirm with bber_bbe_ratio and es_frac\n"
        f"2) Predict alarm_flow rows in this exact CSV format:\n{ALARM_FLOW_SCHEMA}\n"
        "Use alarm names from the rule database OutputAlarm column.\n"
        "Use board names from topology lightpaths. Walk each lightpath and apply matching rules at each hop.\n"
        "3) Provide RCA reasoning based on rules + topology + metrics\n"
        "4) Suggest 3 concrete fixes\n"
    )

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # Extract ground truth metadata from alarm_flow.csv
    acols = {str(c).strip().lower(): c for c in df_alarm.columns}
    c_prop_board = acols.get("proptoboard")
    c_prop_alarm = acols.get("propalarm")
    c_hop = acols.get("hop")
    c_severity = acols.get("severity")

    gt_boards = sorted(df_alarm[c_prop_board].dropna().unique().tolist()) if c_prop_board else []
    gt_alarms = df_alarm[c_prop_alarm].value_counts().to_dict() if c_prop_alarm else {}
    gt_max_hop = int(df_alarm[c_hop].max()) if c_hop and not df_alarm[c_hop].isna().all() else 0
    gt_severity = df_alarm[c_severity].value_counts().to_dict() if c_severity else {}

    result = {
        "prompt": prompt,
        "ground_truth_root_board": board,
        "ground_truth_root_event": event,
        "ground_truth_alarm_flow_csv": df_alarm.to_csv(index=False),
        "ground_truth_boards": json.dumps(gt_boards),
        "ground_truth_alarms": json.dumps(gt_alarms),
        "ground_truth_max_hop": gt_max_hop,
        "ground_truth_row_count": min(len(df_alarm), 20),
        "ground_truth_severity_dist": json.dumps(gt_severity),
    }

    # Multi-failure metadata (always present, n_failures=1 for single)
    result["ground_truth_root_events"] = json.dumps(all_root_events)
    result["ground_truth_root_boards"] = json.dumps(all_root_boards)
    result["n_failures"] = n_failures

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GRPO training data from simulator runs.")
    ap.add_argument("--runs-dir", type=Path, default=Path("outputs/grpo_runs"),
                    help="Directory containing run subdirectories")
    ap.add_argument("--rules", type=Path, default=Path("outputs/Static files/rule_database.csv"))
    ap.add_argument("--timeseries-root", type=Path, default=None,
                    help="Root for data_outputs_* directories (defaults to --runs-dir)")
    ap.add_argument("--timeseries-max-files", type=int, default=5)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/grpo_data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ts_root = args.timeseries_root or args.runs_dir

    run_dirs = sorted([p for p in args.runs_dir.iterdir() if p.is_dir() and not p.name.startswith("data_outputs_")])
    print(f"[info] found {len(run_dirs)} run directories in {args.runs_dir}")

    examples = []
    for run_dir in run_dirs:
        ex = build_grpo_example(run_dir, args.rules, ts_root, args.timeseries_max_files)
        if ex:
            examples.append(ex)
        else:
            print(f"[warn] skipped {run_dir.name} (missing failure.csv or alarm_flow.csv)")

    if not examples:
        raise SystemExit("No examples generated. Check runs dir.")

    random.seed(args.seed)
    random.shuffle(examples)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / "train.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(examples)} GRPO examples -> {out_path}")
    # Print sample keys for verification
    if examples:
        print(f"[info] sample keys: {list(examples[0].keys())}")
        print(f"[info] sample root: {examples[0]['ground_truth_root_board']} / {examples[0]['ground_truth_root_event']}")
        print(f"[info] sample gt rows: {examples[0]['ground_truth_row_count']}, max_hop: {examples[0]['ground_truth_max_hop']}")


if __name__ == "__main__":
    main()
