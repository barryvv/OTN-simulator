#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RULES = Path("outputs/Static files/rule_database.csv")

SYSTEM_PROMPT = (
    "You are a telecom alarm RCA + propagation predictor. "
    "Given ONLY metrics summaries, topology/lightpaths, and the rule database, "
    "infer the most likely root cause and PREDICT the alarm_flow rows. "
    "Do NOT assume failure.csv or alarm_flow.csv is available."
)

METRIC_GUIDE = (
    "Use the following feature-based reasoning to identify the failure type and root board.\n\n"
    "Step 1 — Board type → failure category:\n"
    "  OA/OD/OM/FIU boards → Fiber event (Aging, Crack, or Cut)\n"
    "  XCON boards → XCON event (Port_Down, Buffer_Overflow, or Fabric_Fault)\n"
    "  Line boards → Line_Disconnect\n\n"
    "Step 2 — Temporal pattern → specific event:\n"
    "  'ramp' (gradual rise/fall, 1-2 regions) → Fiber_Aging\n"
    "  'step' (sudden sustained, 1 region) → Fiber_Cut or XCON_Port_Down\n"
    "  'step_recovery' (sustained + recovery spikes, multiple regions) → Fiber_Crack, XCON_Fabric_Fault, or Line_Disconnect\n"
    "  'burst' (many short spikes, high burst_regions) → XCON_Buffer_Overflow\n\n"
    "Step 3 — Secondary features to confirm:\n"
    "  bber_bbe_ratio: >4.0 → Fiber_Crack; <2.0 → Line_Disconnect\n"
    "  es_frac: high → Fiber_Cut; moderate → XCON events\n"
    "  burst_regions: >=5 → burst pattern; <=2 → step or ramp\n\n"
    "Step 4 — peak_board = root cause board (the board with the highest bbe_max)"
)

EVENT_FIXES = {
    "Fiber_Aging": [
        "Check span loss/attenuation and compare to baseline.",
        "Inspect and clean connectors; verify patch panel integrity.",
        "Adjust amplifier gain or replace degraded fiber segment.",
    ],
    "Fiber_Crack": [
        "Inspect fiber route for bends/physical damage.",
        "Check splice points and connector endfaces.",
        "Replace the damaged segment if loss is abnormal.",
    ],
    "Fiber_Cut": [
        "Dispatch field crew to locate cut segment.",
        "Switch traffic to protection path if available.",
        "Splice or replace the cut fiber segment.",
    ],
    "Line_Disconnect": [
        "Verify line card/port status and cabling.",
        "Check cross-connect mappings and patch cords.",
        "Reseat or replace the line interface module.",
    ],
    "XCON_Port_Down": [
        "Validate XCON port health and firmware logs.",
        "Check port cabling and optical power levels.",
        "Reset or replace the affected XCON port.",
    ],
    "XCON_Fabric_Fault": [
        "Inspect fabric health metrics and error counters.",
        "Restart fabric processes if stuck.",
        "Replace faulty fabric module if errors persist.",
    ],
    "XCON_Buffer_Overflow": [
        "Check traffic load and congestion points.",
        "Verify QoS settings and buffer thresholds.",
        "Mitigate by rerouting or rate-limiting traffic.",
    ],
}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _rules_for_event(rules: pd.DataFrame, event: str) -> list[str]:
    if rules.empty:
        return []
    cols = {str(c).strip().lower(): c for c in rules.columns}
    c_fn = cols.get("functiontype") or cols.get("fromtype")
    c_ev = cols.get("eventname") or cols.get("fromalarm")
    c_act = cols.get("actiontype") or cols.get("totype")
    c_tgt = cols.get("actiontarget") or cols.get("edgetype")
    c_out = cols.get("outputalarm") or cols.get("toalarm")
    if not c_ev:
        return []
    sub = rules[rules[c_ev].astype(str).str.strip() == event]
    out = []
    for _, r in sub.iterrows():
        fn = str(r.get(c_fn, "")).strip()
        act = str(r.get(c_act, "")).strip()
        tgt = str(r.get(c_tgt, "")).strip()
        out_alarm = str(r.get(c_out, "")).strip()
        desc = f"{fn} {event} -> {out_alarm}"
        if act:
            desc += f" ({act}"
            if tgt:
                desc += f" to {tgt}"
            desc += ")"
        out.append(desc)
    return out


def _alarm_flow_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "alarm_flow: empty"
    cols = {str(c).strip().lower(): c for c in df.columns}
    c_src = cols.get("sourceboard")
    c_prop = cols.get("proptoboard")
    c_alarm = cols.get("propalarm")
    parts = [f"rows={len(df)}"]
    if c_src:
        parts.append(f"source_boards={df[c_src].nunique()}")
    if c_prop:
        parts.append(f"prop_boards={df[c_prop].nunique()}")
    if c_alarm:
        parts.append(f"top_alarms={df[c_alarm].value_counts().head(5).to_dict()}")
    return "alarm_flow: " + " | ".join(parts)


def _alarm_flow_rows(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "alarm_flow: empty"
    if max_rows > 0 and len(df) > max_rows:
        head = df.head(max_rows).to_csv(index=False).strip()
        return head + f"\n... (truncated {len(df) - max_rows} rows)"
    return df.to_csv(index=False).strip()


def _compute_pattern_features(df: pd.DataFrame) -> dict:
    """Compute temporal pattern features from a timeseries DataFrame.

    Key discriminative features:
    - bbe_max: primary discriminator (unique values per event type)
    - es_frac: secondary discriminator when bbe_max is ambiguous
    - burst_regions: count of distinct failure episodes (threshold = max/2)
    - longest_burst_frac: sustained vs bursty pattern indicator
    """
    import numpy as np
    features = {}
    bbe = pd.to_numeric(df.get("BBE", pd.Series(dtype=float)), errors="coerce").dropna()
    bber = pd.to_numeric(df.get("BBER", pd.Series(dtype=float)), errors="coerce").dropna()
    es = pd.to_numeric(df.get("ES", pd.Series(dtype=float)), errors="coerce").dropna()

    if bbe.empty or len(bbe) < 100:
        return features

    bbe_vals = bbe.values.astype(float)
    max_bbe = float(np.max(bbe_vals))

    # ── bbe_max: primary discriminator ──
    features["bbe_max"] = round(max_bbe, 1)

    # ── burst_regions: count distinct failure episodes ──
    # Use max/2 threshold to only count significant failure spikes
    threshold = max(max_bbe * 0.5, 5.0)
    above = bbe_vals > threshold
    regions = 0
    in_region = False
    region_lengths = []
    current_len = 0
    for v in above:
        if v and not in_region:
            regions += 1
            in_region = True
            current_len = 1
        elif v and in_region:
            current_len += 1
        elif not v and in_region:
            region_lengths.append(current_len)
            in_region = False
            current_len = 0
    if in_region:
        region_lengths.append(current_len)
    features["burst_regions"] = regions

    # ── longest_burst_frac: sustained (near 1.0) vs bursty (near 0) ──
    if region_lengths:
        longest = max(region_lengths)
        total_elevated = sum(region_lengths)
        features["longest_burst_frac"] = round(longest / total_elevated, 3) if total_elevated > 0 else 0.0
    else:
        features["longest_burst_frac"] = 0.0

    # ── es_frac: secondary discriminator ──
    if not es.empty:
        features["es_frac"] = round(float((es.values > 0.5).sum()) / len(es), 4)

    # ── bber_bbe_ratio ──
    if not bber.empty and max_bbe > 0:
        max_bber = float(np.max(bber.values))
        features["bber_bbe_ratio"] = round(max_bber / max_bbe * 1000, 2)

    # ── pattern_kind detection ──
    # Use 0.9*max_bbe threshold to filter out propagated alarm noise (AIS BBE ~24.0).
    # The alarm flow can apply the root event profile multiple times at different
    # times (once per lightpath), so we CLUSTER nearby high-threshold regions and
    # analyze a single cluster to determine the temporal pattern.
    high_thresh = max_bbe * 0.9
    above_high = bbe_vals > high_thresh
    high_regions = []
    start_idx = None
    for i, v in enumerate(above_high):
        if v and start_idx is None:
            start_idx = i
        elif not v and start_idx is not None:
            high_regions.append((start_idx, i))
            start_idx = None
    if start_idx is not None:
        high_regions.append((start_idx, len(bbe_vals)))

    # Cluster high regions by proximity (gap > 500 timesteps = separate event copy)
    CLUSTER_GAP = 500
    clusters = []
    cur_cluster = []
    for reg in high_regions:
        if cur_cluster and (reg[0] - cur_cluster[-1][1] > CLUSTER_GAP):
            clusters.append(cur_cluster)
            cur_cluster = [reg]
        else:
            cur_cluster.append(reg)
    if cur_cluster:
        clusters.append(cur_cluster)

    # Pick the largest cluster (most total elevated width)
    if clusters:
        best = max(clusters, key=lambda c: sum(e - s for s, e in c))
    else:
        best = []

    n_in_cluster = len(best)
    cluster_width = sum(e - s for s, e in best) if best else 0

    if n_in_cluster <= 1:
        # Single region in cluster — step or ramp
        # Step profiles create a WIDE sustained block at max_bbe (width > 15)
        # Ramp profiles create a NARROW peak at max_bbe (only the top of the ramp)
        if n_in_cluster == 1 and cluster_width > 15:
            features["pattern_kind"] = "step"
        else:
            features["pattern_kind"] = "ramp"
    else:
        # Multiple regions in one cluster — burst or step_recovery
        # step_recovery: sustained base step (~0.83*max) between 1.2x spikes
        # burst: baseline (~0) between burst peaks
        inter_vals = []
        for i in range(len(best) - 1):
            gap_s = best[i][1]
            gap_e = best[i + 1][0]
            if gap_e > gap_s:
                inter_vals.extend(bbe_vals[gap_s:gap_e].tolist())
        if inter_vals:
            inter_mean = float(np.mean(inter_vals))
            if inter_mean > 0.7 * max_bbe:
                features["pattern_kind"] = "step_recovery"
            else:
                features["pattern_kind"] = "burst"
        else:
            features["pattern_kind"] = "burst"

    return features


def _timeseries_summary(event: str, root: Path, max_files: int = 3) -> str:
    event_key = event.strip().replace(" ", "_")
    event_key_lower = event_key.lower()
    cand_dirs = [
        root / f"data_outputs_{event_key}",
        root / event_key,
        root / f"data_outputs_{event_key_lower}",
        root / event_key_lower,
        root / "data_outputs",
    ]
    ts_dir = next((d for d in cand_dirs if d.exists()), None)
    if not ts_dir:
        return "timeseries: missing"

    files = sorted(ts_dir.glob("Node_*/timeseries_*.csv"))
    if not files:
        return "timeseries: no timeseries files found"

    # Collect per-board max BBE across all files
    board_max_bbe: dict[str, float] = {}
    all_dfs = []
    for f in files:
        df = _read_csv(f)
        if df.empty or "BBE" not in df.columns:
            continue
        all_dfs.append(df)
        if "board" in df.columns:
            for bname, bdf in df.groupby("board"):
                mx = pd.to_numeric(bdf["BBE"], errors="coerce").max()
                bname_str = str(bname)
                if bname_str not in board_max_bbe or mx > board_max_bbe[bname_str]:
                    board_max_bbe[bname_str] = float(mx)

    # Sort boards by max BBE descending
    sorted_boards = sorted(board_max_bbe.items(), key=lambda x: x[1], reverse=True)

    # Build per-board summary (show top boards)
    max_boards_shown = max_files * 3  # show more boards since they're compact
    board_rows = []
    for bname, mx in sorted_boards[:max_boards_shown]:
        board_rows.append(f"  {bname}: bbe_max={mx:.1f}")
    result = "timeseries per-board bbe_max:\n" + "\n".join(board_rows)

    # Compute pattern features from the highest-BBE board's data only
    peak_board = ""
    peak_node = ""
    if sorted_boards:
        peak_board = sorted_boards[0][0]
        if "-" in peak_board:
            peak_node = peak_board.split("-")[0]

    # Extract peak board's timeseries data from the DataFrames
    peak_features = {}
    if peak_board:
        peak_rows = []
        for df in all_dfs:
            if "board" in df.columns:
                bdf = df[df["board"].astype(str).str.strip() == peak_board]
                if not bdf.empty:
                    peak_rows.append(bdf)
        if peak_rows:
            peak_df = pd.concat(peak_rows, ignore_index=True)
            peak_features = _compute_pattern_features(peak_df)

    # Fallback: compute from all data if peak board extraction failed
    if not peak_features:
        for df in all_dfs:
            pf = _compute_pattern_features(df)
            if pf and pf.get("bbe_max", 0) > peak_features.get("bbe_max", 0):
                peak_features = pf

    if peak_features:
        pb_str = f" peak_board={peak_board}" if peak_board else ""
        pn_str = f" peak_node={peak_node}" if peak_node else ""
        pk_str = f" pattern_kind={peak_features['pattern_kind']}" if "pattern_kind" in peak_features else ""
        result += (
            f"\npattern_features: burst_regions={peak_features.get('burst_regions', 0)}"
            f" longest_burst_frac={peak_features.get('longest_burst_frac', 0.0):.3f}"
            f" bbe_max={peak_features.get('bbe_max', 0.0):.1f}"
            f" es_frac={peak_features.get('es_frac', 0.0):.4f}"
            f" bber_bbe_ratio={peak_features.get('bber_bbe_ratio', 0.0):.2f}"
            f"{pk_str}{pb_str}{pn_str}"
        )

    return result


def _board_catalog(run_dir: Path) -> str:
    """Extract unique boards per ROADM from lightpaths for board localization."""
    import re as _re
    paths = run_dir / "mw_lightpaths_roles.txt"
    if not paths.exists():
        paths = run_dir / "lightpaths.txt"
    if not paths.exists():
        return ""
    text = paths.read_text(encoding="utf-8", errors="ignore")
    # Extract all board names like ROADM0-OA002$0
    boards = set(_re.findall(r"(ROADM\d+-\w+\$\d+)", text))
    # Group by ROADM
    roadm_boards: dict[str, list[str]] = {}
    for b in boards:
        roadm = b.split("-")[0]
        roadm_boards.setdefault(roadm, []).append(b)
    if not roadm_boards:
        return ""
    lines = ["board_catalog:"]
    for roadm in sorted(roadm_boards):
        board_list = sorted(roadm_boards[roadm])
        lines.append(f"  {roadm}: {', '.join(board_list)}")
    return "\n".join(lines)


def _lightpaths_summary(run_dir: Path, max_lines: int = 3) -> str:
    paths = run_dir / "mw_lightpaths_roles.txt"
    if not paths.exists():
        paths = run_dir / "lightpaths.txt"
    if not paths.exists():
        return "topology/lightpaths: missing"
    lines = [ln for ln in paths.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    sample = "\n".join(lines[:max_lines])
    return f"topology/lightpaths lines={len(lines)} sample:\n{sample}"


def _rules_text(path: Path) -> str:
    if not path.exists():
        return "(missing rule database)"
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _build_cot_reasoning(ts_summary: str, event: str, board: str) -> list[str]:
    """Build chain-of-thought reasoning lines from pattern_features using feature-based logic."""
    import re
    lines = []

    # Extract features from the pattern_features line
    pf_match = re.search(r"pattern_features:.*?bbe_max=([\d.]+)", ts_summary)
    pb_match = re.search(r"pattern_features:.*?peak_board=([\w\-\$]+)", ts_summary)
    pk_match = re.search(r"pattern_features:.*?pattern_kind=(\w+)", ts_summary)
    br_match = re.search(r"pattern_features:.*?burst_regions=(\d+)", ts_summary)
    bber_match = re.search(r"pattern_features:.*?bber_bbe_ratio=([\d.]+)", ts_summary)
    es_match = re.search(r"pattern_features:.*?es_frac=([\d.]+)", ts_summary)

    bbe_max = float(pf_match.group(1)) if pf_match else None
    peak_board = pb_match.group(1) if pb_match else None
    pattern_kind = pk_match.group(1) if pk_match else None
    burst_regions = int(br_match.group(1)) if br_match else None
    bber_ratio = float(bber_match.group(1)) if bber_match else None
    es_frac = float(es_match.group(1)) if es_match else None

    if peak_board is None:
        lines.append("Step 1: pattern_features not available, using other evidence.")
        lines.append(f"Conclusion: Based on alarm patterns, event is {event}.")
        return lines

    # Step 1: Board type → category
    board_upper = peak_board.upper()
    # Parse board type — careful: "ROADM" contains "OA" so parse after the dash
    board_type = ""
    dash_idx = board_upper.find("-")
    if dash_idx >= 0:
        after_dash = board_upper[dash_idx + 1:]
        for prefix in ["XCON", "LINE", "TRIBUTARY", "OA", "OD", "OM", "FIU", "SC2"]:
            if after_dash.startswith(prefix):
                board_type = prefix
                break

    if board_type in ("OA", "OD", "OM", "FIU", "SC2"):
        category = "Fiber"
        lines.append(f"Step 1: peak_board={peak_board} is a {board_type} board → Fiber event category.")
    elif board_type == "XCON":
        category = "XCON"
        lines.append(f"Step 1: peak_board={peak_board} is an XCON board → XCON event category.")
    elif board_type in ("LINE", "TRIBUTARY"):
        category = "Line"
        lines.append(f"Step 1: peak_board={peak_board} is a {board_type} board → Line_Disconnect.")
    else:
        category = "unknown"
        lines.append(f"Step 1: peak_board={peak_board}, board type unclear, using temporal features.")

    # Step 2: Temporal pattern → narrow event
    pk_str = f"pattern_kind={pattern_kind}" if pattern_kind else "pattern_kind=unknown"
    br_str = f"burst_regions={burst_regions}" if burst_regions is not None else ""

    if category == "Fiber":
        if pattern_kind == "ramp":
            lines.append(f"Step 2: {pk_str}, {br_str} → gradual degradation → Fiber_Aging.")
        elif pattern_kind == "step":
            lines.append(f"Step 2: {pk_str}, {br_str} → sudden sustained loss → Fiber_Cut.")
        elif pattern_kind == "step_recovery":
            lines.append(f"Step 2: {pk_str}, {br_str} → step with recovery spikes → Fiber_Crack.")
        else:
            lines.append(f"Step 2: {pk_str}, {br_str} → unusual fiber pattern, checking secondary features.")
    elif category == "XCON":
        if pattern_kind == "step":
            lines.append(f"Step 2: {pk_str}, {br_str} → sudden sustained → XCON_Port_Down.")
        elif pattern_kind == "burst":
            lines.append(f"Step 2: {pk_str}, {br_str} → bursty spikes → XCON_Buffer_Overflow.")
        elif pattern_kind == "step_recovery":
            lines.append(f"Step 2: {pk_str}, {br_str} → step with recovery → XCON_Fabric_Fault.")
        else:
            lines.append(f"Step 2: {pk_str}, {br_str} → checking secondary features for XCON subtype.")
    elif category == "Line":
        lines.append(f"Step 2: {pk_str}, {br_str} → Line board with step_recovery → Line_Disconnect.")

    # Step 3: Secondary features confirmation
    confirm_parts = []
    if bber_ratio is not None:
        confirm_parts.append(f"bber_bbe_ratio={bber_ratio:.2f}")
    if es_frac is not None:
        confirm_parts.append(f"es_frac={es_frac:.4f}")
    if bbe_max is not None:
        confirm_parts.append(f"bbe_max={bbe_max:.1f}")
    if confirm_parts:
        lines.append(f"Step 3: Secondary features: {', '.join(confirm_parts)} — consistent with {event}.")

    # Step 4: Conclusion
    lines.append(f"Conclusion: Root cause is {event} on board {peak_board}.")

    return lines


def build_example(
    run_dir: Path,
    rules_df: pd.DataFrame,
    rules_path: Path,
    timeseries_root: Path | None = None,
    timeseries_max_files: int = 3,
    alarm_flow_max_rows: int = 20,
    compact_rules: bool = False,
) -> dict[str, Any] | None:
    failure_path = run_dir / "failure.csv"
    alarm_flow_path = run_dir / "alarm_flow.csv"
    if not failure_path.exists() or not alarm_flow_path.exists():
        return None

    df_fail = _read_csv(failure_path)
    df_alarm = _read_csv(alarm_flow_path)
    if df_fail.empty:
        return None

    fcols = {str(c).strip().lower(): c for c in df_fail.columns}
    c_board = fcols.get("board") or list(df_fail.columns)[0]
    c_event = fcols.get("event") or fcols.get("alarm") or list(df_fail.columns)[1]
    c_time = fcols.get("time")
    c_sev = fcols.get("severity")
    c_dur = fcols.get("durationms") or fcols.get("duration")

    row = df_fail.iloc[0]
    board = str(row.get(c_board))
    event = str(row.get(c_event))
    time = str(row.get(c_time)) if c_time else ""
    severity = str(row.get(c_sev)) if c_sev else ""
    duration = str(row.get(c_dur)) if c_dur else ""

    rule_lines = _rules_for_event(rules_df, event)
    rule_text = "\n".join(f"- {r}" for r in rule_lines) if rule_lines else "- (no matching rules found)"

    ts_summary = ""
    if timeseries_root is not None:
        # Try run_dir name first (for grpo_runs layout), then event name
        ts_summary = _timeseries_summary(run_dir.name, timeseries_root, timeseries_max_files)
        if "missing" in ts_summary:
            ts_summary = _timeseries_summary(event, timeseries_root, timeseries_max_files)

    topo_summary = _lightpaths_summary(run_dir)
    catalog = _board_catalog(run_dir)
    if compact_rules:
        from generate_grpo_data import _rules_text_compact
        rules_text = _rules_text_compact(rules_path)
    else:
        rules_text = _rules_text(rules_path)

    catalog_section = f"\n{catalog}\n" if catalog else ""
    user = (
        "Observations (no failure.csv, no alarm_flow.csv):\n"
        f"{topo_summary}\n"
        f"{catalog_section}\n"
        f"{ts_summary}\n\n"
        f"{METRIC_GUIDE}\n\n"
        "Rule database (CSV):\n"
        f"{rules_text}\n\n"
        "Task:\n"
        "1) Infer likely root cause (board, event, time)\n"
        "2) Predict alarm_flow rows (CSV)\n"
        "3) Provide RCA reasoning based on rules + topology + metrics\n"
    )

    fixes = EVENT_FIXES.get(event, [
        "Verify telemetry alignment around the failure time.",
        "Confirm topology/roles mapping matches rule database types.",
        "Inspect affected boards and links for physical or configuration faults.",
    ])

    # ── Build chain-of-thought reasoning from pattern features ──
    cot_lines = _build_cot_reasoning(ts_summary, event, board)

    alarm_rows = _alarm_flow_rows(df_alarm, alarm_flow_max_rows)
    assistant = (
        "Metric-based diagnosis:\n"
        + "\n".join(cot_lines) + "\n\n"
        "Root cause (predicted):\n"
        f"- Board: {board}\n"
        f"- Event: {event}\n"
        f"- Time: {time}\n"
        f"- Severity: {severity}\n\n"
        "Predicted alarm_flow rows (CSV):\n"
        f"{alarm_rows}\n\n"
        "RCA reasoning:\n"
        f"- Rules used:\n{rule_text}\n"
        f"- Topology evidence:\n{topo_summary}\n"
        "- Metrics evidence: see pattern_features analysis above.\n"
        "Fix ideas:\n"
        + "\n".join(f"{i+1}) {fix}" for i, fix in enumerate(fixes))
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate train/val JSONL for SFT.")
    ap.add_argument("--runs-dir", type=Path, default=Path("outputs/alarm_flows/test_runs"))
    ap.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/llm_data"))
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeseries-root", type=Path, default=None,
                    help="Optional root for data_outputs_<event>/Node_*/timeseries_*.csv")
    ap.add_argument("--timeseries-max-files", type=int, default=3)
    ap.add_argument("--alarm-flow-max-rows", type=int, default=20,
                    help="Max alarm_flow rows to include in label output.")
    ap.add_argument("--compact-rules", action="store_true",
                    help="Use compact rules (strip comments and regen-only rows).")
    args = ap.parse_args()

    rules_df = _read_csv(args.rules)
    run_dirs = sorted([p for p in args.runs_dir.iterdir()
                       if p.is_dir() and not p.name.startswith("data_outputs_")])
    examples = []
    for run_dir in run_dirs:
        ex = build_example(
            run_dir,
            rules_df,
            args.rules,
            args.timeseries_root,
            args.timeseries_max_files,
            args.alarm_flow_max_rows,
            compact_rules=args.compact_rules,
        )
        if ex:
            examples.append(ex)

    if not examples:
        raise SystemExit("No examples generated. Check runs dir and failure.csv files.")

    random.seed(args.seed)
    random.shuffle(examples)
    split_idx = max(1, int(len(examples) * args.train_frac))
    train = examples[:split_idx]
    val = examples[split_idx:] or examples[:1]

    args.outdir.mkdir(parents=True, exist_ok=True)
    train_path = args.outdir / "train.jsonl"
    val_path = args.outdir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(train)} train rows -> {train_path}")
    print(f"[ok] wrote {len(val)} val rows -> {val_path}")


if __name__ == "__main__":
    main()
