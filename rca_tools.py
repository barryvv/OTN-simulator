#!/usr/bin/env python3
"""Tool implementations for the OTN RCA agent.

Provides the agent with structured access to topology, metrics, rules,
and validation capabilities. Each tool returns a plain dict that can
be serialised to JSON for inclusion in the LLM context.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alarm_generator import (
    _board_kind,
    _board_type,
    _compile_rules,
    _parse_paths,
    _role_for_board,
    generate_alarm_flow,
    load_rule_database,
)
from generate_sft_data import _compute_pattern_features, _read_csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOARD_RE = re.compile(r"(ROADM\d+)-(\w+?)(\d+)\$(\d+)")

BOARD_TYPE_CATEGORY = {
    "OA": "optical/fiber",
    "OD": "optical/fiber",
    "OM": "optical/fiber",
    "FIU": "optical/fiber",
    "SC2": "optical/fiber",
    "XCON": "electrical/xcon",
    "LINE": "electrical/line",
    "TRIBUTARY": "electrical/line",
    "REGENIN": "optical/fiber",
    "REGENOUT": "optical/fiber",
}


def _parse_board_name(board_name: str) -> dict[str, str]:
    """Extract board type, ROADM, and category from board name.

    Board format: ROADM{id}-{Type}{idx}$0
    Be careful: 'ROADM' itself contains 'OA', so we parse after the dash.
    """
    m = _BOARD_RE.match(board_name)
    if not m:
        return {"board": board_name, "board_type": "UNKNOWN", "category": "unknown", "roadm": ""}

    roadm = m.group(1)
    raw_type = m.group(2)

    # Normalise to uppercase for matching
    raw_upper = raw_type.upper()

    # Check known prefixes in descending length order to avoid ambiguity
    board_type = "UNKNOWN"
    for prefix in ("REGENIN", "REGENOUT", "TRIBUTARY", "XCON", "LINE", "FIU", "SC2", "OA", "OD", "OM"):
        if raw_upper.startswith(prefix):
            board_type = prefix
            break

    category = BOARD_TYPE_CATEGORY.get(board_type, "unknown")
    return {
        "board": board_name,
        "board_type": board_type,
        "category": category,
        "roadm": roadm,
    }


# ---------------------------------------------------------------------------
# Tool 1: query_topology
# ---------------------------------------------------------------------------

def query_topology(
    board_name: str,
    topology_yaml: str | None = None,
    lightpaths_file: str | None = None,
) -> dict[str, Any]:
    """Query board type, connections, and services for a given board.

    Parameters
    ----------
    board_name : str
        Board ID like ``ROADM0-OA002$0``.
    topology_yaml : str, optional
        Path to topology YAML (currently unused; reserved for future
        expansion with full topology parsing).
    lightpaths_file : str
        Path to lightpaths file (``lightpaths.txt`` or
        ``mw_lightpaths_roles.txt``).

    Returns
    -------
    dict with board metadata, neighbors, services, and path info.
    """
    info = _parse_board_name(board_name)
    result: dict[str, Any] = {
        "board": board_name,
        "board_type": info["board_type"],
        "category": info["category"],
        "roadm": info["roadm"],
        "neighbors": {"upstream": [], "downstream": []},
        "services": [],
        "paths": [],
    }

    if not lightpaths_file or not Path(lightpaths_file).exists():
        result["error"] = f"lightpaths file not found: {lightpaths_file}"
        return result

    paths, role_map = _parse_paths(lightpaths_file)

    upstream_set: set[str] = set()
    downstream_set: set[str] = set()
    services: list[str] = []
    path_entries: list[dict[str, Any]] = []

    for path_idx, path in enumerate(paths):
        boards_in_path = [b for b, _ in path]
        for pos, (b, role) in enumerate(path):
            if b != board_name:
                continue

            service_name = f"S{path_idx + 1}"
            if service_name not in services:
                services.append(service_name)

            # Determine role for this board in this path
            board_role = role or _role_for_board(board_name, role_map)

            path_entries.append({
                "path_index": path_idx,
                "service": service_name,
                "role": board_role,
                "position": pos,
            })

            # Upstream neighbor (previous in path)
            if pos > 0:
                upstream_set.add(boards_in_path[pos - 1])
            # Downstream neighbor (next in path)
            if pos < len(boards_in_path) - 1:
                downstream_set.add(boards_in_path[pos + 1])

    result["neighbors"]["upstream"] = sorted(upstream_set)
    result["neighbors"]["downstream"] = sorted(downstream_set)
    result["services"] = services
    result["paths"] = path_entries

    # Add the functional type used by the rule engine
    board_role = _role_for_board(board_name, role_map)
    result["function_type"] = _board_type(board_name, board_role)
    result["role"] = board_role

    return result


# ---------------------------------------------------------------------------
# Tool 2: get_metrics
# ---------------------------------------------------------------------------

def get_metrics(
    board_name: str,
    data_dir: str,
) -> dict[str, Any]:
    """Get BBE/BBER/ES metrics and pattern features for a board.

    Searches ``data_dir/Node_*/timeseries_*.csv`` for rows matching
    *board_name* (column ``board``) or falling back to the ``vmf_id``
    column if ``board`` is absent.

    Parameters
    ----------
    board_name : str
        Board ID like ``ROADM0-OA002$0``.
    data_dir : str
        Directory containing ``Node_*/timeseries_*.csv`` files.

    Returns
    -------
    dict with metric summary and pattern features.
    """
    result: dict[str, Any] = {"board": board_name}
    data_path = Path(data_dir)

    if not data_path.exists():
        result["error"] = f"data_dir not found: {data_dir}"
        return result

    # Collect all timeseries CSVs
    ts_files = sorted(data_path.glob("Node_*/timeseries_*.csv"))
    if not ts_files:
        # Try patterns with spaces (grpo_runs layout: "Node_1 2/timeseries_1.csv")
        ts_files = sorted(data_path.glob("Node_*/timeseries*.csv"))
    if not ts_files:
        result["error"] = "no timeseries files found"
        return result

    # Collect rows for the target board
    board_rows: list[pd.DataFrame] = []
    all_boards: dict[str, float] = {}

    for f in ts_files:
        df = _read_csv(f)
        if df.empty:
            continue

        # Determine which column holds board names
        if "board" in df.columns:
            col = "board"
        elif "Board" in df.columns:
            col = "Board"
        else:
            # Older format uses vmf_id — skip board-level matching
            continue

        # Track per-board max BBE for context
        if "BBE" in df.columns:
            for bname, bdf in df.groupby(col):
                mx = pd.to_numeric(bdf["BBE"], errors="coerce").max()
                bname_str = str(bname).strip()
                if bname_str not in all_boards or mx > all_boards[bname_str]:
                    all_boards[bname_str] = float(mx)

        # Filter to target board
        mask = df[col].astype(str).str.strip() == board_name
        bdf = df[mask]
        if not bdf.empty:
            board_rows.append(bdf)

    if not board_rows:
        result["error"] = f"no data found for board {board_name}"
        result["available_boards"] = sorted(all_boards.keys())[:20]
        return result

    combined = pd.concat(board_rows, ignore_index=True)

    # Basic metric summaries
    bbe = pd.to_numeric(combined.get("BBE", pd.Series(dtype=float)), errors="coerce").dropna()
    bber = pd.to_numeric(combined.get("BBER", pd.Series(dtype=float)), errors="coerce").dropna()
    es = pd.to_numeric(combined.get("ES", pd.Series(dtype=float)), errors="coerce").dropna()

    result["bbe_max"] = round(float(bbe.max()), 1) if not bbe.empty else 0.0
    result["bber_max"] = round(float(bber.max()), 6) if not bber.empty else 0.0
    result["es_max"] = round(float(es.max()), 4) if not es.empty else 0.0
    result["total_samples"] = len(combined)

    # Elevated sample count (above threshold)
    if not bbe.empty:
        max_bbe = float(bbe.max())
        threshold = max(max_bbe * 0.5, 5.0)
        result["elevated_samples"] = int((bbe.values > threshold).sum())

    # Compute pattern features using the same logic as generate_sft_data.py
    features = _compute_pattern_features(combined)
    for k, v in features.items():
        result[k] = v

    # Include rank among all boards
    if all_boards:
        sorted_boards = sorted(all_boards.items(), key=lambda x: x[1], reverse=True)
        for rank, (bname, mx) in enumerate(sorted_boards):
            if bname == board_name:
                result["bbe_rank"] = rank + 1
                result["bbe_rank_total"] = len(sorted_boards)
                break
        # Top-5 boards for context
        result["top_boards"] = [
            {"board": bname, "bbe_max": round(mx, 1)}
            for bname, mx in sorted_boards[:5]
        ]

    return result


# ---------------------------------------------------------------------------
# Tool 3: lookup_rule
# ---------------------------------------------------------------------------

def lookup_rule(
    board_type: str,
    event_name: str,
    rules_path: str,
) -> list[dict[str, str]]:
    """Return matching propagation rules from rule_database.csv.

    Parameters
    ----------
    board_type : str
        The OTN function type (e.g. ``ODUk-Creation``, ``OTUk-Relay``,
        ``ODUk-Termination``).
    event_name : str
        The alarm/event name (e.g. ``Fiber_Cut``, ``ODUk_PM_AIS``).
    rules_path : str
        Path to the rule_database.csv file.

    Returns
    -------
    list of dicts, one per matching rule.
    """
    path = Path(rules_path)
    if not path.exists():
        return [{"error": f"rules file not found: {rules_path}"}]

    df = load_rule_database(path)
    rules = _compile_rules(df)

    matches: list[dict[str, str]] = []
    for rule in rules:
        # Match by function type and event name
        type_match = (
            rule.from_type == board_type
            or board_type == ""  # wildcard: return all rules for this event
        )
        event_match = rule.from_alarm == event_name

        if type_match and event_match:
            matches.append({
                "rule_idx": rule.idx,
                "from_type": rule.from_type,
                "from_alarm": rule.from_alarm,
                "action": rule.action,
                "to_type": rule.to_type,
                "to_alarm": rule.to_alarm,
            })

    if not matches:
        # Try a looser search: event_name anywhere in from_alarm
        for rule in rules:
            if event_name.lower() in rule.from_alarm.lower():
                matches.append({
                    "rule_idx": rule.idx,
                    "from_type": rule.from_type,
                    "from_alarm": rule.from_alarm,
                    "action": rule.action,
                    "to_type": rule.to_type,
                    "to_alarm": rule.to_alarm,
                    "note": "fuzzy match",
                })

    return matches


# ---------------------------------------------------------------------------
# Tool 4: validate_propagation
# ---------------------------------------------------------------------------

def validate_propagation(
    root_board: str,
    root_event: str,
    rules_path: str,
    lightpaths_file: str,
    severity: str = "Critical",
    time: str = "2025-01-01 00:00:00.000000",
    duration_ms: int = 15000,
) -> dict[str, Any]:
    """Run alarm_generator forward from a hypothesis and return predicted alarm flow.

    Creates a temporary failure.csv, calls ``generate_alarm_flow``, and
    parses the output.

    Parameters
    ----------
    root_board : str
        Hypothesised root cause board (e.g. ``ROADM0-OA002$0``).
    root_event : str
        Hypothesised root event (e.g. ``Fiber_Cut``).
    rules_path : str
        Path to rule_database.csv.
    lightpaths_file : str
        Path to lightpaths file.
    severity : str
        Severity for the seed failure (default ``Critical``).
    time : str
        Timestamp for the seed failure.
    duration_ms : int
        Duration of the seed failure in milliseconds.

    Returns
    -------
    dict with predicted boards, alarms, severity distribution, and raw CSV.
    """
    result: dict[str, Any] = {
        "root_board": root_board,
        "root_event": root_event,
    }

    rules_p = Path(rules_path)
    lp_p = Path(lightpaths_file)

    if not rules_p.exists():
        result["error"] = f"rules file not found: {rules_path}"
        return result
    if not lp_p.exists():
        result["error"] = f"lightpaths file not found: {lightpaths_file}"
        return result

    df_rules = load_rule_database(rules_p)

    # Create temporary failure.csv and output CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        failure_csv = Path(tmpdir) / "failure.csv"
        out_csv = Path(tmpdir) / "alarm_flow.csv"

        # Determine the board role from lightpaths
        paths, role_map = _parse_paths(lightpaths_file)
        board_role = _role_for_board(root_board, role_map)

        failure_csv.write_text(
            "Board,Event,Time,Severity,DurationMs,BoardRole\n"
            f"{root_board},{root_event},{time},{severity},{duration_ms},{board_role}\n",
            encoding="utf-8",
        )

        try:
            generate_alarm_flow(
                graphml_path=str(lp_p),  # not used when paths_file is provided
                df_rules=df_rules,
                out_csv=str(out_csv),
                failures_csv=str(failure_csv),
                paths_file=str(lp_p),
                max_hops=4,
            )
        except Exception as e:
            result["error"] = f"alarm propagation failed: {e}"
            return result

        # Parse the output
        if not out_csv.exists() or out_csv.stat().st_size == 0:
            result["predicted_boards"] = []
            result["predicted_alarms"] = {}
            result["predicted_severity_dist"] = {}
            result["total_rows"] = 0
            result["max_hop"] = 0
            result["alarm_flow_csv"] = ""
            return result

        df_out = pd.read_csv(out_csv)

    # Extract summary statistics
    predicted_boards = sorted(df_out["PropToBoard"].unique().tolist()) if "PropToBoard" in df_out.columns else []
    predicted_alarms = df_out["PropAlarm"].value_counts().to_dict() if "PropAlarm" in df_out.columns else {}
    severity_dist = df_out["Severity"].value_counts().to_dict() if "Severity" in df_out.columns else {}
    max_hop = int(df_out["Hop"].max()) if "Hop" in df_out.columns and not df_out["Hop"].isna().all() else 0

    result["predicted_boards"] = predicted_boards
    result["predicted_alarms"] = predicted_alarms
    result["predicted_severity_dist"] = severity_dist
    result["total_rows"] = len(df_out)
    result["max_hop"] = max_hop
    result["alarm_flow_csv"] = df_out.to_csv(index=False)

    return result


# ---------------------------------------------------------------------------
# Tool 5: search_similar
# ---------------------------------------------------------------------------

def search_similar(
    pattern_kind: str,
    board_type: str,
    bbe_range: tuple[float, float],
    training_data_path: str,
) -> list[dict[str, Any]]:
    """Search training examples for similar past cases.

    Loads the training JSONL and filters by matching criteria. Returns
    the top-3 most similar examples ranked by a simple similarity score
    based on pattern_kind match, board_type match, and bbe_max proximity.

    Parameters
    ----------
    pattern_kind : str
        Temporal pattern kind (``step``, ``ramp``, ``burst``,
        ``step_recovery``).
    board_type : str
        Board type category (e.g. ``OA``, ``XCON``, ``Line``).
    bbe_range : tuple[float, float]
        (min_bbe, max_bbe) range to filter on.
    training_data_path : str
        Path to the training JSONL file.

    Returns
    -------
    list of dicts with matching examples, ranked by similarity.
    """
    path = Path(training_data_path)
    if not path.exists():
        return [{"error": f"training data not found: {training_data_path}"}]

    bbe_min, bbe_max = bbe_range
    candidates: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            gt_board = record.get("ground_truth_root_board", "")
            gt_event = record.get("ground_truth_root_event", "")

            # Extract board type from the ground truth board
            gt_info = _parse_board_name(gt_board)
            gt_board_type = gt_info["board_type"]

            # Extract pattern_kind and bbe_max from the prompt text
            prompt_text = ""
            prompt_msgs = record.get("prompt", [])
            if isinstance(prompt_msgs, list):
                for msg in prompt_msgs:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        prompt_text = msg.get("content", "")
                        break
            elif isinstance(prompt_msgs, str):
                prompt_text = prompt_msgs

            # Parse pattern features from prompt
            pk_match = re.search(r"pattern_kind=(\w+)", prompt_text)
            bbe_match = re.search(r"bbe_max=([\d.]+)", prompt_text)

            gt_pattern_kind = pk_match.group(1) if pk_match else ""
            gt_bbe_max = float(bbe_match.group(1)) if bbe_match else 0.0

            # Filter by bbe_range
            if gt_bbe_max < bbe_min or gt_bbe_max > bbe_max:
                continue

            # Compute similarity score
            score = 0.0

            # Pattern kind match (0.4 weight)
            if gt_pattern_kind == pattern_kind:
                score += 0.4

            # Board type match (0.3 weight)
            if gt_board_type == board_type:
                score += 0.3

            # BBE proximity (0.3 weight) — closer = higher score
            bbe_center = (bbe_min + bbe_max) / 2.0
            bbe_span = max(bbe_max - bbe_min, 1.0)
            bbe_dist = abs(gt_bbe_max - bbe_center) / bbe_span
            score += 0.3 * max(0.0, 1.0 - bbe_dist)

            candidates.append({
                "root_event": gt_event,
                "root_board": gt_board,
                "board_type": gt_board_type,
                "bbe_max": gt_bbe_max,
                "pattern_kind": gt_pattern_kind,
                "similarity_score": round(score, 3),
                "line_num": line_num,
            })

    # Sort by similarity (descending) and return top 3
    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    top = candidates[:3]

    # Remove internal fields
    for c in top:
        c.pop("line_num", None)

    return top


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "query_topology": {
        "function": query_topology,
        "description": (
            "Query board type, connections, and services for a given board. "
            "Returns board type, category (optical/fiber, electrical/xcon, electrical/line), "
            "upstream/downstream neighbors, services passing through, and path positions."
        ),
        "parameters": {
            "board_name": "str - board ID like ROADM0-OA002$0",
            "topology_yaml": "str (optional) - path to topology YAML",
            "lightpaths_file": "str - path to lightpaths.txt or mw_lightpaths_roles.txt",
        },
    },
    "get_metrics": {
        "function": get_metrics,
        "description": (
            "Get BBE/BBER/ES metrics and pattern features for a specific board. "
            "Returns bbe_max, bber_max, es_max, pattern_kind (step/ramp/burst/step_recovery), "
            "burst_regions, longest_burst_frac, es_frac, bber_bbe_ratio, and ranking among all boards."
        ),
        "parameters": {
            "board_name": "str - board ID like ROADM0-OA002$0",
            "data_dir": "str - directory containing Node_*/timeseries_*.csv files",
        },
    },
    "lookup_rule": {
        "function": lookup_rule,
        "description": (
            "Look up propagation rules from the rule database for a given function type and event. "
            "Returns matching rules with action type, target type, and output alarm."
        ),
        "parameters": {
            "board_type": "str - OTN function type (ODUk-Creation, OTUk-Relay, ODUk-Termination)",
            "event_name": "str - alarm/event name (e.g. Fiber_Cut, ODUk_PM_AIS)",
            "rules_path": "str - path to rule_database.csv",
        },
    },
    "validate_propagation": {
        "function": validate_propagation,
        "description": (
            "Run alarm propagation forward from a hypothesised root cause. "
            "Creates a temporary failure, runs the alarm_generator engine, and returns "
            "the predicted alarm flow including affected boards, alarm distribution, "
            "severity distribution, and full CSV output."
        ),
        "parameters": {
            "root_board": "str - hypothesised root cause board (e.g. ROADM0-OA002$0)",
            "root_event": "str - hypothesised root event (e.g. Fiber_Cut)",
            "rules_path": "str - path to rule_database.csv",
            "lightpaths_file": "str - path to lightpaths file",
            "severity": "str (optional) - severity for seed failure (default Critical)",
        },
    },
    "search_similar": {
        "function": search_similar,
        "description": (
            "Search training examples for similar past cases by pattern kind, board type, "
            "and BBE range. Returns top-3 most similar examples with similarity scores."
        ),
        "parameters": {
            "pattern_kind": "str - temporal pattern (step/ramp/burst/step_recovery)",
            "board_type": "str - board type (OA, XCON, Line, etc.)",
            "bbe_range": "tuple[float, float] - (min_bbe, max_bbe) range to search",
            "training_data_path": "str - path to training JSONL file",
        },
    },
}


# ---------------------------------------------------------------------------
# Prompt helper
# ---------------------------------------------------------------------------

def format_tools_for_prompt() -> str:
    """Generate a text description of available tools for the LLM system prompt.

    Returns a multi-line string describing each tool, its purpose, and
    its parameters, suitable for injection into a ReAct-style system
    prompt.
    """
    lines = [
        "You have access to the following tools. To use a tool, respond with a JSON",
        "action block in the format:",
        "",
        '  {"tool": "<tool_name>", "args": {<arguments>}}',
        "",
        "Available tools:",
        "",
    ]

    for name, meta in TOOL_REGISTRY.items():
        lines.append(f"### {name}")
        lines.append(f"  {meta['description']}")
        lines.append("  Parameters:")
        for pname, pdesc in meta["parameters"].items():
            lines.append(f"    - {pname}: {pdesc}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test RCA tools.")
    ap.add_argument("--board", default="ROADM0-OA002$0", help="Board to query")
    ap.add_argument("--lightpaths", default=None, help="Path to lightpaths file")
    ap.add_argument("--data-dir", default=None, help="Path to timeseries data directory")
    ap.add_argument("--rules", default="outputs/Static files/rule_database.csv",
                    help="Path to rule_database.csv")
    ap.add_argument("--training-data", default="outputs/grpo_data/train.jsonl",
                    help="Path to training JSONL")
    ap.add_argument("--tool", choices=list(TOOL_REGISTRY.keys()) + ["all", "prompt"],
                    default="prompt", help="Tool to test")
    args = ap.parse_args()

    if args.tool == "prompt":
        print(format_tools_for_prompt())

    elif args.tool in ("query_topology", "all"):
        if args.lightpaths:
            result = query_topology(args.board, lightpaths_file=args.lightpaths)
            print(json.dumps(result, indent=2))
        else:
            print("[skip] query_topology: --lightpaths not provided")

    elif args.tool in ("get_metrics",) or (args.tool == "all" and args.data_dir):
        if args.data_dir:
            result = get_metrics(args.board, args.data_dir)
            print(json.dumps(result, indent=2))
        else:
            print("[skip] get_metrics: --data-dir not provided")

    elif args.tool in ("lookup_rule",) or args.tool == "all":
        result = lookup_rule("OTUk-Relay", "Fiber_Cut", args.rules)
        print(json.dumps(result, indent=2))

    elif args.tool in ("validate_propagation",) or args.tool == "all":
        if args.lightpaths:
            result = validate_propagation(
                args.board, "Fiber_Cut", args.rules, args.lightpaths,
            )
            # Truncate alarm_flow_csv for display
            if "alarm_flow_csv" in result:
                csv_lines = result["alarm_flow_csv"].splitlines()
                if len(csv_lines) > 5:
                    result["alarm_flow_csv"] = "\n".join(csv_lines[:5]) + f"\n... ({len(csv_lines)-5} more rows)"
            print(json.dumps(result, indent=2))
        else:
            print("[skip] validate_propagation: --lightpaths not provided")

    elif args.tool in ("search_similar",) or args.tool == "all":
        result = search_similar("step", "OA", (40.0, 60.0), args.training_data)
        print(json.dumps(result, indent=2))

    if args.tool == "all":
        print("\n--- Tool prompt ---")
        print(format_tools_for_prompt())
