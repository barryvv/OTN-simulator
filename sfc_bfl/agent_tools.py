#!/usr/bin/env python3
"""SFC-BFL-native agent tools for ReAct agent.

Adapted from rca_tools.py to work with in-memory SFC-BFL data
(parquet DataFrames, JSON lightpaths) instead of file paths.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate_sft_data import _compute_pattern_features

# Import alarm propagation engine
from alarm_generator import (
    _board_kind,
    _board_type,
    _compile_rules,
    _role_for_board,
    generate_alarm_flow,
    load_rule_database,
)

# ---------------------------------------------------------------------------
# Board name parsing (same as rca_tools.py)
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
    """Extract board type, ROADM, and category from board name."""
    m = _BOARD_RE.match(board_name)
    if not m:
        return {"board": board_name, "board_type": "UNKNOWN", "category": "unknown", "roadm": ""}
    roadm = m.group(1)
    raw_upper = m.group(2).upper()
    board_type = "UNKNOWN"
    for prefix in ("REGENIN", "REGENOUT", "TRIBUTARY", "XCON", "LINE", "FIU", "SC2", "OA", "OD", "OM"):
        if raw_upper.startswith(prefix):
            board_type = prefix
            break
    category = BOARD_TYPE_CATEGORY.get(board_type, "unknown")
    return {"board": board_name, "board_type": board_type, "category": category, "roadm": roadm}


# ---------------------------------------------------------------------------
# Tool 1: query_topology_sfc
# ---------------------------------------------------------------------------

def query_topology_sfc(
    board_name: str,
    boards_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    lightpaths: list[dict],
    services: list[dict],
) -> dict[str, Any]:
    """Query board type, neighbors, and services from in-memory SFC-BFL data.

    Returns board metadata, upstream/downstream neighbors, services, and paths.
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

    # Look up in boards_df
    board_row = boards_df[boards_df["board_id"] == board_name]
    if board_row.empty:
        # Try without leading zeros in index
        board_row = boards_df[boards_df["board_id"].str.replace(r"0+(\d)", r"\1", regex=True) ==
                              board_name.replace("$0", "").rstrip("0123456789") + board_name.split("$")[0].split("-")[1].lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") + "$0"]

    if not board_row.empty:
        result["node_id"] = str(board_row.iloc[0].get("node_id", ""))
        result["layer"] = str(board_row.iloc[0].get("layer", ""))
        result["domain_id"] = int(board_row.iloc[0].get("domain_id", 0))

    # Neighbors from edges
    upstream = set()
    downstream = set()
    for _, row in edges_df.iterrows():
        if str(row["dst_board_id"]) == board_name:
            upstream.add(str(row["src_board_id"]))
        if str(row["src_board_id"]) == board_name:
            downstream.add(str(row["dst_board_id"]))
    result["neighbors"]["upstream"] = sorted(upstream)
    result["neighbors"]["downstream"] = sorted(downstream)

    # Services from lightpaths
    svc_list = []
    path_entries = []
    for lp in lightpaths:
        bp = lp.get("board_path", [])
        if board_name in bp:
            pos = bp.index(board_name)
            svc_name = f"S{lp.get('lightpath_id', 0) + 1}"
            if svc_name not in svc_list:
                svc_list.append(svc_name)
            path_entries.append({
                "lightpath_id": lp.get("lightpath_id", 0),
                "service": svc_name,
                "position": pos,
                "total_boards": len(bp),
            })

    result["services"] = svc_list
    result["paths"] = path_entries

    return result


# ---------------------------------------------------------------------------
# Tool 2: get_metrics_sfc
# ---------------------------------------------------------------------------

def get_metrics_sfc(
    board_name: str,
    ow_df: pd.DataFrame,
) -> dict[str, Any]:
    """Get BBE/BBER/ES metrics and pattern features from parquet DataFrame.

    Replaces get_metrics() which reads CSV files from disk.
    """
    result: dict[str, Any] = {"board": board_name}

    if ow_df.empty:
        result["error"] = "no metrics data"
        return result

    # Filter to target board
    board_df = ow_df[ow_df["board_id"].astype(str).str.strip() == board_name]

    # Also track all boards for ranking
    all_boards: dict[str, float] = {}
    if "BBE" in ow_df.columns:
        for bid, bdf in ow_df.groupby("board_id"):
            mx = pd.to_numeric(bdf["BBE"], errors="coerce").max()
            all_boards[str(bid)] = float(mx)

    if board_df.empty:
        result["error"] = f"no data for board {board_name}"
        result["available_boards"] = sorted(all_boards.keys())[:20]
        return result

    # Basic summaries
    bbe = pd.to_numeric(board_df.get("BBE", pd.Series(dtype=float)), errors="coerce").dropna()
    bber = pd.to_numeric(board_df.get("BBER", pd.Series(dtype=float)), errors="coerce").dropna()
    es = pd.to_numeric(board_df.get("ES", pd.Series(dtype=float)), errors="coerce").dropna()

    result["bbe_max"] = round(float(bbe.max()), 1) if not bbe.empty else 0.0
    result["bber_max"] = round(float(bber.max()), 6) if not bber.empty else 0.0
    result["es_max"] = round(float(es.max()), 4) if not es.empty else 0.0
    result["total_samples"] = len(board_df)

    if not bbe.empty:
        max_bbe = float(bbe.max())
        threshold = max(max_bbe * 0.5, 5.0)
        result["elevated_samples"] = int((bbe.values > threshold).sum())

    # Pattern features
    features = _compute_pattern_features(board_df)
    for k, v in features.items():
        result[k] = v

    # Board ranking
    if all_boards:
        sorted_boards = sorted(all_boards.items(), key=lambda x: x[1], reverse=True)
        for rank, (bname, mx) in enumerate(sorted_boards):
            if bname == board_name:
                result["bbe_rank"] = rank + 1
                result["bbe_rank_total"] = len(sorted_boards)
                break
        result["top_boards"] = [
            {"board": bname, "bbe_max": round(mx, 1)}
            for bname, mx in sorted_boards[:5]
        ]

    return result


# ---------------------------------------------------------------------------
# Tool 3: get_top_boards_sfc
# ---------------------------------------------------------------------------

def get_top_boards_sfc(
    ow_df: pd.DataFrame,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Get top-N anomalous boards by max BBE.

    Enables agent exploration without knowing board names a priori.
    """
    if ow_df.empty or "BBE" not in ow_df.columns:
        return [{"error": "no metrics data"}]

    board_max: dict[str, float] = {}
    for bid, bdf in ow_df.groupby("board_id"):
        mx = pd.to_numeric(bdf["BBE"], errors="coerce").max()
        board_max[str(bid)] = float(mx)

    sorted_boards = sorted(board_max.items(), key=lambda x: x[1], reverse=True)

    result = []
    for bname, mx in sorted_boards[:top_n]:
        info = _parse_board_name(bname)
        result.append({
            "board": bname,
            "bbe_max": round(mx, 1),
            "board_type": info["board_type"],
            "category": info["category"],
            "roadm": info["roadm"],
        })
    return result


# ---------------------------------------------------------------------------
# Tool 4: trace_service_path
# ---------------------------------------------------------------------------

def trace_service_path(
    service_id: str,
    lightpaths: list[dict],
    boards_df: pd.DataFrame,
) -> dict[str, Any]:
    """Trace the full board path for a service/lightpath.

    Returns board path with types, node crossings, and domain info.
    """
    # Map service_id to lightpath_id (S1 -> lightpath_id=0)
    lp_idx = None
    if service_id.startswith("S"):
        try:
            lp_idx = int(service_id[1:]) - 1
        except ValueError:
            pass

    if lp_idx is None or lp_idx < 0 or lp_idx >= len(lightpaths):
        return {"error": f"service {service_id} not found (have {len(lightpaths)} lightpaths)"}

    lp = lightpaths[lp_idx]
    bp = lp.get("board_path", [])
    np_ = lp.get("node_path", [])

    # Build board details
    board_details = []
    # Build a lookup for board info
    board_info_map = {}
    for _, row in boards_df.iterrows():
        board_info_map[str(row["board_id"])] = {
            "node_id": str(row.get("node_id", "")),
            "board_type": str(row.get("board_type", "")),
            "layer": str(row.get("layer", "")),
            "domain_id": int(row.get("domain_id", 0)),
        }

    domains_crossed = set()
    for i, bname in enumerate(bp):
        info = board_info_map.get(bname, {})
        parsed = _parse_board_name(bname)
        domain_id = info.get("domain_id", 0)
        domains_crossed.add(domain_id)
        board_details.append({
            "position": i,
            "board": bname,
            "board_type": parsed["board_type"],
            "category": parsed["category"],
            "node_id": info.get("node_id", parsed["roadm"]),
            "domain_id": domain_id,
        })

    return {
        "service_id": service_id,
        "lightpath_id": lp_idx,
        "node_path": np_,
        "board_path": bp,
        "board_details": board_details,
        "total_boards": len(bp),
        "total_nodes": len(np_),
        "domains_crossed": sorted(domains_crossed),
        "is_multi_domain": len(domains_crossed) > 1,
    }


# ---------------------------------------------------------------------------
# Tool 5: lookup_rule_sfc (same as rca_tools.lookup_rule)
# ---------------------------------------------------------------------------

def lookup_rule_sfc(
    board_type: str,
    event_name: str,
    rules_path: str,
) -> list[dict[str, str]]:
    """Return matching propagation rules from rule_database.csv.

    Same implementation as rca_tools.lookup_rule — topology-independent.
    """
    path = Path(rules_path)
    if not path.exists():
        return [{"error": f"rules file not found: {rules_path}"}]

    df = load_rule_database(path)
    rules = _compile_rules(df)

    matches: list[dict[str, str]] = []
    for rule in rules:
        type_match = rule.from_type == board_type or board_type == ""
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
# Tool 6: validate_propagation_sfc
# ---------------------------------------------------------------------------

def validate_propagation_sfc(
    root_board: str,
    root_event: str,
    lightpaths: list[dict],
    rules_path: str,
    severity: str = "Critical",
    time: str = "2025-01-01 00:00:00.000000",
    duration_ms: int = 15000,
) -> dict[str, Any]:
    """Run alarm propagation from a hypothesis using in-memory lightpaths.

    Writes a temporary lightpaths file for the alarm_generator engine.
    """
    result: dict[str, Any] = {
        "root_board": root_board,
        "root_event": root_event,
    }

    rules_p = Path(rules_path)
    if not rules_p.exists():
        result["error"] = f"rules file not found: {rules_path}"
        return result

    df_rules = load_rule_database(rules_p)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)

        # Write lightpaths to temporary file in the format expected by alarm_generator
        paths_file = tmpdir_p / "lightpaths.txt"
        _write_lightpaths_txt(lightpaths, str(paths_file))

        # Write failure CSV
        failure_csv = tmpdir_p / "failure.csv"
        # Determine board role from lightpath position
        board_role = _infer_board_role(root_board, lightpaths)
        failure_csv.write_text(
            "Board,Event,Time,Severity,DurationMs,BoardRole\n"
            f"{root_board},{root_event},{time},{severity},{duration_ms},{board_role}\n",
            encoding="utf-8",
        )

        out_csv = tmpdir_p / "alarm_flow.csv"
        try:
            generate_alarm_flow(
                graphml_path=str(paths_file),
                df_rules=df_rules,
                out_csv=str(out_csv),
                failures_csv=str(failure_csv),
                paths_file=str(paths_file),
                max_hops=4,
            )
        except Exception as e:
            result["error"] = f"alarm propagation failed: {e}"
            return result

        if not out_csv.exists() or out_csv.stat().st_size == 0:
            result["predicted_boards"] = []
            result["predicted_alarms"] = {}
            result["predicted_severity_dist"] = {}
            result["total_rows"] = 0
            result["max_hop"] = 0
            result["alarm_flow_csv"] = ""
            return result

        df_out = pd.read_csv(out_csv)

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


def _write_lightpaths_txt(lightpaths: list[dict], out_path: str) -> None:
    """Write lightpaths.json to lightpaths.txt format for alarm_generator."""
    lines = []
    for lp in lightpaths:
        bp = lp.get("board_path", [])
        if bp:
            lines.append(" -> ".join(bp))
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def _infer_board_role(board_name: str, lightpaths: list[dict]) -> str:
    """Infer board role (SRC/RELAY/SNK) from lightpath position."""
    for lp in lightpaths:
        bp = lp.get("board_path", [])
        if board_name in bp:
            pos = bp.index(board_name)
            if pos == 0:
                return "SRC"
            elif pos == len(bp) - 1:
                return "SNK"
            else:
                return "RELAY"
    return "RELAY"


# ---------------------------------------------------------------------------
# Tool Registry (for SFC-BFL agent)
# ---------------------------------------------------------------------------

SFC_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "get_top_boards": {
        "function": get_top_boards_sfc,
        "description": (
            "Get top-N anomalous boards ranked by max BBE. "
            "Use this first to identify which boards have elevated metrics."
        ),
        "parameters": {
            "top_n": "int (optional, default 10) - number of top boards to return",
        },
    },
    "get_metrics": {
        "function": get_metrics_sfc,
        "description": (
            "Get BBE/BBER/ES metrics and pattern features for a specific board. "
            "Returns bbe_max, pattern_kind, burst_regions, etc."
        ),
        "parameters": {
            "board_name": "str - board ID like ROADM0-OA002$0",
        },
    },
    "query_topology": {
        "function": query_topology_sfc,
        "description": (
            "Query board type, neighbors, and services for a given board. "
            "Returns board type, category, upstream/downstream neighbors, and services."
        ),
        "parameters": {
            "board_name": "str - board ID like ROADM0-OA002$0",
        },
    },
    "trace_service_path": {
        "function": trace_service_path,
        "description": (
            "Trace the full board path for a service/lightpath. "
            "Shows all boards, types, and domain crossings."
        ),
        "parameters": {
            "service_id": "str - service ID like S1, S2, etc.",
        },
    },
    "lookup_rule": {
        "function": lookup_rule_sfc,
        "description": (
            "Look up propagation rules from the rule database. "
            "Returns matching rules with action type, target, and output alarm."
        ),
        "parameters": {
            "board_type": "str - OTN function type (e.g. ODUk-Creation, OTUk-Relay)",
            "event_name": "str - alarm/event name (e.g. Fiber_Cut, ODUk_PM_AIS)",
        },
    },
    "validate_propagation": {
        "function": validate_propagation_sfc,
        "description": (
            "Run alarm propagation forward from a hypothesised root cause. "
            "Returns predicted alarm flow with affected boards and alarms."
        ),
        "parameters": {
            "root_board": "str - hypothesised root cause board",
            "root_event": "str - hypothesised root event (e.g. Fiber_Cut)",
        },
    },
}


def format_sfc_tools_for_prompt() -> str:
    """Generate tool description text for LLM system prompt."""
    lines = [
        "You have access to the following tools. To use a tool, respond with:",
        "",
        '  ACTION: tool_name(param1="value1", param2="value2")',
        "",
        "Available tools:",
        "",
    ]
    for name, meta in SFC_TOOL_REGISTRY.items():
        lines.append(f"### {name}")
        lines.append(f"  {meta['description']}")
        lines.append("  Parameters:")
        for pname, pdesc in meta["parameters"].items():
            lines.append(f"    - {pname}: {pdesc}")
        lines.append("")
    return "\n".join(lines)
