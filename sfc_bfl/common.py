#!/usr/bin/env python3
"""Shared infrastructure for SFC-BFL experiment dataset generation.

Reuses existing simulator components:
- topology_traffic_generator.generate_topology()
- topology_traffic_processor.process_traffic_from_nodeDG()
- convert_graphml_to_topology  (YAML building)
- alarm_generator  (rule-based alarm propagation)
- otn_simulator  (metrics generation, failure injection)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Add project root to path so we can import existing modules
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("ARANGO_SKIP", "1")

from topology_traffic_generator import generate_topology  # noqa: E402
from topology_traffic_processor import process_traffic_from_nodeDG  # noqa: E402
from convert_graphml_to_topology import (  # noqa: E402
    compress_parent_path,
    canonical_seg_id,
    expand_service_to_vmf,
)
from alarm_generator import load_rule_database, generate_alarm_flow  # noqa: E402
from otn_simulator import (  # noqa: E402
    load_yaml,
    safe_mkdir,
    build_graph_and_service_candidates,
    simulate_latent_series,
    latent_to_metrics,
    inject_failures_to_bbe,
    write_paths_from_topology,
    _parse_step_seconds,
    _parse_board_name,
    _role_for_kind,
)
from class_obj import ELECTRICAL_GROUPS  # noqa: E402

import networkx as nx  # noqa: E402

# ---------------------------------------------------------------------------
# Failure event types
# ---------------------------------------------------------------------------
FIBER_EVENTS = ["Fiber_Cut", "Fiber_Crack", "Fiber_Aging"]
XCON_EVENTS = ["XCON_Port_Down", "XCON_Buffer_Overflow", "XCON_Fabric_Fault"]
LINE_EVENTS = ["Line_Disconnect"]
ALL_EVENTS = FIBER_EVENTS + XCON_EVENTS + LINE_EVENTS

# Board kind → event pool
_EVENTS_FOR_KIND = {
    "OA": FIBER_EVENTS, "OM": FIBER_EVENTS, "OD": FIBER_EVENTS,
    "FIU": FIBER_EVENTS,
    "XCON": XCON_EVENTS, "TRIBUTARY": XCON_EVENTS,
    "LINE": LINE_EVENTS,
}

SES_THRESHOLD = 0.5  # ES > threshold => SES=1 (ITU-T simplification)


def event_for_board(board_name: str, board_map: dict, rng) -> str:
    """Determine a valid event type for a given board based on its kind.

    Looks up the board's parent node and kind from board_map, then picks
    a random event from the matching event pool.
    """
    parent = board_name.split("-")[0] if "-" in board_name else board_name
    if parent in board_map:
        for kind, boards in board_map[parent].items():
            if board_name in boards:
                k = kind.upper()
                events = _EVENTS_FOR_KIND.get(k, ALL_EVENTS)
                return events[int(rng.integers(0, len(events)))]
    # Fallback: detect kind from board name
    name_upper = board_name.upper()
    for prefix in ("TRIBUTARY", "XCON", "LINE", "OA", "OD", "OM", "FIU"):
        if prefix in name_upper:
            events = _EVENTS_FOR_KIND.get(prefix, ALL_EVENTS)
            return events[int(rng.integers(0, len(events)))]
    return ALL_EVENTS[int(rng.integers(0, len(ALL_EVENTS)))]


# ============================================================================
# Topology Generation
# ============================================================================

def generate_full_topology(
    num_roadms: int = 5,
    electrical_groups: int | None = None,
    num_lightpaths: int = 15,
    seed: int = 42,
    regen_hops: int = 0,
) -> dict:
    """Generate a complete topology (node graph, board graph, lightpaths, YAML).

    Uses the same pipeline as run.py:
    1. generate_topology() → node_DG
    2. process_traffic_from_nodeDG() → board_DG
    3. _ensure_el_nodes_and_edges() → fix electrical chains
    4. synthesize_lightpaths() → lightpaths from board_DG
    5. Build topology YAML + roles

    Returns dict with keys:
        node_DG, board_DG, lightpath_list, topo, roles, board_list
    """
    eg = electrical_groups if electrical_groups is not None else ELECTRICAL_GROUPS
    os.environ["ELECTRICAL_GROUPS"] = str(eg)
    os.environ["REGEN_HOPS"] = str(regen_hops)

    # Import here to pick up the overridden ELECTRICAL_GROUPS
    import importlib
    import class_obj
    importlib.reload(class_obj)
    import topology_traffic_processor as ttp
    importlib.reload(ttp)

    from topology_traffic_generator import generate_topology as gen_topo
    from topology_traffic_processor import process_traffic_from_nodeDG as proc

    # Use run.py's synthesis approach
    from run import (
        synthesize_lightpaths,
        _ensure_el_nodes_and_edges,
        _annotate_roles_for_path,
        _insert_regen_nodes,
    )

    rng_py = __import__("random")
    rng_py.seed(seed)

    node_DG = gen_topo(num_roadms, ROADM_ratio=1.0, max_degree=2)
    node_list, nc_list, board_DG = proc(node_DG)

    # Ensure electrical chains exist (same as run.py)
    _ensure_el_nodes_and_edges(board_DG)

    # Synthesize lightpaths from board graph (same as run.py)
    # LP_MIN/MAX_PER_NODE controls how many Line boards per node get lightpaths.
    # Set high enough to cover all electrical groups for good board coverage.
    os.environ["E_SEED"] = str(seed)
    lp_need = eg * 4  # total Line boards per node
    os.environ["LP_MIN_PER_NODE"] = str(max(4, lp_need))
    os.environ["LP_MAX_PER_NODE"] = str(max(10, lp_need))
    raw_paths = synthesize_lightpaths(board_DG, limit_per_node=lp_need)

    # Apply regen and role annotations
    annotated_paths = []
    for idx, p in enumerate(raw_paths):
        if regen_hops:
            p = _insert_regen_nodes(board_DG, p, regen_hops, idx)
        p_annot = _annotate_roles_for_path(board_DG, p)
        annotated_paths.append(p_annot)

    # Write lightpaths to temp file for convert_graphml_to_topology compatibility
    lightpath_list = raw_paths  # Use raw paths (board name lists)

    # Build topology YAML structure
    topo, roles = _build_topo_and_roles(
        board_DG, lightpath_list, node_DG, regen_hops
    )

    # Collect all boards
    board_list = list(board_DG.nodes())

    # Store annotated paths for alarm flow generation
    topo["_annotated_paths"] = annotated_paths

    return {
        "node_DG": node_DG,
        "board_DG": board_DG,
        "lightpath_list": lightpath_list,
        "topo": topo,
        "roles": roles,
        "board_list": board_list,
        "node_list": node_list,
        "nc_list": nc_list,
    }


def _build_topo_and_roles(
    board_DG: nx.DiGraph,
    lightpath_list,
    node_DG: nx.DiGraph,
    regen_hops: int = 0,
) -> tuple[dict, list[dict]]:
    """Build topology YAML dict and roles list from board graph + lightpaths."""

    # Build node_map: parent_name -> node_id
    parents = set()
    for n in board_DG.nodes():
        p = n.split("-")[0] if "-" in n else n
        parents.add(p)
    parent_list = sorted(parents)
    node_map = {p: i + 1 for i, p in enumerate(parent_list)}

    # Build board_map: parent -> {kind: [board_names]}
    board_map: dict[str, dict[str, list[str]]] = {}
    for n, data in board_DG.nodes(data=True):
        parent = n.split("-")[0] if "-" in n else n
        ttype = str(data.get("tType", ""))
        if not ttype:
            # Infer from name
            for prefix in ("Tributary", "XCON", "Line", "OA", "OD", "OM", "FIU"):
                if prefix in n:
                    ttype = prefix
                    break
        if not ttype:
            continue
        kind = ttype.upper()
        # Normalize kind
        for k in ("TRIBUTARY", "XCON", "LINE", "OA", "OD", "OM", "FIU"):
            if kind.startswith(k):
                kind = k
                break
        board_map.setdefault(parent, {}).setdefault(kind, []).append(n)

    # Sort board lists
    for parent in board_map:
        for kind in board_map[parent]:
            board_map[parent][kind] = sorted(board_map[parent][kind])

    # Build segments from node graph edges
    segments = []
    segments_index = {}
    seen_segs = set()
    for u, v in node_DG.edges():
        u_name = list(node_DG.nodes())[u] if isinstance(u, int) else u
        v_name = list(node_DG.nodes())[v] if isinstance(v, int) else v
        u_id = node_map.get(u_name, u_name)
        v_id = node_map.get(v_name, v_name)
        seg_id = canonical_seg_id(int(u_id), int(v_id))
        if seg_id not in seen_segs:
            seen_segs.add(seg_id)
            a, b = seg_id.split("-")
            segments.append({"segment_id": seg_id, "a_end": int(a), "z_end": int(b)})
            segments_index[seg_id] = [int(a), int(b)]

    id_rules = {
        "src_fmt": "{node}_SRC_{service}",
        "relay_fmt": "{node}_RELAY_{segment}_{dir}",
        "snk_fmt": "{node}_SNK_{service}",
    }

    # Build services from lightpaths
    services = []
    all_roles = []
    for i, lp in enumerate(lightpath_list):
        sid = f"S{i + 1}"
        board_path = lp.boardPath if hasattr(lp, "boardPath") else lp
        node_path_raw = compress_parent_path(board_path)
        node_path = [int(node_map.get(p, p)) for p in node_path_raw]
        if len(node_path) < 2:
            continue

        path_segments = []
        path_segments_oriented = []
        for j in range(len(node_path) - 1):
            seg = canonical_seg_id(node_path[j], node_path[j + 1])
            path_segments.append(seg)
            path_segments_oriented.append(f"{node_path[j]}->{node_path[j+1]}")

        svc = {
            "service_id": sid,
            "src": node_path[0],
            "dst": node_path[-1],
            "node_path": node_path,
            "path_segments": path_segments,
            "path_segments_oriented": path_segments_oriented,
        }
        vmf_path = expand_service_to_vmf(svc, id_rules, regen_hops)
        svc["vmf_path"] = vmf_path
        services.append(svc)

        # Build roles from vmf_path
        for vmf in vmf_path:
            role_rec = {
                "vmf_id": vmf["vmf_id"],
                "role": vmf["role"],
                "node_id": vmf["node"],
                "service_id": sid,
                "segment_id": vmf.get("segment"),
                "is_regen_boundary": vmf.get("regen", False),
            }
            all_roles.append(role_rec)

    # Deduplicate roles by vmf_id
    roles_deduped = {}
    for r in all_roles:
        vid = r["vmf_id"]
        if vid not in roles_deduped:
            roles_deduped[vid] = dict(r)
            roles_deduped[vid]["service_ids"] = [r["service_id"]]
        else:
            if r["service_id"] not in roles_deduped[vid].get("service_ids", []):
                roles_deduped[vid].setdefault("service_ids", []).append(r["service_id"])
    roles = list(roles_deduped.values())

    topo = {
        "network": {
            "segments": segments,
            "segments_index": segments_index,
        },
        "node_map": node_map,
        "board_map": board_map,
        "services": services,
        "id_rules": id_rules,
    }
    return topo, roles


def generate_topology_for_size(
    target_board_count: int,
    seed: int = 42,
    num_lightpaths: int = 15,
) -> dict:
    """Generate a topology targeting a specific board count.

    Board count formula (approx):
        ROADM: 11 + 12*ELECTRICAL_GROUPS boards per ROADM
        OLA: 5 boards per OLA (not used here, all ROADM)
    """
    # Solve for num_roadms and electrical_groups
    # Target = num_roadms * (11 + 12 * eg)
    # Start with 5 ROADMs, solve for eg
    configs = [
        (3, 4),    # ~3 * (11+48)  = ~177
        (4, 5),    # ~4 * (11+60)  = ~284
        (5, 5),    # ~5 * (11+60)  = ~355
        (5, 8),    # ~5 * (11+96)  = ~535
        (5, 11),   # ~5 * (11+132) = ~715
        (5, 14),   # ~5 * (11+168) = ~895
        (5, 17),   # ~5 * (11+204) = ~1075
        (6, 17),   # ~6 * (11+204) = ~1290
        (7, 17),   # ~7 * (11+204) = ~1505
        (7, 19),   # ~7 * (11+228) = ~1673
        (8, 17),   # ~8 * (11+204) = ~1720
    ]

    best = configs[0]
    best_diff = abs(target_board_count - best[0] * (11 + 12 * best[1]))
    for nr, eg in configs:
        approx = nr * (11 + 12 * eg)
        diff = abs(target_board_count - approx)
        if diff < best_diff:
            best = (nr, eg)
            best_diff = diff

    num_roadms, eg = best
    print(f"[sfc_bfl] target={target_board_count} boards → {num_roadms} ROADMs, "
          f"ELECTRICAL_GROUPS={eg}, approx={num_roadms * (11 + 12*eg)}")

    return generate_full_topology(
        num_roadms=num_roadms,
        electrical_groups=eg,
        num_lightpaths=num_lightpaths,
        seed=seed,
    )


# ============================================================================
# Board / Edge Extraction
# ============================================================================

def extract_boards_csv(board_DG: nx.DiGraph, domain_id: int = 0) -> pd.DataFrame:
    """Extract boards.csv from board graph."""
    rows = []
    for n, data in board_DG.nodes(data=True):
        parent = n.split("-")[0] if "-" in n else n
        ttype = str(data.get("tType", ""))
        layer = data.get("layer", "optical")
        if not ttype:
            for prefix in ("Tributary", "XCON", "Line", "OA", "OD", "OM", "FIU", "RegenIn", "RegenOut"):
                if prefix in n:
                    ttype = prefix
                    break
        rows.append({
            "domain_id": domain_id,
            "node_id": parent,
            "board_id": n,
            "board_type": ttype,
            "layer": layer,
        })
    return pd.DataFrame(rows)


def extract_edges_csv(board_DG: nx.DiGraph, domain_id: int = 0) -> pd.DataFrame:
    """Extract edges.csv from board graph."""
    rows = []
    for u, v, data in board_DG.edges(data=True):
        rows.append({
            "domain_id": domain_id,
            "src_board_id": u,
            "dst_board_id": v,
        })
    return pd.DataFrame(rows)


def extract_lightpaths_json(lightpath_list) -> list[dict]:
    """Extract lightpath info as list of dicts."""
    result = []
    for i, lp in enumerate(lightpath_list):
        bp = lp.boardPath if hasattr(lp, "boardPath") else lp
        np_ = compress_parent_path(bp)
        result.append({
            "lightpath_id": i,
            "board_path": list(bp),
            "node_path": np_,
        })
    return result


def extract_services_json(topo: dict) -> list[dict]:
    """Extract service paths from topology."""
    services = topo.get("services", [])
    result = []
    for svc in services:
        result.append({
            "service_id": svc["service_id"],
            "src": svc["src"],
            "dst": svc["dst"],
            "node_path": svc["node_path"],
            "path_segments": svc.get("path_segments", []),
        })
    return result


# ============================================================================
# Single OW Generation
# ============================================================================

def pick_random_board(
    board_DG: nx.DiGraph,
    topo: dict,
    rng: np.random.Generator,
    board_kind: str | None = None,
) -> tuple[str, str]:
    """Pick a random board and appropriate event type.

    Returns (board_name, event_type).
    """
    board_map = topo.get("board_map", {})
    candidates: list[tuple[str, str]] = []  # (board, kind)
    for parent, kinds in board_map.items():
        for kind, boards in kinds.items():
            k = kind.upper()
            if board_kind and k != board_kind.upper():
                continue
            if k in _EVENTS_FOR_KIND:
                for b in boards:
                    candidates.append((b, k))

    if not candidates:
        # Fallback: pick any board
        all_boards = list(board_DG.nodes())
        b = all_boards[int(rng.integers(0, len(all_boards)))]
        return b, rng.choice(ALL_EVENTS)

    idx = int(rng.integers(0, len(candidates)))
    board, kind = candidates[idx]
    events = _EVENTS_FOR_KIND.get(kind, FIBER_EVENTS)
    event = events[int(rng.integers(0, len(events)))]
    return board, event


def generate_single_ow(
    ow_id: int,
    domain_id: int,
    topo: dict,
    roles: list[dict],
    sim_cfg: dict,
    failure_spec: dict,
    rng: np.random.Generator,
    ow_length: int | None = None,
    alarm_rules_path: str | None = None,
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Generate a single Observation Window.

    Args:
        failure_spec: dict with keys: board, event, severity, duration_ms
        ow_length: override T_total (number of timesteps)
        alarm_rules_path: path to rule_database.csv

    Returns: (metrics_df, alarms_list, label_dict)
    """
    # Override T_total if requested
    sim_cfg_local = _deep_copy_cfg(sim_cfg)
    if ow_length is not None:
        sim_cfg_local["time"]["T_total"] = ow_length

    T_total = int(sim_cfg_local["time"]["T_total"])
    step_seconds = _parse_step_seconds(sim_cfg_local["time"].get("step", "1s"))

    # Build VMF graph
    graph, service_candidates = build_graph_and_service_candidates(
        topo, roles, sim_cfg_local, rng
    )

    # Simulate latent series
    X, meta = simulate_latent_series(graph, sim_cfg_local, rng)

    # Prepare failure CSV in memory
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    failure_time = base_time + timedelta(seconds=max(60, int(T_total * 0.1)))

    board = failure_spec["board"]
    event = failure_spec["event"]
    severity = failure_spec.get("severity", "Critical")
    duration_ms = failure_spec.get("duration_ms", 15000)
    parent, kind = _parse_board_name(board)
    role = _role_for_kind(kind) if kind else "RELAY"

    failure_rows = [{
        "Board": board,
        "Event": event,
        "Severity": severity,
        "Time": failure_time.strftime("%Y-%m-%d %H:%M:%S"),
        "DurationMs": duration_ms,
        "BoardRole": role,
    }]

    # Generate metrics
    metrics = sim_cfg_local["metrics"]["out_channels"]
    metrics_map = latent_to_metrics(X, metrics, rng)

    # Inject failures
    node_map = topo.get("node_map", {})
    alarm_events, alarm_start_time = inject_failures_to_bbe(
        graph, metrics_map, failure_rows, node_map,
        step_seconds=step_seconds,
        lead_sec=60,
        default_duration_sec=int(duration_ms / 1000),
        spike_value=10.0,
        prop_hops=4,
        prop_decay=0.7,
        prop_min_scale=0.2,
        prop_bidir=True,
        prop_delay_min_sec=1,
        prop_delay_max_sec=8,
        seed=int(rng.integers(0, 2**31)),
    )

    # Generate alarm flow if rules available
    alarms_list = []
    alarm_flow_csv_path = None
    if alarm_rules_path and Path(alarm_rules_path).exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_p = Path(tmpdir)

            # Write paths file
            paths_file = tmpdir_p / "paths.txt"
            write_paths_from_topology(topo, str(paths_file), failures=failure_rows)

            # Write failure CSV
            fail_csv = tmpdir_p / "failure.csv"
            pd.DataFrame(failure_rows).to_csv(fail_csv, index=False)

            # Generate alarm flow
            df_rules = load_rule_database(alarm_rules_path)
            alarm_flow_csv = tmpdir_p / "alarm_flow.csv"
            generate_alarm_flow(
                graphml_path=str(paths_file),
                df_rules=df_rules,
                out_csv=str(alarm_flow_csv),
                failures_csv=str(fail_csv),
                paths_file=str(paths_file),
            )

            # Note: alarm spike injection via apply_alarm_spikes_from_flow
            # requires per-node CSV files on disk. The core failure effects are
            # already in metrics_map from inject_failures_to_bbe, so we skip
            # the tmpdir round-trip and just parse the alarm flow for labels.

            # Parse alarm flow
            if alarm_flow_csv.exists():
                try:
                    df_af = pd.read_csv(alarm_flow_csv)
                    if not df_af.empty:
                        for _, row in df_af.iterrows():
                            alarms_list.append({
                                "board": str(row.get("PropToBoard", "")),
                                "alarm": str(row.get("PropAlarm", "")),
                                "severity": str(row.get("Severity", "")),
                                "hop": int(row.get("Hop", 0)),
                                "arrival_time": str(row.get("ArrivalTime", "")),
                                "root_board": str(row.get("RootCauseBoard", "")),
                                "root_event": str(row.get("RootCauseEvent", "")),
                            })
                except Exception:
                    pass

    # Build metrics DataFrame
    vmf_ids = [v["id"] for v in graph.vmf_list]
    vmf_board_map = _build_vmf_board_map(graph, topo, failure_rows)

    T = X.shape[0]
    rows = []
    for gi, vid in enumerate(vmf_ids):
        board_name = vmf_board_map.get(vid, vid)
        for t in range(T):
            row = {
                "ow_id": ow_id,
                "domain_id": domain_id,
                "board_id": board_name,
                "t": t,
            }
            for m in metrics:
                row[m] = float(metrics_map[m][t, gi])
            # Derive SES from ES
            if "ES" in row:
                row["SES"] = 1 if row["ES"] > SES_THRESHOLD else 0
            rows.append(row)

    metrics_df = pd.DataFrame(rows)

    # Build label
    label = {
        "ow_id": ow_id,
        "domain_id": domain_id,
        "root_board": board,
        "root_event": event,
        "severity": severity,
        "failure_time": failure_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_ms": duration_ms,
    }

    return metrics_df, alarms_list, label


def _deep_copy_cfg(cfg: dict) -> dict:
    """Deep copy a YAML config dict."""
    import copy
    return copy.deepcopy(cfg)


def _build_vmf_board_map(graph, topo, failure_rows=None):
    """Map VMF IDs to board names using topology board_map."""
    from otn_simulator import (
        _inverse_node_map, _board_for_role, _build_failure_pref,
    )
    vmf_board_map = {}
    node_map = topo.get("node_map") or {}
    board_map = topo.get("board_map") or {}
    parent_by_id = _inverse_node_map(node_map)
    failure_pref = _build_failure_pref(failure_rows or [])
    vmf_ids = [v["id"] for v in graph.vmf_list]
    for vid in vmf_ids:
        node_id = graph.node_of.get(vid)
        parent = parent_by_id.get(int(node_id)) if node_id is not None else None
        if parent:
            board = _board_for_role(parent, graph.role_of.get(vid), board_map, failure_pref)
            if board:
                vmf_board_map[vid] = board
    return vmf_board_map


# ============================================================================
# Domain Statistics
# ============================================================================

def compute_domain_stats(
    boards_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    lightpaths: list[dict],
) -> dict:
    """Compute per-domain topology statistics.

    Returns a dict (or list of dicts for multi-domain) with:
        num_nodes, num_boards, boards_per_node, board_type_counts,
        avg_degree, num_lightpaths, avg_lightpath_hops, board_traverse_coverage
    """
    domain_ids = sorted(boards_df["domain_id"].unique()) if "domain_id" in boards_df.columns else [0]

    per_domain = []
    for did in domain_ids:
        b = boards_df[boards_df["domain_id"] == did] if len(domain_ids) > 1 else boards_df
        e = edges_df[edges_df["domain_id"] == did] if "domain_id" in edges_df.columns and len(domain_ids) > 1 else edges_df

        num_boards = len(b)
        nodes = b["node_id"].unique()
        num_nodes = len(nodes)

        # boards_per_node: {node_id: count}
        bpn = b.groupby("node_id").size().to_dict()
        # Summarise as min/max/mean
        counts = list(bpn.values())
        boards_per_node = {
            "min": int(min(counts)) if counts else 0,
            "max": int(max(counts)) if counts else 0,
            "mean": round(float(np.mean(counts)), 1) if counts else 0.0,
        }

        # board_type_counts
        board_type_counts = b["board_type"].value_counts().to_dict()
        # Ensure keys are plain strings and values are ints
        board_type_counts = {str(k): int(v) for k, v in board_type_counts.items()}

        # avg_degree from edges (undirected degree per board)
        if len(e) > 0:
            degree_counts: dict[str, int] = {}
            for _, row in e.iterrows():
                src = row["src_board_id"]
                dst = row["dst_board_id"]
                degree_counts[src] = degree_counts.get(src, 0) + 1
                degree_counts[dst] = degree_counts.get(dst, 0) + 1
            avg_degree = round(float(np.mean(list(degree_counts.values()))), 2) if degree_counts else 0.0
        else:
            avg_degree = 0.0

        # Lightpath stats for this domain
        domain_lps = [lp for lp in lightpaths if lp.get("domain_id", 0) == did]
        if not domain_lps:
            # If no domain_id tag, use all lightpaths for single-domain
            domain_lps = lightpaths if len(domain_ids) == 1 else []

        num_lps = len(domain_lps)
        node_hops = [len(lp.get("node_path", [])) - 1 for lp in domain_lps if len(lp.get("node_path", [])) > 1]
        board_hops = [len(lp.get("board_path", [])) - 1 for lp in domain_lps if len(lp.get("board_path", [])) > 1]
        avg_hops = round(float(np.mean(node_hops)), 2) if node_hops else 0.0
        avg_board_hops = round(float(np.mean(board_hops)), 2) if board_hops else 0.0

        # board_traverse_coverage: fraction of boards traversed by at least one lightpath
        traversed = set()
        for lp in domain_lps:
            for board in lp.get("board_path", []):
                traversed.add(board)
        all_board_ids = set(b["board_id"].tolist())
        coverage = round(len(traversed & all_board_ids) / max(1, len(all_board_ids)), 4)

        entry = {
            "domain_id": int(did),
            "num_nodes": int(num_nodes),
            "num_boards": int(num_boards),
            "boards_per_node": boards_per_node,
            "board_type_counts": board_type_counts,
            "avg_degree": avg_degree,
            "num_lightpaths": num_lps,
            "avg_lightpath_node_hops": avg_hops,
            "avg_lightpath_board_hops": avg_board_hops,
            "board_traverse_coverage": coverage,
        }
        per_domain.append(entry)

    if len(per_domain) == 1:
        return per_domain[0]
    return {"domains": per_domain}


# ============================================================================
# Output Writers
# ============================================================================

class StreamingDatasetWriter:
    """Incremental writer that appends OW data to disk without accumulating in memory."""

    def __init__(
        self,
        outdir: str | Path,
        boards_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        lightpaths: list[dict],
        services: list[dict],
    ):
        self.outdir = Path(outdir).resolve()
        safe_mkdir(str(self.outdir))

        # Write static topology files immediately
        boards_df.to_csv(self.outdir / "boards.csv", index=False)
        edges_df.to_csv(self.outdir / "edges.csv", index=False)
        with open(self.outdir / "lightpaths.json", "w") as f:
            json.dump(lightpaths, f, indent=2)
        with open(self.outdir / "services.json", "w") as f:
            json.dump(services, f, indent=2)

        # domain_stats.json
        stats = compute_domain_stats(boards_df, edges_df, lightpaths)
        with open(self.outdir / "domain_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Open streaming files
        self._alarms_f = open(self.outdir / "ow_alarms.jsonl", "w")
        self._labels_f = open(self.outdir / "ow_labels.jsonl", "w")
        self._metrics_parts: list[Path] = []
        self._ow_count = 0
        self._part_idx = 0
        self._metrics_buf: list[pd.DataFrame] = []
        self._metrics_buf_rows = 0
        self._flush_threshold = 5_000_000  # ~5 OWs worth before flushing

        self.boards_df = boards_df
        self.edges_df = edges_df

    def append_ow(
        self,
        metrics_df: pd.DataFrame,
        alarms: list[dict],
        label: dict,
    ) -> None:
        """Append one OW's data. Flushes metrics to disk periodically."""
        # Alarms
        record = {"ow_id": self._ow_count, "alarms": alarms}
        self._alarms_f.write(json.dumps(record) + "\n")

        # Labels
        self._labels_f.write(json.dumps(label) + "\n")

        # Metrics — buffer then flush
        self._metrics_buf.append(metrics_df)
        self._metrics_buf_rows += len(metrics_df)
        self._ow_count += 1

        if self._metrics_buf_rows >= self._flush_threshold:
            self._flush_metrics()

    def _flush_metrics(self) -> None:
        if not self._metrics_buf:
            return
        part_path = self.outdir / f"_metrics_part_{self._part_idx}.parquet"
        chunk = pd.concat(self._metrics_buf, ignore_index=True)
        chunk.to_parquet(part_path, index=False)
        self._metrics_parts.append(part_path)
        self._metrics_buf.clear()
        self._metrics_buf_rows = 0
        self._part_idx += 1

    def finalize(self) -> None:
        """Flush remaining data. Metrics are stored as partitioned parquet files."""
        self._flush_metrics()
        self._alarms_f.close()
        self._labels_f.close()

        # Move part files into ow_metrics/ directory
        metrics_dir = self.outdir / "ow_metrics"
        if self._metrics_parts:
            safe_mkdir(str(metrics_dir))
            for i, p in enumerate(self._metrics_parts):
                dest = metrics_dir / f"part_{i:04d}.parquet"
                p.rename(dest)
        else:
            safe_mkdir(str(metrics_dir))
            pd.DataFrame().to_parquet(metrics_dir / "part_0000.parquet", index=False)

        print(f"[sfc_bfl] wrote dataset to {self.outdir}: "
              f"{self._ow_count} OWs, "
              f"{len(self.boards_df)} boards, "
              f"{len(self.edges_df)} edges, "
              f"{len(self._metrics_parts)} metric parts")


def write_dataset(
    outdir: str | Path,
    boards_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    lightpaths: list[dict],
    services: list[dict],
    ow_metrics_list: list[pd.DataFrame],
    ow_alarms_list: list[list[dict]],
    ow_labels_list: list[dict],
) -> None:
    """Write all dataset files to outdir (batch mode, for small datasets)."""
    writer = StreamingDatasetWriter(outdir, boards_df, edges_df, lightpaths, services)
    for metrics_df, alarms, label in zip(ow_metrics_list, ow_alarms_list, ow_labels_list):
        writer.append_ow(metrics_df, alarms, label)
    writer.finalize()


# ============================================================================
# BFS region expansion (for Dataset A)
# ============================================================================

def bfs_region(
    board_DG: nx.DiGraph,
    root_board: str,
    k: int,
    rng: np.random.Generator,
) -> list[str]:
    """BFS/random walk to expand a connected subgraph of size k from root_board."""
    if k <= 1:
        return [root_board]

    # Build undirected adjacency for BFS
    adj: dict[str, list[str]] = defaultdict(list)
    for u, v in board_DG.edges():
        adj[u].append(v)
        adj[v].append(u)

    visited = {root_board}
    frontier = list(adj.get(root_board, []))
    rng.shuffle(frontier)

    while len(visited) < k and frontier:
        # Pick random from frontier
        idx = int(rng.integers(0, len(frontier)))
        node = frontier.pop(idx)
        if node in visited:
            continue
        visited.add(node)
        # Add neighbors to frontier
        for n in adj.get(node, []):
            if n not in visited:
                frontier.append(n)
        rng.shuffle(frontier)

    return list(visited)


# ============================================================================
# Default config loader
# ============================================================================

def load_default_sim_cfg() -> dict:
    """Load default simulator.yaml from project root."""
    cfg_path = Path(_PROJECT_ROOT) / "simulator.yaml"
    if cfg_path.exists():
        return load_yaml(str(cfg_path))
    # Minimal default
    return {
        "time": {"T_total": 900, "step": "1s", "L_warmup": 60},
        "ar": {"alpha_min": 0.8, "alpha_max": 0.98},
        "propagation": {
            "beta_min": 0.05, "beta_max": 0.2,
            "delay_choices": [1, 2, 3],
            "regen": {"delay": 1, "delay_choices": [1, 2], "weight": 1.0,
                       "otuk_reset_mult": 0.1, "oduk_passthrough_mult": 1.0},
        },
        "noise": {"epsilon_sigma": 0.15},
        "role_priors": {
            "SRC": {"alpha": 0.0, "beta": {"tier": 0.95}, "noise": {"sigma_scale": 1.0},
                    "anomaly_bias": {"p_root": 0.45}},
            "RELAY": {"alpha": 0.0, "beta": {"tier": 0.92}, "noise": {"sigma_scale": 1.0},
                      "anomaly_bias": {"p_root": 0.35}},
            "SNK": {"alpha": 0.0, "beta": {"tier": 0.9}, "noise": {"sigma_scale": 0.8},
                    "anomaly_bias": {"p_root": 0.2}},
        },
        "metrics": {"out_channels": ["BBE", "BBER", "ES"]},
        "anomaly": {
            "amp_range": [2.0, 4.0], "width_steps": [5, 60],
            "type_choices": ["spike", "shift", "burst"],
            "burst": {"mode_choices": ["multi_pulse", "damped_ring", "exp_decay"]},
            "shift_ratio_max": 0.2, "min_gap_steps_per_vmf": 2,
            "max_concurrent_roots_per_fiber": 1, "window_overlap_limit": 2,
            "target_roots_per_mw": [100, 150],
        },
    }


def default_alarm_rules_path() -> str:
    """Return path to default rule_database.csv."""
    return str(Path(_PROJECT_ROOT) / "outputs" / "Static files" / "rule_database.csv")
