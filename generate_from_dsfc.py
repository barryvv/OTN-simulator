#!/usr/bin/env python3
"""Generate Dataset A (alarms + metrics) using external DSFC lightpaths.

Option 2 (downstream-only BFS): regional failure regions are constructed
by walking downstream on the lightpath signal direction starting from
a root board that itself sits on at least one lightpath. This guarantees
every board in `failed_boards` is observable in metrics/alarms.

Usage:
    python generate_from_dsfc.py \
        --dsfc path/to/sfc_cover.paths.txt \
        --outdir outputs/sfc_bfl/dsfc_dataset_a \
        [--pilot]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("ARANGO_SKIP", "1")


# ---------------------------------------------------------------------------
# Service-aware VMF→boards expansion
# ---------------------------------------------------------------------------

def _board_kind(name: str) -> str:
    if "-" not in name:
        return ""
    suffix = name.split("-", 1)[1].split("$")[0]
    return "".join(c for c in suffix if not c.isdigit())


def _node_of(name: str) -> str:
    return name.split("-")[0] if "-" in name else name


def _split_lightpath_by_node(lp: list[str]) -> dict[str, list[str]]:
    """Group consecutive boards by node, preserving order."""
    by_node = {}
    for b in lp:
        n = _node_of(b)
        by_node.setdefault(n, []).append(b)
    return by_node


_ROLE_KINDS = {
    "SRC":   {"TRIBUTARY", "XCON", "LINE", "OM"},          # source-side electrical + first optical mux
    "RELAY": {"OA", "FIU", "OD"},                           # optical pass-through
    "SNK":   {"OD", "LINE", "XCON", "TRIBUTARY"},          # dest-side optical demux + electrical
}


def build_service_aware_board_expansion(
    services: list[dict],
    lightpaths: list[list[str]],
    node_map: dict,
) -> dict[str, set[str]]:
    """For each VMF id in services, return the set of boards it represents on
    its specific lightpath. SRC/SNK are service-specific; RELAY VMFs are
    segment-specific so we union across all services using that segment."""
    parent_by_id = {v: k for k, v in node_map.items()}

    vmf_to_boards: dict[str, set[str]] = {}

    for svc_idx, svc in enumerate(services):
        sid = svc["service_id"]
        if svc_idx >= len(lightpaths):
            continue
        lp = lightpaths[svc_idx]
        by_node = _split_lightpath_by_node(lp)
        node_path = svc.get("node_path", [])

        for vmf in svc.get("vmf_path", []):
            vmf_id = vmf["vmf_id"]
            role = str(vmf.get("role", "")).upper()
            node_id = vmf.get("node")
            parent = parent_by_id.get(int(node_id)) if node_id is not None else None
            if not parent or parent not in by_node:
                continue
            local = by_node[parent]
            target_kinds = _ROLE_KINDS.get(
                "RELAY" if role in ("RELAY_IN", "RELAY_OUT") else role, set()
            )
            picked = {b for b in local if _board_kind(b).upper() in target_kinds}

            # Refine: SRC takes only the front portion (electrical + first OM),
            #         SNK takes the back portion (last OD + electrical),
            #         RELAY takes the optical pass-through.
            if role == "SRC":
                # First node in path: keep boards up to the first OM/FIU exit
                picked = set()
                for b in local:
                    k = _board_kind(b).upper()
                    if k in ("TRIBUTARY", "XCON", "LINE", "OM"):
                        picked.add(b)
                    elif k in ("OA", "FIU"):
                        break
            elif role == "SNK":
                # Last node in path: keep boards from the last OD onward
                seen_od = False
                picked = set()
                for b in local:
                    k = _board_kind(b).upper()
                    if k == "OD":
                        seen_od = True
                    if seen_od and k in ("OD", "LINE", "XCON", "TRIBUTARY"):
                        picked.add(b)
            elif role in ("RELAY_OUT", "RELAY_IN", "RELAY"):
                # Optical layer: OA / FIU / OD / OM
                picked = {b for b in local if _board_kind(b).upper() in
                          ("OA", "FIU", "OD", "OM")}

            if picked:
                vmf_to_boards.setdefault(vmf_id, set()).update(picked)

    return vmf_to_boards


def parse_dsfc_lightpaths(dsfc_path: str) -> list[list[str]]:
    """Read DSFC cover file and return list of board-name lists with $0 suffix."""
    lightpaths = []
    with open(dsfc_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            boards = []
            for b in line.split(","):
                b = b.strip()
                if "$" not in b:
                    b += "$0"
                boards.append(b)
            lightpaths.append(boards)
    return lightpaths


# ---------------------------------------------------------------------------
# Option 2: Downstream-constrained region expansion
# ---------------------------------------------------------------------------

def build_downstream_index(lightpaths: list[list[str]]) -> dict:
    """For each board, record the lightpaths and positions where it appears.

    Returns:
        {board: [(lp_idx, position), ...]}
    """
    index = {}
    for lp_idx, lp in enumerate(lightpaths):
        for i, b in enumerate(lp):
            index.setdefault(b, []).append((lp_idx, i))
    return index


def pick_root_with_downstream(
    lightpaths: list[list[str]],
    topo: dict,
    rng: np.random.Generator,
    min_downstream: int,
) -> tuple[str, str]:
    """Pick a board that lies on a lightpath and has at least `min_downstream`
    boards downstream of it on some lightpath. Returns (board, event)."""
    from sfc_bfl.common import _EVENTS_FOR_KIND

    # max_downstream[board] = max count of boards after this one on any lightpath
    max_downstream = {}
    for lp in lightpaths:
        n = len(lp)
        for i, b in enumerate(lp):
            ds = n - i - 1
            if ds > max_downstream.get(b, -1):
                max_downstream[b] = ds

    board_map = topo.get("board_map", {})
    candidates: list[tuple[str, str]] = []
    for parent, kinds in board_map.items():
        for kind, boards in kinds.items():
            kind_upper = kind.upper()
            if kind_upper not in _EVENTS_FOR_KIND:
                continue
            for b in boards:
                if max_downstream.get(b, -1) >= min_downstream:
                    candidates.append((b, kind_upper))

    if not candidates:
        # Relax: any board on a lightpath
        for b in max_downstream:
            parent = b.split("-")[0] if "-" in b else b
            kind_upper = ""
            for kind in board_map.get(parent, {}):
                if b in board_map[parent][kind]:
                    kind_upper = kind.upper()
                    break
            if kind_upper in _EVENTS_FOR_KIND:
                candidates.append((b, kind_upper))

    if not candidates:
        raise RuntimeError("No lightpath board can satisfy the downstream constraint")

    board, kind = candidates[int(rng.integers(0, len(candidates)))]
    events = _EVENTS_FOR_KIND.get(kind, ["Fiber_Cut"])
    event = events[int(rng.integers(0, len(events)))]
    return board, event


def downstream_region(
    root_board: str,
    k: int,
    lightpaths: list[list[str]],
    rng: np.random.Generator,
) -> list[str]:
    """Pick a contiguous downstream segment of length k starting at root_board.

    Walks signal direction along a lightpath that contains root_board. If
    multiple lightpaths contain root, prefer ones that can supply at least
    k boards starting from the root position. Returns exactly k boards
    (or fewer if no lightpath can supply more)."""
    if k <= 1:
        return [root_board]

    # Find candidate downstream segments of length up to k
    full_options: list[list[str]] = []   # exactly k boards
    short_options: list[list[str]] = []  # fewer than k
    for lp in lightpaths:
        try:
            i = lp.index(root_board)
        except ValueError:
            continue
        seg = list(lp[i : i + k])
        if len(seg) == k:
            full_options.append(seg)
        else:
            short_options.append(seg)

    if full_options:
        chosen = full_options[int(rng.integers(0, len(full_options)))]
    elif short_options:
        # Take the longest available segment
        short_options.sort(key=len, reverse=True)
        chosen = short_options[0]
    else:
        chosen = [root_board]
    return chosen


def generate_single_ow_lightpath_aware(
    ow_id, domain_id, topo, roles, sim_cfg, failure_spec, rng,
    services, lightpaths, ow_length=None, alarm_rules_path=None,
    failed_boards=None,
):
    """Like generate_single_ow but emits metrics for every lightpath board
    using service-aware VMF→board expansion.

    If `failed_boards` is provided (list of boards), inject one failure per
    board with a kind-appropriate event chosen by `event_for_board`. The
    root board uses `failure_spec["event"]`; other boards use a randomly
    picked event from their kind's pool. This makes regional failures
    (k>=2) genuinely multi-failure so each failed board has its own hop=0
    local alarm.
    """
    import tempfile
    from sfc_bfl.common import SES_THRESHOLD, event_for_board
    from otn_simulator import (
        build_graph_and_service_candidates,
        simulate_latent_series,
        latent_to_metrics,
        inject_failures_to_bbe,
        write_paths_from_topology,
        _parse_step_seconds,
        _parse_board_name,
        _role_for_kind,
    )
    from alarm_generator import load_rule_database, generate_alarm_flow, _normalize_board_id

    sim_cfg_local = {**sim_cfg, "time": dict(sim_cfg["time"])}
    if ow_length is not None:
        sim_cfg_local["time"]["T_total"] = ow_length

    T_total = int(sim_cfg_local["time"]["T_total"])
    step_seconds = _parse_step_seconds(sim_cfg_local["time"].get("step", "1s"))

    graph, _ = build_graph_and_service_candidates(topo, roles, sim_cfg_local, rng)
    X, _ = simulate_latent_series(graph, sim_cfg_local, rng)

    base_time = datetime(2025, 1, 15, 10, 0, 0)
    failure_time = base_time + timedelta(seconds=max(60, int(T_total * 0.1)))

    root_board = failure_spec["board"]
    root_event = failure_spec["event"]
    severity = failure_spec.get("severity", "Critical")
    duration_ms = failure_spec.get("duration_ms", 15000)
    board_map = topo.get("board_map", {})

    if failed_boards is None or len(failed_boards) == 0:
        failed_boards_list = [root_board]
    else:
        failed_boards_list = list(failed_boards)
        if root_board not in failed_boards_list:
            failed_boards_list = [root_board] + failed_boards_list

    failure_rows = []
    for fb in failed_boards_list:
        if fb == root_board:
            fb_event = root_event
        else:
            fb_event = event_for_board(fb, board_map, rng)
        _, fb_kind = _parse_board_name(fb)
        fb_role = _role_for_kind(fb_kind) if fb_kind else "RELAY"
        failure_rows.append({
            "Board": fb, "Event": fb_event, "Severity": severity,
            "Time": failure_time.strftime("%Y-%m-%d %H:%M:%S"),
            "DurationMs": duration_ms, "BoardRole": fb_role,
        })

    # Back-compat aliases used later in the function
    board = root_board
    event = root_event

    metrics = sim_cfg_local["metrics"]["out_channels"]
    metrics_map = latent_to_metrics(X, metrics, rng)

    node_map = topo.get("node_map", {})
    inject_failures_to_bbe(
        graph, metrics_map, failure_rows, node_map,
        step_seconds=step_seconds, lead_sec=60,
        default_duration_sec=int(duration_ms / 1000),
        spike_value=10.0, prop_hops=4, prop_decay=0.7, prop_min_scale=0.2,
        prop_bidir=True, prop_delay_min_sec=1, prop_delay_max_sec=8,
        seed=int(rng.integers(0, 2**31)),
    )

    alarms_list = []
    if alarm_rules_path and Path(alarm_rules_path).exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_p = Path(tmpdir)
            paths_file = tmpdir_p / "paths.txt"
            write_paths_from_topology(topo, str(paths_file), failures=failure_rows)
            fail_csv = tmpdir_p / "failure.csv"
            pd.DataFrame(failure_rows).to_csv(fail_csv, index=False)
            df_rules = load_rule_database(alarm_rules_path)
            alarm_flow_csv = tmpdir_p / "alarm_flow.csv"
            generate_alarm_flow(
                graphml_path=str(paths_file), df_rules=df_rules,
                out_csv=str(alarm_flow_csv), failures_csv=str(fail_csv),
                paths_file=str(paths_file),
            )
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

    # Service-aware VMF→boards expansion
    vmf_to_boards = build_service_aware_board_expansion(services, lightpaths, node_map)

    vmf_ids = [v["id"] for v in graph.vmf_list]
    T = X.shape[0]
    rows = []
    seen_board_t = set()  # avoid duplicate (board, t) — when multiple VMFs project onto same board, keep first
    for gi, vid in enumerate(vmf_ids):
        boards = vmf_to_boards.get(vid)
        if not boards:
            continue
        for board_name in boards:
            if (board_name, 0) in seen_board_t:
                continue  # already produced rows for this board
            for t in range(T):
                row = {
                    "ow_id": ow_id, "domain_id": domain_id,
                    "board_id": board_name, "t": t,
                }
                for m in metrics:
                    row[m] = float(metrics_map[m][t, gi])
                if "ES" in row:
                    row["SES"] = 1 if row["ES"] > SES_THRESHOLD else 0
                rows.append(row)
            seen_board_t.add((board_name, 0))

    metrics_df = pd.DataFrame(rows)

    label = {
        "ow_id": ow_id, "domain_id": domain_id,
        "root_board": _normalize_board_id(board), "root_event": event,
        "severity": severity,
        "failure_time": failure_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_ms": duration_ms,
        "failed_boards": [_normalize_board_id(fr["Board"]) for fr in failure_rows],
        "failed_events": [fr["Event"] for fr in failure_rows],
    }
    return metrics_df, alarms_list, label


def generate_dataset_a_from_dsfc(
    dsfc_path: str,
    outdir: str,
    pilot: bool = False,
    seed: int = 42,
    ow_length: int | None = None,
    num_roadms: int = 8,
    electrical_groups: int = 5,
) -> None:
    """Generate Dataset A using external DSFC lightpaths with downstream-only regions."""

    from sfc_bfl.common import (
        generate_full_topology,
        _build_topo_and_roles,
        extract_boards_csv,
        extract_edges_csv,
        extract_lightpaths_json,
        extract_services_json,
        StreamingDatasetWriter,
        load_default_sim_cfg,
        default_alarm_rules_path,
    )

    k_values = [1, 2, 3, 5, 8, 10]
    ow_per_k = 10 if pilot else 200

    rng = np.random.default_rng(seed)
    sim_cfg = load_default_sim_cfg()
    sim_cfg["time"]["T_total"] = ow_length if ow_length else 900
    alarm_rules = default_alarm_rules_path()

    print(f"[dsfc] generating topology: {num_roadms} ROADMs, EG={electrical_groups} ...")
    topo_data = generate_full_topology(
        seed=seed, num_roadms=num_roadms, electrical_groups=electrical_groups,
    )
    board_DG = topo_data["board_DG"]
    node_DG = topo_data["node_DG"]

    print(f"[dsfc] reading DSFC lightpaths from {dsfc_path} ...")
    dsfc_lightpaths = parse_dsfc_lightpaths(dsfc_path)
    print(f"[dsfc] loaded {len(dsfc_lightpaths)} lightpaths")

    all_dg_boards = set(board_DG.nodes())
    missing = set()
    for lp in dsfc_lightpaths:
        for b in lp:
            if b not in all_dg_boards:
                missing.add(b)
    if missing:
        print(f"[dsfc] WARNING: {len(missing)} DSFC boards not in board_DG")

    print(f"[dsfc] rebuilding topology YAML + roles from DSFC lightpaths ...")
    topo, roles = _build_topo_and_roles(board_DG, dsfc_lightpaths, node_DG)

    from run import _annotate_roles_for_path
    annotated = []
    for lp in dsfc_lightpaths:
        try:
            annotated.append(_annotate_roles_for_path(board_DG, lp))
        except Exception:
            annotated.append(lp)
    topo["_annotated_paths"] = annotated

    boards_df = extract_boards_csv(board_DG)
    edges_df = extract_edges_csv(board_DG)
    lightpaths_json = extract_lightpaths_json(dsfc_lightpaths)
    services = extract_services_json(topo)

    print(f"[dsfc] topology: {len(boards_df)} boards, {len(edges_df)} edges, "
          f"{len(dsfc_lightpaths)} lightpaths, {len(services)} services")

    writer = StreamingDatasetWriter(outdir, boards_df, edges_df, lightpaths_json, services)
    ow_id = 0

    for k in k_values:
        print(f"[dsfc] k={k}, generating {ow_per_k} OWs (downstream-only) ...")
        for i in range(ow_per_k):
            root_board, event = pick_root_with_downstream(
                dsfc_lightpaths, topo, rng, min_downstream=k - 1,
            )
            region_boards = downstream_region(root_board, k, dsfc_lightpaths, rng)

            failure_spec = {
                "board": root_board,
                "event": event,
                "severity": "Critical",
                "duration_ms": int(rng.integers(5000, 20001)),
            }

            try:
                metrics_df, alarms, label = generate_single_ow_lightpath_aware(
                    ow_id=ow_id,
                    domain_id=0,
                    topo=topo,
                    roles=roles,
                    sim_cfg=sim_cfg,
                    failure_spec=failure_spec,
                    rng=rng,
                    services=topo.get("services", []),
                    lightpaths=dsfc_lightpaths,
                    ow_length=ow_length,
                    alarm_rules_path=alarm_rules,
                    failed_boards=region_boards,
                )

                label["region_size_k"] = k

                writer.append_ow(metrics_df, alarms, label)
                ow_id += 1
            except Exception as e:
                print(f"[dsfc] k={k} ow={i} failed: {e}")
                continue

    writer.finalize()
    print(f"[dsfc] done: {ow_id} OWs generated → {outdir}")


def main():
    ap = argparse.ArgumentParser(description="Generate Dataset A from DSFC lightpaths (downstream-only)")
    ap.add_argument("--dsfc", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs/sfc_bfl/dsfc_dataset_a")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ow-length", type=int, default=None)
    ap.add_argument("--num-roadms", type=int, default=8)
    ap.add_argument("--eg", type=int, default=5)
    args = ap.parse_args()

    generate_dataset_a_from_dsfc(
        dsfc_path=args.dsfc,
        outdir=args.outdir,
        pilot=args.pilot,
        seed=args.seed,
        ow_length=args.ow_length,
        num_roadms=args.num_roadms,
        electrical_groups=args.eg,
    )


if __name__ == "__main__":
    main()
