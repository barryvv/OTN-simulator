#!/usr/bin/env python3
"""Generate Training Dataset using external DSFC lightpaths per domain.

Usage:
    python generate_train_from_dsfc.py \
        --dsfc-dir /path/to/DSFC_train \
        --outdir outputs/sfc_bfl/dsfc_dataset_train \
        [--pilot]

DSFC_train/ layout:
    domain_0/sfc_cover.paths.txt
    domain_1/sfc_cover.paths.txt
    domain_2/sfc_cover.paths.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("ARANGO_SKIP", "1")


def parse_dsfc_lightpaths(dsfc_path: str) -> list[list[str]]:
    """Read DSFC cover file and return list of board-name lists with $0 suffix."""
    lightpaths = []
    with open(dsfc_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            boards = [b.strip() + "$0" if "$" not in b.strip() else b.strip()
                      for b in line.split(",")]
            lightpaths.append(boards)
    return lightpaths


def extract_node_edges(lightpaths: list[list[str]]) -> list[tuple[str, str]]:
    """Extract unique ROADM-level edges from lightpaths."""
    edges = set()
    for lp in lightpaths:
        nodes = []
        for b in lp:
            n = b.split("-")[0]
            if not nodes or nodes[-1] != n:
                nodes.append(n)
        for i in range(len(nodes) - 1):
            a, b = nodes[i], nodes[i + 1]
            edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def build_node_dg_from_edges(edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Build a DiGraph with the exact ROADM connectivity from DSFC edges."""
    node_DG = nx.DiGraph()
    all_nodes = set()
    for a, b in edges:
        all_nodes.add(a)
        all_nodes.add(b)
    for n in sorted(all_nodes):
        node_DG.add_node(n)
    for a, b in edges:
        node_DG.add_edge(a, b)
    return node_DG


def generate_topology_from_dsfc(
    dsfc_lightpaths: list[list[str]],
    seed: int = 42,
    electrical_groups: int = 5,
) -> dict:
    """Generate topology with node connectivity matching the DSFC lightpaths."""
    import importlib
    import class_obj

    eg = electrical_groups
    os.environ["ELECTRICAL_GROUPS"] = str(eg)
    os.environ["REGEN_HOPS"] = "0"
    importlib.reload(class_obj)

    import topology_traffic_processor as ttp
    importlib.reload(ttp)
    from topology_traffic_processor import process_traffic_from_nodeDG as proc
    from run import (
        _ensure_el_nodes_and_edges,
        _annotate_roles_for_path,
    )
    from sfc_bfl.common import _build_topo_and_roles

    import random
    random.seed(seed)

    # Build node_DG from DSFC edge connectivity
    node_edges = extract_node_edges(dsfc_lightpaths)
    node_DG = build_node_dg_from_edges(node_edges)
    num_nodes = len(node_DG.nodes())
    print(f"    node_DG: {num_nodes} nodes, {len(node_edges)} edges: {node_edges}")

    # Build board graph from node graph
    node_list, nc_list, board_DG = proc(node_DG)
    _ensure_el_nodes_and_edges(board_DG)

    # Use DSFC lightpaths directly
    lightpath_list = dsfc_lightpaths

    # Build topo + roles
    topo, roles = _build_topo_and_roles(board_DG, lightpath_list, node_DG)

    # Annotate paths for alarm flow
    annotated = []
    for lp in dsfc_lightpaths:
        try:
            ann = _annotate_roles_for_path(board_DG, lp)
            annotated.append(ann)
        except Exception:
            annotated.append(lp)
    topo["_annotated_paths"] = annotated

    return {
        "node_DG": node_DG,
        "board_DG": board_DG,
        "lightpath_list": lightpath_list,
        "topo": topo,
        "roles": roles,
        "board_list": list(board_DG.nodes()),
    }


def generate_train_from_dsfc(
    dsfc_dir: str,
    outdir: str,
    pilot: bool = False,
    seed: int = 42,
    ow_length: int | None = None,
    electrical_groups: int = 5,
) -> None:
    """Generate Training Dataset from DSFC lightpaths."""
    from sfc_bfl.common import (
        extract_boards_csv,
        extract_edges_csv,
        extract_lightpaths_json,
        extract_services_json,
        generate_single_ow,
        event_for_board,
        StreamingDatasetWriter,
        load_default_sim_cfg,
        default_alarm_rules_path,
    )
    from otn_simulator import safe_mkdir

    rng = np.random.default_rng(seed)
    sim_cfg = load_default_sim_cfg()
    sim_cfg["time"]["T_total"] = ow_length if ow_length else 900
    alarm_rules = default_alarm_rules_path()

    dsfc_dir_p = Path(dsfc_dir)
    domain_dirs = sorted([d for d in dsfc_dir_p.iterdir()
                          if d.is_dir() and d.name.startswith("domain_")])

    print(f"[dsfc_train] found {len(domain_dirs)} domains in {dsfc_dir}")

    for domain_dir in domain_dirs:
        d = int(domain_dir.name.split("_")[1])
        dsfc_file = domain_dir / "sfc_cover.paths.txt"
        if not dsfc_file.exists():
            print(f"[dsfc_train] WARNING: {dsfc_file} not found, skipping domain {d}")
            continue

        domain_outdir = f"{outdir}/domain_{d}"
        safe_mkdir(domain_outdir)

        print(f"\n[dsfc_train] === domain {d} ===")
        dsfc_lightpaths = parse_dsfc_lightpaths(str(dsfc_file))
        print(f"  loaded {len(dsfc_lightpaths)} DSFC lightpaths")

        # Build topology matching this domain's connectivity
        domain_seed = seed + d * 1000
        topo_data = generate_topology_from_dsfc(
            dsfc_lightpaths, seed=domain_seed, electrical_groups=electrical_groups,
        )
        board_DG = topo_data["board_DG"]
        topo = topo_data["topo"]
        roles = topo_data["roles"]

        # Validate
        all_dg_boards = set(board_DG.nodes())
        dsfc_boards = set()
        for lp in dsfc_lightpaths:
            dsfc_boards.update(lp)
        missing = dsfc_boards - all_dg_boards
        if missing:
            print(f"  WARNING: {len(missing)} DSFC boards not in board_DG")
            for m in sorted(missing)[:5]:
                print(f"    {m}")

        boards_df = extract_boards_csv(board_DG, domain_id=d)
        edges_df = extract_edges_csv(board_DG, domain_id=d)
        lightpaths_json = extract_lightpaths_json(dsfc_lightpaths)
        services = extract_services_json(topo)

        board_map = topo.get("board_map", {})
        all_boards = sorted(board_DG.nodes())
        if pilot:
            all_boards = all_boards[:20]

        print(f"  {len(all_boards)} boards, {len(dsfc_lightpaths)} lightpaths, "
              f"{len(services)} services")
        print(f"  generating 1 OW per board ...")

        writer = StreamingDatasetWriter(
            domain_outdir, boards_df, edges_df, lightpaths_json, services,
        )

        for ow_id, board in enumerate(all_boards):
            event = event_for_board(board, board_map, rng)
            failure_spec = {
                "board": board,
                "event": event,
                "severity": "Critical",
                "duration_ms": int(rng.integers(5000, 20001)),
            }
            try:
                metrics_df, alarms, label = generate_single_ow(
                    ow_id=ow_id,
                    domain_id=d,
                    topo=topo,
                    roles=roles,
                    sim_cfg=sim_cfg,
                    failure_spec=failure_spec,
                    rng=rng,
                    ow_length=ow_length,
                    alarm_rules_path=alarm_rules,
                )
                label["domain_id"] = d
                writer.append_ow(metrics_df, alarms, label)
            except Exception as e:
                print(f"  board={board} failed: {e}")
                continue

        writer.finalize()
        print(f"[dsfc_train] domain {d}: {len(all_boards)} OWs generated")

    print(f"\n[dsfc_train] done → {outdir}")


def main():
    ap = argparse.ArgumentParser(description="Generate Training Dataset from DSFC lightpaths")
    ap.add_argument("--dsfc-dir", type=str, required=True,
                    help="Path to DSFC_train directory with domain_*/sfc_cover.paths.txt")
    ap.add_argument("--outdir", type=str, default="outputs/sfc_bfl/dsfc_dataset_train")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ow-length", type=int, default=None)
    ap.add_argument("--eg", type=int, default=5)
    args = ap.parse_args()

    generate_train_from_dsfc(
        dsfc_dir=args.dsfc_dir,
        outdir=args.outdir,
        pilot=args.pilot,
        seed=args.seed,
        ow_length=args.ow_length,
        electrical_groups=args.eg,
    )


if __name__ == "__main__":
    main()
