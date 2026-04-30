#!/usr/bin/env python3
"""Generate a small-scale debug dataset.

Specs:
    num_nodes: 8
    num_boards: 200-500
    avg_degree: ~2.03
    board_types: reasonable distribution

Usage:
    python -m sfc_bfl.generate_small_debug --outdir outputs/sfc_bfl/small_debug
    python -m sfc_bfl.generate_small_debug --eg 3 --outdir outputs/sfc_bfl/small_debug
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("ARANGO_SKIP", "1")

import numpy as np

from .common import (
    generate_full_topology,
    extract_boards_csv,
    extract_edges_csv,
    extract_lightpaths_json,
    extract_services_json,
    generate_single_ow,
    pick_random_board,
    write_dataset,
    load_default_sim_cfg,
    default_alarm_rules_path,
)


def generate_small_debug(
    outdir: str,
    num_roadms: int = 8,
    electrical_groups: int = 3,
    num_lightpaths: int = 20,
    num_ows: int = 50,
    seed: int = 42,
    ow_length: int = 900,
):
    """Generate a small-scale debug dataset with ~8 nodes, 200-500 boards."""
    rng = np.random.default_rng(seed)

    approx_boards = num_roadms * (11 + 12 * electrical_groups)
    print(f"[small_debug] config: {num_roadms} ROADMs, EG={electrical_groups}")
    print(f"[small_debug] approx boards: {approx_boards}")

    print(f"[small_debug] generating topology ...")
    topo_data = generate_full_topology(
        num_roadms=num_roadms,
        electrical_groups=electrical_groups,
        num_lightpaths=num_lightpaths,
        seed=seed,
    )
    board_DG = topo_data["board_DG"]
    topo = topo_data["topo"]
    roles = topo_data["roles"]
    lightpath_list = topo_data["lightpath_list"]

    actual_boards = len(board_DG.nodes())
    actual_edges = len(board_DG.edges())
    print(f"[small_debug] actual: {actual_boards} boards, {actual_edges} edges")

    boards_df = extract_boards_csv(board_DG)
    edges_df = extract_edges_csv(board_DG)
    lightpaths = extract_lightpaths_json(lightpath_list)
    services = extract_services_json(topo)

    sim_cfg = load_default_sim_cfg()
    sim_cfg["time"]["T_total"] = ow_length
    alarm_rules = default_alarm_rules_path()

    ow_metrics_list = []
    ow_alarms_list = []
    ow_labels_list = []

    print(f"[small_debug] generating {num_ows} OWs (ow_length={ow_length}) ...")
    for i in range(num_ows):
        root_board, event = pick_random_board(board_DG, topo, rng)

        failure_spec = {
            "board": root_board,
            "event": event,
            "severity": "Critical",
            "duration_ms": int(rng.integers(5000, 20001)),
        }

        try:
            metrics_df, alarms, label = generate_single_ow(
                ow_id=i,
                domain_id=0,
                topo=topo,
                roles=roles,
                sim_cfg=sim_cfg,
                failure_spec=failure_spec,
                rng=rng,
                ow_length=ow_length,
                alarm_rules_path=alarm_rules,
            )
            ow_metrics_list.append(metrics_df)
            ow_alarms_list.append(alarms)
            ow_labels_list.append(label)
        except Exception as e:
            print(f"[small_debug] ow={i} failed: {e}")
            continue

    write_dataset(
        outdir, boards_df, edges_df, lightpaths, services,
        ow_metrics_list, ow_alarms_list, ow_labels_list,
    )
    print(f"[small_debug] done: {len(ow_labels_list)} OWs in {outdir}")


def main():
    ap = argparse.ArgumentParser(description="Small-scale debug dataset generator")
    ap.add_argument("--outdir", type=str, default="outputs/sfc_bfl/small_debug")
    ap.add_argument("--num-roadms", type=int, default=8)
    ap.add_argument("--eg", type=int, default=3,
                    help="ELECTRICAL_GROUPS (default 3 → ~376 boards for 8 nodes)")
    ap.add_argument("--num-lightpaths", type=int, default=20)
    ap.add_argument("--num-ows", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ow-length", type=int, default=900)
    args = ap.parse_args()

    generate_small_debug(
        outdir=args.outdir,
        num_roadms=args.num_roadms,
        electrical_groups=args.eg,
        num_lightpaths=args.num_lightpaths,
        num_ows=args.num_ows,
        seed=args.seed,
        ow_length=args.ow_length,
    )


if __name__ == "__main__":
    main()
