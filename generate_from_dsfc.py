#!/usr/bin/env python3
"""Generate Dataset A (alarms + metrics) using external DSFC lightpaths.

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
from pathlib import Path

import numpy as np

# Ensure project root is on path
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
            boards = line.split(",")
            # Add $0 copy suffix if missing
            boards_fixed = []
            for b in boards:
                b = b.strip()
                if "$" not in b:
                    b += "$0"
                boards_fixed.append(b)
            lightpaths.append(boards_fixed)
    return lightpaths


def generate_dataset_a_from_dsfc(
    dsfc_path: str,
    outdir: str,
    pilot: bool = False,
    seed: int = 42,
    ow_length: int | None = None,
    num_roadms: int = 8,
    electrical_groups: int = 5,
) -> None:
    """Generate Dataset A using external DSFC lightpaths."""

    from sfc_bfl.common import (
        generate_full_topology,
        _build_topo_and_roles,
        extract_boards_csv,
        extract_edges_csv,
        extract_lightpaths_json,
        extract_services_json,
        generate_single_ow,
        pick_random_board,
        bfs_region,
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

    # 1. Generate topology (board_DG) — we'll replace lightpaths
    print(f"[dsfc] generating topology: {num_roadms} ROADMs, EG={electrical_groups} ...")
    topo_data = generate_full_topology(
        seed=seed, num_roadms=num_roadms, electrical_groups=electrical_groups,
    )
    board_DG = topo_data["board_DG"]
    node_DG = topo_data["node_DG"]

    # 2. Read external DSFC lightpaths
    print(f"[dsfc] reading DSFC lightpaths from {dsfc_path} ...")
    dsfc_lightpaths = parse_dsfc_lightpaths(dsfc_path)
    print(f"[dsfc] loaded {len(dsfc_lightpaths)} lightpaths")

    # Validate that all boards in lightpaths exist in board_DG
    all_boards = set(board_DG.nodes())
    missing = set()
    for lp in dsfc_lightpaths:
        for b in lp:
            if b not in all_boards:
                missing.add(b)
    if missing:
        print(f"[dsfc] WARNING: {len(missing)} boards in DSFC not found in board_DG:")
        for m in sorted(missing)[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    # 3. Rebuild topo + roles from external lightpaths
    print(f"[dsfc] rebuilding topology YAML + roles from DSFC lightpaths ...")
    topo, roles = _build_topo_and_roles(board_DG, dsfc_lightpaths, node_DG)

    # Copy annotated paths if present (needed for alarm flow)
    if "_annotated_paths" in topo_data.get("topo", {}):
        # Re-annotate with external lightpaths
        from run import _annotate_roles_for_path
        annotated = []
        for lp in dsfc_lightpaths:
            try:
                ann = _annotate_roles_for_path(board_DG, lp)
                annotated.append(ann)
            except Exception:
                annotated.append(lp)
        topo["_annotated_paths"] = annotated

    # 4. Extract static data for writer
    boards_df = extract_boards_csv(board_DG)
    edges_df = extract_edges_csv(board_DG)
    lightpaths_json = extract_lightpaths_json(dsfc_lightpaths)
    services = extract_services_json(topo)

    print(f"[dsfc] topology: {len(boards_df)} boards, {len(edges_df)} edges, "
          f"{len(dsfc_lightpaths)} lightpaths, {len(services)} services")

    # 5. Generate OWs with dataset_a logic
    writer = StreamingDatasetWriter(outdir, boards_df, edges_df, lightpaths_json, services)
    ow_id = 0

    for k in k_values:
        print(f"[dsfc] k={k}, generating {ow_per_k} OWs ...")
        for i in range(ow_per_k):
            root_board, event = pick_random_board(board_DG, topo, rng)
            region_boards = bfs_region(board_DG, root_board, k, rng)

            failure_spec = {
                "board": root_board,
                "event": event,
                "severity": "Critical",
                "duration_ms": int(rng.integers(5000, 20001)),
            }

            try:
                metrics_df, alarms, label = generate_single_ow(
                    ow_id=ow_id,
                    domain_id=0,
                    topo=topo,
                    roles=roles,
                    sim_cfg=sim_cfg,
                    failure_spec=failure_spec,
                    rng=rng,
                    ow_length=ow_length,
                    alarm_rules_path=alarm_rules,
                )

                label["region_size_k"] = k
                label["failed_boards"] = region_boards

                writer.append_ow(metrics_df, alarms, label)
                ow_id += 1
            except Exception as e:
                print(f"[dsfc] k={k} ow={i} failed: {e}")
                continue

    writer.finalize()
    print(f"[dsfc] done: {ow_id} OWs generated → {outdir}")


def main():
    ap = argparse.ArgumentParser(description="Generate Dataset A from external DSFC lightpaths")
    ap.add_argument("--dsfc", type=str, required=True,
                    help="Path to DSFC cover lightpaths file")
    ap.add_argument("--outdir", type=str, default="outputs/sfc_bfl/dsfc_dataset_a",
                    help="Output directory")
    ap.add_argument("--pilot", action="store_true",
                    help="Generate small pilot dataset (10 OWs per k)")
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
