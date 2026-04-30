#!/usr/bin/env python3
"""Dataset D: Single-Domain Scalability (board_count × region_size k sweep)."""

from __future__ import annotations

import numpy as np

from .common import (
    generate_topology_for_size,
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
    safe_mkdir,
)


def generate_dataset_d(
    outdir: str,
    pilot: bool = False,
    seed: int = 42,
    ow_length: int | None = None,
) -> None:
    """Generate Dataset D: Single-Domain Scalability.

    For each board_count × k combination, generate OWs where k boards
    in a connected region are affected by a failure.
    """
    board_counts = [200, 400, 800, 1600]
    k_values = [1, 2, 3, 5, 8, 10]
    ow_per_combo = 10 if pilot else 250

    rng = np.random.default_rng(seed)
    sim_cfg = load_default_sim_cfg()
    if ow_length is not None:
        sim_cfg["time"]["T_total"] = ow_length
    if ow_length is None:
        sim_cfg["time"]["T_total"] = 900

    alarm_rules = default_alarm_rules_path()

    for target_count in board_counts:
        size_outdir = f"{outdir}/size_{target_count}"
        safe_mkdir(size_outdir)

        print(f"[dataset_d] generating topology for ~{target_count} boards ...")
        topo_data = generate_topology_for_size(
            target_board_count=target_count,
            seed=seed + target_count,
            num_lightpaths=max(10, target_count // 30),
        )
        board_DG = topo_data["board_DG"]
        topo = topo_data["topo"]
        roles = topo_data["roles"]
        lightpath_list = topo_data["lightpath_list"]

        actual_count = len(board_DG.nodes())
        print(f"[dataset_d] actual board count: {actual_count}")

        boards_df = extract_boards_csv(board_DG)
        edges_df = extract_edges_csv(board_DG)
        lightpaths = extract_lightpaths_json(lightpath_list)
        services = extract_services_json(topo)

        writer = StreamingDatasetWriter(size_outdir, boards_df, edges_df, lightpaths, services)
        ow_id = 0

        for k in k_values:
            print(f"[dataset_d] size={target_count}, k={k}, generating {ow_per_combo} OWs ...")
            for i in range(ow_per_combo):
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

                    label["target_board_count"] = target_count
                    label["actual_board_count"] = actual_count
                    label["region_size_k"] = k
                    label["failed_boards"] = region_boards

                    writer.append_ow(metrics_df, alarms, label)
                    ow_id += 1
                except Exception as e:
                    print(f"[dataset_d] size={target_count} k={k} ow={i} failed: {e}")
                    continue

        writer.finalize()
        print(f"[dataset_d] size={target_count}: {ow_id} OWs generated")

    print(f"[dataset_d] done")
