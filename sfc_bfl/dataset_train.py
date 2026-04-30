#!/usr/bin/env python3
"""Training Dataset: inject k=1 failure on every board, 3 domains.

Each domain has 8 nodes with independent topology and lightpaths.
For each board in each domain, exactly 1 OW is generated with a
single-board failure (k=1) on that board. Output is one folder
per domain (like Dataset E).
"""

from __future__ import annotations

import numpy as np

from .common import (
    generate_full_topology,
    extract_boards_csv,
    extract_edges_csv,
    extract_lightpaths_json,
    extract_services_json,
    generate_single_ow,
    event_for_board,
    StreamingDatasetWriter,
    load_default_sim_cfg,
    default_alarm_rules_path,
    safe_mkdir,
)


def generate_dataset_train(
    outdir: str,
    pilot: bool = False,
    seed: int = 42,
    ow_length: int | None = None,
    electrical_groups: int | None = None,
) -> None:
    """Generate Training Dataset.

    3 domains × 8 nodes. For each board in each domain, create exactly
    1 OW with k=1 failure on that board. Total OWs = 3 × boards_per_domain.
    """
    NUM_DOMAINS = 3
    NUM_ROADMS = 8

    rng = np.random.default_rng(seed)
    sim_cfg = load_default_sim_cfg()
    if ow_length is not None:
        sim_cfg["time"]["T_total"] = ow_length
    if ow_length is None:
        sim_cfg["time"]["T_total"] = 900

    alarm_rules = default_alarm_rules_path()

    for d in range(NUM_DOMAINS):
        domain_outdir = f"{outdir}/domain_{d}"
        safe_mkdir(domain_outdir)

        domain_seed = seed + d * 1000
        print(f"[dataset_train] generating domain {d} topology (seed={domain_seed}) ...")
        topo_data = generate_full_topology(
            seed=domain_seed,
            num_roadms=NUM_ROADMS,
            electrical_groups=electrical_groups,
        )
        board_DG = topo_data["board_DG"]
        topo = topo_data["topo"]
        roles = topo_data["roles"]
        lightpath_list = topo_data["lightpath_list"]

        boards_df = extract_boards_csv(board_DG, domain_id=d)
        edges_df = extract_edges_csv(board_DG, domain_id=d)
        lightpaths = extract_lightpaths_json(lightpath_list)
        services = extract_services_json(topo)

        board_map = topo.get("board_map", {})
        all_boards = sorted(board_DG.nodes())

        # Pilot: small subset for debug
        if pilot:
            all_boards = all_boards[:20]

        print(f"[dataset_train] domain {d}: {len(all_boards)} boards, "
              f"generating 1 OW per board ...")

        writer = StreamingDatasetWriter(
            domain_outdir, boards_df, edges_df, lightpaths, services,
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
                print(f"[dataset_train] domain={d} board={board} failed: {e}")
                continue

        writer.finalize()
        print(f"[dataset_train] domain {d}: {len(all_boards)} OWs generated")

    print(f"[dataset_train] done")
