#!/usr/bin/env python3
"""Dataset B: Alarm Noise (noise_ratio sweep, variable OW count per level)."""

from __future__ import annotations

import numpy as np

from .common import (
    generate_full_topology,
    extract_boards_csv,
    extract_edges_csv,
    extract_lightpaths_json,
    extract_services_json,
    generate_single_ow,
    pick_random_board,
    StreamingDatasetWriter,
    load_default_sim_cfg,
    default_alarm_rules_path,
    ALL_EVENTS,
)


def generate_dataset_b(
    outdir: str,
    pilot: bool = False,
    seed: int = 42,
    ow_length: int | None = None,
    num_roadms: int = 5,
    electrical_groups: int | None = None,
) -> None:
    """Generate Dataset B: Alarm Noise."""
    noise_ratios = [0, 0.25, 0.5, 1, 2, 4]
    ow_per_level = 15 if pilot else 200

    rng = np.random.default_rng(seed)
    sim_cfg = load_default_sim_cfg()
    if ow_length is None:
        sim_cfg["time"]["T_total"] = 900

    alarm_rules = default_alarm_rules_path()

    print(f"[dataset_b] generating topology ...")
    topo_data = generate_full_topology(
        seed=seed, num_roadms=num_roadms, electrical_groups=electrical_groups,
    )
    board_DG = topo_data["board_DG"]
    topo = topo_data["topo"]
    roles = topo_data["roles"]
    lightpath_list = topo_data["lightpath_list"]

    boards_df = extract_boards_csv(board_DG)
    edges_df = extract_edges_csv(board_DG)
    lightpaths = extract_lightpaths_json(lightpath_list)
    services = extract_services_json(topo)

    all_board_names = list(board_DG.nodes())

    writer = StreamingDatasetWriter(outdir, boards_df, edges_df, lightpaths, services)
    ow_id = 0

    for noise_ratio in noise_ratios:
        print(f"[dataset_b] noise_ratio={noise_ratio}, generating {ow_per_level} OWs ...")
        for i in range(ow_per_level):
            root_board, event = pick_random_board(board_DG, topo, rng)

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

                # Inject false-positive alarms
                n_gt = max(1, len(alarms))
                n_fp = int(noise_ratio * n_gt)
                fp_alarms = []
                for _ in range(n_fp):
                    fp_board = all_board_names[int(rng.integers(0, len(all_board_names)))]
                    fp_alarm = ALL_EVENTS[int(rng.integers(0, len(ALL_EVENTS)))]
                    fp_alarms.append({
                        "board": fp_board,
                        "alarm": fp_alarm,
                        "severity": rng.choice(["Critical", "Major", "Minor"]),
                        "hop": 0,
                        "arrival_time": "",
                        "root_board": "",
                        "root_event": "",
                        "is_fp": True,
                    })

                all_alarms = alarms + fp_alarms

                label["noise_ratio"] = noise_ratio
                label["n_gt_alarms"] = n_gt
                label["n_fp_alarms"] = n_fp

                writer.append_ow(metrics_df, all_alarms, label)
                ow_id += 1
            except Exception as e:
                print(f"[dataset_b] noise={noise_ratio} ow={i} failed: {e}")
                continue

    writer.finalize()
    print(f"[dataset_b] done: {ow_id} OWs generated")
