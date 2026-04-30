#!/usr/bin/env python3
"""Dataset C: Service Coverage Sweep (fraction sweep, variable OW per fraction)."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
)


def _mask_metrics_by_coverage(
    metrics_df: pd.DataFrame,
    topo: dict,
    fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Mask metrics for boards not covered by remaining services.

    When fraction < 1.0, randomly remove services until only `fraction`
    remain. Boards not traversed by any remaining service get NaN metrics.
    """
    if fraction >= 1.0:
        return metrics_df

    services = topo.get("services", [])
    n_services = len(services)
    n_keep = max(1, int(fraction * n_services))

    # Randomly select services to keep
    indices = list(range(n_services))
    rng.shuffle(indices)
    keep_indices = set(indices[:n_keep])

    # Collect boards covered by remaining services
    board_map = topo.get("board_map", {})
    covered_boards = set()
    for idx in keep_indices:
        svc = services[idx]
        node_path = svc.get("node_path", [])
        node_map = topo.get("node_map", {})
        inv_map = {v: k for k, v in node_map.items()}
        for nid in node_path:
            parent = inv_map.get(int(nid), inv_map.get(nid))
            if parent and parent in board_map:
                for kind, boards in board_map[parent].items():
                    covered_boards.update(boards)

    # Mask uncovered boards
    df = metrics_df.copy()
    mask = ~df["board_id"].isin(covered_boards)
    for col in ["BBE", "BBER", "ES", "SES"]:
        if col in df.columns:
            df.loc[mask, col] = float("nan")

    return df


def generate_dataset_c(
    outdir: str,
    pilot: bool = False,
    seed: int = 42,
    ow_length: int | None = None,
    num_roadms: int = 5,
    electrical_groups: int | None = None,
) -> None:
    """Generate Dataset C: Service Coverage Sweep."""
    fractions = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
    ow_per_frac = 10 if pilot else 200

    rng = np.random.default_rng(seed)
    sim_cfg = load_default_sim_cfg()
    if ow_length is None:
        sim_cfg["time"]["T_total"] = 900

    alarm_rules = default_alarm_rules_path()

    print(f"[dataset_c] generating topology ...")
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

    writer = StreamingDatasetWriter(outdir, boards_df, edges_df, lightpaths, services)
    ow_id = 0

    for frac in fractions:
        print(f"[dataset_c] fraction={frac}, generating {ow_per_frac} OWs ...")
        for i in range(ow_per_frac):
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

                # Apply coverage mask
                metrics_df = _mask_metrics_by_coverage(
                    metrics_df, topo, frac, rng
                )

                label["coverage_fraction"] = frac

                writer.append_ow(metrics_df, alarms, label)
                ow_id += 1
            except Exception as e:
                print(f"[dataset_c] frac={frac} ow={i} failed: {e}")
                continue

    writer.finalize()
    print(f"[dataset_c] done: {ow_id} OWs generated")
