#!/usr/bin/env python3
"""Dataset E: Multi-Domain Scalability (domain_count sweep).

All domains inject failures simultaneously per OW. Each domain gets
an independent failure with a random region size k from 1-10.
"""

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
    bfs_region,
    StreamingDatasetWriter,
    load_default_sim_cfg,
    default_alarm_rules_path,
    safe_mkdir,
)


def generate_dataset_e(
    outdir: str,
    pilot: bool = False,
    seed: int = 42,
    ow_length: int | None = None,
    num_roadms: int = 8,
    electrical_groups: int | None = None,
) -> None:
    """Generate Dataset E: Multi-Domain Scalability.

    For each domain_count config, generate independent topologies for each
    domain. Every OW has ALL domains injecting failures simultaneously,
    each with a random k from 1-10.

    OW count = n_domains × ow_per_domain.
    """
    domain_counts = [1, 3, 5, 10]
    ow_per_domain = 10 if pilot else 250

    rng = np.random.default_rng(seed)
    sim_cfg = load_default_sim_cfg()
    if ow_length is not None:
        sim_cfg["time"]["T_total"] = ow_length
    if ow_length is None:
        sim_cfg["time"]["T_total"] = 900

    alarm_rules = default_alarm_rules_path()

    for n_domains in domain_counts:
        config_outdir = f"{outdir}/domains_{n_domains}"
        safe_mkdir(config_outdir)

        total_ow = n_domains * ow_per_domain
        print(f"[dataset_e] generating {n_domains} domain topologies, {total_ow} OWs ...")

        # Generate per-domain topologies
        domain_data = []
        all_boards_dfs = []
        all_edges_dfs = []
        all_lightpaths = []
        all_services = []

        for d in range(n_domains):
            domain_seed = seed + d * 1000 + n_domains
            topo_data = generate_full_topology(
                seed=domain_seed,
                num_roadms=num_roadms,
                electrical_groups=electrical_groups,
            )

            boards_df = extract_boards_csv(topo_data["board_DG"], domain_id=d)
            edges_df = extract_edges_csv(topo_data["board_DG"], domain_id=d)
            lps = extract_lightpaths_json(topo_data["lightpath_list"])
            svcs = extract_services_json(topo_data["topo"])

            # Tag lightpaths and services with domain_id
            for lp in lps:
                lp["domain_id"] = d
            for svc in svcs:
                svc["domain_id"] = d

            all_boards_dfs.append(boards_df)
            all_edges_dfs.append(edges_df)
            all_lightpaths.extend(lps)
            all_services.extend(svcs)

            domain_data.append({
                "domain_id": d,
                "board_DG": topo_data["board_DG"],
                "topo": topo_data["topo"],
                "roles": topo_data["roles"],
            })

        combined_boards = pd.concat(all_boards_dfs, ignore_index=True)
        combined_edges = pd.concat(all_edges_dfs, ignore_index=True)

        writer = StreamingDatasetWriter(
            config_outdir, combined_boards, combined_edges,
            all_lightpaths, all_services,
        )

        print(f"[dataset_e] n_domains={n_domains}, generating {total_ow} OWs "
              f"(all domains fail simultaneously) ...")

        for ow_idx in range(total_ow):
            all_metrics = []
            all_alarms = []
            domain_failures = []

            for dd in domain_data:
                d = dd["domain_id"]
                board_DG = dd["board_DG"]
                topo = dd["topo"]
                roles = dd["roles"]

                # Each domain gets independent random k from 1-10
                k = int(rng.integers(1, 11))
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
                        ow_id=ow_idx,
                        domain_id=d,
                        topo=topo,
                        roles=roles,
                        sim_cfg=sim_cfg,
                        failure_spec=failure_spec,
                        rng=rng,
                        ow_length=ow_length,
                        alarm_rules_path=alarm_rules,
                    )

                    all_metrics.append(metrics_df)
                    all_alarms.extend(alarms)
                    domain_failures.append({
                        "domain_id": d,
                        "root_board": root_board,
                        "root_event": event,
                        "severity": "Critical",
                        "region_size_k": k,
                        "failed_boards": region_boards,
                        "failure_time": label.get("failure_time"),
                        "duration_ms": label.get("duration_ms"),
                    })
                except Exception as e:
                    print(f"[dataset_e] domains={n_domains} ow={ow_idx} d={d} failed: {e}")
                    continue

            if not all_metrics:
                continue

            combined_metrics = pd.concat(all_metrics, ignore_index=True)
            ow_label = {
                "ow_id": ow_idx,
                "n_domains": n_domains,
                "domain_failures": domain_failures,
            }

            writer.append_ow(combined_metrics, all_alarms, ow_label)

        writer.finalize()
        print(f"[dataset_e] domains={n_domains}: {total_ow} OWs generated")

    print(f"[dataset_e] done")
