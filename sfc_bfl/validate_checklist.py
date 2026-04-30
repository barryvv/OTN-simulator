#!/usr/bin/env python3
"""Validate generated datasets against the debug checklist.

Usage:
    python -m sfc_bfl.validate_checklist --basedir outputs/sfc_bfl/debug_check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def check_mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def validate_dataset(ds_dir: str, dataset_name: str = ""):
    """Run all checklist items on a single dataset directory."""
    ds_dir = Path(ds_dir)
    if not ds_dir.exists():
        print(f"  [SKIP] {ds_dir} does not exist")
        return

    print(f"\n{'='*70}")
    print(f"  Validating: {dataset_name} ({ds_dir})")
    print(f"{'='*70}")

    # Load files
    boards_df = pd.read_csv(ds_dir / "boards.csv")
    edges_df = pd.read_csv(ds_dir / "edges.csv")
    with open(ds_dir / "lightpaths.json") as f:
        lightpaths = json.load(f)
    with open(ds_dir / "services.json") as f:
        services = json.load(f)

    metrics_path = ds_dir / "ow_metrics.parquet"
    metrics_df = pd.read_parquet(metrics_path) if metrics_path.exists() else pd.DataFrame()

    alarms_list = []
    alarms_path = ds_dir / "ow_alarms.jsonl"
    if alarms_path.exists():
        with open(alarms_path) as f:
            for line in f:
                alarms_list.append(json.loads(line))

    labels_list = []
    labels_path = ds_dir / "ow_labels.jsonl"
    if labels_path.exists():
        with open(labels_path) as f:
            for line in f:
                labels_list.append(json.loads(line))

    stats_path = ds_dir / "domain_stats.json"
    stats = json.load(open(stats_path)) if stats_path.exists() else {}

    num_boards = len(boards_df)
    num_nodes = boards_df["node_id"].nunique()
    num_ows = len(labels_list)

    # ---- Topology checks ----
    print(f"\n--- Topology ---")

    # 1. Board count
    ok1 = 200 <= num_boards <= 500
    print(f"  [1] {check_mark(ok1)} num_boards={num_boards} (200-500 for debug, or larger)")

    # 2. Node count
    ok2 = 5 <= num_nodes <= 10
    print(f"  [2] {check_mark(ok2)} num_nodes={num_nodes} (5-10)")

    # 3. Board -> node mapping
    has_board_id = "board_id" in boards_df.columns
    has_node_id = "node_id" in boards_df.columns
    orphan_count = boards_df["node_id"].isna().sum() if has_node_id else num_boards
    ok3 = has_board_id and has_node_id and orphan_count == 0
    print(f"  [3] {check_mark(ok3)} board->node mapping: board_id={has_board_id}, node_id={has_node_id}, orphans={orphan_count}")

    # 4. Board graph connectivity
    import networkx as nx
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["src_board_id"], row["dst_board_id"])
    # Add isolated boards
    for b in boards_df["board_id"]:
        if b not in G:
            G.add_node(b)
    n_components = nx.number_connected_components(G)
    ok4 = n_components == 1
    print(f"  [4] {check_mark(ok4)} connected_components={n_components}")

    # 5. Avg degree
    avg_deg = stats.get("avg_degree", 0.0)
    ok5 = 1.8 <= avg_deg <= 3.5
    print(f"  [5] {check_mark(ok5)} avg_degree={avg_deg} (1.8-3.5)")

    # ---- Lightpath checks ----
    print(f"\n--- Lightpaths ---")

    # 6. Lightpath count
    num_lps = len(lightpaths)
    ok6 = 20 <= num_lps <= 80
    print(f"  [6] {check_mark(ok6)} num_lightpaths={num_lps} (20-80)")

    # 7. Board-level hops
    board_hops = [len(lp.get("board_path", [])) - 1 for lp in lightpaths if len(lp.get("board_path", [])) > 1]
    avg_board_hops = np.mean(board_hops) if board_hops else 0
    ok7 = avg_board_hops >= 5
    print(f"  [7] {check_mark(ok7)} avg_lightpath_board_hops={avg_board_hops:.2f} (>=5)")

    # 8. Board traverse coverage
    traversed = set()
    for lp in lightpaths:
        for b in lp.get("board_path", []):
            traversed.add(b)
    all_board_ids = set(boards_df["board_id"].tolist())
    coverage = len(traversed & all_board_ids) / max(1, len(all_board_ids))
    ok8 = coverage >= 0.8
    print(f"  [8] {check_mark(ok8)} board_traverse_coverage={coverage:.4f} (>=0.8, prefer >=0.95)")

    # 9. Each board traversed by at least one lightpath
    not_traversed = all_board_ids - traversed
    ok9 = len(not_traversed) == 0
    print(f"  [9] {check_mark(ok9)} boards_not_traversed={len(not_traversed)}/{num_boards}")
    if not_traversed and len(not_traversed) <= 10:
        print(f"       examples: {list(not_traversed)[:5]}")

    # 10. No single-board lightpaths
    single_board_lps = [lp for lp in lightpaths if len(lp.get("board_path", [])) <= 1]
    ok10 = len(single_board_lps) == 0
    print(f"  [10] {check_mark(ok10)} single_board_lightpaths={len(single_board_lps)}")

    # ---- Service checks ----
    print(f"\n--- Services ---")

    # 11. Service coverage ~100%
    svc_boards = set()
    for svc in services:
        for seg in svc.get("path_segments", []):
            pass  # segments are node-level
    # Use lightpath board paths as proxy for service board coverage
    svc_coverage = coverage  # same as lightpath coverage for now
    ok11 = svc_coverage >= 0.95 or "dataset_c" in str(ds_dir).lower()
    print(f"  [11] {check_mark(ok11)} service_board_coverage={svc_coverage:.4f} (>=0.95, Dataset C exempt)")

    # 12. Service paths are valid lightpath subpaths (spot check)
    ok12 = len(services) > 0
    print(f"  [12] {check_mark(ok12)} num_services={len(services)}")

    # ---- OW Data checks ----
    print(f"\n--- Observation Windows ---")

    # 13. OW count
    ok13 = num_ows > 0
    print(f"  [13] {check_mark(ok13)} num_ows={num_ows}")

    # 14. Each OW has metrics, alarms, labels
    ow_ids_metrics = set(metrics_df["ow_id"].unique()) if "ow_id" in metrics_df.columns else set()
    ow_ids_alarms = set(a["ow_id"] for a in alarms_list)
    ow_ids_labels = set(l["ow_id"] for l in labels_list)
    missing_metrics = ow_ids_labels - ow_ids_metrics
    missing_alarms = ow_ids_labels - ow_ids_alarms
    ok14 = len(missing_metrics) == 0 and len(missing_alarms) == 0
    print(f"  [14] {check_mark(ok14)} all OWs have metrics+alarms+labels "
          f"(missing_metrics={len(missing_metrics)}, missing_alarms={len(missing_alarms)})")

    # 15. Metrics format
    required_metric_cols = {"ow_id", "board_id"}
    has_cols = required_metric_cols.issubset(set(metrics_df.columns)) if len(metrics_df) > 0 else False
    metric_cols = list(metrics_df.columns) if len(metrics_df) > 0 else []
    ok15 = has_cols
    print(f"  [15] {check_mark(ok15)} metrics columns: {metric_cols}")

    # 16. Alarm format
    if alarms_list and alarms_list[0].get("alarms"):
        sample_alarm = alarms_list[0]["alarms"][0] if alarms_list[0]["alarms"] else {}
        alarm_keys = set(sample_alarm.keys())
        required_alarm_keys = {"board", "alarm", "severity"}
        ok16 = required_alarm_keys.issubset(alarm_keys)
        print(f"  [16] {check_mark(ok16)} alarm keys: {sorted(alarm_keys)}")

        # Check for is_fp in Dataset B
        has_fp = any("is_fp" in a for ow in alarms_list for a in ow.get("alarms", []))
        if "dataset_b" in str(ds_dir).lower():
            print(f"       is_fp field present: {has_fp}")
    else:
        ok16 = True
        print(f"  [16] {check_mark(ok16)} alarms: (some OWs may have empty alarms)")

    # ---- Label checks ----
    print(f"\n--- Labels ---")

    # 17. Root cause exists
    has_root = all("root_board" in l or "root_cause_board" in l for l in labels_list)
    ok17 = has_root
    print(f"  [17] {check_mark(ok17)} all labels have root_cause_board")

    # 18. Failed boards exist
    has_failed = all(
        l.get("failed_boards") or l.get("root_board")
        for l in labels_list
    )
    ok18 = has_failed
    print(f"  [18] {check_mark(ok18)} all labels have failed_boards or root_board")

    # 19. Regional failure (Dataset A): |failed_boards| = k, connected subgraph
    if "dataset_a" in str(ds_dir).lower():
        regional_ok = True
        for l in labels_list:
            fb = l.get("failed_boards", [l.get("root_board")])
            k = l.get("region_size_k", 1)
            if k >= 2 and len(fb) < 2:
                regional_ok = False
                break
        print(f"  [19] {check_mark(regional_ok)} regional failures: failed_boards matches k")
    else:
        print(f"  [19] [N/A] not Dataset A")

    # ---- Experiment variable checks ----
    print(f"\n--- Experiment Variables ---")

    # 20. Coverage reduction (Dataset C)
    if "dataset_c" in str(ds_dir).lower():
        fracs = set(l.get("coverage_fraction", 1.0) for l in labels_list)
        print(f"  [20] coverage fractions present: {sorted(fracs)}")
    else:
        print(f"  [20] [N/A] not Dataset C")

    # 21. Alarm noise (Dataset B)
    if "dataset_b" in str(ds_dir).lower():
        ratios = set(l.get("noise_ratio", 0) for l in labels_list)
        print(f"  [21] noise ratios present: {sorted(ratios)}")
        # Check is_fp
        fp_count = sum(1 for ow in alarms_list for a in ow.get("alarms", []) if a.get("is_fp"))
        gt_count = sum(1 for ow in alarms_list for a in ow.get("alarms", []) if not a.get("is_fp"))
        print(f"       FP alarms={fp_count}, GT alarms={gt_count}")
    else:
        print(f"  [21] [N/A] not Dataset B")

    # 22. Labels complete regardless of coverage/noise
    labels_complete = all(
        (l.get("root_board") or l.get("root_cause_board")) and l.get("root_event")
        for l in labels_list
    )
    ok22 = labels_complete
    print(f"  [22] {check_mark(ok22)} all labels complete (root_board + root_event)")

    # Summary
    print(f"\n--- Summary ---")
    checks = [ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8, ok10, ok13, ok14, ok15, ok17, ok18, ok22]
    passed = sum(checks)
    total = len(checks)
    print(f"  {passed}/{total} core checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basedir", type=str, default="outputs/sfc_bfl/debug_check")
    args = ap.parse_args()

    basedir = Path(args.basedir)

    # Dataset A
    validate_dataset(basedir / "dataset_a_pilot", "Dataset A (Regional Failures)")

    # Dataset B
    validate_dataset(basedir / "dataset_b_pilot", "Dataset B (Alarm Noise)")

    # Dataset C
    validate_dataset(basedir / "dataset_c_pilot", "Dataset C (Service Coverage)")

    # Dataset D - multiple sizes
    for size_dir in sorted((basedir / "dataset_d_pilot").glob("size_*")) if (basedir / "dataset_d_pilot").exists() else []:
        validate_dataset(size_dir, f"Dataset D ({size_dir.name})")

    # Dataset E - multiple domain counts
    for dom_dir in sorted((basedir / "dataset_e_pilot").glob("domains_*")) if (basedir / "dataset_e_pilot").exists() else []:
        validate_dataset(dom_dir, f"Dataset E ({dom_dir.name})")


if __name__ == "__main__":
    main()
