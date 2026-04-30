#!/usr/bin/env python3
"""Analyze SFC-BFL benchmark results and generate thesis tables & figures.

Generates:
  1. Method comparison table (5×4: datasets × methods)
  2. Per-event-type accuracy breakdown
  3. Dimension sweep tables & curves
  4. Confusion matrices
  5. LaTeX tables + PNG figures

Usage:
  python -m sfc_bfl.analyze_results \
    --results-root outputs/sfc_bfl_results \
    --out-dir outputs/sfc_bfl_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_root: Path) -> list[dict]:
    """Load all results_summary.csv files from results_root."""
    rows = []
    for results_dir in sorted(results_root.iterdir()):
        summary_csv = results_dir / "results_summary.csv"
        if not summary_csv.exists():
            continue
        with open(summary_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_dir"] = results_dir.name
                rows.append(row)
    return rows


def load_per_example_results(results_root: Path) -> dict[str, list[dict]]:
    """Load all results_per_example.csv files, keyed by experiment name."""
    result = {}
    for results_dir in sorted(results_root.iterdir()):
        per_example_csv = results_dir / "results_per_example.csv"
        if not per_example_csv.exists():
            continue
        rows = []
        with open(per_example_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        result[results_dir.name] = rows
    return result


def load_confusion_matrices(results_root: Path) -> dict[str, dict]:
    """Load confusion matrix CSVs."""
    result = {}
    for results_dir in sorted(results_root.iterdir()):
        cm_csv = results_dir / "confusion_matrix.csv"
        if not cm_csv.exists():
            continue
        matrix: dict[str, dict[str, int]] = {}
        with open(cm_csv) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                continue
            labels = header[1:]
            for row_data in reader:
                gt_label = row_data[0]
                matrix[gt_label] = {}
                for i, pred_label in enumerate(labels):
                    val = int(row_data[i + 1]) if i + 1 < len(row_data) else 0
                    matrix[gt_label][pred_label] = val
        result[results_dir.name] = matrix
    return result


# ---------------------------------------------------------------------------
# Parse method/dataset from experiment name
# ---------------------------------------------------------------------------

def parse_experiment_name(name: str) -> tuple[str, str, str | None]:
    """Parse '{method}_{dataset}[_{subdir}]' into (method, dataset, subdir).

    Examples:
      'zero-shot_dataset_a' -> ('zero-shot', 'dataset_a', None)
      'sft_dataset_d_size_200' -> ('sft', 'dataset_d', 'size_200')
      'agent_dataset_e_domains_3' -> ('agent', 'dataset_e', 'domains_3')
    """
    # Known methods
    for method in ["zero-shot", "sft", "grpo", "agent"]:
        prefix = f"{method}_"
        if name.startswith(prefix):
            remainder = name[len(prefix):]
            # Try to match dataset_X pattern
            for ds in ["dataset_a", "dataset_b", "dataset_c", "dataset_d", "dataset_e"]:
                if remainder == ds:
                    return method, ds, None
                if remainder.startswith(ds + "_"):
                    subdir = remainder[len(ds) + 1:]
                    return method, ds, subdir
            # Fallback: treat entire remainder as dataset
            return method, remainder, None
    return "unknown", name, None


# ---------------------------------------------------------------------------
# Table 1: Method comparison across datasets
# ---------------------------------------------------------------------------

def generate_method_comparison_table(
    summaries: list[dict],
    out_dir: Path,
) -> None:
    """Generate the main 5×4 method comparison table."""
    # Organize by (method, dataset) -> metrics
    data: dict[tuple[str, str], dict] = {}
    for row in summaries:
        method, dataset, subdir = parse_experiment_name(row["_dir"])
        if subdir:
            continue  # Skip subdirectory results for main table
        key = (method, dataset)
        data[key] = row

    datasets = ["dataset_a", "dataset_b", "dataset_c", "dataset_d", "dataset_e"]
    methods = ["zero-shot", "sft", "grpo", "agent"]

    # CSV output
    csv_path = out_dir / "table_method_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method"] + [ds.replace("dataset_", "Dataset ").upper() for ds in datasets])

        for metric_name, metric_key in [
            ("Event Acc", "event_accuracy"),
            ("Board Acc", "board_accuracy"),
            ("E2E Score", "e2e_score_mean"),
        ]:
            for method in methods:
                row_vals = [f"{method} ({metric_name})"]
                for ds in datasets:
                    val = data.get((method, ds), {}).get(metric_key, "")
                    if val:
                        row_vals.append(f"{float(val):.1%}")
                    else:
                        row_vals.append("-")
                writer.writerow(row_vals)
            writer.writerow([])  # blank row between metrics

    # LaTeX output
    tex_path = out_dir / "table_method_comparison.tex"
    with open(tex_path, "w") as f:
        ds_headers = " & ".join(ds.replace("dataset_", "").upper() for ds in datasets)
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Method comparison across SFC-BFL datasets}\n")
        f.write("\\label{tab:sfc-bfl-comparison}\n")
        f.write(f"\\begin{{tabular}}{{l{'c' * len(datasets)}}}\n")
        f.write("\\toprule\n")
        f.write(f"Method & {ds_headers} \\\\\n")
        f.write("\\midrule\n")

        for metric_name, metric_key in [
            ("Event Acc", "event_accuracy"),
            ("Board Acc", "board_accuracy"),
            ("E2E Score", "e2e_score_mean"),
        ]:
            f.write(f"\\multicolumn{{{len(datasets)+1}}}{{l}}{{\\textit{{{metric_name}}}}} \\\\\n")
            for method in methods:
                vals = []
                for ds in datasets:
                    val = data.get((method, ds), {}).get(metric_key, "")
                    if val:
                        vals.append(f"{float(val)*100:.1f}\\%")
                    else:
                        vals.append("--")
                f.write(f"\\quad {method} & {' & '.join(vals)} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"[ok] method comparison table -> {csv_path}, {tex_path}")


# ---------------------------------------------------------------------------
# Table 2: Per-event-type accuracy
# ---------------------------------------------------------------------------

def generate_per_event_table(
    per_example: dict[str, list[dict]],
    out_dir: Path,
) -> None:
    """Generate per-event-type accuracy breakdown."""
    EVENT_TYPES = [
        "Fiber_Aging", "Fiber_Crack", "Fiber_Cut",
        "Line_Disconnect",
        "XCON_Port_Down", "XCON_Buffer_Overflow", "XCON_Fabric_Fault",
    ]

    csv_path = out_dir / "table_per_event_accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "Event Type", "N", "Event Acc", "Board Acc"])

        for exp_name, examples in per_example.items():
            for evt in EVENT_TYPES:
                evt_examples = [e for e in examples if e.get("gt_event") == evt]
                if not evt_examples:
                    continue
                n = len(evt_examples)
                event_acc = sum(int(e.get("event_correct", 0)) for e in evt_examples) / n
                board_acc = sum(int(e.get("board_correct", 0)) for e in evt_examples) / n
                writer.writerow([exp_name, evt, n, f"{event_acc:.3f}", f"{board_acc:.3f}"])

    print(f"[ok] per-event table -> {csv_path}")


# ---------------------------------------------------------------------------
# Table 3: Dimension sweep tables
# ---------------------------------------------------------------------------

def generate_dimension_sweep_tables(
    summaries: list[dict],
    out_dir: Path,
) -> None:
    """Generate tables for dataset D (size) and E (domains) sweeps."""
    # Group by (method, dataset, subdir)
    sweep_data: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in summaries:
        method, dataset, subdir = parse_experiment_name(row["_dir"])
        if subdir:
            sweep_data[dataset][method][subdir] = row

    for dataset, method_data in sweep_data.items():
        if not method_data:
            continue

        # Get dimension values (subdir names)
        all_subdirs = set()
        for subdirs_dict in method_data.values():
            all_subdirs.update(subdirs_dict.keys())

        # Sort subdirs numerically
        def sort_key(s):
            # Extract number from "size_200" or "domains_3"
            parts = s.rsplit("_", 1)
            try:
                return int(parts[-1])
            except ValueError:
                return 0

        sorted_subdirs = sorted(all_subdirs, key=sort_key)
        methods = sorted(method_data.keys())

        csv_path = out_dir / f"table_sweep_{dataset}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Method", "Dimension Value", "Event Acc", "Board Acc", "E2E", "N"])

            for method in methods:
                for sd in sorted_subdirs:
                    row = method_data[method].get(sd, {})
                    if not row:
                        continue
                    writer.writerow([
                        method, sd,
                        row.get("event_accuracy", ""),
                        row.get("board_accuracy", ""),
                        row.get("e2e_score_mean", ""),
                        row.get("n_examples", ""),
                    ])

        print(f"[ok] sweep table -> {csv_path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_difficulty_curves(
    summaries: list[dict],
    out_dir: Path,
) -> None:
    """Generate difficulty curve plots (accuracy vs dimension value)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available, skipping plots")
        return

    # Group sweep data
    sweep_data: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in summaries:
        method, dataset, subdir = parse_experiment_name(row["_dir"])
        if not subdir:
            continue
        # Extract numeric dimension value
        parts = subdir.rsplit("_", 1)
        try:
            dim_val = float(parts[-1])
        except ValueError:
            continue
        event_acc = float(row.get("event_accuracy", 0))
        sweep_data[dataset][method].append((dim_val, event_acc))

    if not sweep_data:
        print("[info] no sweep data for difficulty curves")
        return

    for dataset, method_curves in sweep_data.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for method, points in sorted(method_curves.items()):
            points.sort(key=lambda p: p[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=method)

        # Determine x-axis label from dataset
        if "dataset_a" in dataset:
            xlabel = "Region Size (k)"
        elif "dataset_b" in dataset:
            xlabel = "Noise Ratio"
        elif "dataset_c" in dataset:
            xlabel = "Coverage"
        elif "dataset_d" in dataset:
            xlabel = "Board Count"
        elif "dataset_e" in dataset:
            xlabel = "Number of Domains"
        else:
            xlabel = "Dimension Value"

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Event Accuracy")
        ax.set_title(f"Difficulty Curve: {dataset.replace('dataset_', 'Dataset ').upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        fig_path = out_dir / f"difficulty_curve_{dataset}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] difficulty curve -> {fig_path}")


def generate_bar_chart(
    per_example: dict[str, list[dict]],
    out_dir: Path,
) -> None:
    """Generate per-event-type accuracy bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib not available, skipping bar chart")
        return

    EVENT_TYPES = [
        "Fiber_Aging", "Fiber_Crack", "Fiber_Cut",
        "Line_Disconnect",
        "XCON_Port_Down", "XCON_Buffer_Overflow", "XCON_Fabric_Fault",
    ]
    SHORT_NAMES = [e.replace("_", "\n") for e in EVENT_TYPES]

    # Only use non-sweep experiments for the bar chart
    experiments = {}
    for exp_name, examples in per_example.items():
        method, dataset, subdir = parse_experiment_name(exp_name)
        if subdir:
            continue
        if dataset == "dataset_a":  # Use dataset A as reference
            experiments[method] = examples

    if not experiments:
        print("[info] no dataset_a results for bar chart")
        return

    methods = sorted(experiments.keys())
    n_events = len(EVENT_TYPES)
    n_methods = len(methods)
    x = np.arange(n_events)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        examples = experiments[method]
        accs = []
        for evt in EVENT_TYPES:
            evt_ex = [e for e in examples if e.get("gt_event") == evt]
            if evt_ex:
                acc = sum(int(e.get("event_correct", 0)) for e in evt_ex) / len(evt_ex)
            else:
                acc = 0.0
            accs.append(acc)
        ax.bar(x + i * width - (n_methods - 1) * width / 2, accs,
               width, label=method, alpha=0.8)

    ax.set_xlabel("Event Type")
    ax.set_ylabel("Event Accuracy")
    ax.set_title("Per-Event-Type Accuracy (Dataset A)")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    fig_path = out_dir / "bar_per_event_accuracy.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] bar chart -> {fig_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze SFC-BFL benchmark results."
    )
    ap.add_argument("--results-root", type=Path, required=True,
                    help="Root directory with evaluation results")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/sfc_bfl_analysis"),
                    help="Output directory for tables and figures")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading results from {args.results_root}")

    summaries = load_all_results(args.results_root)
    per_example = load_per_example_results(args.results_root)
    confusion = load_confusion_matrices(args.results_root)

    print(f"[info] found {len(summaries)} summary rows, "
          f"{len(per_example)} experiments with per-example data")

    if summaries:
        generate_method_comparison_table(summaries, args.out_dir)
        generate_dimension_sweep_tables(summaries, args.out_dir)
        generate_difficulty_curves(summaries, args.out_dir)

    if per_example:
        generate_per_event_table(per_example, args.out_dir)
        generate_bar_chart(per_example, args.out_dir)

    # Print summary
    if summaries:
        from sfc_bfl.benchmark import print_summary_table
        print_summary_table(args.results_root)

    print(f"\n[ok] analysis complete -> {args.out_dir}")


if __name__ == "__main__":
    main()
