#!/usr/bin/env python3
"""Generate LaTeX tables from stress test evaluation results.

Produces 3 LaTeX tables for the thesis:
  1. Main comparison: Scenario x 5 Methods x Key Metrics
  2. Multi-failure detailed breakdown (S2)
  3. Per-event accuracy under noise (S3)

Usage:
  python generate_stress_tables.py \
    --results-dir outputs/stress_tests/eval_results \
    --outdir outputs/stress_tests/tables

  # Use placeholder data for testing layout
  python generate_stress_tables.py --placeholder --outdir outputs/stress_tests/tables
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Table 1: Main comparison
# ---------------------------------------------------------------------------

def generate_main_comparison_table(
    data: dict[str, dict[str, dict[str, float]]],
    outdir: Path,
) -> None:
    """Generate LaTeX table: Scenario x Method x Key Metrics.

    data: scenario -> method -> {event_accuracy, board_f1_mean, alarm_f1_mean, e2e_score_mean}
    """
    methods = ["Rule-Based", "Zero-shot", "SFT+GRPO", "Trained Agent", "Claude Agent"]
    scenarios = [
        ("Clean IID", "clean"),
        ("S1: 70\\% Rules", "S1_70"),
        ("S1: 50\\% Rules", "S1_50"),
        ("S2: 2 Failures", "S2_2f"),
        ("S2: 3 Failures", "S2_3f"),
        ("S3: $\\pm$40\\% Noise", "S3"),
        ("S4: 40\\% Masked", "S4"),
    ]

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Event accuracy (\\%) across stress-test scenarios and RCA methods. "
                  "\\textbf{Bold} indicates best self-hosted method per scenario.}")
    lines.append("\\label{tab:stress_comparison}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "c" * len(methods) + "}")
    lines.append("\\toprule")

    # Header
    header = "Scenario"
    for m in methods:
        header += f" & {m}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows
    for scenario_label, scenario_key in scenarios:
        row = scenario_label
        values = []
        for method in methods:
            val = data.get(scenario_key, {}).get(method, {}).get("event_accuracy", 0.0)
            val_pct = val * 100 if val <= 1.0 else val
            values.append(val_pct)

        # Find best self-hosted (exclude Claude Agent = index 4)
        self_hosted_vals = values[:4]
        best_idx = max(range(len(self_hosted_vals)), key=lambda i: self_hosted_vals[i]) if self_hosted_vals else -1

        for i, val in enumerate(values):
            if i == best_idx and val > 0:
                row += f" & \\textbf{{{val:.1f}}}"
            else:
                row += f" & {val:.1f}"
        row += " \\\\"

        # Add horizontal line between scenario groups
        if scenario_key in ("clean", "S1_50", "S2_3f", "S3"):
            row += "\n\\midrule" if scenario_key != "S3" else ""

        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    out_path = outdir / "table1_main_comparison.tex"
    out_path.write_text(tex, encoding="utf-8")
    print(f"[ok] table1_main_comparison -> {out_path}")


# ---------------------------------------------------------------------------
# Table 2: Multi-failure breakdown (S2)
# ---------------------------------------------------------------------------

def generate_multi_failure_table(
    data: dict[str, dict[str, float]],
    outdir: Path,
) -> None:
    """Generate LaTeX table: Multi-failure metrics per method.

    data: method -> {all_roots_identified, partial_root_accuracy, event_accuracy, ...}
    """
    methods = ["Rule-Based", "Zero-shot", "SFT+GRPO", "Trained Agent", "Claude Agent"]
    metrics = [
        ("Event Acc. (\\%)", "event_accuracy"),
        ("All Roots Found (\\%)", "all_roots_identified"),
        ("Partial Root Acc. (\\%)", "partial_root_accuracy"),
        ("Board F1 (\\%)", "board_f1_mean"),
        ("Alarm F1 (\\%)", "alarm_f1_mean"),
    ]

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Multi-failure scenario (S2: 2 concurrent failures) detailed metrics. "
                  "All Roots Found: fraction of test cases where every ground truth root cause was identified.}")
    lines.append("\\label{tab:multi_failure}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "c" * len(methods) + "}")
    lines.append("\\toprule")

    # Header
    header = "Metric"
    for m in methods:
        header += f" & {m}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for metric_label, metric_key in metrics:
        row = metric_label
        for method in methods:
            val = data.get(method, {}).get(metric_key, 0.0)
            val_pct = val * 100 if val <= 1.0 else val
            row += f" & {val_pct:.1f}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    out_path = outdir / "table2_multi_failure.tex"
    out_path.write_text(tex, encoding="utf-8")
    print(f"[ok] table2_multi_failure -> {out_path}")


# ---------------------------------------------------------------------------
# Table 3: Per-event accuracy under noise (S3)
# ---------------------------------------------------------------------------

def generate_noise_per_event_table(
    data: dict[str, dict[str, float]],
    outdir: Path,
) -> None:
    """Generate LaTeX table: Per-event accuracy under high noise.

    data: method -> {event_accuracy_Fiber_Cut, event_accuracy_XCON_Port_Down, ...}
    """
    methods = ["Rule-Based", "Trained Agent", "Claude Agent"]
    events = [
        "Fiber_Aging", "Fiber_Crack", "Fiber_Cut",
        "XCON_Port_Down", "XCON_Buffer_Overflow", "XCON_Fabric_Fault",
        "Line_Disconnect",
    ]

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Per-event event accuracy (\\%) under high noise (S3: $\\pm$40\\%). "
                  "Events with overlapping BBE ranges under noise show the largest rule-based degradation.}")
    lines.append("\\label{tab:noise_per_event}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "c" * len(methods) + "}")
    lines.append("\\toprule")

    header = "Event Type"
    for m in methods:
        header += f" & {m}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for event in events:
        # Use underscores for display
        display_name = event.replace("_", "\\_")
        row = display_name
        for method in methods:
            key = f"event_accuracy_{event}"
            val = data.get(method, {}).get(key, 0.0)
            val_pct = val * 100 if val <= 1.0 else val
            row += f" & {val_pct:.1f}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    out_path = outdir / "table3_noise_per_event.tex"
    out_path.write_text(tex, encoding="utf-8")
    print(f"[ok] table3_noise_per_event -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate LaTeX tables from stress test results."
    )
    ap.add_argument(
        "--results-dir", type=Path,
        default=Path("outputs/stress_tests/eval_results"),
        help="Directory containing evaluation result CSVs",
    )
    ap.add_argument(
        "--outdir", type=Path,
        default=Path("outputs/stress_tests/tables"),
        help="Output directory for LaTeX files",
    )
    ap.add_argument(
        "--placeholder", action="store_true",
        help="Use placeholder data for testing table generation",
    )

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.placeholder:
        print("[info] using placeholder data for table generation")

        # Table 1: Main comparison
        main_data = {
            "clean": {
                "Rule-Based": {"event_accuracy": 0.986},
                "Zero-shot": {"event_accuracy": 0.571},
                "SFT+GRPO": {"event_accuracy": 0.592},
                "Trained Agent": {"event_accuracy": 0.80},
                "Claude Agent": {"event_accuracy": 0.90},
            },
            "S1_70": {
                "Rule-Based": {"event_accuracy": 0.80},
                "Zero-shot": {"event_accuracy": 0.55},
                "SFT+GRPO": {"event_accuracy": 0.58},
                "Trained Agent": {"event_accuracy": 0.78},
                "Claude Agent": {"event_accuracy": 0.88},
            },
            "S1_50": {
                "Rule-Based": {"event_accuracy": 0.65},
                "Zero-shot": {"event_accuracy": 0.50},
                "SFT+GRPO": {"event_accuracy": 0.55},
                "Trained Agent": {"event_accuracy": 0.75},
                "Claude Agent": {"event_accuracy": 0.85},
            },
            "S2_2f": {
                "Rule-Based": {"event_accuracy": 0.0},
                "Zero-shot": {"event_accuracy": 0.05},
                "SFT+GRPO": {"event_accuracy": 0.10},
                "Trained Agent": {"event_accuracy": 0.40},
                "Claude Agent": {"event_accuracy": 0.60},
            },
            "S2_3f": {
                "Rule-Based": {"event_accuracy": 0.0},
                "Zero-shot": {"event_accuracy": 0.03},
                "SFT+GRPO": {"event_accuracy": 0.05},
                "Trained Agent": {"event_accuracy": 0.25},
                "Claude Agent": {"event_accuracy": 0.45},
            },
            "S3": {
                "Rule-Based": {"event_accuracy": 0.55},
                "Zero-shot": {"event_accuracy": 0.40},
                "SFT+GRPO": {"event_accuracy": 0.45},
                "Trained Agent": {"event_accuracy": 0.65},
                "Claude Agent": {"event_accuracy": 0.80},
            },
            "S4": {
                "Rule-Based": {"event_accuracy": 0.05},
                "Zero-shot": {"event_accuracy": 0.15},
                "SFT+GRPO": {"event_accuracy": 0.20},
                "Trained Agent": {"event_accuracy": 0.48},
                "Claude Agent": {"event_accuracy": 0.60},
            },
        }
        generate_main_comparison_table(main_data, args.outdir)

        # Table 2: Multi-failure
        mf_data = {
            "Rule-Based": {"event_accuracy": 0.0, "all_roots_identified": 0.0, "partial_root_accuracy": 0.15, "board_f1_mean": 0.10, "alarm_f1_mean": 0.05},
            "Zero-shot": {"event_accuracy": 0.05, "all_roots_identified": 0.02, "partial_root_accuracy": 0.20, "board_f1_mean": 0.15, "alarm_f1_mean": 0.10},
            "SFT+GRPO": {"event_accuracy": 0.10, "all_roots_identified": 0.05, "partial_root_accuracy": 0.25, "board_f1_mean": 0.20, "alarm_f1_mean": 0.15},
            "Trained Agent": {"event_accuracy": 0.40, "all_roots_identified": 0.20, "partial_root_accuracy": 0.55, "board_f1_mean": 0.45, "alarm_f1_mean": 0.35},
            "Claude Agent": {"event_accuracy": 0.60, "all_roots_identified": 0.35, "partial_root_accuracy": 0.70, "board_f1_mean": 0.60, "alarm_f1_mean": 0.50},
        }
        generate_multi_failure_table(mf_data, args.outdir)

        # Table 3: Per-event noise accuracy
        noise_data = {
            "Rule-Based": {
                "event_accuracy_Fiber_Aging": 0.30, "event_accuracy_Fiber_Crack": 0.50,
                "event_accuracy_Fiber_Cut": 0.80, "event_accuracy_XCON_Port_Down": 0.40,
                "event_accuracy_XCON_Buffer_Overflow": 0.45, "event_accuracy_XCON_Fabric_Fault": 0.35,
                "event_accuracy_Line_Disconnect": 0.70,
            },
            "Trained Agent": {
                "event_accuracy_Fiber_Aging": 0.55, "event_accuracy_Fiber_Crack": 0.65,
                "event_accuracy_Fiber_Cut": 0.85, "event_accuracy_XCON_Port_Down": 0.60,
                "event_accuracy_XCON_Buffer_Overflow": 0.55, "event_accuracy_XCON_Fabric_Fault": 0.50,
                "event_accuracy_Line_Disconnect": 0.80,
            },
            "Claude Agent": {
                "event_accuracy_Fiber_Aging": 0.75, "event_accuracy_Fiber_Crack": 0.80,
                "event_accuracy_Fiber_Cut": 0.95, "event_accuracy_XCON_Port_Down": 0.75,
                "event_accuracy_XCON_Buffer_Overflow": 0.70, "event_accuracy_XCON_Fabric_Fault": 0.65,
                "event_accuracy_Line_Disconnect": 0.90,
            },
        }
        generate_noise_per_event_table(noise_data, args.outdir)

    else:
        print("[warn] loading actual results not yet implemented. Use --placeholder for now.")

    print(f"\n[done] all tables saved to {args.outdir}")


if __name__ == "__main__":
    main()
