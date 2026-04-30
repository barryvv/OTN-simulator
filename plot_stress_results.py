#!/usr/bin/env python3
"""Generate thesis figures from stress test evaluation results.

Produces 5 key figures:
  1. Degradation bar chart: All 5 methods across scenarios + clean baseline
  2. Multi-failure stacked bar: Both/one/zero roots correct (S2)
  3. Noise sensitivity curve: Accuracy vs noise level (S3)
  4. Tool usage heatmap: Which tools the agent calls per scenario
  5. Multi-failure composite: 4-panel publication figure (S2)

Usage:
  python plot_stress_results.py \
    --results-dir outputs/stress_tests/eval_results \
    --outdir outputs/stress_tests/figures

  # Generate the composite multi-failure figure with placeholder data
  python plot_stress_results.py --placeholder --outdir outputs/stress_tests/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Figure style
# ---------------------------------------------------------------------------

METHODS = ["Rule-Based", "Zero-shot", "SFT+GRPO", "Trained Agent", "Claude Agent"]
METHOD_COLORS = ["#808080", "#E8A87C", "#85DCB0", "#E27D60", "#41B3A3"]
SCENARIOS = ["Clean IID", "S1: 50% Rules", "S2: 2 Failures", "S3: ±40% Noise", "S4: 40% Masked"]

EVENT_TYPES = [
    "Fiber_Aging", "Fiber_Crack", "Fiber_Cut",
    "Line_Disconnect", "XCON_Port_Down",
    "XCON_Buffer_Overflow", "XCON_Fabric_Fault",
]
SHORT_EVENT = {
    "Fiber_Aging": "Fib_Age",
    "Fiber_Crack": "Fib_Crk",
    "Fiber_Cut": "Fib_Cut",
    "Line_Disconnect": "Line_Dis",
    "XCON_Port_Down": "XC_Port",
    "XCON_Buffer_Overflow": "XC_Buf",
    "XCON_Fabric_Fault": "XC_Fab",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary(results_dir: Path, scenario: str, method: str) -> dict[str, float] | None:
    """Load a results_summary.csv for a specific scenario/method combo."""
    import csv

    # Try common naming patterns
    candidates = [
        results_dir / scenario / method / "results_summary.csv",
        results_dir / f"{scenario}_{method}" / "results_summary.csv",
        results_dir / scenario / f"results_summary_{method}.csv",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    return {k: float(v) if v.replace(".", "").replace("-", "").isdigit() else v
                            for k, v in row.items()}
    return None


def load_all_results(results_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load all results into nested dict: scenario -> method -> metrics.

    If actual result files aren't found, returns placeholder data for
    figure generation/testing.
    """
    data: dict[str, dict[str, dict]] = {}

    scenario_dirs = {
        "clean": "clean_iid",
        "S1_50": "S1_partial_rules/50pct",
        "S1_70": "S1_partial_rules/70pct",
        "S2_2f": "S2_multi_failure/2f",
        "S2_3f": "S2_multi_failure/3f",
        "S3": "S3_high_noise",
        "S4": "S4_partial_observability",
    }

    method_dirs = {
        "Rule-Based": "rule_based",
        "Zero-shot": "zero_shot",
        "SFT+GRPO": "sft_grpo",
        "Trained Agent": "trained_agent",
        "Claude Agent": "claude_agent",
    }

    for scenario_key, scenario_path in scenario_dirs.items():
        data[scenario_key] = {}
        for method_name, method_path in method_dirs.items():
            summary = load_summary(results_dir, scenario_path, method_path)
            if summary:
                data[scenario_key][method_name] = summary

    return data


# ---------------------------------------------------------------------------
# Figure 1: Degradation bar chart
# ---------------------------------------------------------------------------

def plot_degradation_bars(
    data: dict[str, dict[str, float]],
    outdir: Path,
) -> None:
    """Bar chart showing event accuracy across scenarios and methods.

    data: scenario_name -> method_name -> event_accuracy
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = ["Clean IID", "S1: 50% Rules", "S2: 2 Failures", "S3: High Noise", "S4: 40% Masked"]
    scenario_keys = ["clean", "S1_50", "S2_2f", "S3", "S4"]

    n_methods = len(METHODS)
    n_scenarios = len(scenarios)
    x = np.arange(n_scenarios)
    bar_width = 0.15
    offsets = np.arange(n_methods) - (n_methods - 1) / 2

    for i, (method, color) in enumerate(zip(METHODS, METHOD_COLORS)):
        values = []
        for sk in scenario_keys:
            if sk in data and method in data[sk]:
                val = data[sk][method].get("event_accuracy", 0.0)
                values.append(val * 100 if val <= 1.0 else val)
            else:
                values.append(0)
        ax.bar(
            x + offsets[i] * bar_width,
            values,
            bar_width * 0.9,
            label=method,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_ylabel("Event Accuracy (%)")
    ax.set_title("RCA Method Accuracy Under Stress Scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "fig1_degradation_bars.pdf")
    fig.savefig(outdir / "fig1_degradation_bars.png")
    plt.close(fig)
    print(f"[ok] fig1_degradation_bars -> {outdir}")


# ---------------------------------------------------------------------------
# Figure 2: Multi-failure stacked bar (S2)
# ---------------------------------------------------------------------------

def plot_multi_failure_stacked(
    data: dict[str, dict[str, float]],
    outdir: Path,
) -> None:
    """Stacked bar showing both/one/zero roots correct for S2."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # For multi-failure, we need per-method:
    # all_roots_identified, partial_root_accuracy, (none correct)
    methods_shown = METHODS
    all_correct = []
    partial_correct = []
    none_correct = []

    for method in methods_shown:
        s2_data = data.get("S2_2f", {}).get(method, {})
        all_roots = s2_data.get("all_roots_identified", 0.0)
        partial = s2_data.get("partial_root_accuracy", 0.0)

        # Normalize: all_roots is fraction where all were found
        # partial - all_roots = fraction where only some were found
        # 1 - partial = fraction where none were found
        all_pct = all_roots * 100
        some_pct = max(0, (partial - all_roots) * 100)
        none_pct = max(0, (1.0 - partial) * 100)

        all_correct.append(all_pct)
        partial_correct.append(some_pct)
        none_correct.append(none_pct)

    x = np.arange(len(methods_shown))

    ax.bar(x, all_correct, 0.6, label="All roots found", color="#41B3A3")
    ax.bar(x, partial_correct, 0.6, bottom=all_correct, label="Some roots found", color="#E8A87C")
    bottom2 = [a + p for a, p in zip(all_correct, partial_correct)]
    ax.bar(x, none_correct, 0.6, bottom=bottom2, label="No roots found", color="#C38D9E")

    ax.set_ylabel("Percentage of Test Cases (%)")
    ax.set_title("Multi-Failure Root Cause Identification (S2: 2 concurrent failures)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_shown, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(outdir / "fig2_multi_failure_stacked.pdf")
    fig.savefig(outdir / "fig2_multi_failure_stacked.png")
    plt.close(fig)
    print(f"[ok] fig2_multi_failure_stacked -> {outdir}")


# ---------------------------------------------------------------------------
# Figure 3: Noise sensitivity curve (S3)
# ---------------------------------------------------------------------------

def plot_noise_sensitivity(
    noise_levels: list[float],
    accuracies: dict[str, list[float]],
    outdir: Path,
) -> None:
    """Line chart showing accuracy vs noise level for each method."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, color in zip(METHODS, METHOD_COLORS):
        if method in accuracies:
            ax.plot(
                noise_levels,
                [a * 100 for a in accuracies[method]],
                "o-",
                color=color,
                label=method,
                linewidth=2,
                markersize=6,
            )

    ax.set_xlabel("Noise Scale (1.0 = ±20%, 2.0 = ±40%, 3.0 = ±60%)")
    ax.set_ylabel("Event Accuracy (%)")
    ax.set_title("Noise Sensitivity: Event Accuracy vs. Noise Level")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(outdir / "fig3_noise_sensitivity.pdf")
    fig.savefig(outdir / "fig3_noise_sensitivity.png")
    plt.close(fig)
    print(f"[ok] fig3_noise_sensitivity -> {outdir}")


# ---------------------------------------------------------------------------
# Figure 4: Tool usage heatmap
# ---------------------------------------------------------------------------

def plot_tool_usage_heatmap(
    tool_counts: dict[str, dict[str, float]],
    outdir: Path,
) -> None:
    """Heatmap showing which tools the agent calls per scenario.

    tool_counts: scenario -> tool_name -> avg_calls_per_example
    """
    tools = ["query_topology", "get_metrics", "lookup_rule", "validate_propagation", "search_similar"]
    scenario_labels = list(tool_counts.keys())

    matrix = np.zeros((len(scenario_labels), len(tools)))
    for i, scenario in enumerate(scenario_labels):
        for j, tool in enumerate(tools):
            matrix[i, j] = tool_counts.get(scenario, {}).get(tool, 0.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(tools)))
    ax.set_yticks(np.arange(len(scenario_labels)))
    ax.set_xticklabels([t.replace("_", "\n") for t in tools], fontsize=9)
    ax.set_yticklabels(scenario_labels, fontsize=10)

    # Add text annotations
    for i in range(len(scenario_labels)):
        for j in range(len(tools)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Avg calls per example")
    ax.set_title("Agent Tool Usage Across Stress Scenarios")

    fig.tight_layout()
    fig.savefig(outdir / "fig4_tool_usage_heatmap.pdf")
    fig.savefig(outdir / "fig4_tool_usage_heatmap.png")
    plt.close(fig)
    print(f"[ok] fig4_tool_usage_heatmap -> {outdir}")


# ---------------------------------------------------------------------------
# Figure 5: Multi-failure composite (4-panel)
# ---------------------------------------------------------------------------

def plot_multi_failure_composite(
    data: dict[str, dict[str, dict[str, float]]],
    outdir: Path,
) -> None:
    """Publication-quality 4-panel multi-failure figure.

    Panels:
      (a) Failure combination heatmap — co-occurrence of event pairs in S2
      (b) Primary root identification accuracy per method
      (c) All/some/none roots stacked bar per method
      (d) 2-failure vs 3-failure accuracy comparison

    Parameters
    ----------
    data : nested dict  scenario_key -> method -> metric -> value
        Must include keys ``S2_2f`` and optionally ``S2_3f``.
    """
    # Publication style
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        try:
            plt.style.use("seaborn-paper")
        except OSError:
            pass
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    fig.suptitle(
        "Multi-Failure Root Cause Analysis (S2)",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    # ── Extract per-method data for 2-failure and 3-failure ──
    s2_2f = data.get("S2_2f", {})
    s2_3f = data.get("S2_3f", {})

    methods_present = [m for m in METHODS if m in s2_2f or m in s2_3f]
    if not methods_present:
        methods_present = METHODS  # fall back to all for layout
    colors = [METHOD_COLORS[METHODS.index(m)] for m in methods_present]

    # ================================================================
    # Panel (a): Failure combination heatmap
    # ================================================================
    ax_a = axes[0, 0]

    # Build co-occurrence matrix from S2_2f + S2_3f data
    n_evt = len(EVENT_TYPES)
    cooccur = np.zeros((n_evt, n_evt), dtype=int)

    # Use actual S2 test data if available, otherwise build from placeholder
    s2_test_path = Path("outputs/stress_tests/S2_multi_failure/test.jsonl")
    if s2_test_path.exists():
        with s2_test_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                events = json.loads(rec.get("ground_truth_root_events", "[]"))
                for i_e, ev_a in enumerate(events):
                    for ev_b in events[i_e + 1 :]:
                        if ev_a in EVENT_TYPES and ev_b in EVENT_TYPES:
                            ia = EVENT_TYPES.index(ev_a)
                            ib = EVENT_TYPES.index(ev_b)
                            cooccur[ia, ib] += 1
                            cooccur[ib, ia] += 1
    else:
        # Synthetic co-occurrence based on typical multi-failure scenarios
        pairs = [
            ("Fiber_Cut", "Line_Disconnect", 3),
            ("Fiber_Cut", "Fiber_Crack", 4),
            ("Fiber_Aging", "Fiber_Cut", 3),
            ("Fiber_Aging", "XCON_Buffer_Overflow", 2),
            ("Fiber_Crack", "Fiber_Aging", 3),
            ("XCON_Port_Down", "Fiber_Aging", 2),
            ("Fiber_Cut", "XCON_Buffer_Overflow", 2),
            ("XCON_Fabric_Fault", "Fiber_Crack", 2),
            ("XCON_Fabric_Fault", "Fiber_Cut", 1),
            ("Line_Disconnect", "Fiber_Crack", 1),
            ("XCON_Buffer_Overflow", "XCON_Port_Down", 1),
            ("Fiber_Crack", "XCON_Port_Down", 1),
        ]
        for ev_a, ev_b, cnt in pairs:
            ia = EVENT_TYPES.index(ev_a)
            ib = EVENT_TYPES.index(ev_b)
            cooccur[ia, ib] += cnt
            cooccur[ib, ia] += cnt

    short_labels = [SHORT_EVENT[e] for e in EVENT_TYPES]

    # Plot heatmap (upper triangle only for cleaner look)
    mask = np.tril(np.ones_like(cooccur, dtype=bool), k=-1)
    display = cooccur.copy().astype(float)
    display[mask] = np.nan

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("white")
    im = ax_a.imshow(display, cmap=cmap, aspect="equal", vmin=0)

    # Annotate cells
    for i in range(n_evt):
        for j in range(i, n_evt):
            val = cooccur[i, j]
            if val > 0:
                color = "white" if val > cooccur.max() * 0.6 else "black"
                ax_a.text(j, i, str(val), ha="center", va="center",
                          fontsize=9, fontweight="bold", color=color)

    ax_a.set_xticks(range(n_evt))
    ax_a.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax_a.set_yticks(range(n_evt))
    ax_a.set_yticklabels(short_labels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label("Co-occurrences", fontsize=9)
    ax_a.set_title("(a) Failure Pair Co-occurrence", fontsize=12, fontweight="bold")

    # ================================================================
    # Panel (b): Primary root identification accuracy per method
    # ================================================================
    ax_b = axes[0, 1]

    primary_acc_2f = []
    primary_acc_3f = []
    for m in methods_present:
        primary_acc_2f.append(s2_2f.get(m, {}).get("primary_root_accuracy", 0.0) * 100)
        primary_acc_3f.append(s2_3f.get(m, {}).get("primary_root_accuracy", 0.0) * 100)

    x_b = np.arange(len(methods_present))
    bar_w = 0.35

    bars_2f = ax_b.bar(x_b - bar_w / 2, primary_acc_2f, bar_w,
                       label="2 Failures", color="#5B9BD5", edgecolor="white", linewidth=0.5)
    bars_3f = ax_b.bar(x_b + bar_w / 2, primary_acc_3f, bar_w,
                       label="3 Failures", color="#ED7D31", edgecolor="white", linewidth=0.5)

    # Value labels
    for bar in bars_2f:
        h = bar.get_height()
        if h > 3:
            ax_b.text(bar.get_x() + bar.get_width() / 2, h + 1,
                      f"{h:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar in bars_3f:
        h = bar.get_height()
        if h > 3:
            ax_b.text(bar.get_x() + bar.get_width() / 2, h + 1,
                      f"{h:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(methods_present, rotation=20, ha="right", fontsize=9)
    ax_b.set_ylabel("Primary Root Accuracy (%)")
    ax_b.set_ylim(0, 109)
    ax_b.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_b.grid(axis="y", alpha=0.3, linestyle="--")
    ax_b.set_axisbelow(True)
    ax_b.set_title("(b) Primary Root Cause Identification", fontsize=12, fontweight="bold")

    # ================================================================
    # Panel (c): All/some/none roots stacked bar (combined 2f+3f)
    # ================================================================
    ax_c = axes[1, 0]

    all_correct = []
    partial_correct = []
    none_correct = []

    for m in methods_present:
        # Merge 2f and 3f data
        all_r_2f = s2_2f.get(m, {}).get("all_roots_identified", 0.0)
        partial_2f = s2_2f.get(m, {}).get("partial_root_accuracy", 0.0)
        all_r_3f = s2_3f.get(m, {}).get("all_roots_identified", 0.0)
        partial_3f = s2_3f.get(m, {}).get("partial_root_accuracy", 0.0)

        # Average across 2f and 3f
        all_r = (all_r_2f + all_r_3f) / 2
        partial = (partial_2f + partial_3f) / 2

        all_pct = all_r * 100
        some_pct = max(0, (partial - all_r) * 100)
        none_pct = max(0, (1.0 - partial) * 100)

        all_correct.append(all_pct)
        partial_correct.append(some_pct)
        none_correct.append(none_pct)

    x_c = np.arange(len(methods_present))
    bar_w_c = 0.55

    p1 = ax_c.bar(x_c, all_correct, bar_w_c,
                  label="All roots found", color="#2E7D32", edgecolor="white", linewidth=0.5)
    p2 = ax_c.bar(x_c, partial_correct, bar_w_c, bottom=all_correct,
                  label="Partial (some roots)", color="#FFA726", edgecolor="white", linewidth=0.5)
    bottom2 = [a + p for a, p in zip(all_correct, partial_correct)]
    p3 = ax_c.bar(x_c, none_correct, bar_w_c, bottom=bottom2,
                  label="No roots found", color="#C62828", edgecolor="white", linewidth=0.5)

    # Annotate percentages inside bars (only if segment is tall enough)
    for i in range(len(methods_present)):
        # All correct segment
        if all_correct[i] > 8:
            ax_c.text(x_c[i], all_correct[i] / 2, f"{all_correct[i]:.0f}%",
                      ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")
        # Partial segment
        if partial_correct[i] > 8:
            y_mid = all_correct[i] + partial_correct[i] / 2
            ax_c.text(x_c[i], y_mid, f"{partial_correct[i]:.0f}%",
                      ha="center", va="center", fontsize=7.5, fontweight="bold", color="black")
        # None segment
        if none_correct[i] > 8:
            y_mid = bottom2[i] + none_correct[i] / 2
            ax_c.text(x_c[i], y_mid, f"{none_correct[i]:.0f}%",
                      ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(methods_present, rotation=20, ha="right", fontsize=9)
    ax_c.set_ylabel("Test Cases (%)")
    ax_c.set_ylim(0, 105)
    ax_c.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_c.set_title("(c) Root Cause Completeness", fontsize=12, fontweight="bold")

    # ================================================================
    # Panel (d): 2-failure vs 3-failure event accuracy comparison
    # ================================================================
    ax_d = axes[1, 1]

    evt_acc_2f = []
    evt_acc_3f = []
    for m in methods_present:
        evt_acc_2f.append(s2_2f.get(m, {}).get("event_accuracy", 0.0) * 100)
        evt_acc_3f.append(s2_3f.get(m, {}).get("event_accuracy", 0.0) * 100)

    x_d = np.arange(len(methods_present))

    # Use markers + lines for a clean paired comparison
    ax_d.plot(x_d, evt_acc_2f, "s-", color="#5B9BD5", label="2 Failures",
              markersize=10, linewidth=2, markeredgecolor="white", markeredgewidth=1.5)
    ax_d.plot(x_d, evt_acc_3f, "D-", color="#ED7D31", label="3 Failures",
              markersize=10, linewidth=2, markeredgecolor="white", markeredgewidth=1.5)

    # Annotate deltas
    for i in range(len(methods_present)):
        delta = evt_acc_2f[i] - evt_acc_3f[i]
        if abs(delta) > 0.5:
            mid_y = (evt_acc_2f[i] + evt_acc_3f[i]) / 2
            ax_d.annotate(
                f"{delta:+.0f}",
                xy=(x_d[i] + 0.15, mid_y),
                fontsize=7,
                color="#555555",
                fontstyle="italic",
            )

    # Fill between to show degradation
    ax_d.fill_between(x_d, evt_acc_2f, evt_acc_3f, alpha=0.12, color="#888888")

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(methods_present, rotation=20, ha="right", fontsize=9)
    ax_d.set_ylabel("Event Accuracy (%)")
    ax_d.set_ylim(0, 109)
    ax_d.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_d.grid(axis="y", alpha=0.3, linestyle="--")
    ax_d.set_axisbelow(True)
    ax_d.set_title("(d) Complexity Degradation (2 vs 3 Failures)", fontsize=12, fontweight="bold")

    # ── Final layout ──
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.35, wspace=0.28)

    for fmt in ("png", "pdf"):
        out_path = outdir / f"fig_multi_failure_composite.{fmt}"
        fig.savefig(out_path, dpi=300)
        print(f"[ok] wrote {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate thesis figures from stress test results."
    )
    ap.add_argument(
        "--results-dir", type=Path,
        default=Path("outputs/stress_tests/eval_results"),
        help="Directory containing evaluation result CSVs",
    )
    ap.add_argument(
        "--outdir", type=Path,
        default=Path("outputs/stress_tests/figures"),
        help="Output directory for figures",
    )
    ap.add_argument(
        "--placeholder", action="store_true",
        help="Use placeholder data for testing figure generation",
    )

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.placeholder:
        # Generate figures with expected/placeholder data for testing layout
        print("[info] using placeholder data for figure generation")

        # Fig 1: Degradation bars
        data = {
            "clean": {
                "Rule-Based": {"event_accuracy": 0.986},
                "Zero-shot": {"event_accuracy": 0.571},
                "SFT+GRPO": {"event_accuracy": 0.592},
                "Trained Agent": {"event_accuracy": 0.80},
                "Claude Agent": {"event_accuracy": 0.90},
            },
            "S1_50": {
                "Rule-Based": {"event_accuracy": 0.65},
                "Zero-shot": {"event_accuracy": 0.50},
                "SFT+GRPO": {"event_accuracy": 0.55},
                "Trained Agent": {"event_accuracy": 0.75},
                "Claude Agent": {"event_accuracy": 0.85},
            },
            "S2_2f": {
                "Rule-Based": {"event_accuracy": 0.0, "all_roots_identified": 0.0, "partial_root_accuracy": 0.15},
                "Zero-shot": {"event_accuracy": 0.05, "all_roots_identified": 0.02, "partial_root_accuracy": 0.20},
                "SFT+GRPO": {"event_accuracy": 0.10, "all_roots_identified": 0.05, "partial_root_accuracy": 0.25},
                "Trained Agent": {"event_accuracy": 0.40, "all_roots_identified": 0.20, "partial_root_accuracy": 0.55},
                "Claude Agent": {"event_accuracy": 0.60, "all_roots_identified": 0.35, "partial_root_accuracy": 0.70},
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

        # Add S2_3f placeholder data for composite figure
        data["S2_3f"] = {
            "Rule-Based": {
                "event_accuracy": 0.0,
                "primary_root_accuracy": 0.10,
                "all_roots_identified": 0.0,
                "partial_root_accuracy": 0.10,
            },
            "Zero-shot": {
                "event_accuracy": 0.03,
                "primary_root_accuracy": 0.15,
                "all_roots_identified": 0.0,
                "partial_root_accuracy": 0.12,
            },
            "SFT+GRPO": {
                "event_accuracy": 0.07,
                "primary_root_accuracy": 0.20,
                "all_roots_identified": 0.02,
                "partial_root_accuracy": 0.18,
            },
            "Trained Agent": {
                "event_accuracy": 0.27,
                "primary_root_accuracy": 0.45,
                "all_roots_identified": 0.10,
                "partial_root_accuracy": 0.40,
            },
            "Claude Agent": {
                "event_accuracy": 0.45,
                "primary_root_accuracy": 0.70,
                "all_roots_identified": 0.20,
                "partial_root_accuracy": 0.55,
            },
        }
        # Add primary_root_accuracy to S2_2f
        data["S2_2f"]["Rule-Based"]["primary_root_accuracy"] = 0.15
        data["S2_2f"]["Zero-shot"]["primary_root_accuracy"] = 0.25
        data["S2_2f"]["SFT+GRPO"]["primary_root_accuracy"] = 0.30
        data["S2_2f"]["Trained Agent"]["primary_root_accuracy"] = 0.60
        data["S2_2f"]["Claude Agent"]["primary_root_accuracy"] = 0.80

        plot_degradation_bars(data, args.outdir)
        plot_multi_failure_stacked(data, args.outdir)
        plot_multi_failure_composite(data, args.outdir)

        # Fig 3: Noise sensitivity
        noise_levels = [1.0, 1.5, 2.0, 2.5, 3.0]
        accuracies = {
            "Rule-Based":    [0.986, 0.80, 0.55, 0.35, 0.20],
            "Zero-shot":     [0.571, 0.50, 0.40, 0.30, 0.20],
            "SFT+GRPO":      [0.592, 0.52, 0.45, 0.35, 0.25],
            "Trained Agent": [0.80, 0.75, 0.65, 0.55, 0.45],
            "Claude Agent":  [0.90, 0.87, 0.80, 0.72, 0.65],
        }
        plot_noise_sensitivity(noise_levels, accuracies, args.outdir)

        # Fig 4: Tool usage heatmap
        tool_counts = {
            "Clean IID": {"query_topology": 1.2, "get_metrics": 2.0, "lookup_rule": 1.5, "validate_propagation": 1.0, "search_similar": 0.5},
            "S1: 50% Rules": {"query_topology": 1.5, "get_metrics": 2.5, "lookup_rule": 2.0, "validate_propagation": 1.8, "search_similar": 1.2},
            "S2: 2 Failures": {"query_topology": 2.5, "get_metrics": 3.5, "lookup_rule": 2.0, "validate_propagation": 2.5, "search_similar": 1.0},
            "S3: High Noise": {"query_topology": 1.5, "get_metrics": 3.0, "lookup_rule": 1.5, "validate_propagation": 1.5, "search_similar": 2.0},
            "S4: 40% Masked": {"query_topology": 3.0, "get_metrics": 4.0, "lookup_rule": 1.5, "validate_propagation": 1.5, "search_similar": 1.5},
        }
        plot_tool_usage_heatmap(tool_counts, args.outdir)

    else:
        # Load actual results
        data = load_all_results(args.results_dir)
        if not any(data.values()):
            print("[warn] no result files found. Use --placeholder for testing, "
                  "or ensure results are in the expected directory structure.")
            return

        plot_degradation_bars(data, args.outdir)
        plot_multi_failure_stacked(data, args.outdir)

        # TODO: noise sensitivity requires results at multiple noise levels
        # TODO: tool usage requires parsing agent trace logs

    print(f"\n[done] all figures saved to {args.outdir}")


if __name__ == "__main__":
    main()
