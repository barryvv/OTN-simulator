#!/usr/bin/env python3
"""Generate publication-quality figures from OTN RCA evaluation outputs.

Usage:
  python plot_results.py --results-dir outputs/eval/ --outdir outputs/figures/

Optional:
  --agent-output outputs/eval/agent_trace.json   (for agent reasoning trace)
  --training-log outputs/training_log.json        (for training curves)
  --baselines rule-based:outputs/eval/rule_based zero-shot:outputs/eval/zero_shot ...

Figures produced (when data is available):
  1. confusion_matrix.png        - 7x7 event type confusion heatmap
  2. baseline_comparison.png     - grouped bar chart of all metrics across baselines
  3. category_accuracy.png       - per-category accuracy across baselines
  4. agent_trace.png             - visual flowchart of Think/Act/Observe cycles
  5. training_curves.png         - reward and loss over training steps
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ─────────────────────────────────────────────────────────────
# Style setup
# ─────────────────────────────────────────────────────────────

def _setup_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        try:
            plt.style.use("seaborn-paper")
        except OSError:
            pass  # Fall back to default
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
    })


EVENT_TYPES = [
    "Fiber_Aging",
    "Fiber_Crack",
    "Fiber_Cut",
    "Line_Disconnect",
    "XCON_Port_Down",
    "XCON_Buffer_Overflow",
    "XCON_Fabric_Fault",
]

SHORT_EVENT_NAMES = {
    "Fiber_Aging": "Fib_Age",
    "Fiber_Crack": "Fib_Crk",
    "Fiber_Cut": "Fib_Cut",
    "Line_Disconnect": "Line_Dis",
    "XCON_Port_Down": "XC_Port",
    "XCON_Buffer_Overflow": "XC_Buf",
    "XCON_Fabric_Fault": "XC_Fab",
    "(none)": "(none)",
}

# Baseline display config
BASELINE_COLORS = {
    "Rule-based": "#1f77b4",
    "Zero-shot": "#ff7f0e",
    "SFT": "#2ca02c",
    "SFT+GRPO": "#d62728",
    "Agent": "#9467bd",
}

BASELINE_ORDER = ["Rule-based", "Zero-shot", "SFT", "SFT+GRPO", "Agent"]

METRIC_DISPLAY = {
    "event_accuracy": "Event Acc",
    "board_accuracy": "Board Acc",
    "category_accuracy": "Cat Acc",
    "board_f1_mean": "Board F1",
    "alarm_f1_mean": "Alarm F1",
    "format_score_mean": "Format",
    "e2e_score_mean": "E2E",
}


# ─────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────

def _load_confusion_matrix(path: Path) -> tuple[list[str], np.ndarray] | None:
    """Load confusion_matrix.csv and return (labels, matrix).

    CSV format from evaluate.py:
      gt \\ pred, label1, label2, ...
      label1,     count,  count, ...
    """
    if not path.exists():
        print(f"[skip] confusion matrix not found: {path}")
        return None

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        return None

    header = rows[0][1:]  # skip "gt \\ pred" column
    labels = []
    matrix_rows = []
    for row in rows[1:]:
        if not row:
            continue
        labels.append(row[0])
        vals = [int(v) if v.strip() else 0 for v in row[1:]]
        # Pad or truncate to match header length
        while len(vals) < len(header):
            vals.append(0)
        matrix_rows.append(vals[: len(header)])

    # Filter to only EVENT_TYPES (skip "(none)" rows/cols for cleaner display)
    # but keep them if they have nonzero counts
    active_labels = []
    for lbl in labels:
        if lbl in EVENT_TYPES:
            active_labels.append(lbl)
        elif lbl == "(none)":
            idx = labels.index(lbl)
            if sum(matrix_rows[idx]) > 0:
                active_labels.append(lbl)

    # Also check pred columns for (none)
    if "(none)" in header and "(none)" not in active_labels:
        col_idx = header.index("(none)")
        col_sum = sum(matrix_rows[labels.index(lbl)][col_idx]
                      for lbl in active_labels if lbl in labels)
        if col_sum > 0:
            active_labels.append("(none)")

    # Build filtered matrix
    filtered = np.zeros((len(active_labels), len(active_labels)), dtype=int)
    for i, gt_lbl in enumerate(active_labels):
        if gt_lbl not in labels:
            continue
        gt_idx = labels.index(gt_lbl)
        for j, pred_lbl in enumerate(active_labels):
            if pred_lbl not in header:
                continue
            pred_idx = header.index(pred_lbl)
            if pred_idx < len(matrix_rows[gt_idx]):
                filtered[i, j] = matrix_rows[gt_idx][pred_idx]

    return active_labels, filtered


def _load_summary_csv(path: Path) -> list[dict[str, Any]]:
    """Load results_summary.csv (may contain multiple rows from different models)."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_per_example_csv(path: Path) -> list[dict[str, Any]]:
    """Load results_per_example.csv."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_baselines(
    results_dir: Path,
    baseline_args: list[str] | None,
) -> dict[str, dict[str, float]]:
    """Load baseline results from multiple directories.

    Args:
        results_dir: default results directory
        baseline_args: list of "name:path" strings, e.g. ["Rule-based:outputs/eval/rule_based"]

    Returns:
        dict mapping baseline name -> metric name -> value
    """
    baselines: dict[str, dict[str, float]] = {}

    if baseline_args:
        for spec in baseline_args:
            if ":" in spec:
                name, path_str = spec.split(":", 1)
            else:
                name = Path(spec).stem
                path_str = spec
            summary_path = Path(path_str) / "results_summary.csv"
            rows = _load_summary_csv(summary_path)
            if rows:
                # Take the last row (most recent run)
                row = rows[-1]
                baselines[name] = {
                    k: float(row.get(k, 0)) for k in METRIC_DISPLAY.keys()
                }
    else:
        # Try loading from results_summary.csv which may have multiple model rows
        summary_path = results_dir / "results_summary.csv"
        rows = _load_summary_csv(summary_path)
        for row in rows:
            name = row.get("model_name", "model")
            baselines[name] = {
                k: float(row.get(k, 0)) for k in METRIC_DISPLAY.keys()
            }

    return baselines


# ─────────────────────────────────────────────────────────────
# Figure 1: Confusion Matrix
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(results_dir: Path, outdir: Path) -> None:
    """Generate confusion matrix heatmap from confusion_matrix.csv."""
    result = _load_confusion_matrix(results_dir / "confusion_matrix.csv")
    if result is None:
        return

    labels, matrix = result
    n = len(labels)
    if n == 0:
        return

    short_labels = [SHORT_EVENT_NAMES.get(lbl, lbl) for lbl in labels]

    # Compute percentages (row-normalized)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    pct_matrix = matrix / row_sums * 100

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Heatmap using imshow
    cmap = plt.cm.Blues
    im = ax.imshow(pct_matrix, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=100, aspect="equal")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-Normalized (%)")

    # Annotate cells with count and percentage
    thresh = pct_matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            count = matrix[i, j]
            pct = pct_matrix[i, j]
            color = "white" if pct > thresh else "black"
            if count > 0:
                text = f"{count}\n({pct:.0f}%)"
            else:
                text = "0"
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=8, fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=9)

    ax.set_xlabel("Predicted Event")
    ax.set_ylabel("Ground Truth Event")
    ax.set_title("Event Type Prediction Confusion Matrix")

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        out_path = outdir / f"confusion_matrix.{fmt}"
        fig.savefig(out_path, dpi=300)
        print(f"[ok] wrote {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Figure 2: Baseline Comparison Bar Chart
# ─────────────────────────────────────────────────────────────

def plot_baseline_comparison(
    results_dir: Path,
    outdir: Path,
    baseline_args: list[str] | None,
) -> None:
    """Generate grouped bar chart comparing baselines across metrics."""
    baselines = _load_baselines(results_dir, baseline_args)
    if not baselines:
        print("[skip] no baseline data found for comparison chart")
        return

    # Order baselines by BASELINE_ORDER if possible, otherwise alphabetical
    ordered_names = []
    for name in BASELINE_ORDER:
        if name in baselines:
            ordered_names.append(name)
    for name in sorted(baselines.keys()):
        if name not in ordered_names:
            ordered_names.append(name)

    metrics = list(METRIC_DISPLAY.keys())
    metric_labels = [METRIC_DISPLAY[m] for m in metrics]
    n_metrics = len(metrics)
    n_baselines = len(ordered_names)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(n_metrics)
    total_width = 0.75
    bar_width = total_width / n_baselines

    for i, name in enumerate(ordered_names):
        values = [baselines[name].get(m, 0) * 100 for m in metrics]
        offset = (i - n_baselines / 2 + 0.5) * bar_width
        color = BASELINE_COLORS.get(name, f"C{i}")
        bars = ax.bar(x + offset, values, bar_width, label=name, color=color,
                      edgecolor="white", linewidth=0.5)
        # Value labels on bars
        for bar, val in zip(bars, values):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7,
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score (%)")
    ax.set_title("RCA Performance Comparison Across Methods")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        out_path = outdir / f"baseline_comparison.{fmt}"
        fig.savefig(out_path, dpi=300)
        print(f"[ok] wrote {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Figure 3: Per-Category Accuracy
# ─────────────────────────────────────────────────────────────

def plot_category_accuracy(
    results_dir: Path,
    outdir: Path,
    baseline_args: list[str] | None,
) -> None:
    """Generate per-category accuracy chart (Fiber vs XCON vs Line)."""
    categories = ["Fiber", "XCON", "Line"]

    # Load baselines with category-level metrics
    baselines_full: dict[str, dict[str, float]] = {}

    if baseline_args:
        for spec in baseline_args:
            if ":" in spec:
                name, path_str = spec.split(":", 1)
            else:
                name = Path(spec).stem
                path_str = spec
            summary_path = Path(path_str) / "results_summary.csv"
            rows = _load_summary_csv(summary_path)
            if rows:
                row = rows[-1]
                baselines_full[name] = {}
                for cat in categories:
                    key = f"event_accuracy_{cat}"
                    baselines_full[name][cat] = float(row.get(key, 0))
    else:
        summary_path = results_dir / "results_summary.csv"
        rows = _load_summary_csv(summary_path)
        for row in rows:
            name = row.get("model_name", "model")
            baselines_full[name] = {}
            for cat in categories:
                key = f"event_accuracy_{cat}"
                baselines_full[name][cat] = float(row.get(key, 0))

    if not baselines_full:
        print("[skip] no per-category data found")
        return

    ordered_names = []
    for name in BASELINE_ORDER:
        if name in baselines_full:
            ordered_names.append(name)
    for name in sorted(baselines_full.keys()):
        if name not in ordered_names:
            ordered_names.append(name)

    n_cats = len(categories)
    n_baselines = len(ordered_names)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(n_cats)
    total_width = 0.7
    bar_width = total_width / n_baselines

    for i, name in enumerate(ordered_names):
        values = [baselines_full[name].get(cat, 0) * 100 for cat in categories]
        offset = (i - n_baselines / 2 + 0.5) * bar_width
        color = BASELINE_COLORS.get(name, f"C{i}")
        bars = ax.bar(x + offset, values, bar_width, label=name, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Event Accuracy (%)")
    ax.set_title("Per-Category Event Accuracy")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        out_path = outdir / f"category_accuracy.{fmt}"
        fig.savefig(out_path, dpi=300)
        print(f"[ok] wrote {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Figure 4: Agent Reasoning Trace
# ─────────────────────────────────────────────────────────────

def plot_agent_trace(agent_output_path: Path | None, outdir: Path) -> None:
    """Generate visual flowchart of a ReAct agent reasoning trace.

    Expected JSON format:
    {
        "steps": [
            {"type": "Think", "content": "..."},
            {"type": "Act", "tool": "read_metrics", "args": "..."},
            {"type": "Observe", "content": "..."},
            ...
        ],
        "final_answer": "..."
    }
    """
    if agent_output_path is None or not agent_output_path.exists():
        print("[skip] agent trace not provided or not found")
        return

    try:
        with agent_output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[skip] could not parse agent output: {e}")
        return

    steps = data.get("steps", [])
    final_answer = data.get("final_answer", "")

    if not steps:
        print("[skip] no steps found in agent trace")
        return

    # Layout parameters
    box_width = 0.75
    box_height = 0.08
    gap = 0.03
    left_margin = 0.125

    type_colors = {
        "Think": "#E8F4FD",
        "Act": "#FFF3E0",
        "Observe": "#E8F5E9",
        "Answer": "#FCE4EC",
    }
    type_edge_colors = {
        "Think": "#1565C0",
        "Act": "#E65100",
        "Observe": "#2E7D32",
        "Answer": "#C62828",
    }

    # Add final answer as a step
    all_steps = list(steps)
    if final_answer:
        all_steps.append({"type": "Answer", "content": final_answer})

    n_steps = len(all_steps)
    fig_height = max(4, n_steps * (box_height + gap) * 12 + 2)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y_start = 0.95
    y_step = (box_height + gap)

    for idx, step in enumerate(all_steps):
        stype = step.get("type", "Think")
        content = step.get("content", "")
        tool = step.get("tool", "")

        if stype == "Act" and tool:
            label = f"Act: {tool}"
            args = step.get("args", "")
            if args:
                label += f"({args[:40]}{'...' if len(str(args)) > 40 else ''})"
        else:
            label = f"{stype}: {content[:60]}{'...' if len(content) > 60 else ''}"

        y = y_start - idx * y_step
        color = type_colors.get(stype, "#F5F5F5")
        edge_color = type_edge_colors.get(stype, "#424242")

        # Draw rounded box
        bbox = FancyBboxPatch(
            (left_margin, y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.01",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1.5,
            transform=ax.transAxes,
        )
        ax.add_patch(bbox)

        # Add text
        ax.text(
            left_margin + box_width / 2,
            y,
            label,
            ha="center", va="center",
            fontsize=8, fontfamily="monospace",
            transform=ax.transAxes,
            clip_on=True,
        )

        # Draw arrow to next step
        if idx < n_steps - 1:
            arrow_y_start = y - box_height / 2
            arrow_y_end = y - y_step + box_height / 2
            ax.annotate(
                "",
                xy=(0.5, arrow_y_end),
                xytext=(0.5, arrow_y_start),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="->",
                    color="#616161",
                    lw=1.2,
                    connectionstyle="arc3,rad=0",
                ),
            )

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=type_colors[t], edgecolor=type_edge_colors[t],
                       linewidth=1.5, label=t)
        for t in ["Think", "Act", "Observe", "Answer"]
    ]
    ax.legend(handles=legend_patches, loc="lower center", ncol=4,
              fontsize=9, frameon=True, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.02))

    ax.set_title("Agent ReAct Reasoning Trace", fontsize=13, fontweight="bold", pad=10)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        out_path = outdir / f"agent_trace.{fmt}"
        fig.savefig(out_path, dpi=300)
        print(f"[ok] wrote {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Figure 5: Training Curves
# ─────────────────────────────────────────────────────────────

def plot_training_curves(training_log_path: Path | None, outdir: Path) -> None:
    """Plot training reward/loss curves from TRL training logs.

    Supports two formats:
    1. JSON lines: {"step": N, "reward": R, "loss": L, ...}
    2. TRL trainer_state.json with log_history

    Also supports reading from a directory containing trainer_state.json.
    """
    if training_log_path is None:
        print("[skip] no training log provided")
        return

    # If path is a directory, look for trainer_state.json
    if training_log_path.is_dir():
        candidate = training_log_path / "trainer_state.json"
        if candidate.exists():
            training_log_path = candidate
        else:
            print(f"[skip] no trainer_state.json in {training_log_path}")
            return

    if not training_log_path.exists():
        print(f"[skip] training log not found: {training_log_path}")
        return

    steps = []
    rewards = []
    losses = []

    try:
        with training_log_path.open("r", encoding="utf-8") as f:
            content = f.read().strip()

        if content.startswith("{"):
            # Single JSON object (trainer_state.json)
            data = json.loads(content)
            log_history = data.get("log_history", [])
            for entry in log_history:
                step = entry.get("step", entry.get("global_step"))
                if step is None:
                    continue
                steps.append(int(step))
                # Try multiple reward key names
                reward = (
                    entry.get("reward", None)
                    or entry.get("reward_mean", None)
                    or entry.get("mean_reward", None)
                )
                rewards.append(float(reward) if reward is not None else None)
                loss = entry.get("loss", entry.get("train_loss"))
                losses.append(float(loss) if loss is not None else None)
        else:
            # JSON lines format
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = entry.get("step", entry.get("global_step"))
                if step is None:
                    continue
                steps.append(int(step))
                reward = (
                    entry.get("reward", None)
                    or entry.get("reward_mean", None)
                    or entry.get("mean_reward", None)
                )
                rewards.append(float(reward) if reward is not None else None)
                loss = entry.get("loss", entry.get("train_loss"))
                losses.append(float(loss) if loss is not None else None)

    except (json.JSONDecodeError, IOError) as e:
        print(f"[skip] could not parse training log: {e}")
        return

    if not steps:
        print("[skip] no training steps found in log")
        return

    # Filter out None values
    has_reward = any(r is not None for r in rewards)
    has_loss = any(l is not None for l in losses)

    if not has_reward and not has_loss:
        print("[skip] no reward or loss data in training log")
        return

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color_reward = "#d62728"
    color_loss = "#1f77b4"

    if has_reward:
        valid_steps_r = [s for s, r in zip(steps, rewards) if r is not None]
        valid_rewards = [r for r in rewards if r is not None]
        ax1.plot(valid_steps_r, valid_rewards, color=color_reward, linewidth=1.5,
                 label="Reward", marker="o", markersize=2, alpha=0.8)
        ax1.set_ylabel("Reward", color=color_reward, fontsize=11)
        ax1.tick_params(axis="y", labelcolor=color_reward)

    if has_loss:
        valid_steps_l = [s for s, l in zip(steps, losses) if l is not None]
        valid_losses = [l for l in losses if l is not None]

        if has_reward:
            ax2 = ax1.twinx()
            ax2.plot(valid_steps_l, valid_losses, color=color_loss, linewidth=1.5,
                     label="Loss", marker="s", markersize=2, alpha=0.8)
            ax2.set_ylabel("Loss", color=color_loss, fontsize=11)
            ax2.tick_params(axis="y", labelcolor=color_loss)
        else:
            ax1.plot(valid_steps_l, valid_losses, color=color_loss, linewidth=1.5,
                     label="Loss", marker="s", markersize=2, alpha=0.8)
            ax1.set_ylabel("Loss", color=color_loss, fontsize=11)

    ax1.set_xlabel("Training Step")
    ax1.set_title("Training Curves")
    ax1.grid(alpha=0.3, linestyle="--")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if has_reward and has_loss:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=True)
    else:
        ax1.legend(loc="best", frameon=True)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        out_path = outdir / f"training_curves.{fmt}"
        fig.savefig(out_path, dpi=300)
        print(f"[ok] wrote {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate publication-quality figures from OTN RCA evaluation results."
    )
    ap.add_argument(
        "--results-dir", type=Path, default=Path("outputs/eval"),
        help="Directory containing evaluation output files (confusion_matrix.csv, etc.)",
    )
    ap.add_argument(
        "--outdir", type=Path, default=Path("outputs/figures"),
        help="Output directory for generated figures",
    )
    ap.add_argument(
        "--baselines", nargs="*", default=None,
        help=(
            "Baseline specs as name:path pairs, e.g. "
            "'Rule-based:outputs/eval/rule_based' 'SFT:outputs/eval/sft'"
        ),
    )
    ap.add_argument(
        "--agent-output", type=Path, default=None,
        help="Path to agent trace JSON file for agent_trace.png",
    )
    ap.add_argument(
        "--training-log", type=Path, default=None,
        help="Path to training log (trainer_state.json or JSONL) for training_curves.png",
    )
    args = ap.parse_args()

    _setup_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"[info] results dir: {args.results_dir}")
    print(f"[info] output dir:  {args.outdir}")

    # Figure 1: Confusion Matrix
    plot_confusion_matrix(args.results_dir, args.outdir)

    # Figure 2: Baseline Comparison
    plot_baseline_comparison(args.results_dir, args.outdir, args.baselines)

    # Figure 3: Per-Category Accuracy
    plot_category_accuracy(args.results_dir, args.outdir, args.baselines)

    # Figure 4: Agent Reasoning Trace
    plot_agent_trace(args.agent_output, args.outdir)

    # Figure 5: Training Curves
    plot_training_curves(args.training_log, args.outdir)

    print("[done] figure generation complete")


if __name__ == "__main__":
    main()
