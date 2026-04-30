#!/usr/bin/env python3
"""Plot BBE/BBER/ES timeseries for multi-failure scenarios in one figure.

Generates a single publication-quality figure showing the timeseries metrics
for each failure board side-by-side, with failure onset markers and event labels.

Usage:
  python plot_multi_failure_timeseries.py \
    --run-dir outputs/stress_tests/S2_multi_failure/runs/multi_2f_seed42 \
    --outdir outputs/stress_tests/figures

  # Plot a specific 3-failure run
  python plot_multi_failure_timeseries.py \
    --run-dir outputs/stress_tests/S2_multi_failure/runs/multi_3f_seed42 \
    --outdir outputs/stress_tests/figures

  # Plot multiple runs side-by-side for comparison
  python plot_multi_failure_timeseries.py \
    --run-dir outputs/stress_tests/S2_multi_failure/runs/multi_2f_seed42 \
              outputs/stress_tests/S2_multi_failure/runs/multi_3f_seed42 \
    --outdir outputs/stress_tests/figures
"""

from __future__ import annotations

import argparse
import glob
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

EVENT_COLORS = {
    "Fiber_Aging": "#FF9800",
    "Fiber_Crack": "#F44336",
    "Fiber_Cut": "#D32F2F",
    "Line_Disconnect": "#9C27B0",
    "XCON_Port_Down": "#2196F3",
    "XCON_Buffer_Overflow": "#00BCD4",
    "XCON_Fabric_Fault": "#4CAF50",
}

METRIC_LABELS = {
    "BBE": "Background Block Errors",
    "BBER": "Background Block Error Ratio",
    "ES": "Errored Seconds",
}

SHORT_EVENT = {
    "Fiber_Aging": "Fiber Aging",
    "Fiber_Crack": "Fiber Crack",
    "Fiber_Cut": "Fiber Cut",
    "Line_Disconnect": "Line Disconnect",
    "XCON_Port_Down": "XCON Port Down",
    "XCON_Buffer_Overflow": "XCON Buffer Overflow",
    "XCON_Fabric_Fault": "XCON Fabric Fault",
}


def _setup_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        try:
            plt.style.use("seaborn-paper")
        except OSError:
            pass
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load failure CSV and concatenated timeseries from a run directory.

    Returns (failures_df, timeseries_df).
    """
    fail_path = run_dir / "data_outputs" / "multi_failure.csv"
    if not fail_path.exists():
        # Fallback to failure.csv
        fail_path = run_dir / "failure.csv"
    if not fail_path.exists():
        raise FileNotFoundError(f"No failure CSV in {run_dir}")

    fail_df = pd.read_csv(fail_path)

    ts_files = sorted(glob.glob(str(run_dir / "data_outputs" / "Node_*" / "timeseries_*.csv")))
    if not ts_files:
        raise FileNotFoundError(f"No timeseries CSVs in {run_dir}/data_outputs/")

    ts_df = pd.concat([pd.read_csv(f) for f in ts_files], ignore_index=True)
    return fail_df, ts_df


def parse_failure_time(time_str: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(time_str).strip(), fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------------

def plot_single_run(
    run_dir: Path,
    outdir: Path,
    window_sec: int = 120,
    context_sec: int = 60,
) -> None:
    """Plot BBE/BBER/ES for all failure boards in one figure.

    Layout: N_failures rows x 3 columns (BBE, BBER, ES).
    Each row = one failure board, with failure onset marker.
    """
    fail_df, ts_df = load_run(run_dir)

    n_fail = len(fail_df)
    run_name = run_dir.name

    # Parse failure times
    failures = []
    for _, row in fail_df.iterrows():
        board = row["Board"]
        event = row["Event"]
        time_str = row.get("Time", "")
        dur_ms = row.get("DurationMs", 15000)
        ft = parse_failure_time(time_str)
        failures.append({
            "board": board,
            "event": event,
            "time": ft,
            "duration_sec": float(dur_ms) / 1000.0,
        })

    # Determine base_time (sim start = first failure - warmup)
    fail_times = [f["time"] for f in failures if f["time"]]
    if not fail_times:
        print(f"[skip] no parseable failure times in {run_dir}")
        return

    earliest = min(fail_times)
    # Simulator warmup is 60s typically; base_time = earliest - L_warmup
    sim_path = run_dir.parent.parent.parent / "../../simulator.yaml"
    if not sim_path.exists():
        sim_path = Path("simulator.yaml")
    warmup = 60
    if sim_path.exists():
        with sim_path.open() as f:
            cfg = yaml.safe_load(f)
        warmup = int(cfg.get("time", {}).get("L_warmup", 60))

    base_time = earliest - timedelta(seconds=warmup)

    # Convert timestep to real time
    def t_to_datetime(t: int) -> datetime:
        return base_time + timedelta(seconds=t)

    # Create figure: N_failures rows x 3 cols
    metrics = ["BBE", "BBER", "ES"]
    fig, axes = plt.subplots(
        n_fail, 3,
        figsize=(14, 3.2 * n_fail + 0.8),
        sharex="col",
        squeeze=False,
    )

    # Determine the time window to show: from (earliest_fail - context) to (latest_fail + duration + context)
    latest_end = max(
        f["time"] + timedelta(seconds=f["duration_sec"])
        for f in failures if f["time"]
    )
    t_start = earliest - timedelta(seconds=context_sec)
    t_end = latest_end + timedelta(seconds=context_sec)

    t_start_step = max(0, int((t_start - base_time).total_seconds()))
    t_end_step = int((t_end - base_time).total_seconds())

    for row_idx, failure in enumerate(failures):
        board = failure["board"]
        event = failure["event"]
        f_time = failure["time"]
        f_dur = failure["duration_sec"]
        color = EVENT_COLORS.get(event, "#333333")
        event_label = SHORT_EVENT.get(event, event)

        # Get timeseries for this board
        board_ts = ts_df[ts_df["board"] == board].copy()
        if board_ts.empty:
            for ax in axes[row_idx]:
                ax.text(0.5, 0.5, f"No data for {board}",
                        ha="center", va="center", transform=ax.transAxes)
            continue

        # Filter to window
        board_ts = board_ts[
            (board_ts["t"] >= t_start_step) & (board_ts["t"] <= t_end_step)
        ].copy()

        times = [t_to_datetime(int(t)) for t in board_ts["t"]]

        # Failure onset/end in datetime
        f_onset = f_time
        f_end = f_time + timedelta(seconds=f_dur) if f_time else None

        # Aggregate across services: max per timestep (shows worst-case)
        board_agg = board_ts.groupby("t").agg(
            BBE_max=("BBE", "max"),
            BBE_mean=("BBE", "mean"),
            BBER_max=("BBER", "max"),
            BBER_mean=("BBER", "mean"),
            ES_max=("ES", "max"),
            ES_mean=("ES", "mean"),
        ).reset_index()
        board_agg = board_agg.sort_values("t")
        times = [t_to_datetime(int(t)) for t in board_agg["t"]]

        agg_map = {"BBE": "BBE_max", "BBER": "BBER_max", "ES": "ES_max"}
        mean_map = {"BBE": "BBE_mean", "BBER": "BBER_mean", "ES": "ES_mean"}

        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            values_max = board_agg[agg_map[metric]].values
            values_mean = board_agg[mean_map[metric]].values

            # Plot mean as fill, max as line
            ax.fill_between(times, values_mean, alpha=0.25, color=color)
            ax.plot(times, values_max, color=color, linewidth=1.2, alpha=0.9,
                    label="max across services")
            ax.plot(times, values_mean, color=color, linewidth=0.7, alpha=0.5,
                    linestyle="--", label="mean")

            # Shade the failure window
            if f_onset and f_end:
                ax.axvspan(f_onset, f_end, color=color, alpha=0.15, zorder=0)
                ax.axvline(f_onset, color=color, linestyle="--",
                           linewidth=1.2, alpha=0.8, zorder=2)

            # Also mark OTHER failures as gray dashed lines
            for other_idx, other in enumerate(failures):
                if other_idx == row_idx:
                    continue
                if other["time"]:
                    other_color = EVENT_COLORS.get(other["event"], "#888888")
                    ax.axvline(other["time"], color=other_color,
                               linestyle=":", linewidth=0.8, alpha=0.5, zorder=1)

            # Y-axis label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(
                    f"F{row_idx+1}: {event_label}\n({board})",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                )
            else:
                ax.set_ylabel("")

            # Metric title on top row
            if row_idx == 0:
                ax.set_title(metric, fontsize=12, fontweight="bold")

            ax.grid(alpha=0.25, linestyle="--")
            ax.set_axisbelow(True)

            # Legend on first cell only
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

            # Format x-axis on bottom row
            if row_idx == n_fail - 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
                ax.tick_params(axis="x", rotation=30)

    # Suptitle
    event_list = " + ".join(SHORT_EVENT.get(f["event"], f["event"]) for f in failures)
    fig.suptitle(
        f"Multi-Failure Timeseries: {event_list}  ({run_name})",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # Save
    out_name = f"multi_failure_ts_{run_name}"
    for fmt in ("png", "pdf"):
        out_path = outdir / f"{out_name}.{fmt}"
        fig.savefig(out_path, dpi=300)
        print(f"[ok] wrote {out_path}")
    plt.close(fig)


def plot_comparison(
    run_dirs: list[Path],
    outdir: Path,
    window_sec: int = 120,
    context_sec: int = 60,
) -> None:
    """Plot multiple runs in a single comparison figure.

    Layout: rows = failure boards (max across runs), cols = one per run,
    showing only BBE (the primary metric) for a compact comparison.
    """
    runs_data = []
    for rd in run_dirs:
        try:
            fail_df, ts_df = load_run(rd)
            runs_data.append((rd, fail_df, ts_df))
        except FileNotFoundError as e:
            print(f"[skip] {e}")

    if not runs_data:
        return

    n_runs = len(runs_data)
    max_failures = max(len(fd) for _, fd, _ in runs_data)

    fig, axes = plt.subplots(
        max_failures, n_runs,
        figsize=(5.5 * n_runs, 3.0 * max_failures + 0.8),
        squeeze=False,
    )

    sim_path = Path("simulator.yaml")
    warmup = 60
    if sim_path.exists():
        with sim_path.open() as f:
            cfg = yaml.safe_load(f)
        warmup = int(cfg.get("time", {}).get("L_warmup", 60))

    for col_idx, (rd, fail_df, ts_df) in enumerate(runs_data):
        failures = []
        for _, row in fail_df.iterrows():
            ft = parse_failure_time(row.get("Time", ""))
            failures.append({
                "board": row["Board"],
                "event": row["Event"],
                "time": ft,
                "duration_sec": float(row.get("DurationMs", 15000)) / 1000.0,
            })

        fail_times = [f["time"] for f in failures if f["time"]]
        if not fail_times:
            continue

        earliest = min(fail_times)
        base_time = earliest - timedelta(seconds=warmup)
        latest_end = max(
            f["time"] + timedelta(seconds=f["duration_sec"])
            for f in failures if f["time"]
        )
        t_start = earliest - timedelta(seconds=context_sec)
        t_end = latest_end + timedelta(seconds=context_sec)
        t_start_step = max(0, int((t_start - base_time).total_seconds()))
        t_end_step = int((t_end - base_time).total_seconds())

        def t_to_dt(t):
            return base_time + timedelta(seconds=t)

        # Column title
        event_list = " + ".join(SHORT_EVENT.get(f["event"], f["event"]) for f in failures)
        axes[0, col_idx].set_title(
            f"{rd.name}\n({event_list})",
            fontsize=10, fontweight="bold",
        )

        for row_idx, failure in enumerate(failures):
            board = failure["board"]
            event = failure["event"]
            color = EVENT_COLORS.get(event, "#333333")
            ax = axes[row_idx, col_idx]

            board_ts = ts_df[
                (ts_df["board"] == board) &
                (ts_df["t"] >= t_start_step) &
                (ts_df["t"] <= t_end_step)
            ]

            if board_ts.empty:
                ax.text(0.5, 0.5, f"No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#999")
                continue

            # Aggregate across services
            board_agg = board_ts.groupby("t").agg(
                BBE_max=("BBE", "max"), BBE_mean=("BBE", "mean"),
            ).reset_index().sort_values("t")
            times = [t_to_dt(int(t)) for t in board_agg["t"]]
            ax.fill_between(times, board_agg["BBE_mean"].values, alpha=0.25, color=color)
            ax.plot(times, board_agg["BBE_max"].values, color=color, linewidth=1.0)

            # Shade failure window
            if failure["time"]:
                f_end = failure["time"] + timedelta(seconds=failure["duration_sec"])
                ax.axvspan(failure["time"], f_end, color=color, alpha=0.15)
                ax.axvline(failure["time"], color=color, linestyle="--", linewidth=1.2, alpha=0.8)

            # Mark other failures
            for oi, other in enumerate(failures):
                if oi != row_idx and other["time"]:
                    oc = EVENT_COLORS.get(other["event"], "#888")
                    ax.axvline(other["time"], color=oc, linestyle=":", linewidth=0.8, alpha=0.5)

            if col_idx == 0:
                ax.set_ylabel(
                    f"{SHORT_EVENT.get(event, event)}\n({board})",
                    fontsize=8, fontweight="bold", color=color,
                )

            ax.grid(alpha=0.25, linestyle="--")
            ax.set_axisbelow(True)

            if row_idx == max_failures - 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
                ax.tick_params(axis="x", rotation=30)

        # Hide unused rows
        for row_idx in range(len(failures), max_failures):
            axes[row_idx, col_idx].set_visible(False)

    fig.suptitle(
        "Multi-Failure BBE Timeseries Comparison",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_name = "multi_failure_ts_comparison"
    for fmt in ("png", "pdf"):
        out_path = outdir / f"{out_name}.{fmt}"
        fig.savefig(out_path, dpi=300)
        print(f"[ok] wrote {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot BBE/BBER/ES timeseries for multi-failure runs in one figure."
    )
    ap.add_argument(
        "--run-dir", type=Path, nargs="+", required=True,
        help="One or more multi-failure run directories",
    )
    ap.add_argument(
        "--outdir", type=Path, default=Path("outputs/stress_tests/figures"),
        help="Output directory for figures",
    )
    ap.add_argument(
        "--window-sec", type=int, default=120,
        help="Window around failure to display (seconds)",
    )
    ap.add_argument(
        "--context-sec", type=int, default=60,
        help="Context before/after failure window",
    )
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    _setup_style()

    if len(args.run_dir) == 1:
        plot_single_run(args.run_dir[0], args.outdir,
                        args.window_sec, args.context_sec)
    else:
        # Plot each individually + a comparison
        for rd in args.run_dir:
            plot_single_run(rd, args.outdir, args.window_sec, args.context_sec)
        plot_comparison(args.run_dir, args.outdir,
                        args.window_sec, args.context_sec)


if __name__ == "__main__":
    main()
