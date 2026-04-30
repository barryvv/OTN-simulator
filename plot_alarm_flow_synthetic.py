#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


@dataclass
class AlarmEvent:
    start: datetime
    end: datetime
    alarm: str
    rule_idx: int | None


def parse_time(value: str | None) -> datetime | None:
    if not value or str(value).strip() == "":
        return None
    s = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def to_float(value, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def load_events(df: pd.DataFrame) -> tuple[dict[str, list[AlarmEvent]], datetime | None]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    c_board = cols.get("proptoboard") or cols.get("board")
    c_alarm = cols.get("propalarm") or cols.get("alarm") or cols.get("propevent")
    c_arr = cols.get("arrivaltime")
    c_clear = cols.get("cleartime")
    c_src_start = cols.get("sourcestarttime")
    c_dur = cols.get("sourcedurationsec")
    c_rule = cols.get("ruleidx")
    if not (c_board and c_arr):
        raise SystemExit("alarm_flow missing PropToBoard/ArrivalTime columns")

    events: dict[str, list[AlarmEvent]] = {}
    src_time = None
    if c_src_start:
        for v in df[c_src_start].dropna().astype(str).tolist():
            ts = parse_time(v)
            if ts:
                src_time = ts if src_time is None else min(src_time, ts)

    for _, r in df.iterrows():
        board = str(r.get(c_board)).strip()
        if not board:
            continue
        alarm = str(r.get(c_alarm) or "").strip()
        arr = parse_time(r.get(c_arr))
        if arr is None:
            continue
        clear = parse_time(r.get(c_clear))
        if clear is None:
            dur = to_float(r.get(c_dur), 5.0)
            clear = arr + timedelta(seconds=max(1.0, dur))
        rule_idx = None
        if c_rule and pd.notna(r.get(c_rule)):
            try:
                rule_idx = int(r.get(c_rule))
            except Exception:
                rule_idx = None
        events.setdefault(board, []).append(AlarmEvent(start=arr, end=clear, alarm=alarm, rule_idx=rule_idx))

    return events, src_time


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot synthetic spikes from alarm_flow.csv")
    ap.add_argument("--alarm-flow", default="outputs/alarm_flows/alarm_flow.csv")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--boards", nargs="*", default=None,
                    help="Optional list of boards to plot (default: all)")
    ap.add_argument("--step-sec", type=int, default=1)
    ap.add_argument("--lead-sec", type=int, default=10)
    ap.add_argument("--trail-sec", type=int, default=10)
    ap.add_argument("--bbe-baseline", type=float, default=1.0)
    ap.add_argument("--bbe-spike", type=float, default=8.0)
    ap.add_argument("--lat-sigma", type=float, default=0.25)
    ap.add_argument("--lat-shift", type=float, default=0.8)
    ap.add_argument("--bber-scale", type=float, default=1000.0)
    ap.add_argument("--es-threshold", type=float, default=9.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    alarm_path = Path(args.alarm_flow)
    if not alarm_path.exists():
        raise SystemExit(f"Missing alarm_flow: {alarm_path}")

    df = pd.read_csv(alarm_path)
    if df.empty:
        raise SystemExit("alarm_flow is empty")

    events_by_board, src_time = load_events(df)
    if not events_by_board:
        raise SystemExit("No events found in alarm_flow")

    outdir = Path(args.outdir) if args.outdir else alarm_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    boards = args.boards or list(events_by_board.keys())
    for board in boards:
        events = events_by_board.get(board, [])
        if not events:
            continue
        start = min(e.start for e in events) - timedelta(seconds=max(0, args.lead_sec))
        end = max(e.end for e in events) + timedelta(seconds=max(0, args.trail_sec))
        n_steps = max(2, int((end - start).total_seconds() / max(1, args.step_sec)) + 1)
        times = [start + timedelta(seconds=args.step_sec * i) for i in range(n_steps)]

        bbe = rng.poisson(lam=max(0.1, args.bbe_baseline), size=n_steps).astype(float)
        lat = rng.normal(loc=0.0, scale=max(0.01, args.lat_sigma), size=n_steps).astype(float)

        palette = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                   "tab:purple", "tab:brown", "tab:pink", "tab:gray",
                   "tab:olive", "tab:cyan"]
        alarm_colors = {}
        for ev in events:
            idx0 = max(0, int((ev.start - start).total_seconds() / args.step_sec))
            idx1 = min(n_steps, int((ev.end - start).total_seconds() / args.step_sec))
            if idx1 <= idx0:
                idx1 = min(n_steps, idx0 + 1)
            bbe[idx0:idx1] += float(args.bbe_spike)
            lat[idx0:idx1] += float(args.lat_shift)

        bber = bbe / max(1.0, args.bber_scale)
        es = (bbe >= float(args.es_threshold)).astype(float)

        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        axes[0].plot(times, lat, label=board)
        axes[1].plot(times, bbe, label=board)
        axes[2].plot(times, bber, label=board)
        axes[3].plot(times, es, label=board)

        y_top = axes[1].get_ylim()[1]
        y_range = max(1e-6, axes[1].get_ylim()[1] - axes[1].get_ylim()[0])
        label_rows = 6
        for idx, ev in enumerate(events):
            label = ev.alarm if ev.alarm else "alarm"
            if label not in alarm_colors:
                alarm_colors[label] = palette[len(alarm_colors) % len(palette)]
            color = alarm_colors[label]
            for ax in axes:
                ax.axvspan(ev.start, ev.end, color=color, alpha=0.08)
                ax.axvline(ev.start, color=color, linestyle="--", linewidth=0.8, alpha=0.6)
                ax.axvline(ev.end, color=color, linestyle=":", linewidth=0.8, alpha=0.6)
            mid = ev.start + (ev.end - ev.start) / 2
            note = f"{label}\nARR {ev.start.strftime('%H:%M:%S')}\nCLR {ev.end.strftime('%H:%M:%S')}"
            row = idx % label_rows
            col = idx // label_rows
            axes[1].annotate(
                note,
                xy=(mid, y_top),
                xycoords="data",
                xytext=(1.02 + col * 0.18, 0.95 - row * 0.12),
                textcoords="axes fraction",
                ha="left",
                va="top",
                fontsize=7,
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.6),
                clip_on=False,
            )

        if src_time:
            for ax in axes:
                ax.axvline(src_time, color="red", linestyle="--", linewidth=1.0, alpha=0.7)

        axes[0].set_title(f"{board} - LAT")
        axes[1].set_title(f"{board} - BBE")
        axes[2].set_title(f"{board} - BBER")
        axes[3].set_title(f"{board} - ES")
        axes[0].set_ylabel("LAT")
        axes[1].set_ylabel("BBE")
        axes[2].set_ylabel("BBER")
        axes[3].set_ylabel("ES")
        axes[3].set_xlabel("Time (HH:MM:SS)")
        for ax in axes:
            ax.legend(fontsize=7, ncol=2)
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()
        fig.subplots_adjust(right=0.78)
        fig.tight_layout()

        out_path = outdir / f"{board.replace('$','')}_alarm_flow_synth.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
