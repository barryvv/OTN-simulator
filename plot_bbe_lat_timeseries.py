#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import re
from datetime import datetime, timedelta

import yaml
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


_BOARD_PAT = re.compile(r"^(?P<parent>[^-]+)-(?P<kind>[A-Za-z]+?)(?P<idx>\d+)\$0$")


def parse_board(board: str) -> tuple[str | None, str | None]:
    s = str(board or "").strip()
    m = _BOARD_PAT.match(s)
    if m:
        return m.group("parent"), m.group("kind").upper()
    if "-" not in s and s:
        return s, None
    return None, None


def role_for_kind(kind: str | None) -> str:
    k = (kind or "").upper()
    if k in {"LINE"}:
        return "SNK"
    if k in {"TRIBUTARY", "XCON"}:
        return "SRC"
    if k in {"OA", "OM", "OD", "FIU", "SC2"}:
        return "RELAY"
    return "RELAY"

def normalize_failure_role(role: str | None) -> str | None:
    if not role:
        return None
    r = str(role).strip().upper().replace(" ", "").replace("_", "-")
    if r in {"SRC"}:
        return "SRC"
    if r in {"RELAY"}:
        return "RELAY"
    if r in {"SNK", "TERMINATION"}:
        return "SNK"
    if r in {"OTUK-RELAY"}:
        return "RELAY"
    if r in {"ODUK-TERMINATION"}:
        return "SNK"
    if r in {"ODUK-CREATION"}:
        return "SRC"
    return None


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    s = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def parse_step_seconds(step_val: str) -> int:
    s = str(step_val).strip().lower()
    if s.endswith("ms"):
        return max(1, int(round(float(s[:-2]) / 1000.0)))
    if s.endswith("s"):
        return max(1, int(round(float(s[:-1]))))
    try:
        return max(1, int(round(float(s))))
    except Exception:
        return 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot BBE time series for failure boards.")
    ap.add_argument("--failure", default="outputs/alarm_flows/failure.csv",
                    help="Failure CSV (Board,Event,Severity,Time,DurationMs)")
    ap.add_argument("--topology", default="topology.yaml",
                    help="Topology YAML with node_map")
    ap.add_argument("--data-dir", default="data_outputs",
                    help="MW simulator output directory")
    ap.add_argument("--simulator", default="simulator.yaml",
                    help="Simulator YAML to read step size from")
    ap.add_argument("--outdir", default=None,
                    help="Output directory for plots (defaults to failure CSV dir)")
    ap.add_argument("--max-series", type=int, default=1,
                    help="Max VMFs to plot per failure")
    ap.add_argument("--lead-sec", type=int, default=60,
                    help="Baseline seconds before the first failure time")
    ap.add_argument("--time-axis", action="store_true",
                    help="Plot x-axis as timestamps instead of step index")
    args = ap.parse_args()

    failure_path = Path(args.failure)
    if not failure_path.exists():
        raise SystemExit(f"Missing failure file: {failure_path}")

    topo_path = Path(args.topology)
    if not topo_path.exists():
        raise SystemExit(f"Missing topology file: {topo_path}")

    with topo_path.open("r", encoding="utf-8") as f:
        topo = yaml.safe_load(f)
    node_map = topo.get("node_map") or {}
    board_map = topo.get("board_map") or {}
    if not node_map:
        raise SystemExit("topology.yaml missing node_map; re-run convert_graphml_to_topology.py")

    df_fail = pd.read_csv(failure_path)
    cols = {str(c).strip().lower(): c for c in df_fail.columns}
    c_board = cols.get("board") or cols.get("node") or list(df_fail.columns)[0]
    c_event = cols.get("event") or cols.get("alarm")
    c_time = cols.get("time")
    c_role = cols.get("boardrole") or cols.get("role")

    step_seconds = 1
    sim_path = Path(args.simulator)
    if sim_path.exists():
        with sim_path.open("r", encoding="utf-8") as f:
            sim_cfg = yaml.safe_load(f)
        step_seconds = parse_step_seconds(sim_cfg.get("time", {}).get("step", "1s"))

    failure_times = []
    if c_time:
        for v in df_fail[c_time].tolist():
            t = parse_time(v)
            if t:
                failure_times.append(t)
    base_time = (min(failure_times) - timedelta(seconds=max(0, int(args.lead_sec)))) if failure_times else None

    outdir = Path(args.outdir) if args.outdir else failure_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    parent_by_id = {int(v): k for k, v in node_map.items()}
    failure_pref = {}
    if c_role:
        for _, r in df_fail.iterrows():
            parent, kind = parse_board(r.get(c_board))
            if not parent:
                continue
            role = normalize_failure_role(r.get(c_role)) if c_role else None
            if not role:
                role = role_for_kind(kind)
            failure_pref[(parent, role)] = r.get(c_board)

    for _, row in df_fail.iterrows():
        board = row.get(c_board)
        if pd.isna(board):
            continue
        board = str(board)
        parent, kind = parse_board(board)
        if not parent or parent not in node_map:
            continue
        node_id = int(node_map[parent])
        ts_path = Path(args.data_dir) / f"Node_{node_id}" / f"timeseries_{node_id}.csv"
        if not ts_path.exists():
            continue

        df = pd.read_csv(ts_path)
        role = normalize_failure_role(row.get(c_role)) if c_role else None
        if not role:
            role = role_for_kind(kind)
        role_token = f"_{role}_"
        df_sel = df[df["vmf_id"].str.contains(role_token, na=False)]
        if df_sel.empty:
            df_sel = df

        vmf_ids = list(df_sel["vmf_id"].unique())[: args.max_series]
        if not vmf_ids:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        event_time = parse_time(row.get(c_time)) if c_time else None
        def board_for_vid(vid: str) -> str:
            node_id = None
            try:
                node_id = int(vid.split("_", 1)[0])
            except Exception:
                return vid
            parent = parent_by_id.get(node_id)
            if not parent:
                return vid
            pref = failure_pref.get((parent, role))
            if pref:
                return pref
            kinds = board_map.get(parent, {})
            if role == "SRC":
                for k in ("TRIBUTARY", "XCON"):
                    if kinds.get(k):
                        return kinds[k][0]
            if role == "SNK":
                if kinds.get("LINE"):
                    return kinds["LINE"][0]
            if role == "RELAY":
                for k in ("OA", "OM", "OD", "FIU", "SC2"):
                    if kinds.get(k):
                        return kinds[k][0]
            return vid

        for vid in vmf_ids:
            g = df_sel[df_sel["vmf_id"] == vid]
            label = board_for_vid(vid)
            if args.time_axis and base_time:
                times = [base_time + timedelta(seconds=step_seconds * int(t)) for t in g["t"].tolist()]
                ax.plot(times, g["BBE"], label=label)
            else:
                ax.plot(g["t"], g["BBE"], label=label)

        if args.time_axis and base_time and event_time:
            ax.axvline(event_time, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

        evt = str(row.get(c_event)) if c_event else ""
        ax.set_title(f"{board} ({evt}) - BBE")
        ax.set_xlabel("Time" if (args.time_axis and base_time) else "t (steps)")
        ax.set_ylabel("BBE")
        ax.legend(fontsize=7, ncol=2)
        if args.time_axis and base_time:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            fig.autofmt_xdate()
        fig.tight_layout()

        out_path = outdir / f"{board.replace('$','')}_bbe.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

        print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
