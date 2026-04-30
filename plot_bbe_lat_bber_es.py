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
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
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

def find_intervals(mask, times, min_len: int) -> list[tuple]:
    intervals = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        if (not flag) and start is not None:
            if idx - start >= min_len:
                intervals.append((times[start], times[idx - 1]))
            start = None
    if start is not None and len(mask) - start >= min_len:
        intervals.append((times[start], times[-1]))
    return intervals

def _fmt_time(val) -> str:
    if hasattr(val, "strftime"):
        return val.strftime("%H:%M:%S")
    return str(val)

def role_from_vmf_id(vmf_id: str) -> str | None:
    parts = str(vmf_id).split("_", 2)
    if len(parts) >= 2:
        role = parts[1].upper()
        if role in {"SRC", "SNK", "RELAY"}:
            return role
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot BBE/BBER/ES plus anomaly score for failure boards.")
    ap.add_argument("--failure", default="outputs/alarm_flows/failure.csv",
                    help="Failure CSV (Board,Event,Severity,Time,DurationMs)")
    ap.add_argument("--topology", default="topology.yaml")
    ap.add_argument("--data-dir", default="data_outputs")
    ap.add_argument("--simulator", default="simulator.yaml")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--lead-sec", type=int, default=None,
                    help="Seconds of baseline before failure (default: simulator.yaml time.L_warmup)")
    ap.add_argument("--time-axis", action="store_true", default=True)
    ap.add_argument("--max-series", type=int, default=1)
    ap.add_argument("--affected-all", action="store_true",
                    help="Also plot other boards affected by the failure")
    ap.add_argument("--affected-top", type=int, default=8,
                    help="Max number of affected boards to plot")
    ap.add_argument("--affected-min-delta", type=float, default=1.0,
                    help="Minimum BBE delta to treat a board as affected")
    ap.add_argument("--baseline-sec", type=int, default=300,
                    help="Seconds before failure to use as baseline")
    ap.add_argument("--boards-from-alarm-flow", default=None,
                    help="Optional alarm_flow(_timeline).csv to choose boards to plot")
    ap.add_argument("--spike-sigma", type=float, default=3.0,
                    help="BBE spike threshold in baseline std devs")
    ap.add_argument("--min-span-sec", type=int, default=10,
                    help="Minimum span to label a spike period")
    ap.add_argument("--show-auto-detect", action="store_true",
                    help="Show auto-detected spike/shift markers")
    ap.add_argument("--anomaly-alpha", type=float, default=1.0,
                    help="Weight for anomaly score (higher = more sensitive)")
    ap.add_argument("--center-event", action="store_true",
                    help="Center x-axis around event time for all plots")
    ap.add_argument("--center-window-sec", type=int, default=900,
                    help="Window length in seconds when --center-event is set")
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
        raise SystemExit("topology.yaml missing node_map")

    df_fail = pd.read_csv(failure_path)
    cols = {str(c).strip().lower(): c for c in df_fail.columns}
    c_board = cols.get("board") or cols.get("node") or list(df_fail.columns)[0]
    c_event = cols.get("event") or cols.get("alarm")
    c_time = cols.get("time")
    c_dur = cols.get("durationms") or cols.get("duration")
    c_role = cols.get("boardrole") or cols.get("role")

    step_seconds = 1
    sim_path = Path(args.simulator)
    if sim_path.exists():
        with sim_path.open("r", encoding="utf-8") as f:
            sim_cfg = yaml.safe_load(f)
        step_seconds = parse_step_seconds(sim_cfg.get("time", {}).get("step", "1s"))

    lead_sec = args.lead_sec
    if lead_sec is None:
        lead_sec = int(sim_cfg.get("time", {}).get("L_warmup", 0) or 0)

    failure_times = []
    if c_time:
        for v in df_fail[c_time].tolist():
            t = parse_time(v)
            if t:
                failure_times.append(t)
    base_time = (min(failure_times) - timedelta(seconds=max(0, int(lead_sec)))) if failure_times else None

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

    def board_for_vid(vid: str, role: str | None) -> str:
        try:
            node_id = int(vid.split("_", 1)[0])
        except Exception:
            return vid
        parent = parent_by_id.get(node_id)
        if not parent:
            return vid
        if role:
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

    def node_id_for_board(board_name: str) -> int | None:
        parent, _ = parse_board(board_name)
        if not parent:
            return None
        val = node_map.get(parent)
        return int(val) if val is not None else None

    def _has_vmf_col(df: pd.DataFrame) -> bool:
        return "vmf_id" in df.columns

    def _has_board_col(df: pd.DataFrame) -> bool:
        return "board" in df.columns

    def load_timeseries_for_id(id_val: str) -> pd.DataFrame | None:
        """Load timeseries CSV for a vmf_id (e.g. '1_SRC_S1') or board name."""
        # Try to extract node_id from vmf_id format first
        try:
            node_id = int(id_val.split("_", 1)[0])
        except Exception:
            # id_val is a board name — resolve via node_map
            parent, _ = parse_board(id_val)
            if parent and parent in node_map:
                node_id = int(node_map[parent])
            else:
                return None
        ts_path = Path(args.data_dir) / f"Node_{node_id}" / f"timeseries_{node_id}.csv"
        if not ts_path.exists():
            return None
        return pd.read_csv(ts_path)

    def plot_board(board: str, event: str, event_time: datetime | None,
                   vmf_ids: list[str],
                   alarm_windows: list[tuple[datetime, datetime | None, str]] | None = None) -> None:
        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        for vid in vmf_ids:
            df_sel = load_timeseries_for_id(vid)
            if df_sel is None or df_sel.empty:
                continue
            # Support both vmf_id and board column formats
            if _has_vmf_col(df_sel):
                g = df_sel[df_sel["vmf_id"] == vid]
            elif _has_board_col(df_sel):
                # vid may be a board name or vmf_id; try board match first
                g = df_sel[df_sel["board"] == board]
                if g.empty:
                    g = df_sel[df_sel["board"] == vid]
            else:
                g = df_sel
            if g.empty:
                continue
            label = board
            if args.time_axis and base_time:
                times = [base_time + timedelta(seconds=step_seconds * int(t)) for t in g["t"].tolist()]
            else:
                times = g["t"].tolist()
            bbe_spans = []
            anomaly = None
            if event_time and base_time:
                t0 = int((event_time - base_time).total_seconds() / step_seconds)
            else:
                t0 = int(args.baseline_sec / step_seconds)
            base_steps = max(1, int(args.baseline_sec / step_seconds))
            b0 = max(0, t0 - base_steps)
            b1 = max(b0 + 1, t0)
            base_bbe = g[(g["t"] >= b0) & (g["t"] < b1)]["BBE"]
            base_bber = g[(g["t"] >= b0) & (g["t"] < b1)]["BBER"]
            base_es = g[(g["t"] >= b0) & (g["t"] < b1)]["ES"]
            def _z(series, base):
                mu = float(base.mean()) if not base.empty else float(series.mean())
                sd = float(base.std()) if not base.empty else float(series.std())
                if sd <= 0:
                    sd = 1.0
                return (series - mu).abs() / sd
            z_bbe = _z(g["BBE"], base_bbe)
            z_bber = _z(g["BBER"], base_bber)
            z_es = _z(g["ES"], base_es)
            anomaly = args.anomaly_alpha * pd.concat([z_bbe, z_bber, z_es], axis=1).max(axis=1)
            if args.show_auto_detect:
                if event_time and base_time:
                    t0 = int((event_time - base_time).total_seconds() / step_seconds)
                else:
                    t0 = int(args.baseline_sec / step_seconds)
                base_steps = max(1, int(args.baseline_sec / step_seconds))
                b0 = max(0, t0 - base_steps)
                b1 = max(b0 + 1, t0)
                base_bbe = g[(g["t"] >= b0) & (g["t"] < b1)]["BBE"]
                bbe_mean = float(base_bbe.mean()) if not base_bbe.empty else float(g["BBE"].mean())
                bbe_std = float(base_bbe.std()) if not base_bbe.empty else float(g["BBE"].std())
                spike_thr = bbe_mean + args.spike_sigma * (bbe_std if bbe_std > 0 else 1.0)
                min_len = max(1, int(args.min_span_sec / step_seconds))
                bbe_mask = (g["BBE"] >= spike_thr).tolist()
                bbe_spans = find_intervals(bbe_mask, times, min_len)
            if args.time_axis and base_time:
                axes[0].plot(times, g["BBE"], label=label)
                axes[1].plot(times, g.get("BBER"), label=label)
                axes[2].plot(times, g.get("ES"), label=label)
                axes[3].plot(times, anomaly, label=label)
            else:
                axes[0].plot(g["t"], g["BBE"], label=label)
                axes[1].plot(g["t"], g.get("BBER"), label=label)
                axes[2].plot(g["t"], g.get("ES"), label=label)
                axes[3].plot(g["t"], anomaly, label=label)
            if args.show_auto_detect:
                for s, e in bbe_spans:
                    axes[0].axvspan(s, e, color="red", alpha=0.12)
                    axes[1].axvspan(s, e, color="red", alpha=0.08)
                    axes[2].axvspan(s, e, color="red", alpha=0.08)
                    axes[3].axvspan(s, e, color="red", alpha=0.08)

        if args.time_axis and base_time and event_time:
            for ax in axes:
                ax.axvline(event_time, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
        if args.time_axis and alarm_windows:
            for idx, (start, end, alarm) in enumerate(alarm_windows[:8]):
                label = f"{alarm} { _fmt_time(start) }"
                if end:
                    label += f"–{ _fmt_time(end) }"
                for ax in axes:
                    if end and end >= start:
                        ax.axvspan(start, end, color="tab:blue", alpha=0.08, zorder=0)
                        ax.axvline(start, color="tab:blue", linestyle=":", linewidth=0.9, alpha=0.7)
                        ax.axvline(end, color="tab:blue", linestyle=":", linewidth=0.9, alpha=0.7)
                    else:
                        ax.axvline(start, color="tab:blue", linestyle=":", linewidth=0.9, alpha=0.7)
                axes[0].text(
                    start,
                    0.98 - (idx * 0.06),
                    label,
                    transform=axes[0].get_xaxis_transform(),
                    fontsize=7,
                    color="tab:blue",
                    va="top",
                )

        if args.time_axis and args.show_auto_detect:
            if bbe_spans:
                s, e = bbe_spans[0]
                label = f"BBE spike { _fmt_time(s) }–{ _fmt_time(e) }"
                axes[0].text(s, axes[0].get_ylim()[1], label, color="red", fontsize=8, va="top")
                axes[1].text(s, axes[1].get_ylim()[1], label.replace("BBE", "BBER"), color="red", fontsize=8, va="top")
                axes[2].text(s, axes[2].get_ylim()[1], label.replace("BBE", "ES"), color="red", fontsize=8, va="top")
                axes[3].text(s, axes[3].get_ylim()[1], label.replace("BBE", "Anom"), color="red", fontsize=8, va="top")

        if args.center_event and args.time_axis and event_time:
            half_window = max(1, int(args.center_window_sec)) / 2.0
            start = event_time - timedelta(seconds=half_window)
            end = event_time + timedelta(seconds=half_window)
            for ax in axes:
                ax.set_xlim(start, end)

        axes[0].set_title(f"{board} ({event}) - BBE")
        axes[1].set_title(f"{board} ({event}) - BBER")
        axes[2].set_title(f"{board} ({event}) - ES")
        axes[3].set_title(f"{board} ({event}) - Anomaly Score")
        axes[3].set_xlabel("Time (HH:MM:SS)" if (args.time_axis and base_time) else "t (steps)")
        axes[0].set_ylabel("BBE")
        axes[1].set_ylabel("BBER")
        axes[2].set_ylabel("ES")
        axes[3].set_ylabel("Score")
        axes[0].legend(fontsize=7, ncol=2)
        axes[1].legend(fontsize=7, ncol=2)
        axes[2].legend(fontsize=7, ncol=2)
        if args.time_axis and base_time:
            axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            fig.autofmt_xdate()
        fig.tight_layout()

        out_path = outdir / f"{board.replace('$','')}_bbe_bber_es.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"[ok] wrote {out_path}")

    # Build board→event lookup for correct plot titles in multi-failure scenarios
    _board_event_map: dict[str, str] = {}
    for _, _r in df_fail.iterrows():
        _b = _r.get(c_board)
        _e = _r.get(c_event) if c_event else ""
        if not pd.isna(_b):
            _board_event_map[str(_b)] = str(_e)
    _plotted_boards: set[str] = set()

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

        # Support both vmf_id and board column formats
        if _has_vmf_col(df):
            role_token = f"_{role}_"
            df_sel = df[df["vmf_id"].str.contains(role_token, na=False)]
            if df_sel.empty:
                df_sel = df
            vmf_ids = list(df_sel["vmf_id"].unique())[: args.max_series]
        elif _has_board_col(df):
            df_sel = df[df["board"] == board]
            if df_sel.empty:
                df_sel = df
            vmf_ids = list(df_sel["board"].unique())[: args.max_series]
        else:
            vmf_ids = [board]

        if not vmf_ids:
            continue

        event_time = parse_time(row.get(c_time)) if c_time else None
        evt = str(row.get(c_event)) if c_event else ""

        targets = [(board, vmf_ids)]
        if args.boards_from_alarm_flow:
            alarm_path = Path(args.boards_from_alarm_flow)
            if alarm_path.exists():
                df_alarm = pd.read_csv(alarm_path)
                cols_alarm = {str(c).strip().lower(): c for c in df_alarm.columns}
                c_prop = cols_alarm.get("proptoboard")
                c_src = cols_alarm.get("sourceboard")
                c_arr = cols_alarm.get("arrivaltime")
                c_clear = cols_alarm.get("cleartime")
                c_alarm = cols_alarm.get("propalarm") or cols_alarm.get("sourcealarm") or cols_alarm.get("alarm")
                c_prop_role = cols_alarm.get("proprole")
                c_src_role = cols_alarm.get("sourcerole")
                board_list = []
                role_hint = {}
                if c_src:
                    for v in df_alarm[c_src].dropna().astype(str).tolist():
                        if v not in board_list:
                            board_list.append(v)
                if c_prop:
                    for v in df_alarm[c_prop].dropna().astype(str).tolist():
                        if v not in board_list:
                            board_list.append(v)
                if c_src_role and c_src:
                    for b, r in zip(df_alarm[c_src].astype(str), df_alarm[c_src_role].astype(str)):
                        role_hint.setdefault(b, normalize_failure_role(r))
                if c_prop_role and c_prop:
                    for b, r in zip(df_alarm[c_prop].astype(str), df_alarm[c_prop_role].astype(str)):
                        role_hint.setdefault(b, normalize_failure_role(r))
                alarm_windows_map: dict[str, list[tuple[datetime, datetime | None, str]]] = {}
                if c_arr and c_prop:
                    for _, r in df_alarm.iterrows():
                        b = str(r.get(c_prop, "")).strip()
                        if not b:
                            continue
                        start = parse_time(r.get(c_arr))
                        if not start:
                            continue
                        end = parse_time(r.get(c_clear)) if c_clear else None
                        alarm = str(r.get(c_alarm)) if c_alarm else ""
                        alarm_windows_map.setdefault(b, []).append((start, end, alarm))
                targets = []
                for bname in board_list:
                    node_id_b = node_id_for_board(bname)
                    if node_id_b is None:
                        continue
                    role_b = role_hint.get(bname) or role_for_kind(parse_board(bname)[1])
                    ts_path_b = Path(args.data_dir) / f"Node_{node_id_b}" / f"timeseries_{node_id_b}.csv"
                    if not ts_path_b.exists():
                        continue
                    df_node = pd.read_csv(ts_path_b)
                    if _has_vmf_col(df_node):
                        token = f"_{role_b}_"
                        vmfs = df_node[df_node["vmf_id"].str.contains(token, na=False)]["vmf_id"].unique().tolist()
                    elif _has_board_col(df_node):
                        vmfs = df_node[df_node["board"] == bname]["board"].unique().tolist()
                    else:
                        vmfs = [bname]
                    if not vmfs:
                        continue
                    windows = alarm_windows_map.get(bname)
                    if windows:
                        uniq = {(w[0], w[1], w[2]) for w in windows}
                        windows = sorted(uniq, key=lambda x: x[0])
                    targets.append((bname, [vmfs[0]], windows))
        if args.affected_all and event_time and base_time is not None:
            t0 = int((event_time - base_time).total_seconds() / step_seconds)
            dur_ms = row.get(c_dur)
            if pd.notna(dur_ms):
                win_steps = max(1, int(float(dur_ms) / 1000.0 / step_seconds))
            else:
                win_steps = max(1, int(60 / step_seconds))
            t0 = max(0, t0)
            t1 = min(df["t"].max() + 1, t0 + win_steps)
            base_steps = max(1, int(args.baseline_sec / step_seconds))
            b0 = max(0, t0 - base_steps)
            b1 = t0

            impact = {}
            for ts_file in Path(args.data_dir).glob("Node_*/timeseries_*.csv"):
                df_node = pd.read_csv(ts_file)
                # Support both vmf_id and board column formats
                group_col = "vmf_id" if _has_vmf_col(df_node) else "board" if _has_board_col(df_node) else None
                if group_col is None:
                    continue
                for vid, g in df_node.groupby(group_col):
                    if group_col == "vmf_id":
                        role_v = role_from_vmf_id(vid)
                        board_v = board_for_vid(vid, role_v)
                    else:
                        board_v = vid
                    base_mean = g[(g["t"] >= b0) & (g["t"] < b1)]["BBE"].mean()
                    win_mean = g[(g["t"] >= t0) & (g["t"] < t1)]["BBE"].mean()
                    if pd.isna(base_mean) or pd.isna(win_mean):
                        continue
                    delta = float(win_mean - base_mean)
                    if delta < args.affected_min_delta:
                        continue
                    best = impact.get(board_v)
                    if best is None or delta > best[0]:
                        impact[board_v] = (delta, vid)

            ranked = sorted(impact.items(), key=lambda x: x[1][0], reverse=True)
            if not args.boards_from_alarm_flow:
                targets = []
            for b, (_d, vid) in ranked[: args.affected_top]:
                if b not in [item[0] for item in targets]:
                    targets.append((b, [vid]))
            if board not in [item[0] for item in targets]:
                targets.insert(0, (board, vmf_ids))

        for item in targets:
            if len(item) == 3:
                bname, vids, windows = item
            else:
                bname, vids = item
                windows = None
            if bname in _plotted_boards:
                continue
            _plotted_boards.add(bname)
            # Use the correct event name for this board (from failure.csv lookup)
            board_evt = _board_event_map.get(bname, evt)
            plot_board(bname, board_evt, event_time, vids, windows)


if __name__ == "__main__":
    main()
