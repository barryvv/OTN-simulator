#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import textwrap
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RULES = Path("outputs/Static files/rule_database.csv")
KNOWN_EVENTS = [
    "fiber_crack",
    "fiber_cut",
    "fiber_aging",
    "line_disconnect",
    "xcon_port_down",
    "xcon_fabric_fault",
    "xcon_buffer_overflow",
]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _csv_summary(path: Path, max_rows: int = 8) -> str:
    if not path.exists():
        return f"{path}: missing"
    df = _read_csv(path)
    if df.empty:
        return f"{path}: empty"
    head = df.head(max_rows).to_string(index=False)
    return f"{path} rows={len(df)} cols={list(df.columns)}\n{head}"


def _alarm_flow_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "alarm_flow: empty"
    cols = {str(c).strip().lower(): c for c in df.columns}
    c_src = cols.get("sourceboard")
    c_prop = cols.get("proptoboard")
    c_rule = cols.get("ruleidx")
    c_hop = cols.get("hop")
    c_alarm = cols.get("propalarm")
    parts = [f"alarm_flow rows={len(df)}"]
    if c_src:
        parts.append(f"source boards={df[c_src].nunique()}")
    if c_prop:
        parts.append(f"prop boards={df[c_prop].nunique()}")
    if c_rule:
        parts.append(f"rule_idx counts={df[c_rule].value_counts().head(5).to_dict()}")
    if c_hop:
        parts.append(f"max hop={df[c_hop].max()}")
    if c_alarm:
        parts.append(f"top alarms={df[c_alarm].value_counts().head(5).to_dict()}")
    return " | ".join(parts)


def _timeseries_summary(event: str, root: Path, max_files: int = 3, top_k: int = 3) -> str:
    event_key = event.strip().lower().replace(" ", "_")
    cand_dirs = [
        root / f"data_outputs_{event_key}",
        root / event_key,
    ]
    ts_dir = next((d for d in cand_dirs if d.exists()), None)
    if not ts_dir:
        return "timeseries: no matching timeseries directory found"

    files = sorted(ts_dir.glob("Node_*/timeseries_*.csv"))
    if not files:
        return "timeseries: no timeseries files found"

    def _find_time_col(df: pd.DataFrame) -> str | None:
        for name in ("Time", "time", "timestamp", "t", "T", "step", "Step", "t_sec", "t_s"):
            if name in df.columns:
                return name
        return None

    def _format_time(val: object) -> str:
        if pd.isna(val):
            return "NA"
        if isinstance(val, (int, float)):
            return f"{float(val):.3f}s"
        try:
            t = pd.to_datetime(val, errors="coerce")
            if pd.isna(t):
                return str(val)
            return t.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(val)

    rows = []
    for f in files[:max_files]:
        df = _read_csv(f)
        if df.empty:
            rows.append(f"{f.name}: empty")
            continue
        metrics = [c for c in df.columns if c.upper() in {"BBE", "BBER", "ES", "LAT", "SCORE", "ANOMALY"}]
        stats = []
        for m in metrics:
            series = pd.to_numeric(df[m], errors="coerce").dropna()
            if series.empty:
                continue
            mean = series.mean()
            std = series.std(ddof=0)
            p95 = series.quantile(0.95)
            mx = series.max()
            stats.append(f"{m}: mean={mean:.4g} std={std:.4g} p95={p95:.4g} max={mx:.4g}")

        time_col = _find_time_col(df)
        score_col = None
        for name in ("AnomalyScore", "anomaly_score", "SCORE", "score", "ANOMALY", "anomaly"):
            if name in df.columns:
                score_col = name
                break
        if score_col and time_col:
            s = pd.to_numeric(df[score_col], errors="coerce")
            t = df[time_col]
            valid = s.notna()
            s = s[valid]
            t = t[valid]
            if not s.empty:
                top_idx = s.nlargest(top_k).index
                top_pairs = [(t.loc[i], s.loc[i]) for i in top_idx]
                earliest = min(top_pairs, key=lambda x: pd.to_datetime(x[0], errors="coerce") if not isinstance(x[0], (int, float)) else x[0])
                top_str = ", ".join([f"{_format_time(tv)}@{sv:.3g}" for tv, sv in top_pairs])
                stats.append(f"anomaly_top{top_k}={top_str}")
                stats.append(f"anomaly_earliest={_format_time(earliest[0])}@{earliest[1]:.3g}")
        elif time_col:
            # Fallback: derive a pseudo anomaly score from BBE/BBER/ES if no score column exists
            metric_cols = [c for c in df.columns if c.upper() in {"BBE", "BBER", "ES"}]
            if metric_cols:
                z_cols = []
                for m in metric_cols:
                    series = pd.to_numeric(df[m], errors="coerce")
                    mu = series.mean()
                    sigma = series.std(ddof=0)
                    denom = max(float(sigma) if pd.notna(sigma) else 0.0, 1e-9)
                    z_cols.append((series - mu).abs() / denom)
                pseudo = pd.concat(z_cols, axis=1).max(axis=1)
                top_idx = pseudo.nlargest(top_k).index
                top_pairs = [(df[time_col].loc[i], pseudo.loc[i]) for i in top_idx]
                earliest = min(top_pairs, key=lambda x: pd.to_datetime(x[0], errors="coerce") if not isinstance(x[0], (int, float)) else x[0])
                top_str = ", ".join([f"{_format_time(tv)}@{sv:.3g}" for tv, sv in top_pairs])
                stats.append(f"pseudo_anomaly_top{top_k}={top_str}")
                stats.append(f"pseudo_anomaly_earliest={_format_time(earliest[0])}@{earliest[1]:.3g}")
        rows.append(f"{f.name}: " + "; ".join(stats))
    return "timeseries sample:\n" + "\n".join(rows)


def _infer_event_from_run_dir(run_dir: Path) -> str | None:
    lowered = " ".join([run_dir.name.lower(), str(run_dir.parent).lower()])
    for ev in KNOWN_EVENTS:
        if ev in lowered:
            return ev
    return None


def _rules_summary(path: Path, max_rows: int = 8) -> str:
    if not path.exists():
        return f"{path}: missing"
    df = _read_csv(path)
    if df.empty:
        return f"{path}: empty"
    cols = {str(c).strip().lower(): c for c in df.columns}
    from_alarm = cols.get("fromalarm") or cols.get("eventname")
    from_type = cols.get("fromtype") or cols.get("functiontype")
    parts = [f"rules rows={len(df)} cols={list(df.columns)}"]
    if from_alarm:
        parts.append(f"from_alarm sample={df[from_alarm].dropna().astype(str).head(8).tolist()}")
    if from_type:
        parts.append(f"from_type sample={df[from_type].dropna().astype(str).head(8).tolist()}")
    return " | ".join(parts)


def _compare_failure_rules(failure_df: pd.DataFrame, rules_df: pd.DataFrame) -> str:
    if failure_df.empty or rules_df.empty:
        return "failure_vs_rules: skipped (missing data)"
    fcols = {str(c).strip().lower(): c for c in failure_df.columns}
    rcols = {str(c).strip().lower(): c for c in rules_df.columns}
    c_event = fcols.get("event") or fcols.get("alarm")
    c_rule_alarm = rcols.get("eventname") or rcols.get("fromalarm")
    if not c_event or not c_rule_alarm:
        return "failure_vs_rules: skipped (missing columns)"
    events = set(failure_df[c_event].dropna().astype(str).tolist())
    from_alarms = set(rules_df[c_rule_alarm].dropna().astype(str).tolist())
    missing = sorted([e for e in events if e not in from_alarms])
    return f"failure_vs_rules missing={missing}" if missing else "failure_vs_rules: all events found in rules"


def _split_role(token: str) -> tuple[str, str]:
    parts = token.rsplit("-", 1)
    if len(parts) == 2:
        role = parts[1].strip().upper()
        if role in {"SRC", "RELAY", "TERMINATION"}:
            return parts[0].strip(), role
    return token.strip(), ""


def _parse_lightpaths(path: Path) -> list[list[tuple[str, str]]]:
    if not path.exists():
        return []
    paths = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        nodes = []
        for token in [t.strip() for t in line.split(",") if t.strip()]:
            board, role = _split_role(token)
            nodes.append((board, role))
        if nodes:
            paths.append(nodes)
    return paths


def _topology_summary(paths: list[list[tuple[str, str]]]) -> str:
    if not paths:
        return "topology: missing/empty"
    boards = [b for path in paths for b, _ in path]
    roles = [r for path in paths for _, r in path if r]
    edge_count = sum(max(0, len(path) - 1) for path in paths)
    return (
        f"topology paths={len(paths)} boards={len(set(boards))} edges={edge_count} "
        f"role_counts={Counter(roles)}"
    )


def _failure_boards(failure_df: pd.DataFrame) -> list[str]:
    if failure_df.empty:
        return []
    cols = {str(c).strip().lower(): c for c in failure_df.columns}
    c_board = cols.get("board") or cols.get("sourceboard")
    if not c_board:
        return []
    return sorted(set(failure_df[c_board].dropna().astype(str).tolist()))


def _topology_impact(paths: list[list[tuple[str, str]]], failure_boards: list[str], max_show: int = 6) -> list[str]:
    if not paths or not failure_boards:
        return ["topology_impact: skipped"]
    lines = []
    for board in failure_boards:
        upstream: set[str] = set()
        downstream: set[str] = set()
        for path in paths:
            seq = [b for b, _ in path]
            for idx, node in enumerate(seq):
                if node != board:
                    continue
                upstream.update(seq[:idx])
                downstream.update(seq[idx + 1 :])
        if not upstream and not downstream:
            lines.append(f"{board}: not found in topology")
            continue
        up_list = sorted(upstream)
        down_list = sorted(downstream)
        lines.append(
            f"{board}: upstream={len(up_list)} sample={up_list[:max_show]} | "
            f"downstream={len(down_list)} sample={down_list[:max_show]}"
        )
    return lines


def _propagation_topology_summary(
    alarm_df: pd.DataFrame, paths: list[list[tuple[str, str]]], failure_boards: list[str]
) -> list[str]:
    if alarm_df.empty or not paths or not failure_boards:
        return ["propagation_vs_topology: skipped"]
    cols = {str(c).strip().lower(): c for c in alarm_df.columns}
    c_src = cols.get("sourceboard")
    c_prop = cols.get("proptoboard")
    if not c_prop:
        return ["propagation_vs_topology: skipped (missing PropToBoard)"]
    lines = []
    for board in failure_boards:
        upstream: set[str] = set()
        downstream: set[str] = set()
        for path in paths:
            seq = [b for b, _ in path]
            for idx, node in enumerate(seq):
                if node != board:
                    continue
                upstream.update(seq[:idx])
                downstream.update(seq[idx + 1 :])
        allowed = upstream | downstream | {board}
        if c_src and board in alarm_df[c_src].astype(str).unique():
            subset = alarm_df[alarm_df[c_src].astype(str) == board]
        else:
            subset = alarm_df
        prop_boards = set(subset[c_prop].dropna().astype(str).tolist())
        up_hits = len(prop_boards & upstream)
        down_hits = len(prop_boards & downstream)
        unknown = len(prop_boards - allowed)
        lines.append(
            f"{board}: prop boards={len(prop_boards)} upstream_hits={up_hits} "
            f"downstream_hits={down_hits} unknown={unknown}"
        )
        if unknown:
            sample = sorted(prop_boards - allowed)[:6]
            lines.append(f"{board}: unknown sample={sample}")
    return lines


def _local_alarm_summary(local_df: pd.DataFrame, failure_df: pd.DataFrame, window_minutes: int = 10) -> list[str]:
    if local_df.empty:
        return ["local_alarms: empty"]
    cols = {str(c).strip().lower(): c for c in local_df.columns}
    c_time = cols.get("time")
    c_board = cols.get("board")
    c_alarm = cols.get("localalarm") or cols.get("alarm")
    if not c_time or not c_board:
        return ["local_alarms: skipped (missing Time/Board)"]
    local_df = local_df.copy()
    local_df["_time"] = pd.to_datetime(local_df[c_time], errors="coerce")
    lines = [f"local_alarms rows={len(local_df)} boards={local_df[c_board].nunique()}"]
    if failure_df.empty:
        return lines
    fcols = {str(c).strip().lower(): c for c in failure_df.columns}
    f_board = fcols.get("board")
    f_time = fcols.get("time")
    if not f_board or not f_time:
        return lines
    failure_df = failure_df.copy()
    failure_df["_time"] = pd.to_datetime(failure_df[f_time], errors="coerce")
    for _, row in failure_df.iterrows():
        board = str(row[f_board])
        t0 = row["_time"]
        if pd.isna(t0):
            continue
        t1 = t0 + pd.Timedelta(minutes=window_minutes)
        subset = local_df[(local_df[c_board] == board) & (local_df["_time"] >= t0) & (local_df["_time"] <= t1)]
        if subset.empty:
            lines.append(f"{board}: no local alarms in +{window_minutes}m window")
            continue
        detail = f"{board}: alarms in +{window_minutes}m={len(subset)}"
        if c_alarm:
            top = subset[c_alarm].value_counts().head(5).to_dict()
            detail += f" top={top}"
        lines.append(detail)
    return lines


def build_prompt(run_dir: Path, question: str, rules_path: Path) -> str:
    alarm_flow = run_dir / "alarm_flow.csv"
    failure = run_dir / "failure.csv"
    local_alarms = run_dir / "local_alarms_from_data.csv"
    paths = run_dir / "mw_lightpaths_roles.txt"

    df_alarm = _read_csv(alarm_flow)
    df_fail = _read_csv(failure)
    df_rules = _read_csv(rules_path)

    paths_data = _parse_lightpaths(paths)
    failure_boards = _failure_boards(df_fail)
    context = [
        f"Run dir: {run_dir}",
        _alarm_flow_summary(df_alarm),
        _rules_summary(rules_path),
        _compare_failure_rules(df_fail, df_rules),
        _topology_summary(paths_data),
        _csv_summary(failure),
        _csv_summary(alarm_flow),
        _csv_summary(local_alarms),
    ]
    if paths.exists():
        lines = paths.read_text(encoding="utf-8", errors="ignore").splitlines()
        context.append(f"{paths} lines={len(lines)} sample={lines[:3]}")
    context.extend(_topology_impact(paths_data, failure_boards))
    context.extend(_propagation_topology_summary(df_alarm, paths_data, failure_boards))
    context.extend(_local_alarm_summary(_read_csv(local_alarms), df_fail))

    base = (
        "You are an RCA assistant for a network alarm simulator."
        " Do NOT assume the failure event is known."
    )
    context_block = f"Context:\n{chr(10).join(context)}"
    if "return:" in question.lower():
        prompt = f"""
{base}

{question}

{context_block}
"""
    else:
        prompt = f"""
{base}

User question:
{question}

{context_block}

Return:
1) likely root cause in the simulation (not real-world hardware)
2) 3 concrete fixes (ordered) referencing rule DB + topology
3) topology impact check (upstream/downstream coverage)
4) data anomalies (alarm_flow vs failure vs rules)
5) missing data needed to confirm the analysis
"""
    return textwrap.dedent(prompt).strip()


def build_prompt_predict(
    run_dir: Path,
    question: str,
    rules_path: Path,
    timeseries_root: Path,
    timeseries_event: str | None,
    timeseries_max_files: int,
    timeseries_topk: int,
) -> str:
    paths = run_dir / "mw_lightpaths_roles.txt"
    if not paths.exists():
        paths = run_dir / "lightpaths.txt"

    event = timeseries_event or _infer_event_from_run_dir(run_dir) or "unknown"
    ts_summary = _timeseries_summary(event, timeseries_root, timeseries_max_files, timeseries_topk)
    topo_summary = "topology/lightpaths: missing"
    if paths.exists():
        lines = paths.read_text(encoding="utf-8", errors="ignore").splitlines()
        lines = [ln for ln in lines if ln.strip()]
        topo_summary = f"topology/lightpaths lines={len(lines)} sample:\n" + "\n".join(lines[:3])

    # Include full rule database CSV instead of just a summary
    rules_full = "(missing rule database)"
    if rules_path.exists():
        rules_full = rules_path.read_text(encoding="utf-8", errors="ignore").strip()

    context = [
        ts_summary,
        f"Rule database (CSV):\n{rules_full}",
        topo_summary,
    ]

    alarm_flow_schema = (
        "SourceBoard,SourceAlarm,SourceEvent,SourceStartTime,SourceDurationSec,"
        "SourceRole,PropToBoard,PropAlarm,PropEvent,PropRole,ArrivalTime,ClearTime,"
        "ViaEdgeType,Layer,Direction,Severity,DelayMs,DelaySec,RuleNote,RuleIdx,Hop,PrevBoard"
    )

    base = (
        "You are an RCA assistant for a network alarm simulator. "
        "Do NOT assume the failure event is known. "
        "You only see metrics summaries, topology/lightpaths, and rule database.\n\n"
        "IMPORTANT: Use ONLY board names from the topology/lightpaths (e.g. ROADM0-OA002$0, "
        "ROADM0-FIU002$0, ROADM1-Line021$0). The timeseries files correspond to network nodes — "
        "use the topology to identify which boards are on each node."
    )
    context_block = f"Context:\n{chr(10).join(context)}"
    if "return:" in question.lower():
        prompt = f"""
{base}

{question}

{context_block}
"""
    else:
        prompt = f"""
{base}

User question:
{question}

{context_block}

Return:
1) likely root cause in the simulation (inferred from metrics)
2) 3 concrete fixes (ordered) referencing rule DB + topology
3) topology impact check (upstream/downstream coverage)
4) anomalies vs rules (expected propagation vs rules)
5) missing data needed to confirm the analysis

Predict the alarm_flow rows using rule_db + topology in this exact CSV format:
{alarm_flow_schema}
Use alarm names from the rule database OutputAlarm column (e.g. R_LOS, OTUk_BDI, ODUk_PM_AIS).
Use board names from topology (e.g. ROADM0-OA002$0). Walk the lightpath and apply matching rules at each hop.
"""
    return textwrap.dedent(prompt).strip()


def call_ollama(ollama_url: str, model: str, prompt: str, temperature: float, num_ctx: int) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    data = json.dumps(payload).encode("utf-8")
    url = ollama_url.rstrip("/") + "/api/generate"
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = resp.read().decode("utf-8")
    out = json.loads(body)
    return out.get("response", "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Local LLM log analyzer (Ollama).")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Run directory (used for lightpaths if present)")
    ap.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    ap.add_argument("--model", default="gemma3:27b")
    ap.add_argument("--ollama-url", default="http://localhost:11434",
                    help="Ollama base URL (use with SSH port-forwarding)")
    ap.add_argument("--question", default="Analyze logs and suggest fixes.")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--print-prompt", action="store_true")
    ap.add_argument("--predict-only", action="store_true",
                    help="Use metrics/topology/rules only (no failure/alarm_flow).")
    ap.add_argument("--use-failure-alarm", action="store_true",
                    help="Force inclusion of failure.csv/alarm_flow.csv in the prompt.")
    ap.add_argument("--timeseries-root", type=Path, default=Path("."),
                    help="Root folder containing data_outputs_<event> directories")
    ap.add_argument("--timeseries-event", type=str, default=None,
                    help="Event name to locate timeseries (e.g., Fiber_Cut)")
    ap.add_argument("--timeseries-max-files", type=int, default=3)
    ap.add_argument("--timeseries-topk", type=int, default=3,
                    help="Top-K anomaly times to report per timeseries file")
    args = ap.parse_args()

    if not args.run_dir.exists():
        raise SystemExit(f"Missing run dir: {args.run_dir}")

    q_lower = (args.question or "").lower()
    auto_predict = ("do not assume" in q_lower) or ("predict" in q_lower and "alarm_flow" in q_lower)
    use_predict_only = (args.predict_only or auto_predict) and not args.use_failure_alarm

    if use_predict_only:
        prompt = build_prompt_predict(
            args.run_dir,
            args.question,
            args.rules,
            args.timeseries_root,
            args.timeseries_event,
            args.timeseries_max_files,
            args.timeseries_topk,
        )
    else:
        prompt = build_prompt(args.run_dir, args.question, args.rules)
    if args.print_prompt:
        print(prompt)
        return

    try:
        response = call_ollama(args.ollama_url, args.model, prompt, args.temperature, args.num_ctx)
    except Exception as exc:
        raise SystemExit(f"Ollama request failed: {exc}")

    print(response)


if __name__ == "__main__":
    main()
