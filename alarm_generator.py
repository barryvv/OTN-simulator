#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal rule-driven alarm propagation for otn_simulator.py."""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass
class Rule:
    from_type: str
    from_alarm: str
    action: str
    to_type: str
    to_alarm: str
    idx: int


def _norm(s: object) -> str:
    return str(s).strip()


def _normalize_board_id(board: str) -> str:
    out = board
    patterns = [
        ("RegenIn", 5),
        ("RegenOut", 5),
        ("Tributary", 3),
        ("XCON", 3),
        ("Line", 3),
        ("OM", 3),
        ("OA", 3),
        ("OD", 3),
        ("FIU", 3),
    ]
    for name, width in patterns:
        out = re.sub(
            rf"({name})(\d+)",
            lambda m: f"{m.group(1)}{m.group(2).zfill(width)}",
            out,
            flags=re.IGNORECASE,
        )
    return out


def load_rule_database(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df


def _compile_rules(df: pd.DataFrame) -> list[Rule]:
    rules: list[Rule] = []
    for idx, row in df.iterrows():
        ftype = _norm(row.get("FunctionType", ""))
        if not ftype or ftype.startswith("#"):
            continue
        from_alarm = _norm(row.get("EventName", ""))
        action = _norm(row.get("ActionType", ""))
        to_type = _norm(row.get("ActionTarget", ""))
        to_alarm = _norm(row.get("OutputAlarm", ""))
        if not from_alarm or not action or not to_type or not to_alarm:
            continue
        rules.append(Rule(ftype, from_alarm, action, to_type, to_alarm, idx))
    return rules


def _split_role(token: str) -> tuple[str, str]:
    # Try two-segment suffix first: "...-ODUk-Creation", "...-ODUk-termination"
    parts2 = token.rsplit("-", 2)
    if len(parts2) == 3:
        suffix = f"{parts2[1].strip()}-{parts2[2].strip()}".upper()
        if suffix in {"ODUK-CREATION", "ODUK_CREATION"}:
            return _normalize_board_id(parts2[0].strip()), "SRC"
        if suffix in {"ODUK-TERMINATION", "ODUK_TERMINATION"}:
            return _normalize_board_id(parts2[0].strip()), "TERMINATION"
    # Single-segment suffix: "...-SRC", "...-RELAY", "...-TERMINATION"
    parts = token.rsplit("-", 1)
    if len(parts) == 2:
        role = parts[1].strip().upper()
        if role in {"SRC", "RELAY", "TERMINATION"}:
            return _normalize_board_id(parts[0].strip()), role
    return _normalize_board_id(token.strip()), ""


def _board_kind(board: str) -> str:
    b = board.upper()
    if "REGENIN" in b:
        return "REGENIN"
    if "REGENOUT" in b:
        return "REGENOUT"
    if "TRIBUTARY" in b:
        return "TRIB"
    if "XCON" in b:
        return "XCON"
    if "LINE" in b:
        return "LINE"
    if "OM" in b:
        return "OM"
    if "OA" in b:
        return "OA"
    if "OD" in b:
        return "OD"
    if "FIU" in b:
        return "FIU"
    return "OTHER"


def _board_type(board: str, role: str) -> str:
    role = role.upper()
    if role == "SRC":
        return "ODUk-Creation"
    if role == "TERMINATION":
        return "ODUk-Termination"
    kind = _board_kind(board)
    if kind == "REGENIN":
        return "OTUk-RegenIn"
    if kind == "REGENOUT":
        return "OTUk-RegenOut"
    if kind in {"OM", "OA", "OD", "FIU"}:
        return "OTUk-Relay"
    if kind in {"TRIB", "XCON", "LINE"}:
        return "ODUk-Creation"
    return "OTUk-Relay"


def _edge_type(a: str, b: str) -> str:
    ka, kb = _board_kind(a), _board_kind(b)
    if ka == "REGENIN" and kb == "REGENOUT":
        return "regen_boundary"
    return f"{ka}_{kb}"


def _parse_paths(paths_file: str | Path) -> tuple[list[list[tuple[str, str]]], dict[str, str]]:
    paths: list[list[tuple[str, str]]] = []
    role_map: dict[str, str] = {}
    for raw in Path(paths_file).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        nodes: list[tuple[str, str]] = []
        for token in [t.strip() for t in line.split(",") if t.strip()]:
            board, role = _split_role(token)
            nodes.append((board, role))
            if role and board not in role_map:
                role_map[board] = role
        if nodes:
            paths.append(nodes)
    return paths, role_map


def _build_adjacency(paths: list[list[tuple[str, str]]]) -> dict[str, list[tuple[str, str, str]]]:
    adj: dict[str, list[tuple[str, str, str]]] = {}
    for path in paths:
        boards = [b for b, _ in path]
        for i in range(len(boards) - 1):
            a = boards[i]
            b = boards[i + 1]
            et = _edge_type(a, b)
            adj.setdefault(a, []).append((b, "D", et))
            adj.setdefault(b, []).append((a, "U", et))
    return adj


def _path_targets(
    paths: list[list[tuple[str, str]]],
    role_map: dict[str, str],
    board: str,
    direction: str,
    to_type: str,
) -> list[str]:
    direction = direction.upper()
    targets: set[str] = set()
    for path in paths:
        seq = [b for b, _ in path]
        for idx, node in enumerate(seq):
            if node != board:
                continue
            scan = seq[idx + 1 :] if direction == "D" else list(reversed(seq[:idx]))
            for cand in scan:
                cand_role = _role_for_board(cand, role_map)
                if _board_type(cand, cand_role) == to_type:
                    targets.add(cand)
                    break
    return sorted(targets)


def _path_targets_with_edge(
    paths: list[list[tuple[str, str]]],
    role_map: dict[str, str],
    adj: dict[str, list[tuple[str, str, str]]],
    board: str,
    direction: str,
    to_type: str,
    block_at_regen: bool = False,
) -> list[tuple[str, str]]:
    direction = direction.upper()
    targets: dict[str, str] = {}
    edge_lookup: dict[tuple[str, str], str] = {}
    for src, neighs in adj.items():
        for neigh, edge_dir, et in neighs:
            edge_lookup[(src, neigh, edge_dir)] = et
    for path in paths:
        seq = [b for b, _ in path]
        for idx, node in enumerate(seq):
            if node != board:
                continue
            if direction == "D":
                scan = seq[idx + 1 :]
                next_hop = seq[idx + 1] if idx + 1 < len(seq) else None
                hop_dir = "D"
            else:
                scan = list(reversed(seq[:idx]))
                next_hop = seq[idx - 1] if idx - 1 >= 0 else None
                hop_dir = "U"
            if not scan:
                continue
            edge_type = ""
            if next_hop:
                edge_type = edge_lookup.get((board, next_hop, hop_dir), "")
            for cand in scan:
                cand_role = _role_for_board(cand, role_map)
                cand_type = _board_type(cand, cand_role)
                if cand_type == to_type:
                    targets.setdefault(cand, edge_type)
                    break
                if block_at_regen:
                    cand_kind = _board_kind(cand)
                    if cand_kind in {"REGENIN", "REGENOUT"}:
                        break  # section boundary — don't cross
    return sorted(targets.items())


def _read_failures_csv(path: Optional[str | Path]) -> list[dict[str, str]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _read_local_alarms(path: Optional[str | Path]) -> pd.DataFrame:
    return pd.DataFrame()


def _parse_time(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, errors="coerce")


def _format_time(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d %H:%M:%S.%f")


def _role_for_board(board: str, role_map: dict[str, str]) -> str:
    return role_map.get(board, "")


def _layer_for_type(t: str) -> str:
    return "optical" if "OTUk" in t or "Relay" in t else "electrical"


def _severity_for_alarm(alarm: str) -> str:
    a = alarm.upper()
    if "LOS" in a or "LOF" in a or "AIS" in a:
        return "Critical"
    if "BDI" in a:
        return "Major"
    if "DEG" in a:
        return "Minor"
    return ""


def generate_alarm_flow(
    graphml_path: str,
    df_rules: pd.DataFrame,
    out_csv: str,
    failures_csv: Optional[str] = None,
    paths_file: Optional[str] = None,
    bbe_local_csv: Optional[str] = None,
    max_hops: int = 4,
    rng_seed: int = 2025,
) -> None:
    random.seed(rng_seed)
    rules = _compile_rules(df_rules)
    if not rules:
        Path(out_csv).write_text("", encoding="utf-8")
        return

    if not paths_file:
        paths_file = graphml_path
    paths, role_map = _parse_paths(paths_file)
    adj = _build_adjacency(paths)

    failures = _read_failures_csv(failures_csv)
    local_df = pd.DataFrame()

    rows: list[dict[str, object]] = []
    queue: list[dict[str, object]] = []

    def enqueue_seed(board: str, alarm: str, start: pd.Timestamp, duration_s: float, role: str) -> None:
        queue.append(
            dict(
                board=board,
                alarm=alarm,
                event=alarm,
                start=start,
                duration_s=duration_s,
                role=role,
                hop=0,
                prev="",
                rule_idx=-1,
                rule_note="seed",
                edge_type="",
                direction="",
                delay_s=0.0,
                root_board=board,
                root_event=alarm,
            )
        )

    for fr in failures:
        board = _normalize_board_id(_norm(fr.get("Board", "")))
        event = _norm(fr.get("Event", ""))
        t = _parse_time(_norm(fr.get("Time", "")))
        dur_ms = fr.get("DurationMs")
        try:
            duration_s = float(dur_ms) / 1000.0 if dur_ms else 15.0
        except Exception:
            duration_s = 15.0
        role = _norm(fr.get("BoardRole", "")) or _role_for_board(board, role_map)
        if board and event and not pd.isna(t):
            enqueue_seed(board, event, t, duration_s, role)

    # Intentionally ignore BBE/local alarms; propagate only from explicit failures.

    seen: set[tuple[str, str, str, int]] = set()

    while queue:
        cur = queue.pop(0)
        board = cur["board"]
        alarm = cur["alarm"]
        start = cur["start"]
        duration_s = float(cur["duration_s"])
        role = cur["role"]
        hop = int(cur["hop"])
        prev = cur["prev"]
        rule_idx = cur["rule_idx"]
        rule_note = cur["rule_note"]
        edge_type = cur["edge_type"]
        direction = cur["direction"]
        delay_s = float(cur["delay_s"])

        root_board = cur.get("root_board", board)
        root_event = cur.get("root_event", alarm)

        arrival = start + pd.Timedelta(seconds=delay_s)
        clear = arrival + pd.Timedelta(seconds=duration_s)

        row_key = (board, alarm, _format_time(arrival), hop)
        if row_key in seen:
            continue
        seen.add(row_key)

        rows.append(
            dict(
                SourceBoard=board,
                SourceAlarm=alarm,
                SourceEvent=cur["event"],
                SourceStartTime=_format_time(start),
                SourceDurationSec=duration_s,
                SourceRole=role,
                PropToBoard=board,
                PropAlarm=alarm,
                PropEvent=alarm,
                PropRole=role,
                ArrivalTime=_format_time(arrival),
                ClearTime=_format_time(clear),
                ViaEdgeType=edge_type,
                Layer=_layer_for_type(_board_type(board, role)),
                Direction=direction,
                Severity=_severity_for_alarm(alarm),
                DelayMs=delay_s * 1000.0,
                DelaySec=delay_s,
                RuleNote=rule_note,
                RuleIdx=rule_idx,
                Hop=hop,
                PrevBoard=prev,
                RootCauseBoard=root_board,
                RootCauseEvent=root_event,
            )
        )

        if hop >= max_hops:
            continue

        board_type = _board_type(board, role)
        for rule in rules:
            if rule.from_type != board_type or rule.from_alarm != alarm:
                continue
            action = rule.action.lower()
            to_alarm = rule.to_alarm
            to_type = rule.to_type
            if "report locally" in action:
                queue.append(
                    dict(
                        board=board,
                        alarm=to_alarm,
                        event=alarm,
                        start=arrival,
                        duration_s=duration_s,
                        role=role,
                        hop=hop + 1,
                        prev=board,
                        rule_idx=rule.idx,
                        rule_note=f"{rule.from_type} / {rule.action} / {rule.to_type}",
                        edge_type="",
                        direction="",
                        delay_s=0.0,
                        root_board=root_board,
                        root_event=root_event,
                    )
                )
                continue
            if "regenout" in action:
                # Transmit Downstream RegenOut: target the paired RegenOut
                # via the regen_boundary edge in adjacency
                for neigh, edge_dir, et in adj.get(board, []):
                    if et != "regen_boundary" or edge_dir != "D":
                        continue
                    neigh_role = _role_for_board(neigh, role_map)
                    dly = 1.0
                    queue.append(
                        dict(
                            board=neigh,
                            alarm=to_alarm,
                            event=alarm,
                            start=arrival,
                            duration_s=duration_s,
                            role=neigh_role,
                            hop=hop + 1,
                            prev=board,
                            rule_idx=rule.idx,
                            rule_note=f"{rule.from_type} / {rule.action} / {rule.to_type}",
                            edge_type="regen_boundary",
                            direction="D",
                            delay_s=dly,
                            root_board=root_board,
                            root_event=root_event,
                        )
                    )
                continue
            want_dir = "U" if "upstream" in action else "D"
            block_regen = board_type == "OTUk-Relay"
            targets = _path_targets_with_edge(paths, role_map, adj, board, want_dir, to_type, block_at_regen=block_regen)
            for neigh, et in targets:
                neigh_role = _role_for_board(neigh, role_map)
                dly = random.uniform(1.0, 8.0)
                queue.append(
                    dict(
                        board=neigh,
                        alarm=to_alarm,
                        event=alarm,
                        start=arrival,
                        duration_s=duration_s,
                        role=neigh_role,
                        hop=hop + 1,
                        prev=board,
                        rule_idx=rule.idx,
                        rule_note=f"{rule.from_type} / {rule.action} / {rule.to_type}",
                        edge_type=et,
                        direction=want_dir,
                        delay_s=dly,
                        root_board=root_board,
                        root_event=root_event,
                    )
                )

    out = pd.DataFrame(rows)
    if out.empty:
        Path(out_csv).write_text("", encoding="utf-8")
        return
    cols = [
        "SourceBoard", "SourceAlarm", "SourceEvent", "SourceStartTime", "SourceDurationSec",
        "SourceRole", "PropToBoard", "PropAlarm", "PropEvent", "PropRole", "ArrivalTime",
        "ClearTime", "ViaEdgeType", "Layer", "Direction", "Severity", "DelayMs", "DelaySec",
        "RuleNote", "RuleIdx", "Hop", "PrevBoard", "RootCauseBoard", "RootCauseEvent",
    ]
    out = out[cols]
    out.to_csv(out_csv, index=False)
