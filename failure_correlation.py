#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cascade / correlated failure engine for the OTN simulator.

Given a set of primary failures and configurable cascade rules, this module
resolves secondary (and higher-order) failures that may be triggered.

Cascade is **off by default** — the single-failure pipeline is unchanged
unless the caller explicitly enables it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CascadeRule:
    """One rule describing how a primary failure can trigger a secondary."""
    primary_event: str
    primary_kind: list[str]          # board kinds that can be primary, e.g. ["OA","OD"]
    secondary_event: str
    secondary_kind: list[str]        # board kinds to target, e.g. ["XCON"]
    relationship: str                # downstream | upstream | same_node | adjacent_node | same_board
    probability: float               # 0.0 – 1.0
    delay_sec: tuple[int, int]       # (min_delay, max_delay)
    condition: str | None = None     # null | same_service | same_segment


# ---------------------------------------------------------------------------
# Default cascade rules  (realistic OTN probabilities)
# ---------------------------------------------------------------------------

DEFAULT_CASCADE_RULES: list[dict[str, Any]] = [
    dict(
        primary="Fiber_Cut",
        primary_kind=["OA", "OD", "OM", "FIU"],
        secondary="XCON_Port_Down",
        secondary_kind=["XCON"],
        relationship="downstream",
        probability=0.6,
        delay_sec=[5, 30],
        condition="same_service",
    ),
    dict(
        primary="Fiber_Cut",
        primary_kind=["OA", "OD", "OM", "FIU"],
        secondary="Line_Disconnect",
        secondary_kind=["LINE"],
        relationship="downstream",
        probability=0.45,
        delay_sec=[3, 20],
    ),
    dict(
        primary="XCON_Fabric_Fault",
        primary_kind=["XCON"],
        secondary="XCON_Buffer_Overflow",
        secondary_kind=["XCON"],
        relationship="same_node",
        probability=0.35,
        delay_sec=[2, 10],
    ),
    dict(
        primary="Fiber_Crack",
        primary_kind=["OA", "OD", "OM", "FIU"],
        secondary="Fiber_Cut",
        secondary_kind=["OA", "OD", "OM", "FIU"],
        relationship="same_board",
        probability=0.25,
        delay_sec=[30, 120],
    ),
    dict(
        primary="XCON_Port_Down",
        primary_kind=["XCON"],
        secondary="XCON_Fabric_Fault",
        secondary_kind=["XCON"],
        relationship="adjacent_node",
        probability=0.10,
        delay_sec=[10, 60],
    ),
]


def _dict_to_rule(d: dict[str, Any]) -> CascadeRule:
    pk = d.get("primary_kind", d.get("primaryKind", []))
    if isinstance(pk, str):
        pk = [pk]
    sk = d.get("secondary_kind", d.get("secondaryKind", []))
    if isinstance(sk, str):
        sk = [sk]
    delay = d.get("delay_sec", d.get("delaySec", [5, 30]))
    if isinstance(delay, (list, tuple)) and len(delay) >= 2:
        delay_tup = (int(delay[0]), int(delay[1]))
    else:
        delay_tup = (5, 30)
    return CascadeRule(
        primary_event=d.get("primary", d.get("primary_event", "")),
        primary_kind=[k.upper() for k in pk],
        secondary_event=d.get("secondary", d.get("secondary_event", "")),
        secondary_kind=[k.upper() for k in sk],
        relationship=d.get("relationship", "downstream"),
        probability=float(d.get("probability", 0.0)),
        delay_sec=delay_tup,
        condition=d.get("condition"),
    )


# ---------------------------------------------------------------------------
# Load rules
# ---------------------------------------------------------------------------

def load_cascade_rules(sim_cfg: dict[str, Any] | None = None,
                       yaml_path: str | Path | None = None) -> list[CascadeRule]:
    """Load cascade rules from simulator config or standalone YAML.

    Priority: yaml_path > sim_cfg["cascade"]["rules"] > DEFAULT_CASCADE_RULES.
    """
    raw: list[dict] | None = None

    if yaml_path:
        data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            raw = data.get("cascade", {}).get("rules") or data.get("rules")
        elif isinstance(data, list):
            raw = data

    if raw is None and sim_cfg:
        cascade_sec = sim_cfg.get("cascade", {})
        raw = cascade_sec.get("rules")

    if raw is None:
        raw = DEFAULT_CASCADE_RULES

    return [_dict_to_rule(d) for d in raw]


# ---------------------------------------------------------------------------
# Board kind helper  (mirrors alarm_generator._board_kind)
# ---------------------------------------------------------------------------

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


def _parent_roadm(board: str) -> str:
    """Extract ROADM parent from board name, e.g. 'ROADM0-OA002$0' -> 'ROADM0'."""
    m = re.match(r"(ROADM\d+)", board, re.IGNORECASE)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Parse lightpaths into board → path structure
# ---------------------------------------------------------------------------

def _parse_lightpaths(paths_file: str | Path) -> list[list[str]]:
    """Return list of lightpaths, each a list of bare board names (no role suffix)."""
    paths: list[list[str]] = []
    for raw in Path(paths_file).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        boards: list[str] = []
        for token in line.split(","):
            token = token.strip()
            if not token:
                continue
            # strip role suffix  (-ODUk-Creation, -ODUk-termination, -SRC, -RELAY, etc.)
            board = re.sub(r"-(ODUk-(?:Creation|termination|Termination)|SRC|RELAY|TERMINATION)$",
                           "", token, flags=re.IGNORECASE)
            boards.append(board)
        if boards:
            paths.append(boards)
    return paths


# ---------------------------------------------------------------------------
# find_candidate_boards
# ---------------------------------------------------------------------------

def find_candidate_boards(
    primary_board: str,
    rule: CascadeRule,
    board_map: dict[str, dict[str, list[str]]],
    lightpaths: list[list[str]],
) -> list[str]:
    """Find boards matching the cascade rule's relationship and kind constraints.

    Returns list of candidate board names for the secondary failure.
    """
    primary_parent = _parent_roadm(primary_board)
    target_kinds = set(rule.secondary_kind)
    candidates: list[str] = []

    if rule.relationship == "same_board":
        # Same physical board — secondary event fires on same board
        pk = _board_kind(primary_board)
        if pk in target_kinds:
            candidates.append(primary_board)
        return candidates

    if rule.relationship == "same_node":
        # Different board on same ROADM
        if primary_parent and primary_parent in board_map:
            for kind, boards in board_map[primary_parent].items():
                if kind.upper() in target_kinds:
                    for b in boards:
                        if b != primary_board:
                            candidates.append(b)
        return candidates

    if rule.relationship == "adjacent_node":
        # Boards on neighboring ROADMs (connected via lightpath)
        neighbor_roadms: set[str] = set()
        for path in lightpaths:
            if primary_board in path:
                for b in path:
                    parent = _parent_roadm(b)
                    if parent and parent != primary_parent:
                        neighbor_roadms.add(parent)
        for roadm in neighbor_roadms:
            if roadm in board_map:
                for kind, boards in board_map[roadm].items():
                    if kind.upper() in target_kinds:
                        candidates.extend(boards)
        return candidates

    if rule.relationship in ("downstream", "upstream"):
        is_downstream = rule.relationship == "downstream"
        for path in lightpaths:
            if primary_board not in path:
                continue
            idx = path.index(primary_board)
            scan = path[idx + 1:] if is_downstream else list(reversed(path[:idx]))
            for b in scan:
                bk = _board_kind(b)
                if bk in target_kinds:
                    # condition check
                    if rule.condition == "same_service":
                        # same lightpath implies same service
                        candidates.append(b)
                    elif rule.condition == "same_segment":
                        if _parent_roadm(b) != primary_parent:
                            candidates.append(b)
                    else:
                        candidates.append(b)
                    break  # first match per path

    # deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# resolve_cascades
# ---------------------------------------------------------------------------

def _role_for_kind(kind: str) -> str:
    """Assign a VMF role based on board kind (for failure.csv BoardRole)."""
    k = kind.upper()
    if k in {"OA", "OM", "OD", "FIU", "SC2", "REGENIN", "REGENOUT"}:
        return "RELAY"
    if k in {"TRIB", "TRIBUTARY"}:
        return "SRC"
    return "SRC"


def resolve_cascades(
    primary_failures: list[dict[str, Any]],
    cascade_rules: list[CascadeRule],
    board_map: dict[str, dict[str, list[str]]],
    lightpaths: list[list[str]],
    rng: np.random.Generator,
    max_cascade_depth: int = 2,
) -> list[dict[str, Any]]:
    """BFS cascade resolution.

    Given primary failures and cascade rules, generate secondary failure rows.
    Returns ONLY the additional (cascaded) failure rows — does NOT include primaries.

    Each returned row has the standard failure.csv columns plus:
      - CascadedFrom: board name of the triggering failure
      - CascadeDepth: 1 = direct cascade, 2 = cascade-of-cascade
      - CascadeRuleIdx: index into cascade_rules for audit trail
    """
    cascaded: list[dict[str, Any]] = []
    # Track which (board, event) pairs already exist to avoid duplicates
    existing: set[tuple[str, str]] = set()
    for f in primary_failures:
        existing.add((f.get("Board", ""), f.get("Event", "")))

    # BFS queue: (failure_row_dict, depth)
    queue: list[tuple[dict[str, Any], int]] = [(f, 0) for f in primary_failures]

    while queue:
        failure, depth = queue.pop(0)
        if depth >= max_cascade_depth:
            continue

        f_board = failure.get("Board", "")
        f_event = failure.get("Event", "")
        f_time_str = failure.get("Time", "")
        f_kind = _board_kind(f_board)

        # Try to parse the failure time
        try:
            f_time = datetime.strptime(f_time_str, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            f_time = datetime(2025, 1, 15, 10, 0, 0)

        for rule_idx, rule in enumerate(cascade_rules):
            if rule.primary_event != f_event:
                continue
            if f_kind not in rule.primary_kind:
                continue

            candidates = find_candidate_boards(f_board, rule, board_map, lightpaths)
            if not candidates:
                continue

            for cand_board in candidates:
                # Probability check
                if rng.random() > rule.probability:
                    continue

                # Skip if this (board, event) already exists
                key = (cand_board, rule.secondary_event)
                if key in existing:
                    continue
                existing.add(key)

                # Compute delay
                delay_sec = int(rng.integers(rule.delay_sec[0], rule.delay_sec[1] + 1))
                cascade_time = f_time + timedelta(seconds=delay_sec)

                dur_ms = int(rng.integers(5000, 20001))
                cand_kind = _board_kind(cand_board)
                role = _role_for_kind(cand_kind)

                cascade_row = {
                    "Board": cand_board,
                    "Event": rule.secondary_event,
                    "Severity": "Critical",
                    "Time": cascade_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "DurationMs": dur_ms,
                    "BoardRole": role,
                    "CascadedFrom": f_board,
                    "CascadeDepth": depth + 1,
                    "CascadeRuleIdx": rule_idx,
                }
                cascaded.append(cascade_row)
                # Enqueue for further cascading
                queue.append((cascade_row, depth + 1))

    return cascaded
