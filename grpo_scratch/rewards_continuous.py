"""Continuous-reward variant of the OTN GRPO reward functions.

The original `grpo_scratch.rewards` awards in fixed discrete chunks
(0.25 / 0.15 / 0.10 / 0.05 / 0.03). On structured CSV outputs this
creates plateaus in the reward landscape — many genuinely different
completions land on the same exact total score, so GRPO's group-relative
advantages collapse to zero and there is nothing to learn from.

This module replaces the cliffs with smooth slopes:

- Event match: similarity ratio against any event mention in the text
  (`difflib.SequenceMatcher`), scaled to 0..0.25
- Board match: similarity ratio against any ROADM-name mention in the
  text, scaled to 0..0.15
- Board coverage and alarm overlap: already continuous (Jaccard /
  weighted overlap), unchanged
- Format header: fraction of required-column names found, scaled
  to 0..0.05

Use this module instead of `grpo_scratch.rewards` when the discrete
reward causes flat training:

    from grpo_scratch.rewards_continuous import (
        otn_alarm_reward, otn_format_reward,
    )
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

# Reuse the CSV parsing helpers from the discrete reward module.
from grpo_scratch.rewards import (
    _extract_csv_block,
    _parse_csv_rows,
    _extract_boards_from_csv,
    _extract_alarms_from_csv,
)


# ─────────────────────────────────────────────────────────────
# Continuous similarity helpers
# ─────────────────────────────────────────────────────────────

_EVENT_MENTION = re.compile(
    r"(FIBER[_\s]?(?:CUT|CRACK|AGING|BREAK|DAMAGE|DEGRAD\w*|FAIL\w*)|"
    r"LINE[_\s]?(?:DISCONNECT|FAULT|FAIL\w*|DOWN)|"
    r"XCON[_\s]?(?:PORT[_\s]?DOWN|FABRIC[_\s]?FAULT|BUFFER[_\s]?OVERFLOW|"
    r"FAIL\w*|FAULT|DOWN)|"
    r"REGEN[_\s]?(?:FAIL\w*|FAULT|DOWN))",
    re.IGNORECASE,
)

_BOARD_MENTION = re.compile(r"ROADM\d+[A-Za-z0-9_$\-]*", re.IGNORECASE)


def _event_similarity(gt_event: str, text: str) -> float:
    """0..1 similarity between gt_event and the best event mention in text."""
    if not gt_event:
        return 0.0
    gt_norm = gt_event.upper().replace("_", " ").strip()
    matches = _EVENT_MENTION.findall(text.upper())
    if not matches:
        # Fall back to category fragment: pick up "FIBER" / "XCON" / "LINE"
        # mentions even without a specific event name.
        cat = gt_event.split("_")[0].upper()
        if cat and cat in text.upper():
            return 0.30  # weak signal, not zero
        return 0.0
    best = 0.0
    for m in matches:
        m_norm = m.upper().replace("_", " ").strip()
        ratio = SequenceMatcher(None, gt_norm, m_norm).ratio()
        if ratio > best:
            best = ratio
    return best


def _board_similarity(gt_board: str, text: str) -> float:
    """0..1 similarity between gt_board and the best ROADM-name in text."""
    if not gt_board:
        return 0.0
    gt_norm = gt_board.upper()
    matches = _BOARD_MENTION.findall(text)
    if not matches:
        return 0.0
    best = 0.0
    for m in matches:
        ratio = SequenceMatcher(None, gt_norm, m.upper()).ratio()
        if ratio > best:
            best = ratio
    return best


# Common column names in an alarm_flow CSV header. The reward gives
# proportional credit for how many of these appear.
_REQUIRED_HEADER_COLUMNS = (
    "SourceBoard", "SourceAlarm", "PropToBoard", "PropAlarm",
    "Hop", "Severity", "Service", "Direction", "Layer",
)


def _header_coverage(header_line: str) -> float:
    """0..1 fraction of expected columns that appear in the header line."""
    if not header_line:
        return 0.0
    fields_upper = header_line.upper()
    matched = sum(1 for col in _REQUIRED_HEADER_COLUMNS if col.upper() in fields_upper)
    return matched / len(_REQUIRED_HEADER_COLUMNS)


# ─────────────────────────────────────────────────────────────
# Reward 1: OTN alarm accuracy (continuous, weight 0.8)
# ─────────────────────────────────────────────────────────────

def otn_alarm_reward(completions, **kwargs) -> list[float]:
    """Continuous version. Same max (0.80 = 0.25 + 0.15 + 0.25 + 0.15)
    but every component is a smooth function of similarity."""
    gt_boards_list = kwargs.get("ground_truth_boards", [])
    gt_alarms_list = kwargs.get("ground_truth_alarms", [])
    gt_root_boards = kwargs.get("ground_truth_root_board", [])
    gt_root_events = kwargs.get("ground_truth_root_event", [])

    rewards: list[float] = []
    for i, completion in enumerate(completions):
        text = (
            completion[-1]["content"]
            if isinstance(completion, list) and completion
            else str(completion)
        )

        score = 0.0
        gt_board = gt_root_boards[i] if i < len(gt_root_boards) else ""
        gt_event = gt_root_events[i] if i < len(gt_root_events) else ""

        # ── 1. Event similarity (smooth, 0..0.25) ──
        score += 0.25 * _event_similarity(gt_event, text)

        # ── 2. Board similarity (smooth, 0..0.15) ──
        score += 0.15 * _board_similarity(gt_board, text)

        # ── 3. Board coverage Jaccard (already continuous, 0..0.25) ──
        gt_boards_json = gt_boards_list[i] if i < len(gt_boards_list) else "[]"
        gt_boards = set(json.loads(gt_boards_json)) if gt_boards_json else set()
        pred_boards = _extract_boards_from_csv(text)
        if gt_boards and pred_boards:
            jaccard = len(pred_boards & gt_boards) / len(pred_boards | gt_boards)
            score += 0.25 * jaccard
        elif not gt_boards and not pred_boards:
            score += 0.25

        # ── 4. Alarm name overlap (already continuous, 0..0.15) ──
        gt_alarms_json = gt_alarms_list[i] if i < len(gt_alarms_list) else "{}"
        gt_alarms = json.loads(gt_alarms_json) if gt_alarms_json else {}
        pred_alarms = _extract_alarms_from_csv(text)
        all_names = set(gt_alarms.keys()) | set(pred_alarms.keys())
        if all_names:
            overlap = sum(
                min(gt_alarms.get(k, 0), pred_alarms.get(k, 0)) for k in all_names
            )
            total = sum(
                max(gt_alarms.get(k, 0), pred_alarms.get(k, 0)) for k in all_names
            )
            if total > 0:
                score += 0.15 * (overlap / total)

        rewards.append(score)
    return rewards


# ─────────────────────────────────────────────────────────────
# Reward 2: CSV format compliance (continuous, weight 0.2)
# ─────────────────────────────────────────────────────────────

def otn_format_reward(completions, **kwargs) -> list[float]:
    """Continuous version. Header check becomes a fraction of expected
    columns matched rather than a binary 0/0.05 cliff."""
    gt_row_counts = kwargs.get("ground_truth_row_count", [])

    rewards: list[float] = []
    for i, completion in enumerate(completions):
        text = (
            completion[-1]["content"]
            if isinstance(completion, list) and completion
            else str(completion)
        )

        score = 0.0
        csv_block = _extract_csv_block(text)
        if not csv_block:
            rewards.append(0.0)
            continue

        lines = [ln for ln in csv_block.strip().split("\n") if ln.strip()]
        if not lines:
            rewards.append(0.0)
            continue

        # ── Header coverage (continuous, 0..0.05) ──
        score += 0.05 * _header_coverage(lines[0])

        # ── Valid data rows (already continuous, 0..0.10) ──
        data_lines = lines[1:] if lines[0].count(",") >= 19 else lines
        total_data = len(data_lines)
        if total_data > 0:
            valid_rows = sum(1 for ln in data_lines if ln.count(",") >= 19)
            score += 0.10 * (valid_rows / total_data)

        # ── Row count proximity (already continuous, 0..0.05) ──
        gt_count = gt_row_counts[i] if i < len(gt_row_counts) else 0
        if gt_count > 0 and total_data > 0:
            ratio = min(total_data, gt_count) / max(total_data, gt_count)
            score += 0.05 * ratio

        rewards.append(score)
    return rewards
