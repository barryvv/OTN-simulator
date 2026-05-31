"""OTN-specific reward functions for the from-scratch GRPO trainer.

These functions score a list of completion strings against per-example
ground-truth fields and return a list of floats. Both `otn_alarm_reward`
(accuracy, weight 0.8) and `otn_format_reward` (CSV format, weight 0.2)
are duplicated from `train_grpo_gemma3b.py` on purpose: bundling them in
the `grpo_scratch/` module makes the from-scratch trainer fully
self-contained, so it does not transitively depend on `trl` (which the
production trainer imports at module load time).
"""

from __future__ import annotations

import json
import re


# ─────────────────────────────────────────────────────────────
# CSV parsing helpers
# ─────────────────────────────────────────────────────────────

def _extract_csv_block(text: str) -> str:
    """Extract alarm_flow CSV block from model completion."""
    markers = [
        "alarm_flow rows (CSV):",
        "alarm_flow rows:",
        "alarm_flow CSV:",
        "Predicted alarm_flow",
        "SourceBoard,SourceAlarm",
    ]
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            block = text[idx + len(marker):]
            end_markers = ["\nRCA reasoning:", "\nFix ideas:", "\nFix ", "\n\n\n", "\n---"]
            end_idx = len(block)
            for em in end_markers:
                ei = block.find(em)
                if 0 < ei < end_idx:
                    end_idx = ei
            return block[:end_idx].strip()
    # Fallback: lines with 19+ commas (header has 19+ columns).
    lines = [ln for ln in text.split("\n") if ln.count(",") >= 19]
    return "\n".join(lines) if lines else ""


def _parse_csv_rows(csv_block: str) -> list[dict]:
    lines = [ln.strip() for ln in csv_block.strip().split("\n") if ln.strip()]
    if not lines:
        return []
    header = lines[0].split(",")
    rows: list[dict] = []
    for ln in lines[1:]:
        fields = ln.split(",")
        if len(fields) >= min(len(header), 6):
            row = {
                header[i].strip(): fields[i].strip()
                for i in range(min(len(header), len(fields)))
            }
            rows.append(row)
    return rows


def _extract_boards_from_csv(text: str) -> set[str]:
    csv_block = _extract_csv_block(text)
    rows = _parse_csv_rows(csv_block)
    boards: set[str] = set()
    for r in rows:
        b = r.get("PropToBoard", "").strip()
        if b and b.startswith("ROADM"):
            boards.add(b)
    return boards


def _extract_alarms_from_csv(text: str) -> dict[str, int]:
    csv_block = _extract_csv_block(text)
    rows = _parse_csv_rows(csv_block)
    counts: dict[str, int] = {}
    for r in rows:
        a = r.get("PropAlarm", "").strip()
        if a:
            counts[a] = counts.get(a, 0) + 1
    return counts


# ─────────────────────────────────────────────────────────────
# Reward 1: OTN alarm accuracy (default weight 0.8)
# ─────────────────────────────────────────────────────────────

def otn_alarm_reward(completions, **kwargs) -> list[float]:
    """Score completions on root cause accuracy, board coverage, and alarm
    accuracy. Sub-rewards: event match (0.25), board match (0.15),
    board coverage Jaccard (0.25), alarm-name overlap (0.15)."""
    gt_boards_list = kwargs.get("ground_truth_boards", [])
    gt_alarms_list = kwargs.get("ground_truth_alarms", [])
    gt_root_boards = kwargs.get("ground_truth_root_board", [])
    gt_root_events = kwargs.get("ground_truth_root_event", [])

    rewards: list[float] = []
    for i, completion in enumerate(completions):
        text = completion[-1]["content"] if isinstance(completion, list) and completion else str(completion)

        score = 0.0
        gt_board = gt_root_boards[i] if i < len(gt_root_boards) else ""
        gt_event = gt_root_events[i] if i < len(gt_root_events) else ""

        # ── 1. Root cause event match (0.25) ──
        text_upper = text.upper()
        gt_event_upper = gt_event.upper().replace("_", " ")
        if gt_event_upper in text_upper or gt_event.upper() in text_upper:
            score += 0.25
        else:
            gt_cat = gt_event.split("_")[0].upper()
            specific_events = re.findall(
                r"(FIBER[_ ]?CUT|FIBER[_ ]?CRACK|FIBER[_ ]?AGING|"
                r"LINE[_ ]?DISCONNECT|XCON[_ ]?PORT[_ ]?DOWN|"
                r"XCON[_ ]?FABRIC[_ ]?FAULT|XCON[_ ]?BUFFER[_ ]?OVERFLOW)",
                text_upper,
            )
            if specific_events:
                pred_cat = specific_events[0].split("_")[0].split(" ")[0]
                if pred_cat == gt_cat:
                    score += 0.10
            elif gt_cat in text_upper:
                score += 0.03

        # ── 2. Root cause board match (0.15) ──
        if gt_board and gt_board in text:
            score += 0.15
        elif gt_board:
            roadm = gt_board.split("-")[0] if "-" in gt_board else ""
            if roadm and roadm in text:
                score += 0.05

        # ── 3. Board coverage Jaccard (0.25) ──
        gt_boards_json = gt_boards_list[i] if i < len(gt_boards_list) else "[]"
        gt_boards = set(json.loads(gt_boards_json)) if gt_boards_json else set()
        pred_boards = _extract_boards_from_csv(text)
        if gt_boards and pred_boards:
            jaccard = len(pred_boards & gt_boards) / len(pred_boards | gt_boards)
            score += 0.25 * jaccard
        elif not gt_boards and not pred_boards:
            score += 0.25

        # ── 4. Alarm name accuracy (0.15) ──
        gt_alarms_json = gt_alarms_list[i] if i < len(gt_alarms_list) else "{}"
        gt_alarms = json.loads(gt_alarms_json) if gt_alarms_json else {}
        pred_alarms = _extract_alarms_from_csv(text)
        all_names = set(gt_alarms.keys()) | set(pred_alarms.keys())
        if all_names:
            overlap = sum(min(gt_alarms.get(k, 0), pred_alarms.get(k, 0)) for k in all_names)
            total = sum(max(gt_alarms.get(k, 0), pred_alarms.get(k, 0)) for k in all_names)
            score += 0.15 * (overlap / total) if total > 0 else 0.0

        rewards.append(score)
    return rewards


# ─────────────────────────────────────────────────────────────
# Reward 2: CSV format compliance (default weight 0.2)
# ─────────────────────────────────────────────────────────────

def otn_format_reward(completions, **kwargs) -> list[float]:
    """Score completions on CSV format quality: header check (0.05),
    fraction of valid 19+-comma rows (0.10), row-count proximity (0.05)."""
    gt_row_counts = kwargs.get("ground_truth_row_count", [])

    rewards: list[float] = []
    for i, completion in enumerate(completions):
        text = completion[-1]["content"] if isinstance(completion, list) and completion else str(completion)

        score = 0.0
        csv_block = _extract_csv_block(text)
        if not csv_block:
            rewards.append(0.0)
            continue

        lines = [ln for ln in csv_block.strip().split("\n") if ln.strip()]
        if not lines:
            rewards.append(0.0)
            continue

        # Header check (0.05)
        header = lines[0]
        if header.count(",") >= 19 and "SourceBoard" in header:
            score += 0.05

        # Valid data rows (0.10)
        data_lines = lines[1:] if header.count(",") >= 19 else lines
        total_data = len(data_lines)
        if total_data > 0:
            valid_rows = sum(1 for ln in data_lines if ln.count(",") >= 19)
            score += 0.10 * (valid_rows / total_data)

        # Row count proximity (0.05)
        gt_count = gt_row_counts[i] if i < len(gt_row_counts) else 0
        if gt_count > 0 and total_data > 0:
            ratio = min(total_data, gt_count) / max(total_data, gt_count)
            score += 0.05 * ratio

        rewards.append(score)
    return rewards
