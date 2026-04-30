#!/usr/bin/env python3
"""Automated evaluation framework for the OTN RCA system.

Loads test JSONL or pre-computed model outputs, parses predictions,
computes accuracy / F1 / format metrics, and writes result files.

Usage (cached outputs):
  python evaluate.py \
    --test-file outputs/grpo_data/test.jsonl \
    --outputs-file outputs/eval/model_outputs.jsonl \
    --results-dir outputs/eval/ \
    --model-name "SFT+GRPO v6.1"

Usage (live inference):
  python evaluate.py \
    --test-file outputs/grpo_data/test.jsonl \
    --run-inference \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --adapter adapters/qwen-3b-grpo \
    --results-dir outputs/eval/ \
    --model-name "GRPO v6"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

EVENT_TYPES = [
    "Fiber_Aging",
    "Fiber_Crack",
    "Fiber_Cut",
    "Line_Disconnect",
    "XCON_Port_Down",
    "XCON_Buffer_Overflow",
    "XCON_Fabric_Fault",
]

CATEGORY_MAP = {
    "Fiber_Aging": "Fiber",
    "Fiber_Crack": "Fiber",
    "Fiber_Cut": "Fiber",
    "Line_Disconnect": "Line",
    "XCON_Port_Down": "XCON",
    "XCON_Buffer_Overflow": "XCON",
    "XCON_Fabric_Fault": "XCON",
}


# ─────────────────────────────────────────────────────────────
# CSV extraction helpers (aligned with train_grpo_gemma3b.py)
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
            end_markers = [
                "\nRCA reasoning:",
                "\nFix ideas:",
                "\nFix ",
                "\n\n\n",
                "\n---",
            ]
            end_idx = len(block)
            for em in end_markers:
                ei = block.find(em)
                if 0 < ei < end_idx:
                    end_idx = ei
            return block[:end_idx].strip()
    # Fallback: lines with 19+ commas
    lines = [ln for ln in text.split("\n") if ln.count(",") >= 19]
    return "\n".join(lines) if lines else ""


def _parse_csv_rows(csv_block: str) -> list[dict]:
    """Parse CSV block into list of dicts with basic column mapping."""
    lines = [ln.strip() for ln in csv_block.strip().split("\n") if ln.strip()]
    if not lines:
        return []
    header = lines[0].split(",")
    rows = []
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
    """Extract unique PropToBoard values from CSV in completion."""
    csv_block = _extract_csv_block(text)
    rows = _parse_csv_rows(csv_block)
    boards: set[str] = set()
    for r in rows:
        b = r.get("PropToBoard", "").strip()
        if b and b.startswith("ROADM"):
            boards.add(b)
        # Also capture SourceBoard
        sb = r.get("SourceBoard", "").strip()
        if sb and sb.startswith("ROADM"):
            boards.add(sb)
    return boards


def _extract_alarms_from_csv(text: str) -> dict[str, int]:
    """Extract PropAlarm counts from CSV in completion."""
    csv_block = _extract_csv_block(text)
    rows = _parse_csv_rows(csv_block)
    counts: dict[str, int] = {}
    for r in rows:
        a = r.get("PropAlarm", "").strip()
        if a:
            counts[a] = counts.get(a, 0) + 1
    return counts


def _extract_severity_from_csv(text: str) -> dict[str, int]:
    """Extract Severity counts from CSV in completion."""
    csv_block = _extract_csv_block(text)
    rows = _parse_csv_rows(csv_block)
    counts: dict[str, int] = {}
    for r in rows:
        s = r.get("Severity", "").strip()
        if s:
            counts[s] = counts.get(s, 0) + 1
    return counts


# ─────────────────────────────────────────────────────────────
# Prediction extraction from free-text model output
# ─────────────────────────────────────────────────────────────

def extract_predicted_event(text: str) -> str | None:
    """Extract predicted root cause event type from model output text.

    Prioritises mentions near "Root cause" / "Event:" lines.
    Falls back to the first event type found anywhere in the text.
    """
    if not text:
        return None

    # Pass 1 — look near "Root cause" or "Event:" lines
    rca_patterns = [
        r"[Rr]oot\s*[Cc]ause.*",
        r"[Ee]vent\s*:\s*.*",
        r"[Ff]ailure\s*[Tt]ype\s*:\s*.*",
        r"[Rr]oot\s*[Ee]vent\s*:\s*.*",
    ]
    for pat in rca_patterns:
        m = re.search(pat, text)
        if m:
            snippet = m.group(0)
            for evt in EVENT_TYPES:
                # Match with underscores or spaces
                if evt in snippet or evt.replace("_", " ") in snippet:
                    return evt

    # Pass 2 — first event type anywhere in text
    for evt in EVENT_TYPES:
        if evt in text or evt.replace("_", " ") in text:
            return evt

    # Pass 3 — case-insensitive search with underscores or spaces
    text_upper = text.upper()
    for evt in EVENT_TYPES:
        if evt.upper() in text_upper or evt.upper().replace("_", " ") in text_upper:
            return evt

    return None


def extract_predicted_board(text: str) -> str | None:
    """Extract predicted root cause board from model output text.

    Looks for ROADM board pattern near "Board:" lines first,
    then falls back to the first ROADM board mentioned.
    """
    if not text:
        return None

    # Pass 1 — near "Board:" or "Root cause board:" lines
    board_patterns = [
        r"[Bb]oard\s*:\s*(ROADM\d+-\w+\$\d+)",
        r"[Rr]oot\s*[Cc]ause\s*[Bb]oard\s*:\s*(ROADM\d+-\w+\$\d+)",
        r"[Pp]eak[_ ]?board\s*:\s*(ROADM\d+-\w+\$\d+)",
        r"[Ss]ource[_ ]?[Bb]oard\s*:\s*(ROADM\d+-\w+\$\d+)",
    ]
    for pat in board_patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)

    # Pass 2 — first ROADM board mentioned anywhere
    m = re.search(r"(ROADM\d+-\w+\$\d+)", text)
    if m:
        return m.group(1)

    return None


# ─────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────

def compute_board_f1(pred_boards: set[str], gt_boards: set[str]) -> dict[str, float]:
    """Compute precision, recall, F1 on board sets."""
    if not pred_boards and not gt_boards:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_boards or not gt_boards:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = len(pred_boards & gt_boards)
    precision = overlap / len(pred_boards)
    recall = overlap / len(gt_boards)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_alarm_f1(pred_alarms: dict[str, int], gt_alarms: dict[str, int]) -> float:
    """Compute alarm F1 using overlap/total approach (min/max per alarm type)."""
    all_names = set(pred_alarms.keys()) | set(gt_alarms.keys())
    if not all_names:
        return 1.0  # Both empty
    overlap = sum(
        min(pred_alarms.get(k, 0), gt_alarms.get(k, 0)) for k in all_names
    )
    total = sum(
        max(pred_alarms.get(k, 0), gt_alarms.get(k, 0)) for k in all_names
    )
    if total == 0:
        return 1.0
    return overlap / total


def compute_format_score(text: str, gt_row_count: int) -> float:
    """Score CSV format quality (0.0 to 1.0)."""
    csv_block = _extract_csv_block(text)
    if not csv_block:
        return 0.0

    lines = [ln for ln in csv_block.strip().split("\n") if ln.strip()]
    if not lines:
        return 0.0

    score = 0.0
    header = lines[0]

    # Header check (0.25 of format score)
    has_header = header.count(",") >= 19 and "SourceBoard" in header
    if has_header:
        score += 0.25

    # Valid data rows (0.50 of format score)
    data_lines = lines[1:] if has_header else lines
    total_data = len(data_lines)
    if total_data > 0:
        valid_rows = sum(1 for ln in data_lines if ln.count(",") >= 19)
        score += 0.50 * (valid_rows / total_data)

    # Row count proximity (0.25 of format score)
    if gt_row_count > 0 and total_data > 0:
        ratio = min(total_data, gt_row_count) / max(total_data, gt_row_count)
        score += 0.25 * ratio

    return score


def compute_severity_f1(
    pred_severity: dict[str, int], gt_severity: dict[str, int]
) -> float:
    """Compute severity distribution F1 using overlap/total approach."""
    all_keys = set(pred_severity.keys()) | set(gt_severity.keys())
    if not all_keys:
        return 1.0
    overlap = sum(
        min(pred_severity.get(k, 0), gt_severity.get(k, 0)) for k in all_keys
    )
    total = sum(
        max(pred_severity.get(k, 0), gt_severity.get(k, 0)) for k in all_keys
    )
    if total == 0:
        return 1.0
    return overlap / total


# ─────────────────────────────────────────────────────────────
# Inference runner (optional, requires GPU + transformers)
# ─────────────────────────────────────────────────────────────

def run_inference_batch(
    examples: list[dict],
    base_model: str,
    adapter: str | None,
    max_new_tokens: int = 2048,
    temperature: float = 0.3,
    batch_size: int = 1,
) -> list[dict]:
    """Run model inference on a list of examples. Returns output dicts.

    Requires torch, transformers, peft. Only call with --run-inference.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise SystemExit(
            f"Inference requires torch, transformers, peft. Missing: {e}"
        )

    print(f"[info] loading tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"[info] loading model from {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter:
        print(f"[info] loading adapter from {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    outputs = []
    total = len(examples)
    for idx, ex in enumerate(examples):
        messages = ex["prompt"]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        # Move to model device
        inputs_device = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs_device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
            )
        response = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

        output_record = {
            "sample_index": idx,
            "model_output": response,
            "ground_truth_root_event": ex.get("ground_truth_root_event", ""),
            "ground_truth_root_board": ex.get("ground_truth_root_board", ""),
            "ground_truth_boards": ex.get("ground_truth_boards", "[]"),
            "ground_truth_alarms": ex.get("ground_truth_alarms", "{}"),
            "ground_truth_severity_dist": ex.get("ground_truth_severity_dist", "{}"),
            "ground_truth_row_count": ex.get("ground_truth_row_count", 0),
        }
        outputs.append(output_record)
        print(
            f"  [{idx + 1}/{total}] event={output_record['ground_truth_root_event']}"
            f"  tokens_in={prompt_len}  tokens_out={out[0].shape[0] - prompt_len}"
        )

    return outputs


# ─────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────

def extract_all_predicted_events(text: str) -> list[str]:
    """Extract ALL predicted root cause events from model output (for multi-failure).

    Looks for multiple Event: lines or multiple event type mentions in
    structured output sections.
    """
    if not text:
        return []

    events_found: list[str] = []

    # Look for all "Event: EventType" patterns
    for m in re.finditer(r"[Ee]vent\s*:\s*(\S+)", text):
        candidate = m.group(1).strip().rstrip(",.")
        for evt in EVENT_TYPES:
            if evt == candidate or evt.replace("_", " ") == candidate:
                if evt not in events_found:
                    events_found.append(evt)

    # Also look for "Root cause" sections that list multiple events
    for m in re.finditer(r"[Rr]oot\s*[Cc]ause.*?(?:Event|event)\s*:\s*(\S+)", text):
        candidate = m.group(1).strip().rstrip(",.")
        for evt in EVENT_TYPES:
            if evt == candidate and evt not in events_found:
                events_found.append(evt)

    # If nothing found, fall back to single-event extraction
    if not events_found:
        single = extract_predicted_event(text)
        if single:
            events_found.append(single)

    return events_found


def compute_multi_failure_metrics(
    pred_events: list[str],
    gt_events: list[str],
) -> dict[str, float]:
    """Compute multi-failure evaluation metrics.

    Returns:
        all_roots_identified: 1.0 if ALL ground truth root events were predicted
        partial_root_accuracy: fraction of GT root events that were predicted
        false_positive_rate: fraction of predictions that are not in GT
    """
    gt_set = set(gt_events)
    pred_set = set(pred_events)

    if not gt_set:
        return {
            "all_roots_identified": 1.0 if not pred_set else 0.0,
            "partial_root_accuracy": 1.0,
            "false_positive_rate": 0.0,
        }

    # How many GT roots were predicted?
    identified = len(gt_set & pred_set)
    partial_accuracy = identified / len(gt_set)

    # All roots found?
    all_identified = 1.0 if gt_set <= pred_set else 0.0

    # False positives
    false_pos = len(pred_set - gt_set)
    fpr = false_pos / len(pred_set) if pred_set else 0.0

    return {
        "all_roots_identified": all_identified,
        "partial_root_accuracy": partial_accuracy,
        "false_positive_rate": fpr,
    }


def evaluate(outputs: list[dict]) -> tuple[list[dict], dict[str, float], dict]:
    """Evaluate model outputs against ground truth.

    Returns:
        per_example: list of per-example metric dicts
        summary: aggregate metrics dict
        confusion: event-type confusion matrix (dict of dicts)
    """
    per_example: list[dict] = []
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Accumulators
    n = len(outputs)
    event_correct = 0
    board_correct = 0
    category_correct = 0
    board_f1_sum = 0.0
    alarm_f1_sum = 0.0
    severity_f1_sum = 0.0
    format_sum = 0.0
    e2e_sum = 0.0

    # Multi-failure accumulators
    multi_failure_count = 0
    all_roots_identified_sum = 0.0
    partial_root_accuracy_sum = 0.0

    for record in outputs:
        text = record.get("model_output", "")
        gt_event = record.get("ground_truth_root_event", "")
        gt_board = record.get("ground_truth_root_board", "")
        gt_boards_json = record.get("ground_truth_boards", "[]")
        gt_alarms_json = record.get("ground_truth_alarms", "{}")
        gt_severity_json = record.get("ground_truth_severity_dist", "{}")
        gt_row_count = int(record.get("ground_truth_row_count", 0))
        sample_idx = record.get("sample_index", -1)

        # Parse ground truth
        gt_boards = set(json.loads(gt_boards_json)) if gt_boards_json else set()
        gt_alarms = json.loads(gt_alarms_json) if gt_alarms_json else {}
        gt_severity = json.loads(gt_severity_json) if gt_severity_json else {}

        # Extract predictions
        pred_event = extract_predicted_event(text)
        pred_board = extract_predicted_board(text)
        pred_boards = _extract_boards_from_csv(text)
        pred_alarms = _extract_alarms_from_csv(text)
        pred_severity = _extract_severity_from_csv(text)

        # --- Metric 1: Event accuracy ---
        evt_match = 1.0 if (pred_event and pred_event == gt_event) else 0.0
        event_correct += evt_match

        # --- Metric 2: Board accuracy ---
        brd_match = 1.0 if (pred_board and pred_board == gt_board) else 0.0
        board_correct += brd_match

        # --- Metric 3: Category accuracy ---
        pred_cat = CATEGORY_MAP.get(pred_event, "") if pred_event else ""
        gt_cat = CATEGORY_MAP.get(gt_event, "")
        cat_match = 1.0 if (pred_cat and pred_cat == gt_cat) else 0.0
        category_correct += cat_match

        # --- Metric 4: Board F1 ---
        bf1_result = compute_board_f1(pred_boards, gt_boards)
        bf1 = bf1_result["f1"]
        board_f1_sum += bf1

        # --- Metric 5: Alarm F1 ---
        af1 = compute_alarm_f1(pred_alarms, gt_alarms)
        alarm_f1_sum += af1

        # --- Metric 6: Severity F1 ---
        sf1 = compute_severity_f1(pred_severity, gt_severity)
        severity_f1_sum += sf1

        # --- Metric 7: Format score ---
        fmt = compute_format_score(text, gt_row_count)
        format_sum += fmt

        # --- End-to-end composite ---
        e2e = (
            0.3 * evt_match
            + 0.2 * brd_match
            + 0.2 * bf1
            + 0.2 * af1
            + 0.1 * fmt
        )
        e2e_sum += e2e

        # --- Multi-failure metrics ---
        gt_root_events_json = record.get("ground_truth_root_events", "")
        n_failures = int(record.get("n_failures", 1))
        mf_metrics = {}
        if gt_root_events_json and n_failures > 1:
            gt_root_events = json.loads(gt_root_events_json)
            pred_events_all = extract_all_predicted_events(text)
            mf_metrics = compute_multi_failure_metrics(pred_events_all, gt_root_events)
            multi_failure_count += 1
            all_roots_identified_sum += mf_metrics["all_roots_identified"]
            partial_root_accuracy_sum += mf_metrics["partial_root_accuracy"]

        # Confusion matrix
        pred_label = pred_event if pred_event else "(none)"
        gt_label = gt_event if gt_event else "(none)"
        confusion[gt_label][pred_label] += 1

        ex_record = {
            "sample_index": sample_idx,
            "gt_event": gt_event,
            "gt_board": gt_board,
            "gt_category": gt_cat,
            "pred_event": pred_event or "",
            "pred_board": pred_board or "",
            "pred_category": pred_cat,
            "event_correct": int(evt_match),
            "board_correct": int(brd_match),
            "category_correct": int(cat_match),
            "board_f1": round(bf1, 4),
            "board_precision": round(bf1_result["precision"], 4),
            "board_recall": round(bf1_result["recall"], 4),
            "alarm_f1": round(af1, 4),
            "severity_f1": round(sf1, 4),
            "format_score": round(fmt, 4),
            "e2e_score": round(e2e, 4),
            "pred_boards_count": len(pred_boards),
            "gt_boards_count": len(gt_boards),
            "pred_alarm_types": len(pred_alarms),
            "gt_alarm_types": len(gt_alarms),
            "n_failures": n_failures,
        }
        # Add multi-failure metrics if applicable
        if mf_metrics:
            ex_record["all_roots_identified"] = round(mf_metrics["all_roots_identified"], 4)
            ex_record["partial_root_accuracy"] = round(mf_metrics["partial_root_accuracy"], 4)
            ex_record["false_positive_rate"] = round(mf_metrics["false_positive_rate"], 4)
        per_example.append(ex_record)

    # Aggregate summary
    summary = {
        "n_examples": n,
        "event_accuracy": round(event_correct / n, 4) if n else 0.0,
        "board_accuracy": round(board_correct / n, 4) if n else 0.0,
        "category_accuracy": round(category_correct / n, 4) if n else 0.0,
        "board_f1_mean": round(board_f1_sum / n, 4) if n else 0.0,
        "alarm_f1_mean": round(alarm_f1_sum / n, 4) if n else 0.0,
        "severity_f1_mean": round(severity_f1_sum / n, 4) if n else 0.0,
        "format_score_mean": round(format_sum / n, 4) if n else 0.0,
        "e2e_score_mean": round(e2e_sum / n, 4) if n else 0.0,
    }

    # Multi-failure summary
    if multi_failure_count > 0:
        summary["n_multi_failure"] = multi_failure_count
        summary["all_roots_identified"] = round(
            all_roots_identified_sum / multi_failure_count, 4
        )
        summary["partial_root_accuracy"] = round(
            partial_root_accuracy_sum / multi_failure_count, 4
        )

    # Per-category breakdown
    for cat in ["Fiber", "XCON", "Line"]:
        cat_examples = [
            ex for ex in per_example if ex["gt_category"] == cat
        ]
        cat_n = len(cat_examples)
        if cat_n > 0:
            summary[f"event_accuracy_{cat}"] = round(
                sum(ex["event_correct"] for ex in cat_examples) / cat_n, 4
            )
            summary[f"board_accuracy_{cat}"] = round(
                sum(ex["board_correct"] for ex in cat_examples) / cat_n, 4
            )
            summary[f"e2e_score_{cat}"] = round(
                sum(ex["e2e_score"] for ex in cat_examples) / cat_n, 4
            )
            summary[f"n_{cat}"] = cat_n

    return per_example, summary, dict(confusion)


# ─────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────

def write_per_example_csv(per_example: list[dict], path: Path) -> None:
    """Write per-example results to CSV."""
    if not per_example:
        return
    fieldnames = list(per_example[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_example)
    print(f"[ok] wrote per-example results -> {path}")


def write_summary_csv(summary: dict[str, Any], path: Path, model_name: str) -> None:
    """Write aggregate summary to CSV."""
    summary_with_name = {"model_name": model_name, **summary}
    fieldnames = list(summary_with_name.keys())
    # Append if file exists (to accumulate multiple model runs)
    file_exists = path.exists()
    with path.open("a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary_with_name)
    print(f"[ok] wrote summary -> {path}")


def write_confusion_matrix(
    confusion: dict[str, dict[str, int]], path: Path
) -> None:
    """Write event type confusion matrix to CSV."""
    # Collect all labels
    all_gt = sorted(confusion.keys())
    all_pred = sorted(
        set(p for preds in confusion.values() for p in preds.keys())
    )
    # Ensure EVENT_TYPES are included even if they had zero counts
    all_labels = sorted(
        set(all_gt) | set(all_pred) | set(EVENT_TYPES) | {"(none)"}
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gt \\ pred"] + all_labels)
        for gt in all_labels:
            row = [gt]
            for pred in all_labels:
                row.append(confusion.get(gt, {}).get(pred, 0))
            writer.writerow(row)
    print(f"[ok] wrote confusion matrix -> {path}")


def print_summary_table(summary: dict[str, Any], model_name: str) -> None:
    """Print a formatted summary table to stdout."""
    print()
    print("=" * 70)
    print(f"  Evaluation Results: {model_name}")
    print("=" * 70)
    print(f"  Examples evaluated:   {summary['n_examples']}")
    print("-" * 70)
    print(f"  Event Accuracy:       {summary['event_accuracy']:.1%}")
    print(f"  Board Accuracy:       {summary['board_accuracy']:.1%}")
    print(f"  Category Accuracy:    {summary['category_accuracy']:.1%}")
    print(f"  Board F1 (mean):      {summary['board_f1_mean']:.1%}")
    print(f"  Alarm F1 (mean):      {summary['alarm_f1_mean']:.1%}")
    print(f"  Severity F1 (mean):   {summary['severity_f1_mean']:.1%}")
    print(f"  Format Score (mean):  {summary['format_score_mean']:.1%}")
    print("-" * 70)
    print(f"  End-to-End Score:     {summary['e2e_score_mean']:.1%}")
    print(
        f"    = 0.3*Event({summary['event_accuracy']:.3f})"
        f" + 0.2*Board({summary['board_accuracy']:.3f})"
        f" + 0.2*BoardF1({summary['board_f1_mean']:.3f})"
        f" + 0.2*AlarmF1({summary['alarm_f1_mean']:.3f})"
        f" + 0.1*Format({summary['format_score_mean']:.3f})"
    )
    print("-" * 70)

    # Multi-failure metrics
    if "n_multi_failure" in summary:
        print("-" * 70)
        print(f"  Multi-Failure (n={summary['n_multi_failure']}):")
        print(f"    All roots identified:   {summary['all_roots_identified']:.1%}")
        print(f"    Partial root accuracy:  {summary['partial_root_accuracy']:.1%}")

    # Per-category breakdown
    for cat in ["Fiber", "XCON", "Line"]:
        n_key = f"n_{cat}"
        if n_key in summary:
            print(
                f"  {cat:6s}  (n={summary[n_key]:3d})  "
                f"EventAcc={summary.get(f'event_accuracy_{cat}', 0):.1%}  "
                f"BoardAcc={summary.get(f'board_accuracy_{cat}', 0):.1%}  "
                f"E2E={summary.get(f'e2e_score_{cat}', 0):.1%}"
            )
    print("=" * 70)
    print()


def print_confusion_matrix(confusion: dict[str, dict[str, int]]) -> None:
    """Print a compact confusion matrix to stdout."""
    all_labels = sorted(
        set(EVENT_TYPES) | {"(none)"}
        | set(confusion.keys())
        | set(p for preds in confusion.values() for p in preds.keys())
    )
    # Only show labels with nonzero rows or columns
    active_labels = []
    for lbl in all_labels:
        row_total = sum(confusion.get(lbl, {}).values())
        col_total = sum(
            confusion.get(gt, {}).get(lbl, 0) for gt in all_labels
        )
        if row_total > 0 or col_total > 0:
            active_labels.append(lbl)

    if not active_labels:
        return

    # Short labels for display
    short = {}
    for lbl in active_labels:
        if lbl == "(none)":
            short[lbl] = "(none)"
        else:
            parts = lbl.split("_")
            short[lbl] = "_".join(p[:4] for p in parts)

    col_width = max(len(short[l]) for l in active_labels) + 1
    col_width = max(col_width, 8)

    print("Confusion Matrix (rows=GT, cols=Predicted):")
    header = f"{'':>{col_width}}" + "".join(
        f"{short[l]:>{col_width}}" for l in active_labels
    )
    print(header)
    for gt in active_labels:
        row_str = f"{short[gt]:>{col_width}}"
        for pred in active_labels:
            count = confusion.get(gt, {}).get(pred, 0)
            cell = str(count) if count > 0 else "."
            row_str += f"{cell:>{col_width}}"
        print(row_str)
    print()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate OTN RCA model predictions against ground truth."
    )
    ap.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Test JSONL with prompts and ground truth (same format as GRPO training data)",
    )
    ap.add_argument(
        "--outputs-file",
        type=Path,
        default=None,
        help="Pre-computed model outputs JSONL (each line has model_output + ground truth fields)",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/eval"),
        help="Directory for output result files",
    )
    ap.add_argument(
        "--model-name",
        default="model",
        help="Model name for labeling results",
    )

    # Inference args (only used with --run-inference)
    ap.add_argument(
        "--run-inference",
        action="store_true",
        help="Run live inference instead of loading cached outputs",
    )
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument(
        "--save-outputs",
        type=Path,
        default=None,
        help="Save inference outputs to this JSONL file (for caching)",
    )

    args = ap.parse_args()

    # ── Load or generate outputs ──
    outputs: list[dict] = []

    if args.run_inference:
        # Load test JSONL and run inference
        if not args.test_file:
            ap.error("--test-file is required when using --run-inference")
        if not args.test_file.exists():
            ap.error(f"Test file not found: {args.test_file}")

        print(f"[info] loading test data from {args.test_file}")
        examples = []
        with args.test_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        print(f"[info] loaded {len(examples)} test examples")

        print(f"[info] running inference with {args.base_model}")
        if args.adapter:
            print(f"[info] adapter: {args.adapter}")
        outputs = run_inference_batch(
            examples,
            base_model=args.base_model,
            adapter=args.adapter,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        # Optionally save outputs for caching
        save_path = args.save_outputs
        if save_path is None:
            save_path = args.results_dir / "model_outputs.jsonl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            for rec in outputs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[ok] saved inference outputs -> {save_path}")

    elif args.outputs_file:
        # Load pre-computed outputs
        if not args.outputs_file.exists():
            ap.error(f"Outputs file not found: {args.outputs_file}")
        print(f"[info] loading cached outputs from {args.outputs_file}")
        with args.outputs_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    outputs.append(json.loads(line))
        print(f"[info] loaded {len(outputs)} model outputs")

    elif args.test_file:
        # Load test JSONL — use ground_truth_alarm_flow_csv as "model output"
        # This is useful for sanity-checking the evaluation pipeline itself
        print(f"[info] no --outputs-file or --run-inference; running ground-truth self-eval")
        if not args.test_file.exists():
            ap.error(f"Test file not found: {args.test_file}")
        with args.test_file.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                # Simulate a "perfect" model output from ground truth
                gt_csv = ex.get("ground_truth_alarm_flow_csv", "")
                gt_event = ex.get("ground_truth_root_event", "")
                gt_board = ex.get("ground_truth_root_board", "")
                synthetic_output = (
                    f"Root cause Event: {gt_event}\n"
                    f"Root cause Board: {gt_board}\n\n"
                    f"alarm_flow rows (CSV):\n{gt_csv}\n\n"
                    f"RCA reasoning: Ground truth self-eval.\n"
                )
                outputs.append({
                    "sample_index": idx,
                    "model_output": synthetic_output,
                    "ground_truth_root_event": gt_event,
                    "ground_truth_root_board": gt_board,
                    "ground_truth_boards": ex.get("ground_truth_boards", "[]"),
                    "ground_truth_alarms": ex.get("ground_truth_alarms", "{}"),
                    "ground_truth_severity_dist": ex.get("ground_truth_severity_dist", "{}"),
                    "ground_truth_row_count": ex.get("ground_truth_row_count", 0),
                })
        print(f"[info] built {len(outputs)} self-eval examples from ground truth")

    else:
        ap.error(
            "Provide --outputs-file (cached outputs) or --test-file "
            "(with --run-inference or for self-eval)"
        )

    if not outputs:
        print("[error] no outputs to evaluate")
        sys.exit(1)

    # ── Evaluate ──
    print(f"[info] evaluating {len(outputs)} examples ...")
    per_example, summary, confusion = evaluate(outputs)

    # ── Write results ──
    args.results_dir.mkdir(parents=True, exist_ok=True)

    write_per_example_csv(
        per_example, args.results_dir / "results_per_example.csv"
    )
    write_summary_csv(
        summary, args.results_dir / "results_summary.csv", args.model_name
    )
    write_confusion_matrix(confusion, args.results_dir / "confusion_matrix.csv")

    # ── Print to stdout ──
    print_summary_table(summary, args.model_name)
    print_confusion_matrix(confusion)


if __name__ == "__main__":
    main()
