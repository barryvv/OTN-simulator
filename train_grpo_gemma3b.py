#!/usr/bin/env python3
"""GRPO training script for LLM on OTN alarm RCA.

Usage:
  python train_grpo_gemma3b.py \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --train-file outputs/grpo_data/train.jsonl \
    --output-dir adapters/qwen-3b-grpo
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer


# ─────────────────────────────────────────────────────────────
# Reward helper functions
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
            # Find end of CSV block
            end_markers = ["\nRCA reasoning:", "\nFix ideas:", "\nFix ", "\n\n\n", "\n---"]
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
            row = {header[i].strip(): fields[i].strip() for i in range(min(len(header), len(fields)))}
            rows.append(row)
    return rows


def _extract_boards_from_csv(text: str) -> set[str]:
    """Extract unique PropToBoard values from CSV in completion."""
    csv_block = _extract_csv_block(text)
    rows = _parse_csv_rows(csv_block)
    boards = set()
    for r in rows:
        b = r.get("PropToBoard", "").strip()
        if b and b.startswith("ROADM"):
            boards.add(b)
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


def _extract_max_hop(text: str) -> int:
    """Extract maximum Hop value from CSV in completion."""
    csv_block = _extract_csv_block(text)
    rows = _parse_csv_rows(csv_block)
    max_hop = 0
    for r in rows:
        try:
            h = int(float(r.get("Hop", "0")))
            max_hop = max(max_hop, h)
        except (ValueError, TypeError):
            pass
    return max_hop


# ─────────────────────────────────────────────────────────────
# Reward function 1: OTN Alarm Accuracy (weight 0.8)
# ─────────────────────────────────────────────────────────────

def otn_alarm_reward(completions, **kwargs) -> list[float]:
    """Score completions on root cause accuracy, board coverage, and alarm accuracy."""
    gt_boards_list = kwargs.get("ground_truth_boards", [])
    gt_alarms_list = kwargs.get("ground_truth_alarms", [])
    gt_max_hops = kwargs.get("ground_truth_max_hop", [])
    gt_root_boards = kwargs.get("ground_truth_root_board", [])
    gt_root_events = kwargs.get("ground_truth_root_event", [])

    rewards = []
    for i, completion in enumerate(completions):
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)

        score = 0.0
        gt_board = gt_root_boards[i] if i < len(gt_root_boards) else ""
        gt_event = gt_root_events[i] if i < len(gt_root_events) else ""

        # ── Sub-reward 1: Root cause event match (0.25) ──
        text_upper = text.upper()
        gt_event_upper = gt_event.upper().replace("_", " ")
        # Exact event match
        if gt_event_upper in text_upper or gt_event.upper() in text_upper:
            score += 0.25
        else:
            # Check for specific event names to distinguish same-category predictions
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
                    score += 0.10  # Right category, wrong specific event
            elif gt_cat in text_upper:
                score += 0.03  # Just mentions the category vaguely

        # ── Sub-reward 2: Root cause board match (0.15) ──
        if gt_board and gt_board in text:
            score += 0.15
        elif gt_board:
            # Partial: correct ROADM
            roadm = gt_board.split("-")[0] if "-" in gt_board else ""
            if roadm and roadm in text:
                score += 0.05

        # ── Sub-reward 3: Board coverage (0.25) ──
        gt_boards_json = gt_boards_list[i] if i < len(gt_boards_list) else "[]"
        gt_boards = set(json.loads(gt_boards_json)) if gt_boards_json else set()
        pred_boards = _extract_boards_from_csv(text)
        if gt_boards and pred_boards:
            jaccard = len(pred_boards & gt_boards) / len(pred_boards | gt_boards)
            score += 0.25 * jaccard
        elif not gt_boards and not pred_boards:
            score += 0.25

        # ── Sub-reward 4: Alarm name accuracy (0.15) ──
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
# Reward function 2: CSV Format Compliance (weight 0.2)
# ─────────────────────────────────────────────────────────────

def otn_format_reward(completions, **kwargs) -> list[float]:
    """Score completions on CSV format quality."""
    gt_row_counts = kwargs.get("ground_truth_row_count", [])

    rewards = []
    for i, completion in enumerate(completions):
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)

        score = 0.0
        csv_block = _extract_csv_block(text)

        if not csv_block:
            rewards.append(0.0)
            continue

        lines = [ln for ln in csv_block.strip().split("\n") if ln.strip()]
        if not lines:
            rewards.append(0.0)
            continue

        # ── Header check (0.05) ──
        header = lines[0]
        if header.count(",") >= 19 and "SourceBoard" in header:
            score += 0.05

        # ── Valid data rows (0.10) ──
        data_lines = lines[1:] if header.count(",") >= 19 else lines
        total_data = len(data_lines)
        if total_data > 0:
            valid_rows = sum(1 for ln in data_lines if ln.count(",") >= 19)
            score += 0.10 * (valid_rows / total_data)

        # ── Row count proximity (0.05) ──
        gt_count = gt_row_counts[i] if i < len(gt_row_counts) else 0
        if gt_count > 0 and total_data > 0:
            ratio = min(total_data, gt_count) / max(total_data, gt_count)
            score += 0.05 * ratio

        rewards.append(score)
    return rewards


# ─────────────────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO training for LLM on OTN alarm RCA.")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train-file", default="outputs/grpo_data/train.jsonl")
    ap.add_argument("--output-dir", default="adapters/qwen-3b-grpo")
    ap.add_argument("--max-completion-length", type=int, default=2048)
    ap.add_argument("--num-generations", type=int, default=4,
                    help="G: completions per prompt for GRPO advantage estimation")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Max training steps (-1 = full epoch)")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="KL penalty coefficient (0.0 = no KL, following DAPO)")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load base model in 4-bit quantization (QLoRA) to reduce memory")
    args = ap.parse_args()

    print(f"[info] base model: {args.base_model}")
    print(f"[info] train file: {args.train_file}")
    print(f"[info] device: {args.device}, dtype: {args.dtype}")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ── Dataset ──
    ds = load_dataset("json", data_files={"train": args.train_file}, split="train")
    print(f"[info] loaded {len(ds)} training examples")
    print(f"[info] columns: {ds.column_names}")

    # ── LoRA config ──
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # ── Quantization ──
    model_kwargs = {}
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        print("[info] using 4-bit quantization (QLoRA)")

    # ── GRPO config ──
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        model_init_kwargs=model_kwargs if model_kwargs else None,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=args.beta,
        scale_rewards="group",
        reward_weights=[0.8, 0.2],
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        gradient_checkpointing=True,
        seed=42,
        report_to="none",
        remove_unused_columns=False,
    )

    # ── Trainer ──
    trainer = GRPOTrainer(
        model=args.base_model,
        args=grpo_config,
        reward_funcs=[otn_alarm_reward, otn_format_reward],
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("[info] starting GRPO training ...")
    trainer.train()

    print(f"[info] saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[ok] training complete!")


if __name__ == "__main__":
    main()
