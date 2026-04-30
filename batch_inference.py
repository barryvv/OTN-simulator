#!/usr/bin/env python3
"""Batch inference for zero-shot, SFT, or GRPO models.

Runs on GPU server. Produces JSONL compatible with evaluate.py --outputs-file.

Usage:
  # Zero-shot
  python3 batch_inference.py --test-file test_iid.jsonl \
    --base-model Qwen/Qwen2.5-14B-Instruct --method zero-shot \
    --out eval_outputs/zeroshot_iid.jsonl

  # SFT
  python3 batch_inference.py --test-file test_iid.jsonl \
    --base-model models/qwen-14b-sft-v6.1-merged --method sft \
    --out eval_outputs/sft_iid.jsonl

  # GRPO
  python3 batch_inference.py --test-file test_iid.jsonl \
    --base-model models/qwen-14b-sft-v6.1-merged \
    --adapter adapters/qwen-14b-sft-v6.1-grpo --method grpo \
    --out eval_outputs/grpo_iid.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load_test_examples(path: Path) -> list[dict]:
    """Load test JSONL file."""
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_completed_indices(path: Path) -> set[int]:
    """Load already-completed sample indices for resume support."""
    if not path.exists():
        return set()
    indices = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    indices.add(record["sample_index"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return indices


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch inference for OTN RCA evaluation."
    )
    ap.add_argument("--test-file", type=Path, required=True,
                    help="Test JSONL (from generate_grpo_data.py)")
    ap.add_argument("--base-model", type=str, required=True,
                    help="Base model path or HF model ID")
    ap.add_argument("--adapter", type=str, default=None,
                    help="LoRA adapter path (for GRPO/agent-SFT)")
    ap.add_argument("--method", type=str, required=True,
                    choices=["zero-shot", "sft", "grpo"],
                    help="Method label for output records")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL path")
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="Max generation tokens")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load model in 4-bit quantization")
    ap.add_argument("--resume", action="store_true",
                    help="Skip examples already in output file")
    args = ap.parse_args()

    # ── Load test data ──
    print(f"[info] loading test data from {args.test_file}")
    examples = load_test_examples(args.test_file)
    print(f"[info] loaded {len(examples)} test examples")

    # ── Resume support ──
    completed = set()
    if args.resume:
        completed = load_completed_indices(args.out)
        if completed:
            print(f"[info] resuming — {len(completed)} already completed, "
                  f"{len(examples) - len(completed)} remaining")

    # ── Load model ──
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[info] loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    print(f"[info] loading model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    if args.adapter:
        from peft import PeftModel
        print(f"[info] loading adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    # ── Inference loop ──
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and completed else "w"
    total = len(examples)
    remaining = total - len(completed)
    done = 0
    t0 = time.time()

    # Ground truth keys to copy from each example
    GT_KEYS = [
        "ground_truth_root_event",
        "ground_truth_root_board",
        "ground_truth_boards",
        "ground_truth_alarms",
        "ground_truth_severity_dist",
        "ground_truth_row_count",
        "ground_truth_root_events",
        "ground_truth_root_boards",
        "n_failures",
    ]

    with open(args.out, mode, encoding="utf-8") as fout:
        for idx, ex in enumerate(examples):
            if idx in completed:
                continue

            messages = ex["prompt"]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt")
            prompt_len = inputs["input_ids"].shape[1]

            inputs_device = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **inputs_device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.temperature > 0,
                    top_p=0.9,
                )

            response = tokenizer.decode(
                out[0][prompt_len:], skip_special_tokens=True
            )

            record = {
                "sample_index": idx,
                "model_output": response,
                "method": args.method,
            }
            for k in GT_KEYS:
                if k in ex:
                    record[k] = ex[k]

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            done += 1
            elapsed = time.time() - t0
            per_item = elapsed / done
            eta = per_item * (remaining - done)
            gen_tokens = out[0].shape[0] - prompt_len

            print(
                f"  [{done}/{remaining}] idx={idx} "
                f"event={ex.get('ground_truth_root_event', '?')} "
                f"tokens_in={prompt_len} tokens_out={gen_tokens} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
            )

    total_time = time.time() - t0
    print(f"\n[ok] wrote {done} outputs -> {args.out}")
    print(f"[info] total time: {total_time:.0f}s ({total_time/max(done,1):.1f}s/example)")


if __name__ == "__main__":
    main()
