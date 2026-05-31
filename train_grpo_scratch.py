#!/usr/bin/env python3
"""From-scratch GRPO training entrypoint.

Companion to `train_grpo_gemma3b.py` (TRL-based). This implementation
writes the policy update loop in raw PyTorch and does NOT use
`trl.GRPOTrainer`. The reward functions and dataset are reused so that
metrics can be compared directly between the two trainers.

Usage:
  python train_grpo_scratch.py \\
    --base-model Qwen/Qwen2.5-3B-Instruct \\
    --train-file outputs/grpo_data/train.jsonl \\
    --output-dir adapters/qwen-3b-grpo-scratch \\
    --max-steps 50
"""

from __future__ import annotations

# We only use the PyTorch backend; tell HF / PEFT to skip the optional
# TensorFlow path. On Apple Silicon, the auto-loaded TF wheel is built
# for AVX and crashes the interpreter on import.
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_scratch.trainer import FromScratchGRPOTrainer, GRPOScratchConfig


GROUND_TRUTH_KEYS = (
    "ground_truth_boards",
    "ground_truth_alarms",
    "ground_truth_max_hop",
    "ground_truth_root_board",
    "ground_truth_root_event",
    "ground_truth_row_count",
    "ground_truth_severity_dist",
)


def make_collate(tokenizer):
    """Build a collate_fn that renders chat messages with the model's
    chat template so generation is correctly formatted."""

    def collate(batch):
        out: dict = {"prompt": []}
        for ex in batch:
            msgs = ex["prompt"]
            if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
            else:
                text = str(msgs)
            out["prompt"].append(text)
            for k in GROUND_TRUTH_KEYS:
                out.setdefault(k, []).append(ex.get(k))
        return out

    return collate


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="From-scratch GRPO training entrypoint for OTN RCA.",
    )
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train-file", default="outputs/grpo_data/train.jsonl")
    ap.add_argument("--output-dir", default="adapters/qwen-3b-grpo-scratch")
    ap.add_argument("--group-size", type=int, default=4,
                    help="G: completions per prompt for advantage estimation")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="KL penalty coefficient (0.0 = no KL, following DAPO)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="Trade compute for memory (recommended for MPS / small VRAM)")
    ap.add_argument("--empty-cache-between-phases", action="store_true",
                    help="Call torch.{cuda,mps}.empty_cache() between phases")
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="Gradient norm clip. Default 1.0 is conservative; for RL "
                         "with high raw grad norms, 5-10 often learns faster.")
    ap.add_argument("--reward-fn", default="discrete",
                    choices=["discrete", "continuous"],
                    help="discrete: original tiered rewards (TRL-compatible). "
                         "continuous: difflib similarity slopes that GRPO can "
                         "climb when the discrete tiers cause plateau collapse.")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype] if device == "cuda" else torch.float32

    print(f"[info] device={device}  dtype={dtype}")
    print(f"[info] base model: {args.base_model}")
    print(f"[info] train file: {args.train_file}")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ── Policy model with LoRA adapter ──
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype,
    ).to(device)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # dropout=0 lets us safely switch to train() mode for the
        # new-log-probs forward pass so gradient checkpointing actually
        # activates, without re-introducing the eval-vs-train ratio noise.
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(base, lora_cfg)
    # NOTE: we leave LoRA params in the base dtype (bf16) on purpose. An
    # earlier version cast them to fp32 for AdamW precision, but at LoRA
    # gradient magnitudes around 5e-4 bf16 has ~0.8% relative precision —
    # plenty — and the fp32 cast forced a fp32 LoRA-output copy at every
    # q/k/v/o module which added ~14 GB of activation memory across all
    # layers and tripped OOM on the A100/H100. AdamW in bf16 trains fine
    # here; the slow B-matrix growth seen previously was the LR being
    # too conservative, not the dtype.
    policy.print_trainable_parameters()

    # ── Reference model (only needed if beta > 0) ──
    ref = None
    if args.beta > 0.0:
        print("[info] loading reference model for KL penalty (beta > 0)")
        ref = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=dtype,
        ).to(device).eval()

    # ── Reward functions (discrete tiered vs continuous similarity) ──
    if args.reward_fn == "continuous":
        from grpo_scratch.rewards_continuous import (
            otn_alarm_reward, otn_format_reward,
        )
        print("[info] using continuous (similarity-based) reward functions")
    else:
        from grpo_scratch.rewards import otn_alarm_reward, otn_format_reward
        print("[info] using discrete (tiered) reward functions")

    # ── Dataset ──
    ds = load_dataset(
        "json", data_files={"train": args.train_file}, split="train",
    )
    print(f"[info] loaded {len(ds)} training examples")
    print(f"[info] columns: {ds.column_names}")

    loader = DataLoader(
        ds, batch_size=1, shuffle=True, collate_fn=make_collate(tokenizer),
    )

    # ── Config + trainer ──
    config = GRPOScratchConfig(
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        learning_rate=args.lr,
        epsilon=args.epsilon,
        beta=args.beta,
        output_dir=args.output_dir,
        reward_weights=(0.8, 0.2),
        gradient_checkpointing=args.gradient_checkpointing,
        empty_cache_between_phases=args.empty_cache_between_phases,
        max_grad_norm=args.max_grad_norm,
    )
    trainer = FromScratchGRPOTrainer(
        policy_model=policy,
        ref_model=ref,
        tokenizer=tokenizer,
        reward_funcs=[otn_alarm_reward, otn_format_reward],
        config=config,
    )

    print("[info] starting from-scratch GRPO training ...")
    trainer.train(loader, num_steps=args.max_steps)
    print("[ok] training complete.")


if __name__ == "__main__":
    main()
