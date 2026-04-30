#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def _get_tokenizer(model_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _format_messages(tokenizer: AutoTokenizer, example: dict) -> str:
    messages = example.get("messages")
    if messages:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}:\n{content}")
        return "\n\n".join(parts)
    return example.get("text", "")


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT train Qwen 7B with LoRA adapters.")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train-file", default="outputs/llm_data/train.jsonl")
    ap.add_argument("--val-file", default="outputs/llm_data/val.jsonl")
    ap.add_argument("--output-dir", default="adapters/qwen-7b-failure")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--use-4bit", action="store_true")
    ap.add_argument("--use-8bit", action="store_true")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA",
    )
    args = ap.parse_args()

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)
    if not train_path.exists():
        raise SystemExit(f"Missing train file: {train_path}")
    if not val_path.exists():
        raise SystemExit(f"Missing val file: {val_path}")

    tokenizer = _get_tokenizer(args.base_model)

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
    }
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": args.device}
    if args.use_4bit:
        model_kwargs["load_in_4bit"] = True
    if args.use_8bit:
        model_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        task_type="CAUSAL_LM",
    )

    ds = load_dataset("json", data_files={"train": str(train_path), "validation": str(val_path)})

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        max_length=args.max_length,
        packing=False,
        seed=args.seed,
        fp16=(args.dtype == "float16" and args.device != "cpu"),
        bf16=(args.dtype == "bfloat16"),
        gradient_checkpointing=True,
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": ds["train"],
        "eval_dataset": ds["validation"],
        "peft_config": lora,
        "args": sft_args,
        "formatting_func": lambda ex: _format_messages(tokenizer, ex),
    }
    sig = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
