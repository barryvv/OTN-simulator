#!/usr/bin/env python3
"""Merge a LoRA adapter into a base model and save the result."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--adapter", default="adapters/qwen-14b-sft")
    ap.add_argument("--output-dir", default="models/qwen-14b-sft-merged")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    args = ap.parse_args()

    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"[info] loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dt, device_map="cpu",
    )

    print(f"[info] loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("[info] merging adapter weights ...")
    model = model.merge_and_unload()

    print(f"[info] saving merged model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.save_pretrained(args.output_dir)

    print("[ok] merge complete!")


if __name__ == "__main__":
    main()
