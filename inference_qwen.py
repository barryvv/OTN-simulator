#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with Qwen fine-tuned adapter model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-72B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="adapters/qwen-failure",
        help="Path to adapter weights",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Analyze Fiber_Aging failure on ROADM0-OA002$0 and suggest fixes.",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="",
        help="Read prompt text from file (overrides --prompt)",
    )
    parser.add_argument("--max-tokens", type=int, default=300, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling parameter")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit quantization (requires bitsandbytes)",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda", "none"],
        help="Device map for model loading (use cpu/mps to avoid accelerate balance issues)",
    )
    parser.add_argument(
        "--offload-dir",
        type=str,
        default="",
        help="Folder for disk offload when device_map=auto",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Model dtype",
    )

    args = parser.parse_args()

    print(f"Loading tokenizer from {args.base_model}...")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    print(f"Loading base model from {args.base_model}...")
    device_map = None if args.device_map == "none" else args.device_map
    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": _resolve_dtype(args.dtype),
    }
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if args.offload_dir:
        model_kwargs["offload_folder"] = args.offload_dir
        model_kwargs["offload_state_dict"] = True
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    print(f"Loading adapter from {args.adapter}...")
    peft_kwargs = {}
    if device_map is not None:
        peft_kwargs["device_map"] = device_map
    if args.offload_dir:
        peft_kwargs["offload_folder"] = args.offload_dir
        peft_kwargs["offload_state_dict"] = True
    model = PeftModel.from_pretrained(model, args.adapter, **peft_kwargs)

    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    print(f"\nPrompt: {args.prompt}\n")
    print("Generating response...")

    inputs = tok(args.prompt, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": args.max_tokens,
        "pad_token_id": tok.eos_token_id,
    }
    if args.temperature is not None:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["do_sample"] = True
    if args.top_p is not None:
        generation_kwargs["top_p"] = args.top_p
        generation_kwargs["do_sample"] = True

    with torch.no_grad():
        out = model.generate(**inputs, **generation_kwargs)

    response = tok.decode(out[0], skip_special_tokens=True)
    print("\n" + "=" * 80)
    print("RESPONSE:")
    print("=" * 80)
    print(response)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
