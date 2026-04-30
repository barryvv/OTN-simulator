"""Inference script for GRPO-trained OTN alarm RCA model.

Usage:
  python inference_grpo.py \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --adapter adapters/qwen-3b-grpo \
    --prompt-file outputs/grpo_data/train.jsonl \
    --sample-index 0 \
    --max-new-tokens 1024
"""
import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="adapters/qwen-3b-grpo")
    ap.add_argument("--prompt-file", default="outputs/grpo_data/train.jsonl")
    ap.add_argument("--sample-index", type=int, default=0,
                    help="Which example from the JSONL to use (0-indexed)")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--compare-base", action="store_true",
                    help="Also run base model without adapter for comparison")
    args = ap.parse_args()

    # ── Load prompt ──
    with open(args.prompt_file) as f:
        lines = f.readlines()
    example = json.loads(lines[args.sample_index])
    messages = example["prompt"]
    ground_truth_event = example.get("ground_truth_root_event", "?")
    ground_truth_board = example.get("ground_truth_root_board", "?")

    print(f"[info] Sample {args.sample_index}: event={ground_truth_event}, board={ground_truth_board}")
    print(f"[info] Prompt length: {sum(len(m['content']) for m in messages)} chars")
    print("=" * 80)

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Build input ──
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]
    print(f"[info] Prompt tokens: {prompt_len}")

    # ── Run base model (optional) ──
    if args.compare_base:
        print("\n" + "=" * 80)
        print("BASE MODEL (no adapter):")
        print("=" * 80)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        inputs_device = {k: v.to(base_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = base_model.generate(
                **inputs_device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                top_p=0.9,
            )
        response = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        print(response)
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Run GRPO-finetuned model ──
    print("\n" + "=" * 80)
    print(f"GRPO-FINETUNED MODEL (adapter: {args.adapter}):")
    print("=" * 80)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    inputs_device = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs_device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            top_p=0.9,
        )
    response = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    print(response)

    # ── Ground truth ──
    print("\n" + "=" * 80)
    print("GROUND TRUTH:")
    print("=" * 80)
    print(f"Root Event: {ground_truth_event}")
    print(f"Root Board: {ground_truth_board}")
    gt_flow = example.get("ground_truth_alarm_flow_csv", "")
    if gt_flow:
        print(f"Alarm Flow:\n{gt_flow[:500]}")


if __name__ == "__main__":
    main()
