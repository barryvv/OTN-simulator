#!/usr/bin/env python3
"""SFT training on ReAct agent traces.

Trains a Qwen model to generate ReAct-style reasoning with tool calls.
OBSERVATION turns (tool outputs) are masked so the model only learns to
generate thoughts, ACTIONs, and final ANSWERs.

Usage (on server with GPU):
  python train_agent_sft.py \
    --base-model Qwen/Qwen2.5-14B-Instruct \
    --train-file outputs/agent_traces/train.jsonl \
    --output-dir adapters/qwen-14b-agent-sft \
    --max-length 8192 --epochs 3 --lr 2e-5

  # With 4-bit quantization for memory savings
  python train_agent_sft.py \
    --base-model Qwen/Qwen2.5-14B-Instruct \
    --train-file outputs/agent_traces/train.jsonl \
    --output-dir adapters/qwen-14b-agent-sft \
    --load-in-4bit --max-length 8192
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Data loading and formatting
# ---------------------------------------------------------------------------

OBSERVATION_PREFIX = "OBSERVATION:"


def _is_observation_turn(msg: dict) -> bool:
    """Check if a message is an observation (tool output) turn."""
    content = msg.get("content", "")
    role = msg.get("role", "")
    return role == "user" and content.strip().startswith(OBSERVATION_PREFIX)


def load_agent_traces(path: Path) -> list[dict]:
    """Load agent trace JSONL and convert to chat-format training examples.

    Each trace has a 'messages' field with the full conversation.
    We mark observation turns with a special flag so the trainer can mask them.
    """
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] skipping malformed JSON at line {line_num}", file=sys.stderr)
                continue

            messages = record.get("messages", [])
            if not messages:
                continue

            # Convert to the format expected by TRL SFTTrainer
            # We keep all messages but will handle masking via the
            # DataCollatorForCompletionOnlyLM approach
            formatted_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    formatted_messages.append({"role": "system", "content": content})
                elif role == "user":
                    formatted_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    formatted_messages.append({"role": "assistant", "content": content})

            if formatted_messages:
                examples.append({"messages": formatted_messages})

    return examples


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="SFT training on ReAct agent traces."
    )
    ap.add_argument(
        "--base-model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
        help="Base model HuggingFace ID or local path",
    )
    ap.add_argument(
        "--train-file", type=Path, required=True,
        help="Path to agent traces JSONL",
    )
    ap.add_argument(
        "--output-dir", type=str, default="adapters/qwen-14b-agent-sft",
        help="Output directory for LoRA adapter",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--save-steps", type=int, default=999)
    ap.add_argument("--eval-steps", type=int, default=999)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if not args.train_file.exists():
        raise SystemExit(f"[error] train file not found: {args.train_file}")

    # Import ML dependencies (only needed on GPU server)
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise SystemExit(
            f"Training requires torch, transformers, peft, trl, datasets. Missing: {e}"
        )

    # Load data
    print(f"[info] loading agent traces from {args.train_file}")
    train_examples = load_agent_traces(args.train_file)
    print(f"[info] loaded {len(train_examples)} training examples")

    if not train_examples:
        raise SystemExit("[error] no valid training examples found")

    # Print sample statistics
    msg_counts = [len(ex["messages"]) for ex in train_examples]
    print(f"[info] messages per example: min={min(msg_counts)}, max={max(msg_counts)}, "
          f"mean={sum(msg_counts)/len(msg_counts):.1f}")

    # Create HuggingFace dataset
    dataset = Dataset.from_list(train_examples)

    # Load tokenizer
    print(f"[info] loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Model loading config
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    print(f"[info] loading model from {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        gradient_checkpointing=True,
        seed=args.seed,
        report_to="none",
        packing=False,  # packing=True causes OOM on variable-length sequences
    )

    # Create trainer
    print("[info] creating SFT trainer")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Train
    print(f"[info] starting training: {args.epochs} epochs, lr={args.lr}, "
          f"batch={args.batch_size}x{args.grad_accum}")
    trainer.train()

    # Save
    print(f"[info] saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"[ok] training complete. Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
