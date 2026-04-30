#!/bin/bash
# SFC-BFL Benchmark Inference Script
# Run this on the GPU server (hph8)
set -e

EVAL_ROOT="outputs/sfc_bfl_eval"
BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"
SFT_MODEL="models/qwen-14b-sft-v6.1-merged"
GRPO_ADAPTER="adapters/qwen-14b-sft-v6.1-grpo"

mkdir -p $EVAL_ROOT

# zero-shot on dataset_a
python3 batch_inference.py --test-file dataset_a.jsonl --base-model $BASE_MODEL  --method zero-shot --out $EVAL_ROOT/zero-shot_dataset_a.jsonl --load-in-4bit --resume

# sft on dataset_a
python3 batch_inference.py --test-file dataset_a.jsonl --base-model $SFT_MODEL  --method sft --out $EVAL_ROOT/sft_dataset_a.jsonl --load-in-4bit --resume

# grpo on dataset_a
python3 batch_inference.py --test-file dataset_a.jsonl --base-model $SFT_MODEL --adapter $GRPO_ADAPTER --method grpo --out $EVAL_ROOT/grpo_dataset_a.jsonl --load-in-4bit --resume

# zero-shot on dataset_b
python3 batch_inference.py --test-file dataset_b.jsonl --base-model $BASE_MODEL  --method zero-shot --out $EVAL_ROOT/zero-shot_dataset_b.jsonl --load-in-4bit --resume

# sft on dataset_b
python3 batch_inference.py --test-file dataset_b.jsonl --base-model $SFT_MODEL  --method sft --out $EVAL_ROOT/sft_dataset_b.jsonl --load-in-4bit --resume

# grpo on dataset_b
python3 batch_inference.py --test-file dataset_b.jsonl --base-model $SFT_MODEL --adapter $GRPO_ADAPTER --method grpo --out $EVAL_ROOT/grpo_dataset_b.jsonl --load-in-4bit --resume

# zero-shot on dataset_c
python3 batch_inference.py --test-file dataset_c.jsonl --base-model $BASE_MODEL  --method zero-shot --out $EVAL_ROOT/zero-shot_dataset_c.jsonl --load-in-4bit --resume

# sft on dataset_c
python3 batch_inference.py --test-file dataset_c.jsonl --base-model $SFT_MODEL  --method sft --out $EVAL_ROOT/sft_dataset_c.jsonl --load-in-4bit --resume

# grpo on dataset_c
python3 batch_inference.py --test-file dataset_c.jsonl --base-model $SFT_MODEL --adapter $GRPO_ADAPTER --method grpo --out $EVAL_ROOT/grpo_dataset_c.jsonl --load-in-4bit --resume

# zero-shot on dataset_d_size_1200
python3 batch_inference.py --test-file dataset_d_size_1200.jsonl --base-model $BASE_MODEL  --method zero-shot --out $EVAL_ROOT/zero-shot_dataset_d_size_1200.jsonl --load-in-4bit --resume

# sft on dataset_d_size_1200
python3 batch_inference.py --test-file dataset_d_size_1200.jsonl --base-model $SFT_MODEL  --method sft --out $EVAL_ROOT/sft_dataset_d_size_1200.jsonl --load-in-4bit --resume

# grpo on dataset_d_size_1200
python3 batch_inference.py --test-file dataset_d_size_1200.jsonl --base-model $SFT_MODEL --adapter $GRPO_ADAPTER --method grpo --out $EVAL_ROOT/grpo_dataset_d_size_1200.jsonl --load-in-4bit --resume

# zero-shot on dataset_d_size_200
python3 batch_inference.py --test-file dataset_d_size_200.jsonl --base-model $BASE_MODEL  --method zero-shot --out $EVAL_ROOT/zero-shot_dataset_d_size_200.jsonl --load-in-4bit --resume

# sft on dataset_d_size_200
python3 batch_inference.py --test-file dataset_d_size_200.jsonl --base-model $SFT_MODEL  --method sft --out $EVAL_ROOT/sft_dataset_d_size_200.jsonl --load-in-4bit --resume

# grpo on dataset_d_size_200
python3 batch_inference.py --test-file dataset_d_size_200.jsonl --base-model $SFT_MODEL --adapter $GRPO_ADAPTER --method grpo --out $EVAL_ROOT/grpo_dataset_d_size_200.jsonl --load-in-4bit --resume

# zero-shot on dataset_d_size_400
python3 batch_inference.py --test-file dataset_d_size_400.jsonl --base-model $BASE_MODEL  --method zero-shot --out $EVAL_ROOT/zero-shot_dataset_d_size_400.jsonl --load-in-4bit --resume

# sft on dataset_d_size_400
python3 batch_inference.py --test-file dataset_d_size_400.jsonl --base-model $SFT_MODEL  --method sft --out $EVAL_ROOT/sft_dataset_d_size_400.jsonl --load-in-4bit --resume

# grpo on dataset_d_size_400
python3 batch_inference.py --test-file dataset_d_size_400.jsonl --base-model $SFT_MODEL --adapter $GRPO_ADAPTER --method grpo --out $EVAL_ROOT/grpo_dataset_d_size_400.jsonl --load-in-4bit --resume

# zero-shot on dataset_d_size_800
python3 batch_inference.py --test-file dataset_d_size_800.jsonl --base-model $BASE_MODEL  --method zero-shot --out $EVAL_ROOT/zero-shot_dataset_d_size_800.jsonl --load-in-4bit --resume

# sft on dataset_d_size_800
python3 batch_inference.py --test-file dataset_d_size_800.jsonl --base-model $SFT_MODEL  --method sft --out $EVAL_ROOT/sft_dataset_d_size_800.jsonl --load-in-4bit --resume

# grpo on dataset_d_size_800
python3 batch_inference.py --test-file dataset_d_size_800.jsonl --base-model $SFT_MODEL --adapter $GRPO_ADAPTER --method grpo --out $EVAL_ROOT/grpo_dataset_d_size_800.jsonl --load-in-4bit --resume

# zero-shot on dataset_e_domains_1
python3 batch_inference.py --test-file dataset_e_domains_1.jsonl --base-model $BASE_MODEL  --method zero-shot --out $EVAL_ROOT/zero-shot_dataset_e_domains_1.jsonl --load-in-4bit --resume

# sft on dataset_e_domains_1
python3 batch_inference.py --test-file dataset_e_domains_1.jsonl --base-model $SFT_MODEL  --method sft --out $EVAL_ROOT/sft_dataset_e_domains_1.jsonl --load-in-4bit --resume

# grpo on dataset_e_domains_1
python3 batch_inference.py --test-file dataset_e_domains_1.jsonl --base-model $SFT_MODEL --adapter $GRPO_ADAPTER --method grpo --out $EVAL_ROOT/grpo_dataset_e_domains_1.jsonl --load-in-4bit --resume

# zero-shot on dataset_e_domains_10
python3 batch_inference.py --test-file dataset_e_domains_10.jsonl --base-model $BASE_MODEL  --method zero-shot --out $EVAL_ROOT/zero-shot_dataset_e_domains_10.jsonl --load-in-4bit --resume

# sft on dataset_e_domains_10
python3 batch_inference.py --test-file dataset_e_domains_10.jsonl --base-model $SFT_MODEL  --method sft --out $EVAL_ROOT/sft_dataset_e_domains_10.jsonl --load-in-4bit --resume

# grpo on dataset_e_domains_10
python3 batch_inference.py --test-file dataset_e_domains_10.jsonl --base-model $SFT_MODEL --adapter $GRPO_ADAPTER --method grpo --out $EVAL_ROOT/grpo_dataset_e_domains_10.jsonl --load-in-4bit --resume

# zero-shot on dataset_e_domains_3
python3 batch_inference.py --test-file dataset_e_domains_3.jsonl --base-model $BASE_MODEL  --method zero-shot --out $EVAL_ROOT/zero-shot_dataset_e_domains_3.jsonl --load-in-4bit --resume

# sft on dataset_e_domains_3
python3 batch_inference.py --test-file dataset_e_domains_3.jsonl --base-model $SFT_MODEL  --method sft --out $EVAL_ROOT/sft_dataset_e_domains_3.jsonl --load-in-4bit --resume

# grpo on dataset_e_domains_3
python3 batch_inference.py --test-file dataset_e_domains_3.jsonl --base-model $SFT_MODEL --adapter $GRPO_ADAPTER --method grpo --out $EVAL_ROOT/grpo_dataset_e_domains_3.jsonl --load-in-4bit --resume

# zero-shot on dataset_e_domains_5
python3 batch_inference.py --test-file dataset_e_domains_5.jsonl --base-model $BASE_MODEL  --method zero-shot --out $EVAL_ROOT/zero-shot_dataset_e_domains_5.jsonl --load-in-4bit --resume

# sft on dataset_e_domains_5
python3 batch_inference.py --test-file dataset_e_domains_5.jsonl --base-model $SFT_MODEL  --method sft --out $EVAL_ROOT/sft_dataset_e_domains_5.jsonl --load-in-4bit --resume

# grpo on dataset_e_domains_5
python3 batch_inference.py --test-file dataset_e_domains_5.jsonl --base-model $SFT_MODEL --adapter $GRPO_ADAPTER --method grpo --out $EVAL_ROOT/grpo_dataset_e_domains_5.jsonl --load-in-4bit --resume
