#!/usr/bin/env python3
"""Benchmark orchestrator for SFC-BFL LLM-Agent evaluation.

Runs the full experiment matrix:
  Methods: zero-shot, SFT, GRPO, Agent (ReAct)
  Datasets: A, B, C, D (size sweep), E (domain sweep)

For non-agent methods: convert_to_prompts → batch_inference → evaluate
For agent method: agent_runner → evaluate

Usage:
  # Generate prompts for all datasets
  python -m sfc_bfl.benchmark --stage prompts --dataset-root outputs/sfc_bfl

  # Run evaluation on pre-computed outputs
  python -m sfc_bfl.benchmark --stage evaluate \
    --results-root outputs/sfc_bfl_eval

  # Generate benchmark shell scripts for the server
  python -m sfc_bfl.benchmark --stage scripts \
    --dataset-root outputs/sfc_bfl --prompts-root outputs/sfc_bfl_prompts
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

DATASET_NAMES = ["dataset_a", "dataset_b", "dataset_c", "dataset_d", "dataset_e"]
METHODS = ["zero-shot", "sft", "grpo", "agent"]

# Pilot datasets are smaller (100 OWs) for quick iteration
PILOT_SUFFIX = "_pilot"


def discover_datasets(
    root: Path,
    use_pilot: bool = False,
) -> dict[str, list[Path]]:
    """Discover all SFC-BFL dataset directories.

    Returns {dataset_name: [paths]} where paths may be multiple
    for Dataset D (size_*) and E (domains_*).
    """
    suffix = PILOT_SUFFIX if use_pilot else ""
    result: dict[str, list[Path]] = {}

    for name in DATASET_NAMES:
        ddir = root / f"{name}{suffix}"
        full_dir = root / "full" / name

        # Check pilot first, then full
        for candidate in [ddir, full_dir]:
            if not candidate.exists():
                continue

            # Check if it has subdirectories (D/E) or is a direct dataset
            if (candidate / "ow_labels.jsonl").exists():
                result[name] = [candidate]
                break
            else:
                subdirs = sorted([
                    d for d in candidate.iterdir()
                    if d.is_dir() and (d / "ow_labels.jsonl").exists()
                ])
                if subdirs:
                    result[name] = subdirs
                    break

    return result


# ---------------------------------------------------------------------------
# Stage: Generate prompts
# ---------------------------------------------------------------------------

def generate_prompts(
    dataset_root: Path,
    prompts_root: Path,
    rules_path: Path,
    use_pilot: bool = False,
) -> None:
    """Generate JSONL prompts for all discovered datasets."""
    datasets = discover_datasets(dataset_root, use_pilot=use_pilot)

    if not datasets:
        print(f"[error] no datasets found in {dataset_root}")
        return

    prompts_root.mkdir(parents=True, exist_ok=True)

    for name, paths in datasets.items():
        for dpath in paths:
            # Build output name
            if len(paths) == 1:
                out_name = f"{name}.jsonl"
            else:
                out_name = f"{name}_{dpath.name}.jsonl"

            out_path = prompts_root / out_name
            if out_path.exists():
                print(f"[skip] {out_path} already exists")
                continue

            print(f"[info] converting {dpath} -> {out_path}")
            cmd = [
                sys.executable, "-m", "sfc_bfl.convert_to_prompts",
                "--dataset-dir", str(dpath),
                "--rules", str(rules_path),
                "--out", str(out_path),
            ]
            subprocess.run(cmd, check=True)

    print(f"[ok] prompts generated in {prompts_root}")


# ---------------------------------------------------------------------------
# Stage: Generate server scripts
# ---------------------------------------------------------------------------

def generate_server_scripts(
    prompts_root: Path,
    eval_root: Path,
    output_script: Path,
    base_model: str = "Qwen/Qwen2.5-14B-Instruct",
    sft_model: str = "models/qwen-14b-sft-v6.1-merged",
    grpo_adapter: str = "adapters/qwen-14b-sft-v6.1-grpo",
) -> None:
    """Generate a shell script to run all inference jobs on the server."""
    lines = [
        "#!/bin/bash",
        "# SFC-BFL Benchmark Inference Script",
        "# Run this on the GPU server (hph8)",
        "set -e",
        "",
        f'EVAL_ROOT="{eval_root}"',
        f'BASE_MODEL="{base_model}"',
        f'SFT_MODEL="{sft_model}"',
        f'GRPO_ADAPTER="{grpo_adapter}"',
        "",
        "mkdir -p $EVAL_ROOT",
        "",
    ]

    # Find all prompt files
    prompt_files = sorted(prompts_root.glob("*.jsonl"))
    if not prompt_files:
        print(f"[warn] no prompt files found in {prompts_root}")

    for pf in prompt_files:
        dataset_tag = pf.stem  # e.g. "dataset_a" or "dataset_d_size_200"
        for method in ["zero-shot", "sft", "grpo"]:
            out_file = f"$EVAL_ROOT/{method}_{dataset_tag}.jsonl"

            if method == "zero-shot":
                model_arg = "$BASE_MODEL"
                adapter_arg = ""
            elif method == "sft":
                model_arg = "$SFT_MODEL"
                adapter_arg = ""
            elif method == "grpo":
                model_arg = "$SFT_MODEL"
                adapter_arg = f"--adapter $GRPO_ADAPTER"

            lines.append(f"# {method} on {dataset_tag}")
            lines.append(
                f"python3 batch_inference.py "
                f"--test-file {pf.name} "
                f"--base-model {model_arg} "
                f"{adapter_arg} "
                f"--method {method} "
                f"--out {out_file} "
                f"--load-in-4bit --resume"
            )
            lines.append("")

    output_script.parent.mkdir(parents=True, exist_ok=True)
    output_script.write_text("\n".join(lines), encoding="utf-8")
    output_script.chmod(0o755)
    print(f"[ok] wrote server script -> {output_script}")


# ---------------------------------------------------------------------------
# Stage: Evaluate
# ---------------------------------------------------------------------------

def evaluate_all(
    eval_root: Path,
    results_root: Path,
) -> None:
    """Run evaluate.py on all output files in eval_root."""
    output_files = sorted(eval_root.glob("*.jsonl"))
    if not output_files:
        print(f"[error] no output files found in {eval_root}")
        return

    results_root.mkdir(parents=True, exist_ok=True)

    for of in output_files:
        # Parse method and dataset from filename: {method}_{dataset}.jsonl
        stem = of.stem
        parts = stem.split("_", 1)
        if len(parts) < 2:
            continue

        method = parts[0]
        dataset_tag = parts[1]
        model_name = f"{method}/{dataset_tag}"

        results_dir = results_root / stem
        print(f"[info] evaluating {of.name} -> {results_dir}")

        cmd = [
            sys.executable, "evaluate.py",
            "--outputs-file", str(of),
            "--results-dir", str(results_dir),
            "--model-name", model_name,
        ]
        subprocess.run(cmd, check=True)

    print(f"[ok] all evaluations complete, results in {results_root}")


# ---------------------------------------------------------------------------
# Stage: Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results_root: Path) -> None:
    """Print a combined summary table from all evaluation results."""
    import csv

    rows = []
    for results_dir in sorted(results_root.iterdir()):
        summary_csv = results_dir / "results_summary.csv"
        if not summary_csv.exists():
            continue
        with open(summary_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    if not rows:
        print("[info] no results found")
        return

    # Print summary table
    print()
    print("=" * 100)
    print(f"{'Model/Dataset':<35} {'EventAcc':>9} {'BoardAcc':>9} {'CatAcc':>9} "
          f"{'BoardF1':>9} {'AlarmF1':>9} {'E2E':>9} {'N':>5}")
    print("-" * 100)
    for row in rows:
        name = row.get("model_name", "?")
        print(
            f"{name:<35} "
            f"{float(row.get('event_accuracy', 0)):.1%}     "
            f"{float(row.get('board_accuracy', 0)):.1%}     "
            f"{float(row.get('category_accuracy', 0)):.1%}     "
            f"{float(row.get('board_f1_mean', 0)):.1%}     "
            f"{float(row.get('alarm_f1_mean', 0)):.1%}     "
            f"{float(row.get('e2e_score_mean', 0)):.1%}     "
            f"{int(row.get('n_examples', 0)):>5}"
        )
    print("=" * 100)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="SFC-BFL benchmark orchestrator."
    )
    ap.add_argument("--stage", required=True,
                    choices=["prompts", "scripts", "evaluate", "summary"],
                    help="Stage to run")
    ap.add_argument("--dataset-root", type=Path,
                    default=Path("outputs/sfc_bfl"),
                    help="Root directory for SFC-BFL datasets")
    ap.add_argument("--prompts-root", type=Path,
                    default=Path("outputs/sfc_bfl_prompts"),
                    help="Root directory for generated prompts")
    ap.add_argument("--eval-root", type=Path,
                    default=Path("outputs/sfc_bfl_eval"),
                    help="Root directory for inference outputs")
    ap.add_argument("--results-root", type=Path,
                    default=Path("outputs/sfc_bfl_results"),
                    help="Root directory for evaluation results")
    ap.add_argument("--rules", type=Path,
                    default=Path("outputs/Static files/rule_database.csv"))
    ap.add_argument("--pilot", action="store_true",
                    help="Use pilot datasets (100 OWs each)")
    ap.add_argument("--script-out", type=Path,
                    default=Path("run_sfc_bfl_benchmark.sh"),
                    help="Output path for server script")

    # Model paths (for script generation)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--sft-model", default="models/qwen-14b-sft-v6.1-merged")
    ap.add_argument("--grpo-adapter", default="adapters/qwen-14b-sft-v6.1-grpo")

    args = ap.parse_args()

    if args.stage == "prompts":
        generate_prompts(
            args.dataset_root, args.prompts_root, args.rules,
            use_pilot=args.pilot,
        )
    elif args.stage == "scripts":
        generate_server_scripts(
            args.prompts_root, args.eval_root, args.script_out,
            base_model=args.base_model,
            sft_model=args.sft_model,
            grpo_adapter=args.grpo_adapter,
        )
    elif args.stage == "evaluate":
        evaluate_all(args.eval_root, args.results_root)
    elif args.stage == "summary":
        print_summary_table(args.results_root)


if __name__ == "__main__":
    main()
