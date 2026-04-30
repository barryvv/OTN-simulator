#!/usr/bin/env python3
"""Orchestrator for generating stress test data (S1-S4).

Generates test JSONL files for each stress scenario:
  S1: Equalized bbe_max (all events produce ~35 bbe_max via --equalize-bbe)
  S2: Multi-failure (2-3 concurrent failures via otn_simulator)
  S3: High noise (±80% metric variation via --noise-scale 4.0)
  S4: Partial observability (masks timeseries + pattern_features)

Usage:
  # Generate all stress test scenarios
  python generate_stress_tests.py \
    --test-file outputs/grpo_data/test.jsonl \
    --topology topology_noregen.yaml \
    --roles roles.yaml \
    --simulator simulator.yaml \
    --lightpaths regen_compare/noregen/topology/lightpaths.txt \
    --rules "outputs/Static files/rule_database.csv" \
    --outdir outputs/stress_tests

  # Generate only specific scenarios
  python generate_stress_tests.py --scenarios S2 S3 ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from generate_grpo_data import build_grpo_example


# ---------------------------------------------------------------------------
# S1: Equalized bbe_max (all events produce same ~35 bbe_max)
# ---------------------------------------------------------------------------

def generate_s1(
    topology_yaml: Path,
    roles_yaml: Path,
    simulator_yaml: Path,
    lightpaths_file: Path,
    rules_path: Path,
    outdir: Path,
    n_seeds: int = 10,
    seed: int = 42,
) -> None:
    """S1: Equalized bbe_max — destroys bbe_max as a discriminator.

    Runs the simulator with --equalize-bbe for each event type x seed.
    All root-cause events produce bbe_max ≈ 35, so the baseline's
    closest-match lookup table becomes useless. Only temporal patterns
    (pattern_kind) and secondary features remain for discrimination.
    """
    print("\n[S1] Equalized bbe_max")
    print("=" * 60)

    s1_dir = outdir / "S1_equalized_bbe"
    s1_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = s1_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    events = [
        "Fiber_Aging", "Fiber_Crack", "Fiber_Cut",
        "Line_Disconnect", "XCON_Port_Down",
        "XCON_Buffer_Overflow", "XCON_Fabric_Fault",
    ]

    board_map = {
        "Fiber_Aging": ("ROADM0-OA002$0", "RELAY"),
        "Fiber_Crack": ("ROADM0-OA002$0", "RELAY"),
        "Fiber_Cut": ("ROADM0-OA002$0", "RELAY"),
        "Line_Disconnect": ("ROADM0-Line001$0", "SRC"),
        "XCON_Port_Down": ("ROADM0-XCON001$0", "SRC"),
        "XCON_Buffer_Overflow": ("ROADM0-XCON001$0", "SRC"),
        "XCON_Fabric_Fault": ("ROADM0-XCON001$0", "SRC"),
    }

    examples: list[dict] = []

    for event in events:
        for seed_offset in range(n_seeds):
            run_seed = seed + seed_offset
            run_name = f"eqbbe_{event.lower()}_seed{run_seed}"
            run_dir = runs_dir / run_name

            print(f"  [run] {run_name}")

            run_dir.mkdir(parents=True, exist_ok=True)
            failure_csv = run_dir / "failure.csv"
            board, role = board_map.get(event, ("ROADM0-OA002$0", "RELAY"))
            failure_csv.write_text(
                "Board,Event,Time,Severity,DurationMs,BoardRole\n"
                f"{board},{event},2025-01-15 10:00:00,Critical,15000,{role}\n",
                encoding="utf-8",
            )

            cmd = [
                sys.executable, "otn_simulator.py",
                "--topology", str(topology_yaml),
                "--roles", str(roles_yaml),
                "--simulator", str(simulator_yaml),
                "--outdir", str(run_dir / "data_outputs"),
                "--seed", str(run_seed),
                "--failures", str(failure_csv),
                "--equalize-bbe",
                "--alarm-flow",
                "--alarm-outdir", str(run_dir),
                "--alarm-rules", str(rules_path),
            ]
            if lightpaths_file and lightpaths_file.exists():
                cmd.extend(["--paths-from", str(lightpaths_file)])

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    print(f"    [warn] simulator failed: {result.stderr[:200]}")
                    continue
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                print(f"    [warn] {exc}")
                continue

            ex = build_grpo_example(
                run_dir=run_dir,
                rules_path=rules_path,
                timeseries_root=run_dir,
            )
            if ex:
                ex["scenario"] = "S1_equalized_bbe"
                ex["run_name"] = run_name
                examples.append(ex)
                print(f"    [ok]")
            else:
                print(f"    [warn] could not build example")

    test_path = s1_dir / "test.jsonl"
    with test_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  [ok] wrote {len(examples)} equalized-bbe examples -> {test_path}")


# ---------------------------------------------------------------------------
# S2: Multi-Failure
# ---------------------------------------------------------------------------

def generate_s2(
    topology_yaml: Path,
    roles_yaml: Path,
    simulator_yaml: Path,
    lightpaths_file: Path,
    rules_path: Path,
    outdir: Path,
    n_seeds: int = 15,
    failure_counts: list[int] | None = None,
    seed: int = 42,
) -> None:
    """S2: Generate multi-failure scenarios via otn_simulator.

    Runs otn_simulator with --multi-failure N for N=2,3 with multiple seeds.
    Then builds test JSONL from the resulting alarm flow data.
    """
    print("\n[S2] Multi-Failure Scenarios")
    print("=" * 60)

    if failure_counts is None:
        failure_counts = [2, 3]

    s2_dir = outdir / "S2_multi_failure"
    s2_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = s2_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    examples: list[dict] = []

    for n_fail in failure_counts:
        for seed_offset in range(n_seeds):
            run_seed = seed + seed_offset
            run_name = f"multi_{n_fail}f_seed{run_seed}"
            run_dir = runs_dir / run_name

            print(f"  [run] {run_name}: {n_fail} failures, seed={run_seed}")

            # Run otn_simulator with --multi-failure
            cmd = [
                sys.executable, "otn_simulator.py",
                "--topology", str(topology_yaml),
                "--roles", str(roles_yaml),
                "--simulator", str(simulator_yaml),
                "--outdir", str(run_dir / "data_outputs"),
                "--seed", str(run_seed),
                "--multi-failure", str(n_fail),
                "--alarm-flow",
                "--alarm-outdir", str(run_dir),
                "--alarm-rules", str(rules_path),
            ]
            if lightpaths_file and lightpaths_file.exists():
                cmd.extend(["--paths-from", str(lightpaths_file)])

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    print(f"    [warn] simulator failed: {result.stderr[:200]}")
                    continue
            except subprocess.TimeoutExpired:
                print(f"    [warn] simulator timed out")
                continue
            except FileNotFoundError:
                print(f"    [warn] otn_simulator.py not found")
                continue

            # Multi-failure simulator writes failure_mw.csv; copy as failure.csv
            failure_mw = run_dir / "failure_mw.csv"
            failure_link = run_dir / "failure.csv"
            if failure_mw.exists() and not failure_link.exists():
                shutil.copy2(failure_mw, failure_link)

            # Build GRPO example from the run
            ex = build_grpo_example(
                run_dir=run_dir,
                rules_path=rules_path,
                timeseries_root=run_dir,
            )
            if ex:
                # Add multi-failure metadata
                failure_csv = run_dir / "failure.csv"
                if failure_csv.exists():
                    df_fail = pd.read_csv(failure_csv)
                    fcols = {str(c).strip().lower(): c for c in df_fail.columns}
                    c_board = fcols.get("board", df_fail.columns[0])
                    c_event = fcols.get("event", df_fail.columns[1])

                    # Store ALL root causes (not just the first)
                    root_events = df_fail[c_event].tolist()
                    root_boards = df_fail[c_board].tolist()
                    ex["ground_truth_root_events"] = json.dumps(root_events)
                    ex["ground_truth_root_boards"] = json.dumps(root_boards)
                    ex["n_failures"] = len(df_fail)
                    ex["scenario"] = "S2_multi_failure"
                    ex["run_name"] = run_name

                examples.append(ex)
                print(f"    [ok] {len(root_events)} failures recorded")
            else:
                print(f"    [warn] could not build example from {run_dir}")

    # Write test JSONL
    test_path = s2_dir / "test.jsonl"
    with test_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  [ok] wrote {len(examples)} multi-failure examples -> {test_path}")


# ---------------------------------------------------------------------------
# S3: High Noise
# ---------------------------------------------------------------------------

def generate_s3(
    topology_yaml: Path,
    roles_yaml: Path,
    simulator_yaml: Path,
    lightpaths_file: Path,
    rules_path: Path,
    outdir: Path,
    noise_scale: float = 4.0,
    n_seeds: int = 10,
    seed: int = 42,
) -> None:
    """S3: Generate high-noise scenarios.

    Runs otn_simulator with --noise-scale to double the noise level,
    causing bbe_max ranges to overlap between event types.
    """
    print(f"\n[S3] High Noise (noise_scale={noise_scale})")
    print("=" * 60)

    s3_dir = outdir / "S3_high_noise"
    s3_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = s3_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Event types to test
    events = [
        "Fiber_Aging", "Fiber_Crack", "Fiber_Cut",
        "Line_Disconnect", "XCON_Port_Down",
        "XCON_Buffer_Overflow", "XCON_Fabric_Fault",
    ]

    examples: list[dict] = []

    for event in events:
        for seed_offset in range(n_seeds):
            run_seed = seed + seed_offset
            run_name = f"noise_{event.lower()}_s{noise_scale}_seed{run_seed}"
            run_dir = runs_dir / run_name

            print(f"  [run] {run_name}")

            # We need a failure.csv for this event
            failure_csv = run_dir / "failure.csv"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Create a simple failure CSV
            # Board selection depends on event type
            board_map = {
                "Fiber_Aging": ("ROADM0-OA002$0", "RELAY"),
                "Fiber_Crack": ("ROADM0-OA002$0", "RELAY"),
                "Fiber_Cut": ("ROADM0-OA002$0", "RELAY"),
                "Line_Disconnect": ("ROADM0-Line001$0", "SRC"),
                "XCON_Port_Down": ("ROADM0-XCON001$0", "SRC"),
                "XCON_Buffer_Overflow": ("ROADM0-XCON001$0", "SRC"),
                "XCON_Fabric_Fault": ("ROADM0-XCON001$0", "SRC"),
            }
            board, role = board_map.get(event, ("ROADM0-OA002$0", "RELAY"))
            failure_csv.write_text(
                "Board,Event,Time,Severity,DurationMs,BoardRole\n"
                f"{board},{event},2025-01-15 10:00:00,Critical,15000,{role}\n",
                encoding="utf-8",
            )

            # Run simulator with --noise-scale
            cmd = [
                sys.executable, "otn_simulator.py",
                "--topology", str(topology_yaml),
                "--roles", str(roles_yaml),
                "--simulator", str(simulator_yaml),
                "--outdir", str(run_dir / "data_outputs"),
                "--seed", str(run_seed),
                "--failures", str(failure_csv),
                "--noise-scale", str(noise_scale),
                "--alarm-flow",
                "--alarm-outdir", str(run_dir),
                "--alarm-rules", str(rules_path),
            ]
            if lightpaths_file and lightpaths_file.exists():
                cmd.extend(["--paths-from", str(lightpaths_file)])

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    print(f"    [warn] simulator failed: {result.stderr[:200]}")
                    continue
            except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
                print(f"    [warn] {exc}")
                continue

            # Build example
            ex = build_grpo_example(
                run_dir=run_dir,
                rules_path=rules_path,
                timeseries_root=run_dir,
            )
            if ex:
                ex["scenario"] = "S3_high_noise"
                ex["noise_scale"] = noise_scale
                ex["run_name"] = run_name
                examples.append(ex)
                print(f"    [ok]")
            else:
                print(f"    [warn] could not build example")

    # Write test JSONL
    test_path = s3_dir / "test.jsonl"
    with test_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  [ok] wrote {len(examples)} noisy examples -> {test_path}")


# ---------------------------------------------------------------------------
# S4: Partial Observability
# ---------------------------------------------------------------------------

def generate_s4(
    test_file: Path,
    outdir: Path,
    mask_fraction: float = 0.4,
    mask_peak: bool = True,
    seed: int = 42,
) -> None:
    """S4: Mask timeseries data to simulate partial observability.

    Modifies the user prompts in the test JSONL by removing timeseries
    data for a fraction of boards. Optionally targets the peak board.
    """
    print(f"\n[S4] Partial Observability (mask={mask_fraction:.0%}, peak={mask_peak})")
    print("=" * 60)

    s4_dir = outdir / "S4_partial_observability"
    s4_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Load test examples
    examples: list[dict] = []
    with test_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    masked_examples: list[dict] = []
    for idx, ex in enumerate(examples):
        prompt_msgs = ex.get("prompt", [])
        new_msgs = []
        for msg in prompt_msgs:
            if msg.get("role") != "user":
                new_msgs.append(msg)
                continue

            content = msg["content"]
            masked_content = _mask_timeseries_in_prompt(
                content, mask_fraction, mask_peak, rng,
            )
            new_msgs.append({"role": "user", "content": masked_content})

        masked_ex = dict(ex)
        masked_ex["prompt"] = new_msgs
        masked_ex["scenario"] = "S4_partial_observability"
        masked_ex["mask_fraction"] = mask_fraction
        masked_ex["mask_peak"] = mask_peak
        masked_examples.append(masked_ex)

    # Write test JSONL
    test_path = s4_dir / "test.jsonl"
    with test_path.open("w", encoding="utf-8") as f:
        for ex in masked_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  [ok] wrote {len(masked_examples)} masked examples -> {test_path}")


def _mask_timeseries_in_prompt(
    content: str,
    mask_fraction: float,
    mask_peak: bool,
    rng: random.Random,
) -> str:
    """Remove timeseries data for a fraction of boards from the prompt.

    Finds the timeseries section and removes data blocks for masked boards.
    If mask_peak=True, always masks the peak board (highest BBE).
    """
    import re

    # Find the timeseries summary section
    # It contains lines like: "Board ROADM0-OA002$0: bbe_max=55.0, ..."
    lines = content.split("\n")
    ts_start = None
    ts_end = None
    board_lines: list[tuple[int, str]] = []  # (line_idx, board_name)

    for i, line in enumerate(lines):
        # Detect timeseries section markers (e.g. "timeseries per-board bbe_max:")
        if "timeseries" in line.lower() and ("summary" in line.lower() or "files" in line.lower() or "bbe_max" in line.lower()):
            ts_start = i
        if ts_start is not None:
            # Match board data lines
            m = re.match(r".*?(ROADM\d+-\w+\$\d+).*?bbe_max=([\d.]+)", line)
            if m:
                board_lines.append((i, m.group(1)))

    if not board_lines:
        # Can't find board data — return as-is with a note
        return content + "\n\n[NOTE: Some sensor data is unavailable due to monitoring gaps.]"

    n_boards = len(board_lines)
    n_mask = max(1, int(n_boards * mask_fraction))

    # Determine which boards to mask
    board_indices = list(range(n_boards))
    rng.shuffle(board_indices)
    mask_set = set(board_indices[:n_mask])

    if mask_peak:
        # Find the peak board (first in the list since they're sorted by bbe_max desc)
        # If it's not already masked, add it
        mask_set.add(0)
        # Ensure we don't exceed mask count
        if len(mask_set) > n_mask:
            # Remove one random non-peak board
            non_peak = [i for i in mask_set if i != 0]
            if non_peak:
                mask_set.discard(rng.choice(non_peak))

    # Mask the selected board lines
    mask_line_indices = {board_lines[i][0] for i in mask_set}
    masked_boards = [board_lines[i][1] for i in mask_set]

    new_lines = []
    for i, line in enumerate(lines):
        if i in mask_line_indices:
            # Replace with a sensor unavailable note
            board_name = None
            for bi, bn in board_lines:
                if bi == i:
                    board_name = bn
                    break
            new_lines.append(f"  {board_name}: [SENSOR DATA UNAVAILABLE - monitoring gap]")
        else:
            new_lines.append(line)

    # When peak board is masked, also mask the pattern_features line
    # This prevents the baseline from regex-parsing bbe_max, pattern_kind, etc.
    if mask_peak and 0 in mask_set:
        final_lines = []
        for line in new_lines:
            if line.strip().startswith("pattern_features:"):
                line = re.sub(r"bbe_max=[\d.]+", "bbe_max=[MASKED]", line)
                line = re.sub(r"pattern_kind=\w+", "pattern_kind=[MASKED]", line)
                line = re.sub(r"peak_board=[\w\-\$]+", "peak_board=[MASKED]", line)
                line = re.sub(r"peak_node=\w+", "peak_node=[MASKED]", line)
                line = re.sub(r"burst_regions=\d+", "burst_regions=[MASKED]", line)
                line = re.sub(r"longest_burst_frac=[\d.]+", "longest_burst_frac=[MASKED]", line)
                line = re.sub(r"bber_bbe_ratio=[\d.]+", "bber_bbe_ratio=[MASKED]", line)
                line = re.sub(r"es_frac=[\d.]+", "es_frac=[MASKED]", line)
            final_lines.append(line)
        new_lines = final_lines

    result = "\n".join(new_lines)

    # Also update pattern_features if the peak board was masked
    if mask_peak and board_lines:
        peak_board = board_lines[0][1]
        if peak_board in masked_boards:
            result += (
                "\n\n[WARNING: Peak board sensor data is unavailable. "
                "Use neighboring board data and topology analysis to infer the root cause.]"
            )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate stress test data for S1-S4 scenarios."
    )
    ap.add_argument(
        "--test-file", type=Path,
        default=Path("outputs/grpo_data/test.jsonl"),
        help="Existing IID test JSONL (used by S1, S4)",
    )
    ap.add_argument(
        "--topology", type=Path,
        default=Path("topology_noregen.yaml"),
        help="Topology YAML for simulation (S2, S3)",
    )
    ap.add_argument(
        "--roles", type=Path,
        default=Path("roles.yaml"),
        help="Roles YAML for simulation",
    )
    ap.add_argument(
        "--simulator", type=Path,
        default=Path("simulator.yaml"),
        help="Simulator config YAML",
    )
    ap.add_argument(
        "--lightpaths", type=Path,
        default=Path("regen_compare/noregen/topology/lightpaths.txt"),
        help="Lightpaths file",
    )
    ap.add_argument(
        "--rules", type=Path,
        default=Path("outputs/Static files/rule_database.csv"),
        help="Full rule_database.csv",
    )
    ap.add_argument(
        "--outdir", type=Path,
        default=Path("outputs/stress_tests"),
        help="Output directory for all stress test data",
    )
    ap.add_argument(
        "--scenarios", nargs="*", default=["S1", "S2", "S3", "S4"],
        choices=["S1", "S2", "S3", "S4"],
        help="Which scenarios to generate (default: all)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seeds", type=int, default=15,
                    help="Number of random seeds for S1/S2/S3 (default 15)")
    ap.add_argument("--noise-scale", type=float, default=4.0,
                    help="Noise scale for S3 (default 4.0 = ±80%%)")
    ap.add_argument("--mask-fraction", type=float, default=0.4,
                    help="Fraction of boards to mask in S4 (default 0.4)")

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    scenarios = [s.upper() for s in args.scenarios]

    if "S1" in scenarios:
        missing = []
        for name, path in [("topology", args.topology), ("roles", args.roles),
                           ("simulator", args.simulator)]:
            if not path.exists():
                missing.append(f"{name}={path}")
        if missing:
            print(f"[warn] S1 requires: {', '.join(missing)}, skipping")
        else:
            generate_s1(
                args.topology, args.roles, args.simulator,
                args.lightpaths, args.rules, args.outdir,
                n_seeds=args.n_seeds, seed=args.seed,
            )

    if "S2" in scenarios:
        missing = []
        for name, path in [("topology", args.topology), ("roles", args.roles),
                           ("simulator", args.simulator)]:
            if not path.exists():
                missing.append(f"{name}={path}")
        if missing:
            print(f"[warn] S2 requires: {', '.join(missing)}, skipping")
        else:
            generate_s2(
                args.topology, args.roles, args.simulator,
                args.lightpaths, args.rules, args.outdir,
                n_seeds=args.n_seeds, seed=args.seed,
            )

    if "S3" in scenarios:
        missing = []
        for name, path in [("topology", args.topology), ("roles", args.roles),
                           ("simulator", args.simulator)]:
            if not path.exists():
                missing.append(f"{name}={path}")
        if missing:
            print(f"[warn] S3 requires: {', '.join(missing)}, skipping")
        else:
            generate_s3(
                args.topology, args.roles, args.simulator,
                args.lightpaths, args.rules, args.outdir,
                noise_scale=args.noise_scale, n_seeds=args.n_seeds, seed=args.seed,
            )

    if "S4" in scenarios:
        if not args.test_file.exists():
            print(f"[warn] S4 requires --test-file ({args.test_file} not found), skipping")
        else:
            generate_s4(
                args.test_file, args.outdir,
                mask_fraction=args.mask_fraction,
                mask_peak=True, seed=args.seed,
            )

    print(f"\n[done] stress test data generated in {args.outdir}")


if __name__ == "__main__":
    main()
