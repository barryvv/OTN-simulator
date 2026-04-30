#!/usr/bin/env python3
"""Master runner for SFC-BFL experiment datasets.

Usage:
    # Single dataset, pilot mode (debug scale: 8 nodes, ~368 boards)
    python -m sfc_bfl.run_all --dataset A --pilot --outdir outputs/sfc_bfl/dataset_a_pilot

    # All datasets, full scale
    python -m sfc_bfl.run_all --dataset all --outdir outputs/sfc_bfl/

    # Custom topology size
    python -m sfc_bfl.run_all --dataset A --pilot --num-roadms 8 --eg 3 --outdir /tmp/test_a
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("ARANGO_SKIP", "1")


def main():
    ap = argparse.ArgumentParser(description="SFC-BFL Experiment Dataset Generator")
    ap.add_argument("--dataset", type=str, default="all",
                    help="Dataset to generate: A, B, C, D, E, or 'all' (default: all)")
    ap.add_argument("--outdir", type=str, default="outputs/sfc_bfl",
                    help="Output directory (default: outputs/sfc_bfl)")
    ap.add_argument("--pilot", action="store_true",
                    help="Generate small pilot datasets (~10%% of full OW counts)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--ow-length", type=int, default=None,
                    help="Override OW length in timesteps (default: 900 for datasets)")
    ap.add_argument("--num-roadms", type=int, default=None,
                    help="Number of ROADM nodes (default: 5 for full, 8 for pilot)")
    ap.add_argument("--eg", type=int, default=None,
                    help="ELECTRICAL_GROUPS per ROADM (default: 29 for full, 3 for pilot)")

    args = ap.parse_args()

    # Default topology size: 8 nodes per requirement; EG controls boards/node
    num_roadms = args.num_roadms
    eg = args.eg
    if num_roadms is None:
        num_roadms = 8
    if eg is None:
        eg = 3 if args.pilot else 5

    datasets = args.dataset.upper().split(",") if args.dataset.lower() != "all" else ["TRAIN", "A", "B", "C", "D", "E"]

    topo_kwargs = {"num_roadms": num_roadms, "electrical_groups": eg}

    for ds in datasets:
        ds = ds.strip()
        ds_outdir = os.path.join(args.outdir, f"dataset_{ds.lower()}")
        if args.pilot:
            ds_outdir += "_pilot"

        print(f"\n{'='*60}")
        print(f"  Generating Dataset {ds} → {ds_outdir}")
        print(f"  pilot={args.pilot}, seed={args.seed}, ow_length={args.ow_length}")
        print(f"  num_roadms={num_roadms}, electrical_groups={eg}")
        print(f"{'='*60}\n")

        if ds == "TRAIN":
            from .dataset_train import generate_dataset_train
            generate_dataset_train(ds_outdir, pilot=args.pilot, seed=args.seed,
                                   ow_length=args.ow_length, electrical_groups=eg)
        elif ds == "A":
            from .dataset_a import generate_dataset_a
            generate_dataset_a(ds_outdir, pilot=args.pilot, seed=args.seed,
                               ow_length=args.ow_length, **topo_kwargs)
        elif ds == "B":
            from .dataset_b import generate_dataset_b
            generate_dataset_b(ds_outdir, pilot=args.pilot, seed=args.seed,
                               ow_length=args.ow_length, **topo_kwargs)
        elif ds == "C":
            from .dataset_c import generate_dataset_c
            generate_dataset_c(ds_outdir, pilot=args.pilot, seed=args.seed,
                               ow_length=args.ow_length, **topo_kwargs)
        elif ds == "D":
            from .dataset_d import generate_dataset_d
            generate_dataset_d(ds_outdir, pilot=args.pilot, seed=args.seed,
                               ow_length=args.ow_length)
        elif ds == "E":
            from .dataset_e import generate_dataset_e
            generate_dataset_e(ds_outdir, pilot=args.pilot, seed=args.seed,
                               ow_length=args.ow_length, **topo_kwargs)
        else:
            print(f"[warn] unknown dataset: {ds}")

    print(f"\n[sfc_bfl] all requested datasets generated in {args.outdir}")


if __name__ == "__main__":
    main()
