#!/usr/bin/env python3
"""Mask timeseries files to simulate partial observability (S4).

Takes a data_outputs directory and randomly deletes timeseries CSV files
to simulate sensor failures or monitoring gaps.

Usage:
  # Mask 40% of boards, targeting the peak board
  python mask_metrics.py \
    --data-dir data_outputs_noregen \
    --out-dir data_outputs_masked \
    --mask-fraction 0.4 \
    --mask-peak-board \
    --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd


def mask_timeseries(
    data_dir: Path,
    out_dir: Path,
    mask_fraction: float = 0.4,
    mask_peak_board: bool = True,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Copy data_dir to out_dir, masking a fraction of timeseries files.

    Args:
        data_dir: Source directory with Node_*/timeseries_*.csv
        out_dir: Destination directory (will be created)
        mask_fraction: Fraction of board timeseries to remove
        mask_peak_board: If True, always mask the board with highest BBE
        seed: Random seed

    Returns:
        Dict with 'masked' and 'kept' board lists
    """
    rng = random.Random(seed)

    # Copy entire directory first
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(data_dir, out_dir)

    # Find all timeseries files and their peak BBE
    ts_files = sorted(out_dir.glob("Node_*/timeseries_*.csv"))
    if not ts_files:
        return {"masked": [], "kept": []}

    # Map board names to their files and peak BBE
    board_files: dict[str, list[Path]] = {}
    board_bbe: dict[str, float] = {}

    for f in ts_files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        # Find board column
        board_col = None
        for col in df.columns:
            if col.strip().lower() == "board":
                board_col = col
                break
        if board_col is None:
            continue

        bbe_col = None
        for col in df.columns:
            if col.strip().upper() == "BBE":
                bbe_col = col
                break

        for board_name in df[board_col].dropna().unique():
            bn = str(board_name).strip()
            board_files.setdefault(bn, []).append(f)
            if bbe_col:
                mask = df[board_col].astype(str).str.strip() == bn
                bbe_max = pd.to_numeric(df.loc[mask, bbe_col], errors="coerce").max()
                if bn not in board_bbe or bbe_max > board_bbe[bn]:
                    board_bbe[bn] = float(bbe_max) if pd.notna(bbe_max) else 0.0

    all_boards = list(board_files.keys())
    if not all_boards:
        return {"masked": [], "kept": []}

    n_mask = max(1, int(len(all_boards) * mask_fraction))

    # Determine which boards to mask
    boards_to_mask: set[str] = set()

    if mask_peak_board and board_bbe:
        peak_board = max(board_bbe, key=board_bbe.get)
        boards_to_mask.add(peak_board)

    # Randomly select remaining boards to mask
    remaining = [b for b in all_boards if b not in boards_to_mask]
    rng.shuffle(remaining)
    for b in remaining:
        if len(boards_to_mask) >= n_mask:
            break
        boards_to_mask.add(b)

    # Remove masked boards' data from the files
    masked_boards = sorted(boards_to_mask)
    kept_boards = sorted(set(all_boards) - boards_to_mask)

    for board in boards_to_mask:
        for f in board_files.get(board, []):
            if f.exists():
                try:
                    df = pd.read_csv(f)
                    board_col = None
                    for col in df.columns:
                        if col.strip().lower() == "board":
                            board_col = col
                            break
                    if board_col:
                        # Remove rows for this board
                        mask = df[board_col].astype(str).str.strip() != board
                        df_filtered = df[mask]
                        if df_filtered.empty:
                            f.unlink()
                        else:
                            df_filtered.to_csv(f, index=False)
                except Exception:
                    pass

    return {"masked": masked_boards, "kept": kept_boards}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mask timeseries files to simulate partial observability."
    )
    ap.add_argument(
        "--data-dir", type=Path, required=True,
        help="Source data_outputs directory",
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True,
        help="Output directory for masked data",
    )
    ap.add_argument(
        "--mask-fraction", type=float, default=0.4,
        help="Fraction of boards to mask (default 0.4)",
    )
    ap.add_argument(
        "--mask-peak-board", action="store_true",
        help="Always mask the board with highest BBE",
    )
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"[error] data dir not found: {args.data_dir}")

    result = mask_timeseries(
        args.data_dir, args.out_dir,
        mask_fraction=args.mask_fraction,
        mask_peak_board=args.mask_peak_board,
        seed=args.seed,
    )

    print(f"[ok] masked {len(result['masked'])} boards, kept {len(result['kept'])}")
    print(f"  masked: {', '.join(result['masked'][:5])}{'...' if len(result['masked']) > 5 else ''}")
    print(f"  kept:   {', '.join(result['kept'][:5])}{'...' if len(result['kept']) > 5 else ''}")
    print(f"  output: {args.out_dir}")


if __name__ == "__main__":
    main()
