#!/usr/bin/env python3
"""Create partial rule databases for stress-test S1 (incomplete rule DB).

Removes a fraction of propagation rules from rule_database.csv, simulating
scenarios where documentation is incomplete due to new equipment, firmware
updates, or vendor changes.

Strategy: Remove PROPAGATE rules preferentially over "Report Locally" rules,
since propagation rules are more critical for alarm tracing and their absence
more realistically models incomplete documentation.

Usage:
  python create_partial_rules.py \
    --rules "outputs/Static files/rule_database.csv" \
    --fractions 0.7 0.5 \
    --outdir outputs/stress_tests/partial_rules \
    --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def create_partial_rules(
    rules_path: Path,
    keep_fraction: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a partial rule database by removing rules.

    Preferentially removes PROPAGATE rules over local reporting rules.

    Args:
        rules_path: Path to full rule_database.csv
        keep_fraction: Fraction of rules to keep (e.g., 0.7 = keep 70%)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with reduced rule set
    """
    df = pd.read_csv(rules_path)
    n_total = len(df)
    n_keep = max(1, int(n_total * keep_fraction))
    n_remove = n_total - n_keep

    if n_remove <= 0:
        return df.copy()

    rng = random.Random(seed)

    # Separate rules by type
    # Identify propagation rules (PROPAGATE action) vs local rules
    action_col = None
    for col in df.columns:
        if col.strip().lower() in ("actiontype", "action_type", "action"):
            action_col = col
            break

    if action_col is None:
        # Fallback: random removal
        indices = list(range(n_total))
        rng.shuffle(indices)
        keep_indices = sorted(indices[:n_keep])
        return df.iloc[keep_indices].reset_index(drop=True)

    # Categorize rules
    propagate_indices = []
    local_indices = []
    for idx, row in df.iterrows():
        action = str(row[action_col]).strip().upper()
        if "PROPAGATE" in action:
            propagate_indices.append(idx)
        else:
            local_indices.append(idx)

    # Remove propagation rules first, then local if needed
    rng.shuffle(propagate_indices)
    rng.shuffle(local_indices)

    remove_indices = set()
    # Remove from propagation rules first
    for idx in propagate_indices:
        if len(remove_indices) >= n_remove:
            break
        remove_indices.add(idx)

    # If we still need to remove more, take from local rules
    for idx in local_indices:
        if len(remove_indices) >= n_remove:
            break
        remove_indices.add(idx)

    keep_mask = ~df.index.isin(remove_indices)
    result = df[keep_mask].reset_index(drop=True)

    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create partial rule databases for stress testing."
    )
    ap.add_argument(
        "--rules", type=Path,
        default=Path("outputs/Static files/rule_database.csv"),
        help="Path to full rule_database.csv",
    )
    ap.add_argument(
        "--fractions", type=float, nargs="+", default=[0.7, 0.5],
        help="Fractions of rules to keep (e.g., 0.7 0.5)",
    )
    ap.add_argument(
        "--outdir", type=Path,
        default=Path("outputs/stress_tests/partial_rules"),
        help="Output directory for partial rule CSVs",
    )
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if not args.rules.exists():
        raise SystemExit(f"[error] rules file not found: {args.rules}")

    df_full = pd.read_csv(args.rules)
    print(f"[info] full rule database: {len(df_full)} rules from {args.rules}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    for frac in args.fractions:
        pct = int(frac * 100)
        df_partial = create_partial_rules(args.rules, frac, seed=args.seed)
        out_path = args.outdir / f"rule_database_{pct}pct.csv"
        df_partial.to_csv(out_path, index=False)

        n_removed = len(df_full) - len(df_partial)
        print(
            f"[ok] {pct}% rules: {len(df_partial)} kept, "
            f"{n_removed} removed -> {out_path}"
        )

        # Show what types of rules were removed
        action_col = None
        for col in df_full.columns:
            if col.strip().lower() in ("actiontype", "action_type", "action"):
                action_col = col
                break
        if action_col:
            full_actions = df_full[action_col].value_counts().to_dict()
            partial_actions = df_partial[action_col].value_counts().to_dict()
            for act in full_actions:
                removed = full_actions.get(act, 0) - partial_actions.get(act, 0)
                if removed > 0:
                    print(f"  {act}: {removed} removed ({full_actions[act]} -> {partial_actions.get(act, 0)})")


if __name__ == "__main__":
    main()
