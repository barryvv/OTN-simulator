#!/usr/bin/env python3
"""Convert SFC-BFL dataset → JSONL prompts compatible with batch_inference.py + evaluate.py.

Per OW:
1. Load metrics from parquet (filter by ow_id)
2. Compute pattern features on peak board
3. Build topology/lightpath summary from boards.csv + lightpaths.json
4. Construct prompt using SYSTEM_PROMPT + METRIC_GUIDE
5. Attach ground truth from ow_labels.jsonl + ow_alarms.jsonl

Usage:
  python -m sfc_bfl.convert_to_prompts \
    --dataset-dir outputs/sfc_bfl/dataset_a_pilot \
    --rules outputs/Static\ files/rule_database.csv \
    --out outputs/sfc_bfl_prompts/dataset_a_pilot.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Add project root
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from generate_sft_data import (
    SYSTEM_PROMPT,
    METRIC_GUIDE,
    _compute_pattern_features,
    _read_csv,
)
from generate_grpo_data import ALARM_FLOW_SCHEMA, _rules_text_compact


# ---------------------------------------------------------------------------
# SFC-BFL Dataset Loader
# ---------------------------------------------------------------------------

class SFCBFLDataset:
    """Loads an SFC-BFL dataset directory, provides per-OW access."""

    def __init__(self, dataset_dir: str | Path):
        self.root = Path(dataset_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset dir not found: {self.root}")

        # Load static topology
        self.boards_df = pd.read_csv(self.root / "boards.csv")
        self.edges_df = pd.read_csv(self.root / "edges.csv")
        with open(self.root / "lightpaths.json") as f:
            self.lightpaths = json.load(f)
        with open(self.root / "services.json") as f:
            self.services = json.load(f)

        # Load labels index
        self._labels: dict[int, dict] = {}
        with open(self.root / "ow_labels.jsonl") as f:
            for line in f:
                rec = json.loads(line.strip())
                self._labels[rec["ow_id"]] = rec

        # Load alarms index
        self._alarms: dict[int, list[dict]] = {}
        with open(self.root / "ow_alarms.jsonl") as f:
            for line in f:
                rec = json.loads(line.strip())
                self._alarms[rec["ow_id"]] = rec.get("alarms", [])

        # Metrics: load parquet lazily
        metrics_path = self.root / "ow_metrics.parquet"
        metrics_dir = self.root / "ow_metrics"
        if metrics_path.exists():
            self._metrics_path = str(metrics_path)
        elif metrics_dir.exists():
            self._metrics_path = str(metrics_dir)
        else:
            self._metrics_path = None

        self._metrics_df: pd.DataFrame | None = None

    @property
    def ow_ids(self) -> list[int]:
        return sorted(self._labels.keys())

    @property
    def num_ows(self) -> int:
        return len(self._labels)

    def _ensure_metrics_loaded(self) -> None:
        if self._metrics_df is None and self._metrics_path:
            self._metrics_df = pd.read_parquet(self._metrics_path)

    def get_ow_metrics(self, ow_id: int) -> pd.DataFrame:
        """Get metrics DataFrame for a single OW."""
        self._ensure_metrics_loaded()
        if self._metrics_df is None:
            return pd.DataFrame()
        return self._metrics_df[self._metrics_df["ow_id"] == ow_id]

    def get_ow_label(self, ow_id: int) -> dict:
        return self._labels.get(ow_id, {})

    def get_ow_alarms(self, ow_id: int) -> list[dict]:
        return self._alarms.get(ow_id, [])


# ---------------------------------------------------------------------------
# Parquet-adapted helper functions
# ---------------------------------------------------------------------------

def _timeseries_summary_from_parquet(
    ow_df: pd.DataFrame,
    max_boards_shown: int = 9,
) -> str:
    """Build timeseries summary from parquet OW metrics DataFrame.

    Replaces _timeseries_summary() which reads CSV files from disk.
    """
    if ow_df.empty or "BBE" not in ow_df.columns:
        return "timeseries: no data"

    # Per-board max BBE
    board_max_bbe: dict[str, float] = {}
    for board_id, bdf in ow_df.groupby("board_id"):
        mx = pd.to_numeric(bdf["BBE"], errors="coerce").max()
        board_max_bbe[str(board_id)] = float(mx)

    # Sort by max BBE descending
    sorted_boards = sorted(board_max_bbe.items(), key=lambda x: x[1], reverse=True)

    board_rows = []
    for bname, mx in sorted_boards[:max_boards_shown]:
        board_rows.append(f"  {bname}: bbe_max={mx:.1f}")
    result = "timeseries per-board bbe_max:\n" + "\n".join(board_rows)

    # Compute pattern features from peak board
    peak_board = ""
    peak_node = ""
    peak_features = {}
    if sorted_boards:
        peak_board = sorted_boards[0][0]
        if "-" in peak_board:
            peak_node = peak_board.split("-")[0]

        peak_df = ow_df[ow_df["board_id"].astype(str).str.strip() == peak_board]
        if not peak_df.empty:
            peak_features = _compute_pattern_features(peak_df)

    if peak_features:
        pb_str = f" peak_board={peak_board}" if peak_board else ""
        pn_str = f" peak_node={peak_node}" if peak_node else ""
        pk_str = f" pattern_kind={peak_features['pattern_kind']}" if "pattern_kind" in peak_features else ""
        result += (
            f"\npattern_features: burst_regions={peak_features.get('burst_regions', 0)}"
            f" longest_burst_frac={peak_features.get('longest_burst_frac', 0.0):.3f}"
            f" bbe_max={peak_features.get('bbe_max', 0.0):.1f}"
            f" es_frac={peak_features.get('es_frac', 0.0):.4f}"
            f" bber_bbe_ratio={peak_features.get('bber_bbe_ratio', 0.0):.2f}"
            f"{pk_str}{pb_str}{pn_str}"
        )

    return result


def _lightpaths_summary_from_json(
    lightpaths: list[dict],
    max_lines: int = 10,
) -> str:
    """Build lightpath summary from lightpaths.json.

    Replaces _lightpaths_summary() which reads lightpaths.txt.
    For large topologies (>50 lightpaths), shows condensed summary.
    """
    if not lightpaths:
        return "topology/lightpaths: empty"

    n = len(lightpaths)
    show = min(max_lines, n)
    lines = []
    for lp in lightpaths[:show]:
        bp = lp.get("board_path", [])
        np_ = lp.get("node_path", [])
        lines.append(f"  LP{lp.get('lightpath_id', '?')}: {' -> '.join(np_)} ({len(bp)} boards)")

    result = f"topology/lightpaths: {n} lightpaths\n" + "\n".join(lines)
    if n > show:
        result += f"\n  ... ({n - show} more lightpaths)"
    return result


def _board_catalog_from_csv(
    boards_df: pd.DataFrame,
    max_boards: int = 500,
) -> str:
    """Build board catalog from boards.csv DataFrame.

    Replaces _board_catalog() which parses lightpath files.
    For large topologies (>max_boards), shows only node-level summary.
    """
    if boards_df.empty:
        return ""

    total_boards = len(boards_df)

    # Group by node_id
    node_boards: dict[str, list[str]] = defaultdict(list)
    for _, row in boards_df.iterrows():
        node_boards[str(row["node_id"])].append(str(row["board_id"]))

    if not node_boards:
        return ""

    if total_boards > max_boards:
        # For very large topologies, show summary only
        lines = [f"board_catalog: {total_boards} boards across {len(node_boards)} nodes"]
        for node in sorted(node_boards):
            board_list = sorted(node_boards[node])
            lines.append(f"  {node}: {len(board_list)} boards")
        return "\n".join(lines)

    lines = ["board_catalog:"]
    for node in sorted(node_boards):
        board_list = sorted(node_boards[node])
        lines.append(f"  {node}: {', '.join(board_list)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt_for_ow(
    dataset: SFCBFLDataset,
    ow_id: int,
    rules_path: Path,
) -> dict | None:
    """Build a single JSONL record for one OW.

    Output schema is compatible with batch_inference.py and evaluate.py.
    """
    label = dataset.get_ow_label(ow_id)
    if not label:
        return None

    ow_df = dataset.get_ow_metrics(ow_id)
    alarms = dataset.get_ow_alarms(ow_id)

    # Build prompt components
    ts_summary = _timeseries_summary_from_parquet(ow_df)
    topo_summary = _lightpaths_summary_from_json(dataset.lightpaths)
    catalog = _board_catalog_from_csv(dataset.boards_df)
    rules_full = _rules_text_compact(rules_path)

    catalog_section = f"\n{catalog}\n" if catalog else ""

    user_msg = (
        f"{topo_summary}\n"
        f"{catalog_section}\n"
        f"{ts_summary}\n\n"
        f"{METRIC_GUIDE}\n\n"
        f"Rule database (CSV):\n{rules_full}\n\n"
        "Task:\n"
        "1) Infer the likely root cause (board, event, time, severity) using feature-based reasoning:\n"
        "   a) Identify failure category from peak_board type (OA/OD/OM/FIU→Fiber, XCON→XCON, Line→Line)\n"
        "   b) Use pattern_kind + burst_regions to narrow the specific event\n"
        "   c) Confirm with bber_bbe_ratio and es_frac\n"
        f"2) Predict alarm_flow rows in this exact CSV format:\n{ALARM_FLOW_SCHEMA}\n"
        "Use alarm names from the rule database OutputAlarm column.\n"
        "Use board names from topology lightpaths. Walk each lightpath and apply matching rules at each hop.\n"
        "3) Provide RCA reasoning based on rules + topology + metrics\n"
        "4) Suggest 3 concrete fixes\n"
    )

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # Ground truth from alarms
    gt_boards = sorted(set(a.get("board", "") for a in alarms if a.get("board")))
    gt_alarms: dict[str, int] = {}
    gt_severity: dict[str, int] = {}
    for a in alarms:
        alarm_name = a.get("alarm", "")
        if alarm_name:
            gt_alarms[alarm_name] = gt_alarms.get(alarm_name, 0) + 1
        sev = a.get("severity", "")
        if sev and sev != "nan":
            gt_severity[sev] = gt_severity.get(sev, 0) + 1

    result = {
        "prompt": prompt,
        "ground_truth_root_board": label.get("root_board", ""),
        "ground_truth_root_event": label.get("root_event", ""),
        "ground_truth_boards": json.dumps(gt_boards),
        "ground_truth_alarms": json.dumps(gt_alarms),
        "ground_truth_row_count": min(len(alarms), 20),
        "ground_truth_severity_dist": json.dumps(gt_severity),
        # SFC-BFL metadata
        "dataset": str(dataset.root.name),
        "ow_id": ow_id,
    }

    # Add dimension-specific metadata if present in label
    for key in ("region_size_k", "noise_ratio", "coverage", "board_count",
                "n_domains", "failed_boards"):
        if key in label:
            result[f"dimension_{key}"] = label[key] if not isinstance(label[key], list) else json.dumps(label[key])

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert SFC-BFL dataset to JSONL prompts for inference/evaluation."
    )
    ap.add_argument("--dataset-dir", type=Path, required=True,
                    help="Path to SFC-BFL dataset directory (e.g. outputs/sfc_bfl/dataset_a_pilot)")
    ap.add_argument("--rules", type=Path,
                    default=Path("outputs/Static files/rule_database.csv"),
                    help="Path to rule_database.csv")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL path")
    ap.add_argument("--ow-ids", type=str, default=None,
                    help="OW ID range to convert (e.g. '0-99' or '0,5,10'). Default: all.")
    ap.add_argument("--max-ows", type=int, default=None,
                    help="Max number of OWs to convert")
    args = ap.parse_args()

    # Handle dataset dirs with subdirectories (Dataset D/E)
    dataset_dir = args.dataset_dir
    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()
               and (d / "ow_labels.jsonl").exists()] if dataset_dir.exists() else []

    if (dataset_dir / "ow_labels.jsonl").exists():
        # Single dataset
        dirs_to_process = [(dataset_dir, None)]
    elif subdirs:
        # Dataset with subdirectories (D: size_*, E: domains_*)
        dirs_to_process = [(d, d.name) for d in sorted(subdirs)]
    else:
        raise FileNotFoundError(
            f"No ow_labels.jsonl found in {dataset_dir} or its subdirectories"
        )

    # Parse OW IDs
    ow_id_set = None
    if args.ow_ids:
        ow_id_set = set()
        for part in args.ow_ids.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                ow_id_set.update(range(int(lo), int(hi) + 1))
            else:
                ow_id_set.add(int(part))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for ddir, subdir_name in dirs_to_process:
            print(f"[info] loading dataset from {ddir}")
            ds = SFCBFLDataset(ddir)
            print(f"[info] {ds.num_ows} OWs, {len(ds.boards_df)} boards, "
                  f"{len(ds.lightpaths)} lightpaths")

            for ow_id in ds.ow_ids:
                if ow_id_set is not None and ow_id not in ow_id_set:
                    continue
                if args.max_ows is not None and total_written >= args.max_ows:
                    break

                rec = build_prompt_for_ow(ds, ow_id, args.rules)
                if rec is None:
                    continue

                # Add subdirectory tag for D/E datasets
                if subdir_name:
                    rec["dataset_subdir"] = subdir_name

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_written += 1

            if args.max_ows is not None and total_written >= args.max_ows:
                break

    print(f"[ok] wrote {total_written} prompts -> {args.out}")


if __name__ == "__main__":
    main()
