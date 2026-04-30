#!/usr/bin/env python3
"""Generate LaTeX tables from OTN RCA evaluation results.

Usage:
  python generate_paper_tables.py --results-dir outputs/eval/ --outdir outputs/tables/

Optional:
  --baselines rule-based:outputs/eval/rule_based zero-shot:outputs/eval/zero_shot ...
  --iid-dir outputs/eval/iid  --ood-dir outputs/eval/ood  (for IID vs OOD table)

Tables produced:
  1. main_results.tex       - Main results across all methods
  2. iid_ood_results.tex    - IID vs OOD comparison
  3. per_event_results.tex  - Per-event type breakdown
  4. ablation_results.tex   - Ablation study
  5. error_analysis.tex     - Error analysis and misclassification pairs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

EVENT_TYPES = [
    "Fiber_Aging",
    "Fiber_Crack",
    "Fiber_Cut",
    "Line_Disconnect",
    "XCON_Port_Down",
    "XCON_Buffer_Overflow",
    "XCON_Fabric_Fault",
]

CATEGORY_MAP = {
    "Fiber_Aging": "Fiber",
    "Fiber_Crack": "Fiber",
    "Fiber_Cut": "Fiber",
    "Line_Disconnect": "Line",
    "XCON_Port_Down": "XCON",
    "XCON_Buffer_Overflow": "XCON",
    "XCON_Fabric_Fault": "XCON",
}

BASELINE_ORDER = ["Rule-based", "Zero-shot", "SFT-only", "SFT+GRPO", "SFT+GRPO+Agent"]

# Map directory prefixes to canonical model names
DIR_TO_MODEL = {
    "rule": "Rule-based",
    "zeroshot": "Zero-shot",
    "sft_v6.1": "SFT-only",
    "sft": "SFT-only",
    "grpo_v6.1": "SFT+GRPO",
    "grpo": "SFT+GRPO",
    "agent": "SFT+GRPO+Agent",
}

METRIC_KEYS = [
    "event_accuracy",
    "board_accuracy",
    "category_accuracy",
    "board_f1_mean",
    "alarm_f1_mean",
    "format_score_mean",
    "e2e_score_mean",
]

METRIC_SHORT = {
    "event_accuracy": "Event Acc",
    "board_accuracy": "Board Acc",
    "category_accuracy": "Cat Acc",
    "board_f1_mean": "Board F1",
    "alarm_f1_mean": "Alarm F1",
    "format_score_mean": "Format",
    "e2e_score_mean": "E2E",
}

# Metrics displayed as percentages (multiply by 100)
PCT_METRICS = {"event_accuracy", "board_accuracy", "category_accuracy"}

# Metrics displayed as F1 scores (3 decimal places)
F1_METRICS = {"board_f1_mean", "alarm_f1_mean", "format_score_mean", "e2e_score_mean"}


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def _load_summary_csv(path: Path) -> list[dict[str, Any]]:
    """Load results_summary.csv (may have multiple rows for different models)."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_per_example_csv(path: Path) -> list[dict[str, Any]]:
    """Load results_per_example.csv."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_confusion_matrix(path: Path) -> dict[str, dict[str, int]]:
    """Load confusion_matrix.csv into nested dict: confusion[gt][pred] = count."""
    if not path.exists():
        return {}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) < 2:
        return {}
    header = rows[0][1:]  # skip "gt \\ pred"
    for row in rows[1:]:
        if not row:
            continue
        gt = row[0]
        for j, pred in enumerate(header):
            if j + 1 < len(row):
                val = row[j + 1].strip()
                confusion[gt][pred] = int(val) if val else 0
    return dict(confusion)


def _auto_discover(results_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Scan results_dir for {model}_{split}/ directories.

    Returns: (iid_dirs, ood_dirs) mapping canonical model name -> directory path
    """
    iid_dirs: dict[str, Path] = {}
    ood_dirs: dict[str, Path] = {}

    if not results_dir.is_dir():
        return iid_dirs, ood_dirs

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name.endswith("_iid"):
            model_key = name[:-4]
            model_name = DIR_TO_MODEL.get(model_key, model_key)
            iid_dirs[model_name] = d
        elif name.endswith("_ood"):
            model_key = name[:-4]
            model_name = DIR_TO_MODEL.get(model_key, model_key)
            ood_dirs[model_name] = d

    return iid_dirs, ood_dirs


def _load_one_baseline(d: Path) -> dict[str, float]:
    """Load metrics from a single results directory (uses last row)."""
    summary_path = d / "results_summary.csv"
    rows = _load_summary_csv(summary_path)
    if not rows:
        return {}
    row = rows[-1]  # last row is most recent (handles append duplicates)
    metrics: dict[str, float] = {}
    all_keys = list(METRIC_KEYS)
    for cat in ["Fiber", "XCON", "Line"]:
        all_keys.extend([
            f"event_accuracy_{cat}", f"board_accuracy_{cat}",
            f"e2e_score_{cat}", f"n_{cat}",
        ])
    for k in all_keys:
        try:
            metrics[k] = float(row.get(k, 0))
        except (ValueError, TypeError):
            metrics[k] = 0.0
    return metrics


def _load_baselines(
    results_dir: Path,
    baseline_args: list[str] | None,
) -> dict[str, dict[str, float]]:
    """Load baseline results from multiple directories or from summary CSV.

    Returns: dict mapping baseline_name -> metric_key -> float_value
    """
    baselines: dict[str, dict[str, float]] = {}

    if baseline_args:
        for spec in baseline_args:
            if ":" in spec:
                name, path_str = spec.split(":", 1)
            else:
                name = Path(spec).stem
                path_str = spec
            metrics = _load_one_baseline(Path(path_str))
            if metrics:
                baselines[name] = metrics
    else:
        # Auto-discover: use IID results for main table
        iid_dirs, _ = _auto_discover(results_dir)
        for name, d in iid_dirs.items():
            metrics = _load_one_baseline(d)
            if metrics:
                baselines[name] = metrics

    return baselines


def _load_per_example_baselines(
    results_dir: Path,
    baseline_args: list[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Load per-example results for each baseline."""
    per_example: dict[str, list[dict[str, Any]]] = {}

    if baseline_args:
        for spec in baseline_args:
            if ":" in spec:
                name, path_str = spec.split(":", 1)
            else:
                name = Path(spec).stem
                path_str = spec
            pe = _load_per_example_csv(Path(path_str) / "results_per_example.csv")
            if pe:
                per_example[name] = pe
    else:
        # Auto-discover: use IID results for per-event table
        iid_dirs, _ = _auto_discover(results_dir)
        for name, d in iid_dirs.items():
            pe = _load_per_example_csv(d / "results_per_example.csv")
            if pe:
                per_example[name] = pe

    return per_example


# ─────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────

def _fmt_metric(value: float, metric_key: str) -> str:
    """Format a metric value for LaTeX display."""
    if metric_key in PCT_METRICS:
        return f"{value * 100:.1f}"
    elif metric_key in F1_METRICS:
        return f"{value:.3f}"
    else:
        return f"{value:.2f}"


def _bold_best(values: list[str], raw_values: list[float]) -> list[str]:
    """Bold the best (highest) value in each list."""
    if not raw_values:
        return values
    best_idx = max(range(len(raw_values)), key=lambda i: raw_values[i])
    result = list(values)
    if raw_values[best_idx] > 0:
        result[best_idx] = r"\textbf{" + result[best_idx] + "}"
    return result


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _event_display(event: str) -> str:
    """Format event name for LaTeX display."""
    return _escape_latex(event)


def _write_tex(path: Path, content: str) -> None:
    """Write LaTeX content to file."""
    with path.open("w", encoding="utf-8") as f:
        f.write(content)
    print(f"[ok] wrote {path}")


# ─────────────────────────────────────────────────────────────
# Table 1: Main Results
# ─────────────────────────────────────────────────────────────

def generate_main_results(
    results_dir: Path,
    outdir: Path,
    baseline_args: list[str] | None,
) -> None:
    """Generate main results table comparing all methods."""
    baselines = _load_baselines(results_dir, baseline_args)
    if not baselines:
        print("[skip] no baseline data for main results table")
        return

    # Order baselines
    ordered = []
    for name in BASELINE_ORDER:
        if name in baselines:
            ordered.append(name)
    for name in sorted(baselines.keys()):
        if name not in ordered:
            ordered.append(name)

    # Build header
    header_cols = " & ".join(METRIC_SHORT[k] for k in METRIC_KEYS)
    n_cols = len(METRIC_KEYS) + 1  # +1 for method name

    # Build rows, tracking raw values per column for bolding
    raw_by_col: list[list[float]] = [[] for _ in METRIC_KEYS]
    formatted_rows: list[tuple[str, list[str]]] = []

    for name in ordered:
        vals = baselines[name]
        formatted = []
        for i, k in enumerate(METRIC_KEYS):
            v = vals.get(k, 0.0)
            raw_by_col[i].append(v)
            formatted.append(_fmt_metric(v, k))
        formatted_rows.append((_escape_latex(name), formatted))

    # Bold best in each column
    for col_idx in range(len(METRIC_KEYS)):
        col_raw = raw_by_col[col_idx]
        if not col_raw:
            continue
        best_idx = max(range(len(col_raw)), key=lambda i: col_raw[i])
        if col_raw[best_idx] > 0:
            formatted_rows[best_idx][1][col_idx] = (
                r"\textbf{" + formatted_rows[best_idx][1][col_idx] + "}"
            )

    # Assemble LaTeX
    rows_tex = []
    for name, vals in formatted_rows:
        row = f"    {name} & " + " & ".join(vals) + r" \\"
        rows_tex.append(row)

    tex = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{RCA Performance Across Methods}" "\n"
        r"\label{tab:main-results}" "\n"
        r"\begin{tabular}{l" + "c" * len(METRIC_KEYS) + "}\n"
        r"    \toprule" "\n"
        f"    Method & {header_cols}" r" \\" "\n"
        r"    \midrule" "\n"
        + "\n".join(rows_tex) + "\n"
        r"    \bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )

    _write_tex(outdir / "main_results.tex", tex)


# ─────────────────────────────────────────────────────────────
# Table 2: IID vs OOD Comparison
# ─────────────────────────────────────────────────────────────

def generate_iid_ood_results(
    results_dir: Path,
    outdir: Path,
    iid_dir: Path | None,
    ood_dir: Path | None,
    baseline_args: list[str] | None,
) -> None:
    """Generate IID vs OOD comparison table."""
    # Auto-discover paired IID/OOD directories
    iid_dirs, ood_dirs = _auto_discover(results_dir)

    iid_data: dict[str, dict] = {}
    ood_data: dict[str, dict] = {}

    for name, d in iid_dirs.items():
        rows = _load_summary_csv(d / "results_summary.csv")
        if rows:
            iid_data[name] = rows[-1]
    for name, d in ood_dirs.items():
        rows = _load_summary_csv(d / "results_summary.csv")
        if rows:
            ood_data[name] = rows[-1]

    if not iid_data and not ood_data:
        print("[skip] no IID/OOD data found")
        return

    # Order by BASELINE_ORDER
    all_names_set = set(list(iid_data.keys()) + list(ood_data.keys()))
    all_names = [n for n in BASELINE_ORDER if n in all_names_set]
    for n in sorted(all_names_set):
        if n not in all_names:
            all_names.append(n)

    # Selected metrics for IID/OOD table
    selected_metrics = ["event_accuracy", "board_accuracy", "board_f1_mean",
                        "alarm_f1_mean", "e2e_score_mean"]
    metric_headers = [METRIC_SHORT[m] for m in selected_metrics]

    # Build multicolumn header for IID and OOD
    n_metrics = len(selected_metrics)
    iid_header = r"\multicolumn{" + str(n_metrics) + r"}{c}{IID}"
    ood_header = r"\multicolumn{" + str(n_metrics) + r"}{c}{OOD}"
    gap_header = r"\multicolumn{1}{c}{$\Delta$}"

    sub_header = " & ".join(metric_headers)

    rows_tex = []
    for name in all_names:
        iid_row = iid_data.get(name, {})
        ood_row = ood_data.get(name, {})

        iid_vals = []
        ood_vals = []
        delta_vals = []
        for m in selected_metrics:
            iv = float(iid_row.get(m, 0))
            ov = float(ood_row.get(m, 0))
            iid_vals.append(_fmt_metric(iv, m))
            ood_vals.append(_fmt_metric(ov, m))
            # Delta = OOD - IID (negative means OOD is worse)
            if m in PCT_METRICS:
                d = (ov - iv) * 100
                delta_vals.append(f"{d:+.1f}")
            else:
                d = ov - iv
                delta_vals.append(f"{d:+.3f}")

        row = (
            f"    {_escape_latex(name)} & "
            + " & ".join(iid_vals) + " & "
            + " & ".join(ood_vals) + " & "
            + " & ".join(delta_vals) + r" \\"
        )
        rows_tex.append(row)

    col_spec = "l" + "c" * n_metrics + "c" * n_metrics + "c" * n_metrics
    tex = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{IID vs OOD Generalization Comparison}" "\n"
        r"\label{tab:iid-ood}" "\n"
        r"\resizebox{\textwidth}{!}{%" "\n"
        r"\begin{tabular}{" + col_spec + "}\n"
        r"    \toprule" "\n"
        f"    & {iid_header} & {ood_header} & "
        r"\multicolumn{" + str(n_metrics) + r"}{c}{$\Delta$ (OOD$-$IID)}" r" \\" "\n"
        r"    \cmidrule(lr){2-" + str(1 + n_metrics) + "}"
        r" \cmidrule(lr){" + str(2 + n_metrics) + "-" + str(1 + 2 * n_metrics) + "}"
        r" \cmidrule(lr){" + str(2 + 2 * n_metrics) + "-" + str(1 + 3 * n_metrics) + "}" "\n"
        f"    Method & {sub_header} & {sub_header} & {sub_header}" r" \\" "\n"
        r"    \midrule" "\n"
        + "\n".join(rows_tex) + "\n"
        r"    \bottomrule" "\n"
        r"\end{tabular}}" "\n"
        r"\end{table}" "\n"
    )

    _write_tex(outdir / "iid_ood_results.tex", tex)


# ─────────────────────────────────────────────────────────────
# Table 3: Per-Event Breakdown
# ─────────────────────────────────────────────────────────────

def generate_per_event_results(
    results_dir: Path,
    outdir: Path,
    baseline_args: list[str] | None,
) -> None:
    """Generate per-event type accuracy table."""
    pe_baselines = _load_per_example_baselines(results_dir, baseline_args)
    if not pe_baselines:
        print("[skip] no per-example data for per-event table")
        return

    # Order baselines
    ordered = []
    for name in BASELINE_ORDER:
        if name in pe_baselines:
            ordered.append(name)
    for name in sorted(pe_baselines.keys()):
        if name not in ordered:
            ordered.append(name)

    # Compute per-event accuracy for each baseline
    event_counts: dict[str, int] = Counter()
    event_acc: dict[str, dict[str, float]] = {}  # baseline -> event -> accuracy

    for name in ordered:
        examples = pe_baselines[name]
        event_acc[name] = {}
        by_event: dict[str, list[int]] = defaultdict(list)
        for ex in examples:
            gt = ex.get("gt_event", "")
            correct = int(ex.get("event_correct", 0))
            if gt in EVENT_TYPES:
                by_event[gt].append(correct)
                event_counts[gt] += 1

        for evt in EVENT_TYPES:
            if by_event[evt]:
                event_acc[name][evt] = sum(by_event[evt]) / len(by_event[evt])
            else:
                event_acc[name][evt] = 0.0

    # Normalize counts (divide by number of baselines to get true count)
    n_baselines = len(ordered)
    true_counts = {evt: event_counts.get(evt, 0) // max(n_baselines, 1)
                   for evt in EVENT_TYPES}

    # Build table
    method_headers = " & ".join(_escape_latex(n) for n in ordered)
    n_methods = len(ordered)
    col_spec = "l" + "r" + "c" * n_methods

    rows_tex = []
    for evt in EVENT_TYPES:
        count = true_counts.get(evt, 0)
        raw_vals = [event_acc.get(name, {}).get(evt, 0.0) for name in ordered]
        fmt_vals = [f"{v * 100:.1f}" for v in raw_vals]
        # Bold the best
        if raw_vals:
            best_idx = max(range(len(raw_vals)), key=lambda i: raw_vals[i])
            if raw_vals[best_idx] > 0:
                fmt_vals[best_idx] = r"\textbf{" + fmt_vals[best_idx] + "}"

        vals_str = " & ".join(fmt_vals)
        cat = CATEGORY_MAP.get(evt, "")
        row = f"    {_event_display(evt)} & {count} & {vals_str}" r" \\"
        rows_tex.append(row)

    # Add category separators
    # Fiber events: indices 0-2, Line: index 3, XCON: indices 4-6
    if len(rows_tex) >= 7:
        rows_tex.insert(3, r"    \midrule")
        rows_tex.insert(5, r"    \midrule")  # +1 because we just inserted

    tex = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{Per-Event Type Accuracy (\%)}" "\n"
        r"\label{tab:per-event}" "\n"
        r"\begin{tabular}{" + col_spec + "}\n"
        r"    \toprule" "\n"
        f"    Event Type & Count & {method_headers}" r" \\" "\n"
        r"    \midrule" "\n"
        + "\n".join(rows_tex) + "\n"
        r"    \bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )

    _write_tex(outdir / "per_event_results.tex", tex)


# ─────────────────────────────────────────────────────────────
# Table 4: Ablation Study
# ─────────────────────────────────────────────────────────────

def generate_ablation_table(
    results_dir: Path,
    outdir: Path,
    baseline_args: list[str] | None,
) -> None:
    """Generate ablation table showing incremental improvements."""
    baselines = _load_baselines(results_dir, baseline_args)
    if not baselines:
        print("[skip] no baseline data for ablation table")
        return

    # Define ablation order (cumulative additions — LLM methods only)
    ablation_order = [
        ("Base model (zero-shot)", "Zero-shot"),
        ("+SFT", "SFT-only"),
        ("+SFT +GRPO", "SFT+GRPO"),
        ("+SFT +GRPO +Agent", "SFT+GRPO+Agent"),
    ]

    # Filter to available baselines (exclude non-LLM methods like Rule-based)
    available = []
    for display_name, key in ablation_order:
        if key in baselines:
            available.append((display_name, key))

    if len(available) < 2:
        print("[skip] need at least 2 baselines for ablation table")
        return

    # Selected metrics for ablation
    abl_metrics = ["event_accuracy", "board_accuracy", "board_f1_mean",
                    "alarm_f1_mean", "e2e_score_mean"]
    metric_headers = [METRIC_SHORT[m] for m in abl_metrics]

    header = " & ".join(metric_headers)
    delta_header = " & ".join([r"$\Delta$" + m for m in metric_headers])

    rows_tex = []
    prev_vals: dict[str, float] | None = None

    for display_name, key in available:
        vals = baselines[key]
        fmt_vals = []
        delta_strs = []
        for m in abl_metrics:
            v = vals.get(m, 0.0)
            fmt_vals.append(_fmt_metric(v, m))

            if prev_vals is not None:
                pv = prev_vals.get(m, 0.0)
                if m in PCT_METRICS:
                    d = (v - pv) * 100
                    delta_strs.append(f"{d:+.1f}")
                else:
                    d = v - pv
                    delta_strs.append(f"{d:+.3f}")
            else:
                delta_strs.append("--")

        vals_str = " & ".join(fmt_vals)
        delta_str = " & ".join(delta_strs)
        row = f"    {_escape_latex(display_name)} & {vals_str} & {delta_str}" r" \\"
        rows_tex.append(row)
        prev_vals = vals

    n_metrics = len(abl_metrics)
    col_spec = "l" + "c" * n_metrics + "c" * n_metrics

    tex = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{Ablation Study: Incremental Component Contributions}" "\n"
        r"\label{tab:ablation}" "\n"
        r"\begin{tabular}{" + col_spec + "}\n"
        r"    \toprule" "\n"
        r"    & \multicolumn{" + str(n_metrics) + r"}{c}{Absolute}"
        r" & \multicolumn{" + str(n_metrics) + r"}{c}{$\Delta$ from Previous}" r" \\" "\n"
        r"    \cmidrule(lr){2-" + str(1 + n_metrics) + "}"
        r" \cmidrule(lr){" + str(2 + n_metrics) + "-" + str(1 + 2 * n_metrics) + "}" "\n"
        f"    Configuration & {header} & {header}" r" \\" "\n"
        r"    \midrule" "\n"
        + "\n".join(rows_tex) + "\n"
        r"    \bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )

    _write_tex(outdir / "ablation_results.tex", tex)


# ─────────────────────────────────────────────────────────────
# Table 5: Error Analysis
# ─────────────────────────────────────────────────────────────

def generate_error_analysis(
    results_dir: Path,
    outdir: Path,
    baseline_args: list[str] | None,
) -> None:
    """Generate error analysis table with misclassification pairs and format failures."""
    pe_baselines = _load_per_example_baselines(results_dir, baseline_args)

    # Load confusion matrix from best LLM model (SFT+GRPO IID)
    iid_dirs, _ = _auto_discover(results_dir)
    confusion_data = {}
    for model_name in ["SFT+GRPO", "Zero-shot", "SFT-only"]:
        if model_name in iid_dirs:
            confusion_data = _load_confusion_matrix(
                iid_dirs[model_name] / "confusion_matrix.csv"
            )
            if confusion_data:
                break
    if not confusion_data:
        confusion_data = _load_confusion_matrix(results_dir / "confusion_matrix.csv")

    if not pe_baselines and not confusion_data:
        print("[skip] no data for error analysis table")
        return

    # ── Part A: Top misclassification pairs from confusion matrix ──
    misclass_pairs: list[tuple[str, str, int]] = []
    if confusion_data:
        for gt, preds in confusion_data.items():
            for pred, count in preds.items():
                if gt != pred and count > 0 and gt in EVENT_TYPES:
                    misclass_pairs.append((gt, pred, count))
    misclass_pairs.sort(key=lambda x: x[2], reverse=True)

    # ── Part B: Per-method error breakdown ──
    ordered = []
    for name in BASELINE_ORDER:
        if name in pe_baselines:
            ordered.append(name)
    for name in sorted(pe_baselines.keys()):
        if name not in ordered:
            ordered.append(name)

    method_errors: dict[str, dict[str, int]] = {}
    for name in ordered:
        examples = pe_baselines[name]
        errors = {
            "event_errors": 0,
            "board_errors": 0,
            "format_failures": 0,
            "total": len(examples),
        }
        for ex in examples:
            if int(ex.get("event_correct", 0)) == 0:
                errors["event_errors"] += 1
            if int(ex.get("board_correct", 0)) == 0:
                errors["board_errors"] += 1
            if float(ex.get("format_score", 0)) < 0.25:
                errors["format_failures"] += 1
        method_errors[name] = errors

    # ── Build Part A table: Top Misclassification Pairs ──
    part_a_rows = []
    for gt, pred, count in misclass_pairs[:10]:
        row = (
            f"    {_event_display(gt)} & {_event_display(pred)} & {count}" r" \\"
        )
        part_a_rows.append(row)

    # ── Build Part B table: Per-Method Error Counts ──
    part_b_rows = []
    for name in ordered:
        err = method_errors.get(name, {})
        total = err.get("total", 0)
        evt_err = err.get("event_errors", 0)
        brd_err = err.get("board_errors", 0)
        fmt_fail = err.get("format_failures", 0)
        evt_pct = f"{evt_err / total * 100:.1f}" if total > 0 else "0.0"
        brd_pct = f"{brd_err / total * 100:.1f}" if total > 0 else "0.0"
        fmt_pct = f"{fmt_fail / total * 100:.1f}" if total > 0 else "0.0"
        row = (
            f"    {_escape_latex(name)} & {total} & "
            f"{evt_err} ({evt_pct}\\%) & "
            f"{brd_err} ({brd_pct}\\%) & "
            f"{fmt_fail} ({fmt_pct}\\%)" r" \\"
        )
        part_b_rows.append(row)

    # Combine into one table with subtables
    tex = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{Error Analysis}" "\n"
        r"\label{tab:error-analysis}" "\n"
        "\n"
    )

    # Part A
    if part_a_rows:
        tex += (
            r"\subtable[Top Misclassification Pairs]{" "\n"
            r"\label{tab:misclass}" "\n"
            r"\begin{tabular}{llr}" "\n"
            r"    \toprule" "\n"
            r"    Ground Truth & Predicted & Count \\" "\n"
            r"    \midrule" "\n"
            + "\n".join(part_a_rows) + "\n"
            r"    \bottomrule" "\n"
            r"\end{tabular}}" "\n"
        )

    # Part B
    if part_b_rows:
        if part_a_rows:
            tex += r"\hfill" "\n"
        tex += (
            r"\subtable[Per-Method Error Breakdown]{" "\n"
            r"\label{tab:error-breakdown}" "\n"
            r"\begin{tabular}{lrccc}" "\n"
            r"    \toprule" "\n"
            r"    Method & $N$ & Event Errors & Board Errors & Format Failures \\" "\n"
            r"    \midrule" "\n"
            + "\n".join(part_b_rows) + "\n"
            r"    \bottomrule" "\n"
            r"\end{tabular}}" "\n"
        )

    tex += r"\end{table}" "\n"

    # Fall back to simpler structure if subtable is not available
    # Also produce a standalone version using minipage
    tex_minipage = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{Error Analysis}" "\n"
        r"\label{tab:error-analysis}" "\n"
    )

    if part_a_rows:
        tex_minipage += (
            "\n"
            r"\begin{minipage}[t]{0.45\textwidth}" "\n"
            r"\centering" "\n"
            r"\textbf{(a) Top Misclassification Pairs}" "\n"
            r"\vspace{0.5em}" "\n"
            "\n"
            r"\begin{tabular}{llr}" "\n"
            r"    \toprule" "\n"
            r"    Ground Truth & Predicted & Count \\" "\n"
            r"    \midrule" "\n"
            + "\n".join(part_a_rows) + "\n"
            r"    \bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{minipage}" "\n"
        )

    if part_b_rows:
        tex_minipage += (
            r"\hfill" "\n"
            r"\begin{minipage}[t]{0.52\textwidth}" "\n"
            r"\centering" "\n"
            r"\textbf{(b) Per-Method Error Breakdown}" "\n"
            r"\vspace{0.5em}" "\n"
            "\n"
            r"\begin{tabular}{lrccc}" "\n"
            r"    \toprule" "\n"
            r"    Method & $N$ & Event Err & Board Err & Fmt Fail \\" "\n"
            r"    \midrule" "\n"
            + "\n".join(part_b_rows) + "\n"
            r"    \bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{minipage}" "\n"
        )

    tex_minipage += r"\end{table}" "\n"

    # Write the minipage version (more portable)
    _write_tex(outdir / "error_analysis.tex", tex_minipage)


# ─────────────────────────────────────────────────────────────
# Table 6: Stress Test Comparison
# ─────────────────────────────────────────────────────────────

STRESS_SCENARIOS = {
    "S1": "Multi-failure (2 concurrent)",
    "S2": "Multi-failure (3 concurrent)",
    "S3": "High-noise (2x baseline)",
    "S4": "Cascading failure",
}


def generate_stress_table(
    results_dir: Path,
    outdir: Path,
) -> None:
    """Generate stress test comparison table across methods and scenarios S1-S4."""
    # Look for stress test results in results_dir/stress_*/
    # Each subdir should contain method subdirs with results_summary.csv
    stress_data: dict[str, dict[str, dict[str, float]]] = {}  # scenario -> method -> metrics

    for scenario_key, scenario_desc in STRESS_SCENARIOS.items():
        scenario_dir = results_dir / f"stress_{scenario_key.lower()}"
        if not scenario_dir.is_dir():
            continue

        stress_data[scenario_key] = {}
        for method_dir in sorted(scenario_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method_key = method_dir.name
            model_name = DIR_TO_MODEL.get(method_key, method_key)
            metrics = _load_one_baseline(method_dir)
            if metrics:
                stress_data[scenario_key][model_name] = metrics

    if not stress_data:
        print("[skip] no stress test results found")
        return

    # Collect all methods across scenarios
    all_methods_set: set[str] = set()
    for scenario_methods in stress_data.values():
        all_methods_set.update(scenario_methods.keys())

    ordered_methods = [m for m in BASELINE_ORDER if m in all_methods_set]
    for m in sorted(all_methods_set):
        if m not in ordered_methods:
            ordered_methods.append(m)

    # Build table: rows = scenarios, columns = methods, cells = event_accuracy
    method_headers = " & ".join(_escape_latex(m) for m in ordered_methods)

    rows_tex = []
    for scenario_key in ["S1", "S2", "S3", "S4"]:
        if scenario_key not in stress_data:
            continue
        desc = STRESS_SCENARIOS[scenario_key]
        scenario_methods = stress_data[scenario_key]

        raw_vals = []
        fmt_vals = []
        for m in ordered_methods:
            v = scenario_methods.get(m, {}).get("event_accuracy", 0.0)
            raw_vals.append(v)
            fmt_vals.append(f"{v * 100:.1f}")

        # Bold the best
        if raw_vals:
            best_idx = max(range(len(raw_vals)), key=lambda i: raw_vals[i])
            if raw_vals[best_idx] > 0:
                fmt_vals[best_idx] = r"\textbf{" + fmt_vals[best_idx] + "}"

        vals_str = " & ".join(fmt_vals)
        row = f"    {scenario_key}: {_escape_latex(desc)} & {vals_str}" r" \\"
        rows_tex.append(row)

    col_spec = "l" + "c" * len(ordered_methods)
    tex = (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{Stress Test Event Accuracy (\%) Across Methods}" "\n"
        r"\label{tab:stress}" "\n"
        r"\begin{tabular}{" + col_spec + "}\n"
        r"    \toprule" "\n"
        f"    Scenario & {method_headers}" r" \\" "\n"
        r"    \midrule" "\n"
        + "\n".join(rows_tex) + "\n"
        r"    \bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )

    _write_tex(outdir / "stress_results.tex", tex)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate LaTeX tables from OTN RCA evaluation results."
    )
    ap.add_argument(
        "--results-dir", type=Path, default=Path("outputs/eval"),
        help="Directory containing evaluation output files",
    )
    ap.add_argument(
        "--outdir", type=Path, default=Path("outputs/tables"),
        help="Output directory for generated .tex files",
    )
    ap.add_argument(
        "--baselines", nargs="*", default=None,
        help=(
            "Baseline specs as name:path pairs, e.g. "
            "'Rule-based:outputs/eval/rule_based' 'SFT-only:outputs/eval/sft'"
        ),
    )
    ap.add_argument(
        "--iid-dir", type=Path, default=None,
        help="Directory with IID evaluation results (contains results_summary.csv)",
    )
    ap.add_argument(
        "--ood-dir", type=Path, default=None,
        help="Directory with OOD evaluation results (contains results_summary.csv)",
    )
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"[info] results dir: {args.results_dir}")
    print(f"[info] output dir:  {args.outdir}")

    # Show discovered directories
    iid_dirs, ood_dirs = _auto_discover(args.results_dir)
    print(f"[info] IID baselines: {list(iid_dirs.keys())}")
    print(f"[info] OOD baselines: {list(ood_dirs.keys())}")

    # Table 1: Main Results (uses IID split)
    generate_main_results(args.results_dir, args.outdir, args.baselines)

    # Table 2: IID vs OOD
    generate_iid_ood_results(
        args.results_dir, args.outdir, args.iid_dir, args.ood_dir, args.baselines
    )

    # Table 3: Per-Event Breakdown (uses IID split)
    generate_per_event_results(args.results_dir, args.outdir, args.baselines)

    # Table 4: Ablation (uses IID split)
    generate_ablation_table(args.results_dir, args.outdir, args.baselines)

    # Table 5: Error Analysis (uses IID split)
    generate_error_analysis(args.results_dir, args.outdir, args.baselines)

    # Table 6: Stress Test Comparison
    generate_stress_table(args.results_dir, args.outdir)

    print("[done] table generation complete")


if __name__ == "__main__":
    main()
