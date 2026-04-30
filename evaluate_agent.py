#!/usr/bin/env python3
"""Comparative evaluation: single-shot LLM vs agentic RCA.

Orchestrates evaluation across multiple baselines (rule-based, zero-shot, SFT,
SFT+GRPO, Agent) using metrics from evaluate.py, and produces a comparative
analysis highlighting where the agentic approach adds value.

Usage:
  python evaluate_agent.py \
    --test-file outputs/grpo_data/test.jsonl \
    --results-dir outputs/eval/ \
    --rule-outputs outputs/eval/baseline_rule_outputs.jsonl \
    --sft-outputs outputs/eval/sft_outputs.jsonl \
    --grpo-outputs outputs/eval/grpo_outputs.jsonl \
    --agent-outputs outputs/eval/agent_outputs.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

# Import evaluation metrics from evaluate.py
from evaluate import (
    extract_predicted_event,
    extract_predicted_board,
    compute_board_f1,
    compute_alarm_f1,
    compute_format_score,
    compute_severity_f1,
    evaluate as run_evaluate,
    EVENT_TYPES,
    CATEGORY_MAP,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_DISPLAY_NAMES = {
    "rule": "Rule-Based",
    "zeroshot": "Zero-shot",
    "sft": "SFT",
    "grpo": "SFT+GRPO",
    "agent": "Agent (ReAct)",
}

# Column ordering for summary tables
SUMMARY_METRICS = [
    "n_examples",
    "event_accuracy",
    "board_accuracy",
    "category_accuracy",
    "board_f1_mean",
    "alarm_f1_mean",
    "severity_f1_mean",
    "format_score_mean",
    "e2e_score_mean",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_outputs(path: Path) -> list[dict]:
    """Load a JSONL outputs file and return a list of dicts.

    Each line must be a JSON object with at least ``model_output`` and
    ground truth fields (``ground_truth_root_event``, etc.).  Lines that
    fail to parse are skipped with a warning.
    """
    if not path.exists():
        print(f"[warn] outputs file not found: {path}")
        return []

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] {path}:{line_num}: JSON parse error: {exc}")
    return records


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------

def evaluate_outputs(outputs: list[dict], method_name: str = "") -> pd.DataFrame:
    """Run evaluation metrics on a list of output records.

    Reuses the ``evaluate()`` function from evaluate.py, then converts the
    per-example results to a DataFrame for easier analysis.  The method
    name is attached as a column so that results from multiple methods can
    be concatenated.

    Returns a DataFrame with one row per example and columns for every
    metric produced by evaluate.py plus ``method``.
    """
    if not outputs:
        return pd.DataFrame()

    per_example, summary, confusion = run_evaluate(outputs)

    df = pd.DataFrame(per_example)
    df["method"] = method_name
    df["_summary"] = [json.dumps(summary)] * len(df)
    return df


# ---------------------------------------------------------------------------
# Comparative analysis
# ---------------------------------------------------------------------------

def compare_methods(
    results_dict: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare evaluation results across methods.

    Parameters
    ----------
    results_dict : dict[str, pd.DataFrame]
        Mapping from method name to per-example DataFrame (output of
        ``evaluate_outputs``).

    Returns
    -------
    summary_df : pd.DataFrame
        One row per method with aggregate metrics.
    difficulty_df : pd.DataFrame
        One row per example with a difficulty score (how many methods got
        the event correct).
    agent_advantage_df : pd.DataFrame
        Examples where the agent predicted the event correctly but ALL
        other methods failed.  Empty if ``"agent"`` is not in results_dict.
    """
    if not results_dict:
        empty = pd.DataFrame()
        return empty, empty, empty

    # ---- 1. Summary table ----
    summary_rows: list[dict[str, Any]] = []
    for method_name, df in results_dict.items():
        if df.empty:
            continue
        # The summary JSON was stashed in the _summary column
        summary = json.loads(df["_summary"].iloc[0])
        row: dict[str, Any] = {"method": method_name}
        row["display_name"] = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        for metric in SUMMARY_METRICS:
            row[metric] = summary.get(metric, 0.0)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # ---- 2. Per-example difficulty ----
    #   Merge all methods on sample_index, count how many got event correct.
    all_dfs: list[pd.DataFrame] = []
    for method_name, df in results_dict.items():
        if df.empty:
            continue
        subset = df[["sample_index", "gt_event", "event_correct", "method"]].copy()
        all_dfs.append(subset)

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        # Pivot: rows=sample_index, columns=method, values=event_correct
        pivot = merged.pivot_table(
            index="sample_index",
            columns="method",
            values="event_correct",
            aggfunc="first",
        ).fillna(0)
        pivot["num_methods_correct"] = pivot.sum(axis=1).astype(int)
        pivot["total_methods"] = len(results_dict)

        # Classify difficulty
        def _difficulty_label(row: pd.Series) -> str:
            n_correct = int(row["num_methods_correct"])
            n_total = int(row["total_methods"])
            if n_correct == n_total:
                return "easy"
            if n_correct == 0:
                return "impossible"
            if n_correct == 1:
                return "hard"
            return "medium"

        pivot["difficulty"] = pivot.apply(_difficulty_label, axis=1)

        # Attach ground truth event for context
        gt_map = {}
        for df in results_dict.values():
            if df.empty:
                continue
            for _, row in df[["sample_index", "gt_event"]].iterrows():
                gt_map[row["sample_index"]] = row["gt_event"]
        pivot["gt_event"] = pivot.index.map(gt_map)

        difficulty_df = pivot.reset_index()
    else:
        difficulty_df = pd.DataFrame()

    # ---- 3. Agent advantage ----
    #   Examples where agent got it right but every other method got it wrong.
    agent_advantage_df = pd.DataFrame()
    if "agent" in results_dict and not difficulty_df.empty:
        agent_col = "agent"
        other_cols = [
            m for m in results_dict if m != "agent" and m in difficulty_df.columns
        ]
        if other_cols:
            mask_agent_correct = difficulty_df.get(agent_col, pd.Series(dtype=float)) == 1
            mask_others_wrong = difficulty_df[other_cols].sum(axis=1) == 0
            agent_advantage_df = difficulty_df[mask_agent_correct & mask_others_wrong].copy()

            # Also find cases where agent failed but others succeeded
            mask_agent_wrong = difficulty_df.get(agent_col, pd.Series(dtype=float)) == 0
            mask_others_correct = difficulty_df[other_cols].sum(axis=1) > 0
            agent_regress = difficulty_df[mask_agent_wrong & mask_others_correct].copy()
            if not agent_regress.empty:
                agent_advantage_df["_is_advantage"] = True
                agent_regress["_is_advantage"] = False
                agent_advantage_df = pd.concat(
                    [agent_advantage_df, agent_regress], ignore_index=True
                )
        elif not results_dict.get("agent", pd.DataFrame()).empty:
            # Agent is the only method -- every correct prediction is an "advantage"
            agent_df = results_dict["agent"]
            agent_advantage_df = agent_df[agent_df["event_correct"] == 1][
                ["sample_index", "gt_event", "event_correct"]
            ].copy()
            agent_advantage_df["_is_advantage"] = True

    return summary_df, difficulty_df, agent_advantage_df


# ---------------------------------------------------------------------------
# Tool usage analysis (agent-specific)
# ---------------------------------------------------------------------------

# Regex to extract ACTION: lines from model_output or raw agent traces
_ACTION_LINE_RE = re.compile(r"ACTION:\s*(\w+)\(", re.MULTILINE)


def analyze_tool_usage(agent_outputs: list[dict]) -> pd.DataFrame:
    """Analyze tool usage patterns from agent outputs.

    Reads ``agent_steps``, ``agent_tool_calls``, and ``model_output``
    fields to compute per-example and aggregate tool usage statistics.

    Returns a DataFrame with columns:
      sample_index, gt_event, event_correct, agent_steps, agent_tool_calls,
      tools_used (list), tool_sequence (str)
    """
    rows: list[dict[str, Any]] = []

    for record in agent_outputs:
        sample_idx = record.get("sample_index", -1)
        gt_event = record.get("ground_truth_root_event", "")
        model_output = record.get("model_output", "")

        # Determine correctness
        pred_event = record.get("predicted_event", "")
        if not pred_event:
            pred_event = extract_predicted_event(model_output) or ""
        event_correct = 1 if pred_event == gt_event else 0

        # Get step/tool counts -- prefer explicit fields from rca_agent.py
        agent_steps = record.get("agent_steps", 0)
        agent_tool_calls = record.get("agent_tool_calls", 0)

        # Infer tool calls from ACTION: lines if counts are missing
        action_matches = _ACTION_LINE_RE.findall(model_output)
        if not agent_tool_calls and action_matches:
            agent_tool_calls = len(action_matches)
        if not agent_steps and action_matches:
            agent_steps = len(action_matches) + 1  # actions + final answer step

        # Build tool list/sequence
        tools_used = action_matches if action_matches else []

        rows.append({
            "sample_index": sample_idx,
            "gt_event": gt_event,
            "event_correct": event_correct,
            "agent_steps": agent_steps,
            "agent_tool_calls": agent_tool_calls,
            "agent_elapsed_sec": record.get("agent_elapsed_sec", 0.0),
            "agent_total_tokens": record.get("agent_total_tokens", 0),
            "tools_used": tools_used,
            "tool_sequence": " -> ".join(tools_used) if tools_used else "(none)",
        })

    return pd.DataFrame(rows)


def _summarize_tool_usage(tool_df: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate tool usage statistics from the tool analysis DataFrame.

    Returns a dict with keys suitable for printing and CSV export.
    """
    if tool_df.empty:
        return {}

    n = len(tool_df)
    n_correct = int(tool_df["event_correct"].sum())

    # Overall stats
    stats: dict[str, Any] = {
        "n_examples": n,
        "event_accuracy": round(n_correct / n, 4) if n else 0.0,
        "avg_steps": round(tool_df["agent_steps"].mean(), 2),
        "avg_tool_calls": round(tool_df["agent_tool_calls"].mean(), 2),
        "avg_elapsed_sec": round(tool_df["agent_elapsed_sec"].mean(), 1),
        "avg_tokens": round(tool_df["agent_total_tokens"].mean(), 0),
    }

    # Steps/tools for correct vs incorrect predictions
    correct_mask = tool_df["event_correct"] == 1
    incorrect_mask = tool_df["event_correct"] == 0

    if correct_mask.any():
        stats["avg_steps_correct"] = round(
            tool_df.loc[correct_mask, "agent_steps"].mean(), 2
        )
        stats["avg_tool_calls_correct"] = round(
            tool_df.loc[correct_mask, "agent_tool_calls"].mean(), 2
        )
    else:
        stats["avg_steps_correct"] = 0.0
        stats["avg_tool_calls_correct"] = 0.0

    if incorrect_mask.any():
        stats["avg_steps_incorrect"] = round(
            tool_df.loc[incorrect_mask, "agent_steps"].mean(), 2
        )
        stats["avg_tool_calls_incorrect"] = round(
            tool_df.loc[incorrect_mask, "agent_tool_calls"].mean(), 2
        )
    else:
        stats["avg_steps_incorrect"] = 0.0
        stats["avg_tool_calls_incorrect"] = 0.0

    # Tool frequency across all examples
    all_tools: list[str] = []
    for tools in tool_df["tools_used"]:
        if isinstance(tools, list):
            all_tools.extend(tools)
    tool_counts = Counter(all_tools)
    stats["tool_frequency"] = dict(tool_counts.most_common())

    # Most common tool sequences
    seq_counts = Counter(tool_df["tool_sequence"].tolist())
    stats["top_sequences"] = dict(seq_counts.most_common(10))

    # Correlation: more tools -> higher accuracy?
    if n > 1:
        bins = [(0, 2), (2, 4), (4, 6), (6, float("inf"))]
        bin_stats = {}
        for lo, hi in bins:
            label = f"{lo}-{int(hi) if hi != float('inf') else '+'}"
            mask = (tool_df["agent_tool_calls"] >= lo) & (
                tool_df["agent_tool_calls"] < hi
            )
            subset = tool_df[mask]
            if len(subset) > 0:
                bin_stats[label] = {
                    "count": len(subset),
                    "accuracy": round(subset["event_correct"].mean(), 4),
                }
        stats["tool_count_vs_accuracy"] = bin_stats

    return stats


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_comparison_summary(
    summary_df: pd.DataFrame, results_dir: Path
) -> None:
    """Write the side-by-side comparison summary CSV."""
    path = results_dir / "comparison_summary.csv"
    if summary_df.empty:
        print("[skip] no summary data to write")
        return
    summary_df.to_csv(path, index=False)
    print(f"[ok] wrote comparison summary -> {path}")


def _write_agent_advantage(
    agent_advantage_df: pd.DataFrame, results_dir: Path
) -> None:
    """Write the agent advantage CSV."""
    path = results_dir / "agent_advantage.csv"
    if agent_advantage_df.empty:
        print("[skip] no agent advantage data to write")
        return
    agent_advantage_df.to_csv(path, index=False)
    print(f"[ok] wrote agent advantage -> {path}")


def _write_tool_usage_stats(
    tool_stats: dict[str, Any], tool_df: pd.DataFrame, results_dir: Path
) -> None:
    """Write tool usage statistics CSV."""
    path = results_dir / "tool_usage_stats.csv"
    if tool_df.empty:
        print("[skip] no tool usage data to write")
        return

    # Write per-example tool usage
    export_df = tool_df.drop(columns=["tools_used"], errors="ignore").copy()
    export_df.to_csv(path, index=False)
    print(f"[ok] wrote tool usage stats -> {path}")

    # Also write tool frequency as a separate file
    freq = tool_stats.get("tool_frequency", {})
    if freq:
        freq_path = results_dir / "tool_frequency.csv"
        freq_df = pd.DataFrame(
            [{"tool": k, "count": v} for k, v in freq.items()]
        )
        freq_df.to_csv(freq_path, index=False)
        print(f"[ok] wrote tool frequency -> {freq_path}")


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_comparison_table(summary_df: pd.DataFrame) -> None:
    """Pretty-print the comparison table to stdout."""
    if summary_df.empty:
        return

    print()
    print("=" * 90)
    print("  COMPARATIVE EVALUATION RESULTS")
    print("=" * 90)

    # Determine column widths
    methods = summary_df["display_name"].tolist()
    name_width = max(len(m) for m in methods) + 2

    metric_labels = {
        "event_accuracy": "Event Accuracy",
        "board_accuracy": "Board Accuracy",
        "category_accuracy": "Category Accuracy",
        "board_f1_mean": "Board F1 (mean)",
        "alarm_f1_mean": "Alarm F1 (mean)",
        "severity_f1_mean": "Severity F1 (mean)",
        "format_score_mean": "Format Score (mean)",
        "e2e_score_mean": "End-to-End Score",
    }

    # Header
    header = f"  {'Metric':<22}"
    for _, row in summary_df.iterrows():
        header += f"  {row['display_name']:>{name_width}}"
    print(header)
    print("  " + "-" * (22 + (name_width + 2) * len(summary_df)))

    # N examples row
    n_row = f"  {'N examples':<22}"
    for _, row in summary_df.iterrows():
        n_row += f"  {int(row.get('n_examples', 0)):>{name_width}}"
    print(n_row)
    print("  " + "-" * (22 + (name_width + 2) * len(summary_df)))

    # Metric rows
    for metric_key, metric_label in metric_labels.items():
        line = f"  {metric_label:<22}"
        for _, row in summary_df.iterrows():
            val = row.get(metric_key, 0.0)
            line += f"  {val:>{name_width}.1%}"
        print(line)

    print("  " + "-" * (22 + (name_width + 2) * len(summary_df)))

    # Highlight the best method for each key metric
    best_methods = {}
    for metric_key in ["event_accuracy", "e2e_score_mean"]:
        if metric_key in summary_df.columns:
            best_idx = summary_df[metric_key].idxmax()
            best_methods[metric_key] = summary_df.loc[best_idx, "display_name"]

    if best_methods:
        print()
        print("  Best performers:")
        for metric_key, method_name in best_methods.items():
            label = metric_labels.get(metric_key, metric_key)
            val = summary_df.loc[
                summary_df["display_name"] == method_name, metric_key
            ].iloc[0]
            print(f"    {label}: {method_name} ({val:.1%})")

    print("=" * 90)
    print()


def print_difficulty_analysis(difficulty_df: pd.DataFrame) -> None:
    """Print the difficulty analysis to stdout."""
    if difficulty_df.empty:
        return

    print("=" * 70)
    print("  DIFFICULTY ANALYSIS")
    print("=" * 70)

    counts = difficulty_df["difficulty"].value_counts()
    total = len(difficulty_df)
    for level in ["easy", "medium", "hard", "impossible"]:
        n = counts.get(level, 0)
        pct = 100 * n / total if total > 0 else 0.0
        print(f"  {level:>12s}: {n:3d} / {total} ({pct:5.1f}%)")

    # Per-event breakdown for hard/impossible
    hard_mask = difficulty_df["difficulty"].isin(["hard", "impossible"])
    if hard_mask.any():
        print()
        print("  Hard/impossible examples by event type:")
        hard_events = difficulty_df.loc[hard_mask, "gt_event"].value_counts()
        for evt, cnt in hard_events.items():
            print(f"    {evt:30s}  {cnt}")

    print("=" * 70)
    print()


def print_agent_advantage(
    agent_advantage_df: pd.DataFrame,
    results_dict: dict[str, pd.DataFrame],
) -> None:
    """Print where the agent beat single-shot methods (and where it lost)."""
    if agent_advantage_df.empty:
        print("[info] No agent advantage/regression data to report.")
        return

    print("=" * 70)
    print("  AGENT vs SINGLE-SHOT ANALYSIS")
    print("=" * 70)

    if "_is_advantage" in agent_advantage_df.columns:
        advantages = agent_advantage_df[
            agent_advantage_df["_is_advantage"] == True
        ]
        regressions = agent_advantage_df[
            agent_advantage_df["_is_advantage"] == False
        ]
    else:
        advantages = agent_advantage_df
        regressions = pd.DataFrame()

    print(f"  Agent-only correct (advantage):  {len(advantages)}")
    print(f"  Agent wrong, others correct (regression): {len(regressions)}")

    if not advantages.empty and "gt_event" in advantages.columns:
        print()
        print("  Agent advantage breakdown by event type:")
        for evt, cnt in advantages["gt_event"].value_counts().items():
            print(f"    {evt:30s}  {cnt}")

    if not regressions.empty and "gt_event" in regressions.columns:
        print()
        print("  Agent regression breakdown by event type:")
        for evt, cnt in regressions["gt_event"].value_counts().items():
            print(f"    {evt:30s}  {cnt}")

    net = len(advantages) - len(regressions)
    print()
    print(f"  Net agent advantage: {'+' if net > 0 else ''}{net} examples")
    print("=" * 70)
    print()


def print_tool_usage(tool_stats: dict[str, Any]) -> None:
    """Print tool usage statistics to stdout."""
    if not tool_stats:
        return

    print("=" * 70)
    print("  AGENT TOOL USAGE ANALYSIS")
    print("=" * 70)

    print(f"  Examples:               {tool_stats.get('n_examples', 0)}")
    print(f"  Agent Event Accuracy:   {tool_stats.get('event_accuracy', 0):.1%}")
    print(f"  Avg steps per example:  {tool_stats.get('avg_steps', 0):.1f}")
    print(f"  Avg tool calls:         {tool_stats.get('avg_tool_calls', 0):.1f}")
    print(f"  Avg elapsed time:       {tool_stats.get('avg_elapsed_sec', 0):.1f}s")
    print(f"  Avg tokens:             {tool_stats.get('avg_tokens', 0):.0f}")

    print()
    print("  Steps/tools for correct vs incorrect predictions:")
    print(
        f"    Correct:   avg_steps={tool_stats.get('avg_steps_correct', 0):.1f}  "
        f"avg_tools={tool_stats.get('avg_tool_calls_correct', 0):.1f}"
    )
    print(
        f"    Incorrect: avg_steps={tool_stats.get('avg_steps_incorrect', 0):.1f}  "
        f"avg_tools={tool_stats.get('avg_tool_calls_incorrect', 0):.1f}"
    )

    # Tool frequency
    freq = tool_stats.get("tool_frequency", {})
    if freq:
        print()
        print("  Tool call frequency:")
        total_calls = sum(freq.values())
        for tool_name, count in sorted(freq.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_calls if total_calls > 0 else 0.0
            print(f"    {tool_name:30s}  {count:4d}  ({pct:5.1f}%)")

    # Tool count vs accuracy
    tca = tool_stats.get("tool_count_vs_accuracy", {})
    if tca:
        print()
        print("  Tool count vs accuracy (more tools = better?):")
        for bin_label, bin_data in tca.items():
            print(
                f"    {bin_label:>6s} tools:  "
                f"n={bin_data['count']:3d}  "
                f"accuracy={bin_data['accuracy']:.1%}"
            )

    # Top tool sequences
    seqs = tool_stats.get("top_sequences", {})
    if seqs:
        print()
        print("  Top tool sequences:")
        for seq, count in list(seqs.items())[:5]:
            print(f"    {count:3d}x  {seq}")

    print("=" * 70)
    print()


def print_executive_summary(
    summary_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    agent_advantage_df: pd.DataFrame,
    tool_stats: dict[str, Any],
    results_dict: dict[str, pd.DataFrame],
) -> None:
    """Print a concise executive summary highlighting key findings."""
    print()
    print("#" * 70)
    print("#  EXECUTIVE SUMMARY")
    print("#" * 70)
    print()

    if summary_df.empty:
        print("  No methods evaluated.")
        print()
        return

    methods_evaluated = summary_df["display_name"].tolist()
    print(f"  Methods evaluated: {', '.join(methods_evaluated)}")
    print()

    # Best overall method
    if "e2e_score_mean" in summary_df.columns and len(summary_df) > 0:
        best_idx = summary_df["e2e_score_mean"].idxmax()
        best_method = summary_df.loc[best_idx, "display_name"]
        best_e2e = summary_df.loc[best_idx, "e2e_score_mean"]
        print(f"  Best overall method (E2E Score): {best_method} ({best_e2e:.1%})")

    if "event_accuracy" in summary_df.columns and len(summary_df) > 0:
        best_idx = summary_df["event_accuracy"].idxmax()
        best_method = summary_df.loc[best_idx, "display_name"]
        best_acc = summary_df.loc[best_idx, "event_accuracy"]
        print(f"  Best event accuracy:             {best_method} ({best_acc:.1%})")

    # Agent-specific findings
    if "agent" in results_dict and not summary_df.empty:
        agent_row = summary_df[summary_df["method"] == "agent"]
        if not agent_row.empty:
            agent_e2e = agent_row["e2e_score_mean"].iloc[0]
            agent_evt = agent_row["event_accuracy"].iloc[0]

            # Compare to non-agent methods
            non_agent = summary_df[summary_df["method"] != "agent"]
            if not non_agent.empty:
                best_non_agent_e2e = non_agent["e2e_score_mean"].max()
                best_non_agent_evt = non_agent["event_accuracy"].max()
                best_na_method_e2e = non_agent.loc[
                    non_agent["e2e_score_mean"].idxmax(), "display_name"
                ]
                best_na_method_evt = non_agent.loc[
                    non_agent["event_accuracy"].idxmax(), "display_name"
                ]

                e2e_delta = agent_e2e - best_non_agent_e2e
                evt_delta = agent_evt - best_non_agent_evt

                print()
                print("  Agent vs best single-shot:")
                print(
                    f"    E2E Score:      Agent {agent_e2e:.1%}  vs  "
                    f"{best_na_method_e2e} {best_non_agent_e2e:.1%}  "
                    f"(delta: {'+' if e2e_delta >= 0 else ''}{e2e_delta:.1%})"
                )
                print(
                    f"    Event Accuracy: Agent {agent_evt:.1%}  vs  "
                    f"{best_na_method_evt} {best_non_agent_evt:.1%}  "
                    f"(delta: {'+' if evt_delta >= 0 else ''}{evt_delta:.1%})"
                )

    # Agent advantage
    if not agent_advantage_df.empty and "_is_advantage" in agent_advantage_df.columns:
        advantages = agent_advantage_df[
            agent_advantage_df["_is_advantage"] == True
        ]
        regressions = agent_advantage_df[
            agent_advantage_df["_is_advantage"] == False
        ]
        print()
        print(
            f"  Agent-only correct examples:     {len(advantages)}"
        )
        print(
            f"  Agent regression examples:        {len(regressions)}"
        )
        net = len(advantages) - len(regressions)
        print(
            f"  Net advantage:                    "
            f"{'+' if net > 0 else ''}{net}"
        )

    # Difficulty distribution
    if not difficulty_df.empty and "difficulty" in difficulty_df.columns:
        counts = difficulty_df["difficulty"].value_counts()
        total = len(difficulty_df)
        print()
        print("  Difficulty distribution:")
        for level in ["easy", "medium", "hard", "impossible"]:
            n = counts.get(level, 0)
            pct = 100 * n / total if total > 0 else 0.0
            print(f"    {level:>12s}: {n:3d} ({pct:4.1f}%)")

    # Tool usage highlights
    if tool_stats:
        print()
        print("  Agent tool usage highlights:")
        print(f"    Avg tool calls per example: {tool_stats.get('avg_tool_calls', 0):.1f}")
        print(f"    Avg steps per example:      {tool_stats.get('avg_steps', 0):.1f}")
        freq = tool_stats.get("tool_frequency", {})
        if freq:
            top_tool = max(freq, key=freq.get)
            print(f"    Most used tool:             {top_tool} ({freq[top_tool]}x)")

    print()
    print("#" * 70)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Comparative evaluation of single-shot vs agentic RCA methods. "
            "Loads pre-computed output JSONL files for each method, runs "
            "evaluation metrics, and produces comparative analysis."
        ),
    )

    ap.add_argument(
        "--test-file",
        type=Path,
        default=Path("outputs/grpo_data/test.jsonl"),
        help="Test JSONL with prompts and ground truth (for reference/metadata).",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/eval"),
        help="Directory for output result files.",
    )

    # Per-method output files (all optional)
    ap.add_argument(
        "--rule-outputs",
        type=Path,
        default=None,
        help="JSONL with rule-based baseline outputs.",
    )
    ap.add_argument(
        "--zeroshot-outputs",
        type=Path,
        default=None,
        help="JSONL with zero-shot base model outputs.",
    )
    ap.add_argument(
        "--sft-outputs",
        type=Path,
        default=None,
        help="JSONL with SFT model outputs.",
    )
    ap.add_argument(
        "--grpo-outputs",
        type=Path,
        default=None,
        help="JSONL with SFT+GRPO model outputs.",
    )
    ap.add_argument(
        "--agent-outputs",
        type=Path,
        default=None,
        help="JSONL with agent (ReAct) outputs.",
    )

    args = ap.parse_args()

    # ── Discover and load available outputs ──
    method_files: dict[str, Path] = {}
    if args.rule_outputs:
        method_files["rule"] = args.rule_outputs
    if args.zeroshot_outputs:
        method_files["zeroshot"] = args.zeroshot_outputs
    if args.sft_outputs:
        method_files["sft"] = args.sft_outputs
    if args.grpo_outputs:
        method_files["grpo"] = args.grpo_outputs
    if args.agent_outputs:
        method_files["agent"] = args.agent_outputs

    if not method_files:
        # Auto-discover from results_dir
        results_dir = args.results_dir
        auto_map = {
            "rule": results_dir / "baseline_rule_outputs.jsonl",
            "zeroshot": results_dir / "zeroshot_outputs.jsonl",
            "sft": results_dir / "sft_outputs.jsonl",
            "grpo": results_dir / "grpo_outputs.jsonl",
            "agent": results_dir / "agent_outputs.jsonl",
        }
        for method_name, path in auto_map.items():
            if path.exists():
                method_files[method_name] = path
                print(f"[auto] found {method_name} outputs: {path}")

    if not method_files:
        print(
            "[error] No output files found. Provide at least one "
            "--*-outputs flag or place JSONL files in --results-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load outputs
    print()
    method_outputs: dict[str, list[dict]] = {}
    for method_name, path in method_files.items():
        display = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        outputs = load_outputs(path)
        if outputs:
            method_outputs[method_name] = outputs
            print(f"[ok] loaded {len(outputs):3d} outputs for {display} from {path}")
        else:
            print(f"[skip] no valid outputs for {display} from {path}")

    if not method_outputs:
        print("[error] No valid outputs loaded. Nothing to evaluate.", file=sys.stderr)
        sys.exit(1)

    # ── Evaluate each method ──
    print()
    results_dict: dict[str, pd.DataFrame] = {}
    for method_name, outputs in method_outputs.items():
        display = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        print(f"[info] evaluating {display} ({len(outputs)} examples) ...")
        df = evaluate_outputs(outputs, method_name)
        if not df.empty:
            results_dict[method_name] = df

    # ── Comparative analysis ──
    print()
    print("[info] running comparative analysis ...")
    summary_df, difficulty_df, agent_advantage_df = compare_methods(results_dict)

    # ── Tool usage analysis (agent only) ──
    tool_stats: dict[str, Any] = {}
    tool_df = pd.DataFrame()
    if "agent" in method_outputs:
        print("[info] analyzing agent tool usage ...")
        tool_df = analyze_tool_usage(method_outputs["agent"])
        tool_stats = _summarize_tool_usage(tool_df)

    # ── Write output files ──
    args.results_dir.mkdir(parents=True, exist_ok=True)

    _write_comparison_summary(summary_df, args.results_dir)
    _write_agent_advantage(agent_advantage_df, args.results_dir)
    _write_tool_usage_stats(tool_stats, tool_df, args.results_dir)

    # ── Print to stdout ──
    print_comparison_table(summary_df)
    print_difficulty_analysis(difficulty_df)
    print_agent_advantage(agent_advantage_df, results_dict)
    print_tool_usage(tool_stats)
    print_executive_summary(
        summary_df, difficulty_df, agent_advantage_df, tool_stats, results_dict
    )


if __name__ == "__main__":
    main()
