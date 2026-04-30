#!/usr/bin/env python3
"""Run ReAct agent on SFC-BFL datasets.

Wraps the existing RCAAgent with SFC-BFL-native tools from agent_tools.py.
Dispatches tool calls to in-memory data instead of file paths.

Usage:
  python -m sfc_bfl.agent_runner \
    --dataset-dir outputs/sfc_bfl/dataset_a_pilot \
    --rules outputs/Static\ files/rule_database.csv \
    --backend api --api-url http://localhost:11434/v1 --api-model qwen2.5:14b \
    --ow-ids 0-9 \
    --out outputs/sfc_bfl_eval/agent_dataset_a.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sfc_bfl.convert_to_prompts import SFCBFLDataset, _timeseries_summary_from_parquet
from sfc_bfl.agent_tools import (
    SFC_TOOL_REGISTRY,
    format_sfc_tools_for_prompt,
    get_metrics_sfc,
    get_top_boards_sfc,
    lookup_rule_sfc,
    query_topology_sfc,
    trace_service_path,
    validate_propagation_sfc,
)
from rca_agent import (
    AgentResult,
    AgentStep,
    _parse_response,
    _parse_action,
    _extract_answer,
    _extract_predicted_fields,
    _triage_category,
    _build_specialist_prompt,
)
from generate_sft_data import METRIC_GUIDE


# ---------------------------------------------------------------------------
# SFC-BFL Agent System Prompt
# ---------------------------------------------------------------------------

_SFC_AGENT_SYSTEM_TEMPLATE = """\
You are an OTN Root Cause Analysis Agent. You investigate network failures by \
reasoning step-by-step and using tools to gather evidence.

You have a maximum of {max_steps} tool calls. Plan efficiently.

Available tools:
{tools_description}

To use a tool, output a line in this exact format:
ACTION: tool_name(param1="value1", param2="value2")

Recommended investigation plan:
1. get_top_boards → identify which boards have elevated BBE
2. get_metrics → get pattern_kind, bbe_max for the peak board
3. query_topology → get board type and category
4. validate_propagation → verify alarm flow hypothesis

After 4-5 tool calls, you should have enough evidence. You MUST then produce \
your final answer. Do NOT keep calling tools indefinitely.

When you have enough evidence, output your answer in EXACTLY this format:

ANSWER:
Root cause (predicted):
- Board: ROADM0-OA002$0
- Event: Fiber_Cut
- Time: t=0
- Severity: Critical

Predicted alarm_flow rows (CSV):
SourceBoard,SourceAlarm,SourceEvent,...
{{csv rows}}

RCA reasoning:
{{reasoning}}

Fix ideas:
{{fixes}}

Important rules:
1. Only call ONE tool per response. Wait for the OBSERVATION before continuing.
2. After each OBSERVATION, reason about what you learned before deciding the next action.
3. When you have enough evidence, produce your ANSWER (do NOT output an ACTION line).
4. If a tool returns an error, try an alternative approach.
5. Your ANSWER MUST include "- Board:" and "- Event:" lines with actual values.
"""


def _build_sfc_system_prompt(max_steps: int = 10) -> str:
    tools_desc = format_sfc_tools_for_prompt()
    return _SFC_AGENT_SYSTEM_TEMPLATE.format(
        tools_description=tools_desc,
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# Tool execution dispatcher
# ---------------------------------------------------------------------------

def _execute_sfc_tool(
    tool_name: str,
    kwargs: dict[str, Any],
    *,
    dataset: SFCBFLDataset,
    ow_df: "pd.DataFrame",
    rules_path: str,
) -> str:
    """Execute an SFC-BFL tool by name with given kwargs.

    Routes tool calls to in-memory SFC-BFL data.
    """
    import pandas as pd

    if tool_name not in SFC_TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {tool_name}"}, indent=2)

    try:
        if tool_name == "get_top_boards":
            top_n = int(kwargs.get("top_n", 10))
            result = get_top_boards_sfc(ow_df, top_n=top_n)
        elif tool_name == "get_metrics":
            board_name = kwargs.get("board_name", "")
            result = get_metrics_sfc(board_name, ow_df)
        elif tool_name == "query_topology":
            board_name = kwargs.get("board_name", "")
            result = query_topology_sfc(
                board_name, dataset.boards_df, dataset.edges_df,
                dataset.lightpaths, dataset.services,
            )
        elif tool_name == "trace_service_path":
            service_id = kwargs.get("service_id", "")
            result = trace_service_path(service_id, dataset.lightpaths, dataset.boards_df)
        elif tool_name == "lookup_rule":
            board_type = kwargs.get("board_type", "")
            event_name = kwargs.get("event_name", "")
            result = lookup_rule_sfc(board_type, event_name, rules_path)
        elif tool_name == "validate_propagation":
            root_board = kwargs.get("root_board", "")
            root_event = kwargs.get("root_event", "")
            result = validate_propagation_sfc(
                root_board, root_event, dataset.lightpaths, rules_path,
                severity=kwargs.get("severity", "Critical"),
            )
        else:
            result = {"error": f"Unhandled tool: {tool_name}"}
    except Exception as exc:
        result = {"error": f"Tool {tool_name} raised {type(exc).__name__}: {exc}"}

    try:
        return json.dumps(result, indent=2, default=str)
    except (TypeError, ValueError):
        return str(result)


# ---------------------------------------------------------------------------
# Agent runner (ReAct loop)
# ---------------------------------------------------------------------------

def run_sfc_agent(
    dataset: SFCBFLDataset,
    ow_id: int,
    rules_path: str,
    backend: Any,
    max_steps: int = 8,
    verbose: bool = False,
) -> AgentResult:
    """Run the ReAct agent on a single SFC-BFL OW."""
    ow_df = dataset.get_ow_metrics(ow_id)
    label = dataset.get_ow_label(ow_id)

    # Build user message with metrics summary
    ts_summary = _timeseries_summary_from_parquet(ow_df)
    user_msg = (
        f"Investigate this network failure.\n\n"
        f"Metrics summary:\n{ts_summary}\n\n"
        f"{METRIC_GUIDE}\n\n"
        "Use the tools to gather evidence and determine the root cause board and event type.\n"
    )

    # Build system prompt
    system_prompt = _build_sfc_system_prompt(max_steps=max_steps)

    # Initialize conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    agent_result = AgentResult()
    step_num = 0

    while step_num < max_steps:
        # Call LLM
        response = backend.generate(messages)

        # Parse response
        thought, action_str = _parse_response(response)

        if action_str is None:
            # Final answer
            answer = _extract_answer(response)
            agent_result.final_answer = answer
            agent_result.steps.append(AgentStep(
                step_num=step_num, thought=thought,
            ))
            break

        # Parse and execute tool
        try:
            tool_name, kwargs = _parse_action(action_str)
        except ValueError as e:
            observation = json.dumps({"error": str(e)})
            agent_result.steps.append(AgentStep(
                step_num=step_num, thought=thought,
                action=action_str, observation=observation,
            ))
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
            step_num += 1
            continue

        observation = _execute_sfc_tool(
            tool_name, kwargs,
            dataset=dataset, ow_df=ow_df, rules_path=rules_path,
        )
        agent_result.num_tool_calls += 1

        if verbose:
            print(f"  Step {step_num}: {tool_name}({kwargs}) -> {observation[:200]}...")

        agent_result.steps.append(AgentStep(
            step_num=step_num, thought=thought,
            action=action_str, observation=observation,
        ))

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
        step_num += 1

    # If we exhausted steps without an ANSWER, use last response
    if not agent_result.final_answer and agent_result.steps:
        agent_result.final_answer = agent_result.steps[-1].thought

    # Extract predictions
    event, board = _extract_predicted_fields(agent_result.final_answer)
    agent_result.predicted_event = event
    agent_result.predicted_board = board

    return agent_result


# ---------------------------------------------------------------------------
# LLM Backend (API only for SFC-BFL — training/inference is on server)
# ---------------------------------------------------------------------------

class _APIBackend:
    """OpenAI-compatible API backend."""

    def __init__(self, api_url: str, api_model: str, temperature: float = 0.3,
                 max_tokens: int = 4096):
        self.api_url = api_url.rstrip("/")
        self.api_model = api_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict]) -> str:
        import urllib.request
        url = f"{self.api_url}/chat/completions"
        payload = {
            "model": self.api_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run ReAct agent on SFC-BFL datasets."
    )
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--rules", type=Path,
                    default=Path("outputs/Static files/rule_database.csv"))
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL (compatible with evaluate.py)")
    ap.add_argument("--ow-ids", type=str, default=None,
                    help="OW ID range (e.g. '0-9' or '0,5,10')")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")

    # API backend
    ap.add_argument("--backend", choices=["api"], default="api")
    ap.add_argument("--api-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-model", default="qwen2.5:14b")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=4096)

    args = ap.parse_args()

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

    # Handle subdirectories (Dataset D/E)
    dataset_dir = args.dataset_dir
    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()
               and (d / "ow_labels.jsonl").exists()] if dataset_dir.exists() else []

    if (dataset_dir / "ow_labels.jsonl").exists():
        dirs_to_process = [(dataset_dir, None)]
    elif subdirs:
        dirs_to_process = [(d, d.name) for d in sorted(subdirs)]
    else:
        raise FileNotFoundError(f"No dataset found in {dataset_dir}")

    # Create backend
    backend = _APIBackend(
        args.api_url, args.api_model,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    t0 = time.time()

    with open(args.out, "w", encoding="utf-8") as fout:
        for ddir, subdir_name in dirs_to_process:
            print(f"[info] loading dataset from {ddir}")
            ds = SFCBFLDataset(ddir)

            for ow_id in ds.ow_ids:
                if ow_id_set is not None and ow_id not in ow_id_set:
                    continue

                label = ds.get_ow_label(ow_id)
                alarms = ds.get_ow_alarms(ow_id)

                print(f"  [ow={ow_id}] running agent...")
                result = run_sfc_agent(
                    ds, ow_id, str(args.rules),
                    backend=backend,
                    max_steps=args.max_steps,
                    verbose=args.verbose,
                )

                # Build ground truth
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

                record = {
                    "sample_index": total,
                    "model_output": result.final_answer,
                    "method": "agent",
                    "ground_truth_root_event": label.get("root_event", ""),
                    "ground_truth_root_board": label.get("root_board", ""),
                    "ground_truth_boards": json.dumps(gt_boards),
                    "ground_truth_alarms": json.dumps(gt_alarms),
                    "ground_truth_severity_dist": json.dumps(gt_severity),
                    "ground_truth_row_count": min(len(alarms), 20),
                    "dataset": str(ds.root.name),
                    "ow_id": ow_id,
                    "num_tool_calls": result.num_tool_calls,
                    "predicted_event": result.predicted_event,
                    "predicted_board": result.predicted_board,
                }
                if subdir_name:
                    record["dataset_subdir"] = subdir_name

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                total += 1

                elapsed = time.time() - t0
                print(f"  [ow={ow_id}] event={result.predicted_event} "
                      f"board={result.predicted_board} "
                      f"tools={result.num_tool_calls} "
                      f"elapsed={elapsed:.0f}s")

    print(f"\n[ok] wrote {total} agent outputs -> {args.out}")


if __name__ == "__main__":
    main()
