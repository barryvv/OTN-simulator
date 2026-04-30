#!/usr/bin/env python3
"""Generate ReAct agent training traces via Claude API.

For each training example, sends the prompt to Claude with tool definitions,
executes tool calls locally against real simulator data, and captures the
full multi-turn reasoning trace for SFT training.

Usage:
  # Generate traces from existing training data
  python generate_agent_traces.py \
    --train-file outputs/grpo_data/train.jsonl \
    --out outputs/agent_traces/train.jsonl \
    --lightpaths regen_compare/noregen/topology/lightpaths.txt \
    --rules "outputs/Static files/rule_database.csv" \
    --max-samples 0 \
    --verbose

  # Resume from a partial run
  python generate_agent_traces.py \
    --train-file outputs/grpo_data/train.jsonl \
    --out outputs/agent_traces/train.jsonl \
    --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from rca_tools import (
    TOOL_REGISTRY,
    get_metrics,
    lookup_rule,
    query_topology,
    search_similar,
    validate_propagation,
)
from rca_agent import _build_specialist_prompt

# ---------------------------------------------------------------------------
# Claude API tool definitions (native tool use format)
# ---------------------------------------------------------------------------

CLAUDE_TOOLS = [
    {
        "name": "query_topology",
        "description": (
            "Query board type, connections, and services for a given board. "
            "Returns board type, category (optical/fiber, electrical/xcon, electrical/line), "
            "upstream/downstream neighbors, services passing through, and path positions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "board_name": {
                    "type": "string",
                    "description": "Board ID like ROADM0-OA002$0",
                },
            },
            "required": ["board_name"],
        },
    },
    {
        "name": "get_metrics",
        "description": (
            "Get BBE/BBER/ES metrics and pattern features for a specific board. "
            "Returns bbe_max, bber_max, es_max, pattern_kind (step/ramp/burst/step_recovery), "
            "burst_regions, longest_burst_frac, es_frac, bber_bbe_ratio, and ranking among all boards."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "board_name": {
                    "type": "string",
                    "description": "Board ID like ROADM0-OA002$0",
                },
            },
            "required": ["board_name"],
        },
    },
    {
        "name": "lookup_rule",
        "description": (
            "Look up propagation rules from the rule database for a given function type and event. "
            "Returns matching rules with action type, target type, and output alarm."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "board_type": {
                    "type": "string",
                    "description": "OTN function type (e.g. ODUk-Creation, OTUk-Relay)",
                },
                "event_name": {
                    "type": "string",
                    "description": "Alarm/event name (e.g. Fiber_Cut, ODUk_PM_AIS)",
                },
            },
            "required": ["board_type", "event_name"],
        },
    },
    {
        "name": "validate_propagation",
        "description": (
            "Run alarm propagation forward from a hypothesised root cause. "
            "Returns the predicted alarm flow including affected boards, alarm distribution, "
            "severity distribution, and full CSV output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root_board": {
                    "type": "string",
                    "description": "Hypothesised root cause board (e.g. ROADM0-OA002$0)",
                },
                "root_event": {
                    "type": "string",
                    "description": "Hypothesised root event (e.g. Fiber_Cut)",
                },
            },
            "required": ["root_board", "root_event"],
        },
    },
    {
        "name": "search_similar",
        "description": (
            "Search training examples for similar past cases by pattern kind, board type, "
            "and BBE range. Returns top-3 most similar examples with similarity scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern_kind": {
                    "type": "string",
                    "description": "Temporal pattern (step/ramp/burst/step_recovery)",
                },
                "board_type": {
                    "type": "string",
                    "description": "Board type (OA, XCON, Line, etc.)",
                },
                "bbe_min": {
                    "type": "number",
                    "description": "Minimum BBE value for range filter",
                },
                "bbe_max": {
                    "type": "number",
                    "description": "Maximum BBE value for range filter",
                },
            },
            "required": ["pattern_kind", "board_type", "bbe_min", "bbe_max"],
        },
    },
]

_EVENT_TO_CATEGORY = {
    "Fiber_Aging": "Fiber",
    "Fiber_Crack": "Fiber",
    "Fiber_Cut": "Fiber",
    "XCON_Port_Down": "XCON",
    "XCON_Buffer_Overflow": "XCON",
    "XCON_Fabric_Fault": "XCON",
    "Line_Disconnect": "Line",
}

AGENT_SYSTEM_PROMPT = """\
You are an OTN Root Cause Analysis Agent. You investigate network failures by \
reasoning step-by-step and using tools to gather evidence.

You have a maximum of 8 tool calls. Plan efficiently.

IMPORTANT: You MUST use tools to gather evidence before making your diagnosis. \
Follow this systematic approach:
1. get_metrics → identify peak board, pattern_kind, bbe_max
2. query_topology → get board type and category
3. lookup_rule or search_similar → confirm event hypothesis
4. validate_propagation → verify alarm flow

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
(csv rows)

RCA reasoning:
(reasoning)

Fix ideas:
(fixes)

Important rules:
1. Only call ONE tool per response. Wait for the result before continuing.
2. After each tool result, reason about what you learned before deciding the next action.
3. When you have enough evidence, produce your ANSWER (do NOT call another tool).
4. If a tool returns an error, try an alternative approach.
5. Your ANSWER MUST include "- Board:" and "- Event:" lines with actual values."""


# ---------------------------------------------------------------------------
# Tool execution (local)
# ---------------------------------------------------------------------------

def execute_tool_locally(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    topology_yaml: str | None = None,
    lightpaths_file: str | None = None,
    rules_path: str | None = None,
    data_dir: str | None = None,
    training_data: str | None = None,
) -> str:
    """Execute a tool call locally and return JSON result string."""
    try:
        if tool_name == "query_topology":
            result = query_topology(
                board_name=tool_input["board_name"],
                topology_yaml=topology_yaml,
                lightpaths_file=lightpaths_file,
            )
        elif tool_name == "get_metrics":
            result = get_metrics(
                board_name=tool_input["board_name"],
                data_dir=data_dir or "",
            )
        elif tool_name == "lookup_rule":
            result = lookup_rule(
                board_type=tool_input.get("board_type", ""),
                event_name=tool_input.get("event_name", ""),
                rules_path=rules_path or "",
            )
        elif tool_name == "validate_propagation":
            result = validate_propagation(
                root_board=tool_input["root_board"],
                root_event=tool_input["root_event"],
                rules_path=rules_path or "",
                lightpaths_file=lightpaths_file or "",
            )
        elif tool_name == "search_similar":
            bbe_min = tool_input.get("bbe_min", 0.0)
            bbe_max = tool_input.get("bbe_max", 100.0)
            result = search_similar(
                pattern_kind=tool_input.get("pattern_kind", ""),
                board_type=tool_input.get("board_type", ""),
                bbe_range=(float(bbe_min), float(bbe_max)),
                training_data_path=training_data or "",
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}

    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Claude API interaction
# ---------------------------------------------------------------------------

def _call_claude_api(
    messages: list[dict],
    system: str,
    tools: list[dict],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> dict:
    """Call Claude API with tool use support. Returns the response dict."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "Claude API requires the 'anthropic' package. "
            "Install with: pip install anthropic"
        )

    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=tools,
        messages=messages,
        temperature=temperature,
    )

    return response


def _truncate_observation(obs: str, max_chars: int = 3000) -> str:
    """Truncate long tool observations to keep context manageable."""
    if len(obs) <= max_chars:
        return obs
    # For alarm_flow_csv, truncate the CSV portion
    if "alarm_flow_csv" in obs:
        try:
            data = json.loads(obs)
            if "alarm_flow_csv" in data:
                csv_lines = data["alarm_flow_csv"].splitlines()
                if len(csv_lines) > 10:
                    data["alarm_flow_csv"] = (
                        "\n".join(csv_lines[:10])
                        + f"\n... ({len(csv_lines) - 10} more rows)"
                    )
                return json.dumps(data, indent=2, default=str)
        except (json.JSONDecodeError, KeyError):
            pass
    return obs[:max_chars] + f"\n... (truncated, {len(obs)} total chars)"


# ---------------------------------------------------------------------------
# Trace generation for one example
# ---------------------------------------------------------------------------

def generate_trace(
    user_prompt: str,
    ground_truth: dict[str, Any],
    *,
    topology_yaml: str | None = None,
    lightpaths_file: str | None = None,
    rules_path: str | None = None,
    data_dir: str | None = None,
    training_data: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    max_steps: int = 8,
    verbose: bool = False,
    system_prompt: str | None = None,
) -> dict[str, Any] | None:
    """Generate a ReAct trace for one example using Claude API.

    Returns a dict with the conversation trace in chat format, or None on failure.
    """
    effective_system_prompt = system_prompt or AGENT_SYSTEM_PROMPT

    messages: list[dict] = [
        {"role": "user", "content": user_prompt},
    ]

    trace_turns: list[dict[str, str]] = []
    tool_calls_made = 0

    for step in range(max_steps):
        try:
            response = _call_claude_api(
                messages=messages,
                system=effective_system_prompt,
                tools=CLAUDE_TOOLS,
                model=model,
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as exc:
            print(f"    [API error at step {step}] {exc}", file=sys.stderr)
            return None

        # Process response content blocks
        assistant_content = []
        tool_use_blocks = []
        text_parts = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                assistant_content.append({
                    "type": "text",
                    "text": block.text,
                })
            elif block.type == "tool_use":
                tool_use_blocks.append(block)
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        # Add assistant message to conversation
        messages.append({
            "role": "assistant",
            "content": assistant_content,
        })

        if verbose and text_parts:
            preview = " ".join(text_parts)[:200]
            print(f"    [step {step}] thought: {preview}...", file=sys.stderr)

        # If no tool use, we have the final answer
        if not tool_use_blocks or response.stop_reason == "end_turn":
            # Record final text as assistant response
            final_text = "\n".join(text_parts)
            # Ensure the final answer starts with ANSWER: marker for
            # training/inference format alignment
            if "ANSWER:" not in final_text:
                final_text = "ANSWER:\n" + final_text
            trace_turns.append({
                "role": "assistant",
                "content": final_text,
            })
            break

        # Execute each tool call and collect results
        tool_results = []
        for tool_block in tool_use_blocks:
            tool_calls_made += 1
            if verbose:
                print(
                    f"    [step {step}] ACTION: {tool_block.name}("
                    f"{json.dumps(tool_block.input, default=str)[:100]})",
                    file=sys.stderr,
                )

            obs = execute_tool_locally(
                tool_block.name,
                tool_block.input,
                topology_yaml=topology_yaml,
                lightpaths_file=lightpaths_file,
                rules_path=rules_path,
                data_dir=data_dir,
                training_data=training_data,
            )
            obs = _truncate_observation(obs)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": obs,
            })

            # Record in trace: thought + action + observation
            thought_text = "\n".join(text_parts) if text_parts else ""
            action_str = f"ACTION: {tool_block.name}({_format_kwargs(tool_block.input)})"
            trace_turns.append({
                "role": "assistant",
                "content": f"{thought_text}\n{action_str}" if thought_text else action_str,
            })
            trace_turns.append({
                "role": "user",
                "content": f"OBSERVATION:\n{obs}",
            })

        # Send tool results back to Claude
        messages.append({
            "role": "user",
            "content": tool_results,
        })

        # Clear text_parts for next iteration
        text_parts = []
    else:
        # Exhausted max_steps — force a final answer from Claude
        if verbose:
            print(f"    [warn] max steps reached, forcing final answer", file=sys.stderr)
        # Ask Claude for final answer without tools
        messages.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "You have used all available tool steps. You MUST now provide "
                    "your final diagnosis. Do NOT call any more tools.\n\n"
                    "ANSWER:\nRoot cause (predicted):\n"
                    "- Board: <board_name>\n- Event: <event_type>"
                ),
            }],
        })
        try:
            final_resp = _call_claude_api(
                messages=messages,
                system=effective_system_prompt,
                tools=[],  # No tools available — force text-only
                model=model,
                max_tokens=4096,
                temperature=0.3,
            )
            final_parts = []
            for block in final_resp.content:
                if block.type == "text":
                    final_parts.append(block.text)
            final_text = "\n".join(final_parts)
            if "ANSWER:" not in final_text:
                final_text = "ANSWER:\n" + final_text
            trace_turns.append({
                "role": "assistant",
                "content": final_text,
            })
        except Exception as exc:
            if verbose:
                print(f"    [warn] failed to force final answer: {exc}", file=sys.stderr)

    if not trace_turns:
        return None

    # Build the training record
    # Format: multi-turn conversation with system, user prompt, and ReAct turns
    conversation = [
        {"role": "system", "content": effective_system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    conversation.extend(trace_turns)

    return {
        "messages": conversation,
        "ground_truth_root_board": ground_truth.get("ground_truth_root_board", ""),
        "ground_truth_root_event": ground_truth.get("ground_truth_root_event", ""),
        "ground_truth_alarm_flow_csv": ground_truth.get("ground_truth_alarm_flow_csv", ""),
        "ground_truth_boards": ground_truth.get("ground_truth_boards", "[]"),
        "ground_truth_alarms": ground_truth.get("ground_truth_alarms", "{}"),
        "ground_truth_severity_dist": ground_truth.get("ground_truth_severity_dist", "{}"),
        "ground_truth_row_count": ground_truth.get("ground_truth_row_count", 0),
        "trace_steps": len([t for t in trace_turns if t["role"] == "assistant"]),
        "trace_tool_calls": tool_calls_made,
    }


def _format_kwargs(d: dict) -> str:
    """Format a dict as kwarg string for ACTION: line."""
    parts = []
    for k, v in d.items():
        if isinstance(v, str):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Data dir detection
# ---------------------------------------------------------------------------

def _find_data_dir(run_dir_name: str, search_roots: list[Path]) -> str | None:
    """Try to find the data_outputs directory for a training example."""
    for root in search_roots:
        # Try exact match
        cand = root / f"data_outputs_{run_dir_name}"
        if cand.exists():
            return str(cand)
        # Try partial match
        for d in root.iterdir():
            if d.is_dir() and d.name.startswith("data_outputs_") and run_dir_name.lower() in d.name.lower():
                return str(d)
    return None


def _extract_run_name_from_example(example: dict) -> str | None:
    """Try to extract the run directory name from an example's ground truth."""
    gt_event = example.get("ground_truth_root_event", "")
    if gt_event:
        return gt_event.lower()
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate ReAct agent training traces via Claude API."
    )
    ap.add_argument(
        "--train-file", type=Path, required=True,
        help="Input JSONL with training examples (same format as GRPO train.jsonl)",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("outputs/agent_traces/train.jsonl"),
        help="Output JSONL path for agent traces",
    )
    ap.add_argument(
        "--lightpaths", type=str,
        default="regen_compare/noregen/topology/lightpaths.txt",
        help="Path to lightpaths file for tool execution",
    )
    ap.add_argument(
        "--topology-yaml", type=str,
        default="outputs/noregen_runs/topology_noregen.yaml",
        help="Path to topology YAML for tool execution",
    )
    ap.add_argument(
        "--rules", type=str,
        default="outputs/Static files/rule_database.csv",
        help="Path to rule_database.csv for tool execution",
    )
    ap.add_argument(
        "--training-data", type=str,
        default="outputs/grpo_data/train.jsonl",
        help="Path to training JSONL for search_similar tool",
    )
    ap.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to timeseries data directory (auto-detected if not set)",
    )
    ap.add_argument(
        "--data-search-roots", type=str, nargs="*",
        default=["outputs/grpo_runs", "outputs/alarm_flows/test_runs/noregen"],
        help="Directories to search for data_outputs_* (for auto-detection)",
    )
    ap.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="Claude model to use for trace generation",
    )
    ap.add_argument(
        "--max-steps", type=int, default=8,
        help="Maximum tool-use steps per example",
    )
    ap.add_argument(
        "--max-samples", type=int, default=0,
        help="Max samples to process (0 = all)",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file (skip already-processed indices)",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print detailed step-by-step output",
    )

    args = ap.parse_args()

    # Validate
    if not args.train_file.exists():
        raise SystemExit(f"[error] train file not found: {args.train_file}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "[error] ANTHROPIC_API_KEY environment variable not set. "
            "Set it with: export ANTHROPIC_API_KEY=sk-..."
        )

    # Load training examples
    examples: list[dict] = []
    with args.train_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if args.max_samples > 0:
        examples = examples[: args.max_samples]

    print(f"[info] loaded {len(examples)} training examples from {args.train_file}")

    # Check for resume
    completed_indices: set[int] = set()
    existing_traces: list[str] = []
    if args.resume and args.out.exists():
        with args.out.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_traces.append(line)
                    try:
                        rec = json.loads(line)
                        idx = rec.get("sample_index", -1)
                        if idx >= 0:
                            completed_indices.add(idx)
                    except json.JSONDecodeError:
                        pass
        print(f"[info] resuming: {len(completed_indices)} already completed")

    # Process examples
    args.out.parent.mkdir(parents=True, exist_ok=True)
    search_roots = [Path(p) for p in args.data_search_roots]

    success_count = 0
    fail_count = 0

    with args.out.open("w" if not args.resume else "a", encoding="utf-8") as out_f:
        # Write existing traces first if resuming and opening in write mode
        if not args.resume:
            pass  # Clean start
        # Else we opened in append mode

        for idx, example in enumerate(examples):
            if idx in completed_indices:
                continue

            gt_event = example.get("ground_truth_root_event", "?")
            gt_board = example.get("ground_truth_root_board", "?")
            print(f"\n[{idx + 1}/{len(examples)}] event={gt_event}, board={gt_board}")

            # Extract user prompt
            prompt_msgs = example.get("prompt", [])
            user_prompt = ""
            for msg in prompt_msgs:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_prompt = msg["content"]
                    break
            if not user_prompt:
                print(f"  [skip] no user prompt found")
                fail_count += 1
                continue

            # Auto-detect data_dir
            data_dir = args.data_dir
            if data_dir is None:
                run_name = _extract_run_name_from_example(example)
                if run_name:
                    data_dir = _find_data_dir(run_name, search_roots)

            # Determine specialist prompt from ground truth category
            category = _EVENT_TO_CATEGORY.get(gt_event, "unknown")
            if category != "unknown":
                specialist_prompt = _build_specialist_prompt(category, max_steps=args.max_steps)
            else:
                specialist_prompt = None  # falls back to generic AGENT_SYSTEM_PROMPT

            # Generate trace
            t0 = time.time()
            trace = generate_trace(
                user_prompt=user_prompt,
                ground_truth=example,
                topology_yaml=args.topology_yaml,
                lightpaths_file=args.lightpaths,
                rules_path=args.rules,
                data_dir=data_dir,
                training_data=args.training_data,
                model=args.model,
                max_steps=args.max_steps,
                verbose=args.verbose,
                system_prompt=specialist_prompt,
            )
            elapsed = time.time() - t0

            if trace is None:
                print(f"  [FAIL] trace generation failed ({elapsed:.1f}s)")
                fail_count += 1
                continue

            trace["sample_index"] = idx
            out_f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            out_f.flush()
            success_count += 1

            print(
                f"  [OK] steps={trace['trace_steps']}, "
                f"tools={trace['trace_tool_calls']}, "
                f"time={elapsed:.1f}s"
            )

            # Rate limiting — be gentle with the API
            time.sleep(0.5)

    # Summary
    total = success_count + fail_count
    print(f"\n{'=' * 60}")
    print(f"Trace Generation Summary")
    print(f"{'=' * 60}")
    print(f"Total processed: {total}")
    print(f"Successful:      {success_count}")
    print(f"Failed:          {fail_count}")
    print(f"Output:          {args.out}")
    if completed_indices:
        print(f"Resumed from:    {len(completed_indices)} previous traces")


if __name__ == "__main__":
    main()
