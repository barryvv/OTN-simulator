#!/usr/bin/env python3
"""ReAct-style agent framework for OTN root cause analysis.

Uses tools from rca_tools.py to perform multi-step reasoning to diagnose
network failures. Supports both a fine-tuned local LLM backend (via
transformers) and an API-based backend (via HTTP to any OpenAI-compatible API).

Usage:
  # Run agent on a single test example (API backend)
  python rca_agent.py \
    --test-file outputs/grpo_data/test.jsonl \
    --sample-index 0 \
    --backend api \
    --api-url http://localhost:11434/v1 \
    --api-model qwen2.5:14b \
    --max-steps 8 \
    --verbose

  # Run with local fine-tuned model
  python rca_agent.py \
    --test-file outputs/grpo_data/train.jsonl \
    --sample-index 0 \
    --backend local \
    --base-model Qwen/Qwen2.5-14B-Instruct \
    --adapter adapters/qwen-14b-sft-v6.1-grpo \
    --max-steps 8

  # Run on all examples and save outputs
  python rca_agent.py \
    --test-file outputs/grpo_data/test.jsonl \
    --run-all \
    --backend api \
    --api-url http://localhost:11434/v1 \
    --out outputs/eval/agent_outputs.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rca_tools import (
    TOOL_REGISTRY,
    format_tools_for_prompt,
    get_metrics,
    lookup_rule,
    query_topology,
    search_similar,
    validate_propagation,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """A single step in the ReAct loop."""

    step_num: int
    thought: str
    action: str | None = None  # tool call string, or None if final
    observation: str | None = None  # tool result, or None


@dataclass
class AgentResult:
    """Complete result of an agent run."""

    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_tokens: int = 0
    num_tool_calls: int = 0
    predicted_event: str = ""
    predicted_board: str = ""


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT_TEMPLATE = """\
You are an OTN Root Cause Analysis Agent. You investigate network failures by \
reasoning step-by-step and using tools to gather evidence.

You have a maximum of {max_steps} tool calls. Plan efficiently.

Available tools:
{tools_description}

To use a tool, output a line in this exact format:
ACTION: tool_name(param1="value1", param2="value2")

Recommended investigation plan:
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


def _build_system_prompt(max_steps: int = 10) -> str:
    """Build the system prompt with tool descriptions injected."""
    tools_desc = format_tools_for_prompt()
    return AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        tools_description=tools_desc,
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# Category-based triage
# ---------------------------------------------------------------------------

from rca_tools import _parse_board_name

_PEAK_BOARD_RE = re.compile(r"peak_board=([\w\-\$]+)")

CATEGORY_FROM_BOARD_TYPE = {
    "OA": "Fiber", "OD": "Fiber", "OM": "Fiber",
    "FIU": "Fiber", "SC2": "Fiber",
    "REGENIN": "Fiber", "REGENOUT": "Fiber",
    "XCON": "XCON",
    "LINE": "Line", "TRIBUTARY": "Line",
}


def _triage_category(user_prompt: str) -> tuple[str, dict]:
    """Extract board category from peak_board in prompt.

    Returns (category, features) where category is one of
    'Fiber', 'XCON', 'Line', or 'unknown'.
    """
    m = _PEAK_BOARD_RE.search(user_prompt)
    if not m:
        return "unknown", {}
    peak_board = m.group(1)
    info = _parse_board_name(peak_board)
    category = CATEGORY_FROM_BOARD_TYPE.get(info["board_type"], "unknown")
    return category, {"peak_board": peak_board, "board_type": info["board_type"]}


# ---------------------------------------------------------------------------
# Specialist prompt templates
# ---------------------------------------------------------------------------

_FIBER_SPECIALIST_TEMPLATE = """\
You are an OTN Root Cause Analysis Agent specialising in FIBER failures.

The peak board is optical (OA/OD/OM/FIU/SC2/REGEN), so the failure MUST be \
one of these three events:
- **Fiber_Aging** — gradual degradation (pattern_kind=ramp, bbe_max~28)
- **Fiber_Crack** — sudden damage with partial recovery (pattern_kind=step_recovery, bbe_max~42, bber_bbe_ratio>4)
- **Fiber_Cut** — complete severing (pattern_kind=step, bbe_max~55)

Do NOT consider XCON or Line events. Focus on discriminating the three Fiber subtypes.

Key discrimination features:
1. pattern_kind: ramp→Aging, step→Cut, step_recovery→Crack
2. bbe_max: ~28→Aging, ~42→Crack, ~55→Cut
3. bber_bbe_ratio: >4 strongly suggests Crack
4. es_frac: >0.5 suggests Cut or Crack (severe)

You have a maximum of {max_steps} tool calls. Plan efficiently.

Available tools:
{tools_description}

To use a tool, output a line in this exact format:
ACTION: tool_name(param1="value1", param2="value2")

Recommended investigation plan:
1. get_metrics → identify pattern_kind, bbe_max, bber_bbe_ratio
2. query_topology → confirm board type is optical
3. lookup_rule or search_similar → confirm Fiber event hypothesis
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

_XCON_SPECIALIST_TEMPLATE = """\
You are an OTN Root Cause Analysis Agent specialising in XCON (cross-connect) failures.

The peak board is XCON, so the failure MUST be one of these three events:
- **XCON_Port_Down** — port failure (pattern_kind=step, bbe_max~32)
- **XCON_Buffer_Overflow** — buffer saturation (pattern_kind=burst, bbe_max~46, burst_regions>=5)
- **XCON_Fabric_Fault** — switching fabric issue (pattern_kind=step_recovery, bbe_max~50)

Do NOT confuse with Line_Disconnect. The peak board is XCON, not Line/Tributary.

Key discrimination features:
1. pattern_kind: step→Port_Down, burst→Buffer_Overflow, step_recovery→Fabric_Fault
2. bbe_max: ~32→Port_Down, ~46→Buffer_Overflow, ~50→Fabric_Fault
3. burst_regions: >=5 strongly suggests Buffer_Overflow
4. longest_burst_frac: <0.4 with high burst_regions → Buffer_Overflow

You have a maximum of {max_steps} tool calls. Plan efficiently.

Available tools:
{tools_description}

To use a tool, output a line in this exact format:
ACTION: tool_name(param1="value1", param2="value2")

Recommended investigation plan:
1. get_metrics → identify pattern_kind, bbe_max, burst_regions
2. query_topology → confirm board type is XCON
3. lookup_rule or search_similar → confirm XCON event hypothesis
4. validate_propagation → verify alarm flow

After 4-5 tool calls, you should have enough evidence. You MUST then produce \
your final answer. Do NOT keep calling tools indefinitely.

When you have enough evidence, output your answer in EXACTLY this format:

ANSWER:
Root cause (predicted):
- Board: ROADM0-XCON051$0
- Event: XCON_Port_Down
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

_LINE_SPECIALIST_TEMPLATE = """\
You are an OTN Root Cause Analysis Agent specialising in LINE/TRIBUTARY failures.

The peak board is Line or Tributary, so the failure is **Line_Disconnect** — \
the only event that originates from Line/Tributary boards.

Your task: confirm this is Line_Disconnect and generate the correct alarm flow.

Key features of Line_Disconnect:
- pattern_kind: step_recovery (bbe_max~36, with partial recovery)
- Board type: LINE or TRIBUTARY
- Propagates ODUk_PM_AIS to downstream boards

You have a maximum of {max_steps} tool calls. Plan efficiently.

Available tools:
{tools_description}

To use a tool, output a line in this exact format:
ACTION: tool_name(param1="value1", param2="value2")

Recommended investigation plan:
1. get_metrics → confirm pattern_kind=step_recovery, bbe_max~36
2. query_topology → confirm board type is Line/Tributary
3. validate_propagation → generate alarm flow for Line_Disconnect

After 3-4 tool calls, you should have enough evidence. You MUST then produce \
your final answer. Do NOT keep calling tools indefinitely.

When you have enough evidence, output your answer in EXACTLY this format:

ANSWER:
Root cause (predicted):
- Board: ROADM0-Line001$0
- Event: Line_Disconnect
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

_SPECIALIST_TEMPLATES = {
    "Fiber": _FIBER_SPECIALIST_TEMPLATE,
    "XCON": _XCON_SPECIALIST_TEMPLATE,
    "Line": _LINE_SPECIALIST_TEMPLATE,
}


def _build_specialist_prompt(category: str, max_steps: int = 10) -> str:
    """Build a category-specific specialist prompt."""
    template = _SPECIALIST_TEMPLATES.get(category)
    if template is None:
        return _build_system_prompt(max_steps=max_steps)
    tools_desc = format_tools_for_prompt()
    return template.format(
        tools_description=tools_desc,
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Matches: ACTION: tool_name(arg1="val1", arg2="val2")
# Uses a greedy match up to the LAST closing paren on the line to handle
# nested parens in tuple arguments like bbe_range=(50, 60).
_ACTION_RE = re.compile(r"ACTION:\s*(\w+)\((.*)\)\s*$", re.MULTILINE)

# Matches: ANSWER: (everything after)
_ANSWER_RE = re.compile(r"ANSWER:\s*(.*)", re.DOTALL)

# Matches key="value" or key=value or key=(value, value)
_KWARG_RE = re.compile(
    r"""(\w+)\s*=\s*(?:"""
    r""""([^"]*?)"|"""  # double-quoted string
    r"""'([^']*?)'|"""  # single-quoted string
    r"""\(([^)]*?)\)|"""  # tuple
    r"""(\[[^\]]*?\])|"""  # list
    r"""([\w.\-\$/]+)"""  # bare value (number, identifier, board name with $)
    r""")""",
    re.DOTALL,
)


def _parse_response(response: str) -> tuple[str, str | None]:
    """Parse an LLM response into (thought, action_string | None).

    Returns the full thought text and, if present, the ACTION: line as a
    raw string.  Returns ``(thought, None)`` when the response contains
    no ACTION line (i.e. the agent wants to produce its final answer).
    """
    action_match = _ACTION_RE.search(response)

    if action_match:
        # Everything before the ACTION line is the thought
        thought = response[: action_match.start()].strip()
        action_str = action_match.group(0)
        return thought, action_str

    # No action — the whole response is thought (possibly containing ANSWER)
    return response.strip(), None


def _parse_action(action_str: str) -> tuple[str, dict[str, Any]]:
    """Parse an ACTION string into (tool_name, kwargs).

    Example input:  ACTION: get_metrics(board_name="ROADM0-OA002$0")
    Example output: ("get_metrics", {"board_name": "ROADM0-OA002$0"})
    """
    m = _ACTION_RE.search(action_str)
    if not m:
        raise ValueError(f"Could not parse action: {action_str!r}")

    tool_name = m.group(1)
    raw_args = m.group(2).strip()

    kwargs: dict[str, Any] = {}
    if not raw_args:
        return tool_name, kwargs

    for km in _KWARG_RE.finditer(raw_args):
        key = km.group(1)
        # Groups: 2=double-quoted, 3=single-quoted, 4=tuple, 5=list, 6=bare
        if km.group(2) is not None:
            kwargs[key] = km.group(2)
        elif km.group(3) is not None:
            kwargs[key] = km.group(3)
        elif km.group(4) is not None:
            # Parse tuple: e.g. "50, 60" or "50.0, 60.0"
            parts = [p.strip() for p in km.group(4).split(",")]
            parsed = []
            for p in parts:
                try:
                    parsed.append(float(p))
                except ValueError:
                    parsed.append(p)
            kwargs[key] = tuple(parsed)
        elif km.group(5) is not None:
            # Parse list via ast.literal_eval
            try:
                kwargs[key] = ast.literal_eval(km.group(5))
            except (ValueError, SyntaxError):
                kwargs[key] = km.group(5)
        elif km.group(6) is not None:
            val = km.group(6)
            # Try numeric conversion
            try:
                kwargs[key] = int(val)
            except ValueError:
                try:
                    kwargs[key] = float(val)
                except ValueError:
                    kwargs[key] = val

    return tool_name, kwargs


def _extract_answer(response: str) -> str:
    """Extract the final answer text from a response.

    Looks for an explicit ``ANSWER:`` marker first.  Falls back to
    returning the entire response if no marker is found (the model may
    just produce the answer directly).
    """
    m = _ANSWER_RE.search(response)
    if m:
        return m.group(1).strip()
    return response.strip()


def _extract_predicted_fields(answer: str) -> tuple[str, str]:
    """Extract predicted_event and predicted_board from the final answer.

    Tries multiple patterns to handle varied model output formats:
    - "- Board: ROADM1-XCON051$0"  (structured ANSWER format)
    - "**Board**: ROADM1-XCON051$0" (markdown bold)
    - "Board: ROADM1-XCON051$0"    (plain label)
    - "peak_board=ROADM1-XCON051$0" (inline feature reference)
    - "root cause board is ROADM1-XCON051$0" (natural language)
    """
    board = ""
    event = ""

    # Known event names for fallback matching
    KNOWN_EVENTS = [
        "XCON_Port_Down", "XCON_Fabric_Fault", "XCON_Buffer_Overflow",
        "Fiber_Aging", "Fiber_Crack", "Fiber_Cut", "Line_Disconnect",
    ]

    # Map partial/short event names to canonical names
    EVENT_ALIASES = {
        "Port_Down": "XCON_Port_Down",
        "Fabric_Fault": "XCON_Fabric_Fault",
        "Buffer_Overflow": "XCON_Buffer_Overflow",
        "Fiber_aging": "Fiber_Aging",
        "Fiber_crack": "Fiber_Crack",
        "Fiber_cut": "Fiber_Cut",
        "Line_disconnect": "Line_Disconnect",
        "line_disconnect": "Line_Disconnect",
    }

    # Board patterns (most specific first)
    board_patterns = [
        r"-\s*Board:\s*(\S+)",                          # - Board: X
        r"\*{1,2}Board\*{1,2}\s*:\s*(\S+)",             # **Board**: X (markdown bold from training)
        r"Board\s*:\s*(ROADM\d+-\w+\$\d+)",             # Board: ROADM...$0
        r"peak[_ ]board\s*(?:=|is)\s*[`\"']?(\S+?)(?:[`\"']|\s|$)",  # peak_board=X
        r"root\s+(?:cause\s+)?board\s+(?:is|:)\s*[`\"']?(\S+?)(?:[`\"']|\s|$)",  # root cause board is X
        r"Root cause.*?board.*?(ROADM\d+-\w+\$\d+)",    # "Root cause board: X" (natural language)
        r"(?:diagnosed|identified|determined).*?(ROADM\d+-\w+\$\d+)",  # "diagnosed ... ROADM0-X"
    ]
    for pat in board_patterns:
        m = re.search(pat, answer, re.IGNORECASE)
        if m:
            board = m.group(1).strip().rstrip(".,;:")
            break

    # Event patterns (most specific first)
    event_patterns = [
        r"-\s*Event:\s*(\S+)",                           # - Event: X
        r"\*{1,2}Event\*{1,2}\s*:\s*(\S+)",              # **Event**: X (markdown bold from training)
        r"Event\s*:\s*(\w+_\w+)",                        # Event: Fiber_Cut
        r"root\s+(?:cause\s+)?(?:event\s+)?(?:is|:)\s*[`\"']?(\S+?)(?:[`\"']|\s|$)",
        r"(?:diagnosis|conclusion|root\s+cause).*?(?:is|:)\s*[`\"']?(\w+_\w+)",  # "diagnosis is Fiber_Cut"
        r"(?:diagnosed|identified|determined).*?(\w+_\w+)",  # "diagnosed as Fiber_Cut"
    ]
    for pat in event_patterns:
        m = re.search(pat, answer, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip(".,;:`\"'")
            # Validate it looks like a known event
            if candidate in KNOWN_EVENTS or "_" in candidate:
                event = candidate
                break

    # Normalize partial event names (e.g. "Port_Down" -> "XCON_Port_Down")
    if event and event not in KNOWN_EVENTS:
        if event in EVENT_ALIASES:
            event = EVENT_ALIASES[event]
        else:
            # Try case-insensitive alias match
            for alias, canonical in EVENT_ALIASES.items():
                if event.lower() == alias.lower():
                    event = canonical
                    break

    # Fallback: scan for known event names anywhere in the answer
    if not event:
        for ev in KNOWN_EVENTS:
            if ev in answer:
                event = ev
                break

    # Fallback: scan for partial event names anywhere in the answer
    if not event:
        for alias, canonical in EVENT_ALIASES.items():
            if alias in answer:
                event = canonical
                break

    # Fallback: scan for ROADM board names if still missing — search full text
    # including CSV rows and natural language
    if not board:
        # Find all ROADM mentions and pick the first one
        all_roadm = re.findall(r"(ROADM\d+-\w+\$\d+)", answer)
        if all_roadm:
            board = all_roadm[0]

    return event, board


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _execute_tool(
    tool_name: str,
    kwargs: dict[str, Any],
    *,
    topology_yaml: str | None = None,
    lightpaths_file: str | None = None,
    rules_path: str | None = None,
    data_dir: str | None = None,
    training_data: str | None = None,
) -> str:
    """Execute a tool by name with given kwargs.

    Automatically injects file-path arguments that the agent should not
    need to guess (topology, lightpaths, rules, data_dir, training_data).
    Returns a JSON-formatted string of the tool result.
    """
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {tool_name}"}, indent=2)

    # Force-inject file paths — always override what the model passes,
    # because the model doesn't know the actual paths on the server.
    if tool_name == "query_topology":
        if topology_yaml:
            kwargs["topology_yaml"] = topology_yaml
        if lightpaths_file:
            kwargs["lightpaths_file"] = lightpaths_file
    elif tool_name == "get_metrics":
        if data_dir:
            kwargs["data_dir"] = data_dir
    elif tool_name == "lookup_rule":
        if rules_path:
            kwargs["rules_path"] = rules_path
    elif tool_name == "validate_propagation":
        if rules_path:
            kwargs["rules_path"] = rules_path
        if lightpaths_file:
            kwargs["lightpaths_file"] = lightpaths_file
    elif tool_name == "search_similar":
        if training_data:
            kwargs["training_data_path"] = training_data

    func = TOOL_REGISTRY[tool_name]["function"]

    try:
        result = func(**kwargs)
    except Exception as exc:
        result = {"error": f"Tool {tool_name} raised {type(exc).__name__}: {exc}"}

    # Serialise to JSON string for the observation
    try:
        return json.dumps(result, indent=2, default=str)
    except (TypeError, ValueError):
        return str(result)


# ---------------------------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------------------------


class _LocalBackend:
    """Local transformers + peft backend.  Lazy-loads model on first call."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str | None,
        temperature: float,
        max_tokens: int,
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None
        self._total_tokens = 0

    def _load(self) -> None:
        """Lazy-load model and tokenizer."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local backend requires 'torch' and 'transformers'. "
                "Install them with: pip install torch transformers"
            ) from exc

        if not torch.cuda.is_available():
            print(
                "[warn] CUDA is not available. Local backend will run on CPU "
                "and may be extremely slow.",
                file=sys.stderr,
            )

        device_map = "auto" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print(f"[info] Loading base model: {self.model_name}", file=sys.stderr)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=True
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=device_map
        )

        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "Adapter loading requires 'peft'. Install with: pip install peft"
                ) from exc
            print(f"[info] Loading adapter: {self.adapter_path}", file=sys.stderr)
            model = PeftModel.from_pretrained(model, self.adapter_path)

        model.eval()
        self._model = model
        print("[info] Model loaded.", file=sys.stderr)

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a completion from the message history."""
        import torch

        if self._model is None:
            self._load()

        assert self._tokenizer is not None
        assert self._model is not None

        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(input_text, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                top_p=0.9 if self.temperature > 0 else 1.0,
            )

        gen_len = out.shape[1] - prompt_len
        self._total_tokens += out.shape[1]

        response = self._tokenizer.decode(
            out[0][prompt_len:], skip_special_tokens=True
        )
        return response

    @property
    def total_tokens(self) -> int:
        return self._total_tokens


class _APIBackend:
    """OpenAI-compatible API backend."""

    def __init__(
        self,
        api_url: str,
        api_key: str | None,
        api_model: str | None,
        temperature: float,
        max_tokens: int,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key or ""
        self.api_model = api_model or "default"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._total_tokens = 0

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a completion via the API."""
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "API backend requires 'requests'. Install with: pip install requests"
            ) from exc

        endpoint = f"{self.api_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.api_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            resp = requests.post(
                endpoint, headers=headers, json=payload, timeout=300
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc

        data = resp.json()

        # Track token usage
        usage = data.get("usage", {})
        self._total_tokens += usage.get("total_tokens", 0)

        # Extract response content
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"API returned no choices: {data}")

        content = choices[0].get("message", {}).get("content", "")
        return content

    @property
    def total_tokens(self) -> int:
        return self._total_tokens


class _ClaudeBackend:
    """Claude API backend using native Anthropic SDK.

    Uses the Anthropic API directly for higher-quality reasoning.
    Set ANTHROPIC_API_KEY environment variable before use.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._total_tokens = 0

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a completion via Claude API."""
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Claude backend requires 'anthropic'. "
                "Install with: pip install anthropic"
            ) from exc

        client = anthropic.Anthropic()

        # Separate system message from conversation
        system_text = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                api_messages.append(msg)

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_text,
            messages=api_messages,
            temperature=self.temperature,
        )

        # Track tokens
        self._total_tokens += response.usage.input_tokens + response.usage.output_tokens

        # Extract text content
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        return "\n".join(text_parts)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens


# ---------------------------------------------------------------------------
# RCAAgent
# ---------------------------------------------------------------------------


class RCAAgent:
    """ReAct-style agent for OTN root cause analysis.

    Parameters
    ----------
    backend : str
        ``"local"`` for transformers/peft or ``"api"`` for OpenAI-compatible.
    model_name : str
        HuggingFace model ID (local backend).
    adapter_path : str | None
        Optional LoRA adapter path (local backend).
    api_url : str | None
        API base URL, e.g. ``http://localhost:11434/v1`` (api backend).
    api_key : str | None
        API key (api backend, optional).
    api_model : str | None
        Model name to send to the API (api backend).
    topology_yaml : str
        Path to topology YAML file.
    lightpaths_file : str
        Path to lightpaths.txt.
    rules_path : str
        Path to rule_database.csv.
    data_dir : str | None
        Path to timeseries data directory (Node_*/timeseries_*.csv).
    training_data : str
        Path to training JSONL for search_similar tool.
    max_steps : int
        Maximum ReAct loop iterations.
    temperature : float
        LLM sampling temperature.
    max_tokens : int
        Maximum tokens per LLM generation.
    """

    def __init__(
        self,
        backend: str = "local",
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        adapter_path: str | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        api_model: str | None = None,
        topology_yaml: str = "outputs/noregen_runs/topology_noregen.yaml",
        lightpaths_file: str = "regen_compare/noregen/topology/lightpaths.txt",
        rules_path: str = "outputs/Static files/rule_database.csv",
        data_dir: str | None = None,
        training_data: str = "outputs/grpo_data/train.jsonl",
        max_steps: int = 10,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        use_triage: bool = True,
    ):
        self.backend_name = backend
        self.topology_yaml = topology_yaml
        self.lightpaths_file = lightpaths_file
        self.rules_path = rules_path
        self.data_dir = data_dir
        self.training_data = training_data
        self.max_steps = max_steps
        self.use_triage = use_triage

        self._default_system_prompt = _build_system_prompt(max_steps=max_steps)

        if backend == "local":
            self._backend = _LocalBackend(
                model_name=model_name,
                adapter_path=adapter_path,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif backend == "api":
            if not api_url:
                raise ValueError("--api-url is required for the api backend.")
            self._backend = _APIBackend(
                api_url=api_url,
                api_key=api_key,
                api_model=api_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif backend == "claude":
            self._backend = _ClaudeBackend(
                model=api_model or "claude-sonnet-4-20250514",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'local', 'api', or 'claude'.")

    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Generate next LLM response."""
        return self._backend.generate(messages)

    def _execute(self, action_str: str) -> str:
        """Parse and execute a tool call, returning the observation string."""
        try:
            tool_name, kwargs = _parse_action(action_str)
        except ValueError as exc:
            return json.dumps({"error": str(exc)}, indent=2)

        return _execute_tool(
            tool_name,
            kwargs,
            topology_yaml=self.topology_yaml,
            lightpaths_file=self.lightpaths_file,
            rules_path=self.rules_path,
            data_dir=self.data_dir,
            training_data=self.training_data,
        )

    def run(self, user_prompt: str, verbose: bool = False) -> AgentResult:
        """Run the ReAct loop on a user prompt.

        Parameters
        ----------
        user_prompt : str
            The user message (same format as what the single-shot model gets).
        verbose : bool
            If True, print each step as it executes.

        Returns
        -------
        AgentResult with steps, final answer, token count, etc.
        """
        # Select system prompt: specialist (if triage enabled) or generic
        self._last_triage_category = "unknown"
        if self.use_triage:
            category, features = _triage_category(user_prompt)
            self._last_triage_category = category
            if category != "unknown":
                system_prompt = _build_specialist_prompt(category, self.max_steps)
                if verbose:
                    print(
                        f"[triage] category={category}, "
                        f"board_type={features.get('board_type', '?')}, "
                        f"peak_board={features.get('peak_board', '?')}",
                        file=sys.stderr,
                    )
            else:
                system_prompt = self._default_system_prompt
                if verbose:
                    print("[triage] category=unknown, using generic prompt", file=sys.stderr)
        else:
            system_prompt = self._default_system_prompt

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        steps: list[AgentStep] = []
        final_answer = ""
        num_tool_calls = 0

        for step_num in range(self.max_steps):
            if verbose:
                print(f"\n{'='*60}", file=sys.stderr)
                print(f"STEP {step_num + 1}/{self.max_steps}", file=sys.stderr)
                print(f"{'='*60}", file=sys.stderr)

            # Generate next response
            t0 = time.time()
            response = self._generate(messages)
            elapsed = time.time() - t0

            if verbose:
                print(f"[{elapsed:.1f}s] LLM response:", file=sys.stderr)
                # Show a truncated preview
                preview = response[:500]
                if len(response) > 500:
                    preview += f"\n... ({len(response)} chars total)"
                print(preview, file=sys.stderr)

            # Parse response for ACTION or ANSWER
            thought, action_str = _parse_response(response)

            if action_str is None:
                # No more actions — extract final answer
                final_answer = _extract_answer(response)
                steps.append(AgentStep(
                    step_num=step_num,
                    thought=thought,
                    action=None,
                    observation=None,
                ))

                if verbose:
                    print("\n[FINAL ANSWER]", file=sys.stderr)
                    print(final_answer[:1000], file=sys.stderr)
                break

            # Execute the tool
            if verbose:
                print(f"\n[ACTION] {action_str}", file=sys.stderr)

            observation = self._execute(action_str)
            num_tool_calls += 1

            if verbose:
                obs_preview = observation[:400]
                if len(observation) > 400:
                    obs_preview += f"\n... ({len(observation)} chars total)"
                print(f"[OBSERVATION] {obs_preview}", file=sys.stderr)

            steps.append(AgentStep(
                step_num=step_num,
                thought=thought,
                action=action_str,
                observation=observation,
            ))

            # Append to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"OBSERVATION:\n{observation}",
            })
        else:
            # Exhausted max_steps without producing a final answer.
            if verbose:
                print(
                    f"\n[WARN] Reached max_steps ({self.max_steps}) without "
                    "ANSWER marker. Forcing final answer.",
                    file=sys.stderr,
                )
            if steps and steps[-1].observation:
                # Tell the model it MUST produce ANSWER now — no more tools
                messages.append({
                    "role": "user",
                    "content": (
                        "You have used all available tool steps. You MUST now provide "
                        "your final diagnosis. Do NOT call any more tools. Output your "
                        "ANSWER with the root cause board and event based on the "
                        "evidence you have gathered so far.\n\n"
                        "ANSWER:\nRoot cause (predicted):\n"
                        "- Board: <board_name>\n- Event: <event_type>"
                    ),
                })
                final_response = self._generate(messages)
                final_answer = _extract_answer(final_response)
                steps.append(AgentStep(
                    step_num=self.max_steps,
                    thought=final_response.strip(),
                    action=None,
                    observation=None,
                ))
            elif not final_answer:
                # Fall back to the last thought
                final_answer = steps[-1].thought if steps else ""

        predicted_event, predicted_board = _extract_predicted_fields(final_answer)

        # Last-resort fallback: if extraction failed, scan all step thoughts
        # for board/event mentions (handles ACTION-only final outputs)
        if not predicted_event or not predicted_board:
            all_text = final_answer
            for s in steps:
                if s.thought:
                    all_text += "\n" + s.thought
            if not predicted_event or not predicted_board:
                fb_event, fb_board = _extract_predicted_fields(all_text)
                if not predicted_event and fb_event:
                    predicted_event = fb_event
                if not predicted_board and fb_board:
                    predicted_board = fb_board

        return AgentResult(
            steps=steps,
            final_answer=final_answer,
            total_tokens=self._backend.total_tokens,
            num_tool_calls=num_tool_calls,
            predicted_event=predicted_event,
            predicted_board=predicted_board,
        )


# ---------------------------------------------------------------------------
# Helpers for loading test data
# ---------------------------------------------------------------------------


def _load_test_example(path: str, index: int) -> dict[str, Any]:
    """Load a single example from a JSONL file by index."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line.strip())
    raise IndexError(f"Sample index {index} out of range in {path}")


def _count_examples(path: str) -> int:
    """Count lines in a JSONL file."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _extract_user_prompt(example: dict[str, Any]) -> str:
    """Extract the user prompt text from a GRPO/SFT example.

    Handles both formats:
    - GRPO: example["prompt"] is a list of message dicts
    - SFT:  example["messages"] is a list of message dicts
    """
    messages = example.get("prompt") or example.get("messages") or []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg["content"]
    # Fallback: concatenate all content
    parts = []
    for msg in messages:
        if isinstance(msg, dict):
            parts.append(msg.get("content", ""))
    return "\n".join(parts)


def _extract_data_dir_from_prompt(
    example: dict[str, Any], runs_dir: str | None = None
) -> str | None:
    """Try to infer the data_dir for an example.

    The GRPO pipeline stores timeseries data alongside the run directory.
    We attempt to find it from the prompt metadata or the runs directory.
    """
    if runs_dir is None:
        return None

    # The ground_truth fields can hint at the run directory
    gt_event = example.get("ground_truth_root_event", "")
    if gt_event:
        cand = Path(runs_dir) / f"data_outputs_{gt_event.lower()}"
        if cand.exists():
            return str(cand)

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ReAct-style agent for OTN root cause analysis."
    )

    # Data source
    ap.add_argument(
        "--test-file",
        default="outputs/grpo_data/train.jsonl",
        help="Path to JSONL file with test examples.",
    )
    ap.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Which example to run (0-indexed). Ignored if --run-all is set.",
    )
    ap.add_argument(
        "--run-all",
        action="store_true",
        help="Run on all examples in the test file.",
    )

    # Backend selection
    ap.add_argument(
        "--backend",
        choices=["local", "api", "claude"],
        default="api",
        help="LLM backend: 'local' (transformers), 'api' (OpenAI-compatible), 'claude' (Anthropic API).",
    )

    # Local backend options
    ap.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-14B-Instruct",
        help="HuggingFace model ID (local backend).",
    )
    ap.add_argument(
        "--adapter",
        default=None,
        help="Path to LoRA adapter (local backend, optional).",
    )

    # API backend options
    ap.add_argument(
        "--api-url",
        default=None,
        help="OpenAI-compatible API base URL (e.g. http://localhost:11434/v1).",
    )
    ap.add_argument(
        "--api-key",
        default=None,
        help="API key (optional).",
    )
    ap.add_argument(
        "--api-model",
        default=None,
        help="Model name to send to the API.",
    )

    # File paths
    ap.add_argument(
        "--topology-yaml",
        default="outputs/noregen_runs/topology_noregen.yaml",
        help="Path to topology YAML.",
    )
    ap.add_argument(
        "--lightpaths",
        default="regen_compare/noregen/topology/lightpaths.txt",
        help="Path to lightpaths file.",
    )
    ap.add_argument(
        "--rules",
        default="outputs/Static files/rule_database.csv",
        help="Path to rule_database.csv.",
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help="Path to timeseries data directory. If not set, attempts auto-detection.",
    )
    ap.add_argument(
        "--training-data",
        default="outputs/grpo_data/train.jsonl",
        help="Path to training JSONL for search_similar tool.",
    )

    # Agent options
    ap.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum ReAct loop iterations.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM sampling temperature.",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens per LLM generation.",
    )

    # Triage
    ap.add_argument(
        "--no-triage",
        action="store_true",
        help="Disable category-based triage (use generic prompt for all examples).",
    )

    # Output
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSONL path (for --run-all mode).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed step-by-step output.",
    )

    args = ap.parse_args()

    # Validate
    test_path = Path(args.test_file)
    if not test_path.exists():
        raise SystemExit(f"Test file not found: {args.test_file}")

    # Build agent
    agent = RCAAgent(
        backend=args.backend,
        model_name=args.base_model,
        adapter_path=args.adapter,
        api_url=args.api_url,
        api_key=args.api_key,
        api_model=args.api_model,
        topology_yaml=args.topology_yaml,
        lightpaths_file=args.lightpaths,
        rules_path=args.rules,
        data_dir=args.data_dir,
        training_data=args.training_data,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_triage=not args.no_triage,
    )

    if args.run_all:
        _run_all(agent, args)
    else:
        _run_single(agent, args)


def _run_single(agent: RCAAgent, args: argparse.Namespace) -> None:
    """Run the agent on a single example and print results."""
    example = _load_test_example(args.test_file, args.sample_index)

    gt_event = example.get("ground_truth_root_event", "?")
    gt_board = example.get("ground_truth_root_board", "?")

    print(f"[info] Sample {args.sample_index}: event={gt_event}, board={gt_board}")
    print("=" * 80)

    user_prompt = _extract_user_prompt(example)

    # Try to auto-detect data_dir if not provided
    if agent.data_dir is None:
        detected = _extract_data_dir_from_prompt(example)
        if detected:
            agent.data_dir = detected
            if args.verbose:
                print(f"[info] Auto-detected data_dir: {detected}", file=sys.stderr)

    t0 = time.time()
    result = agent.run(user_prompt, verbose=args.verbose)
    elapsed = time.time() - t0

    # Print results
    print("\n" + "=" * 80)
    print("AGENT RESULT")
    print("=" * 80)
    print(f"Steps taken: {len(result.steps)}")
    print(f"Tool calls:  {result.num_tool_calls}")
    print(f"Tokens:      {result.total_tokens}")
    print(f"Time:        {elapsed:.1f}s")
    print(f"Predicted:   event={result.predicted_event}, board={result.predicted_board}")
    print()
    print("FINAL ANSWER:")
    print("-" * 40)
    print(result.final_answer)

    # Print ground truth for comparison
    print("\n" + "=" * 80)
    print("GROUND TRUTH:")
    print("=" * 80)
    print(f"Root Event: {gt_event}")
    print(f"Root Board: {gt_board}")

    gt_flow = example.get("ground_truth_alarm_flow_csv", "")
    if gt_flow:
        lines = gt_flow.strip().splitlines()
        preview = "\n".join(lines[:6])
        if len(lines) > 6:
            preview += f"\n... ({len(lines) - 6} more rows)"
        print(f"Alarm Flow:\n{preview}")

    # Accuracy check
    event_match = result.predicted_event == gt_event
    board_match = result.predicted_board == gt_board
    triage_cat = getattr(agent, "_last_triage_category", "unknown")
    print(f"\nTriage category: {triage_cat}")
    print(f"Event match: {'YES' if event_match else 'NO'}")
    print(f"Board match: {'YES' if board_match else 'NO'}")


def _run_all(agent: RCAAgent, args: argparse.Namespace) -> None:
    """Run the agent on all examples and save outputs to JSONL."""
    total = _count_examples(args.test_file)
    print(f"[info] Running agent on {total} examples from {args.test_file}")

    out_path = Path(args.out) if args.out else Path("outputs/eval/agent_outputs.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    correct_event = 0
    correct_board = 0

    for idx in range(total):
        example = _load_test_example(args.test_file, idx)
        gt_event = example.get("ground_truth_root_event", "?")
        gt_board = example.get("ground_truth_root_board", "?")
        gt_boards = example.get("ground_truth_boards", "[]")
        gt_alarms = example.get("ground_truth_alarms", "{}")
        gt_severity = example.get("ground_truth_severity_dist", "{}")
        gt_row_count = example.get("ground_truth_row_count", 0)

        print(f"\n[{idx + 1}/{total}] event={gt_event}, board={gt_board}")

        user_prompt = _extract_user_prompt(example)

        # Try to auto-detect data_dir per example
        original_data_dir = agent.data_dir
        if agent.data_dir is None:
            detected = _extract_data_dir_from_prompt(example)
            if detected:
                agent.data_dir = detected

        t0 = time.time()
        try:
            result = agent.run(user_prompt, verbose=args.verbose)
        except Exception as exc:
            print(f"  [ERROR] {exc}", file=sys.stderr)
            result = AgentResult(
                final_answer=f"ERROR: {exc}",
                predicted_event="",
                predicted_board="",
            )
        elapsed = time.time() - t0

        # Restore original data_dir for next iteration
        agent.data_dir = original_data_dir

        event_match = result.predicted_event == gt_event
        board_match = result.predicted_board == gt_board
        if event_match:
            correct_event += 1
        if board_match:
            correct_board += 1

        print(
            f"  predicted: event={result.predicted_event}, board={result.predicted_board} "
            f"| steps={len(result.steps)}, tools={result.num_tool_calls}, "
            f"time={elapsed:.1f}s "
            f"| event={'OK' if event_match else 'MISS'}, board={'OK' if board_match else 'MISS'}"
        )

        record = {
            "sample_index": idx,
            "model_output": result.final_answer,
            "ground_truth_root_event": gt_event,
            "ground_truth_root_board": gt_board,
            "ground_truth_boards": gt_boards,
            "ground_truth_alarms": gt_alarms,
            "ground_truth_severity_dist": gt_severity,
            "ground_truth_row_count": gt_row_count,
            "agent_steps": len(result.steps),
            "agent_tool_calls": result.num_tool_calls,
            "agent_total_tokens": result.total_tokens,
            "agent_elapsed_sec": round(elapsed, 1),
            "predicted_event": result.predicted_event,
            "predicted_board": result.predicted_board,
            "event_correct": event_match,
            "board_correct": board_match,
            "triage_category": getattr(agent, "_last_triage_category", "unknown"),
            "method": "agent",
        }
        results.append(record)

        # Save intermediate outputs every 10 examples
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            with out_path.open("w", encoding="utf-8") as f:
                for rec in results:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [saved] {len(results)} results -> {out_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total examples: {total}")
    print(f"Event accuracy: {correct_event}/{total} ({100*correct_event/max(total,1):.1f}%)")
    print(f"Board accuracy: {correct_board}/{total} ({100*correct_board/max(total,1):.1f}%)")
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    main()
