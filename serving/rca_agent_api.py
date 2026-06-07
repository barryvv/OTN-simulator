"""FastAPI serving wrapper around `rca_agent.RCAAgent` (the `rca-agent-api`).

This is the request-facing entrypoint of the dedicated serving cluster. It runs
the ReAct loop against the vLLM OpenAI-compatible endpoint (`--backend api`) and
returns the predicted root cause + alarm-flow CSV.

The five RCA tools are executed in-process by `RCAAgent` (CPU-cheap file I/O),
which is the simplest correct path. For scale-out, the same tools are also
deployable standalone via `serving/rca_tools_api.py`; this service does not
require that one to be running.

Run locally:
    uvicorn serving.rca_agent_api:app --host 0.0.0.0 --port 8080

Config (env vars):
    VLLM_API_URL    OpenAI-compatible base URL of the vLLM service
                    (e.g. http://vllm-rca-7b.serving.svc/v1)        [required]
    VLLM_MODEL      model/adapter name to request from vLLM
                    (a base model or a promoted LoRA module name)
    TOPOLOGY_YAML / LIGHTPATHS_FILE / RULES_PATH / DATA_DIR / TRAINING_DATA
                    server-side data paths (mounted from PVC / MinIO)
    MAX_STEPS       max ReAct iterations (default 10)
    TEMPERATURE     sampling temperature (default 0.3)
    MAX_TOKENS      max tokens per generation (default 1024)
    USE_TRIAGE      "1" to enable category triage prompts (default 1)

Endpoints:
    GET  /healthz   liveness
    GET  /readyz    readiness (agent constructed)
    POST /rca       run RCA: body = {"user_prompt": "...", "data_dir"?, "model"?}
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rca_agent import RCAAgent


# ── Config ──
VLLM_API_URL = os.getenv("VLLM_API_URL", "")
VLLM_MODEL = os.getenv("VLLM_MODEL", "qwen2.5-7b")
TOPOLOGY_YAML = os.getenv("TOPOLOGY_YAML", "topology.yaml")
LIGHTPATHS_FILE = os.getenv("LIGHTPATHS_FILE", "lightpaths.txt")
RULES_PATH = os.getenv("RULES_PATH", "outputs/Static files/rule_database.csv")
DATA_DIR = os.getenv("DATA_DIR") or None
TRAINING_DATA = os.getenv("TRAINING_DATA", "outputs/grpo_data/train.jsonl")
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
USE_TRIAGE = os.getenv("USE_TRIAGE", "1") == "1"


app = FastAPI(title="OTN RCA Agent", version="1.0.0")


class RCARequest(BaseModel):
    user_prompt: str
    # Per-request timeseries directory (each failure run has its own metrics).
    data_dir: str | None = None
    # Optional override of the vLLM model/adapter to route to (e.g. a canary
    # LoRA module name). Falls back to VLLM_MODEL.
    model: str | None = None
    max_steps: int | None = None


@lru_cache(maxsize=8)
def _get_agent(api_model: str, data_dir: str | None, max_steps: int) -> RCAAgent:
    """Cache one RCAAgent per (model, data_dir, max_steps). The agent is a thin
    stateless ReAct driver; the heavy LLM lives in the vLLM pod."""
    if not VLLM_API_URL:
        raise HTTPException(status_code=503, detail="VLLM_API_URL not configured")
    return RCAAgent(
        backend="api",
        api_url=VLLM_API_URL,
        api_model=api_model,
        topology_yaml=TOPOLOGY_YAML,
        lightpaths_file=LIGHTPATHS_FILE,
        rules_path=RULES_PATH,
        data_dir=data_dir,
        training_data=TRAINING_DATA,
        max_steps=max_steps,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        use_triage=USE_TRIAGE,
    )


# Extract the CSV alarm-flow block from the agent's free-text answer (same
# markers the evaluator looks for).
_CSV_MARKER = re.compile(r"(SourceBoard\s*,.*)", re.IGNORECASE | re.DOTALL)


def _extract_alarm_flow_csv(answer: str) -> str:
    m = _CSV_MARKER.search(answer)
    if not m:
        return ""
    block = m.group(1)
    # Trim trailing prose after the CSV (stop at a blank line followed by text).
    lines = []
    for ln in block.splitlines():
        if ln.strip() == "" and lines:
            break
        lines.append(ln)
    return "\n".join(lines).strip()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    return {"status": "ready", "vllm_api_url": bool(VLLM_API_URL), "model": VLLM_MODEL}


@app.post("/rca")
def run_rca(req: RCARequest) -> dict[str, Any]:
    agent = _get_agent(
        req.model or VLLM_MODEL,
        req.data_dir or DATA_DIR,
        req.max_steps or MAX_STEPS,
    )
    result = agent.run(req.user_prompt, verbose=False)
    return {
        "predicted_event": result.predicted_event,
        "predicted_board": result.predicted_board,
        "alarm_flow_csv": _extract_alarm_flow_csv(result.final_answer),
        "final_answer": result.final_answer,
        "num_tool_calls": result.num_tool_calls,
        "num_steps": len(result.steps),
        "total_tokens": result.total_tokens,
    }
