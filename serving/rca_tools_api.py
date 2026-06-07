"""HTTP wrapper around the RCA tool registry (`rca_tools.TOOL_REGISTRY`).

This is the `rca-tools-svc` from the platform design: a stateless CPU
microservice that exposes the five RCA tools (query_topology, get_metrics,
lookup_rule, validate_propagation, search_similar) over HTTP so they can be
scaled and deployed independently of the LLM serving pod.

It does NOT modify `rca_tools.py` — it imports the registry and the same
path-injection logic the agent uses, so a remote tool call behaves identically
to an in-process one.

Run locally:
    uvicorn serving.rca_tools_api:app --host 0.0.0.0 --port 8081

Config (env vars — server-side file paths, never trusted from the caller):
    TOPOLOGY_YAML   path to topology YAML
    LIGHTPATHS_FILE path to lightpaths.txt / mw_lightpaths_roles.txt
    RULES_PATH      path to rule_database.csv
    DATA_DIR        directory with Node_*/timeseries_*.csv
    TRAINING_DATA   path to training JSONL (for search_similar)

Endpoints:
    GET  /healthz                 liveness
    GET  /tools                   list tool names + descriptions + params
    POST /tools/{tool_name}       execute a tool: body = {"args": {...}}
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Reuse the exact registry and the agent's path-injection rules so a remote
# call is byte-for-byte equivalent to the in-process path.
from rca_tools import TOOL_REGISTRY
from rca_agent import _execute_tool


# ── Server-side paths (env-injected; callers never supply file paths) ──
TOPOLOGY_YAML = os.getenv("TOPOLOGY_YAML", "topology.yaml")
LIGHTPATHS_FILE = os.getenv("LIGHTPATHS_FILE", "lightpaths.txt")
RULES_PATH = os.getenv("RULES_PATH", "outputs/Static files/rule_database.csv")
DATA_DIR = os.getenv("DATA_DIR") or None
TRAINING_DATA = os.getenv("TRAINING_DATA", "outputs/grpo_data/train.jsonl")


app = FastAPI(title="RCA Tools Service", version="1.0.0")


class ToolCall(BaseModel):
    args: dict[str, Any] = {}
    # Optional per-request override of the timeseries dir (different runs /
    # tenants point at different data). Other paths stay server-controlled.
    data_dir: str | None = None


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tools")
def list_tools() -> dict[str, Any]:
    return {
        name: {
            "description": meta["description"],
            "parameters": meta["parameters"],
        }
        for name, meta in TOOL_REGISTRY.items()
    }


@app.post("/tools/{tool_name}")
def run_tool(tool_name: str, call: ToolCall) -> dict[str, Any]:
    if tool_name not in TOOL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"unknown tool: {tool_name}")

    observation = _execute_tool(
        tool_name,
        dict(call.args),
        topology_yaml=TOPOLOGY_YAML,
        lightpaths_file=LIGHTPATHS_FILE,
        rules_path=RULES_PATH,
        data_dir=call.data_dir or DATA_DIR,
        training_data=TRAINING_DATA,
    )
    # `_execute_tool` returns a JSON string (or plain string on failure).
    try:
        return {"tool": tool_name, "result": json.loads(observation)}
    except (TypeError, ValueError):
        return {"tool": tool_name, "result": observation}
