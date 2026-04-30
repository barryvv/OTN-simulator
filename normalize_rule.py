#!/usr/bin/env python3
import sys, pandas as pd
from pathlib import Path

ALIASES = {
    "FromType":  ["fromtype","from_type","srctype","sourcetype","from","src","functiontype","function_type"],
    "FromAlarm": ["fromalarm","from_alarm","srcalarm","sourcealarm","alarmfrom","fromalm","eventname","event_name","alarm"],
    "ToType":    ["totype","to_type","dsttype","desttype","destinationtype","to","dst","actiontarget","action_target","target"],
    "ToAlarm":   ["toalarm","to_alarm","dstalarm","destalarm","alarmto","toalm","propalarm","propagatedalarm","outputalarm","output_alarm"],
    "EdgeType":  ["edgetype","edge_type","etype","edge"],
    "DelayMs":   ["delayms","delay_ms","delay","latencyms","latency_ms"],
    "Severity":  ["severity","level","sev"]
}

EDGE_LUT = {
    ("Tributary","XCON"): "Tributary_XCON",
    ("XCON","Line"):      "XCON_Line",
    ("Line","OM"):        "Line_OM",
    ("OM","OA"):          "OM_OA",
    ("OA","OD"):          "OA_OD",
    ("OD","Line"):        "OD_Line",
    ("FIU","OA"):         "FIU_OA",
    ("SC2","FIU"):        "SC2_FIU",
    ("RegenIn","RegenOut"): "regen_boundary",
}
EXPAND = {
    "Electrical": {"Tributary","XCON","Line"},
    "Optical":    {"OM","OA","OD","FIU","SC2","RegenIn","RegenOut"},
    "Any":        {"Tributary","XCON","Line","OM","OA","OD","FIU","SC2","RegenIn","RegenOut"},
}

# FunctionType values that use mixed-case with internal capitals (e.g. OTUk-RegenIn).
# These must NOT be mangled by .str.title().
_PRESERVE_CASE_TYPES = {
    "otuk-regenin": "OTUk-RegenIn",
    "otuk-regenout": "OTUk-RegenOut",
    "otuk-relay": "OTUk-Relay",
    "oduk-creation": "ODUk-Creation",
    "oduk-termination": "ODUk-Termination",
}

def pick_col(df, logical):
    for cand in ALIASES[logical]:
        for c in df.columns:
            if c.strip().lower().replace("_","") == cand:
                return c
    return None

def main(inp="outputs/Static files/rule_database.csv",
         outp="outputs/Static files/rule_database_canonical.csv"):
    inp = Path(inp); outp = Path(outp)
    if not inp.exists():
        raise FileNotFoundError(f"Rules CSV not found: {inp}")

    df = pd.read_csv(inp)

    cols = {}
    for k in ["FromType","FromAlarm","ToType","ToAlarm","EdgeType","DelayMs","Severity"]:
        cols[k] = pick_col(df, k)

    if not cols["FromType"] and "FunctionType" in df.columns:
        cols["FromType"] = "FunctionType"
    if not cols["FromAlarm"] and "EventName" in df.columns:
        cols["FromAlarm"] = "EventName"
    if not cols["ToType"] and "ActionTarget" in df.columns:
        cols["ToType"] = "ActionTarget"
    if not cols["ToAlarm"] and "OutputAlarm" in df.columns:
        cols["ToAlarm"] = "OutputAlarm"

    needed = ["FromType","FromAlarm","ToType","ToAlarm"]
    missing = [k for k in needed if not cols[k]]
    if missing:
        raise ValueError(f"Cannot normalize rules: missing logical columns {missing} "
                         f"and could not infer from functional schema.")

    def _safe_title(s):
        """Apply .title() but preserve known OTN FunctionType casing."""
        stripped = str(s).strip()
        key = stripped.lower()
        if key in _PRESERVE_CASE_TYPES:
            return _PRESERVE_CASE_TYPES[key]
        return stripped.title() if stripped else stripped

    out = pd.DataFrame({
        "FromType":  df[cols["FromType"]].astype(str).str.strip().apply(_safe_title),
        "FromAlarm": df[cols["FromAlarm"]].astype(str).str.strip(),
        "ToType":    df[cols["ToType"]].astype(str).str.strip().apply(_safe_title),
        "ToAlarm":   df[cols["ToAlarm"]].astype(str).str.strip(),
    })

    def expand_types(s):
        t = s.title()
        return sorted(EXPAND.get(t, {t}))
    out = out.explode("FromType").copy() if False else out

    if cols["EdgeType"]:
        out["EdgeType"] = df[cols["EdgeType"]].astype(str).str.strip()
    else:
        def infer_edge(ft, tt):
            ft, tt = ft.title(), tt.title()
            if ft in EXPAND:
                if tt in {"Xcon"}:     return "XCON_Line"
                if tt in {"Om"}:       return "Line_OM"
                if tt in {"Tributary"}:return "Tributary_XCON"
            return EDGE_LUT.get((ft, tt), f"{ft}_{tt}")
        out["EdgeType"] = [infer_edge(ft, tt) for ft, tt in zip(out["FromType"], out["ToType"])]

    out["DelayMs"] = 500
    if cols["DelayMs"]:
        out["DelayMs"] = pd.to_numeric(df[cols["DelayMs"]], errors="coerce").fillna(500).astype(int)
    out["Severity"] = "Major"
    if cols["Severity"]:
        out["Severity"] = df[cols["Severity"]].astype(str).str.title().replace({"": "Major"})

    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)
    print(f"[Rules] Canonicalized {len(out)} rows → {outp}")
    print("[Rules] Sample EdgeTypes:", sorted(out['EdgeType'].unique())[:8])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/Static files/rule_database.csv")
    ap.add_argument("--out", dest="outp", default="outputs/Static files/rule_database_canonical.csv")
    args = ap.parse_args()
    main(args.inp, args.outp)