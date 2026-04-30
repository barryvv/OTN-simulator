#!/usr/bin/env python3
import sys, argparse
import pandas as pd

KNOWN_TTYPES = {
    "Tributary_XCON", "XCON_Line", "Line_OM", "OD_Line", "OM_OA", "OA_OD",
    "FIU_OA", "OA_FIU", "FIU_SC2", "SC2_FIU", "FIU_FIU", "EL_to_OM", "EL_to_Optical"
}

def infer_from_type(function_type: str) -> str:
    s = (function_type or "").lower()
    if "otuk" in s:  return "Line"
    if "oduk" in s:
        if "relay" in s:      return "XCON"
        return "Tributary"
    return "Tributary"

def infer_to_type(from_type: str, action_type: str, action_target: str) -> str:
    a = (action_type or "").lower()
    t = (action_target or "").lower()
    ft = (from_type or "").lower()
    if "report" in a and "local" in a:
        return from_type
    if "upstream" in a:
        return "XCON" if ft == "line" else ("Tributary" if ft == "xcon" else "Tributary")
    if "downstream" in a:
        if ft == "tributary": return "XCON"
        if ft == "xcon":      return "OM" if ("opt" in t or "om" in t) else "Line"
        if ft == "line":      return "OM" if ("opt" in t or "om" in t) else "Line"
    if "opt" in t or "om" in t: return "OM"
    if "od" in t:               return "OD"
    return from_type

def infer_edge_type(ft: str, tt: str, action_target: str, action_type: str):
    t = (action_target or "").lower()
    if ft == "Tributary" and tt == "XCON": return "Tributary_XCON"
    if ft == "XCON" and tt == "Line":      return "XCON_Line"
    if ft == "Line" and tt == "OM":        return "Line_OM"
    if ft == "OD" and tt == "Line":        return "OD_Line"
    if "opt" in t or "om" in t:            return "Line_OM"
    if "electrical" in t and "optical" in t: return "EL_to_OM"
    return None

def norm(s): return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="Functional rules CSV (FunctionType,EventName,ActionType,ActionTarget,OutputAlarm)")
    ap.add_argument("-o", "--output", default="outputs/Static files/rule_database_canonical.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    cols = {norm(c): c for c in df.columns}

    need = ["functiontype","eventname","actiontype","outputalarm"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise SystemExit(f"Missing required columns in input: {missing}. Found: {list(df.columns)}")

    c_fn  = cols["functiontype"]
    c_ev  = cols["eventname"]
    c_act = cols["actiontype"]
    c_out = cols["outputalarm"]
    c_tgt = cols.get("actiontarget")

    out_rows = []
    for _, r in df.iterrows():
        fn  = str(r.get(c_fn)  or "")
        ev  = str(r.get(c_ev)  or "")
        act = str(r.get(c_act) or "")
        out = str(r.get(c_out) or "")
        tgt = str(r.get(c_tgt) or "") if c_tgt else ""

        ft = infer_from_type(fn)
        tt = infer_to_type(ft, act, tgt)
        et = infer_edge_type(ft, tt, tgt, act)

        if et and et not in KNOWN_TTYPES:
            et = None

        out_rows.append({
            "FromType": ft,
            "FromAlarm": ev or "ANY",
            "ToType": tt,
            "ToAlarm": out or ev or "ANY",
            "EdgeType": et,
            "Layer":   None,
            "Direction": None,
            "Severity": None,
            "DelayMs":  None,
            "Note": f"{fn} / {act} / {tgt}".strip(" /"),
        })

    df_out = pd.DataFrame(out_rows, columns=[
        "FromType","FromAlarm","ToType","ToAlarm","EdgeType","Layer","Direction","Severity","DelayMs","Note"
    ])
    df_out.to_csv(args.output, index=False)
    print(f"Wrote {len(df_out)} canonical rules → {args.output}")

if __name__ == "__main__":
    main()