#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

FLOW = Path("outputs/alarm_flows/alarm_flow.csv")
OUT_CHAIN = Path("outputs/alarm_flows/alarm_chain.csv")
OUT_ALARM = Path("outputs/alarm_flows/alarm.csv")

OUT_CHAIN.parent.mkdir(parents=True, exist_ok=True)

if not FLOW.exists():
    pd.DataFrame(columns=[
        "SourceBoard","SourceAlarm","SourceStartTime","SourceDurationMs",
        "PropToBoard","PropAlarm","ArrivalTime","ClearTime",
        "ViaEdgeType","Layer","Direction","Severity","DelayMs","RuleNote","RuleIdx"
    ]).to_csv(FLOW, index=False)

try:
    df = pd.read_csv(FLOW, parse_dates=["SourceStartTime","ArrivalTime","ClearTime"], keep_default_na=False)
except Exception:
    df = pd.read_csv(FLOW, keep_default_na=False)

if df.empty:
    pd.DataFrame(columns=["ChainId","Step","SourceBoard","SourceAlarm","Board","Alarm","ArrivalTime","ClearTime","ViaEdgeType","Layer","Direction","Severity","RuleIdx"]).to_csv(OUT_CHAIN, index=False)
    pd.DataFrame(columns=["Board","Alarm","StartTime","ClearTime","Severity","SourceBoard","SourceAlarm","RuleIdx"]).to_csv(OUT_ALARM, index=False)
else:
    df = df.copy()
    df["ArrivalSortKey"] = pd.to_datetime(df["ArrivalTime"], errors="coerce")
    chains = []
    chain_id = 0
    for (sb, sa), g in df.groupby(["SourceBoard","SourceAlarm"], dropna=False):
        chain_id += 1
        g2 = g.sort_values(["ArrivalSortKey","PropToBoard"], na_position="first")
        step = 0
        for _, r in g2.iterrows():
            step += 1
            chains.append({
                "ChainId": chain_id,
                "Step": step,
                "SourceBoard": sb,
                "SourceAlarm": sa,
                "Board": r["PropToBoard"],
                "Alarm": r["PropAlarm"],
                "ArrivalTime": r.get("ArrivalTime"),
                "ClearTime": r.get("ClearTime"),
                "ViaEdgeType": r.get("ViaEdgeType"),
                "Layer": r.get("Layer"),
                "Direction": r.get("Direction"),
                "Severity": r.get("Severity"),
                "RuleIdx": r.get("RuleIdx"),
            })
    pd.DataFrame(chains, columns=[
        "ChainId","Step","SourceBoard","SourceAlarm","Board","Alarm","ArrivalTime","ClearTime",
        "ViaEdgeType","Layer","Direction","Severity","RuleIdx"
    ]).to_csv(OUT_CHAIN, index=False)

    flat = df[[
        "PropToBoard","PropAlarm","ArrivalTime","ClearTime","Severity","SourceBoard","SourceAlarm","RuleIdx"
    ]].rename(columns={
        "PropToBoard":"Board","PropAlarm":"Alarm","ArrivalTime":"StartTime"
    })
    flat.to_csv(OUT_ALARM, index=False)

print(f"Wrote {OUT_CHAIN} and {OUT_ALARM}")