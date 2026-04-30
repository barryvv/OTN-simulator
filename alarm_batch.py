#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

DEF_FLOW = Path("outputs/alarm_flows/alarm_flow.csv")
DEF_FAIL = Path("outputs/alarm_flows/failures_timeline.csv")

parser = argparse.ArgumentParser()
parser.add_argument("--topology", default="outputs/topology_preview/board.graphml")
parser.add_argument("--paths", default=None, help="Optional role-based paths file")
parser.add_argument("--rules", default="outputs/Static files/rule_database_canonical.csv")
parser.add_argument("--failures", default=str(DEF_FAIL))
parser.add_argument("--output", default=str(DEF_FLOW))
args = parser.parse_args()

fail = Path(args.failures)
fail.parent.mkdir(parents=True, exist_ok=True)
if not fail.exists():
    with fail.open("w") as f:
        f.write("Board,Alarm,Severity,Time,DurationMs\n")

cmd = ["python3", "alarm_generator.py", "--rules", args.rules, "--output", args.output, "--failures", args.failures]
if args.paths:
    cmd += ["--paths", args.paths]
else:
    cmd += ["--topology", args.topology]
print("[batch] Running:", " ".join(cmd))
subprocess.run(cmd, check=False)

subprocess.run(["python3", "alarm_chain_builder.py"], check=False)

print("Files written under outputs/alarm_flows/:\n - failures_timeline.csv\n - alarm_flow.csv\n - alarm.csv\n - alarm_chain.csv")