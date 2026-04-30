#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RULE_DB = REPO_ROOT / "outputs" / "Static files" / "rule_database.csv"
DEFAULT_RUN_SCRIPT = REPO_ROOT / "run_all.sh"
DEFAULT_ARCHIVE_DIR = REPO_ROOT / "outputs" / "alarm_flows" / "test_runs"


def _clean_line_iter(path: Path) -> Iterable[str]:
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            yield raw


def load_event_names(rule_db: Path) -> List[str]:
    reader = csv.DictReader(_clean_line_iter(rule_db))
    events = []
    for row in reader:
        event = (row.get("EventName") or "").strip()
        if event:
            events.append(event)
    seen = set()
    unique = []
    for ev in events:
        if ev not in seen:
            seen.add(ev)
            unique.append(ev)
    return unique


def slugify(text: str) -> str:
    safe = "".join(
        ch.lower() if ch.isalnum() else "_" for ch in text.strip()
    ).strip("_")
    return safe or "event"


def copy_outputs(archive_dir: Path, event_name: str) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    alarm_dir = REPO_ROOT / "outputs" / "alarm_flows"
    files = [
        ("alarm_flow.csv", "alarm_flow.csv"),
        ("alarm_flow_timeline.csv", "alarm_flow_timeline.csv"),
        ("failure.csv", "failure.csv"),
    ]
    for src_name, dst_name in files:
        src = alarm_dir / src_name
        if src.exists():
            shutil.copy2(src, archive_dir / dst_name)
    data_dir = REPO_ROOT / "outputs" / "data"
    local_alarms = data_dir / "local_alarms_from_data.csv"
    if local_alarms.exists():
        shutil.copy2(local_alarms, archive_dir / "local_alarms_from_data.csv")


def run_event(event: str, args: argparse.Namespace) -> bool:
    env = os.environ.copy()
    env["FAIL_EVENT"] = event
    if args.fail_board:
        env["FAIL_BOARD"] = args.fail_board
    run_script_path = args.run_script
    if not run_script_path.is_absolute():
        run_script_path = (REPO_ROOT / run_script_path).resolve()
    if not run_script_path.exists():
        print(f"[FAIL] run script not found: {run_script_path}")
        return False
    cmd = [str(run_script_path)]
    print(f"\n=== Running {cmd[0]} with FAIL_EVENT={event} ===")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if proc.returncode != 0:
        print(f"[FAIL] Event '{event}' exited with code {proc.returncode}")
        return False
    if args.archive_dir:
        dest = Path(args.archive_dir) / slugify(event)
        copy_outputs(dest, event)
        print(f"[ARCHIVE] Copied outputs → {dest}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run run_all.sh for each alarm event listed in the rule database."
    )
    parser.add_argument(
        "--rules", type=Path, default=DEFAULT_RULE_DB,
        help=f"Path to rule database CSV (default: {DEFAULT_RULE_DB})",
    )
    parser.add_argument(
        "--run-script", type=Path, default=DEFAULT_RUN_SCRIPT,
        help=f"Pipeline script to invoke per event (default: {DEFAULT_RUN_SCRIPT})",
    )
    parser.add_argument(
        "--events", nargs="*", default=None,
        help="Optional subset of events to run (space separated). "
             "If omitted, every EventName in the rule DB is used.",
    )
    parser.add_argument(
        "--fail-board", default=None,
        help="Optional board override (defaults to AUTO selection inside run_all.sh).",
    )
    parser.add_argument(
        "--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR,
        help=f"Directory to store per-event copies of alarm_flow outputs "
             f"(default: {DEFAULT_ARCHIVE_DIR}). Use '' to disable archiving.",
    )
    parser.add_argument(
        "--stop-on-failure", action="store_true",
        help="Abort the suite as soon as one run_all.sh invocation fails.",
    )
    args = parser.parse_args()

    if not args.run_script.exists():
        parser.error(f"run script not found: {args.run_script}")
    if not args.rules.exists():
        parser.error(f"rule database not found: {args.rules}")
    if args.archive_dir and not Path(args.archive_dir).exists():
        Path(args.archive_dir).mkdir(parents=True, exist_ok=True)

    events = args.events or load_event_names(args.rules)
    if not events:
        parser.error("No events found to run.")

    summary = []
    for idx, event in enumerate(events, start=1):
        print(f"\n[{idx}/{len(events)}] Testing event: {event}")
        ok = run_event(event, args)
        summary.append((event, ok))
        if not ok and args.stop_on_failure:
            break

    print("\n=== Test Summary ===")
    for event, ok in summary:
        status = "PASS" if ok else "FAIL"
        print(f"{status:>4}  {event}")

    failed = [ev for ev, ok in summary if not ok]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
