#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import re

PAT_BOARD = re.compile(
    r"^(?P<node>[^-]+)-(?P<kind>[A-Za-z]+?)(?P<idx>\d+)(?P<tag>\$0)?(?:-(?P<label>[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*))?$",
    re.IGNORECASE,
)


def _parse_token(token: str) -> tuple[str, str | None]:
    token = token.strip()
    m = PAT_BOARD.match(token)
    if not m:
        base = token if token.endswith("$0") else f"{token}$0"
        return base, None

    node = m.group("node")
    kind = m.group("kind")
    idx = int(m.group("idx"))
    base = f"{node}-{kind}{idx:03d}$0"

    label = m.group("label")
    role_hint = None
    if label:
        label_upper = label.upper()
        if "CREATION" in label_upper or label_upper.endswith("SRC"):
            role_hint = "SRC"
        elif "TERMINATION" in label_upper or label_upper.endswith("TERMINATION"):
            role_hint = "TERMINATION"
    return base, role_hint


def _roles_for_length(n: int) -> list[str]:
    src_n = min(3, n)
    snk_n = min(3, max(0, n - src_n))
    relay_n = max(0, n - src_n - snk_n)
    return ["SRC"] * src_n + ["RELAY"] * relay_n + ["TERMINATION"] * snk_n


def main():
    ap = argparse.ArgumentParser(description="Generate electrical_lightpaths_roles.txt with canonical roles.")
    ap.add_argument("--input", default="outputs/topology_preview/lightpaths.txt",
                    help="Source lightpaths file (default: outputs/topology_preview/lightpaths.txt)")
    ap.add_argument("--output", default="outputs/topology_preview/electrical_lightpaths_roles.txt",
                    help="Destination role-labelled file (default: outputs/topology_preview/electrical_lightpaths_roles.txt)")
    args = ap.parse_args()

    src_path = Path(args.input)
    out_path = Path(args.output)

    if not src_path.exists():
        raise SystemExit(f"Missing {src_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    skipped = 0
    with src_path.open() as fin, out_path.open("w") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            tokens = [t.strip() for t in raw.split(",") if t.strip()]
            if not tokens:
                continue

            defaults = _roles_for_length(len(tokens))
            labeled: list[str] = []
            for (token, default_role) in zip(tokens, defaults):
                base, hint = _parse_token(token)
                role = hint or default_role
                if role not in {"SRC", "RELAY", "TERMINATION"}:
                    labeled = []
                    break
                labeled.append(f"{base}-{role}")

            if labeled:
                fout.write(",".join(labeled) + "\n")
                wrote += 1
            else:
                skipped += 1

    msg = f"Wrote {wrote} path(s) → {out_path}"
    if skipped:
        msg += f" (skipped {skipped})"
    print(msg)


if __name__ == "__main__":
    main()
