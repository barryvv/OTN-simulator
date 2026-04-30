#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import yaml
import networkx as nx


def canonical_seg_id(u: int, v: int) -> str:
    a, b = (u, v) if u < v else (v, u)
    return f"{a}-{b}"


def compress_parent_path(tokens: Iterable[str]) -> list[str]:
    parents: list[str] = []
    for tok in tokens:
        t = tok.strip()
        if not t or "-" not in t:
            continue
        parent = t.split("-", 1)[0]
        if not parents or parent != parents[-1]:
            parents.append(parent)
    return parents


def normalize_board_name(name: str) -> str:
    if "-" not in name:
        return name
    base = name
    if base.endswith("$0"):
        base = base[:-2]
    parts = base.split("-", 1)
    if len(parts) != 2:
        return name
    parent, rest = parts
    kind = "".join([c for c in rest if not c.isdigit()])
    digits = "".join([c for c in rest if c.isdigit()])
    if not digits:
        return name
    idx = int(digits)
    return f"{parent}-{kind}{idx:03d}$0"


def expand_service_to_vmf(service: dict, id_rules: dict, regen_hops: int) -> list[dict]:
    vmf = []
    src_node = service["src"]
    dst_node = service["dst"]
    segs = service["path_segments"]
    node_path = service["node_path"]

    def add_src(node, service_id, regen=False):
        vmf_id = id_rules["src_fmt"].format(node=node, service=service_id)
        vmf.append({"vmf_id": vmf_id, "role": "SRC", "node": node, "regen": bool(regen)})

    def add_snk(node, service_id, regen=False):
        vmf_id = id_rules["snk_fmt"].format(node=node, service=service_id)
        vmf.append({"vmf_id": vmf_id, "role": "SNK", "node": node, "regen": bool(regen)})

    def add_relay_in(node, seg_id):
        vmf_id = id_rules["relay_fmt"].format(node=node, segment=seg_id, dir="in")
        vmf.append({"vmf_id": vmf_id, "role": "RELAY_IN", "node": node, "segment": seg_id, "dir": "in", "regen": False})

    def add_relay_out(node, seg_id):
        vmf_id = id_rules["relay_fmt"].format(node=node, segment=seg_id, dir="out")
        vmf.append({"vmf_id": vmf_id, "role": "RELAY_OUT", "node": node, "segment": seg_id, "dir": "out", "regen": False})

    service_id = service["service_id"]
    add_src(src_node, service_id, regen=False)

    hop_count = len(segs)
    for i, seg_id in enumerate(segs):
        u, v = service["node_path"][i], service["node_path"][i + 1]
        add_relay_out(u, seg_id)
        add_relay_in(v, seg_id)

        if regen_hops > 0:
            is_boundary = (i + 1) % regen_hops == 0
            not_last = (i + 1) < hop_count
            if is_boundary and not_last:
                add_snk(v, service_id, regen=True)
                add_src(v, service_id, regen=True)

    add_snk(dst_node, service_id, regen=False)
    return vmf


def build_fiber_groups(services: list[dict]) -> list[dict]:
    fiber_groups = {}
    for s in services:
        for v in s.get("vmf_path", []):
            if "segment" not in v:
                continue
            seg = v["segment"]
            a, b = map(int, seg.split("-"))
            fg_id = f"fg_{a}_{b}"
            fg = fiber_groups.setdefault(fg_id, {"id": fg_id, "node_a": a, "node_b": b, "relays": []})
            fg["relays"].append(v["vmf_id"])
    return list(fiber_groups.values())


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert board.graphml + lightpaths.txt into topology.yaml")
    ap.add_argument("--graphml", default="outputs/topology_preview/board.graphml",
                    help="Path to board.graphml")
    ap.add_argument("--lightpaths", default="outputs/topology_preview/lightpaths.txt",
                    help="Path to lightpaths.txt (board-level paths)")
    ap.add_argument("--out", default="topology.yaml",
                    help="Path to write topology.yaml")
    ap.add_argument("--repeat-services", type=int, default=1,
                    help="Duplicate all services this many times (>=1)")
    ap.add_argument("--boost-parent", default=None,
                    help="Parent node name to boost (e.g., ROADM2)")
    ap.add_argument("--boost-factor", type=int, default=0,
                    help="Extra copies per service that includes --boost-parent")
    ap.add_argument("--regen-hops", type=int, default=0,
                    help="Regen hop interval (0 = disabled)")
    ap.add_argument("--H-out", type=int, default=30,
                    help="Propagation window H_out used by otn_simulator")
    args = ap.parse_args()

    graphml = Path(args.graphml)
    lightpaths = Path(args.lightpaths)

    if not graphml.exists():
        raise SystemExit(f"Missing graphml: {graphml}")
    if not lightpaths.exists():
        raise SystemExit(f"Missing lightpaths.txt: {lightpaths}")

    G = nx.read_graphml(graphml)
    parent_names = set()
    board_map = {}
    for n, a in G.nodes(data=True):
        parent = a.get("parentName") or (str(n).split("-", 1)[0] if "-" in str(n) else str(n))
        parent = str(parent)
        parent_names.add(parent)
        kind = str(a.get("tType") or "")
        full = str(a.get("fullName") or n)
        board = normalize_board_name(full)
        if parent and kind and board:
            board_map.setdefault(parent, {}).setdefault(kind.upper(), [])
            board_map[parent][kind.upper()].append(board)

    services = []
    with lightpaths.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            tokens = [t.strip() for t in line.split(",") if t.strip()]
            parents = compress_parent_path(tokens)
            if len(parents) < 2:
                continue
            parent_names.update(parents)
            services.append(parents)

    if not services:
        raise SystemExit("No usable services parsed from lightpaths.txt.")

    parent_sorted = sorted(parent_names)
    parent_to_id = {name: i + 1 for i, name in enumerate(parent_sorted)}
    for parent, kinds in board_map.items():
        for k in list(kinds.keys()):
            kinds[k] = sorted(set(kinds[k]))

    service_rows = []
    seg_ids = set()
    for i, parents in enumerate(services, start=1):
        node_path = [parent_to_id[p] for p in parents]
        if len(node_path) < 2:
            continue
        segs = [canonical_seg_id(node_path[j], node_path[j + 1]) for j in range(len(node_path) - 1)]
        oriented = [f"{node_path[j]}->{node_path[j + 1]}" for j in range(len(node_path) - 1)]
        seg_ids.update(segs)
        service_rows.append({
            "src": node_path[0],
            "dst": node_path[-1],
            "node_path": node_path,
            "path_segments": segs,
            "path_segments_oriented": oriented,
            "_parent_path": parents,
        })

    base_rows = service_rows
    service_rows = []
    repeat = max(1, int(args.repeat_services))
    boost_parent = str(args.boost_parent) if args.boost_parent else None
    boost_factor = max(0, int(args.boost_factor))

    for _ in range(repeat):
        service_rows.extend([dict(r) for r in base_rows])
    if boost_parent and boost_factor > 0:
        for r in base_rows:
            if boost_parent in r.get("_parent_path", []):
                for _ in range(boost_factor):
                    service_rows.append(dict(r))

    segments = []
    segments_index = {}
    for sid in sorted(seg_ids, key=lambda s: tuple(map(int, s.split("-")))):
        a, b = map(int, sid.split("-"))
        segments.append({"segment_id": sid, "a_end": a, "z_end": b})
        segments_index[sid] = (a, b)

    id_rules = {
        "src_fmt": "{node}_SRC_{service}",
        "relay_fmt": "{node}_RELAY_{segment}_{dir}",
        "snk_fmt": "{node}_SNK_{service}",
    }

    for idx, svc in enumerate(service_rows, start=1):
        svc["service_id"] = f"S{idx}"
        svc.pop("_parent_path", None)
        svc["vmf_path"] = expand_service_to_vmf(svc, id_rules, regen_hops=int(args.regen_hops))

    topo = {
        "network": {
            "segments": segments,
            "segments_index": segments_index,
            "segment_index": segments_index,
        },
        "node_map": parent_to_id,
        "board_map": board_map,
        "services": service_rows,
        "id_rules": id_rules,
        "fiber_groups": build_fiber_groups(service_rows),
        "ow_scheme": {"H_out": int(args.H_out)},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(topo, f, sort_keys=False, allow_unicode=True)

    print(f"[ok] wrote {out_path}")
    print(f"  parents={len(parent_sorted)}, services={len(service_rows)}, segments={len(segments)}")


if __name__ == "__main__":
    main()
