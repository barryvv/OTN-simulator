#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except Exception as exc:
    raise SystemExit(f"matplotlib required for plotting ({exc})")


def _node_name(board: str) -> str:
    return board.split("-")[0] if "-" in board else board


def _build_node_graph(board_graph: nx.DiGraph) -> nx.DiGraph:
    node_graph = nx.DiGraph()
    for u, v in board_graph.edges():
        nu = _node_name(u)
        nv = _node_name(v)
        if nu == nv:
            continue
        node_graph.add_edge(nu, nv)
    return node_graph


def _parse_role_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    role_map: dict[str, str] = {}
    suffixes = [
        ("-SRC", "SRC"),
        ("-RELAY", "RELAY"),
        ("-TERMINATION", "TERMINATION"),
        ("-ODUk-Creation", "SRC"),
        ("-ODUk-termination", "TERMINATION"),
        ("-ODUk-Termination", "TERMINATION"),
        ("-OTUk-Relay", "RELAY"),
    ]
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        for token in tokens:
            for suf, role in suffixes:
                if token.endswith(suf):
                    base = token[: -len(suf)]
                    role_map[base] = role
                    break
            else:
                if "-" in token:
                    base, tail = token.rsplit("-", 1)
                    t = tail.upper()
                    if t in {"SRC", "RELAY", "TERMINATION"}:
                        role_map[base] = t
    return role_map


def _board_kind(name: str) -> str:
    if "-" not in name:
        return ""
    suffix = name.split("-", 1)[1]
    for kind in ("Tributary", "XCON", "Line", "OM", "OA", "OD", "FIU", "SC2"):
        if suffix.startswith(kind):
            return kind
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot topology graph from board.graphml")
    ap.add_argument("--graphml", default="outputs/topology_preview/board.graphml")
    ap.add_argument("--out", default="outputs/topology_preview/topology_graph.png")
    ap.add_argument("--mode", choices=["node", "board"], default="node")
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--roles", default="outputs/topology_preview/electrical_lightpaths_roles.txt",
                    help="Optional role-labelled lightpaths file for coloring")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    graphml_path = Path(args.graphml)
    if not graphml_path.exists():
        raise SystemExit(f"Missing graphml: {graphml_path}")

    G = nx.read_graphml(graphml_path)
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    if args.mode == "node":
        H = _build_node_graph(G)
    else:
        H = G.copy()

    role_map = _parse_role_map(Path(args.roles)) if args.mode == "board" else {}
    color_map = {
        "SRC": "#4CAF50",
        "RELAY": "#2196F3",
        "TERMINATION": "#FF9800",
        "REGEN": "#E91E63",
        "DEFAULT": "#9E9E9E",
    }
    node_colors = []
    for n in H.nodes():
        role = role_map.get(n, "")
        if "Regen" in n or G.nodes.get(n, {}).get("is_regen_boundary"):
            role = "REGEN"
        if not role:
            kind = _board_kind(n)
            if kind in {"OM", "OA", "OD", "FIU", "SC2"}:
                role = "RELAY"
            elif kind == "Tributary":
                role = "SRC"
            elif kind == "Line":
                role = "TERMINATION"
            elif kind == "XCON":
                role = "RELAY"
        node_colors.append(color_map.get(role, color_map["DEFAULT"]))

    if H.number_of_nodes() == 0:
        raise SystemExit("Empty graph")

    layout_graph = H.to_undirected()
    pos = nx.spring_layout(layout_graph, seed=args.seed)

    n = H.number_of_nodes()
    figsize = (12, 9) if n > 80 else (10, 7)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f"Topology ({args.mode}-level) nodes={n} edges={H.number_of_edges()}")

    nx.draw_networkx_edges(H, pos, ax=ax, arrows=False, alpha=0.35, width=1.0)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=240, node_color=node_colors, alpha=0.9)
    if args.labels or args.mode == "board":
        nx.draw_networkx_labels(H, pos, ax=ax, font_size=6)

    if args.mode == "board":
        legend_items = [
            Patch(facecolor=color_map["SRC"], label="SRC"),
            Patch(facecolor=color_map["RELAY"], label="RELAY"),
            Patch(facecolor=color_map["TERMINATION"], label="TERMINATION"),
            Patch(facecolor=color_map["REGEN"], label="REGEN"),
            Patch(facecolor=color_map["DEFAULT"], label="OTHER"),
        ]
        ax.legend(handles=legend_items, loc="upper right", frameon=False, fontsize=8)

    ax.axis("off")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
