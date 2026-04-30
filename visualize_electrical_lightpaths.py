#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _load_graph(role_file: Path):
    import networkx as nx

    G = nx.DiGraph()
    with role_file.open() as f:
        for line in f:
            tokens = [tok.strip() for tok in line.strip().split(",") if tok.strip()]
            for a, b in zip(tokens, tokens[1:]):
                G.add_edge(a, b)
    return G


def _color_for(node: str) -> str:
    if node.endswith("-SRC"):
        return "
    if node.endswith("-TERMINATION"):
        return "
    return "


def main():
    parser = argparse.ArgumentParser(description="Visualise electrical lightpaths graph.")
    parser.add_argument("--input", default="outputs/topology_preview/electrical_lightpaths_roles.txt",
                        help="Role-labelled lightpaths file.")
    parser.add_argument("--output", default="outputs/topology_preview/electrical_lightpaths_roles_full.png",
                        help="PNG output path.")
    parser.add_argument("--max-nodes", type=int, default=0,
                        help="Optional cap on number of nodes to draw (0 = no cap).")
    parser.add_argument("--fig-width", type=float, default=24.0,
                        help="Figure width in inches (default: 24).")
    parser.add_argument("--fig-height", type=float, default=16.0,
                        help="Figure height in inches (default: 16).")
    parser.add_argument("--font-size", type=float, default=12.0,
                        help="Node label font size (default: 12).")
    args = parser.parse_args()

    role_path = Path(args.input)
    if not role_path.exists():
        raise SystemExit(f"Input file not found: {role_path}")

    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception as exc:
        raise SystemExit(f"matplotlib/networkx required for graph rendering ({exc})")

    G = _load_graph(role_path)
    if args.max_nodes and G.number_of_nodes() > args.max_nodes:
        nodes = list(sorted(G.nodes()))[:args.max_nodes]
        G = G.subgraph(nodes).copy()

    if G.number_of_nodes() == 0:
        raise SystemExit("Graph is empty; nothing to render.")

    pos = nx.spring_layout(G, seed=42)
    colors = [_color_for(n) for n in G.nodes]

    plt.figure(figsize=(args.fig_width, args.fig_height))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=900,
                           edgecolors="black", linewidths=0.4)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=10, alpha=0.45, width=0.8)
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=args.font_size,
        font_weight="bold",
        font_color="black",
        bbox=dict(facecolor="white", alpha=0.92, edgecolor="
    )
    plt.title("Electrical Lightpaths (role-labelled)", fontsize=args.font_size + 6)
    plt.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="
                   markeredgecolor="black", label="SRC", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="
                   markeredgecolor="black", label="RELAY", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="
                   markeredgecolor="black", label="TERMINATION", markersize=10),
    ]
    plt.legend(handles=legend_handles, loc="upper left", fontsize=args.font_size - 1,
               frameon=True, fancybox=True)
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
