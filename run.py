#!/usr/bin/env python3
from typing import Optional
import os
from pathlib import Path
import csv
import networkx as nx
import matplotlib.pyplot as plt
import re
import random
from topology_traffic_generator import generate_topology
from topology_traffic_processor import process_traffic_from_nodeDG, generate_board_list

_ARANGO_SKIP = os.environ.get("ARANGO_SKIP", "0") == "1"
if not _ARANGO_SKIP:
    try:
        from arango import ArangoClient
    except Exception:
        try:
            from arango.client import ArangoClient
        except Exception as e:
            raise ImportError(
                "ArangoClient not found. You likely installed the wrong package.\n"
                "Fix by running: pip uninstall -y arango && pip install python-arango\n"
                "(The correct PyPI package name is 'python-arango'.)"
            ) from e
    from arango.exceptions import DocumentInsertError
else:
    ArangoClient = None  # type: ignore[assignment,misc]
    DocumentInsertError = Exception  # type: ignore[assignment,misc]

ARANGO_URL = os.environ.get("ARANGO_URL", "http://localhost:8529")
ARANGO_USER = os.environ.get("ARANGO_USER", "root")
ARANGO_PASS = os.environ.get("ARANGO_PASS", "")
ARANGO_DB   = os.environ.get("ARANGO_DB", "topology_db")

GRAPH_NAME = os.environ.get("ARANGO_GRAPH", "electrical_layer_topography")
COLL_PREFIX = os.environ.get("ARANGO_PREFIX", "EL_")

SHOW_LABELS = os.environ.get("PLOT_LABELS", "0") == "1"


def _side_from_index(idx_str: str) -> str:
    """Map Line/XCON/Tributary index to optical side ('1' or '2').

    From class_obj.py, board indices are laid out as idx = 1 + eg*4 + k,
    where k in (0,2) connects to OM1/OD1 (side '1') and k in (1,3)
    connects to OM2/OD2 (side '2').
    """
    try:
        idx = int(idx_str)
    except Exception:
        return '1'
    k = (idx - 1) % 4
    return '1' if k in (0, 2) else '2'

def _neighbors_undirected(G: nx.DiGraph, n: str):
    return set(G.successors(n)) | set(G.predecessors(n))

def _board_parent(n: str) -> str:
    return n.split('-')[0] if '-' in n else ""

def _node_graph_from_board(board_DG: nx.DiGraph) -> nx.DiGraph:
    node_DG = nx.DiGraph()
    for n in board_DG.nodes():
        node = _board_parent(n)
        if node:
            node_DG.add_node(node)
    for u, v in board_DG.edges():
        nu = _board_parent(u)
        nv = _board_parent(v)
        if nu and nv and nu != nv:
            node_DG.add_edge(nu, nv)
    return node_DG

def _find_om_for_side(G: nx.DiGraph, parent: str, side: str):
    cand = f"{parent}-OM{side}$0"
    if G.has_node(cand):
        return cand
    for nn in G.nodes():
        m = _OM_PAT.match(nn)
        if m and m.group('node') == parent:
            return nn
    return None

def _get_triplet_for_index(G: nx.DiGraph, parent: str, idx: int):
    pad = f"{idx:03d}"
    t = f"{parent}-Tributary{pad}$0" if G.has_node(f"{parent}-Tributary{pad}$0") else f"{parent}-Tributary{idx}$0"
    x = f"{parent}-XCON{pad}$0"      if G.has_node(f"{parent}-XCON{pad}$0")      else f"{parent}-XCON{idx}$0"
    l = f"{parent}-Line{pad}$0"      if G.has_node(f"{parent}-Line{pad}$0")      else f"{parent}-Line{idx}$0"
    return (t if G.has_node(t) else None,
            x if G.has_node(x) else None,
            l if G.has_node(l) else None)


def _optical_undirected(board_DG: nx.DiGraph) -> nx.Graph:
    H = nx.Graph()
    for n, a in board_DG.nodes(data=True):
        if a.get("layer") == "optical":
            H.add_node(n, **a)
    for u, v, d in board_DG.edges(data=True):
        if board_DG.nodes[u].get("layer") == "optical" and board_DG.nodes[v].get("layer") == "optical":
            H.add_edge(u, v, **d)
    return H

def _shortest_om_path(board_DG: nx.DiGraph, om_local: str, max_hops: int = 8):
    H = _optical_undirected(board_DG)
    if om_local not in H:
        return None
    local_parent = _board_parent(om_local)
    remote_oms = [n for n in H.nodes() if _OM_PAT.match(n) and _board_parent(n) != local_parent]
    best = None
    best_len = None
    for om in remote_oms:
        try:
            p = nx.shortest_path(H, om_local, om)
            if len(p) - 1 <= max_hops:
                if best is None or len(p) < best_len:
                    best = p
                    best_len = len(p)
        except nx.NetworkXNoPath:
            continue
    return best


# ---------------------------------------------------------------------------
# Directional optical chain tables per ROADM
# ---------------------------------------------------------------------------
# Within a ROADM the optical ring has two directions (U/L):
#   Direction U: FIU1 → OA1 → OD1 → OM1 → OA2 → FIU2
#   Direction L: FIU2 → OA3 → OD2 → OM2 → OA4 → FIU1
#
# For a lightpath the correct sub-chains are:
#   Source (mux):    OM → OA → FIU
#   Relay  (pass):   FIU → OA → OD → OM → OA → FIU
#   Dest   (demux):  FIU → OA → OD
#
# Mux chain (keyed by OM suffix): OM side → [OA, FIU]
_MUX_AFTER_OM = {
    'OM1': ('OA2', 'FIU2'),   # direction U outbound
    'OM2': ('OA4', 'FIU1'),   # direction L outbound
}
# Demux chain (keyed by arriving FIU suffix): FIU → [OA, OD]
_DEMUX_AFTER_FIU = {
    'FIU1': ('OA1', 'OD1'),   # direction U inbound
    'FIU2': ('OA3', 'OD2'),   # direction L inbound
}
# Full pass-through (keyed by arriving FIU): FIU → OA → OD → OM → OA → FIU
_PASSTHROUGH = {
    'FIU1': ('OA1', 'OD1', 'OM1', 'OA2', 'FIU2'),
    'FIU2': ('OA3', 'OD2', 'OM2', 'OA4', 'FIU1'),
}


def _om_suffix(board_name: str) -> str:
    """Extract 'OM1' or 'OM2' from e.g. 'ROADM0-OM1$0'."""
    for s in ('OM1', 'OM2'):
        if s in board_name:
            return s
    return ''


def _fiu_suffix(board_name: str) -> str:
    """Extract 'FIU1' or 'FIU2'."""
    for s in ('FIU1', 'FIU2'):
        if s in board_name:
            return s
    return ''


def _board_at(parent: str, suffix: str) -> str:
    return f"{parent}-{suffix}$0"


def _find_remote_fiu(board_DG: nx.DiGraph, local_fiu: str) -> str | None:
    """Find the FIU at a remote node connected to local_fiu via inter-node fiber."""
    local_parent = _board_parent(local_fiu)
    # Check directed successors first, then predecessors
    for nbr in list(board_DG.successors(local_fiu)) + list(board_DG.predecessors(local_fiu)):
        if _board_parent(nbr) != local_parent and 'FIU' in nbr:
            return nbr
    return None


def _find_remote_fiu_to(board_DG: nx.DiGraph, local_fiu: str, target_parent: str) -> str | None:
    """Find the FIU at target_parent connected to local_fiu."""
    for nbr in list(board_DG.successors(local_fiu)) + list(board_DG.predecessors(local_fiu)):
        if _board_parent(nbr) == target_parent and 'FIU' in nbr:
            return nbr
    return None


def _has_internode_fiu(board_DG: nx.DiGraph, fiu_board: str) -> bool:
    """Check if a FIU board has any inter-node connections."""
    parent = _board_parent(fiu_board)
    for nbr in list(board_DG.successors(fiu_board)) + list(board_DG.predecessors(fiu_board)):
        if _board_parent(nbr) != parent and 'FIU' in nbr:
            return True
    return False


def _build_directional_optical_middle(board_DG: nx.DiGraph, om_local: str,
                                       remote_parent: str) -> list[str] | None:
    """Build correct directional optical path from om_local to OD at remote_parent.

    Returns the optical boards in correct signal order:
      Source:  [OM, OA, FIU]
      Relay:   [FIU, OA, OD, OM, OA, FIU]  (repeated for each relay node)
      Dest:    [FIU, OA, OD]

    If the primary FIU (from the mux chain) has no inter-node connections,
    falls back to the other FIU via internal pass-through.
    """
    src_parent = _board_parent(om_local)
    om_suf = _om_suffix(om_local)
    if om_suf not in _MUX_AFTER_OM:
        return None

    # Source mux chain: OM → OA → FIU
    mux_oa, mux_fiu = _MUX_AFTER_OM[om_suf]
    primary_fiu = _board_at(src_parent, mux_fiu)

    if not _has_internode_fiu(board_DG, primary_fiu):
        # Primary FIU has no inter-node connections — this Line/OM
        # combination cannot physically reach any remote node.
        # In a real network the control plane would provision the
        # service on the other direction instead.
        return None

    result = [
        _board_at(src_parent, om_suf),
        _board_at(src_parent, mux_oa),
        primary_fiu,
    ]

    # Verify boards exist
    for b in result:
        if not board_DG.has_node(b):
            return None

    current_fiu = result[-1]
    visited_nodes = {src_parent}
    max_relay = 20  # safety limit

    while True:
        # Find inter-node FIU connection toward remote_parent
        remote_fiu = _find_remote_fiu_to(board_DG, current_fiu, remote_parent)
        if remote_fiu:
            # Reached destination — demux chain
            fiu_suf = _fiu_suffix(remote_fiu)
            if fiu_suf not in _DEMUX_AFTER_FIU:
                return None
            demux_oa, demux_od = _DEMUX_AFTER_FIU[fiu_suf]
            dest_boards = [
                remote_fiu,
                _board_at(remote_parent, demux_oa),
                _board_at(remote_parent, demux_od),
            ]
            for b in dest_boards:
                if not board_DG.has_node(b):
                    return None
            result.extend(dest_boards)
            return result

        # Not directly connected — find any reachable remote FIU (relay)
        relay_fiu = None
        for nbr in list(board_DG.successors(current_fiu)) + list(board_DG.predecessors(current_fiu)):
            nbr_parent = _board_parent(nbr)
            if nbr_parent not in visited_nodes and 'FIU' in nbr:
                relay_fiu = nbr
                break

        if not relay_fiu:
            return None  # no path found

        relay_parent = _board_parent(relay_fiu)
        visited_nodes.add(relay_parent)
        max_relay -= 1
        if max_relay <= 0:
            return None

        # Pass-through chain at relay
        fiu_suf = _fiu_suffix(relay_fiu)
        if fiu_suf not in _PASSTHROUGH:
            return None
        pt_chain = _PASSTHROUGH[fiu_suf]
        relay_boards = [relay_fiu] + [_board_at(relay_parent, s) for s in pt_chain]

        # If the exit FIU of this relay has no inter-node connections,
        # this path is unroutable (cannot loop through internal ring)
        exit_fiu = relay_boards[-1]
        if not _has_internode_fiu(board_DG, exit_fiu):
            return None

        for b in relay_boards:
            if not board_DG.has_node(b):
                return None
        result.extend(relay_boards)
        current_fiu = relay_boards[-1]  # last FIU

def synthesize_lightpaths(board_DG: nx.DiGraph, limit_per_node: int = 8):
    seed_env = os.environ.get("E_SEED")
    seed = int(seed_env) if seed_env not in (None, "") else None
    rnd = random.Random(seed)

    per = _group_el_by_node_and_index(board_DG)
    paths = []

    lp_min = int(os.environ.get("LP_MIN_PER_NODE", "4"))
    lp_max = int(os.environ.get("LP_MAX_PER_NODE", "10"))
    if lp_max < lp_min:
        lp_max = lp_min
    remote_mode = os.environ.get("LP_REMOTE_PICK", "shortest").lower()

    Hopt = _optical_undirected(board_DG)

    def _reachable_remote_oms(om_local: str):
        local_parent = _board_parent(om_local)
        cands = []
        for node in Hopt.nodes():
            if not _OM_PAT.match(node):
                continue
            if _board_parent(node) == local_parent:
                continue
            try:
                _ = nx.shortest_path(Hopt, om_local, node)
                cands.append(node)
            except nx.NetworkXNoPath:
                continue
        return cands

    for parent, buckets in per.items():
        all_line_idxs = sorted(buckets.get("Line", {}).keys())
        if not all_line_idxs:
            continue
        take = min(len(all_line_idxs), rnd.randint(lp_min, lp_max))
        rnd.shuffle(all_line_idxs)
        chosen = all_line_idxs[:take]

        for idx in chosen:
            line = buckets["Line"].get(idx)
            if not line:
                continue
            side = _side_from_index(str(idx))
            om_local = _find_om_for_side(board_DG, parent, side)
            if not om_local or om_local not in Hopt:
                continue

            remote_om = None
            if remote_mode == "random":
                cands = _reachable_remote_oms(om_local)
                if cands:
                    remote_om = rnd.choice(cands)
            if remote_om is None:
                om_path = _shortest_om_path(board_DG, om_local, max_hops=12)
                if om_path and len(om_path) >= 2:
                    remote_om = om_path[-1]
            if not remote_om:
                continue

            remote_parent = _board_parent(remote_om)

            t1, x1, l1 = _get_triplet_for_index(board_DG, parent, idx)
            left = [p for p in (t1, x1, l1) if p]

            # Build correct directional optical path: OM→OA→FIU ... FIU→OA→OD
            middle = _build_directional_optical_middle(
                board_DG, om_local, remote_parent)
            if middle is None:
                # This Line/OM/FIU combination cannot physically reach
                # remote_parent — skip (no unrealistic loopback paths)
                continue

            # Determine dest OD side from the middle path's last board,
            # then pick a compatible dest Line (OD1→side'1', OD2→side'2')
            dest_side = side  # default: same side as source
            if middle:
                last_opt = middle[-1]
                if 'OD1' in last_opt:
                    dest_side = '1'
                elif 'OD2' in last_opt:
                    dest_side = '2'

            t2, x2, l2 = _get_triplet_for_index(board_DG, remote_parent, idx)
            # Verify dest Line matches OD side; if not, find a compatible one
            if l2 and _side_from_index(str(idx)) != dest_side:
                l2 = None  # force fallback search
            if not l2:
                rb = per.get(remote_parent, {})
                side_matches = [j for j in rb.get("Line", {}).keys()
                                if _side_from_index(str(j)) == dest_side]
                rnd.shuffle(side_matches)
                for j in side_matches:
                    l2 = rb["Line"][j]
                    t2, x2, _ = _get_triplet_for_index(board_DG, remote_parent, j)
                    if l2:
                        break
            if not l2:
                continue

            right = [p for p in (l2, x2, t2) if p]

            if l1 and l2:
                path = (left if left else [l1]) + middle + (right if right else [l2])
                paths.append(path)

    return paths

ROLE_CREATION = "ODUk-Creation"
ROLE_TERMINATION = "ODUk-termination"
ROLE_RELAY = "OTUk-Relay"

_ELECTRICAL_KINDS = {"Tributary", "XCON", "Line"}

def _is_electrical_node(board_DG: nx.DiGraph, name: str) -> bool:
    a = board_DG.nodes.get(name, {}) if hasattr(board_DG, 'nodes') else {}
    layer = a.get("layer")
    if layer:
        return layer == "electrical"
    m = _ID_PAT.match(name)
    return bool(m and m.group("kind") in _ELECTRICAL_KINDS)

def _annotate_roles_for_path(board_DG: nx.DiGraph, path: list[str]) -> list[str]:
    e_idx = [i for i, n in enumerate(path) if _is_electrical_node(board_DG, n)]
    k = len(e_idx)
    if k == 0:
        return path[:]

    first = set(e_idx[:min(3, k)])
    last_cnt = min(3, k)
    last = set(e_idx[-last_cnt:])
    term_only = last - first

    out = []
    for i, n in enumerate(path):
        if i in first:
            out.append(f"{n}-{ROLE_CREATION}")
        elif i in term_only:
            out.append(f"{n}-{ROLE_TERMINATION}")
        elif i in e_idx:
            out.append(f"{n}-{ROLE_RELAY}")
        else:
            out.append(n)
    return out

def _regen_hops_from_env() -> int:
    try:
        return max(0, int(os.environ.get("REGEN_HOPS", "0")))
    except ValueError:
        return 0

def _is_optical_node(board_DG: nx.DiGraph, name: str) -> bool:
    layer = board_DG.nodes.get(name, {}).get("layer")
    return layer is None or layer == "optical"

def _add_regen_node(board_DG: nx.DiGraph, name: str) -> None:
    if board_DG.has_node(name):
        return
    board_DG.add_node(name)
    nx.set_node_attributes(board_DG, {
        name: {
            "tType": "Regen",
            "layer": "optical",
            "direction": "/",
            "is_regen_boundary": True,
        }
    })

def _add_regen_edge(board_DG: nx.DiGraph, src: str, dst: str, base_edge: tuple[str, str]) -> None:
    if board_DG.has_edge(src, dst):
        return
    base = board_DG.get_edge_data(base_edge[0], base_edge[1]) or board_DG.get_edge_data(base_edge[1], base_edge[0]) or {}
    attrs = {
        "layer": base.get("layer", "optical"),
        "direction": base.get("direction", "/"),
        "tType": base.get("tType"),
    }
    board_DG.add_edge(src, dst)
    nx.set_edge_attributes(board_DG, {(src, dst): attrs})

def _node_of(board: str) -> str:
    """Extract ROADM parent name from a board name, e.g. 'ROADM0-OA002$0' -> 'ROADM0'."""
    return board.split("-")[0] if "-" in board else board


def _insert_regen_nodes(board_DG: nx.DiGraph, path: list[str], regen_hops: int, path_id: int) -> list[str]:
    """Insert RegenIn/RegenOut pairs at inter-node fiber span boundaries.

    A 'hop' is counted only when the path crosses from one ROADM node to
    another (i.e. a fiber span).  RegenIn is placed on the receiving node
    right after the span crossing, followed immediately by RegenOut.

    With regen_hops=2 the first regen appears after the 2nd span crossing,
    then every 2 spans after that.
    """
    if regen_hops <= 0 or len(path) < 3:
        return path

    # 1. Find all inter-node span crossing indices.
    #    A crossing is at position i where node_of(path[i-1]) != node_of(path[i]).
    crossings: list[int] = []
    for i in range(1, len(path)):
        if _node_of(path[i - 1]) != _node_of(path[i]):
            crossings.append(i)

    if len(crossings) < regen_hops:
        return path  # not enough spans to place any regen

    # 2. Decide which crossings get a regen pair.
    #    Place after every regen_hops-th crossing, but not at the very last
    #    crossing (that's the final destination node).
    regen_at: set[int] = set()
    for ci, cross_idx in enumerate(crossings, start=1):
        if ci % regen_hops == 0 and cross_idx < len(path) - 1:
            regen_at.add(cross_idx)

    if not regen_at:
        return path

    # 3. Build the new path, inserting RegenIn/RegenOut at each chosen crossing.
    out: list[str] = [path[0]]
    regen_idx = 0
    for i in range(1, len(path)):
        prev = path[i - 1]
        curr = path[i]
        if i in regen_at and _is_optical_node(board_DG, prev) and _is_optical_node(board_DG, curr):
            regen_idx += 1
            # Place regen on the receiving node (curr's side of the span)
            recv_node = _node_of(curr)
            tag = f"{path_id:03d}{regen_idx:02d}"
            regen_in = f"{recv_node}-RegenIn{tag}$0"
            regen_out = f"{recv_node}-RegenOut{tag}$0"
            _add_regen_node(board_DG, regen_in)
            _add_regen_node(board_DG, regen_out)
            _add_regen_edge(board_DG, prev, regen_in, (prev, curr))
            _add_regen_edge(board_DG, regen_in, regen_out, (prev, curr))
            _add_regen_edge(board_DG, regen_out, curr, (prev, curr))
            out.extend([regen_in, regen_out, curr])
        else:
            out.append(curr)
    return out

def export_lightpaths(board_DG: nx.DiGraph, out_path: Path, limit_per_node: int = 8):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = synthesize_lightpaths(board_DG, limit_per_node=limit_per_node)
    regen_hops = _regen_hops_from_env()
    with out_path.open("w") as f:
        for idx, p in enumerate(paths):
            if regen_hops:
                p = _insert_regen_nodes(board_DG, p, regen_hops, idx)
            p_annot = _annotate_roles_for_path(board_DG, p)
            f.write(",".join(p_annot) + "\n")
    print(f"Wrote {len(paths)} lightpaths to {out_path}")

_ID_PAT = re.compile(r"^(?P<node>[^-]+)-(?P<kind>Tributary|XCON|Line)(?P<idx>\d+)(?P<suf>\$\d+)?$")

def _group_el_by_node_and_index(board_DG):
    per = {}
    for n in board_DG.nodes():
        m = _ID_PAT.match(n)
        if not m:
            continue
        node = m.group('node')
        kind = m.group('kind')
        idx = int(m.group('idx'))
        per.setdefault(node, {'Tributary': {}, 'XCON': {}, 'Line': {}})[kind][idx] = n
    return per

def _ensure_el_chains(board_DG):
    per = _group_el_by_node_and_index(board_DG)
    added_tx = added_xl = 0
    for node, b in per.items():
        for i, trib in b['Tributary'].items():
            xnode = b['XCON'].get(i)
            if xnode and not board_DG.has_edge(trib, xnode):
                board_DG.add_edge(trib, xnode, tType='Tributary_XCON', layer='electrical', direction='U')
                board_DG.add_edge(xnode, trib, tType='XCON_Tributary', layer='electrical', direction='L')
                added_tx += 1
        for i, xnode in b['XCON'].items():
            lnode = b['Line'].get(i)
            if lnode and not board_DG.has_edge(xnode, lnode):
                board_DG.add_edge(xnode, lnode, tType='XCON_Line', layer='electrical', direction='U')
                board_DG.add_edge(lnode, xnode, tType='Line_XCON', layer='electrical', direction='L')
                added_xl += 1
    return added_tx, added_xl

def _audit_el_pairs(board_DG, max_examples=10):
    per = _group_el_by_node_and_index(board_DG)
    missing = {}
    missing_edges = []

    for node, b in per.items():
        ms = missing.setdefault(node, {'no_Tributary': [], 'no_XCON': [], 'no_Line': []})
        idxs = set(b['Tributary']) | set(b['XCON']) | set(b['Line'])
        for i in sorted(idxs):
            t = b['Tributary'].get(i)
            x = b['XCON'].get(i)
            l = b['Line'].get(i)
            if not t:
                ms['no_Tributary'].append(i)
            if not x:
                ms['no_XCON'].append(i)
            if not l:
                ms['no_Line'].append(i)
            if t and x and not board_DG.has_edge(t, x):
                missing_edges.append((t, x, 'missing Tributary→XCON'))
            if x and l and not board_DG.has_edge(x, l):
                missing_edges.append((x, l, 'missing XCON→Line'))

    print("[Audit] Electrical pairs check:")
    for node, ms in missing.items():
        parts = []
        for k in ('no_Tributary','no_XCON','no_Line'):
            if ms[k]:
                parts.append(f"{k}={sorted(ms[k])[:10]}{'…' if len(ms[k])>10 else ''}")
        if parts:
            print(f"  {node}: " + "; ".join(parts))
    if missing_edges:
        print(f"[Audit] Missing edges ({min(len(missing_edges), max_examples)} shown):")
        for frm, to, why in missing_edges[:max_examples]:
            print("   -", why, ":", frm, "→", to)
    else:
        print("  All required Tributary↔XCON↔Line edges exist for present endpoints.")


def _normalize_variants(node_name: str, kind: str, idx: int) -> list:
    base = node_name
    idx_padded = f"{idx:03d}"
    return [
        f"{base}-{kind}{idx_padded}$0",
        f"{base}-{kind}{idx}$0",
    ]


def _ensure_el_nodes_and_edges(board_DG):
    per = _group_el_by_node_and_index(board_DG)
    created_nodes = {"Tributary": 0, "XCON": 0, "Line": 0}
    created_edges = {"Tributary_XCON": 0, "XCON_Line": 0}

    def _get_or_create(kind: str, roadm: str, idx: int):
        for key in _normalize_variants(roadm, kind, idx):
            if board_DG.has_node(key):
                return key
        key = f"{roadm}-{kind}{idx:03d}$0"
        attrs = {"tType": kind, "layer": "electrical", "direction": "/"}
        board_DG.add_node(key, **attrs)
        created_nodes[kind] += 1
        return key

    for roadm, buckets in per.items():
        tribs = buckets["Tributary"]
        xcons = buckets["XCON"]
        lines = buckets["Line"]
        idxs = sorted(set(tribs) | set(xcons) | set(lines))
        for idx in idxs:
            trib = tribs.get(idx)
            xcon = xcons.get(idx)
            line = lines.get(idx)

            if not trib and (xcon or line):
                trib = _get_or_create("Tributary", roadm, idx)
                tribs[idx] = trib
            if not xcon and (trib or line):
                xcon = _get_or_create("XCON", roadm, idx)
                xcons[idx] = xcon
            if not line and (trib or xcon):
                line = _get_or_create("Line", roadm, idx)
                lines[idx] = line

            if not (trib and xcon and line):
                continue

            if not board_DG.has_edge(trib, xcon):
                board_DG.add_edge(trib, xcon, tType="Tributary_XCON", layer="electrical", direction="U")
                created_edges["Tributary_XCON"] += 1
            if not board_DG.has_edge(xcon, line):
                board_DG.add_edge(xcon, line, tType="XCON_Line", layer="electrical", direction="U")
                created_edges["XCON_Line"] += 1

    return {"created_nodes": created_nodes, "created_edges": created_edges}


def _prune_electrical_per_node(board_DG: nx.DiGraph,
                               target_min: int = 2,
                               target_max: int = 4,
                               mode: str = "random",
                               seed: Optional[int] = None):
    rnd = random.Random(seed)
    per = _group_el_by_node_and_index(board_DG)
    removed_nodes = {"Tributary": 0, "XCON": 0, "Line": 0}

    def _variants(roadm: str, kind: str, idx: int):
        idx_p = f"{idx:03d}"
        return [f"{roadm}-{kind}{idx_p}$0", f"{roadm}-{kind}{idx}$0"]

    for roadm, buckets in per.items():
        idxs = sorted(set(buckets.get("Tributary", {})) |
                      set(buckets.get("XCON", {})) |
                      set(buckets.get("Line", {})))
        if not idxs:
            continue
        kmin = max(1, int(target_min))
        kmax = max(kmin, int(target_max))
        keep_n = min(len(idxs), rnd.randint(kmin, kmax))
        keep = set(rnd.sample(idxs, k=keep_n)) if mode == "random" else set(idxs[:keep_n])
        drop = [i for i in idxs if i not in keep]
        for idx in drop:
            for kind in ("Tributary", "XCON", "Line"):
                for name in _variants(roadm, kind, idx):
                    if board_DG.has_node(name):
                        board_DG.remove_node(name)
                        removed_nodes[kind] += 1
    print(f"[Prune] Removed electrical nodes -> Tributary:{removed_nodes['Tributary']} XCON:{removed_nodes['XCON']} Line:{removed_nodes['Line']}")
    return {"removed_nodes": removed_nodes}



def _force_exact_el_per_node(board_DG: nx.DiGraph, exact: int = 2, mode: str = "random", seed: Optional[int] = None):
    rnd = random.Random(seed)
    per = _group_el_by_node_and_index(board_DG)

    def _create_triplet(roadm: str, idx: int):
        t = f"{roadm}-Tributary{idx:03d}$0"
        x = f"{roadm}-XCON{idx:03d}$0"
        l = f"{roadm}-Line{idx:03d}$0"
        if not board_DG.has_node(t):
            board_DG.add_node(t, tType="Tributary", layer="electrical", direction="/")
        if not board_DG.has_node(x):
            board_DG.add_node(x, tType="XCON", layer="electrical", direction="/")
        if not board_DG.has_node(l):
            board_DG.add_node(l, tType="Line", layer="electrical", direction="/")
        if not board_DG.has_edge(t, x):
            board_DG.add_edge(t, x, tType="Tributary_XCON", layer="electrical", direction="U")
        if not board_DG.has_edge(x, l):
            board_DG.add_edge(x, l, tType="XCON_Line", layer="electrical", direction="U")

    for roadm, buckets in per.items():
        idxs = sorted(set(buckets.get("Tributary", {})) |
                      set(buckets.get("XCON", {})) |
                      set(buckets.get("Line", {})))
        cur = len(idxs)

        if cur < exact:
            needed = exact - cur
            candidate = 1
            to_make = []
            while len(to_make) < needed:
                if candidate not in idxs and not board_DG.has_node(f"{roadm}-Line{candidate:03d}$0"):
                    to_make.append(candidate)
                candidate += 1
            for idx in to_make:
                _create_triplet(roadm, idx)
            idxs = sorted(idxs + to_make)

        if len(idxs) > exact:
            if mode == "head":
                keep = set(idxs[:exact])
            else:
                keep = set(rnd.sample(idxs, k=exact))
            drop = [i for i in idxs if i not in keep]
            for idx in drop:
                for kind in ("Tributary", "XCON", "Line"):
                    for name in (f"{roadm}-{kind}{idx:03d}$0", f"{roadm}-{kind}{idx}$0"):
                        if board_DG.has_node(name):
                            board_DG.remove_node(name)
    print(f"[Force] Enforced exactly {exact} electrical indices per ROADM.")



def _connect_el_to_optical(board_DG):
    added = 0

    om_by_node = {}
    for n in board_DG.nodes():
        m = _OM_PAT.match(n)
        if m:
            om_by_node.setdefault(m.group('node'), {})[m.group('side')] = n

    def _infer_side_from_idx(idx_str: str) -> str:
        if idx_str:
            if idx_str.startswith('1'):
                return '1'
            if idx_str.startswith('2'):
                return '2'
            try:
                return '1' if int(idx_str) % 2 == 1 else '2'
            except Exception:
                pass
        return '1'

    line_idx_map = {}
    for n in board_DG.nodes():
        m = _LINE_PAT.match(n)
        if m:
            line_idx_map[n] = m.group('idx')

    per = _group_el_by_node_and_index(board_DG)

    for roadm, buckets in per.items():
        om_sides = om_by_node.get(roadm, {})
        if not om_sides:
            continue
        tribs = buckets.get('Tributary', {})
        xcons = buckets.get('XCON', {})
        lines = buckets.get('Line', {})
        idxs = sorted(set(tribs) | set(xcons) | set(lines))
        for idx in idxs:
            line_node = lines.get(idx)
            side = None
            if line_node:
                side = _infer_side_from_idx(line_idx_map.get(line_node, ''))
            else:
                side = _infer_side_from_idx(str(idx))
            om_target = om_sides.get(side) or om_sides.get('1') or next(iter(om_sides.values()), None)
            if not om_target:
                continue

            trib = tribs.get(idx)
            if trib and not board_DG.has_edge(trib, om_target):
                board_DG.add_edge(trib, om_target, tType='EL_to_OM', layer='optical', direction=('U' if side=='1' else 'L'))
                added += 1
            xcon = xcons.get(idx)
            if xcon and not board_DG.has_edge(xcon, om_target):
                board_DG.add_edge(xcon, om_target, tType='EL_to_OM', layer='optical', direction=('U' if side=='1' else 'L'))
                added += 1

    optical_by_parent = {}
    for n, a in board_DG.nodes(data=True):
        if a.get('layer', '') == 'optical':
            parent = n.split('-')[0] if '-' in n else None
            if parent:
                optical_by_parent.setdefault(parent, []).append(n)

    for b, attr in list(board_DG.nodes(data=True)):
        if attr.get('layer') == 'electrical':
            parent = b.split('-')[0] if '-' in b else None
            if not parent:
                continue
            candidates = optical_by_parent.get(parent, [])
            if not candidates:
                continue
            has_optical_neighbor = any(
                (board_DG.has_edge(b, opt) or board_DG.has_edge(opt, b))
                for opt in candidates
            )
            if not has_optical_neighbor:
                target = candidates[0]
                board_DG.add_edge(b, target, tType='EL_to_Optical', layer='optical', direction='U')
                added += 1

    if added:
        print(f"[Fixup] Added {added} EL→Optical edges (EL_to_OM / EL_to_Optical)")
    else:
        print("[Fixup] All electrical boards already have optical neighbors")
    return added


import itertools

_LINE_PAT = re.compile(r"^(?P<node>[^-]+)-Line(?P<idx>\d+)\$0$")
_OM_PAT = re.compile(r"^(?P<node>[^-]+)-OM(?P<side>[12])\$0$")
_OD_PAT = re.compile(r"^(?P<node>[^-]+)-OD(?P<side>[12])\$0$")

def _ensure_line_optical_bridges(board_DG):
    added = 0
    om_by_node = {}
    od_by_node = {}
    for n in board_DG.nodes():
        m = _OM_PAT.match(n)
        if m:
            om_by_node.setdefault(m.group('node'), {})[m.group('side')] = n
            continue
        m = _OD_PAT.match(n)
        if m:
            od_by_node.setdefault(m.group('node'), {})[m.group('side')] = n

    lines_by_node_side = {}
    for n in board_DG.nodes():
        m = _LINE_PAT.match(n)
        if not m:
            continue
        roadm = m.group('node')
        idx = m.group('idx')
        side = '1' if idx.startswith('1') else ('2' if idx.startswith('2') else None)
        if side is None:
            try:
                side = '1' if int(idx) % 2 == 1 else '2'
            except Exception:
                side = '1'
        lines_by_node_side.setdefault(roadm, {}).setdefault(side, []).append(n)

    for roadm, by_side in lines_by_node_side.items():
        for side, line_nodes in by_side.items():
            om = om_by_node.get(roadm, {}).get(side)
            od = od_by_node.get(roadm, {}).get(side)
            if om:
                for ln in line_nodes:
                    if not board_DG.has_edge(ln, om):
                        board_DG.add_edge(ln, om, tType="Line_OM", layer="optical", direction=('U' if side=='1' else 'L'))
                        added += 1
            if od:
                for ln in line_nodes:
                    if not board_DG.has_edge(od, ln):
                        board_DG.add_edge(od, ln, tType="OD_Line", layer="optical", direction=('U' if side=='1' else 'L'))
                        added += 1
    if added:
        print(f"[Fixup] Added {added} Line↔Optical bridge edges (Line→OM and OD→Line).")
    else:
        print("[Fixup] No Line↔Optical bridge edges were added (already present).")
    return added

def _randomize_electrical_layer(board_DG: nx.DiGraph,
                                p_extra: float = 0.25,
                                max_extra_per_node: int = 2,
                                p_skip_triplet_edges: float = 0.1,
                                seed: Optional[int] = None):
    rnd = random.Random(seed)
    tgt_min = int(os.environ.get("E_TARGET_IDX_MIN", "4"))
    tgt_max = int(os.environ.get("E_TARGET_IDX_MAX", "12"))
    gap_span = int(os.environ.get("E_GAP_SPAN", "20"))
    if tgt_min < 1: tgt_min = 1
    if tgt_max < tgt_min: tgt_max = tgt_min

    per = _group_el_by_node_and_index(board_DG)
    added_nodes = added_edges = 0

    for roadm, buckets in per.items():
        existing_idxs = sorted(set(buckets.get("Tributary", {})) |
                               set(buckets.get("XCON", {})) |
                               set(buckets.get("Line", {})))
        target_count = rnd.randint(tgt_min, tgt_max)

        population = list(range(1, max(gap_span, target_count) + 1))
        rnd.shuffle(population)
        desired_idxs = sorted(population[:target_count])

        desired_set = set(desired_idxs) | set(existing_idxs)

        for idx in sorted(desired_set):
            t = f"{roadm}-Tributary{idx:03d}$0"
            x = f"{roadm}-XCON{idx:03d}$0"
            l = f"{roadm}-Line{idx:03d}$0"

            if not board_DG.has_node(t):
                board_DG.add_node(t, tType="Tributary", layer="electrical", direction="/")
                added_nodes += 1
            if not board_DG.has_node(x):
                board_DG.add_node(x, tType="XCON", layer="electrical", direction="/")
                added_nodes += 1
            if not board_DG.has_node(l):
                board_DG.add_node(l, tType="Line", layer="electrical", direction="/")
                added_nodes += 1

            if random.random() >= p_skip_triplet_edges and not board_DG.has_edge(t, x):
                board_DG.add_edge(t, x, tType="Tributary_XCON", layer="electrical", direction="U")
                added_edges += 1
            if random.random() >= p_skip_triplet_edges and not board_DG.has_edge(x, l):
                board_DG.add_edge(x, l, tType="XCON_Line", layer="electrical", direction="U")
                added_edges += 1

    if added_nodes or added_edges:
        print(f"[Randomizer] (counts) Ensured random per-ROADM indices. Added {added_nodes} nodes, {added_edges} edges "
              f"(targets per node ~{tgt_min}..{tgt_max}, gap_span={gap_span}).")
    else:
        print("[Randomizer] (counts) No changes needed (already met targets).")
        
def _randomize_line_to_om_sides(board_DG: nx.DiGraph, swap_prob: float = 0.15, seed: int | None = None):
    rnd = random.Random(seed)
    om_by_node = {}
    for n in board_DG.nodes():
        m = _OM_PAT.match(n)
        if m:
            om_by_node.setdefault(m.group('node'), {})[m.group('side')] = n

    toggled = 0
    for n in list(board_DG.nodes()):
        m = _LINE_PAT.match(n)
        if not m:
            continue
        roadm = m.group('node')
        idx = m.group('idx')
        side = '1' if idx.startswith('1') else ('2' if idx.startswith('2') else None)
        if side is None:
            try:
                side = '1' if int(idx) % 2 == 1 else '2'
            except Exception:
                side = '1'
        other = '2' if side == '1' else '1'
        if rnd.random() < swap_prob:
            om_other = om_by_node.get(roadm, {}).get(other)
            if om_other and not board_DG.has_edge(n, om_other):
                board_DG.add_edge(n, om_other, tType="Line_OM_rand", layer="optical", direction=('L' if other == '2' else 'U'))
                toggled += 1
    if toggled:
        print(f"[Randomizer] Added {toggled} extra Line→OM (cross-side) edges.")
    else:
        print("[Randomizer] No extra Line→OM cross-side edges added (random draw).")

def _connect_db():
    client = ArangoClient(hosts=ARANGO_URL)
    sys_db = client.db("_system", username=ARANGO_USER, password=ARANGO_PASS)
    if not sys_db.has_database(ARANGO_DB):
        sys_db.create_database(ARANGO_DB)
    return client.db(ARANGO_DB, username=ARANGO_USER, password=ARANGO_PASS)

def _name(base: str) -> str:
    return f"{COLL_PREFIX}{base}" if COLL_PREFIX else base

NODES_COLL = _name("Nodes")
BOARDS_COLL = _name("Boards")
NODE_CONNS_COLL = _name("Node_Connections")
NODES_CONTAINS_BOARDS_COLL = _name("Nodes_Contains_Boards")
FIBERS_U_COLL = _name("Fibers_U")
FIBERS_L_COLL = _name("Fibers_L")
FIBERS_ALL_COLL = _name("Fibers")
OMS_TRAILS_COLL = _name("OMS_Trails")

def _create_collections_and_graph(db):
    def create_collection(name, edge):
        if not db.has_collection(name):
            return db.create_collection(name, edge=edge)
        return db.collection(name)

    nodes = create_collection(NODES_COLL, False)
    boards = create_collection(BOARDS_COLL, False)
    node_connections = create_collection(NODE_CONNS_COLL, True)
    nodes_contains_boards = create_collection(NODES_CONTAINS_BOARDS_COLL, True)
    fibers_U = create_collection(FIBERS_U_COLL, True)
    fibers_L = create_collection(FIBERS_L_COLL, True)
    fibers_ALL = create_collection(FIBERS_ALL_COLL, True)
    oms_trails = create_collection(OMS_TRAILS_COLL, True)

    extra_board_colls = []
    try:
        if COLL_PREFIX and db.has_collection("Boards") and "Boards" != BOARDS_COLL:
            extra_board_colls.append("Boards")
    except Exception:
        pass

    boards_vertex_colls = [BOARDS_COLL] + extra_board_colls

    edge_defs = [
        {
            "edge_collection": NODE_CONNS_COLL,
            "from_vertex_collections": [NODES_COLL],
            "to_vertex_collections": [NODES_COLL],
        },
        {
            "edge_collection": NODES_CONTAINS_BOARDS_COLL,
            "from_vertex_collections": [NODES_COLL],
            "to_vertex_collections": [BOARDS_COLL],
        },
        {
            "edge_collection": FIBERS_U_COLL,
            "from_vertex_collections": boards_vertex_colls,
            "to_vertex_collections": boards_vertex_colls,
        },
        {
            "edge_collection": FIBERS_L_COLL,
            "from_vertex_collections": boards_vertex_colls,
            "to_vertex_collections": boards_vertex_colls,
        },
        {
            "edge_collection": FIBERS_ALL_COLL,
            "from_vertex_collections": boards_vertex_colls,
            "to_vertex_collections": boards_vertex_colls,
        },
        {
            "edge_collection": OMS_TRAILS_COLL,
            "from_vertex_collections": boards_vertex_colls,
            "to_vertex_collections": boards_vertex_colls,
        },
    ]
    if not db.has_graph(GRAPH_NAME):
        db.create_graph(GRAPH_NAME, edge_definitions=edge_defs)
    else:
        g = db.graph(GRAPH_NAME)
        existing = set(ed["edge_collection"] for ed in g.edge_definitions())
        for ed in edge_defs:
            if ed["edge_collection"] not in existing:
                g.create_edge_definition(
                    edge_collection=ed["edge_collection"],
                    from_vertex_collections=ed["from_vertex_collections"],
                    to_vertex_collections=ed["to_vertex_collections"],
                )

    return nodes, boards, node_connections, nodes_contains_boards, fibers_U, fibers_L, fibers_ALL, oms_trails

def _obj_to_doc(obj):
    if isinstance(obj, dict):
        d = dict(obj)
    elif hasattr(obj, "get_attributes") and callable(getattr(obj, "get_attributes")):
        try:
            d = dict(obj.get_attributes())
        except Exception:
            d = {}
    elif hasattr(obj, "__dict__"):
        d = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    else:
        return {"_key": str(obj), "name": str(obj)}
    if not d.get("_key"):
        d["_key"] = d.get("name") or str(d.get("ID") or d.get("id") or obj)
    if "name" not in d and d.get("_key"):
        d["name"] = d["_key"]
    return d

def save_topology_to_arango(node_list, board_DG, nc_list):
    db = _connect_db()

    if os.environ.get("ARANGO_RESET", "0") == "1":
        try:
            for cname in [NODES_COLL, BOARDS_COLL, NODE_CONNS_COLL, NODES_CONTAINS_BOARDS_COLL,
                          FIBERS_U_COLL, FIBERS_L_COLL, FIBERS_ALL_COLL, OMS_TRAILS_COLL]:
                if db.has_collection(cname):
                    db.collection(cname).truncate()
                    print(f"[ArangoDB] Truncated {cname}")
        except Exception as e:
            print("[ArangoDB] Reset requested but truncate failed:", e)

    nodes, boards, node_connections, nodes_contains_boards, fibers_U, fibers_L, fibers_ALL, oms_trails = _create_collections_and_graph(db)

    for ndoc in node_list:
        doc = _obj_to_doc(ndoc)
        if not doc.get("_key"):
            continue
        try:
            nodes.insert(doc)
        except DocumentInsertError:
            pass
        except Exception:
            pass

    for b, attr in board_DG.nodes(data=True):
        bdoc = {
            "_key": b,
            "name": b.split('-')[1] if '-' in b else b,
            "parentNodeName": b.split('-')[0] if '-' in b else "",
            "fullName": b,
            "tType": attr.get("tType"),
            "layer": attr.get("layer", "optical"),
            "direction": attr.get("direction", "/"),
        }
        try:
            boards.insert(bdoc)
        except DocumentInsertError:
            pass
        except Exception as e:
            pass

    for e in nc_list:
        try:
            if isinstance(e, dict) and "_from" in e and "_to" in e:
                node_connections.insert(e)
            elif hasattr(e, "connectedNodes"):
                cn = getattr(e, "connectedNodes", None)
                if cn and len(cn) >= 2:
                    edoc = {
                        "_from": f"{NODES_COLL}/{cn[0]}",
                        "_to":   f"{NODES_COLL}/{cn[1]}",
                        "name": getattr(e, "name", f"{cn[0]}_{cn[1]}")
                    }
                    node_connections.insert(edoc)
            else:
                edoc = _obj_to_doc(e)
                if edoc.get("_from") and edoc.get("_to"):
                    node_connections.insert(edoc)
        except DocumentInsertError:
            pass
        except Exception:
            pass

    for b in board_DG.nodes():
        parent = b.split('-')[0] if '-' in b else None
        if parent:
            edoc = {"_from": f"{NODES_COLL}/{parent}", "_to": f"{BOARDS_COLL}/{b}"}
            try:
                nodes_contains_boards.insert(edoc)
            except DocumentInsertError:
                pass

    for u, v, data in board_DG.edges(data=True):
        direction = data.get("direction")
        edoc_common = {
            "_from": f"{BOARDS_COLL}/{u}",
            "_to": f"{BOARDS_COLL}/{v}",
            "name": f"{u}_{v}",
            "fromNode": u.split('-')[0] if '-' in u else "",
            "toNode": v.split('-')[0] if '-' in v else "",
            "fromBoard": u,
            "toBoard": v,
            "tType": data.get("tType"),
            "layer": data.get("layer", "optical"),
            "direction": direction or "/",
        }
        if direction == "U":
            try:
                fibers_U.insert(edoc_common)
            except DocumentInsertError:
                pass
            except Exception:
                pass
            try:
                fibers_ALL.insert(edoc_common)
            except DocumentInsertError:
                pass
            except Exception:
                pass
        elif direction == "L":
            try:
                fibers_L.insert(edoc_common)
            except DocumentInsertError:
                pass
            except Exception:
                pass
            try:
                fibers_ALL.insert(edoc_common)
            except DocumentInsertError:
                pass
            except Exception:
                pass
        else:
            try:
                fibers_U.insert(edoc_common)
            except DocumentInsertError:
                pass
            except Exception:
                pass
            try:
                fibers_ALL.insert(edoc_common)
            except DocumentInsertError:
                pass
            except Exception:
                pass

def main():
    graphml_in = os.environ.get("TOPO_GRAPHML_IN")
    if graphml_in:
        board_path = Path(graphml_in)
        if not board_path.exists():
            raise SystemExit(f"Missing TOPO_GRAPHML_IN: {board_path}")
        board_DG = nx.read_graphml(board_path)
        if not isinstance(board_DG, nx.DiGraph):
            board_DG = nx.DiGraph(board_DG)
        node_DG = _node_graph_from_board(board_DG)
        node_list, nc_list = [], []
        print(f"[GraphML] Loaded {board_path} (boards={len(board_DG.nodes())}, fibers={len(board_DG.edges())})")
    else:
        node_DG = generate_topology(num_nodes=5, ROADM_ratio=1.0, max_degree=2)
        node_list, nc_list, board_DG = process_traffic_from_nodeDG(node_DG)
        summary = _ensure_el_nodes_and_edges(board_DG)
        if summary["created_nodes"]["XCON"] or summary["created_nodes"]["Line"]:
            print("[Fixup] Created missing nodes:", summary["created_nodes"])
        if summary["created_edges"]["Tributary_XCON"] or summary["created_edges"]["XCON_Line"]:
            print("[Fixup] Added missing EL edges:", summary["created_edges"])
        if os.environ.get("E_RANDOM", "1") == "1":
            p_extra = float(os.environ.get("E_P_EXTRA", "0.25"))
            max_extra = int(os.environ.get("E_MAX_EXTRA", "2"))
            p_skip = float(os.environ.get("E_P_SKIP_TRIPLET", "0.10"))
            seed = os.environ.get("E_SEED")
            seed = int(seed) if seed not in (None, "") else None
            _randomize_electrical_layer(board_DG,
                                        p_extra=p_extra,
                                        max_extra_per_node=max_extra,
                                        p_skip_triplet_edges=p_skip,
                                        seed=seed)
            swap_prob = float(os.environ.get("E_SWAP_PROB", "0.15"))
            _randomize_line_to_om_sides(board_DG, swap_prob=swap_prob, seed=seed)
        if os.environ.get("E_PRUNE", "1") == "1":
            prune_min = int(os.environ.get("E_PRUNE_MIN", "2"))
            prune_max = int(os.environ.get("E_PRUNE_MAX", "4"))
            prune_mode = os.environ.get("E_PRUNE_MODE", "random").lower()
            seed = os.environ.get("E_SEED")
            seed = int(seed) if seed not in (None, "") else None
            _prune_electrical_per_node(board_DG, target_min=prune_min, target_max=prune_max, mode=prune_mode, seed=seed)

        e_mode = os.environ.get("E_FORCE_MODE", "random").lower()
        seed = os.environ.get("E_SEED")
        seed = int(seed) if seed not in (None, "") else None
        _force_exact_el_per_node(board_DG, exact=2, mode=e_mode, seed=seed)

        _audit_el_pairs(board_DG)
        _connect_el_to_optical(board_DG)
        _ensure_line_optical_bridges(board_DG)
        _audit_el_pairs(board_DG)

    out_dir = Path(os.environ.get("TOPO_PREVIEW_DIR", "outputs/topology_preview"))
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_board_list(node_DG, board_DG, folderPath=str(out_dir))
    nx.write_graphml(board_DG, out_dir / "board.graphml")

    print(f"Nodes in node_DG: {len(node_DG.nodes())}")
    print(f"Boards in board_DG: {len(board_DG.nodes())}, fibers: {len(board_DG.edges())}")
    print("Wrote:", out_dir / "board.graphml", "and a board_list_...txt in", out_dir)

    with (out_dir / "edges.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fromBoard","toBoard","direction","layer","tType"])
        for u, v, d in board_DG.edges(data=True):
            w.writerow([u, v, d.get("direction",""), d.get("layer",""), d.get("tType","")])
    print("Wrote:", out_dir / "edges.csv")

    export_lightpaths(board_DG, out_dir / "lightpaths.txt", limit_per_node=8)

    regen_hops = _regen_hops_from_env()
    if regen_hops:
        nx.write_graphml(board_DG, out_dir / "board_regen.graphml")
        with (out_dir / "edges_regen.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fromBoard","toBoard","direction","layer","tType"])
            for u, v, d in board_DG.edges(data=True):
                w.writerow([u, v, d.get("direction",""), d.get("layer",""), d.get("tType","")])
        print("Wrote:", out_dir / "board_regen.graphml")



    if os.environ.get("ARANGO_SKIP", "1") != "1":
        try:
            save_topology_to_arango(node_list, board_DG, nc_list)
            print(f"[ArangoDB] Saved to db='{ARANGO_DB}', graph='{GRAPH_NAME}', "
                  f"collections prefix='{COLL_PREFIX}'")
            print(f"[ArangoDB] Open {ARANGO_URL} → DB: {ARANGO_DB} → Graphs: {GRAPH_NAME}")
        except Exception as e:
            print("[ArangoDB] Skipped or failed to save topology:", e)
            print("           Tips:")
            print("            - Ensure ArangoDB is running (e.g., Docker).")
            print("            - Set ARANGO_PASS if your root has a password.")
            print("            - pip install python-arango")
            print("            - Check host/port via ARANGO_URL env var.")

        try:
            db = _connect_db()
            print("[ArangoDB] Connected:", ARANGO_URL, "DB:", ARANGO_DB)
            coll_names = [NODES_COLL, BOARDS_COLL, NODE_CONNS_COLL, NODES_CONTAINS_BOARDS_COLL, FIBERS_U_COLL, FIBERS_L_COLL, FIBERS_ALL_COLL, OMS_TRAILS_COLL]
            for cname in coll_names:
                try:
                    c = db.collection(cname)
                    print(f"[ArangoDB] {cname}: {c.count()} docs")
                except Exception as ce:
                    print(f"[ArangoDB] {cname}: not found ({ce})")
            try:
                print(f"[ArangoDB] Sampling edges from {FIBERS_ALL_COLL} …")
                res = db.aql.execute(q, bind_vars={"@coll": FIBERS_ALL_COLL})
                sampled = 0
                for doc in res:
                    print("   ", doc)
                    sampled += 1
                if sampled == 0:
                    print(f"[ArangoDB] No documents found in {FIBERS_ALL_COLL}. If the graph UI shows 0 edges, ensure the collection is enabled in the Explore panel and that direction/layer filtering didn't exclude edges.")
            except Exception as qe:
                print("[ArangoDB] AQL sample failed:", qe)
            if db.has_graph(GRAPH_NAME):
                g = db.graph(GRAPH_NAME)
                print(f"[ArangoDB] Graph '{GRAPH_NAME}' exists. Edge defs:")
                for ed in g.edge_definitions():
                    print("   -", ed)
            else:
                print(f"[ArangoDB] Graph '{GRAPH_NAME}' NOT found.")
            print("[ArangoDB] Tip: In Graphs → electrical_layer_topography, enable edge collection", FIBERS_ALL_COLL, "to see board-to-board connections.")
        except Exception as ve:
            print("[ArangoDB] Post-save verification failed:", ve)

if __name__ == "__main__":
    main()
