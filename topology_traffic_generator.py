import os
import random
from topology_traffic_processor import *
import networkx as nx

def get_electrical_endpoints(board_DG):
    return [n for n, d in board_DG.nodes(data=True)
            if str(d.get("tType", "")).startswith("Tributary")
            or str(n).split("-")[-1].startswith("Tributary")]

def pick_line_index_for_side(side):
    return 1 if side == "U" else 2

def make_electrical_triplet(node_name, side):
    idx = pick_line_index_for_side(side)
    line = f"{node_name}-Line{idx}$0"
    xcon = f"{node_name}-XCON{idx}$0"
    trib = f"{node_name}-Tributary{idx}$0"
    ingress = [trib, xcon, line]
    egress = [line, xcon, trib]
    return ingress, egress

def infer_side_for_node(board_DG, node_name):
    om1 = f"{node_name}-OM1$0"
    om2 = f"{node_name}-OM2$0"
    od1 = f"{node_name}-OD1$0"
    od2 = f"{node_name}-OD2$0"
    has_u = any(True for _ in board_DG.successors(od1)) or board_DG.has_edge(od1, om1) or board_DG.has_edge(om1, od1)
    has_l = any(True for _ in board_DG.successors(od2)) or board_DG.has_edge(od2, om2) or board_DG.has_edge(om2, od2)
    if has_u and not has_l:
        return "U"
    if has_l and not has_u:
        return "L"
    return "U"

def _assert_simple_board_path(board_DG, path_list):
    if not nx.is_simple_path(board_DG, path_list):
        raise ValueError(f"Board path is not simple or does not exist: {path_list}")

def _regen_hops_from_env() -> int:
    try:
        return max(0, int(os.environ.get("REGEN_HOPS", "0")))
    except ValueError:
        return 0

def _is_optical(board_DG, name: str) -> bool:
    layer = board_DG.nodes.get(name, {}).get("layer")
    return layer is None or layer == "optical"

def _add_regen_node(board_DG, name: str, kind: str) -> None:
    if board_DG.has_node(name):
        return
    board_DG.add_node(name)
    nx.set_node_attributes(board_DG, {
        name: {
            "tType": kind,
            "layer": "optical",
            "direction": "/",
            "is_regen_boundary": True,
        }
    })

def _add_regen_edge(board_DG, src: str, dst: str, base_edge: tuple[str, str], kind: str | None = None) -> None:
    if board_DG.has_edge(src, dst):
        return
    base = board_DG.get_edge_data(base_edge[0], base_edge[1]) or board_DG.get_edge_data(base_edge[1], base_edge[0]) or {}
    attrs = {
        "layer": base.get("layer", "optical"),
        "direction": base.get("direction", "/"),
    }
    if kind:
        attrs["tType"] = kind
    board_DG.add_edge(src, dst)
    nx.set_edge_attributes(board_DG, {(src, dst): attrs})

def _insert_regen_nodes(board_DG, optical_span: list[str], regen_hops: int, path_id: int) -> list[str]:
    if regen_hops <= 0 or len(optical_span) < (regen_hops + 2):
        return optical_span
    out = [optical_span[0]]
    hop = 0
    regen_idx = 0
    for i in range(1, len(optical_span)):
        prev = optical_span[i - 1]
        curr = optical_span[i]
        hop += 1
        if hop % regen_hops == 0 and i < len(optical_span) - 1:
            if _is_optical(board_DG, prev) and _is_optical(board_DG, curr):
                regen_idx += 1
                node_name = curr.split("-")[0]
                tag = f"{path_id:03d}{regen_idx:02d}"
                regen_in = f"{node_name}-RegenIn{tag}$0"
                regen_out = f"{node_name}-RegenOut{tag}$0"
                _add_regen_node(board_DG, regen_in, "RegenIn")
                _add_regen_node(board_DG, regen_out, "RegenOut")
                _add_regen_edge(board_DG, prev, regen_in, (prev, curr))
                _add_regen_edge(board_DG, regen_in, regen_out, (prev, curr), kind="regen_boundary")
                _add_regen_edge(board_DG, regen_out, curr, (prev, curr))
                out.extend([regen_in, regen_out, curr])
                continue
        out.append(curr)
    return out


def generate_single_lightpath(node_DG, board_DG, all_OTU_dict, path_id=0, regen_hops=0):
    eps = get_electrical_endpoints(board_DG)
    if len(eps) < 2:
        return [], -1
    src_board, dst_board = random.sample(eps, 2)
    src_node = src_board.split('-')[0]
    dst_node = dst_board.split('-')[0]

    try:
        nodePath = next(nx.shortest_simple_paths(node_DG, src_node, dst_node))
    except StopIteration:
        return [], -1

    src_side = infer_side_for_node(board_DG, src_node)
    dst_side = infer_side_for_node(board_DG, dst_node)
    ingress_triplet, _ = make_electrical_triplet(src_node, src_side)
    _, egress_triplet = make_electrical_triplet(dst_node, dst_side)

    src_line = ingress_triplet[-1]
    dst_line = egress_triplet[0]
    try:
        optical_span = next(nx.shortest_simple_paths(board_DG, src_line, dst_line))
    except StopIteration:
        return [], -1

    optical_span = _insert_regen_nodes(board_DG, optical_span, regen_hops, path_id)
    boardPath = ingress_triplet + optical_span + egress_triplet

    _assert_simple_board_path(board_DG, boardPath)
    return boardPath, 0


def generate_topology(num_nodes, ROADM_ratio, max_degree=None):
    num_ROADMs = int(num_nodes * ROADM_ratio)
    node_G = nx.fast_gnp_random_graph(num_ROADMs, p=0.72)

    if max_degree is not None:
        all_nodes = list(node_G.nodes)
        random.shuffle(all_nodes)
        for node in all_nodes:
            if node_G.degree[node] > max_degree:
                deleted_num = node_G.degree[node] - max_degree
                node_edges = list(node_G[node])
                sorted_edges = sorted(node_edges, key=lambda n: node_G.degree[n], reverse=True)
                deleted_neighbors = sorted_edges[:deleted_num]
                for d in deleted_neighbors:
                    node_G.remove_edge(node, d)

    node_DG = nx.DiGraph()
    for node in node_G.nodes:
        node_DG.add_node('ROADM' + str(node))
    for edge in node_G.edges:
        node_DG.add_edge('ROADM' + str(edge[0]), 'ROADM' + str(edge[1]))

    num_OLAs = num_nodes - num_ROADMs
    edges_inserted = random.sample(list(node_DG.edges), num_OLAs)
    for i in range(num_OLAs):
        node_DG.add_node('OLA' + str(num_ROADMs + i))
        node_DG.add_edge(edges_inserted[i][0], 'OLA' + str(num_ROADMs + i))
        node_DG.add_edge('OLA' + str(num_ROADMs + i), edges_inserted[i][1])
        node_DG.remove_edge(edges_inserted[i][0], edges_inserted[i][1])

    return node_DG


def generate_lightpaths(node_DG, board_DG, all_OTU_dict, num_lightpaths):
    list_lightpaths = []
    attempts = 0
    regen_hops = _regen_hops_from_env()
    while len(list_lightpaths) < num_lightpaths and attempts < num_lightpaths * 10:
        attempts += 1
        lightpath, _direction_index = generate_single_lightpath(
            node_DG,
            board_DG,
            all_OTU_dict,
            path_id=len(list_lightpaths),
            regen_hops=regen_hops,
        )
        if lightpath:
            list_lightpaths.append(lightpath)
            print(len(list_lightpaths), datetime.now().strftime("%H:%M:%S"))
    if len(list_lightpaths) < num_lightpaths:
        print(f"[warn] Only generated {len(list_lightpaths)} lightpaths out of requested {num_lightpaths}.")
    return list_lightpaths, all_OTU_dict


def generate_static_topology_traffic(num_nodes, num_lightpaths, ROADM_ratio=1):
    node_DG = generate_topology(num_nodes, ROADM_ratio)
    node_list, nc_list, board_DG = process_traffic_from_nodeDG(node_DG)
    write_topo_info(["Number of nodes: " + str(node_DG.number_of_nodes()),
                     "Number of Node connections: " + str(node_DG.number_of_edges()),
                     "Average degree (node-level): " + str(get_average_degree(node_DG)),
                     "Number of boards: " + str(board_DG.number_of_nodes()),
                     "Number of fibers: " + str(board_DG.number_of_edges()),
                     "Average degree (board-level): " + str(get_average_degree(board_DG)), ""],
                    "outputs/static_topology/whole topology info.txt")

    lightpaths, all_OTU_dict = generate_lightpaths(node_DG, board_DG, None, num_lightpaths)
    filepath = 'outputs/static_topology/topology_' + str(num_nodes) + '_nodes_' + str(num_lightpaths) + '_lightpaths.txt'
    with open(filepath, 'w') as file:
        for lightpath in lightpaths:
            lightpath = ','.join(lightpath)
            file.write(lightpath)
            file.write("\n")
    file.close()
    return lightpaths, filepath
