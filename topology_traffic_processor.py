import os
from datetime import datetime

import networkx as nx

from class_obj import Node, Node_Connection, Lightpath, num_copy, OMS_Layer, OTS_Layer, OCh_Layer, OTU_num

def _translate_otu_tokens_to_line(board_list):
    out = []
    for i, name in enumerate(board_list):
        if "-OTU" in name:
            node = name.split('-')[0]
            side = None
            if i + 1 < len(board_list):
                nxt = board_list[i + 1]
                if "OM2" in nxt:
                    side = "L"
                elif "OM1" in nxt:
                    side = "U"
            if side is None and i - 1 >= 0:
                prv = board_list[i - 1]
                if "OD2" in prv:
                    side = "L"
                elif "OD1" in prv:
                    side = "U"
            if side is None:
                side = "U"
            idx = "2" if side == "L" else "1"
            name = f"{node}-Line{idx}$0"
        out.append(name)
    return out

def _is_regen_node(name: str) -> bool:
    return "-RegenIn" in name or "-RegenOut" in name

def _ensure_regen_nodes(board_DG: nx.DiGraph, lightpath_list) -> None:
    for lightpath in lightpath_list:
        path = lightpath.boardPath
        for name in path:
            if _is_regen_node(name) and not board_DG.has_node(name):
                board_DG.add_node(name)
                nx.set_node_attributes(board_DG, {
                    name: {
                        "tType": "Regen",
                        "layer": "optical",
                        "direction": "/",
                        "is_regen_boundary": True,
                    }
                })
        for u, v in zip(path, path[1:]):
            if board_DG.has_edge(u, v):
                continue
            if _is_regen_node(u) or _is_regen_node(v):
                attrs = {
                    "layer": board_DG.nodes.get(u, {}).get("layer", "optical"),
                    "direction": "/",
                }
                board_DG.add_edge(u, v)
                nx.set_edge_attributes(board_DG, {(u, v): attrs})


def write_topo_info(info, filePath, rm=True):
    if rm and os.path.exists(filePath):
        os.remove(filePath)

    with open(filePath, 'a') as file:
        for line in info:
            file.write(line)
            file.write("\n")
    file.close()


def generate_board_list(node_DG, board_DG, folderPath):
    out_path = folderPath + '/board_list_' + str(len(node_DG.nodes)) + '_nodes_' + str(
        len(board_DG.nodes) + len(board_DG.edges)) + '_boards.txt'
    with open(out_path, 'w') as f:
        node_attrs = nx.get_node_attributes(board_DG, None)
        for board in board_DG.nodes:
            bid = nx.get_node_attributes(board_DG, "ID").get(board, "")
            layer = board_DG.nodes[board].get("layer", "optical")
            f.write(f"{board},{bid},{layer}\n")
        for u, v in board_DG.edges:
            eid = nx.get_edge_attributes(board_DG, "ID").get((u, v), "")
            layer = board_DG.edges[(u, v)].get("layer", "optical")
            f.write(f"{u}_{v},{eid},{layer}\n")


def process_traffic_from_nodeDG(node_DG):
    node_list = []
    nc_list = []
    all_nodes = list(node_DG.nodes)
    for n in all_nodes:
        node_list.append(Node(n))

    all_edges = list(node_DG.edges)
    for e in all_edges:
        nc_list.append(Node_Connection(e))

    board_DG = nx.DiGraph()
    for node in node_list:
        for board in node.boards.values():
            board_DG.add_node(board.fullName)
            nx.set_node_attributes(board_DG, {board.fullName: board.get_attributes()})
        for fiber in node.fibers.values():
            board_DG.add_edge(fiber.fullName.split('_')[0], fiber.fullName.split('_')[1])
            nx.set_edge_attributes(board_DG, {tuple(fiber.fullName.split('_')): fiber.get_attributes()})
    for edge in nc_list:
        for fiber in edge.element.values():
            board_DG.add_edge(fiber.fullName.split('_')[0], fiber.fullName.split('_')[1])
            nx.set_edge_attributes(board_DG, {tuple(fiber.fullName.split('_')): fiber.get_attributes()})

    return node_list, nc_list, board_DG




def process_traffic_from_str(lines):
    print("Processing traffic:", datetime.now().strftime("%H:%M:%S"))

    lightpath_list = []
    for i in range(len(lines)):
        lightpath = Lightpath(lines[i])
        lightpath_list.append(lightpath)

    for lightpath in lightpath_list:
        translated = _translate_otu_tokens_to_line(lightpath.boardPath)
        if translated != lightpath.boardPath:
            lightpath.boardPath = translated
            lightpath.fromBoard = translated[0]
            lightpath.toBoard   = translated[-1]

    node_DG = nx.DiGraph()
    for lightpath in lightpath_list:
        nx.add_path(node_DG, lightpath.nodePath)

    node_list, nc_list, board_DG = process_traffic_from_nodeDG(node_DG)
    _ensure_regen_nodes(board_DG, lightpath_list)

    for lightpath in lightpath_list:
        if not nx.is_simple_path(board_DG, lightpath.boardPath):
            raise Exception(lightpath.boardPath, "not simple path/not exist")

    print("Processing traffic finished", datetime.now().strftime("%H:%M:%S"))
    return node_list, nc_list, lightpath_list, node_DG, board_DG


def process_traffic(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].strip()

    node_list, nc_list, lightpath_list, node_DG, board_DG = process_traffic_from_str(lines)
    board_DG, unused_OTUs_count = remove_unused_OTUs(lightpath_list, board_DG)

    lightpath_len_list = [len(lightpath.boardPath) for lightpath in lightpath_list]
    folder_path = '/'.join(filepath.split('/')[:-1])
    write_topo_info(["Number of nodes: " + str(node_DG.number_of_nodes()),
                     "Average degree (board-level): " + str(get_average_degree(board_DG)),
                     "Number of lightpaths: " + str(len(lightpath_list)),
                     "Avg lightpath length (board-level): " + str(sum(lightpath_len_list) / len(lightpath_len_list)),
                     "Max lightpath length (board-level):" + str(max(lightpath_len_list)),
                     "Min lightpath length (board-level):" + str(min(lightpath_len_list)),
                     "Number of OTUs removed: " + str(unused_OTUs_count), ""],
                    folder_path + "/topology info.txt")

    generate_board_list(node_DG, board_DG, folder_path)
    return node_list, nc_list, lightpath_list, node_DG, board_DG








def get_average_degree(DG):
    G = DG.to_undirected()
    degree_list = list(G.degree)
    for i in range(len(degree_list)):
        degree_list[i] = degree_list[i][1]
    average_degree = sum(degree_list) / len(degree_list)
    return average_degree




def get_all_OTUs(node_list):
    OTU_upper_start_list = {}
    OTU_upper_end_list = {}
    OTU_lower_start_list = {}
    OTU_lower_end_list = {}
    for node in node_list:
        if node.nodeType == 'ROADM':
            OTU_upper_end_list[node.name] = []
            OTU_upper_start_list[node.name] = []
            OTU_lower_end_list[node.name] = []
            OTU_lower_start_list[node.name] = []
            for i in range(1, OTU_num * 4 + 1):
                for j in range(num_copy):
                    if i % 4 == 1:
                        OTU_upper_end_list[node.name].append(node.name + '-OTU' + str(i) + '$' + str(j))
                    elif i % 4 == 2:
                        OTU_upper_start_list[node.name].append(node.name + '-OTU' + str(i) + '$' + str(j))
                    elif i % 4 == 3:
                        OTU_lower_end_list[node.name].append(node.name + '-OTU' + str(i) + '$' + str(j))
                    else:
                        OTU_lower_start_list[node.name].append(node.name + '-OTU' + str(i) + '$' + str(j))
    return [OTU_upper_start_list, OTU_upper_end_list, OTU_lower_start_list, OTU_lower_end_list]


def remove_unused_OTUs(lightpath_list, board_DG):
    used_OTUs = []
    for lightpath in lightpath_list:
        used_OTUs.append(lightpath.boardPath[0])
        used_OTUs.append(lightpath.boardPath[-1])

    if len(list(set(used_OTUs))) != len(used_OTUs):
        raise Exception("Duplicate OTU", used_OTUs)

    board_list = list(board_DG.nodes)
    unused_OTUs_count = 0
    for board in board_list:
        if 'OTU' in board and board not in used_OTUs:
            board_DG.remove_node(board)
            unused_OTUs_count += 1

    board_ID = 0
    for board in board_DG.nodes:
        nx.set_node_attributes(board_DG, {board: {"ID": board_ID}})
        board_ID += 1
    for fiber in board_DG.edges:
        nx.set_edge_attributes(board_DG, {fiber: {"ID": board_ID}})
        board_ID += 1

    return board_DG, unused_OTUs_count



def _add_electrical_boards_for_node(board_DG, node_name):
    b_tr = f"{node_name}-Tributary"
    b_xc = f"{node_name}-XCON"
    b_li = f"{node_name}-Line"

    for b in (b_tr, b_xc, b_li):
        if b not in board_DG:
            board_DG.add_node(b, layer="electrical", board_type=b.split("-")[-1])

    board_DG.add_edge(b_tr, b_xc, direction="U")
    board_DG.add_edge(b_xc, b_tr, direction="L")
    board_DG.add_edge(b_xc, b_li, direction="U")
    board_DG.add_edge(b_li, b_xc, direction="L")
def process_trails(lightpath_list, node_DG, board_DG):
    OTS_layer_list = []
    OMS_layer_list = []
    OCh_layer_list = []

    for i in range(len(lightpath_list)):
        lightpath = lightpath_list[i]
        OCh_layer = OCh_Layer(lightpath.fromNode + '-' + lightpath.fromBoard, lightpath.toNode + '-' + lightpath.toBoard, i)
        OCh_layer_list.append(OCh_layer)

    OTS_layer_index = 0
    fiber_list = board_DG.edges
    for fb in fiber_list:
        if not ('OTU' in fb[0] or 'OTU' in fb[1]):
            OTS_layer = OTS_Layer(fb[0], fb[1], OTS_layer_index)
            OTS_layer_list.append(OTS_layer)
            OTS_layer_index += 1

    OMS_layer_index = 0
    node_list = node_DG.nodes
    ROADM_list = list(filter(lambda s: 'ROADM' in s, node_list))
    for node_i in ROADM_list:
        for node_j in ROADM_list:
            if node_i != node_j and nx.has_path(node_DG, node_i, node_j):
                path = next(nx.shortest_simple_paths(node_DG, node_i, node_j))
                if not any('ROADM' in middle for middle in path[1: -1]):
                    for i in range(num_copy):
                        OMS_layer = OMS_Layer(node_i + '-OM1$' + str(i), node_j + '-OD1$' + str(i), OMS_layer_index)
                        OMS_layer_list.append(OMS_layer)
                        OMS_layer_index += 1
                        if not nx.has_path(board_DG, node_i + '-OM1$' + str(i), node_j + '-OD1$' + str(i)):
                            raise Exception('No path between', node_i + '-OM1$' + str(i), node_j + '-OD1$' + str(i))

                        OMS_layer = OMS_Layer(node_j + '-OM2$' + str(i), node_i + '-OD2$' + str(i), OMS_layer_index)
                        OMS_layer_list.append(OMS_layer)
                        OMS_layer_index += 1
                        if not nx.has_path(board_DG, node_j + '-OM2$' + str(i), node_i + '-OD2$' + str(i)):
                            raise Exception('No path between', node_j + '-OM2$' + str(i), node_i + '-OD2$' + str(i))

    return OTS_layer_list, OMS_layer_list, OCh_layer_list
