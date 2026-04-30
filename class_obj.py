import os

ELECTRICAL_GROUPS = int(os.getenv("ELECTRICAL_GROUPS", "29"))

OTU_num = ELECTRICAL_GROUPS

num_copy = int(os.getenv("NUM_COPIES", "1"))
LAYER_OPTICAL = "optical"
LAYER_ELECTRICAL = "electrical"


ELECTRICAL_BOARD_TYPES = ("Tributary", "XCON", "Line")

def _translate_otu_tokens_to_line(board_list):
    out = []
    for i, name in enumerate(board_list):
        if "-OTU" in name:
            node = name.split('-')[0]
            copy = name.split('$')[-1] if '$' in name else "0"
            side = None
            if i + 1 < len(board_list):
                nxt = board_list[i + 1]
                if "-OM2" in nxt or "-OD2" in nxt:
                    side = "L"
                elif "-OM1" in nxt or "-OD1" in nxt:
                    side = "U"
            if side is None and i - 1 >= 0:
                prv = board_list[i - 1]
                if "-OM2" in prv or "-OD2" in prv:
                    side = "L"
                elif "-OM1" in prv or "-OD1" in prv:
                    side = "U"
            if side is None:
                side = "U"
            idx = "2" if side == "L" else "1"
            name = f"{node}-Line{idx}${copy}"
        out.append(name)
    return out


def get_board_fiber_type_dict(parentName):
    if "ROADM" in parentName:
        board_type_dict = {
            "OA": 4, "FIU": 2, "OD": 2, "OM": 2,
            "Tributary": ELECTRICAL_GROUPS * 4,
            "XCON": ELECTRICAL_GROUPS * 4,
            "Line": ELECTRICAL_GROUPS * 4,
        }
        fiber_type_dict = {
            "FIU_OA": ["FIU1_OA1", "FIU2_OA3"],
            "OA_OD": ["OA1_OD1", "OA3_OD2"],
            "OD_OM": ["OD1_OM1", "OD2_OM2"],
            "OM_OA": ["OM1_OA2", "OM2_OA4"],
            "OA_FIU": ["OA2_FIU2", "OA4_FIU1"],
            "Tributary_XCON": [],
            "XCON_Line": [],
            "OD_Line": [],
            "Line_OM": []
        }
        for i in range(ELECTRICAL_GROUPS):
            for k in range(4):
                idx = 1 + i * 4 + k
                fiber_type_dict["Tributary_XCON"].append(f"Tributary{idx}_XCON{idx}")
                fiber_type_dict["XCON_Line"].append(f"XCON{idx}_Line{idx}")
                if k in (0, 2):
                    fiber_type_dict["OD_Line"].append(f"OD1_Line{idx}")
                    fiber_type_dict["Line_OM"].append(f"Line{idx}_OM1")
                if k in (1, 3):
                    fiber_type_dict["OD_Line"].append(f"OD2_Line{idx}")
                    fiber_type_dict["Line_OM"].append(f"Line{idx}_OM2")
    else:
        board_type_dict = {"OA": 2, "FIU": 2}
        fiber_type_dict = {
            "FIU_OA": ["FIU1_OA1", "FIU2_OA2"],
            "OA_FIU": ["OA1_FIU2", "OA2_FIU1"],
        }
    return board_type_dict, fiber_type_dict


def node_init(parentName):
    board_type_dict, fiber_type_dict = get_board_fiber_type_dict(parentName)

    boards = {}
    fibers = {}
    for i in range(num_copy):
        for board_type in board_type_dict.keys():
            for j in range(board_type_dict[board_type]):
                board = Board(board_type + str(j + 1) + "$" + str(i), parentName)
                boards[board.name] = board

        for fiber_type in fiber_type_dict.keys():
            for f in fiber_type_dict[fiber_type]:
                fiber = Fiber(parentName, f.split('_')[0] + "$" + str(i), parentName, f.split('_')[1] + "$" + str(i))
                fibers[fiber.name] = fiber

    return boards, fibers


def node_connection_init(connected_nodes):
    fibers = {}
    for i in range(num_copy):
        # East fiber pair: A.FIU2 ↔ B.FIU1
        fiber1 = Fiber(connected_nodes[0], 'FIU2$' + str(i), connected_nodes[1], 'FIU1$' + str(i))
        fiber2 = Fiber(connected_nodes[1], 'FIU1$' + str(i), connected_nodes[0], 'FIU2$' + str(i))
        # West fiber pair: A.FIU1 ↔ B.FIU2
        fiber3 = Fiber(connected_nodes[0], 'FIU1$' + str(i), connected_nodes[1], 'FIU2$' + str(i))
        fiber4 = Fiber(connected_nodes[1], 'FIU2$' + str(i), connected_nodes[0], 'FIU1$' + str(i))
        # Use fullName as key to avoid collisions (name lacks node prefix)
        fibers[fiber1.fullName] = fiber1
        fibers[fiber2.fullName] = fiber2
        fibers[fiber3.fullName] = fiber3
        fibers[fiber4.fullName] = fiber4
    return fibers


def get_direction(name):
    if '_' in name:
        from_board = name.split('_')[0]
        to_board = name.split('_')[1]
        if 'FIU2' in from_board and 'FIU1' in to_board:
            return 'U'
        elif 'FIU1' in from_board and 'FIU2' in to_board:
            return 'L'
        elif 'ROADM' in from_board:
            direction_dict = {("FIU1", "OA1"): 'U', ("OA1", "OD1"): 'U', ("OD1", "OM1"): 'U', ("OM1", "OA2"): 'U',
                              ("OA2", "FIU2"): 'U',
                              ("FIU2", "OA3"): 'L', ("OA3", "OD2"): 'L', ("OD2", "OM2"): 'L', ("OM2", "OA4"): 'L',
                              ("OA4", "FIU1"): 'L'}
            for d in direction_dict.keys():
                if d[0] in from_board and d[1] in to_board:
                    return direction_dict[d]
            if 'Line' in from_board:
                if 'OM1' in to_board:
                    return 'U'
                if 'OM2' in to_board:
                    return 'L'
            if 'Line' in to_board:
                if 'OD1' in from_board:
                    return 'U'
                if 'OD2' in from_board:
                    return 'L'
            if ('Tributary' in from_board and 'XCON' in to_board) or ('XCON' in from_board and 'Line' in to_board):
                return 'U'
            else:
                raise Exception("illegal name", name)
        elif 'OLA' in name:
            direction_dict = {("FIU1", "OA1"): 'U', ("OA1", "FIU2"): 'U', ("FIU2", "OA2"): 'L', ("OA2", "FIU1"): 'L'}
            for d in direction_dict.keys():
                if d[0] in from_board and d[1] in to_board:
                    return direction_dict[d]
            raise Exception("illegal name", name)
    else:
        if 'ROADM' in name:
            direction_dict = {"OA1": 'U', "OA2": 'U', "OD1": 'U', "OM1": 'U', "FIU2": 'U',
                              "OA3": 'L', "OA4": 'L', "OD2": 'L', "OM2": 'L', "FIU1": 'L'}
            for d in direction_dict.keys():
                if d in name:
                    return direction_dict[d]
            if 'Line' in name:
                num = int(''.join(c for c in name.split('-')[1].split('$')[0] if c.isdigit()) or '0')
                return 'U' if (num % 2 == 1) else 'L'
            if 'Tributary' in name or 'XCON' in name:
                return '/'
            raise Exception("illegal name", name)
        elif 'OLA' in name:
            direction_dict = {"OA1": 'U', "OA2": 'L', "FIU2": 'U', "FIU1": 'L'}
            for d in direction_dict.keys():
                if d in name:
                    return direction_dict[d]
            raise Exception("illegal name", name)


def board_level_to_node_level(board_level_list):
    node_level_list = [board_level_list[0].split('-')[0]]
    for board in board_level_list:
        node = board.split('-')[0]
        if node_level_list[-1] != node:
            node_level_list.append(node)
    return node_level_list


class Node:

    def __init__(self, name):
        self.name = name
        self.nodeType = ''.join(c for c in name if not c.isdigit())
        self.boards, self.fibers = node_init(self.name)
        self.num_copy = num_copy


class Node_Connection:

    def __init__(self, connected_nodes):
        self.name = "(" + connected_nodes[0] + ";" + connected_nodes[1] + ")"
        self.connectedNodes = connected_nodes
        self.element = node_connection_init(connected_nodes)


class Board:

    def __init__(self, name, parentName, attributes_dict=None):
        if name is None:
            self.name = attributes_dict['name']
            self.tType = attributes_dict['tType']
            self.parentName = attributes_dict['parentName']
            self.fullName = attributes_dict['fullName']
            self.direction = attributes_dict['direction']
        else:
            self.name = name
            self.tType = name
            for item in ["FIU", "OA", "OM", "OD", "Line", "XCON", "Tributary"]:
                self.tType = item if item in self.tType else self.tType
            self.parentName = parentName
            self.fullName = parentName + '-' + name
            self.direction = get_direction(self.fullName)
            if any(k in self.tType for k in ["FIU", "OA", "OM", "OD"]):
                self.layer = LAYER_OPTICAL
            elif any(k in self.tType for k in ELECTRICAL_BOARD_TYPES):
                self.layer = LAYER_ELECTRICAL
            else:
                self.layer = LAYER_OPTICAL

    def get_attributes(self):
        return {"name": self.name, "tType": self.tType, "parentName": self.parentName,
                "fullName": self.fullName, "direction": self.direction, "layer": self.layer}


class Fiber:

    def __init__(self, fromNode, fromBoard, toNode, toBoard):
        self.fromNode = fromNode
        self.fromBoard = fromNode + '-' + fromBoard
        self.toNode = toNode
        self.toBoard = toNode + '-' + toBoard
        self.name = fromBoard + '_' + toBoard
        self.fullName = fromNode + '-' + fromBoard + '_' + toNode + '-' + toBoard

        fromBoardType = fromBoard
        toBoardType = toBoard
        for item in ["FIU", "OA", "OM", "OD", "Line", "XCON", "Tributary"]:
            fromBoardType = item if item in fromBoardType else fromBoardType
            toBoardType = item if item in toBoardType else toBoardType
        self.tType = fromBoardType + '_' + toBoardType

        self.direction = get_direction(self.fullName)

        bridge_pairs = (
            ("OD", "Line"),
            ("Line", "OM"),
            ("Tributary", "XCON"),
            ("XCON", "Line"),
        )
        def _has_pair(tt, a, b):
            return (a in tt and b in tt)

        self.is_bridge = any(_has_pair(self.tType, a, b) for (a, b) in bridge_pairs)

        if self.is_bridge:
            self.layer = LAYER_ELECTRICAL
        elif any(k in self.tType for k in ELECTRICAL_BOARD_TYPES) and not any(k in self.tType for k in ["FIU", "OA", "OM", "OD"]):
            self.layer = LAYER_ELECTRICAL
        else:
            self.layer = LAYER_OPTICAL

    def get_attributes(self):
        return {"fromBoard": self.fromBoard, "toBoard": self.toBoard, "name": self.name,
                "fromNode": self.fromNode, "toNode": self.toNode, "fullName": self.fullName,
                "tType": self.tType, "direction": self.direction, "layer": self.layer,
                "is_bridge": getattr(self, "is_bridge", False)}


class Lightpath:

    def __init__(self, lightpath):
        self.boardPath = lightpath.split(',')
        self.boardPath = _translate_otu_tokens_to_line(self.boardPath)
        self.nodePath = board_level_to_node_level(self.boardPath)

        self.fromNode = self.nodePath[0]
        self.fromBoard = self.boardPath[0].split('-')[1]
        self.toNode = self.nodePath[-1]
        self.toBoard = self.boardPath[-1].split('-')[1]

        path_flow = []
        for i in range(len(self.nodePath)):
            path_flow.append(self.nodePath[i])
            if i != len(self.nodePath) - 1:
                path_flow.append((self.nodePath[i], self.nodePath[i + 1]))
        self.path_flow = path_flow

        if int(''.join(c for c in self.fromBoard[:self.fromBoard.index('$')] if c.isdigit())) % 4 == 2:
            self.path = self.path_flow
        else:
            path = []
            for i in range(len(self.nodePath) - 1, -1, -1):
                path.append(self.nodePath[i])
                if i != 0:
                    path.append((self.nodePath[i], self.nodePath[i - 1]))
            self.path = path


class Alarm:

    def __init__(self, alarmType, board, time, faulty_board, failure_index, root_alarm=0, ID=None):
        self.alarmType = alarmType
        self.board = board
        self.time = time
        self.faulty_board = faulty_board
        self.failure_ID = failure_index
        self.ID = ID
        if ID is not None:
            self.name = self.alarmType + '_' + str(ID)
        else:
            self.name = None
        self.is_root_alarm = root_alarm

    def set_ID(self, ID):
        self.ID = ID
        self.name = self.alarmType + '_' + str(ID)


class OTS_Layer:

    def __init__(self, fromBoard, toBoard, ID):
        self.fromBoard = fromBoard
        self.toBoard = toBoard
        self.ID = ID
        self.name = "OTS" + str(ID)


class OMS_Layer:

    def __init__(self, fromBoard, toBoard, ID):
        self.fromBoard = fromBoard
        self.toBoard = toBoard
        self.ID = ID
        self.name = "OMS" + str(ID)


class OCh_Layer:

    def __init__(self, fromBoard, toBoard, ID):
        self.fromBoard = fromBoard
        self.toBoard = toBoard
        self.ID = ID
        self.name = "OCh" + str(ID)
