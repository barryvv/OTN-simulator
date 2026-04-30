import random
from datetime import timedelta, datetime

import networkx as nx
import csv
from class_obj import *

def _coalesce(d, *names):
    for n in names:
        if n in d and d[n] not in (None, ""):
            return d[n]
    return None

def _parse_time_flex(s):
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromtimestamp(float(s))
    except Exception:
        raise ValueError(f"Unrecognized time format: {s}")
    

def time_generator(num, num_min, now):
    now = now.replace(second=0)
    time_list = []
    for i in range(num_min):
        time_list.extend([now + timedelta(minutes=i)] * num)

    return time_list


def get_failure_board():
    nc_failures = [("FIU_FIU", "fiber cut")]
    ROADM_failures = [("SC2_FIU", "fiber cut"), ("FIU_SC2", "fiber cut"), ("FIU_OA", "fiber cut"), ("OA_FIU", "fiber cut"),
                      ("OA_OD", "fiber cut"), ("OD_OM", "fiber cut"), ("OM_OA", "fiber cut"),
                      ("OD_Line", "fiber cut"), ("Line_OM", "fiber cut"),
                      ("OA", "board faulty"), ("OM", "board faulty"), ("OD", "board faulty"),
                      ("SC2", "board faulty"), ("FIU", "board faulty"),
                      ("Line", "board faulty"), ("XCON", "board faulty"), ("Tributary", "board faulty")]
    OLA_failures = [("SC2_FIU", "fiber cut"), ("FIU_SC2", "fiber cut"), ("FIU_OA", "fiber cut"), ("OA_FIU", "fiber cut"),
                    ("OA", "board faulty"), ("SC2", "board faulty"), ("FIU", "board faulty")]
    return nc_failures, ROADM_failures, OLA_failures


def generate_failure_with_failure_type(lightpath_list, board_DG, failure_type_list, num, num_min, now):
    ROADM_board_type_dict, ROADM_fiber_type_dict = get_board_fiber_type_dict("ROADM")
    OLA_board_type_dict, OLA_fiber_type_dict = get_board_fiber_type_dict("OLA")

    fail_time_list = time_generator(num, num_min, now)

    failure_list = []
    fail_location_group = []
    fail_index = 0

    board_ID_dict = nx.get_node_attributes(board_DG, "ID")
    fiber_ID_dict = nx.get_edge_attributes(board_DG, "ID")

    while len(failure_list) < len(failure_type_list):
        failure_type = failure_type_list[fail_index]

        failed_lightpath = random.choice(lightpath_list)
        fail_node = random.choice(failed_lightpath.path)
        if failure_type[0] == "FIU_FIU":
            while type(fail_node) is not tuple:
                fail_node = random.choice(failed_lightpath.path)
        elif failure_type[0] in ["OTU", "OM", "OD", "OA_OD", "OD_OM", "OM_OA", "OD_OTU", "OTU_OM"]:
            while "ROADM" not in fail_node:
                fail_node = random.choice(failed_lightpath.nodePath)
        else:
            while type(fail_node) is not str:
                failed_lightpath = random.choice(lightpath_list)
                fail_node = random.choice(failed_lightpath.nodePath)

        index = str(random.choice(range(num_copy)))
        if failure_type[0] == "FIU_FIU":
            fail_board = [fail_node[0] + '-FIU2$' + index, fail_node[1] + '-FIU1$' + index]
            random.shuffle(fail_board)
            fail_location = Fiber(fail_board[0].split('-')[0], fail_board[0].split('-')[1],
                                  fail_board[1].split('-')[0], fail_board[1].split('-')[1])
        elif failure_type[0] == 'OD_Line':
            fail_location = Fiber(failed_lightpath.boardPath[-2].split('-')[0], failed_lightpath.boardPath[-2].split('-')[1],
                                  failed_lightpath.boardPath[-1].split('-')[0], failed_lightpath.boardPath[-1].split('-')[1])
        elif failure_type[0] == 'Line_OM':
            fail_location = Fiber(failed_lightpath.boardPath[0].split('-')[0], failed_lightpath.boardPath[0].split('-')[1],
                                  failed_lightpath.boardPath[1].split('-')[0], failed_lightpath.boardPath[1].split('-')[1])
        elif failure_type[1] == "fiber cut":
            if "ROADM" in fail_node:
                fail_board = random.choice(ROADM_fiber_type_dict[failure_type[0]])
            else:
                fail_board = random.choice(OLA_fiber_type_dict[failure_type[0]])
            fail_location = Fiber(fail_node, fail_board.split('_')[0] + '$' + index,
                                  fail_node, fail_board.split('_')[1] + '$' + index)
        else:
            if "ROADM" in fail_node:
                fail_location = Board(
                    failure_type[0] + str(
                        random.choice(range(1, ROADM_board_type_dict[failure_type[0]] + 1))) + '$' + index, fail_node)
            else:
                fail_location = Board(
                    failure_type[0] + str(random.choice(range(1, OLA_board_type_dict[failure_type[0]] + 1))) + '$' + index,
                    fail_node)

        if type(fail_location) is Board:
            fail_ID = board_ID_dict[fail_location.fullName]
        else:
            fail_ID = fiber_ID_dict[(fail_location.fromBoard, fail_location.toBoard)]
        failure = Alarm(failure_type[1], fail_location, fail_time_list[fail_index], fail_location.fullName,
                        failure_index=fail_ID, root_alarm=0, ID=fail_ID)

        if fail_location not in fail_location_group:
            failure_list.append(failure)
            fail_location_group.append(fail_location.fullName)
            fail_index += 1

    return failure_list


def generate_failure_with_board_location(board_list, board_DG, num, num_min, now):
    nc_failure_boards, ROADM_failure_boards, OLA_failure_boards = get_failure_board()
    board_ID_dict = nx.get_node_attributes(board_DG, "ID")
    fiber_ID_dict = nx.get_edge_attributes(board_DG, "ID")

    fail_time_list = time_generator(num, num_min, now)

    failure_list = []
    for fail_index in range(len(board_list)):
        if fail_index % 10000 == 0 and fail_index != 0:
            print("Generating failure index", fail_index, datetime.now().strftime("%H:%M:%S"))

        fail_location = board_list[fail_index]
        if '_' in fail_location:
            fail_location = Fiber(fail_location.split('_')[0].split('-')[0], fail_location.split('_')[0].split('-')[1],
                                  fail_location.split('_')[1].split('-')[0], fail_location.split('_')[1].split('-')[1])
            fail_ID = fiber_ID_dict[(fail_location.fromBoard, fail_location.toBoard)]
        else:
            fail_location = Board(fail_location.split('-')[1], fail_location.split('-')[0])
            fail_ID = board_ID_dict[fail_location.fullName]

        if fail_location.tType == 'FIU_FIU':
            fail_name = random.choice([t[1] for t in nc_failure_boards if t[0] == fail_location.tType])
        elif 'ROADM' in fail_location.fullName:
            fail_name = random.choice([t[1] for t in ROADM_failure_boards if t[0] == fail_location.tType])
        else:
            fail_name = random.choice([t[1] for t in OLA_failure_boards if t[0] == fail_location.tType])

        failure = Alarm(fail_name, fail_location, fail_time_list[fail_index], fail_location.fullName,
                        failure_index=fail_ID, root_alarm=0, ID=fail_ID)
        failure_list.append(failure)

    return failure_list


def get_failure_distribution(failure_list):
    failure_type_dict = {}
    for i in range(len(failure_list)):
        failure = failure_list[i]
        if (failure.board.tType, failure.alarmType) in failure_type_dict:
            failure_type_dict[(failure.board.tType, failure.alarmType)].append((failure.board.fullName, failure.ID))
        else:
            failure_type_dict[(failure.board.tType, failure.alarmType)] = [(failure.board.fullName, failure.ID)]
    return failure_type_dict


def process_failure_from_file(filepath, node_list, nc_list):
    rows = []
    with open(filepath, newline='') as inputfile:
        csv_in = csv.DictReader(inputfile)
        for row in csv_in:
            norm = dict(row)
            if _coalesce(norm, "Event", "Failure type") is None and "Alarm" in norm:
                norm["Event"] = norm["Alarm"]
            if _coalesce(norm, "Location", "Board") is None and "Node" in norm:
                norm["Location"] = norm.get("Node")
            if _coalesce(norm, "Time", "Occurrence time") is None and "Start" in norm:
                norm["Time"] = norm.get("Start")
            rows.append(norm)
    return process_failure(rows, node_list, nc_list)


def process_failure(fails, node_list, nc_list):
    failure_list = []
    for i in range(len(fails)):
        fail = dict(fails[i])

        event_name = _coalesce(fail, "Event", "Failure type", "Alarm")
        if not event_name:
            raise Exception(f"Row {i}: missing Event/Failure type field")

        failure_location = _coalesce(fail, "Location", "Board", "Node")
        if not failure_location:
            raise Exception(f"Row {i}: missing Location/Board field")

        failure_node = failure_location
        if '_' in failure_location:
            fail_start_node = failure_location.split('_')[0].split('-')[0]
            fail_end_node = failure_location.split('_')[1].split('-')[0]
            failure_board = failure_location.split('_')[0].split('-')[1] + '_' + failure_location.split('_')[1].split('-')[1]
            if fail_start_node == fail_end_node:
                for j in range(len(node_list)):
                    if node_list[j].name == fail_start_node:
                        failure_node = node_list[j]
                        break
            else:
                for j in range(len(nc_list)):
                    if nc_list[j].connectedNodes == (fail_start_node, fail_end_node) or \
                            nc_list[j].connectedNodes == (fail_end_node, fail_start_node):
                        failure_node = nc_list[j]
                        break
        else:
            failure_board = failure_location.split('-')[1]
            for j in range(len(node_list)):
                if node_list[j].name == failure_location.split('-')[0]:
                    failure_node = node_list[j]
                    break

        if type(failure_node) is Node:
            if '_' in failure_location:
                failure_board = failure_node.fibers[failure_board]
            else:
                failure_board = failure_node.boards[failure_board]
        else:
            failure_board = failure_node.element[failure_board]

        if type(failure_node) is str or type(failure_board) is str:
            raise Exception(failure_node, failure_location, "not found")

        time_raw = _coalesce(fail, "Time", "Occurrence time", "Start")
        failure_time = _parse_time_flex(time_raw) if time_raw else datetime.now().replace(microsecond=0)

        if "Failure ID" in fail and str(fail["Failure ID"]).strip() != "":
            fail_id = int(fail["Failure ID"])
        else:
            try:
                fail_id = getattr(failure_board, "ID") if hasattr(failure_board, "ID") else i
            except Exception:
                fail_id = i

        failure_list.append(Alarm(str(event_name), failure_board, failure_time, failure_board.fullName, int(fail_id),
                                  0, int(fail_id)))
    return failure_list
