#!/usr/bin/env python3
"""Quick demo: compare alarm propagation with and without regenerators.

Uses a minimal standalone reimplementation to avoid heavy imports.
"""

import csv
import random
import re
from pathlib import Path


def _norm(s):
    return str(s).strip()

def _normalize_board_id(board):
    out = board
    patterns = [("RegenIn",5),("RegenOut",5),("Tributary",3),("XCON",3),
                ("Line",3),("OM",3),("OA",3),("OD",3),("FIU",3)]
    for name, width in patterns:
        out = re.sub(rf"({name})(\d+)",
                     lambda m: f"{m.group(1)}{m.group(2).zfill(width)}",
                     out, flags=re.IGNORECASE)
    return out

def _board_kind(board):
    # Parse type from the component after the first dash: ROADM{id}-{Type}{idx}$0
    if "-" in board:
        part = board.split("-", 1)[1].upper()
    else:
        part = board.upper()
    if part.startswith("REGENIN"):  return "REGENIN"
    if part.startswith("REGENOUT"): return "REGENOUT"
    if part.startswith("TRIBUTARY"): return "TRIB"
    if part.startswith("XCON"):     return "XCON"
    if part.startswith("LINE"):     return "LINE"
    if part.startswith("FIU"):      return "FIU"
    if part.startswith("OM"):       return "OM"
    if part.startswith("OA"):       return "OA"
    if part.startswith("OD"):       return "OD"
    if part.startswith("SC"):       return "SC"
    return "OTHER"

def _board_type(board, role):
    role = role.upper()
    if role == "SRC": return "ODUk-Creation"
    if role == "TERMINATION": return "ODUk-Termination"
    kind = _board_kind(board)
    if kind == "REGENIN":  return "OTUk-RegenIn"
    if kind == "REGENOUT": return "OTUk-RegenOut"
    if kind in {"OM","OA","OD","FIU"}: return "OTUk-Relay"
    if kind in {"TRIB","XCON","LINE"}: return "ODUk-Creation"
    return "OTUk-Relay"

def _edge_type(a, b):
    ka, kb = _board_kind(a), _board_kind(b)
    if ka == "REGENIN" and kb == "REGENOUT": return "regen_boundary"
    return f"{ka}_{kb}"

def _severity(alarm):
    a = alarm.upper()
    if any(x in a for x in ["LOS","LOF","AIS"]): return "Critical"
    if "BDI" in a: return "Major"
    if "DEG" in a: return "Minor"
    return ""

def _layer(t):
    return "optical" if "OTUk" in t or "Relay" in t else "electrical"

def _split_role(token):
    parts = token.rsplit("-", 1)
    if len(parts) == 2:
        role = parts[1].strip().upper()
        if role in {"SRC","RELAY","TERMINATION"}: return _normalize_board_id(parts[0].strip()), role
        if role in {"ODUK-CREATION","ODUK_CREATION"}: return _normalize_board_id(parts[0].strip()), "SRC"
        if role in {"ODUK-TERMINATION","ODUK_TERMINATION"}: return _normalize_board_id(parts[0].strip()), "TERMINATION"
    return _normalize_board_id(token.strip()), ""

def load_rules(path):
    rules = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            ft = _norm(row.get("FunctionType",""))
            if not ft or ft.startswith("#"): continue
            ea = _norm(row.get("EventName",""))
            ac = _norm(row.get("ActionType",""))
            tt = _norm(row.get("ActionTarget",""))
            oa = _norm(row.get("OutputAlarm",""))
            if not ea or not ac or not tt or not oa: continue
            rules.append({"ft":ft,"ea":ea,"ac":ac,"tt":tt,"oa":oa,"idx":idx})
    return rules

def parse_paths(path):
    paths, role_map = [], {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line: continue
        nodes = []
        for tok in [t.strip() for t in line.split(",") if t.strip()]:
            board, role = _split_role(tok)
            nodes.append((board, role))
            if role and board not in role_map: role_map[board] = role
        if nodes: paths.append(nodes)
    return paths, role_map

def build_adj(paths):
    adj = {}
    for path in paths:
        boards = [b for b,_ in path]
        for i in range(len(boards)-1):
            a, b = boards[i], boards[i+1]
            et = _edge_type(a, b)
            adj.setdefault(a, []).append((b, "D", et))
            adj.setdefault(b, []).append((a, "U", et))
    return adj

def path_targets(paths, role_map, adj, board, direction, to_type, block_at_regen=False):
    direction = direction.upper()
    targets = {}
    edge_lookup = {}
    for src, neighs in adj.items():
        for neigh, ed, et in neighs:
            edge_lookup[(src, neigh, ed)] = et
    for path in paths:
        seq = [b for b,_ in path]
        for idx, node in enumerate(seq):
            if node != board: continue
            if direction == "D":
                scan = seq[idx+1:]
                nh = seq[idx+1] if idx+1 < len(seq) else None
                hd = "D"
            else:
                scan = list(reversed(seq[:idx]))
                nh = seq[idx-1] if idx-1 >= 0 else None
                hd = "U"
            if not scan: continue
            et = edge_lookup.get((board, nh, hd), "") if nh else ""
            for cand in scan:
                cr = role_map.get(cand, "")
                if _board_type(cand, cr) == to_type:
                    targets.setdefault(cand, et)
                    break
                if block_at_regen and _board_kind(cand) in {"REGENIN","REGENOUT"}:
                    break
    return sorted(targets.items())


def run_alarm_flow(rules, paths, role_map, adj, failure_board, failure_event, max_hops=4):
    random.seed(2025)
    queue = [{"board": failure_board, "alarm": failure_event, "event": failure_event,
              "role": role_map.get(failure_board,""), "hop":0, "prev":"",
              "rule_idx":-1, "rule_note":"seed", "edge_type":"", "direction":"", "delay_s":0.0}]
    seen = set()
    rows = []

    while queue:
        cur = queue.pop(0)
        board = cur["board"]
        alarm = cur["alarm"]
        role = cur["role"]
        hop = cur["hop"]

        key = (board, alarm, hop)
        if key in seen: continue
        seen.add(key)

        bt = _board_type(board, role)
        rows.append({
            "Board": board, "Alarm": alarm, "Event": cur["event"],
            "Role": role, "Layer": _layer(bt), "Severity": _severity(alarm),
            "Hop": hop, "Prev": cur["prev"], "RuleNote": cur["rule_note"],
            "RuleIdx": cur["rule_idx"], "EdgeType": cur["edge_type"],
            "Direction": cur["direction"], "BoardType": bt,
        })

        if hop >= max_hops: continue

        for rule in rules:
            if rule["ft"] != bt or rule["ea"] != alarm: continue
            action = rule["ac"].lower()
            to_alarm = rule["oa"]
            to_type = rule["tt"]

            if "report locally" in action:
                queue.append({"board":board,"alarm":to_alarm,"event":alarm,
                    "role":role,"hop":hop+1,"prev":board,"rule_idx":rule["idx"],
                    "rule_note":f"{rule['ft']} / {rule['ac']} / {rule['tt']}",
                    "edge_type":"","direction":"","delay_s":0.0})
                continue

            if "regenout" in action:
                for neigh, ed, et in adj.get(board, []):
                    if et != "regen_boundary" or ed != "D": continue
                    nr = role_map.get(neigh, "")
                    queue.append({"board":neigh,"alarm":to_alarm,"event":alarm,
                        "role":nr,"hop":hop+1,"prev":board,"rule_idx":rule["idx"],
                        "rule_note":f"{rule['ft']} / {rule['ac']} / {rule['tt']}",
                        "edge_type":"regen_boundary","direction":"D","delay_s":1.0})
                continue

            want_dir = "U" if "upstream" in action else "D"
            block_regen = bt == "OTUk-Relay"
            tgts = path_targets(paths, role_map, adj, board, want_dir, to_type, block_at_regen=block_regen)
            for neigh, et in tgts:
                nr = role_map.get(neigh, "")
                queue.append({"board":neigh,"alarm":to_alarm,"event":alarm,
                    "role":nr,"hop":hop+1,"prev":board,"rule_idx":rule["idx"],
                    "rule_note":f"{rule['ft']} / {rule['ac']} / {rule['tt']}",
                    "edge_type":et,"direction":want_dir,"delay_s":random.uniform(1,8)})

    return rows


def print_flow(rows, title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  Total alarm entries: {len(rows)}")

    boards = sorted(set(r["Board"] for r in rows))
    print(f"  Unique boards affected: {len(boards)}")

    alarms = {}
    for r in rows:
        alarms[r["Alarm"]] = alarms.get(r["Alarm"], 0) + 1
    print(f"\n  Alarm distribution:")
    for a in sorted(alarms, key=lambda x: -alarms[x]):
        print(f"    {a:30s} x {alarms[a]}")

    layers = {}
    for r in rows:
        layers[r["Layer"]] = layers.get(r["Layer"], 0) + 1
    print(f"\n  Layer distribution:")
    for l in sorted(layers):
        print(f"    {l:15s} x {layers[l]}")

    print(f"\n  {'Hop':>3s}  {'Board':35s}  {'Alarm':25s}  {'BoardType':20s}  {'EdgeType':20s}  {'Dir':3s}  RuleNote")
    print(f"  {'---':>3s}  {'-----':35s}  {'-----':25s}  {'---------':20s}  {'--------':20s}  {'---':3s}  --------")
    for r in rows:
        print(f"  {r['Hop']:3d}  {r['Board']:35s}  {r['Alarm']:25s}  {r['BoardType']:20s}  {r['EdgeType']:20s}  {r['Direction']:3s}  {r['RuleNote']}")


def main():
    rule_db = "outputs/Static files/rule_database.csv"
    rules = load_rules(rule_db)
    print(f"Loaded {len(rules)} rules from {rule_db}")

    # === NOREGEN ===
    noregen_lp = "regen_compare/noregen/topology/lightpaths.txt"
    paths_nr, roles_nr = parse_paths(noregen_lp)
    adj_nr = build_adj(paths_nr)
    rows_nr = run_alarm_flow(rules, paths_nr, roles_nr, adj_nr,
                             "ROADM0-OA002$0", "Fiber_Cut")
    print_flow(rows_nr, "NO REGEN: Fiber_Cut at ROADM0-OA002$0")

    # === REGEN ===
    regen_lp = "regen_compare/regen/topology/lightpaths.txt"
    paths_rg, roles_rg = parse_paths(regen_lp)
    adj_rg = build_adj(paths_rg)
    # OA3 is between RegenOut00001 and RegenIn00002 on path 1
    rows_rg = run_alarm_flow(rules, paths_rg, roles_rg, adj_rg,
                             "ROADM0-OA003$0", "Fiber_Cut")
    print_flow(rows_rg, "WITH REGEN: Fiber_Cut at ROADM0-OA003$0 (between regen pairs)")

    # === COMPARISON ===
    print(f"\n{'='*80}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"  No-regen total alarm entries:   {len(rows_nr)}")
    print(f"  With-regen total alarm entries:  {len(rows_rg)}")

    # Check regen behavior
    otuk_rg = [r for r in rows_rg if "OTUk" in r["Alarm"] or "R_LOS" in r["Alarm"]]
    oduk_rg = [r for r in rows_rg if "ODUk" in r["Alarm"]]
    regen_boards = [r for r in rows_rg if "Regen" in r["Board"]]

    print(f"\n  Regen scenario analysis:")
    print(f"    OTUk-level alarm entries: {len(otuk_rg)}")
    print(f"    ODUk-level alarm entries: {len(oduk_rg)}")
    print(f"    Alarms at regen boards:   {len(regen_boards)}")

    # OTUk alarms should NOT appear past the regen boundary
    otuk_past = [r for r in otuk_rg
                 if "Regen" not in r["Board"]
                 and r["Board"] != "ROADM0-OA003$0"  # source board itself
                 and r["BoardType"] not in {"OTUk-Relay"}]
    if otuk_past:
        print(f"\n    WARNING: {len(otuk_past)} OTUk alarms leaked past regen boundary!")
        for r in otuk_past:
            print(f"      {r['Board']} / {r['Alarm']} / {r['RuleNote']}")
    else:
        print(f"\n    PASS: OTUk alarms contained at regen boundary (section terminated)")

    oduk_at_term = [r for r in oduk_rg if r["BoardType"] == "ODUk-Termination"]
    if oduk_at_term:
        print(f"    PASS: {len(oduk_at_term)} ODUk alarms reached termination (transparent passthrough)")
    else:
        print(f"    INFO: No ODUk alarms reached termination in this scenario")

    print()


if __name__ == "__main__":
    main()
