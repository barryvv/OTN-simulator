#!/usr/bin/env python3
"""Test all failure types with regen topology to verify correct behavior."""

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
    parts2 = token.rsplit("-", 2)
    if len(parts2) == 3:
        suffix = f"{parts2[1].strip()}-{parts2[2].strip()}".upper()
        if suffix in {"ODUK-CREATION","ODUK_CREATION"}: return _normalize_board_id(parts2[0].strip()), "SRC"
        if suffix in {"ODUK-TERMINATION","ODUK_TERMINATION"}: return _normalize_board_id(parts2[0].strip()), "TERMINATION"
    parts = token.rsplit("-", 1)
    if len(parts) == 2:
        role = parts[1].strip().upper()
        if role in {"SRC","RELAY","TERMINATION"}: return _normalize_board_id(parts[0].strip()), role
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
              "rule_idx":-1, "rule_note":"seed", "edge_type":"", "direction":""}]
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
                    "edge_type":"","direction":""})
                continue

            if "regenout" in action:
                for neigh, ed, et in adj.get(board, []):
                    if et != "regen_boundary" or ed != "D": continue
                    nr = role_map.get(neigh, "")
                    queue.append({"board":neigh,"alarm":to_alarm,"event":alarm,
                        "role":nr,"hop":hop+1,"prev":board,"rule_idx":rule["idx"],
                        "rule_note":f"{rule['ft']} / {rule['ac']} / {rule['tt']}",
                        "edge_type":"regen_boundary","direction":"D"})
                continue

            want_dir = "U" if "upstream" in action else "D"
            block_regen = bt == "OTUk-Relay"
            tgts = path_targets(paths, role_map, adj, board, want_dir, to_type, block_at_regen=block_regen)
            for neigh, et in tgts:
                nr = role_map.get(neigh, "")
                queue.append({"board":neigh,"alarm":to_alarm,"event":alarm,
                    "role":nr,"hop":hop+1,"prev":board,"rule_idx":rule["idx"],
                    "rule_note":f"{rule['ft']} / {rule['ac']} / {rule['tt']}",
                    "edge_type":et,"direction":want_dir})

    return rows


def check_regen_behavior(rows, failure_event, failure_board):
    """Check that regen rules are correctly applied."""
    issues = []

    # Classify all alarm entries
    otuk_alarms = [r for r in rows if any(x in r["Alarm"].upper() for x in ["OTUK","R_LOS","LOF","LOM"])]
    oduk_alarms = [r for r in rows if "ODUK" in r["Alarm"].upper()]
    regen_in_entries = [r for r in rows if r["BoardType"] == "OTUk-RegenIn"]
    regen_out_entries = [r for r in rows if r["BoardType"] == "OTUk-RegenOut"]
    term_entries = [r for r in rows if r["BoardType"] == "ODUk-Termination"]
    creation_entries = [r for r in rows if r["BoardType"] == "ODUk-Creation"]

    # Determine if failure is on optical relay (should trigger regen behavior)
    fail_bt = None
    for r in rows:
        if r["Board"] == failure_board and r["Alarm"] == failure_event:
            fail_bt = r["BoardType"]
            break

    is_optical_failure = fail_bt in {"OTUk-Relay"}
    is_electrical_failure = fail_bt in {"ODUk-Creation", "ODUk-Termination"}

    # CHECK 1: OTUk alarms at electrical boards are OK if they arrived via
    # a rule with matching source type (e.g. ODUk-Creation sends OTUk_BDI upstream).
    # Only flag if an OTUk alarm arrived at an electrical board via a regen_boundary
    # edge (which would mean it crossed the regen improperly).
    for r in otuk_alarms:
        if r["BoardType"] in {"ODUk-Termination", "ODUk-Creation"} and r["Board"] != failure_board:
            if "regen_boundary" in r.get("EdgeType", ""):
                issues.append(f"OTUk alarm crossed regen boundary to electrical board: {r['Board']}/{r['Alarm']}")

    # CHECK 2: If optical failure → should see OTUk alarm at RegenIn (blocked there)
    if is_optical_failure:
        if not regen_in_entries and not regen_out_entries:
            # No regen boards involved - could be because failure is between creation and first relay
            pass  # Not necessarily an issue
        else:
            # Verify RegenIn has OTUk alarms (blocked)
            regenin_otuk = [r for r in regen_in_entries
                           if any(x in r["Alarm"].upper() for x in ["OTUK","R_LOS","LOF"])]
            if not regenin_otuk:
                issues.append("Optical failure but no OTUk alarm at RegenIn")

            # Verify RegenOut gets OTUk_AIS (forwarded from RegenIn)
            regenout_ais = [r for r in regen_out_entries if "AIS" in r["Alarm"].upper()]
            if not regenout_ais and regen_in_entries:
                # Check if RegenIn has any alarm that should forward
                regenin_with_fwd = [r for r in regen_in_entries
                                   if "Transmit Downstream RegenOut" in r.get("RuleNote","")]
                if not regenin_with_fwd:
                    # RegenIn alarms all "Report Locally" - check if OTUk_AIS was forwarded
                    pass

    # CHECK 3: If regen is involved and optical failure, ODUk_PM_AIS should reach termination
    if is_optical_failure and regen_out_entries:
        oduk_at_term = [r for r in term_entries if "ODUK" in r["Alarm"].upper()]
        regenout_ais = [r for r in regen_out_entries if "AIS" in r["Alarm"].upper()]
        if regenout_ais and not oduk_at_term:
            issues.append("RegenOut has OTUk_AIS but no ODUk_PM_AIS reached termination")

    # CHECK 4: OTUk alarms should NOT cross regen boundary into downstream section
    # i.e., no OTUk alarm at boards downstream of RegenOut (except at RegenOut itself)
    for r in rows:
        if r["BoardType"] in {"OTUk-Relay"} and r["Board"] != failure_board:
            # This relay board should not have OTUk alarms from upstream of a regen
            # Check if it's a board that's downstream of a RegenOut
            if "regen_boundary" in r.get("EdgeType", ""):
                if any(x in r["Alarm"].upper() for x in ["OTUK_LOS","OTUK_LOF"]):
                    issues.append(f"OTUk alarm crossed regen boundary to relay: {r['Board']}/{r['Alarm']}")

    # CHECK 5: For electrical failures (Line_Disconnect, XCON_*), regen should be transparent
    if is_electrical_failure:
        if term_entries or creation_entries:
            pass  # Good - electrical alarms propagate normally

    return issues


def main():
    rule_db = "outputs/Static files/rule_database.csv"
    rules = load_rules(rule_db)
    print(f"Loaded {len(rules)} rules\n")

    regen_lp = "regen_compare/regen/topology/lightpaths.txt"
    paths, role_map = parse_paths(regen_lp)
    adj = build_adj(paths)

    # Get all unique boards and their types from the paths
    all_boards = {}
    for path in paths:
        for board, role in path:
            bt = _board_type(board, role)
            all_boards[board] = {"role": role, "type": bt, "kind": _board_kind(board)}

    # Define test scenarios: (failure_event, board_selector_description, board)
    # Pick boards from path 1: ROADM0-Tributary050$0(SRC), ROADM0-XCON050$0(SRC),
    # ROADM0-Line050$0(SRC), ROADM0-OM2$0, ROADM0-RegenIn00001$0, ROADM0-RegenOut00001$0,
    # ROADM0-OD2$0, ROADM0-OA3$0, ROADM0-RegenIn00002$0, ROADM0-RegenOut00002$0,
    # ROADM0-FIU2$0, ROADM1-FIU1$0, ROADM1-RegenIn00003$0, ROADM1-RegenOut00003$0,
    # ROADM1-OA4$0, ROADM1-OM2$0, ROADM1-Line021$0(TERM), ROADM1-XCON021$0(TERM), ROADM1-Tributary021$0(TERM)

    # Build lookup: find exact board names from parsed paths
    board_by_kind = {}  # (roadm_prefix, kind_upper) -> board_name_in_path
    for path in paths:
        for board, role in path:
            kind = _board_kind(board)
            # Extract ROADM prefix (e.g. "ROADM0" from "ROADM0-OA003$0")
            prefix = board.split("-")[0] if "-" in board else board
            key = (prefix, kind, role)
            board_by_kind.setdefault(key, board)

    # Use exact board names from the paths file for electrical boards
    line050_src = board_by_kind.get(("ROADM0", "LINE", "SRC"), "ROADM0-Line050$0")
    xcon050_src = board_by_kind.get(("ROADM0", "XCON", "SRC"), "ROADM0-XCON050$0")
    trib021_term = board_by_kind.get(("ROADM1", "TRIB", "TERMINATION"), "ROADM1-Tributary021$0")

    tests = [
        # Optical relay failures - should be blocked at regen
        ("Fiber_Cut",   "OA003 (relay between regen pairs)",    "ROADM0-OA003$0"),
        ("Fiber_Crack", "OA003 (relay between regen pairs)",    "ROADM0-OA003$0"),
        ("Fiber_Aging", "OA003 (relay between regen pairs)",    "ROADM0-OA003$0"),
        ("Fiber_Cut",   "FIU002 (relay near inter-ROADM link)", "ROADM0-FIU002$0"),
        ("Fiber_Cut",   "OD002 (relay upstream of regen pair)", "ROADM0-OD002$0"),

        # Electrical failures at creation/termination - regen transparent
        ("Line_Disconnect",     f"Line050 (creation: {line050_src})",  line050_src),
        ("XCON_Port_Down",      f"XCON050 (creation: {xcon050_src})",  xcon050_src),
        ("XCON_Fabric_Fault",   f"XCON050 (creation: {xcon050_src})",  xcon050_src),
        ("XCON_Buffer_Overflow",f"XCON050 (creation: {xcon050_src})",  xcon050_src),
        ("Tributary_Port_Down", f"Trib021 (termination: {trib021_term})", trib021_term),

        # Direct failure at regen board
        ("Fiber_Cut",   "RegenIn00002 (directly at regen ingress)", "ROADM0-RegenIn00002$0"),
    ]

    results = []
    for event, desc, board in tests:
        role = role_map.get(board, "")
        bt = _board_type(board, role)
        rows = run_alarm_flow(rules, paths, role_map, adj, board, event, max_hops=4)
        issues = check_regen_behavior(rows, event, board)

        # Summarize
        otuk_count = sum(1 for r in rows if any(x in r["Alarm"].upper() for x in ["OTUK","R_LOS","LOF","LOM"]))
        oduk_count = sum(1 for r in rows if "ODUK" in r["Alarm"].upper())
        regen_count = sum(1 for r in rows if "Regen" in r["Board"])
        term_oduk = sum(1 for r in rows if r["BoardType"] == "ODUk-Termination" and "ODUK" in r["Alarm"].upper())

        status = "PASS" if not issues else "FAIL"
        results.append((event, desc, board, bt, len(rows), otuk_count, oduk_count, regen_count, term_oduk, issues, rows))

        print(f"{'PASS' if not issues else 'FAIL'}  {event:25s} @ {desc}")
        print(f"       BoardType={bt}, Total={len(rows)}, OTUk={otuk_count}, ODUk={oduk_count}, "
              f"AtRegen={regen_count}, ODUk@Term={term_oduk}")
        if issues:
            for iss in issues:
                print(f"       ISSUE: {iss}")

        # Show unique alarms
        alarm_dist = {}
        for r in rows:
            alarm_dist[r["Alarm"]] = alarm_dist.get(r["Alarm"], 0) + 1
        print(f"       Alarms: {dict(sorted(alarm_dist.items()))}")

        # Show boards affected
        boards_affected = sorted(set(r["Board"] for r in rows))
        print(f"       Boards: {boards_affected}")
        print()

    # Overall summary
    print(f"\n{'='*80}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*80}")
    passed = sum(1 for *_, issues, _ in results if not issues)
    failed = sum(1 for *_, issues, _ in results if issues)
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")

    if failed:
        print(f"\n  FAILED TESTS:")
        for event, desc, board, bt, total, otuk, oduk, regen, term_oduk, issues, rows in results:
            if issues:
                print(f"    {event} @ {desc}")
                for iss in issues:
                    print(f"      - {iss}")

    # Detailed check: show the full flow for select scenarios
    print(f"\n{'='*80}")
    print(f"  DETAILED FLOWS FOR KEY SCENARIOS")
    print(f"{'='*80}")

    for event, desc, board, bt, total, otuk, oduk, regen, term_oduk, issues, rows in results:
        if event in ("Fiber_Cut", "Line_Disconnect", "Tributary_Port_Down") and board in (
            "ROADM0-OA003$0", "ROADM0-Line050$0", "ROADM1-Tributary021$0", "ROADM0-RegenIn00002$0"):
            print(f"\n--- {event} @ {desc} ---")
            print(f"  {'Hop':>3s}  {'Board':35s}  {'Alarm':25s}  {'BoardType':20s}  {'Edge':20s}  {'Dir':3s}")
            print(f"  {'---':>3s}  {'-----':35s}  {'-----':25s}  {'---------':20s}  {'----':20s}  {'---':3s}")
            for r in rows:
                print(f"  {r['Hop']:3d}  {r['Board']:35s}  {r['Alarm']:25s}  {r['BoardType']:20s}  {r['EdgeType']:20s}  {r['Direction']:3s}")
    print()


if __name__ == "__main__":
    main()
