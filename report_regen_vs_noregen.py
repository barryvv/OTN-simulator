#!/usr/bin/env python3
"""Overview report: Regen vs Non-Regen alarm propagation across all failure types.

Compares alarm behavior side-by-side for both topologies, showing how regenerators
affect alarm containment (OTUk blocked) vs transparency (ODUk passthrough).
"""

import csv
import random
import re
from pathlib import Path


# ── helpers (same as demo scripts) ──────────────────────────────────────────

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


# ── analysis helpers ────────────────────────────────────────────────────────

def analyze(rows):
    """Extract key metrics from alarm flow results."""
    total = len(rows)
    unique_boards = sorted(set(r["Board"] for r in rows))
    otuk = [r for r in rows if any(x in r["Alarm"].upper() for x in ["OTUK","R_LOS","LOF","LOM"])]
    oduk = [r for r in rows if "ODUK" in r["Alarm"].upper()]
    at_regen = [r for r in rows if "Regen" in r["Board"]]
    at_term = [r for r in rows if r["BoardType"] == "ODUk-Termination"]
    at_creation = [r for r in rows if r["BoardType"] == "ODUk-Creation"]
    oduk_at_term = [r for r in at_term if "ODUK" in r["Alarm"].upper()]
    otuk_at_term = [r for r in at_term if any(x in r["Alarm"].upper() for x in ["OTUK","R_LOS","LOF","LOM"])]
    crit = [r for r in rows if r["Severity"] == "Critical"]
    major = [r for r in rows if r["Severity"] == "Major"]
    minor = [r for r in rows if r["Severity"] == "Minor"]

    alarm_dist = {}
    for r in rows:
        alarm_dist[r["Alarm"]] = alarm_dist.get(r["Alarm"], 0) + 1

    return {
        "total": total,
        "unique_boards": len(unique_boards),
        "board_list": unique_boards,
        "otuk_count": len(otuk),
        "oduk_count": len(oduk),
        "regen_count": len(at_regen),
        "term_count": len(at_term),
        "creation_count": len(at_creation),
        "oduk_at_term": len(oduk_at_term),
        "otuk_at_term": len(otuk_at_term),
        "critical": len(crit),
        "major": len(major),
        "minor": len(minor),
        "alarm_dist": alarm_dist,
        "max_hop": max((r["Hop"] for r in rows), default=0),
    }


def find_board_by_kind(paths, role_map, kind_filter, role_filter=None, roadm_prefix=None):
    """Find a board matching criteria from parsed paths."""
    for path in paths:
        for board, role in path:
            kind = _board_kind(board)
            if kind != kind_filter:
                continue
            if role_filter and role != role_filter:
                continue
            if roadm_prefix and not board.startswith(roadm_prefix):
                continue
            return board
    return None


# ── main ────────────────────────────────────────────────────────────────────

def main():
    rule_db = "outputs/Static files/rule_database.csv"
    rules = load_rules(rule_db)

    # Load both topologies
    noregen_lp = "regen_compare/noregen/topology/lightpaths.txt"
    regen_lp = "regen_compare/regen/topology/lightpaths.txt"

    paths_nr, roles_nr = parse_paths(noregen_lp)
    adj_nr = build_adj(paths_nr)

    paths_rg, roles_rg = parse_paths(regen_lp)
    adj_rg = build_adj(paths_rg)

    # Find equivalent boards in each topology
    # Optical relays shared (exist in both): OA002, FIU002, OD001 (ROADM4)
    # Electrical boards differ by topology — pick SRC/TERMINATION from each

    # Noregen SRC/TERM boards (from path 2 which goes through OA3)
    nr_line_src = find_board_by_kind(paths_nr, roles_nr, "LINE", "SRC", "ROADM0")
    nr_xcon_src = find_board_by_kind(paths_nr, roles_nr, "XCON", "SRC", "ROADM0")
    nr_trib_term = find_board_by_kind(paths_nr, roles_nr, "TRIB", "TERMINATION", "ROADM1")

    # Regen SRC/TERM boards (from path 1 which goes through OA3)
    rg_line_src = find_board_by_kind(paths_rg, roles_rg, "LINE", "SRC", "ROADM0")
    rg_xcon_src = find_board_by_kind(paths_rg, roles_rg, "XCON", "SRC", "ROADM0")
    rg_trib_term = find_board_by_kind(paths_rg, roles_rg, "TRIB", "TERMINATION", "ROADM1")

    # Define test scenarios: (failure_event, description, noregen_board, regen_board)
    tests = [
        # --- OPTICAL RELAY FAILURES (same boards in both topologies) ---
        ("Fiber_Cut",   "Optical relay (OA)",
         "ROADM0-OA002$0", "ROADM0-OA002$0"),
        ("Fiber_Crack", "Optical relay (OA)",
         "ROADM0-OA002$0", "ROADM0-OA002$0"),
        ("Fiber_Aging", "Optical relay (OA)",
         "ROADM0-OA002$0", "ROADM0-OA002$0"),
        ("Fiber_Cut",   "Optical relay (FIU hub)",
         "ROADM0-FIU002$0", "ROADM0-FIU002$0"),
        ("Fiber_Cut",   "Optical relay (OD demux)",
         "ROADM4-OD001$0", "ROADM4-OD001$0"),

        # --- ELECTRICAL FAILURES (equivalent boards in each topology) ---
        ("Line_Disconnect",      "Electrical (Line SRC)",
         nr_line_src, rg_line_src),
        ("XCON_Port_Down",       "Electrical (XCON SRC)",
         nr_xcon_src, rg_xcon_src),
        ("XCON_Fabric_Fault",    "Electrical (XCON SRC)",
         nr_xcon_src, rg_xcon_src),
        ("XCON_Buffer_Overflow", "Electrical (XCON SRC)",
         nr_xcon_src, rg_xcon_src),
        ("Tributary_Port_Down",  "Electrical (Trib TERM)",
         nr_trib_term, rg_trib_term),
    ]

    # ── Header ──
    W = 100
    print("=" * W)
    print("  OTN ALARM PROPAGATION: REGEN vs NON-REGEN COMPARISON REPORT")
    print("=" * W)
    print(f"  Rule database: {rule_db} ({len(rules)} rules)")
    print(f"  Non-regen topology: {noregen_lp} ({len(paths_nr)} paths)")
    print(f"  Regen topology:     {regen_lp} ({len(paths_rg)} paths)")
    print()

    all_results = []

    for event, desc, nr_board, rg_board in tests:
        rows_nr = run_alarm_flow(rules, paths_nr, roles_nr, adj_nr, nr_board, event)
        rows_rg = run_alarm_flow(rules, paths_rg, roles_rg, adj_rg, rg_board, event)
        a_nr = analyze(rows_nr)
        a_rg = analyze(rows_rg)
        all_results.append((event, desc, nr_board, rg_board, a_nr, a_rg, rows_nr, rows_rg))

    # ── Summary Table ──
    print("-" * W)
    print("  SUMMARY TABLE")
    print("-" * W)
    hdr = (f"  {'Failure':<25s} {'Category':<22s} "
           f"{'NoRegen':>8s} {'Regen':>8s} {'Delta':>7s} "
           f"{'NR-OTUk':>8s} {'RG-OTUk':>8s} "
           f"{'NR-ODUk':>8s} {'RG-ODUk':>8s} "
           f"{'RG@Regen':>9s}")
    print(hdr)
    print("  " + "-" * (W - 2))

    for event, desc, nr_board, rg_board, a_nr, a_rg, _, _ in all_results:
        delta = a_rg["total"] - a_nr["total"]
        delta_str = f"{delta:+d}" if delta != 0 else "0"
        print(f"  {event:<25s} {desc:<22s} "
              f"{a_nr['total']:>8d} {a_rg['total']:>8d} {delta_str:>7s} "
              f"{a_nr['otuk_count']:>8d} {a_rg['otuk_count']:>8d} "
              f"{a_nr['oduk_count']:>8d} {a_rg['oduk_count']:>8d} "
              f"{a_rg['regen_count']:>9d}")
    print()

    # ── OTUk Containment Analysis (key regen metric) ──
    print("-" * W)
    print("  OTUK CONTAINMENT ANALYSIS (Regen Section Termination)")
    print("-" * W)
    print(f"  {'Failure':<25s} {'NR OTUk@Term':>13s} {'RG OTUk@Term':>13s} {'RG ODUk@Term':>13s} {'Status':<15s}")
    print("  " + "-" * (W - 2))

    for event, desc, nr_board, rg_board, a_nr, a_rg, rows_nr, rows_rg in all_results:
        # Check if OTUk alarms leaked past regen to termination in regen scenario
        rg_otuk_leak = a_rg["otuk_at_term"]
        nr_otuk_at_term = a_nr["otuk_at_term"]
        rg_oduk_at_term = a_rg["oduk_at_term"]

        # For optical failures: noregen should have OTUk at term, regen should NOT
        # For electrical failures: OTUk at term in both is fine (generated at electrical boards)
        is_optical = desc.startswith("Optical")
        if is_optical:
            if nr_otuk_at_term > 0 and rg_otuk_leak == 0:
                status = "CONTAINED"
            elif nr_otuk_at_term == 0 and rg_otuk_leak == 0:
                status = "OK (none)"
            elif rg_otuk_leak > 0:
                status = "LEAK!"
            else:
                status = "OK"
        else:
            status = "TRANSPARENT"

        print(f"  {event:<25s} {nr_otuk_at_term:>13d} {rg_otuk_leak:>13d} {rg_oduk_at_term:>13d} {status:<15s}")
    print()

    # ── Detailed Per-Failure Comparison ──
    print("=" * W)
    print("  DETAILED PER-FAILURE COMPARISON")
    print("=" * W)

    for event, desc, nr_board, rg_board, a_nr, a_rg, rows_nr, rows_rg in all_results:
        print()
        print(f"  --- {event} @ {desc} ---")
        if nr_board == rg_board:
            print(f"  Board: {nr_board}")
        else:
            print(f"  NoRegen board: {nr_board}")
            print(f"  Regen board:   {rg_board}")

        # Side-by-side metrics
        print(f"  {'Metric':<30s} {'NoRegen':>10s} {'Regen':>10s} {'Delta':>10s}")
        print(f"  {'-'*30:<30s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s}")

        metrics = [
            ("Total alarm entries",   a_nr["total"],        a_rg["total"]),
            ("Unique boards affected",a_nr["unique_boards"],a_rg["unique_boards"]),
            ("OTUk-layer alarms",     a_nr["otuk_count"],   a_rg["otuk_count"]),
            ("ODUk-layer alarms",     a_nr["oduk_count"],   a_rg["oduk_count"]),
            ("At regen boards",       a_nr["regen_count"],  a_rg["regen_count"]),
            ("At termination",        a_nr["term_count"],   a_rg["term_count"]),
            ("ODUk at termination",   a_nr["oduk_at_term"], a_rg["oduk_at_term"]),
            ("OTUk at termination",   a_nr["otuk_at_term"], a_rg["otuk_at_term"]),
            ("Max propagation hop",   a_nr["max_hop"],      a_rg["max_hop"]),
        ]

        for name, nr_val, rg_val in metrics:
            delta = rg_val - nr_val
            delta_str = f"{delta:+d}" if delta != 0 else "-"
            print(f"  {name:<30s} {nr_val:>10d} {rg_val:>10d} {delta_str:>10s}")

        # Alarm type distribution comparison
        all_alarms = sorted(set(list(a_nr["alarm_dist"].keys()) + list(a_rg["alarm_dist"].keys())))
        print(f"\n  {'Alarm Type':<25s} {'NoRegen':>10s} {'Regen':>10s}")
        print(f"  {'-'*25:<25s} {'-'*10:>10s} {'-'*10:>10s}")
        for alarm in all_alarms:
            nr_c = a_nr["alarm_dist"].get(alarm, 0)
            rg_c = a_rg["alarm_dist"].get(alarm, 0)
            marker = ""
            if nr_c > 0 and rg_c == 0: marker = "  [BLOCKED by regen]"
            elif nr_c == 0 and rg_c > 0: marker = "  [NEW in regen]"
            elif rg_c < nr_c: marker = "  [REDUCED]"
            elif rg_c > nr_c: marker = "  [INCREASED]"
            print(f"  {alarm:<25s} {nr_c:>10d} {rg_c:>10d}{marker}")

    # ── Key Findings ──
    print()
    print("=" * W)
    print("  KEY FINDINGS")
    print("=" * W)

    # 1. OTUk containment
    optical_tests = [(e, d, nr, rg, anr, arg, rnr, rrg)
                     for e, d, nr, rg, anr, arg, rnr, rrg in all_results
                     if d.startswith("Optical")]
    electrical_tests = [(e, d, nr, rg, anr, arg, rnr, rrg)
                        for e, d, nr, rg, anr, arg, rnr, rrg in all_results
                        if d.startswith("Electrical")]

    print(f"\n  1. OTUk Section Termination at Regen Boundary")
    print(f"     Optical failures tested: {len(optical_tests)}")
    leaked = sum(1 for *_, anr, arg, _, _ in optical_tests if arg["otuk_at_term"] > 0)
    # Key metric: in regen topology, OTUk alarms are processed at regen boards
    # instead of propagating as ODUk to termination. Compare ODUk reduction at termination.
    for e, d, nr, rg, anr, arg, rnr, rrg in optical_tests:
        nr_oduk_t = anr["oduk_at_term"]
        rg_oduk_t = arg["oduk_at_term"]
        rg_regen = arg["regen_count"]
        delta = nr_oduk_t - rg_oduk_t
        print(f"     {e:<20s}  NR ODUk@Term={nr_oduk_t:>2d}  RG ODUk@Term={rg_oduk_t:>2d}  "
              f"RG@Regen={rg_regen:>2d}  "
              f"{'CONTAINED' if delta > 0 else 'SAME' if delta == 0 else 'MORE'}")
    if leaked:
        print(f"     WARNING: {leaked} scenarios had OTUk leaks past regen boundary!")
    else:
        print(f"     No OTUk leaks past regen boundary — section termination working correctly")

    # 2. ODUk transparency
    print(f"\n  2. ODUk Payload Transparency Through Regen")
    for e, d, nr, rg, anr, arg, rnr, rrg in optical_tests:
        if arg["oduk_at_term"] > 0:
            print(f"     {e}: ODUk_PM_AIS reached termination (transparent passthrough)")
        else:
            print(f"     {e}: No ODUk alarm at termination (expected for {e})")

    # 3. Electrical failure transparency
    print(f"\n  3. Electrical Failure Transparency (Regen should be invisible)")
    for e, d, nr, rg, anr, arg, rnr, rrg in electrical_tests:
        # Regen should not change electrical alarm behavior significantly
        nr_total = anr["total"]
        rg_total = arg["total"]
        oduk_match = anr["oduk_at_term"] == arg["oduk_at_term"]
        oduk_status = "MATCH" if oduk_match else f"NR={anr['oduk_at_term']} RG={arg['oduk_at_term']}"
        print(f"     {e:<25s}  NR={nr_total:>3d}  RG={rg_total:>3d}  "
              f"ODUk@Term: {oduk_status}  "
              f"{'OK' if oduk_match else 'DIFF'}")

    # 4. Alarm flood reduction
    print(f"\n  4. Alarm Flood Reduction (Regen benefit)")
    for e, d, nr, rg, anr, arg, rnr, rrg in optical_tests:
        nr_total = anr["total"]
        rg_total = arg["total"]
        if nr_total > 0:
            reduction = (1 - rg_total / nr_total) * 100
            print(f"     {e:<25s}  NoRegen: {nr_total:>3d} alarms  "
                  f"Regen: {rg_total:>3d} alarms  "
                  f"{'Reduction' if reduction > 0 else 'Increase'}: {abs(reduction):.1f}%")

    # 5. Board spread
    print(f"\n  5. Board Spread (Blast Radius)")
    for e, d, nr, rg, anr, arg, rnr, rrg in all_results:
        nr_b = anr["unique_boards"]
        rg_b = arg["unique_boards"]
        print(f"     {e:<25s} [{d:<22s}]  NR: {nr_b:>2d} boards  RG: {rg_b:>2d} boards")

    print()
    print("=" * W)
    print("  END OF REPORT")
    print("=" * W)

    # ── Export CSV files ──
    outdir = Path("outputs/reports")
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Summary CSV
    with open(outdir / "summary_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Failure", "Category", "NR_Board", "RG_Board",
                     "NR_Total", "RG_Total", "Delta",
                     "NR_OTUk", "RG_OTUk", "NR_ODUk", "RG_ODUk",
                     "RG_AtRegen", "NR_Boards", "RG_Boards",
                     "NR_ODUk@Term", "RG_ODUk@Term", "NR_OTUk@Term", "RG_OTUk@Term",
                     "NR_MaxHop", "RG_MaxHop"])
        for event, desc, nr_board, rg_board, a_nr, a_rg, _, _ in all_results:
            w.writerow([event, desc, nr_board, rg_board,
                        a_nr["total"], a_rg["total"], a_rg["total"] - a_nr["total"],
                        a_nr["otuk_count"], a_rg["otuk_count"],
                        a_nr["oduk_count"], a_rg["oduk_count"],
                        a_rg["regen_count"],
                        a_nr["unique_boards"], a_rg["unique_boards"],
                        a_nr["oduk_at_term"], a_rg["oduk_at_term"],
                        a_nr["otuk_at_term"], a_rg["otuk_at_term"],
                        a_nr["max_hop"], a_rg["max_hop"]])

    # 2. Alarm distribution CSV (all failures, all alarm types)
    with open(outdir / "alarm_distribution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Failure", "Category", "AlarmType",
                     "NR_Count", "RG_Count", "Delta", "Note"])
        for event, desc, nr_board, rg_board, a_nr, a_rg, _, _ in all_results:
            all_alarms = sorted(set(list(a_nr["alarm_dist"].keys()) + list(a_rg["alarm_dist"].keys())))
            for alarm in all_alarms:
                nr_c = a_nr["alarm_dist"].get(alarm, 0)
                rg_c = a_rg["alarm_dist"].get(alarm, 0)
                delta = rg_c - nr_c
                if nr_c > 0 and rg_c == 0: note = "BLOCKED by regen"
                elif nr_c == 0 and rg_c > 0: note = "NEW in regen"
                elif rg_c < nr_c: note = "REDUCED"
                elif rg_c > nr_c: note = "INCREASED"
                else: note = "SAME"
                w.writerow([event, desc, alarm, nr_c, rg_c, delta, note])

    # 3. Detailed alarm flow CSV (every alarm entry for every scenario)
    with open(outdir / "detailed_alarm_flows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Failure", "Category", "Topology",
                     "Hop", "Board", "Alarm", "BoardType", "Layer",
                     "Severity", "EdgeType", "Direction", "Prev", "RuleNote"])
        for event, desc, nr_board, rg_board, a_nr, a_rg, rows_nr, rows_rg in all_results:
            for r in rows_nr:
                w.writerow([event, desc, "NoRegen",
                            r["Hop"], r["Board"], r["Alarm"], r["BoardType"],
                            r["Layer"], r["Severity"], r["EdgeType"],
                            r["Direction"], r["Prev"], r["RuleNote"]])
            for r in rows_rg:
                w.writerow([event, desc, "Regen",
                            r["Hop"], r["Board"], r["Alarm"], r["BoardType"],
                            r["Layer"], r["Severity"], r["EdgeType"],
                            r["Direction"], r["Prev"], r["RuleNote"]])

    print(f"\nExported to {outdir}/:")
    print(f"  summary_comparison.csv        — one row per failure scenario")
    print(f"  alarm_distribution.csv        — alarm type breakdown per scenario")
    print(f"  detailed_alarm_flows.csv      — every alarm entry for every scenario")
    print(f"  regen_vs_noregen_comparison.txt — this full text report")


if __name__ == "__main__":
    main()
