"""End-to-end integration tests for the OTN simulator pipeline.

Validates: topology structure → alarm propagation → metric profiles.
Run with:  .venv/bin/python -m pytest test_e2e.py -v
"""

import sys
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# ── Imports from existing modules ────────────────────────────────────────────
from report_regen_vs_noregen import (
    load_rules,
    parse_paths,
    build_adj,
    run_alarm_flow,
    analyze,
    find_board_by_kind,
    _board_kind,
)
from demo_all_failures import check_regen_behavior
from otn_simulator import _profile_for_alarm, _apply_profile_to_bbe, _smooth_bbe

# ── Constants ────────────────────────────────────────────────────────────────
RULE_DB = str(REPO / "outputs" / "Static files" / "rule_database.csv")
NOREGEN_LP = str(REPO / "regen_compare" / "noregen" / "topology" / "lightpaths.txt")
REGEN_LP = str(REPO / "regen_compare" / "regen" / "topology" / "lightpaths.txt")
SIMULATOR_YAML = REPO / "simulator.yaml"
BASE_SPIKE = 10.0
AIS_BBE = 6.0 * 2.0 * 2.0  # 24.0 — propagated AIS peak


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rules():
    return load_rules(RULE_DB)


@pytest.fixture(scope="module")
def noregen_ctx():
    paths, role_map = parse_paths(NOREGEN_LP)
    adj = build_adj(paths)
    return paths, role_map, adj


@pytest.fixture(scope="module")
def regen_ctx():
    paths, role_map = parse_paths(REGEN_LP)
    adj = build_adj(paths)
    return paths, role_map, adj


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Fiber_Cut noregen — full pipeline validation
# ══════════════════════════════════════════════════════════════════════════════

def test_fiber_cut_noregen_full_pipeline(noregen_ctx, rules):
    paths, role_map, adj = noregen_ctx

    # ── Stage 1: Topology validation ─────────────────────────────────────────

    # 1A. Find the path containing ROADM0-OA002$0
    failure_board = "ROADM0-OA002$0"
    target_path = None
    for path in paths:
        boards = [b for b, _ in path]
        if failure_board in boards:
            target_path = path
            break
    assert target_path is not None, f"{failure_board} not found in any noregen path"

    boards = [b for b, _ in target_path]
    kinds = [_board_kind(b) for b in boards]

    # 1B. Board sequence: first 3 electrical (TRIB, XCON, LINE), last 3 electrical
    assert kinds[0] == "TRIB", f"First board should be TRIB, got {kinds[0]}"
    assert kinds[1] == "XCON", f"Second board should be XCON, got {kinds[1]}"
    assert kinds[2] == "LINE", f"Third board should be LINE, got {kinds[2]}"
    assert kinds[-3] == "LINE", f"Third-to-last should be LINE, got {kinds[-3]}"
    assert kinds[-2] == "XCON", f"Second-to-last should be XCON, got {kinds[-2]}"
    assert kinds[-1] == "TRIB", f"Last board should be TRIB, got {kinds[-1]}"

    # 1C. Middle boards are optical only
    optical_kinds = set(kinds[3:-3])
    assert optical_kinds.issubset({"OM", "OA", "OD", "FIU"}), (
        f"Non-optical boards in middle: {optical_kinds - {'OM', 'OA', 'OD', 'FIU'}}"
    )

    # 1D. Roles: first 3 = SRC, last 3 = TERMINATION
    for b, role in target_path[:3]:
        assert role == "SRC", f"Board {b} should be SRC, got {role}"
    for b, role in target_path[-3:]:
        assert role == "TERMINATION", f"Board {b} should be TERMINATION, got {role}"

    # 1E. OA002 has empty role (optical relay, no explicit annotation)
    for b, role in target_path:
        if b == failure_board:
            assert role == "", f"OA002 should have empty role (relay), got {role}"

    # 1F. No regen boards in ANY noregen path
    for path in paths:
        for b, _ in path:
            assert "Regen" not in b, f"Regen board in noregen topology: {b}"

    # 1G. Source and destination on different ROADMs
    assert boards[0].split("-")[0] != boards[-1].split("-")[0], (
        "Source and destination should be on different ROADMs"
    )

    # ── Stage 2: Alarm propagation ───────────────────────────────────────────

    rows = run_alarm_flow(rules, paths, role_map, adj, failure_board, "Fiber_Cut", max_hops=4)
    a = analyze(rows)

    # 2A. Seed alarm at hop 0
    seeds = [r for r in rows if r["Hop"] == 0]
    assert len(seeds) == 1, f"Expected 1 seed, got {len(seeds)}"
    assert seeds[0]["Board"] == failure_board
    assert seeds[0]["Alarm"] == "Fiber_Cut"
    assert seeds[0]["BoardType"] == "OTUk-Relay"

    # 2B. R_LOS generated locally at hop 1
    r_los = [r for r in rows if r["Alarm"] == "R_LOS" and r["Hop"] == 1
             and r["Board"] == failure_board]
    assert len(r_los) >= 1, "Missing R_LOS at failure board (Report Locally rule)"
    assert r_los[0]["Severity"] == "Critical"

    # 2C. ODUk_PM_AIS reaches termination
    ais_at_term = [r for r in rows
                   if r["Alarm"] == "ODUk_PM_AIS"
                   and r["BoardType"] == "ODUk-Termination"
                   and r["Hop"] >= 1]
    assert len(ais_at_term) >= 1, "ODUk_PM_AIS must reach at least one termination board"

    # 2D. OTUk_BDI propagates upstream
    bdi_upstream = [r for r in rows
                    if r["Alarm"] == "OTUk_BDI"
                    and r["Direction"] == "U"
                    and r["Hop"] >= 1]
    assert len(bdi_upstream) >= 1, "OTUk_BDI must propagate upstream"

    # 2E. Hop limits and alarm counts
    assert a["max_hop"] <= 4, f"Max hop {a['max_hop']} exceeds limit"
    assert a["max_hop"] >= 2, f"Max hop {a['max_hop']} too low — expect cascading"
    assert 15 <= a["total"] <= 80, f"Unexpected alarm count: {a['total']}"

    # 2F. Both OTUk and ODUk alarms present
    assert a["otuk_count"] >= 2, f"Expected OTUk alarms, got {a['otuk_count']}"
    assert a["oduk_count"] >= 1, f"Expected ODUk alarms, got {a['oduk_count']}"

    # 2G. All boards in alarm_flow exist in lightpaths
    all_lp_boards = set()
    for p in paths:
        for b, _ in p:
            all_lp_boards.add(b)
    for r in rows:
        assert r["Board"] in all_lp_boards, f"Board {r['Board']} not in lightpaths"

    # ── Stage 3: Metric validation ───────────────────────────────────────────

    # 3A. Fiber_Cut profile correctness
    prof = _profile_for_alarm("Fiber_Cut", rng=None)
    assert prof["kind"] == "step"
    assert prof["mult"] == 5.5
    assert prof["bursts"] == 1
    assert prof["ramp_down"] is False
    assert prof["bber_mult"] == 3.0
    assert prof["es_threshold_mult"] == 0.6

    # 3B. BBE peak = 10.0 * 5.5 = 55.0
    expected_bbe = BASE_SPIKE * prof["mult"]
    assert expected_bbe == pytest.approx(55.0, abs=0.01)

    # 3C. All 7 event types produce unique peaks (catches BBE collision bug)
    event_peaks = {
        "Fiber_Cut": 55.0,
        "Fiber_Crack": 42.0,        # step_recovery: 10 * 3.5 * 1.2
        "Fiber_Aging": 28.0,        # ramp: 10 * 2.8
        "Line_Disconnect": 36.0,    # step_recovery: 10 * 3.0 * 1.2
        "XCON_Port_Down": 32.0,     # step: 10 * 3.2
        "XCON_Buffer_Overflow": 46.0,  # burst: 10 * 4.6
        "XCON_Fabric_Fault": 50.4,    # step_recovery: 10 * 4.2 * 1.2
    }
    seen_peaks = set()
    for ev, expected_peak in event_peaks.items():
        p = _profile_for_alarm(ev, rng=None)
        if p["kind"] == "step_recovery":
            actual_peak = BASE_SPIKE * p["mult"] * 1.2
        else:
            actual_peak = BASE_SPIKE * p["mult"]
        assert actual_peak == pytest.approx(expected_peak, abs=0.1), (
            f"{ev}: expected peak {expected_peak}, got {actual_peak}"
        )
        # Uniqueness check
        rounded = round(actual_peak, 1)
        assert rounded not in seen_peaks, (
            f"{ev}: peak {rounded} collides with another event"
        )
        seen_peaks.add(rounded)

    # 3D. All peaks exceed propagated AIS BBE (24.0)
    for ev, peak in event_peaks.items():
        assert peak > AIS_BBE, f"{ev} peak {peak} not > AIS peak {AIS_BBE}"

    # 3E. Apply step profile to synthetic array
    series = np.zeros(200, dtype=np.float32)
    _apply_profile_to_bbe(series, 50, 150, BASE_SPIKE, prof)
    assert np.max(series[50:150]) == pytest.approx(55.0, abs=0.1), "Step plateau wrong"
    assert np.max(series[:50]) == 0.0, "BBE before injection window should be 0"

    # 3F. BBER at peak: 55.0 / 1000.0 * 3.0 = 0.165
    expected_bber = 55.0 / 1000.0 * prof["bber_mult"]
    assert expected_bber == pytest.approx(0.165, abs=0.001)

    # 3G. ES at peak: min(1.0, 55.0 / (9.0 * 0.6)) = 1.0 (saturated)
    expected_es = min(1.0, 55.0 / (9.0 * prof["es_threshold_mult"]))
    assert expected_es == 1.0, "ES should saturate to 1.0 at Fiber_Cut peak"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Regen boundary isolation (OTUk blocked, ODUk transparent)
# ══════════════════════════════════════════════════════════════════════════════

def test_fiber_cut_regen_boundary_isolation(regen_ctx, noregen_ctx, rules):
    paths_rg, role_map_rg, adj_rg = regen_ctx
    paths_nr, role_map_nr, adj_nr = noregen_ctx

    # ── Stage 1: Regen topology structure ────────────────────────────────────

    # 1A. Regen paths contain RegenIn/RegenOut; noregen paths do not
    regen_boards_found = set()
    for path in paths_rg:
        for b, _ in path:
            if "RegenIn" in b:
                regen_boards_found.add(("RegenIn", b))
            if "RegenOut" in b:
                regen_boards_found.add(("RegenOut", b))
    assert len(regen_boards_found) > 0, "Regen topology must contain RegenIn/RegenOut"

    for path in paths_nr:
        for b, _ in path:
            assert "Regen" not in b, f"Noregen topology contains regen board: {b}"

    # 1B. RegenIn always immediately followed by RegenOut
    for path in paths_rg:
        boards = [b for b, _ in path]
        for i, b in enumerate(boards):
            if "RegenIn" in b:
                assert i + 1 < len(boards), f"RegenIn at end of path: {b}"
                next_b = boards[i + 1]
                assert "RegenOut" in next_b, (
                    f"RegenIn {b} not followed by RegenOut, got {next_b}"
                )

    # 1C. Board before RegenIn is FIU (fiber span boundary)
    for path in paths_rg:
        boards = [b for b, _ in path]
        for i, b in enumerate(boards):
            if "RegenIn" in b and i > 0:
                prev_kind = _board_kind(boards[i - 1])
                assert prev_kind == "FIU", (
                    f"Board before RegenIn should be FIU, got {prev_kind} ({boards[i-1]})"
                )

    # 1D. Adjacency: RegenIn→RegenOut has regen_boundary edge
    for src, neighbors in adj_rg.items():
        if "RegenIn" in src:
            regen_edge = [n for n in neighbors
                          if "RegenOut" in n[0] and n[1] == "D"]
            assert len(regen_edge) >= 1, (
                f"RegenIn {src} missing regen_boundary edge to RegenOut"
            )
            for neigh, direction, edge_type in regen_edge:
                assert edge_type == "regen_boundary", (
                    f"Expected regen_boundary edge, got {edge_type}"
                )

    # ── Stage 2: OTUk containment at regen boundary ─────────────────────────

    failure_board = "ROADM0-OA002$0"
    rows_rg = run_alarm_flow(
        rules, paths_rg, role_map_rg, adj_rg, failure_board, "Fiber_Cut", max_hops=4
    )
    a_rg = analyze(rows_rg)

    # 2A. Alarms reach RegenIn
    regen_in_entries = [r for r in rows_rg if r["BoardType"] == "OTUk-RegenIn"]
    assert len(regen_in_entries) >= 1, "No alarm reached RegenIn"

    # 2B. RegenOut gets OTUk_AIS via regen_boundary
    regen_out_entries = [r for r in rows_rg if r["BoardType"] == "OTUk-RegenOut"]
    ais_at_regenout = [r for r in regen_out_entries if "AIS" in r["Alarm"].upper()]
    regenin_otuk = [r for r in regen_in_entries
                    if any(x in r["Alarm"].upper() for x in ["OTUK", "R_LOS", "LOF"])]
    if regenin_otuk:
        assert len(ais_at_regenout) >= 1, (
            "RegenIn has OTUk alarm but RegenOut missing OTUk_AIS"
        )
    for r in ais_at_regenout:
        assert r["EdgeType"] == "regen_boundary", (
            f"OTUk_AIS at RegenOut should have regen_boundary edge, got {r['EdgeType']}"
        )

    # 2C. OTUk alarms must NOT cross regen boundary to ODUk-Termination
    for r in rows_rg:
        if r["BoardType"] == "ODUk-Termination":
            alarm_upper = r["Alarm"].upper()
            if any(x in alarm_upper for x in ["OTUK_LOS", "OTUK_LOF", "OTUK_LOM"]):
                assert "regen_boundary" not in r.get("EdgeType", ""), (
                    f"OTUk alarm {r['Alarm']} crossed regen to termination {r['Board']}"
                )

    # 2D. Hop counts valid
    assert a_rg["max_hop"] <= 4

    # 2E. check_regen_behavior passes (comprehensive invariant check)
    issues = check_regen_behavior(rows_rg, "Fiber_Cut", failure_board)
    assert len(issues) == 0, f"Regen behavior issues: {issues}"

    # ── Stage 3: ODUk transparency for electrical failures ───────────────────

    nr_line_src = find_board_by_kind(paths_nr, role_map_nr, "LINE", "SRC", "ROADM0")
    rg_line_src = find_board_by_kind(paths_rg, role_map_rg, "LINE", "SRC", "ROADM0")
    assert nr_line_src is not None, "No LINE SRC in noregen"
    assert rg_line_src is not None, "No LINE SRC in regen"

    rows_e_nr = run_alarm_flow(
        rules, paths_nr, role_map_nr, adj_nr, nr_line_src, "Line_Disconnect"
    )
    rows_e_rg = run_alarm_flow(
        rules, paths_rg, role_map_rg, adj_rg, rg_line_src, "Line_Disconnect"
    )
    a_e_nr = analyze(rows_e_nr)
    a_e_rg = analyze(rows_e_rg)

    # 3A. Both topologies deliver ODUk to termination
    assert a_e_nr["oduk_at_term"] >= 1, "No ODUk at termination in noregen"
    assert a_e_rg["oduk_at_term"] >= 1, "No ODUk at termination in regen"

    # 3B. No regen_boundary edges for electrical failure (regen is transparent)
    regen_edges_elec = [r for r in rows_e_rg if r["EdgeType"] == "regen_boundary"]
    assert len(regen_edges_elec) == 0, (
        f"Electrical failure should not cross regen boundary, found {len(regen_edges_elec)} entries"
    )

    # 3C. Validate regen config from simulator.yaml
    sim_cfg = yaml.safe_load(SIMULATOR_YAML.read_text())
    regen_cfg = sim_cfg["propagation"]["regen"]
    assert regen_cfg["otuk_reset_mult"] == 0.1, (
        f"OTUk reset mult should be 0.1, got {regen_cfg['otuk_reset_mult']}"
    )
    assert regen_cfg["oduk_passthrough_mult"] == 1.0, (
        f"ODUk passthrough mult should be 1.0, got {regen_cfg['oduk_passthrough_mult']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Parametrized — all event types profile + alarm coverage
# ══════════════════════════════════════════════════════════════════════════════

EVENT_SPECS = [
    # (event, expected_kind, base_mult, board_kind, board_role, peak_bbe)
    ("Fiber_Cut",            "step",          5.5, "OA",   None,  55.0),
    ("Fiber_Crack",          "step_recovery", 3.5, "OA",   None,  42.0),
    ("Fiber_Aging",          "ramp",          2.8, "OA",   None,  28.0),
    ("Line_Disconnect",      "step_recovery", 3.0, "LINE", "SRC", 36.0),
    ("XCON_Port_Down",       "step",          3.2, "XCON", "SRC", 32.0),
    ("XCON_Buffer_Overflow", "burst",         4.6, "XCON", "SRC", 46.0),
    ("XCON_Fabric_Fault",    "step_recovery", 4.2, "XCON", "SRC", 50.4),
]


@pytest.mark.parametrize(
    "event,expected_kind,base_mult,board_kind,board_role,expected_peak",
    EVENT_SPECS,
    ids=[e[0] for e in EVENT_SPECS],
)
def test_all_event_types_profile_and_alarm_coverage(
    event, expected_kind, base_mult, board_kind, board_role, expected_peak,
    noregen_ctx, rules,
):
    paths, role_map, adj = noregen_ctx

    # ── Stage 1: Profile validation ──────────────────────────────────────────

    # 1A. Deterministic profile matches expected kind and mult
    prof = _profile_for_alarm(event, rng=None)
    assert prof["kind"] == expected_kind, (
        f"{event}: expected kind={expected_kind}, got {prof['kind']}"
    )
    assert prof["mult"] == base_mult, (
        f"{event}: expected mult={base_mult}, got {prof['mult']}"
    )

    # 1B. BBE peak matches expected (with ×1.2 for step_recovery bursts)
    if prof["kind"] == "step_recovery":
        actual_peak = BASE_SPIKE * prof["mult"] * 1.2
    else:
        actual_peak = BASE_SPIKE * prof["mult"]
    assert actual_peak == pytest.approx(expected_peak, abs=0.1), (
        f"{event}: expected peak {expected_peak}, got {actual_peak}"
    )

    # 1C. Peak exceeds AIS threshold
    assert actual_peak > AIS_BBE, (
        f"{event}: peak {actual_peak} not > AIS BBE {AIS_BBE}"
    )

    # 1D. Noisy profiles: mult stays within ±22%, kind unchanged
    rng = np.random.default_rng(42)
    for _ in range(50):
        noisy = _profile_for_alarm(event, rng=rng)
        assert noisy["kind"] == expected_kind, (
            f"{event}: noisy profile changed kind to {noisy['kind']}"
        )
        assert base_mult * 0.78 <= noisy["mult"] <= base_mult * 1.22, (
            f"{event}: noisy mult {noisy['mult']} outside ±22% of {base_mult}"
        )

    # ── Stage 2: Profile shape validation ────────────────────────────────────

    T = 200
    series = np.zeros(T, dtype=np.float32)
    t0, t1 = 50, 150
    _apply_profile_to_bbe(series, t0, t1, BASE_SPIKE, prof)

    peak_val = BASE_SPIKE * prof["mult"]

    if expected_kind == "step":
        # Flat plateau at base_spike * mult
        assert np.all(series[t0:t1] >= peak_val - 0.1), (
            f"{event}: step plateau not at {peak_val}"
        )

    elif expected_kind == "ramp":
        # Peak reaches ≥90% of base * mult
        assert np.max(series[t0:t1]) >= peak_val * 0.9, (
            f"{event}: ramp peak too low"
        )

    elif expected_kind == "burst":
        # Has peaks AND gaps (intermittent pattern)
        vals = series[t0:t1]
        at_peak = int(np.sum(vals >= peak_val * 0.9))
        below_half = int(np.sum(vals < peak_val * 0.5))
        assert at_peak >= 1, f"{event}: burst has no peak values"
        assert below_half >= 1, f"{event}: burst should have gaps below peak"

    elif expected_kind == "step_recovery":
        # Plateau at base * mult, with overshoots at mult * 1.2
        assert np.max(series[t0:t1]) >= peak_val, (
            f"{event}: step_recovery below plateau"
        )
        overshoots = int(np.sum(series[t0:t1] > peak_val + 0.01))
        assert overshoots >= 1, (
            f"{event}: step_recovery missing ×1.2 overshoots"
        )

    # 2B. BBER should be > 0 at peak
    if "bber_mult" in prof:
        bbe_smooth = _smooth_bbe(series[t0:t1, np.newaxis], 10).reshape(-1)
        bber_max = float(np.max(bbe_smooth)) / 1000.0 * prof["bber_mult"]
        assert bber_max > 0, f"{event}: BBER max should be > 0"

    # 2C. ES should saturate to 1.0 at peak
    if "es_threshold_mult" in prof:
        bbe_smooth = _smooth_bbe(series[t0:t1, np.newaxis], 10).reshape(-1)
        thresh = 9.0 * prof["es_threshold_mult"]
        es_max = float(np.clip(np.max(bbe_smooth) / thresh, 0, 1))
        assert es_max == pytest.approx(1.0, abs=0.05), (
            f"{event}: ES should saturate to 1.0, got {es_max}"
        )

    # ── Stage 3: Alarm propagation pattern ───────────────────────────────────

    # 3A. Find the correct board for this event
    failure_board = find_board_by_kind(
        paths, role_map, board_kind, board_role, "ROADM0"
    )
    assert failure_board is not None, (
        f"No {board_kind} board (role={board_role}) found on ROADM0"
    )

    # 3B. Run alarm flow
    rows = run_alarm_flow(rules, paths, role_map, adj, failure_board, event, max_hops=4)
    assert len(rows) >= 1, f"{event}: no alarm entries generated"

    # 3C. Seed at hop 0
    seeds = [r for r in rows if r["Hop"] == 0]
    assert len(seeds) == 1, f"{event}: expected 1 seed, got {len(seeds)}"
    assert seeds[0]["Alarm"] == event

    # 3D. Event-specific alarm pattern checks
    local_alarms = [r for r in rows if r["Hop"] == 1 and r["Board"] == failure_board]

    if event == "Fiber_Cut":
        assert any(r["Alarm"] == "R_LOS" for r in local_alarms), (
            f"{event}: missing local R_LOS"
        )
        ds_ais = [r for r in rows
                  if r["Alarm"] == "ODUk_PM_AIS" and r["Direction"] == "D"]
        assert len(ds_ais) >= 1, f"{event}: missing downstream ODUk_PM_AIS"
        bdi = [r for r in rows
               if r["Alarm"] == "OTUk_BDI" and r["Direction"] == "U"]
        assert len(bdi) >= 1, f"{event}: missing upstream OTUk_BDI"

    elif event == "Fiber_Crack":
        assert any(r["Alarm"] == "OTUk_LOF" for r in local_alarms), (
            f"{event}: missing local OTUk_LOF"
        )
        bdi = [r for r in rows
               if r["Alarm"] == "OTUk_BDI" and r["Direction"] == "U"]
        assert len(bdi) >= 1, f"{event}: missing upstream OTUk_BDI"

    elif event == "Fiber_Aging":
        assert any(r["Alarm"] == "OTUk_DEG" for r in local_alarms), (
            f"{event}: missing local OTUk_DEG"
        )
        bdi = [r for r in rows
               if r["Alarm"] == "OTUk_BDI" and r["Direction"] == "U"]
        assert len(bdi) >= 1, f"{event}: missing upstream OTUk_BDI"

    elif event == "Line_Disconnect":
        assert any(r["Alarm"] == "R_LOS" for r in local_alarms), (
            f"{event}: missing local R_LOS"
        )
        ds_ais = [r for r in rows
                  if r["Alarm"] == "ODUk_PM_AIS" and r["Direction"] == "D"]
        assert len(ds_ais) >= 1, f"{event}: missing downstream ODUk_PM_AIS"

    elif event == "XCON_Port_Down":
        assert any(r["Alarm"] == "R_LOS" for r in local_alarms), (
            f"{event}: missing local R_LOS"
        )
        ds_ais = [r for r in rows
                  if r["Alarm"] == "ODUk_PM_AIS" and r["Direction"] == "D"]
        assert len(ds_ais) >= 1, f"{event}: missing downstream ODUk_PM_AIS"

    elif event == "XCON_Buffer_Overflow":
        assert any(r["Alarm"] == "OTUk_EXC" for r in local_alarms), (
            f"{event}: missing local OTUk_EXC"
        )
        bdi = [r for r in rows
               if r["Alarm"] == "OTUk_BDI" and r["Direction"] == "U"]
        assert len(bdi) >= 1, f"{event}: missing upstream OTUk_BDI"
        # Should NOT produce direct downstream AIS
        ds_ais_direct = [r for r in rows
                         if r["Alarm"] == "ODUk_PM_AIS"
                         and r["Hop"] == 1 and r["Direction"] == "D"]
        assert len(ds_ais_direct) == 0, (
            f"{event}: should NOT produce direct downstream ODUk_PM_AIS"
        )

    elif event == "XCON_Fabric_Fault":
        assert any(r["Alarm"] == "OTUk_LOF" for r in local_alarms), (
            f"{event}: missing local OTUk_LOF"
        )
        ds_bdi = [r for r in rows
                  if r["Alarm"] == "ODUk_PM_BDI" and r["Direction"] == "D"
                  and r["Hop"] >= 1]
        assert len(ds_bdi) >= 1, f"{event}: missing downstream ODUk_PM_BDI"

    # 3E. All boards in alarm_flow exist in lightpaths
    all_lp_boards = set()
    for p in paths:
        for b, _ in p:
            all_lp_boards.add(b)
    for r in rows:
        assert r["Board"] in all_lp_boards, (
            f"{event}: board {r['Board']} not in lightpaths"
        )

    # 3F. All severities are valid
    for r in rows:
        if r["Hop"] > 0 and r["Severity"]:
            assert r["Severity"] in ("Critical", "Major", "Minor"), (
                f"{event}: invalid severity {r['Severity']} at {r['Board']}"
            )
