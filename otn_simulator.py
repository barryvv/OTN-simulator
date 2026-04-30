#!/usr/bin/env python3

import argparse
import os, json, math, random, sys, heapq
import re
import subprocess
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timedelta

import yaml
import numpy as np
import pandas as pd


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _run_remote_llm_analysis(local_dir: Path, host: str, remote_base: str,
                             remote_script: str, ollama_url: str,
                             report_name: str, remote_cmd: str | None) -> None:
    run_name = local_dir.name
    remote_base = remote_base.rstrip("/")
    remote_run = f"{remote_base}/{run_name}"

    subprocess.run(["ssh", host, f"mkdir -p {remote_run}"], check=True)

    rsync_up = [
        "rsync", "-az", "--delete",
        f"{str(local_dir)}/",
        f"{host}:{remote_run}/",
    ]
    subprocess.run(rsync_up, check=True)

    local_script = Path(remote_script).expanduser()
    remote_script_remote = remote_script
    if local_script.exists():
        rsync_script = [
            "rsync", "-az",
            str(local_script),
            f"{host}:{remote_run}/",
        ]
        subprocess.run(rsync_script, check=True)
        remote_script_remote = f"{remote_run}/{local_script.name}"

    if remote_cmd:
        report_path = f"{remote_run}/{report_name}"
        cmd = remote_cmd.format(
            run_dir=remote_run,
            report_path=report_path,
            report_name=report_name,
            ollama_url=ollama_url,
            remote_script=remote_script_remote,
        )
        subprocess.run(["ssh", host, cmd], check=True)
        try:
            subprocess.run(["ssh", host, f"test -f {report_path}"], check=True)
        except subprocess.CalledProcessError:
            print(f"[warn] remote LLM report missing: {report_path}")
            return
        rsync_down = [
            "rsync", "-az",
            f"{host}:{report_path}",
            str(local_dir / report_name),
        ]
        subprocess.run(rsync_down, check=True)
        return

    cmd = f"python3 {remote_script_remote} --run-dir {remote_run} --ollama-url {ollama_url}"
    subprocess.run(["ssh", host, cmd], check=True)

    rsync_down = [
        "rsync", "-az",
        f"{host}:{remote_run}/{report_name}",
        str(local_dir / report_name),
    ]
    subprocess.run(rsync_down, check=True)

def canonical_seg_id(u, v):
    a, b = (u, v) if u < v else (v, u)
    return f"{a}-{b}"

def normalize_roles_to_list(obj):
    if isinstance(obj, list):
        if len(obj) == 0 or isinstance(obj[0], dict):
            return obj
        raise TypeError("roles.yaml is a list but items are not dicts.")
    if isinstance(obj, dict):
        if "roles" in obj and isinstance(obj["roles"], list):
            return obj["roles"]
        if all(isinstance(v, dict) for v in obj.values()):
            out = []
            for k, v in obj.items():
                rec = dict(v)
                rec.setdefault("vmf_id", k)
                out.append(rec)
            return out
    raise TypeError("roles.yaml must be a list of dicts, or {'roles': [...]}, or {vmf_id: {...}} mapping.")


def build_adj_from_segment_index(seg_index: dict):
    adj = defaultdict(set)
    for sid, pair in seg_index.items():
        a, b = int(pair[0]), int(pair[1])
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    return adj

def node_path_to_oriented_segments(node_path):
    return [f"{node_path[i]}->{node_path[i+1]}" for i in range(len(node_path) - 1)]

def bfs_shortest_path_hop_leq2(adj, s, t):
    if s == t:
        return [s]
    if t in adj[s]:
        return [s, t]
    for m in adj[s]:
        if m != s and m != t and (t in adj[m]):
            return [s, m, t]
    return None


def lightly_tweak_services_inplace(topo, roles_vmf_ids_set, frac: float, seed: int):
    stats = {"considered": 0, "tweaked": 0, "skipped_no_alt": 0, "skipped_missing_vmf": 0}
    if frac <= 0.0:
        return stats

    seg_index = topo["network"].get("segments_index") or topo["network"].get("segment_index")
    if not seg_index:
        print("[warn] no segments_index in topology; skip topo tweak.", file=sys.stderr)
        return stats

    adj = build_adj_from_segment_index(seg_index)
    id_rules = topo["id_rules"]
    src_fmt   = id_rules["src_fmt"]
    relay_fmt = id_rules["relay_fmt"]
    snk_fmt   = id_rules["snk_fmt"]

    services = topo["services"]
    n = len(services)
    k_target = max(1, int(round(frac * n)))

    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)

    def vmfs_exist_for_path(node_path, service_id):
        src_id = src_fmt.format(node=node_path[0], service=service_id)
        snk_id = snk_fmt.format(node=node_path[-1], service=service_id)
        if src_id not in roles_vmf_ids_set or snk_id not in roles_vmf_ids_set:
            return False
        oriented = node_path_to_oriented_segments(node_path)
        for s in oriented:
            u_str, v_str = s.split("->")
            u, v = int(u_str), int(v_str)
            seg = canonical_seg_id(u, v)
            out_id = relay_fmt.format(node=u, segment=seg, dir="out")
            in_id  = relay_fmt.format(node=v, segment=seg, dir="in")
            if out_id not in roles_vmf_ids_set or in_id not in roles_vmf_ids_set:
                return False
        return True

    for idx in order:
        if stats["tweaked"] >= k_target:
            break
        svc = services[idx]
        stats["considered"] += 1

        s = int(svc["src"]); t = int(svc["dst"])
        old_nodes = list(map(int, svc["node_path"]))

        alt = bfs_shortest_path_hop_leq2(adj, s, t)

        def try_find_another_two_hop():
            for m in rng.permutation(list(adj[s])):
                if m != s and m != t and (t in adj[m]):
                    cand = [s, m, t]
                    if cand != old_nodes:
                        return cand
            return None

        if alt is None or alt == old_nodes:
            alt = try_find_another_two_hop()

        if alt is None or alt == old_nodes:
            stats["skipped_no_alt"] += 1
            continue

        if not vmfs_exist_for_path(alt, svc["service_id"]):
            stats["skipped_missing_vmf"] += 1
            continue

        svc["node_path"] = list(map(int, alt))
        svc["path_segments"] = [canonical_seg_id(alt[i], alt[i+1]) for i in range(len(alt)-1)]
        svc["path_segments_oriented"] = node_path_to_oriented_segments(alt)
        stats["tweaked"] += 1

    return stats


def normalize_role_for_priors(role_str: str) -> str:
    r = str(role_str).upper()
    if r in ("RELAY_IN", "RELAY-IN", "RELAY_OUT", "RELAY-OUT"):
        return "RELAY"
    if r in ("SRC", "SOURCE"):
        return "SRC"
    if r in ("SNK", "SINK", "DEST", "DST"):
        return "SNK"
    return r


class VMFGraph:
    def __init__(self):
        self.vmf_list = []
        self.vmf_index = {}
        self.node_to_vmfs = defaultdict(list)
        self.role_of = {}
        self.node_of = {}
        self.meta = {}
        self.edges = []

    def add_vmf_from_roles(self, rec):
        vmf_id = rec["vmf_id"]
        if vmf_id in self.vmf_index:
            if "service_ids" in rec:
                self.meta[vmf_id]["service_ids"] = sorted(list(set(self.meta[vmf_id].get("service_ids", []) + list(rec["service_ids"]))))
            elif "service_id" in rec:
                sid = rec["service_id"]
                if sid is not None:
                    self.meta[vmf_id]["service_ids"] = sorted(list(set(self.meta[vmf_id].get("service_ids", []) + [sid])))
            return self.vmf_index[vmf_id]

        idx  = len(self.vmf_list)
        node = int(rec["node_id"])
        raw_role  = str(rec["role"]).upper()
        base_role = normalize_role_for_priors(raw_role)
        self.vmf_index[vmf_id] = idx
        self.vmf_list.append({"id": vmf_id, "node": node, "role": base_role})
        self.node_to_vmfs[node].append(idx)
        self.role_of[vmf_id] = base_role
        self.node_of[vmf_id] = node

        m = {
            "node_id": node,
            "segment_id": rec.get("segment_id"),
            "fiber_group_id": rec.get("fiber_group_id"),
            "is_regen_boundary": bool(rec.get("is_regen_boundary", False)),
            "subrole": raw_role,
        }
        if "service_ids" in rec and rec["service_ids"] is not None:
            m["service_ids"] = list(rec["service_ids"])
        else:
            sid = rec.get("service_id")
            m["service_ids"] = [sid] if sid is not None else []
        self.meta[vmf_id] = m
        return idx

    def add_edge(self, src_id, dst_id, kind, delay, weight):
        self.edges.append({
            "src": src_id, "dst": dst_id,
            "kind": kind, "delay": int(delay), "weight": float(weight)
        })


def build_graph_and_service_candidates(topo, roles, sim_cfg, rng):
    g = VMFGraph()
    for rec in roles:
        g.add_vmf_from_roles(rec)

    roles_set = set(g.vmf_index.keys())

    id_rules = topo["id_rules"]
    src_fmt   = id_rules["src_fmt"]
    relay_fmt = id_rules["relay_fmt"]
    snk_fmt   = id_rules["snk_fmt"]

    bmin = float(sim_cfg["propagation"]["beta_min"])
    bmax = float(sim_cfg["propagation"]["beta_max"])
    delay_choices = list(sim_cfg["propagation"]["delay_choices"])
    regen_cfg = sim_cfg["propagation"].get("regen", {})
    regen_delay_choices = regen_cfg.get("delay_choices") or sim_cfg["propagation"].get("regen_delay_choices") or []
    regen_delay_default = int(regen_cfg.get("delay", sim_cfg["propagation"].get("regen_delay", min(delay_choices) if delay_choices else 1)))
    regen_weight = float(regen_cfg.get("weight", sim_cfg["propagation"].get("regen_weight", 1.0)))

    services = topo["services"]
    service_candidates = {}

    prop_edge_param_cache = {}

    def add_edge_if_present(a_id, b_id, kind, delay, weight):
        if (a_id in roles_set) and (b_id in roles_set):
            g.add_edge(a_id, b_id, kind=kind, delay=delay, weight=weight)
            return True
        return False

    def maybe_append(cand_list, vid):
        if vid in roles_set:
            cand_list.append(vid)

    for svc in services:
        sid = svc["service_id"]
        node_path = list(map(int, svc["node_path"]))
        oriented = list(svc["path_segments_oriented"])
        if not oriented:
            continue

        cand = []

        regen_snk = {}
        regen_src = {}
        for vmf in svc.get("vmf_path") or []:
            if not vmf.get("regen"):
                continue
            node = vmf.get("node")
            vid = vmf.get("vmf_id")
            role = str(vmf.get("role") or "").upper()
            if node is None or not vid:
                continue
            if role == "SNK":
                regen_snk[int(node)] = vid
            elif role == "SRC":
                regen_src[int(node)] = vid

        def regen_pair(node_id: int) -> tuple[str | None, str | None]:
            return regen_snk.get(node_id), regen_src.get(node_id)

        src_id = src_fmt.format(node=node_path[0], service=sid)
        maybe_append(cand, src_id)

        u0, v0 = map(int, oriented[0].split("->"))
        seg0 = canonical_seg_id(u0, v0)
        out0 = relay_fmt.format(node=u0, segment=seg0, dir="out")
        if add_edge_if_present(src_id, out0, kind="local", delay=0, weight=1.0):
            maybe_append(cand, out0)

        for k in range(1, len(node_path)-1):
            pu, pv = map(int, oriented[k-1].split("->"))
            nu, nv = map(int, oriented[k].split("->"))
            prev_seg = canonical_seg_id(pu, pv)
            next_seg = canonical_seg_id(nu, nv)
            mid = node_path[k]
            in_id  = relay_fmt.format(node=mid, segment=prev_seg, dir="in")
            out_id = relay_fmt.format(node=mid, segment=next_seg, dir="out")
            snk_id, src_id = regen_pair(int(mid))
            if snk_id and src_id:
                if add_edge_if_present(in_id, snk_id, kind="local", delay=0, weight=1.0):
                    maybe_append(cand, in_id)
                    maybe_append(cand, snk_id)
                regen_delay = int(rng.choice(regen_delay_choices)) if regen_delay_choices else regen_delay_default
                if add_edge_if_present(snk_id, src_id, kind="regen_boundary", delay=regen_delay, weight=regen_weight):
                    maybe_append(cand, src_id)
                if add_edge_if_present(src_id, out_id, kind="local", delay=0, weight=1.0):
                    maybe_append(cand, out_id)
            else:
                if add_edge_if_present(in_id, out_id, kind="local", delay=0, weight=1.0):
                    maybe_append(cand, in_id)
                    maybe_append(cand, out_id)

        ul, vl = map(int, oriented[-1].split("->"))
        last_seg = canonical_seg_id(ul, vl)
        in_last = relay_fmt.format(node=node_path[-1], segment=last_seg, dir="in")
        snk_id  = snk_fmt.format(node=node_path[-1], service=sid)
        if add_edge_if_present(in_last, snk_id, kind="local", delay=0, weight=1.0):
            maybe_append(cand, in_last)
            maybe_append(cand, snk_id)

        for s in oriented:
            a, b = map(int, s.split("->"))
            seg = canonical_seg_id(a, b)
            out_id = relay_fmt.format(node=a, segment=seg, dir="out")
            in_id  = relay_fmt.format(node=b, segment=seg, dir="in")
            key = (out_id, in_id)
            if key not in prop_edge_param_cache:
                w = float(rng.uniform(bmin, bmax))
                d = int(rng.choice(delay_choices))
                prop_edge_param_cache[key] = (d, w)
            d, w = prop_edge_param_cache[key]
            add_edge_if_present(out_id, in_id, kind="prop", delay=d, weight=w)

        cand_unique = sorted(set(cand), key=lambda x: (g.node_of.get(x, 0), x))
        service_candidates[sid] = cand_unique

    return g, service_candidates


def season_value(kind: str, amp: float, t: int, step_seconds: int):
    if kind == "none" or amp == 0.0:
        return 0.0
    day = 24*3600
    week = 7*24*3600
    period = day if kind == "daily" else (week if kind == "weekly" else day)
    x = (t * step_seconds) / period
    return amp * math.sin(2*math.pi * x)

def simulate_latent_series(graph: VMFGraph, sim_cfg, rng):
    T = int(sim_cfg["time"]["T_total"])
    step_seconds = int(str(sim_cfg["time"]["step"]).rstrip("s"))
    L_warmup = int(sim_cfg["time"]["L_warmup"])
    eps_sigma_base = float(sim_cfg["noise"]["epsilon_sigma"])
    a_min = float(sim_cfg["ar"]["alpha_min"])
    a_max = float(sim_cfg["ar"]["alpha_max"])

    N = len(graph.vmf_list)
    X = np.zeros((T, N), dtype=np.float32)

    role_priors = sim_cfg.get("role_priors", {})

    alpha = np.zeros(N, dtype=np.float32)
    a_coef = np.zeros(N, dtype=np.float32)
    sigma  = np.zeros(N, dtype=np.float32)
    seas_kind = ["none"]*N
    seas_amp = np.zeros(N, dtype=np.float32)

    for i, v in enumerate(graph.vmf_list):
        role = v["role"]
        rcfg = role_priors.get(role, {})
        alpha[i] = float(rcfg.get("alpha", 0.0))
        tier = float(rcfg.get("beta", {}).get("tier", (a_min + a_max)/2))
        a_raw = float(rng.uniform(a_min, a_max))
        a_coef[i] = 0.5*a_raw + 0.5*min(max(tier, a_min), a_max)
        sigma_scale = float(rcfg.get("noise", {}).get("sigma_scale", 1.0))
        sigma[i] = eps_sigma_base * sigma_scale

    in_edges = [[] for _ in range(N)]
    for e in graph.edges:
        si = graph.vmf_index[e["src"]]
        di = graph.vmf_index[e["dst"]]
        in_edges[di].append((si, int(e["delay"]), float(e["weight"]), e["kind"]))

    X[0, :] = alpha + rng.normal(0.0, 1.0, size=N).astype(np.float32) * sigma

    def past_value(idx, t, delay):
        tt = t - delay
        if tt >= 0:
            return X[tt, idx]
        else:
            return X[0, idx]

    for t in range(1, T):
        prop_sum = np.zeros(N, dtype=np.float32)
        for di in range(N):
            s = 0.0
            for si, dly, w, _ in in_edges[di]:
                s += w * past_value(si, t, dly)
            prop_sum[di] = s


        noise = rng.normal(0.0, 1.0, size=N).astype(np.float32) * sigma
        X[t, :] = alpha + a_coef * X[t-1, :] + prop_sum + noise

    meta = {
        "alpha": alpha.tolist(),
        "a_coef": a_coef.tolist(),
        "sigma": sigma.tolist(),
        "season_kind": seas_kind,
        "season_amp": seas_amp.tolist(),
        "L_warmup": L_warmup
    }
    return X, meta


def sample_roots_with_constraints(X, graph: VMFGraph, topo, sim_cfg, service_candidates, rng):
    T, N = X.shape
    anom = sim_cfg["anomaly"]
    amp_lo, amp_hi = anom["amp_range"]
    w_lo, w_hi = anom["width_steps"]
    type_choices = list(anom.get("type_choices", ["spike","shift","burst"]))
    burst_modes = list(anom.get("burst", {}).get("mode_choices", ["multi_pulse","damped_ring","exp_decay"]))
    target_min, target_max = anom.get("target_roots_per_mw", [0, 0])
    target = int(rng.integers(int(target_min), int(target_max)+1)) if target_max >= target_min else 0

    shift_ratio_max = float(anom.get("shift_ratio_max", 0.2))
    min_gap = int(anom.get("min_gap_steps_per_vmf", 2))
    overlap_limit = int(anom.get("window_overlap_limit", 2))
    tau_max = int(max(sim_cfg["propagation"]["delay_choices"])) if sim_cfg["propagation"]["delay_choices"] else 3
    H_out = int(topo.get("ow_scheme", {}).get("H_out", 30))
    fiber_conc_max = int(anom.get("max_concurrent_roots_per_fiber", 1))

    priors = sim_cfg.get("role_priors", {})
    p_root = {
        "SRC":   float(priors.get("SRC", {}).get("anomaly_bias", {}).get("p_root", 0.45)),
        "RELAY": float(priors.get("RELAY", {}).get("anomaly_bias", {}).get("p_root", 0.35)),
        "SNK":   float(priors.get("SNK", {}).get("anomaly_bias", {}).get("p_root", 0.20)),
    }
    z = sum(p_root.values()) or 1.0
    for k in p_root: p_root[k] /= z
    role_list = ["SRC","RELAY","SNK"]
    role_probs = np.array([p_root[r] for r in role_list], dtype=np.float64)

    events = []
    vmf_last_t0 = {}
    vmf_windows = defaultdict(list)
    fiber_windows = defaultdict(list)
    shift_used = 0

    svc_ids = list(service_candidates.keys())
    rng.shuffle(svc_ids)

    L_warmup = int(sim_cfg["time"]["L_warmup"])
    t0_lo = max(5, L_warmup) + 1
    t0_hi = max(t0_lo+1, T - 1 - (tau_max + H_out) - 1)

    def window_ok(win_list, L, R, limit):
        cnt = 0
        for l, r in win_list:
            if not (R < l or L > r):
                cnt += 1
                if cnt >= limit:
                    return False
        return True

    def fiber_ok(fid, L, R):
        if fid is None:
            return True
        for l, r in fiber_windows.get(fid, []):
            if not (R < l or L > r):
                return False
        return True

    def push_window(win_list, L, R):
        win_list.append([L, R])

    for sid in svc_ids:
        if len(events) >= target:
            break
        cand = service_candidates[sid]
        if not cand:
            continue

        by_role = defaultdict(list)
        for vid in cand:
            by_role[graph.role_of.get(vid, "RELAY")].append(vid)

        role = rng.choice(role_list, p=role_probs)
        choices = by_role.get(role, []) or cand

        vid = str(rng.choice(choices))
        i = int(graph.vmf_index[vid])

        types_now = list(type_choices)
        if (shift_used + 1) > math.floor(shift_ratio_max * max(1, target)):
            types_now = [t for t in types_now if t != "shift"] or ["spike","burst"]
        typ = str(rng.choice(types_now))
        bmode = str(rng.choice(burst_modes)) if typ == "burst" else None

        if t0_hi <= t0_lo:
            continue
        t0 = int(rng.integers(t0_lo, t0_hi+1))
        width = int(rng.integers(int(w_lo), int(w_hi)+1))
        amp = float(rng.uniform(float(amp_lo), float(amp_hi)))

        L = t0
        R = min(T-1, t0 + tau_max + H_out)

        last = vmf_last_t0.get(vid, -10**9)
        if (t0 - last) < min_gap:
            continue
        if not window_ok(vmf_windows[vid], L, R, overlap_limit):
            continue
        fid = graph.meta.get(vid, {}).get("fiber_group_id")
        if not fiber_ok(fid, L, R):
            continue

        base_std = float(np.std(X[max(0, t0-100):t0+1, i]) + 1e-6)
        if typ == "spike":
            X[t0:t0+width, i] += amp * base_std
        elif typ == "shift":
            X[t0:, i] += amp * base_std
            shift_used += 1
        elif typ == "burst":
            t_idx = np.arange(t0, min(T, t0+width))
            if len(t_idx) > 0:
                mode = bmode
                if mode == "multi_pulse":
                    k = int(rng.integers(2, 4))
                    for _ in range(k):
                        c = int(rng.integers(t0, min(T, t0+width)))
                        w = int(max(1, rng.integers(2, max(3, width//3))))
                        X[c:min(T, c+w), i] += 0.6*amp * base_std
                elif mode == "damped_ring":
                    dt = (t_idx - t0).astype(np.float32)
                    X[t_idx, i] += (amp * base_std) * np.sin(0.5*dt) * np.exp(-0.1*dt)
                else:
                    dt = (t_idx - t0).astype(np.float32)
                    X[t_idx, i] += (amp * base_std) * np.exp(-0.15*dt)
        else:
            X[t0:t0+width, i] += amp * base_std

        events.append({
            "vmf_id": vid,
            "root_service_id": sid,
            "t0": int(t0),
            "width": int(width),
            "amp": float(amp),
            "type": typ,
            "burst_mode": bmode
        })

        vmf_last_t0[vid] = t0
        push_window(vmf_windows[vid], L, R)
        if fid is not None:
            push_window(fiber_windows[fid], L, R)
    

    





    


    return events, {"target": target, "placed": len(events), "shift_used": shift_used,
                    "tau_max": tau_max, "H_out": H_out}


def _smooth_bbe(bbe: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return bbe
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, bbe)

def _role_token(role: str | None) -> str:
    r = str(role or "").strip().upper()
    if r in {"SNK", "TERMINATION"}:
        return "_SNK_"
    if r in {"SRC", "SOURCE"}:
        return "_SRC_"
    return "_RELAY_"

_NOISE_SCALE: float = 1.0  # Global noise scale multiplier (set by --noise-scale)
_EQUALIZE_BBE: bool = False  # When True, all root-cause events use same mult (~3.5)

def _noisy(base: float, frac: float, rng) -> float:
    """Multiplicative uniform noise: base * U(1-frac*scale, 1+frac*scale)."""
    if rng is None:
        return base
    scaled_frac = min(frac * _NOISE_SCALE, 0.95)
    return float(base * rng.uniform(1.0 - scaled_frac, 1.0 + scaled_frac))


def _noisy_int(base: int, delta: int, rng) -> int:
    """Integer noise: base + randint(-delta, +delta), min 1.  Scales with _NOISE_SCALE."""
    if rng is None or delta == 0:
        return base
    scaled_delta = max(1, int(delta * _NOISE_SCALE))
    return max(1, base + int(rng.integers(-scaled_delta, scaled_delta + 1)))


def _profile_for_alarm(name: str, rng=None) -> dict:
    n = (name or "").strip().upper()
    profile = {"kind": "spike", "mult": 1.0, "bursts": 1, "ramp_down": False}
    if not n:
        return profile
    if "LINE_DISCONNECT" in n:
        profile = {
            "kind": "step_recovery",
            "mult": _noisy(3.0, 0.20, rng),
            "bursts": _noisy_int(4, 1, rng),
            "ramp_down": True,
            "tail_factor": 0.7,
            "bber_mult": _noisy(1.2, 0.20, rng),
            "es_threshold_mult": _noisy(0.8, 0.15, rng),
        }
    elif "FIBER_CUT" in n:
        profile = {
            "kind": "step",
            "mult": _noisy(5.5, 0.20, rng),
            "bursts": 1,
            "ramp_down": False,
            "bber_mult": _noisy(3.0, 0.20, rng),
            "es_threshold_mult": _noisy(0.6, 0.15, rng),
        }
    elif "FIBER_CRACK" in n:
        profile = {
            "kind": "step_recovery",
            "mult": _noisy(3.5, 0.20, rng),
            "bursts": _noisy_int(6, 2, rng),
            "ramp_down": True,
            "tail_factor": 0.8,
            "bber_mult": _noisy(5.5, 0.20, rng),
            "es_threshold_mult": _noisy(0.4, 0.15, rng),
        }
    elif "FIBER_AGING" in n or "DEG" in n:
        profile = {
            "kind": "ramp",
            "mult": _noisy(2.8, 0.20, rng),
            "bursts": _noisy_int(1, 1, rng),
            "ramp_down": True,
            "tail_factor": 0.6,
            "bber_mult": _noisy(2.0, 0.20, rng),
            "es_threshold_mult": _noisy(0.7, 0.15, rng),
        }
    elif "BUFFER_OVERFLOW" in n or "EXC" in n:
        profile = {
            "kind": "burst",
            "mult": _noisy(4.6, 0.20, rng),
            "bursts": _noisy_int(8, 2, rng),
            "ramp_down": True,
            "tail_factor": 0.5,
            "bber_mult": _noisy(3.0, 0.20, rng),
            "es_threshold_mult": _noisy(0.5, 0.15, rng),
        }
    elif "AIS" in n:
        profile = {"kind": "step", "mult": 2.0, "bursts": 1, "ramp_down": False}
    elif "BDI" in n:
        profile = {"kind": "spike", "mult": 0.8, "bursts": 1, "ramp_down": False}
    elif "LOS" in n or "LOF" in n:
        profile = {"kind": "step", "mult": 1.9, "bursts": 1, "ramp_down": False}
    elif "XCON_FABRIC_FAULT" in n:
        profile = {
            "kind": "step_recovery",
            "mult": _noisy(4.2, 0.20, rng),
            "bursts": _noisy_int(5, 2, rng),
            "ramp_down": True,
            "tail_factor": 0.6,
            "bber_mult": _noisy(3.2, 0.20, rng),
            "es_threshold_mult": _noisy(0.55, 0.15, rng),
        }
    elif "XCON_PORT_DOWN" in n:
        profile = {
            "kind": "step",
            "mult": _noisy(3.2, 0.20, rng),
            "bursts": 1,
            "ramp_down": False,
            "bber_mult": _noisy(2.8, 0.20, rng),
            "es_threshold_mult": _noisy(0.6, 0.15, rng),
        }
    # Override mult when equalizing bbe_max across all event types
    if _EQUALIZE_BBE:
        profile["mult"] = _noisy(3.5, 0.20, rng)
    return profile

def _apply_profile_to_bbe(series: np.ndarray, t0: int, t1: int, base_spike: float,
                          profile: dict, rng: np.random.Generator | None = None) -> None:
    if t0 >= t1:
        return
    if rng is None:
        rng = np.random.default_rng()
    kind = profile.get("kind", "spike")
    mult = float(profile.get("mult", 1.0))
    bursts = int(profile.get("bursts", 1))
    ramp_down = bool(profile.get("ramp_down", False))
    width = max(1, t1 - t0)
    if kind == "step":
        series[t0:t1] = np.maximum(series[t0:t1], base_spike * mult)
    elif kind == "ramp":
        ramp = np.linspace(0.2, 1.0, width, dtype=np.float32)
        if ramp_down:
            ramp = np.linspace(1.0, 0.2, width, dtype=np.float32)
        series[t0:t1] = np.maximum(series[t0:t1], base_spike * mult * ramp)
    elif kind == "burst":
        for _ in range(max(1, bursts)):
            b_w = max(1, int(width * rng.uniform(0.05, 0.15)))
            b0 = int(rng.integers(t0, max(t0 + 1, t1 - b_w + 1)))
            b1 = min(t1, b0 + b_w)
            series[b0:b1] = np.maximum(series[b0:b1], base_spike * mult)
        if ramp_down:
            tail0 = max(t0, t1 - max(1, int(width * 0.25)))
            tail = np.linspace(1.0, 0.3, t1 - tail0, dtype=np.float32)
            series[tail0:t1] = np.maximum(series[tail0:t1], base_spike * mult * tail)
    elif kind == "step_recovery":
        series[t0:t1] = np.maximum(series[t0:t1], base_spike * mult)
        for _ in range(max(1, bursts)):
            b_w = max(1, int(width * rng.uniform(0.03, 0.08)))
            b0 = int(rng.integers(t0, max(t0 + 1, t1 - b_w + 1)))
            b1 = min(t1, b0 + b_w)
            series[b0:b1] = np.maximum(series[b0:b1], base_spike * (mult * 1.2))
        if ramp_down:
            tail_factor = float(profile.get("tail_factor", 0.5))
            tail_len = max(1, int(width * tail_factor))
            tail0 = t1
            tail1 = min(len(series), t1 + tail_len)
            if tail1 > tail0:
                tail = np.linspace(1.0, 0.1, tail1 - tail0, dtype=np.float32)
                series[tail0:tail1] = np.maximum(series[tail0:tail1], base_spike * mult * tail)
    else:
        series[t0:t1] = np.maximum(series[t0:t1], base_spike * mult)

def _apply_profile_to_bber_es(df_node: pd.DataFrame, idx, profile: dict,
                              bber_scale: float, es_threshold: float) -> None:
    bber_mult = profile.get("bber_mult")
    es_thresh_mult = profile.get("es_threshold_mult")
    if bber_mult is None and es_thresh_mult is None:
        return
    bbe_vals = df_node.loc[idx, "BBE"].to_numpy(dtype=np.float32)
    bbe_smooth = _smooth_bbe(bbe_vals[:, None], int(os.getenv("BBER_SMOOTH_WINDOW", "10"))).reshape(-1)
    if bber_mult is not None and "BBER" in df_node.columns:
        df_node.loc[idx, "BBER"] = (bbe_smooth / max(1.0, bber_scale) * float(bber_mult)).astype(np.float32)
    if es_thresh_mult is not None and "ES" in df_node.columns:
        thresh = es_threshold * float(es_thresh_mult)
        df_node.loc[idx, "ES"] = (bbe_smooth / thresh).clip(0, 1).astype(np.float32)

def apply_alarm_spikes_from_flow(
    alarm_flow_csv: str | Path,
    topo: dict,
    data_dir: str | Path,
    base_time: datetime,
    step_seconds: int,
    bbe_spike: float,
    duration_sec: int,
    stagger_sec: int,
    sim_cfg: dict | None = None,
    root_board: str | None = None,
    root_event: str | None = None,
    seed: int = 42,
) -> None:
    alarm_flow_csv = Path(alarm_flow_csv)
    if not alarm_flow_csv.exists():
        return
    try:
        df_flow = pd.read_csv(alarm_flow_csv)
    except Exception:
        return
    profile_rng = np.random.default_rng(seed)
    # Cache profile for the root event so all boards share the same noisy profile.
    # For multi-failure: cache a profile per root event using provenance columns.
    _root_event_profile = _profile_for_alarm(root_event or "", rng=profile_rng) if root_event else None
    _root_profiles_cache: dict[str, dict] = {}
    if root_event:
        _root_profiles_cache[root_event] = _root_event_profile
    if df_flow.empty:
        return
    node_map = topo.get("node_map") or {}
    if not node_map:
        return
    cols = {str(c).strip().lower(): c for c in df_flow.columns}
    c_board = cols.get("proptoboard") or cols.get("board")
    c_role = cols.get("proprole") or cols.get("role")
    c_arr = cols.get("arrivaltime")
    c_rule = cols.get("ruleidx")
    c_alarm = cols.get("propalarm") or cols.get("sourcealarm") or cols.get("alarm")
    c_clear = cols.get("cleartime")
    c_note = cols.get("rulenote")
    c_edge = cols.get("viaedgetype")
    c_root_board = cols.get("rootcauseboard")
    c_root_event = cols.get("rootcauseevent")
    if not (c_board and c_arr):
        return
    dur_steps = max(1, int(duration_sec / max(1, step_seconds)))
    events_by_node = defaultdict(list)
    for _, r in df_flow.iterrows():
        if c_rule and pd.notna(r.get(c_rule)):
            try:
                if int(r.get(c_rule)) < 0:
                    continue
            except Exception:
                pass
        board = str(r.get(c_board)).strip()
        if not board:
            continue
        parent, _ = _parse_board_name(board)
        if not parent or parent not in node_map:
            continue
        role = r.get(c_role) if c_role else None
        alarm = str(r.get(c_alarm)).strip() if c_alarm else ""
        arr = _parse_time_str(str(r.get(c_arr))) if pd.notna(r.get(c_arr)) else None
        if arr is None:
            continue
        clear = None
        if c_clear and pd.notna(r.get(c_clear)):
            clear = _parse_time_str(str(r.get(c_clear)))
        t0 = int((arr - base_time).total_seconds() / step_seconds)
        if t0 < 0:
            continue
        node_id = int(node_map[parent])
        dur_steps_row = dur_steps
        if clear and clear > arr:
            dur_steps_row = max(1, int((clear - arr).total_seconds() / max(1, step_seconds)))
        note = str(r.get(c_note) or "").lower() if c_note else ""
        edge = str(r.get(c_edge) or "").lower() if c_edge else ""
        is_regen = ("regen_boundary" in note) or ("regen_boundary" in edge)
        row_root_event = str(r.get(c_root_event)).strip() if c_root_event and pd.notna(r.get(c_root_event)) else ""
        row_root_board = str(r.get(c_root_board)).strip() if c_root_board and pd.notna(r.get(c_root_board)) else ""
        events_by_node[node_id].append((t0, role, alarm, dur_steps_row, is_regen, board, row_root_event, row_root_board))

    stagger_steps = max(0, int(stagger_sec / max(1, step_seconds)))
    for node_id, events in events_by_node.items():
        ts_path = Path(data_dir) / f"Node_{node_id}" / f"timeseries_{node_id}.csv"
        if not ts_path.exists():
            continue
        df_node = pd.read_csv(ts_path)
        if df_node.empty:
            continue
        events = sorted(events, key=lambda x: x[0])
        min_gap = max(stagger_steps, dur_steps + 1)
        expanded = []
        last_end = -min_gap
        for t0, role, alarm, dur_steps_row, is_regen, evt_board, row_root_evt, row_root_brd in events:
            t_shift = max(t0, last_end + min_gap)
            expanded.append((t_shift, role, alarm, dur_steps_row, is_regen, evt_board, row_root_evt, row_root_brd))
            last_end = t_shift + dur_steps_row
        ais_bbe_mult = float(os.getenv("AIS_BBE_MULT", "2.0"))
        bdi_bbe_mult = float(os.getenv("BDI_BBE_MULT", "0.7"))
        regen_ais_bbe_mult = float(os.getenv("REGEN_AIS_BBE_MULT", str(ais_bbe_mult)))
        regen_bdi_bbe_mult = float(os.getenv("REGEN_BDI_BBE_MULT", str(bdi_bbe_mult)))
        regen_extra_steps = max(0, int(float(os.getenv("REGEN_EXTRA_SEC", "0")) / max(1, step_seconds)))
        regen_cfg = (sim_cfg or {}).get("propagation", {}).get("regen", {})
        regen_otuk_reset_mult = float(os.getenv("REGEN_OTUK_RESET_MULT", str(regen_cfg.get("otuk_reset_mult", 0.1))))
        regen_oduk_passthrough_mult = float(os.getenv("REGEN_ODUK_PASSTHROUGH_MULT", str(regen_cfg.get("oduk_passthrough_mult", 1.0))))
        for t0, role, alarm, dur_steps_row, is_regen, evt_board, row_root_evt, row_root_brd in expanded:
            t1 = t0 + dur_steps_row + (regen_extra_steps if is_regen else 0)
            id_col = "board" if "board" in df_node.columns else "vmf_id"
            if id_col == "board":
                # Match by exact board name from alarm flow
                mask = df_node[id_col].astype(str).str.strip() == evt_board
                if not mask.any():
                    # Fallback: match by board kind → role mapping
                    target_role = str(role or "").strip().upper()
                    if target_role in {"SNK", "TERMINATION"}:
                        target_role = "SNK"
                    elif target_role in {"SRC", "SOURCE"}:
                        target_role = "SRC"
                    else:
                        target_role = "RELAY"
                    def _board_matches_role(board_name, trole=target_role):
                        _, kind, _ = _parse_token(str(board_name))
                        board_role = _role_for_kind(kind) if kind else None
                        if board_role is None:
                            return False
                        if board_role == "TERMINATION":
                            board_role = "SNK"
                        return board_role == trole
                    mask = df_node[id_col].apply(_board_matches_role)
            else:
                token = _role_token(role)
                mask = df_node[id_col].astype(str).str.contains(token, na=False)
            time_mask = (df_node["t"] >= t0) & (df_node["t"] < t1)
            idx = mask & time_mask
            if idx.any():
                alarm_upper = alarm.upper()
                # Classify alarm layer for regen differentiation
                is_otuk_alarm = any(x in alarm_upper for x in ["OTUK", "R_LOS", "LOF", "LOM"])
                is_oduk_alarm = "ODUK" in alarm_upper
                bbe_val = float(bbe_spike)
                if is_regen:
                    # Layer-specific multipliers at regen boundaries
                    if is_otuk_alarm:
                        bbe_val *= regen_otuk_reset_mult
                    elif is_oduk_alarm:
                        bbe_val *= regen_oduk_passthrough_mult
                    elif "AIS" in alarm_upper:
                        bbe_val *= regen_ais_bbe_mult
                    elif "BDI" in alarm_upper:
                        bbe_val *= regen_bdi_bbe_mult
                else:
                    if "AIS" in alarm_upper:
                        bbe_val *= ais_bbe_mult
                    elif "BDI" in alarm_upper:
                        bbe_val *= bdi_bbe_mult
                # Use per-root event profile from provenance columns (multi-failure aware)
                _ROOT_EVENT_NAMES = {
                    "LINE_DISCONNECT", "FIBER_CUT", "FIBER_CRACK", "FIBER_AGING",
                    "XCON_PORT_DOWN", "XCON_BUFFER_OVERFLOW", "XCON_FABRIC_FAULT",
                }
                # Determine which root event this alarm row belongs to
                effective_root_event = row_root_evt or root_event or ""
                effective_root_board = row_root_brd or root_board or ""
                if effective_root_event and alarm_upper in _ROOT_EVENT_NAMES and evt_board != effective_root_board:
                    prof = {"kind": "spike", "mult": 1.0, "bursts": 1, "ramp_down": False}
                elif effective_root_event and alarm_upper in _ROOT_EVENT_NAMES:
                    # Cache per root event for consistency
                    if effective_root_event not in _root_profiles_cache:
                        _root_profiles_cache[effective_root_event] = _profile_for_alarm(
                            effective_root_event, rng=profile_rng)
                    prof = _root_profiles_cache[effective_root_event]
                elif _root_event_profile and alarm_upper in _ROOT_EVENT_NAMES:
                    prof = _root_event_profile
                else:
                    prof = _profile_for_alarm(alarm, rng=profile_rng)
                for vid, g in df_node.loc[idx].groupby(id_col):
                    series = df_node.loc[g.index, "BBE"].to_numpy().copy()
                    local_t0 = int(g["t"].min())
                    local_t1 = int(g["t"].max()) + 1
                    _apply_profile_to_bbe(series, local_t0 - local_t0, local_t1 - local_t0, bbe_val, prof)
                    df_node.loc[g.index, "BBE"] = series
                _apply_profile_to_bber_es(df_node, idx, prof,
                                          float(os.getenv("BBER_SCALE", "1000.0")),
                                          float(os.getenv("ES_THRESHOLD", "9.0")))
        if "BBER" in df_node.columns or "ES" in df_node.columns:
            bber_scale = float(os.getenv("BBER_SCALE", "1000.0"))
            es_threshold = float(os.getenv("ES_THRESHOLD", "9.0"))
            smooth_window = int(os.getenv("BBER_SMOOTH_WINDOW", "10"))
            updated = []
            id_col2 = "board" if "board" in df_node.columns else "vmf_id"
            for vid, g in df_node.groupby(id_col2):
                bbe_vals = g["BBE"].to_numpy(dtype=np.float32)
                bbe_smooth = _smooth_bbe(bbe_vals[:, None], smooth_window).reshape(-1)
                if "BBER" in df_node.columns:
                    df_node.loc[g.index, "BBER"] = (bbe_smooth / max(1.0, bber_scale)).astype(np.float32)
                if "ES" in df_node.columns:
                    df_node.loc[g.index, "ES"] = (bbe_smooth / es_threshold).clip(0, 1).astype(np.float32)
                updated.append(vid)
        df_node.to_csv(ts_path, index=False)

def latent_to_metrics(X, metrics, rng):
    T, N = X.shape
    out = {}
    Xn = (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-6)
    lam = np.log1p(np.exp(0.8 * Xn))
    metrics_upper = [m.upper() for m in metrics]
    bber_scale = float(os.getenv("BBER_SCALE", "1000.0"))
    es_threshold = float(os.getenv("ES_THRESHOLD", "9.0"))
    smooth_window = int(os.getenv("BBER_SMOOTH_WINDOW", "10"))
    bbe = None
    if any(mu in {"BBE", "BBER", "ES"} for mu in metrics_upper):
        bbe = rng.poisson(lam).astype(np.float32)
    bbe_smooth = None
    if bbe is not None and any(mu in {"BBER", "ES"} for mu in metrics_upper):
        bbe_smooth = _smooth_bbe(bbe, smooth_window)

    for m in metrics:
        mu = m.upper()
        if mu == "LAT":
            continue
        elif mu == "BBE":
            out[m] = bbe
        elif mu == "BBER":
            base = bbe if bbe is not None else rng.poisson(lam).astype(np.float32)
            if bbe_smooth is not None:
                base = bbe_smooth
            out[m] = (base / max(1.0, bber_scale)).astype(np.float32)
        elif mu == "ES":
            base = bbe if bbe is not None else rng.poisson(lam).astype(np.float32)
            if bbe_smooth is not None:
                base = bbe_smooth
            out[m] = (base >= es_threshold).astype(np.float32)
        else:
            out[m] = X.astype(np.float32)
    return out


def dump_per_node(graph: VMFGraph, X, metrics_map, outdir, sidecar_extra,
                  topo=None, failures=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, N = X.shape
    vmf_ids = [v["id"] for v in graph.vmf_list]

    # Build VMF→board mapping using topology board_map
    vmf_board_map = {}
    if topo:
        node_map = topo.get("node_map") or {}
        board_map = topo.get("board_map") or {}
        parent_by_id = _inverse_node_map(node_map)
        failure_pref = _build_failure_pref(failures or [])
        for vid in vmf_ids:
            node_id = graph.node_of.get(vid)
            parent = parent_by_id.get(int(node_id)) if node_id is not None else None
            if parent:
                board = _board_for_role(parent, graph.role_of.get(vid), board_map, failure_pref)
                if board:
                    vmf_board_map[vid] = board

    prop_edges_glob, local_edges_glob = [], []
    for e in graph.edges:
        src_i = graph.vmf_index[e["src"]]
        dst_i = graph.vmf_index[e["dst"]]
        rec = [src_i, dst_i, e["kind"], int(e["delay"]), float(e["weight"])]
        if e["kind"] == "prop":
            prop_edges_glob.append(rec)
        else:
            local_edges_glob.append(rec)

    for node, idxs in graph.node_to_vmfs.items():
        ndir = os.path.join(outdir, f"Node_{node}")
        safe_mkdir(ndir); safe_mkdir(os.path.join(ndir, "viz"))

        idxs_sorted = sorted(idxs, key=lambda i: vmf_ids[i])
        node_vmf_ids = [vmf_ids[i] for i in idxs_sorted]

        rows = []
        cols = list(metrics_map.keys())
        for gi in idxs_sorted:
            vid = vmf_ids[gi]
            board = vmf_board_map.get(vid, vid)
            for t in range(T):
                row = {"t": t, "board": board}
                for m in cols:
                    row[m] = float(metrics_map[m][t, gi])
                rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(ndir, f"timeseries_{node}.csv"), index=False)

        prop_node, local_node = [], []
        idxs_set = set(idxs_sorted)
        for src_i, dst_i, kind, dly, w in prop_edges_glob:
            if src_i in idxs_set or dst_i in idxs_set:
                prop_node.append([src_i, dst_i, kind, dly, w])
        for src_i, dst_i, kind, dly, w in local_edges_glob:
            if src_i in idxs_set or dst_i in idxs_set:
                local_node.append([src_i, dst_i, kind, dly, w])

        prop_edges_by_id = [{
            "src_id": vmf_ids[src_i],
            "dst_id": vmf_ids[dst_i],
            "kind": kind, "delay": int(dly), "weight": float(w)
        } for src_i, dst_i, kind, dly, w in prop_node]
        local_edges_by_id = [{
            "src_id": vmf_ids[src_i],
            "dst_id": vmf_ids[dst_i],
            "kind": kind, "delay": int(dly), "weight": float(w)
        } for src_i, dst_i, kind, dly, w in local_node]

        is_regen_boundary = {vid: bool(graph.meta.get(vid, {}).get("is_regen_boundary", False))
                             for vid in node_vmf_ids}
        is_root_vmf = {vid: False for vid in node_vmf_ids}
        for ev in sidecar_extra.get("anomaly_events", []):
            if ev["vmf_id"] in is_root_vmf:
                is_root_vmf[ev["vmf_id"]] = True

        sidecar = {
            "node_id": node,
            "vmf_ids_order": node_vmf_ids,
            "prop_edges": prop_node,
            "local_edges": local_node,
            "prop_edges_by_id": prop_edges_by_id,
            "local_edges_by_id": local_edges_by_id,
            "is_regen_boundary": is_regen_boundary,
            "eligible_root_roles": ["SRC", "RELAY", "SNK"],
            "is_root_vmf": is_root_vmf,
            "delay_cfg": {"default_tau_max": int(sidecar_extra.get("tau_max", 3))},
            "thresholds": {"resid_p95": None, "bbbe_rate_p99": None},
            "anomaly_events": [ev for ev in sidecar_extra.get("anomaly_events", [])
                               if graph.node_of.get(ev["vmf_id"], -1) == node],
        }
        with open(os.path.join(ndir, f"sidecar_{node}.json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)

        vmf_roles = {vid: graph.role_of[vid] for vid in node_vmf_ids}
        vmf_meta_list = []
        for vid in node_vmf_ids:
            m = graph.meta.get(vid, {}).copy()
            m["node_id"] = graph.node_of.get(vid, m.get("node_id"))
            m["service_ids"] = sorted(list(set(m.get("service_ids", []))))
            vmf_meta_list.append({vid: m})
        topo_node = {
            "node_id": int(node),
            "vmf_roles": vmf_roles,
            "vmf_meta": vmf_meta_list
        }
        with open(os.path.join(ndir, f"topology_{node}.json"), "w", encoding="utf-8") as f:
            json.dump(topo_node, f, ensure_ascii=False, indent=2)

        try:
            if "BBE" in metrics_map and len(idxs_sorted) > 0:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(9, 4))
                for gi in idxs_sorted[:3]:
                    plt.plot(metrics_map["BBE"][:, gi], label=vmf_ids[gi])
                plt.legend(fontsize=8)
                plt.title(f"Node {node} - BBE (up to 3 VMFs)")
                plt.xlabel("t"); plt.ylabel("BBE")
                plt.tight_layout()
                plt.savefig(os.path.join(ndir, "viz", f"propagation_{node}.png"), dpi=140)
                plt.close()
        except Exception as e:
            print(f"[warn] viz for node {node} failed: {e}")


def _parse_step_seconds(step_val) -> int:
    s = str(step_val).strip().lower()
    if s.endswith("ms"):
        return max(1, int(round(float(s[:-2]) / 1000.0)))
    if s.endswith("s"):
        return max(1, int(round(float(s[:-1]))))
    try:
        return max(1, int(round(float(s))))
    except Exception:
        return 1

def _normalize_path_role(role: str, vmf_id: str) -> str:
    r = str(role or "").strip().upper()
    if r in {"RELAY_IN", "RELAY_OUT"}:
        return "RELAY"
    if r in {"SNK", "SINK"}:
        return "TERMINATION"
    if r in {"SRC", "RELAY", "TERMINATION"}:
        return r
    vid = str(vmf_id or "").upper()
    if "_SRC_" in vid:
        return "SRC"
    if "_SNK_" in vid:
        return "TERMINATION"
    if "_RELAY_" in vid:
        return "RELAY"
    return "RELAY"

_TOKEN_PAT = re.compile(
    r"^(?P<node>[^-]+)-(?P<kind>[A-Za-z]+?)(?P<idx>\d+)(?P<tag>\$0)?(?:-(?P<label>[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*))?$",
    re.IGNORECASE,
)

def _parse_token(token: str) -> tuple[str, str | None, str | None]:
    t = str(token or "").strip()
    m = _TOKEN_PAT.match(t)
    if not m:
        return t, None, None
    node = m.group("node")
    kind = m.group("kind")
    idx = int(m.group("idx"))
    base = f"{node}-{kind}{idx:03d}$0"
    label = m.group("label")
    return base, kind.upper(), label

def _role_from_label(kind: str | None, label: str | None) -> str | None:
    if label:
        lu = label.upper()
        if "CREATION" in lu or lu.endswith("SRC"):
            return "SRC"
        if "TERMINATION" in lu:
            return "TERMINATION"
    if kind and kind.upper() in {"OA", "OM", "OD", "FIU", "SC2"}:
        return "RELAY"
    return None

def write_paths_from_lightpaths(lightpaths_file: str | Path, out_path: str) -> int:
    p = Path(lightpaths_file)
    if not p.exists():
        return 0
    lines = []
    with p.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            toks = [t.strip() for t in raw.split(",") if t.strip()]
            if not toks:
                continue
            labeled = []
            for i, tok in enumerate(toks):
                base, kind, label = _parse_token(tok)
                role = _role_from_label(kind, label)
                if role is None:
                    if i < 3:
                        role = "SRC"
                    elif i >= len(toks) - 3:
                        role = "TERMINATION"
                    else:
                        role = "RELAY"
                labeled.append(f"{base}-{role}")
            lines.append(",".join(labeled))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)

def _inverse_node_map(node_map: dict) -> dict[int, str]:
    return {int(v): k for k, v in node_map.items()}

def _board_for_role(parent: str, role: str, board_map: dict, failure_pref: dict | None = None) -> str | None:
    pref = (failure_pref or {}).get((parent, role))
    if pref:
        return pref
    kinds = board_map.get(parent, {}) if board_map else {}
    if role == "SRC":
        for k in ("TRIBUTARY", "XCON"):
            if kinds.get(k):
                return kinds[k][0]
    if role == "SNK":
        if kinds.get("LINE"):
            return kinds["LINE"][0]
    if role == "RELAY":
        for k in ("OA", "OM", "OD", "FIU", "SC2"):
            if kinds.get(k):
                return kinds[k][0]
    return None

def _build_failure_pref(failures: list[dict]) -> dict:
    pref = {}
    for f in failures or []:
        parent, kind = _parse_board_name(f.get("Board"))
        if not parent or not kind:
            continue
        role = _forced_role_for_event(f.get("Event")) or _normalize_failure_role(f.get("BoardRole")) or _role_for_kind(kind)
        pref[(parent, role)] = f.get("Board")
    return pref

def write_paths_from_topology(topo: dict, out_path: str, failures: list[dict] | None = None) -> int:
    services = topo.get("services") or []
    node_map = topo.get("node_map") or {}
    board_map = topo.get("board_map") or {}
    parent_by_id = _inverse_node_map(node_map)
    failure_pref = _build_failure_pref(failures or [])
    lines = []
    for svc in services:
        vmf_path = svc.get("vmf_path") or []
        tokens = []
        last_board = None
        for vmf in vmf_path:
            vid = vmf.get("vmf_id")
            if not vid:
                continue
            role = _normalize_path_role(vmf.get("role"), vid)
            node_id = vmf.get("node")
            parent = parent_by_id.get(int(node_id)) if node_id is not None else None
            board = _board_for_role(parent, "SNK" if role == "TERMINATION" else role, board_map, failure_pref) if parent else None
            if board:
                if board == last_board:
                    continue
                tokens.append(f"{board}-{role}")
                last_board = board
            else:
                tokens.append(f"{vid}-{role}")
        if tokens:
            lines.append(",".join(tokens))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)

def write_local_alarms_from_bbe(graph: VMFGraph, metrics_map: dict, events: list,
                                 step_seconds: int, out_path: str, only_roots: bool = True,
                                 start_time: datetime | None = None,
                                 topo: dict | None = None,
                                 failures: list[dict] | None = None) -> int:
    bbe = metrics_map.get("BBE")
    if bbe is None:
        print("[warn] BBE metric not found; skipping local alarm extraction.")
        return 0
    vmf_ids = [v["id"] for v in graph.vmf_list]
    node_map = topo.get("node_map") if topo else {}
    board_map = topo.get("board_map") if topo else {}
    parent_by_id = _inverse_node_map(node_map or {})
    failure_pref = _build_failure_pref(failures or [])
    root_set = {ev.get("vmf_id") for ev in (events or [])} if only_roots else None
    rows = []
    if start_time is None:
        start_time = datetime.now().replace(microsecond=0)
    for idx, vid in enumerate(vmf_ids):
        if root_set and vid not in root_set:
            continue
        series = bbe[:, idx]
        diffs = np.diff(series, prepend=series[0])
        for t, (v, dv) in enumerate(zip(series, diffs)):
            if v >= 4 and dv >= 2:
                alarm = "BBE_SPIKE"
                sev = "Major"
            elif v >= 2 and dv >= 1:
                alarm = "BBE_SHIFT"
                sev = "Minor"
            else:
                continue
            ts = start_time + timedelta(seconds=step_seconds * t)
            board = None
            node_id = graph.node_of.get(vid)
            parent = parent_by_id.get(int(node_id)) if node_id is not None else None
            if parent:
                board = _board_for_role(parent, graph.role_of.get(vid), board_map, failure_pref)
            rows.append({
                "Time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "Board": board or vid,
                "LocalAlarm": alarm,
                "Severity": sev,
                "Value": float(v),
            })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return len(rows)


_BOARD_PAT = re.compile(r"^(?P<parent>[^-]+)-(?P<kind>[A-Za-z]+?)(?P<idx>\d+)\$0$")

def _parse_board_name(board: str) -> tuple[str | None, str | None]:
    s = str(board or "").strip()
    m = _BOARD_PAT.match(s)
    if m:
        return m.group("parent"), m.group("kind").upper()
    if "-" not in s and s:
        return s, None
    return None, None

def _role_for_kind(kind: str | None) -> str:
    k = (kind or "").upper()
    if k in {"LINE"}:
        return "SNK"
    if k in {"TRIBUTARY", "XCON"}:
        return "SRC"
    if k in {"OA", "OM", "OD", "FIU", "SC2"}:
        return "RELAY"
    return "RELAY"

def _read_failures_csv(path: str | Path) -> list[dict]:
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    c_board = cols.get("board") or cols.get("node") or list(df.columns)[0]
    c_event = cols.get("event") or cols.get("alarm") or list(df.columns)[1]
    c_time = cols.get("time") or cols.get("start") or cols.get("timestamp")
    c_sev = cols.get("severity")
    c_dur = cols.get("durationms") or cols.get("duration") or cols.get("holdms")
    c_role = cols.get("boardrole") or cols.get("role")
    out = []
    for _, r in df.iterrows():
        board = r.get(c_board)
        if pd.isna(board):
            continue
        out.append({
            "Board": str(board),
            "Event": str(r.get(c_event)) if c_event else "",
            "Severity": (str(r.get(c_sev)) if c_sev and pd.notna(r.get(c_sev)) else None),
            "Time": (str(r.get(c_time)) if c_time and pd.notna(r.get(c_time)) else None),
            "DurationMs": float(r.get(c_dur)) if (c_dur and pd.notna(r.get(c_dur))) else None,
            "BoardRole": (str(r.get(c_role)) if (c_role and pd.notna(r.get(c_role))) else None),
        })
    return out

def _parse_time_str(txt: str | None) -> datetime | None:
    if not txt:
        return None
    s = str(txt).strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

def _normalize_failure_role(role: str | None) -> str | None:
    if not role:
        return None
    r = str(role).strip().upper().replace(" ", "").replace("_", "-")
    if r in {"SRC"}:
        return "SRC"
    if r in {"RELAY"}:
        return "RELAY"
    if r in {"SNK", "TERMINATION"}:
        return "SNK"
    if r in {"OTUK-RELAY"}:
        return "RELAY"
    if r in {"ODUK-TERMINATION"}:
        return "SNK"
    if r in {"ODUK-CREATION"}:
        return "SRC"
    return None

def _forced_role_for_event(event: str | None) -> str | None:
    ev = str(event or "").strip().lower()
    if "line_disconnect" in ev:
        return "SRC"
    if "tributary_port_down" in ev:
        return "SNK"
    return None

def _remap_line_disconnect_to_creation(failures: list[dict], topo: dict) -> None:
    """Keep Line_Disconnect on the original Line board (no remap to XCON)."""
    for f in failures:
        ev = str(f.get("Event") or "").strip().lower()
        if "line_disconnect" not in ev:
            continue
        # Ensure SRC role is set for Line_Disconnect but keep the original board
        f["BoardRole"] = f.get("BoardRole") or "SRC"

def _build_vmf_adjacency(graph: VMFGraph, bidir: bool) -> dict[str, list[tuple[str, int]]]:
    adj: dict[str, list[tuple[str, int]]] = {}
    for e in graph.edges:
        src = e["src"]
        dst = e["dst"]
        delay = int(e.get("delay", 0))
        adj.setdefault(src, []).append((dst, delay))
        if bidir:
            adj.setdefault(dst, []).append((src, delay))
    return adj

def _spread_failure_impacts(
    graph: VMFGraph,
    metrics_map: dict,
    root_vmfs: list[str],
    t0: int,
    t1: int,
    spike_value: float,
    max_hops: int,
    decay: float,
    min_scale: float,
    bidir: bool,
    step_seconds: int,
    delay_min_sec: int,
    delay_max_sec: int,
) -> None:
    if not root_vmfs:
        return
    bbe = metrics_map.get("BBE")
    if bbe is None:
        return
    adj = _build_vmf_adjacency(graph, bidir)
    min_steps = max(1, int(delay_min_sec / max(1, step_seconds)))
    max_steps = max(min_steps, int(delay_max_sec / max(1, step_seconds)))
    best: dict[str, tuple[int, int]] = {vid: (0, 0) for vid in root_vmfs}
    pq = [(0, 0, vid) for vid in root_vmfs]
    heapq.heapify(pq)
    while pq:
        cur_arr, hop, vid = heapq.heappop(pq)
        cur_best = best.get(vid)
        if cur_best and cur_arr > cur_best[0]:
            continue
        if hop >= max_hops:
            continue
        for nxt, edge_delay in adj.get(vid, []):
            next_hop = hop + 1
            if next_hop > max_hops:
                continue
            if edge_delay and edge_delay > 0:
                delay_steps = max(1, edge_delay)
            else:
                delay_steps = random.randint(min_steps, max_steps)
            next_arr = cur_arr + delay_steps
            prev = best.get(nxt)
            if prev is None or next_arr < prev[0]:
                best[nxt] = (next_arr, next_hop)
                heapq.heappush(pq, (next_arr, next_hop, nxt))
    for vid, (arr_steps, hop) in best.items():
        if hop == 0:
            continue
        scale = decay ** hop
        if scale < min_scale:
            continue
        gi = graph.vmf_index.get(vid)
        if gi is None:
            continue
        t_start = min(bbe.shape[0] - 1, t0 + arr_steps)
        t_end = min(bbe.shape[0], t_start + (t1 - t0))
        if t_start >= t_end:
            continue
        bbe[t_start:t_end, gi] = np.maximum(bbe[t_start:t_end, gi], float(spike_value) * scale)

def inject_failures_to_bbe(graph: VMFGraph, metrics_map: dict, failures: list[dict],
                           node_map: dict, step_seconds: int, lead_sec: int,
                           default_duration_sec: int, spike_value: float,
                           prop_hops: int,
                           prop_decay: float, prop_min_scale: float,
                           prop_bidir: bool, prop_delay_min_sec: int,
                           prop_delay_max_sec: int,
                           seed: int = 42) -> tuple[list[dict], datetime]:
    """Inject one or more failures into BBE time series.

    Multi-failure support: when multiple failures affect the same VMF, their
    BBE contributions are *summed* (additive combination).  Each failure is
    applied to a snapshot of the pre-failure BBE so that later failures never
    overwrite earlier ones; the final BBE is baseline + sum(deltas).
    """
    if not failures:
        return [], datetime.now().replace(microsecond=0)
    times = [_parse_time_str(f.get("Time")) for f in failures if f.get("Time")]
    times = [t for t in times if t]
    if times:
        start_time = min(times) - timedelta(seconds=max(0, int(lead_sec)))
    else:
        start_time = datetime.now().replace(microsecond=0)

    bbe = metrics_map.get("BBE")
    if bbe is None:
        return [], start_time

    profile_rng = np.random.default_rng(seed)
    events = []
    T = bbe.shape[0]

    # --- Multi-failure: snapshot baseline BBE so each failure's delta is
    # computed independently, then sum all deltas onto the baseline. ---
    bbe_baseline = bbe.copy()
    # Accumulator for per-VMF-column additive deltas from all failures.
    bbe_delta = np.zeros_like(bbe)

    for f in failures:
        parent, kind = _parse_board_name(f.get("Board"))
        if not parent or parent not in node_map:
            continue
        node_id = int(node_map[parent])
        target_role = _forced_role_for_event(f.get("Event")) or _normalize_failure_role(f.get("BoardRole")) or _role_for_kind(kind)
        vmf_ids = [vid for vid in graph.role_of.keys()
                   if graph.node_of.get(vid) == node_id and graph.role_of.get(vid) == target_role]
        if not vmf_ids:
            continue

        t_raw = _parse_time_str(f.get("Time"))
        if t_raw:
            t0 = int((t_raw - start_time).total_seconds() / step_seconds)
        else:
            t0 = 1
        t0 = max(0, min(T - 1, t0))

        dur_ms = f.get("DurationMs")
        if dur_ms is None:
            dur_steps = max(1, int(default_duration_sec / step_seconds))
        else:
            dur_steps = max(1, int(float(dur_ms) / 1000.0 / step_seconds))

        prof = _profile_for_alarm(str(f.get("Event") or ""), rng=profile_rng)
        for vid in vmf_ids:
            gi = graph.vmf_index.get(vid)
            if gi is None:
                continue
            t1 = min(T, t0 + dur_steps)
            # Apply profile to a scratch copy of the baseline so we can
            # compute the *delta* this failure contributes.
            scratch = bbe_baseline[:, gi].copy()
            _apply_profile_to_bbe(scratch, t0, t1, float(spike_value), prof)
            delta = scratch - bbe_baseline[:, gi]
            bbe_delta[:, gi] += delta
            events.append({
                "vmf_id": vid,
                "board": f.get("Board"),
                "event": f.get("Event") or "",
                "time": (t_raw.strftime("%Y-%m-%d %H:%M:%S") if t_raw else ""),
                "duration_ms": dur_ms,
            })
        t1 = min(T, t0 + dur_steps)
        _spread_failure_impacts(
            graph,
            metrics_map,
            vmf_ids,
            t0,
            t1,
            spike_value,
            prop_hops,
            prop_decay,
            prop_min_scale,
            prop_bidir,
            step_seconds,
            prop_delay_min_sec,
            prop_delay_max_sec,
        )

    # Combine: baseline + additive deltas from all failures
    bbe[:] = bbe_baseline + bbe_delta
    if "BBER" in metrics_map or "ES" in metrics_map:
        bber_scale = float(os.getenv("BBER_SCALE", "1000.0"))
        es_threshold = float(os.getenv("ES_THRESHOLD", "9.0"))
        smooth_window = int(os.getenv("BBER_SMOOTH_WINDOW", "10"))
        bbe_smooth = _smooth_bbe(bbe, smooth_window)
        if "BBER" in metrics_map:
            metrics_map["BBER"] = (bbe_smooth / max(1.0, bber_scale)).astype(np.float32)
        if "ES" in metrics_map:
            metrics_map["ES"] = (bbe_smooth / es_threshold).clip(0, 1).astype(np.float32)
    return events, start_time

def dump_effective_topology(topo, path="topology.yaml"):
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(topo, f, sort_keys=False, allow_unicode=True)
        print(f"[info] effective topology written to: {os.path.abspath(path)}")
    except Exception as e:
        print(f"[warn] failed to write effective topology.yaml: {e}")


def generate_multi_failure_csv(
    topology_yaml: str,
    n_failures: int = 2,
    seed: int = 42,
    out_path: str = "multi_failure.csv",
) -> Path:
    """Generate a failure.csv with multiple concurrent failures on different boards.

    Ensures failures are on different segments/ROADMs to create
    interesting overlapping alarm propagation scenarios.

    Args:
        topology_yaml: Path to the topology YAML file.
        n_failures: Number of concurrent failures to generate.
        seed: Random seed for reproducibility.
        out_path: Output path for the CSV file.

    Returns:
        Path to the written CSV file.
    """
    topo = load_yaml(topology_yaml)
    board_map = topo.get("board_map") or {}
    node_map = topo.get("node_map") or {}

    # Event pools keyed by board kind category
    _FIBER_EVENTS = ["Fiber_Cut", "Fiber_Crack", "Fiber_Aging"]
    _XCON_EVENTS = ["XCON_Port_Down", "XCON_Buffer_Overflow", "XCON_Fabric_Fault"]
    _LINE_EVENTS = ["Line_Disconnect"]

    def _events_for_kind(kind: str) -> list[str]:
        k = (kind or "").upper()
        if k in {"OA", "OM", "OD", "FIU", "SC2"}:
            return _FIBER_EVENTS
        if k in {"XCON"}:
            return _XCON_EVENTS
        if k in {"LINE"}:
            return _LINE_EVENTS
        if k in {"TRIBUTARY"}:
            return _XCON_EVENTS  # treat tributary like XCON
        return _FIBER_EVENTS

    # Collect all candidate boards grouped by ROADM parent
    candidates_by_roadm: dict[str, list[tuple[str, str]]] = {}  # parent -> [(board, kind), ...]
    for parent, kinds in board_map.items():
        for kind, boards in kinds.items():
            for board in boards:
                candidates_by_roadm.setdefault(parent, []).append((board, kind.upper()))

    if not candidates_by_roadm:
        raise ValueError(f"No boards found in topology {topology_yaml}")

    rng = np.random.default_rng(seed)
    roadm_list = list(candidates_by_roadm.keys())
    rng.shuffle(roadm_list)

    # Select n_failures boards from DIFFERENT ROADMs
    selected: list[tuple[str, str, str]] = []  # (board, kind, parent)
    for parent in roadm_list:
        if len(selected) >= n_failures:
            break
        boards = candidates_by_roadm[parent]
        idx = int(rng.integers(0, len(boards)))
        board, kind = boards[idx]
        selected.append((board, kind, parent))

    if len(selected) < n_failures:
        print(f"[warn] only {len(selected)} ROADMs available, requested {n_failures} failures")

    # Generate base time and stagger starts within 5-60 seconds
    base_time = datetime(2025, 1, 15, 10, 0, 0)
    rows = []
    for i, (board, kind, _parent) in enumerate(selected):
        events_pool = _events_for_kind(kind)
        event = events_pool[int(rng.integers(0, len(events_pool)))]
        stagger_sec = int(rng.integers(5, 61)) if i > 0 else 0
        t = base_time + timedelta(seconds=stagger_sec)
        role = _role_for_kind(kind)
        dur_ms = int(rng.integers(5000, 20001))  # 5-20 seconds
        rows.append({
            "Board": board,
            "Event": event,
            "Severity": "Critical",
            "Time": t.strftime("%Y-%m-%d %H:%M:%S"),
            "DurationMs": dur_ms,
            "BoardRole": role,
        })

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[info] generated multi-failure CSV with {len(rows)} failures: {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="MW simulator: per-node time series with propagation & constrained anomalies")
    ap.add_argument("--topology", type=str, required=True)
    ap.add_argument("--simulator", type=str, required=True)
    ap.add_argument("--roles", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="data_outputs")

    ap.add_argument("--seed", type=int, default=2025, help="RNG for dynamics/propagation")
    ap.add_argument("--anom-seed", type=int, default=None, help="RNG for anomaly injection (default=--seed)")
    ap.add_argument("--topo-tweak-frac", type=float, default=0.0,
                    help="Light route tweak ratio for MW2/MW3 (hop≤2, no new VMFs)")
    ap.add_argument("--alarm-flow", action="store_true",
                    help="Generate alarm_flow.csv based on BBE changes and rule_database.csv")
    ap.add_argument("--alarm-outdir", type=str, default="outputs/alarm_flows",
                    help="Directory to write alarm flow artifacts")
    ap.add_argument("--alarm-rules", type=str, default="outputs/Static files/rule_database.csv",
                    help="Rule database CSV for alarm propagation")
    ap.add_argument("--alarm-all", action="store_true",
                    help="Use local alarms from all VMFs (default: only root VMFs)")
    ap.add_argument("--paths-from", type=str, default=None,
                    help="Optional lightpaths.txt to build mw_lightpaths_roles.txt")
    ap.add_argument("--failures", type=str, default=None,
                    help="Optional failure.csv (Board,Event,Severity,Time,DurationMs)")
    ap.add_argument("--multi-failure", type=int, default=None, metavar="N",
                    help="Auto-generate a multi-failure scenario with N concurrent failures")
    ap.add_argument("--failure-lead-sec", type=int, default=60,
                    help="Seconds of baseline before the first failure time")
    ap.add_argument("--failure-duration-sec", type=int, default=10,
                    help="Default duration when DurationMs is missing")
    ap.add_argument("--failure-bbe-spike", type=float, default=10.0,
                    help="BBE spike floor injected at failure time")
    ap.add_argument("--failure-prop-hops", type=int, default=4,
                    help="Max hops to spread failure impact to affected boards")
    ap.add_argument("--failure-prop-decay", type=float, default=0.7,
                    help="Decay factor per hop for propagated failure impact")
    ap.add_argument("--failure-prop-min-scale", type=float, default=0.2,
                    help="Minimum scale to apply when propagating impact")
    ap.add_argument("--failure-prop-forward-only", action="store_true",
                    help="Only propagate impacts along directed edges")
    ap.add_argument("--failure-prop-delay-min-sec", type=int, default=1,
                    help="Minimum per-hop delay (seconds) for propagated impact")
    ap.add_argument("--failure-prop-delay-max-sec", type=int, default=8,
                    help="Maximum per-hop delay (seconds) for propagated impact")
    ap.add_argument("--alarm-spikes-from-flow", action="store_true",
                    help="Inject additional spikes based on alarm_flow arrival times")
    ap.add_argument("--alarm-spike-bbe", type=float, default=6.0,
                    help="BBE spike for propagated alarms")
    ap.add_argument("--alarm-spike-duration-sec", type=int, default=5,
                    help="Duration for propagated alarm spikes (seconds)")
    ap.add_argument("--alarm-spike-stagger-sec", type=int, default=3,
                    help="Stagger multiple spikes on the same board (seconds)")
    ap.add_argument("--remote-llm-host", default=None,
                    help="If set, rsync alarm output to this host and run LLM analysis there.")
    ap.add_argument("--remote-llm-base", default="~/sim_runs",
                    help="Remote base directory for run folders (default: ~/sim_runs).")
    ap.add_argument("--remote-llm-script", default="~/finetune-sim/log_llm_assistant.py",
                    help="Remote analysis script path (default: ~/finetune-sim/log_llm_assistant.py).")
    ap.add_argument("--remote-llm-cmd", default=None,
                    help="Remote analysis command template. Supports {run_dir}, {report_path}, {report_name}, {ollama_url}, {remote_script}.")
    ap.add_argument("--remote-llm-ollama-url", default="http://localhost:11434",
                    help="Remote Ollama URL to use on the server.")
    ap.add_argument("--remote-llm-report", default="llm_analysis.md",
                    help="Report filename to pull back into the run folder.")
    ap.add_argument("--remote-llm-qwen32", action="store_true",
                    help="Use Qwen2.5-32B inference on remote host (overrides default Ollama path).")
    ap.add_argument("--remote-llm-venv", default="~/venvs/llm310/bin/activate",
                    help="Remote venv activate script for Qwen inference.")
    ap.add_argument("--remote-llm-inference-script", default="~/finetune-sim/inference_qwen.py",
                    help="Remote inference script path.")
    ap.add_argument("--remote-llm-base-model", default="Qwen/Qwen2.5-32B-Instruct",
                    help="Remote base model for Qwen inference.")
    ap.add_argument("--remote-llm-adapter", default="~/finetune-sim/adapters/qwen-failure",
                    help="Remote adapter path for Qwen inference.")
    ap.add_argument("--remote-llm-max-tokens", type=int, default=300,
                    help="Max tokens for Qwen inference.")
    ap.add_argument("--remote-llm-offload-dir", default="~/finetune-sim/offload",
                    help="Offload directory for Qwen inference (device_map=auto).")
    ap.add_argument("--remote-llm-load-4bit", action="store_true",
                    help="Enable 4-bit quantization for Qwen inference.")
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help="Noise scale multiplier for _noisy() calls (default 1.0; 2.0 doubles noise to ±40%%)")
    ap.add_argument("--equalize-bbe", action="store_true",
                    help="Set all root-cause events to the same mult (~3.5), destroying bbe_max as a discriminator")
    ap.add_argument("--cascading", action="store_true",
                    help="Enable cascading failure generation (off by default)")
    ap.add_argument("--cascade-rules", type=str, default=None,
                    help="Path to cascade rules YAML (default: built-in rules from simulator.yaml or defaults)")
    ap.add_argument("--max-cascade-depth", type=int, default=2,
                    help="Maximum cascade depth for correlated failures (default: 2)")

    args = ap.parse_args()

    # Set global noise scale
    global _NOISE_SCALE, _EQUALIZE_BBE
    _NOISE_SCALE = args.noise_scale
    _EQUALIZE_BBE = args.equalize_bbe
    if _NOISE_SCALE != 1.0:
        print(f"[info] noise scale: {_NOISE_SCALE}x (±{_NOISE_SCALE * 20:.0f}% instead of ±20%)")
    if _EQUALIZE_BBE:
        print("[info] equalize-bbe: all root-cause events will use mult≈3.5")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    anom_seed = args.anom_seed if args.anom_seed is not None else args.seed
    anom_rng = np.random.default_rng(int(anom_seed))

    topo = load_yaml(args.topology)
    sim_cfg = load_yaml(args.simulator)

    # --multi-failure N: auto-generate a multi-failure CSV if no explicit
    # --failures file was provided.
    if args.multi_failure is not None and args.multi_failure > 0 and not args.failures:
        safe_mkdir(args.outdir)
        auto_failure_path = str(Path(args.outdir) / "multi_failure.csv")
        generate_multi_failure_csv(
            topology_yaml=args.topology,
            n_failures=args.multi_failure,
            seed=args.seed,
            out_path=auto_failure_path,
        )
        # Optionally resolve cascading secondary failures
        if args.cascading:
            from failure_correlation import (
                load_cascade_rules, resolve_cascades,
                _parse_lightpaths,
            )
            cascade_rules = load_cascade_rules(
                sim_cfg=sim_cfg,
                yaml_path=args.cascade_rules,
            )
            board_map = topo.get("board_map") or {}
            lp_file = args.paths_from or ""
            lightpaths = _parse_lightpaths(lp_file) if lp_file and Path(lp_file).exists() else []
            # Read primary failures from the CSV we just wrote
            df_primary = pd.read_csv(auto_failure_path)
            primary_failures = df_primary.to_dict("records")
            cascade_rng = np.random.default_rng(args.seed + 7)
            cascaded = resolve_cascades(
                primary_failures=primary_failures,
                cascade_rules=cascade_rules,
                board_map=board_map,
                lightpaths=lightpaths,
                rng=cascade_rng,
                max_cascade_depth=args.max_cascade_depth,
            )
            if cascaded:
                df_cascaded = pd.DataFrame(cascaded)
                df_all = pd.concat([df_primary, df_cascaded], ignore_index=True)
                df_all.to_csv(auto_failure_path, index=False)
                print(f"[info] cascading: {len(cascaded)} secondary failures added ({len(df_all)} total)")
            else:
                print("[info] cascading: no secondary failures triggered (probability dice)")
        args.failures = auto_failure_path
        print(f"[info] auto-generated {args.multi_failure}-failure scenario: {auto_failure_path}")

    roles_raw = load_yaml(args.roles)
    roles = normalize_roles_to_list(roles_raw)

    if args.topo_tweak_frac > 0.0:
        roles_vmf_ids_set = set([r["vmf_id"] for r in roles])
        stats = lightly_tweak_services_inplace(
            topo, roles_vmf_ids_set,
            frac=float(args.topo_tweak_frac),
            seed=int(args.seed) + 77
        )
        print(f"[info] topo lightly tweaked: {stats}")
    else:
        print("[info] topology kept as-is (Baseline).")


    print("[info] building VMF graph and service candidates ...")
    graph, service_candidates = build_graph_and_service_candidates(topo, roles, sim_cfg, rng)
    print(f"[info] total VMFs: {len(graph.vmf_list)}, total edges: {len(graph.edges)}")

    print("[info] simulating latent series ...")
    X, meta = simulate_latent_series(graph, sim_cfg, rng)

    if args.failures:
        tau_max = int(max(sim_cfg["propagation"].get("delay_choices", [0])) if sim_cfg.get("propagation") else 0)
        report = {"placed": 0, "target": 0, "shift_used": 0,
                  "tau_max": tau_max, "H_out": int(topo.get("ow_scheme", {}).get("H_out", 30))}
        events = []
        print("[info] failures provided; skipping random root sampling.")
    else:
        print("[info] sampling constrained root events ...")
        events, report = sample_roots_with_constraints(X, graph, topo, sim_cfg, service_candidates, anom_rng)
        print(f"[info] roots: placed={report['placed']}/{report['target']} (shift_used={report['shift_used']}, "
              f"tau_max={report['tau_max']}, H_out={report['H_out']})")

    print("[info] mapping latent to metrics ...")
    metrics = sim_cfg["metrics"]["out_channels"]
    metrics_map = latent_to_metrics(X, metrics, rng)
    alarm_events = events
    alarm_start_time = None
    failure_rows = []
    if args.failures:
        failure_rows = _read_failures_csv(args.failures)
        node_map = topo.get("node_map") or {}
        if not node_map:
            print("[warn] topology missing node_map; failures may not map to nodes.")
        _remap_line_disconnect_to_creation(failure_rows, topo)
        alarm_events, alarm_start_time = inject_failures_to_bbe(
            graph,
            metrics_map,
            failure_rows,
            node_map,
            step_seconds=_parse_step_seconds(sim_cfg["time"].get("step", "1s")),
            lead_sec=args.failure_lead_sec,
            default_duration_sec=args.failure_duration_sec,
            spike_value=args.failure_bbe_spike,
            prop_hops=args.failure_prop_hops,
            prop_decay=args.failure_prop_decay,
            prop_min_scale=args.failure_prop_min_scale,
            prop_bidir=not args.failure_prop_forward_only,
            prop_delay_min_sec=args.failure_prop_delay_min_sec,
            prop_delay_max_sec=args.failure_prop_delay_max_sec,
            seed=args.seed,
        )

    print(f"[info] dumping per-node outputs to {args.outdir} ...")
    safe_mkdir(args.outdir)
    sidecar_extra = {"anomaly_events": events, "vmf_meta": meta,
                     "tau_max": report["tau_max"]}
    dump_per_node(graph, X, metrics_map, args.outdir, sidecar_extra,
                  topo=topo, failures=failure_rows)

    alarm_outdir = None
    if args.alarm_flow:
        try:
            from alarm_generator import load_rule_database, generate_alarm_flow
        except Exception as e:
            print(f"[warn] failed to import alarm_generator: {e}")
        else:
            alarm_outdir = Path(args.alarm_outdir)
            safe_mkdir(alarm_outdir)
            paths_file = alarm_outdir / "mw_lightpaths_roles.txt"
            if args.paths_from:
                n_paths = write_paths_from_lightpaths(args.paths_from, str(paths_file))
            else:
                n_paths = write_paths_from_topology(topo, str(paths_file), failures=failure_rows)
            if n_paths == 0:
                print("[warn] no paths written from topology; skipping alarm flow.")
            else:
                step_seconds = _parse_step_seconds(sim_cfg["time"].get("step", "1s"))
                local_alarm_csv = alarm_outdir / "local_alarms_from_data.csv"
                n_local = write_local_alarms_from_bbe(
                    graph, metrics_map, alarm_events, step_seconds,
                    str(local_alarm_csv),
                    only_roots=(not args.alarm_all),
                    start_time=alarm_start_time,
                    topo=topo,
                    failures=failure_rows,
                )
                df_rules = load_rule_database(args.alarm_rules)
                alarm_flow_csv = alarm_outdir / "alarm_flow.csv"
                failure_csv = None
                if failure_rows and alarm_events:
                    failure_csv = alarm_outdir / "failure_mw.csv"
                    with failure_csv.open("w", encoding="utf-8") as f:
                        f.write("Board,Event,Severity,Time,DurationMs,BoardRole\n")
                        seen = set()
                        rows = []
                        for fr in failure_rows:
                            board = fr.get("Board")
                            if not board:
                                continue
                            event = fr.get("Event") or ""
                            t_str = fr.get("Time") or ""
                            dur = fr.get("DurationMs")
                            if dur is None or dur == "":
                                dur_val = int(args.failure_duration_sec * 1000)
                            else:
                                dur_val = int(float(dur))
                            parent, kind = _parse_board_name(board)
                            role = _forced_role_for_event(event) or _normalize_failure_role(fr.get("BoardRole")) or _role_for_kind(kind)
                            rows.append((board, event, t_str, dur_val, role))
                        if not rows:
                            for ev in alarm_events:
                                board = ev.get("board") or ev.get("vmf_id")
                                if not board:
                                    continue
                                event = ev.get("event", "")
                                t_str = ev.get("time") or ""
                                dur = ev.get("duration_ms")
                                if dur is None or dur == "":
                                    dur_val = int(args.failure_duration_sec * 1000)
                                else:
                                    dur_val = int(float(dur))
                                rows.append((board, event, t_str, dur_val, ""))
                        for board, event, t_str, dur_val, role in rows:
                            key = (board, event, t_str, dur_val, role)
                            if key in seen:
                                continue
                            seen.add(key)
                            f.write(f"{board},{event},Critical,{t_str},{dur_val},{role}\n")
                # Multi-failure: generate_alarm_flow() handles multiple seed
                # failures natively — _read_failures_csv() returns all rows
                # and each creates a seed in the BFS queue, so concurrent
                # failures propagate independently through the alarm graph.
                generate_alarm_flow(
                    graphml_path=str(paths_file),
                    df_rules=df_rules,
                    out_csv=str(alarm_flow_csv),
                    failures_csv=str(failure_csv) if failure_csv else None,
                    paths_file=str(paths_file),
                    bbe_local_csv=str(local_alarm_csv),
                )
                print(f"[info] alarm flow written to {alarm_flow_csv} (paths={n_paths}, local_alarms={n_local})")
                if args.alarm_spikes_from_flow and alarm_start_time:
                    # Extract root boards/events from failure_rows for correct BBE injection.
                    # Multi-failure: provenance columns in alarm_flow.csv now carry per-root
                    # info, so apply_alarm_spikes_from_flow uses them.  We still pass the
                    # first failure's root_board/root_event for backward compatibility.
                    _root_board = None
                    _root_event = None
                    if failure_rows:
                        _root_board = failure_rows[0].get("Board") or failure_rows[0].get("board")
                        _root_event = failure_rows[0].get("Event") or failure_rows[0].get("event")
                    apply_alarm_spikes_from_flow(
                        alarm_flow_csv,
                        topo,
                        args.outdir,
                        alarm_start_time,
                        step_seconds,
                        args.alarm_spike_bbe,
                        args.alarm_spike_duration_sec,
                        args.alarm_spike_stagger_sec,
                        sim_cfg=sim_cfg,
                        root_board=_root_board,
                        root_event=_root_event,
                        seed=args.seed,
                    )
                    n_roots = len(failure_rows) if failure_rows else 1
                    print(f"[info] applied propagated alarm spikes to time series ({n_roots} root cause(s))")

    if args.remote_llm_host:
        analysis_dir = alarm_outdir if alarm_outdir else Path(args.outdir)
        try:
            remote_cmd = args.remote_llm_cmd
            remote_script = args.remote_llm_script
            if args.remote_llm_qwen32 and not remote_cmd:
                if not Path(remote_script).expanduser().exists() and Path("log_llm_assistant.py").exists():
                    remote_script = "log_llm_assistant.py"
                opts = []
                if args.remote_llm_load_4bit or args.remote_llm_qwen32:
                    opts.append("--load-in-4bit")
                if args.remote_llm_offload_dir:
                    opts.append(f"--offload-dir {args.remote_llm_offload_dir}")
                opt_str = " ".join(opts)
                template = (
                    "bash -lc 'source {venv}; "
                    "PROMPT_FILE=$(mktemp); "
                    "python3 {{remote_script}} --run-dir {{run_dir}} --print-prompt > \"$PROMPT_FILE\"; "
                    "PROMPT=$(cat \"$PROMPT_FILE\"); "
                    "python3 {inference_script} --base-model {base_model} --adapter {adapter} "
                    "{opts} --prompt \"$PROMPT\" --max-tokens {max_tokens} > {{report_path}} 2>&1; "
                    "rm -f \"$PROMPT_FILE\"'"
                )
                remote_cmd = template.format(
                    venv=args.remote_llm_venv,
                    inference_script=args.remote_llm_inference_script,
                    base_model=args.remote_llm_base_model,
                    adapter=args.remote_llm_adapter,
                    opts=opt_str,
                    max_tokens=args.remote_llm_max_tokens,
                )
            _run_remote_llm_analysis(
                analysis_dir,
                args.remote_llm_host,
                args.remote_llm_base,
                remote_script,
                args.remote_llm_ollama_url,
                args.remote_llm_report,
                remote_cmd,
            )
            print(f"[info] LLM report synced to {analysis_dir / args.remote_llm_report}")
        except subprocess.CalledProcessError as e:
            print(f"[warn] remote LLM analysis failed: {e}")
        except Exception as e:
            print(f"[warn] remote LLM analysis error: {e}")

    print("[ok] done.")

if __name__ == "__main__":
    main()
