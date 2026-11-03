"""TOPO-002 §2.3 unified planner: one receding-horizon problem, no L2/L3 phases.

max J = sum_i g_i (T - t_i),  g_i = marginal expected gain sum p(c) over UNKNOWN cells first
seen by stop i (no double counting), t_i = tick + travel ticks (BFS / 3), service 0, hard
budget T - tick.  Candidates: (a) skeleton / chain points every 1.5 m, (b) doors, (c) pocket
interior viewpoints, (d) rim viewpoints of large UNKNOWN areas, (e) WFD frontiers; >= 3 m from
the robot unless nothing else.  Solver: lazy-greedy by dJ/dt with best insertion, 2-opt /
Or-opt, exact Held-Karp re-ordering of the first <= 10 stops.

All arrays in the CORE frame.  `p` is the per-cell expected-free probability from
pc_model.p_free(obs, mean, var) (stub: 1[mean < 0.5]).  No ground truth.
"""
import heapq
import time

import numpy as np
from scipy import ndimage

from . import geometry as geo
from . import tour as tr
from .visibility import Visibility


class R2Config(object):
    def __init__(self, **kw):
        self.ppm = 10.0
        self.T = 1000.0
        self.vp_spacing_m = 1.5
        self.ray_range_m = 20.0
        self.num_rays = 250
        self.min_dist_m = 3.0           # candidates >= 3 m unless nothing else
        self.unlock_m = 1.2             # absolute minimum (native unlocks < 1 m)
        self.p_min = 0.05               # cells with p below this carry no gain
        self.p_block = 0.5              # UNKNOWN cell blocks rays iff p < p_block
        self.p_trav = 0.5               # UNKNOWN cell traversable for travel estimates iff p >= p_trav
        self.min_stop_gain = 20.0       # expected cells
        self.max_stops = 20
        self.max_candidates = 400
        self.pool = 60                  # candidates entering the greedy (by g / ticks)
        self.hk_prefix = 10
        self.rim_min_sum = 300.0        # sum p of an UNKNOWN component to be "large"
        self.rim_band = 2               # cells of observed FREE next to the component
        self.rim_max_per_comp = 30
        self.pocket_vp_max = 4
        self.snap_radius = 4
        self.dedupe_m = 1.0             # candidates closer than this are merged (priority kept)
        self.ls_rounds = 2              # 2-opt / Or-opt rounds per decision
        self.frontier_region_radius = 4
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError("unknown option " + k)
            setattr(self, k, v)

    def as_dict(self):
        return dict(self.__dict__)


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _pts(x):
    return np.asarray(x, dtype=np.float64).reshape(-1, 2)


def parse_chains(topo):
    """L1 v2 `chains` -> [(points N x 2, cls)].  Accepts dicts with rc/points/polyline/
    skeleton_rc and cls/class/kind, or (points, cls) tuples."""
    out = []
    for ch in _get(topo, "chains", None) or []:
        if isinstance(ch, dict):
            pts = next((ch[k] for k in ("polyline_rc", "rc", "points", "polyline", "skeleton_rc", "pts") if k in ch), None)
            cls = next((ch[k] for k in ("cls", "class", "kind", "type") if k in ch), None)
        else:
            pts, cls = ch[0], ch[1] if len(ch) > 1 else None
        if pts is not None and len(pts):
            out.append((_pts(pts), str(cls) if cls is not None else "chain"))
    return out


def parse_viewpoints(topo):
    out = []
    for v in _get(topo, "viewpoints", None) or []:
        if isinstance(v, dict):
            rc = v.get("rc", v.get("pos"))
            kind = v.get("kind", v.get("cls", "viewpoint"))
            region = v.get("region", v.get("region_id"))
        else:
            rc, kind, region = v[0], (v[1] if len(v) > 1 else "viewpoint"), None
        if rc is not None:
            out.append((np.asarray(rc, dtype=np.float64), str(kind), region))
    return out


class _Timer(object):
    def __init__(self):
        self.t = {}
        self._k = None
        self._s = 0.0

    def start(self, k):
        self.stop()
        self._k, self._s = k, time.perf_counter()

    def stop(self):
        if self._k is not None:
            self.t[self._k] = self.t.get(self._k, 0.0) + time.perf_counter() - self._s
            self._k = None


class UnifiedPlanner(object):
    def __init__(self, config=None):
        self.cfg = config or R2Config()

    # ------------------------------------------------------------------ candidates
    def _corridor_regions(self, topo):
        ids = set()
        for r in _get(topo, "regions", []) or []:
            p = r.get("p") or {}
            if r.get("cls") == "corridor" or p.get("corridor", 0) >= 0.5:
                ids.add(int(r["id"]))
        return ids

    def _raw_candidates(self, topo, frontiers, obs, p, region_id, trav, robot):
        cfg = self.cfg
        sp = cfg.vp_spacing_m * cfg.ppm
        pts, kinds = [], []

        def add(points, kind, corridor):
            for q in geo.subsample_points(np.round(_pts(points)).astype(np.int64), sp):
                pts.append(q)
                kinds.append((kind, corridor))

        chains = parse_chains(topo)
        if chains:                                   # (a) L1 v2 chains
            for c, cls in chains:
                add(c, "chain_" + cls, cls == "corridor")
        else:                                        # (a) v1: region skeletons
            cor = self._corridor_regions(topo)
            for r in _get(topo, "regions", []) or []:
                sk = r.get("skeleton_rc")
                if sk is not None and len(sk):
                    add(sk, "skeleton", int(r["id"]) in cor)
        for d in _get(topo, "doors", []) or []:      # (b) doors
            pts.append(np.round(np.asarray(d["rc"], dtype=np.float64)).astype(np.int64))
            kinds.append(("door", False))
        vps = parse_viewpoints(topo)                 # (c) pocket interior viewpoints
        if vps:
            for rc, kind, _ in vps:
                pts.append(np.round(rc).astype(np.int64))
                kinds.append(("vp_" + kind, False))
        else:
            for r in _get(topo, "regions", []) or []:
                c = r.get("centroid_rc")
                if c is not None:
                    pts.append(np.round(np.asarray(c)).astype(np.int64))
                    kinds.append(("pocket_centroid", False))
        rim, rim_info = self._rim_viewpoints(obs, p, trav)   # (d) rim viewpoints
        for q in rim:
            pts.append(q)
            kinds.append(("rim", False))
        for f in frontiers:                          # (e) WFD frontiers
            pts.append(np.asarray(f, dtype=np.int64))
            kinds.append(("frontier", False))
        return pts, kinds, rim_info

    def _rim_viewpoints(self, obs, p, trav):
        """Observed-FREE cells bordering UNKNOWN components with sum p >= rim_min_sum."""
        cfg = self.cfg
        unk = obs == 0.5
        lab, n = ndimage.label(unk, structure=np.ones((3, 3), bool))
        if n == 0:
            return [], {"components": 0, "large": 0}
        sums = ndimage.sum(np.where(unk, p, 0.0), lab, index=np.arange(1, n + 1))
        large = np.flatnonzero(sums >= cfg.rim_min_sum) + 1
        out = []
        free = (obs == 0) & trav
        sp = cfg.vp_spacing_m * cfg.ppm
        for comp in large:
            m = lab == comp
            band = ndimage.binary_dilation(m, iterations=cfg.rim_band) & free
            rr, cc = np.nonzero(band)
            if not len(rr):
                continue
            q = geo.subsample_points(np.stack([rr, cc], axis=1), sp)
            if len(q) > cfg.rim_max_per_comp:
                q = q[np.linspace(0, len(q) - 1, cfg.rim_max_per_comp).astype(int)]
            out.extend([tuple(x) for x in q])
        return out, {"components": int(n), "large": int(len(large)),
                     "large_sum_p": [float(sums[c - 1]) for c in large[:10]]}

    PRIORITY = ("rim", "frontier", "door", "vp_pocket", "pocket_centroid")

    def _filter(self, pts, kinds, ctx, min_dist):
        """Snap, reachability, distance, exclusion; then thin to `dedupe_m` keeping higher-priority
        kinds (rim, frontier, door, pocket viewpoints, then chain / skeleton points)."""
        def prio(k):
            name = k[0]
            for i, p in enumerate(self.PRIORITY):
                if name.startswith(p):
                    return i
            return len(self.PRIORITY) + (0 if name.startswith("chain") or name == "skeleton" else 1)
        order = sorted(range(len(pts)), key=lambda i: prio(kinds[i]))
        pts = [pts[i] for i in order]
        kinds = [kinds[i] for i in order]
        out, seen = [], set()
        cell = max(1, int(self.cfg.dedupe_m * self.cfg.ppm))
        grid = {}
        rr, rc = ctx["robot"]
        for p, kind in zip(pts, kinds):
            q = geo.snap_to_mask(p, ctx["trav"], self.cfg.snap_radius)
            if q is None or q in seen or ctx["dist_r"][q] < 0:
                continue
            if (q[0] - rr) ** 2 + (q[1] - rc) ** 2 < min_dist * min_dist:
                continue
            if any((q[0] - e[0]) ** 2 + (q[1] - e[1]) ** 2 <= 9 for e in ctx["exclude"]):
                continue
            key = (q[0] // cell, q[1] // cell)
            near = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    for o in grid.get((key[0] + dr, key[1] + dc), ()):
                        if (o[0] - q[0]) ** 2 + (o[1] - q[1]) ** 2 < cell * cell:
                            near = True
            if near:
                continue
            grid.setdefault(key, []).append(q)
            seen.add(q)
            out.append((q, kind))
        return out

    # ------------------------------------------------------------------ main
    def plan(self, obs, mean, var, p, robot_rc, topo, plan_finite, frontiers_rc, tick,
             exclude_goals=()):
        cfg = self.cfg
        tm = _Timer()
        tm.start("prep")
        obs = np.asarray(obs)
        p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
        robot = (int(robot_rc[0]), int(robot_rc[1]))
        region_id = _get(topo, "region_id")
        region_id = np.full(obs.shape, -1, np.int32) if region_id is None else np.asarray(region_id)
        unk = obs == 0.5
        gmask = unk & (p >= cfg.p_min)
        gid = np.full(obs.shape, -1, dtype=np.int32)
        gr, gc = np.nonzero(gmask)
        gid[gr, gc] = np.arange(len(gr), dtype=np.int32)
        gw = p[gr, gc]
        occ = ((obs == 1.0) | (unk & (p < cfg.p_block))).astype(np.float64)
        trav = np.asarray(plan_finite, bool) & ((obs == 0) | (unk & (p >= cfg.p_trav)))
        trav[robot] = True
        dist_r = geo.bfs_distance(trav, robot)
        vis = Visibility(occ, gid, int(round(cfg.ray_range_m * cfg.ppm)), cfg.num_rays)
        ctx = {"robot": robot, "trav": trav, "dist_r": dist_r,
               "exclude": [tuple(int(v) for v in e) for e in exclude_goals]}
        budget = max(0.0, cfg.T - float(tick))
        log = {"tick": int(tick), "robot_rc": list(robot), "n_gain_cells": int(len(gr)),
               "expected_unknown_free": float(gw.sum()), "budget": budget,
               "excluded_goals": [list(e) for e in ctx["exclude"]]}

        tm.start("candidates")
        pts, kinds, rim_info = self._raw_candidates(topo, frontiers_rc, obs, p, region_id, trav, robot)
        relaxed = False
        for min_d in (cfg.min_dist_m, cfg.unlock_m):
            tm.start("candidates")
            cands = self._filter(pts, kinds, ctx, min_d * cfg.ppm)
            if len(cands) > cfg.max_candidates:
                cands.sort(key=lambda x: dist_r[x[0]])
                cands = cands[:cfg.max_candidates]
            tm.start("visibility")
            ids = [vis.gain_ids(q) for q, _ in cands]
            g0 = np.array([float(gw[x].sum()) if len(x) else 0.0 for x in ids])
            t0 = np.array([geo.ticks(dist_r[q]) for q, _ in cands]) if cands else np.zeros(0)
            ok = (g0 >= cfg.min_stop_gain) & (t0 < budget)
            order_pool = np.flatnonzero(ok)
            order_pool = order_pool[np.argsort(-(g0[order_pool] / (t0[order_pool] + 1.0)), kind="stable")]
            pool = order_pool[:cfg.pool].tolist()
            if pool:
                break
            relaxed = True      # nothing useful >= 3 m: allow nearer candidates (>= 1.2 m)
        kc = {}
        for _, (k, _) in cands:
            kc[k] = kc.get(k, 0) + 1
        log.update({"raw_candidates": len(pts), "candidates": len(cands), "candidate_kinds": kc,
                    "min_dist_relaxed": relaxed and bool(pool), "rim": rim_info})
        log["pool"] = len(pool)
        if not pool:
            tm.stop()
            log.update({"timing_s": {k: round(v, 4) for k, v in tm.t.items()},
                        "planner_s": round(sum(tm.t.values()), 4), "stops": [], "J": 0.0})
            return {"goals": [], "targets": [], "log": log,
                    "viz": {"tour": [], "cands": [c[0] for c in cands]}}

        # Lazy distances: the grid graph is symmetric, so a candidate's insertion only needs
        # the fields of the robot and of already-selected stops (one early-stopping BFS each).
        tm.start("distances")
        n = len(pool)
        D = np.full((n + 1, n + 1), np.nan)
        D[0, 0] = 0.0
        pts_pool = [cands[i][0] for i in pool]
        for a in range(n):
            D[0, a + 1] = D[a + 1, 0] = t0[pool[a]]
            D[a + 1, a + 1] = 0.0
        n_bfs = [0]

        def fill(k):
            if not np.isnan(D[k, 1:]).any():
                return
            dd = geo.bfs_to_targets(trav, pts_pool[k - 1], pts_pool)
            n_bfs[0] += 1
            row = np.where(dd >= 0, dd / geo.CELLS_PER_TICK, np.inf)
            D[k, 1:] = row
            D[1:, k] = row

        tm.start("select")
        covered = np.zeros(len(gr), dtype=bool)
        g = np.zeros(n + 1)
        model = tr.TourModel(D, g, t0=float(tick), T=cfg.T, budget=budget)
        order = []
        assigned = {}

        def marginal(k):
            x = ids[pool[k - 1]]
            return float(gw[x[~covered[x]]].sum()) if len(x) else 0.0

        def ratio(k):
            gk = marginal(k)
            if gk < cfg.min_stop_gain:
                return None
            model.g[k] = gk
            dJ, dL, pos = tr.best_insertion(model, order, k)
            model.g[k] = 0.0
            if pos is None or dJ <= 0:
                return None
            return dJ / max(dL, 1e-6), gk, pos, dJ, dL

        heap = []
        for k in range(1, n + 1):
            r = ratio(k)
            if r is not None:
                heapq.heappush(heap, (-r[0], k))
        evals = n
        while heap and len(order) < cfg.max_stops:
            _, k = heapq.heappop(heap)
            r = ratio(k)
            evals += 1
            if r is None:
                continue
            if heap and r[0] < -heap[0][0] - 1e-12:
                heapq.heappush(heap, (-r[0], k))
                continue
            _, gk, pos, dJ, dL = r
            model.g[k] = gk
            order.insert(pos, k)
            tm.start("distances")
            fill(k)
            tm.start("select")
            x = ids[pool[k - 1]]
            new = x[~covered[x]]
            covered[new] = True
            assigned[k] = new
        log["select"] = {"stops": len(order), "evaluations": evals, "bfs": n_bfs[0]}

        tm.start("improve")
        J_greedy = model.value(order)
        for _ in range(cfg.ls_rounds):
            prev = list(order)
            order = tr.two_opt(model, order, max_rounds=1)
            order = tr.or_opt(model, order, max_rounds=cfg.ls_rounds)
            if order == prev:
                break
        J_ls = model.value(order)
        order, hk = tr.reorder_prefix(model, order, cfg.hk_prefix)
        J, L, times = model.evaluate(order)
        tm.start("out")
        stops, goals, targets = [], [], []
        for k, tk in zip(order, times):
            q, (kind, corr) = cands[pool[k - 1]]
            regs = region_id[gr[assigned[k]], gc[assigned[k]]] if len(assigned[k]) else np.zeros(0, int)
            stops.append({"rc": list(q), "kind": kind, "on_corridor_chain": bool(corr),
                          "g": float(model.g[k]), "t_pred": tk,
                          "cells": int(len(assigned[k])),
                          "region_at": int(region_id[q]) if region_id[q] >= 0 else None,
                          "gain_outside_regions": float(gw[assigned[k]][regs < 0].sum()) if len(regs) else 0.0})
            goals.append((q, kind))
            x = ids[pool[k - 1]]
            targets.append(np.stack([gr[x], gc[x]], axis=1) if len(x) else np.zeros((0, 2), int))
        tm.stop()
        log.update({"J": J, "ticks": L, "lambda": (J / L) if L > 0 else None,
                    "J_greedy": J_greedy, "J_local_search": J_ls, "hk": hk, "stops": stops,
                    "first_stop_on_corridor_chain": bool(stops[0]["on_corridor_chain"]) if stops else None,
                    "share_stops_on_corridor_chain": (sum(s["on_corridor_chain"] for s in stops) / float(len(stops)))
                    if stops else None,
                    "raycasts": vis.calls,
                    "timing_s": {k: round(v, 4) for k, v in tm.t.items()},
                    "planner_s": round(sum(tm.t.values()), 4)})
        return {"goals": goals, "targets": targets, "log": log,
                "viz": {"tour": [list(q) for q, _ in goals], "cands": [c[0] for c in cands]}}
