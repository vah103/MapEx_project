"""TOPO-001 L2 (backbone tour) + L3 (pocket orienteering) planner.

Input per decision (all CORE frame): obs {0,.5,1}, LaMa mean/var, robot rc,
`plan_finite` (finite cells of the native inflated A* grid), WFD frontier rc,
the L1 TopoMap and the tick.  Output: ordered goal candidates (first = goal)
plus a JSON-able log record.  No ground truth is read anywhere.
"""
import time

import numpy as np

from . import geometry as geo
from . import tour as tr
from .visibility import Visibility, predicted_occupancy


class PlannerConfig(object):
    def __init__(self, **kw):
        self.ppm = 10.0
        self.T = 1000.0
        self.vp_spacing_m = 1.5
        self.ray_range_m = 20.0
        self.num_rays = 250
        self.min_goal_m = 1.2            # native unlocks goals < 1 m; keep a margin
        self.phase1_end_frac = 0.05
        self.phase1_min_gain = 20
        self.cover_stop_frac = 0.05      # L2 set cover stops when < 5 % of U_bb uncovered
        self.min_stop_gain = 20
        self.max_l2_stops = 20
        self.max_vp_candidates = 300
        self.max_pocket_insert = 6
        self.max_pockets_l2 = 8          # pockets evaluated for insertion
        self.max_pockets_l3 = 20
        self.dp_max = 12
        self.hk_K = 32                # Pareto front size; 0 overflow at n=12 in tests
        self.pocket_vp_max = 4
        self.max_pocket_candidates = 60
        self.pocket_cover_frac = 0.9
        self.min_pocket_gain_cells = 20
        self.intra_pocket_factor = 1.2   # Euclid x factor for service inside a pocket
        self.snap_radius = 4
        self.frontier_region_radius = 4
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError("unknown planner option " + k)
            setattr(self, k, v)

    def as_dict(self):
        return dict(self.__dict__)


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class _Timer(object):
    def __init__(self):
        self.t = {}
        self._s = None
        self._k = None

    def start(self, k):
        self.stop()
        self._k, self._s = k, time.perf_counter()

    def stop(self):
        if self._k is not None:
            self.t[self._k] = self.t.get(self._k, 0.0) + time.perf_counter() - self._s
            self._k = None


class TopoPlanner(object):
    def __init__(self, config=None):
        self.cfg = config or PlannerConfig()

    # ------------------------------------------------------------ helpers
    def _region_at(self, region_id, rc):
        r, c = int(rc[0]), int(rc[1])
        h, w = region_id.shape
        if 0 <= r < h and 0 <= c < w and region_id[r, c] >= 0:
            return int(region_id[r, c])
        R = self.cfg.frontier_region_radius
        sub = region_id[max(0, r - R):r + R + 1, max(0, c - R):c + R + 1]
        vals = sub[sub >= 0]
        if not len(vals):
            return -1
        u, n = np.unique(vals, return_counts=True)
        return int(u[np.argmax(n)])

    def _candidates(self, pts, ctx, kinds):
        """Snap to traversable, keep reachable and >= min_goal from robot; dedupe."""
        out = []
        seen = set()
        rr, rc = ctx["robot"]
        ming = self.cfg.min_goal_m * self.cfg.ppm
        for p, kind in zip(pts, kinds):
            q = geo.snap_to_mask(p, ctx["trav"], self.cfg.snap_radius)
            if q is None or q in seen:
                continue
            if ctx["dist_r"][q] < 0:
                continue
            if (q[0] - rr) ** 2 + (q[1] - rc) ** 2 < ming * ming:
                continue
            if any((q[0] - e[0]) ** 2 + (q[1] - e[1]) ** 2 <= 9 for e in ctx["exclude"]):
                continue
            seen.add(q)
            out.append((q, kind))
        return out

    def _region_points(self, topo, rid_set, frontiers, spacing, region_id):
        pts, kinds = [], []
        for reg in _get(topo, "regions", []):
            if int(reg["id"]) not in rid_set:
                continue
            sk = np.asarray(reg.get("skeleton_rc", np.zeros((0, 2))), dtype=np.int64).reshape(-1, 2)
            for p in geo.subsample_points(sk, spacing):
                pts.append(p)
                kinds.append(("skeleton", int(reg["id"])))
            cen = reg.get("centroid_rc")
            if cen is not None:
                pts.append(np.round(np.asarray(cen)).astype(np.int64))
                kinds.append(("centroid", int(reg["id"])))
        for f in frontiers:
            rid = self._region_at(region_id, f)
            if rid in rid_set:
                pts.append(np.asarray(f, dtype=np.int64))
                kinds.append(("frontier", rid))
        return pts, kinds

    # ------------------------------------------------------------ main
    def plan(self, obs, mean, var, robot_rc, topo, plan_finite, frontiers_rc, tick,
             exclude_goals=()):
        cfg = self.cfg
        tm = _Timer()
        tm.start("prep")
        obs = np.asarray(obs)
        mean = np.asarray(mean)
        var = np.asarray(var)
        robot = (int(robot_rc[0]), int(robot_rc[1]))
        region_id = np.asarray(_get(topo, "region_id"))
        backbone_mask = np.asarray(_get(topo, "backbone_mask"), dtype=bool)
        pocket_ids = set(int(p) for p in (_get(topo, "pocket_ids", []) or []))
        regions = _get(topo, "regions", []) or []
        all_ids = set(int(r["id"]) for r in regions)
        backbone_ids = all_ids - pocket_ids

        unk_free = (obs == 0.5) & (mean < 0.5)
        gain_id = np.full(obs.shape, -1, dtype=np.int32)
        gr, gc = np.nonzero(unk_free)
        gain_id[gr, gc] = np.arange(len(gr), dtype=np.int32)
        g_region = region_id[gr, gc]
        g_bb = backbone_mask[gr, gc]
        # var_norm on a fixed scale: unbiased variance of 3 values in [0, 1] is <= 1/3
        g_w = 1.0 - np.clip(var[gr, gc] * 3.0, 0.0, 1.0)

        trav = np.asarray(plan_finite, dtype=bool) & ((obs == 0) | unk_free)
        trav[robot] = True
        dist_r = geo.bfs_distance(trav, robot)
        vis = Visibility(predicted_occupancy(obs, mean), gain_id,
                         range_cells=int(round(cfg.ray_range_m * cfg.ppm)), num_rays=cfg.num_rays)
        ctx = {"robot": robot, "trav": trav, "dist_r": dist_r,
               "exclude": [tuple(int(v) for v in e) for e in exclude_goals]}
        spacing = cfg.vp_spacing_m * cfg.ppm
        bb_free_area = int(np.count_nonzero(backbone_mask & ((obs == 0) | unk_free)))
        U_bb = int(np.count_nonzero(g_bb))
        frac = U_bb / float(bb_free_area) if bb_free_area else 0.0

        log = {"tick": int(tick), "robot_rc": list(robot), "n_gain_cells": int(len(gr)),
               "backbone_free_cells": bb_free_area, "backbone_unknown_free_cells": U_bb,
               "backbone_unknown_frac": frac, "n_regions": len(regions),
               "n_backbone_regions": len(backbone_ids), "n_pockets": len(pocket_ids),
               "excluded_goals": [list(e) for e in ctx["exclude"]]}

        tm.start("l2_candidates")
        pts, kinds = self._region_points(topo, backbone_ids, frontiers_rc, spacing, region_id)
        cands = self._candidates(pts, ctx, kinds)
        if len(cands) > cfg.max_vp_candidates:
            cands.sort(key=lambda x: dist_r[x[0]])
            cands = cands[:cfg.max_vp_candidates]
        tm.start("l2_visibility")
        bb_gain = []
        for q, _ in cands:
            ids = vis.gain_ids(q)
            bb_gain.append(ids[g_bb[ids]] if len(ids) else ids)
        best_bb = max([len(x) for x in bb_gain] or [0])
        log.update({"l2_candidates": len(cands), "l2_best_viewpoint_gain": int(best_bb)})

        phase = "L2" if (frac >= cfg.phase1_end_frac and best_bb >= cfg.phase1_min_gain) else "L3"
        log["phase_rule"] = phase
        result = None
        if phase == "L2":
            result = self._plan_l2(ctx, cands, bb_gain, vis, topo, pocket_ids, frontiers_rc,
                                   region_id, g_region, g_w, tick, tm, log)
            if result is None:
                log["l2_empty_try_l3"] = True
        if result is None:
            result = self._plan_l3(ctx, vis, topo, pocket_ids, frontiers_rc, region_id,
                                   g_region, g_w, tick, tm, log)
            if result is not None:
                phase = "L3"
        tm.stop()
        log["phase"] = phase if result is not None else "NONE"
        log["raycasts"] = vis.calls
        log["timing_s"] = {k: round(v, 4) for k, v in tm.t.items()}
        log["planner_s"] = round(sum(tm.t.values()), 4)
        if result is None:
            log["goal_candidates"] = []
            return {"goals": [], "log": log, "viz": {"tour": [], "cands": [c[0] for c in cands]}}
        result["log"] = log
        result["viz"]["cands"] = [c[0] for c in cands]
        # cells (CORE rc) each goal is expected to observe: used by the hook's replan trigger
        targets = []
        for q, _ in result["goals"]:
            ids = vis.gain_ids(q)
            targets.append(np.stack([gr[ids], gc[ids]], axis=1) if len(ids) else np.zeros((0, 2), int))
        result["targets"] = targets
        return result

    # ------------------------------------------------------------ L2
    def _plan_l2(self, ctx, cands, bb_gain, vis, topo, pocket_ids, frontiers_rc, region_id,
                 g_region, g_w, tick, tm, log):
        cfg = self.cfg
        tm.start("l2_setcover")
        total = int(log["backbone_unknown_free_cells"])
        covered = np.zeros(len(g_region), dtype=bool)
        chosen, gains, gain_ids = [], [], []
        remaining = set(range(len(cands)))
        uncovered = total
        # lazy greedy (CELF): marginal gains only shrink, cost (ticks from robot) is fixed
        import heapq
        cost = {}
        heap = []
        for i in remaining:
            if len(bb_gain[i]) >= cfg.min_stop_gain:
                cost[i] = geo.ticks(ctx["dist_r"][cands[i][0]]) + 1.0
                heapq.heappush(heap, (-len(bb_gain[i]) / cost[i], i))
        while heap and len(chosen) < cfg.max_l2_stops:
            _, i = heapq.heappop(heap)
            new = bb_gain[i][~covered[bb_gain[i]]]
            if len(new) < cfg.min_stop_gain:
                continue
            score = len(new) / cost[i]
            if heap and score < -heap[0][0] - 1e-12:
                heapq.heappush(heap, (-score, i))
                continue
            covered[new] = True
            uncovered -= len(new)
            chosen.append(i)
            gains.append(float(len(new)))
            gain_ids.append(new)
            if total and uncovered < cfg.cover_stop_frac * total:
                break
        log["l2_setcover"] = {"selected": len(chosen), "uncovered_frac":
                              (uncovered / float(total)) if total else 0.0}
        if not chosen:
            return None
        tm.start("l2_distances")
        stops = [cands[i][0] for i in chosen]
        kinds = [cands[i][1] for i in chosen]
        fields = [geo.bfs_distance(ctx["trav"], p) for p in stops]
        n = len(stops)
        D = np.full((n + 1, n + 1), np.inf)
        D[0, 0] = 0.0
        for a in range(n):
            d = ctx["dist_r"][stops[a]]
            D[0, a + 1] = D[a + 1, 0] = geo.ticks(d) if d >= 0 else np.inf
            for b in range(n):
                db = fields[a][stops[b]]
                D[a + 1, b + 1] = geo.ticks(db) if db >= 0 else np.inf
        g = np.concatenate([[0.0], gains])
        tm.start("l2_order")
        model = tr.TourModel(D, g, t0=float(tick), T=cfg.T)
        nodes = list(range(1, n + 1))
        order = tr.greedy_insertion(model, nodes)
        order = tr.improve(model, order)
        J0, L0, _ = model.evaluate(order)
        # ---- pocket insertion (lambda = current tour rate J/L)
        tm.start("l2_pockets")
        pockets = self._pocket_options(ctx, vis, topo, pocket_ids, frontiers_rc, region_id,
                                       g_region, g_w, single_viewpoint=True)
        pockets.sort(key=lambda p: -p["gain"] / (geo.ticks(p["dist_r"]) + 1.0))
        pockets = pockets[:cfg.max_pockets_l2]
        inserted = []
        lam_hist = []
        if pockets and order:
            m = n + len(pockets)
            D2 = np.full((m + 1, m + 1), np.inf)
            D2[:n + 1, :n + 1] = D
            pfields = [geo.bfs_distance(ctx["trav"], p["vp"]) for p in pockets]
            all_pts = stops + [p["vp"] for p in pockets]
            for a, p in enumerate(pockets):
                ia = n + 1 + a
                D2[0, ia] = D2[ia, 0] = geo.ticks(p["dist_r"])
                for b, q in enumerate(all_pts):
                    d = pfields[a][q]
                    D2[ia, b + 1] = D2[b + 1, ia] = geo.ticks(d) if d >= 0 else np.inf
            g2 = np.concatenate([g, [p["gain"] for p in pockets]])
            model = tr.TourModel(D2, g2, t0=float(tick), T=cfg.T)
            rest = list(range(n + 1, m + 1))
            while rest and len(inserted) < cfg.max_pocket_insert:
                J, L, _ = model.evaluate(order)
                lam = J / L if L > 0 else np.inf
                best = None
                for k in rest:
                    dJ, dL, pos = tr.best_insertion(model, order, k)
                    if pos is None or dJ <= 0:
                        continue
                    ratio = dJ / max(dL, 1e-6)
                    if best is None or ratio > best[0]:
                        best = (ratio, k, pos, dJ, dL)
                lam_hist.append(lam)
                if best is None or best[0] <= lam:
                    break
                order.insert(best[2], best[1])
                rest.remove(best[1])
                p = pockets[best[1] - n - 1]
                inserted.append({"pocket": p["id"], "vp": list(p["vp"]), "gain": p["gain"],
                                 "ratio": best[0], "lambda": lam, "dJ": best[3], "dticks": best[4]})
            if inserted:
                order = tr.improve(model, order)
        J, L, times = model.evaluate(order)
        tm.start("l2_out")
        stops_out = []
        goals = []
        for k, tk in zip(order, times):
            if k <= n:
                q, kind = stops[k - 1], kinds[k - 1]
                regs = g_region[gain_ids[k - 1]]
                u, c = np.unique(regs, return_counts=True)
                tag = {str(int(a)): int(b) for a, b in sorted(zip(u, c), key=lambda x: -x[1])[:3]}
                stops_out.append({"rc": list(q), "kind": "backbone_" + kind[0], "region": kind[1],
                                  "gain": float(model.g[k]), "t_pred": tk, "region_gain": tag})
            else:
                p = pockets[k - n - 1]
                q = p["vp"]
                stops_out.append({"rc": list(q), "kind": "pocket_insert", "region": p["id"],
                                  "gain": float(model.g[k]), "t_pred": tk})
            goals.append((q, stops_out[-1]["kind"]))
        log["l2"] = {"J_before_pockets": J0, "ticks_before_pockets": L0, "J": J, "ticks": L,
                     "lambda": (J0 / L0) if L0 > 0 else None, "lambda_hist": lam_hist,
                     "pocket_candidates": len(pockets), "pocket_insertions": inserted,
                     "stops": stops_out}
        return {"goals": goals, "viz": {"tour": [list(g[0]) for g in goals]}}

    # ------------------------------------------------------------ pockets
    def _pocket_options(self, ctx, vis, topo, pocket_ids, frontiers_rc, region_id, g_region,
                        g_w, single_viewpoint):
        """Per pocket: viewpoint set cover of its unknown predicted-free cells.

        single_viewpoint=True (L2 insertion): best single viewpoint, gain = cells seen.
        Otherwise (L3): greedy cover, g_p = sum (1 - var_norm) over covered cells,
        service s_p = round trip from the entry viewpoint (Euclid x factor)."""
        cfg = self.cfg
        if not pocket_ids:
            return []
        spacing = cfg.vp_spacing_m * cfg.ppm
        cells_by_region = {}
        for pid in pocket_ids:
            cells_by_region[pid] = int(np.count_nonzero(g_region == pid))
        doors = {int(d["id"]): d for d in (_get(topo, "doors", []) or [])}
        out = []
        for reg in _get(topo, "regions", []) or []:
            pid = int(reg["id"])
            if pid not in pocket_ids or cells_by_region.get(pid, 0) < cfg.min_pocket_gain_cells:
                continue
            pts, kinds = self._region_points(topo, {pid}, frontiers_rc, spacing, region_id)
            for did in reg.get("door_ids", []) or []:
                d = doors.get(int(did))
                if d is not None:
                    pts.append(np.round(np.asarray(d["rc"])).astype(np.int64))
                    kinds.append(("door", pid))
            cands = self._candidates(pts, ctx, kinds)
            if not cands:
                continue
            if len(cands) > cfg.max_pocket_candidates:
                # keep doors, then the geodesically nearest viewpoints
                cands.sort(key=lambda x: (x[1][0] != "door", ctx["dist_r"][x[0]]))
                cands = cands[:cfg.max_pocket_candidates]
            gsets = []
            for q, _ in cands:
                ids = vis.gain_ids(q)
                gsets.append(ids[g_region[ids] == pid] if len(ids) else ids)
            if single_viewpoint:
                i = int(np.argmax([len(x) for x in gsets]))
                if len(gsets[i]) < cfg.min_pocket_gain_cells:
                    continue
                out.append({"id": pid, "vp": cands[i][0], "gain": float(len(gsets[i])),
                            "dist_r": int(ctx["dist_r"][cands[i][0]])})
                continue
            total = cells_by_region[pid]
            covered = np.zeros(len(g_region), dtype=bool)
            ncov = 0
            sel = []
            while len(sel) < cfg.pocket_vp_max:
                best, bi = 0, None
                for i, ids in enumerate(gsets):
                    if i in sel or len(ids) <= best:
                        continue
                    new = int(np.count_nonzero(~covered[ids]))
                    if new > best:
                        best, bi = new, i
                if bi is None or best < cfg.min_stop_gain:
                    break
                sel.append(bi)
                covered[gsets[bi]] = True
                ncov += best
                if ncov >= cfg.pocket_cover_frac * total:
                    break
            if not sel:
                continue
            cov = np.nonzero(covered)[0]
            gain = float(np.sum(g_w[cov]))
            entry = min(sel, key=lambda i: ctx["dist_r"][cands[i][0]])
            vps = [cands[i][0] for i in sel]
            e = cands[entry][0]
            # nearest-neighbour round trip from entry through the pocket viewpoints
            cur, todo, length = e, [v for v in vps if v != e], 0.0
            while todo:
                nxt = min(todo, key=lambda v: (v[0] - cur[0]) ** 2 + (v[1] - cur[1]) ** 2)
                length += np.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
                cur = nxt
                todo.remove(nxt)
            length += np.hypot(e[0] - cur[0], e[1] - cur[1])
            service = geo.ticks(length * cfg.intra_pocket_factor)
            out.append({"id": pid, "vp": e, "vps": vps, "gain": gain, "cells": int(len(cov)),
                        "service": service, "dist_r": int(ctx["dist_r"][e]),
                        "cls": reg.get("cls")})
        return out

    # ------------------------------------------------------------ L3
    def _plan_l3(self, ctx, vis, topo, pocket_ids, frontiers_rc, region_id, g_region, g_w,
                 tick, tm, log):
        cfg = self.cfg
        tm.start("l3_pockets")
        pockets = self._pocket_options(ctx, vis, topo, pocket_ids, frontiers_rc, region_id,
                                       g_region, g_w, single_viewpoint=False)
        pockets.sort(key=lambda p: -p["gain"] / (geo.ticks(p["dist_r"]) + p["service"] + 1.0))
        pockets = pockets[:cfg.max_pockets_l3]
        log["l3_pocket_options"] = len(pockets)
        if not pockets:
            return None
        tm.start("l3_distances")
        n = len(pockets)
        fields = [geo.bfs_distance(ctx["trav"], p["vp"]) for p in pockets]
        D = np.full((n + 1, n + 1), np.inf)
        D[0, 0] = 0.0
        for a, p in enumerate(pockets):
            D[0, a + 1] = D[a + 1, 0] = geo.ticks(p["dist_r"])
            for b, q in enumerate(pockets):
                d = fields[a][q["vp"]]
                D[a + 1, b + 1] = geo.ticks(d) if d >= 0 else np.inf
        g = np.concatenate([[0.0], [p["gain"] for p in pockets]])
        s = np.concatenate([[0.0], [p["service"] for p in pockets]])
        tm.start("l3_orienteering")
        budget = max(0.0, cfg.T - float(tick))
        model = tr.TourModel(D, g, s, s / 2.0, t0=float(tick), T=cfg.T, budget=budget)
        order, info = tr.orienteering(model, list(range(1, n + 1)), dp_max=cfg.dp_max, K=cfg.hk_K)
        J, L, times = model.evaluate(order)
        log["l3"] = {"solver": info, "J": J, "ticks": L, "budget": budget,
                     "stops": [{"pocket": pockets[k - 1]["id"], "rc": list(pockets[k - 1]["vp"]),
                                "cls": pockets[k - 1]["cls"], "gain": pockets[k - 1]["gain"],
                                "cells": pockets[k - 1]["cells"],
                                "service": pockets[k - 1]["service"], "t_pred": t}
                               for k, t in zip(order, times)]}
        if not order:
            return None
        goals = []
        for k in order:
            p = pockets[k - 1]
            goals.append((p["vp"], "pocket_entry"))
        # inside-pocket viewpoints after the entry of the first pocket (receding horizon)
        first = pockets[order[0] - 1]
        extra = [(v, "pocket_vp") for v in first["vps"] if v != first["vp"]]
        goals = goals[:1] + extra + goals[1:]
        return {"goals": goals, "viz": {"tour": [list(gq[0]) for gq in goals]}}
