"""TOPO-002 native policy hook (R2 unified planner; copied from TOPO-001, same hook API).

TOPO-001 docstring follows.
TOPO-001 native policy hook (drop-in for the MX1000 R4 `PolicyHarness`).

Same call sites as the R4 hook, which was bit-exact with the KTH49 baseline in
passthrough: `choose` at the goal-selection point and in the A*-None reselect
loop, `reject_last`, `track_stale`.  One extra call `goal(centers, index)`
returns the locked goal: the planner goal (any reachable cell) or, on any
fallback, the MapEx frontier chosen exactly as MapEx would (lowest cost among
eligible frontiers, stable index tie break).

No ground truth: `mapper` is used only for `mapper.obs_map`.  The R4 GT
view_leg/view_gain evaluation is removed.

Policies (env TOPO_POLICY): `topo` (default) | `mapex` (no planner, same
fallback code path; for debugging).
"""
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pyastar2d

NATIVE_PAD = 500
POLICIES = {"topo", "mapex"}


def centered_padding(n, divisor=16):
    total = (divisor - int(n) % divisor) % divisor
    return total // 2, total - total // 2


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _jwrite(path, obj):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), sort_keys=True, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(p))


def core_inputs(obs_full, mean_map, var_map, grid, robot, centers, pad=NATIVE_PAD):
    """Crop native padded arrays to the CORE frame used by L1/planner (== logger npz)."""
    H, W = obs_full.shape
    h, w = H - 2 * pad, W - 2 * pad
    obs = np.asarray(obs_full[pad:pad + h, pad:pad + w], dtype=np.float32)
    m = np.asarray(mean_map)
    v = var_map.detach().cpu().numpy() if hasattr(var_map, "detach") else np.asarray(var_map)
    m, v = np.squeeze(m), np.squeeze(v)
    y = pad + centered_padding(H)[0]
    x = pad + centered_padding(W)[0]
    mean = np.asarray(m[y:y + h, x:x + w], dtype=np.float32)
    var = np.asarray(v[y:y + h, x:x + w], dtype=np.float32)
    if mean.shape != obs.shape or var.shape != obs.shape:
        raise ValueError("core crop mismatch %s %s %s" % (obs.shape, mean.shape, var.shape))
    plan_finite = np.isfinite(np.asarray(grid)[pad:pad + h, pad:pad + w])
    rob = (int(robot[0]) - pad, int(robot[1]) - pad)
    fr = []
    for p in np.asarray(centers, dtype=np.int64).reshape(-1, 2):
        q = (int(p[0]) - pad, int(p[1]) - pad)
        if 0 <= q[0] < h and 0 <= q[1] < w:
            fr.append(q)
    return obs, mean, var, plan_finite, rob, fr


class PolicyHarness(object):
    def __init__(self, output_dir, ppm, tick_budget=1000, policy=None):
        self.policy = policy or os.environ.get("TOPO_POLICY", "topo")
        if self.policy not in POLICIES:
            raise ValueError("unknown TOPO_POLICY: " + self.policy)
        self.ppm = float(ppm)
        # planner horizon; TOPO_PLANNER_T keeps T=1000 semantics in shortened smoke runs
        self.T = int(os.environ.get("TOPO_PLANNER_T", tick_budget))
        self.mission_time = int(tick_budget)
        self.out = Path(output_dir)
        self.log = self.out / "topo_decisions.jsonl"
        self.stats_path = self.out / "topo_policy_stats.json"
        self.debug_dir = self.out / "topo_debug"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.debug_png = os.environ.get("TOPO_DEBUG_PNG", "1") == "1"
        self.base_costs = None
        self.excluded = set()
        self.excluded_goals = []      # core-frame topo goals rejected in this decision
        self.last_index = None
        self.last_kind = None
        self.pending_goal = None
        self.decision_no = 0
        self.calls = 0
        self.reselects = 0
        self.stale_ticks = 0
        self.stale_logged = set()
        self.topo_goals = 0
        self.fallbacks = {}
        self.phase_counts = {}
        self.first_stop_corridor = 0      # decisions whose first stop lies on a corridor chain
        self.decisions_with_stops = 0
        self.min_dist_relaxed = 0
        self.p_free = None
        self.pc_source = None
        self.pc_fallback_used = None
        self.l1_state = None
        self.goal_info = None
        self.carry_exclude = []       # goals abandoned by a path blow-up, excluded next decision
        self.replans = {}
        # replan trigger parameters (TOPO goals only)
        self.rp_min_ticks = 5
        self.rp_targets_left = 0.2
        self.rp_blowup_factor = 1.5
        self.rp_blowup_add_m = 5.0
        self.rp_check_every = 3
        self.l1_cache = None          # (tick, topo) reused on reselect within a tick
        self.planner = None
        self.build_topo = None
        self.init_error = None
        if self.policy == "topo":
            try:
                from topo_planner import UnifiedPlanner, R2Config
                self.planner = UnifiedPlanner(R2Config(ppm=self.ppm, T=float(self.T)))
                self.build_topo = load_build_topo()
                self.p_free, self.pc_source = load_p_free()
            except Exception:
                self.init_error = traceback.format_exc()
        self._stats()

    # ---------------------------------------------------------------- bookkeeping
    def _stats(self):
        _jwrite(self.stats_path, {
            "policy": self.policy, "decisions": self.decision_no, "hook_calls": self.calls,
            "reselects": self.reselects, "topo_goals": self.topo_goals,
            "fallbacks_total": int(sum(self.fallbacks.values())), "fallbacks": self.fallbacks,
            "phase_counts": self.phase_counts, "stale_goal_ticks": self.stale_ticks,
            "replan_triggers": self.replans,
            "l1_preset": getattr(self.build_topo, "preset", "spec"),
            "pc_source": self.pc_source, "pc_fallback_used": self.pc_fallback_used,
            "l1_takes_pfree": getattr(self.build_topo, "takes_pfree", None),
            "first_stop_on_corridor_chain": self.first_stop_corridor,
            "decisions_with_stops": self.decisions_with_stops,
            "min_dist_relaxed": self.min_dist_relaxed,
            "init_error": self.init_error})

    def _record(self, event):
        with self.log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(event), sort_keys=True) + "\n")

    def _fallback(self, reason):
        key = reason.split(":")[0]
        self.fallbacks[key] = self.fallbacks.get(key, 0) + 1

    def reject_last(self):
        if self.last_kind is None:
            raise RuntimeError("reselect before a choice")
        if self.last_kind == "topo":
            self.excluded_goals.append(tuple(int(v) - NATIVE_PAD for v in self.pending_goal))
        else:
            self.excluded.add(self.last_index)
        self.reselects += 1
        self._stats()

    def track_stale(self, tick, goal, obs_map):
        """Ticks whose locked goal has no UNKNOWN cell within 2 m (observed info only)."""
        if goal is None or tick in self.stale_logged:
            return
        self.stale_logged.add(tick)
        row, col = np.asarray(goal, dtype=int)
        rad = int(round(2 * self.ppm))
        r0, r1 = max(0, row - rad), min(obs_map.shape[0], row + rad + 1)
        c0, c1 = max(0, col - rad), min(obs_map.shape[1], col + rad + 1)
        yy, xx = np.ogrid[r0:r1, c0:c1]
        if not np.any((obs_map[r0:r1, c0:c1] == 0.5) & ((yy - row) ** 2 + (xx - col) ** 2 <= rad ** 2)):
            self.stale_ticks += 1
            self._stats()

    def should_replan(self, tick, goal, robot, grid, obs_map):
        """Ask for a new decision while a TOPO goal is still natively valid.

        Never for MapEx/fallback goals.  Triggers (>= rp_min_ticks after the decision):
        * targets_observed: < 20 % of the cells the goal was chosen to observe are still UNKNOWN;
        * path_blowup: native A* path > 1.5 x the length at decision time + 5 m
          (checked every 3 ticks); the goal is then excluded at the next decision."""
        gi = self.goal_info
        if self.last_kind != "topo" or gi is None or goal is None:
            return False
        if not np.array_equal(np.asarray(goal, dtype=np.int64), gi["goal"]):
            return False
        if tick - gi["tick"] < self.rp_min_ticks:
            return False
        reason = None
        tg = gi["targets"]
        left = None
        if tg is not None and len(tg):
            left = float(np.mean(obs_map[tg[:, 0], tg[:, 1]] == 0.5))
            if left < self.rp_targets_left:
                reason = "targets_observed"
        cells = None
        if reason is None and tick - gi["last_check"] >= self.rp_check_every:
            gi["last_check"] = int(tick)
            path = pyastar2d.astar_path(grid, np.asarray(robot, dtype=np.int64), gi["goal"],
                                        allow_diagonal=False)
            if path is not None:
                cells = int(len(path) - 1)
                if cells > self.rp_blowup_factor * gi["astar_cells"] + self.rp_blowup_add_m * self.ppm:
                    reason = "path_blowup"
                    self.carry_exclude.append(tuple(int(v) - NATIVE_PAD for v in gi["goal"]))
        if reason is None:
            return False
        self.replans[reason] = self.replans.get(reason, 0) + 1
        self._record({"event": "replan_trigger", "tick": int(tick), "decision": self.decision_no,
                      "reason": reason, "goal_rc": gi["goal"].tolist(), "targets_unknown_frac": left,
                      "astar_cells_now": cells, "astar_cells_at_decision": gi["astar_cells"]})
        self._stats()
        return True

    def goal(self, centers, index):
        if self.pending_goal is not None:
            return self.pending_goal.copy()
        return centers[index]

    # ---------------------------------------------------------------- MapEx choice
    def _mapex_valid(self, points, costs, robot, grid):
        """MapEx choice among eligible frontiers (R4 rule without GT): lowest
        (cost, index) with Euclid >= 1 m, finite grid, A* path >= 1 m."""
        for i in np.argsort(costs, kind="stable"):
            i = int(i)
            p = points[i]
            if i in self.excluded:
                continue
            if float(np.linalg.norm(p - robot)) / self.ppm < 1.0 or not np.isfinite(grid[p[0], p[1]]):
                continue
            path = pyastar2d.astar_path(grid, np.asarray(robot, dtype=np.int64), p,
                                        allow_diagonal=False)
            if path is not None and (len(path) - 1) / self.ppm >= 1.0:
                return i
        return None

    # ---------------------------------------------------------------- topo choice
    def _plan_topo(self, tick, points, robot, grid, mapper, mean_map, var_map, ev):
        t0 = time.perf_counter()
        if self.init_error is not None:
            raise RuntimeError("init: " + self.init_error.strip().splitlines()[-1])
        obs, mean, var, plan_finite, rob, fr = core_inputs(mapper.obs_map, mean_map, var_map,
                                                           grid, robot, points)
        if self.l1_cache is not None and self.l1_cache[0] == tick:
            topo, p = self.l1_cache[1], self.l1_cache[2]
            ev["l1_cached"] = True
            tp = t1 = time.perf_counter()
        else:
            tp0 = time.perf_counter()
            p = self.p_free(obs, mean, var)
            self.pc_fallback_used = getattr(self.p_free, "__globals__", {}).get("fallback_used")
            tp = time.perf_counter()
            ev["pc_s"] = tp - tp0
            topo = self.build_topo(obs, mean, var, rob, self.ppm, self.l1_state, pfree=p)
            self.l1_state = topo.get("state") if isinstance(topo, dict) else getattr(topo, "state", None)
            self.l1_cache = (tick, topo, p)
        t1 = tp
        t2 = time.perf_counter()
        res = self.planner.plan(obs, mean, var, p, rob, topo, plan_finite, fr, tick,
                                exclude_goals=self.excluded_goals)
        log0 = res["log"]
        log0["phase"] = "R2"
        if res["goals"] and ev["reset"]:
            self.decisions_with_stops += 1
            self.first_stop_corridor += int(bool(log0.get("first_stop_on_corridor_chain")))
            self.min_dist_relaxed += int(bool(log0.get("min_dist_relaxed")))
        t3 = time.perf_counter()
        chosen = None
        tried = []
        targets = res.get("targets") or [None] * len(res["goals"])
        for (q, kind), tg in zip(res["goals"], targets):
            g = np.array([q[0] + NATIVE_PAD, q[1] + NATIVE_PAD], dtype=np.int64)
            ok = np.isfinite(grid[g[0], g[1]]) and np.linalg.norm(g - robot) >= self.ppm
            path = None
            if ok:
                path = pyastar2d.astar_path(grid, np.asarray(robot, dtype=np.int64), g,
                                            allow_diagonal=False)
                ok = path is not None
            tried.append({"rc_core": list(q), "kind": kind, "ok": bool(ok),
                          "astar_cells": None if path is None else int(len(path) - 1)})
            if ok:
                chosen = (g, q, kind)
                self.goal_info = {"tick": int(tick), "goal": g.copy(), "astar_cells": int(len(path) - 1),
                                  "targets": None if tg is None else np.asarray(tg, np.int64) + NATIVE_PAD,
                                  "last_check": int(tick)}
                break
            self.excluded_goals.append(tuple(q))
        t4 = time.perf_counter()
        log = res["log"]
        ev.update({"phase": log.get("phase"), "planner": log, "goal_tries": tried,
                   "l1_timing_s": _l1_timing(topo),
                   "timing_s": {"crop": t1 - t0, "l1": t2 - t1, "planner": t3 - t2,
                                "astar_validate": t4 - t3}})
        if self.debug_png:
            try:
                from topo_planner import debug_viz
                tv = time.perf_counter()
                title = "d%d t%d %s goal=%s l1=%.2fs plan=%.2fs" % (
                    self.decision_no, tick, log.get("phase"),
                    chosen[2] if chosen else "FALLBACK", t2 - t1, t3 - t2)
                debug_viz.render(self.debug_dir / ("decision_%04d_t%04d%s.png" % (
                    self.decision_no, tick, "" if ev["reset"] else "_r%d" % self.reselects)),
                    obs, topo, rob, res, chosen[1] if chosen else None, title)
                ev["timing_s"]["debug_png"] = time.perf_counter() - tv
            except Exception:
                ev["debug_png_error"] = traceback.format_exc().strip().splitlines()[-1]
        return chosen

    # ---------------------------------------------------------------- hook
    def choose(self, tick, centers, scored_costs, robot, grid, mapper, mean_map=None,
               var_map=None, trail=None, reset=True):
        """Return (index, costs) like R4; the locked goal is then `goal(centers, index)`."""
        started = time.perf_counter()
        self.calls += 1
        points = np.asarray(centers, dtype=np.int64).reshape(-1, 2)
        costs = np.asarray(scored_costs, dtype=np.float64)
        robot = np.asarray(robot, dtype=np.int64)
        if reset:
            self.base_costs = costs.copy()
            self.excluded.clear()
            self.excluded_goals = list(self.carry_exclude)
            self.carry_exclude = []
            self.goal_info = None
            self.decision_no += 1
        else:
            if self.base_costs is None or len(self.base_costs) != len(points):
                raise RuntimeError("reselect candidates changed")
            costs = self.base_costs.copy()
        self.pending_goal = None
        ev = {"tick": int(tick), "decision": self.decision_no, "reset": bool(reset),
              "robot_rc": robot.tolist(), "n_frontiers": int(len(points)),
              "mapex_argmin": int(np.argmin(costs)) if len(costs) else None}
        chosen = None
        if self.policy == "topo":
            try:
                chosen = self._plan_topo(int(tick), points, robot, grid, mapper, mean_map,
                                         var_map, ev)
                if chosen is None:
                    ev["fallback"] = "no_valid_goal"
            except Exception:
                tb = traceback.format_exc()
                ev["fallback"] = "exception: " + tb.strip().splitlines()[-1]
                ev["traceback"] = tb
        else:
            ev["fallback"] = "policy_mapex"
        if chosen is not None:
            g, q, kind = chosen
            self.pending_goal = g
            self.last_kind = "topo"
            self.topo_goals += 1
            ph = ev.get("phase") or "?"
            self.phase_counts[ph] = self.phase_counts.get(ph, 0) + 1
            # any valid index; goal() returns the planner goal.  Costs untouched.
            idx = int(np.argmin(costs)) if len(costs) else 0
            self.last_index = idx
            ev.update({"goal_rc": g.tolist(), "goal_rc_core": list(q), "goal_kind": kind,
                       "chosen_index": None, "hook_wall_s": time.perf_counter() - started})
            self._record(ev)
            self._stats()
            return idx, costs
        # ---- fallback: plain MapEx choice (counted)
        self._fallback(ev["fallback"])
        self.phase_counts["FALLBACK"] = self.phase_counts.get("FALLBACK", 0) + 1
        mapex_valid = None
        if len(points) and len(points) == len(costs):
            mapex_valid = self._mapex_valid(points, costs, robot, grid)
        ev["mapex_valid_choice"] = mapex_valid
        if mapex_valid is None:
            ev.update({"chosen_index": None, "goal_rc": None,
                       "hook_wall_s": time.perf_counter() - started})
            self._record(ev)
            self._stats()
            return None, costs
        forced = costs.copy()
        forced[mapex_valid] = np.nextafter(float(np.min(costs)), -np.inf)
        self.last_index = mapex_valid
        self.last_kind = "mapex"
        ev.update({"chosen_index": mapex_valid, "goal_rc": points[mapex_valid].tolist(),
                   "goal_kind": "mapex_frontier", "hook_wall_s": time.perf_counter() - started})
        self._record(ev)
        self._stats()
        return mapex_valid, forced


def load_build_topo():
    """L1 entry point: TOPO_L1_FILE, else <run root>/topo_l1/topo_l1.py, else `import topo_l1`."""
    import importlib.util
    cands = [os.environ.get("TOPO_L1_FILE")]
    root = os.environ.get("DELL_NATIVE2D_ROOT")
    if root:
        cands.append(os.path.join(root, "topo_l1", "topo_l1.py"))
    for f in cands:
        if f and os.path.isfile(f):
            spec = importlib.util.spec_from_file_location("topo_l1_frozen", f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return _with_preset(mod)
    import topo_l1
    return _with_preset(topo_l1)


def _with_preset(mod):
    """build_topo bound to TOPO_L1_PRESET (default: 'v2' if the module has it, else 'v1');
    `pfree` is forwarded only if TOPO_L1_PFREE=1 and the L1 accepts it (default: not forwarded)."""
    import inspect
    presets = getattr(mod, "PRESETS", {})
    name = os.environ.get("TOPO_L1_PRESET") or ("v2" if "v2" in presets else "v1")
    params = None if name in ("", "spec", "v0") else dict(presets[name])
    # Coordinator decision (L1 v2 freeze): structure uses its own 3 m range, p(c) is NOT passed to L1
    # unless TOPO_L1_PFREE=1.
    takes_pfree = ("pfree" in inspect.signature(mod.build_topo).parameters
                   and os.environ.get("TOPO_L1_PFREE", "0") == "1")

    def build_topo(obs, mean, var, robot_rc, ppm, prev=None, pfree=None):
        kw = {"params": params} if params is not None else {}
        if takes_pfree:
            kw["pfree"] = pfree
        return mod.build_topo(obs, mean, var, robot_rc, ppm, prev, **kw)
    build_topo.preset = name
    build_topo.takes_pfree = takes_pfree
    return build_topo


def load_p_free():
    """p(c): TOPO_PC_FILE, else <run root>/pc_model/pc_model.py, else the stub 1[mean < 0.5].
    Returns (callable(obs, mean, var) -> float32 map, source tag)."""
    import importlib.util
    cands = [os.environ.get("TOPO_PC_FILE")]
    root = os.environ.get("DELL_NATIVE2D_ROOT")
    if root:
        cands.append(os.path.join(root, "pc_model", "pc_model.py"))
    if os.environ.get("TOPO_PC", "") != "stub":
        for f in cands:
            if f and os.path.isfile(f):
                spec = importlib.util.spec_from_file_location("pc_model_frozen", f)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod.p_free, "pc_model:" + f
    from topo_planner.pc_stub import p_free
    return p_free, "stub_mean_lt_0.5"


def _l1_timing(topo):
    t = topo.get("timing_s") if isinstance(topo, dict) else getattr(topo, "timing_s", None)
    return t
