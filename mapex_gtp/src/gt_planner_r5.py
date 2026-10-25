"""GLOBAL-001 P1: MapEx-GT goal selection (receding coverage tour on the LaMa-predicted map).

Replaces ONLY MapEx's score_frontiers + argmin when a new goal is needed. Everything else (LiDAR,
mapper, inflated A* grid, executor, 1 m lock radius, WFD termination) stays in the parent MapEx code.
Spec: /home/dell/mapex_workspace/handoffs/GLOBAL-001-claude-to-chatgpt-P1-spec-20261021.md (steps 1-12).
Blocks 2-3 are the validated P0 reference `tourlib.py` (byte copy, SHA recorded in the run manifest).
Online inputs only: observed map, robot pose, LaMa ensemble mean. No GT.
Python 3.6 (lama env).
R2 (2026-10-21, ChatGPT REVISE R1): empty traversable plan graph -> logged MapEx fallback instead of a crash.
R5 (2026-10-21, after the R4 KTH smoke): progress rule for viewpoint goals + blacklist of released unreachable goals."""
import time
import numpy as np
import cv2
from scipy.sparse.csgraph import dijkstra
import tourlib as T

# Fixed before any run (spec / RD-029). Not tuned on P1 results.
REPRED_EVERY = 3          # step 2: refresh LaMa after every 3 reached viewpoints
STRIDE_SMALL = 6          # step 5: candidate stride if core area <= 100k cells
STRIDE_LARGE = 8          #         else
STRIDE_AREA_LIMIT = 100000
MIN_GAIN = 10             # step 8: greedy cover stops below 10 cells
MAX_TRIES = 3             # step 7: need set = unknown cells with tries < 3
OCC_THRESHOLD = 0.5       # step 3: unknown cell is occupied iff mean >= 0.5
# R5 progress rule (fixed before the R5 runs; R4 evidence: NR max path ratio 1.11 / tick ratio 1.02, KTH phantom 24.8 / 60.9)
PROGRESS_PATH_FACTOR = 2.0   # release if current native A* length > 2 * L0 + 30 cells
PROGRESS_PATH_SLACK = 30
PROGRESS_TICK_FACTOR = 2.0   # or if ticks since lock > 2 * (L0 / 3) + 20 (executor moves <= 3 cells / tick)
PROGRESS_TICK_SLACK = 20
CELLS_PER_TICK = 3


class GTPlanner(object):
    def __init__(self, core_shape, pad, lock_radius_cells):
        self.H, self.W = int(core_shape[0]), int(core_shape[1])
        self.pad = int(pad)
        self.lock = float(lock_radius_cells)
        self.stride = STRIDE_SMALL if self.H * self.W <= STRIDE_AREA_LIMIT else STRIDE_LARGE
        self.tries = np.zeros(self.H * self.W, np.int8)
        self.mean_core = None          # last LaMa mean on the core (H, W) float32
        self.var_core = None
        self.big = None                # last (pred_maputils, mean_big, var_big) for the MapEx fallback scorer
        self.reached = 0               # viewpoint counter (unlocks by the native 1 m rule while the goal is a viewpoint)
        self.refresh_epoch = 0         # reached // REPRED_EVERY at the last scheduled refresh
        self.cur = None                # current viewpoint goal: dict(goal_rc, planned, tour, decision, all_known_tick)
        self.prev_tour = None          # tour of the previous accepted viewpoint plan (core rc list)
        self.prev_end = None           # how the previous viewpoint goal ended
        self.n_dec = 0
        self.n_pred_sched = 0
        self.n_pred_fallback = 0
        self.n_fallback = 0
        self.n_reject = 0
        self.n_no_graph = 0             # R2: fallbacks because no traversable plan-graph cell exists
        self.n_order_change = 0
        self.n_order_compared = 0
        self.unlocks = {"reached": 0, "invalid": 0, "no_path": 0, "no_progress": 0}
        self.blacklist = set()          # R5: core rc of goals released for no progress (active while still unknown)
        self.n_blacklist_excluded = 0
        self.shadow_all_known = 0      # optional early-replan rule, logged only (not acted on; P0 config)
        self.events = []

    # ---------- prediction ----------
    def _set_prediction(self, out):
        pred_maputils, mean_big, var_big, info = out
        g = info["overlay"]
        y, x = g["core_global_top"], g["core_global_left"]
        v = var_big.detach().cpu().numpy() if hasattr(var_big, "detach") else np.asarray(var_big)
        self.mean_core = np.ascontiguousarray(mean_big[y:y + self.H, x:x + self.W]).astype(np.float32)
        self.var_core = np.ascontiguousarray(v[y:y + self.H, x:x + self.W]).astype(np.float32)
        assert self.mean_core.shape == (self.H, self.W)
        self.big = (pred_maputils, mean_big, var_big)

    def fallback_prediction(self, predict, refreshed_now):
        """MapEx scorer input. MapEx runs LaMa at every decision, so do the same unless this decision already did."""
        t0 = time.time()
        if not refreshed_now:
            self._set_prediction(predict())
            self.n_pred_fallback += 1
        return self.big, time.time() - t0, (not refreshed_now)

    # ---------- unlock bookkeeping ----------
    def on_unlock(self, reason, obs_full, t):
        """Called when the locked goal is released (reached <1 m / invalid cell / no A* path)."""
        self.unlocks[reason] = self.unlocks.get(reason, 0) + 1
        if self.cur is None:          # fallback (MapEx frontier) goal: no viewpoint bookkeeping
            self.events.append(dict(tick=int(t), kind="unlock_fallback_goal", reason=reason))
            return
        core = obs_full[self.pad:self.pad + self.H, self.pad:self.pad + self.W].ravel()
        planned = self.cur["planned"]
        miss = planned[core[planned] == 0.5]
        self.tries[miss] += 1
        if reason == "reached":
            self.reached += 1
        if reason == "no_progress":
            g = self.cur["goal_rc"]
            self.blacklist.add((int(g[0]) - self.pad, int(g[1]) - self.pad))
        self.events.append(dict(tick=int(t), kind="unlock_viewpoint", reason=reason, decision=self.cur["decision"],
                                goal_rc=self.cur["goal_rc"], planned=int(len(planned)), still_unknown=int(len(miss)),
                                reached_total=self.reached, all_known_tick=self.cur["all_known_tick"],
                                L0=self.cur.get("L0"), t0=self.cur.get("t0"), last_path_len=self.cur.get("last_len")))
        self.prev_end = reason
        self.cur = None

    def on_tick(self, obs_full, t):
        """Shadow of the optional early-replan rule: first tick at which all planned cells are known."""
        if self.cur is None or self.cur["all_known_tick"] is not None:
            return
        core = obs_full[self.pad:self.pad + self.H, self.pad:self.pad + self.W].ravel()
        if not (core[self.cur["planned"]] == 0.5).any():
            self.cur["all_known_tick"] = int(t)
            self.shadow_all_known += 1

    def is_viewpoint_goal(self):
        return self.cur is not None

    def progress_exceeded(self, t, path_len):
        """R5: True if the locked viewpoint goal makes no progress (native A* length or elapsed ticks far beyond the
        values expected at lock time). path_len = current native A* path length in cells."""
        if self.cur is None or self.cur.get("L0") is None:
            return False
        self.cur["last_len"] = int(path_len)
        L0 = self.cur["L0"]
        return (path_len > PROGRESS_PATH_FACTOR * L0 + PROGRESS_PATH_SLACK or
                (t - self.cur["t0"]) > PROGRESS_TICK_FACTOR * (L0 / float(CELLS_PER_TICK)) + PROGRESS_TICK_SLACK)

    # ---------- decision (spec steps 2-12; step 1 = WFD termination stays in the caller) ----------
    def decide(self, t, cur_pose, obs_full, occ_grid_pyastar, predict, astar_ok):
        """Returns (goal_padded or None, info). None => caller uses the original MapEx scorer (fallback)."""
        t_all = time.time()
        self.n_dec += 1
        p = self.pad
        core = np.ascontiguousarray(obs_full[p:p + self.H, p:p + self.W])
        robot = np.asarray(cur_pose, dtype=np.int64) - p
        info = dict(decision_index=self.n_dec, tick=int(t), robot_rc_padded=np.asarray(cur_pose).tolist(),
                    robot_rc_core=robot.tolist(), stride=self.stride, reached_viewpoints=self.reached)
        # 2. prediction refresh: first decision, then after every REPRED_EVERY reached viewpoints
        t0 = time.time()
        refreshed = False
        if self.mean_core is None or self.reached // REPRED_EVERY > self.refresh_epoch:
            self._set_prediction(predict())
            self.refresh_epoch = self.reached // REPRED_EVERY
            self.n_pred_sched += 1
            refreshed = True
        info.update(prediction_refreshed=refreshed, time_lama_s=time.time() - t0)
        # 3. plan map: known from observed; unknown occupied iff mean >= 0.5
        known = core != 0.5
        occ_plan = np.where(known, core == 1, self.mean_core >= OCC_THRESHOLD)
        # 4. traversable graph (3x3 dilation, 8-connected, robot component) + 5. candidates
        t0 = time.time()
        # R2 (ChatGPT review R1): tourlib.Graph needs >= 1 traversable cell (its node_near does argmin over the
        # traversable set). Same traversability rule as tourlib.Graph; if empty -> logged MapEx fallback.
        if not (cv2.dilate(occ_plan.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0).any():
            info.update(n_candidates=0, n_candidates_in_lock_radius=0,
                        n_need=int(((~known).ravel() & (self.tries < MAX_TRIES)).sum()),
                        n_unknown=int((~known).sum()), graph_nodes_kept=0, time_visibility_s=time.time() - t0,
                        rejected=[], fallback=True, fallback_reason="no_traversable_plan_graph",
                        time_cover_s=0.0, time_tsp_s=0.0, time_planner_total_s=time.time() - t_all)
            self.n_fallback += 1
            self.n_no_graph += 1
            self.cur = None
            return None, info
        g = T.Graph(occ_plan, (int(robot[0]), int(robot[1])))
        cand_all = T.candidates(g, self.stride)
        # Native-executor adaptation: a goal within the 1 m lock radius is released immediately by MapEx's
        # own rule (is_locked_frontier_center_valid), so such grid points cannot be goals.
        if len(cand_all):
            far = np.hypot(cand_all[:, 0] - robot[0], cand_all[:, 1] - robot[1]) >= self.lock
        else:
            far = np.zeros(0, bool)
        cand = cand_all[far]
        # R5: drop candidates at / next to a goal released for no progress while that cell is still unknown
        self.blacklist = {b for b in self.blacklist if core[b] == 0.5}
        n_bl = 0
        if self.blacklist and len(cand):
            bl = np.array(sorted(self.blacklist))
            near = (np.abs(cand[:, None, :] - bl[None, :, :]).max(axis=2) <= 1).any(axis=1)
            n_bl = int(near.sum()); cand = cand[~near]
            self.n_blacklist_excluded += n_bl
        # 7. need set
        need = (~known).ravel() & (self.tries < MAX_TRIES)
        # 6. visibility (1440 rays, 20 m, stop at first plan-occupied cell), restricted to the need set
        vis = [v[need[v]] for v in (T.visible(occ_plan, int(r), int(c)) for r, c in cand)]
        info.update(n_candidates=int(len(cand)), n_candidates_in_lock_radius=int((~far).sum()), n_candidates_blacklisted=n_bl,
                    blacklist=sorted([list(b) for b in self.blacklist]),
                    n_need=int(need.sum()), n_unknown=int((~known).sum()), n_tries_exhausted=int(((~known).ravel() & ~(self.tries < MAX_TRIES)).sum()),
                    graph_nodes_kept=int(g.keep.sum()), time_visibility_s=time.time() - t0)
        t_cover = t_tsp = 0.0
        rejected = []
        goal = None
        while True:
            # 8. lazy greedy set cover
            t0 = time.time()
            ch = T.greedy_cover(vis, need, MIN_GAIN)
            t_cover += time.time() - t0
            if not ch:
                info["fallback_reason"] = "cover_empty" if not rejected else "no_reachable_viewpoint"
                break
            # 9. tour: Dijkstra on the plan graph + open TSP (NN + 2-opt)
            t0 = time.time()
            pts = cand[ch]
            nodes = [g.node(robot)] + [g.node(q) for q in pts]
            Dm = dijkstra(g.G, indices=nodes)[:, nodes]
            order = T.open_tsp(Dm)
            t_tsp += time.time() - t0
            first = ch[order[0] - 1]
            gp = (cand[first] + p).astype(np.int64)
            # 12. executor check: same validity test and A* as the native executor
            apath = None if occ_grid_pyastar[gp[0], gp[1]] == np.inf else astar_ok(gp)
            if apath is None or apath is False:
                planned = vis[first]
                self.tries[planned] += 1
                need[planned] &= self.tries[planned] < MAX_TRIES
                vis[first] = vis[first][:0]
                rejected.append(dict(goal_rc=gp.tolist(), planned=int(len(planned)),
                                     cause="inf_cell" if occ_grid_pyastar[gp[0], gp[1]] == np.inf else "astar_none"))
                self.n_reject += 1
                continue
            goal = gp
            tour = [(cand[ch[k - 1]] + p).tolist() for k in order]
            info.update(n_viewpoints=int(len(ch)), tour_first5=tour[:5], tour_len=int(len(tour)),
                        tour_cost_cells=float(Dm[0, order[0]] + sum(Dm[order[i], order[i + 1]] for i in range(len(order) - 1))),
                        chosen_goal_rc=goal.tolist(), planned_cells=int(len(vis[first])),
                        planned_gain_first=int(len(vis[first])))
            # tour-order change: first goal vs the previous plan's second viewpoint (only after a reached goal)
            if self.prev_tour is not None and self.prev_end == "reached" and len(self.prev_tour) >= 2:
                self.n_order_compared += 1
                changed = tour[0] != self.prev_tour[1]
                self.n_order_change += int(changed)
                info["order_changed_vs_prev_second"] = bool(changed)
            self.prev_tour = tour
            L0 = int(len(apath) - 1) if hasattr(apath, '__len__') else None   # native A* length in cells (R5)
            info["L0_native_astar_cells"] = L0
            self.cur = dict(goal_rc=goal.tolist(), planned=vis[first].copy(), tour=tour[:5],
                            decision=self.n_dec, all_known_tick=None, L0=L0, t0=int(t), last_len=None)
            break
        info.update(rejected=rejected, fallback=goal is None, time_cover_s=t_cover, time_tsp_s=t_tsp,
                    time_planner_total_s=time.time() - t_all)
        if goal is None:
            self.n_fallback += 1
            self.cur = None
        return goal, info

    def summary(self):
        return dict(decisions=self.n_dec, fallbacks=self.n_fallback, fallbacks_no_traversable_plan_graph=self.n_no_graph,
                    predictions_scheduled=self.n_pred_sched, predictions_fallback=self.n_pred_fallback,
                    predictions_total=self.n_pred_sched + self.n_pred_fallback,
                    reached_viewpoints=self.reached, executor_rejections=self.n_reject,
                    tour_order_changes=self.n_order_change, tour_order_comparisons=self.n_order_compared,
                    unlock_reasons=self.unlocks, blacklist_excluded_candidates=self.n_blacklist_excluded, shadow_early_replan_all_planned_known=self.shadow_all_known,
                    tries_exhausted_cells=int((self.tries >= MAX_TRIES).sum()), stride=self.stride,
                    params=dict(REPRED_EVERY=REPRED_EVERY, STRIDE_SMALL=STRIDE_SMALL, STRIDE_LARGE=STRIDE_LARGE,
                                STRIDE_AREA_LIMIT=STRIDE_AREA_LIMIT, MIN_GAIN=MIN_GAIN, MAX_TRIES=MAX_TRIES,
                                OCC_THRESHOLD=OCC_THRESHOLD, PROGRESS_PATH_FACTOR=PROGRESS_PATH_FACTOR,
                                PROGRESS_PATH_SLACK=PROGRESS_PATH_SLACK, PROGRESS_TICK_FACTOR=PROGRESS_TICK_FACTOR,
                                PROGRESS_TICK_SLACK=PROGRESS_TICK_SLACK, CELLS_PER_TICK=CELLS_PER_TICK, NRAY=T.NRAY, RANGE=T.RANGE, STEP=T.STEP,
                                lock_radius_cells=self.lock, hysteresis="none (x=0)"))
