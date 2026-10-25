"""MapEx-GTP R7: frozen R5 tour planning plus ER, FR-prime, J5, T3 G50 and exact VIS.

Default viewpoint release retains R5's 1m rule and its progress backstop.
verify_observation, cleanup_retry and fallback_guard stay OFF. Feedback uses
observations and native path lengths only. Native LiDAR, mapper, inflated A*,
controller, WFD and LaMa model trio are unchanged. See docs/SPEC_R7.md.
Python 3.6 (lama environment). Frozen R5 tourlib/cover/TSP are used unchanged.
"""
import time
import os
import hashlib
import numpy as np
import cv2
from scipy.sparse.csgraph import dijkstra
import tourlib as T
from gt_feedback import GoalFeedback, CLEANUP_MAX_ATTEMPTS
from gt_visibility import ExactVisibility

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
    def __init__(self, core_shape, pad, lock_radius_cells, features=None, solver="r5", early_release=False):
        if solver not in ('r5', 'al'):
            raise ValueError('solver must be r5 or al')
        if type(early_release) is not bool:
            raise TypeError('early_release must be boolean')
        self.solver = solver
        self.early_release = early_release
        self.n_early_release = 0
        self.H, self.W = int(core_shape[0]), int(core_shape[1])
        self.pad = int(pad)
        self.lock = float(lock_radius_cells)
        self.stride = STRIDE_SMALL if self.H * self.W <= STRIDE_AREA_LIMIT else STRIDE_LARGE
        self.tries = np.zeros(self.H * self.W, np.int8)
        self.mean_core = None          # last LaMa mean on the core (H, W) float32
        self.var_core = None
        self.big = None                # last (pred_maputils, mean_big, var_big) for the MapEx fallback scorer
        self.reached = 0               # completed viewpoint-leg counter; actual observation success is logged separately
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
        self.n_pred_event = 0
        self.feedback = GoalFeedback(core_shape, pad, self.lock, self.stride, features)
        self.visibility = ExactVisibility()
        self.odom_m = 0.0
        self.last_pose = None
        self.prediction_odom_m = None
        self.prediction_hash = None
        self.determinism_checked = False
        self.determinism_checks = []
        self.jump_events = []

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

    @staticmethod
    def _prediction_hash(out):
        h = hashlib.sha256()
        for value in out[:3]:
            a = value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
            h.update(str(a.shape).encode('ascii'))
            h.update(str(a.dtype).encode('ascii'))
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def _record_prediction(self, out, obs_full, t):
        self._set_prediction(out)
        self.prediction_hash = self._prediction_hash(out)
        self.feedback.prediction_recorded(obs_full, t)
        self.prediction_odom_m = self.odom_m

    def _check_dedup(self, predict, obs_full, t):
        if self.determinism_checked or os.environ.get('GTP_R7_DEDUP_CHECK', '1') != '1':
            return
        actual = self._prediction_hash(predict())
        check = dict(tick=int(t), observed_sha256=self.feedback.observed_digest(obs_full),
                     expected=self.prediction_hash, actual=actual, pass_=actual == self.prediction_hash)
        self.determinism_checks.append(check)
        self.determinism_checked = True
        if not check['pass_']:
            raise RuntimeError('GTP_R7_NONDETERMINISTIC_PREDICTION')

    def _observe_pose(self, pose):
        pose = np.asarray(pose, dtype=float)
        if self.last_pose is not None:
            self.odom_m += float(np.linalg.norm(pose - self.last_pose)) * .1
        self.last_pose = pose.copy()

    def fallback_prediction(self, predict, refreshed_now, obs_full=None, t=None):
        """MapEx scorer input. MapEx runs LaMa at every decision, so do the same unless this decision already did."""
        t0 = time.time()
        same_input = (obs_full is not None and self.feedback.features['event_refresh']
                      and self.feedback.prediction_observed is not None
                      and np.array_equal(self.feedback.prediction_observed, self.feedback.core(obs_full)))
        called = not refreshed_now and not same_input
        if called:
            out = predict()
            self._set_prediction(out)
            self.n_pred_fallback += 1
            if obs_full is not None:
                self._record_prediction(out, obs_full, t)
        elif same_input and not refreshed_now:
            self._check_dedup(predict, obs_full, t)
        return self.big, time.time() - t0, called

    # ---------- unlock bookkeeping ----------
    def on_unlock(self, reason, obs_full, t):
        """Record actual observation results when a viewpoint or fallback is released."""
        outcome = self.feedback.finish(reason, obs_full, self.mean_core, t)
        self.unlocks[reason] = self.unlocks.get(reason, 0) + 1
        if self.cur is None:          # fallback (MapEx frontier) goal: no viewpoint bookkeeping
            self.events.append(dict(tick=int(t), kind="unlock_fallback_goal", reason=reason, observation=outcome))
            return
        core = obs_full[self.pad:self.pad + self.H, self.pad:self.pad + self.W].ravel()
        planned = self.cur["planned"]
        miss = planned[core[planned] == 0.5]
        self.tries[miss] = np.minimum(self.tries[miss].astype(np.int16) + 1, MAX_TRIES)
        if reason in ("reached", "observed"):
            self.reached += 1
        if reason == "no_progress":
            g = self.cur["goal_rc"]
            self.blacklist.add((int(g[0]) - self.pad, int(g[1]) - self.pad))
        self.events.append(dict(tick=int(t), kind="unlock_viewpoint", reason=reason, decision=self.cur["decision"],
                                goal_rc=self.cur["goal_rc"], planned=int(len(planned)), still_unknown=int(len(miss)),
                                reached_total=self.reached, all_known_tick=self.cur["all_known_tick"],
                                L0=self.cur.get("L0"), t0=self.cur.get("t0"), last_path_len=self.cur.get("last_len"), observation=outcome))
        self.prev_end = reason
        self.cur = None

    def on_tick(self, obs_full, t, pose=None):
        """Update actual information progress and first tick all planned cells are known."""
        if pose is not None:
            self._observe_pose(pose)
        self.feedback.update(obs_full, t)
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
            return self.feedback.fallback_exceeded(t, path_len)
        previous = self.cur.get('probe_len', self.cur['L0'])
        previous_tick = self.cur.get('probe_tick', self.cur['t0'])
        self.cur['probe_len'], self.cur['probe_tick'] = int(path_len), int(t)
        jumped = (self.feedback.features['jump_release'] and int(t) == previous_tick + 1
                  and path_len - previous > 50)
        if jumped:
            self.jump_events.append(dict(tick=int(t), previous_tick=previous_tick,
                previous_path_cells=int(previous), path_cells=int(path_len),
                growth_cells=int(path_len - previous), decision=self.cur['decision']))
        self.cur["last_len"] = int(path_len)
        if self.feedback.active is not None:
            self.feedback.active["last_len"] = int(path_len)
        L0 = self.cur["L0"]
        return (jumped or path_len > PROGRESS_PATH_FACTOR * L0 + PROGRESS_PATH_SLACK or
                (t - self.cur["t0"]) > PROGRESS_TICK_FACTOR * (L0 / float(CELLS_PER_TICK)) + PROGRESS_TICK_SLACK)

    # R6 native-integration hooks; fallback retains its native proximity rule.
    def goal_release_reason(self, cur_pose, obs_full):
        reason = self.feedback.release_reason(cur_pose, obs_full)
        # Gate 1 ER: proximity retains priority; observation-complete release is
        # counted as reached, so existing reached/tries/refresh bookkeeping stays.
        if (reason is None and self.early_release and self.cur is not None
                and self.cur['all_known_tick'] is not None):
            self.n_early_release += 1
            return 'reached'
        return reason

    def retain_near_viewpoint(self):
        return self.feedback.retain_near_viewpoint()

    def is_active_goal(self):
        return self.cur is not None or (self.feedback.active is not None
                                       and self.feedback.features['fallback_guard'])

    def finalize_feedback(self, obs_full, t):
        outcome = self.feedback.finish('native_terminal', obs_full, self.mean_core, t)
        if outcome is not None:
            self.events.append(dict(tick=int(t), kind='final_goal_observation', observation=outcome))

    def track_execution_goal(self, t, cur_pose, goal, path, obs_full):
        # Called after native A* reselection, so the feedback goal is the executed goal.
        a = self.feedback.active
        if a is not None and a['goal_rc'] != np.asarray(goal).tolist():
            if self.cur is not None and not self.feedback.features['fallback_guard']:
                raise AssertionError('viewpoint changed after accepted native A*')
            self.on_unlock('native_reselection', obs_full, t)
            a = None
        if a is None:
            self.feedback.start('fallback', goal, [], obs_full, t, len(path) - 1, self.n_dec)

    def filter_fallback_candidates(self, candidates, costs, obs_full, t, valid=None):
        q = np.asarray(candidates)
        costs = np.asarray(costs)
        self.feedback.update_regions(obs_full, self.mean_core, t)
        hard_ok = np.array([not self.feedback.exhausted_fallback(g, obs_full) for g in q], dtype=bool)
        hard_excluded = int((~hard_ok).sum())
        q, costs = q[hard_ok], costs[hard_ok]
        allowed = self.feedback.allowed(q - self.pad)
        # A temporary suppression must not create a new exploration terminal.
        excluded_before_relax = int((~allowed).sum())
        viable = allowed.any()
        if excluded_before_relax and viable and valid is not None:
            viable = any(valid(goal) for goal in q[allowed])
        relaxed = bool(len(q) and not viable)
        if relaxed:
            allowed[:] = True
        return q[allowed], costs[allowed], dict(excluded=int((~allowed).sum()),
                                               blocked_before_relax=excluded_before_relax, relaxed=relaxed,
                                               hard_stall_excluded=hard_excluded,
                                               stop_incomplete=bool(hard_excluded and not len(q)))

    # ---------- decision (WFD termination stays in the caller) ----------
    def decide(self, t, cur_pose, obs_full, occ_grid_pyastar, predict, astar_ok):
        """Returns (goal_padded or None, info). None => caller uses the original MapEx scorer (fallback)."""
        t_all = time.time()
        self.n_dec += 1
        self._observe_pose(cur_pose)
        p = self.pad
        core = np.ascontiguousarray(obs_full[p:p + self.H, p:p + self.W])
        robot = np.asarray(cur_pose, dtype=np.int64) - p
        info = dict(decision_index=self.n_dec, tick=int(t), robot_rc_padded=np.asarray(cur_pose).tolist(),
                    robot_rc_core=robot.tolist(), stride=self.stride, reached_viewpoints=self.reached,
                    solver=self.solver, early_release=self.early_release,
                    solver_cover_size=0, solver_tour_cost_before_ls_cells=0.0,
                    solver_tour_cost_after_ls_cells=0.0, time_solver_s=0.0, solver_attempts=[])
        # 2. prediction refresh: first decision, then after every REPRED_EVERY reached viewpoints
        t0 = time.time()
        refreshed = False
        refresh_reasons = self.feedback.refresh_reasons(obs_full)
        scheduled = self.mean_core is None or self.reached // REPRED_EVERY > self.refresh_epoch
        if scheduled or refresh_reasons:
            self._record_prediction(predict(), obs_full, t)
            self.refresh_epoch = self.reached // REPRED_EVERY
            self.n_pred_sched += int(scheduled)
            self.n_pred_event += int(not scheduled and bool(refresh_reasons))
            refreshed = True
        elif self.feedback.features['event_refresh'] and self.feedback.pending_refresh:
            self._check_dedup(predict, obs_full, t)
        info.update(prediction_refreshed=refreshed, time_lama_s=time.time() - t0,
                    prediction_refresh_reasons=(['initial_or_three_completions'] if scheduled else []) + refresh_reasons,
                    prediction_tick=self.feedback.prediction_tick,
                    prediction_age_ticks=None if self.feedback.prediction_tick is None else int(t) - self.feedback.prediction_tick,
                    prediction_age_m=None if self.prediction_odom_m is None else self.odom_m - self.prediction_odom_m,
                    r7_features=dict(self.feedback.features),
                    previous_goal_observation=self.feedback.outcomes[-1] if self.feedback.outcomes else None)
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
        self.feedback.update_regions(obs_full, self.mean_core, t)
        region_allowed = self.feedback.allowed(cand)
        n_region = int((~region_allowed).sum())
        cand = cand[region_allowed]
        # 7. need set
        need = (~known).ravel() & (self.tries < MAX_TRIES)
        # 6. visibility (1440 rays, 20 m, stop at first plan-occupied cell), restricted to the need set
        self.visibility.set_map(occ_plan)
        full_vis = [self._visible(occ_plan, int(r), int(c)) for r, c in cand]
        vis = [v[need[v]] for v in full_vis]
        info.update(n_candidates=int(len(cand)), n_candidates_in_lock_radius=int((~far).sum()), n_candidates_blacklisted=n_bl,
                    blacklist=sorted([list(b) for b in self.blacklist]),
                    n_need=int(need.sum()), n_unknown=int((~known).sum()), n_tries_exhausted=int(((~known).ravel() & ~(self.tries < MAX_TRIES)).sum()),
                    graph_nodes_kept=int(g.keep.sum()), time_visibility_s=time.time() - t0)
        info['n_candidates_failure_region'] = n_region
        t_cover = t_tsp = 0.0
        cleanup = False
        min_gain = MIN_GAIN
        info['cleanup_mode'] = False
        rejected = []
        goal = None
        while True:
            # 8. lazy greedy set cover
            t0 = time.time()
            if self.solver == 'r5':
                ch = T.greedy_cover(vis, need, min_gain)
                solver_info = {}
            else:
                from gt_al_solver import solve_al
                ch, order, Dm, solver_info = solve_al(g, robot, cand, vis, need, min_gain, self.stride)
            solve_cover_s = time.time() - t0
            t_cover += solve_cover_s
            info['time_solver_s'] += solve_cover_s
            attempt = dict(solver_info)
            attempt['cover_size'] = len(ch)
            info['solver_attempts'].append(attempt)
            info['solver_cover_size'] = len(ch)
            if self.solver == 'al':
                info['solver_tour_cost_before_ls_cells'] = solver_info['tour_cost_before_ls_cells']
                info['solver_tour_cost_after_ls_cells'] = solver_info['tour_cost_after_ls_cells']
            if not ch and not cleanup and self.feedback.features['cleanup_retry']:
                # One bounded recovery attempt, on exhausted unknown cells bordering
                # observed FREE. Targets remain unknown if this attempt also fails.
                retry = ((~known).ravel() & (self.tries >= MAX_TRIES)
                         & (self.feedback.cleanup_attempts < CLEANUP_MAX_ATTEMPTS))
                edge = cv2.dilate((core == 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                retry &= edge.ravel()
                if retry.any():
                    cleanup = True
                    need = retry
                    min_gain = 1
                    # Recovery endpoints must already be observed FREE. Native A*
                    # may still traverse UNKNOWN; its semantics are unchanged.
                    use = (core[cand[:, 0], cand[:, 1]] == 0) if len(cand) else np.zeros(0, bool)
                    vis = [v[need[v]] if use[i] else np.zeros(0, np.int64)
                           for i, v in enumerate(T.visible(occ_plan, int(r), int(c)) for r, c in cand)]
                    info.update(cleanup_mode=True, cleanup_need=int(need.sum()), cleanup_min_gain=1)
                    continue
            if not ch:
                info["fallback_reason"] = "cover_empty" if not rejected else "no_reachable_viewpoint"
                break
            # 9. tour: Dijkstra on the plan graph + open TSP (NN + 2-opt)
            t0 = time.time()
            if self.solver == 'r5':
                pts = cand[ch]
                nodes = [g.node(robot)] + [g.node(q) for q in pts]
                Dm = dijkstra(g.G, indices=nodes)[:, nodes]
                order = T.open_tsp(Dm)
                solve_tsp_s = time.time() - t0
                t_tsp += solve_tsp_s
                info['time_solver_s'] += solve_tsp_s
                cost = float(Dm[0, order[0]] + sum(Dm[order[i], order[i + 1]] for i in range(len(order) - 1)))
                # R5 has no covering local search: its NN+2opt result is both values.
                info['solver_tour_cost_before_ls_cells'] = cost
                info['solver_tour_cost_after_ls_cells'] = cost
                attempt.update(tour_cost_before_ls_cells=cost, tour_cost_after_ls_cells=cost)
            first = ch[order[0] - 1]
            gp = (cand[first] + p).astype(np.int64)
            # 12. executor check: same validity test and A* as the native executor
            apath = None if occ_grid_pyastar[gp[0], gp[1]] == np.inf else astar_ok(gp)
            if apath is None or apath is False:
                planned = vis[first]
                self.tries[planned] = np.minimum(self.tries[planned].astype(np.int16) + 1, MAX_TRIES)
                if cleanup:
                    self.feedback.cleanup_attempts[planned] += 1
                    need[planned] = False
                self.feedback.add_region(gp, obs_full, self.mean_core, 'executor_rejection', t)
                if self.feedback.features['event_refresh']:
                    self.feedback.pending_refresh.add('executor_rejection')
                need[planned] &= self.tries[planned] < MAX_TRIES
                vis[first] = vis[first][:0]
                if self.feedback.features['failure_regions']:
                    for ci in np.flatnonzero(~self.feedback.allowed(cand)):
                        vis[ci] = vis[ci][:0]
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
            if self.prev_tour is not None and self.prev_end in ("reached", "observed") and len(self.prev_tour) >= 2:
                self.n_order_compared += 1
                changed = tour[0] != self.prev_tour[1]
                self.n_order_change += int(changed)
                info["order_changed_vs_prev_second"] = bool(changed)
            self.prev_tour = tour
            L0 = int(len(apath) - 1) if hasattr(apath, '__len__') else None   # native A* length in cells (R5)
            info["L0_native_astar_cells"] = L0
            self.feedback.start('viewpoint', goal, vis[first], obs_full, t, L0, self.n_dec)
            if cleanup:
                self.feedback.cleanup_attempts[vis[first]] += 1
            self.cur = dict(goal_rc=goal.tolist(), planned=vis[first].copy(), tour=tour[:5],
                            decision=self.n_dec, all_known_tick=None, L0=L0, t0=int(t), last_len=None)
            break
        info.update(rejected=rejected, fallback=goal is None, time_cover_s=t_cover, time_tsp_s=t_tsp,
                    time_planner_total_s=time.time() - t_all)
        if goal is None:
            self.n_fallback += 1
            self.cur = None
        return goal, info

    def _visible(self, occ_plan, r, c):
        if self.feedback.features['incremental_visibility']:
            return self.visibility.visible(occ_plan, r, c)
        return T.visible(occ_plan, r, c)

    def tail_check(self, frontier_centres, obs_full, info):
        # Empty cover only, after fallback prediction refresh/dedup. Counts ALL
        # visible UNKNOWN, including exhausted tries. Never uses the true map.
        if not self.feedback.features['tail_rule'] or info.get('fallback_reason') != 'cover_empty':
            return False
        core = self.feedback.core(obs_full)
        occ = np.where(core != .5, core == 1, self.mean_core >= OCC_THRESHOLD)
        self.visibility.set_map(occ)
        gains = []
        for goal in np.asarray(frontier_centres).reshape((-1, 2)):
            r, c = (goal.astype(int) - self.pad).tolist()
            if not (0 <= r < self.H and 0 <= c < self.W):
                raise ValueError('WFD frontier centre outside core')
            v = self._visible(occ, r, c)
            gains.append(int((core.ravel()[v] == .5).sum()))
        maximum = max(gains) if gains else 0
        stop = maximum < 50
        info.update(tail_threshold=50, tail_max_visible_unknown=maximum,
                    tail_frontier_gains=gains, tail_sensitivity_stop_G100=maximum < 100,
                    terminate=stop, terminal_class='GTP_TAIL_STOP' if stop else None)
        return stop

    def summary(self):
        return dict(solver=self.solver, early_release=self.early_release, n_early_release=self.n_early_release,
                    r7_feedback=self.feedback.summary(), visibility=self.visibility.summary(),
                    determinism_checks=self.determinism_checks, debug_prediction_calls=len(self.determinism_checks),
                    jump_events=self.jump_events, decisions=self.n_dec, fallbacks=self.n_fallback, fallbacks_no_traversable_plan_graph=self.n_no_graph,
                    predictions_scheduled=self.n_pred_sched, predictions_fallback=self.n_pred_fallback,
                    predictions_total=self.n_pred_sched + self.n_pred_event + self.n_pred_fallback + len(self.determinism_checks),
                    predictions_event_triggered=self.n_pred_event,
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
