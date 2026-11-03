# TOPO-002 WP-D — unified planner R2 (topo-planner session)

Code: `planner/topo_planner/planner_r2.py` (`UnifiedPlanner`, `R2Config`), `tour.reorder_prefix` (Held–Karp on a
prefix), `pc_stub.py`; native hook `native/topo_policy.py` (copy of TOPO-001 hook, same API).
The TOPO-001 L2/L3 planner (`planner.py`) is kept only for reference and is not called.
Development data: synthetic worlds only (no KTH/TEST frames read).

## Algorithm (TOPO-002 §2.3)
* **p(c)** = `pc_model.p_free(obs, mean, var)` (loaded from `<run>/pc_model/pc_model.py`; stub = 1[mean < 0.5]).
  Computed once per decision in the hook and passed to L1 v2 (`pfree=`) and to the planner.
* **Gain cells** = UNKNOWN with p ≥ 0.05; weight p. **Ray blocking**: OCC, or UNKNOWN with p < 0.5 (§2.1).
  **Travel**: BFS on native-finite ∧ (FREE ∨ UNKNOWN with p ≥ 0.5), ticks = cells / 3.
* **Candidates**: (a) L1 v2 `chains` (`polyline_rc`, `cls`) every 1.5 m — v1: region skeletons, corridor flag from
  region class; (b) doors; (c) L1 v2 `viewpoints` (`rc`, `kind`) — v1: region centroids; (d) **rim viewpoints**:
  observed-FREE cells within 0.2 m of each UNKNOWN 8-connected component with Σp ≥ 300, thinned to 1.5 m,
  ≤ 30 per component; (e) WFD frontiers. Snapped to traversable (≤ 0.4 m), reachable, ≥ 3 m from the robot;
  if no candidate ≥ 3 m has gain ≥ 20 expected cells, retried at ≥ 1.2 m (logged `min_dist_relaxed`).
  ≤ 400 candidates; the 80 best by g / (ticks + 1) enter the solver (`pool`).
* **Objective** J = Σ g_i (T − t_i), g_i = marginal expected gain (each cell credited to the first selected stop
  that sees it — no double counting), service 0, hard budget T − tick.
* **Solver**: lazy-greedy (CELF priority queue) on ΔJ/Δt of the best insertion of each pool candidate with its
  current marginal gain, until no positive ΔJ, 25 stops or budget; then 2-opt + Or-opt; then **exact Held–Karp
  re-ordering of the first ≤ 10 stops** with the tail fixed (DP V(S∪k,k) = V(S,j) + W_rest(S)·D[j,k], plus the
  tail's cost from each possible prefix end). The HK order is kept only if the clipped, budgeted J increases.
* Receding horizon: goal = first stop (hook validates finite grid / ≥ 1 m / native A*, else next stop).
  Kept from TOPO-001: replan trigger (targets < 20 % UNKNOWN, A* > 1.5× + 5 m), MapEx fallback, logging.
* "Backbone first" is not coded. Logged per decision: `first_stop_on_corridor_chain`,
  `share_stops_on_corridor_chain`; aggregated in `topo_policy_stats.json`
  (`first_stop_on_corridor_chain` / `decisions_with_stops`).

## Approximations / deviations
1. Lazy-greedy ratio is not strictly monotone (insertion Δt can shrink as the tour grows), CELF is a heuristic here.
2. Held–Karp prefix DP ignores the max(0, ·) clip at T; exact vs brute force when no gain is clipped
   (tests, 60 instances); a DP order that lowers the clipped J is rejected.
3. Gains of stops are fixed at selection time (marginal w.r.t. earlier-selected stops), not recomputed for the final order.
4. BFS travel (lower bound of native weighted A*), as in TOPO-001.
5. Rim threshold Σp ≥ 300 (≈ 3 m² expected free) and band 0.2 m are my choices (not in spec); to be checked on DEV.

## Tests (synthetic only)
`planner/topo_planner/tests`: 24 (tour incl. prefix-HK vs brute force, raycast parity, TOPO-001 planner, 8 R2 tests:
no double counting / budget / order, ≥ 3 m, relaxation, **large UNKNOWN wing outside every L1 region is targeted
via rim viewpoints** (KTH50 failure mode), corridor-chain flag with L1 v2 chains, exclusion, empty, rooms).
`native/tests`: 9 (replica native loop with fake L1, real L1 v1 and real L1 v2-dev, crop parity, replan, reselect,
fallback, no GT access).

## Timing on DEV (50010535_PLAN1, 596×2057 core, 9 recorded decisions, offline)
After lazy distances (BFS only from the robot and selected stops, early stop at pool targets), 1 m candidate
dedupe by priority, ≤ 20 stops, pool 60, 2 local-search rounds: planner median 1.31 s, max 2.11 s;
pc stub + L1 v1 + planner median 4.3 s, max 5.6 s. **L1 v2-dev takes 8.7–12.2 s/decision on this map**
(pc_model 0.2–0.4 s) → hook gate ≤ 6 s not met with v2 as of now (topo-l1's part).

## Smoke TOPO2-50010535_PLAN1-P00-SMOKE-01 (DEV, 150 ticks, L1 v1 + p stub, 2026-11-03)
Baseline dir MXDEV-50010535_PLAN1-P00-T400-01 (topo2-data). Exit 0, 150/150, 0 traceback, no wall hit.
6 decisions (all R2), 0 fallbacks, 0 reselects, replan targets_observed 1, min-dist relaxations 0,
first stop on a corridor-class region 3/6. Hook: 4.81 s (first, numba compile), then 1.93–3.19 s
(L1 v1 1.6–1.9 s, planner 0.28–1.44 s). Peak RSS 3.15 GiB, wall 11 min. PNGs in the run's `topo_debug/`.
HK prefix re-ordering never improved on 2-opt/Or-opt in these 6 decisions (logged `hk.improved`).
Note: with v1, "corridor" = region class p(corridor) ≥ 0.5, not an L1 v2 chain.

## Frozen L1 v2 + p(c) (topo-l1, 2026-11-03) — final configuration
L1 v2 `topo_l1.py` 7aba149a…, preset v2, **p(c) not passed to L1** (coordinator decision; hook flag
`TOPO_L1_PFREE`, default 0; manifest `l1_pfree: false`). pc_model.py fc2c40f8… + pc_frozen.json fe1f75a0…
(`fallback_used` False checked in tests and bench). p(c) is used only for ranking gain (poor cross-building calibration).
Offline on all 6 DEV runs (every 2nd decision): pc 0.45–1.13 s, L1 v2 0.12–1.19 s, planner 0.34–1.28 s,
**total median 1.8–2.6 s, max 2.78 s** per decision. First stop on an L1 v2 corridor chain in 11/27 benched decisions.
Native tests (9) incl. replica loop with frozen L1 v2 + pc_model: OK (16 decisions, 0 fallbacks, first stop on corridor 5/16).
To run the final config: `prepare_run_topo.py ... --l1-dir build/topo002/topo_l1 --l1-preset v2 --pc-dir build/topo002/pc_model`.

## Smoke TOPO2-50015847-P00-SMOKE-01 (DEV 50015847, 150 ticks, FINAL config, 2026-11-03)
L1 v2 frozen (preset v2, no pfree) + pc_model frozen (`fallback_used` False); baseline MXDEV-50015847-P00-T400-01.
Exit 0, 150/150, 0 traceback. 7 decisions, 0 fallbacks, 0 reselects, 0 min-dist relaxations, replan targets_observed 1,
first stop on a corridor chain 1/7 (first-stop kinds: chain_room, vp_chain, rim ×4, chain_corridor).
Hook 5.54 s at t0 (numba compile), then 1.32–2.42 s (crop+p(c) 0.77–0.85 s — logged under `crop`, L1 0.2–0.46 s,
planner 0.21–1.08 s). Peak RSS 4.32 GiB, wall 25 min (large DEV map, slow LaMa steps).
