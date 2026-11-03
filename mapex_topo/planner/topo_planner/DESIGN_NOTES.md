# topo_planner — L2/L3 design notes (TOPO-001)

Owner: topo-planner session. Python 3.6 (`lama` env), numpy/scipy/numba/cv2 only. No GT anywhere.

## Frame & inputs
All planner arrays are in the **CORE frame** (= `plain_mapex_logger` decision npz):
obs = obs_full[500:-500, 500:-500]; mean/var = LaMa maps shifted by the centred pad-to-16 offset
(`native/topo_policy.core_inputs`, unit-tested against the logger formula).
`plan_finite` = finite cells of the native inflated A* grid (`Mapper.get_inflated_planning_maps`;
note MapEx always calls it with unknown_as_occ=False, so UNKNOWN is traversable).

## Modules
| file | content |
|---|---|
| `geometry.py` | numba 4-connected BFS distance fields; ticks = cells / 3; snapping; 1.5 m point thinning |
| `visibility.py` | predicted-map visibility: MapEx ray semantics, polygon fill, gain-cell ids, per-decision cache |
| `tour.py` | J = Σ g·max(0, T − t) model, greedy insertion, 2-opt, Or-opt (seg ≤ 3, both directions), removal, Held–Karp, brute force |
| `planner.py` | `TopoPlanner.plan(...)`: phase rule, L2 tour + pocket insertion, L3 orienteering, JSON log |
| `debug_viz.py` | per-decision PNG (observed, L1 regions by class, backbone outline, doors, candidates, tour, goal) |
| `l1_stub.py` | L1 stand-ins: labelled-fixture TopoMap (tests) and coarse CC stub (timing) |
| `tools/bench_recorded.py` | offline run on recorded baseline decision inputs (stub or real L1) |

## Algorithm (as implemented)
* **Gain cells** U = UNKNOWN ∧ mean < 0.5 (coverage metric counts GT-free cells, so predicted-occupied
  hit cells do not count). Backbone gain cells U_bb = U ∧ backbone_mask.
* **Traversability** for travel estimates = plan_finite ∧ (observed free ∨ U). Ticks = BFS length / 3.
* **L2 candidates**: skeleton points of backbone regions thinned to 1.5 m + region centroids + WFD
  frontiers whose region (±0.4 m) is backbone; snapped to traversable (≤ 0.4 m), reachable, ≥ 1.2 m
  from the robot (native unlocks at < 1 m), not within 0.3 m of a rejected goal. ≤ 300 kept (nearest).
* **Phase rule** (per decision, not latched): L2 iff |U_bb| ≥ 5 % of backbone free area AND some
  candidate sees ≥ 20 U_bb cells; else L3. If L2 yields no tour → L3 is tried; if L3 is empty → no goal
  (native fallback to MapEx, counted).
* **L2 set cover**: lazy-greedy (CELF) on new U_bb cells / (ticks from robot + 1); stop at < 5 % of U_bb
  uncovered, best new gain < 20 or 20 stops. g_i = marginal cells of stop i (fixed after cover).
* **L2 order**: greedy insertion maximising J, then 2-opt + Or-opt until no gain. t_i = tick + cumulative ticks.
* **Pocket insertion (L2)**: the 8 pockets with best gain/ticks; each pocket's best single viewpoint
  (max cells of that pocket seen); insert best ΔJ/Δticks while it exceeds λ = J/L of the current tour
  (λ recomputed after each insertion, ≤ 6 insertions), then 2-opt/Or-opt again.
* **L3 pockets**: per pocket, candidates = its skeleton (1.5 m), centroid, frontiers in it, its doors
  (≤ 60 kept: doors first, then nearest). Greedy max-coverage set cover (≤ 4 viewpoints, stop at 90 %
  or new < 20). g_p = Σ_{covered} (1 − var_norm), service s_p = NN round trip from the entry viewpoint
  (nearest selected) × 1.2 Euclid / 3; gain time = arrival + s_p/2.
* **L3 orienteering**: ≤ 12 pockets → exact Held–Karp; > 12 → greedy insertion + 2-opt/Or-opt +
  removal + re-insertion. Budget = T − tick (hard). Goal list = entry of first pocket, then its other
  viewpoints, then later pockets (receding horizon).
* The planner returns an ordered goal list; the native hook takes the first goal that is finite on the
  native grid, ≥ 1 m from the robot and has a native A* path (others are added to the exclusion list).

## Deviations from the spec (and why)
1. **Raycast**: not a call to `simple_mask_utils.get_free_points` but `visibility.free_points_binary`, a
   numba port of exactly that function (binary mode) without per-ray list building; **bit-identical**
   on 60 random grids (tests/test_visibility.py), ~10× faster (10 ms → ~1 ms). The visible set is the
   ray end-point polygon (cv2.fillPoly) ∪ ray cells, same semantics as MapEx `get_vis_mask`'s
   polygon flood fill, but on a local window. PAPER patch irrelevant (binary mode).
2. **Travel time**: BFS shortest 4-connected length instead of native weighted A* (lower bound of the
   native path; A* is still run once on the chosen goal).
3. **Set-cover cost** = ticks from the robot (not tour-incremental); gains g_i fixed to the greedy
   marginal assignment when ordering (no re-evaluation of overlap per order).
4. **λ** interpreted as J/L (objective per tour tick) of the current tour, compared with ΔJ/Δticks of
   the insertion (both in cells·ticks per tick). Pocket gain in L2 = cells seen from its best viewpoint.
5. **var_norm** on a fixed scale: var·3 clipped to [0, 1] (unbiased variance of 3 values in [0, 1] is
   ≤ 1/3). Per-decision max normalisation was rejected: a uniform variance would zero all gains.
6. **Held–Karp** solves Σ g(T − t) exactly through a backward subset DP on
   C(R, j) = min_k D[j,k]·W(R) + s_k·W(R∖k) + C(R∖k, k) with (C, duration) Pareto fronts (K = 32) for
   the hard budget. Exact vs brute force on 120 random instances n ≤ 7 (with/without budget/service);
   0 front overflow at n = 12 in tests; ~0.45 s at n = 12. Overflow count is logged.
7. **Phase not latched** (re-evaluated every decision) since L1 regions/backbone change over time.
8. **Pocket service time** uses Euclid × 1.2 inside the pocket (rooms are near-convex) to avoid one BFS
   per pocket viewpoint.
9. Candidate caps (300 L2, 60 per pocket, 8 pockets for L2 insertion, 20 for L3) for the ≤ 5 s target.
10. **Native replan trigger** (`native/topo_policy.should_replan`, 2 added lines in the native loop):
   needed because goals may lie in UNKNOWN space where native A* keeps them valid while the real path
   grows (seen in the smoke). Receding horizon is otherwise only "new decision when the native loop
   asks". Not in the spec → needs coordinator review.

## Timing (offline, KTH49 P00 baseline inputs, 20 decisions, real topo_l1 v0 snapshot)
planner median 1.30 s, p90 1.57 s, max 1.67 s; L1 2.0–2.8 s. With the coarse stub (whole free space =
backbone, worst case for L2): median 1.52 s, max 2.19 s.

## Tests
`tests/test_tour.py` (HK vs brute force n ≤ 7, budget, greedy path, local search),
`tests/test_visibility.py` (raycast parity), `tests/test_planner.py` (synthetic corridor + 2×4/2×5/2×8
rooms: L2 heads along corridor, L3 when backbone known with HK, greedy > 12, budget, λ insertion,
exclusion, goal validity, empty case, walls block visibility). Run:
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=/home/dell/DELL_NATIVE2D/vendor
/home/dell/miniforge3/envs/lama/bin/python -m unittest discover -s topo_planner/tests`

## Observations for the coordinator
* With topo_l1 v0 (spec params) LaMa predicts most far-unknown as free → one giant pocket/region; the
  first L3 goals can lie deep in unknown space. topo-l1 is addressing this (v1 PRED_RANGE_M).
* The planner reads `pocket_ids` and treats every other region as backbone; L1's extra `backbone_ids`
  key is not needed.
