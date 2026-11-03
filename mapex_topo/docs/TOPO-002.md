# TOPO-002 — MapEx-TOPO R2: unified expected-gain orienteering + structural L1 v2 + clean dev/test split

Status: IN_PROGRESS (2026-11-03). Method owner: Claude coordinator. Authorization: RD-047. Predecessor: TOPO-001
(wave 1, n = 1: ΔAUC KTH49 P00 +0.155, KTH50 P00 −0.060, KTH51 P01 −0.021; weakness = hard phase rules and gain
restricted to L1 regions). Same native simulator, metric and budget: AUC of GT-FREE coverage over ticks 0..999,
`v0_metrics.coverage_distance`; online only (no GT at run time).

## 1. Data split (binding — decided before any R2 result)
| Set | Maps (`~/MapEx/kth_test_maps`) | Use |
|---|---|---|
| **DEV** (3 buildings, 7 maps) | 50010535_PLAN1, 50010535_PLAN2, 50010536_PLAN3, 50015847, 50015848, 50037764_PLAN1, 50037765_PLAN3 | develop L1 v2, calibrate p(c), planner sanity; anything goes |
| **TEST-IN** (2 KTH buildings) | 50052749 (KTH49) P00/P01/P03, 50052750 (KTH50) P00/P01/P03, 50052751 (KTH51) P00–P03, + 50052748, 50052752, 50052753, 50052754 (P00 each) | final evaluation only; no tuning; frames of these maps must not be read by R2 development code |
| **TEST-OUT** (non-KTH) | New Room P00, Hospital P00 (native maps of the thesis) | cross-domain check |
Wave-1 configs (KTH49 P00, KTH50 P00, KTH51 P01) were used for diagnosis → reported with a flag.
Start poses: KTH maps = first pose for that world in `~/MapEx/clusters/run_explore.job` (paper poses), mapped to the
native core frame; existing KTH49/50/51 poses unchanged. New Room / Hospital = the existing P00 starts.

## 2. Architecture R2 (frozen)
MapEx native loop unchanged (LaMa 3-model, WFD, A* 4-conn, 3 cells/tick, LiDAR, 1 m unlock). Goal selection = one
receding-horizon optimisation; **no L2/L3 phases**.

### 2.1 Expected gain model p(c) (new)
p(c) = P(cell c is GT-FREE | UNKNOWN now; features), features: LaMa mean, var, geodesic/Euclid distance to nearest
observed FREE, distance to nearest observed OCC, (optional) L1 region class. Model: logistic regression or isotonic
binning (≤ 50 parameters), fit on DEV frames only; report reliability diagram, Brier, AUROC on DEV (building-wise
CV) and one-shot on TEST-IN frames. Gain of a viewpoint v at a decision = Σ_{c ∈ Vis(v)} p(c) over UNKNOWN cells,
Vis = MapEx-semantics raycast (250 rays, 20 m) on a predicted occupancy where a cell blocks rays with probability
1 − p(c) (deterministic: blocks if p(c) < 0.5 — same rule for all candidates).

### 2.2 L1 v2 (structural role only)
Produces candidate viewpoints and clusters; makes **no** strategic decision. Changes vs v1:
door evidence from wall **jambs** (pairs of wall ends facing each other 0.7–2.4 m apart, observed or predicted
mean ≥ 0.7) in addition to clearance minima; predicted-wall gap closing (mean ≥ 0.7, gaps ≤ 0.6 m);
corridor = skeleton **chains** ≥ 6 m with ≥ 2 door branches (not region features); PRED_RANGE removed or replaced by p(c).
Interface = TOPO-001 TopoMap + `chains` (list of skeleton polylines with class) + `viewpoints` (rc, kind).

### 2.3 Unified planner
- Candidates: (a) skeleton/chain points every 1.5 m, (b) door points, (c) pocket interior viewpoints (set cover inside
  each L1 region, ≤ 4), (d) **rim viewpoints** on the boundary of large UNKNOWN areas with high Σp (fixes KTH50),
  (e) WFD frontiers. All ≥ 3 m from the robot unless nothing else; reachable on the native planning grid.
- Objective: max J = Σ_i g_i (T − t_i) with g_i = marginal expected gain (no double counting), t_i from travel ticks
  (BFS/A*/3) + 0 service; hard budget T − tick.
- Solver: lazy-greedy selection by ΔJ/Δt, then insertion + 2-opt/Or-opt; exact Held–Karp re-ordering of the first
  ≤ 10 stops; receding horizon (execute first stop). Log J, λ = J/L, stops, solver stats.
- Kept technical rules: replan when < 20 % of targeted cells remain UNKNOWN or A* path > 1.5× + 5 m; fallback = MapEx.
- "Backbone first" is **not** coded; report whether it emerges (share of early stops on corridor chains).

## 3. Work packages
| WP | Owner session | Output |
|---|---|---|
| A — maps, starts, DEV frames, new baselines | **topo2-data** (new) | v3 conversion of DEV + 50052748/52/53/54 with overlay acceptance (memory lesson: check building kept, doors open); start poses; DEV frames = native plain MapEx runs ≤ 400 ticks, 1 per DEV map (decision npz logged); MapEx T1000 baselines on 50052748/52/53/54 P00, New Room P00, Hospital P00 (PAPER raycast, same code as KTH baselines) |
| B — L1 v2 + p(c) | **topo-l1** (continue) | `topo_l1` v2, `pc_model` + reports on DEV; one-shot TEST-IN report after freeze |
| D — unified planner + integration | **topo-planner** (continue) | planner R2 on the TOPO-001 native hook, unit tests, smoke on a DEV map |
| Review, gates, RD | coordinator | |

## 4. Gates
1. A: map acceptance (overlay, building fraction, connectivity) before any run on a new map.
2. B: p(c) reliability on DEV (building-wise CV): ECE ≤ 0.05, beats raw `mean<0.5` in Brier; L1 v2 door recall ≥ 0.75,
   precision ≥ 0.9 on DEV. Then freeze, one-shot on TEST-IN frames (reported, not tuned).
3. D: unit tests; smoke 150 ticks on a DEV map, 0 exceptions; timing ≤ 6 s/decision hook.
4. Code freeze (SHA) → coordinator review → **final wave**: TOPO2 on all TEST-IN (14 configs) + TEST-OUT (2) =
   16 runs × 1000 ticks, sequential, compared with MapEx baselines; no code change during the wave.

## 5. Rules
One native worker at a time (topo2-data and topo-planner coordinate via `pgrep -af dell_worker` + TOPO lock);
no RAM threshold (user); no retry; run IDs `TOPO2-<MAP>-P0x-<tag>-01`, `MX-<MAP>-P00-T1000-01`; baselines read-only.
n = 1 per config, descriptive; report per map and per building.

## Amendment 2026-11-03 (user)
User: test only on the 3 wave-1 configs. Final wave = 3 runs (TOPO2 on KTH49 P00, KTH50 P00, KTH51 P01) vs existing
MapEx baselines and vs TOPO-001 R1. New TEST maps (50052748/52/53/54), TEST-OUT and their 6 baselines are dropped.
DEV set and DEV-only development unchanged. Caveat to report: these 3 configs were used to diagnose R1, so the
result is a development comparison (R1 → R2 on the same configs), not a held-out test.
- 2026-11-03 (Claude): approved deviation — L1 v2 builds structure with the v1 3 m prediction range (pfree as free space was
  ~15× slower and worse corridor labels); p(c) is used only for planner gain. Preliminary DEV (3 maps, 30 frames): L1 v2
  door recall 0.83 / precision 0.995; p(c) logistic (19 params) Brier 0.092 vs 0.166 for mean<0.5, AUROC 0.85 vs 0.60,
  ECE 0.022 overall, 0.056 per building (near-pass, not tuned further). Finding: on UNKNOWN cells LaMa mean ≈ 0 almost
  everywhere; signal is distance to predicted walls.
- 2026-11-03 (Claude): WP-B frozen (build/topo002/eval_dev/REPORT.md, FREEZE_SHA256.txt). Gate B on DEV **FAIL**: p(c)
  building-wise ECE 0.163 (3rd building 5003776 base rate 0.31–0.57 vs 0.75–0.84), L1 v2 door recall 0.709. One-shot
  TEST (3 eval configs' frames): p(c) Brier 0.166 / ECE 0.055 / AUROC 0.80 vs mean<0.5 0.358 / 0.358 / 0.60; L1 v2 door
  recall/precision 0.64/0.97 (≈ v1 on the same frames). Decision: proceed with the frozen models (no post-hoc tuning);
  planner uses p(c) for ranking gains. Report these gate failures with the final wave.
- 2026-11-03 (Claude): final 3-config wave verified independently (global_obs PNG + GT, 10-tick AUC; all runs 1000 ticks,
  3 cells/tick, no GT refs in frozen code). AUC MapEx | R1 | R2: KTH49 P00 0.294 | 0.449 | 0.436 (cov@999 0.491 | 0.807 |
  0.869); KTH50 P00 0.325 | 0.265 | 0.466 (0.690 | 0.462 | 0.757); KTH51 P01 0.739 | 0.718 | 0.758. R2 > MapEx on 3/3
  (n = 1, configs used for R1 diagnosis). Backbone-first emerges partly (first stop on corridor chain 49 % KTH49, 8 % KTH50,
  28 % KTH51). Held–Karp prefix improved 0/99 decisions. Predicted gains poorly calibrated both ways.
