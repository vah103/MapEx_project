# TOPO-002 native integration (WP-D, topo-planner session)

Copied from `build/topo001/native` (TOPO-001 code there is untouched). Changes for TOPO-002:
* `topo_policy.py`: planner = `topo_planner.UnifiedPlanner` (R2, see `../planner/DESIGN_NOTES_R2.md`); p(c) loaded
  from `<run>/pc_model/pc_model.py` (`TOPO_PC=stub` → `pc_stub` 1[mean<0.5]) and computed once per decision,
  forwarded to L1 (`pfree=`, only if its `build_topo` accepts it) and to the planner; L1 preset `TOPO_L1_PRESET`
  (default `v2` if the module has it); stats add `pc_source`, `pc_fallback_used`, `first_stop_on_corridor_chain`,
  `decisions_with_stops`, `min_dist_relaxed`; phase is logged as `R2`. Replan trigger and MapEx fallback unchanged.
* `integrate_topo.py`: unchanged (TOPO script SHA `445608a1…`).
* `prepare_run_topo.py`: run IDs `TOPO2-<MAP>-P0x-<tag>-01`; `--l1-dir/--l1-preset`, optional `--pc-dir` (else stub);
  accepts topo2-data DEV manifests (`map_path`, `core_shape_rc`, start pose parsed from the worker); any
  `opts.mission_time=<int>` in the baseline worker.
* `launch_topo.py`: receipt must say RD-047 / TOPO-002; shared lock `build/topo001/native/TOPO_ONE_WORKER.lock` (same file as topo2-data; their queue also honours `analysis/topo2_data_20261103/PAUSE`) + /proc check
  for other `dell_worker.py`; no RAM threshold (MemAvailable + peak RSS recorded); passes `TOPO_L1_PRESET`, `TOPO_PC`.
* tests: native replica loop with fake L1, real L1 v1 (topo001) and real L1 v2 (topo002, in development).

---- TOPO-001 README (historical) ----

# TOPO-001 native integration (topo-planner session)

Base: MX1000 harness **R4** (`r4_base/`, extracted unchanged from
`staging/MX1000_HARNESS_R4_20261102_PRECOMMIT.zip`; R4 integrated script SHA
`0cac1c42…5026`, whose passthrough was bit-exact with the KTH49 baseline).

## Files
| file | role |
|---|---|
| `integrate_topo.py` | derives `explore_mapex_topo_r1.py` from the SHA-pinned R4 script: (1) import `topo_policy` instead of `mx1000_policy`; (2) the 2 goal locks `frontier_region_centers[chosen_index]` → `mx_policy.goal(frontier_region_centers, chosen_index)`; (3) **added after the smoke**: 2 lines after the native validity check, `if not need_new_locked_frontier and mx_policy.should_replan(...): need_new_locked_frontier = True`. Diff = 5 lines. Current TOPO script SHA `445608a1…2c08` (the smoke ran the earlier `096aa1c7…964d`, edits 1–2 only). |
| `topo_policy.py` | `PolicyHarness` with the R4 hook API (`choose`, `reject_last`, `track_stale`) + `goal()`. Policy `topo` (default) or `mapex` (env `TOPO_POLICY`). |
| `prepare_run_topo.py` | builds an UNLAUNCHED run dir from a baseline run (worker etc. copied, prepared worker forced to `opts.mission_time=<ticks>` with the R4 one-assignment check, TOPO script, `topo_policy.py`, frozen snapshots of `topo_planner/` and `topo_l1/`, manifest with all hashes). Refuses existing destinations; run-id pattern `TOPO1-<MAP>V3-P0x-<tag>-01`; smoke ≤ 150 ticks, eval = 1000. |
| `launch_topo.py` | launches one prepared run. Guards: RD-046 receipt JSON (`authorization`, `task`, `run_ids` containing the run, matching `kind`), no other `dell_worker.py` process (scan of /proc) + flock `TOPO_ONE_WORKER.lock`, no RAM threshold (user decision 2026-11-03: MemAvailable at start and worker peak RSS recorded in RUN_RECEIPT.json), never launched before (no output, no RUN_RECEIPT). No retry. Sets `TOPO_PLANNER_T=1000` so smoke runs keep the T = 1000 objective. |
| `receipts/` | RD-046 receipts (smoke receipt transcribes the coordinator's task instruction) and launcher output. |
| `tests/test_native_hook.py` | replica of the native goal loop with MapEx `Mapper`/`FrontierPlanner`/`psuedo_traj_controller` on a synthetic padded world. |

## Hook behaviour (`topo_policy.PolicyHarness`)
* `choose(reset=True)` (new decision): crop to the CORE frame (`core_inputs`, = logger npz crop),
  `topo_l1.build_topo(obs, mean, var, robot_core, ppm, prev=state)` (loaded from
  `<run>/topo_l1/topo_l1.py`), `TopoPlanner.plan(...)`, then the first planner goal that is finite on the
  native grid, ≥ 1 m from the robot and has a native A* path. The goal may be **any reachable cell**.
  Returns (any valid index, costs unchanged); `goal()` returns the planner goal.
* **Fallback** (planner exception, init/L1 error, no valid goal, policy `mapex`): exactly the MapEx
  choice — lowest (cost, index) among frontiers with Euclid ≥ 1 m, finite grid, A* path ≥ 1 m (R4 rule
  minus the GT evaluation); forced cost as in R4. Counted per reason in `topo_policy_stats.json`.
* **Reselect** (native A* returned None): `reject_last()` excludes the topo goal (or the frontier index
  on fallback); `choose(reset=False)` calls the planner again (L1 result cached for the same tick) with
  the exclusion list.
* **Native lock loop with non-frontier goals** (checked in code and tests): the script unlocks when
  ‖goal − pose‖ < 1 m and re-decides when the goal cell becomes `inf` on the inflated grid
  (`is_locked_frontier_center_valid`); neither check depends on the goal being a frontier. The planner
  only proposes goals ≥ 1.2 m away to avoid an immediate unlock.
* **Replan trigger (new after smoke, needs coordinator review)**: `should_replan` asks the native loop for
  a new decision while a TOPO goal is still natively valid, ≥ 5 ticks after the decision, if
  (a) < 20 % of the cells the goal was chosen to observe are still UNKNOWN (`targets_observed`), or
  (b) the native A* path (checked every 3 ticks) exceeds 1.5 × its length at decision + 5 m
  (`path_blowup`; that goal is excluded at the next decision). Always False for MapEx/fallback goals,
  so fallback behaviour stays plain MapEx. Counted in `replan_triggers`, logged as `event` rows.
* **No ground truth**: only `mapper.obs_map` is read (test `test_mapex_policy_no_gt_access` wraps the
  mapper to raise on any other attribute). The R4 GT `view_leg`/`view_gain` (~95–220 s/decision) is gone.
* Logs in the experiment dir: `topo_decisions.jsonl` (per call: tick, decision, reset, phase, goal
  rc/core rc/kind, planner log with tour stops, gains, predicted ticks, J, λ, pocket insertions, solver
  info, exclusions, A* validation tries, fallback reason/traceback, timing crop/L1/planner/A*/png),
  `topo_policy_stats.json`, `topo_debug/decision_NNNN_tTTTT.png`.

## Tests
`/home/dell/miniforge3/envs/lama/bin/python -W ignore -m unittest discover -s native/tests`
with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=/home/dell/DELL_NATIVE2D/vendor`:
crop parity with the logger, 220-tick replica loop (L2 and L3 goals, non-frontier goals, unlocks,
no wall hit, PNGs), goal→inf replan, reselect re-plans, exception → MapEx fallback, no GT access.

## Running (after coordinator approval only)
```
python3 native/prepare_run_topo.py --baseline runs/<BASELINE> --destination runs/<RUN_ID> \
    --run-id <RUN_ID> --ticks 1000 --kind eval
python3 native/launch_topo.py --run-dir runs/<RUN_ID> --receipt <coordinator RD-046 receipt>
```

## Smoke TOPO1-KTH49V3-P00-SMOKE-01 (150 ticks, KTH49 P00, 2026-11-03)
Prepared from `runs/KTH49V3-P00-MX-T1000-01`, receipt `receipts/RD046_SMOKE_…json`, launched with
`launch_topo.py` (no other worker, 4.25 GiB free). Exit 0, 150/150 ticks, no exception, no wall hit.
Code: TOPO script `096aa1c7…` (no replan trigger), topo_l1 v0 snapshot `topo_l1.py 68279946…`.
* decisions 2 (t0, t37), both L3 (`pocket_entry`), **fallbacks 0**, reselects 0, init_error none.
* timing: decision 1 (cold: numba compile) L1 4.49 s, planner 6.84 s; decision 2 L1 2.79 s, planner
  0.26 s, A* validation 0.01 s, PNG 0.03 s. Sum of step wall times 193 s; LaMa cold forecast 49 s.
* behaviour: L1 v0 gave 2 regions, 0 backbone, 1 pocket of 90k unknown cells (LaMa predicts far
  unknown as free — the issue topo-l1 is fixing with v1). The t37 goal (core 186,70) lies next to an
  unseen wall; A* through UNKNOWN kept it valid, so the robot pursued it 113 ticks (path 12 → 38 m)
  without a new decision → motivated the replan trigger above (tested offline only).
* debug PNGs: `runs/TOPO1-KTH49V3-P00-SMOKE-01/native_plain_lama3small_run05/MAPEX_PLAIN_LAMA3SMALL_RUN05_visvarprob/topo_debug/`.

## Smoke TOPO1-KTH49V3-P00-SMOKE-02 (150 ticks, KTH49 P00, L1 preset v1, replan trigger)
ID -02 because -01 exists (no overwrite/retry). Code: TOPO script `445608a1…`, topo_policy `b1a7ba58…`,
planner.py `888b1eef…`, topo_l1.py `d7088557…` (preset v1). Exit 0, 150/150 ticks, no traceback, no wall hit.
* 8 decisions: 7 L2 (1 pocket_insert, 6 backbone_skeleton) + 1 L3 (pocket_entry); **fallbacks 0**,
  reselects 0, replan triggers 1 (`targets_observed` at t140, 18 % of targets still unknown).
* timing per decision: L1 0.8–1.14 s; planner 0.33–0.58 s (2.1–2.4 s on the first two, numba compile);
  A* validation ≤ 0.01 s; PNG ≤ 0.04 s. Run wall 6.8 min (sum of steps 391 s). Start MemAvailable 4.60 GiB,
  worker peak RSS 2.28 GiB.
* path 44.7 m in 150 ticks. Offline (GT, descriptive, n = 1): GT-free observed at t50/100/149 =
  5.78 / 9.46 / 14.71 % vs MapEx baseline 5.46 / 8.62 / 9.13 % (smoke-01 with L1 v0: 5.38 / 6.00 / 8.82 %).
* debug PNGs: `runs/TOPO1-KTH49V3-P00-SMOKE-02/native_plain_lama3small_run05/MAPEX_PLAIN_LAMA3SMALL_RUN05_visvarprob/topo_debug/`.
