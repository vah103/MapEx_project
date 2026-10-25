# New Room native 2D — current G1 R2 + existing EARLYDOOR-R1

**Date** 2026-10-20. **Status** COMPLETED TECHNICAL / SINGLE-START SCIENTIFIC RESULT / MULTI-MAP GENERALIZATION NOT ACCEPTED.

## Scientific question and frozen method

The user requested a clean comparison using **G1-001 R2, the only current accepted G1**, not the historical V4-R4.1 G1. The new treatment changes one factor relative to `NR-RAYSW-03`: import the **existing unmodified** `g2_early_global_door_r1.py` and add its original portal-anchored prospective external GLOBAL WFD reservation/release into the current G1 R2 + SWEEP source. During each LOCAL decision it evaluates same-epoch full-map frontier IG and observed-FREE passable room boundary portals; on the first subsequent GLOBAL decision it uses the reserved goal only if still a current native-reachable WFD, otherwise falls back to original MapEx. It does not change G1 room inference/frontier eligibility, SWEEP LOCAL goal selection, original scorer, LaMa, native A*, GT visibility, map or start pose.

The exact code diff is reversibly patched from `NR-RAYSW-03` G1 policy, original SHA `b41dccf02a27b2407516ce4273c4309e3fa63441d970464d4920cd1603fc3877`. New integration SHA `af3a8acd2768b19bdfd27bfe2ad76d888fe0a1558aaf308cc5fe30de1a5afa7f`; original EARLYDOOR helper SHA `0a1e45120c010e38b09fbc23b7d28cd6e5798c6825ed4b4350608364a25a084f`. Original SWEEP helper, worker, patch, model checkpoints, start [631,631], paper-cumulative MapEx raycast, New Room P1 203x263 at 0.10 m are frozen and unchanged.

## Execution quality assurance

- Successful full treatment **`NR-RAYEDR2-03`**, registered short alias **NR45**. `RUN_RECEIPT.json`: `SOURCE_TERMINAL`, returncode 0, 468.379 seconds DELL **wall compute/runtime** (not navigation time on COM1), peak process PSS 2139.1 MiB, lowest system MemAvailable 3077.6 MiB; 718 native ticks, 30 decisions.
- Hash/preflight PASS, four original EARLYDOOR unit scenarios PASS, 3-model LaMa `MODEL_SMOKE_PASS`.
- D01–D02 `D01_D02_VARIANT_INTEGRITY_PASS`, **PLUS** independently verified bit-identical D01/D02 observed/mean/variance tensors, pose, GLOBAL/LOCAL mode, exact two executed goals relative to `NR-RAYSW-03`.
- Previously attempted treatment **`NR-RAYEDR2-02` (NR44) aborted** by unchanged RAM guard, did not complete; not ranked or counted. Failure was concurrent high-RAM thesis analysis process, now finished. Rerun NR45 code SHA is byte-identical to NR44.
- `SOURCE_TERMINAL` means the native MapEx algorithm has no selectable frontier, **not proof of full GT map coverage**.

## Four-way same P1 start, corrected paper raycast

| Config / frozen run ID | Total m | LOCAL m | GLOBAL m | Observed known P1 | Decisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain MapEx `NR-RAYMX-02` | 322.323991 | 0 | 322.323991 | 51,304 | 22 |
| Current G1 R2 `NR-RAYRN-03` | 228.850539 | 81.089859 | 147.760680 | 51,305 | 23 |
| Current G1 R2+SWEEP `NR-RAYSW-03` | 233.132505 | 87.853791 | 145.278714 | 51,303 | 30 |
| **Current G1 R2+SWEEP+EARLYDOOR `NR-RAYEDR2-03`** | **202.718728** | **91.219039** | **111.499689** | **51,303** | **30** |

Relative to same-source G1 R2+SWEEP treatment control: final distance **−30.413777 m (−13.05%)**, GLOBAL distance **−33.779024 m**, LOCAL distance **+3.365248 m**. Both final observed known-P1 core-cell binary masks are **bit-identical** (A-only 0, B-only 0, XOR 0). The final G1 R2 vs treatment maps differ by eight P1 cells (G1-only 5, treatment-only 3) and cannot be called spatially equal.

### Early and late matched observed-cell COUNT — distance to first attainment

| Known P1 target | Plain MapEx | G1 R2 | SWEEP | EARLYDOOR |
|---|---:|---:|---:|---:|
| 48,051 (90% of core) | 143.020355 m | 155.270199 m | 152.668884 m | **182.122986 m** |
| 50,720 (95%) | 174.257738 m | 205.696437 m | 203.836762 m | **190.511840 m** |
| 51,254 (96%) | 261.850539 m | 226.832505 m | 230.814471 m | **199.047908 m** |
| 51,303 (minimum terminal count among four) | 322.323991 m | 228.103326 m | 232.085292 m | **200.471515 m** |

At the common 51,303 **count**, EARLYDOOR vs SWEEP saves **31.613777 m**. Counts alone do not guarantee equal spatial masks at the instant of crossing this target. Critically, EARLYDOOR is worse than SWEEP at 90% by **29.454 m**; its advantage comes late, and it is not uniformly superior over the full curve.

## Measured intervention

The saved `worker.stdout.log` records **23 LOCAL EARLYDOOR updates and five LOCAL→GLOBAL releases**. Only **one** release changes the GLOBAL WFD index, at native **tick 147** (native MapEx index 11, EARLYDOOR valid reservation index 2). At tick 84 the stale reservation falls back, and the other three releases retain the original GLOBAL winner. Trajectory divergence after tick 147 changes downstream maps, rankings and travel; do not attribute −30.41 m mechanistically to the isolated segment at 147 without further causal replay. LOCAL+GLOBAL changes are measured at full-path level.

## Evidence

- Frozen successful run: `/home/dell/DELL_NATIVE2D/runs/NR-RAYEDR2-03/` (all scanned maps, per-decision inferences and goal lists, odometry, manifest, code, three-model smoke and native receipts).
- Prior resource aborted run: `/home/dell/DELL_NATIVE2D/runs/NR-RAYEDR2-02/`, labeled incomplete.
- Original scope+SWEEP comparator: `/home/dell/DELL_NATIVE2D/runs/NR-RAYSW-03/`; current G1 R2: `NR-RAYRN-03`; plain MapEx: `NR-RAYMX-02`.
- Immutable precommit: `G1_R2_EARLYDOOR_R1_RETRY_PRECOMMIT_20261020.json` in this report's directory.
- Independent read-only evaluator: `audit_g1r2_earlydoor_retry_4way_20261020.py`.
- Machine comparison: `G1_R2_EARLYDOOR_R1_4WAY_RETRY_COMPARISON_20261020.json`.
- Full 83-decision scroll gallery (23 G1 R2, 30 SWEEP, 30 EARLYDOOR, 249 image panes): `G1_R2_SWEEP_EARLYDOOR_RETRY_3WAY_FULL_DECISIONS_SCROLL.html`.
- DELL-only localhost browser URL while server exists: http://127.0.0.1:18763/G1_R2_SWEEP_EARLYDOOR_RETRY_3WAY_FULL_DECISIONS_SCROLL.html

## Interpretation/next checks

**Technically reproducible single-start New Room case.** This is promising G2 *late travel* evidence, not multi-start or independent Hospital validation. Comparative scientific claims for thesis require preregistered multi-start and cross-map trials, matched coverage curves, obstacle/unknown navigation semantics, and ROS2/Gazebo evaluation. DELL wall time is not a COM1 robot-time comparison.

No changes to original MapEx or frozen comparator runs; no Overleaf/GitHub uploads, no changes to accepted G1 version.
