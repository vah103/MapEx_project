# KTH-001 results — Claude scientific review

- Date: 2026-10-20 ICT
- Inputs: runs synced to DELL `/home/dell/DELL_NATIVE2D/runs/KTH-*` (-05 and -06 retries). Claude audit (read-only): thesis repo `data/ch5/kth001_audit.py` → `kth001_audit.json` (coverage = known reachable-free cells of p1_core.npy, from global_obs PNGs).
- Verdict: results ACCEPTED as descriptive; **KTH-001 does not test G1** on 2 of 3 maps, and it exposed a **G1 bug** (stale LOCAL planning grid).

## Numbers (P00, paper raycast, COM1)
| Map | MX | G1 | G1+SW | G1+SW+ED | Note |
|---|---|---|---|---|---|
| 50052751 | 530.3 m, cov 1.000 | **hit_wall** at 127.7 m (step 442; -06 identical) | 519.1 m | 504.1 m | arms diverge at tick 418 (first LOCAL); LOCAL only 7.4–14.5 m |
| 50052749 | 287.4 m | 287.4 | 287.4 | 287.4 | **odometry identical in all 4 arms** (994 steps): 0 LOCAL decisions |
| 50010535_PLAN1 | 306.0 m, cov 0.9992 | same | same | same | all 4 identical; MapEx core stops with `frontier_region_centers` at step 1056 when every remaining frontier centre is invalid — coverage 99.92 %, effectively terminal, not an algorithm failure |

50052751 distance to 95 / 99 %: MX 243.3 / 391.5 m; SW 229.4 / 357.3; ED 229.4 / 359.3. Coverage at step 1000: MX 0.978, SW 0.982, ED 0.982. MapEx's tail is large here too (95 % → end ≈ 287 m, 54 % of its travel) — but G1 barely acts, so the SW/ED savings (−11 / −26 m) come from a different tour after one LOCAL epoch, single start: no claim.
COM1 compute per decision (median): LaMa 5.4–6.7 s (50052751), 16.3–17.5 s (50052749), 38.5–40.4 s (large); Room Node + scoring 2.0–2.8 s.

## Why G1 does not engage on KTH
At every decision on 50052749 the robot stands on Room Node label 2 (corridor/hall); Room Node does label 300–980 m² as ROOM in 13–30 components, but with a 20 m, 2,500-ray LiDAR the small offices along KTH corridors are observed through their doors, so MapEx never sends the robot into them and the G1 trigger ("robot inside a confirmed room at a decision") never fires. On 50052751 the one LOCAL room is 622 m² (20 × 78 m) — a building half, not a room. Conclusion: G1's benefit depends on rooms that must be entered (furniture/occlusion, rooms deeper than what is visible through the door) — true for New Room and Hospital, not for these KTH office floors with a 20 m sensor.

## G1 bug found (affects all G1-family arms)
`room_native_integration.py`: in LOCAL the A* grid (`scope_grid = plan_grid`) is built once at the decision and reused every tick until the next decision; MapEx rebuilds its planning grid every tick. Walls observed after the decision are not in the LOCAL grid. KTH-50052751 G1: LOCAL goal 72 m away inside the 622 m² "room"; path crossed cells that were unknown at tick 417 and observed as wall by tick 442 → `hit_wall` (core [186,751]; wall observed 255 in the last global_obs). SW/ED did not hit it only because their route differed.
Fix (task G1-005): rebuild the LOCAL grid every tick as the current inflated planning grid with cells outside the frozen room mask set to inf. Must be checked on New Room: replay NAT-001 G1-family runs and test whether any executed LOCAL path ever crossed a cell that was inf in the fresh grid at that tick (if never, NAT-001 results are unaffected).

## Next
1. G1-005 fix + NAT-001 replay check (offline).
2. KTH is not a fair G1 test with 20 m LiDAR; options for the user: (a) report KTH as a "G1 inactive → no regression" result (G1 never worse than MapEx where rooms need no entry); (b) rerun KTH with the TurtleBot-like 12 m SLAM range used in ROS2 (protocol change, needs GO); (c) pick denser KTH plans. Hospital (paper raycast) remains the stronger second scene.

## Correction (Claude, 2026-10-20 evening, re-check of KTH-50052749-P00-G1-05 decision JSON + semantic npz)
"At every decision on 50052749 the robot stands on label 2" is wrong. Robot was inside a confirmed Room-Node ROOM (label 1) at **8 of 16** decisions (D02, D06, D09, D11, D14–D16); every time G1 evaluated it and found `LOCAL_WFD_EXHAUSTED_RETURN_GLOBAL` (no reachable in-room frontier on arrival). Rooms 3–5 have 72–74 k seen free cells (≈ 720 m²), i.e. building halves, not offices. So G1 does engage its check on KTH; it never acts because the residual frontier pockets (17–278 new cells per late leg, after 95 % at 121 m) are not left inside the room the robot is in — they are scattered and cleared on arrival. The MapEx tail on KTH (40–58 % of travel after 95 %, all 3 maps) is therefore a global-ordering problem that G1's finish-the-room rule does not address. Hypothesis (not verified): pockets were already frontier when the robot passed nearby but lost to higher-IG distant frontiers.
