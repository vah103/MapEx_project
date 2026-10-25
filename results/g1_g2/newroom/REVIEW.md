# NAT-001 results — Claude scientific review

- Date: 2026-10-20 ICT
- Inputs: 32 runs (P00 reuse RAYMX-02 / RAYRN-03 / RAYSW-03 / RAYEDR2-03; P01–P07 -03; infra retries G1-P01-04, ED-P06-04 — first attempts aborted on BadZipFile in the guard at ticks 15/20, retried per precommit). P00 determinism exact PASS.
- Claude audit scripts (read-only), thesis repo `data/ch5/`: `nat001_audit.py` (coverage from global_obs, every arm), `nat001_behaviour.py`, `nat001_rooms.py`; outputs `nat001_audit.json`, `nat001_behaviour.json`, `nat001_rooms.json`. Totals match ChatGPT's NAT001_R2_RESULTS_4X8.json.
- Verdict: **ACCEPT results.** G1 claim supported; G2 (SWEEP, EARLYDOOR) claim NOT supported.

## Main numbers (8 starts, New Room, paper raycast)
| Arm | mean ± sd total m | median | min–max |
|---|---|---|---|
| MapEx | 305.8 ± 38.2 | 308.2 | 256.1–376.1 |
| G1-R2 | 220.1 ± 12.6 | 221.7 | 201.9–237.7 |
| G1+SWEEP | 220.6 ± 17.8 | 229.1 | 188.1–236.9 |
| G1+SWEEP+ED | 220.2 ± 22.0 | 220.6 | 190.7–257.9 |

Pair-specific Q (ChatGPT): G1−MX 8/8 negative (median −89.0 m); SW−G1 3/8; ED−SW 3/8; ED−G1 4/8 → mixed.
Median distance to coverage level (MX / G1): 50 % 18.5 / 48.7; 75 % 64.1 / 99.4; 90 % 126.2 / 157.8; 95 % 169.5 / 190.5; **96 % 250.2 / 218.4**. G1 is slower than MapEx up to 95 % at ≥6/8 poses and faster only at the end — it trades early coverage for no tail.

## Does each block behave as designed? Yes.
- G1: 0 of 133 LOCAL decisions with robot or goal outside the room mask. Every room exit leaves only frontier regions ≤ 10 cells (the MapEx size filter) — unknown left in the room is non-frontier (behind furniture). Revisit legs to rooms already left (≥ 10 new cells, ≥ 50 % in left rooms): **0 in all 24 runs**.
- SWEEP: overrides MapEx's in-room choice in 8–17 of 21–23 LOCAL decisions per run; reversals 0–2.
- EARLYDOOR: 5–6 releases per run, 1–3 change the MapEx winner.

## Why distances differ
1. **G1 vs MapEx (cause confirmed):** MapEx makes 56 GLOBAL legs with < 200 new cells, 944.6 m over 8 runs (≈ 118 m/run, 39 % of its travel), mostly after 95 % (tail after 95 %: MX 97–161 m vs G1 7–59 m). G1 family: 0–2 such legs. G1 GLOBAL efficiency 3.58 m/1000 new cells vs MapEx 6.59.
2. **SWEEP vs G1:** identical total LOCAL distance (820.3 vs 820.4 m) but SWEEP sees more cells while LOCAL (6.66 vs 7.52 m/1000 cells, −11 %); total unchanged. Per physical room mixed (medians −5.5…+4.5 m). The top-left room (elongation 1.79) is worse with SWEEP at 7/8 poses → the G2-003 shape-gate premise (sweep good for elongated rooms) is **not supported**.
3. **Pose-to-pose spread of G1/SW/ED (±30 m) = room-tour order.** Any in-room change moves the exit point; greedy GLOBAL then picks a different next room. Total correlates with the sum of GLOBAL legs ≥ 15 m (r = 0.68).
4. **Remaining inefficiency — rooms passed at the door but not entered.** Late (≥ 90 % coverage) GLOBAL transits ≥ 20 m: G1 24.7 m/run, SW 22.8, ED 12.0. In all 14 such transits the robot had earlier passed within ≤ 1.0 m of that room's mask without entering (robot never inside); the room was 44–91 % known then; finishing it later took only 3–16 m of LOCAL. Destinations are the small rooms by the hall (centroids [30,91], [156,236]). EARLYDOOR reduces these late transits but pays the same elsewhere (equal mean, largest variance; P05 +56 m).

## Consequences
- Thesis: G1 is the contribution with evidence (−28 % mean, 8/8 poses, sd 38 → 13). Report the early-coverage trade-off. G2 (SWEEP/ED): no total-distance gain; SWEEP's in-room view efficiency (−11 % m per new cell) is the only positive, modest.
- **G1-003 must be revised:** as specified it triggers only when the robot is inside a room during a GLOBAL route; in all 14 cases the robot never entered. Trigger must be "passing within ~1 m of a Room-Node room (door) that still has a reachable frontier region > 10 cells", with a bounded detour.
- G2-003 (shape-gated SWEEP): deprioritise; premise not supported.
- G2 direction worth keeping: next-room (tour) choice; EARLYDOOR is one greedy step of it and is not robust.

## Not proven
One map (New Room), 8 starts, native only; Hospital and Gazebo pending. Room clustering uses G1's own masks (overlap ≥ 0.5), not GT rooms.

## Correction (Claude, 2026-10-20, after G1-003 shadow)
Item 4 above overstated: in 5 of the 14 late transits the robot was within ≤ 1 m of the destination room only while LOCAL in an adjacent room (never during GLOBAL; min GLOBAL distance 1.1–3.7 m). Two mechanisms: (a) passed near during GLOBAL; (b) seen through the door from an adjacent LOCAL room. See handoffs/G1-003-claude-review-shadow-20261020.md.
