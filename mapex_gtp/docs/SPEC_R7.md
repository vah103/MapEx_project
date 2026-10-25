# MapEx-GTP R7 — implementation specification, 2026-10-23

Authority: RD-037, mapex_workspace commit 52f79cf; the two Claude handoffs copied
next to this document. This candidate replaces the R6/R6.1 *candidate names*;
all old source, unsuccessful tests, strict-gate pilot failures and evidence remain.

## Policy

Frozen R5 goal selection plus five independent booleans:

| Switch | Default | Exact operation |
|---|---|---|
| event_refresh (ER) | true | Refresh on invalid/no_path/no_progress/native_reselection/executor_rejection, zero planned observation, empty reached fallback, when observed core changed. Scheduled refresh stays every three reached viewpoints. First same-input dedup is recomputed once and asserted by full prediction/mean/variance hash. `GTP_R7_DEDUP_CHECK=1` is fixed for the cohort; its extra inference is counted. |
| failure_regions (FR′) | true | In addition to R5 blacklist, suppress a Chebyshev one-stride box. Release only when at least 10 initially UNKNOWN cells inside the box have become known. Prediction changes cannot release. R5 blacklist is never disabled. Fallback suppression relaxes if no executable candidate survives, without a new terminal. |
| jump_release (J5) | true | Consecutive-tick native A* path **cell count** increases by >50 cells for a locked viewpoint: release as no_progress. Exactly +50 does not trigger. R5 cumulative path/tick backstop remains. Fallbacks are excluded. |
| tail_rule (T3) | true | Only `fallback_reason=cover_empty`: after fallback prediction refresh/dedup, before scoring, count all predicted visible UNKNOWN over every current unfiltered WFD centre using observed+mean>=0.5 and unchanged 1440-ray/20m visibility. Includes tries-exhausted cells. Stop GTP_TAIL_STOP iff max<50. Log max<100 as offline sensitivity only. Rejection-only or no-plan-graph fallback does not trigger T3. |
| incremental_visibility (VIS) | true | Cache full raw visible sets, including first occupied samples. On every plan-map transition invalidate each entry if any previously visible cell changed occupancy. Refine to need only after cache lookup. Invalidation covers entries not queried in that decision. |

`verify_observation=False`, `cleanup_retry=False`, `fallback_guard=False`.
Their old optional code stays separate. If fallback_guard is enabled in a later
candidate, its history now accumulates fewer-than-10 newly known cells since the
first stalled attempt, rather than exact-SHA identity; upward A* reanchors are logged.

`track_execution_goal` asserts that accepted viewpoints are never silently
replaced by the native executor when fallback_guard is off. Fallback bookkeeping
may follow native reselection; it does not increment viewpoint tries.

Prediction age is logged at each decision in ticks and actual odometry metres.
The logger keeps actual planned observation outcomes and exact final postscan
mask. Native WFD, LiDAR, map accumulation, inflated A*, controller, model trio,
raycast correction, graph/cover/tour algorithms and baseline constants stay fixed.
Native A* still permits UNKNOWN traversal as in R5.

## Equivalence and evidence boundaries

“All OFF equals R5” means identical decisions, planned sets, tries, blacklist,
reached counters and trajectory/state on the same inputs. Instrumentation,
variant labels, runtime fields and final-state files necessarily differ.
The real-map gate replays *fixed observed inputs and recorded unlock outcomes*;
it is not a new counterfactual trajectory and uses no ground truth. Native-loop
integration uses a small synthetic map and fixed fake models only.

R5 arm A uses the frozen R5 planner/explorer/logger bytes, not an R7-labelled
surrogate. A separate worker captures the native Mapper handle without overriding
any mapping method, so it can save the exact actual final postscan state for A.
B/C use the identical R7 source, with C changing only jump_release to false.

No real-map/native-LaMa cohort is launched by preparation or freezing.
Code ACCEPT must come from Claude, method ACCEPT for T3/J5/VIS from a non-Claude
reviewer, then the user types GO in the executing session. The installer cannot
provide independent code ACCEPT.
