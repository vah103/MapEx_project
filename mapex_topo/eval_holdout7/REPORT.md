# TOPO R2 — seven additional KTH evaluations (RD-047, 2026-11-03)

Frozen source: GitHub `6ad6c67190726f5a60a8fc5bae835e65c3822e9c`.
Seven previously untested map/start configurations, each capped at 1000 ticks.
Coverage = GT-free observed occupancy per frozen V0 metric; AUC = mean ticks 0–999.
GT is used for offline evaluation only, never to choose online robot goals.

| Configuration | MapEx AUC | TOPO R2 AUC | MapEx final coverage | TOPO R2 final coverage | R2 distance (m) |
|---|---:|---:|---:|---:|---:|
| KTH49V3-P01 | 0.282 | 0.318 | 0.596 | 0.756 | 289.69 |
| KTH49V3-P03 | 0.302 | 0.482 | 0.477 | 0.698 | 293.89 |
| KTH50V3-P01 | 0.283 | 0.529 | 0.626 | 0.816 | 291.67 |
| KTH50V3-P03 | 0.398 | 0.489 | 0.621 | 0.666 | 293.35 |
| KTH51V3-P00 | 0.747 | 0.748 | 0.972 | 0.975 | 288.85 |
| KTH51V3-P02 | 0.745 | 0.767 | 0.966 | 0.988 | 288.92 |
| KTH51V3-P03 | 0.756 | 0.781 | 0.965 | 0.981 | 290.99 |

## Evidence and limits

- Seven source-frozen runs: 7/7 execution receipts exit 0, 1000/1000 ticks, freeze checks PASS.
- Every pair is evaluated with the same GT-free mask; both 1000-tick curves are included as CSV.
- Per run: complete TOPO decision JSONL, policy stats, native odometry, receipt, run manifest, worker log, final observed map, all planner debug PNGs, and compressed full step evidence.
- Raw per-tick PNG sequences and LaMa input NPZ stay unchanged on Dell, not committed to Git. `raw_inventory.json` records file counts, bytes and SHA-256 tree digests for verification.
- Three development-evaluation configurations (KTH49 P00, KTH50 P00, KTH51 P01) are archived separately under `../eval_final/`. Together these comprise ten TOPO R2 evaluations.
- TOPO R2 source was NOT modified, no new native experiments were run for this export.
- Results are descriptive; baseline provenance and map versions are in each run manifest. See original TOPO-002 methodology before asserting generalization.

## Recompute

On the original Dell host, with the frozen runtime and raw runs still present:

```bash
python3 mapex_topo/eval_holdout7/scripts/analyze_holdout7.py
```

The script refuses to overwrite an existing export directory.
