#!/usr/bin/env python3
"""TOPO-002 final evaluation: MapEx | TOPO R1 | TOPO R2 (GT offline only).
Coverage = frozen MX1000 v0_metrics.coverage_distance; AUC = mean coverage over ticks 0..999."""
import json
import shutil
from pathlib import Path

import numpy as np

import importlib.util
W1 = Path("/home/dell/DELL_NATIVE2D/build/topo001/eval_wave1/scripts/analyze.py")
spec = importlib.util.spec_from_file_location("w1", str(W1))
w1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w1)
RUNS = Path("/home/dell/DELL_NATIVE2D/runs")
OUT = Path("/home/dell/DELL_NATIVE2D/build/topo002/eval_final/figs")
TRIPLES = [("KTH49 P00", "KTH49V3-P00-MX-T1000-01", "TOPO1-KTH49V3-P00-EVAL-01", "TOPO2-KTH49V3-P00-EVAL-01"),
           ("KTH50 P00", "KTH50V3-P00-MX-T1000-01", "TOPO1-KTH50V3-P00-EVAL-01", "TOPO2-KTH50V3-P00-EVAL-01"),
           ("KTH51 P01", "KTH51V3-V0R2-P01-MX-01", "TOPO1-KTH51V3-P01-EVAL-01", "TOPO2-KTH51V3-P01-EVAL-01")]


def r2_behaviour(exp, root, cov, gt):
    rows = [json.loads(l) for l in (exp / "topo_decisions.jsonl").open()]
    stats = json.loads((exp / "topo_policy_stats.json").read_text())
    rec = json.loads((root / "RUN_RECEIPT.json").read_text())
    dec = [r for r in rows if "event" not in r]
    seq, kinds, hk, relax, tim = [], {}, [0, 0], 0, []
    total = gt.sum()
    ticks = [r["tick"] for r in dec] + [len(cov) - 1]
    gains = []
    for i, r in enumerate(dec):
        p = r.get("planner") or {}
        stops = p.get("stops") or []
        first = stops[0] if stops else None
        k = first["kind"] if first else (r.get("goal_kind") or "fallback")
        kinds[k] = kinds.get(k, 0) + 1
        seq.append((r["tick"], k, p.get("first_stop_on_corridor_chain"), r.get("fallback")))
        h = p.get("hk") or {}
        if h.get("hk_prefix", 0) >= 2:
            hk[0] += 1
            hk[1] += int(bool(h.get("improved")))
        relax += int(bool(p.get("min_dist_relaxed")))
        t = r.get("timing_s") or {}
        tim.append({"pc_crop": t.get("crop"), "l1": t.get("l1"), "planner": t.get("planner"),
                    "hook": r.get("hook_wall_s")})
        t1 = min(ticks[i + 1], len(cov) - 1)
        gains.append({"tick": r["tick"], "ticks_to_next": t1 - r["tick"], "first_kind": k,
                      "pred_first_g": first["g"] if first else None,
                      "actual_new_gt_free": int(round((cov[t1] - cov[r["tick"]]) * total))})
    def q(xs):
        xs = [x for x in xs if x is not None]
        return None if not xs else {"median": float(np.median(xs)), "p90": float(np.percentile(xs, 90)),
                                    "max": float(np.max(xs))}
    corr = [c for _, _, c, _ in seq if c is not None]
    thirds = []
    for a, b in ((0, 333), (333, 666), (666, 1000)):
        cs = [c for t, _, c, _ in seq if c is not None and a <= t < b]
        thirds.append([int(sum(cs)), len(cs)])
    return {"decisions": len([r for r in dec if r["reset"]]), "hook_calls": len(dec),
            "first_kind_counts": kinds, "sequence": seq,
            "first_on_corridor": [int(sum(corr)), len(corr)], "first_on_corridor_by_third": thirds,
            "replan_triggers": stats.get("replan_triggers"), "fallbacks": stats["fallbacks"],
            "fallbacks_total": stats["fallbacks_total"], "reselects": stats["reselects"],
            "min_dist_relaxed": relax, "hk_improved": hk, "stale_goal_ticks": stats["stale_goal_ticks"],
            "pc_source": stats.get("pc_source"), "pc_fallback_used": stats.get("pc_fallback_used"),
            "l1_preset": stats.get("l1_preset"), "init_error": stats.get("init_error"),
            "timing": {k: q([t[k] for t in tim]) for k in ("pc_crop", "l1", "planner", "hook")},
            "gain_pred_vs_actual": gains, "peak_rss_gib": rec.get("worker_peak_rss_gib"),
            "exit_code": rec.get("exit_code"), "launched_at": rec.get("launched_at"),
            "ended_at": rec.get("ended_at")}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, mx, r1, r2 in TRIPLES:
        r = {"mapex": mx, "r1": r1, "r2": r2}
        try:
            c = {}
            for tag, run in (("mapex", mx), ("r1", r1), ("r2", r2)):
                exp, meta, gt, cov, dist = w1.curve_of(run)
                c[tag] = (exp, gt, cov)
                r["m_" + tag] = w1.metrics(cov, dist)
            e2, gt2, c2 = c["r2"]
            r["behaviour_r2"] = r2_behaviour(e2, RUNS / r2, c2, gt2)
            fig, ax = plt.subplots(figsize=(7.5, 4))
            ax.plot(c["mapex"][2], color="#888888", lw=1.6, label="MapEx")
            ax.plot(c["r1"][2], color="#d17a1f", lw=1.5, label="TOPO R1")
            ax.plot(c["r2"][2], color="#1f6fd1", lw=1.9, label="TOPO R2")
            for t, k, corr, fb in r["behaviour_r2"]["sequence"]:
                col = "#c00000" if fb else ("#1f6fd1" if corr else "#9bbbe8")
                ax.axvline(t, ymin=0, ymax=0.04, color=col, lw=1)
            ax.set_xlim(0, 1000); ax.set_ylim(0, 1); ax.grid(alpha=.3)
            ax.set_xlabel("tick"); ax.set_ylabel("GT-free coverage")
            ax.set_title("%s — R2 decisions: dark = first stop on corridor chain, light = other, red = fallback" % name,
                         fontsize=8)
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout(); fig.savefig(str(OUT / ("curve_%s.png" % name.replace(" ", "_"))), dpi=110)
            plt.close(fig)
            dbg = sorted((e2 / "topo_debug").glob("*.png"))
            for k in sorted(set([0, len(dbg) // 3, 2 * len(dbg) // 3, len(dbg) - 1])) if dbg else []:
                shutil.copy2(str(dbg[k]), str(OUT / ("%s_%s" % (r2, dbg[k].name))))
        except Exception:
            import traceback
            r["error"] = traceback.format_exc()
        res[name] = r
    (OUT / "metrics.json").write_text(json.dumps(res, indent=1, default=str))
    for k, r in res.items():
        print(k, {t: (r.get("m_" + t) or {}).get("AUC_0_999") for t in ("mapex", "r1", "r2")},
              r.get("error", "")[-300:])


if __name__ == "__main__":
    main()
