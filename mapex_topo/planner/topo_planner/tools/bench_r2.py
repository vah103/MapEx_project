#!/usr/bin/env python3
"""Offline R2 planner timing/behaviour on recorded DEV decision inputs (DEV maps only: TOPO-002 §1).

Usage: bench_r2.py DECISIONS_DIR OUT_DIR --l1-file PATH --preset v1|v2 [--pc-file PATH] [--every N]
Planning grid rebuilt from the observed core like MapEx Mapper.inflate_map.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from topo_planner import UnifiedPlanner, debug_viz  # noqa: E402
from topo_planner.pc_stub import p_free as stub_p  # noqa: E402

PAD = 500
FORBIDDEN = ("50052749", "50052750", "50052751", "50052748", "50052752", "50052753", "50052754", "KTH")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decisions")
    ap.add_argument("out")
    ap.add_argument("--l1-file", required=True)
    ap.add_argument("--preset", default="v2")
    ap.add_argument("--pc-file")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--l1-pfree", action="store_true", help="forward p to L1 (default off: coordinator decision)")
    a = ap.parse_args()
    if any(f in a.decisions for f in FORBIDDEN):
        raise SystemExit("TEST maps must not be read by R2 development code")
    l1 = load(a.l1_file, "l1_bench")
    params = dict(l1.PRESETS[a.preset])
    import inspect
    takes_pfree = a.l1_pfree and "pfree" in inspect.signature(l1.build_topo).parameters
    pcm = load(a.pc_file, "pc_bench") if a.pc_file else None
    pfn = pcm.p_free if pcm else stub_p
    os.makedirs(a.out, exist_ok=True)
    pl = UnifiedPlanner()
    rows, prev = [], None
    for f in sorted(glob.glob(os.path.join(a.decisions, "decision_*[0-9].json")))[::a.every]:
        d = json.load(open(f))
        z = np.load(os.path.join(a.decisions, os.path.basename(d["decision_input_npz"])))
        obs, mean, var = z["observed"], z["mean"], z["variance"]
        robot = (d["robot_rc"][0] - PAD, d["robot_rc"][1] - PAD)
        fr = [(q[0] - PAD, q[1] - PAD) for q in d["candidates_rc"]]
        pf = ~binary_dilation(obs > 0.5, structure=np.ones((3, 3), bool))
        t0 = time.perf_counter()
        p = pfn(obs, mean, var)
        t1 = time.perf_counter()
        kw = {"pfree": p} if takes_pfree else {}
        topo = l1.build_topo(obs, mean, var, robot, 10.0, prev, params=params, **kw)
        prev = getattr(topo, "state", None)
        t2 = time.perf_counter()
        res = pl.plan(obs, mean, var, p, robot, topo, pf, fr, d["native_tick"])
        t3 = time.perf_counter()
        log = res["log"]
        goal = res["goals"][0][0] if res["goals"] else None
        debug_viz.render(os.path.join(a.out, "decision_%04d.png" % d["decision"]), obs,
                         topo.__dict__ if hasattr(topo, "__dict__") else topo, robot, res, goal,
                         "d%d t%d pc=%.2fs l1=%.2fs plan=%.2fs" % (d["decision"], d["native_tick"],
                                                                  t1 - t0, t2 - t1, t3 - t2))
        row = {"decision": d["decision"], "tick": d["native_tick"], "pc_s": t1 - t0, "l1_s": t2 - t1,
               "planner_s": t3 - t2, "total_s": t3 - t0, "timing": log.get("timing_s"),
               "n_stops": len(log.get("stops", [])), "J": log.get("J"),
               "first_kind": log["stops"][0]["kind"] if log.get("stops") else None,
               "first_on_corridor": log.get("first_stop_on_corridor_chain"),
               "candidate_kinds": log.get("candidate_kinds"), "rim": log.get("rim"),
               "hk": log.get("hk"), "relaxed": log.get("min_dist_relaxed"),
               "pc_fallback_used": getattr(pcm, "fallback_used", None) if pcm else None}
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)
    with open(os.path.join(a.out, "bench.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    tot = [r["total_s"] for r in rows[1:]] or [0]
    print("TOTAL_S (pc+l1+planner) median %.2f p90 %.2f max %.2f" % (np.median(tot), np.percentile(tot, 90), max(tot)))


if __name__ == "__main__":
    main()
