#!/usr/bin/env python3
"""Offline timing/behaviour of the planner on recorded baseline decision inputs.

Usage: bench_recorded.py DECISIONS_DIR OUT_DIR [--l1 stub|real] [--every N]
Planning grid is rebuilt from the observed core exactly like MapEx
Mapper.inflate_map (3x3 dilation of occupied; unknown traversable).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from topo_planner import TopoPlanner, debug_viz  # noqa: E402
from topo_planner.l1_stub import coarse_build_topo  # noqa: E402

PAD = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decisions")
    ap.add_argument("out")
    ap.add_argument("--l1", default="stub")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--preset", default="v1")
    a = ap.parse_args()
    if a.l1 == "real":
        sys.path.insert(0, os.path.expanduser("~/DELL_NATIVE2D/build/topo001/topo_l1"))
        import topo_l1
        _params = dict(topo_l1.PRESETS[a.preset]) if a.preset not in ("", "spec", "v0") else None

        def build_topo(obs, mean, var, rc, ppm, prev=None):
            return topo_l1.build_topo(obs, mean, var, rc, ppm, prev, params=_params)
    else:
        build_topo = coarse_build_topo
    os.makedirs(a.out, exist_ok=True)
    pl = TopoPlanner()
    rows = []
    prev = None
    for f in sorted(glob.glob(os.path.join(a.decisions, "decision_*[0-9].json")))[::a.every]:
        d = json.load(open(f))
        z = np.load(d["decision_input_npz"].replace(
            os.path.dirname(d["decision_input_npz"]), a.decisions))
        obs, mean, var = z["observed"], z["mean"], z["variance"]
        robot = (d["robot_rc"][0] - PAD, d["robot_rc"][1] - PAD)
        fr = [(p[0] - PAD, p[1] - PAD) for p in d["candidates_rc"]]
        pf = ~binary_dilation(obs > 0.5, structure=np.ones((3, 3), bool))
        t0 = time.perf_counter()
        topo = build_topo(obs, mean, var, robot, 10.0, prev)
        t1 = time.perf_counter()
        prev = topo.get("state") if isinstance(topo, dict) else getattr(topo, "state", None)
        res = pl.plan(obs, mean, var, robot, topo, pf, fr, d["native_tick"])
        t2 = time.perf_counter()
        goal = res["goals"][0][0] if res["goals"] else None
        log = res["log"]
        debug_viz.render(os.path.join(a.out, "decision_%04d.png" % d["decision"]), obs, topo,
                         robot, res, goal, "d%d t%d %s l1=%.2fs plan=%.2fs" % (
                             d["decision"], d["native_tick"], log["phase"], t1 - t0, t2 - t1))
        row = {"decision": d["decision"], "tick": d["native_tick"], "phase": log["phase"],
               "l1_s": t1 - t0, "planner_s": t2 - t1, "timing": log["timing_s"],
               "raycasts": log["raycasts"], "goal": list(goal) if goal else None,
               "n_goals": len(res["goals"])}
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(a.out, "bench.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ps = [r["planner_s"] for r in rows[1:]] or [0]
    print("PLANNER_S median %.2f p90 %.2f max %.2f" % (np.median(ps), np.percentile(ps, 90), max(ps)))


if __name__ == "__main__":
    main()
