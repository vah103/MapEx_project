#!/usr/bin/env python3
"""Derive the TOPO native script from the MX1000 R4 integrated script (SHA-pinned).

R4 hook sites are kept byte-identical (choose / reject_last / track_stale).
Only three kinds of edits:
  1. import the TOPO harness instead of the MX1000 one;
  2. the two goal locks `frontier_region_centers[chosen_index]` become
     `mx_policy.goal(frontier_region_centers, chosen_index)` so a goal can be
     any reachable cell (fallback returns exactly the frontier as before);
  3. after the native validity check (non-UPEN branch), the harness may request
     a new decision for a still-valid TOPO goal (`should_replan`, always False
     for MapEx/fallback goals): goal targets observed, or A* path blow-up.
"""
import argparse
import hashlib
from pathlib import Path

R4_SHA = "0cac1c4270685091ac8a8b9f3397dd0607be6813eaca8de30d6986a4a16a5026"
NAME = "explore_mapex_plain_lama3small_r1.py"


def integrate(src):
    if hashlib.sha256(src.encode("utf-8")).hexdigest() != R4_SHA:
        raise ValueError("input is not the R4 integrated MapEx script")
    edits = [
        ("from mx1000_policy import PolicyHarness\n", "from topo_policy import PolicyHarness\n", 1),
        ("locked_frontier_center = frontier_region_centers[chosen_index]\n",
         "locked_frontier_center = mx_policy.goal(frontier_region_centers, chosen_index)\n", 2),
        ("                    need_new_locked_frontier = not is_locked_frontier_center_valid(locked_frontier_center, occ_grid_pyastar, cur_pose, collect_opts, pixel_per_meter)\n",
         "                    need_new_locked_frontier = not is_locked_frontier_center_valid(locked_frontier_center, occ_grid_pyastar, cur_pose, collect_opts, pixel_per_meter)\n"
         "                    if not need_new_locked_frontier and mx_policy.should_replan(t, locked_frontier_center, cur_pose, occ_grid_pyastar, mapper.obs_map):\n"
         "                        need_new_locked_frontier = True\n", 1),
    ]
    out = src
    for old, new, n in edits:
        if out.count(old) != n:
            raise AssertionError("anchor count %d != %d: %r" % (out.count(old), n, old))
        out = out.replace(old, new)
    if "mx1000_policy" in out:
        raise AssertionError("residual MX1000 harness import")
    compile(out, NAME, "exec")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r4-script", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit("refusing to overwrite")
    a.output.write_text(integrate(a.r4_script.read_text(encoding="utf-8")), encoding="utf-8")
    print("R4_SHA", R4_SHA)
    print("TOPO_SHA", hashlib.sha256(a.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
