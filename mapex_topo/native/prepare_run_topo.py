#!/usr/bin/env python3
"""TOPO-002: prepare one UNLAUNCHED TOPO run directory from a frozen MapEx baseline run.

Copies the baseline runtime (worker, LaMa wrapper, logger, raycast patch, ...),
forces the prepared worker copy to `opts.mission_time=<ticks>` (R4 PC7 logic:
exactly one assignment, baseline untouched), installs the TOPO-integrated
script, topo_policy.py and frozen snapshots of topo_planner/ and topo_l1/.
Writes RUN_MANIFEST.json with all hashes.  Fails if the destination exists.
"""
import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOPO = HERE.parent
BASE_SCRIPT_SHA = "41ffcef8c21289054d2e17b1632bf7c8f54c2f29d55c3fc83195233cd910a3bb"
RAY_SHA = "6117e4bbd51205ecf02e8ceff0c9c1382fcb1c2ea96de3d7c90d13591018ba1a"
COPY = ("dell_worker.py", "fast_native_lama_3model.py", "plain_mapex_logger.py",
        "raycast_paper_patch.py", "room_native_integration.py", "sidecar.py")
SCRIPT = "explore_mapex_plain_lama3small_r1.py"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "out", "tests", "tools", "*.png")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def set_worker_mission_time(worker, ticks):
    worker = Path(worker)
    src = worker.read_bytes()
    pat = rb"(?m)^[ \t]*opts\.mission_time[ \t]*=[^\r\n]*"
    found = re.findall(pat, src)
    if len(found) != 1 or not re.match(rb"^opts\.mission_time=\d+$", found[0].strip()):
        raise AssertionError("WORKER_MISSION_TIME_ASSIGNMENT %r" % (found,))
    new = b"opts.mission_time=%d" % int(ticks)
    out = src.replace(found[0].strip(), new, 1)
    if len(re.findall(pat, out)) != 1 or new not in out:
        raise AssertionError("PREPARED_WORKER_MISSION_TIME")
    worker.write_bytes(out)
    return found[0].strip().decode(), new.decode()


def worker_start(worker):
    """Start pose (padded rc) from the worker's run_exploration_for_map call."""
    found = re.findall(r"np\.array\(\[(\d+),\s*(\d+)\],dtype=np\.int64\)", Path(worker).read_text())
    if len(found) != 1:
        raise AssertionError("WORKER_START_POSE %r" % (found,))
    return [int(found[0][0]), int(found[0][1])]


def prepare(baseline, destination, run_id, ticks, topo_script, l1_dir, planner_dir, kind,
            l1_preset, pc_dir):
    baseline, destination = Path(baseline), Path(destination)
    if destination.exists():
        raise RuntimeError("destination exists; no overwrite/retry")
    if not re.match(r"^TOPO2-[A-Za-z0-9_]+-P0\d-[A-Z0-9]+(-[A-Z0-9]+)?-\d\d$", run_id):
        raise ValueError("run id must look like TOPO2-<MAP>-P0x-<tag>-01")
    meta = json.loads((baseline / "RUN_MANIFEST.json").read_text())
    if sha(baseline / SCRIPT) != BASE_SCRIPT_SHA:
        raise AssertionError("BASE_SCRIPT_DRIFT")
    if sha(baseline / "raycast_paper_patch.py") != RAY_SHA:
        raise AssertionError("BASE_RAYCAST_DRIFT")
    if not (Path(l1_dir) / "topo_l1.py").exists():
        raise RuntimeError("topo_l1/topo_l1.py missing: " + str(l1_dir))
    destination.mkdir(parents=True)
    for name in COPY:
        shutil.copy2(str(baseline / name), str(destination / name))
    src_worker = sha(baseline / "dell_worker.py")
    old, new = set_worker_mission_time(destination / "dell_worker.py", ticks)
    if sha(baseline / "dell_worker.py") != src_worker:
        raise AssertionError("BASELINE_WORKER_CHANGED")
    if (baseline / "v41src").exists():
        shutil.copytree(str(baseline / "v41src"), str(destination / "v41src"))
    shutil.copy2(str(topo_script), str(destination / SCRIPT))
    shutil.copy2(str(HERE / "topo_policy.py"), str(destination / "topo_policy.py"))
    shutil.copytree(str(planner_dir), str(destination / "topo_planner"), ignore=IGNORE)
    shutil.copytree(str(l1_dir), str(destination / "topo_l1"), ignore=IGNORE)
    if pc_dir is not None:
        if not (Path(pc_dir) / "pc_model.py").exists():
            raise RuntimeError("pc_model/pc_model.py missing: " + str(pc_dir))
        shutil.copytree(str(pc_dir), str(destination / "pc_model"), ignore=IGNORE)
    manifest = {
        "run_id": run_id, "id": run_id, "kind": kind, "policy": "topo", "authorization": "RD-047",
        "task": "TOPO-002", "planner": "R2_unified", "created_local": datetime.now().isoformat(),
        "source_baseline_run_id": meta.get("run_id", meta.get("id")),
        "source_baseline_dir": str(baseline),
        "map": meta.get("map") or meta.get("map_path"), "map_sha256": meta["map_sha256"],
        "start_padded": meta.get("start_padded") or worker_start(destination / "dell_worker.py"),
        "core_shape": meta.get("core_shape") or meta.get("core_shape_rc"),
        "output_name": meta["output_name"],
        "mission_time": int(ticks), "planner_T": 1000, "l1_preset": l1_preset,
        "pc_source": "pc_model" if pc_dir is not None else "stub",
        "l1_pfree": False,  # coordinator decision at L1 v2 freeze: p(c) not passed to L1
        "l1_dir_source": str(l1_dir), "pc_dir_source": None if pc_dir is None else str(pc_dir),
        "worker_mission_time_line": [old, new],
        "source_worker_sha256": src_worker, "prepared_worker_sha256": sha(destination / "dell_worker.py"),
        "baseline_script_sha256": BASE_SCRIPT_SHA, "r4_integrated_sha256":
            "0cac1c4270685091ac8a8b9f3397dd0607be6813eaca8de30d6986a4a16a5026",
        "topo_script_sha256": sha(destination / SCRIPT),
        "topo_policy_sha256": sha(destination / "topo_policy.py"),
        "topo_planner_sha256": tree_hashes(destination / "topo_planner"),
        "topo_l1_sha256": tree_hashes(destination / "topo_l1"),
        "pc_model_sha256": tree_hashes(destination / "pc_model") if pc_dir is not None else None,
        "run_status": "PREPARED_NOT_LAUNCHED"}
    (destination / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--ticks", type=int, default=1000)
    p.add_argument("--kind", choices=("smoke", "eval"), required=True)
    p.add_argument("--topo-script", type=Path, default=HERE / "explore_mapex_topo_r1.py")
    p.add_argument("--l1-dir", type=Path, default=TOPO / "topo_l1",
                   help="L1 package dir (stub phase: ../topo001/topo_l1 with --l1-preset v1)")
    p.add_argument("--l1-preset", default="v2")
    p.add_argument("--pc-dir", type=Path, default=None, help="pc_model dir; omitted = stub 1[mean<0.5]")
    p.add_argument("--planner-dir", type=Path, default=TOPO / "planner" / "topo_planner")
    a = p.parse_args()
    if a.kind == "smoke" and a.ticks > 150:
        raise SystemExit("smoke runs are <= 150 ticks")
    if a.kind == "eval" and a.ticks != 1000:
        raise SystemExit("eval runs are 1000 ticks")
    print(json.dumps(prepare(a.baseline, a.destination, a.run_id, a.ticks, a.topo_script,
                             a.l1_dir, a.planner_dir, a.kind, a.l1_preset, a.pc_dir),
                     indent=2, sort_keys=True))
