#!/usr/bin/env python3
"""Launch ONE prepared TOPO run.  Never retries.

Guards (TOPO-002 / RD-047):
  * receipt JSON written by the coordinating Claude session:
      {"authorization": "RD-047", "task": "TOPO-002", "run_ids": [...], "kind": "smoke"|"eval"}
    the run id and kind must be listed;
  * no other dell_worker.py process running, plus an exclusive flock;
  * no RAM threshold (user decision 2026-11-03): MemAvailable at start and the
    worker's peak RSS are only recorded in RUN_RECEIPT.json;
  * prepared, never launched (no output dir, no RUN_RECEIPT.json).
"""
import argparse
import fcntl
import hashlib
import json
import os
import resource
import signal
import subprocess
from datetime import datetime
from pathlib import Path

NATIVE_PYTHON = "/home/dell/miniforge3/envs/lama/bin/python"
VENDOR = "/home/dell/DELL_NATIVE2D/vendor"
LIBSTDCPP = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
LOCK = Path("/home/dell/DELL_NATIVE2D/build/topo001/native/TOPO_ONE_WORKER.lock")  # shared with topo2-data (their request)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024. / 1024.
    raise RuntimeError("MemAvailable not reported")


def other_workers():
    me = os.getpid()
    found = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            args = (d / "cmdline").read_bytes().split(b"\0")
        except (OSError, IOError):
            continue
        if any(a.endswith(b"dell_worker.py") for a in args):
            found.append((int(d.name), b" ".join(args).decode("utf-8", "replace")))
    return found


def launch(run_dir, receipt_path):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text())
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get("authorization") != "RD-047" or receipt.get("task") != "TOPO-002":
        raise RuntimeError("RD046_RECEIPT_INVALID")
    if manifest["run_id"] not in receipt.get("run_ids", []):
        raise RuntimeError("RUN_ID_NOT_AUTHORIZED")
    if receipt.get("kind") != manifest.get("kind"):
        raise RuntimeError("RUN_KIND_NOT_AUTHORIZED")
    if manifest.get("run_status") != "PREPARED_NOT_LAUNCHED":
        raise RuntimeError("RUN_MANIFEST_NOT_PREPARED")
    if (run_dir / manifest["output_name"]).exists() or (run_dir / "RUN_RECEIPT.json").exists():
        raise RuntimeError("RUN_ALREADY_LAUNCHED_NO_RETRY")
    with LOCK.open("a+") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        others = other_workers()
        if others:
            raise RuntimeError("OTHER_DELL_WORKER_RUNNING %r" % (others,))
        mem = available_gib()  # recorded only, never blocks
        env = os.environ.copy()
        env.update({"DELL_NATIVE2D_ROOT": str(run_dir), "TOPO_POLICY": manifest["policy"],
                    "TOPO_PLANNER_T": str(manifest.get("planner_T", 1000)),
                    "TOPO_L1_PRESET": str(manifest.get("l1_preset", "v2")),
                    "TOPO_PC": "stub" if manifest.get("pc_source") == "stub" else "model",
                    "TOPO_L1_PFREE": "1" if manifest.get("l1_pfree") else "0",
                    "LD_PRELOAD": LIBSTDCPP, "PYTHONPATH": VENDOR + ":" + str(run_dir),
                    "DELL_NATIVE2D_THREADS": "2", "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                    "NUMBA_CACHE_DIR": str(run_dir / "numba_cache")})
        rec = {"run_id": manifest["run_id"], "kind": manifest["kind"],
               "receipt_sha256": digest(receipt_path), "receipt_path": str(receipt_path),
               "launched_at": datetime.utcnow().isoformat() + "Z", "start_mem_available_gib": mem,
               "worker_sha256": digest(run_dir / "dell_worker.py"), "launched": True}
        with (run_dir / "worker.stdout.log").open("x") as log:
            proc = subprocess.Popen([NATIVE_PYTHON, str(run_dir / "dell_worker.py")],
                                    stdout=log, stderr=subprocess.STDOUT, env=env,
                                    cwd=str(run_dir), start_new_session=True)

            def stop(sig, _):
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                raise KeyboardInterrupt("signal " + str(sig))
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            try:
                rec["exit_code"] = proc.wait()
            except BaseException:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                rec["exit_code"] = proc.wait()
                rec["interrupted"] = True
                raise
            finally:
                # ru_maxrss of waited-for children, KiB on Linux (the worker is the only child)
                rec["worker_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024. / 1024.
                rec["ended_at"] = datetime.utcnow().isoformat() + "Z"
                with (run_dir / "RUN_RECEIPT.json").open("x") as f:
                    json.dump(rec, f, indent=2, sort_keys=True)
        if rec["exit_code"] != 0:
            raise RuntimeError("worker failed; no automatic retry")
        return rec


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(launch(a.run_dir, a.receipt), indent=2, sort_keys=True))
