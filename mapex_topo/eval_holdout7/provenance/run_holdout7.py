#!/usr/bin/env python3
"""USER GO 03/11/2026: seven TOPO-002 R2 holdout configurations.
Uses the original, SHA-pinned TOPO-002 prepare/launch scripts without editing them.
Sequential, one worker, no retry, stop on first failure. Protects topo2-data using PAUSE."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path('/home/dell/DELL_NATIVE2D/build/topo002/native')
STAGE = Path(__file__).resolve().parent
B = HERE.parent
RUNS = Path("/home/dell/DELL_NATIVE2D/runs")
REC = STAGE / "USER_GO_HOLDOUT7_RD047.json"
SMOKE = RUNS / "TOPO2-50015847-P00-SMOKE-01" / "RUN_MANIFEST.json"
PAUSE = Path("/home/dell/DELL_NATIVE2D/analysis/topo2_data_20261103/PAUSE")
FROZEN = ("topo_script_sha256", "topo_policy_sha256", "topo_planner_sha256", "topo_l1_sha256",
          "pc_model_sha256", "l1_preset", "l1_pfree", "pc_source", "planner")


def busy():
    out = subprocess.run(["ps", "-eo", "args"], stdout=subprocess.PIPE, universal_newlines=True).stdout
    return any(l.strip().endswith("dell_worker.py") for l in out.splitlines())


def progress(state, completed, current=None, detail=None):
    info = {"state": state, "completed": completed, "total": 7, "current": current,
            "pid": os.getpid(), "updated_epoch": time.time()}
    if detail is not None:
        info["detail"] = str(detail)
    tmp = STAGE / ".STATUS.pending.json"
    tmp.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    tmp.replace(STAGE / "STATUS.json")


def main():
    rec = json.loads(REC.read_text())
    ref = json.loads(SMOKE.read_text())
    log = STAGE / "holdout7_status.jsonl"
    if len(rec["run_ids"]) != 7 or len(set(rec["run_ids"])) != 7:
        raise RuntimeError("HOLDOUT7_RECEIPT_RUN_IDS")
    blocked = [rid for rid in rec["run_ids"] if (RUNS / rid).exists()]
    if blocked:
        progress("BLOCKED", 0, detail={"existing_run_dirs": blocked})
        raise RuntimeError("HOLDOUT7_EXISTING_RUN_DIRS " + repr(blocked))
    progress("QUEUED", 0)
    if PAUSE.exists():
        raise RuntimeError('PAUSE_ALREADY_EXISTS_NOT_OURS')
    PAUSE.touch(exist_ok=False)
    try:
        for i, rid in enumerate(rec["run_ids"]):
            progress("WAITING_SLOT", i, rid)
            while True:          # free slot, stable for 30 s (other launcher may still hold the lock briefly)
                while busy():
                    time.sleep(15)
                time.sleep(30)
                if not busy():
                    break
            dest = RUNS / rid
            progress("PREPARING", i, rid)
            st = {"run_id": rid, "baseline": rec["baselines"][rid]}
            try:
                out = subprocess.run([sys.executable, str(HERE / "prepare_run_topo.py"), "--baseline",
                                      str(RUNS / rec["baselines"][rid]), "--destination", str(dest),
                                      "--run-id", rid, "--ticks", "1000", "--kind", "eval",
                                      "--l1-dir", str(B / "topo_l1"), "--l1-preset", "v2",
                                      "--pc-dir", str(B / "pc_model")],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                if out.returncode:
                    raise RuntimeError("prepare failed: " + out.stdout[-2000:])
                man = json.loads((dest / "RUN_MANIFEST.json").read_text())
                diff = [k for k in FROZEN if man.get(k) != ref.get(k)]
                if diff:
                    raise RuntimeError("CODE_NOT_FROZEN_AS_SMOKE_50015847 %s" % diff)
                st["frozen_check"] = "PASS"
                progress("RUNNING", i, rid)
                out = subprocess.run([sys.executable, "-W", "ignore", str(HERE / "launch_topo.py"),
                                      "--run-dir", str(dest), "--receipt", str(REC)],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                st["launch_exit"] = out.returncode
                st["launch_out"] = out.stdout[-3000:]
                if out.returncode == 0:
                    rr = json.loads((dest / "RUN_RECEIPT.json").read_text())
                    if rr.get("exit_code") != 0 or rr.get("run_id") != rid:
                        raise RuntimeError("RUN_RECEIPT_INVALID")
                    obs = dest / man["output_name"] / "MAPEX_PLAIN_LAMA3SMALL_RUN05_visvarprob" / "global_obs"
                    expected = {"%08d.png" % t for t in range(1000)}
                    seen = {p.name for p in obs.glob("*.png")}
                    if seen != expected:
                        raise RuntimeError("INCOMPLETE_GLOBAL_OBS missing=%d extra=%d" %
                                           (len(expected - seen), len(seen - expected)))
                    st["global_obs_ticks"] = "1000/1000"
            except Exception as e:  # record and stop batch; never retry
                st["error"] = str(e)
            with log.open("a") as f:
                f.write(json.dumps(st) + "\n")
            print(json.dumps(st)[:600], flush=True)
            if 'error' in st or st.get('launch_exit') != 0:
                progress("FAILED", i, rid, st)
                raise RuntimeError('HOLDOUT7_STOP_ON_FAILURE: ' + str(st))
            progress("WAITING_NEXT", i + 1)
        progress("COMPLETED", 7)
    finally:
        if PAUSE.exists():
            PAUSE.unlink()


if __name__ == "__main__":
    main()
