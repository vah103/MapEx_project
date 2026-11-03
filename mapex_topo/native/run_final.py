#!/usr/bin/env python3
"""TOPO-002 final evaluation (RD047_EVAL_FINAL.json): prepare + launch 3 runs sequentially.
Code must equal the TOPO2-50015847-P00-SMOKE-01 snapshot (hash check before every run);
a failing run is recorded and the next proceeds; no retry.  Pauses topo2-data's queue (PAUSE file)."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = HERE.parent
RUNS = Path("/home/dell/DELL_NATIVE2D/runs")
REC = HERE / "receipts" / "RD047_EVAL_FINAL.json"
SMOKE = RUNS / "TOPO2-50015847-P00-SMOKE-01" / "RUN_MANIFEST.json"
PAUSE = Path("/home/dell/DELL_NATIVE2D/analysis/topo2_data_20261103/PAUSE")
FROZEN = ("topo_script_sha256", "topo_policy_sha256", "topo_planner_sha256", "topo_l1_sha256",
          "pc_model_sha256", "l1_preset", "l1_pfree", "pc_source", "planner")


def busy():
    out = subprocess.run(["ps", "-eo", "args"], stdout=subprocess.PIPE, universal_newlines=True).stdout
    return any(l.strip().endswith("dell_worker.py") for l in out.splitlines())


def main():
    rec = json.loads(REC.read_text())
    ref = json.loads(SMOKE.read_text())
    log = HERE / "receipts" / "final_status.jsonl"
    PAUSE.touch()
    try:
        for rid in rec["run_ids"]:
            while True:          # free slot, stable for 30 s (other launcher may still hold the lock briefly)
                while busy():
                    time.sleep(15)
                time.sleep(30)
                if not busy():
                    break
            dest = RUNS / rid
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
                out = subprocess.run([sys.executable, "-W", "ignore", str(HERE / "launch_topo.py"),
                                      "--run-dir", str(dest), "--receipt", str(REC)],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                st["launch_exit"] = out.returncode
                st["launch_out"] = out.stdout[-3000:]
            except Exception as e:  # recorded; next run proceeds; no retry
                st["error"] = str(e)
            with log.open("a") as f:
                f.write(json.dumps(st) + "\n")
            print(json.dumps(st)[:600], flush=True)
    finally:
        if PAUSE.exists():
            PAUSE.unlink()


if __name__ == "__main__":
    main()
