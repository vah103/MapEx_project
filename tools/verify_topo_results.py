#!/usr/bin/env python3
"""Verify compact TOPO R2 archives without needing Dell/raw runs/model weights."""
import csv
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "mapex_topo"
THREE = {"TOPO2-KTH49V3-P00-EVAL-01": "KTH49 P00",
         "TOPO2-KTH50V3-P00-EVAL-01": "KTH50 P00",
         "TOPO2-KTH51V3-P01-EVAL-01": "KTH51 P01"}
SEVEN = ("TOPO2-KTH49V3-P01-HOLDOUT-01",
         "TOPO2-KTH49V3-P03-HOLDOUT-01",
         "TOPO2-KTH50V3-P01-HOLDOUT-01",
         "TOPO2-KTH50V3-P03-HOLDOUT-01",
         "TOPO2-KTH51V3-P00-HOLDOUT-01",
         "TOPO2-KTH51V3-P02-HOLDOUT-01",
         "TOPO2-KTH51V3-P03-HOLDOUT-01")

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_run(root, expected, reference):
    assert root.is_dir() and root.name == expected
    meta = json.loads((root / "RUN_MANIFEST.json").read_text())
    receipt = json.loads((root / "RUN_RECEIPT.json").read_text())
    inv = json.loads((root / "raw_inventory.json").read_text())
    assert meta["id"] == expected
    assert receipt["run_id"] == expected and receipt["exit_code"] == 0
    assert meta["mission_time"] == 1000 and meta["planner"] == "R2_unified"
    assert inv["original_native_file_sets"]["global_obs"]["count"] == 1000
    assert inv["original_native_file_sets"]["decision_input_and_json"]["count"] > 0
    for relative, digest in inv["published_file_sha256"].items():
        assert sha(root / relative) == digest, (expected, relative)
    assert (root / "final_observed_tick999.png").is_file()
    assert (root / "step_evidence.tar.gz").is_file()
    rows = list(csv.DictReader((root / "coverage_distance_1000ticks.csv").open()))
    assert len(rows) == 1000
    tcov = [float(row["topo_r2_coverage_gtfree"]) for row in rows]
    mcov = [float(row["mapex_coverage_gtfree"]) for row in rows]
    td = [float(row["topo_r2_cumulative_distance_m"]) for row in rows]
    for i, row in enumerate(rows):
        assert int(row["tick"]) == i
        assert all(math.isfinite(x) for x in (tcov[i], mcov[i], td[i]))
        if i:
            assert tcov[i] + 1e-10 >= tcov[i - 1]
            assert mcov[i] + 1e-10 >= mcov[i - 1]
            assert td[i] + 1e-7 >= td[i - 1]
    for key, values in (("topo_r2", tcov), ("mapex", mcov)):
        measures = reference[key]
        assert abs(sum(values) / 1000 - measures["AUC_0_999"]) < 1e-9
        assert abs(values[999] - measures["cov@999"]) < 1e-9
    assert abs(td[-1] - reference["topo_r2"]["distance_m_last"]) < 1e-7
    return len(inv["published_file_sha256"])

def main():
    eval3 = json.loads((BASE / "eval_final/figs/metrics.json").read_text())
    holdout = json.loads((BASE / "eval_holdout7/metrics.json").read_text())
    assert set(holdout) == set(SEVEN)
    count = 0
    for run, name in THREE.items():
        old = eval3[name]
        ref = {"topo_r2": old["m_r2"], "mapex": old["m_mapex"]}
        count += verify_run(BASE / "eval_final/per_run" / run, run, ref)
        print("PASS", run)
    for run in SEVEN:
        count += verify_run(BASE / "eval_holdout7/per_run" / run, run, holdout[run])
        print("PASS", run)
    print("PASS 10/10 compact run archives;", count, "published SHA-256 file checks")

if __name__ == "__main__":
    main()
