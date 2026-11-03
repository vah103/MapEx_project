#!/usr/bin/env python3
"""Export complete compact evidence for the three original TOPO R2 evaluations.

No source/run modification. Must match already archived evaluation metrics exactly.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
TOPO = HERE.parents[2]
SCRIPT = TOPO / "eval_holdout7" / "scripts" / "analyze_holdout7.py"
spec = importlib.util.spec_from_file_location("archive_holdout7", str(SCRIPT))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

OUT = TOPO / "eval_final" / "per_run"
CASES = [
    ("KTH49 P00", "TOPO2-KTH49V3-P00-EVAL-01", "KTH49V3-P00-MX-T1000-01"),
    ("KTH50 P00", "TOPO2-KTH50V3-P00-EVAL-01", "KTH50V3-P00-MX-T1000-01"),
    ("KTH51 P01", "TOPO2-KTH51V3-P01-EVAL-01", "KTH51V3-V0R2-P01-MX-01"),
]

def main():
    assert not OUT.exists(), "Refuse to overwrite existing eval3 evidence"
    published = json.loads((TOPO / "eval_final/figs/metrics.json").read_text())
    OUT.mkdir()
    for name, run, baseline in CASES:
        print("ANALYZING", run, flush=True)
        exp, _, gt, c2, d2 = module.W1.curve_of(run)
        exp0, _, gt0, c0, d0 = module.W1.curve_of(baseline)
        module.valid_run(module.RUNS / run, exp)
        assert len(c2) == len(c0) == 1000 and np.array_equal(gt, gt0)
        m = published[name]
        for arm, c, d in (("r2", c2, d2), ("mapex", c0, d0)):
            new = module.W1.metrics(c, d)
            old = m["m_" + arm]
            for key in ("AUC_0_999", "cov@250", "cov@500", "cov@999", "distance_m_last"):
                assert abs(new[key] - old[key]) < 1e-9, (run, arm, key)
        dest = OUT / run
        module.copy_outputs(run, exp, dest)
        with (dest / "coverage_distance_1000ticks.csv").open("w", newline="") as fp:
            w = csv.writer(fp, lineterminator="\n")
            w.writerow(["tick", "mapex_coverage_gtfree", "topo_r2_coverage_gtfree",
                        "mapex_cumulative_distance_m", "topo_r2_cumulative_distance_m"])
            for i in range(1000):
                w.writerow([i, "%.12f" % c0[i], "%.12f" % c2[i],
                            "%.9f" % d0[i], "%.9f" % d2[i]])
        print("PASS", run, "matches archived metrics", flush=True)
    print("EVAL3_EXPORT_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
