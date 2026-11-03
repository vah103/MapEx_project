#!/usr/bin/env python3
"""Combine the archived 3 development and 7 additional TOPO R2 results."""
import csv
import json
from pathlib import Path

TOPO = Path(__file__).resolve().parents[1] / "mapex_topo"
DEST = TOPO / "RESULTS_10_RUNS.csv"

def main():
    development = json.loads((TOPO / "eval_final/figs/metrics.json").read_text())
    extra = json.loads((TOPO / "eval_holdout7/metrics.json").read_text())
    assert len(development) == 3 and len(extra) == 7
    rows = []
    for name, value in development.items():
        rows.append(("development", name, value["r2"], value["mapex"],
                     value["m_r2"], value["m_mapex"]))
    for name, value in extra.items():
        suffix = name.replace("TOPO2-", "").replace("-HOLDOUT-01", "")
        rows.append(("additional", suffix, name, value["baseline_run_id"],
                     value["topo_r2"], value["mapex"]))
    rows.sort(key=lambda x: (x[1][:5], x[1]))
    columns = ["cohort", "configuration", "topo_run_id", "mapex_baseline_run_id",
               "mapex_auc_0_999", "topo_r2_auc_0_999", "delta_auc",
               "mapex_cov_999", "topo_r2_cov_999", "delta_cov_999",
               "mapex_distance_m", "topo_r2_distance_m", "delta_distance_m"]
    with DEST.open("w", newline="") as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(columns)
        for cohort, config, run, base, topo, mx in rows:
            def v(x):
                return "%.10f" % x
            writer.writerow([cohort, config, run, base,
                             v(mx["AUC_0_999"]), v(topo["AUC_0_999"]),
                             v(topo["AUC_0_999"] - mx["AUC_0_999"]),
                             v(mx["cov@999"]), v(topo["cov@999"]),
                             v(topo["cov@999"] - mx["cov@999"]),
                             v(mx["distance_m_last"]), v(topo["distance_m_last"]),
                             v(topo["distance_m_last"] - mx["distance_m_last"])])
    print("PASS: 10 independent run rows exported:", DEST)

if __name__ == "__main__":
    main()
