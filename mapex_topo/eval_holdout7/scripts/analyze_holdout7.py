#!/usr/bin/env python3
"""Offline export of TOPO-002 R2 holdout7 results; never modifies native run inputs.

Uses the frozen v0_metrics implementation indirectly through analyze.py.
Full native input NPZ/1000 PNGs remain on Dell, with digests in raw_inventory.json.
"""
import csv
import hashlib
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path("/home/dell/DELL_NATIVE2D")
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "mapex_topo" / "eval_holdout7"
RUNS = ROOT / "runs"
STAGE = ROOT / "staging" / "TOPO002_R2_HOLDOUT7_20261103"
SOURCE = REPO / "mapex_topo"
FROZEN = {
    "topo_planner/planner_r2.py": "77f24477bd04695d134964be7b02bdf592ec12424ec7a3a83011ab077a5a4d13",
    "topo_policy.py": "4269077b3792d512e1a1e2bbb488b4d69867786c93a6f4120d83b3009e584ac2",
    "topo_l1/topo_l1.py": "7aba149a4391b1045efdbf4ad2c8f4d81e0c4fddc256a093bf320a47ae794a19",
    "pc_model/pc_model.py": "fc2c40f81307e1ac88b80d2536ac23bb491d57bd8dfcb94d57fb602aa1ab97d6",
    "pc_model/pc_frozen.json": "fe1f75a04bef964d0a5effd832a72ae86595f1dfef906845aeba91446adefb15",
}
ARCHIVED = {
    "topo_planner/planner_r2.py": SOURCE / "planner" / "topo_planner" / "planner_r2.py",
    "topo_policy.py": SOURCE / "native" / "topo_policy.py",
    "topo_l1/topo_l1.py": SOURCE / "topo_l1" / "topo_l1.py",
    "pc_model/pc_model.py": SOURCE / "pc_model" / "pc_model.py",
    "pc_model/pc_frozen.json": SOURCE / "pc_model" / "pc_frozen.json",
}

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

W1 = load_module("topo001_wave1", ROOT / "build/topo001/eval_wave1/scripts/analyze.py")
R2 = load_module("topo002_final", ROOT / "build/topo002/eval_final/scripts/analyze_final.py")

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def inventory(path):
    files = sorted(x for x in path.rglob("*") if x.is_file()) if path.exists() else []
    h = hashlib.sha256()
    size = 0
    for file in files:
        rel = file.relative_to(path).as_posix()
        size += file.stat().st_size
        h.update(rel.encode("utf-8") + b"\0")
        h.update(digest(file).encode("ascii") + b"\n")
    return {"count": len(files), "bytes": size, "tree_digest_sha256": h.hexdigest()}

def valid_run(root, exp):
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    receipt = json.loads((root / "RUN_RECEIPT.json").read_text())
    assert manifest["id"] == root.name
    assert receipt["run_id"] == root.name and receipt["exit_code"] == 0
    assert manifest["mission_time"] == 1000 and manifest["planner"] == "R2_unified"
    for rel, expected in FROZEN.items():
        assert digest(root / rel) == expected, (root.name, rel)
        assert digest(ARCHIVED[rel]) == expected, ("archive source", rel)
    assert manifest["topo_script_sha256"] == digest(SOURCE / "native/explore_mapex_topo_r1.py")
    obs = exp / "global_obs"
    names = sorted(p.name for p in obs.glob("*.png"))
    assert names == [("%08d.png" % i) for i in range(1000)], root.name
    return manifest, receipt

def copy_outputs(run, exp, dest):
    dest.mkdir(parents=True, exist_ok=False)
    root = RUNS / run
    for f in ("RUN_MANIFEST.json", "RUN_RECEIPT.json", "worker.stdout.log"):
        shutil.copy2(root / f, dest / f)
    for f in ("mission_status.json", "topo_policy_stats.json",
              "topo_decisions.jsonl", "odom.npy"):
        shutil.copy2(exp / f, dest / f)
    shutil.copy2(exp / "global_obs" / "00000999.png", dest / "final_observed_tick999.png")
    debug = exp / "topo_debug"
    if debug.exists():
        shutil.copytree(debug, dest / "topo_debug")
    with tarfile.open(dest / "step_evidence.tar.gz", "w:gz") as archive:
        archive.add(exp / "plain_mapex_evidence" / "steps", arcname="steps")
    inv = {
        "native_original_path": str(root),
        "excluded_from_git": ["global_obs/00000000.png..00000998.png",
                              "plain_mapex_evidence/decisions/*_input.npz",
                              "numba_cache and model checkpoints"],
        "original_native_file_sets": {
            "global_obs": inventory(exp / "global_obs"),
            "decision_input_and_json": inventory(exp / "plain_mapex_evidence" / "decisions"),
            "steps": inventory(exp / "plain_mapex_evidence" / "steps"),
        },
        "published_file_sha256": {},
    }
    for path in sorted(dest.rglob("*")):
        if path.is_file() and path.name != "raw_inventory.json":
            inv["published_file_sha256"][path.relative_to(dest).as_posix()] = digest(path)
    (dest / "raw_inventory.json").write_text(json.dumps(inv, indent=2) + "\n")
    return inv

def main():
    assert not (OUT / "per_run").exists(), "Refuse to overwrite existing holdout export"
    state = json.loads((STAGE / "STATUS.json").read_text())
    assert state["state"] == "COMPLETED" and state["completed"] == state["total"] == 7
    ledger = [json.loads(line) for line in (STAGE / "holdout7_status.jsonl").read_text().splitlines()]
    assert len(ledger) == 7 and len({r["run_id"] for r in ledger}) == 7
    for x in ledger:
        assert x["frozen_check"] == "PASS" and x["launch_exit"] == 0 and x["global_obs_ticks"] == "1000/1000"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_run").mkdir()
    (OUT / "figs").mkdir()
    (OUT / "provenance").mkdir()
    for f in ("STATUS.json", "holdout7_status.jsonl", "USER_GO_HOLDOUT7_RD047.json",
              "run_holdout7.py"):
        shutil.copy2(STAGE / f, OUT / "provenance" / f)
    result = {}
    summary = ["# TOPO R2 — seven additional KTH evaluations (RD-047, 2026-11-03)",
               "", "Frozen source: GitHub `6ad6c67190726f5a60a8fc5bae835e65c3822e9c`.",
               "Seven previously untested map/start configurations, each capped at 1000 ticks.",
               "Coverage = GT-free observed occupancy per frozen V0 metric; AUC = mean ticks 0–999.",
               "GT is used for offline evaluation only, never to choose online robot goals.",
               "", "| Configuration | MapEx AUC | TOPO R2 AUC | MapEx final coverage | TOPO R2 final coverage | R2 distance (m) |",
               "|---|---:|---:|---:|---:|---:|"]
    for row in ledger:
        run, base = row["run_id"], row["baseline"]
        print("ANALYZING", run, "vs", base, flush=True)
        exp, meta, gt, cov, distance = W1.curve_of(run)
        exp0, meta0, gt0, cov0, distance0 = W1.curve_of(base)
        valid_run(RUNS / run, exp)
        assert len(cov) == len(cov0) == 1000, (run, len(cov), len(cov0))
        assert gt.shape == gt0.shape and np.array_equal(gt, gt0), (run, "GT mismatch")
        assert np.all(np.isfinite(cov)) and np.all(np.isfinite(distance))
        assert np.all(np.diff(cov) >= -1e-10) and np.all(np.diff(distance) >= -1e-7)
        assert np.all(np.diff(cov0) >= -1e-10) and np.all(np.diff(distance0) >= -1e-7)
        metrics = {"run_id": run, "baseline_run_id": base,
                   "mapex": W1.metrics(cov0, distance0),
                   "topo_r2": W1.metrics(cov, distance),
                   "behaviour_r2": R2.r2_behaviour(exp, RUNS / run, cov, gt)}
        dest = OUT / "per_run" / run
        copy_outputs(run, exp, dest)
        with (dest / "coverage_distance_1000ticks.csv").open("w", newline="") as fp:
            writer = csv.writer(fp, lineterminator="\n")
            writer.writerow(["tick", "mapex_coverage_gtfree", "topo_r2_coverage_gtfree",
                             "mapex_cumulative_distance_m", "topo_r2_cumulative_distance_m"])
            for i in range(1000):
                writer.writerow([i, "%.12f" % cov0[i], "%.12f" % cov[i],
                                 "%.9f" % distance0[i], "%.9f" % distance[i]])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(cov0, label="MapEx", color="#888", lw=1.6)
        ax.plot(cov, label="TOPO R2", color="#1766c2", lw=1.8)
        ax.set(xlim=(0, 1000), ylim=(0, 1), xlabel="tick", ylabel="GT-free coverage",
               title=run)
        ax.grid(alpha=.3); ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "figs" / (run + ".png"), dpi=120)
        plt.close(fig)
        result[run] = metrics
        summary.append("| %s | %.3f | %.3f | %.3f | %.3f | %.2f |" %
                       (run.replace("TOPO2-", "").replace("-HOLDOUT-01", ""),
                        metrics["mapex"]["AUC_0_999"], metrics["topo_r2"]["AUC_0_999"],
                        metrics["mapex"]["cov@999"], metrics["topo_r2"]["cov@999"],
                        metrics["topo_r2"]["distance_m_last"]))
        print("PASS", run, "AUC", round(metrics["topo_r2"]["AUC_0_999"], 4), flush=True)
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary += ["", "## Evidence and limits", "",
                "- Seven source-frozen runs: 7/7 execution receipts exit 0, 1000/1000 ticks, freeze checks PASS.",
                "- Every pair is evaluated with the same GT-free mask; both 1000-tick curves are included as CSV.",
                "- Per run: complete TOPO decision JSONL, policy stats, native odometry, receipt, run manifest, worker log, final observed map, all planner debug PNGs, and compressed full step evidence.",
                "- Raw per-tick PNG sequences and LaMa input NPZ stay unchanged on Dell, not committed to Git. `raw_inventory.json` records file counts, bytes and SHA-256 tree digests for verification.",
                "- Three development-evaluation configurations (KTH49 P00, KTH50 P00, KTH51 P01) are archived separately under `../eval_final/`. Together these comprise ten TOPO R2 evaluations.",
                "- TOPO R2 source was NOT modified, no new native experiments were run for this export.",
                "- Results are descriptive; baseline provenance and map versions are in each run manifest. See original TOPO-002 methodology before asserting generalization.",
                "", "## Recompute", "",
                "On the original Dell host, with the frozen runtime and raw runs still present:",
                "", "```bash",
                "python3 mapex_topo/eval_holdout7/scripts/analyze_holdout7.py",
                "```",
                "", "The script refuses to overwrite an existing export directory.",
                ""]
    (OUT / "REPORT.md").write_text("\n".join(summary))
    print("EXPORT_COMPLETE", len(result), "runs", flush=True)

if __name__ == "__main__":
    main()
