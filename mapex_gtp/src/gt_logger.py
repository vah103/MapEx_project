"""GLOBAL-001 MapEx-GT evidence: extends the frozen PlainRecorder (plain_mapex_logger.py, unchanged).
Per-decision files keep the native names decision_%06d.json / decision_%06d_input.npz so existing gates and
audits read them; the JSON carries the GT fields. steps/ logs (plan, scan, odometry) are the parent's."""
import os
import json
import numpy as np
from plain_mapex_logger import PlainRecorder, sha, atomic


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


class GTRecorder(PlainRecorder):
    def __init__(self, exp_dir):
        PlainRecorder.__init__(self, exp_dir)
        atomic(os.path.join(self.root, "identity_gt.json"),
               {"controller": "GLOBAL-001 MapEx-GT: receding coverage tour on LaMa-predicted map; "
                              "MapEx scorer only as logged fallback",
                "room_node_disabled": True, "no_GT_in_planning": True,
                "GT_read_only_native_sim_collision": True})

    def gt_decision(self, t, pose, obs_full, mean_core, var_core, pad, info):
        self.n += 1
        h, w = mean_core.shape
        core = np.asarray(obs_full[pad:pad + h, pad:pad + w])
        if core.shape != mean_core.shape or var_core.shape != mean_core.shape:
            raise RuntimeError("gt decision crop mismatch")
        base = os.path.join(self.decisions, "decision_%06d" % self.n)
        np.savez_compressed(base + "_input.npz", observed=core.astype(np.float32),
                            mean=mean_core.astype(np.float32), variance=var_core.astype(np.float32))
        rec = dict(decision=self.n, native_tick=int(t), mode="GLOBAL001_GT",
                   robot_rc=np.asarray(pose, dtype=np.int64).tolist(),
                   source="GLOBAL-001 MapEx-GT" if not info.get("fallback") else "MapEx original score_frontiers (GT fallback)",
                   decision_input_npz=base + "_input.npz", input_observed_sha256=sha(core.astype(np.float32)),
                   input_mean_sha256=sha(mean_core.astype(np.float32)), input_var_sha256=sha(var_core.astype(np.float32)),
                   no_room_controller=True)
        rec["variant"] = "MapEx-GTP R7 candidate"
        if info.get("fallback"):
            rec["source"] = "MapEx score_frontiers with R7 failure feedback"
        rec.update(_clean(info))
        atomic(base + ".json", rec)
        self.last_decision = self.n
        if self.last_pose is None:
            self.last_pose = np.asarray(pose, dtype=float)

    def gt_final_observed(self, t, pose, obs_full, pad):
        """Persist the actual final state, including the last completed scan."""
        core = np.ascontiguousarray(obs_full[pad:-pad, pad:-pad] if pad else obs_full).astype(np.float32)
        path = os.path.join(self.root, 'final_observed_core.npz')
        if os.path.exists(path):
            raise RuntimeError('final observed output collision')
        with open(path + '.partial', 'wb') as f:
            np.savez_compressed(f, observed=core)
            f.flush()
            os.fsync(f.fileno())
        os.rename(path + '.partial', path)
        atomic(os.path.join(self.root, 'final_observed_core.json'), dict(
            native_loop_final_tick=int(t), last_observed_pose_rc=np.asarray(pose).tolist(),
            phase='actual final observation; includes all completed scans',
            observed_sha256=sha(core), shape=list(core.shape),
            known_cells=int((core != .5).sum()), free_cells=int((core == 0).sum()),
            unknown_cells=int((core == .5).sum()), ground_truth_used=False))

    def gt_summary(self, obj):
        atomic(os.path.join(self.root, "gt_run_summary.json"), _clean(obj))
