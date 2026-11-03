"""Offline tests of topo_policy in a replica of the native MapEx goal loop.

Uses MapEx sim_utils.Mapper / FrontierPlanner / psuedo_traj_controller and the
same lock / unlock (<1 m) / invalid (inf) / A*-None reselect logic as the
native script, on a synthetic padded world (500-cell free pad like the native
runs).  LaMa is replaced by an oracle-ish mean (synthetic GT with noise) - only
the hook mechanics are tested here, not prediction quality.
"""
import json
import os
import sys
import tempfile
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
TOPO = os.path.dirname(NATIVE)
sys.path.insert(0, NATIVE)
sys.path.insert(0, os.path.join(TOPO, "planner"))
sys.path.insert(0, os.path.join(TOPO, "planner", "topo_planner", "tests"))
os.environ["TOPO_PC"] = "stub"
for p in ("/home/dell/MapEx", "/home/dell/MapEx/scripts"):
    if p not in sys.path:
        sys.path.append(p)

import pyastar2d  # noqa: E402
from scripts import sim_utils  # noqa: E402
import synth  # noqa: E402
from topo_planner.l1_stub import topomap_from_labels  # noqa: E402

PAD = 500


def make_world():
    gt_core, labels, classes, doors = synth.corridor_world(4)
    gt = np.pad(gt_core, PAD, mode="constant", constant_values=0)
    return gt_core, gt, labels, classes, doors


def fake_l1(labels, classes, doors):
    mod = types.ModuleType("topo_l1")

    def build_topo(obs, mean, var, robot_rc, ppm, prev=None, params=None):
        assert obs.shape == labels.shape and mean.shape == labels.shape
        t = topomap_from_labels(labels, classes, doors, ppm, obs, mean)
        t["state"] = {"n": (prev or {}).get("n", 0) + 1}
        return t
    mod.build_topo = build_topo
    mod.PRESETS = {"v1": {}}
    return mod


def prediction(gt_core, obs_full, rng):
    H, W = obs_full.shape
    from topo_policy import centered_padding
    pt, pb = centered_padding(H)
    pl, pr = centered_padding(W)
    mean = np.ones((H + pt + pb, W + pl + pr), dtype=np.float32)
    noisy = np.clip(gt_core + rng.normal(0, 0.15, gt_core.shape), 0, 1).astype(np.float32)
    mean[PAD + pt:PAD + pt + gt_core.shape[0], PAD + pl:PAD + pl + gt_core.shape[1]] = noisy
    var = np.full(mean.shape, 0.02, dtype=np.float32)
    return mean, var


def valid(goal, grid, cur, ppm):
    if goal is None:
        return False
    if grid[goal[0], goal[1]] == np.inf:
        return False
    if np.linalg.norm(goal - cur) < 1 * ppm:
        return False
    return True


def native_loop(harness, gt_core, gt, ticks, seed=0, start=(600, 530)):
    """Replica of the native per-tick goal logic (A* branch of the MapEx script)."""
    rng = np.random.RandomState(seed)
    ppm = 10
    cfg = {"laser_range_m": 20, "num_laser": 2500, "pixel_per_meter": ppm,
           "dilate_diam_for_planning": 3}
    mapper = sim_utils.Mapper(gt, cfg, use_distance_transform_for_planning=True)
    fp = sim_utils.FrontierPlanner(score_mode="visvarprob")
    cur = np.array(start, dtype=np.int64)
    mapper.observe_and_accumulate_given_pose(cur)
    pose_list = np.atleast_2d(cur)
    locked = None
    st = {"unlock_lt_1m": 0, "invalid_inf": 0, "decisions": 0, "reselect": 0,
          "hit_wall": 0, "non_frontier_goals": 0, "ended": None, "replan_trigger": 0}
    for t in range(ticks):
        fr, _, _ = fp.get_frontier_centers_given_obs_map(mapper.obs_map)
        if len(fr) == 0:
            st["ended"] = "no_frontier"
            break
        grid = mapper.get_inflated_planning_maps(unknown_as_occ=True)
        if locked is not None and np.linalg.norm(locked - cur) < 1 * ppm:
            locked = None
            st["unlock_lt_1m"] += 1
        need = not valid(locked, grid, cur, ppm)
        if not need and harness.should_replan(t, locked, cur, grid, mapper.obs_map):
            need = True
            st["replan_trigger"] += 1
        if need:
            if locked is not None and grid[locked[0], locked[1]] == np.inf:
                st["invalid_inf"] += 1
            centers = np.array(fr)
            costs = np.linalg.norm(centers - cur, axis=1) / -100.0 - rng.rand(len(centers))
            mean, var = prediction(gt_core, mapper.obs_map, rng)
            idx, costs = harness.choose(t, centers, costs, cur, grid, mapper, mean, var,
                                        pose_list, reset=True)
            if idx is None:
                st["ended"] = "no_valid_frontier"
                break
            locked = harness.goal(centers, idx)
            st["decisions"] += 1
            if not any(np.array_equal(locked, c) for c in centers):
                st["non_frontier_goals"] += 1
        path = pyastar2d.astar_path(grid, cur, locked, allow_diagonal=False)
        while path is None:
            st["reselect"] += 1
            harness.reject_last()
            idx, costs = harness.choose(t, centers, costs, cur, grid, mapper, mean, var,
                                        pose_list, reset=False)
            if idx is None:
                st["ended"] = "no_reselect"
                return st, mapper
            locked = harness.goal(centers, idx)
            path = pyastar2d.astar_path(grid, cur, locked, allow_diagonal=False)
        harness.track_stale(t, locked, mapper.obs_map)
        cur = sim_utils.psuedo_traj_controller(path[:, 0], path[:, 1], plan_ind_to_use=3)
        if mapper.gt_map[cur[0], cur[1]] == 1:
            st["hit_wall"] += 1
            st["ended"] = "hit_wall"
            break
        pose_list = np.concatenate([pose_list, np.atleast_2d(cur)], axis=0)
        mapper.observe_and_accumulate_given_pose(cur)
    st["known_free_core"] = int(np.count_nonzero(
        mapper.obs_map[PAD:-PAD, PAD:-PAD][gt_core == 0] == 0))
    return st, mapper


class TestNativeHook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gt_core, cls.gt, labels, classes, doors = make_world()
        sys.modules["topo_l1"] = fake_l1(labels, classes, doors)
        import topo_policy
        cls.tp = topo_policy

    def harness(self, policy="topo"):
        d = tempfile.mkdtemp(prefix="topo_hook_")
        return self.tp.PolicyHarness(d, 10, 1000, policy=policy), d

    def test_core_crop_matches_logger(self):
        """core_inputs == plain_mapex_logger crop (pad + native pad_h1/pad_w1)."""
        H, W = 650 + 2 * PAD, 855 + 2 * PAD
        obs = np.random.RandomState(0).choice([0, .5, 1], size=(H, W)).astype(np.float32)
        pad_h, pad_w = H % 16, W % 16
        ph1 = int((16 - pad_h) / 2) if pad_h else 0
        pw1 = int((16 - pad_w) / 2) if pad_w else 0
        from topo_policy import centered_padding
        Hp = H + sum(centered_padding(H))
        Wp = W + sum(centered_padding(W))
        mean = np.random.RandomState(1).rand(Hp, Wp).astype(np.float32)
        var = np.random.RandomState(2).rand(Hp, Wp).astype(np.float32)
        grid = np.zeros((H, W), np.float32)
        o, m, v, pf, rob, fr = self.tp.core_inputs(obs, mean, var, grid, (515, 515),
                                                   np.array([[515, 520], [10, 10]]))
        y, x = PAD + ph1, PAD + pw1
        self.assertTrue(np.array_equal(m, mean[y:y + 650, x:x + 855]))
        self.assertTrue(np.array_equal(v, var[y:y + 650, x:x + 855]))
        self.assertTrue(np.array_equal(o, obs[PAD:PAD + 650, PAD:PAD + 855]))
        self.assertEqual(rob, (15, 15))
        self.assertEqual(fr, [(15, 20)])

    def test_loop_topo(self):
        h, d = self.harness()
        st, mapper = native_loop(h, self.gt_core, self.gt, 220)
        stats = json.load(open(os.path.join(d, "topo_policy_stats.json")))
        print("\nLOOP_STATS", st, stats["fallbacks"], stats["phase_counts"])
        self.assertIsNone(stats["init_error"])
        self.assertEqual(st["hit_wall"], 0)
        self.assertGreater(stats["topo_goals"], 3)
        self.assertGreater(st["non_frontier_goals"], 0)
        self.assertGreater(st["unlock_lt_1m"], 0, "goals reached and unlocked at < 1 m")
        self.assertIn("R2", stats["phase_counts"])
        self.assertEqual(stats["pc_source"], "stub_mean_lt_0.5")
        self.assertGreater(stats["decisions_with_stops"], 0)
        rows = [json.loads(l) for l in open(os.path.join(d, "topo_decisions.jsonl"))]
        self.assertTrue(all("timing_s" in r or "fallback" in r for r in rows if "event" not in r))
        pngs = os.listdir(os.path.join(d, "topo_debug"))
        self.assertGreater(len(pngs), 3)
        self.assertGreater(st["known_free_core"], 0.6 * np.count_nonzero(self.gt_core == 0))

    def _real_l1_loop(self, l1_file, preset):
        os.environ["TOPO_L1_FILE"] = l1_file
        os.environ["TOPO_L1_PRESET"] = preset
        try:
            h, d = self.harness()
        finally:
            os.environ.pop("TOPO_L1_FILE")
            os.environ.pop("TOPO_L1_PRESET")
        self.assertIsNone(h.init_error)
        self.assertEqual(h.build_topo.preset, preset)
        self.assertFalse(h.build_topo.takes_pfree)
        st, _ = native_loop(h, self.gt_core, self.gt, 120)
        stats = json.load(open(os.path.join(d, "topo_policy_stats.json")))
        print("\nREAL_L1_LOOP", preset, st, stats["fallbacks"], stats["phase_counts"],
              stats["replan_triggers"], "first_on_corridor", stats["first_stop_on_corridor_chain"],
              "/", stats["decisions_with_stops"])
        self.assertEqual(st["hit_wall"], 0)
        self.assertGreater(stats["topo_goals"], 0)
        rows = [json.loads(l) for l in open(os.path.join(d, "topo_decisions.jsonl"))]
        self.assertFalse([r for r in rows if "traceback" in r], "no planner/L1 exception")
        return stats

    def test_loop_real_l1_v1(self):
        """Stub phase: TOPO-001 L1 v1 (preset v1) loaded like the native run."""
        self._real_l1_loop(os.path.join(os.path.dirname(TOPO), "topo001", "topo_l1", "topo_l1.py"), "v1")

    @unittest.skipUnless(os.path.isfile(os.path.join(TOPO, "topo_l1", "topo_l1.py")), "L1 v2 absent")
    def test_loop_real_l1_v2(self):
        """L1 v2 (frozen) with the frozen pc_model; pfree not forwarded (coordinator decision)."""
        os.environ["TOPO_PC"] = "model"
        os.environ["TOPO_PC_FILE"] = os.path.join(TOPO, "pc_model", "pc_model.py")
        try:
            stats = self._real_l1_loop(os.path.join(TOPO, "topo_l1", "topo_l1.py"), "v2")
        finally:
            os.environ["TOPO_PC"] = "stub"
            os.environ.pop("TOPO_PC_FILE")
        self.assertTrue(stats["pc_source"].startswith("pc_model:"))
        self.assertIs(stats["pc_fallback_used"], False)

    def test_should_replan_triggers(self):
        h, d = self.harness()
        obs = np.full(self.gt.shape, 0.5, np.float32)
        grid = np.ones(self.gt.shape, np.float32)
        goal = np.array([600, 700])
        tg = np.array([[600, 720 + i] for i in range(50)])
        h.last_kind = "topo"
        h.goal_info = {"tick": 0, "goal": goal.copy(), "astar_cells": 170, "targets": tg,
                       "last_check": 0}
        cur = np.array([600, 530])
        self.assertFalse(h.should_replan(3, goal, cur, grid, obs))      # too early
        self.assertFalse(h.should_replan(6, goal, cur, grid, obs))      # nothing seen, path same
        obs[tg[:45, 0], tg[:45, 1]] = 0.0
        self.assertTrue(h.should_replan(7, goal, cur, grid, obs))       # 90 % of targets seen
        obs[:] = 0.5
        grid[560:640, 600] = np.inf                                      # wall forces detour
        grid[400:800, 650] = np.inf
        h.goal_info.update({"tick": 10, "last_check": 10})
        self.assertFalse(h.should_replan(13, goal, cur, grid, obs))     # < 5 ticks after decision
        self.assertTrue(h.should_replan(16, goal, cur, grid, obs))      # path blow-up
        self.assertEqual(h.carry_exclude, [(100, 200)])
        self.assertFalse(h.should_replan(20, np.array([1, 1]), cur, grid, obs))  # other goal
        h.last_kind = "mapex"
        self.assertFalse(h.should_replan(30, goal, cur, grid, obs))     # never for MapEx goals
        stats = json.load(open(os.path.join(d, "topo_policy_stats.json")))
        self.assertEqual(stats["replan_triggers"], {"targets_observed": 1, "path_blowup": 1})

    def test_inf_goal_triggers_replan(self):
        h, d = self.harness()
        mapper = sim_utils.Mapper(self.gt, {"laser_range_m": 20, "num_laser": 2500,
                                            "pixel_per_meter": 10, "dilate_diam_for_planning": 3},
                                  use_distance_transform_for_planning=True)
        cur = np.array([600, 530])
        mapper.observe_and_accumulate_given_pose(cur)
        grid = mapper.get_inflated_planning_maps()
        fr, _, _ = sim_utils.FrontierPlanner("visvarprob").get_frontier_centers_given_obs_map(mapper.obs_map)
        mean, var = prediction(self.gt_core, mapper.obs_map, np.random.RandomState(0))
        idx, _ = h.choose(0, np.array(fr), np.zeros(len(fr)), cur, grid, mapper, mean, var, None)
        g = h.goal(np.array(fr), idx)
        grid2 = grid.copy()
        grid2[g[0], g[1]] = np.inf      # goal became an obstacle
        self.assertFalse(valid(g, grid2, cur, 10))   # native would call choose(reset=True)
        idx2, _ = h.choose(1, np.array(fr), np.zeros(len(fr)), cur, grid2, mapper, mean, var, None)
        g2 = h.goal(np.array(fr), idx2)
        self.assertTrue(np.isfinite(grid2[g2[0], g2[1]]))

    def test_reselect_calls_planner_again(self):
        h, d = self.harness()
        mapper = sim_utils.Mapper(self.gt, {"laser_range_m": 20, "num_laser": 2500,
                                            "pixel_per_meter": 10, "dilate_diam_for_planning": 3},
                                  use_distance_transform_for_planning=True)
        cur = np.array([600, 530])
        mapper.observe_and_accumulate_given_pose(cur)
        grid = mapper.get_inflated_planning_maps()
        fr = np.array(sim_utils.FrontierPlanner("visvarprob").get_frontier_centers_given_obs_map(mapper.obs_map)[0])
        mean, var = prediction(self.gt_core, mapper.obs_map, np.random.RandomState(0))
        idx, c = h.choose(0, fr, np.zeros(len(fr)), cur, grid, mapper, mean, var, None)
        g1 = h.goal(fr, idx)
        calls = h.calls
        h.reject_last()                       # native: A* returned None
        idx, c = h.choose(0, fr, c, cur, grid, mapper, mean, var, None, reset=False)
        g2 = h.goal(fr, idx)
        self.assertEqual(h.calls, calls + 1)
        self.assertFalse(np.array_equal(g1, g2))
        rows = [json.loads(l) for l in open(os.path.join(d, "topo_decisions.jsonl"))]
        self.assertFalse(rows[-1]["reset"])
        self.assertTrue(rows[-1].get("planner", {}).get("excluded_goals") or rows[-1].get("fallback"))

    def test_planner_exception_falls_back_to_mapex(self):
        h, d = self.harness()

        def boom(*a, **k):
            raise ValueError("synthetic planner failure")
        h.planner.plan = boom
        mapper = sim_utils.Mapper(self.gt, {"laser_range_m": 20, "num_laser": 2500,
                                            "pixel_per_meter": 10, "dilate_diam_for_planning": 3},
                                  use_distance_transform_for_planning=True)
        cur = np.array([600, 530])
        mapper.observe_and_accumulate_given_pose(cur)
        grid = mapper.get_inflated_planning_maps()
        fr = np.array(sim_utils.FrontierPlanner("visvarprob").get_frontier_centers_given_obs_map(mapper.obs_map)[0])
        costs = np.arange(len(fr), dtype=float)[::-1].copy()
        mean, var = prediction(self.gt_core, mapper.obs_map, np.random.RandomState(0))
        idx, forced = h.choose(0, fr, costs, cur, grid, mapper, mean, var, None)
        self.assertEqual(int(np.argmin(forced)), idx)
        self.assertTrue(np.array_equal(h.goal(fr, idx), fr[idx]))
        # MapEx rule: lowest cost among eligible frontiers
        self.assertEqual(idx, h._mapex_valid(fr, costs, cur, grid))
        stats = json.load(open(os.path.join(d, "topo_policy_stats.json")))
        self.assertEqual(stats["fallbacks_total"], 1)
        self.assertIn("exception", stats["fallbacks"])

    def test_mapex_policy_no_gt_access(self):
        h, d = self.harness("mapex")

        class NoGT(object):
            def __init__(self, m):
                self.obs_map = m.obs_map

            def __getattr__(self, k):
                raise AssertionError("GT/mapper attribute accessed: " + k)
        mapper = sim_utils.Mapper(self.gt, {"laser_range_m": 20, "num_laser": 2500,
                                            "pixel_per_meter": 10, "dilate_diam_for_planning": 3},
                                  use_distance_transform_for_planning=True)
        cur = np.array([600, 530])
        mapper.observe_and_accumulate_given_pose(cur)
        grid = mapper.get_inflated_planning_maps()
        fr = np.array(sim_utils.FrontierPlanner("visvarprob").get_frontier_centers_given_obs_map(mapper.obs_map)[0])
        mean, var = prediction(self.gt_core, mapper.obs_map, np.random.RandomState(0))
        h2, d2 = self.harness("topo")
        for hh in (h, h2):
            idx, _ = hh.choose(0, fr, np.zeros(len(fr)), cur, grid, NoGT(mapper), mean, var, None)
            self.assertIsNotNone(idx)


if __name__ == "__main__":
    unittest.main()
