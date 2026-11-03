"""Unified R2 planner on synthetic worlds (no KTH data)."""
import os
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from topo_planner import UnifiedPlanner, R2Config, debug_viz  # noqa: E402
from topo_planner.pc_stub import p_free  # noqa: E402
import synth  # noqa: E402

OUT = os.environ.get("TOPO_TEST_OUT")


def wing_world():
    """corridor + 2x4 rooms (synth) and a large hall ('wing') beyond the corridor end."""
    gt, labels, classes, doors = synth.corridor_world(4)
    H, W = gt.shape
    big = np.ones((H, W + 220), np.float32)
    big[:, :W] = gt
    lab = np.full(big.shape, -1, np.int32)
    lab[:, :W] = labels
    big[90:110, W - 12:W + 10] = 0            # corridor continues through an opening
    big[15:185, W + 10:W + 210] = 0           # the wing (17 x 20 m)
    return big, lab, classes, doors, W


def run(gt, labels, classes, doors, known, robot, tick=0, exclude=(), name=None, chains=None,
        mean=None, cfg=None):
    obs, mean_, var, topo = synth.world_state(gt, labels, classes, doors, known, mean)
    if chains is not None:
        topo["chains"] = chains
    p = p_free(obs, mean_, var)
    pl = UnifiedPlanner(cfg or R2Config())
    t = time.perf_counter()
    res = pl.plan(obs, mean_, var, p, robot, topo, synth.plan_finite(obs), [], tick,
                  exclude_goals=exclude)
    res["wall"] = time.perf_counter() - t
    if OUT and name:
        goal = res["goals"][0][0] if res["goals"] else None
        debug_viz.render(os.path.join(OUT, name + ".png"), obs, topo, robot, res, goal, name, scale=2)
    return res, obs, p


class TestGeometry(unittest.TestCase):
    def test_bfs_to_targets_equals_full_bfs(self):
        from topo_planner import geometry as geo
        rng = np.random.RandomState(4)
        for _ in range(20):
            trav = rng.rand(120, 160) > 0.25
            trav[60, 80] = True
            tg = np.stack([rng.randint(0, 120, 30), rng.randint(0, 160, 30)], 1)
            full = geo.bfs_distance(trav, (60, 80))
            fast = geo.bfs_to_targets(trav, (60, 80), tg)
            self.assertTrue(np.array_equal(fast, full[tg[:, 0], tg[:, 1]]))


class TestR2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = synth.corridor_world(4)
        gt, labels, classes, doors = cls.world
        known = np.zeros(gt.shape, bool)
        known[85:115, 10:120] = True
        run(gt, labels, classes, doors, known, (100, 30))   # numba warm-up

    def test_no_double_counting_and_budget(self):
        gt, labels, classes, doors = self.world
        known = np.zeros(gt.shape, bool)
        known[85:115, 10:120] = True
        res, obs, p = run(gt, labels, classes, doors, known, (100, 100), tick=100, name="r2_start")
        log = res["log"]
        total = float(p[(obs == 0.5) & (p >= 0.05)].sum())
        self.assertTrue(res["goals"])
        self.assertLessEqual(sum(s["g"] for s in log["stops"]), total + 1e-6)
        self.assertLessEqual(log["ticks"], log["budget"] + 1e-6)
        ts = [s["t_pred"] for s in log["stops"]]
        self.assertEqual(ts, sorted(ts))
        self.assertLess(res["wall"], 6.0)
        self.assertIn("hk", log)

    def test_min_distance_3m(self):
        gt, labels, classes, doors = self.world
        known = np.zeros(gt.shape, bool)
        known[85:115, 10:120] = True
        res, _, _ = run(gt, labels, classes, doors, known, (100, 100))
        self.assertFalse(res["log"]["min_dist_relaxed"])
        for q, _ in res["goals"]:
            self.assertGreaterEqual(np.hypot(q[0] - 100, q[1] - 100), 30 - 1e-9)

    def test_relaxes_when_only_near_candidates(self):
        gt, labels, classes, doors = self.world
        known = np.zeros(gt.shape, bool)
        known[85:115, 10:120] = True
        # no candidate lies >= 60 m away on this 30 m map -> nearer ones (>= 1.2 m) are allowed
        res, _, _ = run(gt, labels, classes, doors, known, (100, 100), cfg=R2Config(min_dist_m=60.0))
        self.assertTrue(res["log"]["min_dist_relaxed"])
        self.assertTrue(res["goals"])

    def test_large_unknown_outside_regions_is_targeted(self):
        """KTH50 failure mode: everything known except a big wing outside every L1 region."""
        gt, lab, classes, doors, W = wing_world()
        known = np.zeros(gt.shape, bool)
        known[:, :W - 12] = True
        res, obs, p = run(gt, lab, classes, doors, known, (100, 60), tick=400, name="r2_wing")
        log = res["log"]
        self.assertGreaterEqual(log["rim"]["large"], 1)
        self.assertIn("rim", log["candidate_kinds"])
        first = log["stops"][0]
        self.assertGreater(first["rc"][1], W - 40, "first stop heads to the wing")
        self.assertGreater(sum(s["gain_outside_regions"] for s in log["stops"]), 1000)

    def test_corridor_chain_flag(self):
        gt, labels, classes, doors = self.world
        known = np.zeros(gt.shape, bool)
        known[85:115, 10:120] = True
        chain = {"rc": np.stack([np.full(300, 100), np.arange(15, 315)], 1), "cls": "corridor"}
        res, _, _ = run(gt, labels, classes, doors, known, (100, 60), chains=[chain])
        log = res["log"]
        self.assertTrue(any(k.startswith("chain_corridor") for k in log["candidate_kinds"]))
        self.assertIsNotNone(log["first_stop_on_corridor_chain"])
        self.assertTrue(0.0 <= log["share_stops_on_corridor_chain"] <= 1.0)

    def test_exclusion(self):
        gt, labels, classes, doors = self.world
        known = np.zeros(gt.shape, bool)
        known[85:115, 10:120] = True
        res, _, _ = run(gt, labels, classes, doors, known, (100, 100))
        g0 = res["goals"][0][0]
        res2, _, _ = run(gt, labels, classes, doors, known, (100, 100), exclude=[g0])
        for q, _ in res2["goals"]:
            self.assertGreater((q[0] - g0[0]) ** 2 + (q[1] - g0[1]) ** 2, 9)

    def test_nothing_left(self):
        gt, labels, classes, doors = self.world
        res, _, _ = run(gt, labels, classes, doors, np.ones(gt.shape, bool), (100, 30), tick=500)
        self.assertEqual(res["goals"], [])

    def test_l3_rooms_when_corridor_known(self):
        gt, labels, classes, doors = self.world
        res, _, _ = run(gt, labels, classes, doors, labels == 0, (100, 30), tick=300, name="r2_rooms")
        self.assertTrue(res["goals"])
        kinds = set(s["kind"] for s in res["log"]["stops"])
        self.assertTrue(kinds & {"door", "pocket_centroid", "skeleton", "frontier", "rim"})
        self.assertLessEqual(res["log"]["ticks"], 700 + 1e-6)


if __name__ == "__main__":
    unittest.main()
