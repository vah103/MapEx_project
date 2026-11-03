"""Planner behaviour on the synthetic corridor world."""
import os
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from topo_planner import TopoPlanner, PlannerConfig  # noqa: E402
from topo_planner import debug_viz  # noqa: E402
from topo_planner.visibility import Visibility, predicted_occupancy  # noqa: E402
import synth  # noqa: E402

OUT = os.environ.get("TOPO_TEST_OUT")


def run(gt, labels, classes, doors, known, robot, tick=0, exclude=(), frontiers=(), name=None):
    obs, mean, var, topo = synth.world_state(gt, labels, classes, doors, known)
    pl = TopoPlanner(PlannerConfig())
    t = time.perf_counter()
    res = pl.plan(obs, mean, var, robot, topo, synth.plan_finite(obs), list(frontiers), tick,
                  exclude_goals=exclude)
    res["wall"] = time.perf_counter() - t
    if OUT and name:
        goal = res["goals"][0][0] if res["goals"] else None
        debug_viz.render(os.path.join(OUT, name + ".png"), obs, topo, robot, res, goal,
                         "%s phase=%s" % (name, res["log"]["phase"]), scale=2)
    return res, obs, topo


class TestPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = synth.corridor_world(5)
        gt, labels, _, _ = cls.world
        # warm numba
        known = np.zeros(gt.shape, bool)
        known[85:115, 10:120] = True
        run(*cls.world, known=known, robot=(100, 30))

    def test_visibility_blocked_by_walls(self):
        gt, labels, classes, doors = self.world
        known = np.zeros(gt.shape, bool)
        obs, mean, var, topo = synth.world_state(gt, labels, classes, doors, known)
        gid = np.full(gt.shape, -1, np.int32)
        r, c = np.nonzero(gt == 0)
        gid[r, c] = np.arange(len(r))
        vis = Visibility(predicted_occupancy(obs, mean), gid)
        ids = vis.gain_ids((100, 15))
        seen_lab = labels[r[ids], c[ids]]
        self.assertGreater(np.count_nonzero(seen_lab == 0), 1500)
        # rooms are seen only through doors: far less than room area (4000+ cells)
        for rid in range(1, 11):
            self.assertLess(np.count_nonzero(seen_lab == rid), 1500)
        # nothing behind the wall far from doors
        self.assertTrue(np.all(gt[r[ids], c[ids]] == 0))

    def test_l2_goes_along_corridor(self):
        gt, labels, classes, doors = self.world
        known = np.zeros(gt.shape, bool)
        known[85:115, 10:120] = True
        res, obs, topo = run(gt, labels, classes, doors, known, (100, 100), name="l2_start")
        log = res["log"]
        self.assertEqual(log["phase"], "L2")
        goal = res["goals"][0][0]
        self.assertEqual(labels[goal], 0, "first stop on the backbone (corridor)")
        self.assertGreater(goal[1], 100, "goal heads towards the unknown part of the corridor")
        self.assertLess(res["wall"], 5.0)
        self.assertIn("lambda", log["l2"])
        ts = [s["t_pred"] for s in log["l2"]["stops"]]
        self.assertEqual(ts, sorted(ts))

    def test_l3_when_backbone_known(self):
        gt, labels, classes, doors = self.world
        known = labels == 0
        res, obs, topo = run(gt, labels, classes, doors, known, (100, 30), tick=300,
                             name="l3_rooms")
        log = res["log"]
        self.assertEqual(log["phase"], "L3")
        self.assertEqual(log["l3"]["solver"]["solver"], "held_karp")
        first = log["l3"]["stops"][0]
        self.assertIn(first["pocket"], range(1, 11))
        # nearest rooms first: first pocket is one of the two rooms closest to robot
        self.assertIn(first["pocket"], (1, 6))
        self.assertLessEqual(log["l3"]["ticks"], log["l3"]["budget"] + 1e-6)
        self.assertLess(res["wall"], 5.0)

    def test_l3_budget_limits_tour(self):
        gt, labels, classes, doors = self.world
        known = labels == 0
        res, _, _ = run(gt, labels, classes, doors, known, (100, 30), tick=900)
        l3 = res["log"]["l3"]
        self.assertLessEqual(l3["ticks"], 100 + 1e-6)
        self.assertLess(len(l3["stops"]), 10)

    def test_l3_greedy_many_pockets(self):
        world = synth.corridor_world(8)
        gt, labels, classes, doors = world
        known = labels == 0
        res, _, _ = run(gt, labels, classes, doors, known, (100, 30), tick=200, name="l3_greedy")
        self.assertEqual(res["log"]["l3"]["solver"]["solver"], "greedy")
        self.assertTrue(res["goals"])
        self.assertLess(res["wall"], 5.0)

    def test_pocket_insertion_cheap_room(self):
        gt, labels, classes, doors = self.world
        known = np.zeros(gt.shape, bool)
        known[85:115, 10:200] = True     # corridor start known, door of room 1/6 at col 60
        res, _, _ = run(gt, labels, classes, doors, known, (100, 60), name="l2_insert")
        l2 = res["log"]["l2"]
        self.assertGreater(l2["pocket_candidates"], 0)
        for ins in l2["pocket_insertions"]:
            self.assertGreater(ins["ratio"], ins["lambda"])
        self.assertTrue(l2["pocket_insertions"], "adjacent rooms are cheaper than tour rate")

    def test_exclusion_respected(self):
        gt, labels, classes, doors = self.world
        known = np.zeros(gt.shape, bool)
        known[85:115, 10:120] = True
        res, _, _ = run(gt, labels, classes, doors, known, (100, 30))
        g0 = res["goals"][0][0]
        res2, _, _ = run(gt, labels, classes, doors, known, (100, 30), exclude=[g0])
        for q, _ in res2["goals"]:
            self.assertGreater((q[0] - g0[0]) ** 2 + (q[1] - g0[1]) ** 2, 9)

    def test_goals_valid(self):
        gt, labels, classes, doors = self.world
        known = np.zeros(gt.shape, bool)
        known[85:115, 10:120] = True
        res, obs, _ = run(gt, labels, classes, doors, known, (100, 30))
        pf = synth.plan_finite(obs)
        for q, _ in res["goals"]:
            self.assertTrue(pf[q])
            self.assertGreaterEqual(np.hypot(q[0] - 100, q[1] - 30), 12)

    def test_nothing_left_returns_empty(self):
        gt, labels, classes, doors = self.world
        known = np.ones(gt.shape, bool)
        res, _, _ = run(gt, labels, classes, doors, known, (100, 30), tick=500)
        self.assertEqual(res["goals"], [])
        self.assertEqual(res["log"]["phase"], "NONE")


if __name__ == "__main__":
    unittest.main()
