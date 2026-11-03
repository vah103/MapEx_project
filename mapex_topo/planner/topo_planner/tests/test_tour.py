"""Held-Karp vs brute force (n <= 7) and local-search sanity."""
import os
import sys
import itertools
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from topo_planner import tour  # noqa: E402


def random_model(rng, n, budget, service=True, T=1000.0, t0=None):
    pts = rng.uniform(0, 300, size=(n + 1, 2))
    D = np.linalg.norm(pts[:, None] - pts[None], axis=2) * rng.uniform(1.0, 1.3)
    g = np.concatenate([[0.0], rng.uniform(5, 500, size=n)])
    s = np.concatenate([[0.0], rng.uniform(0, 60, size=n)]) if service else None
    off = s / 2.0 if service else None
    t0 = rng.uniform(0, 600) if t0 is None else t0
    return tour.TourModel(D, g, s, off, t0=t0, T=T, budget=budget)


class TestHeldKarp(unittest.TestCase):
    def test_against_brute_force(self):
        rng = np.random.RandomState(7)
        checked = 0
        for trial in range(120):
            n = int(rng.randint(1, 8))
            budget = [None, rng.uniform(100, 600), rng.uniform(300, 1500)][trial % 3]
            m = random_model(rng, n, budget, service=bool(trial % 2))
            if budget is None:
                m.budget = tour.INF
            nodes = list(range(1, n + 1))
            order, J, dur, info = tour.held_karp(m, nodes, K=64)
            bf_order, bf_J = tour.brute_force(m, nodes)
            self.assertEqual(info["overflow"], 0)
            v = m.value(order) if order else 0.0
            # DP objective equals the clipped evaluation of its own order
            if order:
                self.assertAlmostEqual(v, J, places=6)
            self.assertAlmostEqual(v, bf_J, places=6, msg="trial %d n=%d" % (trial, n))
            checked += 1
        self.assertEqual(checked, 120)

    def test_budget_respected(self):
        rng = np.random.RandomState(3)
        for _ in range(30):
            m = random_model(rng, 6, rng.uniform(50, 300))
            order, J, dur, _ = tour.held_karp(m, list(range(1, 7)))
            self.assertLessEqual(dur, m.budget + 1e-9)
            self.assertAlmostEqual(m.evaluate(order)[1], dur, places=6)

    def test_orienteering_greedy_path(self):
        rng = np.random.RandomState(5)
        m = random_model(rng, 16, 900.0, t0=0.0)
        order, info = tour.orienteering(m, list(range(1, 17)), dp_max=12)
        self.assertEqual(info["solver"], "greedy")
        self.assertGreater(m.value(order), 0)
        self.assertLessEqual(m.evaluate(order)[1], 900.0 + 1e-9)


class TestPrefixHK(unittest.TestCase):
    def test_prefix_reorder_matches_brute_force(self):
        rng = np.random.RandomState(2)
        for trial in range(60):
            n = int(rng.randint(3, 10))
            p = int(rng.randint(2, min(n, 7) + 1))
            # exact when no gain is clipped at T (DP ignores the max(0, .) clip)
            m = random_model(rng, n, None, service=False, t0=0.0, T=1e6)
            m.budget = tour.INF
            order = list(rng.permutation(np.arange(1, n + 1)))
            new, info = tour.reorder_prefix(m, order, p)
            pre, tail = order[:p], order[p:]
            best = max(m.value(list(q) + tail) for q in itertools.permutations(pre))
            self.assertAlmostEqual(max(m.value(new), m.value(order)), max(best, m.value(order)), places=6)
            self.assertEqual(new[p:], tail)


class TestLocalSearch(unittest.TestCase):
    def test_line_order(self):
        # nodes on a line: optimal = visit in increasing distance
        xs = np.array([0, 50, 10, 30, 20, 40], dtype=float)
        D = np.abs(xs[:, None] - xs[None])
        m = tour.TourModel(D, [0, 1, 1, 1, 1, 1], t0=0, T=1000)
        order = tour.greedy_insertion(m, [1, 2, 3, 4, 5])
        order = tour.improve(m, order)
        self.assertEqual(order, [2, 4, 3, 5, 1])

    def test_improve_matches_bf_small(self):
        rng = np.random.RandomState(11)
        worse = 0
        for _ in range(40):
            m = random_model(rng, 6, None, service=False, t0=0.0)
            m.budget = tour.INF
            order = tour.improve(m, tour.greedy_insertion(m, list(range(1, 7))))
            bf = tour.brute_force(m, list(range(1, 7)))[1]
            self.assertLessEqual(m.value(order), bf + 1e-6)
            worse += m.value(order) < bf - 1e-6
        self.assertLessEqual(worse, 8)  # heuristic, mostly optimal


if __name__ == "__main__":
    unittest.main()
