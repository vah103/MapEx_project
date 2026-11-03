"""free_points_binary must equal MapEx get_free_points (binary mode) exactly."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from topo_planner.visibility import free_points_binary, _mapex_smu  # noqa: E402


class TestRaycastParity(unittest.TestCase):
    def test_bit_identical(self):
        smu = _mapex_smu()
        rng = np.random.RandomState(1)
        for trial in range(60):
            H, W = rng.randint(60, 420), rng.randint(60, 420)
            occ = (rng.rand(H, W) < rng.choice([0.0, 0.002, 0.02])).astype(np.float64)
            occ[rng.randint(0, H), :] = 1.0
            pos = np.array([rng.randint(1, H - 1), rng.randint(1, W - 1)], dtype=np.int64)
            occ[pos[0], pos[1]] = 0.0
            R = int(rng.choice([50, 200]))
            n = int(rng.choice([100, 250]))
            a = smu.get_free_points(occ, pos, laser_range=R, num_laser=n)
            b = free_points_binary(occ, pos, R, n)
            for x, y in zip(a, b):
                self.assertTrue(np.array_equal(np.asarray(x).reshape(-1, 2), y), trial)


if __name__ == "__main__":
    unittest.main()
