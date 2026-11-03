"""Predicted-map visibility with the MapEx raycast (TOPO-001 §L2 Visibility).

Rays: MapEx `simple_mask_utils.get_free_points` semantics (250 rays, 20 m, binary
hits) via `free_points_binary`, a list-free numba port that is bit-identical
to the MapEx function (tests/test_visibility.py) and ~10x faster;
cast on the PREDICTED occupancy (LaMa mean >= 0.5 occupied; observed cells
override). As in MapEx `get_vis_mask`, the visible set is the polygon spanned
by the ray end points (filled here with cv2.fillPoly instead of the full-map
shapely+flood fill, same semantics, ~100x cheaper) united with the ray cells.
Only "gain cells" (currently UNKNOWN and predicted free) are returned, as ids.
"""
import os
import sys

import cv2
import numpy as np
from numba import njit

_SMU = None


def _mapex_smu():
    """The MapEx module (already PAPER-patched by the worker when running native)."""
    global _SMU
    if _SMU is None:
        try:
            from scripts import simple_mask_utils as smu  # native worker path
        except Exception:
            root = os.environ.get("MAPEX_ROOT", "/home/dell/MapEx")
            for p in (root, os.path.join(root, "scripts")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            from scripts import simple_mask_utils as smu
        _SMU = smu
    return _SMU


@njit(cache=True)
def _bresenham_into(x1, y1, x2, y2, out):
    """MapEx `simple_mask_utils.bresenham`, same point sequence, no list building."""
    dx = x2 - x1
    dy = y2 - y1
    steep = abs(dy) > abs(dx)
    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1
    dy = y2 - y1
    error = int(dx / 2.0)
    y_step = 1 if y1 < y2 else -1
    y = y1
    n = 0
    for x in range(x1, x2 + 1):
        if steep:
            out[n, 0] = y
            out[n, 1] = x
        else:
            out[n, 0] = x
            out[n, 1] = y
        n += 1
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:
        for i in range(n // 2):
            a0, a1 = out[i, 0], out[i, 1]
            out[i, 0], out[i, 1] = out[n - 1 - i, 0], out[n - 1 - i, 1]
            out[n - 1 - i, 0], out[n - 1 - i, 1] = a0, a1
    return n


@njit(cache=True)
def free_points_binary(occupancy_grid, robot_pos, laser_range, num_laser):
    """Bit-identical to MapEx get_free_points(..., use_prob=False) (tested)."""
    max_points = num_laser * laser_range
    free_points = np.empty((max_points, 2), dtype=np.int64)
    hit_points = np.empty((num_laser, 2), dtype=np.int64)
    actual_hit_points = np.empty((num_laser, 2), dtype=np.int64)
    free_idx, hit_idx, actual_hit_idx = 0, 0, 0
    orientation = np.linspace(0, 2 * np.pi, num_laser)
    end_points = np.zeros((num_laser, 2), dtype=np.int64)
    end_points[:, 0] = robot_pos[0] + laser_range * np.cos(orientation)
    end_points[:, 1] = robot_pos[1] + laser_range * np.sin(orientation)
    buf = np.empty((4 * laser_range + 8, 2), dtype=np.int64)
    for e in range(num_laser):
        n = _bresenham_into(robot_pos[0], robot_pos[1], end_points[e, 0], end_points[e, 1], buf)
        for i in range(n):
            x = buf[i, 0]
            y = buf[i, 1]
            if x < 1 or x >= occupancy_grid.shape[0] - 1 or y < 1 or y >= occupancy_grid.shape[1] - 1:
                if i != 0:
                    hit_points[hit_idx, 0] = buf[i - 1, 0]
                    hit_points[hit_idx, 1] = buf[i - 1, 1]
                    hit_idx += 1
                break
            if occupancy_grid[x, y] == 1:
                actual_hit_points[actual_hit_idx, 0] = x
                actual_hit_points[actual_hit_idx, 1] = y
                actual_hit_idx += 1
                hit_points[hit_idx, 0] = x
                hit_points[hit_idx, 1] = y
                hit_idx += 1
                break
            else:
                free_points[free_idx, 0] = x
                free_points[free_idx, 1] = y
                free_idx += 1
                if i == n - 1:
                    hit_points[hit_idx, 0] = end_points[e, 0]
                    hit_points[hit_idx, 1] = end_points[e, 1]
                    hit_idx += 1
                    break
    return free_points[:free_idx], hit_points[:hit_idx], actual_hit_points[:actual_hit_idx]


def predicted_occupancy(obs, mean):
    occ = (obs == 1.0) | ((obs == 0.5) & (mean >= 0.5))
    return occ.astype(np.float64)


class Visibility(object):
    def __init__(self, occ, gain_id, range_cells=200, num_rays=250):
        self.occ = np.ascontiguousarray(occ, dtype=np.float64)
        self.gain_id = gain_id  # int32 map, -1 = not a gain cell
        self.R = int(range_cells)
        self.n = int(num_rays)
        self.cache = {}
        self.calls = 0

    def visible_mask_window(self, rc):
        """(r0, c0, mask) of visible cells in a (2R+1)^2 window around rc."""
        r, c = int(rc[0]), int(rc[1])
        h, w = self.occ.shape
        free, hit, _ = free_points_binary(self.occ, np.array([r, c], dtype=np.int64),
                                          self.R, self.n)
        r0, c0 = max(0, r - self.R - 1), max(0, c - self.R - 1)
        r1, c1 = min(h, r + self.R + 2), min(w, c + self.R + 2)
        m = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
        if len(hit) >= 3:
            poly = np.stack([hit[:, 1] - c0, hit[:, 0] - r0], axis=1).astype(np.int32)
            cv2.fillPoly(m, [poly.reshape(-1, 1, 2)], 1)
        if len(free):
            fr, fc = free[:, 0] - r0, free[:, 1] - c0
            ok = (fr >= 0) & (fr < m.shape[0]) & (fc >= 0) & (fc < m.shape[1])
            m[fr[ok], fc[ok]] = 1
        if len(hit):
            hr, hc = hit[:, 0] - r0, hit[:, 1] - c0
            ok = (hr >= 0) & (hr < m.shape[0]) & (hc >= 0) & (hc < m.shape[1])
            m[hr[ok], hc[ok]] = 1
        return r0, c0, m.astype(bool)

    def gain_ids(self, rc):
        key = (int(rc[0]), int(rc[1]))
        got = self.cache.get(key)
        if got is not None:
            return got
        self.calls += 1
        if self.occ[key] == 1.0:
            ids = np.zeros(0, dtype=np.int32)
        else:
            r0, c0, m = self.visible_mask_window(key)
            sub = self.gain_id[r0:r0 + m.shape[0], c0:c0 + m.shape[1]]
            ids = sub[m & (sub >= 0)]
            ids = np.unique(ids).astype(np.int32)
        self.cache[key] = ids
        return ids
