"""Synthetic corridor-with-rooms world for planner tests (CORE frame, 10 cells/m)."""
import numpy as np
from scipy.ndimage import binary_dilation

from topo_planner.l1_stub import topomap_from_labels


def corridor_world(n_rooms_per_side=5, room_w=60, length=None):
    """GT occ (1 wall, 0 free), labels, classes, doors. Corridor rows 90..109."""
    length = length or (40 + n_rooms_per_side * (room_w + 2))
    H, W = 200, length + 20
    gt = np.ones((H, W), dtype=np.float32)
    labels = np.full((H, W), -1, dtype=np.int32)
    gt[90:110, 10:W - 10] = 0
    labels[90:110, 10:W - 10] = 0
    classes = {0: "corridor"}
    doors = []
    rid = 1
    for side in (0, 1):
        for i in range(n_rooms_per_side):
            c0 = 30 + i * (room_w + 2)
            c1 = c0 + room_w
            r0, r1 = (20, 89) if side == 0 else (111, 180)
            gt[r0:r1, c0:c1] = 0
            labels[r0:r1, c0:c1] = rid
            dc = c0 + room_w // 2
            dr = 89 if side == 0 else 110
            gt[dr, dc - 6:dc + 6] = 0
            labels[dr, dc - 6:dc + 6] = rid
            classes[rid] = "room"
            doors.append((rid, (dr, dc), (0, rid)))
            rid += 1
    return gt, labels, classes, doors


def observe(gt, known_mask):
    obs = np.full(gt.shape, 0.5, dtype=np.float32)
    obs[known_mask] = gt[known_mask]
    return obs


def plan_finite(obs):
    occ = binary_dilation(obs > 0.5, structure=np.ones((3, 3), bool))
    return ~occ


def world_state(gt, labels, classes, doors, known_mask, mean=None):
    obs = observe(gt, known_mask)
    mean = gt.copy() if mean is None else mean
    var = np.full(gt.shape, 0.01, dtype=np.float32)
    topo = topomap_from_labels(labels, classes, doors, 10.0, obs, mean)
    return obs, mean, var, topo
