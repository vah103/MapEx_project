"""Grid geometry: 4-connected geodesic distance fields (BFS), helpers.

Travel model: native MapEx moves 3 cells per tick along a 4-connected A* path,
so ticks = path cells / 3. A* uses distance-transform weights (1..10); BFS gives
the shortest 4-connected length, i.e. a lower bound of the native path length.
"""
import numpy as np
from numba import njit

CELLS_PER_TICK = 3.0
UNREACH = np.int32(-1)


@njit(cache=True)
def _bfs(trav, sr, sc, max_d):
    h, w = trav.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    if sr < 0 or sr >= h or sc < 0 or sc >= w:
        return dist
    q = np.empty(h * w, dtype=np.int32)
    head = 0
    tail = 0
    dist[sr, sc] = 0
    q[tail] = sr * w + sc
    tail += 1
    while head < tail:
        idx = q[head]
        head += 1
        r = idx // w
        c = idx - r * w
        d = dist[r, c]
        if max_d >= 0 and d >= max_d:
            continue
        for k in range(4):
            if k == 0:
                rr, cc = r - 1, c
            elif k == 1:
                rr, cc = r + 1, c
            elif k == 2:
                rr, cc = r, c - 1
            else:
                rr, cc = r, c + 1
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            if not trav[rr, cc] or dist[rr, cc] >= 0:
                continue
            dist[rr, cc] = d + 1
            q[tail] = rr * w + cc
            tail += 1
    return dist


@njit(cache=True)
def _bfs_targets(trav, sr, sc, tr_, tc_):
    """BFS that stops once every target cell has been reached (or the component is exhausted)."""
    h, w = trav.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    is_t = np.zeros((h, w), dtype=np.bool_)
    left = 0
    for i in range(len(tr_)):
        if not is_t[tr_[i], tc_[i]]:
            is_t[tr_[i], tc_[i]] = True
            left += 1
    q = np.empty(h * w, dtype=np.int32)
    head = 0
    tail = 0
    dist[sr, sc] = 0
    if is_t[sr, sc]:
        left -= 1
    q[tail] = sr * w + sc
    tail += 1
    while head < tail and left > 0:
        idx = q[head]
        head += 1
        r = idx // w
        c = idx - r * w
        d = dist[r, c]
        for k in range(4):
            if k == 0:
                rr, cc = r - 1, c
            elif k == 1:
                rr, cc = r + 1, c
            elif k == 2:
                rr, cc = r, c - 1
            else:
                rr, cc = r, c + 1
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            if not trav[rr, cc] or dist[rr, cc] >= 0:
                continue
            dist[rr, cc] = d + 1
            if is_t[rr, cc]:
                left -= 1
            q[tail] = rr * w + cc
            tail += 1
    out = np.empty(len(tr_), dtype=np.int32)
    for i in range(len(tr_)):
        out[i] = dist[tr_[i], tc_[i]]
    return out


def bfs_to_targets(trav, start_rc, targets_rc):
    """Geodesic cells from start to each target (-1 unreachable), early-terminating BFS."""
    t = np.asarray(targets_rc, dtype=np.int64).reshape(-1, 2)
    trav = np.ascontiguousarray(trav, dtype=np.bool_)
    return _bfs_targets(trav, int(start_rc[0]), int(start_rc[1]), t[:, 0].copy(), t[:, 1].copy())


def bfs_distance(trav, start_rc, max_d=-1):
    """Geodesic 4-connected distance in cells from start_rc (start always seeded)."""
    trav = np.ascontiguousarray(trav, dtype=np.bool_)
    return _bfs(trav, int(start_rc[0]), int(start_rc[1]), int(max_d))


def ticks(cells):
    return float(cells) / CELLS_PER_TICK


def snap_to_mask(rc, mask, radius):
    """Nearest True cell of mask within radius (Euclid), or None."""
    r, c = int(rc[0]), int(rc[1])
    h, w = mask.shape
    if 0 <= r < h and 0 <= c < w and mask[r, c]:
        return (r, c)
    r0, r1 = max(0, r - radius), min(h, r + radius + 1)
    c0, c1 = max(0, c - radius), min(w, c + radius + 1)
    sub = mask[r0:r1, c0:c1]
    ys, xs = np.nonzero(sub)
    if not len(ys):
        return None
    d = (ys + r0 - r) ** 2 + (xs + c0 - c) ** 2
    k = int(np.argmin(d))
    if d[k] > radius * radius:
        return None
    return (int(ys[k] + r0), int(xs[k] + c0))


def subsample_points(points_rc, spacing):
    """Greedy thinning: keep points in given order with pairwise spacing >= spacing."""
    pts = np.asarray(points_rc, dtype=np.int64).reshape(-1, 2)
    if not len(pts):
        return pts
    cell = max(1, int(spacing))
    grid = {}
    keep = []
    s2 = spacing * spacing
    for p in pts:
        key = (int(p[0]) // cell, int(p[1]) // cell)
        ok = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                for q in grid.get((key[0] + dr, key[1] + dc), ()):
                    if (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 < s2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault(key, []).append((int(p[0]), int(p[1])))
            keep.append((int(p[0]), int(p[1])))
    return np.asarray(keep, dtype=np.int64).reshape(-1, 2)
