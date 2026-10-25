"""Viewpoint coverage-tour planner on an occupancy grid (0.1 m cells), for offline feasibility tests.
occ: bool grid of obstacles used for planning (GT or predicted). Robot traverses cells that are free and not
adjacent (3x3) to an obstacle, like MapEx's 3x3 inflation. LiDAR: NR rays, range 20 m, stops at first obstacle."""
import heapq
import numpy as np, cv2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

NRAY, RANGE, STEP = 1440, 200, 0.7
_ang = np.linspace(0, 2 * np.pi, NRAY, endpoint=False)
_s = np.arange(0, RANGE, STEP)
_DR = np.outer(np.sin(_ang), _s); _DC = np.outer(np.cos(_ang), _s)

def visible(occ, r, c):
    """flat indices of cells hit by rays from (r,c) on obstacle grid occ (incl. first obstacle cell)."""
    H, W = occ.shape
    rr = np.rint(r + _DR).astype(np.int32); cc = np.rint(c + _DC).astype(np.int32)
    out = (rr < 0) | (rr >= H) | (cc < 0) | (cc >= W)
    rr = np.clip(rr, 0, H - 1); cc = np.clip(cc, 0, W - 1)
    hit = occ[rr, cc] | out
    first = np.where(hit.any(1), hit.argmax(1), hit.shape[1])
    m = (np.arange(hit.shape[1])[None, :] <= first[:, None]) & ~out
    return np.unique((rr * W + cc)[m]).astype(np.int32)

class Graph:
    def __init__(self, occ, start_rc):
        H, W = occ.shape
        trav = ~(cv2.dilate(occ.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        idx = -np.ones(H * W, np.int64); cells = np.flatnonzero(trav.ravel()); idx[cells] = np.arange(len(cells))
        rows, cols, w = [], [], []
        r0, c0 = np.divmod(cells, W)
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            r1, c1 = r0 + dr, c0 + dc
            ok = (r1 < H) & (c1 >= 0) & (c1 < W)
            j = idx[(r1[ok] * W + c1[ok])]
            i = np.arange(len(cells))[ok]
            k = j >= 0
            rows += [i[k], j[k]]; cols += [j[k], i[k]]; w += [np.full(k.sum(), np.hypot(dr, dc))] * 2
        G = csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(len(cells), len(cells)))
        s = self.node_near(idx, W, H, start_rc, trav)
        _, lab = connected_components(G, directed=False)
        keep = lab == lab[s]
        self.H, self.W, self.G, self.cells, self.idx, self.keep = H, W, G, cells, idx, keep
    @staticmethod
    def node_near(idx, W, H, rc, trav):
        r, c = int(rc[0]), int(rc[1])
        if idx[r * W + c] >= 0:
            return idx[r * W + c]
        ys, xs = np.nonzero(trav)
        k = np.argmin((ys - r) ** 2 + (xs - c) ** 2)
        return idx[ys[k] * W + xs[k]]
    def node(self, rc):
        return self.node_near(self.idx, self.W, self.H, rc, (self.idx >= 0).reshape(self.H, self.W))
    def rc(self, n):
        return divmod(int(self.cells[n]), self.W)

def candidates(g, stride):
    r, c = np.divmod(g.cells[g.keep], g.W); rc = np.stack([r, c], 1)
    sel = (rc[:, 0] % stride == 0) & (rc[:, 1] % stride == 0)
    return rc[sel]

def greedy_cover(vis_list, need, min_gain):
    """need: bool flat array of cells still to see. returns chosen candidate indices."""
    need = need.copy(); heap = [(-int(need[v].sum()), i) for i, v in enumerate(vis_list)]
    heapq.heapify(heap); chosen = []
    while heap:
        g, i = heapq.heappop(heap)
        cur = int(need[vis_list[i]].sum())
        if cur < min_gain:
            if -g < min_gain: break
            continue
        if heap and cur < -heap[0][0]:
            heapq.heappush(heap, (-cur, i)); continue
        chosen.append(i); need[vis_list[i]] = False
    return chosen

def open_tsp(D):
    """D: (n+1)x(n+1), node 0 = start. nearest neighbour + 2-opt on open path. returns order of 1..n"""
    n = D.shape[0]
    if n <= 1: return []
    left = set(range(1, n)); path = [0]
    while left:
        j = min(left, key=lambda k: D[path[-1], k]); path.append(j); left.remove(j)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b = path[i - 1], path[i]
                c = path[k]; d = path[k + 1] if k + 1 < n else None
                old = D[a, b] + (D[c, d] if d is not None else 0)
                new = D[a, c] + (D[b, d] if d is not None else 0)
                if new < old - 1e-9:
                    path[i:k + 1] = path[i:k + 1][::-1]; improved = True
    return path[1:]

def walk(g, src, dst, pred_row):
    seq = [dst]
    while seq[-1] != src:
        seq.append(pred_row[seq[-1]])
        if seq[-1] < 0: return None
    return seq[::-1]

def run_tour(occ_plan, occ_true, start_rc, target, known0=None, stride=5, min_gain=10, replan=True, order_hint=None,
             max_vp=10 ** 9, tick=3, log_every=None):
    """Plan on occ_plan, execute/observe on occ_true. target: bool grid of cells that count.
    Returns list of (distance_m, known_target_count). If replan=False: plan once and execute order (skip unreachable)."""
    H, W = occ_true.shape
    gp = Graph(occ_plan, start_rc)
    gt = gp if occ_plan is occ_true else Graph(occ_true, start_rc)
    seen = np.zeros(H * W, bool) if known0 is None else known0.ravel().copy()
    tflat = target.ravel()
    cand = candidates(gp, stride)
    vis_plan = [v[tflat[v]] for v in (visible(occ_plan, r, c) for r, c in cand)]
    pos = np.array(start_rc); dist = 0.0
    seen[visible(occ_true, *pos)] = True
    curve = [(0.0, int((seen & tflat).sum()))]
    def observe_path(nodes):
        nonlocal dist, pos
        rcs = np.array([gt.rc(n) for n in nodes])
        seg = np.hypot(*np.diff(rcs, axis=0).T) if len(rcs) > 1 else np.zeros(0)
        acc = 0.0
        for i in range(1, len(rcs)):
            acc += seg[i - 1]
            if i % tick == 0 or i == len(rcs) - 1:
                seen[visible(occ_true, *rcs[i])] = True
        dist += acc * 0.1; pos = rcs[-1]
        curve.append((dist, int((seen & tflat).sum())))
    order = None; steps = 0
    while True:
        need = tflat & ~seen
        if order is None or replan:
            ch = greedy_cover(vis_plan, need, min_gain)
            if not ch: break
            pts = cand[ch]
            nodes = [gp.node(pos)] + [gp.node(p) for p in pts]
            D, _ = dijkstra(gp.G, indices=nodes, return_predecessors=True)
            D = D[:, nodes]
            order = [pts[i - 1] for i in open_tsp(D)]
        if not order: break
        nxt = order.pop(0)
        if not replan and not (need[visible(occ_plan, *nxt)]).any():
            continue
        s = gt.node(pos); t = gt.node(nxt)
        d, pr = dijkstra(gt.G, indices=s, return_predecessors=True)
        if not np.isfinite(d[t]):
            continue
        observe_path(walk(gt, s, t, pr))
        steps += 1
        if steps >= max_vp: break
        if not replan and not order: break
    return curve

def dist_at(curve, count):
    for d, k in curve:
        if k >= count: return d
    return None
