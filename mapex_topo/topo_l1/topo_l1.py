"""TOPO-002 L1 v2 (structural role): LaMa topological map -> regions, doors, skeleton chains, viewpoints.
Derived from TOPO-001 L1 v1 (build/topo001/topo_l1, kept unchanged). v2 changes (TOPO-002 §2.2):
- free space of a hypothesis from p(c) (`pfree` array, pc_model) when given; else v1 rule (mean < theta, range 3 m)
- predicted-wall gap closing (mean >= 0.7, gaps <= 0.6 m) on UNKNOWN cells
- door evidence from wall jambs (facing wall ends 0.7-2.4 m apart) in addition to clearance minima
- corridor = region whose skeleton chain (longest skeleton path) >= 6 m with >= 2 doors and width 1.0-3.5 m
- outputs `chains` and `viewpoints` (chain points every 1.5 m, doors, pocket interior points <= 4)

v1 docstring follows.
TOPO-001 L1: LaMa topological map (regions / doors / corridors / rooms / halls).

New code (no Room Node V4 reuse). Spec: ~/mapex_workspace/tasks/TOPO-001.md §L1.

Conventions
- obs: 0 FREE, 1 OCC, 0.5 UNKNOWN. mean: LaMa ensemble occupancy probability (1 = occupied).
- All rc (input robot_rc and every output rc) are in the frame of the arrays passed in
  (core or padded; padded input is cropped internally to the useful bbox for speed).
- ppm: cells per metre (MapEx: 10).

Pipeline per hypothesis h (theta in THETAS):
  free_h = obs FREE | (UNKNOWN & mean < theta), restricted to the robot's 4-connected component
  -> EDT -> thinned skeleton, spurs < SPUR_M pruned
  -> doors = skeleton critical points (width in [DOOR_MIN_M, DOOR_MAX_M], local minimum,
     both sides widen >= WIDEN x within WIDEN_WIN_M) -> 4-connected cut lines -> regions
  -> per-region features and rule class (corridor / room / hall).
Fusion: doors clustered across hypotheses (<= 1 m) -> p_exist; the reference hypothesis
(theta = 0.5) free space is cut at doors with p_exist >= 0.5; region class = vote of the
best-overlapping region of every hypothesis. Temporal: IoU association to prev, EMA, hysteresis.
"""
import time
from collections import deque
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, remove_small_holes

# ---- fixed parameters (spec §L1 / Fixed parameters; changes logged in eval_l1/PARAM_CHANGES.md)
PARAMS = dict(
    THETAS=(0.35, 0.50, 0.65), REF_THETA_IDX=1,
    PF_THETAS=(0.65, 0.50, 0.35),   # with pfree: predicted-free if p(c) >= value (ordered like THETAS: strict->loose)
    GAP_CLOSE_M=0.6, GAP_WALL_MEAN=0.7,
    JAMB=True, JAMB_MIN_M=0.7, JAMB_MAX_M=2.4, JAMB_WALL_MEAN=0.7, JAMB_COS=0.7, JAMB_FREE_FRAC=0.8,
    CHAIN_CORR_M=6.0, CHAIN_CORR_DOORS=2, VP_SPACING_M=1.5, POCKET_VP_MAX=4, POCKET_VP_M2=25.0,
    VAR_GATE=None,            # optional: max variance for predicted-free (None = off, spec default)
    PRED_RANGE_M=3.0,         # v2: used only when no pfree is given (v1 behaviour); optional: predicted-free only within this distance of observed FREE
    OPEN_PRED_M=0.3,          # v1 preset value kept;: morphological opening radius applied to predicted-free cells
    WIDEN_MODE='skeleton',   # 'skeleton' (v0/v1: path along skeleton) | 'euclid' (v2: radius around door)
    SPUR_M=1.0, DOOR_MIN_M=0.7, DOOR_MAX_M=1.6, WIDEN=1.5, WIDEN_WIN_M=1.5,
    LOCALMIN_WIN_M=0.5, DOOR_NMS_M=1.0, DOOR_FUSE_M=1.0, DOOR_PEXIST_CUT=0.5,
    MIN_REGION_M2=1.5, HOLE_FILL_M2=0.09,
    CORR_RATIO=3.0, CORR_SKEL_M=6.0, CORR_CV=0.35, CORR_W_MIN=1.0, CORR_W_MAX=3.5,
    HALL_AREA_M2=40.0, HALL_DEG=3,
    IOU_ASSOC=0.3, EMA_ALPHA=0.5, HYST_HI=0.6, HYST_LO=0.4, BACKBONE_P=0.5, BACKBONE_DEG=3,
)
# Presets evaluated offline (eval_l1/REPORT.md). Default PARAMS = spec (v0).
PRESETS = {
    'v2': {},                                            # defaults
    'v2_nojamb': dict(JAMB=False),
    'v1_compat': dict(JAMB=False, GAP_CLOSE_M=0, CHAIN_CORR_M=None),
}
CLASSES = ("corridor", "room", "hall")
S4 = ndi.generate_binary_structure(2, 1)
S8 = np.ones((3, 3), bool)
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class TopoMap:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return "TopoMap(%d regions, %d doors, %.2fs)" % (len(self.regions), len(self.doors), self.timing_s)


# --------------------------------------------------------------------------------------- skeleton

def _nbr_count(sk):
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    return cv2.filter2D(sk.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) * sk


def _neighbors(sk, r, c):
    h, w = sk.shape
    out = []
    for dr, dc in NB8:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and sk[rr, cc]:
            out.append((rr, cc))
    return out


def _prune_spurs(sk, min_len):
    """Remove end branches (endpoint -> junction) shorter than min_len cells; isolated short pieces too."""
    sk = sk.copy()
    for _ in range(3):
        nb = _nbr_count(sk)
        ends = np.argwhere(sk & (nb == 1))
        removed = 0
        for r, c in ends:
            if not sk[r, c]:
                continue
            path = [(r, c)]; prev = None; cur = (r, c); hit_junction = False
            while len(path) <= min_len:
                nbs = [p for p in _neighbors(sk, *cur) if p != prev and p not in path]
                if not nbs:
                    break
                if len(nbs) > 1 or nb[nbs[0]] >= 3:
                    hit_junction = True
                    break
                prev, cur = cur, nbs[0]
                path.append(cur)
            if len(path) < min_len and (hit_junction or len(path) <= min_len):
                for p in path:
                    sk[p] = False
                removed += 1
        if removed == 0:
            break
    # isolated pixels
    sk &= _nbr_count(sk) > 0
    return sk


# --------------------------------------------------------------------------------------- doors

def _skel_graph(sk):
    """Skeleton pixels as nodes; 8-neighbour adjacency lists (CSR -> python lists)."""
    pts = np.argwhere(sk)
    idx = -np.ones(sk.shape, np.int64); idx[pts[:, 0], pts[:, 1]] = np.arange(len(pts))
    h, w = sk.shape
    I = []; J = []
    for dr, dc in NB8:
        rr = pts[:, 0] + dr; cc = pts[:, 1] + dc
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        j = np.full(len(pts), -1); j[ok] = idx[rr[ok], cc[ok]]
        m = j >= 0
        I.append(np.nonzero(m)[0]); J.append(j[m])
    I = np.concatenate(I); J = np.concatenate(J)
    o = np.argsort(I, kind='stable'); I = I[o]; J = J[o]
    splits = np.searchsorted(I, np.arange(1, len(pts)))
    adj = [a.tolist() for a in np.split(J, splits)]
    return pts, idx, adj


def _side_bfs(adj, wv, start, block, maxd):
    """BFS along skeleton from start (dist 1) not through block. Returns list of (node, dist)."""
    seen = {block, start}; q = deque([(start, 1)]); out = []
    while q:
        p, d = q.popleft(); out.append((p, d))
        if d >= maxd:
            continue
        for n in adj[p]:
            if n not in seen:
                seen.add(n); q.append((n, d + 1))
    return out


def _detect_doors(free, dt, sk, res, P):
    """Critical points on skeleton. Returns list of dicts {rc, width_m, tangent, dt}."""
    wmap = np.maximum(2.0 * dt - 1.0, 0) * res
    pts, idx, adj = _skel_graph(sk)
    if len(pts) == 0:
        return []
    wv = wmap[pts[:, 0], pts[:, 1]]
    deg = np.array([len(a) for a in adj])
    cand = np.nonzero((deg == 2) & (wv >= P['DOOR_MIN_M']) & (wv <= P['DOOR_MAX_M']))[0]
    win = int(round(P['WIDEN_WIN_M'] / res)); lmw = int(round(P['LOCALMIN_WIN_M'] / res))
    out = []
    for i in cand:
        w0 = wv[i]
        a0, a1 = adj[i]
        # cheap local-minimum test first
        loc = _side_bfs(adj, wv, a0, i, lmw) + _side_bfs(adj, wv, a1, i, lmw)
        if min(wv[n] for n, _ in loc) < w0 - 1e-6:
            continue
        ok = True; near = [i]
        sides = []
        for s in (a0, a1):
            side = _side_bfs(adj, wv, s, i, win)
            sides.append(side)
            near += [n for n, d in side if d <= 4]
            if P.get('WIDEN_MODE', 'skeleton') == 'skeleton' and max(wv[n] for n, _ in side) < P['WIDEN'] * w0:
                ok = False
                break
        if not ok:
            continue
        a = pts[near].astype(float)
        if len(a) >= 3:
            a = a - a.mean(0)
            _, _, vt = np.linalg.svd(a, full_matrices=False)
            t = vt[0]
        else:
            t = np.array([1.0, 0.0])
        if P.get('WIDEN_MODE', 'skeleton') == 'euclid':
            # widening measured as the max local width (2*EDT-1) of free cells within a Euclidean radius
            # WIDEN_WIN_M of the door point, separately on each side of the door line (|proj| >= 0.3 m)
            r, c = pts[i]
            r0, r1 = max(0, r - win), min(free.shape[0], r + win + 1)
            c0, c1 = max(0, c - win), min(free.shape[1], c + win + 1)
            rr, cc = np.mgrid[r0:r1, c0:c1]
            disk = ((rr - r) ** 2 + (cc - c) ** 2 <= win * win) & free[r0:r1, c0:c1]
            proj = (rr - r) * t[0] + (cc - c) * t[1]
            wl = wmap[r0:r1, c0:c1]
            m3 = int(round(0.3 / res))
            sa = wl[disk & (proj >= m3)]; sb = wl[disk & (proj <= -m3)]
            if not len(sa) or not len(sb) or min(sa.max(), sb.max()) < P['WIDEN'] * w0:
                continue
        r, c = pts[i]
        out.append(dict(rc=(int(r), int(c)), width_m=float(w0), tangent=t, dt=float(dt[r, c])))
    # NMS: narrowest first
    out.sort(key=lambda d: d['width_m'])
    keep = []
    rad = P['DOOR_NMS_M'] / res
    for d in out:
        if all((d['rc'][0] - k['rc'][0]) ** 2 + (d['rc'][1] - k['rc'][1]) ** 2 > rad ** 2 for k in keep):
            keep.append(d)
    return keep


def _cut_and_label(free, doors, min_cells):
    """Cut free space at doors with 4-connected lines; label regions; validate doors.
    Returns (lab int32 0=none, doors_valid with 'regions'(a,b))."""
    h, w = free.shape
    active = list(doors)
    for _it in range(3):
        cut = np.zeros((h, w), np.uint8)
        for d in active:
            r, c = d['rc']; t = d['tangent']; n = np.array([-t[1], t[0]])
            L = d['dt'] + 3
            p0 = (int(round(c - n[1] * L)), int(round(r - n[0] * L)))
            p1 = (int(round(c + n[1] * L)), int(round(r + n[0] * L)))
            cv2.line(cut, p0, p1, 1, 1, lineType=cv2.LINE_4)
        cutb = cut.astype(bool) & free
        lab, nl = ndi.label(free & ~cutb, structure=S4)
        bad = []
        for i, d in enumerate(active):
            a, b = _door_sides(lab, d)
            d['regions'] = (a, b)
            if a == 0 or b == 0 or a == b:
                bad.append(i)
        if not bad:
            break
        active = [d for i, d in enumerate(active) if i not in bad]
    # merge small components into the neighbour sharing the longest boundary (through cut cells)
    sizes = np.bincount(lab.ravel(), minlength=nl + 1)
    small = np.nonzero((sizes < min_cells) & (np.arange(nl + 1) > 0))[0]
    if len(small):
        objs = ndi.find_objects(lab)
        for s in small:
            sl = objs[s - 1]
            if sl is None:
                continue
            sl2 = tuple(slice(max(0, x.start - 3), x.stop + 3) for x in sl)
            m = lab[sl2] == s
            ring = ndi.binary_dilation(m, S8, iterations=3) & ~m
            nbl = lab[sl2][ring]
            nbl = nbl[(nbl > 0) & (sizes[nbl] >= min_cells)]
            if len(nbl):
                lab[sl2][m] = np.bincount(nbl).argmax()
            else:
                lab[sl2][m] = 0
    # cut cells -> nearest label
    lab = np.where(free, _expand_labels(lab, 4), 0).astype(np.int32)
    lab[~free] = 0
    # re-validate doors after merging
    valid = []
    for d in active:
        a, b = _door_sides(lab, d)
        if a and b and a != b:
            d['regions'] = (a, b); valid.append(d)
    return lab, valid


def _expand_labels(lab, distance):
    """Assign unlabelled cells the nearest label within `distance` cells (skimage<0.17 lacks expand_labels)."""
    d, (ir, ic) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    out = lab[ir, ic]
    out[d > distance] = 0
    return out


def _door_sides(lab, d):
    r, c = d['rc']; t = d['tangent']; R = int(d['dt'] + 5)
    h, w = lab.shape
    r0, r1, c0, c1 = max(0, r - R), min(h, r + R + 1), max(0, c - R), min(w, c + R + 1)
    sub = lab[r0:r1, c0:c1]
    rr, cc = np.mgrid[r0:r1, c0:c1]
    proj = (rr - r) * t[0] + (cc - c) * t[1]
    out = []
    for sgn in (1, -1):
        m = (sub > 0) & (sgn * proj >= 2)
        v = sub[m]
        out.append(int(np.bincount(v).argmax()) if len(v) else 0)
    return out[0], out[1]


# --------------------------------------------------------------------------------------- jamb doors

def _wall_ends(obs, mean, gapw, P):
    """Wall-skeleton end points and their outward directions (independent of the hypothesis)."""
    wall = (obs == 1) | ((obs == 0.5) & (mean >= P['JAMB_WALL_MEAN']))
    if gapw is not None:
        wall |= gapw
    if not wall.any():
        return np.zeros((0, 2), int), np.zeros((0, 2))
    wsk = skeletonize(wall)
    nb = _nbr_count(wsk)
    ends = np.argwhere(wsk & (nb == 1))
    if len(ends) < 2:
        return ends, np.zeros((len(ends), 2))
    # end direction: from the point 5 steps back along the wall skeleton to the end
    dirs = []
    for r, c in ends:
        prev = None; cur = (int(r), int(c))
        for _ in range(5):
            nbs = [q for q in _neighbors(wsk, *cur) if q != prev]
            if len(nbs) != 1:
                break
            prev, cur = cur, nbs[0]
        v = np.array([r - cur[0], c - cur[1]], float); n = np.linalg.norm(v)
        dirs.append(v / n if n > 0 else v)
    return ends, np.array(dirs)


def _jamb_doors(free, ends_dirs, res, P):
    """Doors from wall jambs: pairs of wall-skeleton end points facing each other JAMB_MIN..JAMB_MAX apart,
    the segment between them mostly free and free space on both sides of the segment."""
    ends, dirs = ends_dirs
    if len(ends) < 2:
        return []
    lo, hi = P['JAMB_MIN_M'] / res, P['JAMB_MAX_M'] / res
    from scipy.spatial import cKDTree
    tree = cKDTree(ends)
    out = []
    h, w = free.shape
    for i, j in tree.query_pairs(hi):
        a = ends[i].astype(float); b = ends[j].astype(float); ab = b - a; L = np.linalg.norm(ab)
        if L < lo or not dirs[i].any() or not dirs[j].any():
            continue
        u = ab / L
        if dirs[i] @ u < P['JAMB_COS'] or dirs[j] @ (-u) < P['JAMB_COS']:
            continue
        n = int(np.ceil(L)); ts = np.linspace(0, 1, n + 1)[1:-1]
        pts = np.round(a[None] + ts[:, None] * ab[None]).astype(int)
        if len(pts) == 0 or free[pts[:, 0], pts[:, 1]].mean() < P['JAMB_FREE_FRAC']:
            continue
        mid = (a + b) / 2; t = np.array([-u[1], u[0]])   # passage direction (normal of the segment)
        ok = True
        for sgn in (1, -1):
            q = np.round(mid + sgn * t * max(5, L / 2)).astype(int)
            if not (0 <= q[0] < h and 0 <= q[1] < w and free[q[0], q[1]]):
                ok = False
        if not ok:
            continue
        mr, mc = int(round(mid[0])), int(round(mid[1]))
        if not free[mr, mc]:
            continue
        wfree = float(free[pts[:, 0], pts[:, 1]].sum()) * L / n      # clear width along the segment
        if wfree < lo or wfree > hi:
            continue
        out.append(dict(rc=(mr, mc), width_m=float(wfree * res), tangent=t, dt=float(L / 2), src='jamb'))
    return out


def _merge_doors(primary, extra, res, P):
    """Add extra doors not within DOOR_NMS_M of a primary door (primary kept as is)."""
    rad2 = (P['DOOR_NMS_M'] / res) ** 2
    out = list(primary)
    for d in sorted(extra, key=lambda x: x['width_m']):
        if all((d['rc'][0] - k['rc'][0]) ** 2 + (d['rc'][1] - k['rc'][1]) ** 2 > rad2 for k in out):
            out.append(d)
    return out


# --------------------------------------------------------------------------------------- chains

def _region_chain(lab, i, sk, dt, res, sl=None):
    """Longest geodesic path of the region's skeleton (largest skeleton component). Returns (path Nx2, len_m, wmed_m).
    sl: bbox slices of the region (speed); path returned in full-array coordinates."""
    if sl is None:
        sl = (slice(0, lab.shape[0]), slice(0, lab.shape[1]))
    off = np.array([sl[0].start, sl[1].start])
    lab = lab[sl]; sk = sk[sl]; dt = dt[sl]
    m = sk & (lab == i)
    if m.sum() < 2:
        return None, 0.0, 0.0
    cl, n = ndi.label(m, structure=S8)
    if n > 1:
        m = cl == (np.bincount(cl.ravel())[1:].argmax() + 1)
    pts, idx, adj = _skel_graph(m)
    pl = pts.tolist()

    def far(src):
        dist = [None] * len(pl); par = [-1] * len(pl); dist[src] = 0.0
        q = deque([src])
        while q:
            u = q.popleft(); ru, cu = pl[u]
            for v in adj[u]:
                if dist[v] is None:
                    rv, cv = pl[v]
                    dist[v] = dist[u] + (1.4142 if (ru != rv and cu != cv) else 1.0); par[v] = u; q.append(v)
        dd = np.array([-1.0 if x is None else x for x in dist])
        k = int(np.argmax(dd))
        return k, dd[k], par
        k = int(np.argmax(np.where(np.isfinite(dist), dist, -1)))
        return k, dist[k], par
    a, _, _ = far(0)
    b, L, par = far(a)
    path = [b]
    while path[-1] != a:
        path.append(par[path[-1]])
    path = pts[np.array(path)]
    w = np.maximum(2 * dt[path[:, 0], path[:, 1]] - 1, 0) * res
    return path + off, float(L * res), float(np.median(w))


def _resample(path, step):
    """Points along a polyline every `step` cells (geodesic), incl. both ends."""
    if path is None or len(path) == 0:
        return np.zeros((0, 2), int)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path.astype(float), axis=0), axis=1))]
    tgt = np.arange(0, seg[-1] + 1e-9, step)
    if seg[-1] - tgt[-1] > step / 2:
        tgt = np.r_[tgt, seg[-1]]
    return path[np.searchsorted(seg, tgt).clip(0, len(path) - 1)]


def _pocket_viewpoints(lab, i, dt, res, P, sl):
    """<= POCKET_VP_MAX interior points: most central cell first, then farthest-point sampling on cells
    with clearance >= 0.3 m."""
    cells = np.argwhere(lab[sl] == i) + np.array([sl[0].start, sl[1].start])
    if len(cells) == 0:
        return []
    area = len(cells) * res * res
    k = int(min(P['POCKET_VP_MAX'], max(1, np.ceil(area / P['POCKET_VP_M2']))))
    cl = dt[cells[:, 0], cells[:, 1]]
    good = cells[cl >= max(3, 0.3 / res)] if (cl >= max(3, 0.3 / res)).any() else cells
    out = [cells[int(np.argmax(cl))]]
    dmin = np.linalg.norm(good - out[0], axis=1)
    while len(out) < k:
        j = int(np.argmax(dmin))
        if dmin[j] * res < 2.0:
            break
        out.append(good[j]); dmin = np.minimum(dmin, np.linalg.norm(good - good[j], axis=1))
    return out


# --------------------------------------------------------------------------------------- features

def _region_features(lab, nlab, sk, dt, nb, doors, unknown, res):
    """Vectorised per-label features. Returns dict label -> feature dict."""
    idx = lab.ravel()
    cnt = np.bincount(idx, minlength=nlab + 1).astype(float)
    rr, cc = np.indices(lab.shape)
    sr = np.bincount(idx, rr.ravel(), nlab + 1); sc = np.bincount(idx, cc.ravel(), nlab + 1)
    srr = np.bincount(idx, (rr * rr).ravel(), nlab + 1); scc = np.bincount(idx, (cc * cc).ravel(), nlab + 1)
    src = np.bincount(idx, (rr * cc).ravel(), nlab + 1)
    unk = np.bincount(idx, unknown.ravel().astype(float), nlab + 1)
    skl = lab[sk]; wsk = np.maximum(2 * dt[sk] - 1, 0) * res
    s_n = np.bincount(skl, minlength=nlab + 1).astype(float)
    s_w = np.bincount(skl, wsk, nlab + 1); s_w2 = np.bincount(skl, wsk * wsk, nlab + 1)
    jl = lab[sk & (nb >= 3)]; s_j = np.bincount(jl, minlength=nlab + 1)
    deg = np.zeros(nlab + 1, int)
    for d in doors:
        for x in set(d['regions']):
            deg[x] += 1
    feats = {}
    for i in range(1, nlab + 1):
        n = cnt[i]
        if n == 0:
            continue
        mr, mc = sr[i] / n, sc[i] / n
        cov = np.array([[srr[i] / n - mr * mr, src[i] / n - mr * mc], [src[i] / n - mr * mc, scc[i] / n - mc * mc]])
        ev = np.clip(np.linalg.eigvalsh(cov), 1e-6, None)
        ratio = float(np.sqrt(ev[1] / ev[0]))
        sn = s_n[i]
        mw = s_w[i] / sn if sn else 0.0
        cv = float(np.sqrt(max(s_w2[i] / sn - mw * mw, 0)) / mw) if sn and mw > 0 else 9.9
        feats[i] = dict(area_m2=float(n * res * res), ratio=ratio, skel_m=float(sn * res),
                        width_mean_m=float(mw), width_cv=cv, degree=int(deg[i]), junctions=int(s_j[i]),
                        known_frac=float(1 - unk[i] / n), unknown_free_m2=float(unk[i] * res * res),
                        centroid_rc=(float(mr), float(mc)))
    return feats


def _classify(f, P):
    if P.get('CHAIN_CORR_M') is not None:
        # v2: corridor = skeleton chain >= CHAIN_CORR_M with >= CHAIN_CORR_DOORS doors, median width in range
        corr = (f.get('chain_m', 0) >= P['CHAIN_CORR_M'] and f['degree'] >= P['CHAIN_CORR_DOORS']
                and P['CORR_W_MIN'] <= f.get('chain_wmed_m', 0) <= P['CORR_W_MAX'])
        if corr:
            return 'corridor'
        if f['area_m2'] >= P['HALL_AREA_M2'] and f['degree'] >= P['HALL_DEG']:
            return 'hall'
        return 'room'
    elong = f['ratio'] >= P['CORR_RATIO'] or (f['skel_m'] >= P['CORR_SKEL_M'] and f['width_cv'] <= P['CORR_CV'])
    if elong and P['CORR_W_MIN'] <= f['width_mean_m'] <= P['CORR_W_MAX']:
        return 'corridor'
    if f['area_m2'] >= P['HALL_AREA_M2'] and f['degree'] >= P['HALL_DEG'] and not elong:
        return 'hall'
    return 'room'


# --------------------------------------------------------------------------------------- hypothesis

def _robot_component(free, rc):
    lab, n = ndi.label(free, structure=S4)
    r, c = int(rc[0]), int(rc[1])
    h, w = free.shape
    r = min(max(r, 0), h - 1); c = min(max(c, 0), w - 1)
    k = lab[r, c]
    if k == 0:
        if n == 0:
            return free & False
        d, (ir, ic) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        k = lab[ir[r, c], ic[r, c]]
    return lab == k


def _gap_walls(obs, mean, res, P):
    """Predicted-wall gap closing: UNKNOWN cells inside gaps <= GAP_CLOSE_M between wall evidence
    (observed OCC or predicted mean >= GAP_WALL_MEAN) become wall. Returns bool mask of added wall cells."""
    if not P.get('GAP_CLOSE_M'):
        return np.zeros(obs.shape, bool)
    wall = (obs == 1) | ((obs == 0.5) & (mean >= P['GAP_WALL_MEAN']))
    d = int(round(P['GAP_CLOSE_M'] / res)) + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    closed = cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return closed & ~wall & (obs == 0.5)


def _hypothesis(obs, mean, var, theta, rc, res, P, pfree=None, gapw=None):
    unknown = obs == 0.5
    if pfree is not None:
        pred = unknown & (pfree >= theta)
    else:
        pred = unknown & (mean < theta)
    if gapw is not None:
        pred &= ~gapw
    if P['VAR_GATE'] is not None:
        pred &= var <= P['VAR_GATE']
    if P['PRED_RANGE_M'] is not None and pfree is None:
        dist = P.get('_dfree')
        if dist is None:
            dist = ndi.distance_transform_edt(obs != 0)
        pred &= dist * res <= P['PRED_RANGE_M']
    if P.get('OPEN_PRED_M'):
        # break sub-door gaps in dashed predicted walls: predicted-free kept only where a disk of radius
        # OPEN_PRED_M fits in (observed FREE | predicted-free); observed FREE untouched
        rad = int(round(P['OPEN_PRED_M'] / res))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
        opened = cv2.morphologyEx(((obs == 0) | pred).astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
        pred &= opened
    free = (obs == 0) | pred
    hole = int(round(P['HOLE_FILL_M2'] / res / res))
    if hole > 0:
        free = remove_small_holes(free, area_threshold=hole)
        free &= obs != 1
    free = _robot_component(free, rc)
    fp = np.pad(free, 1).astype(np.uint8)
    dt = cv2.distanceTransform(fp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1].astype(np.float64)
    sk = skeletonize(free)
    sk = _prune_spurs(sk, int(round(P['SPUR_M'] / res)))
    return free, dt, sk


def _segment(free, dt, sk, doors, res, P, unknown):
    min_cells = int(round(P['MIN_REGION_M2'] / res / res))
    lab, doors = _cut_and_label(free, doors, min_cells)
    # relabel consecutively
    u = np.unique(lab); u = u[u > 0]
    remap = np.zeros(lab.max() + 1, np.int32); remap[u] = np.arange(1, len(u) + 1)
    lab = remap[lab]
    for d in doors:
        d['regions'] = (int(remap[d['regions'][0]]), int(remap[d['regions'][1]]))
    nb = _nbr_count(sk)
    feats = _region_features(lab, len(u), sk, dt, nb, doors, unknown & free, res)
    objs = ndi.find_objects(lab)
    for i, f in feats.items():
        f['bbox'] = objs[i - 1]
        if P.get('CHAIN_CORR_M') is not None:
            path, L, wm = _region_chain(lab, i, sk, dt, res, objs[i - 1])
            f['chain'] = path; f['chain_m'] = L; f['chain_wmed_m'] = wm
        f['cls'] = _classify(f, P)
    return lab, doors, feats


# --------------------------------------------------------------------------------------- main

def build_topo(obs, mean, var, robot_rc, ppm, prev=None, params=None, pfree=None):
    """pfree: optional array (same shape as obs) = p(c) from pc_model.p_free; replaces mean thresholds + range."""
    t0 = time.perf_counter()
    P = dict(PARAMS)
    if params:
        P.update(params)
    obs = np.asarray(obs, np.float32); mean = np.asarray(mean, np.float32)
    var = np.zeros_like(mean) if var is None else np.asarray(var, np.float32)
    res = 1.0 / float(ppm)
    H, W = obs.shape
    # crop to the useful bbox (padded inputs)
    if pfree is not None:
        pfree = np.asarray(pfree, np.float32)
        loose = (obs == 0) | ((obs == 0.5) & (pfree >= min(P['PF_THETAS'])))
    else:
        loose = (obs == 0) | ((obs == 0.5) & (mean < max(P['THETAS'])))
        if P['PRED_RANGE_M'] is not None:
            dfree = cv2.distanceTransform((obs != 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            loose &= (obs == 0) | (dfree * res <= P['PRED_RANGE_M'] + 0.5)
    rows = np.nonzero(loose.any(1))[0]; cols = np.nonzero(loose.any(0))[0]
    if len(rows) == 0:
        rows = np.array([int(robot_rc[0])]); cols = np.array([int(robot_rc[1])])
    m = 6
    r0, r1 = max(0, min(rows[0], int(robot_rc[0])) - m), min(H, max(rows[-1], int(robot_rc[0])) + m + 1)
    c0, c1 = max(0, min(cols[0], int(robot_rc[1])) - m), min(W, max(cols[-1], int(robot_rc[1])) + m + 1)
    o = obs[r0:r1, c0:c1]; mu = mean[r0:r1, c0:c1]; va = var[r0:r1, c0:c1]
    pf = pfree[r0:r1, c0:c1] if pfree is not None else None
    gapw = _gap_walls(o, mu, res, P)
    P = dict(P); P['_dfree'] = (cv2.distanceTransform((o != 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                                if pf is None and P['PRED_RANGE_M'] is not None else None)
    ends_dirs = _wall_ends(o, mu, gapw, P) if P.get('JAMB') else None
    rc = (int(robot_rc[0]) - r0, int(robot_rc[1]) - c0)
    unknown = o == 0.5
    timing = {}

    # ---- per hypothesis
    hyps = []
    for th in (P['PF_THETAS'] if pf is not None else P['THETAS']):
        ta = time.perf_counter()
        free, dt, sk = _hypothesis(o, mu, va, th, rc, res, P, pf, gapw)
        doors = _detect_doors(free, dt, sk, res, P)
        if P.get('JAMB'):
            doors = _merge_doors(doors, _jamb_doors(free, ends_dirs, res, P), res, P)
        lab, doors_v, feats = _segment(free, dt, sk, [dict(d) for d in doors], res, P, unknown)
        hyps.append(dict(theta=th, free=free, dt=dt, sk=sk, doors_raw=doors, doors=doors_v, lab=lab, feats=feats))
        timing['hyp_%.2f' % th] = time.perf_counter() - ta

    # ---- fuse doors (validated doors of each hypothesis)
    ta = time.perf_counter()
    nh = len(hyps); ref = hyps[P['REF_THETA_IDX']]
    order = [P['REF_THETA_IDX']] + [i for i in range(nh) if i != P['REF_THETA_IDX']]
    clusters = []
    rad2 = (P['DOOR_FUSE_M'] / res) ** 2
    for hi in order:
        for d in hyps[hi]['doors']:
            best = None; bd = rad2
            for cl in clusters:
                if hi in cl['hyps']:
                    continue
                q = cl['members'][0]['rc']
                dd = (q[0] - d['rc'][0]) ** 2 + (q[1] - d['rc'][1]) ** 2
                if dd <= bd:
                    best, bd = cl, dd
            if best is None:
                clusters.append(dict(members=[d], hyps={hi}))
            else:
                best['members'].append(d); best['hyps'].add(hi)
    cut_doors = []
    for cl in clusters:
        pe = len(cl['hyps']) / nh
        if pe < P['DOOR_PEXIST_CUT']:
            continue
        d = cl['members'][0]
        r, c = d['rc']
        if not ref['free'][r, c]:
            continue
        cut_doors.append(dict(rc=d['rc'], width_m=d['width_m'], tangent=d['tangent'],
                              dt=max(float(ref['dt'][r, c]), d['dt']) if d.get('src') == 'jamb' else float(ref['dt'][r, c]),
                              p_exist=pe, src=d.get('src', 'clearance')))
    lab, fdoors, feats = _segment(ref['free'], ref['dt'], ref['sk'], cut_doors, res, P, unknown)
    nlab = int(lab.max())
    # class vote over hypotheses
    probs = {}
    for i in feats:
        probs[i] = np.zeros(3)
    for h in hyps:
        if h['lab'].max() == 0:
            continue
        pair = lab.astype(np.int64) * (h['lab'].max() + 1) + h['lab']
        bc = np.bincount(pair[(lab > 0) & (h['lab'] > 0)].ravel(), minlength=(nlab + 1) * (h['lab'].max() + 1))
        bc = bc.reshape(nlab + 1, h['lab'].max() + 1)
        for i in feats:
            j = int(bc[i].argmax())
            cls = h['feats'][j]['cls'] if bc[i, j] > 0 and j in h['feats'] else feats[i]['cls']
            probs[i][CLASSES.index(cls)] += 1.0 / nh
    timing['fuse'] = time.perf_counter() - ta

    # ---- temporal association
    ta = time.perf_counter()
    full_lab = np.zeros((H, W), np.int32); full_lab[r0:r1, c0:c1] = lab
    ids = {}; cls_out = {}; p_out = {}
    pst = prev.state if isinstance(prev, TopoMap) else prev
    next_id = 1
    if pst is not None:
        next_id = pst['next_id']
        plab = pst['lab_full']
        if plab.shape == full_lab.shape:
            nprev = int(plab.max())
            pair = full_lab.astype(np.int64) * (nprev + 1) + plab
            inter = np.bincount(pair.ravel(), minlength=(nlab + 1) * (nprev + 1)).reshape(nlab + 1, nprev + 1)
            a_cnt = inter.sum(1); b_cnt = inter.sum(0)
            cand = []
            for i in feats:
                for j in np.nonzero(inter[i, 1:])[0] + 1:
                    iou = inter[i, j] / (a_cnt[i] + b_cnt[j] - inter[i, j])
                    if iou >= P['IOU_ASSOC'] and j in pst['local2id']:
                        cand.append((iou, i, j))
            cand.sort(reverse=True)
            used_i, used_j = set(), set()
            for iou, i, j in cand:
                if i in used_i or j in used_j:
                    continue
                used_i.add(i); used_j.add(j)
                sid = pst['local2id'][j]; ids[i] = sid
                pp = np.asarray(pst['p'][sid])
                pn = P['EMA_ALPHA'] * probs[i] + (1 - P['EMA_ALPHA']) * pp
                pc = pst['cls'][sid]; k = CLASSES.index(pc)
                if pn[k] < P['HYST_LO'] or pn.max() >= P['HYST_HI'] and int(pn.argmax()) != k:
                    pc = CLASSES[int(pn.argmax())]
                p_out[i] = pn; cls_out[i] = pc
    for i in feats:
        if i not in ids:
            ids[i] = next_id; next_id += 1
            p_out[i] = probs[i]; cls_out[i] = CLASSES[int(probs[i].argmax())]
    timing['temporal'] = time.perf_counter() - ta

    # ---- outputs (stable ids)
    lut = np.full(nlab + 1, -1, np.int32)
    for i, s in ids.items():
        lut[i] = s
    region_id = np.full((H, W), -1, np.int32)
    region_id[r0:r1, c0:c1] = lut[lab]
    doors_out = []
    door_ids = {i: [] for i in feats}
    for k, d in enumerate(fdoors):
        a, b = d['regions']
        doors_out.append(dict(id=k, rc=(d['rc'][0] + r0, d['rc'][1] + c0), width_m=d['width_m'],
                              regions=(int(lut[a]), int(lut[b])), p_exist=float(d['p_exist']),
                              tangent=tuple(float(x) for x in d['tangent']), src=d.get('src', 'clearance')))
        door_ids[a].append(k); door_ids[b].append(k)
    sk_lab = lab * ref['sk']
    regions = []; backbone = []; pockets = []
    for i, f in feats.items():
        sid = int(lut[i]); p = p_out[i]
        bb = f['bbox']
        skel = np.argwhere(sk_lab[bb] == i) + np.array([r0 + bb[0].start, c0 + bb[1].start])
        deg = len(set(door_ids[i]))
        reg = dict(id=sid, cls=cls_out[i], p={c: float(p[j]) for j, c in enumerate(CLASSES)},
                   area_m2=f['area_m2'], unknown_free_m2=f['unknown_free_m2'], degree=deg,
                   door_ids=sorted(set(door_ids[i])),
                   centroid_rc=(f['centroid_rc'][0] + r0, f['centroid_rc'][1] + c0),
                   skeleton_rc=skel.astype(np.int32),
                   features=dict(ratio=f['ratio'], skel_m=f['skel_m'], width_mean_m=f['width_mean_m'],
                                 width_cv=f['width_cv'], junctions=f['junctions'], known_frac=f['known_frac']))
        in_bb = (p[0] >= P['BACKBONE_P'] or reg['cls'] in ('corridor', 'hall') or deg >= P['BACKBONE_DEG'])
        reg['backbone'] = bool(in_bb)
        (backbone if in_bb else pockets).append(sid)
        regions.append(reg)
    regions.sort(key=lambda r: r['id'])
    # chains + viewpoints (v2)
    off = np.array([r0, c0])
    chains = []; viewpoints = []
    step = P['VP_SPACING_M'] / res
    for i, f in feats.items():
        sid = int(lut[i])
        path = f.get('chain')
        if path is None and P.get('CHAIN_CORR_M') is None:
            path, L, wm = _region_chain(lab, i, ref['sk'], ref['dt'], res, f['bbox']); f['chain_m'] = L; f['chain_wmed_m'] = wm
        if path is not None and len(path) >= 2:
            chains.append(dict(id=len(chains), region_id=sid, cls=cls_out[i], polyline_rc=(path + off).astype(np.int32),
                               length_m=float(f['chain_m']), width_med_m=float(f['chain_wmed_m']),
                               n_doors=len(set(door_ids[i]))))
            for q in _resample(path, step):
                viewpoints.append(dict(rc=(int(q[0] + r0), int(q[1] + c0)), kind='chain', region_id=sid))
    for d in doors_out:
        viewpoints.append(dict(rc=tuple(d['rc']), kind='door', region_id=d['regions'][0]))
    for i in feats:
        sid = int(lut[i])
        if sid in pockets:
            for q in _pocket_viewpoints(lab, i, ref['dt'], res, P, feats[i]['bbox']):
                viewpoints.append(dict(rc=(int(q[0] + r0), int(q[1] + c0)), kind='pocket', region_id=sid))
    inv = {int(lut[i]): i for i in feats}
    for r in regions:
        f = feats[inv[r['id']]]
        r['features']['chain_m'] = float(f.get('chain_m', 0.0)); r['features']['chain_wmed_m'] = float(f.get('chain_wmed_m', 0.0))
    lut_bb = np.zeros(nlab + 1, bool)
    for i in feats:
        lut_bb[i] = int(lut[i]) in backbone
    backbone_mask = np.zeros((H, W), bool); backbone_mask[r0:r1, c0:c1] = lut_bb[lab]
    state = dict(lab_full=full_lab, local2id={i: ids[i] for i in feats},
                 p={ids[i]: p_out[i].tolist() for i in feats}, cls={ids[i]: cls_out[i] for i in feats},
                 next_id=next_id, timing=timing, crop=(r0, r1, c0, c1),
                 hyp_door_counts=[len(h['doors']) for h in hyps])
    return TopoMap(region_id=region_id, regions=regions, doors=doors_out, backbone_mask=backbone_mask,
                   pocket_ids=sorted(pockets), backbone_ids=sorted(backbone), state=state,
                   chains=chains, viewpoints=viewpoints, gap_wall_mask=None,
                   timing_s=time.perf_counter() - t0)
