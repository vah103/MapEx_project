"""Faithful Gate 1 AL port. No tuning, no runtime dependency on replica variants.
Policy is frozen Gate1; packed exact gains and bounded packed cover queries change storage only.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
LS_MAX_ITERS = 500
EPS = 1e-6

def _gain_rows(vis, keys, u):
    """Exact float32 CSR gains, streamed with a bounded per-row temporary.

    Internal u is binary and each native row has <=201216 unique cells.
    Integer partial sums <=2**24 are exactly representable in float32.
    Outside that contract preserve the original sparse arithmetic.
    """
    if any(len(vis[k]) > 2**24 for k in keys) or not np.all((u==0)|(u==1)):
        return _sparse(vis, keys, u.size).dot(u)
    return np.asarray([np.count_nonzero(u[vis[k]]) for k in keys], dtype=np.float32)


class _PackedCover(object):
    """Exact CSC[:,E].sum(axis=1)==len(E), with bounded storage.

    Native rows are unique; target membership uses one bit. Duplicate external
    rows/queries retain the reference integer-multiplicity predicate by streaming.
    """
    BLOCK_BYTES = 256  # query tiling only, not an algorithm parameter
    MAX_BYTES = 96 * 2**20

    def __init__(self, vis, keys, target, remap, ncols):
        self.vis, self.keys, self.target, self.remap = vis, keys, target, remap
        self.ncols = int(ncols)
        self.stream_only = len(keys)*((self.ncols+7)//8)>self.MAX_BYTES
        self.bits = np.empty((0,0),np.uint8) if self.stream_only else np.zeros((len(keys), (self.ncols+7)//8), np.uint8)
        self.general = False
        if self.stream_only:return
        scratch = np.zeros(self.ncols, np.uint8)
        for j,k in enumerate(keys):
            cols = self._cols(k)
            if len(cols)>1 and (cols[1:]<=cols[:-1]).any():
                ordered=np.sort(cols)
                if (ordered[1:]==ordered[:-1]).any():self.general=True
            scratch.fill(0);scratch[cols]=1
            self.bits[j]=np.packbits(scratch)

    def _cols(self,k):
        v=self.vis[k]
        return self.remap[v[self.target[v]]]

    def covers(self,E):
        E=np.asarray(E,dtype=np.int64)
        duplicate_query=(len(E)>1 and (E[1:]<=E[:-1]).any()
                         and len(np.unique(E))!=len(E))
        if self.stream_only or self.general or duplicate_query:
            out=np.zeros(len(self.keys),bool)
            for j,k in enumerate(self.keys):
                cols=np.sort(self._cols(k))
                count=(np.searchsorted(cols,E,side='right')-
                       np.searchsorted(cols,E,side='left')).sum(dtype=np.int64)
                out[j]=count==len(E)
            return out
        scratch=np.zeros(self.ncols,np.uint8);scratch[E]=1
        packed=np.packbits(scratch);active=np.flatnonzero(packed)
        out=np.ones(len(self.keys),bool)
        for start in range(0,len(active),self.BLOCK_BYTES):
            rows=np.flatnonzero(out)
            if not len(rows):break
            cols=active[start:start+self.BLOCK_BYTES];need=packed[cols]
            block=self.bits[rows[:,None],cols[None,:]]
            out[rows]&=np.all((block & need)==need,axis=1)
        return out



class _GainRows(object):
    """Exact binary CSR.dot using packed membership and tiled byte popcounts."""
    BLOCK_ROWS = 64
    POPCOUNT = np.asarray([bin(i).count('1') for i in range(256)],np.uint8)

    def __init__(self,vis,keys,ncols):
        self.vis,self.keys,self.ncols=vis,keys,int(ncols)
        self.reference=None
        self.packed=None
        # These bounds select storage only, never a goal/scoring parameter.
        if (len(keys)*((self.ncols+7)//8) <= _PackedCover.MAX_BYTES
                and all(len(vis[k])<=2**24 for k in keys)):
            self.packed=_PackedCover(vis,keys,np.ones(ncols,bool),
                                     np.arange(ncols,dtype=np.int64),ncols)

    def dot(self,u):
        if (self.packed is None or self.packed.general
                or not np.all((u==0)|(u==1))):
            return _gain_rows(self.vis,self.keys,u)
        mask=np.packbits(np.asarray(u!=0,np.uint8))
        result=np.empty(len(self.keys),np.float32)
        for begin in range(0,len(self.keys),self.BLOCK_ROWS):
            end=min(begin+self.BLOCK_ROWS,len(self.keys))
            masked=self.packed.bits[begin:end] & mask
            result[begin:end]=self.POPCOUNT[masked].sum(axis=1,dtype=np.int64)
        return result


def two_opt(D, path):
    n = len(path)
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
    return path

def _row(g, node, tgt_nodes):
    return dijkstra(g.G, indices=int(node))[tgt_nodes]

def _cost(Dm, path):
    return float(sum(Dm[path[i], path[i + 1]] for i in range(len(path) - 1)))

class Ctx(object):
    """Distance oracle between the robot (key -1) and candidate indices, via single-source Dijkstra rows (cached)."""
    def __init__(self, g, robot, cand, keys):
        self.g = g
        self.keys = np.asarray(keys, np.int64)                  # candidate indices that may appear in a tour
        self.col = {int(k): j for j, k in enumerate(self.keys)}
        self.nodes = np.array([g.node(q) for q in cand[self.keys]], np.int64) if len(self.keys) else np.zeros(0, np.int64)
        self.rnode = g.node(robot)
        self.rows = {}
        self.n_dij = 0

    def row(self, k):
        if k not in self.rows:
            self.rows[k] = _row(self.g, self.rnode if k == -1 else self.nodes[self.col[k]], self.nodes)
            self.n_dij += 1
        return self.rows[k]

    def d(self, a, b):
        if a == b:
            return 0.0
        if b == -1:
            a, b = b, a
        return float(self.row(a)[self.col[b]])

    def Dm(self, tour):
        n = len(tour)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = self.d(tour[i], tour[j])
        return D

def _insertion(ctx, tour):
    """cheapest insertion cost of every ctx key into the open tour (tour[0] = robot). returns (cost, position)."""
    R = np.stack([ctx.row(k) for k in tour])
    end = R[-1]
    if len(tour) == 1:
        return end, np.full(len(end), 1)
    dab = np.array([ctx.d(tour[i], tour[i + 1]) for i in range(len(tour) - 1)])
    mid = R[:-1] + R[1:] - dab[:, None]
    bm = mid.min(0); pm = mid.argmin(0) + 1
    use_end = end <= bm
    return np.where(use_end, end, bm), np.where(use_end, len(tour), pm)

def _two_opt_tour(ctx, tour):
    D = ctx.Dm(tour)
    p = two_opt(D, list(range(len(tour))))
    return [tour[i] for i in p]

def _sparse(vis, keys, ncells):
    lens = [len(vis[k]) for k in keys]
    r = np.repeat(np.arange(len(keys)), lens)
    c = np.concatenate([vis[k] for k in keys]) if len(keys) else np.zeros(0, np.int64)
    return csr_matrix((np.ones(len(c), np.float32), (r, c)), shape=(len(keys), ncells))

def solve_gpc(g, robot, cand, vis, need, mg, floor):
    gains0 = np.array([int(need[v].sum()) for v in vis]) if len(vis) else np.zeros(0, int)
    keys = np.flatnonzero(gains0 >= mg)
    if not len(keys):
        return [], [], None, {}
    ctx = Ctx(g, robot, cand, keys)
    M = _GainRows(vis, keys, need.size)
    u = need.astype(np.float32)
    tour = [-1]
    inside = np.zeros(len(keys), bool)
    while True:
        gain = M.dot(u)
        ok = (gain >= mg) & ~inside
        if not ok.any():
            break
        ins, pos = _insertion(ctx, tour)
        ratio = np.where(ok, gain / np.maximum(ins, floor), -1.0)
        j = int(np.argmax(ratio))
        tour.insert(int(pos[j]), int(keys[j])); inside[j] = True
        u[vis[keys[j]]] = 0
    tour = _two_opt_tour(ctx, tour)
    ch = tour[1:]
    Dm = ctx.Dm(tour)
    return ch, list(range(1, len(tour))), Dm, dict(n_dijkstra=ctx.n_dij)

def local_search(g, robot, cand, vis, need, ch, order):
    """(b): ch/order = initial cover (cand indices) and its open-tour order (1-based into ch). Hard constraint: every
    need cell covered by the initial cover stays covered."""
    tour = [-1] + [ch[k - 1] for k in order]
    tgt = np.zeros(need.size, bool)
    for k in ch:
        tgt[vis[k]] = True
    tgt &= need
    tcells = np.flatnonzero(tgt)
    remap = -np.ones(need.size, np.int64); remap[tcells] = np.arange(len(tcells))
    gains = np.array([int(tgt[v].sum()) for v in vis])
    keys = np.flatnonzero(gains >= 1)
    ctx = Ctx(g, robot, cand, keys)
    # cover matrix on target cells only (rows = keys)
    MT = _PackedCover(vis, keys, tgt, remap, len(tcells))
    cnt = np.zeros(len(tcells), np.int32)
    tv = {}
    for k in tour[1:]:
        tv[k] = remap[vis[k][tgt[vis[k]]]]
        cnt[tv[k]] += 1
    c0 = _cost(ctx.Dm(tour), list(range(len(tour))))
    n_moves = dict(remove=0, swap=0, reinsert=0)
    it = 0
    while it < LS_MAX_ITERS and len(tour) > 1:
        it += 1
        best = None
        for pos in range(1, len(tour)):
            v = tour[pos]; a = tour[pos - 1]; b = tour[pos + 1] if pos + 1 < len(tour) else None
            save = ctx.d(a, v) + (ctx.d(v, b) - ctx.d(a, b) if b is not None else 0.0)
            E = tv[v][cnt[tv[v]] == 1]
            if len(E) == 0:
                if save > EPS and (best is None or save > best[0]):
                    best = (save, 'remove', pos, None, None)
                continue
            wo = tour[:pos] + tour[pos + 1:]
            ins, ipos = _insertion(ctx, wo)
            # reinsert v
            jv = ctx.col[v]
            imp = save - ins[jv]
            if imp > EPS and (best is None or imp > best[0]):
                best = (imp, 'reinsert', pos, v, int(ipos[jv]))
            # swap v -> u covering all exclusive cells E of v
            covers = MT.covers(E)
            intour = np.isin(keys, wo)
            covers &= ~intour; covers[jv] = False
            if covers.any():
                cand_imp = np.where(covers, save - ins, -np.inf)
                j = int(np.argmax(cand_imp))
                if cand_imp[j] > EPS and (best is None or cand_imp[j] > best[0]):
                    best = (float(cand_imp[j]), 'swap', pos, int(keys[j]), int(ipos[j]))
        if best is None:
            break
        _, kind, pos, u, ip = best
        v = tour.pop(pos)
        n_moves[kind] += 1
        if kind == 'remove':
            cnt[tv.pop(v)] -= 1
        elif kind == 'reinsert':
            tour.insert(ip, v)
        else:
            cnt[tv.pop(v)] -= 1
            tour.insert(ip, u)
            tv[u] = remap[vis[u][tgt[vis[u]]]]
            cnt[tv[u]] += 1
        tour = _two_opt_tour(ctx, tour)
    assert (cnt > 0).all(), 'local search lost coverage'
    Dm = ctx.Dm(tour)
    c1 = _cost(Dm, list(range(len(tour))))
    return tour[1:], list(range(1, len(tour))), Dm, dict(ls_iters=it, ls_moves=n_moves, ls_cost0=c0, ls_cost1=c1,
                                                         n_dijkstra=ctx.n_dij)


def solve_al(g, robot, cand, vis, need, min_gain, stride):
    ch, order, Dm, _ = solve_gpc(g, robot, cand, vis, need, min_gain, float(stride))
    if not ch:
        return [], [], None, dict(cover_size_before_ls=0, cover_size=0,
                                  tour_cost_before_ls_cells=0.0, tour_cost_after_ls_cells=0.0)
    before_size = len(ch)
    ch, order, Dm, info = local_search(g, robot, cand, vis, need, ch, order)
    info.update(cover_size_before_ls=before_size, cover_size=len(ch),
                tour_cost_before_ls_cells=info['ls_cost0'],
                tour_cost_after_ls_cells=info['ls_cost1'])
    return ch, order, Dm, info
