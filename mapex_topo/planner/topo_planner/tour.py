"""Tour optimisation for the objective  J = sum_i g_i * max(0, T - t_i)   (TOPO-001).

Node model (start = index 0 of the distance matrix, nodes 1..n):
  arrive_k = leave_prev + D[prev, k];  gain time t_k = arrive_k + off_k;
  leave_k  = arrive_k + s_k  (service ticks; 0 for a single viewpoint).
`duration` = leave_last - t0.  L3 adds the hard constraint duration <= budget.
"""
import itertools

import numpy as np
from numba import njit

INF = float("inf")


class TourModel(object):
    def __init__(self, D, g, s=None, off=None, t0=0.0, T=1000.0, budget=None):
        self.D = np.asarray(D, dtype=np.float64)
        n = self.D.shape[0]
        self.g = np.zeros(n) if g is None else np.asarray(g, dtype=np.float64)
        self.s = np.zeros(n) if s is None else np.asarray(s, dtype=np.float64)
        self.off = np.zeros(n) if off is None else np.asarray(off, dtype=np.float64)
        self.t0 = float(t0)
        self.T = float(T)
        self.budget = INF if budget is None else float(budget)

    def evaluate(self, order):
        """(J, duration, gain_times) of visiting `order` (node ids >= 1) from start 0."""
        t = self.t0
        prev = 0
        J = 0.0
        times = []
        for k in order:
            d = self.D[prev, k]
            if not np.isfinite(d):
                return -INF, INF, times
            arrive = t + d
            tk = arrive + self.off[k]
            times.append(tk)
            J += self.g[k] * max(0.0, self.T - tk)
            t = arrive + self.s[k]
            prev = k
        return J, t - self.t0, times

    def value(self, order):
        J, dur, _ = self.evaluate(order)
        if dur > self.budget + 1e-9:
            return -INF
        return J


def best_insertion(model, order, k):
    """Best position to insert node k: (delta_J, delta_duration, pos)."""
    J0, dur0, _ = model.evaluate(order)
    best = (-INF, INF, None)
    for pos in range(len(order) + 1):
        cand = order[:pos] + [k] + order[pos:]
        J, dur, _ = model.evaluate(cand)
        if dur > model.budget + 1e-9 or not np.isfinite(J):
            continue
        if J - J0 > best[0]:
            best = (J - J0, dur - dur0, pos)
    return best


def greedy_insertion(model, nodes, order=None):
    """Repeatedly insert the node with the largest positive delta J."""
    order = list(order or [])
    rest = [k for k in nodes if k not in order]
    while rest:
        best = (0.0, None, None)
        for k in rest:
            dJ, _, pos = best_insertion(model, order, k)
            if pos is not None and dJ > best[0] + 1e-9:
                best = (dJ, k, pos)
        if best[1] is None:
            break
        order.insert(best[2], best[1])
        rest.remove(best[1])
    return order


def two_opt(model, order, max_rounds=50):
    order = list(order)
    best = model.value(order)
    for _ in range(max_rounds):
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                v = model.value(cand)
                if v > best + 1e-9:
                    order, best, improved = cand, v, True
        if not improved:
            break
    return order


def or_opt(model, order, max_seg=3, max_rounds=50):
    order = list(order)
    best = model.value(order)
    for _ in range(max_rounds):
        improved = False
        for L in range(1, max_seg + 1):
            for i in range(0, len(order) - L + 1):
                seg = order[i:i + L]
                rest = order[:i] + order[i + L:]
                for p in range(len(rest) + 1):
                    if p == i:
                        continue
                    for s in (seg, seg[::-1]) if L > 1 else (seg,):
                        cand = rest[:p] + s + rest[p:]
                        v = model.value(cand)
                        if v > best + 1e-9:
                            order, best, improved = cand, v, True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return order


def removal(model, order):
    """Drop nodes whose removal increases J (delay they cause exceeds their gain)."""
    order = list(order)
    best = model.value(order)
    changed = True
    while changed and order:
        changed = False
        for k in list(order):
            cand = [x for x in order if x != k]
            v = model.value(cand)
            if v > best + 1e-9:
                order, best, changed = cand, v, True
                break
    return order


def improve(model, order):
    prev = None
    while prev != order:
        prev = list(order)
        order = two_opt(model, order)
        order = or_opt(model, order)
    return order


# ---------------------------------------------------------------- Held-Karp
@njit(cache=True)
def _hk(D, g, s, K):
    """Backward subset DP with (C, Dur) Pareto fronts.

    Nodes 0..n-1, start = n.  For a set R left to visit after LEAVING node j:
      C(R, j)   = min_k D[j,k]*W(R) + s_k*W(R\\k) + C(R\\k, k)   (sum of g_i * rel. arrival)
      Dur(R, j) = D[j,k] + s_k + Dur(R\\k, k)
    Fronts keep up to K non-dominated (C, Dur) pairs (exact unless overflow).
    """
    n = len(g)
    M = 1 << n
    W = np.zeros(M)
    for m in range(1, M):
        low = m & (-m)
        b = 0
        while (1 << b) != low:
            b += 1
        W[m] = W[m ^ low] + g[b]
    C = np.full((M, n + 1, K), np.inf)
    Du = np.full((M, n + 1, K), np.inf)
    NK = np.full((M, n + 1, K), -1, dtype=np.int64)
    NI = np.full((M, n + 1, K), -1, dtype=np.int64)
    cnt = np.zeros((M, n + 1), dtype=np.int64)
    overflow = 0
    for j in range(n + 1):
        C[0, j, 0] = 0.0
        Du[0, j, 0] = 0.0
        cnt[0, j] = 1
    for m in range(1, M):
        for j in range(n + 1):
            if j < n and (m >> j) & 1:
                continue
            for k in range(n):
                if not (m >> k) & 1:
                    continue
                d = D[j, k]
                if not np.isfinite(d):
                    continue
                sub = m ^ (1 << k)
                base_c = d * W[m] + s[k] * W[sub]
                base_d = d + s[k]
                for e in range(cnt[sub, k]):
                    c = base_c + C[sub, k, e]
                    du = base_d + Du[sub, k, e]
                    # Pareto insert into (m, j)
                    dominated = False
                    for q in range(cnt[m, j]):
                        if C[m, j, q] <= c + 1e-12 and Du[m, j, q] <= du + 1e-12:
                            dominated = True
                            break
                    if dominated:
                        continue
                    q = 0
                    while q < cnt[m, j]:
                        if c <= C[m, j, q] and du <= Du[m, j, q]:
                            last = cnt[m, j] - 1
                            C[m, j, q] = C[m, j, last]
                            Du[m, j, q] = Du[m, j, last]
                            NK[m, j, q] = NK[m, j, last]
                            NI[m, j, q] = NI[m, j, last]
                            cnt[m, j] -= 1
                        else:
                            q += 1
                    if cnt[m, j] < K:
                        q = cnt[m, j]
                        cnt[m, j] += 1
                    else:
                        overflow += 1
                        q = 0
                        for r in range(1, K):
                            if C[m, j, r] > C[m, j, q]:
                                q = r
                        if C[m, j, q] <= c:
                            continue
                    C[m, j, q] = c
                    Du[m, j, q] = du
                    NK[m, j, q] = k
                    NI[m, j, q] = e
    return W, C, Du, NK, NI, overflow


def held_karp(model, nodes, K=8):
    """Exact budgeted orienteering on <= 12 nodes for J (gain times include off).

    Returns (order, J, duration, info).  J = W_F (T - t0) - C(F) - sum_F g*off,
    feasible iff duration <= budget; arrivals then lie before T (no clipping)."""
    nodes = list(nodes)
    n = len(nodes)
    if n == 0:
        return [], 0.0, 0.0, {"overflow": 0, "n": 0}
    # local matrix: nodes 0..n-1, start = n
    perm = nodes + [0]
    D = model.D[np.ix_(perm, perm)].copy()
    g = model.g[nodes].copy()
    s = model.s[nodes].copy()
    off = model.off[nodes].copy()
    W, C, Du, NK, NI, overflow = _hk(D, g, s, int(K))
    horizon = model.T - model.t0
    best = (0.0, 0, -1)  # J, mask, front index
    best_dur = 0.0
    for m in range(1, 1 << n):
        goff = sum(g[b] * off[b] for b in range(n) if (m >> b) & 1)
        for e in range(C.shape[2]):
            if not np.isfinite(C[m, n, e]):
                continue
            if Du[m, n, e] > model.budget + 1e-9:
                continue
            J = W[m] * horizon - C[m, n, e] - goff
            if J > best[0] + 1e-9:
                best = (J, m, e)
                best_dur = Du[m, n, e]
    J, m, e = best
    order = []
    j = n
    while m:
        k = int(NK[m, j, e])
        e2 = int(NI[m, j, e])
        order.append(nodes[k])
        m ^= (1 << k)
        j, e = k, e2
    return order, J, best_dur, {"overflow": int(overflow), "n": n}


def brute_force(model, nodes):
    """Reference: all subsets x permutations (tests only, n <= 7)."""
    best = ([], 0.0)
    for r in range(1, len(nodes) + 1):
        for sub in itertools.combinations(nodes, r):
            for perm in itertools.permutations(sub):
                v = model.value(list(perm))
                if v > best[1] + 1e-9:
                    best = (list(perm), v)
    return best


def orienteering(model, nodes, dp_max=12, K=8):
    """L3 solver: exact Held-Karp if <= dp_max nodes, else greedy+2-opt/Or-opt+removal."""
    nodes = list(nodes)
    if len(nodes) <= dp_max:
        order, J, dur, info = held_karp(model, nodes, K=K)
        info["solver"] = "held_karp"
        return order, info
    order = greedy_insertion(model, nodes)
    order = improve(model, order)
    order = removal(model, order)
    order = greedy_insertion(model, nodes, order)
    order = improve(model, order)
    return order, {"solver": "greedy", "n": len(nodes)}


# ------------------------------------------------- R2: exact re-ordering of a prefix
@njit(cache=True)
def _hk_prefix(D, g, wtail, tailc):
    """Order all n prefix nodes (start = index n) minimising
       sum_i g_i a_i + W_tail * a_end + tailc[end]   (a = arrival time from start).
    V(S, j) = sum_{i in S} g_i a_i + W_rest(S) a_j;  V(S+k, k) = V(S, j) + W_rest(S) D[j,k]."""
    n = len(g)
    M = 1 << n
    Wp = np.zeros(M)
    for m in range(1, M):
        low = m & (-m)
        b = 0
        while (1 << b) != low:
            b += 1
        Wp[m] = Wp[m ^ low] + g[b]
    Wall = Wp[M - 1] + wtail
    V = np.full((M, n), np.inf)
    P = np.full((M, n), -1, dtype=np.int64)
    for k in range(n):
        if np.isfinite(D[n, k]):
            V[1 << k, k] = Wall * D[n, k]
    for m in range(1, M):
        wrest = Wall - Wp[m]
        for j in range(n):
            v = V[m, j]
            if not np.isfinite(v):
                continue
            for k in range(n):
                if (m >> k) & 1:
                    continue
                d = D[j, k]
                if not np.isfinite(d):
                    continue
                nv = v + wrest * d
                m2 = m | (1 << k)
                if nv < V[m2, k]:
                    V[m2, k] = nv
                    P[m2, k] = j
    best = np.inf
    be = -1
    for e in range(n):
        tot = V[M - 1, e] + tailc[e]
        if tot < best:
            best = tot
            be = e
    order = np.full(n, -1, dtype=np.int64)
    m = M - 1
    j = be
    for pos in range(n - 1, -1, -1):
        order[pos] = j
        pj = P[m, j]
        m ^= (1 << j)
        j = pj
    return order, best


def reorder_prefix(model, order, k=10):
    """Exact re-ordering of the first <= k stops of `order`, tail kept in its order.
    Returns the new order if it does not lower the (clipped, budgeted) J, else `order`."""
    order = list(order)
    p = min(k, len(order))
    if p < 2:
        return order, {"hk_prefix": p, "improved": False}
    pre, tail = order[:p], order[p:]
    D = np.full((p + 1, p + 1), np.inf)
    idx = pre + [0]
    for a in range(p + 1):
        for b in range(p + 1):
            D[a, b] = model.D[idx[a], idx[b]]
    g = model.g[pre].astype(np.float64)
    wtail = float(sum(model.g[t] for t in tail))
    tailc = np.zeros(p)
    for e in range(p):
        # sum_tail g * (arrival relative to leaving pre[e]), tail order fixed
        t, prev, c = 0.0, pre[e], 0.0
        for x in tail:
            t += model.D[prev, x]
            c += model.g[x] * t
            prev = x
        tailc[e] = c if np.isfinite(c) else np.inf
    perm, _ = _hk_prefix(D, g, wtail, tailc)
    new = [pre[int(i)] for i in perm] + tail
    v0, v1 = model.value(order), model.value(new)
    if v1 > v0 + 1e-9:
        return new, {"hk_prefix": p, "improved": True, "dJ": v1 - v0}
    return order, {"hk_prefix": p, "improved": False}
