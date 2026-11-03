"""TOPO-002 §2.1 expected-gain model p(c) = P(UNKNOWN cell c is GT-FREE | features).

Features per UNKNOWN cell (computed online from obs/mean/var only, no GT):
  mean (LaMa occupancy prob), var, d_free = Euclid distance (m) to nearest observed FREE (cap 10),
  d_occ = distance (m) to nearest observed OCC (cap 5), max7 = max mean in a 0.7 m window,
  d_wall = distance (m) to nearest wall evidence (observed OCC or mean >= 0.5, cap 5).
  (DEV finding: LaMa mean on UNKNOWN is ~0 almost everywhere; wall proximity carries the signal.)
Population: UNKNOWN cells with d_free <= NEAR_M (10 m) — the cells a viewpoint raycast can reach first.
Models (<= 50 parameters):
  'logistic' : logistic regression on a fixed basis of the features (19 weights incl. bias), L2 1e-4
  'bins'     : 10 d_wall-bins x 5 d_free-bins table of smoothed GT-FREE rates (50 parameters)
Fit on DEV frames only (building-wise CV in eval_dev). Frozen model = JSON next to this file.

Planner API:  p_free(obs, mean, var, model=None) -> float32 array, same shape:
  observed FREE -> 1, observed OCC -> 0, UNKNOWN -> model probability.
  model=None loads FROZEN_PATH if present, else falls back to 1 - mean (uncalibrated; flagged in `fallback_used`).
"""
import os, json
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_PATH = os.path.join(HERE, 'pc_frozen.json')
RES = 0.1
D_FREE_CAP = 10.0; D_OCC_CAP = 5.0; D_WALL_CAP = 5.0; NEAR_M = 10.0
DW_EDGES = np.array([0, .15, .3, .5, .75, 1.0, 1.5, 2.0, 3.0, 4.0, 1e9])
DF_EDGES = np.array([0, 1, 2, 4, 8, 1e9])
fallback_used = False


def _dist(mask_zero_where, res):
    """Distance (m) to the nearest cell where mask is False (cv2, fast)."""
    return cv2.distanceTransform(mask_zero_where.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * res


def features(obs, mean, var, res=RES):
    """Returns (unknown mask, X float32 [n, 6] = mean, var, d_free, d_occ, max7, d_wall) for UNKNOWN cells."""
    obs = np.asarray(obs); unk = obs == 0.5
    mean = np.clip(np.nan_to_num(np.asarray(mean, np.float32)), 0, 1)
    df = _dist(obs != 0, res) if (obs == 0).any() else np.full(obs.shape, D_FREE_CAP, np.float32)
    do = _dist(obs != 1, res) if (obs == 1).any() else np.full(obs.shape, D_OCC_CAP, np.float32)
    k = max(1, int(round(0.35 / res)))
    mx = cv2.dilate(mean, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    wall = (obs == 1) | (mean >= 0.5)
    dw = _dist(~wall, res) if wall.any() else np.full(obs.shape, D_WALL_CAP, np.float32)
    X = np.stack([mean[unk], np.clip(np.asarray(var)[unk], 0, 1), np.minimum(df[unk], D_FREE_CAP),
                  np.minimum(do[unk], D_OCC_CAP), mx[unk], np.minimum(dw[unk], D_WALL_CAP)], 1).astype(np.float32)
    return unk, X


def _basis(X):
    m, v, df, do, mx, dw = (X[:, i] for i in range(6))
    lf = np.log1p(df); lo = np.log1p(do); sv = np.sqrt(v); lw = np.log1p(dw)
    B = np.stack([np.ones_like(m), m, m * m, sv, sv * m, lf, df / D_FREE_CAP, lo, do / D_OCC_CAP,
                  m * lf, (m >= 0.5).astype(np.float32), lf * lo, m * lo,
                  mx, lw, dw / D_WALL_CAP, (dw < 0.3).astype(np.float32), mx * lw, lw * lf], 1)
    return B.astype(np.float64)


class PCModel:
    def __init__(self, kind='logistic', params=None, meta=None):
        self.kind = kind; self.params = None if params is None else np.asarray(params, float); self.meta = meta or {}

    # ---- fit
    def fit(self, X, y, w=None):
        y = np.asarray(y, float); w = np.ones_like(y) if w is None else np.asarray(w, float)
        if self.kind == 'logistic':
            B = _basis(X); mu = B[:, 1:].mean(0); sd = B[:, 1:].std(0) + 1e-9
            Z = np.c_[B[:, :1], (B[:, 1:] - mu) / sd]
            lam = 1e-4

            def f(beta):
                z = Z @ beta; p = 1 / (1 + np.exp(-z))
                ll = -(w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum() / w.sum()
                g = Z.T @ (w * (p - y)) / w.sum()
                return ll + lam * beta[1:] @ beta[1:], g + np.r_[0, 2 * lam * beta[1:]]
            r = minimize(f, np.zeros(Z.shape[1]), jac=True, method='L-BFGS-B', options=dict(maxiter=500))
            beta = r.x.copy()
            # fold standardisation into raw-basis weights
            braw = np.r_[beta[0] - (beta[1:] * mu / sd).sum(), beta[1:] / sd]
            self.params = braw
        elif self.kind == 'bins':
            i, j = self._bin(X)
            n = np.zeros((10, 5)); k = np.zeros((10, 5))
            np.add.at(n, (i, j), w); np.add.at(k, (i, j), w * y)
            prior = (w * y).sum() / w.sum()
            self.params = ((k + 2 * prior) / (n + 2)).ravel()      # Beta smoothing toward the global rate
        else:
            raise ValueError(self.kind)
        return self

    def _bin(self, X):
        i = np.clip(np.searchsorted(DW_EDGES, X[:, 5], side='right') - 1, 0, 9)
        j = np.clip(np.searchsorted(DF_EDGES, X[:, 2], side='right') - 1, 0, 4)
        return i, j

    def predict(self, X):
        if self.kind == 'logistic':
            z = _basis(X) @ self.params
            return (1 / (1 + np.exp(-np.clip(z, -30, 30)))).astype(np.float32)
        i, j = self._bin(X)
        return self.params.reshape(10, 5)[i, j].astype(np.float32)

    @property
    def n_params(self):
        return 0 if self.params is None else int(self.params.size)

    # ---- io
    def save(self, path):
        json.dump(dict(kind=self.kind, params=self.params.tolist(), meta=self.meta), open(path, 'w'), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path)); return cls(d['kind'], d['params'], d.get('meta'))


_cached = {}


def get_model(path=FROZEN_PATH):
    if path not in _cached:
        _cached[path] = PCModel.load(path) if os.path.exists(path) else None
    return _cached[path]


def p_free(obs, mean, var, model=None, res=RES):
    """p(c) map (float32, obs shape): FREE 1, OCC 0, UNKNOWN model probability."""
    global fallback_used
    obs = np.asarray(obs)
    out = np.where(obs == 0, 1.0, 0.0).astype(np.float32)
    model = model if model is not None else get_model()
    unk, X = features(obs, mean, var, res)
    if model is None:
        fallback_used = True
        out[unk] = 1 - X[:, 0]
    else:
        fallback_used = False
        out[unk] = model.predict(X)
    return out


# ---- metrics (used by eval_dev)
def brier(p, y):
    return float(np.mean((p - y) ** 2))


def ece(p, y, nb=10):
    e = 0.0; edges = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, nb - 1)
    for b in range(nb):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def auroc(p, y):
    y = np.asarray(y).astype(bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    from scipy.stats import rankdata
    ranks = rankdata(p)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(p, y, nb=10):
    edges = np.linspace(0, 1, nb + 1); idx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, nb - 1)
    return [(float(p[idx == b].mean()), float(y[idx == b].mean()), int((idx == b).sum())) for b in range(nb) if (idx == b).any()]
