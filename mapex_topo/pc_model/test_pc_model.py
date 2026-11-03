"""Unit tests for pc_model on synthetic data. Run: python test_pc_model.py"""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pc_model as M


def synth(seed=0, n=60000):
    """Ground truth: P(free) = sigmoid(3 - 6*mean - 0.5*d_free); a calibrated model must recover it."""
    rs = np.random.RandomState(seed)
    m = rs.beta(0.5, 1.5, n); dw = rs.exponential(1.0, n).clip(0, 5)
    X = np.c_[m, rs.rand(n) * 0.1, rs.exponential(3, n).clip(0, 10), rs.rand(n) * 5, np.maximum(m, rs.rand(n) * 0.3), dw].astype(np.float32)
    pt = 1 / (1 + np.exp(-(-1 + 3 * np.log1p(dw) - 2 * X[:, 0] + 0.1 * X[:, 2])))
    return X, (rs.rand(n) < pt).astype(float), pt


def test_fit_calibrated():
    X, y, pt = synth(0); Xt, yt, ptt = synth(1)
    for kind in ('logistic', 'bins'):
        m = M.PCModel(kind).fit(X, y)
        assert m.n_params <= 50
        p = m.predict(Xt)
        assert M.ece(p, yt) < 0.03, (kind, M.ece(p, yt))
        assert M.brier(p, yt) < M.brier((Xt[:, 0] < 0.5).astype(float), yt), kind
        assert M.auroc(p, yt) > 0.75
        print(kind, 'ECE %.3f Brier %.3f AUROC %.3f' % (M.ece(p, yt), M.brier(p, yt), M.auroc(p, yt)))


def test_metrics():
    y = np.array([0, 0, 1, 1.]); assert M.auroc(np.array([.1, .2, .8, .9]), y) == 1.0
    assert M.auroc(np.array([.5, .5, .5, .5]), y) == 0.5
    assert abs(M.brier(np.array([1, 0, 1, 0.]), np.array([1, 0, 0, 1.])) - 0.5) < 1e-9
    assert M.ece(np.array([.2] * 10), np.array([0.] * 8 + [1.] * 2)) < 1e-9


def test_p_free_map_and_io():
    obs = np.full((50, 60), 0.5, np.float32); obs[20:30, 10:40] = 0; obs[19, 10:40] = 1
    mean = np.full(obs.shape, 0.2, np.float32); var = np.zeros_like(mean)
    X, y, _ = synth(0, 20000)
    m = M.PCModel('logistic').fit(X, y)
    f = os.path.join(tempfile.mkdtemp(), 'm.json'); m.save(f); m2 = M.PCModel.load(f)
    p = M.p_free(obs, mean, var, model=m2)
    assert p.shape == obs.shape and p.dtype == np.float32
    assert (p[obs == 0] == 1).all() and (p[obs == 1] == 0).all()
    u = obs == 0.5
    assert 0 <= p[u].min() and p[u].max() <= 1
    assert p[18, 25] < p[40, 25], 'cell next to a wall must be less likely free than one far from walls'
    pf = M.p_free(obs, mean, var, model=None)
    if not os.path.exists(M.FROZEN_PATH):
        assert M.fallback_used and np.allclose(pf[u], 0.8)
    print('p_free OK')


if __name__ == '__main__':
    test_metrics(); test_fit_calibrated(); test_p_free_map_and_io(); print('ALL PASS')
