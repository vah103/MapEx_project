"""Unit tests for topo_l1 v2 (TOPO-002) on synthetic maps (corridor + rooms + doors). Run: python test_topo_l1.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topo_l1 import build_topo

PPM = 10


def synth(unknown_rooms=False):
    """30 x 70 m map: horizontal corridor (2 m wide, 50 m long) at rows 140-160,
    4 rooms 5x6 m above and 4 below, each with a 0.9 m door; one 10x10 m hall at the east end
    with 3 exits. Walls 0.2 m. Returns obs, mean, var, robot_rc, gt (room ids), door list."""
    H, W = 300, 700
    occ = np.ones((H, W), bool)
    gt = np.zeros((H, W), int)
    rid = 1
    occ[140:160, 50:550] = False; gt[140:160, 50:550] = rid; corr = rid; rid += 1
    doors = []
    for k in range(4):
        x0 = 80 + k * 110
        # room above: rows 78..138 (walls at 138..140)
        occ[78:138, x0:x0 + 50] = False; gt[78:138, x0:x0 + 50] = rid; rid += 1
        occ[138:140, x0 + 20:x0 + 29] = False; gt[138:140, x0 + 20:x0 + 29] = gt[100, x0 + 1]
        doors.append((139, x0 + 24))
        # room below
        occ[162:222, x0:x0 + 50] = False; gt[162:222, x0:x0 + 50] = rid; rid += 1
        occ[160:162, x0 + 20:x0 + 29] = False; gt[160:162, x0 + 20:x0 + 29] = gt[200, x0 + 1]
        doors.append((161, x0 + 24))
    # hall 10 x 10 m east, joined by a 1.0 m door
    occ[100:200, 553:653] = False; gt[100:200, 553:653] = rid; hall = rid; rid += 1
    occ[145:155, 550:553] = False; gt[145:155, 550:553] = hall
    doors.append((150, 551))
    # two dead-end small rooms off the hall (so the hall has degree >= 3)
    occ[60:97, 580:620] = False; gt[60:97, 580:620] = rid; rid += 1
    occ[97:100, 595:604] = False; gt[97:100, 595:604] = rid - 1; doors.append((98, 599))
    occ[203:240, 580:620] = False; gt[203:240, 580:620] = rid; rid += 1
    occ[200:203, 595:604] = False; gt[200:203, 595:604] = rid - 1; doors.append((201, 599))
    obs = np.where(occ, 1.0, 0.0).astype(np.float32)
    mean = obs.copy()
    if unknown_rooms:
        # rooms below the corridor unknown, LaMa predicts them correctly-ish (walls 0.8, free 0.1)
        sel = np.zeros_like(occ); sel[161:240, 60:540] = True
        obs[sel] = 0.5
        mean[sel] = np.where(occ[sel], 0.8, 0.1)
    var = np.zeros_like(mean)
    return obs, mean, var, (150, 60), gt, doors, corr, hall


def _match(doors_pred, doors_gt, tol=10):
    hit = 0
    for g in doors_gt:
        if any(np.hypot(d['rc'][0] - g[0], d['rc'][1] - g[1]) <= tol for d in doors_pred):
            hit += 1
    prec = sum(any(np.hypot(d['rc'][0] - g[0], d['rc'][1] - g[1]) <= tol for g in doors_gt) for d in doors_pred)
    return hit / len(doors_gt), prec / max(1, len(doors_pred))


def test_known_map():
    obs, mean, var, rc, gt, doors, corr, hall = synth()
    tm = build_topo(obs, mean, var, rc, PPM)
    rec, prec = _match(tm.doors, doors)
    assert rec >= 0.9 and prec >= 0.9, (rec, prec, [d['rc'] for d in tm.doors])
    # corridor region
    cid = tm.region_id[150, 300]
    creg = [r for r in tm.regions if r['id'] == cid][0]
    assert creg['cls'] == 'corridor', creg
    assert creg['degree'] >= 8, creg['degree']
    # rooms are rooms and pure
    for (r, c) in [(100, 100), (200, 320), (80, 600)]:
        reg = [x for x in tm.regions if x['id'] == tm.region_id[r, c]][0]
        assert reg['cls'] == 'room', reg
        cells = tm.region_id == reg['id']
        ids, cnt = np.unique(gt[cells], return_counts=True)
        assert cnt.max() / cnt.sum() > 0.95
    hreg = [x for x in tm.regions if x['id'] == tm.region_id[150, 600]][0]
    assert hreg['cls'] == 'hall', hreg
    assert tm.backbone_mask[150, 300] and tm.backbone_mask[150, 600] and not tm.backbone_mask[100, 100]
    assert len(tm.regions) == 1 + 8 + 1 + 2, len(tm.regions)
    assert tm.timing_s < 3.0
    # interface fields
    for r in tm.regions:
        for k in ('id', 'cls', 'p', 'area_m2', 'unknown_free_m2', 'degree', 'door_ids', 'centroid_rc', 'skeleton_rc'):
            assert k in r
        assert abs(sum(r['p'].values()) - 1) < 1e-6
    for d in tm.doors:
        assert d['regions'][0] != d['regions'][1] and 0 < d['p_exist'] <= 1
    assert tm.region_id.dtype == np.int32 and tm.backbone_mask.dtype == bool
    print('known map OK', tm)


def test_unknown_predicted_rooms():
    obs, mean, var, rc, gt, doors, corr, hall = synth(unknown_rooms=True)
    tm = build_topo(obs, mean, var, rc, PPM)
    rec, prec = _match(tm.doors, doors)
    assert rec >= 0.9 and prec >= 0.9, (rec, prec)
    reg = [x for x in tm.regions if x['id'] == tm.region_id[175, 100]][0]
    assert reg['cls'] == 'room' and reg['unknown_free_m2'] > 10, reg
    assert reg['id'] in tm.pocket_ids
    print('predicted rooms OK', tm)


def test_temporal_and_padding():
    obs, mean, var, rc, gt, doors, corr, hall = synth()
    t1 = build_topo(obs, mean, var, rc, PPM)
    t2 = build_topo(obs, mean, var, rc, PPM, prev=t1)
    a = {r['id']: r['cls'] for r in t1.regions}; b = {r['id']: r['cls'] for r in t2.regions}
    assert a == b, 'ids/classes must be stable on identical input'
    # padded input (pad 500 with UNKNOWN / mean 1, like MapEx) gives the same structure, shifted
    pad = 500
    po = np.pad(obs, pad, constant_values=0.5); pm = np.pad(mean, pad, constant_values=1.0)
    pv = np.pad(var, pad)
    t3 = build_topo(po, pm, pv, (rc[0] + pad, rc[1] + pad), PPM)
    assert t3.region_id.shape == po.shape
    assert np.array_equal(t3.region_id[pad:-pad, pad:-pad] >= 0, t1.region_id >= 0)
    assert len(t3.doors) == len(t1.doors)
    print('temporal + padding OK')


def test_pfree_path():
    """p(c) array replaces mean thresholds + range: full predicted rooms are recovered."""
    obs, mean, var, rc, gt, doors, corr, hall = synth(unknown_rooms=True)
    pf = np.where(obs == 0.5, 1 - mean, 0).astype(np.float32)
    tm = build_topo(obs, mean, var, rc, PPM, pfree=pf)
    rec, prec = _match(tm.doors, doors)
    assert rec >= 0.9 and prec >= 0.9, (rec, prec)
    reg = [x for x in tm.regions if x['id'] == tm.region_id[210, 100]][0]
    assert reg['cls'] == 'room' and reg['unknown_free_m2'] > 25, reg
    print('pfree OK', tm)


def test_chains_viewpoints():
    obs, mean, var, rc, gt, doors, corr, hall = synth()
    tm = build_topo(obs, mean, var, rc, PPM)
    cid = tm.region_id[150, 300]
    ch = [c for c in tm.chains if c['region_id'] == cid]
    assert len(ch) == 1 and ch[0]['length_m'] >= 40 and ch[0]['cls'] == 'corridor', ch
    assert 1.5 <= ch[0]['width_med_m'] <= 2.5
    kinds = {}
    for v in tm.viewpoints:
        kinds[v['kind']] = kinds.get(v['kind'], 0) + 1
        assert tm.region_id[v['rc']] >= 0 or v['kind'] == 'door'
    assert kinds.get('chain', 0) >= 30 and kinds.get('door', 0) == len(tm.doors) and kinds.get('pocket', 0) >= 10, kinds
    # room 5x6 m with one door is never a corridor even if its chain is long-ish
    for r in tm.regions:
        if r['degree'] < 2:
            assert r['cls'] != 'corridor'
    print('chains/viewpoints OK', kinds)


def synth_jamb():
    """Two 8x8 m rooms side by side, 0.2 m wall with a 2.0 m opening (too wide for clearance doors)."""
    H, W = 120, 200
    occ = np.ones((H, W), bool)
    occ[20:100, 20:100] = False; occ[20:100, 102:182] = False
    occ[50:70, 100:102] = False                       # 2.0 m opening
    obs = np.where(occ, 1.0, 0.0).astype(np.float32)
    return obs, obs.copy(), np.zeros_like(obs), (60, 50)


def test_jamb_door():
    obs, mean, var, rc = synth_jamb()
    t0 = build_topo(obs, mean, var, rc, PPM, params=dict(JAMB=False))
    assert len(t0.doors) == 0
    tm = build_topo(obs, mean, var, rc, PPM)
    assert len(tm.doors) == 1 and tm.doors[0]['src'] == 'jamb', tm.doors
    d = tm.doors[0]
    assert abs(d['rc'][0] - 60) <= 3 and abs(d['rc'][1] - 101) <= 3 and 1.8 <= d['width_m'] <= 2.2, d
    assert len(tm.regions) == 2 and tm.region_id[60, 50] != tm.region_id[60, 150]
    print('jamb OK', tm)


def test_gap_closing():
    """Two unknown predicted rooms separated by a dashed predicted wall (mean 0.8, gaps 0.4 m): no leak."""
    H, W = 160, 220
    obs = np.full((H, W), 0.5, np.float32); mean = np.full((H, W), 0.05, np.float32)
    obs[:, :10] = 1; obs[:, -10:] = 1; obs[:10] = 1; obs[-10:] = 1
    obs[70:90, 10:210] = 0                             # observed corridor
    obs[69, 10:210] = 1; obs[90, 10:210] = 1
    obs[69, 50:59] = 0; obs[69, 150:159] = 0            # two 0.9 m doors from corridor to the rooms above
    mean[10:69, 108:112] = 0.8                          # predicted wall between rooms above...
    for r in range(14, 69, 10):
        mean[r:r + 4, 108:112] = 0.2                    # ...with 0.4 m gaps every 1 m
    t_no = build_topo(obs, mean, None, (80, 100), PPM, params=dict(GAP_CLOSE_M=0, OPEN_PRED_M=None, PRED_RANGE_M=None))
    t_gc = build_topo(obs, mean, None, (80, 100), PPM, params=dict(OPEN_PRED_M=None, PRED_RANGE_M=None))
    assert t_no.region_id[40, 60] == t_no.region_id[40, 160], 'precondition: rooms leak without gap closing'
    assert t_gc.region_id[40, 60] != t_gc.region_id[40, 160] and t_gc.region_id[40, 60] >= 0, 'gap closing failed'
    print('gap closing OK')


if __name__ == '__main__':
    test_known_map(); test_unknown_predicted_rooms(); test_temporal_and_padding()
    test_pfree_path(); test_chains_viewpoints(); test_jamb_door(); test_gap_closing()
    print('ALL PASS')
