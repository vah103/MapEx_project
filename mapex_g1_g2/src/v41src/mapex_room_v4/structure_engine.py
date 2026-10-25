from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import math
import numpy as np
import cv2
from scipy import ndimage as ndi
from .geometry_helpers import find_objects, extrapolate_to_junction, nested_bay_hypotheses
ROOM, CORRIDOR, OBSTACLE, WALL = (1, 2, 3, 4)

@dataclass(frozen=True)
class Params:
    pred_wall: float = 0.73
    max_wall_var: float = 0.11
    unknown_free_mean: float = 0.65
    wall_min_m: float = 0.7
    virtual_gap_m: float = 2.2
    junction_reach_m: float = 1.2
    min_region_cells: int = 45
    min_room_cells: int = 110
    door_min_m: float = 0.5
    door_max_m: float = 2.6
    pair_same: float = 0.22
    room_door_penalty: float = 0.18
    corridor_link_reward: float = 0.24
    corridor_min_evidence: float = 0.38

def _morph(mask, typ, shape):
    return cv2.morphologyEx(mask.astype(np.uint8), typ, np.ones(shape, np.uint8)).astype(bool)

def _components(mask):
    return cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=4)

def _extract_structural(obs, mean, var, res, cfg):
    ob = obs == 2
    predicted = (obs == 1) & (mean >= cfg.pred_wall) & (var <= cfg.max_wall_var)
    solid = ob | predicted
    furniture, objects = find_objects(solid)
    wall = solid & ~furniture
    k = max(5, round(cfg.wall_min_m / res))
    horizontal = _morph(wall, cv2.MORPH_OPEN, (1, k))
    vertical = _morph(wall, cv2.MORPH_OPEN, (k, 1))
    kw = max(3, round(cfg.virtual_gap_m / res)) | 1
    hc = _morph(horizontal, cv2.MORPH_CLOSE, (1, kw))
    vc = _morph(vertical, cv2.MORPH_CLOSE, (kw, 1))
    projected = extrapolate_to_junction(vc, hc, max(3, round(cfg.junction_reach_m / res)))
    closure = hc | vc | projected
    barrier = ndi.binary_dilation(closure, iterations=1)
    virtual = barrier & ~solid
    floor = ((obs == 0) | (obs == 1) & (mean < cfg.unknown_free_mean)) & ~solid
    return {'solid': solid, 'observed_wall': ob, 'pred_wall': predicted, 'furniture': furniture, 'wall': wall, 'horizontal': hc, 'vertical': vc, 'closure': closure, 'barrier': barrier, 'virtual': virtual, 'floor': floor, 'objects': objects}

def _make_regions(state, obs, cfg):
    n, lbl, stats, cent = _components(~state['barrier'])
    regions = {}
    for ix in range(1, n):
        x, y, w, h, area = map(int, stats[ix])
        if area < cfg.min_region_cells:
            continue
        idx = lbl == ix
        floor_cells = int(np.sum(idx & state['floor']))
        if floor_cells < max(20, int(0.18 * area)):
            continue
        boundary = idx & ~ndi.binary_erosion(idx, iterations=1)
        surround = ndi.binary_dilation(state['wall'], iterations=2)
        support = float(np.sum(boundary & surround)) / max(1, int(boundary.sum()))
        observed_floor = float(np.sum(idx & (obs == 0))) / max(1, floor_cells)
        aspect = max(w, h) / max(1, min(w, h))
        contour, _ = cv2.findContours(idx.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = []
        if contour:
            p = max(contour, key=cv2.contourArea)
            poly = cv2.approxPolyDP(p, max(1, 0.007 * cv2.arcLength(p, True)), True).reshape(-1, 2).tolist()
        regions[ix] = {'id': ix, 'bbox_xywh': [x, y, w, h], 'area_cells': floor_cells, 'face_area': int(area), 'centroid_rc': [round(float(cent[ix, 1]), 1), round(float(cent[ix, 0]), 1)], 'aspect': round(float(aspect), 3), 'boundary_support': round(support, 3), 'observed_support': round(observed_floor, 3), 'polygon_xy': poly}
    return (lbl, regions)

def _build_connectivity(state, lbl, regions, obs, res, cfg):
    valid = set(regions)
    if not valid:
        return []
    pair_pixels = defaultdict(list)
    H, W = lbl.shape
    for rr, cc in np.argwhere(state['virtual']):
        rr = int(rr)
        cc = int(cc)
        if obs[rr, cc] == 2:
            continue
        pairs = []
        for d in (2, 3, 4):
            if cc - d >= 0 and cc + d < W:
                a = int(lbl[rr, cc - d])
                b = int(lbl[rr, cc + d])
                if a != b and a in valid and (b in valid):
                    pairs.append((a, b))
            if rr - d >= 0 and rr + d < H:
                a = int(lbl[rr - d, cc])
                b = int(lbl[rr + d, cc])
                if a != b and a in valid and (b in valid):
                    pairs.append((a, b))
        for a, b in set(((min(a, b), max(a, b)) for a, b in pairs)):
            pair_pixels[a, b].append((rr, cc))
    edges = []
    for (a, b), cells in sorted(pair_pixels.items()):
        if len(cells) < 3:
            continue
        pix = np.asarray(cells, dtype=int)
        ymin, xmin = np.min(pix, axis=0)
        ymax, xmax = np.max(pix, axis=0)
        local = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
        local[pix[:, 0] - ymin, pix[:, 1] - xmin] = 1
        ns, lab, st, ctr = _components(ndi.binary_dilation(local, iterations=1))
        for i in range(1, ns):
            yy, xx = np.nonzero(lab == i)
            if len(yy) < 3:
                continue
            ycoords = yy + ymin
            xcoords = xx + xmin
            extent = max(int(np.max(ycoords) - np.min(ycoords) + 1), int(np.max(xcoords) - np.min(xcoords) + 1))
            if extent > max(60, round(4.0 / res)):
                continue
            center_r = float(np.mean(ycoords))
            center_c = float(np.mean(xcoords))
            yy0 = max(0, int(center_r) - 3)
            yy1 = min(H, int(center_r) + 4)
            xx0 = max(0, int(center_c) - 3)
            xx1 = min(W, int(center_c) + 4)
            local_obs = obs[yy0:yy1, xx0:xx1]
            obs_open = float(np.mean(local_obs == 0))
            pred_uncertain = float(np.mean(state['pred_wall'][yy0:yy1, xx0:xx1]))
            boundary_virtual = float(np.mean(state['virtual'][yy0:yy1, xx0:xx1]))
            opening_score = round(float(0.63 * obs_open + 0.25 * (1 - pred_uncertain) + 0.12 * boundary_virtual), 3)
            status = 'OBSERVED_FREE_GAP_HYPOTHESIS' if obs_open >= 0.45 else 'UNKNOWN_OPENING_HYPOTHESIS'
            edges.append({'u': a, 'v': b, 'at_rc': [round(center_r, 1), round(center_c, 1)], 'opening_m': round(float(extent * res), 2), 'observed_gap_fraction': round(obs_open, 3), 'opening_score': opening_score, 'status': status, 'connector_cells': int(len(yy))})
    return edges

def _graph_reason(regions, edges, cfg):
    ids = sorted(regions)
    if not ids:
        return ({}, {}, {})
    contacts = defaultdict(dict)
    for e in edges:
        a, b = (e['u'], e['v'])
        q = e['opening_score']
        contacts[a][b] = max(contacts[a].get(b, 0), q)
        contacts[b][a] = max(contacts[b].get(a, 0), q)
    scores = {}
    for k in ids:
        r = regions[k]
        degree = len(contacts[k])
        area = r['area_cells']
        aspect = r['aspect']
        supp = r['boundary_support']
        w, h = r['bbox_xywh'][2:]
        narrow = min(w, h) <= 29
        slender = aspect >= 1.55 and narrow and (degree >= 2)
        hub = degree >= 5
        mid = degree >= 4
        room = 0.46 + 0.38 * supp + 0.08 * r['observed_support'] + 0.07 * min(area, 1200) / 1200 - (0.25 if mid else 0) - (0.3 if hub else 0)
        hall = 0.12 + (0.91 if hub else 0.22 if mid else 0) + (0.78 if slender else 0) + 0.06 * min(degree, 5) / 5 + 0.02 * min(area, 3000) / 3000
        scores[k] = {'ROOM': round(room, 4), 'CORRIDOR': round(hall, 4), 'degree': degree, 'narrow_passage': bool(slender), 'hub': bool(hub)}
    choices = {k: 'CORRIDOR' if scores[k]['CORRIDOR'] > scores[k]['ROOM'] else 'ROOM' for k in ids}
    for _ in range(12):
        updated = {}
        for k in ids:
            room_val = scores[k]['ROOM']
            hall_val = scores[k]['CORRIDOR']
            for other, q in contacts[k].items():
                if choices[other] == 'CORRIDOR':
                    hall_val += 0.055 * q
                    room_val += 0.02 * q
                else:
                    hall_val += 0.02 * q
                    room_val += 0.005 * q
            if not contacts[k]:
                hall_val -= 0.22
            if len(contacts[k]) == 1:
                hall_val -= 0.11
            updated[k] = 'CORRIDOR' if hall_val > room_val else 'ROOM'
        if updated == choices:
            break
        choices = updated
    margins = {}
    for k in ids:
        d = scores[k]['CORRIDOR'] - scores[k]['ROOM']
        for other, q in contacts[k].items():
            d += (0.035 if choices[other] == 'CORRIDOR' else 0.015) * q
        margins[k] = round(float(d), 3)
    return (choices, scores, margins)

def _nested_rooms(state, obs, floor, sem, res):
    proposals = nested_bay_hypotheses(state['horizontal'], state['vertical'], sem, floor, obs)
    H, W = obs.shape
    picked = []
    occupied = np.zeros((H, W), bool)
    for score, (x0, y0, x1, y1), left, right, obsfrac in proposals:
        inner = (slice(y0 + 2, y1 - 2), slice(x0 + 2, x1 - 2))
        if (y1 - y0) * (x1 - x0) < 160 or y1 - y0 < 12:
            continue
        proposed = (sem[inner] == CORRIDOR) & floor[inner]
        if proposed.sum() < 100:
            continue
        if np.sum(occupied[inner] & proposed) > 0.18 * proposed.sum():
            continue
        occupied[inner] |= proposed
        picked.append({'bbox_xyxy': [x0, y0, x1, y1], 'boundary_support': round(float((left + right) / 2), 3), 'observed_support': round(float(obsfrac), 3), 'score': round(float(score), 3), 'mask_inner': inner})
    return picked

def _polygons_for_mask(m, min_area=20):
    ct, _ = cv2.findContours(m.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(ct, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area:
            continue
        p = cv2.approxPolyDP(c, epsilon=1.3, closed=True).reshape(-1, 2)
        if len(p) >= 3:
            out.append(p.astype(int).tolist())
    return out

def infer(observed: np.ndarray, mean: np.ndarray, variance: np.ndarray, *, resolution_m: float=0.1, config: Params=Params(), snapshot_id: str=''):
    obs = np.asarray(observed)
    mean = np.asarray(mean)
    var = np.asarray(variance)
    if obs.ndim != 2 or obs.shape != mean.shape or obs.shape != var.shape:
        raise ValueError('shape mismatch')
    if obs.size > 1000000 or resolution_m <= 0:
        raise ValueError('invalid grid/resolution')
    if not np.isin(obs, [0, 1, 2]).all():
        raise ValueError('observed must encode FREE=0 UNKNOWN=1 OCCUPIED=2')
    if not np.isfinite(mean).all() or not np.isfinite(var).all():
        raise ValueError('invalid probabilities')
    if mean.min() < 0 or mean.max() > 1 or var.min() < 0:
        raise ValueError('out of bounds')
    before = hashlib.sha256(obs.tobytes()).hexdigest()
    st = _extract_structural(obs, mean, var, resolution_m, config)
    labels, regions = _make_regions(st, obs, config)
    edges = _build_connectivity(st, labels, regions, obs, resolution_m, config)
    choices, unaries, margins = _graph_reason(regions, edges, config)
    face_sem = np.zeros(obs.shape, np.uint8)
    for k, cls in choices.items():
        face_sem[labels == k] = CORRIDOR if cls == 'CORRIDOR' else ROOM
    if not np.any(face_sem):
        face_sem[st['floor']] = ROOM
    else:
        inds = ndi.distance_transform_edt(face_sem == 0, return_distances=False, return_indices=True)
        face_sem = face_sem[tuple(inds)]
    sem = np.zeros(obs.shape, np.uint8)
    sem[st['floor']] = face_sem[st['floor']]
    sem[st['solid'] & ~st['furniture']] = WALL
    sem[st['furniture']] = OBSTACLE
    nested = _nested_rooms(st, obs, st['floor'], sem, resolution_m)
    nested_masks = []
    for n in nested:
        mask = np.zeros(obs.shape, bool)
        sel = n.pop('mask_inner')
        mask[sel] = (sem[sel] == CORRIDOR) & st['floor'][sel]
        if mask.sum() < 100:
            continue
        sem[mask] = ROOM
        nested_masks.append((n, mask))
    rooms = []
    corridors = []
    nodes = []
    room_n = 0
    hall_n = 0
    for k in sorted(regions):
        r = regions[k].copy()
        typ = choices[k]
        r['structural_role'] = typ
        r['role_margin_heuristic'] = margins[k]
        r['unary_evidence'] = unaries[k]
        r['local_region_id'] = str(k)
        if typ == 'ROOM' and r['area_cells'] >= config.min_room_cells:
            region_mask = (labels == k) & (sem == ROOM)
            polys = _polygons_for_mask(region_mask, 80)
            if polys:
                room_n += 1
                r['id'] = f'R{room_n:03d}'
                r['polygons_xy'] = polys
                r['tentative'] = bool(r['observed_support'] < 0.6 or r['boundary_support'] < 0.38 or abs(margins[k]) < 0.12)
                rooms.append(r)
        if typ == 'CORRIDOR':
            region_mask = (labels == k) & (sem == CORRIDOR)
            polys = _polygons_for_mask(region_mask, 80)
            if polys:
                hall_n += 1
                r['id'] = f'C{hall_n:03d}'
                r['polygons_xy'] = polys
                corridors.append(r)
        nodes.append({'local_region_id': k, 'role': typ, 'centroid_rc': r['centroid_rc'], 'area_cells': r['area_cells'], 'degree': unaries[k]['degree'], 'role_margin_heuristic': margins[k]})
    doors = [{**e, 'u_role': choices[e['u']], 'v_role': choices[e['v']]} for e in edges]
    for n, mask in nested_masks:
        p = _polygons_for_mask(mask, 75)
        if not p:
            continue
        room_n += 1
        nr = {'id': f'R{room_n:03d}', 'local_region_id': 'NESTED', 'structural_role': 'ROOM', 'area_cells': int(mask.sum()), 'bbox_xyxy': n['bbox_xyxy'], 'boundary_support': n['boundary_support'], 'observed_support': n['observed_support'], 'tentative': bool(n['observed_support'] < 0.65 or n['boundary_support'] < 0.65), 'source': 'OPPOSING_WALL_BAY_HYPOTHESIS', 'polygons_xy': p}
        rooms.append(nr)
        x0, y0, x1, y1 = n['bbox_xyxy']
        cy = (y0 + y1) / 2
        cx = (x0 + x1) / 2
        nodes.append({'local_region_id': nr['id'], 'role': 'ROOM', 'centroid_rc': [cy, cx], 'area_cells': nr['area_cells'], 'degree': 1, 'role_margin_heuristic': None})
        doors.append({'u': nr['id'], 'v': 'CIRCULATION_PARENT_UNRESOLVED', 'at_rc': [cy, cx], 'status': 'ROOM_BAY_OPENING_HYPOTHESIS', 'opening_score': None, 'warning': 'Corridor parent is not reliably associated'})
    assert np.isin(sem[st['floor']], [ROOM, CORRIDOR]).all()
    assert before == hashlib.sha256(obs.tobytes()).hexdigest()
    report = {'version': 'V4_STRUCTURAL_GRAPH_R1', 'snapshot_id': snapshot_id, 'rooms': rooms, 'corridors': corridors, 'door_hypotheses': doors, 'graph': {'nodes': nodes, 'edges': [{'u': e['u'], 'v': e['v'], 'status': e['status'], 'opening_score': e['opening_score']} for e in doors]}, 'obstacle_candidates': st['objects'], 'diagnostics': {'status': 'OFFLINE_SHADOW_UNVALIDATED', 'rooms': len(rooms), 'corridors': len(corridors), 'graph_regions': len(regions), 'graph_edges': len(edges), 'door_hypotheses': len(doors), 'nested_bay_rooms': len(nested_masks), 'virtual_barrier_cells': int(st['virtual'].sum()), 'observed_free': int((obs == 0).sum()), 'classified_observed_free': int(np.isin(sem[obs == 0], [ROOM, CORRIDOR]).sum()), 'estimated_floor_cells': int(st['floor'].sum()), 'classified_estimated_floor_cells': int(np.isin(sem[st['floor']], [ROOM, CORRIDOR]).sum()), 'input_hashes': {'observed': before, 'mean': hashlib.sha256(mean.tobytes()).hexdigest(), 'variance': hashlib.sha256(var.tobytes()).hexdigest()}, 'warnings': ['Manual E22/reference forbidden as inference input', 'Geometric/graph role scores are uncalibrated', 'Door edges are hypotheses, NOT verified physical doors', 'Manhattan bias and map-specific development history', 'Graph cannot prove correctness or actionability']}}
    return {'semantic': sem, 'virtual_barriers': st['virtual'], 'wall_observed': st['observed_wall'], 'wall_predicted': st['pred_wall'], 'floor': st['floor'], 'report': report}
