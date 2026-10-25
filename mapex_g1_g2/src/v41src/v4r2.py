from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import math
import numpy as np
import cv2
from scipy import ndimage as ndi
from mapex_room_v4.structure_engine import _extract_structural, _make_regions, _build_connectivity, Params, ROOM, CORRIDOR, WALL, OBSTACLE
from mapex_room_v4.geometry_helpers import nested_bay_hypotheses

@dataclass(frozen=True)
class RoleParams:
    door_min_m: float = 0.5
    door_max_m: float = 2.6
    pair_same: float = 0.22
    room_door_penalty: float = 0.18
    corridor_link_reward: float = 0.24
    corridor_min_evidence: float = 0.38
    room_wall_reward: float = 0.35
    max_iterations: int = 8

def _t_roles(st, obs, resolution_m):
    barrier = st['barrier']
    floor = st['floor']
    nr, lab, stats, cent = cv2.connectedComponentsWithStats((~barrier).astype('uint8'), 4)
    faces = []
    for k in range(1, nr):
        x, y, w, h, area = map(int, stats[k])
        if area < 55:
            continue
        faces.append({'id': k, 'area': area, 'bbox': [x, y, w, h], 'elongation': round(max(w, h) / max(1, min(w, h)), 3)})
    face_ids = set((f['id'] for f in faces))
    adj = defaultdict(set)
    vl, nv = ndi.label(st['virtual'], structure=np.ones((3, 3), bool))
    for j, s in enumerate(ndi.find_objects(vl), 1):
        if s is None:
            continue
        component = vl[s] == j
        if component.sum() > 500:
            continue
        yy, xx = s
        pad = (slice(max(0, yy.start - 3), min(lab.shape[0], yy.stop + 3)), slice(max(0, xx.start - 3), min(lab.shape[1], xx.stop + 3)))
        near = [int(v) for v in np.unique(lab[pad]) if int(v) > 0]
        if len(near) > 7:
            continue
        for i in near:
            adj[i].update((v for v in near if v != i))
    if faces:
        biggest = max(faces, key=lambda f: f['area'])
        biggest_area = biggest['area']
        corridor_ids = {biggest['id']}
        for f in faces:
            if f['id'] in corridor_ids:
                continue
            neighborhood = sum((other in face_ids for other in adj[f['id']]))
            if neighborhood >= 3 and f['area'] >= 600 and (f['elongation'] >= 2.3) and (f['area'] >= biggest_area * 0.12):
                corridor_ids.add(f['id'])
        bx, by, bw, bh = biggest['bbox']
        for f in faces:
            if f['id'] in corridor_ids:
                continue
            x, y, w, h = f['bbox']
            if min(w, h) > max(24, round(2.4 / resolution_m)) or f['elongation'] < 2.1 or f['area'] < 430:
                continue
            overlap_y = max(0, min(y + h, by + bh) - max(y, by))
            overlap_x = max(0, min(x + w, bx + bw) - max(x, bx))
            near_x = abs(x + w - bx) <= 8 or abs(x - (bx + bw)) <= 8
            near_y = abs(y + h - by) <= 8 or abs(y - (by + bh)) <= 8
            if near_x and overlap_y >= min(h, bh) * 0.5 or (near_y and overlap_x >= min(w, bw) * 0.5):
                corridor_ids.add(f['id'])
    else:
        corridor_ids = set()
    base = np.zeros_like(obs, dtype=np.uint8)
    for f in faces:
        base[lab == f['id']] = CORRIDOR if f['id'] in corridor_ids else ROOM
    if not np.any(base):
        base[floor] = ROOM
    else:
        inds = ndi.distance_transform_edt(base == 0, return_distances=False, return_indices=True)
        base = base[tuple(inds)]
    return (base, lab, faces, adj, corridor_ids)

def _graph_roles(st, obs, resolution_m, legacy_base, cfg: RoleParams):
    if cfg.door_min_m >= cfg.door_max_m or cfg.max_iterations < 1:
        raise ValueError('invalid graph configuration')
    labels, regions = _make_regions(st, obs, Params(min_region_cells=55))
    edges = _build_connectivity(st, labels, regions, obs, resolution_m, Params())
    filtered = [e for e in edges if cfg.door_min_m <= e['opening_m'] <= cfg.door_max_m]
    contacts = defaultdict(dict)
    door_centers = defaultdict(list)
    for e in filtered:
        a, b = (e['u'], e['v'])
        q = e['opening_score']
        contacts[a][b] = max(q, contacts[a].get(b, 0))
        contacts[b][a] = max(q, contacts[b].get(a, 0))
        door_centers[a].append(e['at_rc'])
        door_centers[b].append(e['at_rc'])
    unaries = {}
    initial = {}
    through = {}
    for k, r in regions.items():
        rm = (labels == k) & st['floor']
        nroom = int(np.count_nonzero((legacy_base == ROOM) & rm))
        nhall = int(np.count_nonzero((legacy_base == CORRIDOR) & rm))
        prior = 'CORRIDOR' if nhall > nroom else 'ROOM'
        initial[k] = prior
        degree = len(contacts[k])
        width, height = r['bbox_xywh'][2:]
        aspect = r['aspect']
        narrow = min(width, height) * resolution_m <= 2.9
        slender = float(narrow and aspect >= 1.55)
        hub = float(min(max(degree - 2, 0) / 3, 1))
        cr, cc = r['centroid_rc']
        ports = door_centers[k]
        angular_opposition = 0.0
        if len(ports) >= 2:
            dirs = []
            for rr, xx in ports:
                v = np.array([rr - cr, xx - cc], float)
                norm = np.linalg.norm(v)
                if norm > 1.0:
                    dirs.append(v / norm)
            if len(dirs) >= 2:
                angular_opposition = max((float((-np.dot(a, b) + 1) / 2) for i, a in enumerate(dirs) for b in dirs[i + 1:]))
        passage = angular_opposition if degree >= 2 else 0.0
        through[k] = round(float(passage), 4)
        evidence = 0.45 * slender + 0.3 * passage + 0.25 * hub
        room_score = 0.36 + cfg.room_wall_reward * r['boundary_support'] + 0.1 * r['observed_support'] + (0.12 if prior == 'ROOM' else 0) - cfg.room_door_penalty * hub * passage * 0.5
        hall_score = 0.16 + 0.35 * slender + 0.22 * passage + 0.18 * hub + (0.2 if prior == 'CORRIDOR' else 0) + cfg.corridor_link_reward * math.log1p(degree) / math.log(6)
        unaries[k] = {'room': float(room_score), 'hall': float(hall_score), 'evidence': float(evidence), 'prior': prior, 'degree': degree, 'passage': passage, 'slender': bool(slender), 'hub': hub, 'observed_support': r['observed_support'], 'wall_support': r['boundary_support']}
    roles = dict(initial)
    margins = {}
    for iteration in range(cfg.max_iterations):
        proposed = {}
        for k, e in unaries.items():
            neighbor = contacts[k]
            hall_links = sum((q for other, q in neighbor.items() if roles.get(other) == 'CORRIDOR'))
            room_links = sum((q for other, q in neighbor.items() if roles.get(other) == 'ROOM'))
            degree = max(1, len(neighbor))
            hall = e['hall'] + cfg.pair_same * 0.5 * hall_links / degree
            room = e['room'] + cfg.pair_same * 0.5 * room_links / degree
            margin = hall - room
            margins[k] = float(margin)
            valid_corr = e['evidence'] >= cfg.corridor_min_evidence and e['degree'] >= 2 and (e['slender'] or e['hub'] >= 0.66) and (e['passage'] >= 0.35)
            demote_corr = margin < -0.25 and e['wall_support'] >= 0.45 and (e['observed_support'] >= 0.55) and (e['degree'] <= 1) and (regions[k]['aspect'] <= 1.6)
            if initial[k] == 'ROOM' and (not valid_corr):
                proposed[k] = 'ROOM'
            elif initial[k] == 'CORRIDOR' and (not demote_corr):
                proposed[k] = 'CORRIDOR'
            else:
                proposed[k] = 'CORRIDOR' if margin > 0 else 'ROOM'
        if proposed == roles:
            break
        roles = proposed
    out = legacy_base.copy()
    for k, role in roles.items():
        out[labels == k] = CORRIDOR if role == 'CORRIDOR' else ROOM
    return (out, {'regions': regions, 'edges': edges, 'filtered_edges': filtered, 'unaries': unaries, 'roles': roles, 'margins': margins, 'iterations': iteration + 1, 'changed_roles': sum((roles[k] != initial[k] for k in roles))})

def infer(observed, mean, variance, *, resolution_m=0.1, geometry='T', classifier='R2', role_config=RoleParams(), snapshot_id=''):
    obs = np.asarray(observed)
    mean = np.asarray(mean)
    var = np.asarray(variance)
    if obs.ndim != 2 or obs.shape != mean.shape or obs.shape != var.shape:
        raise ValueError('shape mismatch')
    if obs.size > 1000000 or resolution_m <= 0:
        raise ValueError('invalid resolution')
    if not np.isin(obs, [0, 1, 2]).all():
        raise ValueError('observed values 0,1,2 required')
    if not np.isfinite(mean).all() or not np.isfinite(var).all():
        raise ValueError('nonfinite map')
    if mean.min() < 0 or mean.max() > 1 or var.min() < 0 or (var.max() > 1):
        raise ValueError('mean/variance range')
    if geometry not in ('T', 'R1') or classifier not in ('T', 'R2'):
        raise ValueError('unknown experimental arm')
    sha_obs = hashlib.sha256(obs.tobytes()).hexdigest()
    geo_params = Params(virtual_gap_m=2.5, junction_reach_m=1.4) if geometry == 'T' else Params(virtual_gap_m=2.2, junction_reach_m=1.2)
    st = _extract_structural(obs, mean, var, resolution_m, geo_params)
    base, lab, faces, adj, corridor_ids = _t_roles(st, obs, resolution_m)
    graph = {}
    if classifier == 'R2':
        base, graph = _graph_roles(st, obs, resolution_m, base, role_config)
    sem = np.zeros_like(obs, dtype=np.uint8)
    sem[st['floor']] = base[st['floor']]
    sem[st['solid'] & ~st['furniture']] = WALL
    sem[st['furniture']] = OBSTACLE
    bays = nested_bay_hypotheses(st['horizontal'], st['vertical'], sem, st['floor'], obs)
    nested_added = []
    for score, (x0, y0, x1, y1), left, right, obsfrac in bays:
        sub = sem[y0 + 2:y1 - 2, x0 + 2:x1 - 2]
        replaced = int((sub == CORRIDOR).sum())
        sub[sub == CORRIDOR] = ROOM
        if replaced:
            nested_added.append({'bbox_xyxy': [int(x0), int(y0), int(x1), int(y1)], 'promoted_floor_cells': replaced, 'boundary_support': round(float((left + right) / 2), 3), 'observed_support': round(float(obsfrac), 3), 'is_verified_door': False})
    if not np.isin(sem[st['floor']], [ROOM, CORRIDOR]).all():
        raise AssertionError('unclassified floor')
    if sha_obs != hashlib.sha256(obs.tobytes()).hexdigest():
        raise AssertionError('mutated observed input')
    diag = {'version': 'V4R2_ABLATION_R1', 'snapshot_id': snapshot_id, 'geometry': geometry, 'classifier': classifier, 'status': 'OFFLINE_SHADOW_IN_SAMPLE', 'room_floor_cells': int((sem == ROOM).sum()), 'corridor_floor_cells': int((sem == CORRIDOR).sum()), 'floor_cells': int(st['floor'].sum()), 'virtual_wall_cells': int(st['virtual'].sum()), 'face_count': len(faces), 'legacy_corridor_faces': len(corridor_ids), 'graph_regions': len(graph.get('regions', {})), 'graph_edges': len(graph.get('filtered_edges', [])), 'graph_role_changes': graph.get('changed_roles', 0), 'nested_bays': len(nested_added), 'observed_sha256': sha_obs, 'mean_sha256': hashlib.sha256(mean.tobytes()).hexdigest(), 'variance_sha256': hashlib.sha256(var.tobytes()).hexdigest(), 'warnings': ['R2 graph classifier begins with V4-T current-snapshot prior', 'Scores and thresholds uncalibrated; not Luperto paper reproduction', 'No unseen-building validation; no online navigation claims']}
    if graph:
        diag['graph_nodes'] = [{'id': int(k), 'role': graph['roles'][k], 'unary': graph['unaries'][k], 'margin': round(float(graph['margins'].get(k, 0)), 4)} for k in sorted(graph['roles'])]
        diag['door_hypotheses'] = [{'u': int(e['u']), 'v': int(e['v']), 'opening_m': e['opening_m'], 'status': e['status'], 'opening_score': e['opening_score']} for e in graph['filtered_edges']]
    diag['nested_bay_hypotheses'] = nested_added
    return {'semantic': sem, 'floor': st['floor'], 'virtual_barriers': st['virtual'], 'pred_wall': st['pred_wall'], 'observed_wall': st['observed_wall'], 'report': diag}
