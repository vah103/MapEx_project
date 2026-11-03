"""L1 stand-ins (development only; the native run uses the real topo_l1).

* `topomap_from_labels`: build a TopoMap with the TOPO-001 interface from a
  labelled synthetic map (test fixture).
* `coarse_build_topo`: crude runtime stand-in with the `build_topo` signature:
  connected components of predicted free space, every region = corridor
  (backbone only, no pockets).  Used for timing on recorded inputs.
"""
import time

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def _region_dict(rid, mask, cls, ppm, obs=None, mean=None, door_ids=()):
    rr, cc = np.nonzero(mask)
    sk = skeletonize(mask)
    sr, sc = np.nonzero(sk)
    unk = 0.0
    if obs is not None and mean is not None:
        unk = float(np.count_nonzero(mask & (obs == 0.5) & (mean < 0.5))) / ppm ** 2
    p = {"corridor": 0.0, "room": 0.0, "hall": 0.0}
    p[cls] = 1.0
    return {"id": int(rid), "cls": cls, "p": p, "area_m2": len(rr) / ppm ** 2,
            "unknown_free_m2": unk, "degree": len(door_ids), "door_ids": list(door_ids),
            "centroid_rc": [float(rr.mean()), float(cc.mean())] if len(rr) else [0.0, 0.0],
            "skeleton_rc": np.stack([sr, sc], axis=1).astype(np.int64)}


def topomap_from_labels(labels, classes, doors=(), ppm=10.0, obs=None, mean=None):
    """labels: int map (-1 none); classes: {id: cls}; doors: [(id, rc, (a, b))]."""
    labels = np.asarray(labels, dtype=np.int32)
    door_list = []
    by_region = {}
    for did, rc, ab in doors:
        door_list.append({"id": int(did), "rc": [int(rc[0]), int(rc[1])], "width_m": 1.0,
                          "regions": [int(ab[0]), int(ab[1])], "p_exist": 1.0})
        for a in ab:
            by_region.setdefault(int(a), []).append(int(did))
    regions = []
    for rid, cls in sorted(classes.items()):
        regions.append(_region_dict(rid, labels == rid, cls, ppm, obs, mean,
                                    by_region.get(int(rid), [])))
    backbone = set(r["id"] for r in regions
                   if r["p"]["corridor"] >= 0.5 or r["cls"] == "hall" or r["degree"] >= 3)
    pockets = [r["id"] for r in regions if r["id"] not in backbone]
    bmask = np.isin(labels, list(backbone)) if backbone else np.zeros(labels.shape, bool)
    return {"region_id": labels, "regions": regions, "doors": door_list,
            "backbone_mask": bmask, "pocket_ids": pockets, "state": None,
            "timing_s": 0.0}


def coarse_build_topo(obs, mean, var, robot_rc, ppm, prev=None):
    t0 = time.perf_counter()
    free = (obs == 0) | ((obs == 0.5) & (mean < 0.5))
    lab, n = ndimage.label(free)
    sizes = np.bincount(lab.ravel())
    labels = np.full(obs.shape, -1, dtype=np.int32)
    classes = {}
    k = 0
    for i in range(1, n + 1):
        if sizes[i] / ppm ** 2 < 1.5:
            continue
        labels[lab == i] = k
        classes[k] = "corridor"
        k += 1
    topo = topomap_from_labels(labels, classes, (), ppm, obs, mean)
    topo["timing_s"] = time.perf_counter() - t0
    return topo
