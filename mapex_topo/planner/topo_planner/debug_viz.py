"""Per-decision debug PNG: observed map + L1 regions coloured by class + tour + goal."""
import cv2
import numpy as np

CLS_BGR = {"corridor": (230, 140, 40), "hall": (200, 60, 200), "room": (40, 160, 240)}


def render(path, obs, topo, robot_rc, plan, goal_rc=None, title="", scale=1):
    h, w = obs.shape
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    img[obs == 0] = (245, 245, 245)
    img[obs == 1] = (25, 25, 25)
    if topo is not None:
        rid = np.asarray(topo["region_id"] if isinstance(topo, dict) else topo.region_id)
        regions = topo["regions"] if isinstance(topo, dict) else topo.regions
        pockets = set(topo["pocket_ids"] if isinstance(topo, dict) else topo.pocket_ids)
        overlay = img.copy()
        for reg in regions:
            m = rid == int(reg["id"])
            overlay[m] = CLS_BGR.get(reg.get("cls"), (120, 200, 120))
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        bb = np.asarray(topo["backbone_mask"] if isinstance(topo, dict) else topo.backbone_mask)
        edge = bb ^ cv2.erode(bb.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        img[edge] = (200, 90, 0)
        for reg in regions:
            c = reg.get("centroid_rc")
            if c is None:
                continue
            tag = ("P" if int(reg["id"]) in pockets else "B") + str(reg["id"])
            cv2.putText(img, tag, (int(c[1]), int(c[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (60, 60, 60), 1, cv2.LINE_AA)
        doors = topo["doors"] if isinstance(topo, dict) else topo.doors
        for d in doors or []:
            cv2.circle(img, (int(d["rc"][1]), int(d["rc"][0])), 3, (0, 200, 255), -1)
    viz = plan.get("viz", {}) if plan else {}
    for q in viz.get("cands", []):
        cv2.circle(img, (int(q[1]), int(q[0])), 1, (150, 150, 0), -1)
    pts = [tuple(int(v) for v in robot_rc)] + [tuple(int(v) for v in q) for q in viz.get("tour", [])]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, (a[1], a[0]), (b[1], b[0]), (0, 140, 0), 1, cv2.LINE_AA)
    for i, q in enumerate(pts[1:], 1):
        cv2.circle(img, (q[1], q[0]), 3, (0, 110, 0), -1)
        cv2.putText(img, str(i), (q[1] + 4, q[0] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 90, 0), 1, cv2.LINE_AA)
    r = pts[0]
    cv2.circle(img, (r[1], r[0]), 5, (0, 0, 255), -1)
    if goal_rc is not None:
        cv2.drawMarker(img, (int(goal_rc[1]), int(goal_rc[0])), (0, 220, 0),
                       cv2.MARKER_STAR, 14, 2)
    if scale != 1:
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    bar = np.full((22, img.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(bar, title[:160], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                cv2.LINE_AA)
    out = np.concatenate([bar, img], axis=0)
    if not cv2.imwrite(str(path), out):
        raise IOError("cannot write " + str(path))
    return path
