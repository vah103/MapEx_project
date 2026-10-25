import numpy as np
import cv2
from scipy import ndimage as ndi
CORRIDOR = 2

def find_objects(solid):
    lbl, num = ndi.label(solid, structure=np.ones((3, 3), bool))
    furniture = np.zeros_like(solid)
    records = []
    for i, s in enumerate(ndi.find_objects(lbl), 1):
        if s is None:
            continue
        sy, sx = s
        h = sy.stop - sy.start
        w = sx.stop - sx.start
        sub = lbl[s] == i
        area = int(sub.sum())
        holes = int((ndi.binary_fill_holes(sub) & ~sub).sum())
        boundary = sy.start <= 2 or sx.start <= 2 or sy.stop >= solid.shape[0] - 2 or (sx.stop >= solid.shape[1] - 2)
        compact = max(h, w) / max(1, min(h, w)) < 5.0
        slender_island = min(h, w) <= 3 and max(h, w) < 32
        probable_object = not boundary and (compact and area < 230 and (holes >= 3 or (h <= 7 and w <= 8)) or slender_island)
        if probable_object:
            furniture[lbl == i] = True
            records.append({'bbox_rc': [sy.start, sx.start, sy.stop, sx.stop], 'area_cells': area, 'interior_cells': holes})
    return (furniture, records)

def extrapolate_to_junction(vline, hline, limit_cells):
    hh, ww = vline.shape
    ext = np.zeros_like(vline)
    n, lab, stat, _ = cv2.connectedComponentsWithStats(vline.astype('uint8'), 8)
    for i in range(1, n):
        x, y, w, h, area = map(int, stat[i])
        if h < 13 or h < w * 2.5:
            continue
        midx = x + w // 2
        for endpoint, direction in [(y, -1), (y + h - 1, +1)]:
            for dist in range(2, limit_cells + 1):
                ry = endpoint + direction * dist
                if ry < 1 or ry >= hh - 1:
                    break
                local = hline[ry, max(0, midx - 2):min(ww, midx + 3)]
                if not np.any(local):
                    continue
                lo = max(0, midx - 10)
                hi = min(ww, midx + 11)
                if hline[ry, lo:hi].sum() < 8:
                    continue
                ext[min(ry, endpoint):max(ry, endpoint) + 1, max(0, midx - 1):min(ww, midx + 2)] = True
                break
    n, lab, stat, _ = cv2.connectedComponentsWithStats(hline.astype('uint8'), 8)
    for i in range(1, n):
        x, y, w, h, area = map(int, stat[i])
        if w < 13 or w < h * 2.5:
            continue
        midy = y + h // 2
        for endpoint, direction in [(x, -1), (x + w - 1, 1)]:
            for dist in range(2, limit_cells + 1):
                rx = endpoint + direction * dist
                if rx < 1 or rx >= ww - 1:
                    break
                if not np.any(vline[max(0, midy - 2):min(hh, midy + 3), rx]):
                    continue
                lo = max(0, midy - 10)
                hi = min(hh, midy + 11)
                if vline[lo:hi, rx].sum() < 8:
                    continue
                ext[max(0, midy - 1):min(hh, midy + 2), min(rx, endpoint):max(rx, endpoint) + 1] = True
                break
    return ext

def nested_bay_hypotheses(hclose, vclose, sem, floor, observed):
    H, W = sem.shape
    segments = []
    for y in range(H):
        a = hclose[y].astype(np.int8)
        diff = np.diff(np.pad(a, (1, 1)))
        for x0, x1 in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
            if x1 - x0 < 19:
                continue
            if segments and y - segments[-1][0] <= 2 and (abs(x0 - segments[-1][1]) <= 3) and (abs(x1 - segments[-1][2]) <= 3):
                continue
            segments.append((int(y), int(x0), int(x1)))
    proposals = []
    for i, (y0, a, b) in enumerate(segments):
        for y1, c, d in segments[i + 1:]:
            height = y1 - y0
            if height < 12 or height > 65:
                continue
            x0 = max(a, c)
            x1 = min(b, d)
            width = x1 - x0
            if width < 18 or width / height > 0.0 + 4.5 or width / height < 0.5 or (width * height > 6000):
                continue
            if min(x0, y0, W - x1, H - y1) < 4:
                continue
            yy = slice(y0 + 2, y1 - 2)
            xx = slice(x0 + 2, x1 - 2)
            if yy.stop - yy.start < 8 or xx.stop - xx.start < 14:
                continue
            crop = sem[yy, xx]
            frac = float((crop == CORRIDOR).sum() / max(1, crop.size))
            if frac < 0.72:
                continue
            left = float(np.max(vclose[y0:y1, max(0, x0 - 3):min(W, x0 + 4)], axis=1).mean())
            right = float(np.max(vclose[y0:y1, max(0, x1 - 3):min(W, x1 + 4)], axis=1).mean())
            if min(left, right) < 0.24 or left + right < 1.12:
                continue
            area = width * height
            obs_frac = float((observed[yy, xx] == 0).sum() / max(1, crop.size))
            score = left + right + 0.08 * frac + 0.04 * obs_frac
            proposals.append((score, (x0, y0, x1, y1), left, right, obs_frac))
    proposals.sort(key=lambda t: -t[0])
    chosen = []
    for score, box, left, right, obsfrac in proposals:
        x0, y0, x1, y1 = box
        if any((max(0, min(x1, B[2]) - max(x0, B[0])) * max(0, min(y1, B[3]) - max(y0, B[1])) > 0.35 * min((x1 - x0) * (y1 - y0), (B[2] - B[0]) * (B[3] - B[1])) for _, B, *_ in chosen)):
            continue
        chosen.append((score, box, left, right, obsfrac))
    return chosen
