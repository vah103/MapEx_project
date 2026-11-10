#!/usr/bin/env python3
"""Visual replay of the locked DEV51 RoomNode S0/S1a comparison.

Read-only on the 51 archived DEV inputs and frozen RoomNode source. Produces
cropped RGB region maps, predicted door marks, GT-match diagnostic marks, and
a mobile-friendly scroll gallery. No native exploration or KTH work.
"""

import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


COLORS = [
    (249, 191, 181), (167, 211, 243), (181, 226, 187), (248, 220, 154),
    (210, 194, 241), (245, 188, 222), (157, 221, 214), (245, 208, 179),
    (183, 204, 244), (204, 222, 180), (231, 203, 202), (174, 227, 227),
]
WALL = (41, 57, 75)
UNKNOWN = (226, 234, 242)
FREE = (251, 253, 255)


def make_roi(obs, map0, map1, robot, pad=18):
    active = (obs != 0.5) | (map0.region_id > 0) | (map1.region_id > 0)
    rows, cols = np.nonzero(active)
    if not len(rows):
        rows, cols = np.array([robot[0]]), np.array([robot[1]])
    h, w = obs.shape
    r0 = max(0, min(int(rows.min()), int(robot[0])) - pad)
    r1 = min(h, max(int(rows.max()), int(robot[0])) + pad + 1)
    c0 = max(0, min(int(cols.min()), int(robot[1])) - pad)
    c1 = min(w, max(int(cols.max()), int(robot[1])) + pad + 1)
    return (r0, r1, c0, c1)


def gt_matches(doors, gt_doors):
    if not doors or not gt_doors:
        return [False] * len(doors)
    points = np.concatenate([np.asarray(g['cells'], dtype=np.int32) for g in gt_doors])
    query = np.asarray([d['rc'] for d in doors], dtype=float)
    return (cKDTree(points).query(query, k=1)[0] <= 10.0).tolist()


def render(obs, mean, topo, robot, roi, matches, path):
    r0, r1, c0, c1 = roi
    o = obs[r0:r1, c0:c1]
    m = mean[r0:r1, c0:c1]
    rid = topo.region_id[r0:r1, c0:c1]
    rgb = np.empty((*rid.shape, 3), dtype=np.uint8)
    rgb[:] = UNKNOWN
    rgb[o == 0] = FREE
    rgb[o == 1] = WALL
    rgb[(o == 0.5) & (m < 0.5)] = (207, 230, 243)

    # Region palette is categorical; observed and predicted portions retain
    # different opacity so LaMa-only geometry remains visibly distinguishable.
    for k in np.unique(rid):
        if k <= 0:
            continue
        color = np.asarray(COLORS[(int(k) * 7 + 3) % len(COLORS)], dtype=float)
        mask = rid == k
        obs_mask = mask & (o == 0)
        pred_mask = mask & (o == 0.5)
        rgb[obs_mask] = np.rint(rgb[obs_mask] * 0.42 + color * 0.58).astype(np.uint8)
        rgb[pred_mask] = np.rint(rgb[pred_mask] * 0.24 + color * 0.76).astype(np.uint8)

    # Dark outlines expose individual region boundaries, including non-observed
    # predicted regions; 4-neighbor comparison avoids diagonal speckling.
    boundaries = np.zeros(rid.shape, bool)
    adjacent = (rid[1:, :] != rid[:-1, :]) & (rid[1:, :] > 0) & (rid[:-1, :] > 0)
    boundaries[1:, :] |= adjacent
    boundaries[:-1, :] |= adjacent
    adjacent = (rid[:, 1:] != rid[:, :-1]) & (rid[:, 1:] > 0) & (rid[:, :-1] > 0)
    boundaries[:, 1:] |= adjacent
    boundaries[:, :-1] |= adjacent
    rgb[boundaries] = (67, 79, 94)

    image = Image.fromarray(rgb, 'RGB')
    draw = ImageDraw.Draw(image)
    # Light skeleton accent, where the algorithm classifies a backbone.
    backbone = topo.backbone_mask[r0:r1, c0:c1]
    coords = np.argwhere(backbone)
    for rr, cc in coords[::2]:
        draw.point((int(cc), int(rr)), fill=(255, 255, 255))

    for door, matched in zip(topo.doors, matches):
        rr, cc = map(int, door['rc'])
        x, y = cc - c0, rr - r0
        if x < -8 or y < -8 or x >= c1-c0+8 or y >= r1-r0+8:
            continue
        center_obs = obs[rr, cc] == 0
        fill = (245, 158, 59) if center_obs else (148, 87, 199)
        draw.ellipse((x-5, y-5, x+5, y+5), fill=(255, 255, 255))
        draw.ellipse((x-4, y-4, x+4, y+4), fill=fill, outline=(69, 57, 82), width=1)
        if not matched:
            # Mark *no annotation match within 1 m*, not proven fake doors.
            draw.line((x-7, y-7, x+7, y+7), fill=(212, 57, 75), width=2)
            draw.line((x-7, y+7, x+7, y-7), fill=(212, 57, 75), width=2)
    rx, ry = robot[1] - c0, robot[0] - r0
    draw.ellipse((rx-8, ry-8, rx+8, ry+8), fill=(255, 255, 255))
    draw.ellipse((rx-6, ry-6, rx+6, ry+6), fill=(245, 175, 46), outline=(125, 81, 27), width=1)
    image.save(path, format='WEBP', quality=90, method=6)


def write_gallery(root, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['map_id']].append(row)
    map_list = list(groups)
    nav = ''.join(f'<button data-map="{m}">{html.escape(m)}</button>' for m in map_list)
    sections = []
    for mid, entries in groups.items():
        cards = []
        for row in entries:
            s0 = row['methods']['S0']
            s1 = row['methods']['S1a']
            caption = f'D{row["decision"]:02d} · timestep {row["tick"]}'
            cards.append(f'''<article id="{mid}-D{row['decision']}" class="decision">
<h3>{html.escape(caption)}</h3>
<div class="pair">
<figure><figcaption>RoomNode L1 cũ (S0) · {s0['regions']} vùng · {s0['doors']} cửa</figcaption>
<a href="{s0['image']}" target="_blank"><img loading="lazy" src="{s0['image']}" alt="RoomNode L1, {html.escape(caption)}"></a></figure>
<figure><figcaption>RoomNode R1 mới (S1a) · {s1['regions']} vùng · {s1['doors']} cửa</figcaption>
<a href="{s1['image']}" target="_blank"><img loading="lazy" src="{s1['image']}" alt="RoomNode R1, {html.escape(caption)}"></a></figure>
</div></article>''')
        chips = ''.join(f'<a href="#{mid}-D{x["decision"]}">D{x["decision"]:02d}</a>' for x in entries)
        sections.append(f'<section class="map-group" id="group-{mid}" data-map="{mid}">'
                        f'<h2>{html.escape(mid)} · {len(entries)} decision</h2>'
                        f'<div class="chips">{chips}</div>{"".join(cards)}</section>')

    page = f'''<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RoomNode DEV51 · S0 vs S1a</title>
<style>:root{{color-scheme:light;--ink:#243749;--line:#d3e0ed;--bg:#edf4fa}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth;scroll-padding-top:145px}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px system-ui,-apple-system,sans-serif}}
header{{max-width:1400px;margin:auto;padding:22px 14px 12px}}
h1{{margin:0;font-size:clamp(22px,4vw,31px)}}header p{{margin:8px 0;color:#526c86;font-size:13px}}
nav{{position:sticky;top:0;z-index:5;background:#ffffffed;padding:10px max(10px,calc((100% - 1375px)/2));
border-block:1px solid var(--line);display:flex;gap:7px;overflow:auto}}
nav button,.chips a{{flex:none;border:1px solid #cddded;border-radius:7px;background:#f6faff;color:#245f87;
font:inherit;font-size:12px;padding:8px 11px;cursor:pointer;text-decoration:none}}
nav button.active{{background:#246e9d;color:white;border-color:#246e9d;font-weight:bold}}
main{{max-width:1400px;margin:auto;padding:9px 12px 60px}}
.legend{{display:flex;flex-wrap:wrap;gap:9px 20px;padding:8px 0 13px;color:#536a80;font-size:12px}}
.dot{{display:inline-block;width:11px;height:11px;border-radius:50%;vertical-align:middle;margin-right:5px}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0 20px}}
.chips a{{padding:5px 8px}}h2{{margin:15px 0 0;font-size:21px}}
.decision{{background:white;border:1px solid var(--line);border-radius:12px;margin-bottom:18px;
padding:12px;scroll-margin-top:115px}}
h3{{margin:2px 0 12px;color:#3b617e;font-size:15px}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
figure{{min-width:0;margin:0;border:1px solid #e1eaf3;border-radius:9px;overflow:hidden;background:#f4f8fc}}
figcaption{{padding:9px 10px;background:#f6faff;font-size:12px;font-weight:700}}
img{{display:block;width:100%;height:auto;object-fit:contain}}a img{{cursor:zoom-in}}
@media(max-width:700px){{.pair{{grid-template-columns:1fr}}.decision{{padding:7px}}nav{{padding:8px}}
header{{padding:15px 12px 8px}}}}
</style></head><body>
<header><h1>RoomNode · L1 cũ so với R1 mới</h1>
<p>51 snapshot DEV · cùng đầu vào LaMa, không chạy closed-loop · chạm ảnh để phóng to</p></header>
<nav id="map-nav">{nav}</nav><main>
<div class="legend">
<span><i class="dot" style="background:#f5af2e"></i>Robot</span>
<span><i class="dot" style="background:#f59e3b"></i>Cửa tâm quan sát</span>
<span><i class="dot" style="background:#9457c7"></i>Cửa tâm UNKNOWN</span>
<span><b style="color:#d4394b">✕</b> Cửa không khớp chú thích GT (≤ 1 m)</span>
<span>Vùng được tô màu khác nhau; màu đậm hơn là vùng dự đoán</span>
</div>
{''.join(sections)}</main>
<script>const buttons=[...document.querySelectorAll('nav button')];
const groups=[...document.querySelectorAll('.map-group')];
function activate(key){{buttons.forEach(b=>b.classList.toggle('active',b.dataset.map===key));
groups.forEach(g=>g.hidden=g.dataset.map!==key);window.scrollTo({{top:0,behavior:'instant'}});}}
buttons.forEach(b=>b.onclick=()=>activate(b.dataset.map));activate(buttons[0].dataset.map);
</script></body></html>'''
    (root / 'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Smoke first N DEV frames')
    a = parser.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    (a.out / 'assets').mkdir()

    sys.path.insert(0, str(a.code_root))
    from p1_score import load_dev
    from prototype_r1.run_offline import load_frozen
    from prototype_r1.variant import build_variant
    refs = {}
    for item in a.reference.read_text(encoding='utf-8').splitlines():
        x = json.loads(item)
        key = (str(x['map_id']), int(x['decision']), x['variant'])
        if key in refs:
            raise AssertionError(f'Duplicate reference {key}')
        refs[key] = x
    if len(refs) != 102:
        raise AssertionError('DEV51 expected 102 reference metrics')

    l1 = load_frozen()
    records = []
    gt_cache = {}
    counts = defaultdict(lambda: [0, 0])
    frames = list(load_dev(a.source_root))
    if len(frames) != 51:
        raise AssertionError(f'DEV51 expected; got {len(frames)}')
    if a.limit:
        frames = frames[:a.limit]
    for j, f in enumerate(frames, 1):
        mid, dec = f['map_id'], int(f['decision'])
        if mid not in gt_cache:
            gt_cache[mid] = f['gt_loader']()[2]
        maps = {}
        for variant in ('S0', 'S1a'):
            maps[variant] = build_variant(
                l1, f['obs'], f['mean'], f['var'], f['robot'], 10.,
                prev=None, variant=variant, with_evidence=False,
            )
            ref = refs[(mid, dec, variant)]
            if (f['input_sha256'] != ref['input_sha256'] or
                    len(maps[variant].doors) != ref['n_output_doors'] or
                    len(maps[variant].regions) != ref['n_output_regions']):
                raise AssertionError(f'Scorer parity mismatch {mid} D{dec} {variant}')
        roi = make_roi(f['obs'], maps['S0'], maps['S1a'], f['robot'])
        methods = {}
        for variant in ('S0', 'S1a'):
            topo = maps[variant]
            matches = gt_matches(topo.doors, gt_cache[mid])
            image_name = f'assets/{mid}_D{dec:02d}_{variant}.webp'
            render(f['obs'], f['mean'], topo, f['robot'], roi, matches, a.out / image_name)
            methods[variant] = {
                'image': image_name, 'regions': len(topo.regions),
                'doors': len(topo.doors), 'door_unmatched_gt': matches.count(False),
                'doors_in_unknown': int(sum(f['obs'][tuple(d['rc'])] == 0.5 for d in topo.doors)),
            }
            counts[variant][0] += len(topo.doors)
            counts[variant][1] += matches.count(False)
        records.append({
            'map_id': mid, 'decision': dec, 'tick': int(f['tick']),
            'input_sha256': f['input_sha256'], 'roi_rc': list(map(int, roi)),
            'methods': methods,
        })
        print(f'{j}/{len(frames)} {mid} D{dec}: '
              f'{methods["S0"]["regions"]}/{methods["S1a"]["regions"]} regions, '
              f'{methods["S0"]["doors"]}/{methods["S1a"]["doors"]} doors', flush=True)
    manifest = {
        'source': 'frozen DEV51 snapshot replay; GT used for offline door-match marks only',
        'n_frames': len(records), 'n_images': len(records)*2,
        'variants': ['S0', 'S1a'], 'aggregate_doors_and_unmatched': counts,
        'frames': records,
    }
    (a.out / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8'
    )
    write_gallery(a.out, records)
    print('DONE', len(records), len(records)*2, flush=True)


if __name__ == '__main__':
    main()
