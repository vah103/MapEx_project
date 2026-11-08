#!/usr/bin/env python3
"""Build bright decision galleries for 10 native MapEx/Tour/Tour R2 runs.

Sources: completed Dell native runs only. No closed-loop execution, no changes
to algorithm output. Blue circles mark native frontier candidates logged for
each decision (candidates_rc).
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/home/dell/DELL_NATIVE2D')
INNER = Path('native_plain_lama3small_run05/MAPEX_PLAIN_LAMA3SMALL_RUN05_visvarprob')
POSES = {
    'kth49_p00': (
        'runs/KTH49V3-P00-MX-T1000-01',
        'sandbox_phase5_jf_r4_kth49/runs/P5JF-KTH49V3-P00-FEAS-01',
        'sandbox_topo_mean_kth49_p00_eval_20261108',
    ),
    'kth49_p01': (
        'runs/KTH49V3-P01-MX-T1000-01',
        'sandbox_phase5_jf_r4_kth49/runs/P5JF-KTH49V3-P01-FEAS-01',
        'sandbox_topo_mean_kth49_p01_eval_20261108',
    ),
    'kth49_p03': (
        'runs/KTH49V3-P03-MX-T1000-01',
        'sandbox_phase5_jf_r3/runs/P5JF-KTH49V3-P03-FEAS-01',
        'sandbox_topo_mean_kth49_p03_eval_20261108',
    ),
    'kth50_p00': (
        'runs/KTH50V3-P00-MX-T1000-01',
        'sandbox_phase5_jf_r4_kth50/runs/P5JF-KTH50V3-P00-FEAS-01',
        'sandbox_topo_mean_kth50_p00_eval_20261108',
    ),
    'kth50_p01': (
        'runs/KTH50V3-P01-MX-T1000-01',
        'sandbox_phase5_jf_r4_kth50/runs/P5JF-KTH50V3-P01-FEAS-01',
        'sandbox_topo_mean_kth50_p01_resume_eval_20261108',
    ),
    'kth50_p03': (
        'runs/KTH50V3-P03-MX-T1000-01',
        'sandbox_phase5_jf_r3/runs/P5JF-KTH50V3-P03-FEAS-01',
        'sandbox_topo_mean_kth50_p03_resume_eval_20261108',
    ),
    'kth51_p00': (
        'runs/FURN-KTH51V3-P00-MX-NOOBJ-01',
        'sandbox_phase5_jf_r4_kth51/runs/P5JF-KTH51V3-P00-FEAS-01',
        'sandbox_topo_mean_kth51_p00_eval_20261108',
    ),
    'kth51_p01': (
        'runs/KTH51V3-V0R2-P01-MX-01',
        'sandbox_phase5_jf_r4_kth51/runs/P5JF-KTH51V3-P01-FEAS-01',
        'sandbox_topo_mean_kth51_p01_resume_eval_20261108',
    ),
    'kth51_p02': (
        'runs/KTH51V3-V0R2-P02-MX-01',
        'sandbox_phase5_jf_r4_kth51/runs/P5JF-KTH51V3-P02-FEAS-01',
        'sandbox_topo_mean_kth51_p02_resume_eval_20261108',
    ),
    'kth51_p03': (
        'runs/KTH51V3-V0R2-P03-MX-01',
        'sandbox_phase5_jf_r4_kth51/runs/P5JF-KTH51V3-P03-FEAS-01',
        'sandbox_topo_mean_kth51_p03_resume_eval_20261108',
    ),
}
METHODS = (
    ('mapex', 'MapEx thường'),
    ('tour', 'MapEx Tour'),
    ('tour_r2', 'MapEx Tour R2'),
)
PALETTE = {
    0: (255, 255, 255),
    128: (226, 237, 249),
    255: (35, 54, 74),
}


def xy(rc, crop):
    return round(float(rc[1])) - crop[0], round(float(rc[0])) - crop[1]


def line(draw, rcs, crop, color, width):
    if len(rcs) >= 2:
        draw.line([xy(rc, crop) for rc in rcs], fill=color, width=width, joint='curve')


def marker(draw, rc, crop, color, radius):
    x, y = xy(rc, crop)
    draw.ellipse((x-radius-2, y-radius-2, x+radius+2, y+radius+2), fill='#ffffff')
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)


def paint(frame_path, crop, trajectory, tick, robot, goal, planned, frontiers):
    with Image.open(frame_path) as source:
        raw = np.asarray(source.convert('L').crop(crop), dtype=np.uint8)
    rgb = np.empty((*raw.shape, 3), dtype=np.uint8)
    rgb[:] = (243, 246, 250)
    for shade, color in PALETTE.items():
        rgb[raw == shade] = color
    image = Image.fromarray(rgb, 'RGB')
    draw = ImageDraw.Draw(image)
    # Frontier candidate centers from the logged native decision, not reconstructed.
    for rc in frontiers:
        marker(draw, rc, crop, '#168acf', 3)
    past = trajectory[:tick+1:3].tolist()
    past.append(trajectory[tick].tolist())
    line(draw, past, crop, '#5ab6ad', 3)
    if planned:
        line(draw, planned, crop, '#9e88dc', 3)
    if goal is not None:
        marker(draw, goal, crop, '#e76b8b', 7)
    marker(draw, robot, crop, '#f2ae35', 8)
    return image


def collect(relative, key, label, folder):
    run = ROOT / relative / INNER
    decisions_dir = run / 'plain_mapex_evidence/decisions'
    steps_dir = run / 'plain_mapex_evidence/steps'
    obs_dir = run / 'global_obs'
    source_files = sorted(decisions_dir.glob('decision_*.json'))
    if not source_files:
        raise RuntimeError(f'Missing decisions: {run}')
    if not (obs_dir / '00000999.png').is_file():
        raise RuntimeError(f'Not terminal at 999: {run}')
    trajectory = np.load(run / 'odom.npy', allow_pickle=False)
    with Image.open(obs_dir / '00000999.png') as last:
        crop = (480, 480, last.width-480, last.height-480)
    if not (crop[2] > crop[0] and crop[3] > crop[1]):
        raise AssertionError(f'Invalid image extent: {run}, {crop}')
    assets = folder / 'assets' / key
    assets.mkdir(parents=True)
    frames = []
    seen = set()
    trace = hashlib.sha256()
    for source_file in source_files:
        raw = source_file.read_bytes()
        decision = json.loads(raw)
        tick = int(decision['native_tick'])
        if tick >= 1000:
            continue
        if tick in seen:
            raise AssertionError(f'Duplicate decision tick {tick}: {run}')
        seen.add(tick)
        trace.update(raw)
        frontiers = decision['candidates_rc']
        if len(frontiers) != int(decision['frontier_candidate_count']):
            raise AssertionError(f'Non-frontier candidate mixed in: {run} tick {tick}')
        if tick >= len(trajectory):
            raise AssertionError(f'Tick beyond trajectory: {run} tick {tick}')
        robot = decision['robot_rc']
        if np.linalg.norm(trajectory[tick] - np.asarray(robot)) >= 0.01:
            raise AssertionError(f'Odom mismatch: {run} tick {tick}')
        step = json.loads((steps_dir / f'tick_{tick:06d}_plan.json').read_text())
        goal = step['actual_selected_goal_rc']
        obs = obs_dir / f'{tick:08d}.png'
        if not obs.is_file():
            raise FileNotFoundError(obs)
        frame = paint(obs, crop, trajectory, tick, robot, goal,
                      step.get('astar_path_rc') or [], frontiers)
        name = f'D{len(frames)+1:02d}_t{tick:04d}.webp'
        frame.save(assets / name, format='WEBP', quality=84, method=6)
        frames.append({
            'n': len(frames)+1, 'tick': tick,
            'frontier_count': len(frontiers),
            'image': f'assets/{key}/{name}',
        })
    if not frames or frames[0]['tick'] != 0:
        raise AssertionError(f'Missing first decision: {run}')
    return {
        'key': key, 'label': label, 'count': len(frames),
        'source_run': relative, 'decision_sha256': trace.hexdigest(),
        'frames': frames,
    }


def make_page(template, name, manifest, output):
    original = template.read_text(encoding='utf-8')
    safe = json.dumps(manifest, ensure_ascii=False).replace('</', '<\\/')
    original, n1 = re.subn(
        r'<title>[^<]*</title>',
        f'<title>{name} · MapEx / Tour / Tour R2</title>',
        original, count=1,
    )
    original, n2 = re.subn(
        r'<header><h1>.*?</h1><p>.*?</p></header>',
        f'<header><h1>{name} · ảnh từng decision</h1>'
        '<p>MapEx thường · MapEx Tour · MapEx Tour R2</p></header>',
        original, count=1, flags=re.DOTALL,
    )
    original, n3 = re.subn(
        r'(<script id="dataset" type="application/json">).*?(</script>)',
        lambda m: m.group(1) + safe + m.group(2),
        original, count=1, flags=re.DOTALL,
    )
    original, n4 = re.subn(
        r'(<div class="key">)',
        r'\1<span><i class="dot" style="background:#168acf"></i>Frontier</span>',
        original, count=1,
    )
    if (n1, n2, n3, n4) != (1, 1, 1, 1):
        raise AssertionError(f'HTML template mismatch: {(n1,n2,n3,n4)}')
    (output / 'index.html').write_text(original, encoding='utf-8')


def make_landing(output, counts):
    cards = []
    for pose in POSES:
        label = pose.upper().replace('_', ' ')
        values = counts[pose]
        cards.append(
            f'<a href="{pose}/index.html"><strong>{label}</strong>'
            f'<small>{values[0]} / {values[1]} / {values[2]} decision</small></a>'
        )
    body = '\n'.join(cards)
    html = f'''<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MapEx · 10 run · 3 phương pháp</title>
<style>:root{{color-scheme:light}}*{{box-sizing:border-box}}
body{{margin:0;background:#eef4f9;color:#22374b;font-family:system-ui,-apple-system,sans-serif}}
main{{max-width:900px;margin:auto;padding:22px 16px 65px}}
h1{{font-size:clamp(24px,3vw,34px)}}p{{color:#658099}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:13px;margin-top:23px}}
a{{display:flex;flex-direction:column;gap:9px;background:white;color:#225c86;text-decoration:none;
padding:20px 18px;border:1px solid #d6e1ec;border-radius:12px;box-shadow:0 3px 12px #2a507014}}
a:hover{{border-color:#2789c6}}small{{color:#687f94}}</style></head><body>
<main><h1>MapEx · 10 run</h1><p>MapEx thường · MapEx Tour · MapEx Tour R2</p>
<p>Ảnh từng decision, bản đồ sáng. Chấm xanh dương là frontier ứng viên từ log native.</p>
<div class="grid">{body}</div></main></body></html>'''
    (output / 'index.html').write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('html_template', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    counts = {}
    for pose, paths in POSES.items():
        folder = args.output / pose
        folder.mkdir()
        methods = [
            collect(relative, key, label, folder)
            for (key, label), relative in zip(METHODS, paths)
        ]
        manifest = {
            'pose': pose, 'source': 'completed Dell native runs',
            'frontier_overlay': 'logged native candidates_rc',
            'methods': methods,
        }
        (folder / 'manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8'
        )
        make_page(args.html_template, pose.upper().replace('_', ' '), manifest, folder)
        counts[pose] = [m['count'] for m in methods]
        print(pose, counts[pose], flush=True)
    make_landing(args.output, counts)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
