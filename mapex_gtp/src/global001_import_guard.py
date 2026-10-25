"""GLOBAL-001 P1 R3 runtime dependency guard, called by dell_worker.py (ChatGPT review of PR #2, issues 2-3).
Fail-closed, before any exploration tick:
  check_checkpoints(root): SHA256 of the 3 LaMa best.ckpt + config.yaml, the map and MapEx configs/base.yaml ==
     RUN_MANIFEST checkpoint_pins (before model load).
  check_imports(root):     after models are loaded and every staged module is imported:
     - each staged module (gt_planner, gt_logger, tourlib, fast_native_lama_3model, raycast_paper_patch,
       plain_mapex_logger, global001_import_guard) resolves to ROOT (no shadowing);
     - every loaded module file under ROOT, /home/dell/MapEx or DELL_NATIVE2D/vendor has exactly the pinned SHA256
       (RUN_MANIFEST import_pins), and no pinned file is missing;
     - no loaded module file lives outside ROOT, those two roots, or this interpreter's prefix (conda env + stdlib);
     - torch/numpy/scipy/cv2/python versions == RUN_MANIFEST runtime_versions.
  Writes IMPORT_GUARD_<mode>_<time>.json next to the manifest. GLOBAL001_IMPORT_PROBE=1 only records origins (used
  once by the protocol to create the pins; never for a run).
Python 3.6."""
import os, sys, json, hashlib, time

NATIVE_ROOTS = ('/home/dell/MapEx', '/home/dell/DELL_NATIVE2D/vendor')
IMPORT_CWD = '/home/dell/MapEx/scripts'   # the worker chdirs here before importing MapEx (relative sys.path entries, e.g. '../lama')
STAGED_MODULES = ('gt_planner', 'gt_feedback', 'gt_visibility', 'gt_logger', 'tourlib', 'fast_native_lama_3model', 'raycast_paper_patch',
                  'plain_mapex_logger', 'global001_import_guard')


class DependencyDrift(RuntimeError):
    pass


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _under(p, root):
    return p == root or p.startswith(root.rstrip('/') + '/')


def _src(m):
    f = getattr(m, '__file__', None)
    if not f:
        return None
    if not os.path.isabs(f):                                 # e.g. saicinpainting imported via sys.path '../lama'
        f = os.path.join(IMPORT_CWD, f)
    f = os.path.realpath(f)
    if f.endswith('.pyc') and '__pycache__' in f:          # report the source the .pyc was compiled from
        d = os.path.dirname(os.path.dirname(f)); base = os.path.basename(f).split('.')[0]
        cand = os.path.join(d, base + '.py')
        f = cand if os.path.exists(cand) else f
    return f


def _manifest(root):
    with open(os.path.join(root, 'RUN_MANIFEST.json')) as f:
        return json.load(f)


def _write(root, mode, obj):
    p = os.path.join(root, 'IMPORT_GUARD_%s_%s.json' % (mode, time.strftime('%Y%m%d_%H%M%S')))
    with open(p, 'w') as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    return p


def runtime_versions():
    import numpy, scipy, cv2, torch
    return dict(python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__,
                cv2=cv2.__version__, torch=torch.__version__, executable=os.path.realpath(sys.executable))


def origins(root):
    root = os.path.realpath(root)
    allowed = [root] + list(NATIVE_ROOTS)
    prefixes = sorted({os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)})
    pinned, outside, staged, unresolved = {}, [], {}, []
    for name, m in sorted(list(sys.modules.items())):
        f = _src(m)
        if not f:
            continue
        if not os.path.isfile(f):
            # zipped eggs inside the interpreter prefix (e.g. site-packages/cycler-*.egg/cycler.py) are part of the env
            if not any(_under(f, p) for p in prefixes):
                unresolved.append(dict(module=name, file=f))
            continue
        if any(_under(f, r) for r in allowed):
            pinned[f] = _sha(f)
        elif not any(_under(f, p) for p in prefixes):
            outside.append(dict(module=name, file=f))
    for n in STAGED_MODULES:
        m = sys.modules.get(n)
        staged[n] = _src(m) if m is not None else None
    return dict(root=root, files=pinned, outside_allowed_roots=outside, unresolved=unresolved, staged_origins=staged, prefixes=prefixes)


def check_checkpoints(root):
    if os.environ.get('GLOBAL001_IMPORT_PROBE') == '1':
        return None
    pins = _manifest(root)['checkpoint_pins']
    bad = {p: (s, _sha(p) if os.path.isfile(p) else None) for p, s in pins.items()}
    bad = {p: v for p, v in bad.items() if v[0] != v[1]}
    if len(pins) < 8 or bad:            # 3 best.ckpt + 3 config.yaml + map + MapEx configs/base.yaml
        _write(root, 'CHECKPOINT_FAIL', dict(bad=bad, n=len(pins)))
        raise DependencyDrift('CHECKPOINT_PIN_MISMATCH %s' % sorted(bad))
    print('GLOBAL001_CHECKPOINT_PINS_OK', len(pins), flush=True)
    return True


def check_imports(root, mode):
    o = origins(root)
    o['runtime_versions'] = runtime_versions()
    if os.environ.get('GLOBAL001_IMPORT_PROBE') == '1':
        p = _write(root, 'PROBE', o)
        print('GLOBAL001_IMPORT_PROBE_WRITTEN', p, flush=True)
        return o
    m = _manifest(root)
    pins = m['import_pins']
    errs = []
    rr = os.path.realpath(root)
    for n, f in o['staged_origins'].items():
        if f is None or os.path.dirname(f) != rr:
            errs.append('STAGED_MODULE_NOT_FROM_ROOT %s %s' % (n, f))
    for f, s in o['files'].items():
        key = os.path.relpath(f, rr) if _under(f, rr) else f
        if key not in pins:
            errs.append('UNPINNED_IMPORT %s' % key)
        elif pins[key] != s:
            errs.append('IMPORT_PIN_MISMATCH %s' % key)
    seen = {os.path.relpath(f, rr) if _under(f, rr) else f for f in o['files']}
    for key in pins:
        if key not in seen:
            errs.append('PINNED_NOT_IMPORTED %s' % key)
    for x in o['outside_allowed_roots']:
        errs.append('IMPORT_OUTSIDE_ALLOWED_ROOTS %s %s' % (x['module'], x['file']))
    for x in o['unresolved']:
        errs.append('UNRESOLVED_MODULE_FILE %s %s' % (x['module'], x['file']))
    ex = os.path.join(rr, 'explore_mapex_gt_r1.py')         # exec'd by the worker, not in sys.modules
    if _sha(ex) != m['staged_sha256']['explore_mapex_gt_r1.py']:
        errs.append('EXPLORE_SOURCE_MISMATCH')
    if o['runtime_versions'] != m['runtime_versions']:
        errs.append('RUNTIME_VERSIONS %s != %s' % (o['runtime_versions'], m['runtime_versions']))
    o['errors'] = errs
    p = _write(root, mode, o)
    if errs:
        raise DependencyDrift('IMPORT_GUARD_FAIL %d errors, see %s: %s' % (len(errs), p, errs[:5]))
    print('GLOBAL001_IMPORT_GUARD_PASS', mode, len(o['files']), p, flush=True)
    return o
