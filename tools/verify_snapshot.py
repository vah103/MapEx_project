#!/usr/bin/env python3
"""Verify archived source bytes without importing research modules."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
errors = []
files_checked = 0
legacy = set()
sources = [(root / 'SOURCE_MANIFEST.json', root)]
original = root / 'mapex_original' / 'SOURCE_MANIFEST.json'
if original.exists():
    sources.append((original, original.parent))
for manifest_path, base in sources:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for row in manifest['files']:
        path = base / row['path']
        if not path.is_file():
            errors.append('MISSING ' + str(path.relative_to(root)))
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            errors.append('HASH_MISMATCH ' + str(path.relative_to(root)))
        files_checked += 1
    for note in manifest.get('python_syntax_notes', []):
        # Legacy syntax is allowed only for byte-verified upstream files.
        name = 'src/' + note['path']
        if name not in {row['path'] for row in manifest['files']}:
            errors.append('UNVERIFIED_LEGACY_FILE ' + name)
        else:
            legacy.add(base / name)
python_files = sorted(root.rglob('*.py'))
parsed = 0
for path in python_files:
    if path in legacy:
        continue
    try:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        parsed += 1
    except (SyntaxError, UnicodeError) as exc:
        errors.append(str(path.relative_to(root)) + ': ' + str(exc))
if errors:
    raise SystemExit('\n'.join(errors))
print('PASS: %d archived files match SHA256; %d Python files parse; %d declared legacy upstream files preserved. No native/model run.' %
      (files_checked, parsed, len(legacy)))
