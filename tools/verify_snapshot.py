#!/usr/bin/env python3
"""Verify copied bytes and parse Python without importing research modules."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
errors = []
for row in manifest['files']:
    path = root / row['path']
    if not path.is_file():
        errors.append('MISSING ' + row['path'])
    elif hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        errors.append('HASH_MISMATCH ' + row['path'])
python_files = sorted(root.rglob('*.py'))
for path in python_files:
    try:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except (SyntaxError, UnicodeError) as exc:
        errors.append(str(path.relative_to(root)) + ': ' + str(exc))
if errors:
    raise SystemExit('\n'.join(errors))
print('PASS: %d copied files match SHA256; %d Python files parse. No native/model run.' %
      (len(manifest['files']), len(python_files)))
