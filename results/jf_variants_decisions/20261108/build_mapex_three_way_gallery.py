#!/usr/bin/env python3
"""Build minimal three-method pages from completed eight-method galleries.

The source images remain unchanged and are shared with index.html.
"""

import argparse
import json
import re
from pathlib import Path


METHODS = (
    ('mapex', 'MapEx thường'),
    ('feas', 'MapEx Tour (JF-FEAS)'),
    ('topo_mean', 'MapEx Tour R2 (TOPO R2 + LaMa mean)'),
)


def generate(folder: Path) -> None:
    source = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    by_key = {method['key']: method for method in source['methods']}
    if len(source['methods']) != 8 or not all(key in by_key for key, _ in METHODS):
        raise AssertionError(f'Unexpected source methods: {folder}')
    methods = []
    for key, label in METHODS:
        method = dict(by_key[key])
        method['label'] = label
        for frame in method['frames']:
            if not (folder / frame['image']).is_file():
                raise FileNotFoundError(folder / frame['image'])
        methods.append(method)
    data = {'map': source['map'], 'source': source['source'], 'methods': methods}
    json_data = json.dumps(data, indent=2, ensure_ascii=False)
    (folder / 'three_way_manifest.json').write_text(json_data + '\n', encoding='utf-8')

    original = (folder / 'index.html').read_text(encoding='utf-8')
    new_title = f"{source['map']} · 3 phương pháp · Ảnh theo decision"
    original, title_count = re.subn(r'<title>[^<]*</title>', f'<title>{new_title}</title>', original, count=1)
    original, header_count = re.subn(
        r'<header><h1>.*?</h1><p>.*?</p></header>',
        f'<header><h1>{source["map"]} · MapEx / Tour / Tour R2</h1>'
        '<p>3 phương pháp · dữ liệu run thực · ảnh từng decision</p></header>',
        original, count=1, flags=re.DOTALL,
    )
    safe_json = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    original, data_count = re.subn(
        r'(<script id="dataset" type="application/json">).*?(</script>)',
        lambda match: match.group(1) + safe_json + match.group(2),
        original, count=1, flags=re.DOTALL,
    )
    if (title_count, header_count, data_count) != (1, 1, 1):
        raise AssertionError(f'Unexpected source HTML: {folder}')
    (folder / 'three_way.html').write_text(original, encoding='utf-8')
    print(f'{source["map"]}: ' + ', '.join(f'{m["label"]}: {m["count"]}' for m in methods))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('gallery_root', type=Path)
    args = parser.parse_args()
    for dirname in ('kth49_p01', 'kth51_p00'):
        generate(args.gallery_root / dirname)


if __name__ == '__main__':
    main()
