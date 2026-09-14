"""Verify recorded file bytes without running any numerical calculation."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / 'REPOSITORY_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise SystemExit(f'Missing or unsafe path: {name}')
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise SystemExit(f'Integrity mismatch: {name}')
    print(f"Verified {len(manifest['files'])} repository files; no simulations executed.")


if __name__ == '__main__':
    main()
