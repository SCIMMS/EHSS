"""Safely extract and verify the frozen evidence in a new directory."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = '22e9f66d61447cb20e02b28784c2eb0e35de652b1b23dbe2bde2665380784645'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('work/evidence'))
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        parser.error('Output must be a new directory; existing data is never overwritten.')
    archive = ROOT / 'data/evidence/ehss_evidence_snapshot_v4.zip'
    if digest(archive) != EXPECTED:
        raise SystemExit('Evidence archive SHA256 mismatch.')
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = PurePosixPath(member.filename)
            path = (dest / member.filename).resolve()
            if (name.is_absolute() or '..' in name.parts or ':' in member.filename
                    or '\\' in member.filename or not path.is_relative_to(dest)):
                raise SystemExit(f'Unsafe member path: {member.filename}')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise SystemExit(f'Symlink member rejected: {member.filename}')
        dest.mkdir(parents=True)
        z.extractall(dest)
    manifest = json.loads((dest / 'snapshot_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (dest / name).resolve()
        if not path.is_relative_to(dest) or not path.is_file():
            raise SystemExit(f'Missing or unsafe manifested path: {name}')
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise SystemExit(f'Extracted evidence mismatch: {name}')
    print(f"Verified {len(manifest['files'])} evidence files in {dest}.")
    print('No analysis, scattering simulation or vendor executable was run.')


if __name__ == '__main__':
    main()
