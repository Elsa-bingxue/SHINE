"""Check the extracted figure-data files against the public SHA-256 inventory."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data'),
                        help='Directory containing revision_experiments/ and outputs/.')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    inventory = json.loads((repo / 'data/replay_files.json').read_text())['files']
    missing, changed = [], []
    for relative, expected in inventory.items():
        path = args.root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(2 ** 20), b''):
                digest.update(block)
        if path.stat().st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
            changed.append(relative)
    for label, paths in [('Missing', missing), ('Changed', changed)]:
        for path in paths:
            print(label + ': ' + path)
    if missing or changed:
        raise SystemExit('Data check failed: %d missing, %d changed.' % (len(missing), len(changed)))
    print('Verified all %d files. The seven section notebooks are ready.' % len(inventory))


if __name__ == '__main__':
    main()
