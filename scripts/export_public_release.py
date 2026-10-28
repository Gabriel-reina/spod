"""Export only portable SPOD sources and public data; omit local cluster adapters."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ('README.md', 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'requirements.txt',
         'spod_analyze.py', 'spod_metrics.py', '.gitignore')
TREES = ('entry', 'configs', 'scripts', 'roll_spod', 'data', 'data_pipeline', 'rustc_perf', 'tests')
OMIT_RAW = {'prs.jsonl', 'issues.jsonl', 'records.jsonl', 'README_snapshot.md'}


def export(destination):
    paths = set(ROOT / p for p in FILES)
    for tree in TREES:
        for path in (ROOT / tree).rglob('*'):
            if path.is_file() and not any(part.startswith('.') or part == '__pycache__' or part.startswith('output')
                                          for part in path.relative_to(ROOT).parts):
                if path.name not in OMIT_RAW and path.suffix != '.pyc':
                    paths.add(path)
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksums = {}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):
            if path == destination:
                continue
            rel = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            z.writestr(rel, data)
            checksums[rel] = hashlib.sha256(data).hexdigest()
        z.writestr('RELEASE_MANIFEST.json', json.dumps(checksums, indent=2))
    print(f'{destination}: {len(checksums)} files')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'dist/spod_public_seed42.zip')
    export(p.parse_args().output)
