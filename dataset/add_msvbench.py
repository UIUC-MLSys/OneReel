"""Add the five MSVBench stories to dataset/multishot_bench.json, giving the 40 prompts of the paper.

The MSVBench release states no license, so its text is not redistributed in this repository. This script reads the
ten files it needs from the official release on Hugging Face (a fixed revision, with HTTP range requests, so the
742 MB archive is not downloaded), puts them under data/external/msvbench, builds the rows with the rules of
build_multishot_bench.py and inserts them in the paper's order. Running it again changes nothing.

Usage: python dataset/add_msvbench.py [--bench dataset/multishot_bench.json] [--raw data/external]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from build_multishot_bench import MSVBENCH, msvbench

REVISION = 'a15da13e6f5809b3c7dd8edac59de403e39f66e4'
URL = f'https://huggingface.co/datasets/MrSunshy/MSVBench/resolve/{REVISION}/Dataset.zip'
FULL_SHA256 = '6acf9d40f5fb9feffcb0df62db341264d754a7b878baa1ee4993702ac94ae705'     # of the 40-prompt bench used in the paper


class HttpFile(io.RawIOBase):
    """the read-only seekable file zipfile needs, backed by HTTP range requests"""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-0'})) as r:
            self.size = int(r.headers['Content-Range'].rsplit('/', 1)[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        if end <= self.pos:
            return b''
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end - 1}'})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data


def fetch(raw: Path) -> None:
    """Dataset/script/<id>.json and Dataset/prompt/<id>.txt of the five stories, unless they are already there"""
    root = raw / 'msvbench' / 'Dataset'
    wanted = {f'{kind}/{sid}.{ext}' for sid in MSVBENCH for kind, ext in (('script', 'json'), ('prompt', 'txt'))}
    if all((root / name).exists() for name in wanted):
        return
    with zipfile.ZipFile(HttpFile(URL)) as z:
        members = {name: m for m in z.namelist() for name in wanted if m == name or m.endswith('/' + name)}
        missing = wanted - set(members)
        if missing:
            raise SystemExit(f'not in the archive: {sorted(missing)}')
        for name, member in members.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(z.read(member))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', type=Path, default=Path('dataset/multishot_bench.json'))
    ap.add_argument('--raw', type=Path, default=Path('data/external'))
    a = ap.parse_args()
    rows = json.loads(a.bench.read_text())['prompts']
    if any(r['source'] == 'MSVBench' for r in rows):
        print(a.bench, 'already holds the MSVBench stories')
        return
    fetch(a.raw)
    at = max(i for i, r in enumerate(rows) if r['source'] == 'HoloCine') + 1      # the builder's order: HoloCine, MSVBench, ...
    text = json.dumps({'prompts': rows[:at] + msvbench(a.raw) + rows[at:]}, indent=1, ensure_ascii=False)
    a.bench.write_text(text)
    same = hashlib.sha256(text.encode()).hexdigest() == FULL_SHA256
    print(f'wrote {a.bench}: {len(rows) + len(MSVBENCH)} prompts,', 'identical to the paper\'s bench' if same else
          'WARNING: differs from the paper\'s bench')


if __name__ == '__main__':
    main()
