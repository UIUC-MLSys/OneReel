"""Joint narrative metrics: a shot succeeds only if it follows its own sentence AND keeps the subject's identity.

Per shot j: `correct` = its mean CLIP similarity is highest for sentence j (as seg_acc); `emb` = mean DINOv2 embedding
of the GroundingDINO subject crops in 4 frames (as subj_across). Identity threshold tau = the 95th percentile of
subject similarity between shots of DIFFERENT prompts whose subjects differ (pooled over every setting of the run), i.e.
"more alike than 95% of different-character pairs". Per video:
  same_id      share of shot pairs whose subject similarity >= tau
  joint_pair   share of shot pairs where both shots are correct AND same identity      (main joint metric)
  joint_shot   share of shots j >= 2 that are correct AND same identity as a correct shot 1 (narrative success)
Shots without a subject crop count as failures. Caches per-shot results in OUT_DIR/narrative_cache.

Usage (onereel or onereel_metrics env, HF_HUB_OFFLINE=1): python eval/event_order/narrative_score.py RUN_SETTINGS_DIR
    --bench B.json --settings a,b,c --out OUT_DIR [--same-subject-key subject]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import segment_index  # noqa: E402
from models import Grounder, Scorer, read_video  # noqa: E402


def shots(sc, gd, video, r):
    f = read_video(video)
    n, K = len(f), len(r['sentences'])
    if str(n) in r.get('segments', {}):
        ranges = [x['pixel'] for x in r['segments'][str(n)]['per_sentence']]
    else:
        b = [round(n * j / K) for j in range(K + 1)]
        ranges = [[b[j], b[j + 1]] for j in range(K)]
    seg = segment_index(n, ranges)
    sim = sc.clip_sim(f, r['sentences'])
    out = []
    for j in range(K):
        idx = np.flatnonzero(seg == j)
        es = []
        for i in idx[np.linspace(2, len(idx) - 3, 4).round().astype(int)]:
            e, c = gd.crop_emb([f[i]], r['subject'])
            if c:
                es.append(e[0] / np.linalg.norm(e[0]))
        emb = np.mean(es, 0) if es else None
        out.append({'correct': bool(sim[seg == j].mean(0).argmax() == j),
                     'emb': (emb / np.linalg.norm(emb)).tolist() if emb is not None else None})
    return out


def cos(a, b):
    return float(np.dot(a, b)) if a is not None and b is not None else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path)
    ap.add_argument('--bench', required=True)
    ap.add_argument('--settings', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tau', type=float, default=None,
                    help='fixed identity threshold (e.g. that of another report) instead of the 95th percentile')
    ap.add_argument('--anchor', choices=('first', 'first_detected'), default='first',
                    help='joint_shot anchor: shot 1, or the first shot with a subject crop (requests opening on an empty shot)')
    a = ap.parse_args()
    bench = {r['id']: r for r in json.loads(Path(a.bench).read_text())['prompts']}
    settings = a.settings.split(',')
    cache = a.out / 'narrative_cache'
    cache.mkdir(parents=True, exist_ok=True)
    sc = gd = None
    data = {}
    for s in settings:
        for v in sorted((a.run / s).glob('*/video.mp4')):
            pid = v.parent.name
            if pid not in bench:
                continue
            cf = cache / f'{s}_{pid}.json'
            if not cf.exists():
                if sc is None:
                    sc = Scorer()
                    gd = Grounder(sc)
                cf.write_text(json.dumps(shots(sc, gd, v, bench[pid])))
            data[(s, pid)] = json.loads(cf.read_text())
            print(s, pid, [x['correct'] for x in data[(s, pid)]], flush=True)
    # identity threshold from shots of different prompts with different subjects
    neg = []
    for s in settings:
        keys = [k for k in data if k[0] == s]
        for (_, p1), (_, p2) in itertools.combinations(keys, 2):
            if bench[p1]['subject'].lower() == bench[p2]['subject'].lower():
                continue
            for x in data[(s, p1)]:
                for y in data[(s, p2)]:
                    c = cos(x['emb'], y['emb'])
                    if c == c:
                        neg.append(c)
    tau = float(np.percentile(neg, 95)) if a.tau is None else a.tau
    rows = []
    for (s, pid), sh in data.items():
        K = len(sh)
        emb = [None if x['emb'] is None else np.array(x['emb']) for x in sh]
        same = {(i, j): cos(emb[i], emb[j]) >= tau for i, j in itertools.combinations(range(K), 2)}
        ok = [x['correct'] for x in sh]
        a0 = 0                                       # anchor shot of joint_shot
        if a.anchor == 'first_detected':
            a0 = next((j for j, e in enumerate(emb) if e is not None), 0)
        rows.append({'setting': s, 'id': pid, 'seg_acc': float(np.mean(ok)),
                     'same_id': float(np.mean(list(same.values()))),
                     'joint_pair': float(np.mean([ok[i] and ok[j] and v for (i, j), v in same.items()])),
                     'joint_shot': float(np.mean([ok[a0] and ok[j] and same[(a0, j)] for j in range(a0 + 1, K)]))})
    keys = ['seg_acc', 'same_id', 'joint_pair', 'joint_shot']
    lines = [f'identity threshold tau = {tau:.3f} (' + ('fixed' if a.tau is not None else f'95th pct of {len(neg)} different-subject shot pairs') + ')', '',
             '| setting | n | ' + ' | '.join(keys) + ' |', '|---|---|' + '---|' * len(keys)]
    for s in settings:
        rs = [r for r in rows if r['setting'] == s]
        if rs:
            lines.append(f'| {s} | {len(rs)} | ' + ' | '.join(f'{np.mean([r[k] for r in rs]):.3f}' for k in keys) + ' |')
    print('\n'.join(lines))
    (a.out / 'narrative.json').write_text(json.dumps({'tau': tau, 'rows': rows, 'table': lines}, indent=1))


if __name__ == '__main__':
    main()
