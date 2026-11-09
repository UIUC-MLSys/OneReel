"""Transition accuracy: are the cuts in a multi-shot video at the intended shot boundaries?

Cut detection re-implements PySceneDetect's AdaptiveDetector (defaults: HSV content value, window 2, adaptive
threshold 3.0, min content 15, frames downscaled to 256 px wide); min_scene_len is 8 frames here because shots are 25-28
frames. The intended boundaries are the first pixel frame of shots 2..K (prompt's segments[<frames>], else an even split).
Detected cuts are matched one-to-one to intended boundaries within +-tol frames (greedy by distance). Per video:
  recall      matched boundaries / (K - 1)      transition accuracy: the intended cut happened
  precision   matched cuts / detected cuts      cuts that were not asked for lower it (a shot split in two)
  f1, dev     mean |frame offset| of matched cuts
  exact       1 if every boundary is matched and there is no extra cut
Only hard cuts are detected. A gradual transition (dissolve / morph over several frames) counts as missed; telling it
from ordinary motion needs a learned detector such as TransNetV2.

Usage: python eval/event_order/transition_acc.py RUN_SETTINGS_DIR --bench B.json [--tol 4 8] [--out OUT.json]
RUN_SETTINGS_DIR holds <setting>/<id>/video.mp4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np

WINDOW, ADAPTIVE, MIN_CONTENT, MIN_LEN, WIDTH = 2, 3.0, 15.0, 8, 256


def content_values(path):
    """mean |delta| of H, S, V between consecutive frames; c[t] compares frame t-1 and t (c[0] = 0)"""
    prev, out = None, [0.0]
    for f in av.open(str(path)).decode(video=0):
        x = f.to_ndarray(format='bgr24')
        h = int(round(x.shape[0] * WIDTH / x.shape[1]))
        hsv = cv2.cvtColor(cv2.resize(x, (WIDTH, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV).astype(np.float32)
        if prev is not None:
            out.append(float(np.abs(hsv - prev).reshape(-1, 3).mean(0).mean()))
        prev = hsv
    return np.array(out)


def detect_cuts(c):
    cuts, last = [], 0
    for t in range(1 + WINDOW, len(c) - WINDOW):
        nb = np.r_[c[t - WINDOW:t], c[t + 1:t + 1 + WINDOW]]
        ratio = c[t] / max(nb.mean(), 1e-5)
        if ratio >= ADAPTIVE and c[t] >= MIN_CONTENT and t - last >= MIN_LEN:
            cuts.append(t)
            last = t
    return cuts


def boundaries(r, n):
    seg = r.get('segments', {}).get(str(n))
    if seg:
        return [x['pixel'][0] for x in seg['per_sentence'][1:]]
    k = len(r['sentences'])
    return [int(round(i * n / k)) for i in range(1, k)]


def match(gt, det, tol):
    pairs = sorted((abs(g - d), i, j) for i, g in enumerate(gt) for j, d in enumerate(det) if abs(g - d) <= tol)
    ug, ud, devs = set(), set(), []
    for dist, i, j in pairs:
        if i not in ug and j not in ud:
            ug.add(i), ud.add(j), devs.append(dist)
    m = len(devs)
    rec = m / len(gt) if gt else 1.0
    prec = m / len(det) if det else (1.0 if not gt else 0.0)
    return dict(recall=rec, precision=prec, f1=2 * rec * prec / (rec + prec) if rec + prec else 0.0,
                dev=float(np.mean(devs)) if devs else float('nan'), exact=float(m == len(gt) == len(det)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path)
    ap.add_argument('--bench', required=True)
    ap.add_argument('--tol', type=int, nargs='+', default=[4, 8])
    ap.add_argument('--settings', default='', help='comma list; default: every subdir with videos')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    rows = {r['id']: r for r in json.loads(Path(a.bench).read_text())['prompts']}
    settings = a.settings.split(',') if a.settings else sorted(
        p.name for p in a.run.iterdir() if p.is_dir() and any(p.glob('*/video.mp4')))
    per_video = []
    for s in settings:
        for v in sorted((a.run / s).glob('*/video.mp4')):
            pid = v.parent.name
            if pid not in rows:
                continue
            c = content_values(v)
            gt, det = boundaries(rows[pid], len(c)), detect_cuts(c)
            rec = dict(setting=s, id=pid, frames=len(c), gt=gt, cuts=det)
            for t in a.tol:
                rec.update({f'{k}@{t}': x for k, x in match(gt, det, t).items()})
            per_video.append(rec)
            print(f'{s:34s} {pid:6s} gt={gt} cuts={det}', flush=True)
    keys = [f'{k}@{t}' for t in a.tol for k in ('recall', 'precision', 'f1', 'exact', 'dev')]
    lines = ['| setting | n | ' + ' | '.join(keys) + ' | cuts/video |', '|---|---|' + '---|' * (len(keys) + 1)]
    for s in settings:
        rs = [r for r in per_video if r['setting'] == s]
        if rs:
            lines.append(f'| {s} | {len(rs)} | ' + ' | '.join(f'{np.nanmean([r[k] for r in rs]):.2f}' for k in keys)
                         + f' | {np.mean([len(r["cuts"]) for r in rs]):.1f} (want {np.mean([len(r["gt"]) for r in rs]):.1f}) |')
    print('\n'.join(lines))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({'per_video': per_video, 'table': lines}, indent=1))


if __name__ == '__main__':
    main()
