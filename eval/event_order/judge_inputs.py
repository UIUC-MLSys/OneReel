"""Extract the images that eval/event_order/gemini_judge.py sends to Gemini, following two published protocols.

EntityBench (arXiv 2605.15199), single subject per prompt:
  - presence: GroundingDINO (box 0.25, text 0.20) on 5 evenly spaced frames per shot; a detection whose crop has CLIP
    similarity >= 0.20 to the subject phrase is `present`, a lower one `weak`, no detection `absent`
  - canonical crop per shot: the detection maximising CLIP x sharpness x box area, 10% padding, 224x224
  - DINOv2 CLS of every canonical crop (the judge picks the anchor closest to the gate-passing centroid)
  - action grid per shot: 6 evenly spaced frames in a 2x3 grid, the subject box drawn and labelled
CineWeaver (arXiv 2607.26529), narrative coherence: 20 frames sampled across shots in proportion to shot length, at
least one per shot.

Writes OUT/<setting>/<id>/{inputs.json, crop_<j>.jpg, grid_<j>.jpg, narr_<i>.jpg}.
Usage (onereel_metrics env, HF_HUB_OFFLINE=1): python eval/event_order/judge_inputs.py RUN_SETTINGS_DIR --bench B.json
    --settings a,b --out OUT
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import Grounder, Scorer, read_video  # noqa: E402

N_PRESENCE, N_GRID, N_NARR, PAD, CROP = 5, 6, 20, 0.10, 224


def shot_ranges(r, n):
    if str(n) in r.get('segments', {}):
        return [x['pixel'] for x in r['segments'][str(n)]['per_sentence']]
    K = len(r['sentences'])
    b = [round(n * j / K) for j in range(K + 1)]
    return [[b[j], b[j + 1]] for j in range(K)]


def main_entity(r):
    """prompts with an entities list (dataset/multishot_bench.json): the entity whose description appears in the most
    sentences, and a short detector query from its head noun phrase; else None"""
    ents = r.get('entities') or []
    if not ents:
        return None
    e = max(ents, key=lambda e: sum(e['description'].lower()[:40] in s.lower() for s in r['sentences']))
    head = re.split(r',| with | wearing | in (?:a|an|his|her|practical) ', e['description'], maxsplit=1)[0]
    return e['description'].rstrip('.'), re.sub(r'^(a|an|the)\s+', '', head.strip(), flags=re.I).lower()


def description(sentences, subject):
    """the entity card: the words every sentence starts with (pronoun prompts have none: 'a <subject>')"""
    w = [s.split() for s in sentences]
    k = 0
    while all(len(x) > k and x[k] == w[0][k] for x in w):
        k += 1
    return ' '.join(w[0][:k]).rstrip(',') if k >= 3 else f'a {subject}'


def even(lo, hi, n):
    return np.linspace(lo + 1, hi - 2, n).round().astype(int)


@torch.no_grad()
def detect(gd, frames, subject):
    ims = [Image.fromarray(f) for f in frames]
    inp = gd.proc(images=ims, text=[subject.lower() + '.'] * len(ims), return_tensors='pt').to('cuda')
    res = gd.proc.post_process_grounded_object_detection(gd.model(**inp), inp.input_ids, threshold=0.25,
                                                         text_threshold=0.20, target_sizes=[im.size[::-1] for im in ims])
    return [r['boxes'][int(r['scores'].argmax())].tolist() if len(r['scores']) else None for r in res]


def padded(frame, box):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * PAD, (y1 - y0) * PAD
    x0, y0, x1, y1 = int(max(x0 - px, 0)), int(max(y0 - py, 0)), int(min(x1 + px, w)), int(min(y1 + py, h))
    return frame[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path)
    ap.add_argument('--bench', required=True)
    ap.add_argument('--settings', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    bench = {r['id']: r for r in json.loads(Path(a.bench).read_text())['prompts']}
    sc = Scorer()
    gd = Grounder(sc)
    for s in a.settings.split(','):
        for v in sorted((a.run / s).glob('*/video.mp4')):
            pid = v.parent.name
            if pid not in bench:
                continue
            r, d = bench[pid], a.out / s / pid
            if (d / 'inputs.json').exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            f = read_video(v)
            ranges = shot_ranges(r, len(f))
            me = main_entity(r)
            if me:
                desc, subj = me
            else:
                subj = r['subject']
                desc = description(r['sentences'], subj)
            shots = []
            for j, (lo, hi) in enumerate(ranges):
                idx = even(lo, hi, N_PRESENCE)
                boxes = detect(gd, [f[i] for i in idx], subj)
                best, status = None, 'absent'
                for i, b in zip(idx, boxes):
                    if b is None or b[2] - b[0] < 8 or b[3] - b[1] < 8:
                        continue
                    c = padded(f[i], b)
                    clip = float(sc.clip_sim([c], [subj])[0, 0])
                    sharp = float(cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
                    area = (b[2] - b[0]) * (b[3] - b[1]) / (f[i].shape[0] * f[i].shape[1])
                    status = 'present' if clip >= 0.20 or status == 'present' else 'weak'
                    cand = dict(frame=int(i), box=b, clip=clip, sharp=sharp, area=area, crop=c)
                    if best is None or clip * sharp * area > best['clip'] * best['sharp'] * best['area']:
                        best = cand
                rec = dict(shot=j, range=[int(lo), int(hi)], status=status)
                if best is not None:
                    Image.fromarray(best.pop('crop')).resize((CROP, CROP)).save(d / f'crop_{j}.jpg', quality=92)
                    rec.update(best, emb=sc.dino_cls([np.asarray(Image.open(d / f'crop_{j}.jpg'))])[0].tolist())
                gi = even(lo, hi, N_GRID)
                gb = detect(gd, [f[i] for i in gi], subj)
                tiles = []
                for i, b in zip(gi, gb):
                    im = Image.fromarray(f[i])
                    if b is not None:
                        dr = ImageDraw.Draw(im)
                        dr.rectangle(b, outline=(255, 0, 0), width=3)
                        dr.text((b[0] + 4, b[1] + 4), subj, fill=(255, 0, 0))
                    tiles.append(np.asarray(im.resize((480, int(480 * im.height / im.width)))))
                grid = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
                Image.fromarray(grid).save(d / f'grid_{j}.jpg', quality=88)
                shots.append(rec)
            # narrative frames: proportional to shot length, at least one per shot
            lens = np.array([hi - lo for lo, hi in ranges], float)
            k = np.maximum(1, np.floor(lens / lens.sum() * N_NARR)).astype(int)
            while k.sum() < N_NARR:
                k[np.argmax(lens / k)] += 1
            narr = []
            for j, ((lo, hi), kj) in enumerate(zip(ranges, k)):
                for i in even(lo, hi, kj):
                    Image.fromarray(f[i]).resize((480, int(480 * f[i].shape[0] / f[i].shape[1]))).save(
                        d / f'narr_{len(narr):02d}.jpg', quality=88)
                    narr.append(dict(frame=int(i), shot=j))
            (d / 'inputs.json').write_text(json.dumps(dict(setting=s, id=pid, subject=subj, description=desc,
                                                          sentences=r['sentences'], shots=shots, narrative=narr)))
            print(s, pid, [x['status'] for x in shots], flush=True)


if __name__ == '__main__':
    main()
