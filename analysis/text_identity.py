"""CLIP Text against identity consistency, checked with Gemini (which judges each shot against its own sentence only).

Reads a quality run (narrative_score.py report, gemini_judge.py v2 rows, gemini_pairwise.py rows against separate):
  1. per setting, Spearman correlation over prompts of same_id with CLIP Text (seg_acc) and with Gemini Action
     (two-sided permutation p);
  2. per OneReel setting, the prompts split by whether same_id rose by more than --rise over separate, with the mean
     change of same_id, Text, Text+ID (joint_shot) and Action per group, and for --pairwise-setting the blind pairwise
     shot-following wins / ties / losses against separate (sign-test p).
The appendix numbers: python analysis/text_identity.py RUN_DIR
"""
import argparse
import json
import math
import random
import statistics as S
from pathlib import Path


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for k, i in enumerate(o):
        r[i] = k
    return r


def spear(a, b):
    x, y = rank(a), rank(b)
    mx, my = S.mean(x), S.mean(y)
    return sum((p - mx) * (q - my) for p, q in zip(x, y)) / (sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)) ** 0.5


def perm_p(a, b, n=20000):
    """two-sided permutation p for Spearman"""
    random.seed(0)
    r0 = abs(spear(a, b))
    b = list(b)
    hit = 0
    for _ in range(n):
        random.shuffle(b)
        hit += abs(spear(a, b)) >= r0
    return hit / n


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path, help='run directory with report/<model>/ and judge/<model>/')
    ap.add_argument('--model', default='hunyuan')
    ap.add_argument('--settings', default='separate,joint_sep_fm4,joint_sep_fm4_sw5,baseline')
    ap.add_argument('--ours', default='joint_sep_fm4,joint_sep_fm4_sw5', help='settings split against separate')
    ap.add_argument('--pairwise-setting', default='joint_sep_fm4_sw5', help='the setting of judge/<model>/pairwise_rows.json')
    ap.add_argument('--rise', type=float, default=0.2)
    a = ap.parse_args()
    d = json.loads((a.run / f'report/{a.model}/narrative.json').read_text())['rows']
    g = {(r['setting'], r['id']): r for r in json.loads((a.run / f'judge/{a.model}/gemini_rows_v2.json').read_text())}
    pw = {r['id']: r for r in json.loads((a.run / f'judge/{a.model}/pairwise_rows.json').read_text())['separate']}

    for s in a.settings.split(','):
        rs = [r for r in d if r['setting'] == s]
        x, t = [r['same_id'] for r in rs], [r['seg_acc'] for r in rs]
        act = [g[(s, r['id'])]['action_depicted'] for r in rs]
        print(f"{s:18s} Text {100 * S.mean(t):5.1f} same_id {100 * S.mean(x):5.1f} Action {100 * S.mean(act):5.1f}  "
              f"rho(same_id, Text) {spear(x, t):+.2f} (p={perm_p(x, t):.3f})  rho(same_id, Action) {spear(x, act):+.2f} "
              f"(p={perm_p(x, act):.3f})")
    sep = {r['id']: r for r in d if r['setting'] == 'separate'}
    for s in a.ours.split(','):
        ours = {r['id']: r for r in d if r['setting'] == s}
        ids = sorted(sep)
        rose = [i for i in ids if ours[i]['same_id'] - sep[i]['same_id'] > a.rise]
        for name, sel in ((f'same_id rose > {a.rise}', rose), ('other', [i for i in ids if i not in rose])):
            dT = 100 * S.mean(ours[i]['seg_acc'] - sep[i]['seg_acc'] for i in sel)
            dI = 100 * S.mean(ours[i]['same_id'] - sep[i]['same_id'] for i in sel)
            dJ = 100 * S.mean(ours[i]['joint_shot'] - sep[i]['joint_shot'] for i in sel)
            dA = 100 * S.mean(g[(s, i)]['action_depicted'] - g[('separate', i)]['action_depicted'] for i in sel)
            line = f'{s:18s} {name:18s} n={len(sel):2d}: same_id {dI:+5.1f} Text {dT:+5.1f} Text+ID {dJ:+5.1f} Action {dA:+5.1f}'
            if s == a.pairwise_setting:
                sf = [pw[i]['shot_following'] for i in sel]
                w, t, l = sf.count(1), sf.count(0), sf.count(-1)
                line += f'  pairwise shot_following vs separate {w}/{t}/{l} (p={sign_p(w, l):.2f})'
            print(line)


if __name__ == '__main__':
    main()
