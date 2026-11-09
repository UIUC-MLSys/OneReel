"""Blind pairwise preference (Good / Same / Bad) between two settings, judged by Gemini on the same multi-shot script.

For every prompt, the judge sees the script (shot descriptions in order and the recurring characters) and the 20
narrative frames of two videos (judge_inputs.py: sampled across the shots in proportion to their length, each frame
labelled with the shot whose time span it comes from), shown as "Video A" and "Video B". It never learns which method
made a video. Every pair is judged twice with A and B swapped; a dimension counts as a win only when both orders pick
the same video, otherwise as a tie (position-bias control). Dimensions, each with a short reason stated first:
  shot_following         every shot shows its own description, in order
  character_consistency  recurring characters stay the same individuals, judged over the shots that show them as
                         described (repeating one image instead of the described shots earns nothing; EntityBench's
                         gating, in words)
  visual_quality         artifacts, distortions, blur, flicker, independent of the script
  overall                which video better realises the script as a whole
The prompt text is fixed (PROMPT below) and identical for every comparison.

Output: OUT/<model>/pairwise/<ours>__vs__<other>/<id>_<AB|BA>.json (cache), pairwise_summary.md and
pairwise_rows.json (per prompt: winner per dimension from <ours>'s side and both reasons).

Usage: python eval/event_order/gemini_pairwise.py OUT --model-dir hunyuan --ours joint_sep_grope_adj2
    --others baseline,separate,radial,svg,solk --bench dataset/multishot_bench.json --credentials KEY.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

from gemini_judge import Gemini, img

DIMS = ['shot_following', 'character_consistency', 'visual_quality', 'overall']
PROMPT = (
    'You compare two generated videos made from the same multi-shot script.\n'
    'Script ({k} shots, in order):\n{shots}\n'
    'Recurring characters:\n{chars}\n'
    'For each video you see 20 frames in time order; each frame is labelled with the shot of the script its time span '
    'belongs to.\n'
    'Judge each dimension separately and answer "A", "B" or "tie":\n'
    '- shot_following: which video better shows every shot as its description asks (subject, action, setting, '
    'framing), in the given order, with each shot showing its own content?\n'
    '- character_consistency: which video better keeps each recurring character the same individual (face, hair, '
    'clothing, build) across the shots in which that character appears as described? Only shots that show the '
    'character as described count; repeating one image instead of showing the described shots does not count as '
    'consistency.\n'
    '- visual_quality: which video has fewer artifacts, distortions, blur or flicker, regardless of the script?\n'
    '- overall: which video better realises the script as a whole?\n'
    'Answer "tie" when the videos are equally good or equally poor on a dimension.\n'
    'Before each verdict, state your reason in one or two sentences that name what you see.\n'
    'Answer JSON: {{"reasons": {{"shot_following": str, "character_consistency": str, "visual_quality": str, '
    '"overall": str}}, "shot_following": "A"|"B"|"tie", "character_consistency": "A"|"B"|"tie", '
    '"visual_quality": "A"|"B"|"tie", "overall": "A"|"B"|"tie"}}')


def characters(row) -> str:
    ents = row.get('entities') or []
    return '\n'.join(f'- {e["description"]}' for e in ents) or '- (none named)'


def frames(d: Path, tag: str):
    x = json.loads((d / 'inputs.json').read_text())
    parts = [{'text': f'Video {tag}:'}]
    for i, n in enumerate(x['narrative']):
        parts += [{'text': f'{tag} frame {i + 1} (shot {n["shot"] + 1})'}, img(d / f'narr_{i:02d}.jpg')]
    return parts


def judge_pair(g: Gemini, row, ours: Path, other: Path, cache: Path) -> dict:
    """both orders; per dimension +1 ours wins, -1 other wins, 0 tie (or the orders disagree)"""
    text = PROMPT.format(k=len(row['sentences']), shots='\n'.join(f'Shot {j + 1}: {s}' for j, s in enumerate(row['sentences'])),
                         chars=characters(row))
    out = {}
    for order, (a, b) in (('AB', (ours, other)), ('BA', (other, ours))):
        out[order] = g([{'text': text}] + frames(a, 'A') + frames(b, 'B'), cache / f'{row["id"]}_{order}.json')
    res = {'id': row['id']}
    if any('blocked' in r for r in out.values()):
        return {**res, 'blocked': True}
    for k in DIMS:
        ab = {'A': 1, 'B': -1}.get(out['AB'][k], 0)                       # ours is A in AB
        ba = {'A': -1, 'B': 1}.get(out['BA'][k], 0)                       # ours is B in BA
        res[k] = ab if ab == ba else 0
        res[k + '_orders'] = [ab, ba]
    res['reasons'] = {o: r.get('reasons', {}) for o, r in out.items()}
    return res


def sign_p(w: int, l: int) -> float:
    """two-sided exact sign test over the decided prompts"""
    n = w + l
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(w, l) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', type=Path, help='judge root (OUT/<model>/<setting>/<id>/ from judge_inputs.py)')
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--ours', required=True)
    ap.add_argument('--others', required=True)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--model', default='gemini-2.5-pro')
    ap.add_argument('--credentials', type=Path, required=True, help='Vertex AI service-account JSON')
    ap.add_argument('--location', default='us-central1')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='first N prompts only (pilot)')
    a = ap.parse_args()
    g = Gemini(a.model, credentials=a.credentials, location=a.location)
    bench = {r['id']: r for r in json.loads(a.bench.read_text())['prompts']}
    root = a.out / a.model_dir
    ids = sorted(p.name for p in (root / a.ours).iterdir() if (p / 'inputs.json').exists())
    if a.limit:
        ids = ids[:a.limit]
    lines = [f'model {a.model_dir}, ours = {a.ours}, judge {a.model}; per dimension: Good / Same / Bad for ours '
             '(a win needs both A/B orders to agree), sign-test p over decided prompts', '',
             '| vs | n | ' + ' | '.join(DIMS) + ' |', '|---|---|' + '---|' * len(DIMS)]
    all_rows = {}
    for other in a.others.split(','):
        pair_ids = [i for i in ids if (root / other / i / 'inputs.json').exists()]
        cache = root / 'pairwise' / f'{a.ours}__vs__{other}'
        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(lambda i: judge_pair(g, bench[i], root / a.ours / i, root / other / i, cache), pair_ids))
        all_rows[other] = rows
        ok = [r for r in rows if not r.get('blocked')]
        cells = []
        for k in DIMS:
            w, t, l = (sum(r[k] == v for r in ok) for v in (1, 0, -1))
            cells.append(f'{w}/{t}/{l} (p={sign_p(w, l):.3f})')
        lines.append(f'| {other} | {len(ok)} | ' + ' | '.join(cells) + ' |')
        print(lines[-1], flush=True)
    blocked = sum(r.get('blocked', False) for rs in all_rows.values() for r in rs)
    lines.append(f'\nGemini refused (safety filter) {blocked} pairs; they are excluded.')
    (root / 'pairwise_summary.md').write_text('\n'.join(lines) + '\n')
    (root / 'pairwise_rows.json').write_text(json.dumps(all_rows, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
