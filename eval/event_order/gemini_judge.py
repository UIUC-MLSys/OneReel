"""Gemini judge for multi-shot videos, following EntityBench (arXiv 2605.15199) and CineWeaver (arXiv 2607.26529).

Reads OUT/<model>/<setting>/<id>/ written by eval/event_order/judge_inputs.py. Gemini REST API. Auth: a Vertex AI
service-account JSON (--credentials; needs google-auth) or a Gemini API key ($GEMINI_API_KEY or --key-file; standard
library only). Every response is cached in <id>/gemini/<call>.json; delete a file to redo that call.

EntityBench (gemini-2.5-pro in the paper):
  fidelity   per shot: canonical crop + entity description -> overall phi in [0, 1]
  gate       a shot is admitted if its crop is `present` and phi >= 0.5 (tau_fid in the paper)
  identity   anchor = admitted crop closest (DINOv2) to the admitted centroid; pairwise anchor-vs-other -> same in {0, 1}
             llm_id_raw = mean(same) over evaluated pairs; llm_id = raw x coverage, coverage = evaluated / (K - 1)
             (shots that fail the gate count as zero, as in the paper's gate-corrected mean)
  action     per shot: 2x3 frame grid with the subject box + the shot sentence -> depicted in {0, 1}, overall in [0, 1]
CineWeaver narrative coherence (Gemini-2.5 in the paper): 20 frames across shots + global and per-shot text -> binary
scene / subject / action / spatial consistency; score = their mean. The papers do not publish their prompt texts; the
prompts below follow the criteria the papers list.

Protocol v2 (default, cache <id>/gemini_v2/): every prompt is the v1 prompt plus one instruction to state a short reason
before the verdict, and every answer carries it ("reason"; the narrative call one per dimension). The reasons are
collected in OUT/<model>/gemini_reasons.jsonl. --v1 reproduces the v1 prompts and cache (<id>/gemini/).

Usage: python eval/event_order/gemini_judge.py OUT --models hunyuan,h3 [--model gemini-2.5-pro] [--workers 8] [--v1]
"""
from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

URL = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
VERTEX_URL = ('https://{host}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}'
              ':generateContent')
TAU_FID = 0.5
DIMS = ['scene_consistency', 'subject_consistency', 'action_coherence', 'spatial_consistency']
REASON = ('Before the verdict, state your reason in one or two sentences that name what you see in the image(s).\n')

FID = ('You are checking whether a generated character matches its description.\n'
       'Character description: "{desc}"\nThe image is a crop of the character from one shot of a video.\n'
       'Rate how faithfully the crop matches the description on face, hair, clothing and build, each from 1 to 10, '
       'and give an overall score from 1 to 10. If no person is visible, give 1 everywhere.\n'
       'Answer JSON: {{"face": int, "hair": int, "clothing": int, "build": int, "overall": int}}')
PAIR = ('Two crops of a character, taken from two different shots of the same video.\n'
        'Intended character: "{desc}"\n'
        'Are they the same individual (same identity)? Clothing may be the same by design, so judge identity mainly by '
        'face, hair and build; a different person in the same outfit is NOT the same. Also rate face, hair, clothing '
        'and build similarity and an overall similarity, each from 1 to 10.\n'
        'Answer JSON: {{"same": bool, "sim": int, "face": int, "hair": int, "clothing": int, "build": int}}')
ACTION = ('The image is a 2x3 grid of 6 frames, in time order, from one shot of a generated video. The red box marks '
          'the {subj}.\nShot description: "{sentence}"\n'
          'Is the described action, in the described setting, actually depicted in this shot? Also rate overall '
          'agreement with the description, subject identity (does the character match "{desc}"), subject action, and '
          'motion quality, each from 1 to 10.\n'
          'Answer JSON: {{"depicted": bool, "overall": int, "identity": int, "action": int, "motion": int}}')
NARR = ('You are evaluating a generated multi-shot video. It has {k} shots; the frames below are sampled in time order '
        'and each is labelled with its shot.\nGlobal description: the {subj} described as "{desc}" goes through these '
        'shots in order.\nShot descriptions:\n{shots}\n'
        'Give a binary judgment (true/false) for each dimension:\n'
        '- scene_consistency: every shot shows the setting its description asks for, and the setting stays coherent '
        'within the shot.\n'
        '- subject_consistency: the main character is recognisably the same individual in every shot.\n'
        '- action_coherence: every shot shows the action its description asks for, in the given order.\n'
        '- spatial_consistency: the spatial layout within each shot is physically plausible (no impossible jumps, '
        'merged scenes or duplicated characters).\n'
        'Answer JSON: {{"scene_consistency": bool, "subject_consistency": bool, "action_coherence": bool, '
        '"spatial_consistency": bool}}')


def img(path):
    return {'inline_data': {'mime_type': 'image/jpeg', 'data': base64.b64encode(Path(path).read_bytes()).decode()}}


class Gemini:
    def __init__(self, model, key='', credentials=None, location='us-central1'):
        self.key, self.creds, self.lock = key, None, threading.Lock()
        if credentials:
            from google.oauth2 import service_account
            self.creds = service_account.Credentials.from_service_account_file(
                str(credentials), scopes=['https://www.googleapis.com/auth/cloud-platform'])
            host = 'aiplatform.googleapis.com' if location == 'global' else f'{location}-aiplatform.googleapis.com'
            self.url = VERTEX_URL.format(host=host, project=self.creds.project_id, location=location, model=model)
        else:
            self.url = URL.format(model=model)

    def headers(self):
        if self.creds is None:
            return {'Content-Type': 'application/json', 'x-goog-api-key': self.key}
        with self.lock:
            if not self.creds.valid:
                from google.auth.transport.requests import Request
                self.creds.refresh(Request())
            return {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.creds.token}'}

    def __call__(self, parts, cache: Path):
        if cache.exists():
            return json.loads(cache.read_text())
        body = json.dumps({'contents': [{'role': 'user', 'parts': parts}],
                           'generationConfig': {'temperature': 0, 'responseMimeType': 'application/json'}}).encode()
        for attempt in range(6):
            req = urllib.request.Request(self.url, body, self.headers())
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    resp = json.loads(r.read())
                cand = (resp.get('candidates') or [{}])[0]
                if 'content' not in cand and (resp.get('promptFeedback', {}).get('blockReason') or
                                              cand.get('finishReason') in ('SAFETY', 'PROHIBITED_CONTENT', 'BLOCKLIST')):
                    # refused by the safety filter: cached as blocked, excluded from that metric (not scored as 0)
                    out = {'blocked': resp.get('promptFeedback', {}).get('blockReason') or cand.get('finishReason')}
                else:
                    out = json.loads(''.join(p.get('text', '') for p in cand['content']['parts']))
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(out))
                return out
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                    raise RuntimeError(f'{e.code}: {e.read()[:300]!r}') from e
            except (urllib.error.URLError, OSError, http.client.HTTPException, KeyError, json.JSONDecodeError):
                if attempt == 5:
                    raise
            time.sleep(2 ** attempt * 5)


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / ((sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5)


def v2(prompt: str) -> str:
    """the v1 prompt with the reason instruction and a reason field (one per dimension for the narrative call)"""
    head, tail = prompt.split('Answer JSON: {{', 1)
    field = ('"reasons": {{' + ', '.join(f'"{k}": str' for k in DIMS) + '}}, ') if '"spatial_consistency"' in tail \
        else '"reason": str, '
    return head + REASON + 'Answer JSON: {{' + field + tail


def judge_video(g: Gemini, d: Path, reasons: bool = True):
    x = json.loads((d / 'inputs.json').read_text())
    FID_, PAIR_, ACTION_, NARR_ = (v2(t) if reasons else t for t in (FID, PAIR, ACTION, NARR))
    c, desc, subj, shots = d / ('gemini_v2' if reasons else 'gemini'), x['description'], x['subject'], x['shots']
    log = []
    ask = lambda parts, cache: log.append((cache.stem, g(parts, cache))) or log[-1][1]
    K = len(shots)
    phi = {}
    for s in shots:
        if (d / f'crop_{s["shot"]}.jpg').exists():
            r = ask([{'text': FID_.format(desc=desc)}, img(d / f'crop_{s["shot"]}.jpg')], c / f'fid_{s["shot"]}.json')
            if 'blocked' not in r:
                phi[s['shot']] = r['overall'] / 10
    pool = [s for s in shots if s['status'] == 'present' and phi.get(s['shot'], 0) >= TAU_FID]
    same = []
    if len(pool) >= 2:
        cen = [mean(v) for v in zip(*[s['emb'] for s in pool])]
        anchor = max(pool, key=lambda s: cos(s['emb'], cen))
        for s in pool:
            if s is not anchor:
                r = ask([{'text': PAIR_.format(desc=desc)}, img(d / f'crop_{anchor["shot"]}.jpg'),
                       img(d / f'crop_{s["shot"]}.jpg')], c / f'pair_{anchor["shot"]}_{s["shot"]}.json')
                if 'blocked' not in r:
                    same.append(float(bool(r['same'])))
    act = [ask([{'text': ACTION_.format(subj=subj, desc=desc, sentence=x['sentences'][s['shot']])},
              img(d / f'grid_{s["shot"]}.jpg')], c / f'action_{s["shot"]}.json') for s in shots]
    blocked = sum('blocked' in r for r in act)
    act = [a for a in act if 'blocked' not in a]
    parts = [{'text': NARR_.format(k=K, subj=subj, desc=desc,
                                  shots='\n'.join(f'Shot {j + 1}: {t}' for j, t in enumerate(x['sentences'])))}]
    for i, n in enumerate(x['narrative']):
        parts += [{'text': f'Frame {i + 1} (shot {n["shot"] + 1})'}, img(d / f'narr_{i:02d}.jpg')]
    nr = ask(parts, c / 'narrative.json')
    blocked += 'blocked' in nr
    dims = DIMS
    nan = float('nan')
    nd = {k: nan for k in dims} if 'blocked' in nr else {k: float(bool(nr[k])) for k in dims}
    return {'setting': x['setting'], 'id': x['id'], 'fidelity': mean(phi.values()) if phi else 0.0,
            'gate_pass': len(pool) / K, 'n_pairs': len(same), 'n_pairs_max': K - 1,
            'llm_id_raw': mean(same) if same else nan, 'llm_id': sum(same) / (K - 1),
            'action_depicted': mean(float(bool(a['depicted'])) for a in act) if act else nan,
            'action_overall': mean(a['overall'] / 10 for a in act) if act else nan,
            **nd, 'narrative': nan if 'blocked' in nr else mean(nd.values()), 'blocked_calls': blocked,
            'calls': [{'call': k, **v} for k, v in log]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', type=Path)
    ap.add_argument('--models', default='hunyuan,h3')
    ap.add_argument('--model', default='gemini-2.5-pro')
    ap.add_argument('--key-file', type=Path)
    ap.add_argument('--credentials', type=Path, help='Vertex AI service-account JSON')
    ap.add_argument('--location', default='us-central1', help='Vertex AI location')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='judge only the first N videos per model (smoke test)')
    ap.add_argument('--settings', default='', help='comma list; default: every setting with judge inputs')
    ap.add_argument('--v1', action='store_true', help='the v1 prompts (no reasons) and their cache')
    a = ap.parse_args()
    if a.credentials:
        g = Gemini(a.model, credentials=a.credentials, location=a.location)
    else:
        key = os.environ.get('GEMINI_API_KEY') or (a.key_file.read_text().strip() if a.key_file else '')
        assert key, 'set GEMINI_API_KEY, pass --key-file, or pass --credentials'
        g = Gemini(a.model, key)
    keys = ['fidelity', 'gate_pass', 'llm_id_raw', 'llm_id', 'action_depicted', 'action_overall', 'scene_consistency',
            'subject_consistency', 'action_coherence', 'spatial_consistency', 'narrative']
    for m in a.models.split(','):
        dirs = sorted(p.parent for p in (a.out / m).glob('*/*/inputs.json'))
        if a.settings:
            dirs = [d for d in dirs if d.parent.name in a.settings.split(',')]
        if a.limit:
            dirs = dirs[:a.limit]
        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(lambda d: judge_video(g, d, not a.v1), dirs))
        calls = [dict(setting=r['setting'], id=r['id'], **c) for r in rows for c in r.pop('calls')]
        lines = [f'model {m}, judge {a.model}', '', '| setting | n | ' + ' | '.join(keys) + ' |',
                 '|---|---|' + '---|' * len(keys)]
        for s in sorted({r['setting'] for r in rows}):
            rs = [r for r in rows if r['setting'] == s]
            # benchmark-level gate-corrected identity: raw mean over evaluated pairs x pooled coverage
            ev = sum(r['n_pairs'] for r in rs)
            raw = sum(r['llm_id_raw'] * r['n_pairs'] for r in rs if r['n_pairs']) / ev if ev else 0.0
            agg = {k: mean(r[k] for r in rs if r[k] == r[k]) if any(r[k] == r[k] for r in rs) else float('nan')
                   for k in keys}
            agg['llm_id_raw'], agg['llm_id'] = raw, raw * ev / sum(r['n_pairs_max'] for r in rs)
            lines.append(f'| {s} | {len(rs)} | ' + ' | '.join(f'{agg[k]:.3f}' for k in keys) + ' |')
        lines.append(f'\nGemini refused (safety filter) {sum(r["blocked_calls"] for r in rows)} calls; their items are '
                     'excluded from the metric they belong to.')
        print('\n'.join(lines), flush=True)
        tag = '' if a.v1 else '_v2'
        (a.out / m / f'gemini_rows{tag}.json').write_text(json.dumps(rows, indent=1))
        (a.out / m / f'gemini_summary{tag}.md').write_text('\n'.join(lines) + '\n')
        if not a.v1:
            (a.out / m / 'gemini_reasons.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in calls))


if __name__ == '__main__':
    main()
