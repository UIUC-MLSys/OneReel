"""HoloCine (Wan2.2-T2V-A14B fine-tuned for multi-shot video, sparse inter-shot attention) on the multishot bench, one
GPU per process, the official pipeline (third_party/HoloCine) with every model resident on the GPU (no CPU offload).

Videos: OUT/holocine/<variant>/<id>/{video.mp4, meta.json} (the layout eval/event_order/judge_inputs.py reads).
Input in HoloCine's structured format: the global caption introduces every entity as [characterK] (its bench
description); the shot captions are the bench sentences with each entity description replaced by its tag (HoloCine's
native style, and it keeps the prompt inside the T5 window of 512 tokens; meta records the token count and whether it
was truncated). Frames and shot cuts are the bench's pixel segments (4t + 1). Sampling as
HoloCine_inference_sparse_attention.py: 480x832, 50 steps, CFG 5, its negative prompt, seed 0, 15 fps.
Usage: CUDA_VISIBLE_DEVICES=0 python experiments/holocine_gen.py --ids holo_1,holo_2 --out OUT [--dry-run]
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HC = ROOT / 'third_party/HoloCine'
CKPT = ROOT / 'data/models/HoloCine'
TOKENIZER = ROOT / 'data/models/Wan2.1-T2V-1.3B/google/umt5-xxl'
NEG = ('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，'
       '画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走')
TEXT_LEN = 512


def holocine_prompt(r, short=False):
    """[global caption] ... This scene contains N shots. [per shot caption] s1 [shot cut] s2 ...; short: introduce each
    entity by the first sentence of its description only"""
    shots = list(r['sentences'])
    ents = r.get('entities') or []
    intro = []
    for k, e in enumerate(ents, 1):
        tag, desc = f'[character{k}]', e['description'].strip().rstrip('.')
        bare = re.sub(r'^(a|an|the)\s+', '', desc, flags=re.I)
        d = r'(?:(?:a|an|the)\s+)?' + re.escape(bare) + r'\.?'
        name = re.escape(e['name'])
        for pat in (rf'\b{name}\s*\({d}\)', rf'\b{d}'):          # "David (a lively boy ...)", then the bare description
            shots = [re.sub(pat, tag, s, flags=re.I) for s in shots]
        who = desc[0].lower() + desc[1:] if re.match(r'(a|an|the)\s', desc, re.I) else desc
        if short:
            who = re.split(r'(?<=[.!?])\s', who + '.')[0].rstrip('.')
        named = e['name'] if e['name'][:1].isupper() else None    # bench names like 'old' / 'young' are role keys
        intro.append(f'{tag} is {named + ", " if named else ""}{who}.')
    glob = ' '.join(intro + [f'This scene contains {len(shots)} shots.'])
    return f"[global caption] {glob} [per shot caption] {' [shot cut] '.join(s.strip() for s in shots)}"


def fit_prompt(r, tok):
    """the HoloCine prompt, with the short entity introductions when the full one exceeds the T5 window; returns
    (prompt, tokens)"""
    p = holocine_prompt(r)
    n = len(tok(p).input_ids)
    if n > TEXT_LEN:
        p = holocine_prompt(r, short=True)
        n = len(tok(p).input_ids)
    return p, n


def transformers_compat():
    """diffsynth (written for transformers 4.x) imports PretrainedConfig from transformers.modeling_utils, which
    transformers 5 no longer re-exports"""
    import transformers
    import transformers.modeling_utils as mu
    if not hasattr(mu, 'PretrainedConfig'):
        mu.PretrainedConfig = transformers.PretrainedConfig


def shot_cuts(r):
    seg = r['segments'][str(r['frames'])]['per_sentence']
    return [s['pixel'][0] for s in seg[1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variant', default='holocine_sparse', choices=('holocine_sparse', 'holocine_full'))
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--dry-run', action='store_true', help='print prompts and token counts only')
    a = ap.parse_args()
    bench = {r['id']: r for r in json.load(open(ROOT / 'dataset/multishot_bench.json'))['prompts']}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    if a.dry_run:
        for pid in a.ids.split(','):
            p, n = fit_prompt(bench[pid], tok)
            print(f'{pid}: {n} tokens{" TRUNCATED" if n > TEXT_LEN else ""}, frames {bench[pid]["frames"]}, '
                  f'cuts {shot_cuts(bench[pid])}\n  {p[:400]}')
        return

    sys.path[:0] = [str(ROOT / 'experiments/stubs'), str(HC), str(ROOT / 'data/envs/holocine_extra')]
    import torch
    transformers_compat()
    from diffsynth import save_video
    from diffsynth.pipelines.wan_video_holocine import WanVideoHoloCinePipeline, ModelConfig
    kind = a.variant.split('_')[1]
    pipe = WanVideoHoloCinePipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[ModelConfig(path=str(CKPT / 'Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth')),
                       ModelConfig(path=str(CKPT / f'HoloCine_dit/{kind}/{kind}_high_noise.safetensors')),
                       ModelConfig(path=str(CKPT / f'HoloCine_dit/{kind}/{kind}_low_noise.safetensors')),
                       ModelConfig(path=str(CKPT / 'Wan2.2-T2V-A14B/Wan2.1_VAE.pth'))],
        tokenizer_config=ModelConfig(path=str(TOKENIZER)))
    pipe.dit.use_sparse_self_attn = pipe.dit2.use_sparse_self_attn = kind == 'sparse'
    pipe.to('cuda')
    for pid in a.ids.split(','):
        d = a.out / 'holocine' / a.variant / pid
        if (d / 'video.mp4').exists():
            continue
        r = bench[pid]
        prompt, n_tok = fit_prompt(r, tok)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        video = pipe(prompt=prompt, negative_prompt=NEG, seed=0, tiled=True, height=480, width=832,
                     num_inference_steps=a.steps, num_frames=r['frames'], shot_cut_frames=shot_cuts(r))
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        d.mkdir(parents=True, exist_ok=True)
        save_video(video, str(d / 'video.mp4'), fps=15, quality=5)
        meta = dict(model='holocine', setting=a.variant, id=pid, seed=0, frames=len(video), seconds=sec, steps=a.steps,
                    lengths=r['lengths'], shot_cuts=shot_cuts(r), prompt=prompt, prompt_tokens=n_tok,
                    truncated=n_tok > TEXT_LEN, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
        (d / 'meta.json').write_text(json.dumps(meta, indent=1))
        print(f'[done] holocine {a.variant}/{pid} {sec:.0f}s frames {len(video)} tokens {n_tok}', flush=True)


if __name__ == '__main__':
    main()
