"""MiniMax-H3 t2va on two GPUs (conditioner on cuda:1, the rest on cuda:0), timed per phase.

Follows the two-card recipe of the diffusers docs: nothing is offloaded once resident. Each prompt runs --repeat
times; the reported times are from the last run. Phases: text encoding (the Qwen3-VL conditioner call), denoising
(sum of transformer forwards, CUDA events) and the rest of the second half (latent preparation, VAE decodes).
Saves OUT/<setting>/<id>/{video.mp4, audio.wav, meta.json} (the layout eval/event_order expects) and OUT/timing.json.
A 2-step warm-up run first moves every weight onto its device so the first prompt's timing is not inflated.

Usage: data/envs/h3/bin/python experiments/minimax_h3_two_gpu.py --out DIR [--ids L1_CAB] [--height 544 --width 960]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

BENCH = Path(__file__).resolve().parents[1] / 'dataset' / 'event_order_bench.json'
REPO = 'MiniMaxAI/MiniMax-H3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ids', default='L1_CAB')
    ap.add_argument('--height', type=int, default=544)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--frames', type=int, default=124)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--setting', default='baseline')
    ap.add_argument('--warmup', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.utils import export_to_video
    bench = {r['id']: r for r in json.loads(BENCH.read_text())['prompts']}
    t0 = time.perf_counter()
    workflow = ModularPipeline.from_pretrained(REPO).blocks.get_workflow('t2va')
    text_manager = ComponentsManager()
    text_manager.enable_auto_cpu_offload(device='cuda:1')
    conditioner = workflow.sub_blocks.pop('text_encoder').init_pipeline(REPO, components_manager=text_manager)
    conditioner.load_components(dtype=torch.bfloat16)
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device='cuda:0')
    rest = workflow.init_pipeline(REPO, components_manager=manager)
    rest.load_components(dtype=torch.bfloat16)
    load_s = time.perf_counter() - t0
    fwd = []

    def pre(m, *_):
        torch.cuda.synchronize('cuda:0')
        m._t0 = time.perf_counter()

    def post(m, *_):
        torch.cuda.synchronize('cuda:0')
        fwd.append(time.perf_counter() - m._t0)
    rest.transformer.register_forward_pre_hook(pre)
    rest.transformer.register_forward_hook(post)
    if a.warmup:
        state = conditioner(prompt='A warm-up clip of a quiet beach.')
        rest(state=state, height=a.height, width=a.width, num_frames=a.frames, num_inference_steps=2,
             generator=torch.Generator().manual_seed(0), output=['videos'])
        print('[warmup] done', flush=True)
    res = {'load_s': load_s, 'height': a.height, 'width': a.width, 'frames': a.frames, 'runs': []}
    for pid in a.ids.split(','):
        prompt = bench[pid]['prompt']
        for rep in range(a.repeat):
            fwd.clear()
            for d in ('cuda:0', 'cuda:1'):
                torch.cuda.synchronize(d)
            t0 = time.perf_counter()
            state = conditioner(prompt=prompt)
            torch.cuda.synchronize('cuda:1')
            t1 = time.perf_counter()
            out = rest(state=state, height=a.height, width=a.width, num_frames=a.frames,
                       generator=torch.Generator().manual_seed(a.seed), output=['videos', 'audio', 'sampling_rate'])
            torch.cuda.synchronize('cuda:0')
            t2 = time.perf_counter()
            run = {'id': pid, 'repeat': rep, 'text_s': t1 - t0, 'second_half_s': t2 - t1, 'denoise_s': sum(fwd),
                   'forwards': len(fwd), 'per_forward_s': float(np.median(fwd)) if fwd else None, 'total_s': t2 - t0}
            run['other_s'] = run['second_half_s'] - run['denoise_s']
            res['runs'].append(run)
            print(json.dumps(run), flush=True)
        video, audio, sr = out['videos'], out['audio'], out['sampling_rate']
        v = video[0] if isinstance(video, (list, tuple)) else video
        frames = list(v) if not torch.is_tensor(v) else list(v.float().cpu().numpy())
        d = a.out / a.setting / pid
        d.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(d / 'video.mp4'), fps=24)
        (d / 'meta.json').write_text(json.dumps({'model': 'h3', 'setting': a.setting, 'id': pid, 'seed': a.seed, 'frames': len(frames),
                                                 'height': a.height, 'width': a.width, 'seconds': run['total_s'], **run}, indent=1))
        try:
            import soundfile as sf
            x = audio[0] if isinstance(audio, (list, tuple)) else audio
            x = x.float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
            x = x.squeeze()
            sf.write(str(d / 'audio.wav'), x.T if x.ndim == 2 and x.shape[0] <= 2 else x, int(sr))
        except Exception as ex:
            print('[audio] not saved:', ex, flush=True)
    res['peak_mem_GB'] = {d: torch.cuda.max_memory_allocated(d) / 2 ** 30 for d in ('cuda:0', 'cuda:1')}
    (a.out / 'timing.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
