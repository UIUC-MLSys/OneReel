"""Where the time of one 12-scene request goes, and what crosses the GPUs (one multi-GPU instance, USP degree = world).

Methods (experiments/fig1_latency.py's Runner): dense (Ulysses, full attention), onereel (scene attention, path by
Eq. 2: K/V gathering), onereel_ulysses and onereel_ring (the same scene attention on USP's Ulysses, or on Ring with
every scene split evenly over the ranks: the USP baselines of the communication study), separate (12 scene requests,
dense each; scene 0 is measured and scaled by 12, the scenes being equally long).

Per method, after two untimed one-step passes (compilation, allocator):
  request   one request of STEPS denoising steps: text encoding, every transformer forward and the VAE decode are timed
            (CUDA-synchronised); denoising is scaled from STEPS to 20 steps (steps after the first are alike)
  kernels   one conditional forward of a later step under torch.profiler, kernels split into attention (FlashAttention),
            GEMM, communication (NCCL) and other; exposed communication = NCCL kernel time not overlapped by a compute
            kernel; idle = forward wall time - union of kernel time
  bytes     bytes each rank sends per forward, counted at the torch.distributed calls (all-to-all, point-to-point,
            all-gather, all-reduce)
  memory    frame memory (Hunyuan OneReel): pooling and per-step selection (all-reduce + k-means), CUDA-synchronised
Writes OUT/<model>_<res>[_<tag>].json (rank 0, after every run).
--layouts / --partitions (system ablations; pl<long>_<short>_<k>_<i>: one long scene at index i): every method once per scene layout and K/V gathering partition, as
<layout>/<partition>/<method>, with per-rank kernel times (kernels_ranks) and bytes (bytes_ranks).
--fm-budgets 0,1,2,4,8 (memory-budget ablation): OneReel only, once per frame-memory budget M, as method
onereel_m{M}; M = 0 reads only the scene's own frames (no halo, no memory). Also 'halo' (the model's halo frames,
no memory) and 'full' (every scene reads every frame of every other scene) as onereel_halo / onereel_full.

--fast (request-property sweeps): no CFG (one forward per step, the same for every method), no warm-up passes, no
profile; one 3-step request per run, step_s = the faster of steps 1 and 2 (step 0 meets every shape first; frame memory
is read from step 1 on). Layouts dur<s>: an s-second request (6 latent frames per second at 24 fps) of shots of 7 latent
frames on average (the bench's real requests), shot lengths lq-spread (sigma 0.5); ks<k>[_<f>]: k shots over 84 (or f)
latent frames, lq-spread (sigma 0.5).

Usage: torchrun --nproc_per_node 2 experiments/e2e_breakdown.py --model hunyuan --res 720 --out DIR [--tag nop2p]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import fig1_latency  # noqa: E402
from fig1_latency import BENCH, D, Runner  # noqa: E402
from onereel.layout import SceneLayout  # noqa: E402

STEPS = 3


LAYOUTS = {'equal': [7] * 12, 'uneven': [4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5], 'skew': [2] * 6 + [12] * 6,
           'one_long': [4] * 9 + [48]}                                 # 84 latent frames each


def lognormal_lengths(sigma, n=12, total=84, seed=0):
    """n scene lengths summing to total, proportional to exp(sigma * z) with z ~ N(0, 1) drawn once (seed): the same
    scene order for every sigma, so sigma alone sets the skew (0 = equal lengths); at least one frame each, rounded
    by largest remainder"""
    z = np.random.default_rng(seed).standard_normal(n)
    w = np.exp(sigma * z)
    x = 1 + w / w.sum() * (total - n)
    lengths = np.floor(x).astype(int)
    for i in np.argsort(-(x - lengths))[:total - lengths.sum()]:
        lengths[i] += 1
    return [int(v) for v in lengths]


def quantile_lengths(sigma, n, total, seed=0):
    """as lognormal_lengths, but z takes the evenly spaced normal quantiles in the rank order of the seeded draw, so a
    few scenes still span the distribution (the longest / shortest ratio is set by sigma, not by the luck of the draw)"""
    from statistics import NormalDist
    rank = np.argsort(np.argsort(np.random.default_rng(seed).standard_normal(n)))
    z = np.array([NormalDist().inv_cdf((r + 0.5) / n) for r in rank])
    w = np.exp(sigma * z)
    x = 1 + w / w.sum() * (total - n)
    lengths = np.floor(x).astype(int)
    for i in np.argsort(-(x - lengths))[:total - lengths.sum()]:
        lengths[i] += 1
    return [int(v) for v in lengths]


def layout_row(row, name):
    """the T_12 request with the named scene lengths (its first sentences, cycled past 12 scenes); K<k>: k equal (84
    frames, the remainder on the first scenes); ln<sigma>: 12 lognormal scene lengths (lognormal_lengths); bench:<json>#<id>: a request of another bench
    file as it is"""
    if name.startswith('K'):                                       # 84 frames, the remainder on the first scenes
        k = int(name[1:])
        lengths = [84 // k + (i < 84 % k) for i in range(k)]
    elif name.startswith('ln'):
        lengths = lognormal_lengths(float(name[2:]))
    elif name.startswith('ol'):                                      # ol<long>_<short>_<scenes>: one long scene last
        long, short, k = map(int, name[2:].split('_'))
        lengths = [short] * (k - 1) + [long]
    elif name.startswith('pl'):                                      # pl<long>_<short>_<scenes>_<i>: one long scene, index i
        long, short, k, i = map(int, name[2:].split('_'))
        lengths = [short] * (k - 1)
        lengths.insert(i, long)
    elif name.startswith('bench:'):                                 # bench:<json under the repo>#<id>: that request
        path, pid = name[6:].split('#')
        r = {x['id']: x for x in json.loads((fig1_latency.ROOT / path).read_text())['prompts']}[pid]
        return dict(r, id=pid)
    elif name.startswith('dur'):                                     # dur<seconds>: shots of 7 frames on average
        total = 6 * int(name[3:])
        lengths = quantile_lengths(0.5, max(1, round(total / 7)), total)
    elif name.startswith('ks'):                                      # ks<k>[_<frames>]: k shots over 84 (or frames)
        k, _, tot = name[2:].partition('_')
        lengths = quantile_lengths(0.5, int(k), int(tot or 84))
    elif name.startswith('lq'):                                      # lq<sigma>_<scenes>_<frames>
        sg, n, tot = name[2:].split('_')
        lengths = quantile_lengths(float(sg), int(n), int(tot))
    else:
        lengths = LAYOUTS[name]
    sentences = [row['sentences'][i % len(row['sentences'])] for i in range(len(lengths))]   # reused past 12 scenes
    return dict(row, id=name, sentences=sentences, lengths=lengths)


class FullLayout(SceneLayout):
    """every scene also reads every frame of every other scene (full video attention, per-scene text)"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.extra = {j: [f for f in range(self.n_frames) if f not in self.frames[j]] for j in range(self.n_scenes)}
SENT = defaultdict(float)                # bytes this rank sent, by call


def _count(name, nbytes):
    SENT[name] += nbytes


def install_counters():
    """wrap the torch.distributed calls the engine makes; the wrappers only count bytes sent"""
    a2a, bisr, ag, ar = dist.all_to_all_single, dist.batch_isend_irecv, dist.all_gather, dist.all_reduce

    def all_to_all_single(out, inp, *a, **k):
        n = dist.get_world_size(k.get('group'))
        _count('all_to_all', inp.numel() * inp.element_size() * (n - 1) / n)
        return a2a(out, inp, *a, **k)

    def batch_isend_irecv(ops):
        for op in ops:
            if op.op in (dist.isend,):
                _count('p2p', op.tensor.numel() * op.tensor.element_size())
        return bisr(ops)

    def all_gather(lst, t, *a, **k):
        _count('all_gather', t.numel() * t.element_size() * (len(lst) - 1))
        return ag(lst, t, *a, **k)

    def all_reduce(t, *a, **k):
        n = dist.get_world_size(k.get('group'))
        _count('all_reduce', 2 * t.numel() * t.element_size() * (n - 1) / n)
        return ar(t, *a, **k)
    dist.all_to_all_single, dist.batch_isend_irecv, dist.all_gather, dist.all_reduce = \
        all_to_all_single, batch_isend_irecv, all_gather, all_reduce


def sync():
    torch.cuda.synchronize()


class Timers:
    """CUDA-synchronised wall times of the transformer forwards, the VAE decode and the frame-memory calls"""

    def __init__(self, R):
        self.fwd, self.vae, self.fm, self.R = [], [], [], R
        tr = R.pipe.transformer
        t0 = {}

        def pre(m, args, kw):
            sync()
            t0['f'] = time.perf_counter()

        def post(m, args, kw, out):
            sync()
            self.fwd.append(time.perf_counter() - t0['f'])
        tr.register_forward_pre_hook(pre, with_kwargs=True)
        tr.register_forward_hook(post, with_kwargs=True)
        vae = R.pipe.vae
        dec = vae.decode

        def decode(*a, **k):
            sync()
            t = time.perf_counter()
            out = dec(*a, **k)
            sync()
            self.vae.append(time.perf_counter() - t)
            return out
        vae.decode = decode
        from onereel import frame_memory as FM
        for name in ('collect', 'finish'):
            orig = getattr(FM.FrameMemory, name)

            def wrapped(self_, *a, _orig=orig, **k):
                sync()
                t = time.perf_counter()
                out = _orig(self_, *a, **k)
                sync()
                self.fm.append(time.perf_counter() - t)
                return out
            setattr(FM.FrameMemory, name, wrapped)

    def reset(self):
        self.fwd, self.vae, self.fm = [], [], []


ATTN = ('flash', 'fmha', 'attention')
GEMM = ('gemm', 'cutlass', 'xmma', 'nvjet', 'cublas', 'sm90_', 'sm80_', 'ampere_', 'Kernel2')


def kind(name):
    n = name.lower()
    if 'nccl' in n:
        return 'comm'
    if any(a in n for a in ATTN):
        return 'attention'
    if any(g.lower() in n for g in GEMM):
        return 'gemm'
    return 'other'


def union(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def length(iv):
    return sum(b - a for a, b in iv)


def overlap(a_iv, b_iv):
    """total length of a_iv covered by b_iv (both unions)"""
    i = j = 0
    s = 0.0
    while i < len(a_iv) and j < len(b_iv):
        lo, hi = max(a_iv[i][0], b_iv[j][0]), min(a_iv[i][1], b_iv[j][1])
        s += max(0.0, hi - lo)
        if a_iv[i][1] < b_iv[j][1]:
            i += 1
        else:
            j += 1
    return s


def profile_forward(R, method, row, scene, target, steps=None):
    """profile the target-th transformer forward of a short request (steps: its length, default enough for target
    at 2 forwards per step plus one); returns the kernel breakdown in ms"""
    from torch.profiler import ProfilerActivity, profile
    tr = R.pipe.transformer
    count, state = [0], {}

    def pre(m, args, kw):
        count[0] += 1
        if count[0] == target:
            sync()
            state['p'] = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])
            state['p'].__enter__()
            state['t'] = time.perf_counter()

    def post(m, args, kw, out):
        if count[0] == target:
            sync()
            state['wall'] = (time.perf_counter() - state['t']) * 1e3
            state['p'].__exit__(None, None, None)
    h1 = tr.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = tr.register_forward_hook(post, with_kwargs=True)
    try:
        R.run(method, row, scene=scene, steps=steps or max(2, (target + 1) // 2 + 1), latent=True)
    finally:
        h1.remove()
        h2.remove()
    ev = [e for e in state['p'].events() if e.device_type.name == 'CUDA' and e.time_range.end > e.time_range.start]
    by = defaultdict(list)
    for e in ev:
        by[kind(e.name)].append((e.time_range.start, e.time_range.end))
    total = {k: sum(b - a for a, b in v) / 1e3 for k, v in by.items()}
    comp = union([iv for k in ('attention', 'gemm', 'other') for iv in by[k]])
    comm = union(by['comm'])
    busy = union([iv for v in by.values() for iv in v])
    return dict(wall_ms=state['wall'], attention_ms=total.get('attention', 0.0), gemm_ms=total.get('gemm', 0.0),
                other_ms=total.get('other', 0.0), comm_ms=length(comm) / 1e3,
                comm_exposed_ms=(length(comm) - overlap(comm, comp)) / 1e3,
                idle_ms=state['wall'] - length(busy) / 1e3)


NO_VAE = False                  # --no-vae: requests stop at the latents (vae_s = 0; the decode is a fixed cost per length)


def measure(R, timers, method, row, scene=None):
    """one request of STEPS steps, timed by part; text encoding = request - forwards - VAE"""
    timers.reset()
    SENT.clear()
    sync()
    D.barrier()
    t = time.perf_counter()
    R.run(method, row, scene=scene, steps=STEPS, latent=NO_VAE)
    sync()
    D.barrier()
    total = time.perf_counter() - t
    per_step = len(timers.fwd) // STEPS
    steps_s = [sum(timers.fwd[i * per_step:(i + 1) * per_step]) for i in range(STEPS)]
    steady = timers.fwd[per_step:]                        # steps after the first
    step_s = sum(steady) / (STEPS - 1) / 1.0
    fwd_s, vae_s, fm_s = sum(timers.fwd), sum(timers.vae), sum(timers.fm)
    return dict(forwards_per_step=per_step, step_s=step_s, steps_s=steps_s, denoise_20_s=sum(timers.fwd[:per_step]) + 19 * step_s,
                text_and_rest_s=total - fwd_s - vae_s, vae_s=vae_s, frame_memory_per_step_ms=fm_s / STEPS * 1e3,
                bytes_per_forward={k: v / len(timers.fwd) for k, v in SENT.items()}, path=dict(R.ctx.path or {}),
                run_mode=getattr(R.ctx, 'run_mode', None))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=['hunyuan', 'wan'], required=True)
    ap.add_argument('--res', type=int, choices=[480, 720], default=720)
    ap.add_argument('--methods', default='dense,onereel,onereel_ulysses,onereel_ring,separate')
    ap.add_argument('--tag', default='')
    ap.add_argument('--fm-budgets', default='')
    ap.add_argument('--layouts', default='', help='scene layouts (LAYOUTS, K<k>: k equal scenes, ln<sigma>: lognormal)')
    ap.add_argument('--partitions', default='balanced', help='K/V gathering partitions: balanced,tokens,scenes')
    ap.add_argument('--no-vae', action='store_true', help='stop each request at the latents (no VAE decode)')
    ap.add_argument('--quick', action='store_true', help='one warm-up pass, 2 timed steps, step 1 profiled in a '
                    '2-step request (long sweeps; the default is 2 warm-up passes, 3 timed steps, step 2 profiled)')
    ap.add_argument('--fast', action='store_true', help='no CFG, no warm-up, no profile: one 3-step request per run, '
                    'step_s = min(step 1, step 2)')
    ap.add_argument('--fast-steps', type=int, default=3, help='--fast: steps per request (step_s = the fastest after step 0; '
                    'more steps when kernels compile per shape, e.g. Sol-Attn per sequence length)')
    ap.add_argument('--overlap', default=None, choices=['none', 'lse'], help='K/V gathering: overlap of the exchange '
                    'with the scenes that read remote keys (default: the model Context\'s)')
    ap.add_argument('--warm-steps', type=int, default=0, help='--fast: one untimed request of this many steps before each '
                    'timed one (kernels compiled per shape, e.g. Sol-Attn per sequence length; frame memory is '
                    'deterministic, so the timed request meets the same shapes)')
    ap.add_argument('--kv-chunks', type=int, default=None, help='K/V gathering: head chunks of the pipelined '
                    'exchange (default: the model Context\'s; 1 = one exchange, waited for before the non-local scenes)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    global NO_VAE, STEPS
    NO_VAE = a.no_vae
    if a.quick:
        STEPS = 2
    if a.fast:
        STEPS = a.fast_steps
        fig1_latency.CFG[a.model] = 1.0
    D.init()
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = torch._dynamo.config.recompile_limit = 64
    install_counters()
    bench = {r['id']: r for r in json.loads(BENCH.read_text())['prompts']}
    base = bench['T_12']
    R = Runner(a.model, a.res)
    if a.kv_chunks:
        R.ctx.kv_chunks = a.kv_chunks
    if a.overlap:
        R.ctx.overlap = a.overlap
    overlap0 = getattr(R.ctx, 'overlap', None)
    timers = Timers(R)
    res = {}
    runs = [(m, None) for m in a.methods.split(',')]
    if a.fm_budgets:
        runs = [('onereel', b if b in ('halo', 'full') else int(b)) for b in a.fm_budgets.split(',')]
        halo0 = fig1_latency.HALO[a.model]
    layouts = a.layouts.split(',') if a.layouts else [None]
    partitions = a.partitions.split(',')
    for lay in layouts:
        row = base if lay is None else layout_row(base, lay)
        for part in partitions:
            fig1_latency.PARTITION = part
            for method, budget in runs:
                if budget is not None:                                              # frame-memory budget M
                    fig1_latency.SceneLayout = SceneLayout
                    if budget in (0, 'halo', 'full'):
                        fig1_latency.FRAME_MEMORY.pop(a.model, None)
                        fig1_latency.HALO[a.model] = halo0 if budget == 'halo' else (0, 'adj')
                        if budget == 'full':
                            fig1_latency.SceneLayout = FullLayout
                    else:
                        fig1_latency.FRAME_MEMORY[a.model] = budget
                        fig1_latency.HALO[a.model] = halo0
                key = method if budget is None else f'onereel_{budget}' if isinstance(budget, str) else f'onereel_m{budget}'
                if a.layouts or a.partitions != 'balanced':
                    key = f'{lay}/{part}/{key}'
                method, _, opt = method.partition('+')         # onereel+lse: that overlap; onereel_ulysses+sol:
                R.configure(method)                             # intra-scene Sol-Attn (experiments/intra_scene_sparse)
                if overlap0 is not None:
                    R.ctx.overlap = overlap0 if opt == 'sol' else opt or overlap0
                if opt == 'sol':
                    from intra_scene_sparse import IntraSceneSol
                    assert method == 'onereel_ulysses', 'intra-scene Sol-Attn runs on the Ulysses path'
                    R.ctx.attn_override = IntraSceneSol(R.ctx)
                scene = 0 if method == 'separate' else None
                for _ in range(0 if a.fast else 1 if a.quick else 2):
                    R.run(method, row, scene=scene, steps=1, latent=True)             # compile, allocator
                if a.fast and a.warm_steps:
                    R.run(method, row, scene=scene, steps=a.warm_steps, latent=True)
                m = measure(R, timers, method, row, scene)
                if a.fast:
                    m['step_s'] = min(m['steps_s'][1:])
                    m['kernels'] = {}
                elif a.quick:                                                            # step 1's first forward
                    m['kernels'] = profile_forward(R, method, row, scene, m['forwards_per_step'] + 1, steps=2)
                else:
                    target = 2 * m['forwards_per_step'] + 1                            # step 2's first forward
                    m['kernels'] = profile_forward(R, method, row, scene, target)
                m['kernels_ranks'] = D.all_gather_object(m['kernels'])               # per rank: load balance
                m['bytes_ranks'] = D.all_gather_object(m['bytes_per_forward'])
                m['own'] = [list(x) for x in R.ctx.own]
                m['kv_chunks'] = R.ctx.kv_chunks
                m['overlap'] = getattr(R.ctx, 'overlap', None)
                m['lengths'] = row.get('lengths')
                if method == 'separate':                                             # equal scene requests
                    for k in ('denoise_20_s', 'text_and_rest_s', 'vae_s'):
                        m[k] *= len(row['sentences'])
                    m['scale'] = len(row['sentences'])
                m['e2e_est_s'] = m['denoise_20_s'] + m['text_and_rest_s'] + m['vae_s']
                m['fm_budget'] = budget
                res[key] = m
                if D.rank() == 0:
                    print(f'[breakdown] {a.model} {a.res} {key}: {json.dumps(m)}', flush=True)
                    a.out.mkdir(parents=True, exist_ok=True)                         # written after every run
                    name = f'{a.model}_{a.res}' + (f'_{a.tag}' if a.tag else '') + '.json'
                    (a.out / name).write_text(json.dumps(dict(model=a.model, res=a.res, world=D.world(), steps=STEPS,
                                                              methods=res), indent=1))

if __name__ == '__main__':
    main()
