"""CogVideoX1.5 OneReel port, one transformer forward on random inputs (480 x 848, CFG batch of 2):
  1. one scene, no halo == the original diffusers attention (every rank, before the OneReel processors are installed)
  2. two scenes of 4 token frames with 1 halo frame: K/V gathering ('scene') == Ulysses == the single-rank result
Run: torchrun --nproc_per_node 2 tests/dist_onereel_cogvideox.py   (world 1 also works: checks 1 and the 1-rank path)
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from onereel.layout import SceneLayout  # noqa: E402
from onereel.models import cogvideox as C  # noqa: E402
from onereel.parallel import dist as D  # noqa: E402

H_, W_ = 480, 848


def forward(pipe, x, e, t, rot):
    with torch.no_grad():
        return pipe.transformer(hidden_states=x, encoder_hidden_states=e, timestep=t, image_rotary_emb=rot,
                                return_dict=False)[0]


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    D.init()
    dev = f'cuda:{torch.cuda.current_device()}'                      # set by D.init from LOCAL_RANK
    pipe = C.load(dev)
    g = torch.Generator(device='cpu').manual_seed(0)
    lat = 16                                                       # latent frames = 8 token frames
    x = torch.randn(2, lat, 16, H_ // 8, W_ // 8, generator=g).to(dev, torch.bfloat16)
    t = torch.tensor([500, 500], device=dev)
    rot = pipe._prepare_rotary_positional_embeddings(H_, W_, lat, torch.device(dev))
    rank0 = D.rank() == 0

    # 1. one scene == original attention
    e1 = (torch.randn(2, C.TEXT_LEN, 4096, generator=g) * 0.1).to(dev, torch.bfloat16)
    ref = forward(pipe, x, e1, t, rot)
    ctx = C.Context(SceneLayout([8], halo=0), height=H_, width=W_)
    C.install(pipe, ctx)
    ctx.T, ctx.counts = (H_ // 16) * (W_ // 16), {True: [C.TEXT_LEN], False: [C.TEXT_LEN]}
    ctx.mode = 'scene'
    C.H._plan_parallel(ctx, 96)
    out = forward(pipe, x, e1, t, rot)
    if rank0:
        print(f'[1] one scene vs original: rel err {rel(out, ref):.2e} (path {ctx.run_mode})', flush=True)

    # 2. two scenes, halo 1: scene == ulysses (== world-1 run of the same script)
    e2 = (torch.randn(2, 2 * C.TEXT_LEN, 4096, generator=g) * 0.1).to(dev, torch.bfloat16)
    res = {}
    for mode in ('scene', 'ulysses') if D.world() > 1 else ('scene',):
        ctx.layout, ctx.mode = SceneLayout([4, 4], halo=1), mode
        ctx.counts = {True: [C.TEXT_LEN] * 2, False: [C.TEXT_LEN] * 2}
        C.H._plan_parallel(ctx, 96)
        res[mode] = forward(pipe, x, e2, t, rot)
        if rank0:
            print(f'[2] {mode}: own {ctx.own} path {ctx.path}', flush=True)
    if rank0:
        (ROOT / 'outputs/tests').mkdir(parents=True, exist_ok=True)
        torch.save(res['scene'].cpu(), ROOT / f'outputs/tests/test_two_scenes_w{D.world()}.pt')
        if 'ulysses' in res:
            print(f'[2] scene vs ulysses: rel err {rel(res["scene"], res["ulysses"]):.2e}', flush=True)
        other = ROOT / f'outputs/tests/test_two_scenes_w{3 - D.world()}.pt'
        if other.exists():
            print(f'[2] world {D.world()} vs world {3 - D.world()}: rel err {rel(res["scene"].cpu(), torch.load(other)):.2e}', flush=True)


if __name__ == '__main__':
    main()
