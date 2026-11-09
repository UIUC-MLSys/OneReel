# OneReel

Prompt-structured scene attention and scene parallelism for multi-shot video DiTs.
Multi-GPU runs use xfuser's sequence-parallel group from `third_party/xDiT`.

## What it does

Each sentence of a multi-shot prompt is encoded on its own and owns one **scene**, which is a contiguous run of latent
frames (`SceneLayout`). The rules below apply at every layer and every step.

- **Video queries of scene j** attend to:
  - scene j's frames;
  - `halo` frames of each neighbouring scene (both neighbours, `prev` only, or `next` only);
  - sentence j's tokens.
- **Text queries of sentence j** attend to scene j's video and sentence j.
- **Subject links** are optional. At one step, a subject mask is built from the logits of the video tokens to the
  tokens of the subject phrase. From then on, scene j also reads the subject tokens of `n` key frames of every other
  scene.
- **Frame memory** (`frame_memory.py`, `--frame-memory M`) replaces the halo frames. After each step's conditional
  pass, the frames are clustered by k-means on one layer's mean-pooled hidden states, and the M medoid frames become
  the shared frames that every scene reads at the next step. Step 0 reads the halo frames.
- **Temporal positions** (HunyuanVideo-1.5 only):
  - `--scene-rope` restarts the temporal RoPE index at every scene. Scenes after the first start at 1, because
    index 0 is the causal VAE's single-image latent.
  - `--shared-pos` sets where a scene's shared frames sit:
    - `source`: at their own index;
    - `adjacent`: next to the scene; with `--scene-rope`, just after its last frame;
    - `switch`: `source` before step `--shared-switch`, then `adjacent`.
  - Moving a shared key re-rotates it by the position difference (`tests/test_shared_pos.py`).
  - Wan reads its text through cross-attention without RoPE, and keeps the global index.

| Model | Module | Attention | Multi-GPU | Subject links |
|---|---|---|---|---|
| HunyuanVideo-1.5 480p t2v | `models/hunyuan15.py` | joint [video \| text] varlen | K/V gathering / Ulysses | yes, on 1 GPU |
| Wan2.2-TI2V-5B, 704x1280 | `models/wan.py` | self-attention: one contiguous key range per scene; cross-attention: own sentence | K/V gathering / Ulysses | no |
| MiniMax-H3 t2va | `models/minimax_h3.py` | packed [text \| audio \| video] varlen | K/V gathering / Ulysses | yes |

### Attention backend

`onereel.attention.BACKEND` selects the backend. It is `fa2` by default, which reproduces the experiment scripts.

| Setting | Behaviour |
|---|---|
| `fa2` | FlashAttention-2 (the default) |
| `fa3` | FlashAttention-3 (`flash_attn_interface`, built from `third_party/flash-attention/hopper`) |
| `auto` | FA3 when it is available, otherwise FA2 |

With FA3, the Wan path reads each scene's overlapping key range in place, with no copy (`cu_seqlens_k` gives the
offsets and `seqused_k` the lengths).

### Parallel modes for HunyuanVideo-1.5 (torchrun)

**`scene`**
- The frames are cut into 2 contiguous ranges, balanced by per-token cost.
- Each rank receives only the K/V frames of the other rank that its scenes and halos read. This is a
  `batch_isend_irecv` on contiguous slices.
- Each rank computes the scenes whose keys are all local while the transfer runs.

**`ulysses`**
- Tokens are split equally.
- q, k and v go through an all-to-all to (all tokens, local heads), then the same scene attention runs, then an
  all-to-all brings the results back.

**Both modes**
- Text tokens are replicated on every rank, and their outputs are all-reduced.
- The output matches the 1-GPU result bit for bit.

## Usage

Use the onereel env (`data/envs/onereel`: torch 2.8, diffusers 0.41, FA2 + FA3, xfuser), and run
`source scripts/env.sh` first. For H3, unset `HF_HUB_OFFLINE`, because its sharded weights resolve through the hub.

```bash
python -m onereel.run --model hunyuan15 --bench B.json --id M6 --lengths 8,8,8,8,8,8 --out DIR
torchrun --nproc_per_node 2 -m onereel.run --model hunyuan15 --mode scene --bench B.json --id M6 --lengths 3,5,7,9,11,13 --out DIR
python -m onereel.run --model wan --bench B.json --id M6 --frames 81 --sides next --out DIR
python -m onereel.run --model h3 --bench B.json --id M6 --frames 124 --subject-links 2 --out DIR [--any-duration]
```

The quality settings on `dataset/multishot_bench.json` (480p / 704x1280, 20 steps, seed 0). The scene lengths come
from the bench row:

```bash
# HunyuanVideo-1.5, CFG 6: frame memory M = 4, scene-local RoPE, shared frames adjacent from step 5
python -m onereel.run --model hunyuan15 --bench dataset/multishot_bench.json --id holo_1 --frame-memory 4 \
  --scene-rope --shared-pos switch --shared-switch 5 --backend fa2 --out DIR
# Wan2.2: 2 halo frames of the next scene, CFG 5 with wan.NEGATIVE
python -m onereel.run --model wan --bench dataset/multishot_bench.json --id holo_1 --sides next --backend fa2 --out DIR
```

`--cfg` defaults to 6 for HunyuanVideo-1.5 and to 5 for Wan.

From Python:

```python
from onereel import SceneLayout
from onereel.models import hunyuan15 as M

ctx = M.Context(SceneLayout([8] * 6, halo=2))
pipe = M.load()
M.install(pipe, ctx)
video = M.generate(pipe, ctx, sentences, steps=20, guidance=6.0)
```

## Tests

The unit tests are in `tests/test_onereel_layout.py`. They cover the layout, the halo order, the partition and the
exchange sets.

## Limits

- Scene parallelism handles 2 ranks. Ulysses handles any rank count that divides the frame count.
- Subject links run on 1 GPU for HunyuanVideo-1.5.
- `xfuser.model_executor.models.transformers.transformer_minimax_h3` (dense H3 with Ulysses) stays in xDiT as the
  baseline. It is not wired into `onereel.run`.
