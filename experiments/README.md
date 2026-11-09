# Experiment drivers

The engine is `src/onereel` (`python -m onereel.run` generates one video with any of its settings). The scripts
below drive it in batches, time it, or hold variants and baselines that stay out of the package. Videos go to
`OUT/<model>/<setting>/<id>/{video.mp4, meta.json}`, the layout that `eval/event_order/` scores. Run them from the
repository root with `data/envs/onereel/bin/python`; each documents its arguments in its docstring and `--help`.

## Quality

| Script | Purpose |
|---|---|
| `frame_memory_smoke.py` | Batch generation of OneReel variants on the multishot bench: halo, local, full, frame memory, and its ablations (budget, boundary candidates, per-layer choice, sink). The quality runs use it: HunyuanVideo-1.5 `fm_all --budget 4 --scene-rope --shared-pos switch --shared-switch 5`, Wan `halo`. |
| `event_order_attention.py` | Dense, Separate and the first `joint_sep_*` scene-attention settings on HunyuanVideo-1.5 and Wan. `src/onereel` was taken from it. |
| `prompt_time_attention.py` | Wan helpers that `event_order_attention.py` and `fig1_latency.py` import (RoPE, sentence token ranges, the Wan negative prompt). |
| `sparse_baselines.py` | Training-free sparse-attention baselines (Radial, SVG, Sol-Attn) on HunyuanVideo-1.5 and Wan. |
| `intra_scene_sparse.py` | OneReel with Sol-Attn inside each scene. It needs the kernel from `third_party/Sana`. |
| `holocine_gen.py`, `holocine_parallel.py` | HoloCine on the multishot bench, on one GPU and on N GPUs (`stubs/` holds a modelscope stub it imports). |
| `minimax_h3_scene.py` | MiniMax-H3 scene attention (with `minimax_h3_two_gpu.py`; its optional probes import `block_attention_stats.py`, `tail_compression.py` and `official_warmup/sol_h3_profile.py`). |
| `minute_gen.py` | One-minute requests on N GPUs, dense and OneReel. |

## Latency

| Script | Purpose |
|---|---|
| `fig1_latency.py` | Figure 1: end-to-end latency of every method on one multi-GPU instance. Its `Runner` is shared by `minute_gen.py`, `e2e_breakdown.py` and `scene_count_latency.py`. |
| `e2e_breakdown.py` | Time and communication breakdown of one 12-scene request, and the system ablations. |
| `scene_count_latency.py` | Latency against the number of scenes at a fixed video length. |

## Kernels

| Script | Purpose |
|---|---|
| `kernel_breakdown.py` | Scene attention per kernel on the scopes it runs, against FlashInfer block-sparse (the packing-overhead appendix). |

The CLIP Text vs identity appendix is computed by `analysis/text_identity.py`.
