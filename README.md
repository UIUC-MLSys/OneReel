# OneReel

OneReel generates multi-scene videos with a pretrained video DiT, without training. Each scene keeps full attention over
its own frames and its own sentence, and scenes exchange information only through a small memory of representative
frames. Scene scopes are packed into dense variable-length attention, and multi-GPU runs balance attention work while
choosing between K/V gathering and Ulysses-style parallelism.

**OneReel: Efficient Multi-Scene Video Generation with Scene-Structured Attention**
<!-- TODO before release: paper and project page links -->
Wei Zhang\*, Yuchen Wang\*, Tianrui Feng, Hongrui Wu, Chenfeng Xu, Fan Lai (\* equal contribution)

Models: HunyuanVideo-1.5 480p t2v, Wan2.2-TI2V-5B, MiniMax-H3 t2va, CogVideoX1.5-5B. The engine is `src/onereel`;
[its README](src/onereel/README.md) describes the attention, the shared frames and the parallel paths.

## Install

Generation uses Python 3.10, PyTorch 2.8.0 (CUDA 12.8) and a GPU. `envs/generation_packages.txt` lists every package
version used for the paper.

```bash
python3.10 -m venv data/envs/onereel && source data/envs/onereel/bin/activate
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install flash_attn==2.8.3.post1 "git+https://github.com/huggingface/diffusers@e0415c41f8434f73177127f7bd7a3c96eb9d609f" \
  transformers==5.19.0 huggingface_hub==1.33.0 peft==0.21.2 torchao==0.15.0 tokenizers==0.23.2 soundfile \
  yunchang==0.6.4 distvae==0.1.0 cuda-python==12.9.10 nvidia-cutlass-dsl==4.8.1
pip install -e third_party/xDiT
pip install -e . --no-deps
```

FlashAttention 3, which the latency experiments use on Hopper GPUs, is built from the `hopper` directory of
[flash-attention](https://github.com/Dao-AILab/flash-attention) (`pip install --no-build-isolation third_party/flash-attention/hopper`).

Metrics use two conda environments: `onereel_metrics` (CLIP, DINOv2, GroundingDINO, Gemini judge) and `videoeval`
(VBench). `bash scripts/setup_env.sh` creates them and downloads the evaluation models. `source scripts/env.sh` sets the
paths and interpreters; override them in the git-ignored `env.local.sh`.

Weights are public on Hugging Face; `envs/model_revisions.txt` lists the snapshots used.

## Generate

```bash
source scripts/env.sh
P=data/envs/onereel/bin/python
# one request, HunyuanVideo-1.5 quality setting
$P -m onereel.run --model hunyuan15 --bench dataset/multishot_bench.json --id holo_1 --frame-memory 4 \
  --scene-rope --shared-pos switch --shared-switch 5 --out DIR
# Wan2.2: two frames of the next scene
$P -m onereel.run --model wan --bench dataset/multishot_bench.json --id holo_1 --sides next --out DIR
# two GPUs, Eq. 2 picks K/V gathering or Ulysses per request
$P -m torch.distributed.run --nproc_per_node 2 -m onereel.run --model hunyuan15 --mode auto ...
```

## Reproduce the paper

The paper uses 40 prompts, 20 steps and seed 0. HunyuanVideo-1.5 runs at 480p with CFG 6, and Wan2.2 at 704x1280 with
CFG 5. `dataset/multishot_bench.json` holds 35 of the prompts. The other five come from MSVBench, whose release states
no license, so they are not redistributed here. This command adds them from the official release, and the script
checks that the result is the bench used in the paper:

```bash
python dataset/add_msvbench.py
```

| Result | Script |
|---|---|
| Quality: OneReel | `experiments/frame_memory_smoke.py` |
| Quality: Dense, Separate | `experiments/event_order_attention.py` |
| Quality: Radial, SVG, Sol-Attn | `experiments/sparse_baselines.py` |
| End-to-end latency and GPU scaling | `experiments/fig1_latency.py` |
| Time and communication breakdown, ablations | `experiments/e2e_breakdown.py` |
| Packing overhead against block-sparse kernels | `experiments/kernel_breakdown.py` |
| Latency against scene count | `experiments/scene_count_latency.py` |
| One-minute videos | `experiments/minute_gen.py` |
| Sol-Attn inside scenes | `experiments/intra_scene_sparse.py` |
| MiniMax-H3 | `experiments/minimax_h3_scene.py` |
| HoloCine runtime reuse | `experiments/holocine_gen.py`, `experiments/holocine_parallel.py` |
| CLIP Text and identity appendix | `analysis/text_identity.py` |

Each script documents its arguments in its docstring and `--help`; see also [experiments/README.md](experiments/README.md).

Some experiments need code that is not in this repository. Put it under `third_party/`:
`third_party/Sana` for the Sol-Attn kernel (`solk`, `intra_scene_sparse.py`) and `third_party/HoloCine` for HoloCine.

## Evaluate

Videos are laid out as `RUN/<model>/<setting>/<id>/video.mp4`. The metrics are in `eval/event_order/`. The Gemini judge
reads its credentials from a file you supply.

```bash
$PY_METRICS eval/event_order/narrative_score.py RUN/hunyuan --bench dataset/multishot_bench.json \
  --settings separate,joint_sep_fm4_sw5 --tau 0.3524431409162932 --out RUN/report/hunyuan   # Text, Text+ID
$PY_METRICS eval/event_order/judge_inputs.py RUN/hunyuan --bench dataset/multishot_bench.json --settings ... --out RUN/judge/hunyuan
$PY_METRICS eval/event_order/gemini_judge.py RUN/judge --models hunyuan --credentials .config/gemini/api_key.json
$PY_METRICS eval/event_order/gemini_pairwise.py RUN/judge --model-dir hunyuan --ours joint_sep_fm4_sw5 --others separate ...
bash eval/event_order/vbench.sh RUN/vb hunyuan 0,1                                            # Img., Aes., Dyn.
```

The identity thresholds of the paper are 0.3524431409162932 (HunyuanVideo-1.5) and 0.35822427960626835 (Wan2.2).

## Tests

```bash
$P -m pytest -q tests                                                          # CPU unit tests
$P -m torch.distributed.run --nproc_per_node 2 tests/dist_onereel_gather.py    # multi-GPU checks, one per file
```

## License

The code is released under the [Apache License 2.0](LICENSE). `third_party/xDiT` is also Apache-2.0; our changes are listed in its [MODIFICATIONS.md](third_party/xDiT/MODIFICATIONS.md).
The Apache license does not cover the benchmark prompts. Each prompt in `dataset/multishot_bench.json` has its own
`license` field: Shot2Story annotations are CC BY-NC-SA 4.0, HoloCine examples Apache-2.0, and our scripts CC BY 4.0.
See [dataset/README.md](dataset/README.md).

## Citation

<!-- TODO before release: arXiv number -->
```bibtex
@article{zhang2026onereel,
  title   = {OneReel: Efficient Multi-Scene Video Generation with Scene-Structured Attention},
  author  = {Zhang, Wei and Wang, Yuchen and Feng, Tianrui and Wu, Hongrui and Xu, Chenfeng and Lai, Fan},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
