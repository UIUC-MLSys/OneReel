#!/usr/bin/env bash
# One-command environment setup for OneReel (OneReel) on a new cluster.
#
# Creates the conda environments for evaluation (onereel_metrics) and VBench (videoeval), downloads
# model weights into a data directory, wires the symlinks the code expects, writes env.local.sh,
# and verifies the result. The generation environment data/envs/onereel is set up separately
# (src/onereel/README.md). Every step is idempotent: rerun it after
# an interruption and finished steps are skipped.
#
# Usage:
#   bash scripts/setup_env.sh [options]
#
# Options:
#   --data DIR      root for models and caches (default: <repo>/data); put it on a large
#                   file system, e.g. /path/to/large/disk/onereel_data
#   --envs LIST     comma list of onereel_metrics,videoeval  (default: both)
#   --models LIST   none | core | all | comma list of model groups (default: core)
#                   groups: wan22 eval cogvideox
#                   core = wan22,eval   all = every group
#   --conda DIR     conda base directory; Miniconda is installed there if missing
#                   (default: the active conda, else $HOME/miniconda3)
#   --force         recreate conda environments that already exist
#   --check         only verify the current installation
#   -h, --help      show this help
#
# Model groups (sizes approximate):
#   wan22      Wan-AI/Wan2.2-TI2V-5B-Diffusers (HF cache, ~32 GB)    OneReel generation
#   eval       dinov2-base, clip-vit-base-patch32, LPIPS AlexNet     DINO/CLIP/LPIPS metrics
#   cogvideox  THUDM/CogVideoX1.5-5B (HF cache)                       CogVideoX1.5
# Gated or private models need `export HF_TOKEN=...` before running.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$REPO/data"
ENVS="onereel_metrics,videoeval"
MODELS="core"
FORCE=0
CHECK_ONLY=0
CONDA_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    --envs) ENVS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --conda) CONDA_DIR="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$MODELS" in
  core) MODELS="wan22,eval" ;;
  all) MODELS="wan22,eval,cogvideox" ;;
  none) MODELS="" ;;
esac
has() { [[ ",$1," == *",$2,"* ]]; }
log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARN: %s\033[0m\n' "$*" >&2; }

# ---------------------------------------------------------------- conda
setup_conda() {
  if [ -z "$CONDA_DIR" ]; then
    if command -v conda >/dev/null 2>&1; then CONDA_DIR="$(conda info --base)"; else CONDA_DIR="$HOME/miniconda3"; fi
  fi
  if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    log "Installing Miniconda into $CONDA_DIR"
    local inst; inst="$(mktemp -d)/miniconda.sh"
    curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh" -o "$inst"
    bash "$inst" -b -p "$CONDA_DIR"
  fi
  # shellcheck disable=SC1091
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  echo "conda: $CONDA_DIR ($(conda --version))"
}

env_exists() { [ -x "$CONDA_DIR/envs/$1/bin/python" ]; }

prepare_env() {   # name -> 0 if it must be (re)built
  if env_exists "$1"; then
    if [ "$FORCE" = 1 ]; then conda env remove -y -n "$1"; return 0; fi
    echo "$1 exists (use --force to recreate)"; return 1
  fi
  return 0
}

build_onereel_metrics() {
  log "Environment onereel_metrics (generation, CUDA 12.1)"
  if prepare_env onereel_metrics; then
    conda env create -f "$REPO/envs/onereel_metrics.yaml"
  fi
  local pip="$CONDA_DIR/envs/onereel_metrics/bin/pip"
  "$CONDA_DIR/envs/onereel_metrics/bin/python" -c "import flash_attn" 2>/dev/null \
    || "$pip" install flash-attn==2.7.3 --no-build-isolation \
    || warn "flash-attn build failed; rerun on a GPU node with nvcc (module load cuda) - xDiT falls back to SDPA"
  "$pip" install -e "$REPO/third_party/xDiT"
  "$pip" install -e "$REPO"
}

build_videoeval() {
  log "Environment videoeval (VBench/DINO/CLIP evaluation, CUDA 11.8)"
  if prepare_env videoeval; then
    conda env create -f "$REPO/envs/videoeval.yaml" \
      || warn "videoeval creation failed (detectron2 builds from source and needs the env's nvcc; retry on a GPU node)"
  fi
}

# ---------------------------------------------------------------- models
hf() {   # python snippet run with huggingface_hub in an env that has it
  local py="$CONDA_DIR/envs/onereel_metrics/bin/python"
  HF_HOME="$DATA/hf" TORCH_HOME="$DATA/torch" HF_HUB_ENABLE_HF_TRANSFER=0 "$py" - "$@"
}

link() {   # link TARGET LINKNAME (replace an existing link, never a real directory)
  mkdir -p "$(dirname "$2")"
  if [ -e "$2" ] && [ ! -L "$2" ]; then warn "$2 exists and is not a symlink; leaving it"; return; fi
  ln -sfn "$1" "$2"
}

download_models() {
  [ -n "$MODELS" ] || return 0
  log "Model weights -> $DATA (groups: $MODELS)"
  mkdir -p "$DATA/hf" "$DATA/models" "$DATA/torch"
  local M="$DATA/models"
  if has "$MODELS" wan22; then
    local snap
    snap="$(hf <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download('Wan-AI/Wan2.2-TI2V-5B-Diffusers'))
EOF
)"
    link "$(echo "$snap" | tail -1)" "$M/Wan2.2-TI2V-5B-Diffusers"
  fi
  if has "$MODELS" eval; then
    hf <<'EOF'
from huggingface_hub import snapshot_download
for repo in ('facebook/dinov2-base', 'openai/clip-vit-base-patch32'):
    print(snapshot_download(repo))
import torchvision
torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)   # LPIPS backbone -> TORCH_HOME
print('alexnet ok')
EOF
  fi
  if has "$MODELS" cogvideox; then
    hf <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download('THUDM/CogVideoX1.5-5B'))
EOF
  fi
}

write_env_local() {
  cat > "$REPO/env.local.sh" <<EOF
# Written by scripts/setup_env.sh on $(date +%F). Sourced by scripts/env.sh.
export CONDA_ROOT="$CONDA_DIR"
export ONEREEL_ROOT="$REPO"
export ONEREEL_DATA="$DATA"
export ONEREEL_MODELS="$DATA/models"
export HF_HOME="$DATA/hf"
export TORCH_HOME="$DATA/torch"
EOF
  echo "wrote $REPO/env.local.sh"
}

# ---------------------------------------------------------------- verification
verify() {
  log "Verification"
  local ok=1
  check_env() {   # name python-snippet
    if env_exists "$1"; then
      if HF_HOME="$DATA/hf" "$CONDA_DIR/envs/$1/bin/python" -c "$2" 2>/dev/null; then echo "  [ok]   $1"
      else echo "  [FAIL] $1: imports failed"; ok=0; fi
    else echo "  [--]   $1 not installed"; fi
  }
  check_env onereel_metrics "import torch, diffusers, transformers, xfuser, onereel; print('       torch', torch.__version__, 'diffusers', diffusers.__version__)"
  check_env videoeval "import torch, vbench, detectron2; print('       torch', torch.__version__)"
  [ -e "$DATA/models/Wan2.2-TI2V-5B-Diffusers/model_index.json" ] \
    && echo "  [ok]   Wan2.2-TI2V-5B-Diffusers" || echo "  [--]   Wan2.2-TI2V-5B-Diffusers"
  local gpu
  if gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"; then echo "  GPU: ${gpu%%$'\n'*}"
  else echo "  (no GPU visible here; GPU tests must run on a compute node)"; fi
  [ "$ok" = 1 ] || { warn "some environments failed verification"; return 1; }
}

# ---------------------------------------------------------------- main
setup_conda
mkdir -p "$DATA"
DATA="$(cd "$DATA" && pwd)"
if [ "$CHECK_ONLY" = 0 ]; then
  has "$ENVS" onereel_metrics && build_onereel_metrics
  has "$ENVS" videoeval && build_videoeval
  download_models
  write_env_local
fi
verify
log "Done. Use: source scripts/env.sh   (sets HF_HOME, TORCH_HOME, PY_METRICS, PY_VIDEOEVAL, ...)"
