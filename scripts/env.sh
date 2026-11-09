# Shared paths for job scripts. Source it from the repository root:  source scripts/env.sh
#
# Every value can be overridden from the environment. scripts/setup_env.sh writes the
# values it used to env.local.sh (git-ignored), which is sourced first.
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$_repo/env.local.sh" ] && source "$_repo/env.local.sh"

export ONEREEL_ROOT="${ONEREEL_ROOT:-$_repo}"
export ONEREEL_DATA="${ONEREEL_DATA:-$ONEREEL_ROOT/data}"           # models, HF cache, torch cache
export ONEREEL_MODELS="${ONEREEL_MODELS:-$ONEREEL_DATA/models}"     # non-HF-cache model directories
export ONEREEL_OUTPUTS="${ONEREEL_OUTPUTS:-$ONEREEL_ROOT/outputs}"  # experiment results
export HF_HOME="${HF_HOME:-$ONEREEL_DATA/hf}"
export TORCH_HOME="${TORCH_HOME:-$ONEREEL_DATA/torch}"

if [ -z "${CONDA_ROOT:-}" ]; then
  CONDA_ROOT="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
fi
export CONDA_ROOT
export PY_METRICS="${PY_METRICS:-$CONDA_ROOT/envs/onereel_metrics/bin/python}"
export PY_VIDEOEVAL="${PY_VIDEOEVAL:-$CONDA_ROOT/envs/videoeval/bin/python}"
unset _repo
