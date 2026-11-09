#!/bin/bash
# VBench quality for an event-order run: links RUN_DIR/<model>/<setting>/<id>/video.mp4 into
# RUN_DIR/vbench_in/<model>__<setting>/<id>.mp4 and runs eval/run_vbench.py (videoeval env), one process per GPU.
# Settings whose dir already holds vbench_results are skipped.
# Usage: bash eval/event_order/vbench.sh RUN_DIR MODEL [GPU_LIST, default 0,1]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=$(realpath "$1"); MODEL=$2; IFS=, read -ra GPUS <<< "${3:-0,1}"
source "$ROOT/scripts/env.sh"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
DIMS="imaging_quality aesthetic_quality motion_smoothness dynamic_degree subject_consistency background_consistency"

TODO=()
for s in "$RUN/$MODEL"/*/; do
  d="$RUN/vbench_in/${MODEL}__$(basename "$s")"
  mkdir -p "$d"
  for v in "$s"*/video.mp4; do ln -sf "$(realpath "$v")" "$d/$(basename "$(dirname "$v")").mp4"; done
  compgen -G "$d/vbench_results/*_eval_results.json" > /dev/null || TODO+=("$d")
done
echo "[vbench] ${#TODO[@]} settings to score on GPUs ${GPUS[*]}"
[ ${#TODO[@]} -eq 0 ] && exit 0

cd "$ROOT/eval"
n=${#GPUS[@]}
for i in "${!GPUS[@]}"; do
  part=()
  for j in "${!TODO[@]}"; do (( j % n == i )) && part+=("${TODO[$j]}"); done
  [ ${#part[@]} -eq 0 ] && continue
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} MASTER_PORT=$((29650 + i)) $PY_VIDEOEVAL run_vbench.py --video-dirs "${part[@]}" \
    --output-dir "$RUN/vbench_out$i" --dimensions $DIMS > "$RUN/vbench$i.log" 2>&1 &
done
wait
echo "[vbench] done"
