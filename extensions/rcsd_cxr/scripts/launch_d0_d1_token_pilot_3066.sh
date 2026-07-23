#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101
DATA_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_data
MODEL_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/model
MIMIC_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/021_260129VIVID/data/dataset/mimic-cxr
PYTHON=/ipfs/inspurfileset/home/dqxy/dqxy11/miniforge3/envs/vivid_med310/bin/python
PYTHON_ENV=/ipfs/inspurfileset/home/dqxy/dqxy11/miniforge3/envs/vivid_med310
INPUT_ROOT="$PROJECT_ROOT/local_runs/d0_d1_20260723"
RUN_ROOT="$PROJECT_ROOT/local_runs/d0_d1_token_pilot_qwen35_2b_20260723_s2"
TEACHER="$MODEL_ROOT/Qwen3.5-2B"
BACKBONE="$DATA_ROOT/model_aux/vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors"
LOCK="$PROJECT_ROOT/audit/rcsd_d0_d1_review_lock.json"

test -x "$PYTHON"
test -f "$LOCK"
test -f "$INPUT_ROOT/hard_ums.jsonl"
test -f "$INPUT_ROOT/d1_reliability.jsonl"
test -f "$TEACHER/model.safetensors.index.json"
test -f "$TEACHER/config.json"
test -f "$BACKBONE"
test -d "$MIMIC_ROOT/mimic-cxr-images"
if [[ -e "$RUN_ROOT" ]]; then
  echo "refusing to overwrite: $RUN_ROOT" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH="$PYTHON_ENV/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PYTHON" "$PROJECT_ROOT/scripts/check_qwen35_acceleration.py"

exec "$PYTHON" "$PROJECT_ROOT/scripts/run_d0_d1_local_queue.py" \
  --mode pilot \
  --lock "$LOCK" \
  --hard-manifest "$INPUT_ROOT/hard_ums.jsonl" \
  --reliability-manifest "$INPUT_ROOT/d1_reliability.jsonl" \
  --image-root "$MIMIC_ROOT/mimic-cxr-images" \
  --teacher-path "$TEACHER" \
  --backbone-weights "$BACKBONE" \
  --output-root "$RUN_ROOT"
