#!/usr/bin/env bash
set -euo pipefail

ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_vivid_gds_fresh_replication
GDS=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_vivid_gds
RCSD=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101
DATA=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/021_260129VIVID/data/dataset
PY=/ipfs/inspurfileset/home/dqxy/dqxy11/miniforge3/envs/vivid_med310/bin/python
RUN_ROOT="$ROOT/local_runs/vivid_gds_fresh_replication_qwen35_2b_20260724"

if [[ -e "$RUN_ROOT" ]]; then
  echo "run root already exists: $RUN_ROOT" >&2
  exit 17
fi

cd "$ROOT"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

while squeue -s -j 3066 -h -o '%i' \
  | grep -v '^3066\.batch$' \
  | grep -q '^3066\.'; do
  echo "$(date -Is) waiting for existing allocation-3066 steps to finish"
  squeue -s -j 3066 -h -o '%i|%N|%j|%M'
  sleep 60
done

echo "$(date -Is) allocation-3066 step queue is clear; launching replication"

exec srun \
  --jobid=3066 \
  --exclusive \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --gres=gpu:1 \
  "$PY" scripts/run_fresh_replication_queue.py \
    --run-root "$RUN_ROOT" \
    --lock "$ROOT/audit/vivid_gds_fresh_replication_lock.json" \
    --split-audit "$ROOT/private_manifests/fresh_split_audit.json" \
    --split-dir "$ROOT/private_manifests/manifests" \
    --hard-manifest "$RCSD/local_runs/d0_d1_20260723/hard_ums.jsonl" \
    --mimic-image-root "$DATA/mimic-cxr/mimic-cxr-images" \
    --chexpert-image-root "$DATA/CheXpert-v1.0-small" \
    --teacher-path /ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/model/Qwen3.5-2B \
    --backbone-weights /ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_data/model_aux/vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors \
    --training-script "$GDS/scripts/train_vivid_gds_stage_a.py" \
    --device cuda:0
