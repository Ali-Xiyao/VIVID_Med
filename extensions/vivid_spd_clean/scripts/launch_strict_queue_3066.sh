#!/usr/bin/env bash
set -euo pipefail

ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_vivid_spd_clean
PY=/ipfs/inspurfileset/home/dqxy/dqxy11/miniforge3/envs/vivid_med310/bin/python
RUN_ROOT="$ROOT/local_runs/strict_vivid_spd_qwen35_2b_20260723_s0_s3"

if [[ -e "$RUN_ROOT" ]]; then
  echo "run root already exists: $RUN_ROOT" >&2
  exit 17
fi

cd "$ROOT"
export PYTHONUNBUFFERED=1

exec srun \
  --jobid=3066 \
  --exclusive \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --gres=gpu:1 \
  "$PY" scripts/run_strict_vivid_spd_queue.py \
    --run-root "$RUN_ROOT" \
    --hard-manifest /ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101/local_runs/d0_d1_20260723/hard_ums.jsonl \
    --overfit-ids /ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101/local_runs/d0_d1_20260723/overfit_ids.json \
    --mimic-image-root /ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/021_260129VIVID/data/dataset/mimic-cxr/mimic-cxr-images \
    --teacher-path /ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/model/Qwen3.5-2B \
    --backbone-weights /ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_data/model_aux/vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors \
    --probe-train-manifest /ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101/local_runs/d0_d1_20260723/chexpert_probe_train.csv \
    --expert-manifest /ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101/local_runs/d0_d1_20260723/chexpert_expert_dev.csv \
    --chexpert-root /ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/021_260129VIVID/data/dataset/CheXpert-v1.0-small \
    --lock "$ROOT/audit/vivid_spd_clean_lock.json" \
    --device cuda:0
