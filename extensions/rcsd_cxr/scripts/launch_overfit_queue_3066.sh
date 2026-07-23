#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101
DATA_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_data
MIMIC_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/021_260129VIVID/data/dataset/mimic-cxr
PYTHON=/ipfs/inspurfileset/home/dqxy/dqxy11/miniforge3/envs/vivid_med310/bin/python
RUN_ROOT="$PROJECT_ROOT/local_runs/gate1_simplified_20260723"
MANIFEST="$RUN_ROOT/mimic_chexbert_overfit_256.csv"
BACKBONE="$DATA_ROOT/model_aux/vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors"
SESSION=rcsd_overfit_queue_3066

test -f "$MANIFEST"
test -f "$BACKBONE"
mkdir -p "$RUN_ROOT"

if screen -ls | grep -q "[.]$SESSION"; then
  echo "session already running: $SESSION"
  exit 3
fi

screen -dmS "$SESSION" bash -lc "
  set -euo pipefail
  srun --jobid=3066 --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 \
    bash -lc '
      set -euo pipefail
      cd \"$PROJECT_ROOT\"
      export CUDA_VISIBLE_DEVICES=0
      export PYTHONPATH=\"$PROJECT_ROOT\"
      export PYTHONUNBUFFERED=1
      export HF_HUB_OFFLINE=1
      \"$PYTHON\" scripts/train_visual_state_overfit.py \
        --variant spd \
        --manifest \"$MANIFEST\" \
        --image-root \"$MIMIC_ROOT/mimic-cxr-images\" \
        --backbone-weights \"$BACKBONE\" \
        --output-dir \"$RUN_ROOT/overfit_spd\" \
        --max-steps 2000 \
        --eval-every 100 \
        --batch-size 16 \
        --num-workers 2 \
        --seed 0 \
        > \"$RUN_ROOT/overfit_spd.log\" 2>&1

      \"$PYTHON\" scripts/train_visual_state_overfit.py \
        --variant field_anchor \
        --manifest \"$MANIFEST\" \
        --image-root \"$MIMIC_ROOT/mimic-cxr-images\" \
        --backbone-weights \"$BACKBONE\" \
        --output-dir \"$RUN_ROOT/overfit_field_anchor\" \
        --max-steps 2000 \
        --eval-every 100 \
        --batch-size 16 \
        --num-workers 2 \
        --seed 0 \
        > \"$RUN_ROOT/overfit_field_anchor.log\" 2>&1
    '
"

echo "launched session: $SESSION"
