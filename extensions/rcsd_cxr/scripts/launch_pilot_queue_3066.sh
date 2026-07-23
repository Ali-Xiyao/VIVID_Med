#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101
DATA_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_data
MIMIC_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/021_260129VIVID/data/dataset/mimic-cxr
PYTHON=/ipfs/inspurfileset/home/dqxy/dqxy11/miniforge3/envs/vivid_med310/bin/python
INPUT_ROOT="$PROJECT_ROOT/local_runs/gate1_simplified_20260723"
RUN_ROOT="$PROJECT_ROOT/local_runs/gate3_simplified_20260723_r2"
MANIFEST="$INPUT_ROOT/mimic_chexbert_20k_plus_val.csv"
BACKBONE="$DATA_ROOT/model_aux/vit_base_patch16_224.augreg2_in21k_ft_in1k/model.safetensors"
PROTOTYPES="$INPUT_ROOT/qwen35_2b_field_prototypes.pt"
SESSION=rcsd_pilot_queue_3066

test -f "$MANIFEST"
test -f "$BACKBONE"
test -f "$PROTOTYPES"
test -f "$INPUT_ROOT/overfit_spd/summary.json"
test -f "$INPUT_ROOT/overfit_field_anchor/summary.json"
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
      export PYTHONPATH=\"$PROJECT_ROOT:$PROJECT_ROOT/scripts\"
      export PYTHONUNBUFFERED=1
      export HF_HUB_OFFLINE=1
      export CUBLAS_WORKSPACE_CONFIG=:4096:8

      \"$PYTHON\" scripts/train_visual_state_pilot.py \
        --variant spd \
        --manifest \"$MANIFEST\" \
        --image-root \"$MIMIC_ROOT/mimic-cxr-images\" \
        --backbone-weights \"$BACKBONE\" \
        --prototypes \"$PROTOTYPES\" \
        --output-dir \"$RUN_ROOT/pilot_spd\" \
        --max-steps 1000 \
        --eval-every 200 \
        --batch-size 16 \
        --grad-accumulation 4 \
        --num-workers 2 \
        --seed 0 \
        --field-weight 1.0 \
        > \"$RUN_ROOT/pilot_spd.log\" 2>&1

      \"$PYTHON\" scripts/train_visual_state_pilot.py \
        --variant field_anchor \
        --manifest \"$MANIFEST\" \
        --image-root \"$MIMIC_ROOT/mimic-cxr-images\" \
        --backbone-weights \"$BACKBONE\" \
        --prototypes \"$PROTOTYPES\" \
        --output-dir \"$RUN_ROOT/pilot_field_anchor\" \
        --max-steps 1000 \
        --eval-every 200 \
        --batch-size 16 \
        --grad-accumulation 4 \
        --num-workers 2 \
        --seed 0 \
        --field-weight 1.0 \
        > \"$RUN_ROOT/pilot_field_anchor.log\" 2>&1

      \"$PYTHON\" scripts/audit_visual_state_pilot.py \
        --spd \"$RUN_ROOT/pilot_spd/summary.json\" \
        --field-anchor \"$RUN_ROOT/pilot_field_anchor/summary.json\" \
        --output \"$RUN_ROOT/pilot_gate.json\" \
        > \"$RUN_ROOT/pilot_gate.log\" 2>&1
    '
"

echo "launched session: $SESSION"
