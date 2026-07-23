#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101
DATA_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_data
MIMIC_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/021_260129VIVID/data/dataset/mimic-cxr
PYTHON=/ipfs/inspurfileset/home/dqxy/dqxy11/miniforge3/envs/vivid_med310/bin/python
CHECKPOINT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/model/CheXbert/checkpoints/chexbert.pth
RUN_DIR="$PROJECT_ROOT/local_runs/gate1_simplified_20260723"
SESSION=rcsd_mimic_chexbert_3066

test -f "$CHECKPOINT"
test -f "$PROJECT_ROOT/local_runs/gate0_20260723/mimic_canonical_train_validate.csv"
mkdir -p "$RUN_DIR"

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
      export TOKENIZERS_PARALLELISM=false
      \"$PYTHON\" scripts/build_mimic_chexbert_pilot.py \
        --canonical-manifest \"$PROJECT_ROOT/local_runs/gate0_20260723/mimic_canonical_train_validate.csv\" \
        --report-root \"$MIMIC_ROOT/mimic-cxr-reports\" \
        --checkpoint \"$CHECKPOINT\" \
        --auxiliary-dir \"$DATA_ROOT/model_aux/chexbert_bert_config\" \
        --output \"$RUN_DIR/mimic_chexbert_20k_plus_val.csv\" \
        --overfit-output \"$RUN_DIR/mimic_chexbert_overfit_256.csv\" \
        --audit-output \"$RUN_DIR/mimic_chexbert_pilot_audit.json\" \
        --train-count 20000 \
        --batch-size 16 \
        --device cuda
    ' > \"$RUN_DIR/mimic_chexbert_pilot.log\" 2>&1
"

echo "launched session: $SESSION"
