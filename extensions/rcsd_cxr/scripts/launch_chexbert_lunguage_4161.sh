#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101
DATA_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_data
PYTHON=/ipfs/inspurfileset/home/dqxy/dqxy11/miniforge3/envs/vivid_med310/bin/python
CHECKPOINT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/model/CheXbert/checkpoints/chexbert.pth
RUN_DIR="$PROJECT_ROOT/local_runs/gate2_20260723"
SESSION=rcsd_g2_chexbert_20260723

test -f "$CHECKPOINT"
test -f "$DATA_ROOT/LUNGUAGE/Lunguage.csv"
test -f "$DATA_ROOT/model_aux/chexbert_bert_config/config.json"
test -f "$DATA_ROOT/model_aux/chexbert_bert_config/vocab.txt"
mkdir -p "$RUN_DIR"

if screen -ls | grep -q "[.]$SESSION"; then
  echo "session already running: $SESSION"
  exit 3
fi

screen -dmS "$SESSION" bash -lc "
  set -euo pipefail
  srun --jobid=4161 --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 \
    bash -lc '
      set -euo pipefail
      cd \"$PROJECT_ROOT\"
      export CUDA_VISIBLE_DEVICES=0
      export PYTHONPATH=\"$PROJECT_ROOT\"
      export TOKENIZERS_PARALLELISM=false
      \"$PYTHON\" scripts/run_chexbert_lunguage_source.py \
        --lunguage \"$DATA_ROOT/LUNGUAGE/Lunguage.csv\" \
        --checkpoint \"$CHECKPOINT\" \
        --auxiliary-dir \"$DATA_ROOT/model_aux/chexbert_bert_config\" \
        --output \"$RUN_DIR/lunguage_chexbert_source.csv\" \
        --audit-output \"$RUN_DIR/lunguage_chexbert_source_audit.json\" \
        --batch-size 16 \
        --device cuda
    ' > \"$RUN_DIR/chexbert_inference.log\" 2>&1
"

echo "launched session: $SESSION"
