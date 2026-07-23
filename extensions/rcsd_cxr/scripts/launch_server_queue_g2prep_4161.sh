#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101
PYTHON=/ipfs/inspurfileset/home/dqxy/dqxy11/miniforge3/envs/vivid_med310/bin/python
QUEUE="$PROJECT_ROOT/configs/server_queue_4161.yaml"
STATE_DIR="$PROJECT_ROOT/local_runs/server_queue_4161_20260723_r3"
CONTROLLER_LOG="$STATE_DIR/controller_g2prep.log"

cd "$PROJECT_ROOT"
exec srun \
  --jobid=4161 \
  --overlap \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  bash -lc "
    set -euo pipefail
    cd '$PROJECT_ROOT'
    export RCSD_DATA_REGISTRY='$PROJECT_ROOT/data_refs/datasets.server.yaml'
    export PYTHONUNBUFFERED=1
    exec '$PYTHON' scripts/run_gate_queue.py \
      --queue '$QUEUE' \
      --state-dir '$STATE_DIR' \
      --project-root '$PROJECT_ROOT' \
      --from-task G2_official_source_manifest
  " >"$CONTROLLER_LOG" 2>&1
