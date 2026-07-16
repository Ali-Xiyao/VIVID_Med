#!/bin/bash
# KiTS21 Segmentation Transfer — 4组实验批量运行
# 用法: bash scripts/run_seg_kits_all.sh
# 或单独运行某个: bash scripts/run_seg_kits_all.sh imagenet

set -e
cd "$(dirname "$0")/.."

SCRIPT="scripts/train_seg_transfer.py"

run_experiment() {
    local name=$1
    local config=$2
    echo "============================================"
    echo "  Running: ${name}"
    echo "  Config:  ${config}"
    echo "============================================"
    python ${SCRIPT} --config ${config} 2>&1 | tee "outputs/seg_kits_${name}.log"
    echo ""
    echo "  ${name} finished."
    echo ""
}

# 如果指定了参数，只跑对应实验
if [ -n "$1" ]; then
    case "$1" in
        imagenet)
            run_experiment "imagenet" "configs/seg_kits_imagenet.yaml"
            ;;
        biomedclip)
            run_experiment "biomedclip" "configs/seg_kits_biomedclip.yaml"
            ;;
        a_ums)
            run_experiment "a_ums" "configs/seg_kits_a_ums.yaml"
            ;;
        spd)
            run_experiment "spd" "configs/seg_kits_spd.yaml"
            ;;
        *)
            echo "Unknown experiment: $1"
            echo "Usage: $0 [imagenet|biomedclip|a_ums|spd]"
            exit 1
            ;;
    esac
    exit 0
fi

# 默认：按顺序跑全部4组
echo "Starting all 4 KiTS segmentation transfer experiments..."
echo ""

# 1) ImageNet supervised baseline
run_experiment "imagenet" "configs/seg_kits_imagenet.yaml"

# 2) BiomedCLIP baseline
run_experiment "biomedclip" "configs/seg_kits_biomedclip.yaml"

# 3) A+UMS (no SPD) ablation
run_experiment "a_ums" "configs/seg_kits_a_ums.yaml"

# 4) SPD (ours)
run_experiment "spd" "configs/seg_kits_spd.yaml"

echo "============================================"
echo "  All 4 experiments completed!"
echo "============================================"
