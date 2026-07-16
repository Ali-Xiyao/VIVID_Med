@echo off
cd /d H:\Xiyao_Wang\021_260129VIVID
set CUDA_VISIBLE_DEVICES=0
(
echo [%date% %time%] start counterfactual prefix dependency eval on CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
call C:\Users\Admin\anaconda3\condabin\conda.bat run --no-capture-output -n vivid python scripts\eval_counterfactual_prefix_dependency.py --config configs\ablation_A_ums_12label.yaml --checkpoint outputs\ablation_A_ums_12label\checkpoints\best.pt --output outputs\counterfactual_prefix_dependency_A_ums_12label_128.json --max-samples 128 --batch-size 2 --num-workers 0
echo [%date% %time%] exitcode %ERRORLEVEL%
) > outputs\logs\counterfactual_prefix_dependency_A_ums_12label_128_gpu0.log 2>&1
