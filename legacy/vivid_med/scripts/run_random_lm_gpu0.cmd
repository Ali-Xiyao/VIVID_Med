@echo off
cd /d H:\Xiyao_Wang\021_260129VIVID
set CUDA_VISIBLE_DEVICES=0
(
echo [%date% %time%] start random-LM UMS control on CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
call C:\Users\Admin\anaconda3\condabin\conda.bat run --no-capture-output -n vivid python scripts\train_cxr.py --config configs\ablation_ums_random_lm_12label.yaml
echo [%date% %time%] exitcode %ERRORLEVEL%
) > outputs\logs\ablation_ums_random_lm_12label_train_gpu0.log 2>&1
