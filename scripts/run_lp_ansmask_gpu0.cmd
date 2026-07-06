@echo off
cd /d H:\Xiyao_Wang\021_260129VIVID
set CUDA_VISIBLE_DEVICES=0
(
echo [%date% %time%] start lp ansmask on CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
call C:\Users\Admin\anaconda3\condabin\conda.bat run --no-capture-output -n vivid python scripts\train_vit_baseline.py --config configs\lp_ums_ansmask_12label.yaml
echo [%date% %time%] exitcode %ERRORLEVEL%
) > outputs\logs\lp_ums_ansmask_12label_train_gpu0.log 2>&1
