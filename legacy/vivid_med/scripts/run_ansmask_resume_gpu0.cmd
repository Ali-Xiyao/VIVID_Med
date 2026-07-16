@echo off
cd /d H:\Xiyao_Wang\021_260129VIVID
set CUDA_VISIBLE_DEVICES=0
set CKPT=outputs\ablation_ums_ansmask_12label\checkpoints\best.pt
for /f "delims=" %%F in ('powershell -NoProfile -ExecutionPolicy Bypass -File scripts\select_latest_checkpoint.ps1 -Dir outputs\ablation_ums_ansmask_12label\checkpoints -Default outputs\ablation_ums_ansmask_12label\checkpoints\best.pt 2^>nul') do (
  set CKPT=%%F
)
(
echo [%date% %time%] start ansmask resume on CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES% checkpoint=%CKPT%
call C:\Users\Admin\anaconda3\condabin\conda.bat run --no-capture-output -n vivid python scripts\train_cxr.py --config configs\ablation_ums_ansmask_12label.yaml --resume %CKPT%
echo [%date% %time%] exitcode %ERRORLEVEL%
) > outputs\logs\ablation_ums_ansmask_12label_resume_from_best_gpu0.log 2>&1
