@echo off
cd /d H:\Xiyao_Wang\021_260129VIVID
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=1
set PYTHONUNBUFFERED=1
set CKPT=outputs\ablation_ums_ansmask_12label\checkpoints\best.pt
for /f "delims=" %%F in ('powershell -NoProfile -ExecutionPolicy Bypass -File scripts\select_latest_checkpoint.ps1 -Dir outputs\ablation_ums_ansmask_12label\checkpoints -Default outputs\ablation_ums_ansmask_12label\checkpoints\best.pt 2^>nul') do (
  set CKPT=%%F
)
(
echo.
echo [%date% %time%] start ansmask resume on CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES% checkpoint=%CKPT%
call C:\Users\Admin\anaconda3\condabin\conda.bat run --no-capture-output -n vivid python scripts\train_cxr.py --config configs\ablation_ums_ansmask_12label_resume_gpu1.yaml --resume %CKPT%
echo [%date% %time%] exitcode %ERRORLEVEL%
) >> outputs\logs\ablation_ums_ansmask_12label_resume_from_best_gpu1.log 2>&1
