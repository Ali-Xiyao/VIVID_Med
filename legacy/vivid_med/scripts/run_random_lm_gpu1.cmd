@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d H:\Xiyao_Wang\021_260129VIVID
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=1
set PYTHONUNBUFFERED=1
set "RESUME_CKPT="
for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "$d='outputs\ablation_ums_random_lm_12label\checkpoints'; if (Test-Path -LiteralPath $d) { Get-ChildItem -LiteralPath $d -File | Where-Object { $_.Name -like 'step_*.pt' -or $_.Name -eq 'best.pt' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName }"`) do (
    if not defined RESUME_CKPT set "RESUME_CKPT=%%F"
)
(
echo.
echo [%date% %time%] start random-LM UMS control on CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
if defined RESUME_CKPT (
    echo [%date% %time%] resume random-LM from !RESUME_CKPT!
    call C:\Users\Admin\anaconda3\condabin\conda.bat run --no-capture-output -n vivid python scripts\train_cxr.py --config configs\ablation_ums_random_lm_12label.yaml --resume "!RESUME_CKPT!"
) else (
    call C:\Users\Admin\anaconda3\condabin\conda.bat run --no-capture-output -n vivid python scripts\train_cxr.py --config configs\ablation_ums_random_lm_12label.yaml
)
echo [%date% %time%] exitcode !ERRORLEVEL!
) >> outputs\logs\ablation_ums_random_lm_12label_train.log 2>&1
