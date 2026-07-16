@echo off
setlocal EnableDelayedExpansion
cd /d H:\Xiyao_Wang\021_260129VIVID
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=1
set PYTHONUNBUFFERED=1
set "CONFIG=configs\ablation_ums_null_as_negative_12label_thermal_resume_gpu1.yaml"
set "CHECKPOINT_DIR=outputs\ablation_ums_null_as_negative_12label\checkpoints"
set "RESUME="
for /f "usebackq delims=" %%F in (`powershell -NoProfile -ExecutionPolicy Bypass -File scripts\select_latest_checkpoint.ps1 -Dir "%CHECKPOINT_DIR%"`) do (
set "RESUME=%%F"
)
if defined RESUME if not exist "!RESUME!" set "RESUME="
(
echo.
echo [%date% %time%] start null-as-negative on CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo [%date% %time%] config %CONFIG%
if defined RESUME (
echo [%date% %time%] resume checkpoint !RESUME!
call C:\Users\Admin\anaconda3\condabin\conda.bat run --no-capture-output -n vivid python scripts\train_cxr.py --config "%CONFIG%" --resume "!RESUME!"
) else (
echo [%date% %time%] no checkpoint found; starting from scratch
call C:\Users\Admin\anaconda3\condabin\conda.bat run --no-capture-output -n vivid python scripts\train_cxr.py --config "%CONFIG%"
)
set RC=!ERRORLEVEL!
echo [%date% %time%] exitcode !RC!
) >> outputs\logs\ablation_ums_null_as_negative_12label_train.log 2>&1
exit /b !RC!
