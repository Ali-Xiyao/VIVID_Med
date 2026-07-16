@echo off
setlocal EnableDelayedExpansion
cd /d H:\Xiyao_Wang\021_260129VIVID
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=1
set PYTHONUNBUFFERED=1
(
echo.
echo [%date% %time%] start lp ansmask on CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
call C:\Users\Admin\anaconda3\condabin\conda.bat run --no-capture-output -n vivid python scripts\train_vit_baseline.py --config configs\lp_ums_ansmask_12label.yaml
set RC=!ERRORLEVEL!
echo [%date% %time%] exitcode !RC!
) >> outputs\logs\lp_ums_ansmask_12label_train.log 2>&1
exit /b !RC!
