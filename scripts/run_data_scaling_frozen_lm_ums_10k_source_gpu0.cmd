@echo off
setlocal
cd /d H:\Xiyao_Wang\021_260129VIVID
set HF_ENDPOINT=https://huggingface.co
set HF_HUB_DISABLE_XET=1
set CUDA_VISIBLE_DEVICES=0
set PYTHONIOENCODING=utf-8
if not exist outputs\logs mkdir outputs\logs
set LOG=outputs\logs\data_scaling_frozen_lm_ums_10k_source_gpu0.log
echo START %DATE% %TIME% > "%LOG%"
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES% >> "%LOG%"
echo COMMAND python -u scripts\train_cxr.py --config configs\data_scaling\frozen_lm_ums_10k.yaml >> "%LOG%"
python -u scripts\train_cxr.py --config configs\data_scaling\frozen_lm_ums_10k.yaml >> "%LOG%" 2>&1
set EXITCODE=%ERRORLEVEL%
echo EXITCODE %EXITCODE% >> "%LOG%"
echo END %DATE% %TIME% >> "%LOG%"
exit /b %EXITCODE%
