@echo off
cd /d H:\Xiyao_Wang\021_260129VIVID

set HF_ENDPOINT=https://huggingface.co
set HF_HUB_DISABLE_XET=1
set CUDA_VISIBLE_DEVICES=0
set PYTHONIOENCODING=utf-8

if not exist outputs\logs mkdir outputs\logs
set LOG=outputs\logs\data_scaling_frozen_lm_ums_1k_source.log

echo START %DATE% %TIME% > %LOG%
echo COMMAND python -u scripts\train_cxr.py --config configs\data_scaling\frozen_lm_ums_1k.yaml >> %LOG%

python -u scripts\train_cxr.py --config configs\data_scaling\frozen_lm_ums_1k.yaml >> %LOG% 2>&1
set CODE=%ERRORLEVEL%
echo EXITCODE %CODE% >> %LOG%
exit /b %CODE%
