@echo off
setlocal
cd /d H:\Xiyao_Wang\021_260129VIVID
if not exist outputs\logs mkdir outputs\logs
set CUDA_VISIBLE_DEVICES=1
set LOG=outputs\logs\data_scaling_no_lm_ums_30k_source_gpu1.log
echo START %DATE% %TIME% > "%LOG%"
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES% >> "%LOG%"
echo COMMAND python -u scripts\train_ums_classifier.py --config configs\data_scaling\no_lm_ums_30k.yaml >> "%LOG%"
python -u scripts\train_ums_classifier.py --config configs\data_scaling\no_lm_ums_30k.yaml >> "%LOG%" 2>&1
set EXITCODE=%ERRORLEVEL%
echo EXITCODE %EXITCODE% >> "%LOG%"
echo END %DATE% %TIME% >> "%LOG%"
exit /b %EXITCODE%
