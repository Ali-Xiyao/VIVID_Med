@echo off
cd /d H:\Xiyao_Wang\021_260129VIVID
if not exist outputs\logs mkdir outputs\logs
set CUDA_VISIBLE_DEVICES=0
set LOG=outputs\logs\schema_no_lm_s3_source_resume3_gpu0.log
echo START %DATE% %TIME% > %LOG%
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES% >> %LOG%
echo COMMAND python scripts\train_ums_classifier.py --config configs\schema_sweep\no_lm_s3_state_uncertainty.yaml --resume outputs\schema_sweep\no_lm_s3_state_uncertainty\best.pt >> %LOG%
python scripts\train_ums_classifier.py --config configs\schema_sweep\no_lm_s3_state_uncertainty.yaml --resume outputs\schema_sweep\no_lm_s3_state_uncertainty\best.pt >> %LOG% 2>&1
echo EXITCODE %ERRORLEVEL% >> %LOG%
echo END %DATE% %TIME% >> %LOG%
