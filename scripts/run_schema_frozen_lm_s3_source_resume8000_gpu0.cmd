@echo off
setlocal
cd /d H:\Xiyao_Wang\021_260129VIVID
if not exist outputs\logs mkdir outputs\logs
set CUDA_VISIBLE_DEVICES=0
set LOG=outputs\logs\schema_frozen_lm_s3_source_resume8000_gpu0.log
set CKPT=outputs\schema_sweep\frozen_lm_s3_state_uncertainty\checkpoints\step_8000.pt
echo START %DATE% %TIME% > "%LOG%"
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES% >> "%LOG%"
echo RESUME_CHECKPOINT=%CKPT% >> "%LOG%"
echo COMMAND python scripts\train_cxr.py --config configs\schema_sweep\frozen_lm_s3_state_uncertainty.yaml --resume %CKPT% >> "%LOG%"
python scripts\train_cxr.py --config configs\schema_sweep\frozen_lm_s3_state_uncertainty.yaml --resume %CKPT% >> "%LOG%" 2>&1
set EXITCODE=%ERRORLEVEL%
echo EXITCODE %EXITCODE% >> "%LOG%"
echo END %DATE% %TIME% >> "%LOG%"
exit /b %EXITCODE%
