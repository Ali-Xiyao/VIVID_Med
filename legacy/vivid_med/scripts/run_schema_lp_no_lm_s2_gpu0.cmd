@echo off
setlocal
cd /d H:\Xiyao_Wang\021_260129VIVID
if not exist outputs\logs mkdir outputs\logs
set CUDA_VISIBLE_DEVICES=0
set LOG=outputs\logs\schema_lp_no_lm_s2_gpu0.log
echo START %DATE% %TIME% > "%LOG%"
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES% >> "%LOG%"
echo COMMAND python scripts\train_vit_baseline.py --config configs\schema_sweep\lp_no_lm_s2_state_answerability.yaml >> "%LOG%"
python scripts\train_vit_baseline.py --config configs\schema_sweep\lp_no_lm_s2_state_answerability.yaml >> "%LOG%" 2>&1
set EXITCODE=%ERRORLEVEL%
echo EXITCODE %EXITCODE% >> "%LOG%"
echo END %DATE% %TIME% >> "%LOG%"
exit /b %EXITCODE%
