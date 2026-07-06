@echo off
setlocal
cd /d H:\Xiyao_Wang\021_260129VIVID
if not exist outputs\logs mkdir outputs\logs
set LOG=outputs\logs\data_scaling_lp_frozen_lm_ums_1k.log
echo START %DATE% %TIME% > %LOG%
echo CMD python -u scripts\train_vit_baseline.py --config configs\data_scaling\lp_frozen_lm_ums_1k.yaml >> %LOG%
python -u scripts\train_vit_baseline.py --config configs\data_scaling\lp_frozen_lm_ums_1k.yaml >> %LOG% 2>&1
set EXITCODE=%ERRORLEVEL%
echo EXITCODE %EXITCODE% >> %LOG%
echo END %DATE% %TIME% >> %LOG%
exit /b %EXITCODE%
