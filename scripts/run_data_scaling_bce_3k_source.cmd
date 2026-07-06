@echo off
setlocal

cd /d H:\Xiyao_Wang\021_260129VIVID

if not exist outputs\logs mkdir outputs\logs

set LOG=outputs\logs\data_scaling_bce_3k_source.log

echo START %DATE% %TIME% > %LOG%
echo COMMAND python scripts\train_vit_baseline.py --config configs\data_scaling\bce_3k.yaml >> %LOG%

python scripts\train_vit_baseline.py --config configs\data_scaling\bce_3k.yaml >> %LOG% 2>&1

set EXITCODE=%ERRORLEVEL%
echo EXITCODE %EXITCODE% >> %LOG%
echo END %DATE% %TIME% >> %LOG%

exit /b %EXITCODE%
