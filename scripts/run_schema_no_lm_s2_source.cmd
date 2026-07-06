@echo off
setlocal

cd /d H:\Xiyao_Wang\021_260129VIVID

if not exist outputs\logs mkdir outputs\logs

set LOG=outputs\logs\schema_no_lm_s2_source.log

echo START %DATE% %TIME% > %LOG%
echo COMMAND python scripts\train_ums_classifier.py --config configs\schema_sweep\no_lm_s2_state_answerability.yaml >> %LOG%

python scripts\train_ums_classifier.py --config configs\schema_sweep\no_lm_s2_state_answerability.yaml >> %LOG% 2>&1

set EXITCODE=%ERRORLEVEL%
echo EXITCODE %EXITCODE% >> %LOG%
echo END %DATE% %TIME% >> %LOG%

exit /b %EXITCODE%
