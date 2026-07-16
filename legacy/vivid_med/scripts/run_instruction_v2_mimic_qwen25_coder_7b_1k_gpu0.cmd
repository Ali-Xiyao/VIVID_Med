@echo off
cd /d H:\Xiyao_Wang\021_260129VIVID
call conda run -n vivid python scripts/train_cxr_instruction.py --config configs/instruction_v2_mimic_qwen25_coder_7b_1k.yaml > outputs/logs/instruction_v2_mimic_qwen25_coder_7b_1k_gpu0.log 2>&1
echo EXITCODE %ERRORLEVEL%>> outputs/logs/instruction_v2_mimic_qwen25_coder_7b_1k_gpu0.log
exit /b %ERRORLEVEL%
