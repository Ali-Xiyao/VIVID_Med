@echo off
cd /d H:\Xiyao_Wang\021_260129VIVID
set CUDA_VISIBLE_DEVICES=1
call conda run -n vivid python scripts/train_vit_baseline.py --config configs/lp_instruction_v4_mimic_counterfactual_weighted_qwen25_coder_7b_1k.yaml > outputs\logs\lp_instruction_v4_mimic_counterfactual_weighted_qwen25_coder_7b_1k_gpu1.log 2>&1
echo EXITCODE %ERRORLEVEL%>> outputs\logs\lp_instruction_v4_mimic_counterfactual_weighted_qwen25_coder_7b_1k_gpu1.log
exit /b %ERRORLEVEL%
