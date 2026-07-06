@echo off
cd /d H:\Xiyao_Wang\021_260129VIVID
set CUDA_VISIBLE_DEVICES=0
call conda run -n vivid python scripts/evaluate_instruction_visual_dependence.py --config configs/instruction_v2_mimic_qwen25_coder_7b_1k.yaml --checkpoint outputs/instruction_runs/v2_mimic_qwen25_coder_7b_1k/checkpoints/best.pt --output outputs/instruction_runs/v2_mimic_qwen25_coder_7b_1k/visual_dependence_best.json --max-samples 796 --batch-size 2 > outputs\logs\instruction_v2_mimic_visual_dependence_best_gpu0.log 2>&1
echo EXITCODE %ERRORLEVEL%>> outputs\logs\instruction_v2_mimic_visual_dependence_best_gpu0.log
exit /b %ERRORLEVEL%
