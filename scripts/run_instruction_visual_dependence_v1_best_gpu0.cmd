@echo off
cd /d H:\Xiyao_Wang\021_260129VIVID
set CUDA_VISIBLE_DEVICES=0
call conda run -n vivid python scripts/evaluate_instruction_visual_dependence.py --config configs/instruction_v1_qwen25_coder_7b_1k.yaml --checkpoint outputs/instruction_runs/v1_qwen25_coder_7b_1k/checkpoints/best.pt --output outputs/instruction_runs/v1_qwen25_coder_7b_1k/visual_dependence_best.json --max-samples 1000 --batch-size 2 --device cuda > outputs/logs/instruction_visual_dependence_v1_best_gpu0.log 2>&1
echo EXITCODE %ERRORLEVEL%>> outputs/logs/instruction_visual_dependence_v1_best_gpu0.log
exit /b %ERRORLEVEL%
