@echo off
cd /d H:\Xiyao_Wang\021_260129VIVID
set PYTHONUNBUFFERED=1
call python scripts\generate_clinical_instructions.py --mode glm --api-key-env ZHIPU_API_KEY --prompt prompts\glm_instruction_generation_report_grounded_v2.txt --version v2_mimic_report_grounded --input data\instructions\manifests\mimic_report_train_1k.jsonl --output data\instructions\raw\v2_mimic_report_grounded_train1k\shard1.jsonl --api-log outputs\instruction_generation\v2_mimic_report_grounded_train1k\api_log_shard1.jsonl --parse-errors outputs\instruction_generation\v2_mimic_report_grounded_train1k\parse_errors_shard1.jsonl --skip-samples 250 --max-samples 250 --max-instructions-per-sample 4 --temperature 0.1 --timeout 180 --sleep 0.2 --stream-output --resume > outputs\logs\mimic_v2_glm_train_shard1.log 2>&1
echo EXITCODE %ERRORLEVEL%>> outputs\logs\mimic_v2_glm_train_shard1.log
exit /b %ERRORLEVEL%
