@echo off
cd /d H:\Xiyao_Wang\021_260129VIVID
python scripts\generate_clinical_instructions.py --mode glm --input data\instructions\manifests\mimic_report_train_1k.jsonl --output data\instructions\raw\v3_mimic_report_grounded_counterfactual_train1k\shard0.jsonl --api-log outputs\instruction_generation\v3_mimic_report_grounded_counterfactual_train1k\train_shard0_api_log.jsonl --parse-errors outputs\instruction_generation\v3_mimic_report_grounded_counterfactual_train1k\train_shard0_parse_errors.jsonl --prompt prompts\glm_instruction_generation_report_grounded_v3_counterfactual.txt --version v3_mimic_report_grounded_counterfactual --api-key-env ZHIPU_API_KEY --max-samples 250 --skip-samples 0 --max-instructions-per-sample 4 --temperature 0.1 --timeout 120 --stream-output --resume > outputs\logs\mimic_v3_glm_train_shard0.log 2>&1
echo EXITCODE %ERRORLEVEL%>> outputs\logs\mimic_v3_glm_train_shard0.log
exit /b %ERRORLEVEL%
