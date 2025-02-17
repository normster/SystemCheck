#!/bin/bash

MODEL=${1}

python -m llm_rules.scripts.read_scores --model_name $MODEL --log_dir logs_use_system_remove_precedence --bootstrap 10000
# python -m llm_rules.scripts.read_scores --model_name $MODEL --log_dir logs_use_system --bootstrap 10000
# python -m llm_rules.scripts.read_scores --model_name $MODEL --log_dir logs --bootstrap 10000
