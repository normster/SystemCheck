#!/bin/bash

set -x

MODEL_DIR=${1}

MODEL_NAME=$(basename $MODEL_DIR)

for SUITE in benign basic redteam; do
    python -m llm_rules.scripts.evaluate_batched \
        --provider vllm_batched \
        --model $MODEL_DIR \
        --model_name $MODEL_NAME \
        --test_suite $SUITE \
        --output_dir logs_use_system_remove_precedence/${SUITE} \
        --system_instructions \
        --remove_precedence_reminders \
        --concurrency 100
done