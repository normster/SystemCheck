#!/bin/bash

set -x

MODEL_NAME=${1}

for SUITE in benign basic redteam; do
    python -m llm_rules.scripts.evaluate \
        --provider openai \
        --model $MODEL_NAME \
        --test_suite $SUITE \
        --output_dir logs_use_system_remove_precedence/${SUITE} \
        --system_instructions \
        --remove_precedence_reminders \
        --concurrency 100
done