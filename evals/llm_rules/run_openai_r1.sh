#!/bin/bash

set -x

for SUITE in benign basic redteam; do
    python -m llm_rules.scripts.evaluate \
        --provider openai \
        --model deepseek-ai/DeepSeek-R1 \
        --model_name DeepSeek-R1 \
        --test_suite $SUITE \
        --output_dir logs_use_system_remove_precedence/${SUITE} \
        --system_instructions \
        --remove_precedence_reminders \
        --temperature 0.6 \
        --model_kwargs max_tokens=8192 \
        --model_kwargs top_p=0.95 \
        --concurrency 10

    python -m llm_rules.scripts.evaluate \
        --provider openai \
        --model deepseek-ai/DeepSeek-V3 \
        --model_name DeepSeek-V3 \
        --test_suite $SUITE \
        --output_dir logs_use_system_remove_precedence/${SUITE} \
        --system_instructions \
        --remove_precedence_reminders \
        --concurrency 50
done