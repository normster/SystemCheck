#!/bin/bash

set -x

MODEL_PATH=${1}
CONFIG=${2}
BATCH_SIZE=${3}

MODEL_NAME=$(basename $(dirname $MODEL_PATH))

for SUITE in benign basic redteam; do
    python -m llm_rules.scripts.evaluate_batched \
        --provider torchllms \
        --model $MODEL_PATH \
        --model_name ${MODEL_NAME}_eager \
        --model_kwargs template_config=${CONFIG} \
        --model_kwargs attention_impl=eager \
        --model_kwargs batch_size=$BATCH_SIZE \
        --test_suite $SUITE \
        --output_dir logs_use_system_remove_precedence/${SUITE} \
        --system_instructions \
        --remove_precedence_reminders
done
