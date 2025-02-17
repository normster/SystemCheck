#!/bin/bash

set -x

MODEL_PATH=${1}
CONFIG=${2}
BATCH_SIZE=${3:-1}

MODEL_NAME=$(basename $(dirname $MODEL_PATH))

cd "$(dirname "$0")"

for SUITE in benign basic redteam; do
    python -m llm_rules.scripts.evaluate_batched \
        --provider torchllms \
        --model $MODEL_PATH \
        --model_name no_system_neg_${MODEL_NAME}_eager_cd \
        --model_kwargs template_config=${CONFIG} \
        --model_kwargs batch_size=${BATCH_SIZE} \
        --model_kwargs attention_impl=eager \
        --model_kwargs lp_kwargs="{\"type\": \"cd\", \"plausibility_threshold\": 0.1, \"prompt_builder\": \"no_system\"}" \
        --test_suite $SUITE \
        --output_dir logs_use_system_remove_precedence/${SUITE} \
        --system_instructions \
        --remove_precedence_reminders
done

cd -
