#!/bin/bash

set -x

MODEL_PATH=${1}
CONFIG=${2}
BATCH_SIZE=${3:-1}

MODEL_NAME=$(basename $(dirname $MODEL_PATH))

for SUITE in extraction hijacking helpful; do
    python -m torchllms.inference.generate \
        --input_path inputs/${SUITE}.jsonl \
        --output_path outputs/${MODEL_NAME}_eager/${SUITE}.jsonl \
        --provider torchllms \
        --provider_kwargs "{\"model_path\": \"${MODEL_PATH}\", \"max_model_len\": 4096, \"template_config\": \"${CONFIG}\", \"model_kwargs\": {\"attention_impl\": \"eager\"}, \"batched\": true}" \
        --generate_kwargs "{\"temperature\": 0.0, \"max_new_tokens\": 500, \"batch_size\": ${BATCH_SIZE}}"
done
