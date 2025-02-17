#!/bin/bash

set -x

MODEL_PATH=${1}
CONFIG=${2}
VARIANT=${3:-sys_ifeval}
BATCH_SIZE=${4:-1}

MODEL_NAME=$(basename $(dirname $MODEL_PATH))

cd "$(dirname "$0")"

python -m torchllms.inference.generate \
    --input_path inputs/${VARIANT}.jsonl \
    --output_path outputs/${MODEL_NAME}_eager/${VARIANT}.jsonl \
    --provider torchllms \
    --provider_kwargs "{\"model_path\": \"${MODEL_PATH}\", \"max_model_len\": 4096, \"template_config\": \"${CONFIG}\", \"model_kwargs\": {\"attention_impl\": \"eager\"}, \"batched\": true}" \
    --generate_kwargs "{\"temperature\": 0.0, \"max_new_tokens\": 1280, \"batch_size\": ${BATCH_SIZE}}"

cd -
