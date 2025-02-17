#!/bin/bash

set -x

MODEL_PATH=${1}
CONFIG=${2}
BATCH_SIZE=${3:-1}
GUARDRAILS=${4:-3}
SAMPLES=${5:-100}

MODEL_NAME=$(basename $(dirname $MODEL_PATH))

python stresstest.py \
    --outputs_dir outputs/${MODEL_NAME}_eager \
    --warmup_rounds 0 \
    --samples ${SAMPLES} \
    --guardrails ${GUARDRAILS} \
    --provider torchllms \
    --provider_kwargs "{\"model_path\": \"${MODEL_PATH}\", \"max_model_len\": 4096, \"template_config\": \"${CONFIG}\", \"model_kwargs\": {\"attention_impl\": \"eager\"}, \"batched\": true}" \
    --generate_kwargs "{\"temperature\": 0.0, \"max_new_tokens\": 1000, \"batch_size\": ${BATCH_SIZE}}"
