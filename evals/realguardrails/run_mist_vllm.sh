#!/bin/bash

set -x

MODEL_DIR=${1}
NUM_GPU=${2:-1}
GUARDRAILS=${3:-3}
SAMPLES=${4:-100}

MODEL_NAME=$(basename $MODEL_DIR)

python stresstest.py \
    --outputs_dir outputs/${MODEL_NAME} \
    --warmup_rounds 0 \
    --samples ${SAMPLES} \
    --guardrails ${GUARDRAILS} \
    --provider vllm \
    --provider_kwargs "{\"model_path\": \"${MODEL_DIR}\", \"max_model_len\": 4096, \"tensor_parallel_size\": ${NUM_GPU}}" \
    --generate_kwargs "{\"temperature\": 0.0, \"max_tokens\": 1000}"