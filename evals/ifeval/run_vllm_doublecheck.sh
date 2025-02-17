#!/bin/bash

set -x

MODEL_DIR=${1}
VARIANT=${2:-sys_ifeval}
NUM_GPU=${3:-1}

MODEL_NAME=$(basename $MODEL_DIR)

cd "$(dirname "$0")"

python -m torchllms.inference.generate \
    --input_path inputs/${VARIANT}.jsonl \
    --output_path outputs/${MODEL_NAME}_doublecheck/${VARIANT}.jsonl \
    --provider vllm_doublecheck \
    --provider_kwargs "{\"model_path\": \"${MODEL_DIR}\", \"max_model_len\": 4096, \"tensor_parallel_size\": ${NUM_GPU}}" \
    --generate_kwargs "{\"temperature\": 0.0, \"max_new_tokens\": 1280}"

cd -
