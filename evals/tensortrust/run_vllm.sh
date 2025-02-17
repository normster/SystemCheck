#!/bin/bash

set -x

MODEL_DIR=${1}
NUM_GPU=${2:-1}

MODEL_NAME=$(basename $MODEL_DIR)

for SUITE in extraction hijacking helpful; do
    python -m torchllms.inference.generate \
        --input_path inputs/${SUITE}.jsonl \
        --output_path outputs/${MODEL_NAME}/${SUITE}.jsonl \
        --provider vllm \
        --provider_kwargs "{\"model_path\": \"${MODEL_DIR}\", \"max_model_len\": 4096, \"tensor_parallel_size\": ${NUM_GPU}}" \
        --generate_kwargs "{\"temperature\": 0.0, \"max_new_tokens\": 500}"
done
