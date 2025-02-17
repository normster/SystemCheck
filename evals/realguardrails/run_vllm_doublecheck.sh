#!/bin/bash

set -x

MODEL_DIR=${1}
NUM_GPU=${2:-1}

MODEL_NAME=$(basename $MODEL_DIR)

for SUITE in handwritten distractors; do
    python -m torchllms.inference.generate \
        --input_path inputs/${SUITE}.jsonl \
        --output_path outputs/${MODEL_NAME}_doublecheck/${SUITE}.jsonl \
        --provider vllm_doublecheck \
        --provider_kwargs "{\"model_path\": \"${MODEL_DIR}\", \"max_model_len\": 4096, \"tensor_parallel_size\": ${NUM_GPU}}" \
        --generate_kwargs "{\"temperature\": 0.0, \"max_tokens\": 1000}"
done

python evaluate.py \
    --outputs_dir outputs/${MODEL_NAME}_doublecheck \
    --results_dir results/${MODEL_NAME}_doublecheck \
    --judge_model gpt-4o-2024-08-06 \
    --concurrency 100
