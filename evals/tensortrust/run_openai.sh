#!/bin/bash

set -x

MODEL_NAME=${1:-gpt-4o-mini-2024-07-18}

for SUITE in extraction hijacking helpful; do
    python -m torchllms.inference.generate \
        --input_path inputs/${SUITE}.jsonl \
        --output_path outputs/${MODEL_NAME}/${SUITE}.jsonl \
        --provider openai \
        --provider_kwargs "{\"model\": \"${MODEL_NAME}\", \"concurrency\": 100}" \
        --generate_kwargs "{\"temperature\": 0.0, \"max_tokens\": 500}"
done
