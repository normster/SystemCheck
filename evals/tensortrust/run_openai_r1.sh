#!/bin/bash

set -x

for SUITE in extraction hijacking helpful; do
    python -m torchllms.inference.generate \
        --input_path inputs/${SUITE}.jsonl \
        --output_path outputs/DeepSeek-V3/${SUITE}.jsonl \
        --provider openai \
        --provider_kwargs "{\"model\": \"deepseek-ai/DeepSeek-V3\", \"concurrency\": 50}" \
        --generate_kwargs "{\"temperature\": 0.0, \"max_tokens\": 500}"

    python -m torchllms.inference.generate \
        --input_path inputs/${SUITE}.jsonl \
        --output_path outputs/DeepSeek-R1/${SUITE}.jsonl \
        --provider openai \
        --provider_kwargs "{\"model\": \"deepseek-ai/DeepSeek-R1\", \"concurrency\": 10}" \
        --generate_kwargs "{\"temperature\": 0.6, \"max_tokens\": 8192, \"top_p\": 0.95}"
done
