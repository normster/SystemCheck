#!/bin/bash

set -x

VARIANT=${1:-sys_ifeval}

python -m torchllms.inference.generate \
    --input_path inputs/${VARIANT}.jsonl \
    --output_path outputs/DeepSeek-R1/${VARIANT}.jsonl \
    --provider openai \
    --provider_kwargs "{\"model\": \"deepseek-ai/DeepSeek-R1\", \"concurrency\": 10}" \
    --generate_kwargs "{\"temperature\": 0.6, \"max_tokens\": 8192, \"top_p\": 0.95}"

python -m torchllms.inference.generate \
    --input_path inputs/${VARIANT}.jsonl \
    --output_path outputs/DeepSeek-V3/${VARIANT}.jsonl \
    --provider openai \
    --provider_kwargs "{\"model\": \"deepseek-ai/DeepSeek-V3\", \"concurrency\": 50}" \
    --generate_kwargs "{\"max_completion_tokens\": 2000}"