#!/bin/bash

set -x

for SUITE in handwritten distractors; do
    python -m torchllms.inference.generate \
        --input_path inputs/${SUITE}.jsonl \
        --output_path outputs/DeepSeek-R1/${SUITE}.jsonl \
        --provider openai \
        --provider_kwargs "{\"model\": \"deepseek-ai/DeepSeek-R1\", \"concurrency\": 10}" \
        --generate_kwargs "{\"temperature\": 0.6, \"max_tokens\": 8192, \"top_p\": 0.95}"

    python -m torchllms.inference.generate \
        --input_path inputs/${SUITE}.jsonl \
        --output_path outputs/DeepSeek-V3/${SUITE}.jsonl \
        --provider openai \
        --provider_kwargs "{\"model\": \"deepseek-ai/DeepSeek-V3\", \"concurrency\": 50}" \
        --generate_kwargs "{\"temperature\": 0.0, \"max_tokens\": 1000}"
done

python evaluate.py \
    --outputs_dir outputs/DeepSeek-R1 \
    --results_dir results/DeepSeek-R1 \
    --judge_model gpt-4o-2024-08-06 \
    --concurrency 100

python evaluate.py \
    --outputs_dir outputs/DeepSeek-V3 \
    --results_dir results/DeepSeek-V3 \
    --judge_model gpt-4o-2024-08-06 \
    --concurrency 100