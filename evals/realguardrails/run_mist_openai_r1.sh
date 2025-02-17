#!/bin/bash

set -x

GUARDRAILS=${1:-3}
SAMPLES=${2:-100}

python stresstest.py \
    --outputs_dir outputs/DeepSeek-R1 \
    --warmup_rounds 0 \
    --samples ${SAMPLES} \
    --guardrails ${GUARDRAILS} \
    --provider openai \
    --provider_kwargs "{\"model\": \"deepseek-ai/DeepSeek-R1\", \"concurrency\": 10}" \
    --generate_kwargs "{\"temperature\": 0.6, \"max_tokens\": 8192, \"top_p\": 0.95}"

python stresstest.py \
    --outputs_dir outputs/DeepSeek-V3 \
    --warmup_rounds 0 \
    --samples ${SAMPLES} \
    --guardrails ${GUARDRAILS} \
    --provider openai \
    --provider_kwargs "{\"model\": \"deepseek-ai/DeepSeek-V3\", \"concurrency\": 50}" \
    --generate_kwargs "{\"max_completion_tokens\": 2000}"
