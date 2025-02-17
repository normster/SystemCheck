#!/bin/bash

set -x

MODEL=${1}
GUARDRAILS=${2:-3}
SAMPLES=${3:-100}

python stresstest.py \
    --outputs_dir outputs/${MODEL} \
    --warmup_rounds 0 \
    --samples ${SAMPLES} \
    --guardrails ${GUARDRAILS} \
    --provider openai \
    --provider_kwargs "{\"model\": \"${MODEL}\", \"concurrency\": 100}" \
    --generate_kwargs "{\"max_completion_tokens\": 2000}"