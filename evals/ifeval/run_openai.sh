#!/bin/bash

set -x

MODEL_NAME=${1}
VARIANT=${2:-sys_ifeval}

cd "$(dirname "$0")"

python -m torchllms.inference.generate \
    --input_path inputs/${VARIANT}.jsonl \
    --output_path outputs/${MODEL_NAME}/${VARIANT}.jsonl \
    --provider openai \
    --provider_kwargs "{\"model\": \"${MODEL_NAME}\", \"concurrency\": 100}" \
    --generate_kwargs "{\"temperature\": 0.0, \"max_completion_tokens\": 1280}"

cd -
