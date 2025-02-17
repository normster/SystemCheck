#!/bin/bash

set -x

MODEL_PATH=${1}
CONFIG=${2}
VARIANT=${3:-sys_ifeval}
BATCH_SIZE=${4:-1}

MODEL_NAME=$(basename $(dirname $MODEL_PATH))

cd "$(dirname "$0")"

for SCALE in 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2; do
    python -m torchllms.inference.generate \
        --input_path inputs/${VARIANT}.jsonl \
        --output_path outputs/no_system_neg_${MODEL_NAME}_${SCALE}_eager/${VARIANT}.jsonl \
        --provider torchllms \
        --provider_kwargs "{\"model_path\": \"${MODEL_PATH}\", \"max_model_len\": 4096, \"template_config\": \"${CONFIG}\", \"model_kwargs\": {\"attention_impl\": \"eager\"}, \"batched\": true, \"lp_kwargs\": {\"plausibility_threshold\": 0.1, \"guidance_scale\": ${SCALE}, \"type\": \"cfg\"}}" \
        --generate_kwargs "{\"temperature\": 0.0, \"max_new_tokens\": 1280, \"batch_size\": ${BATCH_SIZE}}"
done
cd -
