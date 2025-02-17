#!/bin/bash

set -x

MODEL_PATH=${1}
CONFIG=${2}
BATCH_SIZE=${3:-1}

MODEL_NAME=$(basename $(dirname $MODEL_PATH))

cd "$(dirname "$0")"

for SCALE in 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2; do
    for SUITE in handwritten distractors; do
        python -m torchllms.inference.generate \
            --input_path inputs/${SUITE}.jsonl \
            --output_path outputs/no_system_neg_${MODEL_NAME}_${SCALE}_eager/${SUITE}.jsonl \
            --provider torchllms \
            --provider_kwargs "{\"model_path\": \"${MODEL_PATH}\", \"max_model_len\": 4096, \"template_config\": \"${CONFIG}\", \"model_kwargs\": {\"attention_impl\": \"eager\"}, \"batched\": true, \"lp_kwargs\": {\"plausibility_threshold\": 0.1, \"guidance_scale\": ${SCALE}, \"type\": \"cfg\"}}" \
            --generate_kwargs "{\"temperature\": 0.0, \"max_new_tokens\": 1000, \"batch_size\": ${BATCH_SIZE}}"
    done
done

for SCALE in 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2; do
    python evaluate.py \
        --outputs_dir outputs/no_system_neg_${MODEL_NAME}_${SCALE}_eager \
        --results_dir results/no_system_neg_${MODEL_NAME}_${SCALE}_eager \
        --judge_model gpt-4o-2024-08-06 \
        --concurrency 100
done
cd -
