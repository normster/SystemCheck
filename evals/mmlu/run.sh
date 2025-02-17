#!/bin/bash

shopt -s globstar

MODEL_DIR=${1}

MODEL_NAME=$(basename $MODEL_DIR)
SAFE_NAME=$(echo $MODEL_NAME | sha256sum | head -c 10)

mkdir -p results/${SAFE_NAME}
mkdir -p results/${MODEL_NAME}
ln -sf $MODEL_DIR $SAFE_NAME

lm_eval \
    --model hf \
    --model_args pretrained=\"${SAFE_NAME}\",dtype=bfloat16,parallelize=True \
    --tasks mmlu \
    --batch_size auto \
    --output_path results/${SAFE_NAME}

mv results/${SAFE_NAME}/**/results_* results/${MODEL_NAME}

rm $SAFE_NAME
rm -r results/${SAFE_NAME}