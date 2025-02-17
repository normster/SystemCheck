#!/bin/bash

shopt -s globstar

MODEL_PATH=${1}  # path to checkpoint file
CONFIG=${2}      # path to .yaml template config file (expected in torchllms/messages/configs/)

MODEL_DIR=$(dirname $MODEL_PATH)
MODEL_NAME=$(basename $MODEL_DIR)
SAFE_NAME=$(echo $MODEL_NAME| sha256sum | head -c 10)
CKPT=$(basename $MODEL_PATH)

mkdir -p results/${SAFE_NAME}
mkdir -p results/${MODEL_NAME}_eager

ln -sf $MODEL_DIR $SAFE_NAME

lm_eval \
    --model torchllms \
    --model_args base_path=\"${SAFE_NAME}/${CKPT}\",template_config=${CONFIG} \
    --tasks mmlu \
    --batch_size 1 \
    --output_path results/${SAFE_NAME}

mv results/${SAFE_NAME}/**/results_* results/${MODEL_NAME}_eager

rm $SAFE_NAME
rm -r results/${SAFE_NAME}