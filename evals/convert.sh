#!/bin/bash

set -x

BASE_DIR=/home/ubuntu/models

MODELS=(
)

for MODEL in ${MODELS[@]}; do
    python -m torchllms.models.checkpoint_converter \
        --ckpt_paths $BASE_DIR/$MODEL/model_final.pth \
        --output_dir $BASE_DIR/$MODEL --to_hf

    cp ../torchllms/messages/configs/llama3_instruct.json $BASE_DIR/$MODEL/tokenizer_config.json
done