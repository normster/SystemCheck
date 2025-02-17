#!/bin/bash

MODEL=${1}
VARIANT=${2:-sys_ifeval}

python evaluate.py \
    --input_data inputs/${VARIANT}.jsonl \
    --input_response_data outputs/${MODEL}/${VARIANT}.jsonl \
    --instruction_ids_to_skip language:response_language,combination:repeat_prompt \
    --bootstrap 10000
