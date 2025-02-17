#!/bin/bash

MODEL=${1}

python evaluate.py \
    --results_dir results/${MODEL} \
    --judge_model gpt-4o-2024-08-06 \
    --concurrency 100 \
    --bootstrap 10000
