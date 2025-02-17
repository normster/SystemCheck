#!/bin/bash

set -x
set -e

# using Haiku 3 and low concurrency limit for this demo
python src/get_queries_anthropic.py \
    --input_file sample_prompts.jsonl \
    --output_file sample_prompts_and_queries.jsonl \
    --max_concurrency 5 \
    --model claude-3-haiku-20240307 \
    --temperature 0

# concurrency bottleneck here due to scrapfly
python src/get_responses_openai.py \
    --input_file sample_prompts_and_queries.jsonl \
    --output_file sample_prompts_and_queries_and_responses.jsonl \
    --max_concurrency 5 \
    --model gpt-4o-mini \
    --temperature 0