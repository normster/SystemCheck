#!/bin/bash

MODEL=${1}
GUARDRAILS=${2:-3}
SAMPLES=${3:-100}

python stresstest.py \
    --outputs_dir outputs/${MODEL} \
    --guardrails ${GUARDRAILS} \
    --samples ${SAMPLES} \
    --bootstrap 10000
