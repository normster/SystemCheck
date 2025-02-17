#!/bin/bash

shopt -s globstar

MODEL=${1}

LATEST=$(ls results/${MODEL}/**/results* 2>/dev/null)
if [ -z "$LATEST" ]; then
    echo "missing"
    exit 1
fi
LATEST=$(echo "$LATEST" | tail -n 1)

MEAN=$(jq -r '.results.mmlu."acc,none"' $LATEST)
SE=$(jq -r '.results.mmlu."acc_stderr,none"' $LATEST)

CI_WIDTH=$(echo "$SE * 1.96" | bc -l)
LOWER=$(echo "$MEAN - $CI_WIDTH" | bc -l)
UPPER=$(echo "$MEAN + $CI_WIDTH" | bc -l)

MEAN=$(printf "%.5f" $MEAN)
LOWER=$(printf "%.5f" $LOWER)
UPPER=$(printf "%.5f" $UPPER)

echo -e "${MEAN},(${LOWER}-${UPPER})"
