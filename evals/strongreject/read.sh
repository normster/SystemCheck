#!/usr/bin/env bash
# Evaluate StrongREJECT responses using the 3-class judge.
#
# Usage:
#   bash SystemCheck/evals/strongreject/read.sh <model_name> [judge_model] [bootstrap]
#
# Examples:
#   bash SystemCheck/evals/strongreject/read.sh gpt-oss-20b
#   bash SystemCheck/evals/strongreject/read.sh gpt-oss-20b gemini-2.5-pro 1000
#
# Notes on "missing" responses:
#   During generation (run.sh), responses that hit the max_tokens limit or
#   error out are recorded as "missing". This is most common with rot_13
#   (model burns tokens decoding ROT13) and cot_forgery (reasoning loops).
#   These are sent to the judge and typically classified as REFUSAL, which
#   is conservative (undercounts ASR). Check the run.sh logs for
#   "[OpenAI] WARNING: N/M responses are 'missing'" to see how many.
#   The judge may also return "missing" if Gemini's content_filter blocks
#   the judge request (logged as finish_reason=content_filter).

set -euo pipefail

MODEL_NAME=${1:?Usage: read.sh <model_name> [judge_model] [bootstrap]}
JUDGE_MODEL=${2:-"gemini-2.5-pro"}
BOOTSTRAP=${3:-0}

# File layout:
#   SystemCheck/evals/strongreject/outputs/<model>/<attack>.jsonl  <- input (from run.sh)
#   SystemCheck/evals/strongreject/results/<model>/<attack>.jsonl  <- judged records (cached)
#   SystemCheck/evals/strongreject/results/<model>/summary.json    <- final scores + breakdown
EVAL_DIR="SystemCheck/evals/strongreject"
OUTPUT_DIR="${EVAL_DIR}/outputs/${MODEL_NAME}"
RESULTS_DIR="${EVAL_DIR}/results/${MODEL_NAME}"

if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: No outputs found at ${OUTPUT_DIR}"
    echo "Run run.sh first to generate model responses."
    exit 1
fi

# Reads:  outputs/<model>/<attack>.jsonl
# Writes: results/<model>/<attack>.jsonl (per-record judgements, cached)
#         results/<model>/summary.json   (ASR scores + classification breakdown)
echo "=== Evaluating ${MODEL_NAME} with ${JUDGE_MODEL} judge ==="

uv run python -m SystemCheck.evals.strongreject.evaluate \
    --outputs_dir "${OUTPUT_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --judge_model "${JUDGE_MODEL}" \
    --bootstrap "${BOOTSTRAP}"
