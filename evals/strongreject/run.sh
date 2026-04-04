#!/usr/bin/env bash
# Run StrongREJECT evaluation pipeline for a given model.
#
# Step 1: Prepare inputs (apply attacks to StrongREJECT prompts)
# Step 2: Generate model responses via torchllms.inference.generate
#
# Usage:
#   # With vLLM (API server):
#   bash SystemCheck/evals/strongreject/run.sh vllm gpt-oss-20b \
#       '{"model_path": "/path/to/gpt-oss-20b"}' \
#       '{"temperature": 0, "max_tokens": 1024}'
#
#   # With torchllms (native inference):
#   bash SystemCheck/evals/strongreject/run.sh torchllms gpt-oss-20b \
#       '{"model_path": "/path/to/gpt-oss-20b", "max_model_len": 4096}' \
#       '{"temperature": 0, "max_new_tokens": 1024}'
#
#   # With OpenAI API:
#   bash SystemCheck/evals/strongreject/run.sh openai gpt-4o-mini \
#       '{"model": "gpt-4o-mini"}' \
#       '{"temperature": 0, "max_tokens": 1024}'
#
#   # Specific attacks only:
#   ATTACKS="none,base64,cot_forgery" bash SystemCheck/evals/strongreject/run.sh ...

set -euo pipefail

PROVIDER=${1:?Usage: run.sh <provider> <model_name> [provider_kwargs] [generate_kwargs]}
MODEL_NAME=${2:?Usage: run.sh <provider> <model_name> [provider_kwargs] [generate_kwargs]}
PROVIDER_KWARGS=${3:-'{}'}
GENERATE_KWARGS=${4:-'{"temperature": 0, "max_tokens": 1024}'}

# File layout:
#   SystemCheck/evals/strongreject/cot_forgeries.json                          <- Step 0: pre-generated forgeries (Gemini)
#   SystemCheck/evals/strongreject/inputs/*.jsonl     <- Step 1: attack prompts (one file per attack)
#   SystemCheck/evals/strongreject/outputs/<model>/*.jsonl <- Step 2: model responses (one file per attack)
EVAL_DIR="SystemCheck/evals/strongreject"
INPUT_DIR="${EVAL_DIR}/inputs"
OUTPUT_DIR="${EVAL_DIR}/outputs/${MODEL_NAME}"
FORGERIES_PATH="SystemCheck/evals/strongreject/cot_forgeries.json"

# Optional: restrict to specific attacks
ATTACKS=${ATTACKS:-""}

mkdir -p "${OUTPUT_DIR}"

# Step 0 (optional): Generate CoT forgeries with Gemini 2.5 Pro via Google 1P API.
# Reads:  SystemCheck/evals/strongreject/strongreject_dataset.csv
# Writes: SystemCheck/evals/strongreject/cot_forgeries.json
# uv run python -m SystemCheck.evals.strongreject.generate_forgeries \
#     --output "${FORGERIES_PATH}" \
#     --model gemini-2.5-pro \
#     --concurrency 20

# Step 1: Prepare inputs (apply attacks to StrongREJECT prompts)
# Reads:  SystemCheck/evals/strongreject/strongreject_dataset.csv, SystemCheck/evals/strongreject/cot_forgeries.json
# Writes: SystemCheck/evals/strongreject/inputs/{none,cot_forgery,...}.jsonl
if [ ! -f "${INPUT_DIR}/none.jsonl" ]; then
    echo "=== Preparing inputs ==="
    PREPARE_ARGS="--output_dir ${INPUT_DIR}"
    if [ -f "${FORGERIES_PATH}" ]; then
        PREPARE_ARGS="${PREPARE_ARGS} --forgeries_path ${FORGERIES_PATH}"
    fi
    if [ -n "${ATTACKS}" ]; then
        PREPARE_ARGS="${PREPARE_ARGS} --attacks ${ATTACKS}"
    fi
    uv run python -m SystemCheck.evals.strongreject.prepare_inputs ${PREPARE_ARGS}
fi

# Step 2: Generate model responses for each attack
# Reads:  SystemCheck/evals/strongreject/inputs/<attack>.jsonl
# Writes: SystemCheck/evals/strongreject/outputs/<model>/<attack>.jsonl
echo "=== Generating responses with ${MODEL_NAME} (${PROVIDER}) ==="

for INPUT_FILE in "${INPUT_DIR}"/*.jsonl; do
    ATTACK_NAME=$(basename "${INPUT_FILE}" .jsonl)

    # Skip combined file — we process individual attack files
    [ "${ATTACK_NAME}" = "all_attacks" ] && continue

    # If ATTACKS is set, only process listed attacks
    if [ -n "${ATTACKS}" ]; then
        if ! echo ",${ATTACKS}," | grep -q ",${ATTACK_NAME},"; then
            continue
        fi
    fi

    OUTPUT_FILE="${OUTPUT_DIR}/${ATTACK_NAME}.jsonl"

    if [ -f "${OUTPUT_FILE}" ]; then
        echo "  ${ATTACK_NAME}: already exists, skipping"
        continue
    fi

    echo "  ${ATTACK_NAME}: generating..."
    uv run python -m torchllms.inference.generate \
        --input_path "${INPUT_FILE}" \
        --output_path "${OUTPUT_FILE}" \
        --provider "${PROVIDER}" \
        --provider_kwargs "${PROVIDER_KWARGS}" \
        --generate_kwargs "${GENERATE_KWARGS}"
done

echo "=== Generation complete. Outputs in ${OUTPUT_DIR} ==="
echo "Next: bash SystemCheck/evals/strongreject/read.sh ${MODEL_NAME}"
