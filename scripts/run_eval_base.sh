#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

EVAL_CONFIG="${ROOT_DIR}/configs/eval/default.yaml"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

DATASETS="${DATASETS:-}"
TEMPERATURE="${TEMPERATURE:-0.6}"
BASE_MODEL="${BASE_MODEL:-}"

python "${ROOT_DIR}/src/evaluate.py" \
    --config "${EVAL_CONFIG}" \
    --datasets "${DATASETS}" \
    --temperature "${TEMPERATURE}" \
    --base_model "${BASE_MODEL}" \
    "$@"
