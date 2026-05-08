#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

EVAL_CONFIG="${ROOT_DIR}/configs/eval/default.yaml"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

DATASETS="${DATASETS:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
TEMPERATURE="${TEMPERATURE:-0.6}"

nohup python "${ROOT_DIR}/src/evaluate.py" \
    --config "${EVAL_CONFIG}" \
    --all_checkpoints \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --datasets "${DATASETS}" \
    --temperature "${TEMPERATURE}" \
    "$@" > "${ROOT_DIR}/logs/eval_$(basename $CHECKPOINT_DIR).log" 2>&1 &
