#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

EVAL_CONFIG="${ROOT_DIR}/configs/eval/default.yaml"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

DATASETS="${DATASETS:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CKPT_STEP="${CKPT_STEP:-}"

TEMPERATURE="${TEMPERATURE:-0.6}"

CHECKPOINT_ARGS=()
if [ -n "${CHECKPOINT_DIR:-}" ]; then
    CHECKPOINT_ARGS+=(--checkpoint_dir "${CHECKPOINT_DIR}")
fi
if [ -n "${CKPT_STEP:-}" ]; then
    CHECKPOINT_ARGS+=(--checkpoint_step "${CKPT_STEP}")
fi

nohup python "${ROOT_DIR}/src/evaluate.py" \
    --config "${EVAL_CONFIG}" \
    --datasets "${DATASETS}" \
    --temperature "${TEMPERATURE}" \
    "${CHECKPOINT_ARGS[@]}" \
    "$@" > "${ROOT_DIR}/logs/eval_$(basename $CHECKPOINT_DIR)_step${CKPT_STEP}.log" 2>&1 &
