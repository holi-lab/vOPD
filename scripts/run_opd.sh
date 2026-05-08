#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
NUM_PROCESSES="${NUM_PROCESSES:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAINING_CONFIG="${ROOT_DIR}/configs/training/default.yaml"

accelerate launch \
    --config_file "${ROOT_DIR}/configs/accelerate.yaml" \
    --num_processes ${NUM_PROCESSES} \
    "${ROOT_DIR}/src/opd_train.py" \
    training_config="${TRAINING_CONFIG}"