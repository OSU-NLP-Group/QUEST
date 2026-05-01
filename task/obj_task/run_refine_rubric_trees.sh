#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepresearch

FORMATTED_TRAJ_DIR="${FORMATTED_TRAJ_DIR:-${TRAJ_DIR:-./outputs/objective_trajectories}/formatted}"
REFINED_TRAJ_DIR="${REFINED_TRAJ_DIR:-${FORMATTED_TRAJ_DIR}/refined}"
REFINE_LOG_DIR="${REFINE_LOG_DIR:-${FORMATTED_TRAJ_DIR}/refine_logs}"

export REFINE_MODEL_NAME="${REFINE_MODEL_NAME:-openai/gpt-5.2}"

python "${SCRIPT_DIR}/refine_rubric_trees.py" \
  --input-dir "$FORMATTED_TRAJ_DIR" \
  --output-dir "$REFINED_TRAJ_DIR" \
  --log-dir "$REFINE_LOG_DIR" \
  --model "$REFINE_MODEL_NAME" \
  --workers "${REFINE_WORKERS:-20}" \
  --max-refine-iterations "${REFINE_MAX_ITERATIONS:-3}"
