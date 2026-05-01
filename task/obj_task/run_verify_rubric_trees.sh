#!/bin/bash
set -euo pipefail

# Rubric verifier model configuration
export RUBRIC_VERIFIER_AWS_ACCESS_KEY_ID="${RUBRIC_VERIFIER_AWS_ACCESS_KEY_ID:-your_aws_access_key_id}"
export RUBRIC_VERIFIER_AWS_SECRET_ACCESS_KEY="${RUBRIC_VERIFIER_AWS_SECRET_ACCESS_KEY:-your_aws_secret_access_key}"
export RUBRIC_VERIFIER_AWS_REGION_NAME="${RUBRIC_VERIFIER_AWS_REGION_NAME:-us-east-2}"
export RUBRIC_VERIFIER_MODEL_NAME="${RUBRIC_VERIFIER_MODEL_NAME:-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0}"

# Execution paths
MODEL_ID="${RUBRIC_VERIFIER_MODEL_NAME}"
if [ -z "${FORMATTED_TRAJ_DIR:-}" ]; then
  BASE_FORMATTED_TRAJ_DIR="${TRAJ_DIR:-./outputs/objective_trajectories}/formatted"
  if [ -d "${BASE_FORMATTED_TRAJ_DIR}/refined" ]; then
    FORMATTED_TRAJ_DIR="${BASE_FORMATTED_TRAJ_DIR}/refined"
  else
    FORMATTED_TRAJ_DIR="${BASE_FORMATTED_TRAJ_DIR}"
  fi
fi
LOG_DIR="${LOG_DIR:-${FORMATTED_TRAJ_DIR}/verifier/rubric-tree-verifier-logs}"
OUTPUT_FILE="${OUTPUT_FILE:-${FORMATTED_TRAJ_DIR}/verifier/rubrc-tree-verification-results.json}"

mkdir -p "$LOG_DIR"

python verify_rubric_trees.py \
  --folder "$FORMATTED_TRAJ_DIR" \
  --output "$OUTPUT_FILE" \
  --model "$MODEL_ID" \
  --region "${RUBRIC_VERIFIER_AWS_REGION_NAME}" \
  --max-tasks 9999999 \
  --log-dir "$LOG_DIR"
