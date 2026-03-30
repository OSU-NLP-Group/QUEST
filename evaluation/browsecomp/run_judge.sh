#!/usr/bin/env bash
set -euo pipefail

# Configure judge environment here.
# Uncomment and fill in the provider you use.
# export JUDGE_MODEL_NAME="gpt-4o-mini"
# export JUDGE_OPENAI_API_KEY="..."
#
export JUDGE_MODEL_NAME="gpt-4.1-2025-04-14"
export JUDGE_OPENAI_API_KEY="${JUDGE_OPENAI_API_KEY:-your_openai_api_key}"
#
# export JUDGE_MODEL_NAME="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
# export JUDGE_AWS_ACCESS_KEY_ID="..."
# export JUDGE_AWS_SECRET_ACCESS_KEY="..."
# export JUDGE_AWS_REGION_NAME="..."

# Paths and workers
# Define multiple target_dir entries. Add more directories as needed.
TARGET_DIRS=(
  "/fs/scratch/PAS1576/jianxie/DeepResearch/evaluation/datasets/browsecomp/a3b-results/qwen3-moe-base-vanilla-post-training-20260227-2k-20k-traj-plut-drb-5k-5ep-20k-output-80k-memory-400turns/results/deepresearch/browsecomp"
)

# BrowseComp dataset (defaults to the official remote CSV, but a local path can also be used).
export DATASET_PATH="https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
export WORKERS=50

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Process each target_dir in order.
for TARGET_DIR in "${TARGET_DIRS[@]}"; do
  echo "=========================================="
  echo "Processing: $TARGET_DIR"
  echo "=========================================="

  if [ ! -d "$TARGET_DIR" ]; then
    echo "Warning: Directory does not exist, skipping: $TARGET_DIR"
    continue
  fi

  python "$SCRIPT_DIR/eval.py" \
    --target-dir "$TARGET_DIR" \
    --dataset "$DATASET_PATH" \
    --workers "$WORKERS"

  echo ""
done

echo "All target directories processed!"
