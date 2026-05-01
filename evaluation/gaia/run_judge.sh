#!/usr/bin/env bash
set -euo pipefail

# Configure judge environment here.
# Uncomment and fill in the provider you use.
# export JUDGE_MODEL_NAME="gpt-4o-mini"
# export JUDGE_OPENAI_API_KEY="..."
#
# export JUDGE_MODEL_NAME="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
# export JUDGE_AWS_ACCESS_KEY_ID="..."
# export JUDGE_AWS_SECRET_ACCESS_KEY="..."
# export JUDGE_AWS_REGION_NAME="..."

TARGET_DIRS=(
  "your_output_dir/results/deepresearch/gaia-text-only-103"
)


export DATASET_PATH="gaia-103-org.json" # please unzip gaia-103-org.zip, password: 8sK9pR2xQ7bT5gA3
export WORKERS=150

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for TARGET_DIR in "${TARGET_DIRS[@]}"; do
  echo "=========================================="
  echo "Processing: $TARGET_DIR"
  echo "=========================================="

  if [ ! -d "$TARGET_DIR" ]; then
    echo "Warning: Directory does not exist, skipping: $TARGET_DIR"
    continue
  fi

  python "$SCRIPT_DIR/judge.py" \
    --target-dir "$TARGET_DIR" \
    --dataset "$DATASET_PATH" \
    --workers "$WORKERS"

  echo ""
done

echo "All target directories processed!"
