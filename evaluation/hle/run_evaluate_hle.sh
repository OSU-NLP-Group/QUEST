#!/usr/bin/env bash
set -euo pipefail

# evaluate_hle.py uses the standard OpenAI API configuration.
export API_KEY="${API_KEY:-your_openai_api_key}"
export BASE_URL="${BASE_URL:-}"

# If unset, you can assign values here locally. Do not commit real keys.
# export API_KEY="sk-..."
# export BASE_URL="https://..."

# Paths and worker count
TARGET_DIRS=(
  "//fs/scratch/PAS1576/jianxie/DeepResearch/evaluation/datasets/hle/a3b-results/qwen3-moe-rl-drb-5k-5ep-python-tool-20k-output-32k-memory-200turns/results/deepresearch/hle_text_only_130"
)

DATASET_PATH="/fs/scratch/PAS1576/jianxie/DeepResearch/evaluation/datasets/hle/hle_text_only_130.jsonl"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKERS=100

# Process each target_dir in order
for TARGET_DIR in "${TARGET_DIRS[@]}"; do
  echo "=========================================="
  echo "Processing: $TARGET_DIR"
  echo "=========================================="

  if [ ! -d "$TARGET_DIR" ]; then
    echo "Warning: Directory does not exist, skipping: $TARGET_DIR"
    continue
  fi

  python "$SCRIPT_DIR/evaluate_hle.py" \
    --target-dir "$TARGET_DIR" \
    --dataset "$DATASET_PATH" \
    --workers "$WORKERS"
  # Add --quiet for less logging.

  echo ""
done

echo "All target directories processed!"
