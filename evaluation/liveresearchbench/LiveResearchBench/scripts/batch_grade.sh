#!/bin/bash
# Batch grade all models with a single provider
# Usage: ./scripts/batch_grade.sh [run_id]

set -e

RUN_ID=${1:-"run_$(date +%Y%m%d_%H%M%S)"}
OUTPUT_DIR="results/${RUN_ID}"

echo "🚀 Starting batch grading: ${RUN_ID}"
echo "📁 Output directory: ${OUTPUT_DIR}"

# Activate virtual environment (if using venv instead of uv)
# source .venv/bin/activate

# Run batch grading
python main.py \
    --batch \
    --config configs/batch_config.yaml \
    --provider gemini \
    --model gemini-2.5-pro \
    --output-dir "${OUTPUT_DIR}" \
    --max-concurrent 5 \
    --verbose

echo ""
echo "✅ Batch grading complete!"
echo "📊 Results saved to: ${OUTPUT_DIR}"

