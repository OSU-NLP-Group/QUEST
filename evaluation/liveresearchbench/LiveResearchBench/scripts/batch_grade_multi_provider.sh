#!/bin/bash
# Batch grade with both providers and average results
# Usage: ./scripts/batch_grade_multi_provider.sh [run_id]

set -e

RUN_ID=${1:-"run_$(date +%Y%m%d_%H%M%S)"}
BASE_OUTPUT_DIR="results/${RUN_ID}"

echo "🚀 Starting multi-provider batch grading: ${RUN_ID}"
echo "📁 Base output directory: ${BASE_OUTPUT_DIR}"

# Step 1: Grade with GPT-5
echo ""
echo "═══════════════════════════════════════════"
echo "Step 1/3: Grading with GPT-5"
echo "═══════════════════════════════════════════"
python main.py \
    --batch \
    --config configs/batch_config.yaml \
    --provider openai \
    --model gpt-5-mini \
    --output-dir "${BASE_OUTPUT_DIR}/gpt5" \
    --max-concurrent 5 \
    --verbose

# Step 2: Grade with Gemini
echo ""
echo "═══════════════════════════════════════════"
echo "Step 2/3: Grading with Gemini 2.5 Pro"
echo "═══════════════════════════════════════════"
python main.py \
    --batch \
    --config configs/batch_config.yaml \
    --provider gemini \
    --model gemini-2.5-pro \
    --output-dir "${BASE_OUTPUT_DIR}/gemini" \
    --max-concurrent 5 \
    --verbose

# Step 3: Average results (if both completed successfully)
echo ""
echo "═══════════════════════════════════════════"
echo "Step 3/3: Averaging results"
echo "═══════════════════════════════════════════"

# Find and average matching JSON files
mkdir -p "${BASE_OUTPUT_DIR}/averaged"

for gpt5_file in "${BASE_OUTPUT_DIR}/gpt5"/*.json; do
    if [ -f "$gpt5_file" ]; then
        filename=$(basename "$gpt5_file")
        # Try to find corresponding gemini file
        gemini_file=$(echo "$gpt5_file" | sed 's|gpt5|gemini|' | sed 's|openai|gemini|' | sed 's|gpt-5-mini|gemini-2.5-pro|')
        
        if [ -f "$gemini_file" ]; then
            output_file="${BASE_OUTPUT_DIR}/averaged/${filename//_graded_*/_averaged}"
            echo "Averaging: $(basename "$gpt5_file") + $(basename "$gemini_file")"
            python average_results.py \
                --input-a "$gpt5_file" \
                --input-b "$gemini_file" \
                --output "$output_file"
        fi
    fi
done

echo ""
echo "✅ Multi-provider batch grading complete!"
echo "📊 Results:"
echo "  - GPT-5:    ${BASE_OUTPUT_DIR}/gpt5/"
echo "  - Gemini:   ${BASE_OUTPUT_DIR}/gemini/"
echo "  - Averaged: ${BASE_OUTPUT_DIR}/averaged/"
