#!/bin/bash
# Preprocess reports from all models or specific models
# Usage: 
#   ./scripts/preprocess_all.sh                    # Process all models
#   ./scripts/preprocess_all.sh gpt-5 gemini-pro   # Process specific models

set -e

# Default base path, can be overridden by environment variable
BASE_PATH="${BASE_PATH:-/path/to/model_outputs}"
OUTPUT_DIR="extracted_reports_$(date +%Y%m%d_%H%M%S)"

# Validate BASE_PATH
if [ "$BASE_PATH" = "/path/to/model_outputs" ] || [ ! -d "$BASE_PATH" ]; then
    echo "❌ Error: BASE_PATH is not set or directory does not exist."
    echo "Please set the BASE_PATH environment variable or modify this script."
    echo ""
    echo "Expected structure:"
    echo "  BASE_PATH/"
    echo "    model_name_1/"
    echo "      qid_<qid>_report.md"
    echo "    model_name_2/"
    echo "      qid_<qid>_report.md"
    echo ""
    echo "Example: export BASE_PATH=/export/xgen-small/mas_eval/extracted_model_output"
    exit 1
fi

echo "🔍 Preprocessing reports..."
echo "📁 Base path: ${BASE_PATH}"
echo "💾 Output: ${OUTPUT_DIR}"

# Check if specific models were provided
if [ $# -gt 0 ]; then
    echo "🎯 Models: $@"
    # Run preprocessing for specific models
    python preprocess.py "${BASE_PATH}" \
        -m "$@" \
        -o "${OUTPUT_DIR}" \
        --verbose
else
    echo "🎯 Processing all models in directory"
    # Run preprocessing for all models
    python preprocess.py "${BASE_PATH}" \
        -o "${OUTPUT_DIR}" \
        --verbose
fi

echo ""
echo "✅ Preprocessing complete!"
echo "📄 JSON file created in: ${OUTPUT_DIR}/"
