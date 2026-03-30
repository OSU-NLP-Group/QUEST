#!/bin/bash

# =============================================================================
# DeepResearch Benchmark - Multi-Model Runner
# =============================================================================
# This script allows you to easily run benchmarks with different LLM providers
# Uncomment the model configuration section you want to use
# =============================================================================

# Target model to evaluate
TARGET_MODEL="claude-3-7-sonnet-latest"

# Common parameters
RAW_DATA_DIR="data/test_data/raw_data"
CLEANED_DATA_DIR="data/test_data/cleaned_data"
OUTPUT_DIR="results"
QUERY_DATA_PATH="data/prompt_data/query.jsonl"
MAX_WORKERS=5

# Optional flags (uncomment to enable)
# LIMIT="--limit 2"
# SKIP_CLEANING="--skip_cleaning"
# ONLY_ZH="--only_zh"
# ONLY_EN="--only_en"
# FORCE="--force"

# =============================================================================
# MODEL CONFIGURATIONS - Uncomment ONE section below
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Google Gemini (Default)
# -----------------------------------------------------------------------------
export GEMINI_API_KEY="${GEMINI_API_KEY:-your-gemini-api-key}"
EVAL_MODEL="gemini/gemini-2.5-pro-preview-06-05"
CLEAN_MODEL="gemini/gemini-2.5-flash-preview-05-20"
echo "Using Google Gemini models"

# -----------------------------------------------------------------------------
# 2. OpenAI GPT
# -----------------------------------------------------------------------------
# export OPENAI_API_KEY="${OPENAI_API_KEY:-your-openai-api-key}"
# EVAL_MODEL="gpt-4"
# CLEAN_MODEL="gpt-3.5-turbo"
# echo "Using OpenAI models"

# -----------------------------------------------------------------------------
# 3. OpenAI GPT-4 Turbo
# -----------------------------------------------------------------------------
# export OPENAI_API_KEY="${OPENAI_API_KEY:-your-openai-api-key}"
# EVAL_MODEL="gpt-4-turbo"
# CLEAN_MODEL="gpt-3.5-turbo"
# echo "Using OpenAI GPT-4 Turbo"

# -----------------------------------------------------------------------------
# 4. Anthropic Claude
# -----------------------------------------------------------------------------
# export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-your-anthropic-api-key}"
# EVAL_MODEL="claude-3-opus-20240229"
# CLEAN_MODEL="claude-3-sonnet-20240229"
# echo "Using Anthropic Claude models"

# -----------------------------------------------------------------------------
# 5. DeepSeek
# -----------------------------------------------------------------------------
# export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-your-deepseek-api-key}"
# EVAL_MODEL="deepseek/deepseek-chat"
# CLEAN_MODEL="deepseek/deepseek-chat"
# echo "Using DeepSeek models"

# -----------------------------------------------------------------------------
# 6. Azure OpenAI
# -----------------------------------------------------------------------------
# export AZURE_API_KEY="your-azure-api-key"
# export AZURE_API_BASE="https://your-resource.openai.azure.com/"
# export AZURE_API_VERSION="2024-02-15-preview"
# EVAL_MODEL="azure/gpt-4-deployment"
# CLEAN_MODEL="azure/gpt-35-turbo-deployment"
# echo "Using Azure OpenAI models"

# -----------------------------------------------------------------------------
# 7. AWS Bedrock
# -----------------------------------------------------------------------------
# export AWS_ACCESS_KEY_ID="your-aws-access-key-id"
# export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
# export AWS_REGION_NAME="us-east-1"
# EVAL_MODEL="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
# CLEAN_MODEL="bedrock/anthropic.claude-3-haiku-20240307-v1:0"
# echo "Using AWS Bedrock models"

# -----------------------------------------------------------------------------
# 8. Mixed Providers (e.g., Claude for eval, GPT for cleaning)
# -----------------------------------------------------------------------------
# export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-your-anthropic-api-key}"
# export OPENAI_API_KEY="${OPENAI_API_KEY:-your-openai-api-key}"
# EVAL_MODEL="claude-3-opus-20240229"
# CLEAN_MODEL="gpt-3.5-turbo"
# echo "Using mixed providers: Claude for eval, OpenAI for cleaning"

# =============================================================================
# Run Benchmark
# =============================================================================

echo "========================================="
echo "DeepResearch Benchmark Configuration"
echo "========================================="
echo "Target Model:      $TARGET_MODEL"
echo "Evaluation Model:  $EVAL_MODEL"
echo "Cleaning Model:    $CLEAN_MODEL"
echo "Max Workers:       $MAX_WORKERS"
echo "Output Directory:  $OUTPUT_DIR"
echo "========================================="

# Create output directory
RACE_OUTPUT="$OUTPUT_DIR/race/$TARGET_MODEL"
mkdir -p "$RACE_OUTPUT"

# Build command
PYTHON_CMD="python -u deepresearch_bench_race.py \"$TARGET_MODEL\" \
    --raw_data_dir $RAW_DATA_DIR \
    --cleaned_data_dir $CLEANED_DATA_DIR \
    --max_workers $MAX_WORKERS \
    --query_file $QUERY_DATA_PATH \
    --output_dir $RACE_OUTPUT \
    --eval_model $EVAL_MODEL \
    --clean_model $CLEAN_MODEL"

# Add optional parameters
if [[ -n "$LIMIT" ]]; then
    PYTHON_CMD="$PYTHON_CMD $LIMIT"
    echo "Limit:             ${LIMIT#--limit }"
fi

if [[ -n "$SKIP_CLEANING" ]]; then
    PYTHON_CMD="$PYTHON_CMD $SKIP_CLEANING"
    echo "Skip Cleaning:     Yes"
fi

if [[ -n "$ONLY_ZH" ]]; then
    PYTHON_CMD="$PYTHON_CMD $ONLY_ZH"
    echo "Language:          Chinese only"
fi

if [[ -n "$ONLY_EN" ]]; then
    PYTHON_CMD="$PYTHON_CMD $ONLY_EN"
    echo "Language:          English only"
fi

if [[ -n "$FORCE" ]]; then
    PYTHON_CMD="$PYTHON_CMD $FORCE"
    echo "Force:             Yes"
fi

echo "========================================="
echo ""

# Execute
echo "Starting benchmark..."
eval $PYTHON_CMD

echo ""
echo "========================================="
echo "Benchmark completed!"
echo "Results saved to: $RACE_OUTPUT"
echo "========================================="
