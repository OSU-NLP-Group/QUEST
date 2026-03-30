#!/bin/bash

# =============================================================================
# Configuration examples - do not run this file directly.
# =============================================================================
# This file contains example configurations for different model providers.
# Copy the configuration you need into run_with_models.sh or use it directly on the command line.
# =============================================================================

# -----------------------------------------------------------------------------
# Example 1: Google Gemini (default configuration)
# -----------------------------------------------------------------------------
# Best for: general evaluation with a balanced quality/cost tradeoff
# Cost: medium
export GEMINI_API_KEY="your-gemini-api-key"
EVAL_MODEL="gemini/gemini-2.5-pro-preview-06-05"
CLEAN_MODEL="gemini/gemini-2.5-flash-preview-05-20"

# Run command:
# python deepresearch_bench_race.py "model-name" \
#     --eval_model "$EVAL_MODEL" \
#     --clean_model "$CLEAN_MODEL"

# -----------------------------------------------------------------------------
# Example 2: OpenAI GPT-4 (high accuracy)
# -----------------------------------------------------------------------------
# Best for: production environments that need the highest-quality evaluation
# Cost: high
export OPENAI_API_KEY="your-openai-api-key"
EVAL_MODEL="gpt-4"
CLEAN_MODEL="gpt-3.5-turbo"

# Run command:
# python deepresearch_bench_race.py "model-name" \
#     --eval_model "gpt-4" \
#     --clean_model "gpt-3.5-turbo"

# -----------------------------------------------------------------------------
# Example 3: OpenAI GPT-3.5 (cost-optimized)
# -----------------------------------------------------------------------------
# Best for: development testing and fast iteration
# Cost: low
export OPENAI_API_KEY="your-openai-api-key"
EVAL_MODEL="gpt-3.5-turbo"
CLEAN_MODEL="gpt-3.5-turbo"

# Run command:
# python deepresearch_bench_race.py "model-name" \
#     --eval_model "gpt-3.5-turbo" \
#     --clean_model "gpt-3.5-turbo" \
#     --limit 10

# -----------------------------------------------------------------------------
# Example 4: Anthropic Claude Opus (highest quality)
# -----------------------------------------------------------------------------
# Best for: research-grade evaluation requiring the deepest analysis
# Cost: highest
export ANTHROPIC_API_KEY="your-anthropic-api-key"
EVAL_MODEL="claude-3-opus-20240229"
CLEAN_MODEL="claude-3-sonnet-20240229"

# Run command:
# python deepresearch_bench_race.py "model-name" \
#     --eval_model "claude-3-opus-20240229" \
#     --clean_model "claude-3-sonnet-20240229"

# -----------------------------------------------------------------------------
# Example 5: DeepSeek (ultra-low cost)
# -----------------------------------------------------------------------------
# Best for: large-scale experiments with tight budgets
# Cost: very low
export DEEPSEEK_API_KEY="your-deepseek-api-key"
EVAL_MODEL="deepseek/deepseek-chat"
CLEAN_MODEL="deepseek/deepseek-chat"

# Run command:
# python deepresearch_bench_race.py "model-name" \
#     --eval_model "deepseek/deepseek-chat" \
#     --clean_model "deepseek/deepseek-chat"

# -----------------------------------------------------------------------------
# Example 6: Azure OpenAI (enterprise deployment)
# -----------------------------------------------------------------------------
# Best for: enterprise environments that need data sovereignty and compliance
# Cost: medium-high
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"
# Note: use your actual deployment names.
EVAL_MODEL="azure/gpt-4-deployment"
CLEAN_MODEL="azure/gpt-35-turbo-deployment"

# Run command:
# python deepresearch_bench_race.py "model-name" \
#     --eval_model "azure/gpt-4-deployment" \
#     --clean_model "azure/gpt-35-turbo-deployment"

# -----------------------------------------------------------------------------
# Example 7: AWS Bedrock (enterprise deployment)
# -----------------------------------------------------------------------------
# Best for: AWS ecosystems that need enterprise-grade support
# Cost: medium-high
export AWS_ACCESS_KEY_ID="your-aws-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export AWS_REGION_NAME="us-east-1"
EVAL_MODEL="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
CLEAN_MODEL="bedrock/anthropic.claude-3-haiku-20240307-v1:0"

# Run command:
# python deepresearch_bench_race.py "model-name" \
#     --eval_model "bedrock/anthropic.claude-3-sonnet-20240229-v1:0" \
#     --clean_model "bedrock/anthropic.claude-3-haiku-20240307-v1:0"

# -----------------------------------------------------------------------------
# Example 8: Mixed configuration (Claude for evaluation + GPT for cleaning)
# -----------------------------------------------------------------------------
# Best for: combining the strengths of different models
# Cost: medium
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export OPENAI_API_KEY="your-openai-api-key"
EVAL_MODEL="claude-3-opus-20240229"
CLEAN_MODEL="gpt-3.5-turbo"

# Run command:
# python deepresearch_bench_race.py "model-name" \
#     --eval_model "claude-3-opus-20240229" \
#     --clean_model "gpt-3.5-turbo"

# =============================================================================
# Common parameter combinations
# =============================================================================

# Quick test (2 tasks)
# --limit 2

# Process Chinese only
# --only_zh

# Process English only
# --only_en

# Skip the cleaning step
# --skip_cleaning

# Force re-evaluation
# --force

# Reduce concurrency (to avoid rate limits)
# --max_workers 2

# Increase concurrency (to finish faster)
# --max_workers 10

# =============================================================================
# Full example commands
# =============================================================================

# Full example 1: high-quality production evaluation
# export OPENAI_API_KEY="your-key"
# python deepresearch_bench_race.py "claude-3-7-sonnet-latest" \
#     --eval_model "gpt-4" \
#     --clean_model "gpt-3.5-turbo" \
#     --max_workers 5

# Full example 2: quick development test
# export OPENAI_API_KEY="your-key"
# python deepresearch_bench_race.py "test-model" \
#     --eval_model "gpt-3.5-turbo" \
#     --clean_model "gpt-3.5-turbo" \
#     --limit 5 \
#     --only_en

# Full example 3: cost-optimized Chinese evaluation
# export DEEPSEEK_API_KEY="your-key"
# python deepresearch_bench_race.py "model-name" \
#     --eval_model "deepseek/deepseek-chat" \
#     --clean_model "deepseek/deepseek-chat" \
#     --only_zh \
#     --max_workers 3

# Full example 4: full enterprise evaluation
# export AZURE_API_KEY="your-key"
# export AZURE_API_BASE="https://your-resource.openai.azure.com/"
# export AZURE_API_VERSION="2024-02-15-preview"
# python deepresearch_bench_race.py "production-model" \
#     --eval_model "azure/gpt-4-deployment" \
#     --clean_model "azure/gpt-35-turbo-deployment" \
#     --max_workers 10 \
#     --force

# =============================================================================
# Cost estimate reference (for guidance only; actual prices may change)
# =============================================================================

# GPT-4:              ~$0.03/1K tokens (input), ~$0.06/1K tokens (output)
# GPT-3.5-turbo:      ~$0.0005/1K tokens (input), ~$0.0015/1K tokens (output)
# Claude-3-opus:      ~$0.015/1K tokens (input), ~$0.075/1K tokens (output)
# Claude-3-sonnet:    ~$0.003/1K tokens (input), ~$0.015/1K tokens (output)
# Gemini Pro:         similar to GPT-4
# Gemini Flash:       similar to GPT-3.5
# DeepSeek:           ~$0.0001/1K tokens (very low cost)

# Recommendations:
# - Development testing: use GPT-3.5, Gemini Flash, or DeepSeek
# - Production evaluation: use GPT-4, Claude-3-opus, or Gemini Pro
# - Cleaning tasks: always use a cheaper model (GPT-3.5 or Gemini Flash)

echo "This is a configuration example file. Do not run it directly."
echo "Copy the configuration you need into run_with_models.sh or use it on the command line."
