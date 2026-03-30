#!/bin/bash

# =============================================================================
# Quick Model Configuration Test Script
# =============================================================================
# This script tests that your model configurations are working correctly
# Run this before starting the full benchmark
# =============================================================================

set -e  # Exit on error

echo "========================================="
echo "Model Configuration Test"
echo "========================================="
echo ""

# Test script path
TEST_SCRIPT="test_models.py"

# Check if test script exists
if [ ! -f "$TEST_SCRIPT" ]; then
    echo "Error: $TEST_SCRIPT not found!"
    exit 1
fi

# Function to test a model
test_model() {
    local model_name=$1
    local description=$2

    echo "----------------------------------------"
    echo "Testing: $description"
    echo "Model:   $model_name"
    echo "----------------------------------------"

    if python "$TEST_SCRIPT" --model "$model_name"; then
        echo "✓ $description - PASSED"
        return 0
    else
        echo "✗ $description - FAILED"
        return 1
    fi
}

# Track results
PASSED=0
FAILED=0
TESTS=()

# =============================================================================
# Test Model Configurations
# Uncomment the sections for models you want to test
# =============================================================================

# -----------------------------------------------------------------------------
# Test 1: Google Gemini
# -----------------------------------------------------------------------------
if [ -n "$GEMINI_API_KEY" ]; then
    echo ""
    echo "Testing Google Gemini..."
    if test_model "gemini/gemini-2.5-flash-preview-05-20" "Google Gemini Flash"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
else
    echo ""
    echo "⊘ Skipping Gemini (GEMINI_API_KEY not set)"
fi

# -----------------------------------------------------------------------------
# Test 2: OpenAI
# -----------------------------------------------------------------------------
if [ -n "$OPENAI_API_KEY" ]; then
    echo ""
    echo "Testing OpenAI..."
    if test_model "gpt-3.5-turbo" "OpenAI GPT-3.5"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
else
    echo ""
    echo "⊘ Skipping OpenAI (OPENAI_API_KEY not set)"
fi

# -----------------------------------------------------------------------------
# Test 3: Anthropic Claude
# -----------------------------------------------------------------------------
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "Testing Anthropic Claude..."
    if test_model "claude-3-sonnet-20240229" "Anthropic Claude 3 Sonnet"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
else
    echo ""
    echo "⊘ Skipping Anthropic Claude (ANTHROPIC_API_KEY not set)"
fi

# -----------------------------------------------------------------------------
# Test 4: DeepSeek
# -----------------------------------------------------------------------------
if [ -n "$DEEPSEEK_API_KEY" ]; then
    echo ""
    echo "Testing DeepSeek..."
    if test_model "deepseek/deepseek-chat" "DeepSeek Chat"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
else
    echo ""
    echo "⊘ Skipping DeepSeek (DEEPSEEK_API_KEY not set)"
fi

# -----------------------------------------------------------------------------
# Test 5: Azure OpenAI
# -----------------------------------------------------------------------------
if [ -n "$AZURE_API_KEY" ] && [ -n "$AZURE_API_BASE" ]; then
    echo ""
    echo "Testing Azure OpenAI..."
    # Note: Replace 'your-deployment-name' with your actual deployment name
    if [ -n "$AZURE_DEPLOYMENT_NAME" ]; then
        if test_model "azure/$AZURE_DEPLOYMENT_NAME" "Azure OpenAI"; then
            ((PASSED++))
        else
            ((FAILED++))
        fi
    else
        echo "⊘ Skipping Azure OpenAI (AZURE_DEPLOYMENT_NAME not set)"
        echo "  Set AZURE_DEPLOYMENT_NAME to your deployment name to test"
    fi
else
    echo ""
    echo "⊘ Skipping Azure OpenAI (AZURE_API_KEY or AZURE_API_BASE not set)"
fi

# -----------------------------------------------------------------------------
# Test 6: AWS Bedrock
# -----------------------------------------------------------------------------
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    echo ""
    echo "Testing AWS Bedrock..."
    if test_model "bedrock/anthropic.claude-3-haiku-20240307-v1:0" "AWS Bedrock Claude 3 Haiku"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
else
    echo ""
    echo "⊘ Skipping AWS Bedrock (AWS credentials not set)"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total Tests: $((PASSED + FAILED))"
echo "Passed:      $PASSED"
echo "Failed:      $FAILED"
echo "========================================="

if [ $FAILED -eq 0 ] && [ $PASSED -gt 0 ]; then
    echo "✓ All tests passed!"
    exit 0
elif [ $PASSED -eq 0 ]; then
    echo "⊘ No tests were run. Please configure API keys."
    exit 1
else
    echo "✗ Some tests failed. Please check your configuration."
    exit 1
fi
