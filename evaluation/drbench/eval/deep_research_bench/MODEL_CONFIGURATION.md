# Model Configuration Guide

This document describes how to configure different LLM models for DeepResearch Benchmark evaluation.

## Supported Models

The benchmark now supports multiple LLM providers through litellm:

### 1. Google Gemini

**Model Format:**
- `gemini/gemini-2.5-pro-preview-06-05`
- `gemini/gemini-2.0-flash-exp`
- `gemini/gemini-2.5-flash-preview-05-20`

**Environment Variables:**
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

**Example:**
```bash
python deepresearch_bench_race.py model_name --eval_model gemini/gemini-2.5-pro-preview-06-05
```

### 2. OpenAI

**Model Format:**
- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`
- `openai/gpt-4o`

**Environment Variables:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

**Example:**
```bash
python deepresearch_bench_race.py model_name --eval_model gpt-4
```

### 3. Anthropic Claude

**Model Format:**
- `claude-3-sonnet-20240229`
- `claude-3-opus-20240229`
- `anthropic/claude-3-5-sonnet-20241022`

**Environment Variables:**
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

**Example:**
```bash
python deepresearch_bench_race.py model_name --eval_model claude-3-sonnet-20240229
```

### 4. DeepSeek

**Model Format:**
- `deepseek/deepseek-chat`
- `deepseek/deepseek-reasoner`

**Environment Variables:**
```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

**Example:**
```bash
python deepresearch_bench_race.py model_name --eval_model deepseek/deepseek-chat
```

### 5. Azure OpenAI

**Model Format:**
- `azure/<your-deployment-name>`

**Environment Variables:**
```bash
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"
```

**Example:**
```bash
python deepresearch_bench_race.py model_name --eval_model azure/gpt-4-deployment
```

### 6. AWS Bedrock

**Model Format:**
- `bedrock/anthropic.claude-3-sonnet-20240229-v1:0`
- `bedrock/anthropic.claude-3-opus-20240229-v1:0`
- `bedrock/anthropic.claude-3-haiku-20240307-v1:0`

**Environment Variables:**
```bash
export AWS_ACCESS_KEY_ID="your-aws-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-access-key"
export AWS_REGION_NAME="us-east-1"  # or your preferred region
```

**Example:**
```bash
python deepresearch_bench_race.py model_name --eval_model bedrock/anthropic.claude-3-sonnet-20240229-v1:0
```

## Using Different Models for Evaluation and Cleaning

You can specify different models for evaluation (scoring) and article cleaning:

```bash
python deepresearch_bench_race.py model_name \
    --eval_model gpt-4 \
    --clean_model gpt-3.5-turbo
```

This allows you to use a more capable (and expensive) model for evaluation while using a faster/cheaper model for article cleaning.

## Script Usage

Update the model configuration in `run_benchmark.sh`:

```bash
# For evaluation (scoring)
EVAL_MODEL="gemini/gemini-2.5-pro-preview-06-05"

# For article cleaning (can use faster/cheaper model)
CLEAN_MODEL="gemini/gemini-2.5-flash-preview-05-20"
```

Then run:
```bash
bash run_benchmark.sh
```

## Troubleshooting

### API Key Not Found
Make sure you've exported the correct environment variable for your chosen model provider.

### Azure Configuration
For Azure, ensure all three environment variables are set:
- `AZURE_API_KEY`
- `AZURE_API_BASE` (must end with `/`)
- `AZURE_API_VERSION`

### AWS Bedrock
For Bedrock, ensure your AWS credentials have permission to invoke the Bedrock models and the region is correct.

### Model Not Found
Check that you're using the correct model format for litellm. Refer to the [litellm documentation](https://docs.litellm.ai/docs/providers) for the full list of supported models.
