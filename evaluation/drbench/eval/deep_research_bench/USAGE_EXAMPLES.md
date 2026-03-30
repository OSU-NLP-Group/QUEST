# Usage Examples

This document provides practical examples for using different LLM models with DeepResearch Benchmark.

## Quick Start

### 1. Test Your Model Configuration

Before running the full benchmark, test your model configuration:

```bash
# Test Gemini
export GEMINI_API_KEY="your-key"
python test_models.py --model "gemini/gemini-2.5-pro-preview-06-05"

# Test OpenAI
export OPENAI_API_KEY="your-key"
python test_models.py --model "gpt-4"

# Test Claude
export ANTHROPIC_API_KEY="your-key"
python test_models.py --model "claude-3-sonnet-20240229"
```

### 2. Run Benchmark with Different Models

#### Example 1: Using Google Gemini (Default)

```bash
export GEMINI_API_KEY="your-gemini-api-key"

python deepresearch_bench_race.py "claude-3-7-sonnet-latest" \
    --eval_model "gemini/gemini-2.5-pro-preview-06-05" \
    --clean_model "gemini/gemini-2.5-flash-preview-05-20" \
    --max_workers 5
```

#### Example 2: Using OpenAI GPT-4

```bash
export OPENAI_API_KEY="your-openai-api-key"

python deepresearch_bench_race.py "claude-3-7-sonnet-latest" \
    --eval_model "gpt-4" \
    --clean_model "gpt-3.5-turbo" \
    --max_workers 5
```

#### Example 3: Using Anthropic Claude

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"

python deepresearch_bench_race.py "gpt-4-turbo" \
    --eval_model "claude-3-opus-20240229" \
    --clean_model "claude-3-sonnet-20240229" \
    --max_workers 5
```

#### Example 4: Using DeepSeek

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"

python deepresearch_bench_race.py "claude-3-7-sonnet-latest" \
    --eval_model "deepseek/deepseek-chat" \
    --clean_model "deepseek/deepseek-chat" \
    --max_workers 5
```

#### Example 5: Using Azure OpenAI

```bash
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"

python deepresearch_bench_race.py "claude-3-7-sonnet-latest" \
    --eval_model "azure/gpt-4-deployment" \
    --clean_model "azure/gpt-35-turbo-deployment" \
    --max_workers 5
```

#### Example 6: Using AWS Bedrock

```bash
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export AWS_REGION_NAME="us-east-1"

python deepresearch_bench_race.py "gpt-4-turbo" \
    --eval_model "bedrock/anthropic.claude-3-sonnet-20240229-v1:0" \
    --clean_model "bedrock/anthropic.claude-3-haiku-20240307-v1:0" \
    --max_workers 5
```

### 3. Advanced Options

#### Test with Limited Data

```bash
python deepresearch_bench_race.py "model-name" \
    --eval_model "gpt-4" \
    --limit 10  # Only process 10 tasks
```

#### Process Only Chinese or English

```bash
# Only Chinese
python deepresearch_bench_race.py "model-name" \
    --eval_model "gpt-4" \
    --only_zh

# Only English
python deepresearch_bench_race.py "model-name" \
    --eval_model "gpt-4" \
    --only_en
```

#### Skip Article Cleaning

```bash
python deepresearch_bench_race.py "model-name" \
    --eval_model "gpt-4" \
    --skip_cleaning
```

#### Force Re-evaluation

```bash
python deepresearch_bench_race.py "model-name" \
    --eval_model "gpt-4" \
    --force
```

### 4. Using the Shell Script

Edit `run_benchmark.sh` to configure your models:

```bash
# In run_benchmark.sh
TARGET_MODELS=("claude-3-7-sonnet-latest" "gpt-4-turbo")
EVAL_MODEL="gpt-4"
CLEAN_MODEL="gpt-3.5-turbo"
```

Then run:

```bash
bash run_benchmark.sh
```

## Mixed Model Usage

You can mix different providers for evaluation and cleaning:

```bash
# Use Claude for evaluation, GPT-3.5 for cleaning
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."

python deepresearch_bench_race.py "model-name" \
    --eval_model "claude-3-opus-20240229" \
    --clean_model "gpt-3.5-turbo"
```

## Troubleshooting

### Rate Limiting

If you encounter rate limiting issues, reduce the number of workers:

```bash
python deepresearch_bench_race.py "model-name" \
    --eval_model "gpt-4" \
    --max_workers 2  # Reduce from default 5
```

### Timeout Issues

The default timeout is 600 seconds. If you experience timeouts, check your network connection or model availability.

### Azure Deployment Names

For Azure OpenAI, use your actual deployment name (not the model name):

```bash
# If your deployment is named "my-gpt4-deployment"
--eval_model "azure/my-gpt4-deployment"
```

### AWS Region

Make sure your AWS region supports the Bedrock model you're trying to use:

```bash
# Claude models are available in us-east-1 and us-west-2
export AWS_REGION_NAME="us-east-1"
```

## Cost Optimization Tips

1. **Use cheaper models for cleaning**: Article cleaning is less critical than evaluation
   ```bash
   --eval_model "gpt-4" --clean_model "gpt-3.5-turbo"
   ```

2. **Start with a small sample**: Use `--limit` to test with a few tasks first
   ```bash
   --limit 5
   ```

3. **Process one language at a time**: Use `--only_zh` or `--only_en` to split the workload
   ```bash
   --only_zh  # Process Chinese first, then run again with --only_en
   ```

4. **Use flash models**: For Gemini, use flash models for faster/cheaper processing
   ```bash
   --eval_model "gemini/gemini-2.0-flash-exp"
   ```
