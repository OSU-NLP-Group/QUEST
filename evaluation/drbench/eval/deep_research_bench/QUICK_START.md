# Quick Start Guide

This guide will help you get started with running DeepResearch Benchmark using different LLM models.

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API keys (choose the provider you want to use):

```bash
# Google Gemini
export GEMINI_API_KEY="your-gemini-api-key"

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# Azure OpenAI
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"
export AZURE_DEPLOYMENT_NAME="your-deployment-name"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION_NAME="us-east-1"
```

## Method 1: Quick Test (Recommended First Step)

Test that your model configuration is working:

```bash
# Make scripts executable
chmod +x test_all_models.sh

# Run configuration test
./test_all_models.sh
```

Or test a specific model:

```bash
python test_models.py --model "gemini/gemini-2.5-pro-preview-06-05"
```

## Method 2: Simple Single Run

Use the easy configuration script:

```bash
# Make script executable
chmod +x run_with_models.sh

# Edit the script to select your model (uncomment the section you want)
vim run_with_models.sh

# Run the benchmark
./run_with_models.sh
```

### Quick Edit Guide for `run_with_models.sh`:

1. Open the file
2. Find the "MODEL CONFIGURATIONS" section
3. Comment out the default (Gemini) by adding `#` at the start of each line
4. Uncomment the section for your chosen provider (remove `#`)
5. Save and run

Example - switching to OpenAI:
```bash
# Comment out Gemini (add # at start):
# export GEMINI_API_KEY="${GEMINI_API_KEY:-your-gemini-api-key}"
# EVAL_MODEL="gemini/gemini-2.5-pro-preview-06-05"
# CLEAN_MODEL="gemini/gemini-2.5-flash-preview-05-20"

# Uncomment OpenAI (remove # from start):
export OPENAI_API_KEY="${OPENAI_API_KEY:-your-openai-api-key}"
EVAL_MODEL="gpt-4"
CLEAN_MODEL="gpt-3.5-turbo"
```

## Method 3: Direct Python Command

Run directly with Python for maximum control:

```bash
# Basic usage
python deepresearch_bench_race.py "claude-3-7-sonnet-latest" \
    --eval_model "gemini/gemini-2.5-pro-preview-06-05" \
    --clean_model "gemini/gemini-2.5-flash-preview-05-20"

# With additional options
python deepresearch_bench_race.py "claude-3-7-sonnet-latest" \
    --eval_model "gpt-4" \
    --clean_model "gpt-3.5-turbo" \
    --max_workers 5 \
    --limit 10
```

## Method 4: Compare Multiple Models

Run benchmark with multiple model configurations automatically:

```bash
# Make script executable
chmod +x run_multiple_models.sh

# Edit to select which models to compare
vim run_multiple_models.sh

# Run
./run_multiple_models.sh
```

This will run the benchmark with each selected model and generate a comparison summary.

## Common Options

Add these flags to any method:

```bash
--limit 10              # Test with only 10 tasks (for quick testing)
--only_zh              # Process only Chinese data
--only_en              # Process only English data
--skip_cleaning        # Skip article cleaning step
--force                # Force re-evaluation even if results exist
--max_workers 10       # Use more parallel workers (default: 5)
```

## Example Workflows

### Workflow 1: First Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key
export GEMINI_API_KEY="your-key"

# 3. Test configuration
python test_models.py --model "gemini/gemini-2.5-pro-preview-06-05"

# 4. Run a small test
python deepresearch_bench_race.py "test-model" \
    --eval_model "gemini/gemini-2.5-pro-preview-06-05" \
    --limit 2

# 5. Run full benchmark
./run_with_models.sh
```

### Workflow 2: Quick Cost-Effective Test

```bash
# Use cheaper models and limit scope
export OPENAI_API_KEY="your-key"

python deepresearch_bench_race.py "model-to-test" \
    --eval_model "gpt-3.5-turbo" \
    --clean_model "gpt-3.5-turbo" \
    --limit 5 \
    --only_en
```

### Workflow 3: Production Run

```bash
# Use best models for accurate evaluation
export OPENAI_API_KEY="your-key"

python deepresearch_bench_race.py "production-model" \
    --eval_model "gpt-4" \
    --clean_model "gpt-3.5-turbo" \
    --max_workers 10
```

## Output Files

Results are saved to:
- `results/race/<target-model>/raw_results.jsonl` - Detailed results
- `results/race/<target-model>/race_result.txt` - Summary scores

## Troubleshooting

### Error: API Key Not Found
Make sure you've exported the correct environment variable for your model.

```bash
# Check if set
echo $GEMINI_API_KEY
echo $OPENAI_API_KEY

# Set if missing
export GEMINI_API_KEY="your-key"
```

### Error: Model Not Found
Check that you're using the correct model format. See [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) for all supported formats.

### Rate Limiting
Reduce `--max_workers`:
```bash
--max_workers 2
```

### Out of Memory
Process one language at a time:
```bash
--only_zh  # First run
--only_en  # Second run
```

## Next Steps

- Read [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) for detailed model configuration
- Read [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for more examples
- Check the original `run_benchmark.sh` for the full benchmark including FACT evaluation

## Script Summary

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `test_models.py` | Test a single model | Before first run |
| `test_all_models.sh` | Test all configured models | Check all API keys |
| `run_with_models.sh` | Run with one model config | Simple single run |
| `run_multiple_models.sh` | Compare multiple models | Model comparison |
| `run_benchmark.sh` | Full benchmark (RACE + FACT) | Production evaluation |
