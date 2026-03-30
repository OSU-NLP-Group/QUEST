# LiteLLM Migration Guide

This document describes the migration from Google Gemini-only to LiteLLM multi-model support.

## What Changed

### Before (Gemini Only)
```python
# Only supported Google Gemini
from google import genai
client = genai.Client(api_key=api_key)
```

### After (Multi-Model Support)
```python
# Now supports multiple providers through litellm
from litellm import completion
response = completion(model="gemini/...", messages=messages)
response = completion(model="gpt-4", messages=messages)
response = completion(model="claude-3-opus", messages=messages)
# ... and many more
```

## Supported Models

| Provider | Model Examples | API Key Environment Variable |
|----------|---------------|------------------------------|
| **Google Gemini** | `gemini/gemini-2.5-pro-preview-06-05` | `GEMINI_API_KEY` |
| **OpenAI** | `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo` | `OPENAI_API_KEY` |
| **Anthropic Claude** | `claude-3-opus-20240229`, `claude-3-sonnet-20240229` | `ANTHROPIC_API_KEY` |
| **DeepSeek** | `deepseek/deepseek-chat`, `deepseek/deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| **Azure OpenAI** | `azure/<deployment-name>` | `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` |
| **AWS Bedrock** | `bedrock/anthropic.claude-3-sonnet-20240229-v1:0` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME` |

## Modified Files

### Core Changes

1. **`utils/api.py`** - Main API client
   - Replaced Google Gemini SDK with litellm
   - Added support for multiple model providers
   - Maintained backward compatibility with existing code

2. **`requirements.txt`**
   - Removed: `google-genai`
   - Added: `litellm>=1.0.0`

3. **`deepresearch_bench_race.py`**
   - Added `--eval_model` parameter (model for scoring)
   - Added `--clean_model` parameter (model for article cleaning)

### New Files

4. **`run_with_models.sh`** - Easy model configuration script
   - Simple interface to switch between model providers
   - Commented configuration sections for each provider

5. **`test_models.py`** - Model configuration tester
   - Test individual model configurations
   - Verify API keys and connectivity

6. **`test_all_models.sh`** - Batch configuration tester
   - Test all configured model providers at once

7. **`run_multiple_models.sh`** - Multi-model comparison runner
   - Run benchmark with multiple models automatically
   - Generate comparison summary

8. **Documentation**
   - `MODEL_CONFIGURATION.md` - Detailed configuration guide
   - `USAGE_EXAMPLES.md` - Practical usage examples
   - `QUICK_START.md` - Quick start guide
   - `LITELLM_MIGRATION.md` - This file

## Migration Steps

### For New Users

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Choose a model provider and set API key:
```bash
export GEMINI_API_KEY="your-key"  # or any other provider
```

3. Run:
```bash
./run_with_models.sh
```

### For Existing Users

1. **Update dependencies:**
```bash
pip install -r requirements.txt
```

2. **No code changes needed!** The default behavior remains the same (Gemini models)

3. **To use other models**, add new parameters:
```bash
python deepresearch_bench_race.py "model-name" \
    --eval_model "gpt-4" \
    --clean_model "gpt-3.5-turbo"
```

## Backward Compatibility

✅ **100% backward compatible** - All existing scripts and commands work without modification

- Default models remain Gemini (same as before)
- All existing parameters work the same way
- Output format unchanged

## New Capabilities

### 1. Model Selection
```bash
# Use different models for evaluation vs cleaning
--eval_model "gpt-4" --clean_model "gpt-3.5-turbo"
```

### 2. Cost Optimization
```bash
# Use cheaper models for testing
--eval_model "gpt-3.5-turbo" --limit 10
```

### 3. Provider Comparison
```bash
# Compare results across different providers
./run_multiple_models.sh
```

### 4. Enterprise Support
```bash
# Use Azure OpenAI for enterprise deployments
--eval_model "azure/gpt-4-deployment"
```

## API Cost Considerations

Different models have different costs. Here's a rough guide:

| Tier | Models | Use Case |
|------|--------|----------|
| **Premium** | GPT-4, Claude Opus, Gemini Pro | Final evaluation, production |
| **Standard** | GPT-3.5 Turbo, Claude Sonnet, Gemini Flash | Development, testing |
| **Budget** | DeepSeek | Large-scale testing, experiments |

**Recommendation:** Use premium models for `--eval_model` and standard models for `--clean_model` to balance accuracy and cost.

## Environment Variable Reference

### Google Gemini
```bash
export GEMINI_API_KEY="your-key"
```

### OpenAI
```bash
export OPENAI_API_KEY="your-key"
```

### Anthropic Claude
```bash
export ANTHROPIC_API_KEY="your-key"
```

### DeepSeek
```bash
export DEEPSEEK_API_KEY="your-key"
```

### Azure OpenAI
```bash
export AZURE_API_KEY="your-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"
```

### AWS Bedrock
```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION_NAME="us-east-1"
```

## Testing Your Configuration

### Quick Test
```bash
python test_models.py --model "gpt-4"
```

### Test All Configured Models
```bash
./test_all_models.sh
```

### Small Benchmark Test
```bash
python deepresearch_bench_race.py "test-model" \
    --eval_model "gpt-4" \
    --limit 2
```

## Troubleshooting

### Issue: "API key not found"
**Solution:** Make sure you've exported the correct environment variable
```bash
echo $GEMINI_API_KEY  # Check if set
export GEMINI_API_KEY="your-key"  # Set if needed
```

### Issue: "Model not found"
**Solution:** Check the model name format in [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md)

### Issue: Rate limiting
**Solution:** Reduce parallel workers
```bash
--max_workers 2
```

### Issue: Azure deployment name
**Solution:** Use your actual deployment name, not the model name
```bash
--eval_model "azure/my-gpt4-deployment"
```

## Support

- **Documentation:** See [QUICK_START.md](QUICK_START.md) for getting started
- **Examples:** See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for detailed examples
- **Configuration:** See [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) for all options

## Credits

This multi-model support is powered by [LiteLLM](https://github.com/BerriAI/litellm), which provides a unified interface to 100+ LLM providers.
