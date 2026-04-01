# llm_client — LLM Provider Abstraction

Unified interface for making LLM API calls across different providers.

## Architecture

### base_client.py — Provider Router
`LLMClient` is a simple router that delegates to provider-specific clients:
- `provider="openai"` → `OpenAIClient` / `AsyncOpenAIClient`
- `provider="azure_openai"` → `AzureOpenAIClient` / `AsyncAzureOpenAIClient`
- `provider="bedrock_anthropic"` → `BedrockAnthropicClient` / `AsyncBedrockAnthropicClient`

Key methods:
- `response(**kwargs)`: Synchronous call (raises if `is_async=True`)
- `async_response(**kwargs)`: Async call (raises if `is_async=False`)

### openai_client.py — OpenAI API Client
`OpenAIClient` (sync) and `AsyncOpenAIClient` (async):
- Wraps OpenAI Python SDK v1+
- **Structured output**: If `response_format` kwarg is provided, routes to `client.beta.chat.completions.parse()` for Pydantic model parsing
- **Plain text**: Otherwise uses standard `client.chat.completions.create()`
- **Retry**: Exponential backoff via `@backoff.on_exception` for `RateLimitError`, `APIConnectionError`, `InternalServerError`, `APITimeoutError`
- `count_token=True` returns `(result, token_dict)` tuple

### azure_openai_client.py — Azure OpenAI Client
Same interface as OpenAI client but uses `AzureOpenAI` / `AsyncAzureOpenAI`:
- Requires env vars: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT_URL`, `AZURE_OPENAI_API_VERSION`
- Same backoff retry logic

### bedrock_anthropic_client.py — AWS Bedrock Anthropic Client
For Claude models via AWS Bedrock:
- Requires env vars: `AWS_ACCESS_KEY`, `AWS_SECRET_KEY`, `AWS_REGION`
- Note: Does NOT support structured output (`response_format`); returns raw text only
- Note: No backoff retry currently implemented (unlike OpenAI clients)

### api_cost.py — Cost Calculation
`calculate_api_cost(input_tokens, output_tokens, model_name)`:
- Lookup table for per-million-token pricing
- Supported models: `gpt-4.1`, `o4-mini`, `gpt-4o`, Claude 3.7 Sonnet on Bedrock

## Usage Pattern
```python
# In eval pipeline (async)
client = LLMClient(provider="openai", is_async=True)
result = await client.async_response(
    model="o4-mini",
    messages=[...],
    response_format=MyPydanticModel,  # optional structured output
)
```

## Notes
- The default evaluation model throughout the codebase is `o4-mini`
- Temperature is automatically set to 0.0 for non-reasoning models (those without "o" in name) in `BaseEvaluator.call_llm_with_semaphore()`
- All clients suppress noisy httpx logging: `logging.getLogger("httpx").setLevel(logging.WARNING)`
