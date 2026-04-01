

class LLMClient():
    def __init__(self, provider, is_async=False):
        self.provider = provider
        self.is_async = is_async
        if provider == 'azure_openai':
            if is_async:
                from mind2web2.llm_client.azure_openai_client import AsyncAzureOpenAIClient
                self.client = AsyncAzureOpenAIClient()
            else:
                from mind2web2.llm_client.azure_openai_client import AzureOpenAIClient
                self.client = AzureOpenAIClient()
        elif provider == 'openai':
            if is_async:
                from mind2web2.llm_client.openai_client import AsyncOpenAIClient
                self.client = AsyncOpenAIClient()
            else:
                from mind2web2.llm_client.openai_client import OpenAIClient
                self.client = OpenAIClient()
        elif provider == 'bedrock_anthropic':
            if is_async:
                from mind2web2.llm_client.bedrock_anthropic_client import AsyncBedrockAnthropicClient
                self.client = AsyncBedrockAnthropicClient()
            else:
                from mind2web2.llm_client.bedrock_anthropic_client import BedrockAnthropicClient
                self.client = BedrockAnthropicClient()
        else:
            raise ValueError(f'Provider {provider} not supported')
    
    def response(self, **kwargs):
        # insure that the provider is not async
        if self.is_async:
            raise ValueError(f'Provider {self.provider} is async and does not support synchronous response')
        return self.client.response(**kwargs)

    async def async_response(self, **kwargs):
        # insure that the provider is async
        if not self.is_async:
            raise ValueError(f'Provider {self.provider} is not async and does not support asynchronous response')
        return await self.client.response(**kwargs)
