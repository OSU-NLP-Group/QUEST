import os
from anthropic import AnthropicBedrock, AsyncAnthropicBedrock


def completion_with_backoff(client, **kwargs):
    return client.messages.create(**kwargs)


async def acompletion_with_backoff(client, **kwargs):
    return await client.messages.create(**kwargs)


class BedrockAntrhopicClient():
    def __init__(self):
        self.client = AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY"), 
            aws_secret_key=os.getenv("AWS_SECRET_KEY"),
            aws_region=os.getenv("AWS_REGION")
        )
    
    def response(self, count_token=False, **kwargs):
        response = completion_with_backoff(self.client, **kwargs)
        if count_token:
            tokens = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
            return response.content[0].text, tokens
        else:
            return response.content[0].text


class AsyncBedrockAntrhopicClient():
    def __init__(self):
        self.client = AsyncAnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY"), 
            aws_secret_key=os.getenv("AWS_SECRET_KEY"),
            aws_region=os.getenv("AWS_REGION")
        )
    
    async def response(self, count_token=False, **kwargs):
        response = await acompletion_with_backoff(self.client, **kwargs)
        if count_token:
            tokens = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
            return response.content[0].text, tokens
        else:
            return response.content[0].text
