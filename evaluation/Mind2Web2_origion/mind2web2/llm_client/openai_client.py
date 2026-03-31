"""
mind2web2/llm_client/openai_client.py

A thin wrapper around the OpenAI Python SDK (v1+) that
adds exponential-backoff retry logic, unified synchronous
and asynchronous interfaces, and optional token usage stats.
"""

import os
import backoff
from openai import OpenAI, AsyncOpenAI
from openai import (
    OpenAIError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    APITimeoutError,
)
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _log_backoff(details):
    """Log retry attempts triggered by backoff."""
    exc = details.get("exception")
    tries = details.get("tries")
    wait = details.get("wait")
    target = details.get("target")
    target_name = getattr(target, "__name__", str(target))
    kwargs = details.get("kwargs") or {}
    model = kwargs.get("model")
    if exc is not None:
        logger.warning(
            "OpenAI retry #%s after %.1fs in %s (model=%s) due to %s: %s",
            tries,
            wait or 0,
            target_name,
            model,
            type(exc).__name__,
            exc,
        )
    else:
        logger.warning(
            "OpenAI retry #%s after %.1fs in %s (model=%s, no exception info)",
            tries,
            wait or 0,
            target_name,
            model,
        )


def _log_giveup(details):
    exc = details.get("exception")
    target = details.get("target")
    target_name = getattr(target, "__name__", str(target))
    kwargs = details.get("kwargs") or {}
    model = kwargs.get("model")
    if exc is not None:
        logger.error(
            "OpenAI retries exhausted in %s (model=%s) due to %s: %s",
            target_name,
            model,
            type(exc).__name__,
            exc,
        )
    else:
        logger.error(
            "OpenAI retries exhausted in %s (model=%s, no exception info)",
            target_name,
            model,
        )


# --------------------------------------------------------------------------- #
# Retry helpers                                                               #
# --------------------------------------------------------------------------- #


@backoff.on_exception(
    backoff.expo,
    (OpenAIError, APIConnectionError, RateLimitError, InternalServerError, APITimeoutError),
    on_backoff=_log_backoff,
    on_giveup=_log_giveup,
)
def completion_with_backoff(client: OpenAI, **kwargs):
    """
    Synchronous completion request with exponential-backoff retry.

    If `response_format` is supplied the call is routed to the
    structured-output beta endpoint; otherwise the regular endpoint is used.
    """
    if "response_format" in kwargs:
        return client.beta.chat.completions.parse(**kwargs)  # structured JSON
    return client.chat.completions.create(**kwargs)


@backoff.on_exception(
    backoff.expo,
    (OpenAIError, APIConnectionError, RateLimitError, InternalServerError, APITimeoutError),
    on_backoff=_log_backoff,
    on_giveup=_log_giveup,
)
async def acompletion_with_backoff(client: AsyncOpenAI, **kwargs):
    """
    Asynchronous completion request with exponential-backoff retry.
    """
    if "response_format" in kwargs:
        return await client.beta.chat.completions.parse(**kwargs)
    return await client.chat.completions.create(**kwargs)


# --------------------------------------------------------------------------- #
# Synchronous client                                                          #
# --------------------------------------------------------------------------- #


class OpenAIClient:
    """
    Synchronous OpenAI client.

    Example:
        client = OpenAIClient()
        result = client.response(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
            temperature=0.2,
            # response_format={"type": "json_object"}  # optional
        )
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def response(self, count_token: bool = False, **kwargs):
        """
        Wrapper around `chat.completions.create`.

        Args:
            count_token: If True, also return a dict with token usage.
            **kwargs: Arguments accepted by the OpenAI `/chat/completions` API.

        Returns:
            Either the content/parsed JSON, or a tuple
            (content_or_parsed_json, token_dict) when `count_token=True`.
        """
        response = completion_with_backoff(self.client, **kwargs)

        tokens = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        if "response_format" in kwargs:  # structured-output mode
            return (response.choices[0].message.parsed, tokens) if count_token else response.choices[0].message.parsed

        # plain-text mode
        return (response.choices[0].message.content, tokens) if count_token else response.choices[0].message.content


# --------------------------------------------------------------------------- #
# Asynchronous client                                                         #
# --------------------------------------------------------------------------- #


class AsyncOpenAIClient:
    """
    Asynchronous OpenAI client.

    Example:
        client = AsyncOpenAIClient()
        result = await client.response(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Ping"}],
        )
    """

    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def response(self, count_token: bool = False, **kwargs):
        """
        Async wrapper around `chat.completions.create`.

        Behavior mirrors `OpenAIClient.response`.
        """
        response = await acompletion_with_backoff(self.client, **kwargs)

        tokens = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

        if "response_format" in kwargs:
            return (response.choices[0].message.parsed, tokens) if count_token else response.choices[0].message.parsed

        return (response.choices[0].message.content, tokens) if count_token else response.choices[0].message.content
