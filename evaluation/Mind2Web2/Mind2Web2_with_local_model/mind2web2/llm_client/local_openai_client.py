import os
import asyncio
import threading
import inspect
import time
import backoff
import logging
import httpx
from openai import OpenAI, AsyncOpenAI
from openai import (
    OpenAIError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    APITimeoutError,
    LengthFinishReasonError,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _normalize_base_url(raw: str) -> str:
    url = (raw or "").strip()
    if not url:
        return ""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def _parse_default_ports() -> list[int]:
    ports_raw = (os.getenv("VLLM_PORTS") or "6000,6001,6002,6003").strip()
    ports: list[int] = []
    for token in ports_raw.split(","):
        t = token.strip()
        if t.isdigit():
            p = int(t)
            if p not in ports:
                ports.append(p)
    return ports or [6000, 6001, 6002, 6003]


def _parse_endpoints_file(path: str) -> list[str]:
    if not path:
        return []
    if not os.path.isfile(path):
        return []

    urls: list[str] = []
    seen: set[str] = set()

    def _append(endpoint: str):
        normalized = _normalize_base_url(endpoint)
        if normalized and normalized not in seen:
            urls.append(normalized)
            seen.add(normalized)

    with open(path, "r", encoding="utf-8") as f:
        default_ports = _parse_default_ports()
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = [t.strip() for t in line.split(",") if t.strip()]
            for token in tokens:
                if token.startswith("http://") or token.startswith("https://"):
                    _append(token)
                    continue
                if ":" in token:
                    _append(token)
                    continue
                parts = token.split()
                if len(parts) == 2 and parts[1].isdigit():
                    _append(f"{parts[0]}:{parts[1]}")
                    continue
                # Host-only entry: expand to default VLLM ports.
                for port in default_ports:
                    _append(f"{token}:{port}")
    return urls

def _log_backoff(details):
    exc = details.get("exception")
    tries = details.get("tries")
    wait = details.get("wait")
    target = details.get("target")
    target_name = getattr(target, "__name__", str(target))
    logger.warning(
        "Local OpenAI retry #%s after %.1fs in %s due to %s: %s",
        tries,
        wait or 0,
        target_name,
        type(exc).__name__ if exc else "UnknownError",
        exc,
    )


def _log_giveup(details):
    exc = details.get("exception")
    target = details.get("target")
    target_name = getattr(target, "__name__", str(target))
    # LengthFinishReasonError is expected when max_tokens is hit in structured mode.
    if isinstance(exc, LengthFinishReasonError):
        logger.warning("Local OpenAI giving up in %s due to length limit (treating as fallback).", target_name)
        return
    logger.error("Local OpenAI retries exhausted in %s due to %s: %s", target_name, type(exc).__name__, exc)


def _giveup_on_length_limit(exc: Exception) -> bool:
    # In structured-output mode (`response_format=...`), if generation hits the
    # length limit the returned JSON is often truncated and `.parse(...)` raises
    # LengthFinishReasonError. Retrying usually repeats the failure, so give up.
    return isinstance(exc, LengthFinishReasonError)


def _is_binary_eval_schema(response_format) -> bool:
    # Heuristic: verification schema has `result: bool` and `reasoning: str`.
    fields = getattr(response_format, "model_fields", None)
    return isinstance(fields, dict) and "result" in fields and "reasoning" in fields


def _fallback_binary_eval_result(response_format, *, raw_content: str = ""):
    # Return a deterministic "fail" result when structured output is truncated.
    raw_content = raw_content or ""
    max_chars_raw = int(os.getenv("M2W2_TRACE_TRUNCATED_MAX_CHARS", "20000") or 20000)
    if max_chars_raw > 0 and len(raw_content) > max_chars_raw:
        raw_content = raw_content[:max_chars_raw] + "\n...[raw output truncated]..."
    reasoning = "Length limit reached; treating as not supported."
    if raw_content:
        reasoning += "\n\n[RAW_TRUNCATED_OUTPUT]\n" + raw_content
    try:
        return response_format(
            result=False,
            reasoning=reasoning,
        )
    except Exception:
        # Last resort: attempt default init (may fail if required fields).
        return response_format()


def _fallback_structured_result(response_format):
    """
    Best-effort fallback for structured outputs when `.parse(...)` fails due to
    truncation. For eval we prefer returning an empty object and continuing.
    """
    # Pydantic v2
    mc = getattr(response_format, "model_construct", None)
    if callable(mc):
        try:
            return mc()
        except Exception:
            pass
    # Pydantic v1
    c = getattr(response_format, "construct", None)
    if callable(c):
        try:
            return c()
        except Exception:
            pass
    return response_format()

def _extract_truncated_content(exc: LengthFinishReasonError):
    """
    Best-effort extraction of the raw (likely-truncated) assistant content from
    the `ChatCompletion` attached to LengthFinishReasonError.
    """
    try:
        completion = getattr(exc, "completion", None)
        if completion is None:
            return None
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        if message is None:
            return None
        content = getattr(message, "content", None)
        return content if isinstance(content, str) and content else None
    except Exception:
        return None


def _extract_usage_tokens(exc: LengthFinishReasonError):
    """Best-effort extraction of prompt/completion tokens from the attached completion."""
    try:
        completion = getattr(exc, "completion", None)
        usage = getattr(completion, "usage", None) if completion is not None else None
        if usage is None:
            return None
        prompt = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt is None and completion_tokens is None:
            return None
        return {
            "input_tokens": int(prompt or 0),
            "output_tokens": int(completion_tokens or 0),
        }
    except Exception:
        return None


@backoff.on_exception(
    backoff.expo,
    (OpenAIError, APIConnectionError, RateLimitError, InternalServerError, APITimeoutError),
    giveup=_giveup_on_length_limit,
    on_backoff=_log_backoff,
    on_giveup=_log_giveup,
    max_tries=50, #todo
)
def completion_with_backoff(client_or_getter, **kwargs):
    client = client_or_getter() if callable(client_or_getter) else client_or_getter
    if "response_format" in kwargs:
        return client.beta.chat.completions.parse(**kwargs)
    return client.chat.completions.create(**kwargs)


@backoff.on_exception(
    backoff.expo,
    (OpenAIError, APIConnectionError, RateLimitError, InternalServerError, APITimeoutError),
    giveup=_giveup_on_length_limit,
    on_backoff=_log_backoff,
    on_giveup=_log_giveup,
    max_tries=50, #todo
)
async def acompletion_with_backoff(client_or_getter, **kwargs):
    if callable(client_or_getter):
        maybe_client = client_or_getter()
        client = await maybe_client if inspect.isawaitable(maybe_client) else maybe_client
    else:
        client = client_or_getter
    if "response_format" in kwargs:
        return await client.beta.chat.completions.parse(**kwargs)
    return await client.chat.completions.create(**kwargs)


class LocalOpenAIClient:
    def __init__(self) -> None:
        api_key = os.getenv("LOCAL_OPENAI_API_KEY", "dummy")
        self._api_key = api_key
        self._timeout = httpx.Timeout(connect=5.0, read=900.0, write=900.0, pool=900.0)
        self._reload_interval_seconds = max(
            1,
            int(os.getenv("LOCAL_OPENAI_ENDPOINTS_RELOAD_SECONDS", "15") or 15),
        )
        self._last_reload_at = 0.0
        base_urls = _get_local_openai_base_urls()
        if not base_urls:
            raise RuntimeError(
                "LOCAL_OPENAI_BASE_URL (or LOCAL_OPENAI_BASE_URLS or LOCAL_OPENAI_ENDPOINTS_FILE) is not set."
            )
        self._base_urls = list(base_urls)
        self._clients = [OpenAI(api_key=api_key, base_url=url, timeout=self._timeout, max_retries=0) for url in base_urls]
        self._lock = threading.Lock()
        self._next_index = 0
    
    def _refresh_clients_if_needed(self) -> None:
        now = time.monotonic()
        if now - self._last_reload_at < self._reload_interval_seconds:
            return
        self._last_reload_at = now

        latest_urls = _get_local_openai_base_urls()
        if not latest_urls:
            return
        if latest_urls == self._base_urls:
            return

        logger.info("Reloading Local OpenAI endpoints: %s", ", ".join(latest_urls))
        self._base_urls = list(latest_urls)
        self._clients = [
            OpenAI(api_key=self._api_key, base_url=url, timeout=self._timeout, max_retries=0)
            for url in latest_urls
        ]
        self._next_index = 0

    def _get_client(self) -> OpenAI:
        self._refresh_clients_if_needed()
        if len(self._clients) == 1:
            return self._clients[0]
        with self._lock:
            self._refresh_clients_if_needed()
            client = self._clients[self._next_index]
            self._next_index = (self._next_index + 1) % len(self._clients)
            return client

    def response(self, count_token: bool = False, **kwargs):
        kwargs.pop("_trace", None)
        try:
            # Pass the getter so each backoff retry can rotate to the next endpoint.
            response = completion_with_backoff(self._get_client, **kwargs)
        except LengthFinishReasonError as e:
            raw = _extract_truncated_content(e) or ""
            if raw:
                logger.warning("Local OpenAI truncated output (raw, may be incomplete):\n%s", raw)
            if "response_format" in kwargs:
                rf = kwargs.get("response_format")
                if _is_binary_eval_schema(rf):
                    out = _fallback_binary_eval_result(rf, raw_content=raw)
                else:
                    out = _fallback_structured_result(rf)
                tokens = _extract_usage_tokens(e) or {"input_tokens": 0, "output_tokens": 0}
                return (out, tokens) if count_token else out
            raise

        tokens = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
        if "response_format" in kwargs:
            out = response.choices[0].message.parsed
            return (out, tokens) if count_token else out
        out = response.choices[0].message.content
        return (out, tokens) if count_token else out


class AsyncLocalOpenAIClient:
    def __init__(self) -> None:
        api_key = os.getenv("LOCAL_OPENAI_API_KEY", "dummy")
        self._api_key = api_key
        self._timeout = httpx.Timeout(connect=5.0, read=900.0, write=900.0, pool=900.0)
        self._reload_interval_seconds = max(
            1,
            int(os.getenv("LOCAL_OPENAI_ENDPOINTS_RELOAD_SECONDS", "15") or 15),
        )
        self._last_reload_at = 0.0
        base_urls = _get_local_openai_base_urls()
        if not base_urls:
            raise RuntimeError(
                "LOCAL_OPENAI_BASE_URL (or LOCAL_OPENAI_BASE_URLS or LOCAL_OPENAI_ENDPOINTS_FILE) is not set."
            )
        self._base_urls = list(base_urls)
        self._clients = [AsyncOpenAI(api_key=api_key, base_url=url, timeout=self._timeout, max_retries=0) for url in base_urls]
        self._lock = asyncio.Lock()
        self._next_index = 0
    
    async def _refresh_clients_if_needed(self) -> None:
        now = time.monotonic()
        if now - self._last_reload_at < self._reload_interval_seconds:
            return
        self._last_reload_at = now

        latest_urls = _get_local_openai_base_urls()
        if not latest_urls:
            return
        if latest_urls == self._base_urls:
            return

        logger.info("Reloading Local OpenAI endpoints: %s", ", ".join(latest_urls))
        self._base_urls = list(latest_urls)
        self._clients = [
            AsyncOpenAI(api_key=self._api_key, base_url=url, timeout=self._timeout, max_retries=0)
            for url in latest_urls
        ]
        self._next_index = 0

    async def _get_client(self) -> AsyncOpenAI:
        await self._refresh_clients_if_needed()
        if len(self._clients) == 1:
            return self._clients[0]
        async with self._lock:
            await self._refresh_clients_if_needed()
            client = self._clients[self._next_index]
            self._next_index = (self._next_index + 1) % len(self._clients)
            return client

    async def response(self, count_token: bool = False, **kwargs):
        kwargs.pop("_trace", None)
        try:
            # Pass the getter so each backoff retry can rotate to the next endpoint.
            response = await acompletion_with_backoff(self._get_client, **kwargs)
        except LengthFinishReasonError as e:
            raw = _extract_truncated_content(e) or ""
            if raw:
                logger.warning("Local OpenAI truncated output (raw, may be incomplete):\n%s", raw)
            if "response_format" in kwargs:
                rf = kwargs.get("response_format")
                if _is_binary_eval_schema(rf):
                    out = _fallback_binary_eval_result(rf, raw_content=raw)
                else:
                    out = _fallback_structured_result(rf)
                tokens = _extract_usage_tokens(e) or {"input_tokens": 0, "output_tokens": 0}
                return (out, tokens) if count_token else out
            raise

        tokens = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
        if "response_format" in kwargs:
            out = response.choices[0].message.parsed
            return (out, tokens) if count_token else out
        out = response.choices[0].message.content
        return (out, tokens) if count_token else out


def _get_local_openai_base_urls() -> list[str]:
    """
    Resolve local OpenAI-compatible base URLs.

    Supports:
      - LOCAL_OPENAI_ENDPOINTS_FILE: path to endpoint file (hot-reload source)
      - LOCAL_OPENAI_BASE_URLS: comma-separated list of base URLs
      - LOCAL_OPENAI_BASE_URL: single base URL (fallback)
    """
    endpoints_file = (os.getenv("LOCAL_OPENAI_ENDPOINTS_FILE") or "").strip()
    if endpoints_file:
        file_urls = _parse_endpoints_file(endpoints_file)
        if file_urls:
            return file_urls

    base_urls_raw = (os.getenv("LOCAL_OPENAI_BASE_URLS") or "").strip()
    if base_urls_raw:
        urls = [_normalize_base_url(u) for u in base_urls_raw.split(",")]
        urls = [u for u in urls if u]
        if urls:
            return urls
    base_url = (os.getenv("LOCAL_OPENAI_BASE_URL") or "").strip()
    normalized = _normalize_base_url(base_url)
    return [normalized] if normalized else []
