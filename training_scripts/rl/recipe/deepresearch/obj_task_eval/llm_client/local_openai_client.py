import os
import asyncio
import threading
import logging
import random
import time
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI, AsyncOpenAI
from openai import (
    OpenAIError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    APITimeoutError,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


_RETRYABLE_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}
_RETRYABLE_TEXT_HINTS = (
    "timeout",
    "timed out",
    "temporarily unavailable",
    "connection refused",
    "connection reset",
    "pool timeout",
    "rate limit",
    "overloaded",
    "busy",
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, RateLimitError, InternalServerError, APITimeoutError, TimeoutError)):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in _RETRYABLE_HTTP_STATUS:
        return True

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int) and response_status in _RETRYABLE_HTTP_STATUS:
        return True

    text = str(exc).lower()
    return any(hint in text for hint in _RETRYABLE_TEXT_HINTS)


def _extract_tokens(response) -> dict:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            value = value.strip()
            if value:
                return value
    return ""


def _env_override(name: str, fallback: str = "") -> str:
    # Respect explicit env overrides even when set to the empty string.
    return os.environ[name] if name in os.environ else fallback


def _resolve_local_openai_fallback_config(requested_model: str) -> Dict[str, str]:
    model_name = (
        _env_first("LOCAL_OPENAI_FALLBACK_MODEL_NAME")
        or _env_first("EVAL_LLM_FALLBACK_MODEL_NAME")
        or _env_first("MEMORY_MODEL_NAME", "SUMMARY_MODEL_NAME")
        or requested_model
        or "gpt-4o-mini"
    )
    azure_endpoint = _env_override(
        "LOCAL_OPENAI_FALLBACK_AZURE_ENDPOINT",
        _env_override(
            "EVAL_LLM_FALLBACK_AZURE_ENDPOINT",
            os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        ),
    )
    return {
        "api_key": _env_first(
            "LOCAL_OPENAI_FALLBACK_API_KEY",
            "EVAL_LLM_FALLBACK_API_KEY",
            "MEMORY_API_KEY",
            "API_KEY",
        ),
        "api_base": _env_first(
            "LOCAL_OPENAI_FALLBACK_API_BASE",
            "EVAL_LLM_FALLBACK_API_BASE",
            "MEMORY_API_BASE",
            "API_BASE",
        ),
        "model_name": model_name,
        "azure_endpoint": azure_endpoint,
        "azure_api_version": _env_override(
            "LOCAL_OPENAI_FALLBACK_AZURE_API_VERSION",
            _env_override(
                "EVAL_LLM_FALLBACK_AZURE_API_VERSION",
                os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview",
            ),
        ),
        "azure_deployment": _env_override(
            "LOCAL_OPENAI_FALLBACK_AZURE_DEPLOYMENT",
            _env_override(
                "EVAL_LLM_FALLBACK_AZURE_DEPLOYMENT",
                os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""),
            ),
        ),
    }


def _resolve_local_openai_secondary_fallback_config(requested_model: str) -> Dict[str, str]:
    model_name = (
        _env_first("LOCAL_OPENAI_SECONDARY_FALLBACK_MODEL_NAME")
        or _env_first("EVAL_LLM_MODEL_NAME", "OPENAI_MODEL_NAME")
        or requested_model
        or "gpt-4o-mini"
    )
    return {
        "api_key": _env_first(
            "LOCAL_OPENAI_SECONDARY_FALLBACK_API_KEY",
            "EVAL_LLM_API_KEY",
            "API_KEY",
            "LOCAL_OPENAI_FALLBACK_API_KEY",
        ),
        "api_base": _env_first(
            "LOCAL_OPENAI_SECONDARY_FALLBACK_API_BASE",
            "EVAL_LLM_API_BASE",
            "API_BASE",
        ),
        "model_name": model_name,
        "azure_endpoint": _env_override(
            "LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_ENDPOINT",
            os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        ),
        "azure_api_version": _env_override(
            "LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_API_VERSION",
            os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview",
        ),
        "azure_deployment": _env_override(
            "LOCAL_OPENAI_SECONDARY_FALLBACK_AZURE_DEPLOYMENT",
            os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""),
        ),
    }


def _fallback_cfg_key(cfg: Dict[str, str]) -> Tuple[str, str, str, str, str]:
    return (
        str(cfg.get("api_key") or ""),
        str(cfg.get("api_base") or ""),
        str(cfg.get("model_name") or ""),
        str(cfg.get("azure_endpoint") or ""),
        str(cfg.get("azure_deployment") or ""),
    )


def _has_fallback_target(cfg: Dict[str, str]) -> bool:
    if not cfg.get("api_key"):
        return False
    if cfg.get("azure_endpoint"):
        return True
    return bool(cfg.get("api_base") or cfg.get("model_name"))


def _iter_local_openai_fallback_candidates(requested_model: str) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    primary_cfg = _resolve_local_openai_fallback_config(requested_model)
    secondary_cfg = _resolve_local_openai_secondary_fallback_config(requested_model)

    if _has_fallback_target(primary_cfg):
        candidates.append(primary_cfg)
    if _has_fallback_target(secondary_cfg):
        if not candidates or _fallback_cfg_key(secondary_cfg) != _fallback_cfg_key(candidates[0]):
            candidates.append(secondary_cfg)
    return candidates


def _extract_json_payload(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return stripped

    patterns = [
        r"```json\s*\n(.*?)\n```",
        r"```\s*\n(.*?)\n```",
        r"```json\s*(.*?)```",
        r"```\s*(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return stripped[first_brace:last_brace + 1]

    return stripped


def _parse_fallback_content(content: Optional[str], response_format: Any):
    if response_format is None:
        return content
    if content is None:
        raise RuntimeError("LOCAL_OPENAI fallback returned empty content for structured response")

    payload = _extract_json_payload(content)
    if hasattr(response_format, "model_validate_json"):
        try:
            return response_format.model_validate_json(payload)
        except Exception:
            return response_format.model_validate(json.loads(payload))
    if hasattr(response_format, "parse_raw"):
        return response_format.parse_raw(payload)
    raise RuntimeError(f"Unsupported response_format for LOCAL_OPENAI fallback: {response_format!r}")


def _fallback_backend_label(cfg: Dict[str, str]) -> str:
    if cfg["azure_endpoint"]:
        return f"azure:{cfg['azure_endpoint']}"
    return f"openai:{cfg['api_base'] or 'default'}"


def _build_sync_fallback_client(cfg: Dict[str, str]):
    if cfg["azure_endpoint"]:
        from openai import AzureOpenAI

        deployment = cfg["azure_deployment"] or cfg["model_name"]
        if not deployment:
            raise RuntimeError("LOCAL_OPENAI fallback missing Azure deployment/model name")
        client = AzureOpenAI(
            api_key=cfg["api_key"],
            api_version=cfg["azure_api_version"],
            azure_endpoint=cfg["azure_endpoint"],
        )
        return client, deployment

    client_kwargs = {"api_key": cfg["api_key"]}
    if cfg["api_base"]:
        client_kwargs["base_url"] = cfg["api_base"]
    client = OpenAI(**client_kwargs)
    return client, cfg["model_name"]


def _build_async_fallback_client(cfg: Dict[str, str]):
    if cfg["azure_endpoint"]:
        from openai import AsyncAzureOpenAI

        deployment = cfg["azure_deployment"] or cfg["model_name"]
        if not deployment:
            raise RuntimeError("LOCAL_OPENAI fallback missing Azure deployment/model name")
        client = AsyncAzureOpenAI(
            api_key=cfg["api_key"],
            api_version=cfg["azure_api_version"],
            azure_endpoint=cfg["azure_endpoint"],
        )
        return client, deployment

    client_kwargs = {"api_key": cfg["api_key"]}
    if cfg["api_base"]:
        client_kwargs["base_url"] = cfg["api_base"]
    client = AsyncOpenAI(**client_kwargs)
    return client, cfg["model_name"]


def _fallback_sync_request(messages, response_format: Any, requested_model: str):
    candidate_cfgs = _iter_local_openai_fallback_candidates(requested_model)
    if not candidate_cfgs:
        raise RuntimeError("LOCAL_OPENAI fallback missing API key")

    if response_format is not None:
        logger.warning(
            "LOCAL_OPENAI fallback is using condenser-compatible plain chat completion; response_format will be parsed locally."
        )

    max_retries = max(1, _env_int("LOCAL_OPENAI_FALLBACK_MAX_RETRIES", 2))
    last_error: Optional[Exception] = None
    for candidate_idx, cfg in enumerate(candidate_cfgs):
        client, call_model = _build_sync_fallback_client(cfg)
        backend_label = _fallback_backend_label(cfg)
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=call_model,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=16384,
                )
                parsed = _parse_fallback_content(response.choices[0].message.content, response_format)
                if candidate_idx > 0:
                    logger.warning(
                        "LOCAL_OPENAI secondary fallback succeeded via %s model=%s",
                        backend_label,
                        call_model,
                    )
                return response, parsed, call_model, backend_label
            except Exception as exc:
                last_error = exc
                logger.error(
                    "LOCAL_OPENAI fallback attempt %d/%d failed via %s model=%s: %s",
                    attempt + 1,
                    max_retries,
                    backend_label,
                    call_model,
                    exc,
                )
                if attempt >= max_retries - 1:
                    break
        if candidate_idx < len(candidate_cfgs) - 1:
            logger.warning(
                "LOCAL_OPENAI fallback backend %s exhausted; trying secondary backend %s",
                backend_label,
                _fallback_backend_label(candidate_cfgs[candidate_idx + 1]),
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("LOCAL_OPENAI fallback failed without explicit exception")


async def _fallback_async_request(messages, response_format: Any, requested_model: str):
    candidate_cfgs = _iter_local_openai_fallback_candidates(requested_model)
    if not candidate_cfgs:
        raise RuntimeError("LOCAL_OPENAI fallback missing API key")

    if response_format is not None:
        logger.warning(
            "LOCAL_OPENAI fallback is using condenser-compatible plain chat completion; response_format will be parsed locally."
        )

    max_retries = max(1, _env_int("LOCAL_OPENAI_FALLBACK_MAX_RETRIES", 2))
    last_error: Optional[Exception] = None
    for candidate_idx, cfg in enumerate(candidate_cfgs):
        client, call_model = _build_async_fallback_client(cfg)
        backend_label = _fallback_backend_label(cfg)
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=call_model,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=16384,
                )
                parsed = _parse_fallback_content(response.choices[0].message.content, response_format)
                if candidate_idx > 0:
                    logger.warning(
                        "LOCAL_OPENAI secondary fallback succeeded via %s model=%s",
                        backend_label,
                        call_model,
                    )
                return response, parsed, call_model, backend_label
            except Exception as exc:
                last_error = exc
                logger.error(
                    "LOCAL_OPENAI fallback attempt %d/%d failed via %s model=%s: %s",
                    attempt + 1,
                    max_retries,
                    backend_label,
                    call_model,
                    exc,
                )
                if attempt >= max_retries - 1:
                    break
        if candidate_idx < len(candidate_cfgs) - 1:
            logger.warning(
                "LOCAL_OPENAI fallback backend %s exhausted; trying secondary backend %s",
                backend_label,
                _fallback_backend_label(candidate_cfgs[candidate_idx + 1]),
            )

    if last_error is not None:
        raise last_error
    raise RuntimeError("LOCAL_OPENAI fallback failed without explicit exception")


def completion_request(client: OpenAI, **kwargs):
    if "response_format" in kwargs:
        return client.beta.chat.completions.parse(**kwargs)
    return client.chat.completions.create(**kwargs)


async def acompletion_request(client: AsyncOpenAI, **kwargs):
    if "response_format" in kwargs:
        return await client.beta.chat.completions.parse(**kwargs)
    return await client.chat.completions.create(**kwargs)


class LocalOpenAIClient:
    def __init__(self) -> None:
        api_key = os.getenv("LOCAL_OPENAI_API_KEY", "dummy")
        base_urls = _get_local_openai_base_urls()
        if not base_urls:
            raise RuntimeError(
                "LOCAL_OPENAI_BASE_URL (or LOCAL_OPENAI_BASE_URLS) is not set."
            )
        self._clients = [OpenAI(api_key=api_key, base_url=url) for url in base_urls]
        self._lock = threading.Lock()
        self._active_requests = [0] * len(self._clients)
        self._cooldown_until = [0.0] * len(self._clients)
        self._next_index = random.randrange(len(self._clients)) if len(self._clients) > 1 else 0
        self._busy_cooldown_seconds = max(0.0, _env_float("LOCAL_OPENAI_BUSY_COOLDOWN_SECONDS", 2.0))
        self._retry_backoff_seconds = max(0.0, _env_float("LOCAL_OPENAI_RETRY_BACKOFF_SECONDS", 0.2))
        self._max_retries = max(
            1,
            _env_int("LOCAL_OPENAI_MAX_RETRIES", min(10, max(1, len(self._clients) * 2))),
        )
        self._request_timeout_seconds = max(1.0, _env_float("LOCAL_OPENAI_TIMEOUT_SECONDS", 30.0))
        self._fallback_enabled = os.getenv("LOCAL_OPENAI_FALLBACK_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}

    def _acquire_client(self) -> tuple[int, OpenAI]:
        if len(self._clients) == 1:
            with self._lock:
                self._active_requests[0] += 1
            return 0, self._clients[0]

        with self._lock:
            now = time.monotonic()
            candidate = None
            min_active = float("inf")
            start = self._next_index

            for offset in range(len(self._clients)):
                idx = (start + offset) % len(self._clients)
                if self._cooldown_until[idx] > now:
                    continue
                active = self._active_requests[idx]
                if active < min_active:
                    min_active = active
                    candidate = idx
                    if active == 0:
                        break

            if candidate is None:
                candidate = start

            self._next_index = (candidate + 1) % len(self._clients)
            self._active_requests[candidate] += 1
            return candidate, self._clients[candidate]

    def _release_client(self, idx: int, mark_busy: bool) -> None:
        with self._lock:
            self._active_requests[idx] = max(0, self._active_requests[idx] - 1)
            if mark_busy:
                cooldown_deadline = time.monotonic() + self._busy_cooldown_seconds
                self._cooldown_until[idx] = max(self._cooldown_until[idx], cooldown_deadline)
            else:
                self._cooldown_until[idx] = 0.0

    def response(self, count_token: bool = False, **kwargs):
        kwargs.pop("_trace", None)
        kwargs.setdefault("timeout", self._request_timeout_seconds)
        retries = kwargs.pop("max_retries", self._max_retries)
        try:
            retries = max(1, int(retries))
        except (TypeError, ValueError):
            retries = self._max_retries

        last_error = None
        for attempt in range(retries):
            idx, client = self._acquire_client()
            endpoint = str(getattr(client, "base_url", "unknown"))
            try:
                response = completion_request(client, **kwargs)
                self._release_client(idx, mark_busy=False)
                tokens = _extract_tokens(response)
                if "response_format" in kwargs:
                    out = response.choices[0].message.parsed
                    return (out, tokens) if count_token else out
                out = response.choices[0].message.content
                return (out, tokens) if count_token else out
            except Exception as exc:
                retryable = _is_retryable_error(exc)
                self._release_client(idx, mark_busy=retryable)
                last_error = exc
                if not retryable:
                    raise
                if attempt >= retries - 1:
                    break

                sleep_seconds = self._retry_backoff_seconds * (attempt + 1)
                sleep_seconds += random.uniform(0.0, self._retry_backoff_seconds)
                logger.warning(
                    "LocalOpenAI retryable error on %s (attempt %d/%d): %s",
                    endpoint,
                    attempt + 1,
                    retries,
                    exc,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

        if self._fallback_enabled and last_error is not None:
            logger.warning(
                "LocalOpenAI exhausted %d/%d retries; activating fallback backend for model=%s",
                retries,
                retries,
                kwargs.get("model"),
            )
            response, parsed, fallback_model, fallback_backend = _fallback_sync_request(
                messages=kwargs.get("messages"),
                response_format=kwargs.get("response_format"),
                requested_model=str(kwargs.get("model") or ""),
            )
            logger.warning(
                "LocalOpenAI fallback succeeded via %s model=%s",
                fallback_backend,
                fallback_model,
            )
            tokens = _extract_tokens(response)
            if "response_format" in kwargs:
                out = parsed
                return (out, tokens) if count_token else out
            out = response.choices[0].message.content
            return (out, tokens) if count_token else out

        if last_error is not None:
            raise last_error
        raise RuntimeError("LocalOpenAI request failed without explicit exception")


class AsyncLocalOpenAIClient:
    def __init__(self) -> None:
        api_key = os.getenv("LOCAL_OPENAI_API_KEY", "dummy")
        base_urls = _get_local_openai_base_urls()
        if not base_urls:
            raise RuntimeError(
                "LOCAL_OPENAI_BASE_URL (or LOCAL_OPENAI_BASE_URLS) is not set."
            )
        self._clients = [AsyncOpenAI(api_key=api_key, base_url=url) for url in base_urls]
        self._lock = asyncio.Lock()
        self._active_requests = [0] * len(self._clients)
        self._cooldown_until = [0.0] * len(self._clients)
        self._next_index = random.randrange(len(self._clients)) if len(self._clients) > 1 else 0
        self._busy_cooldown_seconds = max(0.0, _env_float("LOCAL_OPENAI_BUSY_COOLDOWN_SECONDS", 2.0))
        self._retry_backoff_seconds = max(0.0, _env_float("LOCAL_OPENAI_RETRY_BACKOFF_SECONDS", 0.2))
        self._max_retries = max(
            1,
            _env_int("LOCAL_OPENAI_MAX_RETRIES", min(10, max(1, len(self._clients) * 2))),
        )
        self._request_timeout_seconds = max(1.0, _env_float("LOCAL_OPENAI_TIMEOUT_SECONDS", 30.0))
        self._fallback_enabled = os.getenv("LOCAL_OPENAI_FALLBACK_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}

    async def _acquire_client(self) -> tuple[int, AsyncOpenAI]:
        if len(self._clients) == 1:
            async with self._lock:
                self._active_requests[0] += 1
            return 0, self._clients[0]

        async with self._lock:
            now = time.monotonic()
            candidate = None
            min_active = float("inf")
            start = self._next_index

            for offset in range(len(self._clients)):
                idx = (start + offset) % len(self._clients)
                if self._cooldown_until[idx] > now:
                    continue
                active = self._active_requests[idx]
                if active < min_active:
                    min_active = active
                    candidate = idx
                    if active == 0:
                        break

            if candidate is None:
                candidate = start

            self._next_index = (candidate + 1) % len(self._clients)
            self._active_requests[candidate] += 1
            return candidate, self._clients[candidate]

    async def _release_client(self, idx: int, mark_busy: bool) -> None:
        async with self._lock:
            self._active_requests[idx] = max(0, self._active_requests[idx] - 1)
            if mark_busy:
                cooldown_deadline = time.monotonic() + self._busy_cooldown_seconds
                self._cooldown_until[idx] = max(self._cooldown_until[idx], cooldown_deadline)
            else:
                self._cooldown_until[idx] = 0.0

    async def response(self, count_token: bool = False, **kwargs):
        kwargs.pop("_trace", None)
        kwargs.setdefault("timeout", self._request_timeout_seconds)
        retries = kwargs.pop("max_retries", self._max_retries)
        try:
            retries = max(1, int(retries))
        except (TypeError, ValueError):
            retries = self._max_retries

        last_error = None
        for attempt in range(retries):
            idx, client = await self._acquire_client()
            endpoint = str(getattr(client, "base_url", "unknown"))
            try:
                response = await acompletion_request(client, **kwargs)
                await self._release_client(idx, mark_busy=False)
                tokens = _extract_tokens(response)
                if "response_format" in kwargs:
                    out = response.choices[0].message.parsed
                    return (out, tokens) if count_token else out
                out = response.choices[0].message.content
                return (out, tokens) if count_token else out
            except Exception as exc:
                retryable = _is_retryable_error(exc)
                await self._release_client(idx, mark_busy=retryable)
                last_error = exc
                if not retryable:
                    raise
                if attempt >= retries - 1:
                    break

                sleep_seconds = self._retry_backoff_seconds * (attempt + 1)
                sleep_seconds += random.uniform(0.0, self._retry_backoff_seconds)
                logger.warning(
                    "AsyncLocalOpenAI retryable error on %s (attempt %d/%d): %s",
                    endpoint,
                    attempt + 1,
                    retries,
                    exc,
                )
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)

        if self._fallback_enabled and last_error is not None:
            logger.warning(
                "AsyncLocalOpenAI exhausted %d/%d retries; activating fallback backend for model=%s",
                retries,
                retries,
                kwargs.get("model"),
            )
            response, parsed, fallback_model, fallback_backend = await _fallback_async_request(
                messages=kwargs.get("messages"),
                response_format=kwargs.get("response_format"),
                requested_model=str(kwargs.get("model") or ""),
            )
            logger.warning(
                "AsyncLocalOpenAI fallback succeeded via %s model=%s",
                fallback_backend,
                fallback_model,
            )
            tokens = _extract_tokens(response)
            if "response_format" in kwargs:
                out = parsed
                return (out, tokens) if count_token else out
            out = response.choices[0].message.content
            return (out, tokens) if count_token else out

        if last_error is not None:
            raise last_error
        raise RuntimeError("AsyncLocalOpenAI request failed without explicit exception")


def _get_local_openai_base_urls() -> list[str]:
    """
    Resolve local OpenAI-compatible base URLs.

    Supports:
      - LOCAL_OPENAI_BASE_URLS: comma-separated list of base URLs
      - LOCAL_OPENAI_BASE_URL: single base URL (fallback)
    """
    base_urls_raw = (os.getenv("LOCAL_OPENAI_BASE_URLS") or "").strip()
    if base_urls_raw:
        urls = [u.strip() for u in base_urls_raw.split(",") if u.strip()]
        if urls:
            return urls
    base_url = (os.getenv("LOCAL_OPENAI_BASE_URL") or "").strip()
    return [base_url] if base_url else []
