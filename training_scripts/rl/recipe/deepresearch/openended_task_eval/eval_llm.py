import asyncio
import logging
import os
import random
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

EvalLLMAddresses = Optional[Union[str, List[str]]]
VLLMChatCallable = Callable[..., Awaitable[Optional[str]]]

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
_local_openai_pools: Dict[tuple, "_AsyncLocalOpenAIPool"] = {}
_local_openai_pools_lock = threading.Lock()


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return ""


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


def _normalize_api_model_override(model_name: str) -> str:
    normalized = (model_name or "").strip()
    if not normalized or normalized == "default":
        return ""

    local_aliases = {"eval_model"}
    env_local_alias = (os.environ.get("EVAL_LLM_MODEL") or "").strip()
    if env_local_alias:
        local_aliases.add(env_local_alias)

    if normalized in local_aliases:
        return ""
    return normalized


def _parse_eval_addresses(eval_llm_addresses: EvalLLMAddresses) -> List[str]:
    if isinstance(eval_llm_addresses, str):
        return [addr.strip() for addr in eval_llm_addresses.split(",") if addr.strip()]
    if isinstance(eval_llm_addresses, list):
        return [str(addr).strip() for addr in eval_llm_addresses if str(addr).strip()]
    return []


def _normalize_local_openai_base_url(address: str) -> str:
    normalized = address.strip()
    if not normalized.startswith("http"):
        normalized = f"http://{normalized}"
    normalized = normalized.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in _RETRYABLE_HTTP_STATUS:
        return True

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int) and response_status in _RETRYABLE_HTTP_STATUS:
        return True

    text = str(exc).lower()
    return any(hint in text for hint in _RETRYABLE_TEXT_HINTS)


def _prefers_max_completion_tokens(model_name: str) -> bool:
    """Heuristic for modern reasoning/chat models that prefer max_completion_tokens."""
    normalized = (model_name or "").strip().lower()
    return normalized.startswith(("o1", "o3", "o4", "gpt-5"))


def _is_unsupported_token_param_error(exc: Exception, param_name: str) -> bool:
    text = str(exc).lower()
    if "unsupported parameter" not in text:
        return False
    token = f"'{param_name}'"
    quoted = f"\"{param_name}\""
    return token in text or quoted in text or param_name in text


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if "temperature" not in text:
        return False
    if "unsupported value" in text:
        return True
    if "does not support" in text and "temperature" in text:
        return True
    return False


def _force_default_temperature(request_kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "temperature" not in request_kwargs:
        return None
    try:
        current = float(request_kwargs["temperature"])
    except (TypeError, ValueError):
        current = None
    if current is not None and abs(current - 1.0) < 1e-9:
        return None
    patched = dict(request_kwargs)
    patched["temperature"] = 1.0
    return patched


def _swap_token_limit_key(request_kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Swap max_tokens <-> max_completion_tokens for one compatibility retry."""
    if "max_tokens" in request_kwargs:
        swapped = dict(request_kwargs)
        value = swapped.pop("max_tokens")
        swapped["max_completion_tokens"] = value
        return swapped
    if "max_completion_tokens" in request_kwargs:
        swapped = dict(request_kwargs)
        value = swapped.pop("max_completion_tokens")
        swapped["max_tokens"] = value
        return swapped
    return None


def _request_kwargs_compat_variants(request_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = [dict(request_kwargs)]

    swapped = _swap_token_limit_key(request_kwargs)
    if swapped and swapped not in variants:
        variants.append(swapped)

    temp_default = _force_default_temperature(request_kwargs)
    if temp_default and temp_default not in variants:
        variants.append(temp_default)
        swapped_temp = _swap_token_limit_key(temp_default)
        if swapped_temp and swapped_temp not in variants:
            variants.append(swapped_temp)

    return variants


def _is_compat_request_error(exc: Exception) -> bool:
    return (
        _is_unsupported_token_param_error(exc, "max_tokens")
        or _is_unsupported_token_param_error(exc, "max_completion_tokens")
        or _is_unsupported_temperature_error(exc)
    )


def _profile_env_candidates(profile: str, key: str) -> List[str]:
    normalized = (profile or "eval").strip().lower()
    if normalized == "citation":
        return [f"CITATION_EVAL_LLM_{key}"]
    if normalized in {"openended", "open-ended"}:
        return [f"OPENENDED_EVAL_LLM_{key}"]
    if normalized in {"eval", "default"}:
        return [f"EVAL_LLM_{key}"]
    return [f"{normalized.upper()}_EVAL_LLM_{key}"]


def _profile_fallback_env_candidates(profile: str, key: str) -> List[str]:
    normalized = (profile or "eval").strip().lower()
    if normalized == "citation":
        return [f"CITATION_EVAL_LLM_FALLBACK_{key}"]
    if normalized in {"openended", "open-ended"}:
        return [f"OPENENDED_EVAL_LLM_FALLBACK_{key}"]
    if normalized in {"eval", "default"}:
        return [f"EVAL_LLM_FALLBACK_{key}"]
    return [f"{normalized.upper()}_EVAL_LLM_FALLBACK_{key}"]


def _profile_local_fallback_env_candidates(profile: str, key: str) -> List[str]:
    normalized = (profile or "eval").strip().lower()
    if normalized == "citation":
        return [f"CITATION_EVAL_LLM_LOCAL_FALLBACK_{key}"]
    if normalized in {"openended", "open-ended"}:
        return [f"OPENENDED_EVAL_LLM_LOCAL_FALLBACK_{key}"]
    if normalized in {"eval", "default"}:
        return [f"EVAL_LLM_LOCAL_FALLBACK_{key}"]
    return [f"{normalized.upper()}_EVAL_LLM_LOCAL_FALLBACK_{key}"]


def _resolve_eval_llm_timeout_seconds(profile: str, explicit_timeout: Optional[int]) -> int:
    if explicit_timeout is not None:
        return explicit_timeout
    raw_timeout = _env_first(*_profile_env_candidates(profile, "TIMEOUT_SECONDS"))
    if not raw_timeout:
        return 120
    try:
        return int(raw_timeout)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid eval LLM timeout for profile=%s: %r. Falling back to 120s.",
            profile,
            raw_timeout,
        )
        return 120


def resolve_eval_llm_api_config(model_name: str = "default", profile: str = "eval") -> Dict[str, str]:
    model_from_arg = _normalize_api_model_override(model_name)
    model_from_env = _env_first(*_profile_env_candidates(profile, "MODEL_NAME"))
    resolved_model = model_from_arg or model_from_env or "gpt-4o-mini"

    cfg = {
        "provider": (
            _env_first(*_profile_env_candidates(profile, "PROVIDER")) or "auto"
        ).lower(),
        "api_key": _env_first(*_profile_env_candidates(profile, "API_KEY")),
        "api_base": _env_first(*_profile_env_candidates(profile, "API_BASE")),
        "model_name": resolved_model,
        "azure_endpoint": _env_first(*_profile_env_candidates(profile, "AZURE_ENDPOINT")),
        "azure_api_version": _env_first(*_profile_env_candidates(profile, "AZURE_API_VERSION"))
        or "2024-08-01-preview",
        "azure_deployment": _env_first(*_profile_env_candidates(profile, "AZURE_DEPLOYMENT")),
    }
    return cfg


def resolve_eval_llm_api_fallback_config(model_name: str = "default", profile: str = "eval") -> Dict[str, str]:
    model_from_arg = _normalize_api_model_override(model_name)
    model_from_env = _env_first(*_profile_fallback_env_candidates(profile, "MODEL_NAME"))
    resolved_model = model_from_env or model_from_arg or ""

    cfg = {
        "provider": (
            _env_first(*_profile_fallback_env_candidates(profile, "PROVIDER")) or "auto"
        ).lower(),
        "api_key": _env_first(*_profile_fallback_env_candidates(profile, "API_KEY")),
        "api_base": _env_first(*_profile_fallback_env_candidates(profile, "API_BASE")),
        "model_name": resolved_model,
        "azure_endpoint": _env_first(*_profile_fallback_env_candidates(profile, "AZURE_ENDPOINT")),
        "azure_api_version": _env_first(*_profile_fallback_env_candidates(profile, "AZURE_API_VERSION"))
        or "2024-08-01-preview",
        "azure_deployment": _env_first(*_profile_fallback_env_candidates(profile, "AZURE_DEPLOYMENT")),
    }
    return cfg


def resolve_eval_llm_local_fallback_model_name(model_name: str = "default", profile: str = "eval") -> str:
    model_from_arg = _normalize_api_model_override(model_name)
    model_from_env = _env_first(
        *_profile_local_fallback_env_candidates(profile, "MODEL_NAME"),
    )
    return model_from_env or model_from_arg or ""


def _has_eval_llm_api_target(cfg: Dict[str, str]) -> bool:
    if not cfg["api_key"]:
        return False
    provider = cfg["provider"]
    if provider == "vllm":
        return False
    if provider == "azure":
        return bool(cfg["azure_endpoint"])
    if provider in {"openai", "api", "local_openai"}:
        return True
    return bool(cfg["azure_endpoint"] or cfg["api_base"] or cfg["api_key"])


def _iter_eval_llm_api_candidate_configs(model_name: str = "default", profile: str = "eval") -> List[tuple[str, Dict[str, str]]]:
    primary_cfg = resolve_eval_llm_api_config(model_name=model_name, profile=profile)
    fallback_cfg = resolve_eval_llm_api_fallback_config(model_name=model_name, profile=profile)

    candidates: List[tuple[str, Dict[str, str]]] = []
    if _has_eval_llm_api_target(primary_cfg):
        candidates.append(("primary", primary_cfg))

    if _has_eval_llm_api_target(fallback_cfg):
        same_target = (
            primary_cfg["provider"] == fallback_cfg["provider"]
            and primary_cfg["api_key"] == fallback_cfg["api_key"]
            and primary_cfg["api_base"] == fallback_cfg["api_base"]
            and primary_cfg["model_name"] == fallback_cfg["model_name"]
            and primary_cfg["azure_endpoint"] == fallback_cfg["azure_endpoint"]
            and primary_cfg["azure_deployment"] == fallback_cfg["azure_deployment"]
            and primary_cfg["azure_api_version"] == fallback_cfg["azure_api_version"]
        )
        if not same_target:
            candidates.append(("fallback", fallback_cfg))

    return candidates


def has_eval_llm_api_client_config(model_name: str = "default", profile: str = "eval") -> bool:
    return bool(_iter_eval_llm_api_candidate_configs(model_name=model_name, profile=profile))


def _has_eval_llm_api_fallback_target(profile: str = "eval") -> bool:
    return bool(_iter_eval_llm_api_candidate_configs(model_name="default", profile=profile))


def has_eval_llm_backend(
    eval_llm_addresses: EvalLLMAddresses,
    model_name: str = "default",
    profile: str = "eval",
) -> bool:
    if isinstance(eval_llm_addresses, str):
        if any(addr.strip() for addr in eval_llm_addresses.split(",")):
            return True
    elif isinstance(eval_llm_addresses, list):
        if len(eval_llm_addresses) > 0:
            return True
    return has_eval_llm_api_client_config(model_name=model_name, profile=profile)


class _AsyncLocalOpenAIPool:
    """Async local-openai pool with least-active routing and cooldown."""

    def __init__(self, base_urls: List[str], api_key: str):
        from openai import AsyncOpenAI

        self._clients = [AsyncOpenAI(api_key=api_key, base_url=url) for url in base_urls]
        self._lock = asyncio.Lock()
        self._active_requests = [0] * len(self._clients)
        self._cooldown_until = [0.0] * len(self._clients)
        self._next_index = random.randrange(len(self._clients)) if len(self._clients) > 1 else 0
        self._busy_cooldown_seconds = max(0.0, _env_float("LOCAL_OPENAI_BUSY_COOLDOWN_SECONDS", 2.0))
        self._retry_backoff_seconds = max(0.0, _env_float("LOCAL_OPENAI_RETRY_BACKOFF_SECONDS", 0.2))
        self._default_max_retries = max(
            1,
            _env_int("LOCAL_OPENAI_MAX_RETRIES", min(10, max(1, len(self._clients) * 2))),
        )

    async def _acquire_client(self) -> tuple[int, Any]:
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

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        timeout_seconds: int,
        max_retries: int,
        profile: str,
        **kwargs,
    ) -> Optional[str]:
        try:
            retries = max(1, int(max_retries))
        except (TypeError, ValueError):
            retries = self._default_max_retries

        last_error: Optional[Exception] = None
        for attempt in range(retries):
            idx, client = await self._acquire_client()
            endpoint = str(getattr(client, "base_url", "unknown"))

            try:
                request_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                request_kwargs.update(kwargs)
                request_kwargs.setdefault("timeout", timeout_seconds)
                if "max_tokens" not in request_kwargs and "max_completion_tokens" not in request_kwargs:
                    if _prefers_max_completion_tokens(model):
                        request_kwargs["max_completion_tokens"] = max_tokens
                    else:
                        request_kwargs["max_tokens"] = max_tokens

                variants = _request_kwargs_compat_variants(request_kwargs)
                response = None
                last_exc: Optional[Exception] = None
                for idx_variant, req_kwargs in enumerate(variants):
                    try:
                        response = await client.chat.completions.create(**req_kwargs)
                        break
                    except Exception as exc_variant:
                        last_exc = exc_variant
                        if (not _is_compat_request_error(exc_variant)) or idx_variant == len(variants) - 1:
                            raise
                if response is None:
                    if last_exc is not None:
                        raise last_exc
                    raise RuntimeError("LocalOpenAI eval request failed without response")
                await self._release_client(idx, mark_busy=False)
                content = response.choices[0].message.content
                return str(content) if content is not None else None
            except Exception as exc:
                retryable = _is_retryable_error(exc)
                await self._release_client(idx, mark_busy=retryable)
                last_error = exc

                if not retryable or attempt >= retries - 1:
                    logger.error(
                        "LocalOpenAI eval request failed on %s (profile=%s): %s",
                        endpoint,
                        profile,
                        exc,
                    )
                    break

                sleep_seconds = self._retry_backoff_seconds * (attempt + 1)
                sleep_seconds += random.uniform(0.0, self._retry_backoff_seconds)
                logger.warning(
                    "LocalOpenAI retryable error on %s (profile=%s, attempt %d/%d): %s",
                    endpoint,
                    profile,
                    attempt + 1,
                    retries,
                    exc,
                )
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)

        if last_error is not None:
            logger.error("LocalOpenAI eval request exhausted retries (profile=%s): %s", profile, last_error)
        return None


def _get_local_openai_pool(eval_llm_addresses: EvalLLMAddresses, api_key: str) -> Optional[_AsyncLocalOpenAIPool]:
    addresses = _parse_eval_addresses(eval_llm_addresses)
    if not addresses:
        return None
    base_urls = [_normalize_local_openai_base_url(addr) for addr in addresses]
    cache_key = (tuple(base_urls), api_key)
    with _local_openai_pools_lock:
        pool = _local_openai_pools.get(cache_key)
        if pool is None:
            pool = _AsyncLocalOpenAIPool(base_urls=base_urls, api_key=api_key)
            _local_openai_pools[cache_key] = pool
        return pool


async def _chat_completions_local_openai_async(
    eval_llm_addresses: EvalLLMAddresses,
    messages: List[Dict[str, str]],
    model: str,
    profile: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    max_retries: int,
    api_key: str,
    **kwargs,
) -> Optional[str]:
    resolved_api_key = api_key or os.environ.get("LOCAL_OPENAI_API_KEY", "dummy")
    pool = _get_local_openai_pool(eval_llm_addresses=eval_llm_addresses, api_key=resolved_api_key)
    if pool is None:
        logger.error("No local_openai addresses configured for profile=%s.", profile)
        return None
    result = await pool.complete(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        profile=profile,
        **kwargs,
    )
    if result is not None:
        return result

    if not _has_eval_llm_api_fallback_target(profile=profile):
        logger.error(
            "LocalOpenAI exhausted retries for profile=%s and no API fallback target is configured.",
            profile,
        )
        return None

    logger.warning(
        "LocalOpenAI exhausted retries for profile=%s model=%s; falling back to OpenAI/Azure API client.",
        profile,
        model,
    )
    loop = asyncio.get_running_loop()
    fallback_result = await loop.run_in_executor(
        None,
        lambda: _chat_completions_openai_style_sync(
            messages=messages,
            model="default",
            profile=profile,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            **kwargs,
        ),
    )
    if fallback_result is None:
        logger.error(
            "OpenAI/Azure API fallback failed after LocalOpenAI exhaustion (profile=%s).",
            profile,
        )
        return None

    logger.warning(
        "OpenAI/Azure API fallback succeeded after LocalOpenAI exhaustion (profile=%s).",
        profile,
    )
    return fallback_result


def _chat_completions_openai_style_sync(
    messages: List[Dict[str, str]],
    model: str = "default",
    profile: str = "eval",
    max_tokens: int = 16384,
    temperature: float = 1.0,
    timeout_seconds: int = 120,
    **kwargs,
) -> Optional[str]:
    candidate_cfgs = _iter_eval_llm_api_candidate_configs(model_name=model, profile=profile)
    if not candidate_cfgs:
        logger.error("No API key configured for eval LLM API-client mode (profile=%s).", profile)
        return None

    last_error: Optional[Exception] = None
    for candidate_idx, (candidate_label, cfg) in enumerate(candidate_cfgs):
        provider = cfg["provider"]
        if provider == "vllm":
            continue
        if provider == "azure":
            use_azure = True
        elif provider in {"openai", "api", "local_openai"}:
            use_azure = False
        else:
            use_azure = bool(cfg["azure_endpoint"])

        try:
            from openai import AzureOpenAI, OpenAI

            call_model = cfg["model_name"]
            if use_azure:
                if not cfg["azure_endpoint"]:
                    logger.error(
                        "Azure eval LLM selected but no azure endpoint is configured (profile=%s, target=%s).",
                        profile,
                        candidate_label,
                    )
                    continue
                deployment = cfg["azure_deployment"] or call_model
                if not deployment:
                    logger.error(
                        "Azure eval LLM requires deployment/model name (profile=%s, target=%s).",
                        profile,
                        candidate_label,
                    )
                    continue
                client = AzureOpenAI(
                    api_key=cfg["api_key"],
                    api_version=cfg["azure_api_version"] or "2024-08-01-preview",
                    azure_endpoint=cfg["azure_endpoint"],
                )
                call_model = deployment
            else:
                client_kwargs: Dict[str, Any] = {"api_key": cfg["api_key"]}
                if cfg["api_base"]:
                    client_kwargs["base_url"] = cfg["api_base"]
                client = OpenAI(**client_kwargs)

            request_kwargs: Dict[str, Any] = {
                "model": call_model,
                "messages": messages,
                "temperature": temperature,
            }
            if use_azure:
                # Azure chat-completions expects temperature=1 for these eval paths.
                request_kwargs["temperature"] = 1.0
            request_kwargs.update(kwargs)
            request_kwargs.setdefault("timeout", timeout_seconds)
            if "max_tokens" not in request_kwargs and "max_completion_tokens" not in request_kwargs:
                if _prefers_max_completion_tokens(call_model):
                    request_kwargs["max_completion_tokens"] = max_tokens
                else:
                    request_kwargs["max_tokens"] = max_tokens

            variants = _request_kwargs_compat_variants(request_kwargs)
            response = None
            last_exc: Optional[Exception] = None
            for idx_variant, req_kwargs in enumerate(variants):
                try:
                    response = client.chat.completions.create(**req_kwargs)
                    break
                except Exception as exc_variant:
                    last_exc = exc_variant
                    if (not _is_compat_request_error(exc_variant)) or idx_variant == len(variants) - 1:
                        raise
            if response is None:
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError("OpenAI/Azure eval request failed without response")
            content = response.choices[0].message.content
            if content is None:
                return None
            if candidate_idx > 0:
                logger.warning(
                    "Eval LLM API-client fallback succeeded (profile=%s, target=%s).",
                    profile,
                    candidate_label,
                )
            return str(content)
        except Exception as e:
            last_error = e
            if candidate_idx < len(candidate_cfgs) - 1:
                logger.warning(
                    "Eval LLM API client failed (profile=%s, target=%s): %s. Falling back to next API target.",
                    profile,
                    candidate_label,
                    e,
                )
                continue
            logger.error(
                "Error calling eval LLM via OpenAI/Azure client (profile=%s): %s",
                profile,
                e,
            )
            return None

    if last_error is not None:
        logger.error(
            "Error calling eval LLM via OpenAI/Azure client (profile=%s): %s",
            profile,
            last_error,
        )
    return None


async def chat_completions_eval_llm(
    eval_llm_addresses: EvalLLMAddresses,
    messages: List[Dict[str, str]],
    model: str = "default",
    profile: str = "eval",
    max_tokens: int = 16384,
    temperature: float = 0.0,
    timeout_seconds: Optional[int] = None,
    max_retries: int = 3,
    load_balance_strategy: str = "least_connections",
    vllm_chat_fn: Optional[VLLMChatCallable] = None,
    **kwargs,
) -> Optional[str]:
    timeout_seconds = _resolve_eval_llm_timeout_seconds(profile, timeout_seconds)
    cfg = resolve_eval_llm_api_config(model_name=model, profile=profile)
    provider = cfg["provider"]

    if provider in {"azure", "openai", "api"}:
        use_api_client = True
    elif provider in {"vllm", "local_openai"}:
        use_api_client = False
    else:
        use_api_client = False if eval_llm_addresses else has_eval_llm_api_client_config(model_name=model, profile=profile)

    if use_api_client:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _chat_completions_openai_style_sync(
                messages=messages,
                model=model,
                profile=profile,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                **kwargs,
            ),
        )
        if result is not None:
            return result

        if eval_llm_addresses:
            local_model = resolve_eval_llm_local_fallback_model_name(model_name=model, profile=profile)
            if local_model:
                logger.warning(
                    "Eval LLM API-client chain failed (profile=%s); falling back to local_openai with model=%s.",
                    profile,
                    local_model,
                )
                return await _chat_completions_local_openai_async(
                    eval_llm_addresses=eval_llm_addresses,
                    messages=messages,
                    model=local_model,
                    profile=profile,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    api_key=os.environ.get("LOCAL_OPENAI_API_KEY", "dummy"),
                    **kwargs,
                )
        return None

    if provider == "local_openai":
        return await _chat_completions_local_openai_async(
            eval_llm_addresses=eval_llm_addresses,
            messages=messages,
            model=model,
            profile=profile,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            api_key=cfg["api_key"],
            **kwargs,
        )

    if not eval_llm_addresses:
        logger.error(
            "No eval LLM backend configured for profile=%s. Provide addresses or API-client config.",
            profile,
        )
        return None
    if vllm_chat_fn is None:
        logger.error("vLLM caller is not provided for profile=%s.", profile)
        return None

    return await vllm_chat_fn(
        address=eval_llm_addresses,
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        load_balance_strategy=load_balance_strategy,
        **kwargs,
    )
