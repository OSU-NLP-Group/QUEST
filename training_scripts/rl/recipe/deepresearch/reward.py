# Copyright 2025 DeepResearch authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reward function for DeepResearch RL training.
Supports dynamic eval scripts per task with async API calls.
Migrated from slime/examples/deepresearch/deepresearch_reward.py
"""

import asyncio
import base64
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import logging
import os
import re
import string
import sys
import threading
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
from typing import Union  # noqa: F811  (also imported below; kept for locality)
from urllib.request import Request, urlopen

import aiohttp

logger = logging.getLogger(__name__)
try:
    from recipe.deepresearch.openended_task_eval.eval_llm import (
        chat_completions_eval_llm as drb_chat_completions_eval_llm,
        has_eval_llm_backend as drb_has_eval_llm_backend,
    )
    from recipe.deepresearch.citation_task_eval.inline_citation import (
        INLINE_CITATION_MAX_SCORE_DEFAULT as DRB_INLINE_CITATION_MAX_SCORE_DEFAULT,
        compute_inline_citation_score as drb_compute_inline_citation_score,
    )
    from recipe.deepresearch.openended_task_eval.openended.scoring import compute_score_openended as drb_compute_score_openended
except Exception:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)
    from openended_task_eval.eval_llm import (
        chat_completions_eval_llm as drb_chat_completions_eval_llm,
        has_eval_llm_backend as drb_has_eval_llm_backend,
    )
    from citation_task_eval.inline_citation import (
        INLINE_CITATION_MAX_SCORE_DEFAULT as DRB_INLINE_CITATION_MAX_SCORE_DEFAULT,
        compute_inline_citation_score as drb_compute_inline_citation_score,
    )
    from openended_task_eval.openended.scoring import compute_score_openended as drb_compute_score_openended

_INLINE_CITATION_MAX_SCORE_DEFAULT = DRB_INLINE_CITATION_MAX_SCORE_DEFAULT
_BASE_SCORE_WEIGHT_DEFAULT = 0.75


def _normalize_eval_profile_name(raw_name: str) -> str:
    normalized = (raw_name or "").strip().lower()
    alias_map = {
        "default": "obj",
        "eval": "obj",
        "objective": "obj",
        "obj": "obj",
        "main": "obj",
        "openended": "openended",
        "open-ended": "openended",
        "citation": "citation",
        "cite": "citation",
    }
    return alias_map.get(normalized, normalized)


def _is_openended_type(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"openended", "open-ended"}


def _is_openended_eval_type(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized.startswith(("openended", "open-ended"))


def _split_conf_list_values(raw_value: str) -> List[str]:
    value = (raw_value or "").strip()
    if not value:
        return []
    return [part for part in re.split(r"[,\s]+", value) if part]


def _dedup_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def load_eval_llm_profiles_from_file(conf_path: str) -> Dict[str, Dict[str, str]]:
    """Load per-profile eval LLM config from a text conf file.

    Supported formats (can be mixed):
    1) Legacy one-IP-per-line (applies to obj profile)
    2) Section format:
       [obj] / [openended] / [citation]
       ips=...
       ports=...
       model=...
       addresses=...
    3) Prefixed keys:
       obj_ips=..., openended_model=..., citation_addresses=...
       obj.model=..., openended.ports=...
    """
    p = Path(conf_path)
    if not p.is_file():
        return {}

    structured: Dict[str, Dict[str, Any]] = {}

    def _entry(profile_name: str) -> Dict[str, Any]:
        profile = _normalize_eval_profile_name(profile_name)
        if profile not in structured:
            structured[profile] = {
                "ips": [],
                "ports": [],
                "addresses": [],
                "model": "",
            }
        return structured[profile]

    current_profile = "obj"
    for raw_line in p.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            normalized_section = _normalize_eval_profile_name(section)
            if normalized_section in {"obj", "openended", "citation"}:
                current_profile = normalized_section
            continue

        target_profile = current_profile
        key = ""
        value = ""
        if "=" in line:
            raw_key, raw_value = line.split("=", 1)
            key = raw_key.strip().lower()
            value = raw_value.strip()
        else:
            _entry(target_profile)["ips"].extend(_split_conf_list_values(line))
            continue

        if not key:
            continue

        parsed_profile: Optional[str] = None
        parsed_key = key

        if "." in parsed_key:
            maybe_profile, maybe_key = parsed_key.split(".", 1)
            normalized_profile = _normalize_eval_profile_name(maybe_profile)
            if normalized_profile in {"obj", "openended", "citation"}:
                parsed_profile = normalized_profile
                parsed_key = maybe_key.strip().lower()
        elif "_" in parsed_key:
            maybe_profile, maybe_key = parsed_key.split("_", 1)
            normalized_profile = _normalize_eval_profile_name(maybe_profile)
            if normalized_profile in {"obj", "openended", "citation"}:
                parsed_profile = normalized_profile
                parsed_key = maybe_key.strip().lower()

        if parsed_profile is not None:
            target_profile = parsed_profile

        entry = _entry(target_profile)
        if parsed_key in {"ip", "ips", "nodes", "hosts"}:
            entry["ips"].extend(_split_conf_list_values(value))
        elif parsed_key in {"port", "ports"}:
            entry["ports"].extend(_split_conf_list_values(value))
        elif parsed_key in {"address", "addresses", "base_url", "base_urls"}:
            entry["addresses"].extend(_split_conf_list_values(value))
        elif parsed_key in {"model", "model_name"}:
            if value:
                entry["model"] = value

    out: Dict[str, Dict[str, str]] = {}
    for profile, cfg in structured.items():
        profile_cfg: Dict[str, str] = {}
        ips = _dedup_keep_order([x.strip() for x in cfg.get("ips", []) if str(x).strip()])
        if ips:
            profile_cfg["ips"] = ",".join(ips)
        ports = _dedup_keep_order([x.strip() for x in cfg.get("ports", []) if str(x).strip()])
        if ports:
            profile_cfg["ports"] = ",".join(ports)
        addresses = _dedup_keep_order([x.strip() for x in cfg.get("addresses", []) if str(x).strip()])
        if addresses:
            profile_cfg["addresses"] = ",".join(addresses)
        model = str(cfg.get("model", "")).strip()
        if model:
            profile_cfg["model"] = model
        if profile_cfg:
            out[profile] = profile_cfg
    return out


def load_eval_llm_ips_from_file(conf_path: str) -> Optional[str]:
    """Backward-compatible helper for obj-profile IP loading."""
    profiles = load_eval_llm_profiles_from_file(conf_path)
    obj_cfg = profiles.get("obj", {})
    ips = obj_cfg.get("ips")
    return ips.strip() if isinstance(ips, str) and ips.strip() else None


def _resolve_eval_llm_profile_runtime_config(
    profile: str,
    eval_llm_nodes_conf: Optional[str],
    default_addresses: Optional[Union[str, List[str]]],
    default_model: str,
) -> tuple[Optional[Union[str, List[str]]], str]:
    """Resolve the latest backend config for one eval profile."""
    resolved_addresses = default_addresses
    resolved_model = default_model

    if not eval_llm_nodes_conf:
        return resolved_addresses, resolved_model

    normalized_profile = _normalize_eval_profile_name(profile)
    if normalized_profile not in {"obj", "openended", "citation"}:
        return resolved_addresses, resolved_model

    profile_cfg = load_eval_llm_profiles_from_file(eval_llm_nodes_conf).get(normalized_profile, {})
    if not profile_cfg:
        return resolved_addresses, resolved_model

    if profile_cfg.get("addresses"):
        resolved_addresses = profile_cfg["addresses"]
    elif profile_cfg.get("ips") and profile_cfg.get("ports"):
        resolved_addresses = generate_server_addresses(
            profile_cfg["ips"],
            profile_cfg["ports"],
        )

    if profile_cfg.get("model"):
        resolved_model = str(profile_cfg["model"]).strip() or resolved_model

    return resolved_addresses, resolved_model
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Cache eval scripts that failed to import/compile so we can
# avoid repeated expensive traceback spam for the same broken file.
_EVAL_SCRIPT_LOAD_FAILURES: Dict[str, str] = {}


# ============== Default Scoring Utilities ==============

def normalize_answer(s: str) -> str:
    """
    Normalize answer for comparison.
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Fix whitespace
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers: List[str]) -> bool:
    """Check if prediction matches any golden answer (exact match)."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return True
    return False


def fuzzy_match(prediction: str, golden_answers: List[str]) -> bool:
    """Check if prediction contains any golden answer (fuzzy match)."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        normalized_gold = normalize_answer(golden_answer)
        if normalized_gold in normalized_prediction or normalized_prediction in normalized_gold:
            return True
    return False


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from the response."""
    pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(pattern, text, re.DOTALL))

    if not matches:
        return None

    return matches[-1].group(1).strip()


def extract_tool_responses(text: str) -> List[str]:
    """Extract all tool response blocks from the text."""
    pattern = r"<tool_response>(.*?)</tool_response>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches]


def is_valid_format(text: str) -> tuple:
    """Check if the response follows the expected format."""
    # Check for think tags
    think_open = len(re.findall(r"<think>", text))
    think_close = len(re.findall(r"</think>", text))
    if think_open != think_close:
        return False, f"Unbalanced think tags: {think_open} open, {think_close} close"

    # Check for answer tags
    answer_open = len(re.findall(r"<answer>", text))
    answer_close = len(re.findall(r"</answer>", text))
    if answer_open != answer_close:
        return False, f"Unbalanced answer tags: {answer_open} open, {answer_close} close"

    # Check for tool_call tags
    tool_open = len(re.findall(r"<tool_call>", text))
    tool_close = len(re.findall(r"</tool_call>", text))
    if tool_open != tool_close:
        return False, f"Unbalanced tool_call tags: {tool_open} open, {tool_close} close"

    # Must have answer or tool calls
    if answer_open > 0:
        return True, "Valid format with answer"
    if tool_open > 0:
        return True, "Valid format with tool calls"

    return False, "No answer or tool calls found"


def count_tool_calls(text: str) -> Dict[str, int]:
    """Count the number of each type of tool call."""
    counts = {"search": 0, "google_scholar": 0, "PythonInterpreter": 0, "visit": 0, "total": 0}

    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(tool_call_pattern, text, re.DOTALL)

    for match in matches:
        counts["total"] += 1
        if '"name": "search"' in match or '"name":"search"' in match:
            counts["search"] += 1
        elif '"name": "google_scholar"' in match or '"name":"google_scholar"' in match:
            counts["google_scholar"] += 1
        elif '"name": "PythonInterpreter"' in match or '"name":"PythonInterpreter"' in match:
            counts["PythonInterpreter"] += 1
        elif '"name": "visit"' in match or '"name":"visit"' in match:
            counts["visit"] += 1

    return counts


def check_retrieval_quality(text: str, golden_answers: List[str]) -> bool:
    """Check if any tool response contains relevant information."""
    tool_responses = extract_tool_responses(text)
    for response in tool_responses:
        for golden in golden_answers:
            if normalize_answer(golden) in normalize_answer(response):
                return True
    return False


def _coerce_bool_flag(value: Any, default: bool = False) -> bool:
    """Parse bool-like runtime flags from bool/int/str."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    return default


def _coerce_int_flag(
    value: Any,
    default: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """Parse int-like runtime flags with optional clamping."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _coerce_float_flag(
    value: Any,
    default: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """Parse float-like runtime flags with optional clamping."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


# ============== Load Balancer for Multiple vLLM Servers ==============

import random
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Union


def generate_server_addresses(
    ips: Union[str, List[str]],
    ports: Union[str, int, List[int]],
) -> List[str]:
    """
    Generate all combinations of IP addresses and ports.

    Args:
        ips: Single IP, comma-separated IPs, or list of IPs
             Examples: "10.0.0.1", "10.0.0.1,10.0.0.2,10.0.0.3", ["10.0.0.1", "10.0.0.2"]
        ports: Single port, comma-separated ports, or list of ports
               Examples: 8000, "8000,8001,8002", [8000, 8001, 8002]

    Returns:
        List of server addresses like ["http://10.0.0.1:8000", "http://10.0.0.1:8001", ...]

    Example:
        >>> generate_server_addresses("10.0.0.1,10.0.0.2", "8000,8001")
        ['http://10.0.0.1:8000', 'http://10.0.0.1:8001',
         'http://10.0.0.2:8000', 'http://10.0.0.2:8001']
    """
    # Parse IPs
    if isinstance(ips, str):
        ip_list = [ip.strip() for ip in ips.split(",") if ip.strip()]
    else:
        ip_list = list(ips)

    # Parse ports
    if isinstance(ports, int):
        port_list = [ports]
    elif isinstance(ports, str):
        port_list = [int(p.strip()) for p in ports.split(",") if p.strip()]
    else:
        port_list = list(ports)

    # Generate all combinations
    addresses = []
    for ip in ip_list:
        # Remove http:// prefix if present
        ip = ip.replace("http://", "").replace("https://", "")
        # Remove port if already in IP
        if ":" in ip:
            ip = ip.split(":")[0]

        for port in port_list:
            addresses.append(f"http://{ip}:{port}")

    return addresses


@dataclass
class ServerStats:
    """Statistics for a single server."""
    address: str
    active_requests: int = 0
    total_requests: int = 0
    total_errors: int = 0
    is_healthy: bool = True
    last_error_time: float = 0.0


class VLLMLoadBalancer:
    """
    Load balancer for multiple vLLM servers.

    Supports:
    - Round-robin load balancing
    - Least-connections load balancing
    - Health checking with automatic failover
    - Thread-safe request counting

    Usage:
        # Single address (backward compatible)
        balancer = VLLMLoadBalancer("http://127.0.0.1:8000")

        # Multiple addresses (comma-separated string)
        balancer = VLLMLoadBalancer("http://10.0.0.1:8000,http://10.0.0.2:8000,http://10.0.0.3:8000")

        # Multiple addresses (list)
        balancer = VLLMLoadBalancer(["http://10.0.0.1:8000", "http://10.0.0.2:8000"])
    """

    def __init__(
        self,
        addresses: Union[str, List[str]],
        strategy: str = "least_connections",  # "round_robin" or "least_connections"
        health_check_interval: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize the load balancer.

        Args:
            addresses: Single address, comma-separated addresses, or list of addresses
            strategy: Load balancing strategy ("round_robin" or "least_connections")
            health_check_interval: Seconds before retrying an unhealthy server
            max_retries: Maximum retries across all servers
        """
        # Parse addresses
        if isinstance(addresses, str):
            # Support comma-separated addresses
            self.addresses = [addr.strip() for addr in addresses.split(",") if addr.strip()]
        else:
            self.addresses = list(addresses)

        # Normalize addresses
        self.addresses = [self._normalize_address(addr) for addr in self.addresses]

        if not self.addresses:
            raise ValueError("At least one server address is required")

        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.max_retries = max_retries

        # Server stats
        self._stats: Dict[str, ServerStats] = {
            addr: ServerStats(address=addr) for addr in self.addresses
        }
        self._lock = threading.Lock()
        self._round_robin_index = 0

        logger.info(f"VLLMLoadBalancer initialized with {len(self.addresses)} servers: {self.addresses}")

    def _normalize_address(self, address: str) -> str:
        """Normalize server address."""
        address = address.strip()
        if not address.startswith("http"):
            address = f"http://{address}"
        return address

    def _get_healthy_servers(self) -> List[str]:
        """Get list of healthy servers."""
        import time
        current_time = time.time()
        healthy = []

        with self._lock:
            for addr, stats in self._stats.items():
                # Re-enable server after health_check_interval
                if not stats.is_healthy:
                    if current_time - stats.last_error_time > self.health_check_interval:
                        stats.is_healthy = True
                        logger.info(f"Server {addr} marked as healthy (retry)")

                if stats.is_healthy:
                    healthy.append(addr)

        # If all servers are unhealthy, return all (forced retry)
        if not healthy:
            logger.warning("All servers unhealthy, forcing retry on all")
            return self.addresses

        return healthy

    def get_next_server(self) -> str:
        """Get the next server based on load balancing strategy."""
        healthy_servers = self._get_healthy_servers()

        if self.strategy == "round_robin":
            with self._lock:
                # Filter to only healthy servers
                self._round_robin_index = self._round_robin_index % len(healthy_servers)
                server = healthy_servers[self._round_robin_index]
                self._round_robin_index = (self._round_robin_index + 1) % len(healthy_servers)
            return server

        elif self.strategy == "least_connections":
            with self._lock:
                # Find server with least active requests
                min_requests = float('inf')
                best_server = healthy_servers[0]

                for addr in healthy_servers:
                    stats = self._stats[addr]
                    if stats.active_requests < min_requests:
                        min_requests = stats.active_requests
                        best_server = addr

                return best_server

        else:
            # Random fallback
            return random.choice(healthy_servers)

    def acquire(self, address: str):
        """Mark a request as starting on a server."""
        with self._lock:
            self._stats[address].active_requests += 1
            self._stats[address].total_requests += 1

    def release(self, address: str, success: bool = True):
        """Mark a request as completed on a server."""
        import time
        with self._lock:
            stats = self._stats[address]
            stats.active_requests = max(0, stats.active_requests - 1)

            if not success:
                stats.total_errors += 1
                stats.is_healthy = False
                stats.last_error_time = time.time()
                logger.warning(f"Server {address} marked as unhealthy (errors: {stats.total_errors})")

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all servers."""
        with self._lock:
            return {
                addr: {
                    "active_requests": stats.active_requests,
                    "total_requests": stats.total_requests,
                    "total_errors": stats.total_errors,
                    "is_healthy": stats.is_healthy,
                }
                for addr, stats in self._stats.items()
            }


# Global load balancer instance (lazy initialized)
_global_load_balancer: Optional[VLLMLoadBalancer] = None
_load_balancer_lock = threading.Lock()
_shared_local_openai_clients: Dict[tuple[Any, ...], Any] = {}
_shared_local_openai_clients_lock = threading.Lock()
_shared_citation_visit_tool: Optional[Any] = None
_shared_citation_visit_tool_lock = threading.Lock()


def get_load_balancer(addresses: Union[str, List[str]], **kwargs) -> VLLMLoadBalancer:
    """Get or create a global load balancer instance. Recreates if addresses change."""
    global _global_load_balancer

    # Normalize for comparison
    if isinstance(addresses, str):
        new_addrs = sorted(addr.strip() for addr in addresses.split(",") if addr.strip())
    else:
        new_addrs = sorted(addresses)

    with _load_balancer_lock:
        if _global_load_balancer is not None:
            if sorted(_global_load_balancer.addresses) != new_addrs:
                logger.info("Load balancer addresses changed, recreating")
                _global_load_balancer = None

        if _global_load_balancer is None:
            _global_load_balancer = VLLMLoadBalancer(addresses, **kwargs)
        return _global_load_balancer


class _HotReloadingLocalOpenAIClientProxy:
    """Resolve latest local_openai endpoints on every eval-script LLM call."""

    provider = "local_openai"
    is_async = True

    def __init__(
        self,
        *,
        profile: str,
        eval_llm_nodes_conf: Optional[str],
        default_addresses: Optional[Union[str, List[str]]],
        default_model: str,
    ) -> None:
        self._profile = profile
        self._eval_llm_nodes_conf = eval_llm_nodes_conf
        self._default_addresses = default_addresses
        self._default_model = str(default_model or "default").strip() or "default"

    def _resolve_runtime_backend(self) -> tuple[Optional[Union[str, List[str]]], str]:
        return _resolve_eval_llm_profile_runtime_config(
            profile=self._profile,
            eval_llm_nodes_conf=self._eval_llm_nodes_conf,
            default_addresses=self._default_addresses,
            default_model=self._default_model,
        )

    def _get_cached_client(self, resolved_addresses: Optional[Union[str, List[str]]]) -> Any:
        # Import lazily to keep module import side effects minimal.
        from obj_task_eval.llm_client.base_client import LLMClient

        thread_id = threading.get_ident()
        try:
            loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            loop_id = 0

        with _shared_local_openai_clients_lock:
            if resolved_addresses:
                setup_llm_client_env(addresses=resolved_addresses)

            base_urls = (os.environ.get("LOCAL_OPENAI_BASE_URLS") or os.environ.get("LOCAL_OPENAI_BASE_URL") or "").strip()
            api_key = (os.environ.get("LOCAL_OPENAI_API_KEY") or "dummy").strip()
            timeout_seconds = (os.environ.get("LOCAL_OPENAI_TIMEOUT_SECONDS") or "").strip()
            max_retries = (os.environ.get("LOCAL_OPENAI_MAX_RETRIES") or "").strip()
            busy_cooldown = (os.environ.get("LOCAL_OPENAI_BUSY_COOLDOWN_SECONDS") or "").strip()
            retry_backoff = (os.environ.get("LOCAL_OPENAI_RETRY_BACKOFF_SECONDS") or "").strip()
            cache_key = (
                thread_id,
                loop_id,
                base_urls,
                api_key,
                timeout_seconds,
                max_retries,
                busy_cooldown,
                retry_backoff,
            )

            client = _shared_local_openai_clients.get(cache_key)
            if client is None:
                client = LLMClient(provider="local_openai", is_async=True)
                _shared_local_openai_clients[cache_key] = client
                logger.info(
                    "Created hot-reload local_openai client for thread=%s loop=%s with %d endpoint chars",
                    thread_id,
                    loop_id,
                    len(base_urls),
                )

            # Only keep one active client per thread/loop to avoid stale endpoint reuse.
            stale_keys = [
                key
                for key in list(_shared_local_openai_clients.keys())
                if len(key) >= 2 and key[0] == thread_id and key[1] == loop_id and key != cache_key
            ]
            for key in stale_keys:
                _shared_local_openai_clients.pop(key, None)

            return client

    @staticmethod
    def _maybe_override_model(kwargs: Dict[str, Any], resolved_model: str, default_model: str) -> None:
        requested_model = str(kwargs.get("model") or "").strip()
        if not requested_model or requested_model == default_model:
            kwargs["model"] = resolved_model

    async def async_response(self, **kwargs):
        resolved_addresses, resolved_model = self._resolve_runtime_backend()
        client = self._get_cached_client(resolved_addresses)
        request_kwargs = dict(kwargs)
        self._maybe_override_model(request_kwargs, resolved_model, self._default_model)
        return await client.async_response(**request_kwargs)


def get_shared_local_openai_client(
    *,
    profile: str = "obj",
    eval_llm_nodes_conf: Optional[str] = None,
    default_addresses: Optional[Union[str, List[str]]] = None,
    default_model: str = "default",
):
    """Get a local_openai client proxy that hot-reloads backend routing on each call."""
    return _HotReloadingLocalOpenAIClientProxy(
        profile=profile,
        eval_llm_nodes_conf=eval_llm_nodes_conf,
        default_addresses=default_addresses,
        default_model=default_model,
    )


# ============== Async LLM API Utilities (vLLM Compatible with Load Balancing) ==============

def _default_visit_tool_schema_dict() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit webpage(s) and return the summary of the content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The URL(s) of the webpage(s) to visit.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "The specific information goal for visiting the webpage.",
                    },
                },
                "required": ["url", "goal"],
            },
        },
    }


def _load_visit_tool_config_from_yaml() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load visit-tool config/schema from deepresearch config/tools.yaml."""
    tools_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config",
        "tools.yaml",
    )
    default_config: Dict[str, Any] = {"type": "native"}
    default_schema = _default_visit_tool_schema_dict()
    if not os.path.exists(tools_config_path):
        return default_config, default_schema

    try:
        from omegaconf import OmegaConf

        tools_config = OmegaConf.load(tools_config_path)
        for tool_item in tools_config.get("tools", []):
            class_name = str(tool_item.get("class_name", ""))
            if not class_name.endswith("visit_tool.DeepResearchVisitTool"):
                continue
            tool_config = OmegaConf.to_container(tool_item.get("config", {}), resolve=True) or {}
            tool_schema = OmegaConf.to_container(tool_item.get("tool_schema", {}), resolve=True) or default_schema
            if not isinstance(tool_config, dict):
                tool_config = default_config
            if not isinstance(tool_schema, dict):
                tool_schema = default_schema
            tool_config.setdefault("type", "native")
            return tool_config, tool_schema
    except Exception as e:
        logger.warning("Failed to load visit tool config from %s: %s", tools_config_path, e)

    return default_config, default_schema


def get_shared_citation_visit_tool() -> Optional[Any]:
    """Get a shared visit tool instance for inline-citation URL visits."""
    global _shared_citation_visit_tool
    with _shared_citation_visit_tool_lock:
        if _shared_citation_visit_tool is not None:
            return _shared_citation_visit_tool

        try:
            try:
                from recipe.deepresearch.tools.visit_tool import DeepResearchVisitTool
            except Exception:
                from .tools.visit_tool import DeepResearchVisitTool
            from verl.tools.schemas import OpenAIFunctionToolSchema

            tool_config, tool_schema_dict = _load_visit_tool_config_from_yaml()
            tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
            _shared_citation_visit_tool = DeepResearchVisitTool(
                config=tool_config,
                tool_schema=tool_schema,
            )
        except Exception as e:
            logger.error("Failed to initialize shared citation visit tool: %s", e)
            _shared_citation_visit_tool = None

        return _shared_citation_visit_tool


async def _visit_url_via_tool_async(url: str, max_retries: int = 3) -> Dict[str, Any]:
    """Visit URL via tools/visit_tool.py using create/execute/release pattern."""
    visit_tool = get_shared_citation_visit_tool()
    if visit_tool is None:
        return {"url": url, "url_content": "scrape failed: visit_tool unavailable"}

    goal = "Extract complete factual evidence from this page for inline citation verification."
    last_error = "unknown error"

    for _ in range(max_retries):
        instance_id = None
        try:
            instance_id, _ = await visit_tool.create()
            tool_response, _, metrics = await visit_tool.execute(
                instance_id=instance_id,
                parameters={"url": [url], "goal": goal},
            )

            response_text = (tool_response.text or "").strip()
            status = str((metrics or {}).get("status", "")).lower() if isinstance(metrics, dict) else ""
            if response_text and status != "error":
                return {"url": url, "url_content": response_text}
            last_error = response_text or "empty visit response"
        except Exception as e:
            last_error = str(e)
        finally:
            if instance_id is not None:
                try:
                    await visit_tool.release(instance_id)
                except Exception:
                    pass

        await asyncio.sleep(1)

    return {"url": url, "url_content": f"scrape failed: {last_error}"}


def _has_eval_llm_backend(
    eval_llm_addresses: Optional[Union[str, List[str]]],
    model_name: str = "default",
    profile: str = "eval",
) -> bool:
    return drb_has_eval_llm_backend(
        eval_llm_addresses=eval_llm_addresses,
        model_name=model_name,
        profile=profile,
    )


async def chat_completions_eval_llm(
    eval_llm_addresses: Optional[Union[str, List[str]]],
    messages: List[Dict[str, str]],
    model: str = "default",
    profile: str = "eval",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout_seconds: int = 120,
    max_retries: Optional[int] = None,
    load_balance_strategy: str = "least_connections",
    **kwargs,
) -> Optional[str]:
    if max_retries is None:
        try:
            max_retries = max(1, int(os.environ.get("LOCAL_OPENAI_MAX_RETRIES", "3")))
        except (TypeError, ValueError):
            max_retries = 3
    return await drb_chat_completions_eval_llm(
        eval_llm_addresses=eval_llm_addresses,
        messages=messages,
        model=model,
        profile=profile,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        load_balance_strategy=load_balance_strategy,
        vllm_chat_fn=chat_completions_vllm,
        **kwargs,
    )


async def chat_completions_eval_llm_hot_reload(
    eval_llm_addresses: Optional[Union[str, List[str]]],
    messages: List[Dict[str, str]],
    model: str = "default",
    profile: str = "eval",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout_seconds: int = 120,
    max_retries: Optional[int] = None,
    load_balance_strategy: str = "least_connections",
    eval_llm_nodes_conf: Optional[str] = None,
    **kwargs,
) -> Optional[str]:
    if max_retries is None:
        try:
            max_retries = max(1, int(os.environ.get("LOCAL_OPENAI_MAX_RETRIES", "3")))
        except (TypeError, ValueError):
            max_retries = 3
    resolved_addresses, resolved_model = _resolve_eval_llm_profile_runtime_config(
        profile=profile,
        eval_llm_nodes_conf=eval_llm_nodes_conf,
        default_addresses=eval_llm_addresses,
        default_model=model,
    )
    return await chat_completions_eval_llm(
        eval_llm_addresses=resolved_addresses,
        messages=messages,
        model=resolved_model,
        profile=profile,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        load_balance_strategy=load_balance_strategy,
        **kwargs,
    )


async def chat_completions_vllm_single(
    address: str,
    messages: List[Dict[str, str]],
    model: str = "default",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout_seconds: int = 120,
    **kwargs
) -> Optional[str]:
    """
    Make async chat completion request to a single vLLM server.

    Args:
        address: vLLM server address (already normalized)
        messages: Chat messages
        model: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        timeout_seconds: Request timeout

    Returns:
        Generated text or None on error
    """
    try:
        # vLLM uses /v1/chat/completions endpoint
        if "/v1" not in address:
            request_url = f"{address}/v1/chat/completions"
        else:
            request_url = f"{address}/chat/completions"

        timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        # vLLM typically doesn't require API key, but support it if provided
        api_key = os.environ.get("VLLM_API_KEY", "") or os.environ.get("REWARD_API_KEY", "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request_body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        request_body.update(kwargs)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url=request_url,
                json=request_body,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"vLLM API error {resp.status} from {address}: {error_text[:500]}")
                    return None

                output = await resp.json()
                return output["choices"][0]["message"]["content"]

    except asyncio.TimeoutError:
        logger.error(f"vLLM request to {address} timed out after {timeout_seconds}s")
        return None
    except Exception as e:
        logger.error(f"Error calling vLLM at {address}: {e}")
        return None


async def chat_completions_vllm(
    address: Union[str, List[str]],
    messages: List[Dict[str, str]],
    model: str = "default",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout_seconds: int = 120,
    max_retries: int = 3,
    load_balance_strategy: str = "least_connections",
    **kwargs
) -> Optional[str]:
    """
    Make async chat completion request to vLLM with load balancing support.

    Supports multiple servers for high availability and load distribution.

    Args:
        address: Single address, comma-separated addresses, or list of addresses
                 Examples:
                 - "http://127.0.0.1:8000"
                 - "http://10.0.0.1:8000,http://10.0.0.2:8000"
                 - ["http://10.0.0.1:8000", "http://10.0.0.2:8000"]
        messages: Chat messages
        model: Model name (vLLM uses the loaded model by default)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for deterministic)
        timeout_seconds: Request timeout per attempt
        max_retries: Maximum retry attempts across servers
        load_balance_strategy: "round_robin" or "least_connections"

    Returns:
        Generated text or None on error
    """
    # Parse addresses
    if isinstance(address, str):
        addresses = [addr.strip() for addr in address.split(",") if addr.strip()]
    else:
        addresses = list(address)

    # Single server: direct call without load balancer overhead
    if len(addresses) == 1:
        addr = addresses[0]
        if not addr.startswith("http"):
            addr = f"http://{addr}"
        return await chat_completions_vllm_single(
            addr, messages, model, max_tokens, temperature, timeout_seconds, **kwargs
        )

    # Multiple servers: use load balancer
    balancer = get_load_balancer(addresses, strategy=load_balance_strategy, max_retries=max_retries)

    tried_servers = set()
    last_error = None

    for attempt in range(max_retries):
        server = balancer.get_next_server()

        # Avoid retrying the same failed server immediately
        if server in tried_servers and len(tried_servers) < len(addresses):
            # Try to get a different server
            for _ in range(len(addresses)):
                server = balancer.get_next_server()
                if server not in tried_servers:
                    break

        tried_servers.add(server)
        balancer.acquire(server)

        try:
            result = await chat_completions_vllm_single(
                server, messages, model, max_tokens, temperature, timeout_seconds, **kwargs
            )

            if result is not None:
                balancer.release(server, success=True)
                return result
            else:
                balancer.release(server, success=False)
                last_error = f"Server {server} returned None"

        except Exception as e:
            balancer.release(server, success=False)
            last_error = str(e)
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed on {server}: {e}")

    logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
    return None


# Alias for backward compatibility
chat_completions_aiohttp = chat_completions_vllm


# ============== Dynamic Eval Script Loading ==============

def load_eval_script(task_id: str, eval_scripts_dir: Optional[str] = None) -> Optional[tuple]:
    """
    Load task-specific evaluation script.

    Looks for: eval_scripts/<task_id>.py
    The script must define: evaluate_answer(...)

    Supports two signatures:
    1. Simple: evaluate_answer(solution_str, ground_truth, task_id, **kwargs) -> Dict
    2. Async with client: async evaluate_answer(client, answer, agent_name, answer_name,
                                                 cache, semaphore, logger, model) -> Dict

    Args:
        task_id: Task identifier
        eval_scripts_dir: Directory containing eval scripts

    Returns:
        Tuple of (evaluate_answer function, is_async, needs_client) if found, None otherwise
    """
    if eval_scripts_dir is None:
        # Default: look in eval_scripts/ relative to this file
        eval_scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_scripts")

    # Sanitize task_id to prevent path traversal (e.g., "../../malicious")
    safe_task_id = os.path.basename(task_id)
    script_path = os.path.join(eval_scripts_dir, f"{safe_task_id}.py")

    # Verify resolved path is within eval_scripts_dir
    real_script = os.path.realpath(script_path)
    real_dir = os.path.realpath(eval_scripts_dir)
    if not real_script.startswith(real_dir + os.sep):
        logger.error(f"Eval script path escapes eval_scripts_dir: {script_path}")
        return None

    if not os.path.exists(script_path):
        return None

    # If this task's eval script has already failed to import/compile in this
    # worker process, short-circuit and let caller mark this sample as dropped.
    if safe_task_id in _EVAL_SCRIPT_LOAD_FAILURES:
        return (None, False, False)

    try:
        # Add deepresearch directory to sys.path so eval scripts can import from it
        deepresearch_dir = os.path.dirname(os.path.abspath(__file__))
        if deepresearch_dir not in sys.path:
            sys.path.insert(0, deepresearch_dir)

        spec = importlib.util.spec_from_file_location(f"eval_script_{safe_task_id}", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "evaluate_answer"):
            eval_func = module.evaluate_answer

            # Check if it's async
            is_async = asyncio.iscoroutinefunction(eval_func)

            # Check if it needs a client (async signature with client parameter)
            sig = inspect.signature(eval_func)
            params = set(sig.parameters.keys())
            needs_client = "client" in params

            _EVAL_SCRIPT_LOAD_FAILURES.pop(safe_task_id, None)
            return (eval_func, is_async, needs_client)
        else:
            logger.warning(f"Warning: {script_path} does not define evaluate_answer function")
            return None

    except Exception as e:
        import traceback
        _EVAL_SCRIPT_LOAD_FAILURES[safe_task_id] = f"{type(e).__name__}: {e}"
        logger.error(f"Error loading eval script {script_path}: {e}")
        print(f"[load_eval_script] ERROR loading {script_path}: {e}\n{traceback.format_exc()}")
        # Return a typed sentinel (eval_func=None) so caller can distinguish
        # compile/import failures from "file not found".
        return (None, False, False)


def setup_llm_client_env(
    ips: Optional[Union[str, List[str]]] = None,
    ports: Optional[Union[str, int, List[int]]] = None,
    addresses: Optional[Union[str, List[str]]] = None,
):
    """
    Set up environment variables for LLMClient to use vLLM servers.

    Args:
        ips: IP addresses (e.g., "10.0.0.1,10.0.0.2")
        ports: Ports (e.g., "8000,8001")
        addresses: Full addresses (e.g., "http://10.0.0.1:8000,http://10.0.0.2:8000")
    """
    # Generate addresses from IPs + ports if provided
    if ips and ports:
        server_addresses = generate_server_addresses(ips, ports)
    elif addresses:
        if isinstance(addresses, str):
            server_addresses = [addr.strip() for addr in addresses.split(",") if addr.strip()]
        else:
            server_addresses = list(addresses)
    else:
        return

    # Ensure addresses have /v1 suffix for OpenAI-compatible API
    formatted_urls = []
    for addr in server_addresses:
        if not addr.startswith("http"):
            addr = f"http://{addr}"
        if not addr.endswith("/v1"):
            addr = f"{addr}/v1"
        formatted_urls.append(addr)

    # Set environment variable for LLMClient
    os.environ["LOCAL_OPENAI_BASE_URLS"] = ",".join(formatted_urls)
    os.environ["LOCAL_OPENAI_API_KEY"] = os.environ.get("VLLM_API_KEY", "dummy")

    logger.info(f"Set LOCAL_OPENAI_BASE_URLS to {len(formatted_urls)} servers")


async def call_async_eval_with_client(
    eval_func: Callable,
    solution_str: str,
    ground_truth: Dict[str, Any],
    task_id: str,
    model: str = "default",
    **kwargs
) -> Dict[str, Any]:
    """
    Call an async eval function that requires an LLM client.

    This function handles the complex async eval scripts that use the
    obj_task_eval Evaluator pattern with client, cache, semaphore, etc.

    Args:
        eval_func: The async evaluate_answer function
        solution_str: The solution string to evaluate
        ground_truth: Ground truth for evaluation
        task_id: Task identifier
        model: Model name for LLM calls
        **kwargs: Additional arguments (may contain IPs, ports, etc.)

    Returns:
        Dict with evaluation results
    """
    try:
        resolved_addresses, resolved_model = _resolve_eval_llm_profile_runtime_config(
            profile="obj",
            eval_llm_nodes_conf=kwargs.get("eval_llm_nodes_conf"),
            default_addresses=kwargs.get("eval_llm_addresses"),
            default_model=model,
        )
        if resolved_addresses:
            setup_llm_client_env(addresses=resolved_addresses)
        model = resolved_model

        # Import from the packaged obj_task_eval library only (no fallback path mutation).
        from obj_task_eval.utils.cache_filesys import CacheFileSys

        # Use a proxy so each underlying LLM call can pick up node/config changes.
        client = get_shared_local_openai_client(
            profile="obj",
            eval_llm_nodes_conf=kwargs.get("eval_llm_nodes_conf"),
            default_addresses=resolved_addresses,
            default_model=model,
        )

        # Create cache (use temp directory if not specified)
        cache_dir = kwargs.get("cache_dir", "/tmp/eval_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache = CacheFileSys(cache_dir)

        # Create semaphore for concurrency control
        max_concurrent = kwargs.get("max_concurrent", 10)
        semaphore = asyncio.Semaphore(max_concurrent)

        answer_name = kwargs.get("answer_name")
        if not answer_name:
            answer_digest = hashlib.sha1(solution_str.encode("utf-8")).hexdigest()[:16]
            answer_name = f"{task_id}_{answer_digest}"

        # Create logger
        eval_logger = logging.getLogger(f"eval_{task_id}")

        # Call the async eval function
        result = await eval_func(
            client=client,
            answer=solution_str,
            agent_name=kwargs.get("agent_name", "deepresearch"),
            answer_name=answer_name,
            cache=cache,
            semaphore=semaphore,
            logger=eval_logger,
            model=model,
        )

        return result

    except Exception as e:
        logger.error(f"Error calling async eval with client for {task_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "score": 0.0,
            "acc": 0.0,
            "error": str(e),
            "error_type": "eval_script_runtime_failed",
            "eval_script_runtime_failed": True,
            "drop_from_training": True,
        }


# ============== Synchronous Compute Score Function ==============

def compute_score_sync(
    solution_str: str,
    ground_truth: Optional[Dict[str, Any]],
    task_id: str = "unknown",
    eval_scripts_dir: Optional[str] = None,
    use_default_scoring: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute the reward score for a DeepResearch response (synchronous version).

    NOTE: This function only handles simple sync eval scripts.
    For async eval scripts that need LLM client, use compute_score() instead.

    Args:
        solution_str: Full model response including tool interactions
        ground_truth: Dict with evaluation criteria
        task_id: Task identifier for loading eval script
        eval_scripts_dir: Directory containing eval scripts
        use_default_scoring: Whether to use default scoring if no eval script
        **kwargs: Additional arguments passed to eval function

    Returns:
        Dict with score and additional metrics
    """
    if ground_truth is None:
        ground_truth = {}

    # Load task-specific eval script
    eval_result = load_eval_script(task_id, eval_scripts_dir)

    if eval_result is not None:
        eval_func, is_async, needs_client = eval_result

        # Eval script exists but failed to import/compile. Mark sample as unusable
        # so trainer can drop it and avoid retry loops on broken external scripts.
        if eval_func is None:
            safe_task_id = os.path.basename(task_id)
            error_msg = _EVAL_SCRIPT_LOAD_FAILURES.get(safe_task_id, "eval_script_load_failed")
            return {
                "score": 0.0,
                "acc": 0.0,
                "error": error_msg,
                "error_type": "eval_script_load_failed",
                "eval_script_load_failed": True,
                "drop_from_training": True,
            }

        # If async or needs client, return special marker for async handling
        if is_async or needs_client:
            return {
                "score": None,
                "_requires_async": True,
                "_eval_func": eval_func,
                "_is_async": is_async,
                "_needs_client": needs_client,
            }

        # Simple sync eval script
        try:
            result = eval_func(solution_str, ground_truth, task_id, **kwargs)
            if isinstance(result, dict):
                return {"score": result.get("final_score", result.get("score", 0.0)), **result}
            elif isinstance(result, (int, float)):
                return {"score": float(result)}
            else:
                logger.warning(f"Eval script returned unexpected type: {type(result)}")
                return {"score": 0.0, "error": "unexpected_return_type"}
        except Exception as e:
            logger.error(f"Error running eval script for {task_id}: {e}")
            return {
                "score": 0.0,
                "acc": 0.0,
                "error": str(e),
                "error_type": "eval_script_runtime_failed",
                "eval_script_runtime_failed": True,
                "drop_from_training": True,
            }

    # No eval script found
    if not use_default_scoring:
        raise ValueError(
            f"No eval script found for task_id '{task_id}' and use_default_scoring=False"
        )

    if "answer" not in ground_truth and "golden_answers" not in ground_truth:
        raise ValueError(
            f"No eval script found for task_id '{task_id}' and ground_truth is missing answers "
            "(expected 'answer' or 'golden_answers')"
        )

    # Default scoring: exact match on extracted answer
    extracted = extract_answer(solution_str)
    if not extracted:
        return {"score": 0.0, "extracted_answer": None, "has_answer": False}

    # Get golden answers
    golden_answers = ground_truth.get("answer", ground_truth.get("golden_answers", []))
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    # Check exact match
    em = em_check(extracted, golden_answers)

    # Check fuzzy match
    fm = fuzzy_match(extracted, golden_answers)

    # Format check
    valid_format, format_msg = is_valid_format(solution_str)

    # Tool usage stats
    tool_counts = count_tool_calls(solution_str)

    return {
        "score": 1.0 if em else (0.5 if fm else 0.0),
        "acc": em,
        "exact_match": em,
        "fuzzy_match": fm,
        "extracted_answer": extracted,
        "has_answer": True,
        "valid_format": valid_format,
        "format_message": format_msg,
        "tool_counts": tool_counts,
    }


# ============== BrowseComp Evaluation ==============

_BROWSECOMP_DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
_BROWSECOMP_GRADER_MODEL_DEFAULT = "gpt-5-mini"
_BROWSECOMP_GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

_HLE_JUDGE_TEMPLATE = """Please determine whether the predicted answer matches the correct answer.

Question: {question}

Correct answer: {ground_truth}

Predicted answer:
{predicted_answer}

Compare the predicted answer and the correct answer carefully and decide whether they are consistent.

If the predicted answer explicitly contains the correct answer, or its core meaning matches the correct answer, respond "true".
If the predicted answer does not match or contradicts the correct answer, respond "false".
For multiple-choice questions, if the predicted answer mentions the correct option (e.g., "A", "B"), treat it as consistent.
For numerical answers, if the predicted answer contains the number, treat it as consistent.
Reply with only "true" or "false". Do not add any other content."""

# Module-level cache for BrowseComp gold QA (question_text -> gold_answer)
_browsecomp_gold_qa_cache: Optional[Dict[str, str]] = None
_browsecomp_gold_qa_lock = threading.Lock()


def _resolve_shared_azure_config(model_name: str) -> Dict[str, Any]:
    raw_model = (model_name or "").strip()
    azure_endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
    azure_api_version = (
        os.environ.get("AZURE_OPENAI_API_VERSION")
        or "2024-08-01-preview"
    ).strip()
    azure_deployment = (
        os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        or (raw_model.split("/", 1)[1] if raw_model.startswith("azure/") else raw_model)
    ).strip()
    return {
        "endpoint": azure_endpoint,
        "api_version": azure_api_version,
        "deployment": azure_deployment,
        "enabled": bool(azure_endpoint and azure_deployment),
    }


def _resolve_openai_or_azure_model_name(model_name: str) -> str:
    raw_model = (model_name or "").strip()
    if raw_model.startswith("azure/"):
        return raw_model.split("/", 1)[1].strip()
    return raw_model


def _is_bedrock_model_name(model_name: str) -> bool:
    return (model_name or "").strip().startswith("bedrock/")


def _create_async_openai_chat_client(
    api_key: str,
    model_name: str,
    base_url: Optional[str] = None,
):
    from openai import AsyncAzureOpenAI, AsyncOpenAI

    azure_cfg = _resolve_shared_azure_config(model_name)
    if azure_cfg["enabled"]:
        return (
            AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_cfg["endpoint"],
                api_version=azure_cfg["api_version"],
            ),
            azure_cfg["deployment"],
        )

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return AsyncOpenAI(**client_kwargs), _resolve_openai_or_azure_model_name(model_name)


def _browsecomp_derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _browsecomp_decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _browsecomp_derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def _load_browsecomp_gold_qa() -> Dict[str, str]:
    """Load and cache BrowseComp gold QA from remote encrypted CSV. Thread-safe."""
    global _browsecomp_gold_qa_cache
    if _browsecomp_gold_qa_cache is not None:
        return _browsecomp_gold_qa_cache
    with _browsecomp_gold_qa_lock:
        if _browsecomp_gold_qa_cache is not None:
            return _browsecomp_gold_qa_cache
        logger.info("Fetching BrowseComp gold QA from %s", _BROWSECOMP_DATASET_URL)
        req = Request(
            _BROWSECOMP_DATASET_URL,
            headers={"User-Agent": "DeepResearch_verl_reward"},
        )
        with urlopen(req) as resp:
            data = resp.read()
        text = data.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        qa: Dict[str, str] = {}
        for row in reader:
            problem = _browsecomp_decrypt(
                row.get("problem", ""), row.get("canary", "")
            ).strip()
            answer = _browsecomp_decrypt(
                row.get("answer", ""), row.get("canary", "")
            ).strip()
            if problem:
                qa[problem] = answer
        _browsecomp_gold_qa_cache = qa
        logger.info("Loaded %d BrowseComp gold QA pairs", len(qa))
        return qa


async def _compute_browsecomp_score(
    extracted_answer: str,
    question: str,
    openai_api_key: str,
    grader_model: str = _BROWSECOMP_GRADER_MODEL_DEFAULT,
) -> Dict[str, Any]:
    """
    Grade a BrowseComp answer by:
    1. Looking up the gold answer from the remote encrypted CSV (cached).
    2. Calling OpenAI/Azure chat completions to judge correctness.
    """
    # Load gold QA in a thread executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    try:
        gold_qa = await loop.run_in_executor(None, _load_browsecomp_gold_qa)
    except Exception as e:
        logger.error("Failed to load BrowseComp gold QA: %s", e)
        return {"score": 0.0, "error": f"browsecomp_gold_load_failed: {e}"}

    gold_answer = gold_qa.get(question.strip())
    if not gold_answer:
        logger.warning("BrowseComp question not found in gold QA cache; question prefix: %s", question[:80])
        return {"score": 0.0, "error": "browsecomp_question_not_in_gold"}

    grader_prompt = _BROWSECOMP_GRADER_TEMPLATE.format(
        question=question,
        correct_answer=gold_answer,
        response=extracted_answer,
    )

    try:
        client, resolved_model = _create_async_openai_chat_client(
            api_key=openai_api_key,
            model_name=grader_model,
        )
        response = await client.chat.completions.create(
            model=resolved_model,
            messages=[{"role": "user", "content": grader_prompt}],
            temperature=1.0,
            max_completion_tokens=2048,
        )
        text = (
            getattr(response.choices[0].message, "content", "") or ""
        ).strip()
    except Exception as e:
        logger.error("BrowseComp grader API call failed: %s", e)
        return {"score": 0.0, "error": f"browsecomp_grader_api_failed: {e}"}

    m = re.search(r"correct:\s*(yes|no)\b", text, flags=re.IGNORECASE)
    if m is None:
        logger.warning("BrowseComp grader parse failure; raw output: %s", text[:200])
        return {"score": 0.0, "error": "browsecomp_grader_parse_failure", "grader_output": text}

    verdict = m.group(1).lower()
    is_correct = verdict == "yes"
    return {
        "score": 1.0 if is_correct else 0.0,
        "acc": is_correct,
        "exact_match": is_correct,
        "grader_verdict": verdict,
        "gold_answer": gold_answer,
        "grader_output": text,
    }


async def _compute_hle_score(
    extracted_answer: str,
    question: str,
    ref_answer: str,
    judge_model: str,
    judge_api_key: Optional[str] = None,
    judge_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Grade an HLE answer using the OpenAI/Azure chat completions API.
    """
    if not ref_answer or not ref_answer.strip():
        return {"score": 0.0, "error": "hle_ref_answer_missing"}
    if not judge_model or not judge_model.strip():
        return {"score": 0.0, "error": "hle_missing_judge_model"}
    if not judge_api_key or not judge_api_key.strip():
        return {"score": 0.0, "error": "hle_missing_api_key"}
    if _is_bedrock_model_name(judge_model):
        return {
            "score": 0.0,
            "error": "hle_raw_openai_requires_non_bedrock_model_name",
            "judge_model": judge_model,
        }

    judgment_prompt = _HLE_JUDGE_TEMPLATE.format(
        question=question,
        ground_truth=ref_answer,
        predicted_answer=extracted_answer,
    )

    try:
        client, resolved_model = _create_async_openai_chat_client(
            api_key=judge_api_key,
            model_name=judge_model,
            base_url=judge_base_url,
        )
        response = await client.chat.completions.create(
            model=resolved_model,
            messages=[{"role": "user", "content": judgment_prompt}],
            temperature=1.0,
            max_completion_tokens=128,
        )
        judgment = (
            getattr(response.choices[0].message, "content", "") or ""
        ).strip()
    except Exception as e:
        logger.error("HLE OpenAI judge call failed: %s", e)
        return {"score": 0.0, "error": f"hle_judge_api_failed: {e}"}

    # Parse true/false
    normalized = judgment.strip().lower()
    if normalized in {"true", "consistent"} or ("true" in normalized and "false" not in normalized):
        is_correct = True
    elif normalized in {"false", "inconsistent"} or ("false" in normalized and "true" not in normalized):
        is_correct = False
    else:
        logger.warning("HLE judge parse failure; raw: %s", judgment[:100])
        return {"score": 0.0, "error": "hle_judge_parse_failure", "judge_output": judgment}

    return {
        "score": 1.0 if is_correct else 0.0,
        "acc": is_correct,
        "exact_match": is_correct,
        "judge_verdict": judgment,
        "ref_answer": ref_answer,
    }


# ============== DRB Evaluation Modules ==============


async def _compute_base_score(
    solution_str: str,
    extracted_answer: str,
    ground_truth: Dict[str, Any],
    task_id: str,
    question: str,
    eval_scripts_dir: Optional[str],
    use_default_scoring: bool,
    eval_llm_addresses: Optional[Union[str, List[str]]],
    eval_llm_model: str,
    openended_eval_llm_addresses: Optional[Union[str, List[str]]],
    openended_eval_llm_model: str,
    eval_llm_nodes_conf: Optional[str] = None,
    eval_source: Optional[str] = None,
    # BrowseComp judge config
    browsecomp_openai_api_key: Optional[str] = None,
    browsecomp_grader_model: str = _BROWSECOMP_GRADER_MODEL_DEFAULT,
    # HLE judge config
    hle_judge_model: Optional[str] = None,
    hle_judge_api_key: Optional[str] = None,
    hle_judge_base_url: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Compute base task score without citation bonus."""
    extra_info = extra_info or {}

    # ---- BrowseComp: fetch gold answer from remote encrypted CSV, judge with OpenAI ----
    if eval_source == "browsecomp":
        api_key = (
            browsecomp_openai_api_key
            or os.environ.get("BROWSECOMP_OPENAI_API_KEY")
            or os.environ.get("API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not api_key:
            logger.error("BrowseComp grading requires BROWSECOMP_OPENAI_API_KEY, API_KEY, or OPENAI_API_KEY")
            return {"score": 0.0, "error": "browsecomp_missing_api_key"}
        return await _compute_browsecomp_score(
            extracted_answer=extracted_answer,
            question=question,
            openai_api_key=api_key,
            grader_model=browsecomp_grader_model,
        )

    # ---- HLE: judge with the OpenAI Responses API using ref_answer from ground_truth / extra_info ----
    if eval_source == "hle":
        ref_answer = (
            ground_truth.get("ref_answer")
            or extra_info.get("ref_answer", "")
        )
        judge_model = (
            hle_judge_model
            or os.environ.get("HLE_JUDGE_MODEL_NAME")
            or os.environ.get("JUDGE_MODEL_NAME", "")
        )
        if not judge_model:
            logger.error("HLE grading requires HLE_JUDGE_MODEL_NAME or JUDGE_MODEL_NAME env var")
            return {"score": 0.0, "error": "hle_missing_judge_model"}
        api_key = (
            hle_judge_api_key
            or os.environ.get("HLE_JUDGE_OPENAI_API_KEY")
            or os.environ.get("JUDGE_OPENAI_API_KEY")
            or os.environ.get("API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        return await _compute_hle_score(
            extracted_answer=extracted_answer,
            question=question,
            ref_answer=ref_answer,
            judge_model=judge_model,
            judge_api_key=api_key,
            judge_base_url=(
                hle_judge_base_url
                or os.environ.get("HLE_JUDGE_OPENAI_BASE_URL")
                or os.environ.get("API_BASE")
                or os.environ.get("OPENAI_BASE_URL")
            ),
        )

    # Open-ended type: criteria-based evaluation (uses criterions + dimension_weight from ground_truth)
    if _is_openended_type(ground_truth.get("type")):
        if not _has_eval_llm_backend(
            openended_eval_llm_addresses,
            model_name=openended_eval_llm_model,
            profile="openended",
        ):
            logger.error("Open-ended evaluation requires an eval LLM backend, but none is configured")
            return {"score": 0.0, "error": "no_eval_llm_for_openended"}
        return await drb_compute_score_openended(
            extracted_answer=extracted_answer,
            ground_truth=ground_truth,
            question=question,
            eval_llm_addresses=openended_eval_llm_addresses,
            eval_llm_model=openended_eval_llm_model,
            llm_chat_fn=lambda **chat_kwargs: chat_completions_eval_llm_hot_reload(
                eval_llm_nodes_conf=eval_llm_nodes_conf,
                **chat_kwargs,
            ),
            profile="openended",
        )

    # Objective/default path
    result = compute_score_sync(
        solution_str=solution_str,
        ground_truth=ground_truth,
        task_id=task_id,
        eval_scripts_dir=eval_scripts_dir,
        use_default_scoring=use_default_scoring,
    )

    # Check if eval script requires async handling with LLM client
    if not result.get("_requires_async"):
        return result

    eval_func = result.pop("_eval_func")
    is_async = result.pop("_is_async")
    needs_client = result.pop("_needs_client")
    result.pop("_requires_async")
    result.pop("score")  # Will be set by async eval

    if needs_client:
        # Call async eval with LLM client
        try:
            async_result = await call_async_eval_with_client(
                eval_func=eval_func,
                solution_str=solution_str,
                ground_truth=ground_truth,
                task_id=task_id,
                model=eval_llm_model,
                eval_llm_addresses=eval_llm_addresses,
                eval_llm_nodes_conf=eval_llm_nodes_conf,
                **kwargs
            )
            # Merge results
            result.update(async_result)
            # Normalize score field
            if "final_score" in result and "score" not in result:
                result["score"] = result["final_score"]
        except Exception as e:
            logger.error(f"Async eval with client failed for {task_id}: {e}")
            result["score"] = 0.0
            result["error"] = str(e)
            result["acc"] = 0.0
            result["error_type"] = "eval_script_runtime_failed"
            result["eval_script_runtime_failed"] = True
            result["drop_from_training"] = True

    elif is_async:
        # Simple async eval without client
        try:
            async_result = await eval_func(solution_str, ground_truth, task_id, **kwargs)
            if isinstance(async_result, dict):
                result.update(async_result)
                if "final_score" in result and "score" not in result:
                    result["score"] = result["final_score"]
            elif isinstance(async_result, (int, float)):
                result["score"] = float(async_result)
        except Exception as e:
            logger.error(f"Async eval failed for {task_id}: {e}")
            result["score"] = 0.0
            result["error"] = str(e)
            result["acc"] = 0.0
            result["error_type"] = "eval_script_runtime_failed"
            result["eval_script_runtime_failed"] = True
            result["drop_from_training"] = True

    return result


# ============== Async Compute Score Function (Main Entry) ==============

async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[Dict[str, Any]],
    extra_info: Optional[Dict[str, Any]] = None,
    eval_scripts_dir: Optional[str] = None,
    use_default_scoring: bool = True,
    # LLM client configuration for eval scripts (optional)
    eval_llm_address: Optional[Union[str, List[str]]] = None,
    eval_llm_ips: Optional[Union[str, List[str]]] = None,
    eval_llm_ports: Optional[Union[str, int, List[int]]] = None,
    eval_llm_model: str = "default",
    eval_llm_nodes_conf: Optional[str] = None,
    # BrowseComp judge config (falls back to BROWSECOMP_OPENAI_API_KEY / API_KEY / OPENAI_API_KEY env vars)
    browsecomp_openai_api_key: Optional[str] = None,
    browsecomp_grader_model: Optional[str] = None,
    # HLE judge config (falls back to HLE_JUDGE_MODEL_NAME / JUDGE_MODEL_NAME env vars)
    hle_judge_model: Optional[str] = None,
    hle_judge_api_key: Optional[str] = None,
    hle_judge_base_url: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute the reward score for a DeepResearch response (async version).

    This is the main entry point for reward computation, following the verl
    async reward function pattern (similar to fapo's compute_score_fapo).

    Supports:
    1. Simple sync eval scripts with (solution_str, ground_truth, task_id) signature
    2. Complex async eval scripts that need LLM client for verification
    3. Multiple vLLM servers with load balancing

    Args:
        data_source: Name of the data source
        solution_str: Full model response including tool interactions
        ground_truth: Dict with evaluation criteria (or string for simple cases)
        extra_info: Additional info from agent loop (contains task_id, question, etc.)
        eval_scripts_dir: Directory containing eval scripts
        use_default_scoring: Whether to use default scoring if no eval script
        # Option 1: Provide full addresses
        eval_llm_address: vLLM server address(es). Supports:
            - Single: "http://127.0.0.1:8000"
            - Multiple (comma-separated): "http://10.0.0.1:8000,http://10.0.0.2:8000"
            - Multiple (list): ["http://10.0.0.1:8000", "http://10.0.0.2:8000"]

        # Option 2: Provide IPs and ports separately (auto-generates all combinations)
        eval_llm_ips: IP addresses (e.g., "10.0.0.1,10.0.0.2,10.0.0.3")
        eval_llm_ports: Ports (e.g., "8000,8001,8002" or [8000, 8001, 8002])
            Example: ips="10.0.0.1,10.0.0.2" + ports="8000,8001" generates:
                http://10.0.0.1:8000, http://10.0.0.1:8001,
                http://10.0.0.2:8000, http://10.0.0.2:8001

        eval_llm_model: Model name for LLM client (default: "default" for vLLM)
    eval_llm_nodes_conf: Path to config file for obj/openended eval backends.
            Supports legacy one-IP-per-line format and profile sections
            like [obj]/[openended] with keys: ips/ports/model/addresses.
            If set, this file is re-read on every call (hot-update).
        **kwargs: Additional arguments

    Returns:
        Dict with score and additional metrics
    """
    extra_info = extra_info or {}

    # Backward compatibility: accept legacy llm_judge_* kwargs
    if eval_llm_address is None:
        eval_llm_address = kwargs.pop("llm_judge_address", None)
    if eval_llm_ips is None:
        eval_llm_ips = kwargs.pop("llm_judge_ips", None)
    if eval_llm_ports is None:
        eval_llm_ports = kwargs.pop("llm_judge_ports", None)
    if eval_llm_model == "default":
        eval_llm_model = kwargs.pop("llm_judge_model", eval_llm_model)
    kwargs.pop("llm_judge_max_retries", None)
    kwargs.pop("llm_judge_strategy", None)
    kwargs.pop("use_llm_judge", None)

    # Inline citation fact-checking controls
    enable_inline_citation_score = _coerce_bool_flag(
        kwargs.pop("enable_inline_citation_score", True),
        default=True,
    )
    inline_citation_max_score = _coerce_float_flag(
        kwargs.pop("inline_citation_max_score", _INLINE_CITATION_MAX_SCORE_DEFAULT),
        _INLINE_CITATION_MAX_SCORE_DEFAULT,
        min_value=0.0,
    )
    inline_citation_min_required = _coerce_int_flag(
        kwargs.pop("inline_citation_min_required", 2),
        2,
        min_value=1,
        max_value=16,
    )
    inline_citation_max_urls = _coerce_int_flag(
        kwargs.pop("inline_citation_max_urls", 3),
        3,
        min_value=1,
        max_value=10,
    )
    inline_citation_max_facts_per_url = _coerce_int_flag(
        kwargs.pop("inline_citation_max_facts_per_url", 3),
        3,
        min_value=1,
        max_value=10,
    )
    parallel_base_and_citation = _coerce_bool_flag(
        kwargs.pop("parallel_base_and_citation", True),
        default=True,
    )
    inline_citation_eval_llm_model = str(
        kwargs.pop(
            "inline_citation_eval_llm_model",
            kwargs.pop("citation_eval_llm_model", os.environ.get("CITATION_EVAL_LLM_MODEL_NAME", eval_llm_model)),
        )
    ).strip() or eval_llm_model
    base_score_weight = _coerce_float_flag(
        kwargs.pop("base_score_weight", _BASE_SCORE_WEIGHT_DEFAULT),
        _BASE_SCORE_WEIGHT_DEFAULT,
        min_value=0.0,
    )
    openended_eval_llm_address = kwargs.pop(
        "openended_eval_llm_address",
        kwargs.pop("openended_eval_llm_addresses", openended_eval_llm_addresses),
    )
    openended_eval_llm_ips = kwargs.pop("openended_eval_llm_ips", None)
    openended_eval_llm_ports = kwargs.pop("openended_eval_llm_ports", eval_llm_ports)
    openended_eval_llm_model = str(
        kwargs.pop(
            "openended_eval_llm_model",
            os.environ.get("OPENENDED_EVAL_LLM_MODEL_NAME", openended_eval_llm_model),
        )
    ).strip() or openended_eval_llm_model

    if ground_truth is None:
        ground_truth = {}

    # Handle ground_truth as string
    if isinstance(ground_truth, str):
        ground_truth = {"answer": ground_truth}

    # For multi-session rollouts, use full_response from extra_info if available
    # This ensures all sessions from the same rollout get the same reward
    # full_response contains the complete response with final answer
    if "full_response" in extra_info and extra_info["full_response"]:
        solution_str = extra_info["full_response"]
        logger.debug(f"Using full_response for reward computation (session_id={extra_info.get('session_id', 'N/A')})")

    # Early exit: extract <answer> and check validity
    extracted_answer = extract_answer(solution_str)
    if not extracted_answer or not extracted_answer.strip():
        return {
            "score": 0.0,
            "extracted_answer": extracted_answer,
            "has_answer": False,
        }

    # Get task_id
    task_id = ground_truth.get("task_id", extra_info.get("task_id", "unknown"))

    # Hot-reload obj/openended profile config from eval_llm_nodes_conf.
    if eval_llm_nodes_conf:
        profile_cfg = load_eval_llm_profiles_from_file(eval_llm_nodes_conf)
        obj_cfg = profile_cfg.get("obj", {})
        openended_cfg = profile_cfg.get("openended", {})

        if obj_cfg.get("addresses"):
            eval_llm_address = obj_cfg["addresses"]
        if obj_cfg.get("ips"):
            eval_llm_ips = obj_cfg["ips"]
        if obj_cfg.get("ports"):
            eval_llm_ports = obj_cfg["ports"]
        if obj_cfg.get("model"):
            eval_llm_model = str(obj_cfg["model"]).strip() or eval_llm_model

        if openended_cfg.get("addresses"):
            openended_eval_llm_address = openended_cfg["addresses"]
        if openended_cfg.get("ips"):
            openended_eval_llm_ips = openended_cfg["ips"]
        if openended_cfg.get("ports"):
            openended_eval_llm_ports = openended_cfg["ports"]
        if openended_cfg.get("model"):
            openended_eval_llm_model = str(openended_cfg["model"]).strip() or openended_eval_llm_model

        logger.debug(
            "Loaded eval_llm_nodes_conf=%s obj_cfg=%s openended_cfg=%s",
            eval_llm_nodes_conf,
            obj_cfg,
            openended_cfg,
        )

    # Generate objective addresses from IPs + Ports if provided.
    eval_llm_addresses = eval_llm_address
    if isinstance(eval_llm_addresses, str) and not eval_llm_addresses.strip():
        eval_llm_addresses = None
    if eval_llm_ips and eval_llm_ports:
        eval_llm_addresses = generate_server_addresses(eval_llm_ips, eval_llm_ports)
        logger.info(f"Generated {len(eval_llm_addresses)} server addresses from IPs and ports")

    # Generate open-ended eval addresses independently (falls back to objective addresses).
    openended_eval_llm_addresses = openended_eval_llm_address
    if isinstance(openended_eval_llm_addresses, str) and not openended_eval_llm_addresses.strip():
        openended_eval_llm_addresses = None
    if openended_eval_llm_ips and openended_eval_llm_ports:
        openended_eval_llm_addresses = generate_server_addresses(openended_eval_llm_ips, openended_eval_llm_ports)
        logger.info(f"Generated {len(openended_eval_llm_addresses)} open-ended eval addresses from IPs and ports")
    if openended_eval_llm_addresses is None:
        openended_eval_llm_addresses = eval_llm_addresses

    # Set up environment for LLM client (used by async eval scripts)
    if eval_llm_addresses or (eval_llm_ips and eval_llm_ports):
        setup_llm_client_env(
            ips=eval_llm_ips,
            ports=eval_llm_ports,
            addresses=eval_llm_addresses,
        )

    # Resolve eval_source for routing (browsecomp / hle / m2w2 / etc.)
    eval_source = extra_info.get("eval_source") or data_source.replace("eval_", "")

    # Base and citation scoring can run concurrently to reduce end-to-end latency.
    base_task = asyncio.create_task(
        _compute_base_score(
            solution_str=solution_str,
            extracted_answer=extracted_answer,
            ground_truth=ground_truth,
            task_id=task_id,
            question=extra_info.get("question", ""),
            eval_scripts_dir=eval_scripts_dir,
            use_default_scoring=use_default_scoring,
            eval_llm_addresses=eval_llm_addresses,
            eval_llm_model=eval_llm_model,
            openended_eval_llm_addresses=openended_eval_llm_addresses,
            openended_eval_llm_model=openended_eval_llm_model,
            eval_llm_nodes_conf=eval_llm_nodes_conf,
            eval_source=eval_source,
            browsecomp_openai_api_key=browsecomp_openai_api_key,
            browsecomp_grader_model=browsecomp_grader_model or _BROWSECOMP_GRADER_MODEL_DEFAULT,
            hle_judge_model=hle_judge_model,
            hle_judge_api_key=hle_judge_api_key,
            hle_judge_base_url=hle_judge_base_url,
            extra_info=extra_info,
            **kwargs,
        )
    )
    # BrowseComp and HLE are exact-answer tasks; skip citation scoring for them.
    _skip_citation_for_source = eval_source in {"browsecomp", "hle"}

    citation_task: Optional[asyncio.Task] = None
    if enable_inline_citation_score and parallel_base_and_citation and not _skip_citation_for_source:
        citation_task = asyncio.create_task(
            drb_compute_inline_citation_score(
                solution_str=solution_str,
                extracted_answer=extracted_answer,
                eval_llm_addresses=eval_llm_addresses,
                eval_llm_model=inline_citation_eval_llm_model,
                llm_chat_fn=chat_completions_eval_llm,
                visit_url_fn=_visit_url_via_tool_async,
                has_backend_fn=_has_eval_llm_backend,
                profile="citation",
                max_score=inline_citation_max_score,
                min_required_citations=inline_citation_min_required,
                max_urls=inline_citation_max_urls,
                max_facts_per_url=inline_citation_max_facts_per_url,
            )
        )

    result = await base_task

    base_score = _coerce_float_flag(result.get("score"), 0.0)
    result["base_score"] = base_score
    result["score_without_citation"] = base_score

    if result.get("drop_from_training"):
        citation_result = {
            "citation_score": 0.0,
            "citation_score_max": inline_citation_max_score,
            "citation_score_applied": False,
            "citation_eval_status": "skipped_drop_from_training",
        }
        if citation_task and not citation_task.done():
            citation_task.cancel()
            try:
                await citation_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
    elif enable_inline_citation_score and not _skip_citation_for_source:
        if citation_task is not None:
            try:
                citation_result = await citation_task
            except Exception as e:
                logger.error(f"Inline citation scoring failed for {task_id}: {e}")
                citation_result = {
                    "citation_score": 0.0,
                    "citation_score_max": inline_citation_max_score,
                    "citation_score_applied": False,
                    "citation_eval_status": "error",
                    "citation_error": str(e),
                }
        else:
            citation_result = await drb_compute_inline_citation_score(
                solution_str=solution_str,
                extracted_answer=extracted_answer,
                eval_llm_addresses=eval_llm_addresses,
                eval_llm_model=inline_citation_eval_llm_model,
                llm_chat_fn=chat_completions_eval_llm,
                visit_url_fn=_visit_url_via_tool_async,
                has_backend_fn=_has_eval_llm_backend,
                profile="citation",
                max_score=inline_citation_max_score,
                min_required_citations=inline_citation_min_required,
                max_urls=inline_citation_max_urls,
                max_facts_per_url=inline_citation_max_facts_per_url,
            )
    else:
        citation_eval_status = (
            "skipped_exact_answer_task" if _skip_citation_for_source else "disabled"
        )
        citation_result = {
            "citation_score": 0.0,
            "citation_score_max": inline_citation_max_score,
            "citation_score_applied": False,
            "citation_eval_status": citation_eval_status,
        }

    result.update(citation_result)
    citation_score = _coerce_float_flag(result.get("citation_score"), 0.0)
    citation_applied = _coerce_bool_flag(result.get("citation_score_applied"), default=False)
    citation_base_scale = min(max(base_score, 0.0), 1.0)
    result["citation_score_raw"] = citation_score
    result["citation_score_base_scale"] = citation_base_scale
    result["citation_score_added"] = (citation_score * citation_base_scale) if citation_applied else 0.0
    result["score"] = (base_score * base_score_weight) + result["citation_score_added"]
    result["score_with_citation"] = result["score"]
    return result


# ============== Async Wrapper (for async contexts) ==============

async def compute_score_wrapper(
    data_source: str,
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Async wrapper for compute_score.
    Use this inside an existing event loop.
    """
    return await compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


# ============== Synchronous Wrapper (for non-async contexts) ==============

def compute_score_wrapper_sync(
    data_source: str,
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Synchronous wrapper for compute_score.
    Use this when you need to call from non-async context.
    """
    # Always use asyncio.run() in a new thread to avoid event loop conflicts.
    # This ensures a clean event loop regardless of the calling context
    # (Ray workers, Jupyter, nested async, etc.).
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            asyncio.run,
            compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
        )
        return future.result()
