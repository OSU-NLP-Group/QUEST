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
Python execution tool for DeepResearch.
Runs code in sandbox_fusion. This tool is intentionally stateless and does not use cache.
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

try:
    import json5
except ImportError:
    json5 = None

try:
    from requests.exceptions import Timeout as RequestsTimeout
except ImportError:
    RequestsTimeout = Exception

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _split_conf_list_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_split_conf_list_values(item))
        return out
    return [item.strip() for item in re.split(r"[,\s]+", str(value)) if item.strip()]


def _dedup_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _normalize_sandbox_endpoint(endpoint: str) -> str:
    endpoint = (endpoint or "").strip()
    if not endpoint:
        return ""

    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"

    parsed = urlsplit(endpoint)
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc
    if not netloc and parsed.path:
        # Defensive fallback for malformed inputs.
        path_parts = parsed.path.split("/", 1)
        netloc = path_parts[0]
        path = f"/{path_parts[1]}" if len(path_parts) > 1 else ""
    else:
        path = parsed.path

    path = (path or "").rstrip("/")
    if path == "/run_code":
        path = ""

    return urlunsplit((scheme, netloc, path, parsed.query, parsed.fragment))


def _generate_sandbox_endpoints(hosts: list[str], ports: list[str]) -> list[str]:
    endpoints = []
    for host in hosts:
        host = str(host).strip()
        if not host:
            continue

        host_for_parse = host if "://" in host else f"http://{host}"
        parsed = urlsplit(host_for_parse)
        hostname = parsed.hostname
        if not hostname:
            continue
        scheme = parsed.scheme or "http"

        for port in ports:
            port = str(port).strip()
            if not port:
                continue
            endpoints.append(f"{scheme}://{hostname}:{port}")

    return _dedup_keep_order([_normalize_sandbox_endpoint(endpoint) for endpoint in endpoints if endpoint])


def _load_sandbox_nodes_conf(conf_path: str) -> dict[str, list[str]]:
    parsed = {"hosts": [], "ports": [], "endpoints": []}
    if not conf_path:
        return parsed

    try:
        with open(conf_path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue

                if "=" not in line:
                    if "://" in line or "/" in line or re.search(r":\d+$", line):
                        parsed["endpoints"].extend(_split_conf_list_values(line))
                    else:
                        parsed["hosts"].extend(_split_conf_list_values(line))
                    continue

                raw_key, raw_value = line.split("=", 1)
                key = raw_key.strip().lower()
                value = raw_value.strip()
                if key in {"ip", "ips", "node", "nodes", "host", "hosts"}:
                    parsed["hosts"].extend(_split_conf_list_values(value))
                elif key in {"port", "ports"}:
                    parsed["ports"].extend(_split_conf_list_values(value))
                elif key in {
                    "endpoint",
                    "endpoints",
                    "url",
                    "urls",
                    "address",
                    "addresses",
                    "base_url",
                    "base_urls",
                }:
                    parsed["endpoints"].extend(_split_conf_list_values(value))
    except Exception as e:
        logger.warning("[python] Failed to read python_nodes_conf %s: %s", conf_path, e)
        return {"hosts": [], "ports": [], "endpoints": []}

    parsed["hosts"] = _dedup_keep_order(parsed["hosts"])
    parsed["ports"] = _dedup_keep_order(parsed["ports"])
    parsed["endpoints"] = _dedup_keep_order(
        [_normalize_sandbox_endpoint(endpoint) for endpoint in parsed["endpoints"] if endpoint]
    )
    return parsed


def _load_static_sandbox_endpoints(config: dict[str, Any]) -> list[str]:
    raw_endpoints = []
    for value in (
        os.environ.get("SANDBOX_FUSION_ENDPOINTS"),
        config.get("sandbox_fusion_endpoints"),
        os.environ.get("SANDBOX_FUSION_ENDPOINT"),
        config.get("sandbox_fusion_endpoint"),
        os.environ.get("PYTHON_SERVICE_URLS"),
        config.get("python_service_urls"),
        os.environ.get("PYTHON_SERVICE_URL"),
        config.get("python_service_url"),
    ):
        raw_endpoints.extend(_split_conf_list_values(value))

    return _dedup_keep_order(
        [_normalize_sandbox_endpoint(endpoint) for endpoint in raw_endpoints if endpoint]
    )


def _extract_code_from_text(text: str) -> str:
    if not text:
        return ""
    code = text
    xml_match = re.search(r"<code>(.*?)</code>", code, re.DOTALL)
    if xml_match:
        code = xml_match.group(1)
    triple_match = re.search(r"```[^\n]*\n(.*?)```", code, re.DOTALL)
    if triple_match:
        code = triple_match.group(1)
    return code.strip()


def _normalize_code_param(params: Any) -> str:
    if isinstance(params, dict):
        return _extract_code_from_text(str(params.get("code", "") or params.get("raw", "")))
    if isinstance(params, str):
        raw = params
        try:
            parsed = json.loads(params)
            if isinstance(parsed, dict):
                raw = str(parsed.get("code", "") or parsed.get("raw", "") or raw)
        except Exception:
            if json5 is not None:
                try:
                    parsed = json5.loads(params)
                    if isinstance(parsed, dict):
                        raw = str(parsed.get("code", "") or parsed.get("raw", "") or raw)
                except Exception:
                    pass
        return _extract_code_from_text(raw)
    return _extract_code_from_text(str(params))


class DeepResearchPythonTool(BaseTool):
    """Python execution tool backed by sandbox_fusion."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.timeout = int(config.get("timeout", 50))
        self.max_retries = int(config.get("max_retries", 5))
        self._python_nodes_conf = str(os.environ.get("PYTHON_NODES_CONF") or config.get("python_nodes_conf") or "").strip()
        self._static_sandbox_endpoints = _load_static_sandbox_endpoints(config)
        if self._python_nodes_conf or self._static_sandbox_endpoints:
            logger.info(
                "[python] sandbox backend configured (nodes_conf=%s, static_endpoints=%s)",
                self._python_nodes_conf or "(none)",
                ",".join(self._static_sandbox_endpoints) or "(none)",
            )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()

    def _get_sandbox_endpoints(self) -> list[str]:
        if self._python_nodes_conf:
            conf = _load_sandbox_nodes_conf(self._python_nodes_conf)
            if conf["endpoints"]:
                return conf["endpoints"]
            if conf["hosts"] and conf["ports"]:
                return _generate_sandbox_endpoints(conf["hosts"], conf["ports"])
        return self._static_sandbox_endpoints

    def _run_python(self, parameters: Any) -> tuple[str, dict]:
        code = _normalize_code_param(parameters)
        if not code:
            return "[Python Interpreter Error]: Empty code.", {"error": "empty_code"}

        endpoints = self._get_sandbox_endpoints()
        if not endpoints:
            return (
                "[Python Interpreter Error]: sandbox endpoint is not configured. Set SANDBOX_FUSION_ENDPOINT or python_nodes_conf.",
                {"error": "missing_sandbox_endpoint"},
            )

        try:
            from sandbox_fusion import RunCodeRequest, run_code
        except ImportError as e:
            return (
                f"[Python Interpreter Error]: sandbox_fusion is not installed: {e}",
                {"error": "missing_sandbox_fusion"},
            )

        last_error = None
        attempts = max(1, self.max_retries)
        retry_endpoints = list(endpoints)
        if len(retry_endpoints) > 1:
            random.shuffle(retry_endpoints)
        for attempt in range(attempts):
            endpoint = retry_endpoints[attempt % len(retry_endpoints)]
            if attempt >= len(retry_endpoints) and len(retry_endpoints) > 1 and attempt % len(retry_endpoints) == 0:
                random.shuffle(retry_endpoints)
            try:
                started_at = time.time()
                code_result = run_code(
                    RunCodeRequest(code=code, language="python", run_timeout=self.timeout),
                    max_attempts=1,
                    client_timeout=self.timeout,
                    endpoint=endpoint,
                )
                elapsed = time.time() - started_at
                result_parts = []
                stdout = getattr(code_result.run_result, "stdout", "") or ""
                stderr = getattr(code_result.run_result, "stderr", "") or ""
                execution_time = getattr(code_result.run_result, "execution_time", 0.0) or 0.0
                if stdout:
                    result_parts.append(f"stdout:\n{stdout}")
                if stderr:
                    result_parts.append(f"stderr:\n{stderr}")
                if execution_time >= self.timeout - 1:
                    result_parts.append("[Python Interpreter Error] TimeoutError: Execution timed out.")
                text = "\n".join(result_parts).strip() or "Finished execution."
                return text, {
                    "endpoint": endpoint,
                    "execution_time": round(elapsed, 6),
                    "status": "success",
                }
            except RequestsTimeout:
                last_error = f"[Python Interpreter Error] TimeoutError: Execution timed out on endpoint {endpoint}."
            except Exception as e:
                last_error = f"[Python Interpreter Error]: {e} on endpoint {endpoint}"
            logger.warning("[python] attempt %s/%s failed: %s", attempt + 1, attempts, last_error)

        return last_error or "[Python Interpreter Error]: All attempts failed.", {"error": "execution_failed"}

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        loop = asyncio.get_running_loop()
        text, metrics = await loop.run_in_executor(None, self._run_python, parameters)
        return ToolResponse(text=text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
