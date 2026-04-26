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
Visit tool for DeepResearch using Jina API with SQLite cache.
Supports optional sharded cache.
"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
import atexit
import hashlib
from typing import Any, Optional, Tuple
from uuid import uuid4

import requests
from openai import AzureOpenAI, OpenAI

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_RECIPE_DIR = os.path.dirname(_TOOL_DIR)
_DEFAULT_CACHE_DIR = os.path.join(_RECIPE_DIR, "database")


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y"}


def _get_leader_flag(config: dict) -> bool:
    if config.get("cache_shard_leader") is not None:
        return bool(config.get("cache_shard_leader"))
    if os.getenv("VISIT_CACHE_SHARD_LEADER") is not None:
        return _bool_env("VISIT_CACHE_SHARD_LEADER", False)
    # Auto-detect common rank envs: only rank 0 is leader
    for env in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if os.getenv(env) is not None:
            try:
                return int(os.getenv(env, "-1")) == 0
            except ValueError:
                return False
    return False


def _is_visit_error_result(result: str) -> bool:
    if not result:
        return True
    lowered = result.strip().lower()
    error_prefixes = (
        "[visit error]",
        "[visit] error",
        "[visit] invalid",
    )
    if lowered.startswith(error_prefixes):
        return True
    if "failed to read page" in lowered:
        return True
    if "could not be accessed" in lowered:
        return True
    return False


def _is_summarization_failure_result(result: str) -> bool:
    if not result:
        return True
    lowered = result.lower()
    failure_markers = (
        "summarization failed:",
        "failed to summarize content",
    )
    return any(marker in lowered for marker in failure_markers)


EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""


class VisitCache:
    """SQLite cache storing URL content and (url, goal) summaries. Supports sharded DB files."""

    def __init__(
        self,
        cache_dir: str,
        cache_file: str,
        shards: int = 1,
        resume: bool = True,
        auto_shard: bool = False,
        auto_merge: bool = False,
        is_leader: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.master_file = cache_file
        self.shards = max(1, int(shards))
        self.resume = resume
        self.auto_shard = auto_shard
        self.auto_merge = auto_merge
        self.is_leader = is_leader

        base, _ = os.path.splitext(os.path.basename(self.master_file))
        self._shard_files = [
            os.path.join(self.cache_dir, f"{base}_shard{idx}.db") for idx in range(self.shards)
        ]

        self._conns: dict[int, sqlite3.Connection] = {}
        self._locks: dict[int, threading.Lock] = {}

        if self.shards == 1:
            self._get_conn(0)

        if self.shards > 1 and self.auto_shard and self.is_leader:
            self._split_master()

        if self.shards > 1 and self.auto_merge and self.is_leader:
            atexit.register(self.merge_shards)

        atexit.register(self.close)

    def close(self) -> None:
        for conn in self._conns.values():
            try:
                conn.close()
            except Exception:
                pass

    def _open_conn(self, path: str) -> sqlite3.Connection:
        db_exists = os.path.exists(path)
        conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-8000")
        conn.commit()

        if self.resume and db_exists:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row["name"] for row in cursor.fetchall()}
            has_new = "url_content" in tables and "url_goal_info" in tables
            has_old = "visit_cache" in tables
            if has_new:
                self._ensure_tables(conn)
            elif has_old:
                self._ensure_tables(conn)
                self._migrate_from_old_structure(conn)
            else:
                self._ensure_tables(conn)
        else:
            self._ensure_tables(conn)
        return conn

    def _ensure_tables(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS url_content (
                url TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS url_goal_info (
                url TEXT NOT NULL,
                goal TEXT NOT NULL,
                useful_information TEXT NOT NULL,
                timestamp REAL NOT NULL,
                PRIMARY KEY (url, goal)
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_url_goal_info_url
            ON url_goal_info(url)
            """
        )
        conn.commit()

    def _migrate_from_old_structure(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='visit_cache'"
        )
        if cursor.fetchone() is None:
            return
        cursor.execute(
            "SELECT url, goal, content, useful_information, timestamp FROM visit_cache ORDER BY timestamp DESC"
        )
        old_records = cursor.fetchall()
        migrated_urls = set()
        for row in old_records:
            url = row["url"]
            goal = row["goal"]
            content = row["content"]
            useful_information = row["useful_information"]
            timestamp = row["timestamp"]

            if url not in migrated_urls and content:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO url_content (url, content, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (url, content, timestamp),
                )
                migrated_urls.add(url)
            if useful_information:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO url_goal_info (url, goal, useful_information, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    (url, goal, useful_information, timestamp),
                )
        conn.commit()

    def _get_conn(self, shard_id: int) -> sqlite3.Connection:
        if shard_id not in self._conns:
            if shard_id == 0 and self.shards == 1:
                path = self.master_file
            else:
                path = self._shard_files[shard_id]
            self._conns[shard_id] = self._open_conn(path)
            self._locks[shard_id] = threading.Lock()
        return self._conns[shard_id]

    def _shard_id(self, url: str) -> int:
        if self.shards == 1:
            return 0
        digest = hashlib.md5(url.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.shards

    def _split_master(self) -> None:
        if not os.path.exists(self.master_file):
            return

        if not self.resume:
            for shard_file in self._shard_files:
                if os.path.exists(shard_file):
                    os.remove(shard_file)

        master_conn = self._open_conn(self.master_file)
        master_cursor = master_conn.cursor()

        master_cursor.execute("SELECT url, content, timestamp FROM url_content")
        content_rows = master_cursor.fetchall()
        rows_by_shard: dict[int, list[tuple[str, str, float]]] = {}
        for row in content_rows:
            shard_id = self._shard_id(row["url"])
            rows_by_shard.setdefault(shard_id, []).append(
                (row["url"], row["content"], row["timestamp"])
            )
        for shard_id, rows in rows_by_shard.items():
            conn = self._get_conn(shard_id)
            with self._locks[shard_id]:
                cursor = conn.cursor()
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO url_content (url, content, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()

        master_cursor.execute(
            "SELECT url, goal, useful_information, timestamp FROM url_goal_info"
        )
        goal_rows = master_cursor.fetchall()
        rows_by_shard = {}
        for row in goal_rows:
            shard_id = self._shard_id(row["url"])
            rows_by_shard.setdefault(shard_id, []).append(
                (row["url"], row["goal"], row["useful_information"], row["timestamp"])
            )
        for shard_id, rows in rows_by_shard.items():
            conn = self._get_conn(shard_id)
            with self._locks[shard_id]:
                cursor = conn.cursor()
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO url_goal_info (url, goal, useful_information, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()

    def merge_shards(self) -> None:
        if self.shards <= 1:
            return
        master_conn = self._open_conn(self.master_file)
        master_cursor = master_conn.cursor()

        for shard_id, shard_file in enumerate(self._shard_files):
            if not os.path.exists(shard_file):
                continue
            conn = self._open_conn(shard_file)
            cursor = conn.cursor()

            cursor.execute("SELECT url, content, timestamp FROM url_content")
            content_rows = cursor.fetchall()
            if content_rows:
                master_cursor.executemany(
                    """
                    INSERT OR REPLACE INTO url_content (url, content, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    [(row["url"], row["content"], row["timestamp"]) for row in content_rows],
                )
                master_conn.commit()

            cursor.execute(
                "SELECT url, goal, useful_information, timestamp FROM url_goal_info"
            )
            goal_rows = cursor.fetchall()
            if goal_rows:
                master_cursor.executemany(
                    """
                    INSERT OR REPLACE INTO url_goal_info (url, goal, useful_information, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (row["url"], row["goal"], row["useful_information"], row["timestamp"])
                        for row in goal_rows
                    ],
                )
                master_conn.commit()

    def get_content_by_url(self, url: str) -> Optional[str]:
        shard_id = self._shard_id(url)
        conn = self._get_conn(shard_id)
        with self._locks[shard_id]:
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM url_content WHERE url = ?", (url,))
            row = cursor.fetchone()
            return row["content"] if row else None

    def get_useful_information(self, url: str, goal: str) -> Optional[str]:
        shard_id = self._shard_id(url)
        conn = self._get_conn(shard_id)
        with self._locks[shard_id]:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT useful_information FROM url_goal_info WHERE url = ? AND goal = ?",
                (url, goal),
            )
            row = cursor.fetchone()
            return row["useful_information"] if row else None

    def set(self, url: str, goal: str, content: str, useful_information: str) -> None:
        shard_id = self._shard_id(url)
        conn = self._get_conn(shard_id)
        current_time = time.time()
        with self._locks[shard_id]:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO url_content (url, content, timestamp)
                VALUES (?, ?, ?)
                """,
                (url, content, current_time),
            )
            cursor.execute(
                """
                INSERT OR REPLACE INTO url_goal_info (url, goal, useful_information, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (url, goal, useful_information, current_time),
            )
            conn.commit()


def truncate_content(text: str, max_chars: int = 100000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Content truncated due to length...]"


def _format_useful_information(url: str, goal: str, evidence: str, summary: str) -> str:
    return (
        f"The useful information in {url} for user goal {goal} as follows: \n\n"
        f"Evidence in page: \n{evidence}\n\n"
        f"Summary: \n{summary}\n\n"
    )


def jina_fetch_sync(url: str, jina_api_key: str, timeout: int = 50) -> str:
    if not jina_api_key:
        return "[visit] Error: JINA_API_KEY environment variable not set."

    headers = {"Authorization": f"Bearer {jina_api_key}"}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.text
            error_info = response.text
            try:
                error_json = json.loads(response.text)
                error_message = error_json.get("readableMessage") or error_json.get("message", response.text)
                error_info = f"[visit] Error (HTTP {response.status_code}): {error_message}"
            except Exception:
                error_info = f"[visit] Error (HTTP {response.status_code}): {response.text[:500]}"

            if attempt == max_retries - 1:
                return error_info
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                return f"[visit] Error: Request timed out after {timeout}s"
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return f"[visit] Error: Network error: {str(e)}"
        except Exception as e:
            if attempt == max_retries - 1:
                return f"[visit] Error: Unexpected error: {str(e)}"

    return "[visit] Failed to read page after all retries."


def summarize_content(
    content: str,
    url: str,
    goal: str,
    api_key: str = "",
    api_base: str = "",
    model_name: str = "gpt-4o-mini",
    azure_endpoint: str = "",
    azure_api_version: str = "2024-08-01-preview",
    fallback_api_key: str = "",
    fallback_api_base: str = "",
    fallback_model_name: str = "",
    fallback_azure_endpoint: str = "",
    fallback_azure_api_version: str = "2024-08-01-preview",
    timeout_seconds: float = 300,
    max_content_length: int = 100000,
) -> str:
    if content.startswith("[visit]"):
        return _format_useful_information(
            url,
            goal,
            content,
            "The webpage could not be accessed. Please check the URL or file format.",
        )

    candidate_clients = []
    if azure_endpoint:
        candidate_clients.append(
            (
                "visit-summary primary",
                AzureOpenAI(
                    api_key=api_key,
                    api_version=azure_api_version,
                    azure_endpoint=azure_endpoint,
                    timeout=timeout_seconds,
                ),
                model_name,
            )
        )
    elif api_key:
        client_kwargs = {"api_key": api_key, "timeout": timeout_seconds}
        if api_base:
            client_kwargs["base_url"] = api_base
        candidate_clients.append(("visit-summary primary", OpenAI(**client_kwargs), model_name))

    if fallback_azure_endpoint:
        candidate_clients.append(
            (
                "visit-summary fallback",
                AzureOpenAI(
                    api_key=fallback_api_key,
                    api_version=fallback_azure_api_version,
                    azure_endpoint=fallback_azure_endpoint,
                    timeout=timeout_seconds,
                ),
                fallback_model_name,
            )
        )
    elif fallback_api_key:
        fallback_client_kwargs = {"api_key": fallback_api_key, "timeout": timeout_seconds}
        if fallback_api_base:
            fallback_client_kwargs["base_url"] = fallback_api_base
        candidate_clients.append(("visit-summary fallback", OpenAI(**fallback_client_kwargs), fallback_model_name))

    if not candidate_clients:
        content = truncate_content(content, 5000)
        return _format_useful_information(
            url,
            goal,
            content,
            "The webpage content could not be processed, and therefore, no information is available.",
        )

    content = truncate_content(content, max_content_length)

    messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]

    max_retries = 3
    last_error = None
    for client_label, client, current_model_name in candidate_clients:
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=current_model_name,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=16384,
                )
                raw_result = response.choices[0].message.content
                if not raw_result:
                    continue
                raw_result = raw_result.replace("```json", "").replace("```", "").strip()
                try:
                    parsed = json.loads(raw_result)
                except json.JSONDecodeError:
                    left = raw_result.find("{")
                    right = raw_result.rfind("}")
                    if left != -1 and right != -1 and left < right:
                        parsed = json.loads(raw_result[left : right + 1])
                    else:
                        return _format_useful_information(url, goal, raw_result, "")

                evidence = parsed.get("evidence", "N/A")
                summary = parsed.get("summary", "N/A")
                return _format_useful_information(url, goal, evidence, summary)
            except Exception as e:
                last_error = e
                logger.warning(
                    "[visit] %s attempt %s/%s failed for %s: %s",
                    client_label,
                    attempt + 1,
                    max_retries,
                    url,
                    e,
                )
        logger.warning("[visit] %s exhausted; trying next summarizer path for %s", client_label, url)

    content = truncate_content(content, 3000)
    return _format_useful_information(
        url,
        goal,
        content,
        f"Summarization failed: {str(last_error)}" if last_error else "Failed to summarize content",
    )


class DeepResearchVisitTool(BaseTool):
    """Visit tool with optional SQLite cache."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        def _env_override(name: str, fallback: str = "") -> str:
            # Respect explicit env overrides even when they are set to the empty string.
            return os.environ[name] if name in os.environ else fallback

        self.jina_api_key = (
            config.get("jina_api_key")
            or os.environ.get("JINA_API_KEY", "")
            or (os.environ.get("JINA_API_KEYS", "").split(",")[0].strip() if os.environ.get("JINA_API_KEYS") else "")
        )
        self.timeout = config.get("timeout", 50)

        # Prefer visit-summary-specific env vars over static tool config so the launcher
        # can switch providers without being overridden by tools.yaml defaults.
        self.api_key = _env_override("VISIT_SUMMARY_API_KEY", config.get("api_key") or "")
        self.api_base = _env_override("VISIT_SUMMARY_API_BASE", config.get("api_base") or "")
        self.model_name = (
            _env_override(
                "VISIT_SUMMARY_MODEL_NAME",
                config.get("summary_model_name") or "",
            )
        )
        self.azure_endpoint = _env_override("VISIT_SUMMARY_AZURE_ENDPOINT", config.get("azure_endpoint") or "")
        self.azure_api_version = (
            _env_override("VISIT_SUMMARY_AZURE_API_VERSION", config.get("azure_api_version") or "2024-08-01-preview")
        )
        self.fallback_api_key = _env_override(
            "VISIT_SUMMARY_FALLBACK_API_KEY",
            config.get("fallback_api_key") or "",
        )
        self.fallback_api_base = _env_override(
            "VISIT_SUMMARY_FALLBACK_API_BASE",
            config.get("fallback_api_base") or "",
        )
        self.fallback_model_name = _env_override(
            "VISIT_SUMMARY_FALLBACK_MODEL_NAME",
            config.get("fallback_model_name") or "",
        )
        self.fallback_azure_endpoint = _env_override(
            "VISIT_SUMMARY_FALLBACK_AZURE_ENDPOINT",
            config.get("fallback_azure_endpoint") or "",
        )
        self.fallback_azure_api_version = _env_override(
            "VISIT_SUMMARY_FALLBACK_AZURE_API_VERSION",
            config.get("fallback_azure_api_version") or "2024-08-01-preview",
        )
        self.summary_timeout_seconds = float(os.environ.get("VISIT_SUMMARY_TIMEOUT_SECONDS", "300"))
        self.max_content_length = config.get("max_content_length", 100000)

        cache_enabled = config.get("cache_enabled")
        if cache_enabled is None:
            cache_enabled = _bool_env("VISIT_CACHE_ENABLED", True)
        cache_resume = config.get("cache_resume")
        if cache_resume is None:
            cache_resume = _bool_env("VISIT_CACHE_RESUME", True)

        cache_dir = config.get("cache_dir") or os.getenv("VISIT_CACHE_DIR", "")
        if not cache_dir:
            cache_dir = _DEFAULT_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = config.get("cache_file") or os.getenv("VISIT_CACHE_FILE", "")
        if not cache_file:
            cache_file = os.path.join(cache_dir, "visit_cache.db")

        cache_shards = config.get("cache_shards")
        if cache_shards is None:
            cache_shards = int(os.getenv("VISIT_CACHE_SHARDS", "1"))

        cache_auto_shard = config.get("cache_auto_shard")
        if cache_auto_shard is None:
            cache_auto_shard = _bool_env("VISIT_CACHE_AUTO_SHARD", False)

        cache_auto_merge = config.get("cache_auto_merge")
        if cache_auto_merge is None:
            cache_auto_merge = _bool_env("VISIT_CACHE_AUTO_MERGE", False)

        is_leader = _get_leader_flag(config)

        self.cache = (
            VisitCache(
                cache_dir=cache_dir,
                cache_file=cache_file,
                shards=cache_shards,
                resume=cache_resume,
                auto_shard=cache_auto_shard,
                auto_merge=cache_auto_merge,
                is_leader=is_leader,
            )
            if cache_enabled
            else None
        )

        if not self.jina_api_key:
            logger.warning("JINA_API_KEY not set, visit tool will not work")

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"visited_urls": [], "results": []}
        return instance_id, ToolResponse()

    def _validate_url(self, url: Any) -> Tuple[bool, str]:
        if not isinstance(url, str):
            return False, "[Visit] URL must be a string."
        url = url.strip()
        if not url:
            return False, "[Visit] URL cannot be empty."
        lowered = url.lower()
        if lowered.startswith("view-source:"):
            return False, "[Visit] Invalid URL protocol: 'view-source:' is not allowed."
        if lowered.startswith("javascript:"):
            return False, "[Visit] Invalid URL protocol: 'javascript:' is not allowed."
        if lowered.startswith("data:"):
            return False, "[Visit] Invalid URL protocol: 'data:' is not allowed."
        if lowered.startswith("file:"):
            return False, "[Visit] Invalid URL protocol: 'file:' is not allowed."
        return True, ""

    def _fetch_and_summarize(self, url: str, goal: str) -> str:
        content = jina_fetch_sync(url, self.jina_api_key, self.timeout)
        useful_info = summarize_content(
            content,
            url,
            goal,
            self.api_key,
            self.api_base,
            self.model_name,
            self.azure_endpoint,
            self.azure_api_version,
            self.fallback_api_key,
            self.fallback_api_base,
            self.fallback_model_name,
            self.fallback_azure_endpoint,
            self.fallback_azure_api_version,
            self.summary_timeout_seconds,
            self.max_content_length,
        )

        if self.cache and content and not content.startswith("[visit]"):
            if (
                not _is_visit_error_result(useful_info)
                and not _is_summarization_failure_result(useful_info)
            ):
                self.cache.set(url, goal, content, useful_info)

        return useful_info

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        url = parameters.get("url", "")
        goal = parameters.get("goal", "Extract relevant information")

        if not url:
            return ToolResponse(text="[Visit Error] URL cannot be empty."), 0.0, {"error": "empty_url"}

        urls = url if isinstance(url, list) else [url]
        loop = asyncio.get_running_loop()
        results = []
        valid_urls = []

        for u in urls:
            is_valid, error_msg = self._validate_url(u)
            if not is_valid:
                results.append(error_msg)
                continue
            u = u.strip()
            valid_urls.append(u)

            if self.cache:
                cached_info = self.cache.get_useful_information(u, goal)
                if cached_info:
                    logger.info("[visit] cache hit (url+goal): %s %s", u, goal)
                    results.append(cached_info)
                    continue

                cached_content = self.cache.get_content_by_url(u)
                if cached_content:
                    logger.info("[visit] cache hit (url) %s, summarizing: %s", u, goal)
                    result = await loop.run_in_executor(
                        None,
                        summarize_content,
                        cached_content,
                        u,
                        goal,
                        self.api_key,
                        self.api_base,
                        self.model_name,
                        self.azure_endpoint,
                        self.azure_api_version,
                        self.fallback_api_key,
                        self.fallback_api_base,
                        self.fallback_model_name,
                        self.fallback_azure_endpoint,
                        self.fallback_azure_api_version,
                        self.summary_timeout_seconds,
                        self.max_content_length,
                    )
                    if (
                        result
                        and not _is_visit_error_result(result)
                        and not _is_summarization_failure_result(result)
                    ):
                        self.cache.set(u, goal, cached_content, result)
                    results.append(result)
                    continue

            result = await loop.run_in_executor(None, self._fetch_and_summarize, u, goal)
            results.append(result)

        self._instance_dict[instance_id]["visited_urls"].extend(valid_urls)
        self._instance_dict[instance_id]["results"].extend(results)

        combined_result = "\n=======\n".join(results)
        error_count = sum(1 for result in results if _is_visit_error_result(result))
        if error_count == len(results):
            status = "error"
        elif error_count > 0:
            status = "partial_success"
        else:
            status = "success"
        metrics = {
            "url_count": len(urls),
            "valid_url_count": len(valid_urls),
            "error_count": error_count,
            "status": status,
        }

        return ToolResponse(text=combined_result), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
