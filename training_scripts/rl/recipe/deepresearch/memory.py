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
Memory/Condenser system for DeepResearch.
Maintains state summary to prevent redundant search/visit actions.
Migrated from slime/examples/deepresearch/tool_memory.py
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _build_openai_client_config(
    *,
    api_key: str,
    api_base: str = "",
    model_name: str = "",
    azure_endpoint: str = "",
    azure_api_version: str = "",
    timeout_seconds: float = 300.0,
):
    if azure_endpoint:
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", model_name)
        client = AzureOpenAI(
            api_key=api_key,
            api_version=azure_api_version or "2024-08-01-preview",
            azure_endpoint=azure_endpoint,
            timeout=timeout_seconds,
        )
        return client, azure_deployment

    client_kwargs = {"api_key": api_key, "timeout": timeout_seconds}
    if api_base:
        client_kwargs["base_url"] = api_base
    return OpenAI(**client_kwargs), model_name


def _call_condenser_with_client(
    *,
    client,
    model_name: str,
    messages: List[Dict[str, str]],
    max_retries: int,
    label: str,
) -> tuple[Optional[str], Optional[Exception]]:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=1,
                max_completion_tokens=16384,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from API.")
            if parse_memory_result(content) is None:
                raise ValueError("Response was not valid condenser JSON.")
            return content, None
        except Exception as e:
            last_error = e
            logger.warning(f"[Memory] {label} attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 4))
    return None, last_error


def _normalize_local_openai_base_url(address: str) -> str:
    addr = (address or "").strip()
    if not addr:
        return ""
    if addr.startswith("http://") or addr.startswith("https://"):
        return addr.rstrip("/")
    if ":" in addr:
        host, port = addr.rsplit(":", 1)
        if port.isdigit():
            return f"http://{host}:{port}/v1"
    return f"http://{addr}:6000/v1"


def _resolve_local_fallback_base_urls() -> List[str]:
    base_urls = (os.environ.get("LOCAL_OPENAI_BASE_URLS") or os.environ.get("LOCAL_OPENAI_BASE_URL") or "").strip()
    if base_urls:
        urls = [_normalize_local_openai_base_url(x) for x in re.split(r"[,\s]+", base_urls) if x.strip()]
        return [u for u in urls if u]

    ips_raw = (os.environ.get("EVAL_LLM_IPS") or "").strip()
    ports_raw = (os.environ.get("EVAL_LLM_PORTS") or "6000,6001,6002,6003").strip()
    ips = [x for x in re.split(r"[,\s]+", ips_raw) if x]
    ports = [x for x in re.split(r"[,\s]+", ports_raw) if x]
    urls: List[str] = []
    if not ips:
        return urls
    for ip in ips:
        for port in ports:
            if port.isdigit():
                urls.append(f"http://{ip}:{port}/v1")
    return urls


def _call_condenser_with_local_base_urls(
    *,
    base_urls: List[str],
    api_key: str,
    model_name: str,
    messages: List[Dict[str, str]],
    max_retries: int,
    timeout_seconds: float,
) -> tuple[Optional[str], Optional[Exception]]:
    base_url = base_urls[0]
    try:
        client = OpenAI(api_key=api_key or "dummy", base_url=base_url, timeout=timeout_seconds)
        content, err = _call_condenser_with_client(
            client=client,
            model_name=model_name,
            messages=messages,
            max_retries=1,
            label=f"Local memory fallback via {base_url}",
        )
        return content, err
    except Exception as e:
        logger.warning(f"[Memory] Local memory fallback via {base_url} failed: {e}")
        return None, e


def build_fallback_memory_state(events: List[Dict], prev_state: Optional[Dict] = None) -> Dict:
    """Return an empty bounded memory state when condenser fallback is needed."""
    return {
        "version": "dr_state",
        "search_queries": [],
        "visited_sources": [],
        "information_state": {
            "trusted": [],
            "untrusted": [],
            "uncertain": [],
        },
    }

# Memory system prompt for state summarization
MEMORY_SYSTEM_PROMPT = """You are a State Summarizer for a DeepResearch agent.
Your ONLY job is to maintain a compact, parseable, context-aware state JSON for memory management.

Your primary objective is to prevent redundant search and redundant visit actions by
extracting useful, answer-ready information from tool responses and preserving it
in a structured state.

You will be given:
1) events: a chronological list of interaction events (user/assistant messages and tool calls/responses)
2) prev_state: the previous state JSON (may be empty or null)

You MUST output ONLY a single JSON object that conforms EXACTLY to the schema below.
No markdown, no extra text, no code fences, no explanations.

========================
OUTPUT JSON SCHEMA (STRICT)

{
  "version": "dr_state",
  "search_queries": [
    { "q": "string", "intent": "string" }
  ],
  "visited_sources": [
    { "url": "string", "note": "string" }
  ],
  "information_state": {
    "trusted": [
      { "id": "T1", "claim": "string", "sources": ["string"], "reason": "string" }
    ],
    "untrusted": [
      { "id": "U1", "claim": "string", "sources": ["string"], "reason": "string" }
    ],
    "uncertain": [
      { "id": "C1", "claim": "string", "sources": ["string"], "reason": "string", "need": "string" }
    ]
  }
}

========================
CORE PRINCIPLE (CRITICAL)

Visited pages alone are NOT useful memory.

For every visit() tool_response, you MUST attempt to extract at least one
useful, concrete fact into information_state unless the page is irrelevant.

The goal is that the DeepResearch agent can rely on information_state.trusted
to answer questions directly, and rely on information_state.uncertain.need
to know the exact next step without re-searching.

========================
UPDATE RULES (IMPORTANT)

0) Anti-redundancy objective:
- The state must clearly encode:
  a) what is already verified and final (trusted),
  b) what is false or contradicted (untrusted),
  c) what is missing AND the exact next action to resolve it (uncertain.need).
- Prefer concrete actions such as:
  "visit <exact URL>" or "search <exact query>".

1) Merge with prev_state:
- Start from prev_state if provided; update it using new events.
- Never delete past entries except for:
  a) exact duplicates, or
  b) bucket migration (moving the same claim between uncertain/trusted/untrusted).

2) De-duplication:
- search_queries: dedupe by exact "q" string.
- visited_sources: dedupe by exact "url".
- information_state: dedupe by exact "claim" string ACROSS ALL BUCKETS with priority:
  trusted > untrusted > uncertain.

3) Tool extraction (evidence-driven):
- search / google_scholar tool_call:
  - Add each query to search_queries with a concise intent.
  - If search snippets reveal candidate authoritative URLs, reference them
    inside uncertain.need, but do NOT add them to visited_sources unless visited.

- visit tool_call + tool_response:
  - Add each visited URL to visited_sources.
  - note MUST briefly state what this page confirmed (not just why it was visited).
  - Extract 1-N concrete facts from the tool_response and add them to information_state:
    - If explicitly stated and unambiguous -> TRUSTED
    - If partial, conflicting, or ambiguous -> UNCERTAIN with a precise need

4) Information triage (fact-centric):
- TRUSTED:
  - Claims must be directly supported by visited sources.
  - Claims must be answer-ready and specific (numbers, dates, limits, rules).
  - reason must state where and why the fact is settled.

- UNTRUSTED:
  - Claims contradicted by visited sources or clearly unreliable.
  - reason should briefly state what contradicts it.

- UNCERTAIN:
  - Claims with conflicting or insufficient evidence.
  - reason must state what is missing or conflicting.
  - need MUST specify the next concrete step:
    - Prefer "visit <exact URL>" if a candidate URL exists.
    - Otherwise "search <exact query>".

5) Output constraints:
- Output EXACTLY the keys shown in the schema. No extra keys.
- If a list has no items, output [].
- Keep strings concise but sufficiently informative.

Return ONLY the updated JSON object."""


def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    Extract JSON object from LLM response.
    Handles markdown wrapping (```json ... ```)

    Args:
        response: LLM response string

    Returns:
        Parsed JSON dict, or None if parsing fails
    """
    if not response:
        return None

    # Try direct parsing
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    json_patterns = [
        r'```json\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```',
        r'```json\s*(.*?)```',
        r'```\s*(.*?)```',
    ]

    for pattern in json_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                json_str = match.group(1).strip()
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

    # Try finding first { to last }
    first_brace = response.find('{')
    last_brace = response.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            json_str = response[first_brace:last_brace + 1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return None


def format_memory_input(events: List[Dict], prev_state: Optional[Dict] = None) -> str:
    """
    Format memory tool input for LLM.

    Args:
        events: List of interaction events
        prev_state: Previous state JSON (may be None)

    Returns:
        Formatted user prompt string
    """
    events_str = json.dumps(events, ensure_ascii=False, indent=2)

    if prev_state is None:
        prev_state_str = "null"
    else:
        prev_state_str = json.dumps(prev_state, ensure_ascii=False, indent=2)

    user_prompt = f"""You are given the following inputs for state summarization.

events:
{events_str}

prev_state:
{prev_state_str}

IMPORTANT: If no state changes are warranted, return prev_state unchanged."""

    return user_prompt


def call_condenser_sync(
    events: List[Dict],
    prev_state: Optional[Dict] = None,
    max_retries: int = 2,
    api_key: str = "",
    api_base: str = "",
    model_name: str = "",
    azure_endpoint: str = "",
    azure_api_version: str = "",
) -> str:
    """
    Call condenser/memory API synchronously.

    Args:
        events: List of interaction events
        prev_state: Previous state JSON
        max_retries: Maximum retry attempts
        api_key: OpenAI API key
        api_base: OpenAI API base URL
        model_name: Model name
        azure_endpoint: Azure OpenAI endpoint
        azure_api_version: Azure API version

    Returns:
        API response string (should be JSON state)
    """
    # Get model configuration from environment if not provided
    if not model_name:
        model_name = os.environ.get("MEMORY_MODEL_NAME", "")
    if not api_key:
        api_key = os.environ.get("MEMORY_API_KEY", "")
    if not api_base:
        api_base = os.environ.get("MEMORY_API_BASE", "")
    if not azure_endpoint:
        azure_endpoint = os.environ.get("MEMORY_AZURE_ENDPOINT", "")
    if not azure_api_version:
        azure_api_version = os.environ.get("MEMORY_AZURE_API_VERSION") or "2024-08-01-preview"
    primary_timeout_seconds = float(os.environ.get("MEMORY_TIMEOUT_SECONDS", "300"))
    fallback_timeout_seconds = float(
        os.environ.get("MEMORY_FALLBACK_TIMEOUT_SECONDS", str(primary_timeout_seconds))
    )
    local_fallback_timeout_seconds = float(
        os.environ.get(
            "MEMORY_LOCAL_FALLBACK_TIMEOUT_SECONDS",
            os.environ.get("LOCAL_OPENAI_TIMEOUT_SECONDS", str(primary_timeout_seconds)),
        )
    )

    fallback_model_name = os.environ.get("MEMORY_FALLBACK_MODEL_NAME", "deepseek.v3.2")
    fallback_api_key = os.environ.get("MEMORY_FALLBACK_API_KEY", "")
    fallback_api_base = os.environ.get("MEMORY_FALLBACK_API_BASE", "")
    fallback_enabled = bool(fallback_api_key and fallback_api_base and fallback_model_name)
    local_fallback_model_name = os.environ.get(
        "MEMORY_LOCAL_FALLBACK_MODEL_NAME",
        "",
    )
    local_fallback_api_key = os.environ.get("MEMORY_LOCAL_FALLBACK_API_KEY", "")
    local_fallback_base_urls = _resolve_local_fallback_base_urls()
    local_fallback_enabled = bool(local_fallback_base_urls and local_fallback_model_name)

    # Format input
    user_content = format_memory_input(events, prev_state)

    messages = [
        {"role": "system", "content": MEMORY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    candidate_apis: List[Dict[str, Any]] = []
    if fallback_enabled:
        candidate_apis.append(
            {
                "label": "Bedrock DeepSeek primary candidate",
                "client_kwargs": {
                    "api_key": fallback_api_key,
                    "api_base": fallback_api_base,
                    "model_name": fallback_model_name,
                    "timeout_seconds": fallback_timeout_seconds,
                },
                "max_retries": max_retries,
            }
        )
    else:
        logger.warning("[Memory] Bedrock DeepSeek primary candidate unavailable: missing API key/base/model.")

    if api_key:
        candidate_apis.append(
            {
                "label": "Azure/OpenAI fallback candidate",
                "client_kwargs": {
                    "api_key": api_key,
                    "api_base": api_base,
                    "model_name": model_name,
                    "azure_endpoint": azure_endpoint,
                    "azure_api_version": azure_api_version,
                    "timeout_seconds": primary_timeout_seconds,
                },
                "max_retries": max_retries,
            }
        )
    else:
        logger.warning("[Memory] Azure/OpenAI fallback candidate unavailable: missing API key.")

    primary_error: Optional[Exception] = None
    for idx, candidate in enumerate(candidate_apis):
        try:
            client, selected_model_name = _build_openai_client_config(**candidate["client_kwargs"])
            content, candidate_error = _call_condenser_with_client(
                client=client,
                model_name=selected_model_name,
                messages=messages,
                max_retries=candidate["max_retries"],
                label=candidate["label"],
            )
            if content:
                return content
            primary_error = candidate_error
            if idx < len(candidate_apis) - 1:
                logger.warning(
                    "[Memory] %s failed; falling back to %s",
                    candidate["label"],
                    candidate_apis[idx + 1]["label"],
                )
        except Exception as e:
            primary_error = e
            if idx < len(candidate_apis) - 1:
                logger.warning(
                    "[Memory] %s raised before completion; falling back to %s: %s",
                    candidate["label"],
                    candidate_apis[idx + 1]["label"],
                    e,
                )

    if local_fallback_enabled:
        logger.warning(
            "[Memory] Falling back to local eval nodes model=%s urls=%s",
            local_fallback_model_name,
            ",".join(local_fallback_base_urls),
        )
        content, local_fallback_error = _call_condenser_with_local_base_urls(
            base_urls=local_fallback_base_urls,
            api_key=local_fallback_api_key,
            model_name=local_fallback_model_name,
            messages=messages,
            max_retries=max_retries,
            timeout_seconds=local_fallback_timeout_seconds,
        )
        if content:
            return content
        return (
            f"[Memory Error] Primary/Bedrock/local fallback all failed. "
            f"last_primary={primary_error}; local={local_fallback_error}"
        )

    return f"[Memory Error] Failed to call primary memory API: {primary_error}"


async def call_condenser_async(
    events: List[Dict],
    prev_state: Optional[Dict] = None,
    max_retries: int = 2,
    api_key: str = "",
    api_base: str = "",
    model_name: str = "",
    azure_endpoint: str = "",
    azure_api_version: str = "",
) -> str:
    """
    Call condenser/memory API asynchronously.

    Args:
        events: List of interaction events
        prev_state: Previous state JSON
        max_retries: Maximum retry attempts
        api_key: OpenAI API key
        api_base: OpenAI API base URL
        model_name: Model name
        azure_endpoint: Azure OpenAI endpoint
        azure_api_version: Azure API version

    Returns:
        API response string (should be JSON state)
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        call_condenser_sync,
        events,
        prev_state,
        max_retries,
        api_key,
        api_base,
        model_name,
        azure_endpoint,
        azure_api_version
    )


def parse_memory_result(result: str) -> Optional[Dict]:
    """
    Parse memory tool result to extract state JSON.

    Args:
        result: Memory tool response string

    Returns:
        Parsed state dict, or None if parsing fails
    """
    if not result or result.startswith("[Memory"):
        return None

    return extract_json_from_response(result)


def format_prev_state_section(state: Dict) -> str:
    """
    Format state JSON as prev_state section for user message.

    Args:
        state: State JSON dict

    Returns:
        Formatted prev_state section string
    """
    state_json = json.dumps(state, ensure_ascii=False, indent=2)

    return f"""

====================
RESEARCH STATE SUMMARY (prev_state)
====================
The following is a compressed summary of your research progress so far. Use this to:
- Avoid repeating searches you've already done (check search_queries)
- Reference information you've already verified (check information_state.trusted)
- Build upon previous findings rather than starting from scratch

{state_json}

IMPORTANT: This state summary is maintained automatically. You can reference it to avoid redundant work and build upon previous research findings.
"""


def extract_events_from_messages(messages: List[Dict]) -> List[Dict]:
    """Extract events from message history for condenser."""
    events = []
    for msg in messages:
        role = msg.get("role", "")
        if role in ["user", "assistant"]:
            events.append({
                "role": role,
                "content": msg.get("content", "")
            })
    return events
