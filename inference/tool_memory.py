import json
import os
import re
from typing import Union, Dict, List, Optional
from qwen_agent.tools.base import BaseTool, register_tool
from litellm import completion
from transformers import AutoTokenizer
import time
from datetime import datetime
from prompt import (
    MEMORY_LOCAL_SYSTEM_PROMPT,
    choose_local_openai_base_url,
    get_local_served_model_name,
    use_memory_local_prompt,
)
from openai import OpenAI


# System prompt for memory tool
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
TRIGGER NOTE (IMPORTANT)

This summarizer is invoked automatically when CONTEXT_THRESHOLD is reached:
- The system invokes summarization when context tokens reach a threshold.
- Focus on extracting evidence, deduplicating tool usage, and making the state more actionable.

Note: Agent-initiated condenser tool calls are ignored for memory updates.
Only automatic CONTEXT_THRESHOLD triggers will update the memory state.

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
- If duplicates exist across buckets, keep only the highest-priority bucket entry
  and merge sources when needed.

3) Tool extraction (evidence-driven):
- search tool_call:
  - Add each query to search_queries with a concise intent.
  - If search snippets reveal candidate authoritative URLs, you may reference them
    inside uncertain.need, but do NOT add them to visited_sources unless visited.

- visit tool_call + tool_response:
  - Add each visited URL to visited_sources.
  - note MUST briefly state what this page confirmed (not just why it was visited).
  - Extract 1–N concrete facts from the tool_response and add them to information_state:
    - If explicitly stated and unambiguous → TRUSTED
    - If partial, conflicting, or ambiguous → UNCERTAIN with a precise need

4) Information triage (fact-centric):
- TRUSTED:
  - Claims must be directly supported by visited sources.
  - Claims must be answer-ready and specific (numbers, dates, limits, rules).
  - reason must state where and why the fact is settled.
  - You MAY include a short preventive claim (e.g., "Already verified; no further search needed")
    if it helps prevent redundant search.

- UNTRUSTED:
  - Claims contradicted by visited sources or clearly unreliable.
  - reason should briefly state what contradicts it.

- UNCERTAIN:
  - Claims with conflicting or insufficient evidence.
  - reason must state what is missing or conflicting.
  - need MUST specify the next concrete step:
    - Prefer "visit <exact URL>" if a candidate URL exists.
    - Otherwise "search <exact query>".
  - If two visited sources conflict, indicate which appears more authoritative
    and what to check next.

- Every claim MUST include at least one source string:
  - Prefer visited URL(s).
  - Otherwise use labels like "tool_search_snippet" or "user_statement".

- Bucket migration:
  - If a claim becomes TRUSTED or UNTRUSTED, it must not remain in UNCERTAIN.

5) Output constraints:
- Output EXACTLY the keys shown in the schema. No extra keys.
- If a list has no items, output [].
- Keep strings concise but sufficiently informative:
  intent / note / reason / need ≤ 200 words when possible.
- Claim IDs:
  - Reuse existing IDs for identical claims if present.
  - Otherwise assign incremental IDs within each bucket prefix (T/U/C).

========================
INPUT HINTS

- search() calls: tool_call with name "search" and arguments { "query": [...] }.
- visit() calls: tool_call with name "visit" and arguments { "url": [...], "goal": "..." }.
- Tool responses: extract facts directly from them.
- Final answers: only promote to TRUSTED if backed by visited sources.

Return ONLY the updated JSON object."""


MEMORY_SYSTEM_PROMPT_NO_VISIT = """You are a State Summarizer for a DeepResearch agent.
Your ONLY job is to maintain a compact, parseable, context-aware state JSON for memory management.

Your primary objective is to prevent redundant search actions by
extracting useful, answer-ready information from tool responses and preserving it
in a structured state.

NOTE: The visit tool is DISABLED in this run. There are no visit() calls or visit responses.
All evidence comes from search results only. Do NOT suggest "visit <URL>" anywhere in the state.

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
  "visited_sources": [],
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
TRIGGER NOTE (IMPORTANT)

This summarizer is invoked automatically when CONTEXT_THRESHOLD is reached:
- The system invokes summarization when context tokens reach a threshold.
- Focus on extracting evidence, deduplicating tool usage, and making the state more actionable.

Note: Agent-initiated condenser tool calls are ignored for memory updates.
Only automatic CONTEXT_THRESHOLD triggers will update the memory state.

========================
CORE PRINCIPLE (CRITICAL)

Search snippets alone may be incomplete. For every search() tool_response, extract
every useful, concrete fact into information_state. Snippets from search results
are the ONLY evidence source in this run.

The goal is that the DeepResearch agent can rely on information_state.trusted
to answer questions directly, and rely on information_state.uncertain.need
to know the exact next search step without repeating queries.

========================
UPDATE RULES (IMPORTANT)

0) Anti-redundancy objective:
- The state must clearly encode:
  a) what is already verified and final (trusted),
  b) what is false or contradicted (untrusted),
  c) what is missing AND the exact next action to resolve it (uncertain.need).
- All uncertain.need values MUST be "search <exact query>" - do NOT suggest visiting URLs.

1) Merge with prev_state:
- Start from prev_state if provided; update it using new events.
- Never delete past entries except for:
  a) exact duplicates, or
  b) bucket migration (moving the same claim between uncertain/trusted/untrusted).

2) De-duplication:
- search_queries: dedupe by exact "q" string.
- visited_sources: always [].
- information_state: dedupe by exact "claim" string ACROSS ALL BUCKETS with priority:
  trusted > untrusted > uncertain.
- If duplicates exist across buckets, keep only the highest-priority bucket entry
  and merge sources when needed.

3) Tool extraction (evidence-driven):
- search tool_call:
  - Add each query to search_queries with a concise intent.
  - Extract any concrete facts from snippets into information_state.
  - If snippets mention a candidate URL but you cannot visit it, reference the URL
    only as a source label - do NOT add it to visited_sources or suggest visiting it.

4) Information triage (fact-centric):
- TRUSTED:
  - Claims clearly and unambiguously supported by search snippets.
  - Claims must be answer-ready and specific (numbers, dates, limits, rules).
  - reason must state which search result supports the fact.

- UNTRUSTED:
  - Claims contradicted by search results or clearly unreliable.
  - reason should briefly state what contradicts it.

- UNCERTAIN:
  - Claims with conflicting or insufficient evidence.
  - reason must state what is missing or conflicting.
  - need MUST specify the next concrete search step: "search <exact query>".
  - Do NOT use "visit <URL>" in need under any circumstances.

- Every claim MUST include at least one source string:
  - Use the search query or snippet label (e.g., "search_snippet:<query>").
  - Otherwise use labels like "tool_search_snippet" or "user_statement".

5) Output constraints:
- Output EXACTLY the keys shown in the schema. No extra keys.
- visited_sources MUST always be [].
- If a list has no items, output [].
- Keep strings concise but sufficiently informative.

========================
INPUT HINTS

- search() calls: tool_call with name "search" and arguments { "query": [...] }.
- Tool responses: extract facts directly from search snippets.
- Final answers: only promote to TRUSTED if clearly backed by search snippets.
- There are NO visit() calls in this run. Ignore any "visit" entries in prev_state.

Return ONLY the updated JSON object."""


@register_tool('condenser', allow_overwrite=True)
class Memory(BaseTool):
    """
    Condenser tool: calls the API through litellm and maintains the DeepResearch agent's state summary.
    """
    name = 'condenser'
    description = 'Access memory API through litellm. Maintains state summary for DeepResearch agent.'
    parameters = {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "description": "A chronological list of interaction events (user/assistant messages and tool calls/responses)."
            },
            "prev_state": {
                "type": ["object", "null"],
                "description": "The previous state JSON (may be empty or null)."
            }
        },
        "required": ["events"]
    }
    
    def __init__(self, *args, **kwargs):
        """Initialize the Memory tool"""
        super().__init__(*args, **kwargs)
        
        # Initialize the tokenizer for token counting
        tokenizer_path = os.environ.get("MEMORY_TOKENIZER_PATH", "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except Exception as e:
            print(f"[Memory] Warning: Failed to load tokenizer from {tokenizer_path}: {e}")
            print(f"[Memory] Will proceed without token threshold check")
            self.tokenizer = None
        
        # Read the threshold from environment variables. Prefer the shared script variable,
        # while keeping legacy names as fallback for older environments.
        threshold_str = (
            os.environ.get("MEMORY_THRESHOLD", "")
            or os.environ.get("MEMORY_CONTEXT_THRESHOLD", "")
            or os.environ.get("MEMORY_TOKEN_THRESHOLD", "")
        )
        if threshold_str:
            try:
                self.token_threshold = int(threshold_str)
            except ValueError:
                print(f"[Memory] Warning: Invalid memory threshold value: {threshold_str}, ignoring threshold")
                self.token_threshold = None
        else:
            self.token_threshold = None
        
    
    def call_api_with_litellm(self, messages: List[Dict], max_retries: int = 2) -> str:
        """
        Call the API via litellm
        
        Args:
            messages: message list, formatted as [{"role": "user", "content": "..."}]
            max_retries: maximum retry count
            
        Returns:
            content string returned by the API
        """
        visit_disabled = os.environ.get("DISABLE_VISIT_TOOL", "").lower() in ("1", "true", "yes")
        memory_system_prompt = MEMORY_SYSTEM_PROMPT_NO_VISIT if visit_disabled else MEMORY_SYSTEM_PROMPT

        if use_memory_local_prompt():
            model_name = get_local_served_model_name()
            full_messages = messages.copy()
            has_system = any(msg.get("role") == "system" for msg in full_messages)
            if not has_system:
                full_messages = [
                    {"role": "system", "content": memory_system_prompt},
                    {"role": "system", "content": MEMORY_LOCAL_SYSTEM_PROMPT},
                ] + full_messages

            last_error = None
            for attempt in range(max_retries):
                try:
                    client = OpenAI(
                        api_key="EMPTY",
                        base_url=choose_local_openai_base_url(),
                        timeout=600.0,
                    )
                    chat_response = client.chat.completions.create(
                        model=model_name,
                        messages=full_messages,
                        temperature=1
                    )
                    content = chat_response.choices[0].message.content
                    if content:
                        return content
                except Exception as e:
                    last_error = e
                    print(f"[Memory] local server call error (attempt {attempt + 1}/{max_retries}): {e}")
                    continue
            raise RuntimeError(f"Memory local server call failed after retries: {last_error}")

        # Use MEMORY_* prefixed environment variables, independent from other model settings
        model_name = os.environ.get("MEMORY_MODEL_NAME", "")
        
        if not model_name:
            raise ValueError("MEMORY_MODEL_NAME environment variable must be set")
        
        # Preserve the original memory prompt and optionally append the local
        # training-style prompt as an extra system message for A/B alignment.
        full_messages = messages.copy()
        has_system = any(msg.get("role") == "system" for msg in full_messages)
        if not has_system:
            system_messages = [{
                "role": "system",
                "content": memory_system_prompt
            }]
            if use_memory_local_prompt():
                system_messages.append({
                    "role": "system",
                    "content": MEMORY_LOCAL_SYSTEM_PROMPT
                })
            full_messages = system_messages + full_messages
        
        # Prepare litellm call arguments
        call_kwargs = {
            "model": model_name,
            "messages": full_messages,
            "temperature": 1,
            "num_retries": max_retries
        }
        
        # Use the shared API/Azure configuration, with legacy memory-specific vars as fallback.
        api_key = (
            os.environ.get("MEMORY_API_KEY")
            or os.environ.get("API_KEY")
            or os.environ.get("MEMORY_OPENAI_API_KEY")
        )
        if api_key:
            call_kwargs["api_key"] = api_key
        api_base = (
            os.environ.get("MEMORY_API_BASE")
            or os.environ.get("API_BASE")
        )
        if api_base:
            call_kwargs["api_base"] = api_base
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint:
            call_kwargs["api_base"] = azure_endpoint
            call_kwargs["api_version"] = (
                os.environ.get("AZURE_OPENAI_API_VERSION")
                or "2024-08-01-preview"
            )
            if not model_name.startswith("azure/"):
                azure_deployment = (
                    os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                    or model_name
                )
                call_kwargs["model"] = f"azure/{azure_deployment}"
        
        # Record metadata before the call
        call_start_time = time.time()
        input_token_count = 0
        if self.tokenizer is not None:
            try:
                full_text = "\n".join([msg.get("content", "") for msg in full_messages])
                input_token_count = self.count_tokens(full_text)
            except Exception:
                pass
        
        response_content = ""
        output_token_count = 0
        api_error = None
        
        for attempt in range(max_retries):
            try:
                # Use the unified litellm calling interface
                response = completion(**call_kwargs)
                content = response.choices[0].message.content
                
                # Try to get token usage information from the response
                if hasattr(response, 'usage'):
                    if hasattr(response.usage, 'prompt_tokens'):
                        input_token_count = response.usage.prompt_tokens
                    if hasattr(response.usage, 'completion_tokens'):
                        output_token_count = response.usage.completion_tokens
                
                if content:
                    response_content = content
                    break
                else:
                    response_content = ""
                    break
            except Exception as e:
                api_error = str(e)
                print(f"[Memory] API call attempt {attempt + 1} failed: {e}")
                if attempt == (max_retries - 1):
                    response_content = f"[Memory] Failed to call API after {max_retries} attempts: {str(e)}"
                time.sleep(0.5)
                continue
        
        if not response_content:
            response_content = "[Memory] Failed to call API"
        
        # If token counts are not returned by the API, estimate output tokens with the tokenizer
        if output_token_count == 0 and self.tokenizer is not None and response_content:
            try:
                output_token_count = self.count_tokens(response_content)
            except Exception:
                pass
        
        call_end_time = time.time()
        call_duration = call_end_time - call_start_time
        
        # Try to parse the JSON output
        parsed_json = None
        json_parse_error = None
        if response_content and not api_error:
            parsed_json = self._extract_json_from_response(response_content)
            if parsed_json is None:
                json_parse_error = "Failed to parse JSON from response"
                print(f"[Memory] Warning: {json_parse_error}")
        
        return response_content
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text
        
        Args:
            text: text to count
            
        Returns:
            token count; return 0 if the tokenizer is not initialized
        """
        if self.tokenizer is None:
            return 0
        
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            print(f"[Memory] Warning: Failed to count tokens: {e}")
            return 0
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """
        Extract a JSON object from the response
        Handle possible markdown wrappers(such as ```json ... ```)
        
        Args:
            response: response string returned by the LLM
            
        Returns:
            parsed JSON dict, or None if parsing fails
        """
        if not response:
            return None
        
        # Try parsing directly
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from a markdown code block
        # Match ```json ... ``` or ``` ... ```
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
        
        # Try to find the content between the first { and the last }
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                json_str = response[first_brace:last_brace + 1]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _format_memory_input(self, events: List, 
                            prev_state: Optional[Dict] = None) -> str:
        """
        Format the input for the memory tool
        
        Args:
            events: interaction event list(required)
            prev_state: previous state JSON(optional)
            
        Returns:
            formatted user prompt string
        """
        # Format events as a JSON string
        events_str = json.dumps(events, ensure_ascii=False, indent=2)
        
        # Format prev_state as a JSON string
        if prev_state is None:
            prev_state_str = "null"
        else:
            prev_state_str = json.dumps(prev_state, ensure_ascii=False, indent=2)
        
        # Use template formatting
        user_prompt = f"""You are given the following inputs for state summarization.

events:
<<<EVENTS>>>

prev_state:
<<<PREV_STATE>>>

IMPORTANT: If no state changes are warranted, return prev_state unchanged."""
        
        # Replace placeholders
        user_prompt = user_prompt.replace("<<<EVENTS>>>", events_str)
        user_prompt = user_prompt.replace("<<<PREV_STATE>>>", prev_state_str)
        
        return user_prompt
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Main call method
        
        Args:
            params: parameter dict; must contain:
                - events: interaction event list(required)
                - prev_state: previous state JSON (optional, may be null)
            
        Returns:
            API return value (should be state in JSON format)
        """
        try:
            if isinstance(params, str):
                params = json.loads(params)
            
            # Get required parameters
            events = params.get("events")
            prev_state = params.get("prev_state")
            
            # Validate required parameters
            if events is None:
                return "[Memory] Invalid request format: 'events' field is required"
            
            if not isinstance(events, list):
                return "[Memory] Invalid request format: 'events' must be a list"
            
            # Format the input
            formatted_content = self._format_memory_input(
                events=events,
                prev_state=prev_state
            )
            
            # Build the messages
            messages = [{
                "role": "user",
                "content": formatted_content
            }]
            
            # Check the token threshold
            if self.token_threshold is not None and self.tokenizer is not None:
                token_count = self.count_tokens(formatted_content)
                if token_count <= self.token_threshold:
                    print(f"[Memory] Token count ({token_count}) <= threshold ({self.token_threshold}), skipping LLM call")
                    result = f"[Memory] Content token count ({token_count}) is below threshold ({self.token_threshold}), no LLM call made."
                    return result
            
            # Call the API
            result = self.call_api_with_litellm(messages)
            return result
                
        except json.JSONDecodeError as e:
            return f"[Memory] Invalid request format: Input must be a valid JSON object: {str(e)}"
        except Exception as e:
            return f"[Memory] Error processing request: {str(e)}"
