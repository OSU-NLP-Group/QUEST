import json
import json5
import os
import sys
import re
import hashlib
import threading
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer
from datetime import datetime, date
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from prompt import *
import time
import asyncio
import copy
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
# Also add the current directory so modules in this directory can be imported(such as tool_memory.py)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from tool_scholar import *
from tool_search import *
from tool_visit import *
from tool_memory import *

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

def is_python_tool_enabled():
    enable_python = os.getenv("ENABLE_PYTHON_TOOL", "true").strip().lower()
    return enable_python not in {"0", "false", "no", "off"}


PYTHON_TOOL_ENABLED = is_python_tool_enabled()

if PYTHON_TOOL_ENABLED:
    try:
        from tool_python import PythonInterpreter
    except ImportError as exc:
        raise RuntimeError(
            "ENABLE_PYTHON_TOOL=true requires the optional Python tool dependencies. "
            "Install sandbox_fusion or set ENABLE_PYTHON_TOOL=false."
        ) from exc
else:
    PythonInterpreter = None


def is_scholar_tool_enabled():
    enable_scholar = os.getenv("ENABLE_SCHOLAR_TOOL", "false").strip().lower()
    return enable_scholar not in {"0", "false", "no", "off"}


SCHOLAR_TOOL_ENABLED = is_scholar_tool_enabled()

TOOL_CLASS = [Visit(), Search(), Memory()]
if SCHOLAR_TOOL_ENABLED:
    TOOL_CLASS.append(Scholar())
if PYTHON_TOOL_ENABLED:
    TOOL_CLASS.append(PythonInterpreter())
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

# Remove visit from TOOL_MAP when disabled to prevent execution even if a model emits it.
if os.environ.get("DISABLE_VISIT_TOOL", "").lower() in ("1", "true", "yes"):
    TOOL_MAP.pop("visit", None)

import random

PYTHON_TOOL_PROMPT = """{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:\n1. The 'arguments' JSON object must be empty: {}.\n2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.\n\nIMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.\n\nExample of a correct call:\n<tool_call>\n{\"name\": \"PythonInterpreter\", \"arguments\": {}}\n<code>\nimport numpy as np\n# Your code here\nprint(f\"The result is: {np.mean([1,2,3])}\")\n</code>\n</tool_call>", "parameters": {"type": "object", "properties": {}}}}
"""

def today_date():
    return date.today().strftime("%Y-%m-%d")


def _visit_disabled_notice() -> str:
    return """

# CRITICAL - visit tool is DISABLED in this run

- Do not call `visit`. Never emit a `<tool_call>` whose `"name"` is `"visit"`.
- The `visit` tool is not available; such calls fail and waste context.
- Gather new evidence using `search` only. If `condenser` appears in <tools> above, you may still call it for memory / state summarization.
- If `prev_state`, instructions, or your own plan mention visiting a URL, issue another `search` query instead.

"""


def _apply_visit_disabled_text_overrides(prompt: str) -> str:
    replacements = [
        (
            "- Check `visited_sources` to avoid visiting URLs that have already been visited.",
            "- Check `visited_sources` to avoid redoing work already reflected in state (there is no `visit` tool in this run).",
        ),
        (
            "You can use these directly in your answer without re-searching or re-visiting.",
            "You can use these directly in your answer without re-running the same searches.",
        ),
        (
            'The `need` field specifies the exact next action (e.g., "visit <URL>" or "search <query>") to resolve the uncertainty.',
            'The `need` field may suggest a next action; if it mentions visiting a URL, use an additional `search` query instead (`visit` is disabled).',
        ),
        (
            "IMPORTANT: Do NOT search for or visit information that is already in `prev_state`, unless it's insufficient to answer the user's question. Only in this case, you are encouraged to search for more information or even visit the same URL. Instead, use the information from `prev_state` directly, or follow the specific actions suggested in `information_state.uncertain.need` if more information is needed.",
            "IMPORTANT: Do NOT search for information that is already in `prev_state`, unless it's insufficient to answer the user's question. Only in this case, issue additional `search` queries (`visit` is disabled). Use `prev_state` directly where possible, or follow `information_state.uncertain.need` but map any \"visit\" suggestion to `search`.",
        ),
        (
            "- The visit tool can ONLY open bm25://<docid> URLs from search results. External URLs (https://, http://) will fail.",
            "- Visit is disabled in this run; rely on search snippets and metadata from the local index only.",
        ),
        (
            "- Do NOT attempt to visit Wikipedia, Google, or any other external website.",
            "- Do not call `visit` for external sites or bm25 links; use search only.",
        ),
        (
            "- When search returns relevant documents, use visit to read their full content via the bm25:// links before drawing conclusions.",
            "- When search returns relevant documents, base conclusions on returned snippets and scores; do not call `visit`.",
        ),
    ]
    for old, new in replacements:
        if old in prompt:
            prompt = prompt.replace(old, new, 1)
    return prompt


def build_system_prompt(enable_python_tool: bool) -> str:
    prompt_name = os.getenv("SYSTEM_PROMPT_NAME", "").strip()
    if prompt_name and prompt_name != "SYSTEM_PROMPT":
        prompt = globals().get(prompt_name)
        if prompt is None:
            print(
                f"[build_system_prompt] Warning: SYSTEM_PROMPT_NAME={prompt_name!r} "
                "not found in prompt.py, falling back to SYSTEM_PROMPT"
            )
            prompt = SYSTEM_PROMPT
    elif os.environ.get('BM25_INDEX_PATH', '') or os.environ.get('FAISS_INDEX_PATH', ''):
        prompt = BROWSECOMP_PLUS_SYSTEM_PROMPT
    else:
        prompt = SYSTEM_PROMPT
    if SCHOLAR_TOOL_ENABLED:
        prompt = prompt.replace("</tools>", f"{SCHOLAR_TOOL_PROMPT}</tools>")
    if enable_python_tool:
        prompt = prompt.replace("</tools>", f"{PYTHON_TOOL_PROMPT}</tools>")
    visit_disabled = os.environ.get("DISABLE_VISIT_TOOL", "").lower() in ("1", "true", "yes")
    if visit_disabled:
        prompt = re.sub(r'\{"type": "function", "function": \{"name": "visit".*?\}\}\n?', '', prompt)
        if "</tools>" in prompt:
            prompt = prompt.replace("</tools>", "</tools>" + _visit_disabled_notice(), 1)
        prompt = _apply_visit_disabled_text_overrides(prompt)
    return prompt


class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model_path"]
        self.python_tool_enabled = PYTHON_TOOL_ENABLED
        self.scholar_tool_enabled = SCHOLAR_TOOL_ENABLED
        self.function_list = function_list or []
        
        # Base configuration (not modified concurrently)
        memory_threshold_str = (
            os.getenv('MEMORY_THRESHOLD')
            or os.getenv('MEMORY_CONTEXT_THRESHOLD')
            or os.getenv('MEMORY_TOKEN_THRESHOLD')
            or '16000'
        )
        self.base_memory_context_threshold = int(memory_threshold_str)
        print(f"Base memory threshold: {self.base_memory_context_threshold}")
        print(f"Python tool enabled: {self.python_tool_enabled}")
        print(f"Scholar tool enabled: {self.scholar_tool_enabled}")
        self.memory_enabled = os.getenv('MEMORY_ENABLED', 'true').lower() == 'true'
        self.memory_strategy = os.getenv('MEMORY_STRATEGY', 'condenser').strip().lower()
        print(f"Memory strategy: {self.memory_strategy}")
        
        # Base log directory(not modified concurrently)
        self.base_log_dir = os.getenv('TASK_LOG_DIR', './task_logs')
        Path(self.base_log_dir).mkdir(parents=True, exist_ok=True)
        
        # ========== Key fix: use threading.local() to store thread-specific state ==========
        # this gives each thread its own independent state and prevents interference
        self._thread_local = threading.local()

        # Cache tokenizer to avoid reloading every time
        self._tokenizer = None
    
    def _get_thread_state(self):
        """Get the current thread state, initializing it if needed"""
        if not hasattr(self._thread_local, 'initialized'):
            self._init_thread_state()
        return self._thread_local
    
    def _init_thread_state(self):
        """Initialize the current thread state"""
        tl = self._thread_local
        tl.initialized = True
        tl.url_filter_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }
        tl.memory_state = None  # prev_state for memory tool
        tl.memory_context_threshold = self.base_memory_context_threshold
        tl.context_threshold_triggered = False
        tl.condenser_processing = False
        tl.condenser_call_count = 0
        tl.original_messages_snapshots = []
        tl.full_messages = []  # full message history; not compressed by condenser; used for trajectory_no_memory
        tl.task_log_dir = None
        tl.current_question = None
        tl.current_filename = ""
        tl.trajectory_update_count = 0
        tl.model = None
        tl.preferred_endpoint = None  # (host, port) for endpoint affinity to utilize KV cache
        tl.endpoint_failure_count = 0  # consecutive failure count for current endpoint
        tl.user_prompt = None

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content

    
    def _parse_memory_result(self, result: str) -> Optional[Dict]:
        """
        Parse the result returned by the memory tool and extract the JSON state
        
        Args:
            result: string returned by the memory tool
            
        Returns:
            parsed state JSON, or None if parsing fails
        """
        if not result or result.startswith("[Memory]"):
            return None
        
        try:
            # Try parsing directly
            new_state = json.loads(result)
            return new_state
        except json.JSONDecodeError:
            # Try extracting JSON from a markdown code block
            json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', result, re.DOTALL)
            if json_match:
                try:
                    new_state = json.loads(json_match.group(1))
                    return new_state
                except json.JSONDecodeError:
                    pass
            
            # Try to find the content from the first { to the last }
            first_brace = result.find('{')
            last_brace = result.rfind('}')
            if first_brace != -1 and last_brace != -1:
                try:
                    new_state = json.loads(result[first_brace:last_brace + 1])
                    return new_state
                except json.JSONDecodeError:
                    pass
            
            print(f"[Memory] Warning: Failed to parse JSON from memory tool response: {result[:200]}")
            return None
    
    def _save_memory_log(self, original_messages: List[Dict], compressed_messages: List[Dict], 
                        state: Dict, trigger_reason: str, round_num: int):
        """
        Save logs for memory operations(simplified version, keeping only necessary debug information)
        
        Args:
            original_messages: original messages before compression(actually seen by the model)
            compressed_messages: messages after compression(new prev_state added and later messages removed)
            state: generated state JSON
            trigger_reason: trigger reason (CONTEXT_THRESHOLD, final, or timeout)
            round_num: current round number
        """
        try:
            tl = self._get_thread_state()
            tl.condenser_call_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Save only one simplified log file with the necessary information(save it directly under the task directory)
            log_data = {
                "condenser_call_number": tl.condenser_call_count,
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "round": round_num,
                "trigger_reason": trigger_reason,
                "state": state,
                "original_message_count": len(original_messages),
                "compressed_message_count": len(compressed_messages),
                "messages_removed": len(original_messages) - len(compressed_messages)
            }
            
            log_file = os.path.join(tl.task_log_dir, f"condenser_call_{tl.condenser_call_count}_{timestamp}.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            print(f"[Memory] Log saved to: {log_file}")
            
        except Exception as e:
            print(f"[Memory] Warning: Failed to save memory log: {e}")
            import traceback
            traceback.print_exc()
    
    def _remove_prev_state_from_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Remove all prev_state sections from messages and return the original messages(without the memory mechanism)
        
        Args:
            messages: messages containing prev_state
            
        Returns:
            messages after removing prev_state
        """
        cleaned_messages = copy.deepcopy(messages)
        
        for msg in cleaned_messages:
            if msg.get("role") == "user" and "content" in msg:
                content = msg["content"]
                # Find and remove the prev_state section
                prev_state_start = content.find("====================\nRESEARCH STATE SUMMARY")
                if prev_state_start != -1:
                    # Remove everything from prev_state to the end
                    msg["content"] = content[:prev_state_start].rstrip()
        
        return cleaned_messages

    def _normalize_tool_call(self, raw_tool_call: str) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
        """
        Parse and normalize a tool_call payload.

        Returns:
            (tool_name, tool_args, error_message)
            - On success: (normalized_name, normalized_args_dict, None)
            - On failure: (None, None, "[Tool Error] ...")
        """
        # Handle <code>...</code> tags for PythonInterpreter:
        # Model may generate: {"name": "PythonInterpreter", "arguments": {}}\n<code>...\n</code>
        extracted_code = None
        json_part = raw_tool_call
        code_match = re.search(r'<code>(.*?)</code>', raw_tool_call, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
            # Remove <code>...</code> from the string before JSON parsing
            json_part = raw_tool_call[:code_match.start()].strip()

        try:
            tool_call = json.loads(json_part)
        except Exception:
            raw_tool_name = ["__invalid_tool_call_json__", raw_tool_call]
            return (
                None,
                None,
                f"[Tool Error] Invalid tool call format: name must be string, got {raw_tool_name!r}",
            )

        # Align with training-side parser:
        # - dict: read {"name", "arguments"}
        # - non-dict: pass through as tool_name with empty args
        if isinstance(tool_call, dict):
            raw_tool_name = tool_call.get("name", "")
            raw_tool_args = tool_call.get("arguments", {})
        else:
            raw_tool_name = tool_call
            raw_tool_args = {}

        tool_name = raw_tool_name
        if isinstance(tool_name, list):
            if len(tool_name) == 1 and isinstance(tool_name[0], str):
                tool_name = tool_name[0]
            else:
                return (
                    None,
                    None,
                    f"[Tool Error] Invalid tool call format: name must be string, got {raw_tool_name!r}",
                )

        if not isinstance(tool_name, str) or not tool_name.strip():
            return (
                None,
                None,
                f"[Tool Error] Invalid tool call format: name must be string, got {raw_tool_name!r}",
            )
        tool_name = tool_name.strip()

        tool_args = raw_tool_args
        if isinstance(tool_args, str):
            try:
                # Keep strict JSON parsing for stringified arguments to mirror training loop.
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                return (
                    None,
                    None,
                    f"[Tool Error] Invalid tool arguments JSON for {tool_name}: {raw_tool_args!r}",
                )
        elif isinstance(tool_args, list):
            if len(tool_args) == 1 and isinstance(tool_args[0], dict):
                tool_args = tool_args[0]
            else:
                return (
                    None,
                    None,
                    f"[Tool Error] Invalid tool call format for {tool_name}: arguments must be object, got {raw_tool_args!r}",
                )
        elif tool_args is None:
            tool_args = {}

        if not isinstance(tool_args, dict):
            return (
                None,
                None,
                f"[Tool Error] Invalid tool call format for {tool_name}: arguments must be object, got {raw_tool_args!r}",
            )

        # Inject extracted <code> into tool_args for PythonInterpreter
        if extracted_code and tool_name and tool_name.strip() == "PythonInterpreter":
            tool_args["code"] = extracted_code

        return tool_name, tool_args, None

    def _save_trajectory_for_distillation(self, messages: List[Dict], state: Optional[Dict] = None, 
                                         trigger_reason: str = "unknown", round_num: int = 0,
                                         is_final: bool = False,
                                         no_memory_messages: Optional[List[Dict]] = None):
        """
        Save the trajectory used for distillation training
        Save the messages actually seen by the model (before updating prev_state), rather than the updated messages
        This ensures distillation training uses the actual inputs processed by the model
        
        Args:
            messages: messages actually seen by the model (original messages before updating prev_state)
            state: newly generated state JSON (used for distillation training; can be None for the final trajectory)
            trigger_reason: trigger reason (CONTEXT_THRESHOLD, final, timeout, or token_limit)
            round_num: current round number
            is_final: whether this is the final trajectory
        """
        try:
            tl = self._get_thread_state()
            
            if tl.current_question is None:
                print("[Memory] Warning: current_question is None, skipping trajectory save")
                return
            
            if tl.task_log_dir is None:
                print("[Memory] Warning: task_log_dir is None, skipping trajectory save")
                return
            
            if not is_final:
                tl.trajectory_update_count += 1
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Build trajectory data(the version containing prev_state)
            trajectory_data = {
                "question": tl.current_question,
                "is_final": is_final,
                "condenser_update_id": tl.trajectory_update_count if not is_final else None,
                "condenser_call_id": tl.condenser_call_count if not is_final else None,
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "round": round_num,
                "trigger_reason": trigger_reason,
                "prev_state": state,  # Newly generated prev_state (used for distillation training); may be None for the final trajectory
                "messages": copy.deepcopy(messages),  # Messages actually seen by the model (original messages before updating prev_state)
                "message_count": len(messages),
                "token_count": self.count_tokens(messages) if hasattr(self, 'count_tokens') else None
            }
            
            # Save to a JSONL file(append mode for later training use)
            trajectory_file = os.path.join(tl.task_log_dir, "trajectories.jsonl")
            with open(trajectory_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trajectory_data, ensure_ascii=False) + "\n")
            
            should_save_no_memory = is_final or no_memory_messages is not None
            if should_save_no_memory:
                if no_memory_messages is not None:
                    messages_no_memory = copy.deepcopy(no_memory_messages)
                else:
                    full_msgs = tl.full_messages if tl.full_messages else messages
                    messages_no_memory = self._remove_prev_state_from_messages(full_msgs)
                trajectory_data_no_memory = {
                    "question": tl.current_question,
                    "is_final": is_final,
                    "condenser_update_id": tl.trajectory_update_count if not is_final else None,
                    "condenser_call_id": tl.condenser_call_count if not is_final else None,
                    "timestamp": timestamp,
                    "datetime": datetime.now().isoformat(),
                    "round": round_num,
                    "trigger_reason": trigger_reason,
                    "prev_state": None,  # Without the memory mechanism, prev_state is None
                    "messages": messages_no_memory,  # Full conversation history with prev_state removed
                    "message_count": len(messages_no_memory),
                    "token_count": self.count_tokens(messages_no_memory) if hasattr(self, 'count_tokens') else None
                }
                
                trajectory_file_no_memory = os.path.join(tl.task_log_dir, "trajectories_no_memory.jsonl")
                with open(trajectory_file_no_memory, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(trajectory_data_no_memory, ensure_ascii=False) + "\n")
            
            if is_final:
                print(f"[Memory] Final trajectory saved for distillation (round {round_num}, trigger: {trigger_reason})")
            else:
                print(f"[Memory] Trajectory saved for distillation (update #{tl.trajectory_update_count}, round {round_num}, trigger: {trigger_reason})")
            
        except Exception as e:
            print(f"[Memory] Warning: Failed to save trajectory: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_first_user_message_with_state(self, messages: List[Dict], state: Dict, 
                                             original_messages: List[Dict] = None,
                                             trigger_reason: str = "unknown",
                                             round_num: int = 0):
        """
        Update the first user message by appending prev_state and the prompt at the end
        and delete all later interaction messages (user and assistant), because this information has already been summarized into prev_state
        
        Args:
            messages: message list(will be modified)
            state: state JSON to add
            original_messages: original messages before compression(used for logging)
            trigger_reason: trigger reason(used for logging)
            round_num: current round number(used for logging)
        """
        # Save a snapshot of the messages before compression(used for logging)
        if original_messages is None:
            original_messages = copy.deepcopy(messages)
        
        # Find the first user message(usually messages[1], since messages[0] is the system message)
        first_user_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                first_user_idx = i
                break
        
        if first_user_idx is None:
            print("[Memory] Warning: No user message found to update")
            return
        
        # Save the raw content of the first user message (with any previously added prev_state removed)
        first_user_content = messages[first_user_idx]["content"]
        # If prev_state was added previously, remove it first
        if "RESEARCH STATE SUMMARY (prev_state)" in first_user_content:
            # Find the prev_state section and remove it
            prev_state_start = first_user_content.find("====================\nRESEARCH STATE SUMMARY")
            if prev_state_start != -1:
                first_user_content = first_user_content[:prev_state_start].rstrip()
        
        # Decide how many messages to keep based on the trigger type(before adding prev_state)
        # This lets us compute the token count after deletion first
        # CONTEXT_THRESHOLD when triggered, delete all subsequent messages(keep only the system prompt and the first user message)
        messages_to_keep = first_user_idx + 1
        removed_count = len(messages) - messages_to_keep
        if removed_count > 0:
            messages[:] = messages[:messages_to_keep]
            print(f"[Memory] {trigger_reason}: Removed {removed_count} subsequent messages (summarized in prev_state)")
        else:
            print(f"[Memory] {trigger_reason}: No messages to remove")
        
        # Compute the token count after deleting messages(before prev_state is added)
        token_count_after_deletion = self.count_tokens(messages)
        
        # Format state as a JSON string
        state_json = json.dumps(state, ensure_ascii=False, indent=2)
        
        # Add prev_state and the prompt
        state_section = f"""

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
        
        # Add the full prev_state
        messages[first_user_idx]["content"] = first_user_content + state_section
        
        # Compute the token count after adding prev_state
        token_count_with_state = self.count_tokens(messages)
        print(f"[Memory] Added prev_state to first user message (tokens: {token_count_after_deletion} -> {token_count_with_state})")
        
        # Save a snapshot of the messages after compression
        compressed_messages = copy.deepcopy(messages)
        
        # Save logs
        self._save_memory_log(original_messages, compressed_messages, state, trigger_reason, round_num)
        
        # Save the trajectory used for distillation training (messages actually seen by the model, i.e. original_messages before the update)
        # Note:This saves the messages seen by the model before calling condenser, including the old prev_state(if any)
        self._save_trajectory_for_distillation(original_messages, state, trigger_reason, round_num)

    def _discard_all_memory_context(self, messages: List[Dict],
                                   original_messages: List[Dict] = None,
                                   trigger_reason: str = "CONTEXT_THRESHOLD_DISCARD_ALL",
                                   round_num: int = 0):
        """
        Drop the current context and keep only the system prompt plus the first user message.
        Any existing prev_state is removed, so the next round restarts without injected memory.
        """
        if original_messages is None:
            original_messages = copy.deepcopy(messages)

        tl = self._get_thread_state()
        tl.memory_state = None

        first_user_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                first_user_idx = i
                break

        if first_user_idx is None:
            print("[Memory] Warning: No user message found to reset")
            return

        first_user_content = messages[first_user_idx]["content"]
        if "RESEARCH STATE SUMMARY (prev_state)" in first_user_content:
            prev_state_start = first_user_content.find("====================\nRESEARCH STATE SUMMARY")
            if prev_state_start != -1:
                first_user_content = first_user_content[:prev_state_start].rstrip()
                messages[first_user_idx]["content"] = first_user_content

        messages_to_keep = first_user_idx + 1
        removed_count = len(messages) - messages_to_keep
        if removed_count > 0:
            messages[:] = messages[:messages_to_keep]
            print(f"[Memory] {trigger_reason}: Removed {removed_count} subsequent messages and cleared prev_state")
        else:
            print(f"[Memory] {trigger_reason}: No messages to remove")

        compressed_messages = copy.deepcopy(messages)
        discard_state = {
            "strategy": "discard_all",
            "discarded": True,
            "threshold": self.base_memory_context_threshold,
        }
        self._save_memory_log(original_messages, compressed_messages, discard_state, trigger_reason, round_num)
        self._save_trajectory_for_distillation(
            original_messages,
            None,
            trigger_reason,
            round_num,
            no_memory_messages=copy.deepcopy(tl.full_messages if tl.full_messages else original_messages),
        )

    def _hide_old_tool_results(self, messages: List[Dict],
                              original_messages: List[Dict] = None,
                              trigger_reason: str = "CONTEXT_THRESHOLD_HIDE_TOOL_RESULT",
                              round_num: int = 0):
        """
        Retain the reasoning chain but prune bulky tool results.
        Keep:
        - system message(s)
        - the first user message
        - all assistant messages (think + tool calls)
        - only the most recent tool_response user message
        Drop:
        - older tool_response user messages
        """
        if original_messages is None:
            original_messages = copy.deepcopy(messages)

        first_user_idx = None
        tool_response_indices = []
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user" and first_user_idx is None:
                first_user_idx = i
            if role == "user" and isinstance(content, str) and content.startswith(OBS_START) and content.rstrip().endswith("</tool_response>"):
                tool_response_indices.append(i)

        if first_user_idx is None:
            print("[Memory] Warning: No user message found to prune")
            return

        latest_tool_response_idx = tool_response_indices[-1] if tool_response_indices else None

        first_user_msg = copy.deepcopy(messages[first_user_idx])
        first_user_content = first_user_msg.get("content", "")
        if "RESEARCH STATE SUMMARY (prev_state)" in first_user_content:
            prev_state_start = first_user_content.find("====================\nRESEARCH STATE SUMMARY")
            if prev_state_start != -1:
                first_user_msg["content"] = first_user_content[:prev_state_start].rstrip()

        pruned_messages = []
        removed_tool_results = 0
        for i, msg in enumerate(messages):
            if i == first_user_idx:
                pruned_messages.append(first_user_msg)
                continue

            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user" and isinstance(content, str) and content.startswith(OBS_START) and content.rstrip().endswith("</tool_response>"):
                if i == latest_tool_response_idx:
                    pruned_messages.append(copy.deepcopy(msg))
                else:
                    removed_tool_results += 1
                continue

            pruned_messages.append(copy.deepcopy(msg))

        old_message_count = len(messages)
        messages[:] = pruned_messages
        new_message_count = len(messages)
        print(
            f"[Memory] {trigger_reason}: removed {removed_tool_results} old tool result messages "
            f"({old_message_count} -> {new_message_count} messages)"
        )

        compressed_messages = copy.deepcopy(messages)
        hide_state = {
            "strategy": "hide_tool_result",
            "removed_tool_results": removed_tool_results,
            "kept_latest_tool_result": latest_tool_response_idx is not None,
            "threshold": self.base_memory_context_threshold,
        }
        self._save_memory_log(original_messages, compressed_messages, hide_state, trigger_reason, round_num)
        self._save_trajectory_for_distillation(
            original_messages,
            None,
            trigger_reason,
            round_num,
            no_memory_messages=compressed_messages,
        )
    
    def _call_condenser_directly(self, messages: List[Dict], round_num: int = 0) -> Optional[Dict]:
        """
        Directly call the condenser tool to update the state summary
        
        Args:
            messages: current message list (used to extract events)
            round_num: current round number(used for logging)
            
        Returns:
            updated state JSON, or None if the update fails
        """
        try:
            tl = self._get_thread_state()
            
            # Prevent duplicate calls:If processing is already in progress, return directly
            if tl.condenser_processing:
                print("[Memory] Warning: Condenser is already processing, skipping duplicate call")
                return tl.memory_state
            
            # Mark as processing
            tl.condenser_processing = True
            
            # Save a copy of the original messages(used for logging)
            original_messages = copy.deepcopy(messages)
            
            # Prepare events(extracting them from messages while excluding the system prompt)
            events = []
            for msg in messages:
                role = msg.get("role", "")
                if role in ["user", "assistant"]:
                    events.append({
                        "role": role,
                        "content": msg.get("content", "")
                    })
            
            # Call the condenser tool
            condenser_tool = TOOL_MAP.get("condenser")
            if not condenser_tool:
                print("[Memory] Warning: Condenser tool not found in TOOL_MAP")
                return None
            
            params = {
                "events": events,
                "prev_state": tl.memory_state
            }
            
            print(f"[Memory] Directly calling condenser tool (CONTEXT_THRESHOLD trigger)")
            result = condenser_tool.call(params)
            
            # Print the raw summary output
            print(f"[Memory] Condenser output (raw):")
            print("=" * 80)
            # If the output is too long, show only the first 2000 characters
            if len(result) > 2000:
                print(result[:2000])
                print(f"... [truncated, total length: {len(result)} chars]")
            else:
                print(result)
            print("=" * 80)
            
            # Parse the returned JSON state
            new_state = self._parse_memory_result(result)
            if new_state:
                tl.memory_state = new_state
                print(f"[Memory] State updated successfully from direct condenser call")
                # Print the parsed state JSON
                print(f"[Memory] Parsed state JSON:")
                print(json.dumps(new_state, ensure_ascii=False, indent=2))
                # Update the first user message, add prev_state, and save logs
                self._update_first_user_message_with_state(
                    messages, new_state, 
                    original_messages=original_messages,
                    trigger_reason="CONTEXT_THRESHOLD",
                    round_num=round_num
                )
                # Reset the processing flag
                tl.condenser_processing = False
                return new_state
            else:
                print(f"[Memory] Warning: Failed to parse state from condenser result")
                # Reset the processing flag
                tl.condenser_processing = False
                return None
                
        except Exception as e:
            print(f"[Memory] Error calling condenser tool directly: {e}")
            import traceback
            traceback.print_exc()
            # Reset the processing flag
            tl = self._get_thread_state()
            tl.condenser_processing = False
            return None

    def _strip_wrapping_quotes(self, value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            return value[1:-1].strip()
        return value

    def _load_server_endpoints(self) -> Tuple[List[str], List[int], str]:
        """Reload endpoint configuration on every call to support runtime hot-swapping.

        Priority:SERVER_ENDPOINTS_FILE file > environment variables HOSTNAME_LIST / PORTS.
        Changes to server_endpoints.conf take effect without restarting the process.
        """
        hostname_list = os.getenv('HOSTNAME_LIST', 'localhost')
        port_list = os.getenv('PORTS', '6000,6001,6002,6003')
        endpoint_source = "env"

        config_file = os.getenv('SERVER_ENDPOINTS_FILE', '').strip()
        if config_file:
            config_path = Path(config_file).expanduser()
            if config_path.is_file():
                try:
                    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
                        line = raw_line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = self._strip_wrapping_quotes(value)
                        if key == "HOSTNAME_LIST" and value:
                            hostname_list = value
                        elif key == "PORTS" and value:
                            port_list = value
                    endpoint_source = f"file:{config_path}"
                except Exception as e:
                    print(f"Warning: failed to read endpoint config {config_path}: {e}. Falling back to env values.")
            else:
                print(f"Warning: SERVER_ENDPOINTS_FILE not found: {config_path}. Falling back to env values.")

        hosts = [h.strip() for h in hostname_list.split(',') if h.strip()]
        if not hosts:
            hosts = ['localhost']

        ports: List[int] = []
        invalid_ports: List[str] = []
        for raw_port in port_list.split(','):
            raw_port = raw_port.strip()
            if not raw_port:
                continue
            try:
                ports.append(int(raw_port))
            except ValueError:
                invalid_ports.append(raw_port)

        if invalid_ports:
            print(f"Warning: invalid ports ignored: {', '.join(invalid_ports)}")

        if not ports:
            ports = [6000, 6001, 6002, 6003]

        return hosts, ports, endpoint_source

    def _select_endpoint(self, available_endpoints: List[Tuple[str, int]]) -> Tuple[str, int]:
        """Select endpoint with affinity to utilize KV cache.

        - First call: randomly select and bind an endpoint
        - Subsequent calls: reuse the bound endpoint if still available
        - If bound endpoint is removed (hot-swap): re-select from available endpoints
        """
        tl = self._get_thread_state()

        if tl.preferred_endpoint and tl.preferred_endpoint in available_endpoints:
            return tl.preferred_endpoint

        selected = random.choice(available_endpoints)
        tl.preferred_endpoint = selected
        tl.endpoint_failure_count = 0
        return selected

    def _failover_endpoint(self, current_endpoint: Tuple[str, int],
                           all_endpoints: List[Tuple[str, int]],
                           hosts: List[str], ports: List[int]) -> Optional[Tuple[str, int]]:
        """Select a new endpoint for failover.

        Priority:
        1. Try a different host first (to avoid host-level failures)
        2. If no other hosts, try a different port on the same host
        3. If no alternatives, return None
        """
        current_host, current_port = current_endpoint

        other_host_endpoints = [ep for ep in all_endpoints
                                if ep[0] != current_host and ep != current_endpoint]
        if other_host_endpoints:
            return random.choice(other_host_endpoints)

        same_host_other_ports = [ep for ep in all_endpoints
                                 if ep[0] == current_host and ep[1] != current_port]
        if same_host_other_ports:
            return random.choice(same_host_other_ports)

        return None

    def call_server(self, msgs, max_tries=10):

        openai_api_key = "EMPTY"
        FAILOVER_THRESHOLD = 2  # Switch endpoint after 2 consecutive failures

        tl = self._get_thread_state()
        base_sleep_time = 1
        for attempt in range(max_tries):
            # Reload endpoint configuration on each attempt to support runtime hot-swapping
            hosts, ports, endpoint_source = self._load_server_endpoints()
            all_endpoints = [(host, port) for host in hosts for port in ports]
            # Use endpoint affinity to utilize KV cache
            selected_host, selected_port = self._select_endpoint(all_endpoints)
            openai_api_base = f"http://{selected_host}:{selected_port}/v1"

            print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
            print(f"--- Endpoint source: {endpoint_source} (hosts={hosts}, ports={ports}) ---")
            print(f"--- Using endpoint: {selected_host}:{selected_port} (preferred={tl.preferred_endpoint}, f_count={tl.endpoint_failure_count}) ---")

            call_failed = False
            try:
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base,
                    timeout=600.0,
                )
                max_tokens = int(os.getenv('LLM_MAX_TOKENS', '10000'))
                chat_response = client.chat.completions.create(
                    model=tl.model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    logprobs=True,
                    max_tokens=max_tokens,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
                )
                content = chat_response.choices[0].message.content

                # OpenRouter provides API calling. If you want to use OpenRouter, you need to uncomment line 89 - 90.
                # reasoning_content = "<think>\n" + chat_response.choices[0].message.reasoning.strip() + "\n</think>"
                # content = reasoning_content + content                

                if content and content.strip():
                    print(f"--- Service call successful, received a valid response from {selected_host}:{selected_port} ---")
                    tl.endpoint_failure_count = 0
                    return content.strip()
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response from {selected_host}:{selected_port}.")
                    call_failed = True

            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error on {selected_host}:{selected_port}: {e}")
                call_failed = True
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error on {selected_host}:{selected_port}: {e}")
                call_failed = True

            if call_failed:
                tl.endpoint_failure_count += 1
                print(f"--- Endpoint {selected_host}:{selected_port} failure count: {tl.endpoint_failure_count}/{FAILOVER_THRESHOLD} ---")

                if tl.endpoint_failure_count >= FAILOVER_THRESHOLD:
                    new_endpoint = self._failover_endpoint(
                        tl.preferred_endpoint, all_endpoints, hosts, ports
                    )
                    if new_endpoint:
                        old_endpoint = tl.preferred_endpoint
                        tl.preferred_endpoint = new_endpoint
                        tl.endpoint_failure_count = 0
                        print(f"--- Failover: switched from {old_endpoint} to {new_endpoint} ---")
                    else:
                        print(f"--- Failover: no alternative endpoints available, continuing with {tl.preferred_endpoint} ---")
                        tl.endpoint_failure_count = 0

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30) 
                
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")
        
        return f"vllm server error!!!"

    def count_tokens(self, messages):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
        full_prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = self._tokenizer(full_prompt, return_tensors="pt")
        token_count = len(tokens["input_ids"][0])

        return token_count

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        # ========== Key fix: use thread-local storage to ensure each thread/task has independent state ==========
        # Initialize the current thread state
        self._init_thread_state()
        tl = self._thread_local
        
        tl.model = model
        # Reset all memory-related state(it is now thread-local and will not be affected by other threads)
        tl.memory_state = None
        tl.memory_context_threshold = self.base_memory_context_threshold
        tl.context_threshold_triggered = False
        tl.condenser_processing = False
        tl.condenser_call_count = 0
        tl.original_messages_snapshots = []
        tl.full_messages = []  # Reset the full message history
        tl.current_question = None
        tl.trajectory_update_count = 0
        tl.task_log_dir = None
        tl.current_filename = ""
        # ===================================================================
        
        try:
            question = data["item"]["question"]
        except Exception:
            raw_msg = data["item"]["messages"][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

        resume_messages = data.get("resume_messages")
        resume_num_rounds = int(data.get("resume_num_rounds") or 0)
        resume_enabled = isinstance(resume_messages, list) and len(resume_messages) > 0

        def attach_resume_metadata(result: Dict) -> Dict:
            if resume_enabled:
                result["resumed_from_messages"] = True
                result["resume_start_num_rounds"] = resume_num_rounds
                result["resume_previous_termination"] = data.get("resume_previous_termination")
                result["resume_previous_prediction"] = data.get("resume_previous_prediction")
            return result

        # Save the current question for trajectory persistence(thread-local)
        tl.current_question = question
        
        # Create a unified log directory for the current task
        # Prefer filename, then task_id, and finally a hash of the question
        filename = data.get("filename", "")
        task_id = data.get("task_id", None)
        
        # Save filename so it can later be added to the result(thread-local)
        tl.current_filename = filename
        
        if filename:
            # Use filename as the directory name(the .jsonl suffix has already been removed)
            task_dir_name = filename
        elif task_id:
            # Use task_id as the directory name
            task_dir_name = f"task_{task_id}"
        else:
            # If neither filename nor task_id exists, use a hash of the question
            question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_dir_name = f"task_{question_hash}_{timestamp}"
        
        # If rollout_idx exists(iteration number), create an iter{rollout_idx} subdirectory under task_dir_name
        rollout_idx = data.get("rollout_idx", None)
        if rollout_idx is not None:
            # create an iter{rollout_idx} subdirectory under task_dir_name
            iter_dir_name = f"iter{rollout_idx}"
            tl.task_log_dir = os.path.join(self.base_log_dir, task_dir_name, iter_dir_name)
        else:
            # If rollout_idx does not exist, use task_dir_name directly
            tl.task_log_dir = os.path.join(self.base_log_dir, task_dir_name)
        
        Path(tl.task_log_dir).mkdir(parents=True, exist_ok=True)
        
        if rollout_idx is not None:
            print(f"[Memory] Task log directory created: {tl.task_log_dir} (filename: {filename or 'N/A'}, task_id: {task_id or 'N/A'}, rollout_idx: {rollout_idx})")
        else:
            print(f"[Memory] Task log directory created: {tl.task_log_dir} (filename: {filename or 'N/A'}, task_id: {task_id or 'N/A'})")

        start_time = time.time()
        answer = data["item"]["answer"]
        # If the sample contains a reference solution(for example, loaded from 51_100_with_solution.jsonl), append it to the prompt
        ref_solution = data["item"].get("solution")
        # If the sample contains historical interactions (for example, loaded from *_with_solution.jsonl), append them to the prompt as well
        ref_interactions = data["item"].get("interactions")

        tl.user_prompt = question
        system_prompt = build_system_prompt(self.python_tool_enabled)
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)

        # Provide the structured solution to the model as an internal reference/checklist
        # Add extra_ref only when ref_solution or ref_interactions exists
        extra_ref = ""
        
        if ref_solution is not None or ref_interactions:
            solution_text = ""
            interactions_text = ""
            
            if ref_solution is not None:
                try:
                    solution_text = json.dumps(ref_solution, ensure_ascii=False)
                except Exception:
                    solution_text = str(ref_solution)
            
            if ref_interactions:
                try:
                    interactions_text = json.dumps(ref_interactions, ensure_ascii=False)
                except Exception:
                    interactions_text = str(ref_interactions)

            visit_off = os.environ.get("DISABLE_VISIT_TOOL", "").lower() in ("1", "true", "yes")
            if visit_off:
                tool_usage_clause = (
                    "perform your own search tool calls only (the visit tool is disabled in this run), "
                    "and base your explanation and final answer only on evidence actually obtained from tools. "
                    "You must never fabricate search queries, search results, "
                    "or intermediate reasoning steps that you did not genuinely perform, "
                )
                chain_clause = (
                    "full chain-of-thought from scratch using only information obtained "
                    "via your own search tool calls.\n\n"
                )
            else:
                tool_usage_clause = (
                    "perform your own searches and webpage visits, and base your explanation "
                    "and final answer only on evidence actually obtained from tools. "
                    "You must never fabricate search queries, search results, visit outputs, "
                    "or intermediate reasoning steps that you did not genuinely perform, "
                )
                chain_clause = (
                    "full chain-of-thought from scratch using only URLs and information obtained "
                    "via your own search and webpage visit tool calls.\n\n"
                )

            extra_ref = (
                    "\n\nYou are also given some possibly relevant solution information that you may refer to while thinking. "
                    "Use it as a helpful reference, but write your own answer from scratch and DO NOT copy or quote this solution verbatim.\n\n "
                    "You must always start your reasoning and tool usage from scratch: "
                    + tool_usage_clause +
                    "even if they would be consistent with any reference solution.\n"
                    "The reference solution is only a potential guide. Even if you cannot find "
                    "enough evidence through search or tools, you are strictly forbidden from "
                    "directly copying, paraphrasing, or reverse‑engineering the reference "
                    "solution to fabricate an answer. You must follow your own research path, "
                    "reason from the information you actually retrieved, and then give the "
                    "answer you genuinely believe is correct.\n\n"
                    f"{solution_text}\n\n"
                    "The above reference information (including any solution and interaction logs) "
                    "must NOT be directly used as evidence in your reasoning or final answer. "
                    "You should behave as if you had never seen it, and instead reconstruct your "
                    + chain_clause +
                    f"{interactions_text}\n"
                )
        
        user_prompt = tl.user_prompt + extra_ref

        if resume_enabled:
            messages = copy.deepcopy(resume_messages)
            print(
                f"[Resume] Continuing from saved messages: "
                f"message_count={len(messages)}, start_num_rounds={resume_num_rounds}, "
                f"previous_termination={data.get('resume_previous_termination')}"
            )
        else:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        # Initialize the full message history (not compressed by condenser), used for trajectory_no_memory
        tl.full_messages = copy.deepcopy(messages)
        num_llm_calls_available = max(0, MAX_LLM_CALL_PER_RUN - resume_num_rounds)
        round = resume_num_rounds
        # Note:memory state has already been reset at the start of the function, so it does not need to be reset again here
        
        while num_llm_calls_available > 0:
            # Check whether time is reached
            max_minutes = int(os.getenv('MAX_RUNTIME_MINUTES', '1440'))
            if time.time() - start_time > max_minutes * 60:
                prediction = f'No answer found after {max_minutes}min'
                termination = f'No answer found after {max_minutes}min'
                
                # Save the final trajectory(timeout case)
                if self.memory_enabled and tl.task_log_dir:
                    try:
                        self._save_trajectory_for_distillation(
                            messages, 
                            state=tl.memory_state,
                            trigger_reason="timeout",
                            round_num=round,
                            is_final=True
                        )
                    except Exception as e:
                        print(f"[Memory] Warning: Failed to save final trajectory: {e}")
                
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "num_rounds": round,
                    "url_filter_prompt_tokens": tl.url_filter_stats["prompt_tokens"],
                    "url_filter_completion_tokens": tl.url_filter_stats["completion_tokens"],
                    "url_filter_total_tokens": tl.url_filter_stats["total_tokens"],
                    "url_filter_total_cost_usd": tl.url_filter_stats["total_cost"],
                    "task_log_dir": tl.task_log_dir,  # Add the log directory path to the result
                    "filename": tl.current_filename,  # Add the filename field for resume checks
                }
                return attach_resume_metadata(result)
            round += 1
            num_llm_calls_available -= 1
            # Reset the CONTEXT_THRESHOLD trigger flag at the start of each round so it can trigger again in the new round
            tl.context_threshold_triggered = False
            content = self.call_server(messages)   #todo!!raw initial answer; may contain URLs in tool calls
            print(f'Round {round}: {content}')
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            assistant_msg = {"role": "assistant", "content": content.strip()}
            messages.append(assistant_msg)
            tl.full_messages.append(copy.deepcopy(assistant_msg))  # Sync to the full message history
            
            if '<tool_call>' in content and '</tool_call>' in content:
                matches = list(re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL))
                raw_tool_call = matches[-1].group(1).strip()
                tool_name, tool_args, tool_error = self._normalize_tool_call(raw_tool_call)

                if tool_error is not None:
                    result = tool_error
                else:
                    result = self.custom_call_tool(tool_name, tool_args)

                    # Note:memory updates triggered by agent_call have been removed
                    # Only when CONTEXT_THRESHOLD is triggered will condenser be called automatically to update memory
                    # If the agent calls the condenser tool proactively, only return the result and do not trigger a memory update
                    if tool_name == "condenser":
                        print(f"[Memory] Condenser tool called by agent, but memory update is disabled for agent_call triggers")
                        print(f"[Memory] Only CONTEXT_THRESHOLD triggers will update memory automatically")

                result = "<tool_response>\n" + str(result) + "\n</tool_response>"
                # print(result)
                user_msg = {"role": "user", "content": result}
                messages.append(user_msg)  # search results, used as user input
                tl.full_messages.append(copy.deepcopy(user_msg))  # Sync to the full message history
            
            # After completing the tool call and tool response, check whether CONTEXT_THRESHOLD should be triggered
            max_tokens = 256 * 1024
            token_count = self.count_tokens(messages)
            
            # Use the current dynamic threshold(if it has already been adjusted, use the adjusted value; otherwise use the base threshold)
            dynamic_threshold = tl.memory_context_threshold
            
            print(f"round: {round}, token count: {token_count}, dynamic threshold: {dynamic_threshold}")

            # Check whether <answer> has already appeared; if so, no further summary is needed
            has_answer = False
            if '<answer>' in content and '</answer>' in content:
                has_answer = True

            # Trigger 2: CONTEXT_THRESHOLD - When the token count reaches the dynamic threshold, call condenser directly
            # Check after completing the tool call and tool response, and use a flag to avoid repeated triggers(within the same round)
            # If <answer> has already appeared, skip the CONTEXT_THRESHOLD check
            if (not has_answer and
                token_count >= dynamic_threshold and 
                self.memory_enabled and 
                not tl.context_threshold_triggered):
                print(f"[Memory] CONTEXT_THRESHOLD trigger: token_count ({token_count}) >= dynamic threshold ({dynamic_threshold}, base: {self.base_memory_context_threshold})")
                if self.memory_strategy == "discard_all":
                    original_token_count = token_count
                    self._discard_all_memory_context(messages, round_num=round)
                    token_count_after = self.count_tokens(messages)
                    reduction = original_token_count - token_count_after
                    reduction_percent = (reduction / original_token_count * 100) if original_token_count > 0 else 0
                    print(f"[Memory] Discard-all reset completed, continuing from fresh context")
                    print(f"[Memory] Token count reduced: {original_token_count} -> {token_count_after} (reduced by {reduction} tokens, {reduction_percent:.1f}%)")
                    token_count = token_count_after
                    tl.memory_context_threshold = self.base_memory_context_threshold
                    tl.context_threshold_triggered = False
                elif self.memory_strategy == "hide_tool_result":
                    original_token_count = token_count
                    self._hide_old_tool_results(messages, round_num=round)
                    token_count_after = self.count_tokens(messages)
                    reduction = original_token_count - token_count_after
                    reduction_percent = (reduction / original_token_count * 100) if original_token_count > 0 else 0
                    print(f"[Memory] Hide-tool-result pruning completed, keeping only the latest tool result")
                    print(f"[Memory] Token count reduced: {original_token_count} -> {token_count_after} (reduced by {reduction} tokens, {reduction_percent:.1f}%)")
                    token_count = token_count_after
                    if token_count_after >= dynamic_threshold:
                        multiplier = random.uniform(1.2, 1.5)
                        old_threshold = dynamic_threshold
                        new_threshold = int(dynamic_threshold * multiplier)
                        tl.memory_context_threshold = new_threshold
                        print(f"[Memory] Token count after pruning ({token_count_after}) still >= threshold ({old_threshold})")
                        print(f"[Memory] Adjusting threshold: {old_threshold} -> {new_threshold} (multiplier: {multiplier:.3f})")
                    else:
                        tl.memory_context_threshold = self.base_memory_context_threshold
                    tl.context_threshold_triggered = False
                else:
                    new_state = self._call_condenser_directly(messages, round_num=round)
                    if new_state:
                        token_count_after = self.count_tokens(messages)
                        reduction = token_count - token_count_after
                        reduction_percent = (reduction / token_count * 100) if token_count > 0 else 0
                        print(f"[Memory] Context summarized successfully, continuing with updated state")
                        print(f"[Memory] Token count reduced: {token_count} -> {token_count_after} (reduced by {reduction} tokens, {reduction_percent:.1f}%)")
                        token_count = token_count_after

                        if token_count_after >= dynamic_threshold:
                            multiplier = random.uniform(1.2, 1.5)
                            old_threshold = dynamic_threshold
                            new_threshold = int(dynamic_threshold * multiplier)
                            tl.memory_context_threshold = new_threshold
                            print(f"[Memory] Token count after compression ({token_count_after}) still >= threshold ({old_threshold})")
                            print(f"[Memory] Adjusting threshold: {old_threshold} -> {new_threshold} (multiplier: {multiplier:.3f})")
                        else:
                            tl.memory_context_threshold = self.base_memory_context_threshold

                        tl.context_threshold_triggered = False

            
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                print("LLM call count reached the limit, forcing a final answer.")

                llm_call_limit_prompt = (
                    "You have now reached the maximum number of reasoning turns you can use. "
                    "You must stop making tool calls and, based on all the information above, "
                    "think again and provide what you consider the most likely answer in the "
                    "following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                )
                user_limit_msg = {"role": "user", "content": llm_call_limit_prompt}
                messages.append(user_limit_msg)
                tl.full_messages.append(copy.deepcopy(user_limit_msg))  # Sync to the full message history
                content = self.call_server(messages)
                final_assistant_msg = {"role": "assistant", "content": content.strip()}
                messages.append(final_assistant_msg)
                tl.full_messages.append(copy.deepcopy(final_assistant_msg))  # Sync to the full message history

                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as llm call limit reached'
                else:
                    prediction = 'answer not found.'
                    termination = 'answer not found.'

                # Save the final trajectory(LLM call-count limit case)
                if self.memory_enabled and tl.task_log_dir:
                    try:
                        self._save_trajectory_for_distillation(
                            messages,
                            state=tl.memory_state,
                            trigger_reason="llm_call_limit",
                            round_num=round,
                            is_final=True
                        )
                    except Exception as e:
                        print(f"[Memory] Warning: Failed to save final trajectory: {e}")

                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "num_rounds": round,
                    "task_log_dir": tl.task_log_dir,  # Add the log directory path to the result
                    "filename": tl.current_filename,  # Add the filename field for resume checks
                }
                return attach_resume_metadata(result)

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")

                token_limit_prompt = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                user_limit_msg = {"role": "user", "content": token_limit_prompt}
                messages.append(user_limit_msg)
                tl.full_messages.append(copy.deepcopy(user_limit_msg))  # Sync to the full message history
                content = self.call_server(messages)
                final_assistant_msg = {"role": "assistant", "content": content.strip()}
                messages.append(final_assistant_msg)
                tl.full_messages.append(copy.deepcopy(final_assistant_msg))  # Sync to the full message history
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = 'answer not found.'
                    termination = 'answer not found.'
                
                # Save the final trajectory(token-limit case)
                if self.memory_enabled and tl.task_log_dir:
                    try:
                        self._save_trajectory_for_distillation(
                            messages, 
                            state=tl.memory_state,
                            trigger_reason="token_limit",
                            round_num=round,
                            is_final=True
                        )
                    except Exception as e:
                        print(f"[Memory] Warning: Failed to save final trajectory: {e}")
                
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "num_rounds": round,
                    "task_log_dir": tl.task_log_dir,  # Add the log directory path to the result
                    "filename": tl.current_filename,  # Add the filename field for resume checks
                }
                return attach_resume_metadata(result)

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        
        # Save the final trajectory(regardless of whether an answer was found successfully)
        if self.memory_enabled and tl.task_log_dir:
            try:
                self._save_trajectory_for_distillation(
                    messages, 
                    state=tl.memory_state,
                    trigger_reason="final",
                    round_num=round,
                    is_final=True
                )
            except Exception as e:
                print(f"[Memory] Warning: Failed to save final trajectory: {e}")
        
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "num_rounds": round,
            "task_log_dir": tl.task_log_dir,  # Add the log directory path to the result
            "filename": tl.current_filename,  # Add the filename field for resume checks
        }
        return attach_resume_metadata(result)

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name not in TOOL_MAP:
            return f"[Tool Error] Unknown tool: {tool_name}. Available tools: {list(TOOL_MAP.keys())}"

        if not isinstance(tool_args, dict):
            return f"[Tool Error] Invalid tool call format for {tool_name}: arguments must be object, got {tool_args!r}"

        # Align key validation messages with training-time tools.
        if tool_name == "search":
            query = tool_args.get("query", "")
            if not query:
                return "[Tool Error] Search query cannot be empty."
        elif tool_name == "visit":
            if not tool_args.get("goal"):
                tool_args["goal"] = "Extract relevant information"
            url = tool_args.get("url", "")
            if not url:
                return "[Visit Error] URL cannot be empty."

        try:
            tool_args["params"] = tool_args
            raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
            return raw_result
        except Exception as e:
            return f"[Tool Error] {str(e)}"
