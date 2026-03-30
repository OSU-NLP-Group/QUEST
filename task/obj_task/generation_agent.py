import json
import json5
import os
import re
import copy
import hashlib
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union
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
from generation_prompts import *
from generation_prompts import build_self_refine_rubric_prompt
from verify_rubric_trees import flatten_rubric_tree, extract_judge_sections
import time
import asyncio
import boto3
from botocore.config import Config
from litellm import completion
import litellm
from pathlib import Path
# litellm.set_verbose = True
# try:
#     litellm._turn_on_debug()
# except AttributeError:
#     import logging
#     logging.basicConfig(level=logging.DEBUG)
#     litellm.set_verbose = True

from tool_search import *
from tool_visit import *

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

TOOL_CLASS = [
    Visit(),
    Search(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

import random


def today_date():
    return date.today().strftime("%Y-%m-%d")

class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self.model = llm["model"]
        self.model_type = llm.get("model_type", "qwen_dashscope")
        self.save_traj = os.getenv("SAVE_TRAJ", "false").lower() == "true"
        self.traj_dir = os.getenv("TRAJ_DIR", "./trajectories")
        if self.save_traj:
            os.makedirs(self.traj_dir, exist_ok=True)
        self.bedrock_client = None
        self.aws_credentials_list = self._parse_aws_credentials()
        
        # Memory state management
        self.memory_state = None  # prev_state for memory tool
        self.base_memory_context_threshold = int(os.getenv('MEMORY_CONTEXT_THRESHOLD', '16000'))  # Base threshold
        self.memory_context_threshold = self.base_memory_context_threshold  # Dynamic threshold (will be adjusted based on prev_state size)
        print(f"Base memory context threshold: {self.base_memory_context_threshold}")
        self.memory_enabled = os.getenv('MEMORY_ENABLED', 'true').lower() == 'true'
        
        # Logging for memory operations
        self.base_log_dir = os.getenv('TASK_LOG_DIR', './task_logs')
        Path(self.base_log_dir).mkdir(parents=True, exist_ok=True)
        self.condenser_call_count = 0  # Track number of condenser calls
        self.original_messages_snapshots = []  # Store snapshots of original messages before compression
        self.task_log_dir = None
        self.current_question = None  # Store current question for trajectory saving
        self.trajectory_update_count = 0  # Track number of trajectory updates
        self.trajectory_timestamp = None # Added for consistent timestamp
        
    
    def _parse_aws_credentials(self):
        """Documentation omitted."""
        credentials_list = []
        aws_credentials_json = os.environ.get("DEEPRESEARCH_AWS_CREDENTIALS")
        if aws_credentials_json:
            try:
                credentials_list = json.loads(aws_credentials_json)
                if not isinstance(credentials_list, list):
                    raise ValueError("DEEPRESEARCH_AWS_CREDENTIALS must be a JSON array")
                for i, cred in enumerate(credentials_list):
                    if not isinstance(cred, dict):
                        raise ValueError(f"Credential at index {i} must be a JSON object")
                    if "access_key_id" not in cred or "secret_access_key" not in cred:
                        raise ValueError(f"Credential at index {i} must contain 'access_key_id' and 'secret_access_key'")
                print(f"[AWS Credentials] Loaded {len(credentials_list)} credential pair(s) from DEEPRESEARCH_AWS_CREDENTIALS")
                return credentials_list
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse DEEPRESEARCH_AWS_CREDENTIALS as JSON: {e}. Falling back to individual environment variables.")
            except ValueError as e:
                print(f"Warning: Invalid DEEPRESEARCH_AWS_CREDENTIALS format: {e}. Falling back to individual environment variables.")
        aws_access_key_id = os.environ.get("DEEPRESEARCH_AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("DEEPRESEARCH_AWS_SECRET_ACCESS_KEY")
        aws_region_name = os.environ.get("DEEPRESEARCH_AWS_REGION_NAME", "us-east-2")
        
        if aws_access_key_id and aws_secret_access_key:
            credentials_list = [{
                "access_key_id": aws_access_key_id,
                "secret_access_key": aws_secret_access_key,
                "region": aws_region_name
            }]
            print(f"[AWS Credentials] Loaded 1 credential pair from individual environment variables (backward compatibility)")
            return credentials_list
        print("Warning: No AWS credentials found. Bedrock calls may fail.")
        return []
    
    def _get_random_aws_credentials(self):
        """Documentation omitted."""
        if not self.aws_credentials_list:
            return None
        selected_cred = random.choice(self.aws_credentials_list)
        return {
            "access_key_id": selected_cred["access_key_id"],
            "secret_access_key": selected_cred["secret_access_key"],
            "region": selected_cred.get("region", "us-east-2")
        }

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content
    
    def call_server(self, msgs, planning_port=None, max_tries=20):
        """Documentation omitted."""
        base_sleep_time = 1
        model_name = self.model
        if self.model_type == "bedrock" and not model_name.startswith("bedrock/"):
            model_name = f"bedrock/{model_name}"
        print(f"DEBUG: Using model_name = {model_name}, model_type = {self.model_type}")
        api_base = os.environ.get("DEEPRESEARCH_API_BASE")
        if not api_base and planning_port:
            api_base = f"http://127.0.0.1:{planning_port}/v1"
        call_kwargs = {
            "model": model_name,
            "messages": msgs,
            "max_tokens": self.llm_generate_cfg.get('max_tokens', 20000),
            "temperature": self.llm_generate_cfg.get('temperature', 1),
            "stop": ["\n<tool_response>", "<tool_response>"],
            "num_retries": max_tries,
        }
        # if 'top_p' in self.llm_generate_cfg:
        #     call_kwargs["top_p"] = self.llm_generate_cfg.get('top_p', 0.95)
        if 'presence_penalty' in self.llm_generate_cfg:
            call_kwargs["presence_penalty"] = self.llm_generate_cfg.get('presence_penalty', 1.1)
        if model_name.startswith("azure/"):
            api_key = os.environ.get("DEEPRESEARCH_AZURE_API_KEY")
            api_base_azure = os.environ.get("DEEPRESEARCH_AZURE_API_BASE")
            api_version = os.environ.get("DEEPRESEARCH_AZURE_API_VERSION")
            if api_key:
                call_kwargs["api_key"] = api_key
            if api_base_azure:
                call_kwargs["api_base"] = api_base_azure
            if api_version:
                call_kwargs["api_version"] = api_version
        elif model_name.startswith("bedrock/"):
            aws_cred = self._get_random_aws_credentials()
            if aws_cred:
                call_kwargs["aws_access_key_id"] = aws_cred["access_key_id"]
                call_kwargs["aws_secret_access_key"] = aws_cred["secret_access_key"]
                call_kwargs["aws_region_name"] = aws_cred["region"]
                print(f"[AWS Bedrock] Using credentials: {aws_cred['access_key_id'][:8]}... (region: {aws_cred['region']})")
            else:
                aws_access_key_id = os.environ.get("DEEPRESEARCH_AWS_ACCESS_KEY_ID")
                aws_secret_access_key = os.environ.get("DEEPRESEARCH_AWS_SECRET_ACCESS_KEY")
                aws_region_name = os.environ.get("DEEPRESEARCH_AWS_REGION_NAME")
                if aws_access_key_id:
                    call_kwargs["aws_access_key_id"] = aws_access_key_id
                if aws_secret_access_key:
                    call_kwargs["aws_secret_access_key"] = aws_secret_access_key
                if aws_region_name:
                    call_kwargs["aws_region_name"] = aws_region_name
        elif model_name.startswith("openai/"):
            api_key = os.environ.get("DEEPRESEARCH_OPENAI_API_KEY")
            if api_key:
                call_kwargs["api_key"] = api_key
            else:
                print("WARNING: DEEPRESEARCH_OPENAI_API_KEY is not set. LiteLLM may use default OpenAI API key from environment.")
            if api_base:
                print(f"WARNING: API_BASE is set for OpenAI model. This may indicate vLLM usage. Using api_base: {api_base}")
                call_kwargs["api_base"] = api_base
        elif model_name.startswith("vllm/"):
            api_key = os.environ.get("DEEPRESEARCH_OPENAI_API_KEY", "EMPTY")
            call_kwargs["api_key"] = api_key
            if api_base:
                call_kwargs["api_base"] = api_base
            else:
                raise ValueError(f"vLLM model '{model_name}' requires DEEPRESEARCH_API_BASE environment variable to be set")
        else:
            print(f"WARNING: Model name '{model_name}' does not start with a known prefix (openai/, azure/, bedrock/, vllm/). "
                  f"Attempting to use as OpenAI model. Please use the correct format.")
            api_key = os.environ.get("DEEPRESEARCH_OPENAI_API_KEY")
            if api_key:
                call_kwargs["api_key"] = api_key
            if api_base:
                call_kwargs["api_base"] = api_base
        platform_name = "Unknown"
        if model_name.startswith("azure/"):
            platform_name = "Azure OpenAI"
        elif model_name.startswith("bedrock/"):
            platform_name = "AWS Bedrock"
        elif api_base:
            platform_name = f"Local vLLM/OpenAI-compatible ({api_base})"
        else:
            platform_name = "OpenAI"
        empty_response_count = 0
        max_empty_retries = 3
        
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call {platform_name} via LiteLLM, try {attempt + 1}/{max_tries} ---")
                response = completion(**call_kwargs)
                content = response.choices[0].message.content
                if content is None:
                    content = ""
                
                print(content)
                stop_sequences = ["\n<tool_response>", "<tool_response>"]
                for stop_seq in stop_sequences:
                    if content and stop_seq in content:
                        content = content.split(stop_seq)[0]
                
                if content and content.strip():
                    print(f"--- {platform_name} call successful via LiteLLM, received a valid response ---")
                    empty_response_count = 0
                    token_count = 0
                    if hasattr(response, 'usage') and response.usage:
                        token_count = response.usage.total_tokens
                    cost_info = {
                        "cost": 0.0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                    
                    try:
                        if hasattr(response, '_hidden_params') and response._hidden_params:
                            hidden_params = response._hidden_params
                            if 'response_cost' in hidden_params:
                                cost_info["cost"] = float(hidden_params['response_cost'])
                            if 'prompt_tokens' in hidden_params:
                                cost_info["prompt_tokens"] = int(hidden_params['prompt_tokens'])
                            if 'completion_tokens' in hidden_params:
                                cost_info["completion_tokens"] = int(hidden_params['completion_tokens'])
                            if 'total_tokens' in hidden_params:
                                cost_info["total_tokens"] = int(hidden_params['total_tokens'])
                        if cost_info["prompt_tokens"] == 0 and hasattr(response, 'usage') and response.usage:
                            if hasattr(response.usage, 'prompt_tokens'):
                                cost_info["prompt_tokens"] = response.usage.prompt_tokens
                            if hasattr(response.usage, 'completion_tokens'):
                                cost_info["completion_tokens"] = response.usage.completion_tokens
                            if hasattr(response.usage, 'total_tokens'):
                                cost_info["total_tokens"] = response.usage.total_tokens
                        if cost_info["total_tokens"] == 0 and cost_info["prompt_tokens"] > 0 and cost_info["completion_tokens"] > 0:
                            cost_info["total_tokens"] = cost_info["prompt_tokens"] + cost_info["completion_tokens"]
                            
                    except Exception as e:
                        print(f"Warning: Failed to extract cost information: {e}")
                    
                    return content.strip(), token_count, cost_info
                else:
                    empty_response_count += 1
                    print(f"Warning: Attempt {attempt + 1} received an empty response. (Empty response count: {empty_response_count}/{max_empty_retries})")
                    if empty_response_count >= max_empty_retries:
                        print(f"Error: Received {max_empty_retries} consecutive empty responses. Stopping retries.")
                        break
                    
            except Exception as e:
                if "consecutive empty responses" in str(e):
                    print(f"Error: {e}")
                    break
                
                print(f"Error: Attempt {attempt + 1} failed with error: {e}")
                import traceback
                traceback.print_exc()
            if empty_response_count >= max_empty_retries:
                break
                
            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Error: All retry attempts have been exhausted. The {platform_name} call has failed.")
        if empty_response_count >= max_empty_retries:
            raise ValueError(f"Received {max_empty_retries} consecutive empty responses from {platform_name}. Stopping all retries.")
        empty_cost_info = {
            "cost": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        return f"LLM server error!!!", 0, empty_cost_info

    def count_tokens(self, messages, last_token_count=0):
        """Documentation omitted."""
        if last_token_count > 0:
            return last_token_count
        tokenizer_path = os.getenv('MEMORY_TOKENIZER_PATH', None)
        if tokenizer_path:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                tokens = tokenizer(full_prompt, return_tensors="pt")
                token_count = len(tokens["input_ids"][0])
                return token_count
            except Exception as e:
                print(f"[Memory] Warning: Failed to count tokens with MEMORY_TOKENIZER_PATH ({tokenizer_path}): {e}")
        if self.llm_local_path and not self.llm_local_path.startswith(('bedrock/', 'azure/', 'openai/', 'vllm/')):
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path, trust_remote_code=True)
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                tokens = tokenizer(full_prompt, return_tensors="pt")
                token_count = len(tokens["input_ids"][0])
                return token_count
            except Exception as e:
                print(f"[Memory] Warning: Failed to count tokens with llm_local_path ({self.llm_local_path}): {e}")
        print(f"[Memory] Warning: Cannot count tokens - llm_local_path is '{self.llm_local_path}' (may be a remote model name). "
              f"Please set MEMORY_TOKENIZER_PATH environment variable to a local tokenizer path for accurate token counting.")
        return 0

    def _self_refine_if_needed(self, messages, planning_port=None):
        """Use a single rubric-tree self-refine step to produce a corrected rubric tree (one extra LLM call)."""
        try:
            last_content = messages[-1]["content"]
            start = last_content.index("<answer>") + len("<answer>")
            end = last_content.index("</answer>", start)
            answer_json_str = last_content[start:end].strip()
            parsed_answer = json.loads(answer_json_str)

            question = parsed_answer.get("proposed_question", "") or self.user_prompt
            constraints = parsed_answer.get("constraints", [])
            rubric_tree = parsed_answer.get("rubric_tree", {})
            def _attach_node_names(tree: Dict[str, Any]) -> Dict[str, Any]:
                def clone_node(node: Dict[str, Any], node_name: str) -> Dict[str, Any]:
                    new_node: Dict[str, Any] = dict(node)
                    new_node.setdefault("node_name", node_name)
                    children = new_node.get("children")
                    if isinstance(children, dict):
                        new_children = {}
                        for child_name, child_node in children.items():
                            if isinstance(child_node, dict):
                                new_children[child_name] = clone_node(child_node, child_name)
                            else:
                                new_children[child_name] = child_node
                        new_node["children"] = new_children
                    elif isinstance(children, list):
                        new_children_list = []
                        for idx, child_node in enumerate(children):
                            if isinstance(child_node, dict):
                                child_name = child_node.get("node_name") or child_node.get("name") or f"child_{idx}"
                                new_children_list.append(clone_node(child_node, child_name))
                            else:
                                new_children_list.append(child_node)
                        new_node["children"] = new_children_list
                    return new_node

                if not isinstance(tree, dict):
                    return {}
                return {name: clone_node(node, name) for name, node in tree.items() if isinstance(node, dict)}

            formatted_tree = _attach_node_names(rubric_tree)
            rubric_summary = flatten_rubric_tree(formatted_tree)
            refine_prompt = build_self_refine_rubric_prompt(
                question=question,
                constraints=constraints,
                rubric_summary=rubric_summary,
            )

            refine_msgs = list(messages)
            refine_msgs.append({"role": "user", "content": refine_prompt})
            try:
                revised_tree_text, _, cost_info = self.call_server(refine_msgs, planning_port)
                if hasattr(self, '_total_cost_info'):
                    self._total_cost_info["cost"] += cost_info.get("cost", 0.0)
                    self._total_cost_info["prompt_tokens"] += cost_info.get("prompt_tokens", 0)
                    self._total_cost_info["completion_tokens"] += cost_info.get("completion_tokens", 0)
                    self._total_cost_info["total_tokens"] += cost_info.get("total_tokens", 0)
            except Exception as e:  # noqa: BLE001
                print(f"[Self-Refine] Failed to get revised rubric tree: {e}")
                return messages
            tree_text = revised_tree_text.strip()
            try:
                if not tree_text.startswith("{"):
                    start = tree_text.find("{")
                    if start != -1:
                        tree_text = tree_text[start:]
                if not tree_text.endswith("}"):
                    end = tree_text.rfind("}")
                    if end != -1:
                        tree_text = tree_text[: end + 1]
                obj = json.loads(tree_text)
            except Exception as e:
                print(f"[Self-Refine] Failed to parse revised rubric JSON: {e}")
                return messages
            if isinstance(obj, dict) and "rubric_tree" in obj:
                new_rubric_tree = obj["rubric_tree"]
                final_obj = obj
            else:
                new_rubric_tree = obj
                final_obj = {"reasoning": "", "rubric_tree": new_rubric_tree}
            refined_json = json.dumps(final_obj, ensure_ascii=False, indent=2)
            messages.append({"role": "user", "content": refine_prompt})
            messages.append({"role": "assistant", "content": refined_json})
            return messages

        except Exception as e:  # noqa: BLE001
            print(f"[Self-Refine] Unexpected error: {e}")
            return messages
            
    def save_trajectory(self, result):
        """Documentation omitted."""
        if not self.save_traj:
            return
        iteration_id = result.get("iteration_id", "unknown")
        timestamp = getattr(self, 'trajectory_timestamp', None)
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        complexity_class = result.get("complexity_class", "None")
        subcategory = result.get("subcategory", "unknown")
        subcategory_clean = subcategory.replace("/", "_").replace(" ", "_")
        filename = f"traj_{iteration_id}_{timestamp}_{complexity_class}_{subcategory_clean}.json"
        filepath = os.path.join(self.traj_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Trajectory saved to: {filepath}")
        except Exception as e:
            print(f"Failed to save trajectory: {e}")
    
    def _parse_memory_result(self, result: str) -> Optional[Dict]:
        """Documentation omitted."""
        if not result or result.startswith("[Memory]"):
            return None
        
        try:
            new_state = json.loads(result)
            return new_state
        except json.JSONDecodeError:
            json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', result, re.DOTALL)
            if json_match:
                try:
                    new_state = json.loads(json_match.group(1))
                    return new_state
                except json.JSONDecodeError:
                    pass
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
        """Documentation omitted."""
        try:
            self.condenser_call_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_data = {
                "condenser_call_number": self.condenser_call_count,
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "round": round_num,
                "trigger_reason": trigger_reason,
                "state": state,
                "original_message_count": len(original_messages),
                "compressed_message_count": len(compressed_messages),
                "messages_removed": len(original_messages) - len(compressed_messages)
            }
            
            log_file = os.path.join(self.task_log_dir, f"condenser_call_{self.condenser_call_count}_{timestamp}.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            print(f"[Memory] Log saved to: {log_file}")
            
        except Exception as e:
            print(f"[Memory] Warning: Failed to save memory log: {e}")
            import traceback
            traceback.print_exc()
    
    def _remove_prev_state_from_messages(self, messages: List[Dict]) -> List[Dict]:
        """Documentation omitted."""
        cleaned_messages = copy.deepcopy(messages)
        
        for msg in cleaned_messages:
            if msg.get("role") == "user" and "content" in msg:
                content = msg["content"]
                prev_state_start = content.find("====================\nRESEARCH STATE SUMMARY")
                if prev_state_start != -1:
                    msg["content"] = content[:prev_state_start].rstrip()
        
        return cleaned_messages
    
    def _save_trajectory_for_distillation(self, messages: List[Dict], state: Optional[Dict] = None, 
                                         trigger_reason: str = "unknown", round_num: int = 0,
                                         is_final: bool = False):
        """Documentation omitted."""
        try:
            if self.current_question is None:
                print("[Memory] Warning: current_question is None, skipping trajectory save")
                return
            
            if self.task_log_dir is None:
                print("[Memory] Warning: task_log_dir is None, skipping trajectory save")
                return
            
            if not is_final:
                self.trajectory_update_count += 1
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            trajectory_data = {
                "question": self.current_question,
                "is_final": is_final,
                "condenser_update_id": self.trajectory_update_count if not is_final else None,
                "condenser_call_id": self.condenser_call_count if not is_final else None,
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "round": round_num,
                "trigger_reason": trigger_reason,
                "prev_state": state,
                "messages": copy.deepcopy(messages),
                "message_count": len(messages),
                "token_count": self.count_tokens(messages) if hasattr(self, 'count_tokens') else None
            }
            trajectory_file = os.path.join(self.task_log_dir, "trajectories.jsonl")
            with open(trajectory_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trajectory_data, ensure_ascii=False) + "\n")
            if is_final:
                messages_no_memory = self._remove_prev_state_from_messages(messages)
                trajectory_data_no_memory = {
                    "question": self.current_question,
                    "is_final": True,
                    "condenser_update_id": None,
                    "condenser_call_id": None,
                    "timestamp": timestamp,
                    "datetime": datetime.now().isoformat(),
                    "round": round_num,
                    "trigger_reason": trigger_reason,
                    "prev_state": None,
                    "messages": messages_no_memory,
                    "message_count": len(messages_no_memory),
                    "token_count": self.count_tokens(messages_no_memory) if hasattr(self, 'count_tokens') else None
                }
                
                trajectory_file_no_memory = os.path.join(self.task_log_dir, "trajectories_no_memory.jsonl")
                with open(trajectory_file_no_memory, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(trajectory_data_no_memory, ensure_ascii=False) + "\n")
            
            if is_final:
                print(f"[Memory] Final trajectory saved for distillation (round {round_num}, trigger: {trigger_reason})")
            else:
                print(f"[Memory] Trajectory saved for distillation (update #{self.trajectory_update_count}, round {round_num}, trigger: {trigger_reason})")
            
        except Exception as e:
            print(f"[Memory] Warning: Failed to save trajectory: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_first_user_message_with_state(self, messages: List[Dict], state: Dict, 
                                             original_messages: List[Dict] = None,
                                             trigger_reason: str = "unknown",
                                             round_num: int = 0):
        """Documentation omitted."""
        if original_messages is None:
            original_messages = copy.deepcopy(messages)
        first_user_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                first_user_idx = i
                break
        
        if first_user_idx is None:
            print("[Memory] Warning: No user message found to update")
            return
        first_user_content = messages[first_user_idx]["content"]
        if "RESEARCH STATE SUMMARY (prev_state)" in first_user_content:
            prev_state_start = first_user_content.find("====================\nRESEARCH STATE SUMMARY")
            if prev_state_start != -1:
                first_user_content = first_user_content[:prev_state_start].rstrip()
        messages_to_keep = first_user_idx + 1
        removed_count = len(messages) - messages_to_keep
        if removed_count > 0:
            messages[:] = messages[:messages_to_keep]
            print(f"[Memory] {trigger_reason}: Removed {removed_count} subsequent messages (summarized in prev_state)")
        else:
            print(f"[Memory] {trigger_reason}: No messages to remove")
        token_count_after_deletion = self.count_tokens(messages)
        state_json = json.dumps(state, ensure_ascii=False, indent=2)
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
        messages[first_user_idx]["content"] = first_user_content + state_section
        token_count_with_state = self.count_tokens(messages)
        print(f"[Memory] Added prev_state to first user message (tokens: {token_count_after_deletion} -> {token_count_with_state})")
        compressed_messages = copy.deepcopy(messages)
        self._save_memory_log(original_messages, compressed_messages, state, trigger_reason, round_num)
        self._save_trajectory_for_distillation(original_messages, state, trigger_reason, round_num)
    
    def _call_condenser_directly(self, messages: List[Dict], round_num: int = 0) -> Optional[Dict]:
        """Documentation omitted."""
        try:
            if getattr(self, '_condenser_processing', False):
                print("[Memory] Warning: Condenser is already processing, skipping duplicate call")
                return self.memory_state
            self._condenser_processing = True
            original_messages = copy.deepcopy(messages)
            events = []
            for msg in messages:
                role = msg.get("role", "")
                if role in ["user", "assistant"]:
                    events.append({
                        "role": role,
                        "content": msg.get("content", "")
                    })
            condenser_tool = TOOL_MAP.get("condenser")
            if not condenser_tool:
                print("[Memory] Warning: Condenser tool not found in TOOL_MAP")
                return None
            
            params = {
                "events": events,
                "prev_state": self.memory_state
            }
            
            print(f"[Memory] Directly calling condenser tool (CONTEXT_THRESHOLD trigger)")
            result = condenser_tool.call(params)
            print(f"[Memory] Condenser output (raw):")
            print("=" * 80)
            if len(result) > 2000:
                print(result[:2000])
                print(f"... [truncated, total length: {len(result)} chars]")
            else:
                print(result)
            print("=" * 80)
            new_state = self._parse_memory_result(result)
            if new_state:
                self.memory_state = new_state
                print(f"[Memory] State updated successfully from direct condenser call")
                print(f"[Memory] Parsed state JSON:")
                print(json.dumps(new_state, ensure_ascii=False, indent=2))
                self._update_first_user_message_with_state(
                    messages, new_state, 
                    original_messages=original_messages,
                    trigger_reason="CONTEXT_THRESHOLD",
                    round_num=round_num
                )
                self._condenser_processing = False
                return new_state
            else:
                print(f"[Memory] Warning: Failed to parse state from condenser result")
                self._condenser_processing = False
                return None
                
        except Exception as e:
            print(f"[Memory] Error calling condenser tool directly: {e}")
            import traceback
            traceback.print_exc()
            self._condenser_processing = False
            return None

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        start_time = time.time()
        planning_port = data.get('planning_port')
        complexity_class = data.get('complexity_class')
        sampled_keywords = data.get('sampled_keywords', [])
        iteration_id = data.get('iteration_id')
        subcategory = data.get('subcategory')
        self.complexity_class = complexity_class
        answer = data['item']['answer']
        self.user_prompt = question
        self.current_question = question
        self.trajectory_update_count = 0
        iteration_id = data.get('iteration_id', "unknown")
        complexity_class = data.get('complexity_class', "None")
        subcategory = data.get('subcategory', "unknown")
        self.trajectory_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subcategory_clean = subcategory.replace("/", "_").replace(" ", "_")
        task_dir_name = f"traj_{iteration_id}_{self.trajectory_timestamp}_{complexity_class}_{subcategory_clean}"
        if self.memory_enabled:
            self.task_log_dir = os.path.join(self.base_log_dir, task_dir_name)
            Path(self.task_log_dir).mkdir(parents=True, exist_ok=True)
            print(f"[Memory] Task log directory created: {self.task_log_dir}")
        else:
            self.task_log_dir = None
            print(f"[Memory] Memory system is disabled, task log directory will not be created")
        total_cost_info = {
            "cost": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        self._total_cost_info = total_cost_info
        system_prompt, examples_source = build_system_prompt()
        self.examples_source = examples_source
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        user_content = "Topic: " + question
        if sampled_keywords:
            keywords_str = ", ".join(sampled_keywords)
            user_content += f"\n\nInitial Keywords (10 keywords provided): {keywords_str}"
            user_content += "\n\nNote: You have been provided with 10 initial keywords above. In STEP 1 — Brainstorm Keywords, you should use these as a starting point and brainstorm 10 additional keywords yourself. The final keywords list should combine both the provided keywords and your brainstormed keywords."
        
        if complexity_class:
            complexity_instruction = f"\n\nIMPORTANT: You must generate a task with complexity class {complexity_class}. The rubric tree must strictly follow the Breadth and Depth requirements for {complexity_class}:\n"
            if complexity_class == 'C1':
                complexity_instruction += "Breadth: 1-3 nodes per layer, Depth: 2 layers (including root)"
            elif complexity_class == 'C2':
                complexity_instruction += "Breadth: 1-3 nodes per layer, Depth: 3-4 layers (including root)"
            elif complexity_class == 'C3':
                complexity_instruction += "Breadth: 1-3 nodes per layer, Depth: 5-6 layers (including root)"
            elif complexity_class == 'C4':
                complexity_instruction += "Breadth: 4-11 nodes per layer, Depth: 2 layers (including root)"
            elif complexity_class == 'C5':
                complexity_instruction += "Breadth: 4-11 nodes per layer, Depth: 3-4 layers (including root)"
            elif complexity_class == 'C6':
                complexity_instruction += "Breadth: 4-11 nodes per layer, Depth: 5-6 layers (including root)"
            elif complexity_class == 'C7':
                complexity_instruction += "Breadth: ≥12 nodes per layer, Depth: 2 layers (including root)"
            elif complexity_class == 'C8':
                complexity_instruction += "Breadth: ≥12 nodes per layer, Depth: 3-4 layers (including root)"
            elif complexity_class == 'C9':
                complexity_instruction += "Breadth: ≥12 nodes per layer, Depth: 5-6 layers (including root)"
            user_content += complexity_instruction
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        self.memory_state = None
        self.memory_context_threshold = self.base_memory_context_threshold
        self._context_threshold_triggered = False
        self._condenser_processing = False
        self.condenser_call_count = 0
        self.original_messages_snapshots = []
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = {
                    "text": 'No answer found after 2h30mins',
                    "json": 'No answer found after 2h30mins'
                }
                termination = 'No answer found after 2h30mins'
                if self.memory_enabled and self.task_log_dir:
                    try:
                        self._save_trajectory_for_distillation(
                            messages, 
                            state=self.memory_state,
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
                    "complexity_class": getattr(self, 'complexity_class', None),
                    "iteration_id": iteration_id,
                    "subcategory": subcategory,
                    "examples_source": getattr(self, 'examples_source', None),
                    "cost_info": total_cost_info.copy(),
                    "task_log_dir": self.task_log_dir,
                }
                self.save_trajectory(result)
                return result
            round += 1
            num_llm_calls_available -= 1
            self._context_threshold_triggered = False
            content, token_count, cost_info = self.call_server(messages, planning_port)
            total_cost_info["cost"] += cost_info.get("cost", 0.0)
            total_cost_info["prompt_tokens"] += cost_info.get("prompt_tokens", 0)
            total_cost_info["completion_tokens"] += cost_info.get("completion_tokens", 0)
            total_cost_info["total_tokens"] += cost_info.get("total_tokens", 0)
            print(f'Round {round}: {content}')
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
                tool_calls = re.findall(tool_call_pattern, content, re.DOTALL)
                print(tool_calls)
                tool_results = []
                if len(tool_calls) > 5:
                    print(f"Warning: Too many tool calls: {len(tool_calls)}. Requesting agent to reduce the number.")
                    feedback_message = f"<tool_response>\nError: You have generated {len(tool_calls)} tool calls, which exceeds the maximum limit of 5. Please reduce the number of tool calls to 5 or fewer and try again. Focus on the most important tool calls first.\n</tool_response>"
                    messages.append({"role": "user", "content": feedback_message})
                    continue
                
                for tool_call_raw in tool_calls:
                    tool_call_raw = tool_call_raw.strip()
                    try:
                        if "python" in tool_call_raw.lower():
                            try:
                                code_pattern = r'<code>(.*?)</code>'
                                code_match = re.search(code_pattern, tool_call_raw, re.DOTALL)
                                if code_match:
                                    code_raw = code_match.group(1).strip()
                                    result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                                    tool_result = {
                                        "tool_name": "PythonInterpreter",
                                        "query": {"code": code_raw},
                                        "result": result
                                    }
                                else:
                                    result = "[Python Interpreter Error]: No <code> tag found."
                                    tool_result = {
                                        "tool_name": "PythonInterpreter",
                                        "query": {"raw": tool_call_raw},
                                        "result": result
                                    }
                            except Exception as e:
                                result = f"[Python Interpreter Error]: {str(e)}"
                                tool_result = {
                                    "tool_name": "PythonInterpreter",
                                    "query": {"raw": tool_call_raw},
                                    "result": result
                                }
                        else:
                            cleaned_call = tool_call_raw.strip()
                            if cleaned_call.startswith('\n'):
                                cleaned_call = cleaned_call[1:]
                            if cleaned_call.count('}') > cleaned_call.count('{'):
                                brace_count = 0
                                last_valid_pos = -1
                                for i, char in enumerate(cleaned_call):
                                    if char == '{':
                                        brace_count += 1
                                    elif char == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            last_valid_pos = i
                                if last_valid_pos > 0:
                                    cleaned_call = cleaned_call[:last_valid_pos + 1]
                            
                            try:
                                tool_call = json5.loads(cleaned_call)
                            except (ValueError, json.JSONDecodeError) as parse_error:
                                try:
                                    tool_call = json.loads(cleaned_call)
                                except:
                                    raise parse_error
                            
                            tool_name = tool_call.get('name', '')
                            tool_args = tool_call.get('arguments', {})
                            query_copy = copy.deepcopy(tool_args)
                            if "params" in query_copy:
                                del query_copy["params"]
                            result = self.custom_call_tool(tool_name, tool_args)
                            tool_result = {
                                "tool_name": tool_name,
                                "query": query_copy,
                                "result": result
                            }
                        
                        tool_results.append(tool_result)
                    except (ValueError, json.JSONDecodeError) as e:
                        tool_result = {
                            "tool_name": "unknown",
                            "query": {"raw": tool_call_raw},
                            "result": f'Error: Tool call is not a valid JSON: {str(e)}'
                        }
                        tool_results.append(tool_result)
                    except Exception as e:
                        tool_result = {
                            "tool_name": "unknown",
                            "query": {"raw": tool_call_raw},
                            "result": f'Error: Failed to execute tool call: {str(e)}'
                        }
                        tool_results.append(tool_result)
                if tool_results:
                    combined_result = json.dumps(tool_results, ensure_ascii=False, indent=2)
                    combined_result = "<tool_response>\n" + combined_result + "\n</tool_response>"
                    messages.append({"role": "user", "content": combined_result})
                    for tool_result in tool_results:
                        if tool_result.get("tool_name") == "condenser":
                            print(f"[Memory] Condenser tool called by agent, but memory update is disabled for agent_call triggers")
                            print(f"[Memory] Only CONTEXT_THRESHOLD triggers will update memory automatically")
            max_tokens = 110 * 1024
            current_token_count = self.count_tokens(messages, token_count)
            dynamic_threshold = self.memory_context_threshold
            
            print(f"round: {round}, token count: {current_token_count}, dynamic threshold: {dynamic_threshold}")
            has_answer = False
            if '<answer>' in content and '</answer>' in content:
                has_answer = True
            if (not has_answer and
                current_token_count >= dynamic_threshold and 
                self.memory_enabled and 
                not getattr(self, '_context_threshold_triggered', False)):
                print(f"[Memory] CONTEXT_THRESHOLD trigger: token_count ({current_token_count}) >= dynamic threshold ({dynamic_threshold}, base: {self.base_memory_context_threshold})")
                new_state = self._call_condenser_directly(messages, round_num=round)
                if new_state:
                    token_count_after = self.count_tokens(messages)
                    reduction = current_token_count - token_count_after
                    reduction_percent = (reduction / current_token_count * 100) if current_token_count > 0 else 0
                    print(f"[Memory] Context summarized successfully, continuing with updated state")
                    print(f"[Memory] Token count reduced: {current_token_count} -> {token_count_after} (reduced by {reduction} tokens, {reduction_percent:.1f}%)")
                    current_token_count = token_count_after
                    if token_count_after >= dynamic_threshold:
                        multiplier = random.uniform(1.05, 1.1)
                        old_threshold = dynamic_threshold
                        new_threshold = int(dynamic_threshold * multiplier)
                        self.memory_context_threshold = new_threshold
                        print(f"[Memory] Token count after compression ({token_count_after}) still >= threshold ({old_threshold})")
                        print(f"[Memory] Adjusting threshold: {old_threshold} -> {new_threshold} (multiplier: {multiplier:.3f})")
                    else:
                        self.memory_context_threshold = self.base_memory_context_threshold
                    self._context_threshold_triggered = False
            
            if '<answer>' in content and '</answer>' in content:
                if '<tool_call>' not in content and '</tool_call>' not in content:
                    termination = 'answer'
                    break
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            if current_token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {current_token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                content, _, cost_info = self.call_server(messages, planning_port)
                total_cost_info["cost"] += cost_info.get("cost", 0.0)
                total_cost_info["prompt_tokens"] += cost_info.get("prompt_tokens", 0)
                total_cost_info["completion_tokens"] += cost_info.get("completion_tokens", 0)
                total_cost_info["total_tokens"] += cost_info.get("total_tokens", 0)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    answer_text = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    try:
                        answer_json = json.loads(answer_text)
                    except json.JSONDecodeError:
                        answer_json = answer_text
                    prediction = {
                        "text": messages[-1]['content'],
                        "json": answer_json
                    }
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = {
                        "text": messages[-1]['content'],
                        "json": messages[-1]['content']
                    }
                    termination = 'format error: generate an answer as token limit reached'
                if self.memory_enabled and self.task_log_dir:
                    try:
                        self._save_trajectory_for_distillation(
                            messages, 
                            state=self.memory_state,
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
                    "complexity_class": getattr(self, 'complexity_class', None),
                    "iteration_id": iteration_id,
                    "subcategory": subcategory,
                    "examples_source": getattr(self, 'examples_source', None),
                    "cost_info": total_cost_info.copy(),
                    "task_log_dir": self.task_log_dir,
                }
                self.save_trajectory(result)
                return result



        messages = self._self_refine_if_needed(messages, planning_port)

        last_content = messages[-1]['content']
        if self.memory_enabled and self.task_log_dir:
            try:
                self._save_trajectory_for_distillation(
                    messages, 
                    state=self.memory_state,
                    trigger_reason="final",
                    round_num=round,
                    is_final=True
                )
            except Exception as e:
                print(f"[Memory] Warning: Failed to save final trajectory: {e}")

        if len(messages) >= 2 and '<answer>' in messages[-2]['content']:
            try:
                answer_content = messages[-2]['content']
                start_idx = answer_content.find('<answer>')
                end_idx = answer_content.find('</answer>', start_idx)
                if start_idx != -1 and end_idx != -1:
                    answer_json_str = answer_content[start_idx + len('<answer>'):end_idx].strip()
                    prediction_json = json.loads(answer_json_str)
                else:
                    prediction_json = json.loads(answer_content)
                refined_content = messages[-1]['content'].strip()
                try:
                    refined_obj = json.loads(refined_content)
                    if isinstance(refined_obj, dict) and 'rubric_tree' in refined_obj:
                        rubric_tree = refined_obj['rubric_tree']
                    else:
                        rubric_tree = refined_obj
                except json.JSONDecodeError:
                    start = refined_content.find('{')
                    end = refined_content.rfind('}')
                    if start != -1 and end != -1:
                        refined_obj = json.loads(refined_content[start:end+1])
                        rubric_tree = refined_obj.get('rubric_tree', refined_obj)
                    else:
                        raise ValueError("Cannot parse refined rubric tree from messages[-1]")
                prediction_json['rubric_tree'] = rubric_tree
                if start_idx != -1 and end_idx != -1:
                    updated_answer_json_str = json.dumps(prediction_json, ensure_ascii=False, indent=2)
                    prediction_text = (
                        answer_content[:start_idx + len('<answer>')] + 
                        '\n' + updated_answer_json_str + '\n' + 
                        answer_content[end_idx:]
                    )
                else:
                    prediction_text = json.dumps(prediction_json, ensure_ascii=False, indent=2)
                prediction = {
                    "text": prediction_text,
                    "json": prediction_json
                }
                termination = 'answer'
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"[Error] Failed to extract prediction and rubric_tree: {e}")
                answer_content = messages[-2]['content']
                prediction = {
                    "text": answer_content,
                    "json": {}
                }
                termination = 'format error: failed to parse answer'
        else:
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
            prediction = {
                "text": messages[-1]['content'] if len(messages) > 0 else "",
                "json": {}
            }
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "complexity_class": getattr(self, 'complexity_class', None),
            "iteration_id": iteration_id,
            "subcategory": subcategory,
            "examples_source": getattr(self, 'examples_source', None),
            "cost_info": total_cost_info.copy(),
            "task_log_dir": self.task_log_dir,
        }
        self.save_trajectory(result)
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            if "python" in tool_name.lower():
                result = TOOL_MAP['PythonInterpreter'].call(tool_args)
            elif tool_name == "parse_file":
                params = {"files": tool_args["files"]}
                
                raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
                result = raw_result

                if not isinstance(raw_result, str):
                    result = str(raw_result)
            else:
                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
                result = raw_result
            return result

        else:
            return f"Error: Tool {tool_name} not found"
