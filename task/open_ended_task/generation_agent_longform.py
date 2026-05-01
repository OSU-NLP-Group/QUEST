import json
import json5
import os
import re
import copy
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer 
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from generation_prompt_longform import *
import time
import asyncio
import boto3
from botocore.config import Config
from litellm import completion
import litellm

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
import datetime


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self.model = llm["model"]  # Add self.model for call_server method
        self.model_type = llm.get("model_type", "qwen_dashscope")
        
        # Auto-save trajectory settings
        self.save_traj = os.getenv("SAVE_TRAJ", "false").lower() == "true"
        self.traj_dir = os.getenv("TRAJ_DIR", "./trajectories")
        if self.save_traj:
            os.makedirs(self.traj_dir, exist_ok=True)
        
        # Use LiteLLM for unified management, no need to initialize Bedrock client separately
        # LiteLLM automatically reads AWS credentials from environment variables
        self.bedrock_client = None

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content
    
    def call_server(self, msgs, planning_port=None, max_tries=5):
        """Use LiteLLM to uniformly call all platforms (OpenAI, Azure OpenAI, Bedrock, local vLLM), returns (content, token_count)"""
        base_sleep_time = 1
        
        # Get model name
        model_name = self.model
        
        # If using old bedrock model_type, ensure model name uses litellm format
        if self.model_type == "bedrock" and not model_name.startswith("bedrock/"):
            model_name = f"bedrock/{model_name}"
        
        # Debug info: print model name
        print(f"DEBUG: Using model_name = {model_name}, model_type = {self.model_type}")
        
        # Check if there's a custom base_url (for local vLLM or other OpenAI-compatible services)
        # Prioritize DEEPRESEARCH_API_BASE, if not available check planning_port (for backward compatibility)
        api_base = os.environ.get("DEEPRESEARCH_API_BASE")
        if not api_base and planning_port:
            api_base = f"http://127.0.0.1:{planning_port}/v1"
        
        # Prepare litellm call parameters, use DEEPRESEARCH_* environment variables (independent from Summary Model)
        call_kwargs = {
            "model": model_name,
            "messages": msgs,
            "max_tokens": self.llm_generate_cfg.get('max_tokens', 20000),
            "temperature": self.llm_generate_cfg.get('temperature', 1),
            "stop": ["\n<tool_response>", "<tool_response>"],
            "num_retries": max_tries,
        }
        
        # If top_p is configured, add it as well
        # if 'top_p' in self.llm_generate_cfg:
        #     call_kwargs["top_p"] = self.llm_generate_cfg.get('top_p', 0.95)
        
        # If presence_penalty is configured, add it as well (supported by some platforms)
        if 'presence_penalty' in self.llm_generate_cfg:
            call_kwargs["presence_penalty"] = self.llm_generate_cfg.get('presence_penalty', 1.1)
        
        # Add corresponding API configuration based on model type and configuration
        if model_name.startswith("azure/"):
            # Azure OpenAI configuration
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
            # AWS Bedrock configuration
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
            # OpenAI configuration
            api_key = os.environ.get("DEEPRESEARCH_OPENAI_API_KEY")
            if api_key:
                call_kwargs["api_key"] = api_key
            else:
                print("WARNING: DEEPRESEARCH_OPENAI_API_KEY is not set. LiteLLM may use default OpenAI API key from environment.")
            # OpenAI should not set api_base (unless using custom endpoint)
            if api_base:
                print(f"WARNING: API_BASE is set for OpenAI model. This may indicate vLLM usage. Using api_base: {api_base}")
                call_kwargs["api_base"] = api_base
        elif model_name.startswith("vllm/"):
            # Local vLLM (OpenAI compatible format) configuration
            api_key = os.environ.get("DEEPRESEARCH_OPENAI_API_KEY", "EMPTY")
            call_kwargs["api_key"] = api_key
            # vLLM must set api_base
            if api_base:
                call_kwargs["api_base"] = api_base
            else:
                raise ValueError(f"vLLM model '{model_name}' requires DEEPRESEARCH_API_BASE environment variable to be set")
        else:
            # Unknown format, try to handle as OpenAI (for backward compatibility)
            print(f"WARNING: Model name '{model_name}' does not start with a known prefix (openai/, azure/, bedrock/, vllm/). "
                  f"Attempting to use as OpenAI model. Please use the correct format.")
            api_key = os.environ.get("DEEPRESEARCH_OPENAI_API_KEY")
            if api_key:
                call_kwargs["api_key"] = api_key
            if api_base:
                call_kwargs["api_base"] = api_base
        
        # Determine platform name (for log output)
        platform_name = "Unknown"
        if model_name.startswith("azure/"):
            platform_name = "Azure OpenAI"
        elif model_name.startswith("bedrock/"):
            platform_name = "AWS Bedrock"
        elif api_base:
            platform_name = f"Local vLLM/OpenAI-compatible ({api_base})"
        else:
            platform_name = "OpenAI"
        
        # Use litellm to uniformly call all platforms
        empty_response_count = 0  # Track consecutive empty response count
        max_empty_retries = 3  # Maximum allowed consecutive empty responses
        
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call {platform_name} via LiteLLM, try {attempt + 1}/{max_tries} ---")
                
                # Use litellm unified call interface
                response = completion(**call_kwargs)
                
                # Extract response content
                content = response.choices[0].message.content
                
                # Handle case where content is None
                if content is None:
                    content = ""
                
                print(content)
                
                # Extra processing for stop sequences (in case API doesn't handle correctly)
                stop_sequences = ["\n<tool_response>", "<tool_response>"]
                for stop_seq in stop_sequences:
                    if content and stop_seq in content:
                        content = content.split(stop_seq)[0]
                
                if content and content.strip():
                    print(f"--- {platform_name} call successful via LiteLLM, received a valid response ---")
                    # Reset empty response counter (since valid response received)
                    empty_response_count = 0
                    
                    # Get token count (if available)
                    token_count = 0
                    if hasattr(response, 'usage') and response.usage:
                        token_count = response.usage.total_tokens
                    
                    # Extract cost information
                    cost_info = {
                        "cost": 0.0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                    
                    try:
                        # LiteLLM stores cost information in response._hidden_params
                        if hasattr(response, '_hidden_params') and response._hidden_params:
                            hidden_params = response._hidden_params
                            # Get cost (USD)
                            if 'response_cost' in hidden_params:
                                cost_info["cost"] = float(hidden_params['response_cost'])
                            # Get token information
                            if 'prompt_tokens' in hidden_params:
                                cost_info["prompt_tokens"] = int(hidden_params['prompt_tokens'])
                            if 'completion_tokens' in hidden_params:
                                cost_info["completion_tokens"] = int(hidden_params['completion_tokens'])
                            if 'total_tokens' in hidden_params:
                                cost_info["total_tokens"] = int(hidden_params['total_tokens'])
                        
                        # If token info not in _hidden_params, try to get from response.usage
                        if cost_info["prompt_tokens"] == 0 and hasattr(response, 'usage') and response.usage:
                            if hasattr(response.usage, 'prompt_tokens'):
                                cost_info["prompt_tokens"] = response.usage.prompt_tokens
                            if hasattr(response.usage, 'completion_tokens'):
                                cost_info["completion_tokens"] = response.usage.completion_tokens
                            if hasattr(response.usage, 'total_tokens'):
                                cost_info["total_tokens"] = response.usage.total_tokens
                        
                        # If total_tokens is 0, try to calculate
                        if cost_info["total_tokens"] == 0 and cost_info["prompt_tokens"] > 0 and cost_info["completion_tokens"] > 0:
                            cost_info["total_tokens"] = cost_info["prompt_tokens"] + cost_info["completion_tokens"]
                            
                    except Exception as e:
                        print(f"Warning: Failed to extract cost information: {e}")
                    
                    return content.strip(), token_count, cost_info
                else:
                    empty_response_count += 1
                    print(f"Warning: Attempt {attempt + 1} received an empty response. (Empty response count: {empty_response_count}/{max_empty_retries})")
                    
                    # If consecutive empty responses reach limit, exit early
                    if empty_response_count >= max_empty_retries:
                        print(f"Error: Received {max_empty_retries} consecutive empty responses. Stopping retries.")
                        # Break directly to exit loop, no more retries
                        break
                    
            except Exception as e:
                # Check if exception is caused by empty response (shouldn't happen, but for safety)
                if "consecutive empty responses" in str(e):
                    print(f"Error: {e}")
                    break  # If empty response exception, exit directly
                
                print(f"Error: Attempt {attempt + 1} failed with error: {e}")
                import traceback
                traceback.print_exc()
            
            # If already exited due to empty response, don't continue retrying
            if empty_response_count >= max_empty_retries:
                break
                
            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Error: All retry attempts have been exhausted. The {platform_name} call has failed.")
        
        # Check if exited due to consecutive empty responses
        if empty_response_count >= max_empty_retries:
            raise ValueError(f"Received {max_empty_retries} consecutive empty responses from {platform_name}. Stopping all retries.")
        
        # Return empty cost_info when returning error
        empty_cost_info = {
            "cost": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        return f"LLM server error!!!", 0, empty_cost_info

    def count_tokens(self, messages, last_token_count=0):
        """Use API returned token count, if not available fallback to local calculation"""
        if last_token_count > 0:
            return last_token_count
        
        # Fallback to local calculation
        tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path) 
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(full_prompt, return_tensors="pt")
        token_count = len(tokens["input_ids"][0])
        
        return token_count
            
    def save_trajectory(self, result):
        """Save trajectory to file"""
        if not self.save_traj:
            return
        
        # Generate filename: traj_{iteration_id}_{timestamp}_{complexity_class}_{subcategory}.json
        iteration_id = result.get("iteration_id", "unknown")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        complexity_class = result.get("complexity_class", "None")
        subcategory = result.get("subcategory", "unknown")
        # Clean subcategory name, replace characters unsuitable for filenames (preserve & symbol)
        subcategory_clean = subcategory.replace("/", "_").replace(" ", "_")
        filename = f"traj_{iteration_id}_{timestamp}_{complexity_class}_{subcategory_clean}.json"
        filepath = os.path.join(self.traj_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Trajectory saved to: {filepath}")
        except Exception as e:
            print(f"Failed to save trajectory: {e}")

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        start_time = time.time()
        planning_port = data.get('planning_port')  # May be None for Bedrock
        complexity_class = data.get('complexity_class')  # Get complexity class
        sampled_keywords = data.get('sampled_keywords', [])  # Get sampled keywords
        iteration_id = data.get('iteration_id')  # Get iteration ID
        subcategory = data.get('subcategory')  # Get subcategory name
        self.complexity_class = complexity_class  # Save to instance variable for later trajectory saving
        answer = data['item']['answer']
        self.user_prompt = question

        # Initialize cumulative cost info
        total_cost_info = {
            "cost": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        self._total_cost_info = total_cost_info  # Save to instance variable
        # Reconstruct system prompt each run to randomly select an examples source
        system_prompt = build_system_prompt()
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        
        # Build user_content
        user_content = "Topic: " + question
        
        # If sampled keywords exist, add to user_content
        if sampled_keywords:
            keywords_str = ", ".join(sampled_keywords)
            user_content += f"\n\nInitial Keyword: {keywords_str}"
            user_content += "\n\nNote: You have been provided with 1 initial keyword above. In STEP 1 — Brainstorm Topic, you should use these as a starting point and brainstorm a research topic that needs multi-step reasoning, cross-document synthesis, and the generation of evidence-backed, long-form answers yourself."
        
        if complexity_class:
            complexity_instruction = f"\n\nIMPORTANT: You must generate a task with complexity class {complexity_class}.\n"
            user_content += complexity_instruction
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = {
                    "text": 'No answer found after 2h30mins',
                    "json": 'No answer found after 2h30mins'
                }
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "complexity_class": getattr(self, 'complexity_class', None),
                    "iteration_id": iteration_id,
                    "subcategory": subcategory,
                    "cost_info": total_cost_info.copy(),  # Record cumulative cost info
                }
                self.save_trajectory(result)
                return result
            round += 1
            num_llm_calls_available -= 1
            content, token_count, cost_info = self.call_server(messages, planning_port)
            # Accumulate cost info
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
                # Extract all tool_call blocks
                tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
                tool_calls = re.findall(tool_call_pattern, content, re.DOTALL)
                print(tool_calls)
                tool_results = []
                if len(tool_calls) > 5:
                    print(f"Warning: Too many tool calls: {len(tool_calls)}. Requesting agent to reduce the number.")
                    # Send feedback to agent, requesting to reduce tool call count
                    feedback_message = f"<tool_response>\nError: You have generated {len(tool_calls)} tool calls, which exceeds the maximum limit of 5. Please reduce the number of tool calls to 5 or fewer and try again. Focus on the most important tool calls first.\n</tool_response>"
                    messages.append({"role": "user", "content": feedback_message})
                    # Skip executing tool calls, continue loop to let agent regenerate
                    continue
                
                for tool_call_raw in tool_calls:
                    tool_call_raw = tool_call_raw.strip()
                    try:
                        if "python" in tool_call_raw.lower():
                            # Handle Python code call
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
                            # Handle JSON format tool call
                            # Clean up possible format issues: remove extra closing brackets, quotes, etc.
                            cleaned_call = tool_call_raw.strip()
                            # Try to fix common format errors
                            # If starts with newline, remove it
                            if cleaned_call.startswith('\n'):
                                cleaned_call = cleaned_call[1:]
                            # If there are extra closing brackets at the end, try to fix
                            if cleaned_call.count('}') > cleaned_call.count('{'):
                                # Find the last complete JSON object
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
                                # If still parsing fails, try using standard json library
                                try:
                                    tool_call = json.loads(cleaned_call)
                                except:
                                    raise parse_error
                            
                            tool_name = tool_call.get('name', '')
                            tool_args = tool_call.get('arguments', {})
                            # Create deep copy of tool_args for saving query, avoid circular reference
                            query_copy = copy.deepcopy(tool_args)
                            # Remove possible circular reference fields
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
                        # json5.loads throws ValueError when parsing fails, not JSONDecodeError
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
                
                # Assemble all results into JSON format
                if tool_results:
                    combined_result = json.dumps(tool_results, ensure_ascii=False, indent=2)
                    combined_result = "<tool_response>\n" + combined_result + "\n</tool_response>"
                    messages.append({"role": "user", "content": combined_result})
            if '<answer>' in content and '</answer>' in content:
                # Ensure content no longer has tool calls before terminating
                if '<tool_call>' not in content and '</tool_call>' not in content:
                    termination = 'answer'
                    break
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = 110 * 1024
            # Use API returned token count, if not available fallback to local calculation
            current_token_count = self.count_tokens(messages, token_count)
            print(f"round: {round}, token count: {current_token_count}")

            if current_token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {current_token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                content, _, cost_info = self.call_server(messages, planning_port)
                # Accumulate cost info
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
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "complexity_class": getattr(self, 'complexity_class', None),
                    "iteration_id": iteration_id,
                    "subcategory": subcategory
                }
                self.save_trajectory(result)
                return result

        content = messages[-1]['content']
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
        else:
            prediction = {
                "text": messages[-1]['content'],
                "json": messages[-1]['content']
            }
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "complexity_class": "_".join(getattr(self, 'complexity_class', None).values()),
            "iteration_id": iteration_id,
            "subcategory": subcategory,
            "cost_info": total_cost_info.copy(),  # Record cumulative cost info
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
