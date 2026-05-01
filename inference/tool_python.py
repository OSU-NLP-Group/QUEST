import re
from typing import Any, Dict, List, Optional, Union
import json5
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import extract_code
from sandbox_fusion import run_code, RunCodeRequest, RunStatus
from requests.exceptions import Timeout
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

CHINESE_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')
DEFAULT_PYTHON_NODES_CONF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "python_nodes.conf",
)


def has_chinese_chars(data: Any) -> bool:
    text = f'{data}'
    return bool(CHINESE_CHAR_RE.search(text))

def _normalize_csv_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.strip()


def _parse_python_nodes_conf(config_path: str) -> List[str]:
    endpoints: List[str] = []
    hosts: List[str] = []
    ports: List[str] = []

    if not config_path or not os.path.exists(config_path):
        return endpoints

    with open(config_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = _normalize_csv_value(value)
            if not value:
                continue

            if key in {"endpoint", "url"}:
                endpoints.append(value)
            elif key in {"endpoints", "urls"}:
                endpoints.extend(
                    item.strip() for item in value.split(",") if item.strip()
                )
            elif key == "hosts":
                hosts = [item.strip() for item in value.split(",") if item.strip()]
            elif key == "ports":
                ports = [item.strip() for item in value.split(",") if item.strip()]

    if hosts and ports:
        for host in hosts:
            for port in ports:
                endpoints.append(f"http://{host}:{port}")

    deduped: List[str] = []
    seen = set()
    for endpoint in endpoints:
        if endpoint not in seen:
            seen.add(endpoint)
            deduped.append(endpoint)
    return deduped


def _get_sandbox_fusion_endpoints() -> List[str]:
    config_path = os.getenv("PYTHON_NODES_CONF", DEFAULT_PYTHON_NODES_CONF)
    endpoints = _parse_python_nodes_conf(config_path)
    if endpoints:
        return endpoints

    env_endpoints = os.getenv("SANDBOX_FUSION_ENDPOINTS", "")
    if env_endpoints.strip():
        return [item.strip() for item in env_endpoints.split(",") if item.strip()]

    env_endpoint = os.getenv("SANDBOX_FUSION_ENDPOINT", "")
    if env_endpoint.strip():
        return [item.strip() for item in env_endpoint.split(",") if item.strip()]

    return []


@register_tool('PythonInterpreter', allow_overwrite=True)
class PythonInterpreter(BaseToolWithFileAccess):
    name = "PythonInterpreter"
    description = 'Execute Python code in a sandboxed environment. Use this to run Python code and get the execution results.\n**Make sure to use print() for any output you want to see in the results.**\nFor code parameters, use placeholders first, and then put the code within <code></code> XML tags, such as:\n<tool_call>\n{"purpose": <detailed-purpose-of-this-tool-call>, "name": <tool-name>, "arguments": {"code": ""}}\n<code>\nHere is the code.\n</code>\n</tool_call>\n'

    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Must be provided within <code></code> XML tags. Remember to use print() statements for any output you want to see.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        # self.summary_mapping = SummaryMapping()

    @property
    def args_format(self) -> str:
        fmt = self.cfg.get('args_format')
        if fmt is None:
            if has_chinese_chars([self.name_for_human, self.name, self.description, self.parameters]):
                fmt = 'The input for this tool should be a Markdown code block.'

            else:
                fmt = 'Enclose the code within triple backticks (`) at the beginning and end of the code.'
        return fmt

    def observation(self, tool: dict, tool_dict: dict, tool_results, empty_mode: bool=False, readpage: bool=False, max_observation_length: int=None, tokenizer=None):
        print('test')
        assert isinstance(tool_results, str), f"result of python code should be str, instead of {type(tool_results)}. {tool_results}"
        return tool_results

    @property
    def function(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        }

    def call(self, params, files= None, timeout = 50, **kwargs) -> str:
        try:
            endpoints = _get_sandbox_fusion_endpoints()
            if not endpoints:
                return "[Python Interpreter Error]: No sandbox endpoints configured."

            if isinstance(params, dict):
                code = params.get('code', '') or params.get('raw', '')
            elif isinstance(params, str):
                code = params
                try:
                    parsed = json5.loads(params)
                    if isinstance(parsed, dict):
                        code = parsed.get('code', '') or parsed.get('raw', '') or code
                except Exception:
                    pass
            else:
                code = extract_code(params)

            triple_match = re.search(r'```[^\n]*\n(.+?)```', code, re.DOTALL)
            if triple_match:
                code = triple_match.group(1)

            if not code or not code.strip():
                return '[Python Interpreter Error]: Empty code.'

            last_error = None
            for attempt in range(8):
                try:
                    # Randomly sample an endpoint for each attempt
                    endpoint = random.choice(endpoints)
                    print(f"Attempt {attempt + 1}/5 using endpoint: {endpoint}")

                    code_result = run_code(RunCodeRequest(code=code, language='python', run_timeout=timeout), max_attempts=1, client_timeout=timeout, endpoint=endpoint)
                    print("[Python] Code Result", code_result)
                    result = []
                    if code_result.run_result.stdout:
                        result.append(f"stdout:\n{code_result.run_result.stdout}")
                    if code_result.run_result.stderr:
                        result.append(f"stderr:\n{code_result.run_result.stderr}")
                    if code_result.run_result.execution_time >= timeout-1:
                        result.append(f"[PythonInterpreter Error] TimeoutError: Execution timed out.")
                    result = '\n'.join(result)
                    print('SUCCESS RUNNING TOOL')
                    return result if result.strip() else 'Finished execution.'

                except Timeout as e:
                    last_error = f'[Python Interpreter Error] TimeoutError: Execution timed out on endpoint {endpoint}.'
                    print(f"Timeout on attempt {attempt + 1}: {last_error}")
                    if attempt == 4:  # Last attempt
                        return last_error
                    continue

                except Exception as e:
                    last_error = f'[Python Interpreter Error]: {str(e)} on endpoint {endpoint}'
                    print(f"Error on attempt {attempt + 1}: {last_error}")
                    if attempt == 4:  # Last attempt
                        return last_error
                    continue

            return last_error if last_error else '[Python Interpreter Error]: All attempts failed.'

        except Exception as e:
            return f"[Python Interpreter Error]: {str(e)}"

    def call_specific_endpoint(self, params: Union[str, dict], endpoint: str, timeout: Optional[int] = 30, **kwargs) -> tuple:
        """Test a specific endpoint directly"""
        try:
            if type(params) is str:
                params = json5.loads(params)
            code = params.get('code', '')
            if not code:
                code = params.get('raw', '')
            triple_match = re.search(r'```[^\n]*\n(.+?)```', code, re.DOTALL)
            if triple_match:
                code = triple_match.group(1)
        except Exception:
            code = extract_code(params)

        if not code.strip():
            return False, '[Python Interpreter Error]: Empty code.'

        try:
            start_time = time.time()
            code_result = run_code(RunCodeRequest(code=code, language='python', run_timeout=timeout),
                                 max_attempts=1, client_timeout=timeout, endpoint=endpoint)
            end_time = time.time()

            result = []
            if code_result.run_result.stdout:
                result.append(f"stdout:\n{code_result.run_result.stdout}")
            if code_result.run_result.stderr:
                result.append(f"stderr:\n{code_result.run_result.stderr}")

            result = '\n'.join(result)
            execution_time = end_time - start_time
            return True, result if result.strip() else 'Finished execution.', execution_time

        except Timeout as e:
            return False, f'[Python Interpreter Error] TimeoutError: Execution timed out.', None
        except Exception as e:
            return False, f'[Python Interpreter Error]: {str(e)}', None
