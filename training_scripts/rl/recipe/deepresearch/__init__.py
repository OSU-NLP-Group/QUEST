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
DeepResearch Recipe for verl framework.

This recipe implements the DeepResearch agent with:
- Multi-turn tool calling (search, google_scholar, PythonInterpreter, visit)
- Memory/condenser system for context compression
- Task-specific evaluation scripts
"""

from .agent_loop import DeepResearchAgentLoop
from .memory import (
    call_condenser_async,
    extract_events_from_messages,
    format_prev_state_section,
    parse_memory_result,
)
from .reward import (
    compute_score,
    compute_score_sync,
    compute_score_wrapper,
    compute_score_wrapper_sync,
)
from .tools import DeepResearchPythonTool, DeepResearchScholarTool, DeepResearchSearchTool, DeepResearchVisitTool

__all__ = [
    # Agent loop
    "DeepResearchAgentLoop",
    # Memory
    "call_condenser_async",
    "extract_events_from_messages",
    "format_prev_state_section",
    "parse_memory_result",
    # Reward
    "compute_score",
    "compute_score_sync",
    "compute_score_wrapper",
    "compute_score_wrapper_sync",
    # Tools
    "DeepResearchSearchTool",
    "DeepResearchScholarTool",
    "DeepResearchPythonTool",
    "DeepResearchVisitTool",
]
