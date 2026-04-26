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
DeepResearch agent loop with partial rollout interruption/resume support.
"""

import asyncio
import copy
import logging
import os
from typing import Any, Dict, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from .deepresearch_agent_loop import AgentData, AgentState, DeepResearchAgentLoop, SessionData

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("deepresearch_async_partial_agent")
class DeepResearchAsyncPartialAgentLoop(DeepResearchAgentLoop):
    """DeepResearch agent loop adapted for fully_async partial rollout."""

    _class_initialized = False

    def __init__(self, trainer_config, server_manager, tokenizer, processor, dataset_cls, dataset_config, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, dataset_cls, dataset_config, **kwargs)
        self.enable_partial_rollout = trainer_config.config.async_training.get("partial_rollout", False)

    @rollout_trace_op
    async def run(
        self, sampling_params: Dict[str, Any], *, cancellation_event: asyncio.Event = None, **kwargs
    ) -> AgentLoopOutput:
        param_version = kwargs.get("param_version", 0)
        output: Optional[AgentLoopOutput] = kwargs.get("output", None)

        if output and output.extra_fields.get("is_cancel", False):
            agent_data, state, consecutive_invalid_count = self._restore_from_output(output)
        else:
            if output and not output.extra_fields.get("is_cancel", False):
                return output
            agent_data = await self._init_agent_data(kwargs, param_version)
            state = AgentState.PENDING
            consecutive_invalid_count = 0

        state, consecutive_invalid_count = await self._run_state_machine(
            agent_data, state, consecutive_invalid_count, sampling_params, cancellation_event
        )

        if state == AgentState.TERMINATED:
            return self._build_completed_output(agent_data, param_version)
        return self._build_cancelled_output(agent_data, state, consecutive_invalid_count)

    async def _init_agent_data(self, kwargs: Dict[str, Any], param_version: int) -> AgentData:
        messages = list(kwargs["raw_prompt"])
        metrics: Dict[str, Any] = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        trajectory_rollout_n = kwargs.get("trajectory_rollout_n")

        original_user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                original_user_content = msg.get("content", "")
                break

        agent_data = AgentData(
            messages=messages,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            original_user_content=original_user_content,
        )
        agent_data.extra_fields["param_version_start"] = param_version
        agent_data.extra_fields["param_version_end"] = param_version
        if trajectory_rollout_n is not None:
            agent_data.extra_fields["trajectory_rollout_n"] = trajectory_rollout_n
        return agent_data

    def _restore_from_output(self, output: AgentLoopOutput) -> tuple[AgentData, AgentState, int]:
        agent_data = output.extra_fields.get("agent_data", None)
        agent_state = output.extra_fields.get("agent_state", None)
        consecutive_invalid_count = int(output.extra_fields.get("consecutive_invalid_count", 0))

        if isinstance(agent_state, str):
            agent_state = AgentState(agent_state)

        if agent_data is None or agent_state is None:
            raise ValueError(
                f"Unexpected recovery payload: agent_data={type(agent_data)}, agent_state={agent_state!r}"
            )

        if not hasattr(agent_data, "extra_fields"):
            agent_data.extra_fields = {}

        return agent_data, agent_state, consecutive_invalid_count

    async def _run_state_machine(
        self,
        agent_data: AgentData,
        state: AgentState,
        consecutive_invalid_count: int,
        sampling_params: Dict[str, Any],
        cancellation_event: asyncio.Event = None,
    ) -> tuple[AgentState, int]:
        while state != AgentState.TERMINATED:
            if cancellation_event and cancellation_event.is_set():
                logger.info("Cancellation detected at state=%s", state)
                return state, consecutive_invalid_count

            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state, consecutive_invalid_count = await self._handle_generating_state_partial(
                    agent_data, sampling_params, consecutive_invalid_count
                )
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.MEMORY_TRIGGER:
                state = await self._handle_memory_trigger_state(agent_data)
            else:
                logger.error("Invalid state: %s", state)
                state = AgentState.TERMINATED

        return state, consecutive_invalid_count

    async def _handle_generating_state_partial(
        self,
        agent_data: AgentData,
        sampling_params: Dict[str, Any],
        consecutive_invalid_count: int,
    ) -> tuple[AgentState, int]:
        if not self.enable_partial_rollout:
            return await super()._handle_generating_state(agent_data, sampling_params, consecutive_invalid_count)

        if self._reached_max_turns(agent_data):
            logger.info(
                "Terminating due to max_turns=%s (user_turns=%s, assistant_turns=%s)",
                self.max_turns,
                agent_data.user_turns,
                agent_data.assistant_turns,
            )
            return AgentState.TERMINATED, consecutive_invalid_count

        budgeted_sampling_params = self._build_budgeted_sampling_params(agent_data, sampling_params)
        if budgeted_sampling_params is None:
            logger.warning(
                "Generation budget exhausted in partial rollout: prompt_ids=%d, response_tokens=%d, response_limit=%d, turn_limit=%d, model_max_len=%d",
                len(agent_data.prompt_ids),
                len(agent_data.response_mask),
                self.response_length,
                self.turn_response_length,
                self.model_max_len,
            )
            return self._budget_exhausted_state(agent_data), consecutive_invalid_count

        with simple_timer("generate_sequences", agent_data.metrics):
            response_ids, log_probs, is_cancel = await self.server_manager.generate_for_partial(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=budgeted_sampling_params,
                image_data=None,
            )

        if is_cancel:
            partial_tokens = list(response_ids) if response_ids else []
            if partial_tokens:
                agent_data.response_ids += partial_tokens
                agent_data.prompt_ids += partial_tokens
                agent_data.response_mask += [1] * len(partial_tokens)
                if log_probs:
                    agent_data.response_logprobs += list(log_probs)

            return AgentState.GENERATING, consecutive_invalid_count

        final_tokens = list(response_ids) if response_ids else []
        agent_data.assistant_turns += 1
        agent_data.response_ids += final_tokens
        agent_data.prompt_ids += final_tokens
        agent_data.response_mask += [1] * len(final_tokens)
        if log_probs:
            agent_data.response_logprobs += list(log_probs)

        cur_response = await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(final_tokens, skip_special_tokens=True))
        cur_response = self._postprocess_response(cur_response)

        if self._print_turns:
            text = (
                cur_response
                if len(cur_response) <= self._print_turns_max_chars
                else cur_response[: self._print_turns_max_chars] + "...<truncated>"
            )
            print(
                f"[TURN] req={agent_data.request_id[:8]} assistant_turn={agent_data.assistant_turns} "
                f"tokens={len(final_tokens)}\n{text}",
                flush=True,
            )

        agent_data.full_response += cur_response
        agent_data.session_response += cur_response
        agent_data.messages.append({"role": "assistant", "content": cur_response})
        agent_data.full_messages.append({"role": "assistant", "content": cur_response})

        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED, consecutive_invalid_count
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED, consecutive_invalid_count
        if self._reached_max_turns(agent_data):
            return AgentState.TERMINATED, consecutive_invalid_count

        if self._check_for_answer(cur_response):
            return AgentState.TERMINATED, consecutive_invalid_count

        tool_name, tool_args = self._parse_tool_call(cur_response)
        if tool_name is not None:
            consecutive_invalid_count = 0
            agent_data.tool_calls = [{"name": tool_name, "arguments": tool_args}]
            return AgentState.PROCESSING_TOOLS, consecutive_invalid_count

        if self._remaining_response_budget(agent_data) <= 0:
            logger.info(
                "Response budget exhausted after assistant turn in partial rollout: response_tokens=%d, response_limit=%d",
                len(agent_data.response_mask),
                self.response_length,
            )
            return self._budget_exhausted_state(agent_data), consecutive_invalid_count

        consecutive_invalid_count += 1
        if consecutive_invalid_count >= 3:
            logger.warning("Terminating: 3 consecutive invalid responses")
            return AgentState.TERMINATED, consecutive_invalid_count

        error_msg = (
            "\n[System] Invalid response. Please either:\n"
            "1) Call a tool with <tool_call>{\"name\": <function-name>, \"arguments\": <args-json-object>}</tool_call>\n"
            "2) Or provide the final answer inside <answer>...</answer> tags.\n"
        )

        env_tokens = await self._compute_env_turn_tokens(agent_data, error_msg)
        if self._remaining_response_budget(agent_data, extra_tokens=len(env_tokens)) < 0:
            logger.info(
                "Skipping retry prompt in partial rollout because it would overflow response budget: response_tokens=%d, env_tokens=%d, response_limit=%d",
                len(agent_data.response_mask),
                len(env_tokens),
                self.response_length,
            )
            return self._budget_exhausted_state(agent_data), consecutive_invalid_count

        agent_data.messages.append({"role": "user", "content": error_msg})
        agent_data.full_messages.append({"role": "user", "content": error_msg})
        self._append_env_tokens(agent_data, env_tokens)
        agent_data.full_response += error_msg
        agent_data.session_response += error_msg
        agent_data.user_turns += 1
        agent_data.request_id = uuid4().hex

        if self._reached_max_turns(agent_data):
            return AgentState.TERMINATED, consecutive_invalid_count

        return AgentState.GENERATING, consecutive_invalid_count

    def _build_completed_output(self, agent_data: AgentData, param_version: int) -> AgentLoopOutput:
        final_session_prompt_ids = list(agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_ids)])
        final_session = SessionData(
            session_id=agent_data.session_id,
            prompt_ids=final_session_prompt_ids,
            original_user_content=agent_data.original_user_content,
            memory_state=agent_data.memory_state,
        )
        final_session.response_ids = list(agent_data.response_ids)
        final_session.response_mask = list(agent_data.response_mask)
        final_session.response_logprobs = list(agent_data.response_logprobs) if agent_data.response_logprobs else []
        final_session.response_text = agent_data.session_response
        final_session.messages = copy.deepcopy(agent_data.messages)

        all_sessions = agent_data.completed_sessions + [final_session]
        session_outputs = []
        for session in all_sessions:
            session_outputs.append(
                {
                    "session_id": session.session_id,
                    "prompt_ids": session.prompt_ids,
                    "response_ids": session.response_ids[: self.response_length],
                    "response_mask": session.response_mask[: self.response_length],
                    "response_logprobs": (
                        session.response_logprobs[: self.response_length] if session.response_logprobs else None
                    ),
                    "memory_state": session.memory_state,
                    "messages": session.messages,
                }
            )

        output = AgentLoopOutput(
            prompt_ids=final_session.prompt_ids,
            response_ids=final_session.response_ids[: self.response_length],
            response_mask=final_session.response_mask[: self.response_length],
            multi_modal_data={},
            response_logprobs=(
                final_session.response_logprobs[: self.response_length] if final_session.response_logprobs else None
            ),
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update(
            {
                "session_id": final_session.session_id,
                "num_sessions": len(all_sessions),
                "session_outputs": session_outputs,
                "full_response": agent_data.full_response,
                "memory_state": final_session.memory_state,
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                "is_cancel": False,
                "param_version_start": agent_data.extra_fields.get("param_version_start", param_version),
                "param_version_end": param_version,
                "trajectory_rollout_n": agent_data.extra_fields.get("trajectory_rollout_n"),
                "agent_messages": copy.deepcopy(agent_data.messages),
                "full_messages": copy.deepcopy(agent_data.full_messages),
            }
        )
        return output

    def _build_cancelled_output(
        self, agent_data: AgentData, state: AgentState, consecutive_invalid_count: int
    ) -> AgentLoopOutput:
        return AgentLoopOutput(
            prompt_ids=[],
            response_ids=[],
            response_mask=[],
            multi_modal_data={},
            response_logprobs=None,
            num_turns=0,
            metrics=agent_data.metrics,
            extra_fields={
                "is_cancel": True,
                "agent_data": agent_data,
                "agent_state": state.value,
                "consecutive_invalid_count": consecutive_invalid_count,
                "trajectory_rollout_n": agent_data.extra_fields.get("trajectory_rollout_n"),
            },
        )
