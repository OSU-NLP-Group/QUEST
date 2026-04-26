# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import asyncio
import logging
import os

import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OneStepOffAgentLoopManager(AgentLoopManager):
    def __init__(
        self,
        config,
        worker_group=None,
        rollout_resource_pool=None,
        reward_loop_worker_handles=None,
    ):
        super().__init__(
            config=config,
            worker_group=worker_group,
            rollout_resource_pool=rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        self._session_level_reward = bool(config.actor_rollout_ref.rollout.multi_turn.get("session_level_reward", True))
        self._session_tokenizer = None
        self._prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self._response_length = config.actor_rollout_ref.rollout.response_length

    def _maybe_init_session_tokenizer(self):
        if self._session_tokenizer is not None:
            return
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
        self._session_tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

    def _maybe_expand_sessions(self, output: DataProto) -> DataProto:
        if not self._session_level_reward:
            return output
        if output.non_tensor_batch.get("session_outputs") is None:
            return output

        self._maybe_init_session_tokenizer()
        from verl.experimental.fully_async_policy.detach_utils import expand_multi_session_output

        return expand_multi_session_output(
            output,
            tokenizer=self._session_tokenizer,
            prompt_length=self._prompt_length,
            response_length=self._response_length,
            session_level_reward=self._session_level_reward,
        )

    async def generate_sequences_async(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers (async version).

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        # Use asyncio.gather with ray.get wrapped in asyncio.to_thread to avoid blocking
        import asyncio

        outputs = await asyncio.gather(
            *[
                asyncio.to_thread(ray.get, worker.generate_sequences.remote(chunk))
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        output = self._maybe_expand_sessions(output)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    async def wake_up(self):
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])

    async def sleep(self):
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_replicas])

    async def clear_kv_cache(self):
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])
