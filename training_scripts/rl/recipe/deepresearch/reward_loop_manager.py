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
DeepResearch reward-loop manager for experimental reward loop workers.

This class extends the experimental NaiveRewardManager and enriches `extra_info`
with DeepResearch-specific fields (e.g., full_response/session metadata), so the
custom async reward function can compute rollout-level rewards consistently.
"""

import asyncio
import copy
import logging
import math
import os
from numbers import Real

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _is_openended_type(value) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"openended", "open-ended"}


def _is_openended_eval_type(value) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized.startswith(("openended", "open-ended"))


@register("deepresearch")
class DeepResearchRewardLoopManager(NaiveRewardManager):
    """Reward loop manager with DeepResearch-specific extra_info propagation."""

    @staticmethod
    def _coerce_metric_scalar(value) -> float | None:
        """Convert supported scalar metric values to finite float."""
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, Real):
            val = float(value)
            if math.isfinite(val):
                return val
        return None

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        # Avoid mutating original fields in-place.
        raw_extra_info = data_item.non_tensor_batch.get("extra_info", {})
        extra_info = copy.deepcopy(raw_extra_info) if isinstance(raw_extra_info, dict) else {}

        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None and hasattr(tool_extra_fields, "items"):
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        # Preserve DeepResearch session-level metadata if present.
        for key in (
            "full_response",
            "session_id",
            "num_sessions",
            "rollout_index",
            "memory_state",
            "tool_rewards",
        ):
            if key in data_item.non_tensor_batch and data_item.non_tensor_batch.get(key) is not None:
                extra_info[key] = data_item.non_tensor_batch.get(key)

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        # Per-response eval timeout.  The clock starts fresh on every run_single()
        # invocation.  In async_reward_cancel_mode=exit the reward is re-submitted as a
        # new run_single() call, so the timer resets naturally.  In save_state mode the
        # in-flight Ray future is re-awaited by the agent loop, but run_single() itself
        # still runs only once; restarting the trainer from a checkpoint causes a fresh
        # run_single() call, which also resets the timer.
        _eval_timeout = float(os.environ.get("EVAL_PER_RESPONSE_TIMEOUT_SECONDS", "3600"))

        if self.is_async_reward_score:
            _coro = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            _coro = self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        _eval_timed_out = False
        try:
            result = await asyncio.wait_for(_coro, timeout=_eval_timeout)
        except asyncio.TimeoutError:
            _task_id = extra_info.get("task_id", "unknown")
            logger.warning(
                "Eval per-response timeout (%.0fs) for task_id=%s data_source=%s. Returning score=0.",
                _eval_timeout,
                _task_id,
                data_source,
            )
            result = {"score": 0.0, "acc": 0.0, "eval_timeout": True}
            _eval_timed_out = True

        reward_extra_info = {}
        requires_google_maps = 1.0 if bool(extra_info.get("requires_google_maps", False)) else 0.0
        missing_google_maps_api_key = (
            1.0 if requires_google_maps > 0.0 and not os.environ.get("GOOGLE_MAPS_API_KEY") else 0.0
        )
        gt_type = ""
        if isinstance(ground_truth, dict):
            gt_type = str(ground_truth.get("type", "")).strip().lower()
        if isinstance(result, dict):
            # Keep metrics numeric and stable for downstream aggregation.
            # `acc` is required by algorithm.filter_groups.metric=acc.
            reward = self._coerce_metric_scalar(result.get("score"))
            if reward is None:
                reward = self._coerce_metric_scalar(result.get("final_score"))
            if reward is None:
                reward = 0.0
            acc = self._coerce_metric_scalar(result.get("acc"))
            if acc is None:
                acc = reward
            base_score = self._coerce_metric_scalar(result.get("score_without_citation"))
            if base_score is None:
                base_score = self._coerce_metric_scalar(result.get("base_score"))
            if base_score is None:
                base_score = reward
            citation_score = self._coerce_metric_scalar(result.get("citation_score_added"))
            if citation_score is None:
                citation_score = 0.0
            citation_score_raw = self._coerce_metric_scalar(result.get("citation_score_raw"))
            if citation_score_raw is None:
                citation_score_raw = citation_score
            citation_applied = 1.0 if bool(result.get("citation_score_applied", False)) else 0.0
            drop_from_training = bool(result.get("drop_from_training", False))
            eval_script_load_failed = bool(result.get("eval_script_load_failed", False))
            eval_script_runtime_failed = bool(result.get("eval_script_runtime_failed", False))
            eval_type = str(result.get("eval_type", "")).strip().lower()
            is_openended = 1.0 if (_is_openended_type(gt_type) or _is_openended_eval_type(eval_type)) else 0.0
            is_obj = 1.0 - is_openended
            reward_extra_info["score"] = reward
            reward_extra_info["acc"] = acc
            reward_extra_info["score_obj"] = base_score if is_obj > 0.0 else 0.0
            reward_extra_info["score_openended"] = base_score if is_openended > 0.0 else 0.0
            reward_extra_info["score_citation"] = citation_score
            reward_extra_info["score_citation_raw"] = citation_score_raw
            reward_extra_info["citation_score_applied"] = citation_applied
            reward_extra_info["is_obj"] = is_obj
            reward_extra_info["is_openended"] = is_openended
            reward_extra_info["drop_from_training"] = 1.0 if drop_from_training else 0.0
            reward_extra_info["eval_script_load_failed"] = 1.0 if eval_script_load_failed else 0.0
            reward_extra_info["eval_script_runtime_failed"] = 1.0 if eval_script_runtime_failed else 0.0
            reward_extra_info["requires_google_maps"] = requires_google_maps
            reward_extra_info["missing_google_maps_api_key"] = missing_google_maps_api_key
            reward_extra_info["eval_timeout"] = 1.0 if _eval_timed_out else 0.0
        else:
            reward = self._coerce_metric_scalar(result)
            if reward is None:
                reward = 0.0
            is_openended = 1.0 if _is_openended_type(gt_type) else 0.0
            is_obj = 1.0 - is_openended
            reward_extra_info["score"] = reward
            reward_extra_info["acc"] = reward
            reward_extra_info["score_obj"] = reward if is_obj > 0.0 else 0.0
            reward_extra_info["score_openended"] = reward if is_openended > 0.0 else 0.0
            reward_extra_info["score_citation"] = 0.0
            reward_extra_info["score_citation_raw"] = 0.0
            reward_extra_info["citation_score_applied"] = 0.0
            reward_extra_info["is_obj"] = is_obj
            reward_extra_info["is_openended"] = is_openended
            reward_extra_info["drop_from_training"] = 0.0
            reward_extra_info["eval_script_load_failed"] = 0.0
            reward_extra_info["eval_script_runtime_failed"] = 0.0
            reward_extra_info["requires_google_maps"] = requires_google_maps
            reward_extra_info["missing_google_maps_api_key"] = missing_google_maps_api_key
            reward_extra_info["eval_timeout"] = 1.0 if _eval_timed_out else 0.0

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
