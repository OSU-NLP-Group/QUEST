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
DeepResearch Reward Manager.

This reward manager extends NaiveRewardManager to properly handle
multi-session rollouts where all sessions should receive the same reward.
"""

from collections import defaultdict
import logging
import math
import os
from numbers import Real
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager.naive import NaiveRewardManager
from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY, register

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _coerce_metric_scalar(value: Any) -> float | None:
    """Convert supported scalar metric values to finite float."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Real):
        val = float(value)
        if math.isfinite(val):
            return val
    return None


def _is_openended_type(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in {"openended", "open-ended"}


def _is_openended_eval_type(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized.startswith(("openended", "open-ended"))


class DeepResearchRewardManager(NaiveRewardManager):
    """Reward manager for DeepResearch with multi-session support.

    This manager extends NaiveRewardManager to pass `full_response` from
    the agent loop's extra_fields to the compute_score function. This ensures
    all sessions from the same rollout receive the same reward.
    """

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute rewards for each sample in the batch.

        Extends the parent implementation to pass full_response and other
        agent loop fields to the compute_score function.
        """
        # If there is rm score, we directly return rm score
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
            if valid_prompt_length > 0:
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            else:
                valid_prompt_ids = prompt_ids[:0]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum().item())
            if valid_response_length > 0:
                valid_response_ids = response_ids[:valid_response_length]
            else:
                valid_response_ids = response_ids[:0]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            reward_model = data_item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model.get("ground_truth")
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            # Get extra_info from dataset and merge with agent loop fields
            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))

            # Add agent loop fields to extra_info
            # These are important for DeepResearch reward computation
            extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})

            # Add full_response for multi-session reward computation
            # When sessions are expanded, all sessions share the same full_response
            full_response = data_item.non_tensor_batch.get("full_response")
            if full_response is not None:
                extra_info["full_response"] = full_response

            # Add session tracking info
            extra_info["session_id"] = data_item.non_tensor_batch.get("session_id")
            extra_info["num_sessions"] = data_item.non_tensor_batch.get("num_sessions")
            extra_info["rollout_index"] = data_item.non_tensor_batch.get("rollout_index")

            # Add tool rewards if available
            tool_rewards = data_item.non_tensor_batch.get("tool_rewards")
            if tool_rewards is not None:
                extra_info["tool_rewards"] = tool_rewards

            # Check if terminated by consecutive invalid responses
            terminated_by_consecutive_invalid = data_item.non_tensor_batch.get("terminated_by_consecutive_invalid", False)

            if terminated_by_consecutive_invalid:
                score = {"score": 0.0, "terminated_by_consecutive_invalid": True}
                print(f"[RewardManager] Sample {i}: terminated_by_consecutive_invalid, score=0.0")
            else:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

            gt_type = ""
            if isinstance(ground_truth, dict):
                gt_type = str(ground_truth.get("type", "")).strip().lower()

            if isinstance(score, dict):
                reward = _coerce_metric_scalar(score.get("score"))
                if reward is None:
                    reward = _coerce_metric_scalar(score.get("final_score"))
                if reward is None:
                    reward = 0.0
                base_score = _coerce_metric_scalar(score.get("score_without_citation"))
                if base_score is None:
                    base_score = _coerce_metric_scalar(score.get("base_score"))
                if base_score is None:
                    base_score = reward
                citation_score = _coerce_metric_scalar(score.get("citation_score_added"))
                if citation_score is None:
                    citation_score = 0.0
                citation_score_raw = _coerce_metric_scalar(score.get("citation_score_raw"))
                if citation_score_raw is None:
                    citation_score_raw = citation_score
                citation_applied = 1.0 if bool(score.get("citation_score_applied", False)) else 0.0
                drop_from_training = bool(score.get("drop_from_training", False))
                eval_script_load_failed = bool(score.get("eval_script_load_failed", False))
                eval_script_runtime_failed = bool(score.get("eval_script_runtime_failed", False))
                eval_type = str(score.get("eval_type", "")).strip().lower()
                is_openended = 1.0 if (_is_openended_type(gt_type) or _is_openended_eval_type(eval_type)) else 0.0
                is_obj = 1.0 - is_openended
                requires_google_maps = 1.0 if bool(extra_info.get("requires_google_maps", False)) else 0.0
                missing_google_maps_api_key = (
                    1.0 if requires_google_maps > 0.0 and not os.environ.get("GOOGLE_MAPS_API_KEY") else 0.0
                )
                # Keep only stable numeric metrics for downstream aggregation.
                reward_extra_info["score"].append(reward)
                reward_extra_info["score_obj"].append(base_score if is_obj > 0.0 else 0.0)
                reward_extra_info["score_openended"].append(base_score if is_openended > 0.0 else 0.0)
                reward_extra_info["score_citation"].append(citation_score)
                reward_extra_info["score_citation_raw"].append(citation_score_raw)
                reward_extra_info["citation_score_applied"].append(citation_applied)
                reward_extra_info["is_obj"].append(is_obj)
                reward_extra_info["is_openended"].append(is_openended)
                reward_extra_info["drop_from_training"].append(1.0 if drop_from_training else 0.0)
                reward_extra_info["eval_script_load_failed"].append(1.0 if eval_script_load_failed else 0.0)
                reward_extra_info["eval_script_runtime_failed"].append(1.0 if eval_script_runtime_failed else 0.0)
                reward_extra_info["requires_google_maps"].append(requires_google_maps)
                reward_extra_info["missing_google_maps_api_key"].append(missing_google_maps_api_key)
            else:
                reward = _coerce_metric_scalar(score)
                if reward is None:
                    reward = 0.0
                is_openended = 1.0 if _is_openended_type(gt_type) else 0.0
                is_obj = 1.0 - is_openended
                requires_google_maps = 1.0 if bool(extra_info.get("requires_google_maps", False)) else 0.0
                missing_google_maps_api_key = (
                    1.0 if requires_google_maps > 0.0 and not os.environ.get("GOOGLE_MAPS_API_KEY") else 0.0
                )
                reward_extra_info["score"].append(reward)
                reward_extra_info["score_obj"].append(reward if is_obj > 0.0 else 0.0)
                reward_extra_info["score_openended"].append(reward if is_openended > 0.0 else 0.0)
                reward_extra_info["score_citation"].append(0.0)
                reward_extra_info["score_citation_raw"].append(0.0)
                reward_extra_info["citation_score_applied"].append(0.0)
                reward_extra_info["is_obj"].append(is_obj)
                reward_extra_info["is_openended"].append(is_openended)
                reward_extra_info["drop_from_training"].append(0.0)
                reward_extra_info["eval_script_load_failed"].append(0.0)
                reward_extra_info["eval_script_runtime_failed"].append(0.0)
                reward_extra_info["requires_google_maps"].append(requires_google_maps)
                reward_extra_info["missing_google_maps_api_key"].append(missing_google_maps_api_key)

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = reward
            else:
                logger.warning(
                    "Sample %d has empty valid response (data_source=%s); reward is computed but cannot be assigned to a token.",
                    i,
                    data_source,
                )

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str[:500] + "..." if len(response_str) > 500 else response_str)
                if full_response:
                    print("[full_response length]", len(full_response))
                print("[ground_truth]", ground_truth)
                print("[session_id]", extra_info.get("session_id"))
                print("[num_sessions]", extra_info.get("num_sessions"))
                if isinstance(score, dict):
                    for key, value in score.items():
                        if key != "score":
                            print(f"[{key}]", value)
                print("[score]", reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


def _register_deepresearch_reward_manager() -> None:
    """Register once to avoid duplicate-register errors under importlib reloads."""
    if REWARD_MANAGER_REGISTRY.get("deepresearch") is None:
        register("deepresearch")(DeepResearchRewardManager)


_register_deepresearch_reward_manager()
