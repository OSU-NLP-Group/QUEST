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
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.trainer.ppo.ray_trainer import compute_response_mask


@dataclass
class RolloutSample:
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information
    full_batch: Any

    # AgentLoopOutput from generation
    agent_loop_output_list: list[AgentLoopOutput]

    # Metadata
    sample_id: str
    epoch: int

    # Processing metadata
    processing_times: list[float]
    tool_calls: list[float]
    param_version: int
    param_version_start: list[int]
    param_version_end: list[int]
    rollout_status: dict[str, Any]


@dataclass
class ValidateMetrics:
    """Metrics for validation"""

    timing_raw: dict[str, Any]
    metrics: Optional[dict[str, Any]] = None
    global_steps: Optional[int] = None
    param_version: Optional[int] = None


def prepare_single_generation_data(batch_dict, config) -> DataProto:
    """
    Similar to the logic of ray_trainer._prepare_generate_batch, but for a single sample.
    Separate the data used for generation from the original data.

    Returns:
        tuple: (original_batch_dict, gen_data_for_single_sample)
    """

    full_batch = DataProto.from_single_dict(batch_dict)

    batch_keys_to_pop = []
    non_tensor_batch_keys_to_pop = []

    existing_batch_keys = [k for k in batch_keys_to_pop if k in full_batch.batch.keys()]
    existing_non_tensor_keys = [k for k in non_tensor_batch_keys_to_pop if k in full_batch.non_tensor_batch.keys()]

    if existing_batch_keys or existing_non_tensor_keys:
        full_batch.pop(
            batch_keys=existing_batch_keys,
            non_tensor_batch_keys=existing_non_tensor_keys,
        )

    # Set selected agent.
    # Respect explicit default agent when provided (e.g. deepresearch_agent),
    # and keep the original fully-async defaults as fallback.
    agent_name = None
    rollout_agent_cfg = getattr(config.actor_rollout_ref.rollout, "agent", None)
    if rollout_agent_cfg is not None:
        agent_name = rollout_agent_cfg.get("default_agent_loop", None)
    if not agent_name:
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            agent_name = "async_partial_tool_agent"
        else:
            agent_name = "partial_single_turn_agent"
    full_batch.non_tensor_batch["agent_name"] = np.array([agent_name] * len(full_batch), dtype=object)

    # Fully-async feeds one sample at a time, then repeats it rollout.n times.
    # Ensure all repeated rows share the same sample index so get_trajectory_info()
    # can derive rollout_n=0..n-1 deterministically.
    if "index" not in full_batch.non_tensor_batch:
        full_batch.non_tensor_batch["index"] = np.zeros(len(full_batch), dtype=np.int32)

    # Add global step count to generated data
    full_batch = full_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
    # Preserve a stable per-rollout slot id across later session expansion and dumping.
    full_batch.non_tensor_batch["rollout_index"] = np.arange(len(full_batch), dtype=np.int32)
    return full_batch


def expand_multi_session_output(
    result: DataProto,
    tokenizer,
    prompt_length: int,
    response_length: int,
    session_level_reward: bool = True,
    session_weight_correction: bool = False,
) -> DataProto:
    """Expand multi-session outputs into separate batch rows.

    This is a function-form variant of DeepResearchAgentLoopManager._expand_sessions,
    used by fully_async_policy so each session can be trained as an independent trajectory.
    """
    if not session_level_reward:
        return result

    session_outputs_list = result.non_tensor_batch.get("session_outputs")
    if session_outputs_list is None:
        if session_weight_correction:
            result.batch["session_row_weight"] = torch.ones(len(result), dtype=torch.float32)
        return result

    original_batch_size = len(result)
    total_sessions = 0
    for session_outputs in session_outputs_list:
        if session_outputs is not None and len(session_outputs) > 1:
            total_sessions += len(session_outputs)
        else:
            total_sessions += 1

    if total_sessions == original_batch_size:
        # No expansion needed, but still add session_row_weight=1.0 for key consistency
        # when other batches in the same concat DO have multi-session expansion.
        if session_weight_correction:
            result.batch["session_row_weight"] = torch.ones(original_batch_size, dtype=torch.float32)
        return result

    expanded_prompts = []
    expanded_responses = []
    expanded_response_masks = []
    expanded_attention_masks = []
    expanded_input_ids = []
    expanded_position_ids = []

    has_rollout_log_probs = "rollout_log_probs" in result.batch.keys()
    expanded_log_probs = [] if has_rollout_log_probs else None
    has_rm_scores = "rm_scores" in result.batch.keys()
    expanded_rm_scores = [] if has_rm_scores else None

    expanded_non_tensor = {key: [] for key in result.non_tensor_batch.keys()}
    rollout_indices = []

    for i in range(original_batch_size):
        session_outputs = session_outputs_list[i]

        if session_outputs is None or len(session_outputs) <= 1:
            expanded_prompts.append(result.batch["prompts"][i])
            expanded_responses.append(result.batch["responses"][i])
            expanded_response_masks.append(result.batch["response_mask"][i])
            expanded_attention_masks.append(result.batch["attention_mask"][i])
            expanded_input_ids.append(result.batch["input_ids"][i])
            expanded_position_ids.append(result.batch["position_ids"][i])

            if has_rollout_log_probs:
                expanded_log_probs.append(result.batch["rollout_log_probs"][i])
            if has_rm_scores:
                expanded_rm_scores.append(result.batch["rm_scores"][i])

            for key in result.non_tensor_batch.keys():
                expanded_non_tensor[key].append(result.non_tensor_batch[key][i])

            rollout_indices.append(i)
            continue

        original_padding_side = tokenizer.padding_side
        for session_data in session_outputs:
            prompt_ids = session_data["prompt_ids"]
            tokenizer.padding_side = "left"
            prompt_output = tokenizer.pad(
                {"input_ids": prompt_ids},
                padding="max_length",
                max_length=prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            response_ids = session_data["response_ids"]
            tokenizer.padding_side = "right"
            response_output = tokenizer.pad(
                {"input_ids": response_ids},
                padding="max_length",
                max_length=response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask = session_data["response_mask"]
            response_mask_output = tokenizer.pad(
                {"input_ids": response_mask},
                padding="max_length",
                max_length=response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

            response_mask_tensor = response_mask_output["input_ids"] * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

            from verl.utils.model import compute_position_id_with_mask

            position_ids = compute_position_id_with_mask(attention_mask)

            expanded_prompts.append(prompt_output["input_ids"].squeeze(0))
            expanded_responses.append(response_output["input_ids"].squeeze(0))
            expanded_response_masks.append(response_mask_tensor.squeeze(0))
            expanded_attention_masks.append(attention_mask.squeeze(0))
            expanded_input_ids.append(input_ids.squeeze(0))
            expanded_position_ids.append(position_ids.squeeze(0))

            if has_rollout_log_probs and session_data.get("response_logprobs") is not None:
                log_probs = session_data["response_logprobs"]
                pad_size = response_length - len(log_probs)
                padded_log_probs = torch.tensor(log_probs + [0.0] * pad_size, dtype=torch.float32)
                expanded_log_probs.append(padded_log_probs)
            elif has_rollout_log_probs:
                expanded_log_probs.append(torch.zeros(response_length, dtype=torch.float32))

            if has_rm_scores:
                expanded_rm_scores.append(result.batch["rm_scores"][i])

            for key in result.non_tensor_batch.keys():
                if key == "session_id":
                    expanded_non_tensor[key].append(session_data["session_id"])
                elif key == "memory_state":
                    expanded_non_tensor[key].append(session_data.get("memory_state"))
                elif key == "agent_messages":
                    # Use per-session messages instead of rollout-level messages
                    expanded_non_tensor[key].append(session_data.get("messages"))
                else:
                    expanded_non_tensor[key].append(result.non_tensor_batch[key][i])

            rollout_indices.append(i)

        tokenizer.padding_side = original_padding_side

    expanded_batch = TensorDict(
        {
            "prompts": torch.stack(expanded_prompts),
            "responses": torch.stack(expanded_responses),
            "response_mask": torch.stack(expanded_response_masks),
            "input_ids": torch.stack(expanded_input_ids),
            "attention_mask": torch.stack(expanded_attention_masks),
            "position_ids": torch.stack(expanded_position_ids),
        },
        batch_size=total_sessions,
    )

    if has_rollout_log_probs and expanded_log_probs:
        expanded_batch["rollout_log_probs"] = torch.stack(expanded_log_probs)
    if has_rm_scores and expanded_rm_scores:
        expanded_batch["rm_scores"] = torch.stack(expanded_rm_scores)

    # Compute session_row_weight for rollout-equivalent loss weighting.
    # row_weight = row_assistant_tokens / rollout_total_assistant_tokens
    # so that split sessions collectively behave like one unsplit trajectory under token-mean.
    if session_weight_correction:
        from collections import defaultdict

        response_mask_stacked = expanded_batch["response_mask"]
        token_counts = response_mask_stacked.sum(dim=-1).float()  # (total_sessions,)

        rollout_groups = defaultdict(list)
        for j, ri in enumerate(rollout_indices):
            rollout_groups[ri].append(j)

        weights = torch.ones(total_sessions, dtype=torch.float32)
        num_unique_rollouts = len(rollout_groups)
        for ri, row_ids in rollout_groups.items():
            if len(row_ids) <= 1:
                continue  # non-expanded row, weight stays 1.0
            total_tokens = sum(token_counts[j].item() for j in row_ids)
            if total_tokens > 0:
                for j in row_ids:
                    weights[j] = token_counts[j].item() / total_tokens

        # Pre-scale so that sum(weights) = B (expanded rows) instead of R (unique rollouts).
        # This makes the weights compatible with the existing global_batch_size denominator,
        # ensuring DP-correct normalization without needing cross-rank all-reduce.
        # Math: loss = sum(w_i * token_mean_i) / global_batch_size
        #     → after DDP avg: total / global_batch_size = (B/R)*sum_r(token_mean_r)/B = sum_r(token_mean_r)/R
        if num_unique_rollouts > 0:
            weights = weights * (total_sessions / num_unique_rollouts)

        expanded_batch["session_row_weight"] = weights

    # Force 1D object arrays to avoid numpy auto-inferring 2D when nested
    # lists happen to share lengths across samples.
    expanded_non_tensor_batch = {}
    for key, values in expanded_non_tensor.items():
        arr = np.empty(len(values), dtype=object)
        arr[:] = values
        expanded_non_tensor_batch[key] = arr
    expanded_non_tensor_batch["rollout_index"] = np.asarray(rollout_indices, dtype=np.int64)

    return DataProto(
        batch=expanded_batch,
        non_tensor_batch=expanded_non_tensor_batch,
        meta_info=result.meta_info,
    )


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None
) -> DataProto:
    """
    Assemble gen_batch_output from RolloutSample objects
    Assembles batches from RolloutSample objects, similar to the _post_generate_batch logic in ray_trainer.

    Args:
        rollout_samples: List of RolloutSample objects
        tokenizer: Tokenizer instance
        config: Configuration object containing trainer settings
        balance_batch: Whether to balance the batch (simplified version)

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_samples is empty
    """
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided for batch assembly")

    print(f"[BatchUtils] Assembling batch from {len(rollout_samples)} RolloutSample objects")

    rollout_samples_batch = []
    processing_times = []
    tool_calls = []
    rollout_status = rollout_samples[0].rollout_status
    # Add a prefix to all rollout_status keys
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    for rs in rollout_samples:
        rollout_samples_batch.append(rs.full_batch)
    final_batch = DataProto.concat(rollout_samples_batch)

    # Calculate response_mask (if not present)
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Calculate the global valid token number
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    processing_times = final_batch.non_tensor_batch["processing_times"]
    tool_calls = final_batch.non_tensor_batch["tool_calls_times"]
    # Collect statistics

    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    tool_calls_stats = {}
    if len(tool_calls) > 0:
        tool_calls_stats = {
            "timing_s/agent_loop/tool_calls/max": np.max(tool_calls),
            "timing_s/agent_loop/tool_calls/min": np.min(tool_calls),
            "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls),
        }
    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}

    param_version_start = final_batch.non_tensor_batch["param_version_start"]
    param_version_end = final_batch.non_tensor_batch["param_version_end"]
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
    num_diff0 = param_version_diff.count(0)
    partial_stats = {
        "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
        "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
        "fully_async/partial/max_partial_span": max(param_version_diff),
    }
    # add meta_info
    param_versions = [rs.param_version for rs in rollout_samples]
    trajectorys_param_versions = final_batch.non_tensor_batch["param_version_end"]

    final_batch.meta_info.update(
        {
            "rollout_param_versions": param_versions,
            "param_version_diversity": len(set(param_versions)) if param_versions else 0,
            "trajectory_param_versions": trajectorys_param_versions,
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
            **tool_calls_stats,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch


class MetricsAggregator:
    """Metrics aggregator, used to combine metrics from multiple training steps"""

    def __init__(self, total_gpus: int):
        # Store all values ​​for each metric
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        # Store the number of samples at each step for weighted averaging
        self.sample_counts: list[int] = []
        # Store the timestamp of each step for time-related calculations
        self.timestamps: list[float] = []
        # Step Count
        self.step_count = 0
        # total num gpus used
        self.total_gpus = total_gpus

        # Metric aggregation rule configuration
        self.aggregation_rules = self._init_aggregation_rules()

    def _init_aggregation_rules(self) -> dict[str, dict[str, list[str]]]:
        """Initialize metrics aggregation rules"""
        return {
            # Time-Based metrics, can add metrics here
            "time_sum": ["perf/time_per_step"],
            "min": ["timing_s/agent_loop/tool_calls/min"],
            "avg": ["timing_s/agent_loop/tool_calls/mean"],
            "max": ["timing_s/agent_loop/tool_calls/max"],
            "last": [
                "fully_async/count/total_generated_samples",
                "fully_async/count/stale_samples_processed",
                "fully_async/count/stale_trajectory_processed",
                "fully_async/count/current_param_version",
                "fully_async/count/dropped_stale_samples",
                "training/global_step",  # TODO change name to: total_step
            ],
        }

    def add_step_metrics(self, metrics: dict[str, Any], sample_count: int, timestamp: float = None):
        """Adding a single-step metrics"""
        if timestamp is None:
            timestamp = time.time()

        self.sample_counts.append(sample_count)
        self.timestamps.append(timestamp)
        self.step_count += 1

        # Store all metrics values
        for key, value in metrics.items():
            if isinstance(value, int | float | np.number):
                self.metric_values[key].append(float(value))
            elif isinstance(value, torch.Tensor):
                self.metric_values[key].append(float(value.item()))

    def _get_aggregation_type(self, metric_name: str) -> str:
        """Determine the aggregation type based on the metric name"""
        for agg_type, metric_list in self.aggregation_rules.items():
            if metric_name in metric_list:
                return agg_type

        metric_lower = metric_name.lower()
        if any(keyword in metric_lower for keyword in ["timing_s/"]):
            return "time_sum"
        if any(keyword in metric_lower for keyword in ["mean", "avg", "average"]):
            return "avg"
        if any(keyword in metric_lower for keyword in ["max", "maximum"]):
            return "max"
        if any(keyword in metric_lower for keyword in ["min", "minimum"]):
            return "min"
        if any(keyword in metric_lower for keyword in ["sum", "total"]):
            return "sum"
        if any(keyword in metric_lower for keyword in ["weighted_avg"]):
            return "weighted_avg"

        return "avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        """Aggregating a single metric"""
        if not values:
            return 0.0

        agg_type = self._get_aggregation_type(metric_name)

        if agg_type == "last":
            return values[-1]

        elif agg_type == "weighted_avg":
            # Weighted average
            if len(values) != len(self.sample_counts):
                # If the lengths do not match, use a simple average
                return sum(values) / len(values)

            total_samples = sum(self.sample_counts)
            if total_samples == 0:
                return sum(values) / len(values)

            weighted_sum = sum(v * c for v, c in zip(values, self.sample_counts, strict=False))
            return weighted_sum / total_samples

        elif agg_type == "sum" or agg_type == "time_sum":
            return sum(values)

        elif agg_type == "avg":
            return sum(values) / len(values)

        elif agg_type == "max":
            return max(values)

        elif agg_type == "min":
            return min(values)

        else:
            # Default average
            return sum(values) / len(values)

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """aggregated metrics"""
        t = time.time()
        if self.step_count == 0:
            return {}

        aggregated = {}

        # Aggregate all metrics
        for metric_name, values in self.metric_values.items():
            aggregated[metric_name] = self._aggregate_single_metric(metric_name, values)

        # Aggregate special metrics
        aggregated = self._special_metrics_aggergate(aggregated)

        print(f"aggregated metrics done. cost {time.time() - t}")

        return aggregated

    def _special_metrics_aggergate(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """calculate special metrics"""

        # global_seqlen/minmax_diff
        if "global_seqlen/minmax_diff" in aggregated.keys():
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        # perf/throughput
        REQUIRED_PERF_KEYS = {"perf/throughput", "perf/total_num_tokens", "perf/time_per_step"}
        if REQUIRED_PERF_KEYS.issubset(aggregated):
            aggregated["perf/throughput"] = aggregated["perf/total_num_tokens"] / (
                aggregated["perf/time_per_step"] * self.total_gpus
            )

        # trainer/idle_ratio
        if "timing_s/gen" in aggregated.keys() and "timing_s/step" in aggregated.keys():
            aggregated["trainer/idle_ratio"] = aggregated["timing_s/gen"] / aggregated["timing_s/step"]

        return aggregated

    def reset(self):
        """Reset Aggregator"""
        self.metric_values.clear()
        self.sample_counts.clear()
        self.timestamps.clear()
        self.step_count = 0

    def get_current_stats(self) -> dict[str, Any]:
        """Get statistics about the current aggregation state (for debugging)"""
        return {
            "step_count": self.step_count,
            "metric_count": len(self.metric_values),
            "total_samples": sum(self.sample_counts),
            "metric_names": list(self.metric_values.keys()),
        }
