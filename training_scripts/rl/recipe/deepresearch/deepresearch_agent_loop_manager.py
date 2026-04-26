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
DeepResearch Agent Loop Manager for async rollout generation.

Each rollout may produce multiple sessions (when condenser triggers).
Each session is treated as an independent trajectory, all receiving
the same final reward from the rollout.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, fallback to default=%s", name, raw, default)
        return default


class DeepResearchAgentLoopManager(AgentLoopManager):
    """Agent loop manager for DeepResearch with memory and multi-turn support.

    This manager extends the base AgentLoopManager to support:
    - DeepResearch-specific agent loops
    - Memory/condenser integration
    - Multi-session generation

    Session handling:
    When condenser triggers, the agent loop stores session data in extra_fields.
    After generation, this manager expands sessions into separate batch rows.
    Each session is an independent trajectory that receives the same final reward.

    Example:
        1 rollout → condenser triggers 2 times → 3 sessions
        Each session is a separate trajectory with the same reward
    """

    def __init__(
        self,
        config,
        worker_group,
        rollout_resource_pool=None,
        reward_loop_worker_handles=None,
    ):
        """Initialize the DeepResearch agent loop manager.

        Args:
            config: Training configuration
            worker_group: Actor rollout worker group
            rollout_resource_pool: Resource pool for actor rollout
            reward_loop_worker_handles: Reward loop workers for async scoring
        """
        super().__init__(
            config=config,
            worker_group=worker_group,
            rollout_resource_pool=rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        # DeepResearch-specific configuration
        self.memory_enabled = _env_bool(
            "MEMORY_ENABLED", config.actor_rollout_ref.rollout.multi_turn.get("memory_enabled", True)
        )
        self.context_threshold = _env_int(
            "CONTEXT_THRESHOLD", config.actor_rollout_ref.rollout.multi_turn.get("context_threshold", 16000)
        )

        # Session-level reward configuration
        self.session_level_reward = config.actor_rollout_ref.rollout.multi_turn.get("session_level_reward", True)

        # Get tokenizer for padding operations
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.print_rollout = os.getenv("DEEPRESEARCH_PRINT_ROLLOUT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        try:
            self.print_rollout_max_samples = max(1, int(os.getenv("DEEPRESEARCH_PRINT_ROLLOUT_MAX_SAMPLES", "2")))
        except ValueError:
            self.print_rollout_max_samples = 2
        try:
            self.print_rollout_max_chars = max(64, int(os.getenv("DEEPRESEARCH_PRINT_ROLLOUT_MAX_CHARS", "2000")))
        except ValueError:
            self.print_rollout_max_chars = 2000
        logger.warning(
            "Rollout preview config: enabled=%s max_samples=%s max_chars=%s",
            self.print_rollout,
            self.print_rollout_max_samples,
            self.print_rollout_max_chars,
        )

        logger.info(
            f"Initialized DeepResearchAgentLoopManager with "
            f"memory_enabled={self.memory_enabled}, "
            f"session_level_reward={self.session_level_reward}"
        )

    def _expand_sessions(self, result: DataProto) -> DataProto:
        """Expand multi-session outputs into separate batch rows.

        When a rollout produces multiple sessions (via condenser), this method
        expands them into separate rows in the batch. Each session becomes
        an independent trajectory.

        Args:
            result: DataProto from parent generate_sequences

        Returns:
            Expanded DataProto with separate rows for each session
        """
        if not self.session_level_reward:
            return result

        # Check if any samples have multiple sessions
        session_outputs_list = result.non_tensor_batch.get("session_outputs")
        if session_outputs_list is None:
            return result

        # Count total sessions
        total_sessions = 0
        for session_outputs in session_outputs_list:
            if session_outputs is not None and len(session_outputs) > 1:
                total_sessions += len(session_outputs)
            else:
                total_sessions += 1

        # If no expansion needed, return as-is
        original_batch_size = len(result)
        if total_sessions == original_batch_size:
            return result

        logger.info(f"Expanding {original_batch_size} rollouts into {total_sessions} session trajectories")

        # Build expanded tensors
        expanded_prompts = []
        expanded_responses = []
        expanded_response_masks = []
        expanded_attention_masks = []
        expanded_input_ids = []
        expanded_position_ids = []

        # Track optional tensors
        has_rollout_log_probs = "rollout_log_probs" in result.batch.keys()
        expanded_log_probs = [] if has_rollout_log_probs else None
        has_rm_scores = "rm_scores" in result.batch.keys()
        expanded_rm_scores = [] if has_rm_scores else None

        # Build expanded non-tensor batch
        expanded_non_tensor = {key: [] for key in result.non_tensor_batch.keys()}

        # Track rollout indices for reward assignment
        rollout_indices = []

        for i in range(original_batch_size):
            session_outputs = session_outputs_list[i]

            if session_outputs is None or len(session_outputs) <= 1:
                # Single session - copy as-is
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
            else:
                # Multiple sessions - expand each one
                original_padding_side = self.tokenizer.padding_side
                for session_data in session_outputs:
                    # Pad prompt_ids (left-padded)
                    prompt_ids = session_data["prompt_ids"]
                    self.tokenizer.padding_side = "left"
                    prompt_output = self.tokenizer.pad(
                        {"input_ids": prompt_ids},
                        padding="max_length",
                        max_length=self.prompt_length,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                    if prompt_output["input_ids"].dim() == 1:
                        prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                        prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

                    # Pad response_ids (right-padded)
                    response_ids = session_data["response_ids"]
                    self.tokenizer.padding_side = "right"
                    response_output = self.tokenizer.pad(
                        {"input_ids": response_ids},
                        padding="max_length",
                        max_length=self.response_length,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                    if response_output["input_ids"].dim() == 1:
                        response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                        response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

                    # Pad response_mask
                    response_mask = session_data["response_mask"]
                    response_mask_output = self.tokenizer.pad(
                        {"input_ids": response_mask},
                        padding="max_length",
                        max_length=self.response_length,
                        return_tensors="pt",
                        return_attention_mask=False,
                    )
                    if response_mask_output["input_ids"].dim() == 1:
                        response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

                    # Compute combined tensors
                    response_mask_tensor = response_mask_output["input_ids"] * response_output["attention_mask"]
                    attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
                    input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

                    # Compute position_ids
                    from verl.utils.model import compute_position_id_with_mask
                    position_ids = compute_position_id_with_mask(attention_mask)

                    expanded_prompts.append(prompt_output["input_ids"].squeeze(0))
                    expanded_responses.append(response_output["input_ids"].squeeze(0))
                    expanded_response_masks.append(response_mask_tensor.squeeze(0))
                    expanded_attention_masks.append(attention_mask.squeeze(0))
                    expanded_input_ids.append(input_ids.squeeze(0))
                    expanded_position_ids.append(position_ids.squeeze(0))

                    # Handle log probs
                    if has_rollout_log_probs and session_data.get("response_logprobs") is not None:
                        log_probs = session_data["response_logprobs"]
                        pad_size = self.response_length - len(log_probs)
                        padded_log_probs = torch.tensor(log_probs + [0.0] * pad_size)
                        expanded_log_probs.append(padded_log_probs)
                    elif has_rollout_log_probs:
                        # Use zeros if log probs not available for this session
                        expanded_log_probs.append(torch.zeros(self.response_length))

                    # Session-level trajectories share the rollout-level final reward.
                    if has_rm_scores:
                        expanded_rm_scores.append(result.batch["rm_scores"][i])

                    # Copy non-tensor batch with session-specific updates
                    for key in result.non_tensor_batch.keys():
                        if key == "session_id":
                            expanded_non_tensor[key].append(session_data["session_id"])
                        elif key == "memory_state":
                            expanded_non_tensor[key].append(session_data.get("memory_state"))
                        elif key == "agent_messages":
                            # Use per-session messages instead of rollout-level messages
                            expanded_non_tensor[key].append(session_data.get("messages"))
                        else:
                            # Copy from original (e.g., full_response, data_source, etc.)
                            expanded_non_tensor[key].append(result.non_tensor_batch[key][i])

                    rollout_indices.append(i)

                self.tokenizer.padding_side = original_padding_side

        # Stack tensors
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

        # Convert non-tensor lists to arrays
        expanded_non_tensor_batch = {}
        for key, values in expanded_non_tensor.items():
            expanded_non_tensor_batch[key] = np.array(values, dtype=object)

        # Add rollout_index for reward assignment
        expanded_non_tensor_batch["rollout_index"] = np.array(rollout_indices)

        return DataProto(
            batch=expanded_batch,
            non_tensor_batch=expanded_non_tensor_batch,
            meta_info=result.meta_info,
        )

    def _print_rollout_preview(self, result: DataProto) -> None:
        if not self.print_rollout:
            return
        if "responses" not in result.batch.keys():
            return

        total = len(result)
        preview_num = min(total, self.print_rollout_max_samples)
        has_response_mask = "response_mask" in result.batch.keys()
        has_uid = "uid" in result.non_tensor_batch
        has_session_id = "session_id" in result.non_tensor_batch

        logger.warning("[ROLLOUT] batch_size=%s preview_samples=%s", total, preview_num)
        print(f"[ROLLOUT] batch_size={total}, preview_samples={preview_num}", flush=True)
        for i in range(preview_num):
            token_ids = result.batch["responses"][i]
            if has_response_mask:
                valid_len = int(result.batch["response_mask"][i].sum().item())
                token_ids = token_ids[:valid_len]

            token_list = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
            response_text = self.tokenizer.decode(token_list, skip_special_tokens=True)
            if len(response_text) > self.print_rollout_max_chars:
                response_text = response_text[: self.print_rollout_max_chars] + "...<truncated>"

            uid = result.non_tensor_batch["uid"][i] if has_uid else "NA"
            session_id = result.non_tensor_batch["session_id"][i] if has_session_id else "NA"
            if hasattr(uid, "item"):
                uid = uid.item()
            if hasattr(session_id, "item"):
                session_id = session_id.item()

            logger.warning("[ROLLOUT][%s] uid=%s session_id=%s chars=%s", i, uid, session_id, len(response_text))
            print(f"[ROLLOUT][{i}] uid={uid} session_id={session_id} chars={len(response_text)}", flush=True)
            print(response_text, flush=True)
            print("[ROLLOUT] ---", flush=True)

    def generate_sequences(self, gen_batch: DataProto) -> DataProto:
        """Generate sequences using the DeepResearch agent loop.

        The agent loop may produce multiple sessions per rollout (when condenser triggers).
        After generation, sessions are expanded into separate batch rows.
        Each session becomes an independent trajectory with the same reward.

        Args:
            gen_batch: Input batch for generation

        Returns:
            DataProto with generated sequences (may have more rows than input if
            rollouts produced multiple sessions)
        """
        # Call parent implementation
        result = super().generate_sequences(gen_batch)

        # Expand multi-session outputs into separate batch rows (training only).
        # Skip during validation to avoid inflating best@k/worst@k metrics.
        is_validate = gen_batch.meta_info.get("validate", False)
        if self.session_level_reward and not is_validate:
            result = self._expand_sessions(result)

        # Log session statistics
        if "num_sessions" in result.non_tensor_batch:
            num_sessions = result.non_tensor_batch["num_sessions"]
            total_sessions = sum(num_sessions) if hasattr(num_sessions, '__iter__') else num_sessions
            logger.info(f"Generated {len(result.batch['responses'])} trajectories from rollouts")

        self._print_rollout_preview(result)

        return result
