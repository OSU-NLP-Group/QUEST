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

import json
import os
import re
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from math import gcd
from pprint import pprint
from typing import Any

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    ValidateMetrics,
    assemble_batch_from_rollout_samples,
)
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import compute_reward_async, load_reward_manager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer


class TrainingStopException(Exception):
    """Exception raised to signal training should stop"""

    pass


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(SeparateRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        device_name=None,
    ):
        # ==================== RayPPOTrainer config ====================

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # ==================== SeparateRayPPOTrainer config ====================
        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.logger = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        # ==================== fully async config ====================

        self.message_queue_client = None
        self.param_synchronizer = None
        self.reward_loop_manager = None
        self.rollouter = None
        self.wait_last_update = None
        self.wait_last_resume = None
        self.validate_task = None

        # Statistics
        # we start from step 1
        self.global_steps = 1
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_samples_processed = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        self.train_val_metrics = None
        self.train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        # use trainer to do validation
        if self.config.async_training.use_trainer_do_validate:
            from verl.trainer.main_ppo import create_rl_dataset
            from verl.utils.dataset.rl_dataset import collate_fn

            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
            rollout_gpus = config.rollout.nnodes * config.rollout.n_gpus_per_node
            print(f"[FullyAsyncTrainer] split before val_dataset total len: {len(val_dataset)}")
            split_dataset = val_dataset.split(total_gpus)
            rollout_val_dataset0 = split_dataset[rollout_gpus:]
            from torch.utils.data import ConcatDataset

            val_dataset = ConcatDataset(rollout_val_dataset0)
            print(f"[FullyAsyncTrainer] split after val_dataset total len: {len(val_dataset)}")
            self.val_dataset = val_dataset
            # update val_dataloader
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(val_dataset)
            from torchdata.stateful_dataloader import StatefulDataLoader

            print(f"[FullyAsyncTrainer] create val_dataloader with batch_size: {val_batch_size}")
            self.val_dataloader = StatefulDataLoader(
                dataset=val_dataset,
                batch_size=val_batch_size,
                num_workers=self.config.data["dataloader_num_workers"],
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    def set_parameter_synchronizer(self, param_synchronizer):
        """Set parameter synchronizer"""
        self.param_synchronizer = param_synchronizer

    def set_rollouter(self, rollouter):
        """Set rollouter reference and replicate ParameterSynchronizer initialization."""
        from ray.util.collective import collective
        from verl.utils.device import get_nccl_backend

        self.rollouter = rollouter
        self.rollout_wg = ray.get(rollouter.get_rollout_wg.remote())
        self.sync_group_name = "actor_rollout"

        # 1. propagate weights info from actor to rollout workers
        weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(weights_info)

        # 2. create NCCL sync group between actor and rollout workers
        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        n_workers = len(actor_rollout_workers)
        if self.config.trainer.device == "npu":
            master_address = ray.get(self.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
            master_port = ray.get(self.actor_wg.workers[0]._get_free_port.remote())
            self.actor_wg.create_weight_sync_group(master_address, master_port, 0, n_workers)
            ray.get(self.rollout_wg.create_weight_sync_group(
                master_address, master_port, len(self.actor_wg.workers), n_workers
            ))
        else:
            collective.create_collective_group(
                actor_rollout_workers,
                n_workers,
                list(range(n_workers)),
                backend=get_nccl_backend(),
                group_name=self.sync_group_name,
            )

        # 3. init checkpoint engine on workers if enabled
        if self.config.async_training.checkpoint_engine.enable:
            actor_num = len(self.actor_wg.workers)
            rollout_num = len(self.rollout_wg.workers)
            ray.get(self.actor_wg.init_checkpoint_engine(
                rank_offset=0, actor_num=actor_num, rollout_num=rollout_num
            ))
            ray.get(self.rollout_wg.init_checkpoint_engine(
                rank_offset=actor_num, actor_num=actor_num, rollout_num=rollout_num
            ))

        print(
            f"[FullyAsyncTrainer] set_rollouter done: sync_group='{self.sync_group_name}', "
            f"n_workers={n_workers}, checkpoint_engine={self.config.async_training.checkpoint_engine.enable}"
        )

    def set_total_train_steps(self, total_train_steps):
        self.total_train_steps = total_train_steps
        self.progress_bar = tqdm(total=self.total_train_steps, initial=self.current_param_version, desc="Training Progress")

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    @staticmethod
    def _env_flag(name: str, default: str = "0") -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _checkpoint_name_rank(name: str) -> tuple[int, int] | None:
        patterns = (
            (2, r"savefreq_step_(\d+)"),
            (1, r"step_(\d+)"),
            (0, r"global_step_(\d+)"),
        )
        for rank, pattern in patterns:
            match = re.fullmatch(pattern, name)
            if match:
                return rank, int(match.group(1))
        return None

    @classmethod
    def _extract_checkpoint_step(cls, checkpoint_path: str) -> int:
        info = cls._checkpoint_name_rank(os.path.basename(os.path.normpath(checkpoint_path)))
        if info is None:
            raise ValueError(f"Unsupported checkpoint path: {checkpoint_path}")
        return info[1]

    @staticmethod
    def _save_every_step_ckpt_enabled() -> bool:
        return FullyAsyncTrainer._env_flag("DEEPRESEARCH_SAVE_EVERY_STEP_CKPT", "1")

    def _checkpoint_dir_name(self, step: int, *, formal: bool) -> str:
        return f"savefreq_step_{step}" if formal else f"step_{step}"

    @staticmethod
    def _needs_old_log_prob_anchor(config) -> bool:
        rollout_corr_config = config.algorithm.get("rollout_correction", None)
        return bool(rollout_corr_config and not rollout_corr_config.get("bypass_mode", False))

    def _anchor_checkpoint_dir(self, checkpoint_folder: str) -> str:
        return os.path.join(checkpoint_folder, "actor_anchor")

    def _save_rollout_anchor_checkpoint(
        self,
        checkpoint_folder: str,
        *,
        remote_checkpoint_folder: str | None,
        max_actor_ckpt_to_keep,
    ) -> None:
        if self.local_trigger_step <= 1 or not self._needs_old_log_prob_anchor(self.config):
            return

        if not all(self.actor_rollout_wg.has_cpu_model(1)):
            print(
                "[FullyAsyncTrainer] local_trigger_step > 1 but old-log-prob anchor is missing in CPU cache. "
                "Skipping actor_anchor checkpoint save."
            )
            return

        anchor_local_path = self._anchor_checkpoint_dir(checkpoint_folder)
        anchor_remote_path = None if remote_checkpoint_folder is None else os.path.join(remote_checkpoint_folder, "actor_anchor")
        current_slot = int(self.local_trigger_step)

        self.actor_rollout_wg.save_model_to_cpu(current_slot)
        try:
            self.actor_rollout_wg.restore_model_from_cpu(1)
            self.actor_rollout_wg.save_checkpoint(
                anchor_local_path,
                anchor_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_actor_ckpt_to_keep,
            )
        finally:
            self.actor_rollout_wg.restore_model_from_cpu(current_slot)
            self.actor_rollout_wg.clear_cpu_model(current_slot)

    def _restore_rollout_anchor_checkpoint(
        self,
        checkpoint_folder: str,
        *,
        actor_path: str,
        del_local_after_load: bool,
    ) -> None:
        if self.local_trigger_step <= 1 or not self._needs_old_log_prob_anchor(self.config):
            return

        anchor_path = self._anchor_checkpoint_dir(checkpoint_folder)
        if not self._has_complete_actor_checkpoint(anchor_path):
            print(
                "[FullyAsyncTrainer] Missing actor_anchor checkpoint for resumed local_trigger_step > 1. "
                "Resetting local_trigger_step to 1 to avoid using an invalid old-log-prob anchor."
            )
            self.local_trigger_step = 1
            return

        self.actor_rollout_wg.load_checkpoint(anchor_path, del_local_after_load=del_local_after_load)
        self.actor_rollout_wg.save_model_to_cpu(1)
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=del_local_after_load)
        print(f"[FullyAsyncTrainer] Restored old-log-prob anchor from {anchor_path}")

    def _remove_previous_temporary_checkpoint(self, current_step: int) -> None:
        if current_step <= 1:
            return
        previous_temp_dir = os.path.join(self.config.trainer.default_local_dir, f"step_{current_step - 1}")
        if os.path.isdir(previous_temp_dir):
            shutil.rmtree(previous_temp_dir)
            print(f"[FullyAsyncTrainer] Removed previous temporary checkpoint: {previous_temp_dir}")

    @staticmethod
    def _jsonable(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [FullyAsyncTrainer._jsonable(v) for v in value.tolist()]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): FullyAsyncTrainer._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [FullyAsyncTrainer._jsonable(v) for v in value]
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except TypeError:
            return str(value)

    @staticmethod
    def _remove_prev_state_from_messages(messages: list[dict]) -> list[dict]:
        marker = "====================\nRESEARCH STATE SUMMARY (prev_state)"
        cleaned = deepcopy(messages)
        for msg in cleaned:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                break
            idx = content.find(marker)
            if idx != -1:
                msg["content"] = content[:idx].rstrip()
            break
        return cleaned

    @staticmethod
    def _has_value(value) -> bool:
        if value is None:
            return False
        if isinstance(value, str) and value.strip() == "":
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        return True

    @staticmethod
    def _path_component(value, fallback: str) -> str:
        text = str(value).strip() if value is not None else ""
        if not text:
            text = fallback
        text = text.replace(os.sep, "_")
        if os.altsep:
            text = text.replace(os.altsep, "_")
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        text = text.strip("._-")
        return text[:128] if text else fallback

    @staticmethod
    def _coerce_dict(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return None
        return None

    @staticmethod
    def _get_non_tensor_value(non_tensor_batch: dict, key: str, idx: int):
        if key not in non_tensor_batch:
            return None
        value = non_tensor_batch[key]
        try:
            value = value[idx]
        except Exception:
            pass
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        return value

    @staticmethod
    def _to_float_array(values):
        if values is None:
            return None
        if isinstance(values, np.ndarray):
            arr = values.reshape(-1)
        else:
            try:
                arr = np.asarray(values).reshape(-1)
            except Exception:
                return None
        if arr.size == 0:
            return None
        try:
            arr = arr.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            return None
        return arr

    @classmethod
    def _compute_masked_metric_stats(cls, values, mask, *, prefix: str) -> dict[str, float]:
        metrics = {}
        value_arr = cls._to_float_array(values)
        mask_arr = cls._to_float_array(mask)
        if value_arr is None or mask_arr is None:
            return metrics
        n = min(value_arr.size, mask_arr.size)
        if n <= 0:
            return metrics
        selected = value_arr[:n][mask_arr[:n] > 0.0]
        metrics[f"{prefix}/count"] = float(selected.size)
        if selected.size > 0:
            metrics[f"{prefix}/mean"] = float(np.mean(selected))
            metrics[f"{prefix}/max"] = float(np.max(selected))
            metrics[f"{prefix}/min"] = float(np.min(selected))
        return metrics

    @classmethod
    def _valid_rollout_metric_rows(cls, batch: DataProto) -> np.ndarray:
        valid = np.ones(len(batch), dtype=bool)

        if "__is_pad__" in batch.batch.keys():
            is_pad = batch.batch["__is_pad__"].detach().cpu().numpy().astype(bool)
            valid[: is_pad.size] &= ~is_pad[: len(batch)]

        if "response_mask" in batch.batch.keys():
            response_len = batch.batch["response_mask"].sum(dim=-1).detach().cpu().numpy()
            valid[: response_len.size] &= response_len[: len(batch)] > 0

        return valid

    @classmethod
    def _rollout_metric_groups(cls, batch: DataProto) -> list[list[int]]:
        """Group expanded session rows back to their original rollout slot."""
        valid_rows = cls._valid_rollout_metric_rows(batch)
        groups: dict[tuple[Any, ...], list[int]] = {}

        for i in range(len(batch)):
            if not valid_rows[i]:
                continue

            uid = cls._get_non_tensor_value(batch.non_tensor_batch, "uid", i)
            prompt_key = str(uid) if cls._has_value(uid) else None
            if prompt_key is None:
                for key in ("index", "task_id", "id"):
                    value = cls._get_non_tensor_value(batch.non_tensor_batch, key, i)
                    if cls._has_value(value):
                        prompt_key = f"{key}:{value}"
                        break

            rollout_key = None
            for key in ("rollout_index", "trajectory_rollout_n", "rollout_n"):
                value = cls._get_non_tensor_value(batch.non_tensor_batch, key, i)
                if cls._has_value(value):
                    rollout_key = f"{key}:{value}"
                    break

            if prompt_key is not None and rollout_key is not None:
                metric_key = ("rollout", prompt_key, rollout_key)
            else:
                # Without both prompt and rollout identifiers, do not risk merging
                # independent rows that merely share a partial fallback key.
                metric_key = ("row", i)

            groups.setdefault(metric_key, []).append(i)

        return list(groups.values())

    @classmethod
    def _group_metric_values(cls, values, groups: list[list[int]]) -> np.ndarray | None:
        value_arr = cls._to_float_array(values)
        if value_arr is None:
            return None

        grouped = []
        for rows in groups:
            idxs = [idx for idx in rows if idx < value_arr.size]
            if not idxs:
                continue
            selected = value_arr[idxs]
            selected = selected[np.isfinite(selected)]
            if selected.size > 0:
                grouped.append(float(np.mean(selected)))

        if not grouped:
            return None
        return np.asarray(grouped, dtype=np.float64)

    @classmethod
    def _compute_grouped_masked_metric_stats(
        cls, values, mask, groups: list[list[int]], *, prefix: str
    ) -> dict[str, float]:
        metrics = {}
        value_arr = cls._to_float_array(values)
        mask_arr = cls._to_float_array(mask)
        if value_arr is None or mask_arr is None:
            return metrics

        n = min(value_arr.size, mask_arr.size)
        selected_by_rollout = []
        for rows in groups:
            idxs = [idx for idx in rows if idx < n and mask_arr[idx] > 0.0]
            if not idxs:
                continue
            selected = value_arr[idxs]
            selected = selected[np.isfinite(selected)]
            if selected.size > 0:
                selected_by_rollout.append(float(np.mean(selected)))

        metrics[f"{prefix}/count"] = float(len(selected_by_rollout))
        if selected_by_rollout:
            arr = np.asarray(selected_by_rollout, dtype=np.float64)
            metrics[f"{prefix}/mean"] = float(np.mean(arr))
            metrics[f"{prefix}/max"] = float(np.max(arr))
            metrics[f"{prefix}/min"] = float(np.min(arr))
        return metrics

    @classmethod
    def _compute_rollout_sequence_metrics(cls, batch: DataProto) -> dict[str, float]:
        metrics = {}
        groups = cls._rollout_metric_groups(batch)
        if not groups or "token_level_scores" not in batch.batch.keys():
            return metrics

        sequence_score = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()
        grouped_score = cls._group_metric_values(sequence_score, groups)
        if grouped_score is not None and grouped_score.size > 0:
            metrics["critic/score/mean"] = float(np.mean(grouped_score))
            metrics["critic/score/max"] = float(np.max(grouped_score))
            metrics["critic/score/min"] = float(np.min(grouped_score))

        if "token_level_rewards" in batch.batch.keys():
            sequence_reward = batch.batch["token_level_rewards"].sum(-1).detach().cpu().numpy()
            grouped_reward = cls._group_metric_values(sequence_reward, groups)
            if grouped_reward is not None and grouped_reward.size > 0:
                metrics["critic/rewards/mean"] = float(np.mean(grouped_reward))
                metrics["critic/rewards/max"] = float(np.max(grouped_reward))
                metrics["critic/rewards/min"] = float(np.min(grouped_reward))

        return metrics

    @classmethod
    def _compute_training_reward_metrics(cls, batch: DataProto) -> dict[str, float]:
        metrics = {}
        reward_metrics = batch.non_tensor_batch
        groups = cls._rollout_metric_groups(batch)
        if not groups:
            return metrics

        score = cls._group_metric_values(reward_metrics.get("score"), groups)
        if score is not None and score.size > 0:
            clipped_acc = np.maximum(score, 0.0)
            metrics["critic/acc/mean"] = float(np.mean(clipped_acc))
            metrics["critic/acc/max"] = float(np.max(clipped_acc))
            metrics["critic/acc/min"] = float(np.min(clipped_acc))

        metrics.update(
            cls._compute_grouped_masked_metric_stats(
                reward_metrics.get("score_obj"),
                reward_metrics.get("is_obj"),
                groups,
                prefix="critic/base_score_obj",
            )
        )
        metrics.update(
            cls._compute_grouped_masked_metric_stats(
                reward_metrics.get("score_openended"),
                reward_metrics.get("is_openended"),
                groups,
                prefix="critic/base_score_openended",
            )
        )
        score_citation = cls._group_metric_values(reward_metrics.get("score_citation"), groups)
        if score_citation is not None and score_citation.size > 0:
            metrics.update(
                cls._compute_masked_metric_stats(
                    score_citation,
                    np.ones_like(score_citation),
                    prefix="critic/score_citation",
                )
            )
        score_citation_raw = cls._group_metric_values(reward_metrics.get("score_citation_raw"), groups)
        if score_citation_raw is not None and score_citation_raw.size > 0:
            metrics.update(
                cls._compute_masked_metric_stats(
                    score_citation_raw,
                    np.ones_like(score_citation_raw),
                    prefix="critic/score_citation_raw",
                )
            )
        citation_applied = cls._group_metric_values(reward_metrics.get("citation_score_applied"), groups)
        if citation_applied is not None and citation_applied.size > 0:
            metrics["critic/score_citation_applied_rate"] = float(np.mean(citation_applied))
        return metrics

    @staticmethod
    def _augment_validation_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
        metrics = dict(metric_dict)
        score_obj_keys_to_remove = []
        score_openended_keys_to_remove = []
        additions = {}
        for key, value in list(metrics.items()):
            if not isinstance(key, str):
                continue
            if "/score_obj/" in key:
                denom_key = key.replace("/score_obj/", "/is_obj/")
                denom = metrics.get(denom_key)
                if denom:
                    additions[key.replace("/score_obj/", "/base_score_obj/")] = value / denom
                    if "/val-aux/" in key and key.endswith("/mean@1"):
                        additions["val/critic/base_score_obj/mean"] = value / denom
                score_obj_keys_to_remove.append(key)
            elif "/score_openended/" in key:
                denom_key = key.replace("/score_openended/", "/is_openended/")
                denom = metrics.get(denom_key)
                if denom:
                    additions[key.replace("/score_openended/", "/base_score_openended/")] = value / denom
                    if "/val-aux/" in key and key.endswith("/mean@1"):
                        additions["val/critic/base_score_openended/mean"] = value / denom
                score_openended_keys_to_remove.append(key)
            elif "/score_citation/" in key and "/val-aux/" in key and key.endswith("/mean@1"):
                additions["val/critic/score_citation/mean"] = value
            elif "/score_citation_raw/" in key and "/val-aux/" in key and key.endswith("/mean@1"):
                additions["val/critic/score_citation_raw/mean"] = value
            elif "/citation_score_applied/" in key and "/val-aux/" in key and key.endswith("/mean@1"):
                additions["val/critic/score_citation_applied_rate"] = value
            elif "/acc/" in key and key.startswith("val-core/") and key.endswith("/mean@1"):
                additions["val/critic/acc/mean"] = value

        for key in score_obj_keys_to_remove + score_openended_keys_to_remove:
            metrics.pop(key, None)
        metrics.update(additions)
        return metrics

    @staticmethod
    def _select_console_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
        keep_keys = [
            "training/global_step",
            "training/epoch",
            "fully_async/count/current_param_version",
            "critic/score/mean",
            "critic/acc/mean",
            "critic/base_score_obj/mean",
            "critic/base_score_openended/mean",
            "critic/score_citation/mean",
            "critic/score_citation_applied_rate",
            "val/critic/acc/mean",
            "val/critic/base_score_obj/mean",
            "val/critic/base_score_openended/mean",
            "val/critic/score_citation/mean",
            "val/critic/score_citation_applied_rate",
        ]
        return {key: metric_dict[key] for key in keep_keys if key in metric_dict}

    @staticmethod
    def _rewrite_non_console_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
        rewritten = dict(metric_dict)
        if "training/global_step" in rewritten:
            rewritten["training/local_global_step"] = rewritten.pop("training/global_step")
        return rewritten

    def _extract_task_id(self, batch: DataProto, idx: int) -> str:
        reward_model = self._coerce_dict(self._get_non_tensor_value(batch.non_tensor_batch, "reward_model", idx))
        if reward_model is not None:
            for key in ("task_id", "original_task_id", "question_id"):
                value = reward_model.get(key)
                if self._has_value(value):
                    return self._path_component(value, f"sample_{idx}")

            ground_truth = self._coerce_dict(reward_model.get("ground_truth"))
            if ground_truth is not None:
                for key in ("task_id", "original_task_id", "question_id"):
                    value = ground_truth.get(key)
                    if self._has_value(value):
                        return self._path_component(value, f"sample_{idx}")

        extra_info = self._coerce_dict(self._get_non_tensor_value(batch.non_tensor_batch, "extra_info", idx))
        if extra_info is not None:
            for key in ("task_id", "original_task_id", "question_id", "id"):
                value = extra_info.get(key)
                if self._has_value(value):
                    return self._path_component(value, f"sample_{idx}")

        for key in ("task_id", "id"):
            value = self._get_non_tensor_value(batch.non_tensor_batch, key, idx)
            if self._has_value(value):
                return self._path_component(value, f"sample_{idx}")

        for key in ("index",):
            value = self._get_non_tensor_value(batch.non_tensor_batch, key, idx)
            if self._has_value(value):
                return self._path_component(value, f"sample_{idx}")

        uid = self._get_non_tensor_value(batch.non_tensor_batch, "uid", idx)
        if self._has_value(uid):
            return self._path_component(uid, f"sample_{idx}")
        return f"sample_{idx}"

    def _compute_rollout_numbers(self, batch: DataProto, task_ids: list[str]) -> list[int]:
        rollout_numbers = []
        rollout_index_map = {}
        counters = defaultdict(int)
        has_trajectory_rollout_n = "trajectory_rollout_n" in batch.non_tensor_batch
        has_rollout_index = "rollout_index" in batch.non_tensor_batch

        for i, task_id in enumerate(task_ids):
            uid = self._get_non_tensor_value(batch.non_tensor_batch, "uid", i)
            group_key = str(uid) if self._has_value(uid) else task_id

            if has_trajectory_rollout_n:
                trajectory_rollout_n = self._get_non_tensor_value(batch.non_tensor_batch, "trajectory_rollout_n", i)
                if self._has_value(trajectory_rollout_n):
                    rollout_no = int(trajectory_rollout_n)
                else:
                    rollout_no = counters[group_key]
                    counters[group_key] += 1
            elif has_rollout_index:
                rollout_index = self._get_non_tensor_value(batch.non_tensor_batch, "rollout_index", i)
                key = (group_key, rollout_index)
                if key not in rollout_index_map:
                    rollout_index_map[key] = counters[group_key]
                    counters[group_key] += 1
                rollout_no = rollout_index_map[key]
            else:
                rollout_no = counters[group_key]
                counters[group_key] += 1
            rollout_numbers.append(int(rollout_no))

        return rollout_numbers

    def _select_final_rollout_rows(self, batch: DataProto, rollout_numbers: list[int]) -> list[int]:
        """Keep one representative row per rollout slot after session expansion.

        Session-level expansion can turn one rollout into multiple batch rows with
        different session_id / per-session scores. For saved rollout artifacts we
        want one final record per rollout, so keep the row with the largest
        session_id for each rollout slot.
        """
        selected_by_key: dict[tuple[str, int], tuple[int, int]] = {}
        task_ids = [self._extract_task_id(batch, i) for i in range(len(batch))]

        for i, rollout_no in enumerate(rollout_numbers):
            uid = self._get_non_tensor_value(batch.non_tensor_batch, "uid", i)
            group_key = str(uid) if self._has_value(uid) else task_ids[i]
            session_id = self._get_non_tensor_value(batch.non_tensor_batch, "session_id", i)
            try:
                session_rank = int(session_id) if self._has_value(session_id) else -1
            except (TypeError, ValueError):
                session_rank = -1

            key = (group_key, int(rollout_no))
            prev = selected_by_key.get(key)
            if prev is None or session_rank >= prev[0]:
                selected_by_key[key] = (session_rank, i)

        return sorted(idx for _, idx in selected_by_key.values())

    def _dump_trajectories(
        self,
        *,
        batch: DataProto,
        inputs: list[str],
        outputs: list[str],
        scores: list[float],
        dump_path: str,
    ) -> None:
        os.makedirs(dump_path, exist_ok=True)

        n = len(outputs)
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
        task_ids = [self._extract_task_id(batch, i) for i in range(n)]
        rollout_numbers = self._compute_rollout_numbers(batch, task_ids)
        selected_rows = set(self._select_final_rollout_rows(batch, rollout_numbers))

        for i in range(n):
            # Prefer structured agent_messages from the agent loop (proper multi-turn format)
            agent_messages = self._get_non_tensor_value(batch.non_tensor_batch, "agent_messages", i)
            if agent_messages is not None and isinstance(agent_messages, (list, tuple)) and len(agent_messages) > 0:
                if isinstance(agent_messages, tuple):
                    agent_messages = list(agent_messages)
                messages_with_assistant = list(agent_messages)
            else:
                # Fallback: reconstruct from raw_prompt + decoded flat output
                messages = []
                if raw_prompts is not None:
                    raw_prompt = self._get_non_tensor_value(batch.non_tensor_batch, "raw_prompt", i)
                    if isinstance(raw_prompt, np.ndarray):
                        raw_prompt = raw_prompt.tolist()
                    if isinstance(raw_prompt, tuple):
                        raw_prompt = list(raw_prompt)
                    if isinstance(raw_prompt, list):
                        for msg in raw_prompt:
                            if isinstance(msg, dict) and ("role" in msg) and ("content" in msg):
                                messages.append({"role": str(msg.get("role")), "content": msg.get("content")})

                if not messages:
                    messages = [{"role": "user", "content": inputs[i]}]

                messages_with_assistant = deepcopy(messages)
                messages_with_assistant.append({"role": "assistant", "content": outputs[i]})

            # For no_memory version, prefer full_messages so the final saved rollout
            # contains the full cross-session history without condenser state.
            full_msgs = self._get_non_tensor_value(batch.non_tensor_batch, "full_messages", i)
            if full_msgs is not None and isinstance(full_msgs, (list, tuple)) and len(full_msgs) > 0:
                if isinstance(full_msgs, tuple):
                    full_msgs = list(full_msgs)
                no_memory_messages = self._remove_prev_state_from_messages(list(full_msgs))
            else:
                no_memory_messages = self._remove_prev_state_from_messages(deepcopy(messages_with_assistant))

            entry = {
                "step": self.global_steps,
                "task_id": task_ids[i],
                "rollout_number": rollout_numbers[i],
                "rollout_index": self._get_non_tensor_value(batch.non_tensor_batch, "rollout_index", i),
                "trajectory_rollout_n": self._get_non_tensor_value(batch.non_tensor_batch, "trajectory_rollout_n", i),
                "score": scores[i],
                "uid": self._get_non_tensor_value(batch.non_tensor_batch, "uid", i),
                "session_id": self._get_non_tensor_value(batch.non_tensor_batch, "session_id", i),
                "num_sessions": self._get_non_tensor_value(batch.non_tensor_batch, "num_sessions", i),
                "prev_state": self._get_non_tensor_value(batch.non_tensor_batch, "memory_state", i),
                "messages": messages_with_assistant,
            }
            no_memory_entry = deepcopy(entry)
            no_memory_entry["prev_state"] = None
            no_memory_entry["messages"] = no_memory_messages

            step_dir = f"step_{self.global_steps}"
            task_dir = f"task_{task_ids[i]}"
            rollout_dir = f"rollout_{rollout_numbers[i]}"
            output_dir = os.path.join(dump_path, step_dir, task_dir, rollout_dir)
            os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(output_dir, "trajectories.jsonl"), "a") as f:
                f.write(json.dumps(self._jsonable(entry), ensure_ascii=False) + "\n")
            if i in selected_rows:
                with open(os.path.join(output_dir, "trajectories_no_memory.jsonl"), "a") as f:
                    f.write(json.dumps(self._jsonable(no_memory_entry), ensure_ascii=False) + "\n")

    @staticmethod
    def _coerce_filter_metric_scalar(value, metric_name: str) -> float:
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(
                    f"Filter metric {metric_name} must be scalar per trajectory, but got array with shape {value.shape}"
                )
            value = value.item()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Filter metric {metric_name} must be numeric, got {type(value)} with value {value!r}") from exc

    @staticmethod
    def _coerce_binary_flag(value) -> bool:
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return False
            if value.size == 1:
                value = value.item()
            else:
                return bool(np.any(value))
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) != 0.0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return False

    def _drop_flagged_samples(
        self, batch: DataProto, flag_key: str = "drop_from_training"
    ) -> tuple[DataProto, dict[str, int]]:
        stats = {
            "dropped_prompt_groups": 0,
            "dropped_trajs": 0,
        }
        flag_values = batch.non_tensor_batch.get(flag_key, None)
        if flag_values is None:
            return batch, stats

        if len(flag_values) != len(batch):
            print(
                f"[WARN] Ignore {flag_key}: expected {len(batch)} flags but got {len(flag_values)}."
            )
            return batch, stats

        keep_idxs = []
        dropped_idxs = []
        for idx, flag in enumerate(flag_values):
            if self._coerce_binary_flag(flag):
                dropped_idxs.append(idx)
            else:
                keep_idxs.append(idx)

        if not dropped_idxs:
            return batch, stats

        stats["dropped_trajs"] = len(dropped_idxs)
        if "uid" in batch.non_tensor_batch and len(batch.non_tensor_batch["uid"]) == len(batch):
            stats["dropped_prompt_groups"] = len({batch.non_tensor_batch["uid"][idx] for idx in dropped_idxs})
        else:
            stats["dropped_prompt_groups"] = len(dropped_idxs)

        filtered_batch = batch[keep_idxs] if keep_idxs else batch[:0]
        return filtered_batch, stats

    def _resolve_filter_prompt_bsz(self) -> int:
        # Fully async has train_batch_size=0 by design.
        # Keep legacy filtering behavior via env override if present.
        env_bsz = os.getenv("DEEPRESEARCH_FILTER_PROMPT_BSZ")
        if env_bsz:
            try:
                val = int(env_bsz)
                if val > 0:
                    return val
            except ValueError:
                pass
        cfg_bsz = int(self.config.data.get("train_batch_size", 0))
        if cfg_bsz > 0:
            return cfg_bsz
        return int(self.required_samples)

    def _apply_filter_groups(
        self,
        new_batch: DataProto,
        accumulated_batch: DataProto | None,
        num_prompt_in_batch: int,
        num_gen_batches: int,
    ) -> tuple[DataProto, int, bool, dict[str, int]]:
        filter_cfg = self.config.algorithm.get("filter_groups", None)
        if not filter_cfg or not filter_cfg.enable:
            return (
                new_batch,
                num_prompt_in_batch,
                False,
                {
                    "total_prompt_groups": 0,
                    "kept_prompt_groups": 0,
                    "total_trajs": 0,
                    "kept_trajs": 0,
                },
            )

        metric_name = filter_cfg.metric
        if not metric_name:
            raise ValueError("algorithm.filter_groups.metric must be set when filter_groups is enabled.")

        if metric_name == "seq_final_reward":
            if "token_level_rewards" not in new_batch.batch:
                raise ValueError(
                    "metric=seq_final_reward requires token_level_rewards in batch. "
                    "For DeepResearch, use metric=score/acc/seq_reward or disable use_kl_in_reward."
                )
            new_batch.non_tensor_batch["seq_final_reward"] = (
                new_batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
            )
        elif metric_name == "seq_reward":
            if "token_level_scores" not in new_batch.batch:
                raise ValueError("metric=seq_reward requires token_level_scores in batch.")
            new_batch.non_tensor_batch["seq_reward"] = (
                new_batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()
            )

        if "uid" not in new_batch.non_tensor_batch:
            raise KeyError("uid not found in non_tensor_batch; filter_groups requires per-prompt uid.")
        if metric_name not in new_batch.non_tensor_batch:
            available = sorted(new_batch.non_tensor_batch.keys())
            raise KeyError(f"filter_groups.metric={metric_name!r} not found. Available keys: {available}")

        prompt_uid2metric_vals = defaultdict(list)
        for uid, metric_val in zip(
            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
        ):
            prompt_uid2metric_vals[uid].append(self._coerce_filter_metric_scalar(metric_val, metric_name))

        prompt_uid2metric_std = {
            prompt_uid: np.std(metric_vals) for prompt_uid, metric_vals in prompt_uid2metric_vals.items()
        }
        kept_prompt_uids = {
            uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
        }

        filter_stats = {
            "total_prompt_groups": len(prompt_uid2metric_vals),
            "kept_prompt_groups": len(kept_prompt_uids),
            "total_trajs": len(new_batch.non_tensor_batch["uid"]),
            "kept_trajs": 0,
        }
        num_prompt_in_batch += len(kept_prompt_uids)

        kept_traj_idxs = [
            idx for idx, traj_uid in enumerate(new_batch.non_tensor_batch["uid"]) if traj_uid in kept_prompt_uids
        ]
        filter_stats["kept_trajs"] = len(kept_traj_idxs)

        filtered_batch = new_batch[kept_traj_idxs]
        batch = filtered_batch if accumulated_batch is None else DataProto.concat([accumulated_batch, filtered_batch])

        prompt_bsz = self._resolve_filter_prompt_bsz()
        if num_prompt_in_batch < prompt_bsz:
            max_num_gen_batches = filter_cfg.max_num_gen_batches
            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                print(f"{num_prompt_in_batch=} < {prompt_bsz=}, {num_gen_batches=}. Keep generating...")
                return batch, num_prompt_in_batch, True, filter_stats
            raise ValueError(
                f"{num_gen_batches=} >= {max_num_gen_batches=}. "
                "Generated too many batches while filtering groups. "
                "Please check if rewards are too homogeneous or set max_num_gen_batches=0 for unlimited retries."
            )

        traj_bsz = prompt_bsz * self.config.actor_rollout_ref.rollout.n
        batch = batch[:traj_bsz]
        return batch, num_prompt_in_batch, False, filter_stats

    def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """
        Get samples from message queue and compose gen_batch_output
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        # Collect samples using a simple loop calling get_sample
        consumer_start = time.time()
        queue_samples = []
        queue_len = 0
        while len(queue_samples) < self.required_samples:
            # Get a single sample and wait until there is a sample or None is received
            sample, queue_len = self.message_queue_client.get_sample_sync()

            if sample is None:
                # None is sent by Rollouter in fully_async_rollouter._streaming_generation_main() after
                # _feed_samples() and _processor_worker() finish. Getting it before any real sample usually
                # means: train_dataloader was empty (0 batches), so _feed_samples() put 0 samples then "DONE".
                print(
                    f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                    f"Collected {len(queue_samples)}/{self.required_samples} samples"
                )
                break

            queue_samples.append(sample)

            if len(queue_samples) % 64 == 0:
                print(
                    f"[FullyAsyncTrainer] Collected {len(queue_samples)}/{self.required_samples} samples. "
                    f"mq_len: {queue_len}"
                )

        consumer_end = time.time()

        if not queue_samples or len(queue_samples) < self.required_samples:
            print("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None
        total_wait_time = consumer_end - consumer_start

        print(
            f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, "
            f"total wait time: {total_wait_time:.2f} seconds."
            f"mq_len: {queue_len}"
        )

        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        # Assemble batch - balance is applied later after reward/filtering.
        batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        return 0, batch

    def _pad_and_balance_batch_for_dp(self, batch: DataProto, do_balance: bool = True) -> DataProto:
        """Pad to DP divisibility and (optionally) balance batch across DP ranks.

        Padding always runs to ensure batch size is divisible by the minibatch iterator's
        expected divisor.  Balance is only applied when do_balance=True.
        """
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size", None)
        rollout_n = int(self.config.actor_rollout_ref.rollout.get("n", 1))
        if ppo_mini_batch_size and int(ppo_mini_batch_size) > 0:
            mini_batch_total = int(ppo_mini_batch_size) * rollout_n
            # Megatron normalizes per-rank mini_batch via floor division:
            #   ppo_mini_batch_size = (ppo_mini_batch_size * rollout_n) // dp_size
            # The make_iterator assertion is: per_rank_batch % per_rank_mini == 0
            # which requires: total % (dp_size * per_rank_mini) == 0.
            # lcm(dp_size, M) over-pads when dp_size does not divide M
            # (e.g. dp=3, M=128 -> lcm=384, per_rank=128, but 128%(128//3=42)!=0).
            per_rank_mini = mini_batch_total // dp_size
            pad_divisor = dp_size * per_rank_mini if per_rank_mini > 0 else dp_size
        else:
            mini_batch_total = None
            pad_divisor = dp_size

        batch_size_before_dp_pad = len(batch)
        dp_pad_size = 0
        # Mark all real rows before padding so we can find them after _balance_batch reorders.
        batch.batch["__is_pad__"] = torch.zeros(batch_size_before_dp_pad, dtype=torch.bool)
        if batch_size_before_dp_pad % pad_divisor != 0:
            # pad_dataproto_to_divisor internally calls DataProto.concat, which merges meta_info
            # via `==`. Some meta_info fields are numpy arrays, and array equality is ambiguous.
            # Build a meta-less view for padding, then restore original meta_info.
            original_meta_info = dict(batch.meta_info)
            batch_for_pad = type(batch)(batch=batch.batch, non_tensor_batch=batch.non_tensor_batch, meta_info={})
            batch, dp_pad_size = pad_dataproto_to_divisor(batch_for_pad, pad_divisor)
            batch.meta_info.update(original_meta_info)
            batch_size_after_dp_pad = len(batch)
            print(
                "[WARN] Padded train batch for DP/minibatch divisibility: "
                f"original={batch_size_before_dp_pad}, dp_size={dp_size}, pad_divisor={pad_divisor}, "
                f"mini_batch_total={mini_batch_total}, "
                f"pad_size={dp_pad_size}, padded={batch_size_after_dp_pad}"
            )
            pad_row_slice = slice(batch_size_before_dp_pad, batch_size_after_dp_pad)
            # Ensure padded rows never contribute to optimization.
            batch.batch["__is_pad__"][pad_row_slice] = True
            if "response_mask" in batch.batch:
                batch.batch["response_mask"][pad_row_slice] = 0
            # Keep padded rows forward-valid so Megatron DP replicas preserve the same
            # micro-batch schedule; zero response/loss tensors make them no-op samples.
            for tensor_key in ("token_level_scores", "token_level_rewards", "session_mask", "session_row_weight"):
                if tensor_key in batch.batch:
                    batch.batch[tensor_key][pad_row_slice] = 0
        else:
            batch_size_after_dp_pad = batch_size_before_dp_pad

        if do_balance:
            self._balance_batch(batch, metrics={})
        batch.meta_info["fully_async/dp_size"] = int(dp_size)
        batch.meta_info["fully_async/pad_divisor"] = int(pad_divisor)
        batch.meta_info["fully_async/dp_pad_size"] = int(dp_pad_size)
        batch.meta_info["fully_async/batch_size_before_dp_pad"] = int(batch_size_before_dp_pad)
        batch.meta_info["fully_async/batch_size_after_dp_pad"] = int(batch_size_after_dp_pad)
        return batch

    def _create_actor_rollout_classes(self):
        # create actor
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        # self._init_async_objects()
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        self._init_reward_loop()
        await self._init_async_rollout_manager()
        rollout_replicas = []
        if getattr(self, "async_rollout_manager", None) is not None:
            rollout_replicas = self.async_rollout_manager.rollout_replicas
        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=rollout_replicas,
        )
        print(
            "[FullyAsyncTrainer] checkpoint_manager initialized "
            f"backend={self.config.actor_rollout_ref.rollout.checkpoint_engine.backend} "
            f"replicas={len(rollout_replicas)}"
        )

    async def _init_async_rollout_manager(self):
        # use async rollout do validate
        print(f"[FullyAsyncTrainer] use_trainer_do_validate: {self.config.async_training.use_trainer_do_validate}")
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] Init async rollout manager")

            # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
            # agent_reward_loop: streaming reward computation with actor rollout
            # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
            enable_agent_reward_loop = self.use_reward_loop and (
                not self.use_rm or self.config.reward_model.enable_resource_pool
            )
            # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
            # to stream reward computation with actor rollout
            reward_loop_worker_handles = None
            if enable_agent_reward_loop:
                if self.reward_loop_manager is None:
                    raise RuntimeError(
                        "reward_loop_manager is not initialized. "
                        "Call _init_reward_loop() before _init_async_rollout_manager()."
                    )
                reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers

            # create async rollout manager and request scheduler
            assert self.config.actor_rollout_ref.rollout.mode == "async"
            from verl.experimental.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                reward_loop_worker_handles=reward_loop_worker_handles,
            )

            print("[FullyAsyncTrainer] async_rollout_manager sleep")
            await self.async_rollout_manager.sleep()
        else:
            print("[FullyAsyncTrainer] Skip async rollout manager (use_trainer_do_validate=False)")

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.rollouter is None:
            raise ValueError("rollouter client not set. Call set_rollouter() first.")

        filter_cfg = self.config.algorithm.get("filter_groups", None)
        filter_enabled = bool(filter_cfg and filter_cfg.enable)
        if filter_enabled and (not self.use_reward_loop) and self.config.reward_model.launch_reward_fn_async:
            raise ValueError(
                "DeepResearch filter_groups does not support reward_model.launch_reward_fn_async=True "
                "when use_reward_loop=False. Please disable launch_reward_fn_async."
            )

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.max_steps_duration = 0

        # get validate data before training
        self._log_validation_data()

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                print("[FullyAsyncTrainer] Training stopped by queue termination signal")
                break

        # final parameter sync and validate
        # 1. waiting remaining validate task
        self._wait_last_rollouter_update()
        self._log_validation_data()
        # 2. perform addtional parameter_sync and validate if trainer already updated
        test_freq = int(self.config.rollout.test_freq)
        needs_final_sync = self.local_trigger_step > 1 or (
            test_freq > 0 and self.current_param_version % test_freq != 0
        )
        if needs_final_sync:
            await self._trigger_parameter_sync_after_step(validate=test_freq > 0)
            self._wait_last_rollouter_update()
            self._log_validation_data()
        self.progress_bar.close()
        await self._fit_save_checkpoint()

    async def fit_step(self, batch_dict: dict = None):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_dict: Raw data dictionary
        """
        print("[FullyAsyncTrainer] fit_step")
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        filter_cfg = self.config.algorithm.get("filter_groups", None)
        filter_enabled = bool(filter_cfg and filter_cfg.enable)

        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            if filter_enabled:
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                filter_total_prompt_groups = 0
                filter_kept_prompt_groups = 0
                filter_total_trajs = 0
                filter_kept_trajs = 0
                filter_dropped_prompt_groups = 0
                filter_dropped_trajs = 0

                while True:
                    new_batch = self._fit_generate(None)
                    new_batch = self._fit_compute_reward(new_batch)
                    new_batch, drop_stats = self._drop_flagged_samples(new_batch, flag_key="drop_from_training")
                    filter_dropped_prompt_groups += drop_stats["dropped_prompt_groups"]
                    filter_dropped_trajs += drop_stats["dropped_trajs"]
                    if len(new_batch) == 0:
                        if drop_stats["dropped_trajs"] > 0:
                            print(
                                "[WARN] All trajectories in this generation batch were dropped "
                                "due to eval script load failures; fetching another batch."
                            )
                        continue
                    num_gen_batches += 1
                    batch, num_prompt_in_batch, need_more_generation, filter_stats = self._apply_filter_groups(
                        new_batch=new_batch,
                        accumulated_batch=batch,
                        num_prompt_in_batch=num_prompt_in_batch,
                        num_gen_batches=num_gen_batches,
                    )
                    filter_total_prompt_groups += filter_stats["total_prompt_groups"]
                    filter_kept_prompt_groups += filter_stats["kept_prompt_groups"]
                    filter_total_trajs += filter_stats["total_trajs"]
                    filter_kept_trajs += filter_stats["kept_trajs"]
                    if not need_more_generation:
                        break

                self.metrics.update(
                    {
                        "filter_groups/num_gen_batches": num_gen_batches,
                        "filter_groups/total_prompt_groups": filter_total_prompt_groups,
                        "filter_groups/kept_prompt_groups": filter_kept_prompt_groups,
                        "filter_groups/total_trajs": filter_total_trajs,
                        "filter_groups/kept_trajs": filter_kept_trajs,
                        "filter_groups/kept_prompt_ratio": (
                            filter_kept_prompt_groups / filter_total_prompt_groups
                            if filter_total_prompt_groups > 0
                            else 0.0
                        ),
                        "filter_groups/kept_traj_ratio": (
                            filter_kept_trajs / filter_total_trajs if filter_total_trajs > 0 else 0.0
                        ),
                        "filter_groups/prompt_bsz_target": self._resolve_filter_prompt_bsz(),
                        "filter_groups/dropped_prompt_groups_eval_script": filter_dropped_prompt_groups,
                        "filter_groups/dropped_trajs_eval_script": filter_dropped_trajs,
                    }
                )
            else:
                batch = self._fit_generate(None)
                batch = self._fit_compute_reward(batch)

            reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
            if reward_extra_keys:
                self.reward_extra_infos_dict = {
                    key: batch.non_tensor_batch[key] for key in reward_extra_keys if key in batch.non_tensor_batch
                }

            # Dump data BEFORE padding/balancing to avoid losing real rows.
            # _pad_and_balance_batch_for_dp reorders all rows, then unpad_dataproto
            # blindly removes the last dp_pad_size rows which may include real rollout rows.
            self._fit_dump_data(batch)

            # Padding always runs to ensure divisibility by the minibatch iterator.
            # Balance is only applied when balance_batch=True.
            batch = self._pad_and_balance_batch_for_dp(
                batch, do_balance=self.config.trainer.balance_batch
            )
            self.metrics.update(
                {
                    "training/dp_size": batch.meta_info.get("fully_async/dp_size", 0),
                    "training/dp_pad_size": batch.meta_info.get("fully_async/dp_pad_size", 0),
                    "training/batch_size_before_dp_pad": batch.meta_info.get(
                        "fully_async/batch_size_before_dp_pad", len(batch)
                    ),
                    "training/batch_size_after_dp_pad": batch.meta_info.get(
                        "fully_async/batch_size_after_dp_pad", len(batch)
                    ),
                }
            )

            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            await self._fit_update_weights()
            dp_pad_size = int(batch.meta_info.get("fully_async/dp_pad_size", 0) or 0)
            if dp_pad_size > 0:
                # Use the __is_pad__ sentinel to remove pad rows by index rather than
                # tail-trimming, since _balance_batch may have reordered the rows.
                if "__is_pad__" in batch.batch.keys():
                    is_pad = batch.batch["__is_pad__"]
                    real_indices = (~is_pad).nonzero(as_tuple=True)[0].tolist()
                    batch = batch[real_indices]
                else:
                    batch = unpad_dataproto(batch, pad_size=dp_pad_size)
            # Clean up sentinel column if present.
            if "__is_pad__" in batch.batch.keys():
                batch.batch.pop("__is_pad__")

        await self._fit_save_checkpoint()
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_update_curriculum(batch)
        self._fit_torch_memory()
        self._fit_postprocess_step()

    def _fit_generate(self, batch: DataProto = None) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: queue returned None")
            self._collect_metrics_from_samples(batch, metrics)
        return batch

    def _fit_compute_reward(self, batch: DataProto) -> DataProto:
        timing_raw = self.timing_raw
        with marked_timer("reward", timing_raw, color="yellow"):
            if self.use_rm and "rm_scores" not in batch.batch.keys():
                batch_reward = self._compute_reward_colocate(batch)
                batch = batch.union(batch_reward)

            if not self.use_reward_loop:
                if self.config.reward_model.launch_reward_fn_async:
                    self.future_reward = compute_reward_async.remote(
                        data=batch, config=self.config, tokenizer=self.tokenizer
                    )
                    self.reward_tensor = None
                    self.reward_extra_infos_dict = {}
                else:
                    self.reward_tensor, self.reward_extra_infos_dict = self._compute_reward_legacy(
                        batch, reward_fn=self.reward_fn, reward_for_val=False
                    )
            else:
                self.reward_tensor = batch.batch["rm_scores"]
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                self.reward_extra_infos_dict = {
                    key: batch.non_tensor_batch[key] for key in reward_extra_keys if key in batch.non_tensor_batch
                }

            # For filtering and rollout-data dumping, eagerly attach available reward fields.
            if self.use_reward_loop or not self.config.reward_model.launch_reward_fn_async:
                batch.batch["token_level_scores"] = self.reward_tensor
                if self.reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in self.reward_extra_infos_dict.items()})
                if not self.config.algorithm.use_kl_in_reward:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        return batch

    def _fit_compute_advantage(self, batch: DataProto) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw

        with marked_timer("adv", timing_raw, color="brown"):
            reward_extra_infos_dict: dict[str, list]

            if "token_level_scores" not in batch.batch:
                if self.config.reward_model.launch_reward_fn_async:
                    reward_tensor, reward_extra_infos_dict = ray.get(self.future_reward)
                else:
                    reward_tensor = self.reward_tensor
                    reward_extra_infos_dict = self.reward_extra_infos_dict
                batch.batch["token_level_scores"] = reward_tensor
                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                self.reward_extra_infos_dict = reward_extra_infos_dict
            else:
                reward_extra_infos_dict = self.reward_extra_infos_dict

            if self.config.algorithm.use_kl_in_reward:
                from verl.trainer.ppo.ray_trainer import apply_kl_penalty

                batch, kl_metrics = apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update(kl_metrics)
            else:
                if "token_level_rewards" not in batch.batch:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
            bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
            if rollout_corr_config is not None and "rollout_log_probs" in batch.batch and not bypass_recomputing_logprobs:
                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                metrics.update(is_metrics)

            from verl.trainer.ppo.ray_trainer import compute_advantage

            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )
        return batch

    def _compute_old_log_prob(self, batch: DataProto):
        """
        If algorithm.rollout_correction.bypass_mode is False,
        use model engine and first version model params to re-calculate old_log_prob.

        If local_trigger_step == 1, load the training engine's parameters to the CPU
          and save a copy for subsequent MIS use.

        If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
        then restore the parameters of the current version.
        """
        if self.local_trigger_step == 1:
            self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _fit_collect_metrics(self, batch):
        super()._fit_collect_metrics(batch)
        # The rollout side may expand one logical rollout into multiple session
        # rows. Override reward/score log metrics with rollout-level aggregation
        # so split sessions do not count as independent samples in W&B/console.
        self.metrics.update(self._compute_rollout_sequence_metrics(batch))
        self.metrics.update(self._compute_training_reward_metrics(batch))
        self.metrics_aggregator.add_step_metrics(
            metrics=self.metrics, sample_count=self.required_samples, timestamp=time.time()
        )
        self._log_validation_data()

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        reward_extra_infos_dict.pop("acc", None)
        super()._dump_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        from verl.utils.debug import marked_timer as _marked_timer

        with _marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault("request_id", batch.non_tensor_batch["request_id"].tolist())

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

        if self._env_flag("DEEPRESEARCH_DUMP_TRAJECTORY_JSONL", "0"):
            with _marked_timer("dump_rollout_trajectories", timing_raw, color="green"):
                self._dump_trajectories(
                    batch=batch,
                    inputs=inputs,
                    outputs=outputs,
                    scores=scores,
                    dump_path=rollout_data_dir,
                )

    async def _fit_update_weights(self):
        # with marked_timer("update_weights", self.timing_raw, color="red"):
        #     self.checkpoint_manager.update_weights()

        # Trigger parameter synchronization after training step
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
            f"local_trigger_step: {self.local_trigger_step} "
            f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
            f"{time_str}"
        )
        await self._trigger_parameter_sync_after_step()

    def _wait_last_rollouter_update(self):
        print("[FullyAsyncTrainer] Waiting last rollout update and validate...")
        start_time = time.time()
        if self.wait_last_update:
            ray.get(self.wait_last_update)
        if self.wait_last_resume:
            ray.get(self.wait_last_resume)
        if self.validate_task:
            ray.get(self.validate_task)
        self.wait_last_update = None
        self.wait_last_resume = None
        self.validate_task = None
        print(f"[FullyAsyncTrainer] Wait last rollout update cost: {time.time() - start_time:.2f} seconds")

    async def sync_rollout_with_current_version(self, validate: bool = False):
        if self.rollouter is None:
            raise ValueError("rollouter client not set. Call set_rollouter() first.")
        timing_param_sync = {}
        start_time = time.time()
        ray.get(self.rollouter.pause.remote())
        pause_time = time.time()

        rollout_name = getattr(self.config.actor_rollout_ref.rollout, "name", None)
        use_checkpoint_engine = self.config.async_training.checkpoint_engine.enable and rollout_name != "sglang"
        print(
            "[FullyAsyncTrainer] begin rollout weight sync. "
            f"rollout_name={rollout_name}, use_checkpoint_engine={use_checkpoint_engine}, "
            f"version={self.current_param_version}, pause_cost={pause_time - start_time:.2f}s"
        )
        with marked_timer("timing_s/param_sync", timing_param_sync):
            if use_checkpoint_engine:
                self.actor_wg.sync_rollout_weights_by_checkpoint(self.sync_group_name)
                ray.get(self.rollout_wg.sync_rollout_weights_by_checkpoint(self.sync_group_name))
            else:
                self.actor_wg.sync_rollout_weights(self.sync_group_name)
                ray.get(self.rollout_wg.sync_rollout_weights(self.sync_group_name))
            self.message_queue_client.update_param_version_sync(self.current_param_version)

        total_time = timing_param_sync["timing_s/param_sync"]
        print(
            "[FullyAsyncTrainer] sync_rollout_with_current_version success. "
            f"version={self.current_param_version}, total_cost={total_time:.2f}s, "
            f"sync_cost={time.time() - pause_time:.2f}s"
        )

        self.wait_last_update = self.rollouter.update_param_version.remote(
            self.current_param_version,
            validate,
            self.global_steps,
            self.config.async_training.use_trainer_do_validate,
        )
        self.wait_last_resume = self.rollouter.resume.remote(self.wait_last_update)
        return timing_param_sync

    async def _fit_save_checkpoint(self):
        timing_raw = self.timing_raw
        current_step = int(self.current_param_version)
        if current_step <= 0:
            return

        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        save_freq = int(self.config.trainer.save_freq)
        periodic_save = save_freq > 0 and current_step % save_freq == 0
        save_every_step = self._save_every_step_ckpt_enabled()
        should_save = save_every_step or periodic_save or esi_close_to_expiration
        save_formal = periodic_save or esi_close_to_expiration

        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. It's the last training step.
        # 3. The current step number is a multiple of the save frequency.
        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if should_save:
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                checkpoint_manager = getattr(self, "checkpoint_manager", None)
                if checkpoint_manager is not None:
                    print(
                        "[SAVE_STAGE=TRAINER_BEFORE_SLEEP_REPLICAS][FullyAsyncTrainer] "
                        f"backend={checkpoint_manager.backend} replicas={len(checkpoint_manager.replicas)} "
                        f"current_param_version={self.current_param_version}"
                    )
                    await checkpoint_manager.sleep_replicas()
                    print(
                        "[SAVE_STAGE=TRAINER_AFTER_SLEEP_REPLICAS][FullyAsyncTrainer] "
                        f"current_param_version={self.current_param_version}"
                    )
                else:
                    print("[SAVE_STAGE=TRAINER_SKIP_SLEEP_REPLICAS][FullyAsyncTrainer] checkpoint_manager unavailable")
                try:
                    self._save_checkpoint(formal=save_formal)
                    self._remove_previous_temporary_checkpoint(current_step)
                finally:
                    if checkpoint_manager is not None:
                        print(
                            "[SAVE_STAGE=TRAINER_BEFORE_UPDATE_WEIGHTS][FullyAsyncTrainer] "
                            f"backend={checkpoint_manager.backend} replicas={len(checkpoint_manager.replicas)} "
                            f"current_param_version={self.current_param_version}"
                        )
                        await checkpoint_manager.update_weights()
                        print(
                            "[SAVE_STAGE=TRAINER_AFTER_UPDATE_WEIGHTS][FullyAsyncTrainer] "
                            f"current_param_version={self.current_param_version}"
                        )
                    else:
                        print("[SAVE_STAGE=TRAINER_SKIP_UPDATE_WEIGHTS][FullyAsyncTrainer] checkpoint_manager unavailable")

    def _fit_update_curriculum(self, batch: DataProto):
        """Update dynamic curriculum state with reward feedback from this training step."""
        # Per-category score metrics (training)
        try:
            from recipe.deepresearch.curriculum_sampler import compute_category_score_metrics
        except ImportError:
            compute_category_score_metrics = None
        if compute_category_score_metrics is not None:
            self.metrics.update(compute_category_score_metrics(batch, prefix="train"))

        # Curriculum Q-value update
        try:
            from recipe.deepresearch.curriculum_sampler import update_curriculum_from_batch

            curriculum_metrics = update_curriculum_from_batch(batch)
            if curriculum_metrics:
                self.metrics.update(curriculum_metrics)
        except (ImportError, Exception):
            # Curriculum sampler not configured or not available - silently skip
            pass

    def _fit_postprocess_step(self):
        if self.metrics:
            console_metrics = self._select_console_metrics(self.metrics)
            if console_metrics:
                self.logger.log(data=console_metrics, step=self.current_param_version, backend=["console"])
        self.global_steps += 1

    def _save_checkpoint(self, *, formal: bool):
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        step = int(self.current_param_version)
        ckpt_dir_name = self._checkpoint_dir_name(step, formal=formal)
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, ckpt_dir_name)

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, ckpt_dir_name, "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, ckpt_dir_name, str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        if self.rollouter is None:
            raise ValueError("rollouter client not set. Call set_rollouter() first.")
        ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
        self._save_trainer_state(local_global_step_folder)
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    def _has_complete_actor_checkpoint(self, checkpoint_folder: str) -> bool:
        actor_path = os.path.join(checkpoint_folder, "actor")
        if not os.path.isdir(actor_path):
            return False

        dist_ckpt_common = os.path.join(actor_path, "dist_ckpt", "common.pt")
        if os.path.exists(dist_ckpt_common):
            return True

        adapter_ckpt_path = os.path.join(actor_path, "adapter_checkpoint")
        if os.path.isdir(adapter_ckpt_path):
            return True

        hf_ckpt_path = os.path.join(actor_path, "huggingface")
        if not os.path.isdir(hf_ckpt_path):
            return False

        for entry in os.listdir(hf_ckpt_path):
            if entry.endswith(".safetensors") or entry.endswith(".safetensors.index.json"):
                return True
        return False

    def _has_complete_async_checkpoint(self, checkpoint_folder: str) -> bool:
        return self._has_complete_actor_checkpoint(checkpoint_folder) and os.path.exists(
            os.path.join(checkpoint_folder, "data.pt")
        )

    def _trainer_state_path(self, checkpoint_folder: str) -> str:
        return os.path.join(checkpoint_folder, "trainer_state.json")

    def _save_trainer_state(self, checkpoint_folder: str) -> None:
        trainer_state = {
            # Checkpoints are written after update_weights but before _fit_postprocess_step().
            # Store the next step index so resume starts from the next fit iteration.
            "global_steps": int(self.global_steps + 1),
            "local_trigger_step": int(self.local_trigger_step),
            "current_param_version": int(self.current_param_version),
            "last_ckpt_version": int(self.last_ckpt_version),
            "epoch": int(self.epoch),
        }
        trainer_state_path = self._trainer_state_path(checkpoint_folder)
        with open(trainer_state_path, "w") as f:
            json.dump(trainer_state, f)
        print(f"[FullyAsyncTrainer] Saved trainer state to {trainer_state_path}: {trainer_state}")

    def _load_trainer_state(self, checkpoint_folder: str) -> bool:
        trainer_state_path = self._trainer_state_path(checkpoint_folder)
        if not os.path.exists(trainer_state_path):
            return False

        with open(trainer_state_path) as f:
            trainer_state = json.load(f)

        self.current_param_version = int(trainer_state["current_param_version"])
        self.global_steps = int(trainer_state["global_steps"])
        self.local_trigger_step = int(trainer_state["local_trigger_step"])
        self.last_ckpt_version = int(trainer_state.get("last_ckpt_version", self.current_param_version))
        self.epoch = int(trainer_state.get("epoch", self.epoch))
        print(f"[FullyAsyncTrainer] Loaded trainer state from {trainer_state_path}: {trainer_state}")
        return True

    def _find_latest_complete_checkpoint(self, checkpoint_folder: str) -> str | None:
        if not os.path.isdir(checkpoint_folder):
            print(f"[FullyAsyncTrainer] Checkpoint folder does not exist: {checkpoint_folder}")
            return None

        candidates = []
        for entry in os.listdir(checkpoint_folder):
            info = self._checkpoint_name_rank(entry)
            if info is not None:
                rank, step = info
                candidates.append((step, rank, os.path.join(checkpoint_folder, entry)))

        tracker_candidate = find_latest_ckpt_path(checkpoint_folder)
        if tracker_candidate:
            tracker_name = os.path.basename(os.path.normpath(tracker_candidate))
            info = self._checkpoint_name_rank(tracker_name)
            if info is not None:
                step, rank = info[1], info[0]
                candidates.append((step, rank, tracker_candidate))

        seen = set()
        ordered_candidates = []
        for step, rank, candidate in sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True):
            normalized = os.path.normpath(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered_candidates.append((step, rank, candidate))

        for _, _, candidate in ordered_candidates:
            if self._has_complete_async_checkpoint(candidate):
                print(f"[FullyAsyncTrainer] Found latest complete fully-async checkpoint: {candidate}")
                return candidate

        return None

    def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = self._find_latest_complete_checkpoint(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("[FullyAsyncTrainer] Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert self._checkpoint_name_rank(
                    os.path.basename(os.path.normpath(self.config.trainer.resume_from_path))
                ) is not None, "resume ckpt must be step_<n>, savefreq_step_<n>, or legacy global_step_<n>"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
                assert self._has_complete_async_checkpoint(global_step_folder), (
                    f"Incomplete fully-async checkpoint at {global_step_folder}"
                )
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        if not self._load_trainer_state(global_step_folder):
            # Older checkpoints do not have trainer_state.json. Fall back to the previous heuristic.
            self.current_param_version = self._extract_checkpoint_step(global_step_folder)
            self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
            self.local_trigger_step = 1
            self.last_ckpt_version = self.current_param_version
            print(
                "[FullyAsyncTrainer] trainer_state.json not found. "
                "Falling back to heuristic step reconstruction."
            )
        loaded_local_trigger_step = int(self.local_trigger_step)
        if loaded_local_trigger_step != 1:
            print(
                "[FullyAsyncTrainer] Resetting local_trigger_step from "
                f"{loaded_local_trigger_step} to 1 on resume to start a fresh trigger cycle."
            )
            self.local_trigger_step = 1

        print(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"local_trigger_step to {self.local_trigger_step}, "
            f"current_param_version to {self.current_param_version}"
        )
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")
        if self.progress_bar is not None:
            self.progress_bar.n = self.current_param_version
            self.progress_bar.refresh()

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        return self.current_param_version

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            samples_param_versions = batch.meta_info["rollout_param_versions"]
            stale_count = sum(1 for v in samples_param_versions if self.current_param_version - v >= 1)
            self.stale_samples_processed += stale_count
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_samples_processed": self.stale_samples_processed,
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value

    async def _trigger_parameter_sync_after_step(self, validate: bool = False):
        """
        Trigger parameter synchronization after training step
        This ensures rollouter always uses the latest trained parameters
        """
        if self.local_trigger_step < self.trigger_parameter_sync_step and not validate:
            self.local_trigger_step += 1
            return

        self.current_param_version += 1
        self.local_trigger_step = 1
        aggregated_metrics = self.metrics_aggregator.get_aggregated_metrics()
        if aggregated_metrics:
            self.logger.log(data=aggregated_metrics, step=self.current_param_version, backend=["console"])
        other_backends = [name for name in self.logger.logger.keys() if name != "console"]
        if other_backends:
            self.logger.log(
                data=self._rewrite_non_console_metrics(aggregated_metrics),
                step=self.current_param_version,
                backend=other_backends,
            )
        self.progress_bar.update(1)
        self.metrics_aggregator.reset()
        timing_param_sync = {}
        with marked_timer("timing_s/wait_last_valid", timing_param_sync):
            self._wait_last_rollouter_update()
        new_timing = await self.sync_rollout_with_current_version(validate=validate)
        timing_param_sync.update(new_timing)

        #  do trainer validate
        do_validate_param = (
            self.config.rollout.test_freq > 0
            and self.current_param_version % self.config.rollout.test_freq == 0
            and self.current_param_version > 0
        )
        print(f"do_validate_param: {do_validate_param}")
        if do_validate_param and self.reward_fn is not None and self.config.async_training.use_trainer_do_validate:
            print(f"[FullyAsyncTrainer] validate param version: {self.current_param_version}")
            await self._validate_process()
        else:
            self.train_val_metrics = None
        self.logger.log(data=timing_param_sync, step=self.current_param_version)

    def _log_validation_data(self):
        """
        Log validation data
        """
        val_data = self.message_queue_client.get_validate_sync()
        if not val_data:
            return

        val_metrics: ValidateMetrics = ray.cloudpickle.loads(val_data)
        if self.train_val_metrics and self.config.async_training.use_trainer_do_validate:
            # merge info
            timing_param_sync = {}
            with marked_timer("timing_s/merge_val", timing_param_sync):
                new_metrics = self._merge_validation_results(self.train_val_metrics, val_metrics.metrics)
                new_metrics = self._augment_validation_metrics(new_metrics)
            if new_metrics:
                console_metrics = self._select_console_metrics(new_metrics)
                if console_metrics:
                    self.logger.log(data=console_metrics, step=val_metrics.param_version, backend=["console"])
                other_backends = [name for name in self.logger.logger.keys() if name != "console"]
                if other_backends:
                    self.logger.log(data=new_metrics, step=val_metrics.param_version, backend=other_backends)
                pprint(
                    f"[FullyAsyncTrainer] parameter version: {val_metrics.param_version} "
                    f"Validation metrics: {new_metrics}, timing_param_sync: {timing_param_sync['timing_s/merge_val']}"
                )
                self.logger.log(data=val_metrics.timing_raw, step=val_metrics.param_version)
        else:
            if val_metrics.metrics:
                logged_metrics = self._augment_validation_metrics(val_metrics.metrics)
                console_metrics = self._select_console_metrics(logged_metrics)
                if console_metrics:
                    self.logger.log(data=console_metrics, step=val_metrics.param_version, backend=["console"])
                other_backends = [name for name in self.logger.logger.keys() if name != "console"]
                if other_backends:
                    self.logger.log(data=logged_metrics, step=val_metrics.param_version, backend=other_backends)
                pprint(
                    f"[FullyAsyncTrainer] parameter version: {val_metrics.param_version} "
                    f"Validation metrics: {logged_metrics}"
                )
        self.logger.log(data=val_metrics.timing_raw, step=val_metrics.param_version)

    async def _validate_process(self):
        if self.config.async_training.use_trainer_do_validate:
            print("[FullyAsyncTrainer] _validate_process")
            from verl.utils.profiler import marked_timer

            timing_raw = {}
            await self.async_rollout_manager.wake_up()
            with marked_timer("trainer/validate_time", timing_raw):
                self.train_val_metrics = self._validate(True)
            await self.async_rollout_manager.sleep()
            print(f"[FullyAsyncTrainer] validate timing_raw validate: {timing_raw['trainer/validate_time']}")
        else:
            self.train_val_metrics = None
            print("[FullyAsyncTrainer] _validate_process without async_rollout_manager")
