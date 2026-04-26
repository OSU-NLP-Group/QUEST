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
DeepResearch Ray Trainer extending PPO Trainer.
This trainer inherits from the base PPO trainer and adds DeepResearch-specific logic.
"""

import json
import os
import re
import shutil
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer as BaseRayPPOTrainer
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward_async
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.fs import local_mkdir_safe
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

# Re-export for backward compatibility
__all__ = [
    "RayPPOTrainer",
    "ResourcePoolManager",
    "Role",
    "apply_kl_penalty",
    "compute_advantage",
    "compute_response_mask",
]


class RayPPOTrainer(BaseRayPPOTrainer):
    """DeepResearch-specific PPO trainer that extends the base trainer.

    This trainer inherits most functionality from the base RayPPOTrainer and adds:
    - DeepResearch-specific generation batch handling
    - Full response tracking for evaluation
    - Memory state management
    """

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        reward_extra_infos_dict.pop("acc", None)
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    @staticmethod
    def _env_flag(name: str, default: str = "0") -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _parse_checkpoint_folder_name(folder_name: str):
        match = re.fullmatch(r"(savefreq_step|step|global_step)_(\d+)", folder_name)
        if match is None:
            return None
        ckpt_type, step_str = match.groups()
        step = int(step_str)
        # Prefer formal savefreq checkpoints over rolling step checkpoints.
        # Keep legacy global_step support only for explicit resume_path compatibility.
        ckpt_priority_map = {"global_step": -1, "step": 0, "savefreq_step": 1}
        ckpt_priority = ckpt_priority_map[ckpt_type]
        return step, ckpt_priority, ckpt_type

    @staticmethod
    def _jsonable(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [RayPPOTrainer._jsonable(v) for v in value.tolist()]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): RayPPOTrainer._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [RayPPOTrainer._jsonable(v) for v in value]
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

    def _save_every_step_ckpt_enabled(self) -> bool:
        return self._env_flag("DEEPRESEARCH_SAVE_EVERY_STEP_CKPT", "0")

    def _get_sync_step_span(self) -> int:
        return max(1, int(self.config.async_training.get("trigger_parameter_sync_step", 1)))

    def _get_sync_step(self, trainer_update_step: int | None = None) -> int:
        if trainer_update_step is None:
            trainer_update_step = self.global_steps
        sync_step_span = self._get_sync_step_span()
        return ((trainer_update_step - 1) // sync_step_span) + 1

    def _is_sync_step_boundary(self, trainer_update_step: int | None = None, is_last_step: bool = False) -> bool:
        if is_last_step:
            return True
        if trainer_update_step is None:
            trainer_update_step = self.global_steps
        return trainer_update_step % self._get_sync_step_span() == 0

    def _checkpoint_is_async(self) -> bool:
        checkpoint_cfg = self.config.actor_rollout_ref.actor.checkpoint
        return bool(
            getattr(checkpoint_cfg, "async_save", False)
            or ("async_save" in checkpoint_cfg and checkpoint_cfg["async_save"])
        )

    def _checkpoint_root_dir(self) -> str:
        checkpoint_root = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_root):
            checkpoint_root = os.path.join(os.getcwd(), checkpoint_root)
        return checkpoint_root

    def _remove_local_checkpoint_tree(self, checkpoint_path: str | None) -> None:
        if not checkpoint_path:
            return
        abs_path = os.path.abspath(checkpoint_path)
        print(f"Removing checkpoint tree: {abs_path}")
        shutil.rmtree(abs_path, ignore_errors=True)

    def _write_checkpoint_tracker(self, tracker_name: str, step: int) -> None:
        if self._checkpoint_is_async():
            print(f"skip write {tracker_name} when async_save is True")
            return
        tracker_path = os.path.join(self._checkpoint_root_dir(), tracker_name)
        with open(tracker_path, "w") as f:
            f.write(str(step))

    def _write_trainer_state(self, checkpoint_dir: str, checkpoint_step: int) -> None:
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        trainer_state = {
            "trainer_update_step": int(self.global_steps),
            "sync_step": int(checkpoint_step),
        }
        with open(trainer_state_path, "w") as f:
            json.dump(trainer_state, f)

    def _save_checkpoint_to_tag(self, checkpoint_tag: str, checkpoint_step: int, tracker_name: str | None = None) -> str:
        local_global_step_folder = os.path.join(self._checkpoint_root_dir(), checkpoint_tag)

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, checkpoint_tag, "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, checkpoint_step, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, checkpoint_tag, str(Role.Critic))
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, checkpoint_step, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        self._write_trainer_state(local_global_step_folder, checkpoint_step)

        if tracker_name is not None:
            self._write_checkpoint_tracker(tracker_name, checkpoint_step)

        return local_global_step_folder

    def _save_checkpoint(self, checkpoint_step: int):
        checkpoint_path = self._save_checkpoint_to_tag(
            checkpoint_tag=f"savefreq_step_{checkpoint_step}",
            checkpoint_step=checkpoint_step,
            tracker_name="latest_checkpointed_iteration.txt",
        )
        previous_step_ckpt_path = getattr(self, "_latest_step_ckpt_path", None)
        if previous_step_ckpt_path is not None:
            self._remove_local_checkpoint_tree(previous_step_ckpt_path)
            self._latest_step_ckpt_path = None
        return checkpoint_path

    def _save_latest_step_checkpoint(self, checkpoint_step: int):
        checkpoint_path = self._save_checkpoint_to_tag(
            checkpoint_tag=f"step_{checkpoint_step}",
            checkpoint_step=checkpoint_step,
            tracker_name="latest_step_checkpointed_iteration.txt",
        )
        previous_step_ckpt_path = getattr(self, "_latest_step_ckpt_path", None)
        if previous_step_ckpt_path is not None and os.path.abspath(previous_step_ckpt_path) != os.path.abspath(
            checkpoint_path
        ):
            self._remove_local_checkpoint_tree(previous_step_ckpt_path)
        self._latest_step_ckpt_path = checkpoint_path
        return checkpoint_path

    def _find_latest_local_checkpoint_path(self) -> str | None:
        checkpoint_root = self._checkpoint_root_dir()
        if not os.path.isdir(checkpoint_root):
            return None

        candidates = []
        for entry in os.listdir(checkpoint_root):
            parsed = self._parse_checkpoint_folder_name(entry)
            if parsed is None:
                continue
            _, _, ckpt_type = parsed
            if ckpt_type == "global_step":
                continue
            step, ckpt_priority, _ = parsed
            full_path = os.path.join(checkpoint_root, entry)
            if os.path.isdir(full_path):
                candidates.append((step, ckpt_priority, full_path))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[-1][2]

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")

        if self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            resume_path = self.config.trainer.resume_from_path
            if not os.path.isabs(resume_path):
                resume_path = os.path.join(os.getcwd(), resume_path)
            checkpoint_folder = resume_path
        else:
            checkpoint_folder = self._find_latest_local_checkpoint_path()
            if checkpoint_folder is None:
                print("Training from scratch")
                return 0

        folder_name = os.path.basename(checkpoint_folder.rstrip("/"))
        parsed = self._parse_checkpoint_folder_name(folder_name)
        assert parsed is not None, (
            f"resume ckpt must be named as step_<n>, savefreq_step_<n>, or legacy global_step_<n>, got {checkpoint_folder}"
        )
        step, _, ckpt_type = parsed

        print(f"Load from checkpoint folder: {checkpoint_folder}")
        trainer_state_path = os.path.join(checkpoint_folder, "trainer_state.json")
        trainer_update_step = None
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path) as f:
                trainer_state = json.load(f)
            trainer_update_step = int(trainer_state.get("trainer_update_step", 0))

        if trainer_update_step is not None and trainer_update_step > 0:
            self.global_steps = trainer_update_step
        elif ckpt_type == "global_step":
            self.global_steps = step
        else:
            self.global_steps = step * self._get_sync_step_span()
        self._latest_step_ckpt_path = checkpoint_folder if ckpt_type == "step" else None

        print(f"Setting trainer update step to {self.global_steps}")
        print(f"Resuming from {checkpoint_folder}")

        actor_path = os.path.join(checkpoint_folder, "actor")
        critic_path = os.path.join(checkpoint_folder, str(Role.Critic))
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        dataloader_local_path = os.path.join(checkpoint_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    @staticmethod
    def _to_int(value, field_name: str) -> int:
        if hasattr(value, "item"):
            value = value.item()
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be int-like, got {value!r} ({type(value)})") from exc

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
    def _compute_obj_openended_avg_score_metrics(
        cls,
        reward_metrics: dict,
        prefix: str,
        groups: list[list[int]] | None = None,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        score_obj = cls._to_float_array(reward_metrics.get("score_obj"))
        score_openended = cls._to_float_array(reward_metrics.get("score_openended"))
        is_obj = cls._to_float_array(reward_metrics.get("is_obj"))
        is_openended = cls._to_float_array(reward_metrics.get("is_openended"))

        def _add_bucket(score_arr, mask_arr, bucket_name: str) -> None:
            if score_arr is None or mask_arr is None:
                return
            n = min(score_arr.size, mask_arr.size)
            if n <= 0:
                return
            if groups is not None:
                vals = []
                for rows in groups:
                    idxs = [idx for idx in rows if idx < n and mask_arr[idx] > 0.0]
                    if not idxs:
                        continue
                    selected = score_arr[idxs]
                    selected = selected[np.isfinite(selected)]
                    if selected.size > 0:
                        vals.append(float(np.mean(selected)))
                metrics[f"{prefix}/{bucket_name}_count"] = float(len(vals))
                if vals:
                    metrics[f"{prefix}/{bucket_name}_avg_score"] = float(np.mean(vals))
                return
            score_arr = score_arr[:n]
            mask_arr = mask_arr[:n]
            denom = float(np.sum(mask_arr))
            metrics[f"{prefix}/{bucket_name}_count"] = denom
            if denom > 0.0:
                metrics[f"{prefix}/{bucket_name}_avg_score"] = float(np.sum(score_arr) / denom)

        _add_bucket(score_obj, is_obj, "obj")
        _add_bucket(score_openended, is_openended, "openended")
        return metrics

    @classmethod
    def _compute_masked_metric_stats(
        cls,
        values,
        mask,
        *,
        prefix: str,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
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
        cls,
        values,
        mask,
        groups: list[list[int]],
        *,
        prefix: str,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
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
        metrics: dict[str, float] = {}
        groups = cls._rollout_metric_groups(batch)
        if not groups or "token_level_scores" not in batch.batch:
            return metrics

        sequence_score = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()
        grouped_score = cls._group_metric_values(sequence_score, groups)
        if grouped_score is not None and grouped_score.size > 0:
            metrics["critic/score/mean"] = float(np.mean(grouped_score))
            metrics["critic/score/max"] = float(np.max(grouped_score))
            metrics["critic/score/min"] = float(np.min(grouped_score))

        if "token_level_rewards" in batch.batch:
            sequence_reward = batch.batch["token_level_rewards"].sum(-1).detach().cpu().numpy()
            grouped_reward = cls._group_metric_values(sequence_reward, groups)
            if grouped_reward is not None and grouped_reward.size > 0:
                metrics["critic/rewards/mean"] = float(np.mean(grouped_reward))
                metrics["critic/rewards/max"] = float(np.max(grouped_reward))
                metrics["critic/rewards/min"] = float(np.min(grouped_reward))

        return metrics

    @classmethod
    def _compute_training_reward_metrics(cls, batch: DataProto) -> dict[str, float]:
        metrics: dict[str, float] = {}
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
        citation_applied = cls._group_metric_values(reward_metrics.get("citation_score_applied"), groups)
        if citation_applied is not None and citation_applied.size > 0:
            metrics["critic/score_citation_applied_rate"] = float(np.mean(citation_applied))
        return metrics

    @classmethod
    def _compute_validation_reward_metrics(cls, reward_metrics: dict) -> dict[str, float]:
        metrics: dict[str, float] = {}
        score = cls._to_float_array(reward_metrics.get("score"))
        if score is not None and score.size > 0:
            clipped_acc = np.maximum(score, 0.0)
            metrics["val/critic/acc/mean"] = float(np.mean(clipped_acc))
            metrics["val/critic/acc/max"] = float(np.max(clipped_acc))
            metrics["val/critic/acc/min"] = float(np.min(clipped_acc))

        metrics.update(
            cls._compute_masked_metric_stats(
                reward_metrics.get("score_obj"),
                reward_metrics.get("is_obj"),
                prefix="val/critic/base_score_obj",
            )
        )
        metrics.update(
            cls._compute_masked_metric_stats(
                reward_metrics.get("score_openended"),
                reward_metrics.get("is_openended"),
                prefix="val/critic/base_score_openended",
            )
        )
        metrics.update(
            cls._compute_masked_metric_stats(
                reward_metrics.get("score_citation"),
                np.ones_like(cls._to_float_array(reward_metrics.get("score_citation")))
                if cls._to_float_array(reward_metrics.get("score_citation")) is not None
                else None,
                prefix="val/critic/score_citation",
            )
        )
        citation_applied = cls._to_float_array(reward_metrics.get("citation_score_applied"))
        if citation_applied is not None and citation_applied.size > 0:
            metrics["val/critic/score_citation_applied_rate"] = float(np.mean(citation_applied))
        return metrics

    @staticmethod
    def _is_openended_ground_truth(ground_truth) -> bool:
        if not isinstance(ground_truth, dict):
            return False
        return str(ground_truth.get("type", "")).strip().lower() in {"openended", "open-ended"}

    @classmethod
    def _compute_obj_category_avg_score_metrics_from_batch(
        cls,
        batch: DataProto,
        *,
        scores: list[float] | None = None,
        prefix: str,
        groups: list[list[int]] | None = None,
    ) -> dict[str, float]:
        try:
            from recipe.deepresearch.curriculum_sampler import CATEGORIES, _get_category as _get_cat
        except ImportError:
            return {}

        if scores is None:
            if "token_level_scores" not in batch.batch:
                return {}
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

        if len(scores) != len(batch):
            return {}

        obj_cat_scores: dict[str, list[float]] = defaultdict(list)
        if groups is None:
            groups = [[i] for i in range(len(batch))]

        for rows in groups:
            idxs = [idx for idx in rows if idx < len(batch)]
            if not idxs:
                continue
            i = idxs[0]
            item = batch[i]
            cat = _get_cat(item)
            if not cat:
                continue
            reward_model = item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
            if cls._is_openended_ground_truth(ground_truth):
                continue
            vals = [float(scores[idx]) for idx in idxs if idx < len(scores)]
            if vals:
                obj_cat_scores[cat].append(float(np.mean(vals)))

        metrics: dict[str, float] = {}
        for cat in CATEGORIES:
            vals = obj_cat_scores.get(cat, [])
            metrics[f"{prefix}/obj_{cat}_count"] = float(len(vals))
            if vals:
                metrics[f"{prefix}/obj_{cat}_avg_score"] = float(np.mean(vals))
        return metrics

    def _align_batch_for_rollout_union(
        self,
        batch: DataProto,
        rollout_output: DataProto,
        *,
        context: str,
    ) -> tuple[DataProto, DataProto]:
        """Align input batch rows to rollout output rows before DataProto.union().

        Session-level expansion may increase rollout_output size. When that happens,
        rollout_output.non_tensor_batch["rollout_index"] maps each expanded row back
        to the source row in `batch`.
        """
        # Keep dataset-side raw_prompt as source-of-truth. Agent-loop-side raw_prompt
        # may be semantically equivalent but structurally different (e.g., list/tuple),
        # which can trigger strict union_numpy_dict equality assertions.
        if "raw_prompt" in batch.non_tensor_batch and "raw_prompt" in rollout_output.non_tensor_batch:
            rollout_output.non_tensor_batch.pop("raw_prompt", None)

        rollout_indices = rollout_output.non_tensor_batch.get("rollout_index", None)
        if rollout_indices is None:
            if len(batch) == len(rollout_output):
                return batch, rollout_output
            raise ValueError(
                f"{context}: batch size mismatch ({len(batch)} vs {len(rollout_output)}) "
                "and rollout_index is missing in rollout output."
            )

        if len(rollout_indices) != len(rollout_output):
            raise ValueError(
                f"{context}: rollout_index length mismatch "
                f"({len(rollout_indices)} vs {len(rollout_output)})."
            )

        valid_output_rows: list[int] = []
        source_rows: list[int] = []
        dropped_rows = 0

        for out_i, raw_idx in enumerate(rollout_indices):
            src_i = self._to_int(raw_idx, "rollout_index")
            if 0 <= src_i < len(batch):
                valid_output_rows.append(out_i)
                source_rows.append(src_i)
            else:
                dropped_rows += 1

        if dropped_rows > 0:
            print(
                f"[WARN] {context}: dropped {dropped_rows} rollout rows with out-of-range rollout_index "
                f"(batch_size={len(batch)})."
            )
            rollout_output = rollout_output[valid_output_rows]

        if not source_rows:
            raise ValueError(
                f"{context}: no valid rollout_index rows remain after filtering; "
                f"batch_size={len(batch)}, rollout_size={len(rollout_output)}."
            )

        batch = batch[source_rows]
        if len(batch) != len(rollout_output):
            raise RuntimeError(
                f"{context}: alignment failed ({len(batch)} vs {len(rollout_output)})."
            )
        return batch, rollout_output

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

        # Last-resort fallback if task_id-like fields are unavailable.
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
        has_rollout_index = "rollout_index" in batch.non_tensor_batch

        for i, task_id in enumerate(task_ids):
            uid = self._get_non_tensor_value(batch.non_tensor_batch, "uid", i)
            group_key = str(uid) if self._has_value(uid) else task_id

            if has_rollout_index:
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
                                messages.append(
                                    {
                                        "role": str(msg.get("role")),
                                        "content": msg.get("content"),
                                    }
                                )

                if not messages:
                    messages = [{"role": "user", "content": inputs[i]}]

                messages_with_assistant = deepcopy(messages)
                messages_with_assistant.append({"role": "assistant", "content": outputs[i]})

            # For no_memory version, prefer full_messages (spans all sessions without condenser rewrite)
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
            with open(os.path.join(output_dir, "trajectories_no_memory.jsonl"), "a") as f:
                f.write(json.dumps(self._jsonable(no_memory_entry), ensure_ascii=False) + "\n")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk and optionally emit trajectory JSONL variants."""
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

        if self._env_flag("DEEPRESEARCH_DUMP_TRAJECTORY_JSONL", "0"):
            with marked_timer("dump_rollout_trajectories", timing_raw, color="green"):
                self._dump_trajectories(
                    batch=batch,
                    inputs=inputs,
                    outputs=outputs,
                    scores=scores,
                    dump_path=rollout_data_dir,
                )

    def _on_val_batch_complete(self, batch, inputs, outputs, scores):
        """Dump trajectory JSONL for each validation batch when enabled."""
        if not self._env_flag("DEEPRESEARCH_DUMP_TRAJECTORY_JSONL", "0"):
            return
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if not rollout_data_dir:
            return
        val_traj_dir = os.path.join(rollout_data_dir, "val")
        self._dump_trajectories(
            batch=batch,
            inputs=inputs,
            outputs=outputs,
            scores=scores,
            dump_path=val_traj_dir,
        )

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """Override: Get generation batch with DeepResearch-specific keys.

        DeepResearch modification: Includes "raw_prompt" in reward_model_keys.
        """
        reward_model_keys = (
            set({"data_source", "reward_model", "extra_info", "uid", "raw_prompt"}) & batch.non_tensor_batch.keys()
        )

        # pop those keys for generation
        batch_keys_to_pop = [k for k in ["input_ids", "attention_mask", "position_ids"] if k in batch.batch.keys()]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        # Per-category score accumulator
        _val_cat_scores: dict[str, list[float]] = defaultdict(list)
        _val_obj_cat_scores: dict[str, list[float]] = defaultdict(list)
        # Per-source score accumulator (browsecomp / hle / m2w2 etc.)
        _val_source_scores: dict[str, list[float]] = defaultdict(list)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights()

            # Unpad direct trailing duplicates first; if session expansion happened,
            # we additionally align by rollout_index below.
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            test_batch, test_output_gen_batch = self._align_batch_for_rollout_union(
                test_batch,
                test_output_gen_batch,
                context="validate_union",
            )

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Keep labels aligned with possibly expanded session rows.
            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            if not self.use_reward_loop:
                reward_tensor, reward_extra_info = self._compute_reward_legacy(
                    test_batch, reward_fn=self.val_reward_fn, reward_for_val=True
                )
            else:
                reward_tensor = test_batch.batch["rm_scores"]
                reward_extra_keys = test_batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: test_batch.non_tensor_batch[key] for key in reward_extra_keys}

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            # Accumulate per-category scores for val metrics
            try:
                from recipe.deepresearch.curriculum_sampler import _get_category as _get_cat
            except ImportError:
                _get_cat = None
            if _get_cat is not None:
                for _i, _s in enumerate(scores):
                    _item = test_batch[_i]
                    _cat = _get_cat(_item)
                    _val_cat_scores["mean"].append(float(_s))
                    if _cat:
                        _val_cat_scores[_cat].append(float(_s))
                        _reward_model = _item.non_tensor_batch.get("reward_model", {})
                        _ground_truth = _reward_model.get("ground_truth") if isinstance(_reward_model, dict) else None
                        if not self._is_openended_ground_truth(_ground_truth):
                            _val_obj_cat_scores[_cat].append(float(_s))

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            _batch_data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            data_source_lst.append(_batch_data_sources)

            # Accumulate per-source scores
            for _src, _s in zip(_batch_data_sources, scores):
                _val_source_scores[str(_src)].append(float(_s))

            # Per-batch hook for subclass extensions (e.g. trajectory dumping).
            self._on_val_batch_complete(test_batch, input_texts, output_texts, scores)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        # Compute per-category score metrics from accumulator
        _cat_score_metrics: dict[str, float] = {
            f"val/score_{k}": float(np.mean(v)) for k, v in _val_cat_scores.items() if v
        }
        _obj_cat_metrics: dict[str, float] = {}
        try:
            from recipe.deepresearch.curriculum_sampler import CATEGORIES
        except ImportError:
            CATEGORIES = []
        for _cat in CATEGORIES:
            _vals = _val_obj_cat_scores.get(_cat, [])
            _obj_cat_metrics[f"val/reward_split/obj_{_cat}_count"] = float(len(_vals))
            if _vals:
                _obj_cat_metrics[f"val/reward_split/obj_{_cat}_avg_score"] = float(np.mean(_vals))

        # Per-source score metrics
        _source_score_metrics: dict[str, float] = {}
        for _src, _vals in _val_source_scores.items():
            if _vals:
                _source_score_metrics[f"val/source/{_src}/score"] = float(np.mean(_vals))
                _source_score_metrics[f"val/source/{_src}/count"] = float(len(_vals))

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
                "_cat_scores": dict(_val_cat_scores),   # raw lists for proper merging
                "_obj_cat_scores": dict(_val_obj_cat_scores),  # raw lists for proper merging
                "_source_scores": dict(_val_source_scores),  # raw lists for proper merging
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        metrics = self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
        metrics.update(_cat_score_metrics)
        metrics.update(_obj_cat_metrics)
        metrics.update(_source_score_metrics)
        metrics.update(
            self._compute_obj_openended_avg_score_metrics(
                reward_extra_infos_dict,
                prefix="val/reward_split",
            )
        )
        return metrics

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        metrics = super()._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
        metrics.update(self._compute_validation_reward_metrics(reward_extra_infos_dict))
        return metrics

    def _merge_validation_results(self, result_a, result_b):
        """Override: merge per-category score raw lists in addition to base reward_extra_infos_dict."""
        base_metrics = super()._merge_validation_results(result_a, result_b)

        # Merge _cat_scores raw lists from both halves
        cat_scores_a = (result_a or {}).get("_cat_scores", {})
        cat_scores_b = (result_b or {}).get("_cat_scores", {})
        merged_cat: dict[str, list] = defaultdict(list)
        for k, v in cat_scores_a.items():
            merged_cat[k].extend(v if isinstance(v, list) else [v])
        for k, v in cat_scores_b.items():
            merged_cat[k].extend(v if isinstance(v, list) else [v])

        for cat, vals in merged_cat.items():
            if vals:
                base_metrics[f"val/score_{cat}"] = float(np.mean(vals))

        try:
            from recipe.deepresearch.curriculum_sampler import CATEGORIES
        except ImportError:
            CATEGORIES = []
        obj_cat_scores_a = (result_a or {}).get("_obj_cat_scores", {})
        obj_cat_scores_b = (result_b or {}).get("_obj_cat_scores", {})
        merged_obj_cat: dict[str, list] = defaultdict(list)
        for k, v in obj_cat_scores_a.items():
            merged_obj_cat[k].extend(v if isinstance(v, list) else [v])
        for k, v in obj_cat_scores_b.items():
            merged_obj_cat[k].extend(v if isinstance(v, list) else [v])
        for cat in CATEGORIES:
            vals = merged_obj_cat.get(cat, [])
            base_metrics[f"val/reward_split/obj_{cat}_count"] = float(len(vals))
            if vals:
                base_metrics[f"val/reward_split/obj_{cat}_avg_score"] = float(np.mean(vals))

        # Merge _source_scores raw lists
        source_scores_a = (result_a or {}).get("_source_scores", {})
        source_scores_b = (result_b or {}).get("_source_scores", {})
        merged_source: dict[str, list] = defaultdict(list)
        for k, v in source_scores_a.items():
            merged_source[k].extend(v if isinstance(v, list) else [v])
        for k, v in source_scores_b.items():
            merged_source[k].extend(v if isinstance(v, list) else [v])
        for src, vals in merged_source.items():
            if vals:
                base_metrics[f"val/source/{src}/score"] = float(np.mean(vals))
                base_metrics[f"val/source/{src}/count"] = float(len(vals))

        merged_reward_extra_infos_dict = {}
        merged_reward_info_a = (result_a or {}).get("reward_extra_infos_dict", {})
        merged_reward_info_b = (result_b or {}).get("reward_extra_infos_dict", {})
        for key in ("score_obj", "score_openended", "is_obj", "is_openended"):
            vals_a = merged_reward_info_a.get(key, [])
            vals_b = merged_reward_info_b.get(key, [])
            merged_reward_extra_infos_dict[key] = list(vals_a) + list(vals_b)
        base_metrics.update(
            self._compute_obj_openended_avg_score_metrics(
                merged_reward_extra_infos_dict,
                prefix="val/reward_split",
            )
        )

        return base_metrics

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # create rollout at the end so vllm can better estimate kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create reward loop manager
        self.reward_loop_manager = None
        if self.use_reward_loop:
            from verl.experimental.reward_loop import RewardLoopManager

            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
            self.reward_loop_manager = RewardLoopManager(
                config=self.config,
                rm_resource_pool=rm_resource_pool,
            )

        # create async rollout manager for DeepResearch agent loop
        self.async_rollout_mode = False
        rollout_replicas = []
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from .deepresearch_agent_loop_manager import DeepResearchAgentLoopManager

            self.async_rollout_mode = True
            enable_agent_reward_loop = self.use_reward_loop and (
                not self.use_rm or self.config.reward_model.enable_resource_pool
            )
            reward_loop_worker_handles = (
                self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
            )
            self.async_rollout_manager = DeepResearchAgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rollout_resource_pool=actor_rollout_resource_pool,
                reward_loop_worker_handles=reward_loop_worker_handles,
            )
            rollout_replicas = self.async_rollout_manager.rollout_replicas

        # Base fit() always calls self.checkpoint_manager.update_weights(); initialize it here.
        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=rollout_replicas,
        )
        self.checkpoint_manager.sleep_replicas()

    @staticmethod
    def _coerce_filter_metric_scalar(value, metric_name: str) -> float:
        """Convert filter metric values to scalar float."""
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
        """Convert common scalar flag encodings into bool."""
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
        """Drop trajectories flagged by reward logic as unusable for training."""
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

    def _apply_filter_groups(
        self,
        new_batch: DataProto,
        accumulated_batch: DataProto | None,
        num_prompt_in_batch: int,
        num_gen_batches: int,
    ) -> tuple[DataProto, int, bool, dict[str, int]]:
        """Apply DAPO-style group filtering on trajectories.

        Returns:
            (batch_to_use, updated_prompt_count, should_continue_generating, filter_stats)
        """
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
            idx for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]) if traj_from_prompt_uid in kept_prompt_uids
        ]
        filter_stats["kept_trajs"] = len(kept_traj_idxs)
        filtered_batch = new_batch[kept_traj_idxs]
        batch = filtered_batch if accumulated_batch is None else DataProto.concat([accumulated_batch, filtered_batch])

        prompt_bsz = self.config.data.train_batch_size
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

    def fit(self):
        """
        PPO training loop with optional DAPO-style filter_groups support.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        filter_cfg = self.config.algorithm.get("filter_groups", None)
        filter_enabled = bool(filter_cfg and filter_cfg.enable)
        if filter_enabled and (not self.use_reward_loop) and self.config.reward_model.launch_reward_fn_async:
            raise ValueError(
                "DeepResearch filter_groups does not support reward_model.launch_reward_fn_async=True "
                "when use_reward_loop=False. Please disable launch_reward_fn_async."
            )

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # filter_groups accumulators
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        filter_total_prompt_groups = 0
        filter_kept_prompt_groups = 0
        filter_total_trajs = 0
        filter_kept_trajs = 0
        filter_dropped_prompt_groups = 0
        filter_dropped_trajs = 0

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}
                need_more_generation = False
                dp_size = None
                dp_pad_size = 0
                batch_size_before_dp_pad = 0
                batch_size_after_dp_pad = 0

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                new_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(new_batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                if curr_step_profile:
                                    self.async_rollout_manager.start_profile()
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                self.checkpoint_manager.sleep_replicas()
                                if curr_step_profile:
                                    self.async_rollout_manager.stop_profile()
                            new_batch, gen_baseline_output = self._align_batch_for_rollout_union(
                                new_batch,
                                gen_baseline_output,
                                context="remax_baseline_union",
                            )
                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(new_batch)
                                new_batch = new_batch.union(batch_reward)

                            # Compute or extract reward for REMAX baseline
                            if not self.use_reward_loop:
                                reward_baseline_tensor = self._compute_reward_legacy(
                                    new_batch, reward_fn=self.reward_fn, sum_reward=True
                                )
                            else:
                                reward_baseline_tensor = new_batch.batch["rm_scores"].sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch, gen_batch_output = self._align_batch_for_rollout_union(
                        new_batch,
                        gen_batch_output,
                        context="train_union",
                    )
                    new_batch = new_batch.union(gen_batch_output)

                    if "response_mask" not in new_batch.batch.keys():
                        new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(new_batch)
                            new_batch = new_batch.union(batch_reward)

                        # Compute or extract reward_tensor and reward_extra_infos_dict for training
                        if not self.use_reward_loop:
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(
                                    data=new_batch, config=self.config, tokenizer=self.tokenizer
                                )
                            else:
                                reward_tensor, reward_extra_infos_dict = self._compute_reward_legacy(
                                    new_batch, reward_fn=self.reward_fn, reward_for_val=False
                                )
                        else:
                            reward_tensor = new_batch.batch["rm_scores"]
                            reward_extra_keys = new_batch.meta_info.get("reward_extra_keys", [])
                            reward_extra_infos_dict = {key: new_batch.non_tensor_batch[key] for key in reward_extra_keys}

                    # For non-async reward path, attach score fields now so filtering can use them.
                    if self.use_reward_loop or not self.config.reward_model.launch_reward_fn_async:
                        new_batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        if not self.config.algorithm.use_kl_in_reward:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if filter_enabled:
                        new_batch, drop_stats = self._drop_flagged_samples(new_batch, flag_key="drop_from_training")
                        filter_dropped_prompt_groups += drop_stats["dropped_prompt_groups"]
                        filter_dropped_trajs += drop_stats["dropped_trajs"]

                        if len(new_batch) == 0:
                            if drop_stats["dropped_trajs"] > 0:
                                print(
                                    "[WARN] All trajectories in this generation batch were dropped "
                                    "due to eval script load failures; fetching another batch."
                                )
                            need_more_generation = True
                        else:
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
                    else:
                        num_gen_batches += 1
                        batch = new_batch

                    if need_more_generation:
                        # keep accumulating additional generation batches before one optimizer step
                        continue

                    # Session expansion can make train batch size dynamic and not divisible by DP size.
                    # Pad before any DP-dispatched compute/update (balance, logprob, value, actor/critic update).
                    # For Megatron, make_iterator also requires (batch/dp_size) % (ppo_mini_batch_size*n/dp_size) == 0,
                    # i.e. batch_size % (ppo_mini_batch_size * n) == 0. Use lcm to satisfy both constraints.
                    dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
                    _ppo_mini_bsz = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size", None)
                    _rollout_n = self.config.actor_rollout_ref.rollout.get("n", 1)
                    if _ppo_mini_bsz:
                        _mini_bsz_total = _ppo_mini_bsz * _rollout_n
                        _per_rank_mini = _mini_bsz_total // dp_size
                        pad_divisor = dp_size * _per_rank_mini if _per_rank_mini > 0 else dp_size
                    else:
                        pad_divisor = dp_size
                    batch_size_before_dp_pad = len(batch)
                    if batch_size_before_dp_pad % pad_divisor != 0:
                        batch, dp_pad_size = pad_dataproto_to_divisor(batch, pad_divisor)
                        batch_size_after_dp_pad = len(batch)
                        print(
                            "[WARN] Padded train batch for DP/minibatch divisibility: "
                            f"original={batch_size_before_dp_pad}, dp_size={dp_size}, "
                            f"pad_divisor={pad_divisor}, "
                            f"pad_size={dp_pad_size}, padded={batch_size_after_dp_pad}"
                        )
                        pad_row_slice = slice(batch_size_before_dp_pad, batch_size_after_dp_pad)
                        # Ensure padded rows never contribute to optimization.
                        batch.batch["response_mask"][pad_row_slice] = 0
                        batch.batch["attention_mask"][pad_row_slice] = 0
                        for tensor_key in ("token_level_scores", "token_level_rewards", "session_mask", "session_row_weight"):
                            if tensor_key in batch.batch:
                                batch.batch[tensor_key][pad_row_slice] = 0
                    else:
                        batch_size_after_dp_pad = batch_size_before_dp_pad

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                router_mode = getattr(
                                    self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                )
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if not filter_enabled:
                            if (not self.use_reward_loop) and self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            if "token_level_rewards" not in batch.batch:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        if dp_pad_size > 0:
                            # Exclude padded rows from advantage estimation to avoid
                            # perturbing GRPO/PASS@k group statistics.
                            adv_batch = compute_advantage(
                                batch[:batch_size_before_dp_pad],
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )
                            pad_adv = torch.zeros_like(batch.batch["response_mask"][batch_size_before_dp_pad:]).to(
                                dtype=adv_batch.batch["advantages"].dtype
                            )
                            pad_ret = torch.zeros_like(batch.batch["response_mask"][batch_size_before_dp_pad:]).to(
                                dtype=adv_batch.batch["returns"].dtype
                            )
                            batch.batch["advantages"] = torch.cat([adv_batch.batch["advantages"], pad_adv], dim=0)
                            batch.batch["returns"] = torch.cat([adv_batch.batch["returns"], pad_ret], dim=0)
                        else:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current sync_step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        current_sync_step = self._get_sync_step()
                        checkpoint_on_this_update = self._is_sync_step_boundary(is_last_step=is_last_step)
                        if checkpoint_on_this_update or esi_close_to_expiration:
                            if self.config.trainer.save_freq > 0 and (
                                is_last_step
                                or current_sync_step % self.config.trainer.save_freq == 0
                                or esi_close_to_expiration
                            ):
                                if esi_close_to_expiration:
                                    print("Force saving checkpoint: ESI instance expiration approaching.")
                                with marked_timer("save_checkpoint", timing_raw, color="green"):
                                    self._save_checkpoint(current_sync_step)
                            elif self._save_every_step_ckpt_enabled():
                                with marked_timer("save_step_checkpoint", timing_raw, color="green"):
                                    self._save_latest_step_checkpoint(current_sync_step)

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if dp_pad_size > 0:
                        batch = unpad_dataproto(batch, pad_size=dp_pad_size)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if need_more_generation:
                    with marked_timer("stop_profile", timing_raw):
                        next_step_profile = (
                            self.global_steps + 1 in self.config.global_profiler.steps
                            if self.config.global_profiler.steps is not None
                            else False
                        )
                        self._stop_profiling(
                            curr_step_profile and not next_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                        prev_step_profile = curr_step_profile
                        curr_step_profile = next_step_profile
                    continue

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "training/num_gen_batches": num_gen_batches,
                    }
                )
                if dp_size is not None:
                    metrics["training/dp_size"] = dp_size
                    metrics["training/dp_pad_size"] = dp_pad_size
                    metrics["training/batch_size_before_dp_pad"] = batch_size_before_dp_pad
                    metrics["training/batch_size_after_dp_pad"] = batch_size_after_dp_pad
                if filter_enabled:
                    metrics["training/filter_total_prompt_groups"] = filter_total_prompt_groups
                    metrics["training/filter_kept_prompt_groups"] = filter_kept_prompt_groups
                    metrics["training/filter_total_trajectories"] = filter_total_trajs
                    metrics["training/filter_kept_trajectories"] = filter_kept_trajs
                    metrics["training/filter_dropped_prompt_groups_eval_script"] = filter_dropped_prompt_groups
                    metrics["training/filter_dropped_trajectories_eval_script"] = filter_dropped_trajs
                    metrics["training/filter_kept_prompt_ratio"] = (
                        filter_kept_prompt_groups / filter_total_prompt_groups if filter_total_prompt_groups > 0 else 0.0
                    )
                    metrics["training/filter_kept_traj_ratio"] = (
                        filter_kept_trajs / filter_total_trajs if filter_total_trajs > 0 else 0.0
                    )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # The rollout side may expand one logical rollout into multiple
                # session rows. Override reward/score log metrics with
                # rollout-level aggregation so split sessions do not count as
                # independent samples in W&B/console.
                rollout_metric_groups = self._rollout_metric_groups(batch)
                metrics.update(self._compute_rollout_sequence_metrics(batch))
                metrics.update(self._compute_training_reward_metrics(batch))
                metrics.update(
                    self._compute_obj_openended_avg_score_metrics(
                        batch.non_tensor_batch,
                        prefix="reward_split",
                        groups=rollout_metric_groups,
                    )
                )
                metrics.update(
                    self._compute_obj_category_avg_score_metrics_from_batch(
                        batch,
                        prefix="reward_split",
                        groups=rollout_metric_groups,
                    )
                )
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # Per-category score metrics (training)
                try:
                    from recipe.deepresearch.curriculum_sampler import compute_category_score_metrics
                except ImportError:
                    compute_category_score_metrics = None
                if compute_category_score_metrics is not None:
                    metrics.update(compute_category_score_metrics(batch, prefix="train"))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)
                    # Log curriculum metrics if available
                    try:
                        from recipe.deepresearch.curriculum_sampler import CURRICULUM_STATE_ACTOR_NAME
                        import ray as _ray
                        _curriculum_actor = _ray.get_actor(CURRICULUM_STATE_ACTOR_NAME, namespace="curriculum")
                        _curriculum_metrics = _ray.get(_curriculum_actor.get_metrics.remote())
                        metrics.update(_curriculum_metrics)
                    except (ValueError, ImportError):
                        pass

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

                # reset accumulators after one optimizer step
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                filter_total_prompt_groups = 0
                filter_kept_prompt_groups = 0
                filter_total_trajs = 0
                filter_kept_trajs = 0
                filter_dropped_prompt_groups = 0
                filter_dropped_trajs = 0
