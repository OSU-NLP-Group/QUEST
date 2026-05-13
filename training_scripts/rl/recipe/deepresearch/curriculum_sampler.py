# Copyright 2025 DeepResearch Authors
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
Dynamic Curriculum Learning Sampler for DeepResearch.

Objective and subjective tasks share the C1-C9 categories and one Boltzmann bandit.

- Q-value update: Q = (1 - lr) * Q + lr * new_Q
- Sampling: Boltzmann (softmax over Q-values / temperature)
- objective='adv' (default): masked_mean(|advantage|, response_mask)
- objective='progress': reward[t] - reward[t-1]

Usage (enabled by default in the run script):
    data.sampler.class_path='recipe/deepresearch/curriculum_sampler.py'
    data.sampler.class_name='DynamicCurriculumSampler'
    data.dataloader_num_workers=0
    +data.curriculum.objective=adv
    +data.curriculum.lr=0.1
    +data.curriculum.temperature=1.0
    +data.curriculum.min_weight=0.02
    +data.curriculum.replacement=False
"""

import copy
import math
import random
import re
from collections import defaultdict
from typing import Optional

import numpy as np
import ray
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler

# -------------------------------------------------------------------------
# Category definitions (C1-C9 shared by objective and subjective tasks)
# -------------------------------------------------------------------------
_CATEGORY_RE = re.compile(r"_(C\d)_")
CATEGORIES = [f"C{i}" for i in range(1, 10)]

CURRICULUM_STATE_ACTOR_NAME = "deepresearch_curriculum_state"


def _extract_category(task_id: str) -> Optional[str]:
    """Extract C1-C9 from an obj task_id string."""
    m = _CATEGORY_RE.search(task_id)
    return m.group(1) if m else None


def _extract_category_from_item(item: dict) -> Optional[str]:
    """Return complexity_class (C1-C9) for a dataset item dict."""
    rm = item.get("reward_model")
    if isinstance(rm, dict):
        cat = _extract_category(rm.get("task_id", ""))
        if cat:
            return cat
    extra = item.get("extra_info")
    if isinstance(extra, dict):
        cat = _extract_category(extra.get("original_task_id", ""))
        if cat:
            return cat
        cat = extra.get("complexity_class")
        if cat in CATEGORIES:
            return cat
    return None


# -------------------------------------------------------------------------
# CurriculumState: named Ray actor shared between trainer and rollout workers
# -------------------------------------------------------------------------
@ray.remote(num_cpus=0)
class CurriculumState:
    """
    Lightweight Ray actor that stores each category Q-value and Boltzmann sampling weight.

    The rollout-side DynamicCurriculumSampler reads the weights;
    The trainer calls update() after each training step to update Q-values.
    """

    def __init__(
        self,
        categories: list[str],
        objective: str = "adv",     # 'adv' or 'progress'
        lr: float = 0.1,
        temperature: float = 1.0,
        min_weight: float = 0.02,
    ):
        self.categories = categories
        self.num_arms = len(categories)
        self.cat_to_arm = {cat: i for i, cat in enumerate(categories)}
        self.objective = objective
        self.lr = lr
        self.temperature = temperature
        self.min_weight = min_weight

        # Q-values initialized to 0, which corresponds to uniform sampling.
        self.q_values = np.zeros(self.num_arms)

        # For the 'progress' objective: record the previous reward.
        self.last_reward: dict[str, float] = {cat: 0.0 for cat in categories}

        self.update_count: dict[str, int] = {cat: 0 for cat in categories}
        self.total_updates = 0

        # Initialize uniform weights.
        self._weights: dict[str, float] = {cat: 1.0 / self.num_arms for cat in categories}

    def get_objective(self) -> str:
        return self.objective

    # ------------------------------------------------------------------
    # Core update: receive per-sample (category, signal) pairs from the trainer.
    # ------------------------------------------------------------------
    def update(
        self,
        categories: list[str],
        signals: list[float],       # |advantage| or reward，depends on the objective
    ) -> None:
        """
        Update Q-values.

        Args:
            categories: Category label for each sample
            signals:    Signal value for each sample
                        - objective='adv':      masked_mean(|advantage|, mask) per sample
                        - objective='progress': sequence reward per sample
        """
        # Aggregate signals by category.
        cat_signals: dict[str, list[float]] = defaultdict(list)
        for cat, sig in zip(categories, signals):
            if cat in self.cat_to_arm:
                cat_signals[cat].append(sig)

        if self.objective == "adv":
            new_q = np.zeros(self.num_arms)
            for cat, sigs in cat_signals.items():
                arm = self.cat_to_arm[cat]
                new_q[arm] = float(np.mean(np.abs(sigs)))
                self.update_count[cat] += 1

        elif self.objective == "progress":
            # Compute reward improvement relative to the previous step.
            new_q = np.zeros(self.num_arms)
            for cat, sigs in cat_signals.items():
                arm = self.cat_to_arm[cat]
                cur_reward = float(np.mean(sigs))
                new_q[arm] = cur_reward - self.last_reward[cat]
                self.last_reward[cat] = cur_reward
                self.update_count[cat] += 1
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # EMA update.
        self.q_values = (1 - self.lr) * self.q_values + self.lr * new_q
        self.total_updates += 1
        self._recompute_weights()

    def get_weights(self) -> dict[str, float]:
        return self._weights

    def get_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        weights = self._weights
        for cat in self.categories:
            arm = self.cat_to_arm[cat]
            metrics[f"curriculum/q_value_{cat}"] = float(self.q_values[arm])
            metrics[f"curriculum/weight_{cat}"] = float(weights[cat])
            metrics[f"curriculum/update_count_{cat}"] = float(self.update_count[cat])
        metrics["curriculum/total_updates"] = float(self.total_updates)
        return metrics

    # ------------------------------------------------------------------
    def _recompute_weights(self):
        """Boltzmann sampling (softmax over Q-values / temperature) with a min_weight floor."""
        q = self.q_values / self.temperature
        q = q - q.max()                      # Numerical stability.
        exp_q = np.exp(q)
        raw = exp_q / exp_q.sum()

        # Apply the min_weight floor and renormalize.
        n = self.num_arms
        total_floor = self.min_weight * n
        if total_floor >= 1.0:
            self._weights = {cat: 1.0 / n for cat in self.categories}
            return

        remaining = 1.0 - total_floor
        above = np.maximum(raw - self.min_weight, 0.0)
        above_sum = above.sum()
        if above_sum < 1e-12:
            self._weights = {cat: 1.0 / n for cat in self.categories}
            return

        final = self.min_weight + remaining * above / above_sum
        self._weights = {cat: float(final[self.cat_to_arm[cat]]) for cat in self.categories}


# -------------------------------------------------------------------------
# Extract (categories, signals) from a DataProto batch.
#
# Key detail: one prompt can be expanded into multiple rows by session expansion,
# and rows with the same uid share the same task_id/category. Aggregate by uid first
# (using the mean across sessions), so each prompt contributes only one signal value,
# which avoids over-weighting categories with more sessions.
# -------------------------------------------------------------------------

def _get_uid(item) -> str:
    """Extract the unique prompt identifier (uid) for session aggregation."""
    uid = item.non_tensor_batch.get("uid", None)
    if uid is not None:
        return str(uid.item() if hasattr(uid, "item") else uid)
    # fallback: Use the index as a fallback when uid is unavailable.
    return str(id(item))


def _get_category(item) -> Optional[str]:
    """Extract the category (C1-C9) from one sample."""
    rm = item.non_tensor_batch.get("reward_model", {})
    task_id = rm.get("task_id", "") if isinstance(rm, dict) else ""
    cat = _extract_category(task_id)
    if cat:
        return cat
    extra = item.non_tensor_batch.get("extra_info", {})
    if isinstance(extra, dict):
        cat = _extract_category(extra.get("original_task_id", ""))
        if cat:
            return cat
        cat = extra.get("complexity_class")
        if isinstance(cat, str) and cat in CATEGORIES:
            return cat
    return None


def _aggregate_by_uid(
    batch: DataProto,
    per_sample_values: list[float],
) -> tuple[list[str], list[float]]:
    """
    Aggregate per_sample_values by uid: average rows with the same uid,
    and return (categories, signals), with each uid appearing once.
    """
    # uid → (category, [values])
    uid_data: dict[str, tuple[str, list[float]]] = {}

    for i in range(len(batch)):
        item = batch[i]
        cat = _get_category(item)
        if cat is None:
            continue
        uid = _get_uid(item)
        if uid not in uid_data:
            uid_data[uid] = (cat, [])
        uid_data[uid][1].append(per_sample_values[i])

    categories: list[str] = []
    signals: list[float] = []
    for cat, vals in uid_data.values():
        categories.append(cat)
        signals.append(float(np.mean(vals)))

    return categories, signals


def _extract_adv_signals(batch: DataProto) -> tuple[list[str], list[float]]:
    """
    objective='adv'：masked_mean(|advantage|, response_mask) per sample，
    Return output after uid aggregation.
    """
    if "advantages" not in batch.batch or "response_mask" not in batch.batch:
        return [], []

    adv = batch.batch["advantages"]          # (B, T)
    mask = batch.batch["response_mask"]      # (B, T)

    abs_adv = adv.abs()
    mask_sum = mask.sum(-1).clamp(min=1)
    per_sample = ((abs_adv * mask).sum(-1) / mask_sum).cpu().tolist()  # (B,)

    return _aggregate_by_uid(batch, per_sample)


def _extract_progress_signals(batch: DataProto) -> tuple[list[str], list[float]]:
    """
    objective='progress'：sequence reward per sample，Return output after uid aggregation.
    """
    if "token_level_scores" in batch.batch:
        per_sample = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
    elif "token_level_rewards" in batch.batch:
        per_sample = batch.batch["token_level_rewards"].sum(-1).cpu().tolist()
    else:
        return [], []

    return _aggregate_by_uid(batch, per_sample)


# -------------------------------------------------------------------------
# DynamicCurriculumSampler
# -------------------------------------------------------------------------
class DynamicCurriculumSampler(AbstractCurriculumSampler):
    """
    Dynamic curriculum sampler. Objective and subjective tasks share C1-C9 categories and one Boltzmann bandit.

    - objective='adv' (default): prioritize categories with larger |advantage|.
    - objective='progress': prioritize categories whose reward improves faster.

    Synchronous mode: the trainer calls sampler.update(batch) after each step.
    Fully asynchronous mode: the trainer calls update_curriculum_from_batch(batch).
    """

    def __init__(self, data_source, data_config: DictConfig):
        self.data_source = data_source
        self.data_config = data_config
        self.dataset_size = len(data_source)

        curriculum_cfg = data_config.get("curriculum", {})
        self.objective   = str(curriculum_cfg.get("objective", "adv"))
        self.lr          = float(curriculum_cfg.get("lr", 0.1))
        self.temperature = float(curriculum_cfg.get("temperature", 1.0))
        self.min_weight  = float(curriculum_cfg.get("min_weight", 0.02))
        self.replacement = bool(curriculum_cfg.get("replacement", False))

        # category-to-indices mapping
        self.category_to_indices: dict[str, list[int]] = {cat: [] for cat in CATEGORIES}
        self.uncategorized_indices: list[int] = []

        for idx in range(self.dataset_size):
            item = data_source[idx]
            cat = _extract_category_from_item(item)
            if cat in CATEGORIES:
                self.category_to_indices[cat].append(idx)
            else:
                self.uncategorized_indices.append(idx)

        dist = ", ".join(f"{c}={len(self.category_to_indices[c])}" for c in CATEGORIES)
        print(f"[DynamicCurriculumSampler] {dist}")
        if self.uncategorized_indices:
            print(f"[DynamicCurriculumSampler] WARNING: {len(self.uncategorized_indices)} uncategorized samples")

        self.curriculum_state = CurriculumState.options(
            name=CURRICULUM_STATE_ACTOR_NAME,
            get_if_exists=True,
            namespace="curriculum",
        ).remote(
            categories=CATEGORIES,
            objective=self.objective,
            lr=self.lr,
            temperature=self.temperature,
            min_weight=self.min_weight,
        )

        self._cached_weights: dict[str, float] = {c: 1.0 / len(CATEGORIES) for c in CATEGORIES}
        self._refresh_interval = 10
        self._sample_counter = 0

    def __len__(self) -> int:
        return self.dataset_size

    def _refresh_weights(self):
        try:
            self._cached_weights = ray.get(self.curriculum_state.get_weights.remote())
        except Exception:
            pass

    def __iter__(self):
        # Emit uncategorized samples first.
        uncategorized = list(self.uncategorized_indices)
        random.shuffle(uncategorized)
        yield from uncategorized

        self._sample_counter = 0

        if self.replacement:
            # ---- Sampling with replacement ----
            avail = [c for c in CATEGORIES if self.category_to_indices[c]]
            total = sum(len(v) for v in self.category_to_indices.values())

            for _ in range(total):
                if self._sample_counter % self._refresh_interval == 0:
                    self._refresh_weights()
                self._sample_counter += 1

                probs = [self._cached_weights.get(c, 1.0 / len(CATEGORIES)) for c in avail]
                total_p = sum(probs)
                probs = [p / total_p for p in probs]
                cat = random.choices(avail, weights=probs, k=1)[0]
                yield random.choice(self.category_to_indices[cat])

        else:
            # ---- Sampling without replacement (default) ----
            remaining = {c: list(v) for c, v in self.category_to_indices.items()}
            for lst in remaining.values():
                random.shuffle(lst)

            total = sum(len(v) for v in remaining.values())
            while total > 0:
                if self._sample_counter % self._refresh_interval == 0:
                    self._refresh_weights()
                self._sample_counter += 1

                avail = [c for c in CATEGORIES if remaining[c]]
                if not avail:
                    break

                probs = [self._cached_weights.get(c, 1.0 / len(CATEGORIES)) for c in avail]
                total_p = sum(probs)
                probs = [p / total_p for p in probs]
                cat = random.choices(avail, weights=probs, k=1)[0]
                yield remaining[cat].pop()
                total -= 1

    def update(self, batch: DataProto) -> None:
        """Synchronous training mode: called by the trainer after each step."""
        categories, signals = _get_signals(batch, self.objective)
        if not categories:
            return
        ray.get(self.curriculum_state.update.remote(categories, signals))
        self._refresh_weights()


# -------------------------------------------------------------------------
# Per-category score metrics (shared by training and eval)
# -------------------------------------------------------------------------
def compute_category_score_metrics(
    batch: DataProto,
    scores: Optional[list[float]] = None,
    prefix: str = "train",
) -> dict[str, float]:
    """Compute per-category mean sequence score for wandb logging.

    Returns:
        e.g. {"train/score_C1": 0.3, ..., "train/score_mean": 0.45}
    """
    if scores is None:
        if "token_level_scores" not in batch.batch:
            return {}
        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

    if len(scores) != len(batch):
        return {}

    cat_scores: dict[str, list[float]] = defaultdict(list)
    for i in range(len(batch)):
        cat = _get_category(batch[i])
        s = float(scores[i])
        cat_scores["mean"].append(s)
        if cat:
            cat_scores[cat].append(s)

    return {f"{prefix}/score_{k}": float(np.mean(v)) for k, v in cat_scores.items() if v}


# -------------------------------------------------------------------------
# Fully asynchronous mode only: standalone function called by the trainer.
# -------------------------------------------------------------------------
def update_curriculum_from_batch(batch: DataProto) -> dict[str, float]:
    """
    The fully asynchronous trainer calls this function after each step to update CurriculumState.
    Return the curriculum metrics dict, or {} if curriculum learning is disabled.
    """
    try:
        state = ray.get_actor(CURRICULUM_STATE_ACTOR_NAME, namespace="curriculum")
    except ValueError:
        return {}

    try:
        objective = ray.get(state.get_objective.remote())
    except Exception:
        objective = "adv"

    categories, signals = _get_signals(batch, objective)
    if not categories:
        return {}

    ray.get(state.update.remote(categories, signals))
    return ray.get(state.get_metrics.remote())


# -------------------------------------------------------------------------
# Internal utilities
# -------------------------------------------------------------------------
def _get_signals(batch: DataProto, objective: str) -> tuple[list[str], list[float]]:
    if objective == "adv":
        return _extract_adv_signals(batch)
    elif objective == "progress":
        return _extract_progress_signals(batch)
    else:
        raise ValueError(f"Unknown curriculum objective: {objective!r}")
