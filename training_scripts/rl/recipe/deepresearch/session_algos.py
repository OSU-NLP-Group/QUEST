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
Session-level algorithms for DeepResearch training.

This module provides functions for:
1. Session-level loss aggregation
2. Session-level advantage estimation
3. Session-level reward shaping
"""

from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch


def agg_loss_session_level(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    session_mask: torch.Tensor,
    dp_size: int = 1,
    global_num_sessions: Optional[int] = None,
) -> torch.Tensor:
    """
    Aggregate loss at the session level.

    Instead of averaging over all tokens, this function:
    1. Sums the loss for each session
    2. Averages over the number of sessions

    This ensures that each session contributes equally to the gradient,
    regardless of the number of tokens in the session.

    Args:
        loss_mat: Per-token loss matrix [batch_size, response_length]
        loss_mask: Mask for valid tokens [batch_size, response_length]
        session_mask: Mask indicating session end positions [batch_size, response_length]
            1 at the last token of each session, 0 otherwise
        dp_size: Data parallel size for scaling
        global_num_sessions: Total number of sessions across all workers
            If None, computed from session_mask

    Returns:
        loss: Scalar aggregated loss
    """
    # Compute per-token loss with mask
    masked_loss = loss_mat * loss_mask  # [batch_size, response_length]

    # Cumulative sum of losses along sequence dimension
    cumsum_loss = masked_loss.cumsum(dim=-1)  # [batch_size, response_length]

    # Values at session end positions
    session_end_cumsum = cumsum_loss * session_mask  # [batch_size, response_length]

    # Build prev_session_cumsum vectorized:
    # For each session end, we need the cumsum at the *previous* session end (0 for the first).
    # Strategy: cumulative-max of (session_end_cumsum) shifted right by one position.
    # Since session_end_cumsum is 0 at non-boundary positions and monotonically
    # increasing at boundary positions, cummax propagates the last boundary value.
    shifted_end_cumsum = torch.zeros_like(session_end_cumsum)
    shifted_end_cumsum[:, 1:] = session_end_cumsum[:, :-1]
    # cummax fills forward the last session-end cumsum value
    prev_session_cumsum, _ = shifted_end_cumsum.cummax(dim=-1)

    # Per-session loss = cumsum at this session end - cumsum at previous session end
    per_session_loss = (session_end_cumsum - prev_session_cumsum * session_mask)

    # Sum all session losses
    total_session_loss = per_session_loss.sum()

    # Count total sessions
    if global_num_sessions is None:
        global_num_sessions = session_mask.sum()

    # Average over sessions
    loss = total_session_loss / (global_num_sessions + 1e-8) * dp_size

    return loss


def compute_session_level_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    session_ids: np.ndarray,
    session_boundaries: List[List[Tuple[int, int]]],
    index: np.ndarray,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute session-level advantage using GRPO-style group normalization.

    For each trajectory, all tokens within a session receive the same
    advantage value (the sum of rewards in that session minus the group mean).

    Args:
        token_level_rewards: Rewards at each token position [batch_size, response_length]
        response_mask: Mask for valid tokens [batch_size, response_length]
        session_ids: Session ID for each token (numpy object array of lists)
        session_boundaries: List of (start, end) tuples for each trajectory
        index: Group indices for GRPO normalization
        epsilon: Small constant for numerical stability

    Returns:
        advantages: Session-level advantages [batch_size, response_length]
        returns: Returns (reward-to-go) [batch_size, response_length]
    """
    batch_size, response_length = token_level_rewards.shape

    # Compute returns (reward-to-go from each position)
    returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

    # Compute session-level rewards
    session_rewards = torch.zeros(batch_size, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
    session_counts = torch.zeros(batch_size, dtype=torch.long, device=token_level_rewards.device)

    for i in range(batch_size):
        boundaries = session_boundaries[i] if i < len(session_boundaries) else []
        valid_length = response_mask[i].sum().long().item()

        if not boundaries:
            # No session boundaries - use full trajectory reward
            session_rewards[i] = (token_level_rewards[i] * response_mask[i]).sum()
            session_counts[i] = 1
        else:
            # Sum rewards for all sessions (which should be the same due to session-level reward)
            session_rewards[i] = (token_level_rewards[i] * response_mask[i]).sum()
            session_counts[i] = len(boundaries)

    # Group by prompt (index) and compute mean
    prompt_groups = defaultdict(list)
    for i in range(batch_size):
        prompt_groups[index[i]].append(i)

    group_means = torch.zeros(batch_size, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
    group_stds = torch.zeros(batch_size, dtype=token_level_rewards.dtype, device=token_level_rewards.device)

    for prompt_idx, traj_indices in prompt_groups.items():
        traj_rewards = session_rewards[traj_indices]
        mean_reward = traj_rewards.mean()
        std_reward = traj_rewards.std() + epsilon

        for i in traj_indices:
            group_means[i] = mean_reward
            group_stds[i] = std_reward

    # Compute advantages: (reward - group_mean) / group_std
    trajectory_advantages = (session_rewards - group_means) / group_stds

    # Expand to token level - all tokens in trajectory get same advantage
    advantages = torch.zeros_like(token_level_rewards)
    for i in range(batch_size):
        advantages[i] = trajectory_advantages[i] * response_mask[i]

    return advantages, returns


def compute_session_level_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    session_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GRPO-style advantage at session level.

    Each session's tokens receive the same advantage value, computed as
    the trajectory reward minus the group mean (normalized by std if enabled).

    Args:
        token_level_rewards: Rewards at each token [batch_size, response_length]
        response_mask: Mask for valid tokens [batch_size, response_length]
        session_mask: Mask for session ends [batch_size, response_length]
        index: Group indices for normalization [batch_size]
        epsilon: Numerical stability constant
        norm_adv_by_std_in_grpo: Whether to normalize by std

    Returns:
        advantages: Session-level advantages [batch_size, response_length]
        returns: Token-level returns [batch_size, response_length]
    """
    # Compute returns
    returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

    # Compute trajectory-level rewards (sum over all tokens)
    trajectory_rewards = (token_level_rewards * response_mask).sum(dim=-1)  # [batch_size]

    # Group normalization
    unique_indices = np.unique(index)
    trajectory_advantages = torch.zeros_like(trajectory_rewards)

    for prompt_idx in unique_indices:
        mask = torch.tensor(index == prompt_idx, device=trajectory_rewards.device)
        group_rewards = trajectory_rewards[mask]

        mean_reward = group_rewards.mean()
        if norm_adv_by_std_in_grpo:
            std_reward = group_rewards.std() + epsilon
            trajectory_advantages[mask] = (group_rewards - mean_reward) / std_reward
        else:
            trajectory_advantages[mask] = group_rewards - mean_reward

    # Expand to token level
    # All tokens in the same trajectory get the same advantage
    advantages = trajectory_advantages.unsqueeze(-1) * response_mask

    return advantages, returns
