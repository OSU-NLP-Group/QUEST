"""
Toy verification: session weight correction makes split sessions equivalent to unsplit trajectory.

Setup:
  Rollout A: split into 3 sessions (20, 40, 80 assistant tokens)
  Rollout B: unsplit, 1 session (140 tokens)
  B = 4 (expanded rows), R = 2 (unique rollouts), scale_factor = B/R = 2.0

Weights are pre-scaled by B/R at expansion time so that sum(weights) = B.
This makes the existing global_batch_size denominator DP-correct.

We verify:
  - Without correction: rollout A has ~3x the weight of rollout B
  - With correction: rollout A and B have equal weight (each contributes token_mean)
  - The result is identical to processing 2 unsplit rollouts
"""

import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verl.trainer.ppo.core_algos import agg_loss


def make_prescaled_weights(token_counts_per_rollout: list[list[int]]):
    """Simulate what expand_multi_session_output produces.

    Args:
        token_counts_per_rollout: e.g. [[20, 40, 80], [140]]
            Rollout A has 3 sessions, Rollout B has 1.

    Returns:
        flat_token_counts: [20, 40, 80, 140]
        session_row_weight: pre-scaled so sum = B
    """
    flat_counts = []
    weights = []
    for sessions in token_counts_per_rollout:
        rollout_total = sum(sessions)
        for n in sessions:
            flat_counts.append(n)
            if len(sessions) == 1:
                weights.append(1.0)
            else:
                weights.append(n / rollout_total)

    B = len(flat_counts)
    R = len(token_counts_per_rollout)
    weights = torch.tensor(weights, dtype=torch.float32)
    # Pre-scale: sum(weights) goes from R to B
    weights = weights * (B / R)
    return flat_counts, weights


def make_loss_mat_and_mask(token_counts: list[int], response_length: int = 200):
    """Create loss_mat (all 1s for simplicity) and response_mask for given token counts."""
    bs = len(token_counts)
    loss_mat = torch.ones(bs, response_length)
    loss_mask = torch.zeros(bs, response_length)
    for i, n in enumerate(token_counts):
        loss_mask[i, :n] = 1.0
    return loss_mat, loss_mask


def test_without_correction():
    """Without correction: 3 split rows + 1 unsplit row, seq-mean-token-mean."""
    # Rollout A: 3 sessions (20, 40, 80 tokens)
    # Rollout B: 1 session (140 tokens)
    token_counts = [20, 40, 80, 140]
    loss_mat, loss_mask = make_loss_mat_and_mask(token_counts)

    loss = agg_loss(loss_mat, loss_mask, loss_agg_mode="seq-mean-token-mean")

    # Each row's seq_loss = sum(loss*mask) / n = n/n = 1.0 (since loss_mat is all 1s)
    # So loss = (1 + 1 + 1 + 1) / 4 = 1.0
    # Rollout A total weight = 3/4, Rollout B total weight = 1/4
    print(f"Without correction: loss = {loss.item():.6f}")
    print(f"  Rollout A total weight: 3/4 = {3/4:.4f}")
    print(f"  Rollout B total weight: 1/4 = {1/4:.4f}")
    print(f"  Ratio A:B = 3:1 (biased!)")
    print()


def test_with_correction():
    """With pre-scaled correction: weights sum to B, denominator = B."""
    token_counts, session_row_weight = make_prescaled_weights([[20, 40, 80], [140]])
    loss_mat, loss_mask = make_loss_mat_and_mask(token_counts)

    # B=4, R=2, scale=2.0
    # Weights: [20/140*2, 40/140*2, 80/140*2, 1.0*2] = [2/7, 4/7, 8/7, 2.0]
    # sum(weights) = 14/7 + 2 = 4 = B ✓

    loss = agg_loss(
        loss_mat, loss_mask,
        loss_agg_mode="seq-mean-token-mean",
        session_row_weight=session_row_weight,
    )

    # Weighted seq_losses: [2/7*1, 4/7*1, 8/7*1, 2*1] = [2/7, 4/7, 8/7, 2]
    # masked_sum = 14/7 + 2 = 4
    # denominator = seq_mask.sum() = 4  (or global_batch_size = 4)
    # loss = 4 / 4 = 1.0
    # Unsplit equiv: 2 rollouts each token_mean=1.0, loss = (1+1)/2 = 1.0 ✓
    expected = 1.0
    assert abs(loss.item() - expected) < 1e-6, f"Expected {expected}, got {loss.item()}"
    print(f"With correction: loss = {loss.item():.6f} (expected {expected})")
    print(f"  sum(weights) = {session_row_weight.sum().item():.1f} = B")
    print(f"  Ratio A:B = 1:1 (fair!)")
    print()


def test_with_global_batch_size():
    """Verify correction works when global_batch_size is explicitly set (DP scenario)."""
    token_counts, session_row_weight = make_prescaled_weights([[20, 40, 80], [140]])
    loss_mat, loss_mask = make_loss_mat_and_mask(token_counts)

    # Simulate: global_batch_size = 4 (set by trainer), dp_size = 2 (2 GPUs)
    # Each rank has 2 rows. We simulate rank 0 having all 4 rows for simplicity,
    # but pass global_batch_size and dp_size as if it were distributed.
    loss = agg_loss(
        loss_mat, loss_mask,
        loss_agg_mode="seq-mean-token-mean",
        session_row_weight=session_row_weight,
        global_batch_size=4,
        dp_size=1,  # single rank for test, but denominator is global
    )

    # loss = masked_sum(weighted_seq_losses, seq_mask) / global_batch_size * dp_size
    #      = 4.0 / 4 * 1 = 1.0
    expected = 1.0
    assert abs(loss.item() - expected) < 1e-6, f"Expected {expected}, got {loss.item()}"
    print(f"With global_batch_size=4, dp_size=1: loss = {loss.item():.6f} (expected {expected})")

    # Now simulate what one rank of a 2-GPU setup would compute:
    # Rank 0 gets rows [A_s1, A_s2], rank 1 gets [A_s3, B]
    # global_batch_size = 4 (same on both ranks), dp_size = 2
    # Rank 0 loss: (2/7*1 + 4/7*1) / 4 * 2 = (6/7) / 4 * 2 = 12/28 = 3/7
    rank0_mat, rank0_mask = make_loss_mat_and_mask([20, 40])
    rank0_weight = session_row_weight[:2]
    loss_rank0 = agg_loss(
        rank0_mat, rank0_mask,
        loss_agg_mode="seq-mean-token-mean",
        session_row_weight=rank0_weight,
        global_batch_size=4,
        dp_size=2,
    )

    rank1_mat, rank1_mask = make_loss_mat_and_mask([80, 140])
    rank1_weight = session_row_weight[2:]
    loss_rank1 = agg_loss(
        rank1_mat, rank1_mask,
        loss_agg_mode="seq-mean-token-mean",
        session_row_weight=rank1_weight,
        global_batch_size=4,
        dp_size=2,
    )

    # DDP gradient average: (loss_rank0 + loss_rank1) / 2
    effective_loss = (loss_rank0.item() + loss_rank1.item()) / 2
    print(f"  DP simulation: rank0={loss_rank0.item():.6f}, rank1={loss_rank1.item():.6f}")
    print(f"  After DDP average: {effective_loss:.6f} (expected {expected})")
    assert abs(effective_loss - expected) < 1e-6, f"DP mismatch: {effective_loss} != {expected}"
    print()


def test_with_varying_loss():
    """Verify with non-trivial loss values."""
    response_length = 200
    token_counts = [20, 40, 80, 140]
    bs = len(token_counts)

    # Random per-token losses
    torch.manual_seed(42)
    loss_mat = torch.rand(bs, response_length)
    loss_mask = torch.zeros(bs, response_length)
    for i, n in enumerate(token_counts):
        loss_mask[i, :n] = 1.0

    # Without correction
    loss_no_corr = agg_loss(loss_mat, loss_mask, loss_agg_mode="seq-mean-token-mean")

    # With pre-scaled correction
    _, session_row_weight = make_prescaled_weights([[20, 40, 80], [140]])
    loss_with_corr = agg_loss(
        loss_mat, loss_mask, loss_agg_mode="seq-mean-token-mean",
        session_row_weight=session_row_weight,
    )

    # Manually verify: compute token_mean for each rollout
    all_A_losses = torch.cat([loss_mat[0, :20], loss_mat[1, :40], loss_mat[2, :80]])
    token_mean_A = all_A_losses.mean().item()
    token_mean_B = loss_mat[3, :140].mean().item()

    # Corrected loss should equal unsplit: (token_mean_A + token_mean_B) / 2
    expected_loss = (token_mean_A + token_mean_B) / 2.0
    assert abs(loss_with_corr.item() - expected_loss) < 1e-5, \
        f"Expected {expected_loss:.6f}, got {loss_with_corr.item():.6f}"

    print(f"With varying loss values:")
    print(f"  Loss without correction: {loss_no_corr.item():.6f}")
    print(f"  Loss with correction:    {loss_with_corr.item():.6f}")
    print(f"  Rollout A token_mean: {token_mean_A:.6f}")
    print(f"  Rollout B token_mean: {token_mean_B:.6f}")
    print(f"  Expected (unsplit equiv): {expected_loss:.6f}")
    print(f"  Match: {abs(loss_with_corr.item() - expected_loss) < 1e-5}")
    print()


def test_single_session_unchanged():
    """Non-expanded rows: all weights = B/R = 1.0, behaves identically to no correction."""
    token_counts = [50, 100, 150]
    loss_mat, loss_mask = make_loss_mat_and_mask(token_counts)

    loss_no_weight = agg_loss(loss_mat, loss_mask, loss_agg_mode="seq-mean-token-mean")

    # All single-session: R = B = 3, scale = 1.0, weights all 1.0
    _, session_row_weight = make_prescaled_weights([[50], [100], [150]])
    loss_with_weight = agg_loss(
        loss_mat, loss_mask, loss_agg_mode="seq-mean-token-mean",
        session_row_weight=session_row_weight,
    )

    print(f"Single session (no split):")
    print(f"  Without weight: {loss_no_weight.item():.6f}")
    print(f"  With weight=1:  {loss_with_weight.item():.6f}")
    print(f"  Match: {abs(loss_no_weight.item() - loss_with_weight.item()) < 1e-7}")
    assert abs(loss_no_weight.item() - loss_with_weight.item()) < 1e-7
    print()


if __name__ == "__main__":
    test_without_correction()
    test_with_correction()
    test_with_global_batch_size()
    test_with_varying_loss()
    test_single_session_unchanged()
    print("All tests passed!")
