"""Pairwise grader for comparative evaluation."""

import logging
from typing import Dict, Any, Optional, Tuple

from .base_grader import BaseGrader
from liveresearchbench.criteria.depth import (
    DEPTH_COMPARISON_SCHEMA,
    create_depth_comparison_prompt,
    create_depth_user_prompt
)

logger = logging.getLogger(__name__)


class PairwiseGrader(BaseGrader):
    """Grader for pairwise comparison evaluation (Analysis Depth)."""
    
    def __init__(self):
        """Initialize the pairwise grader."""
        super().__init__()
    
    async def compare_depth_async(self, query: str, report_a: str, report_b: str,
                                 provider: str = "gemini", model: str = None,
                                 swap_positions: bool = True) -> Dict[str, Any]:
        """
        Compare two reports for analysis depth using position-swap averaging.
        
        Args:
            query: Original research query
            report_a: First report content
            report_b: Second report content
            provider: AI provider (gemini or openai)
            model: Specific model to use
            swap_positions: If True (default), performs position-swap averaging to mitigate position bias
            
        Returns:
            Dictionary with comparison results
        """
        if swap_positions:
            return await self._compare_with_position_swap(
                query, report_a, report_b, provider, model
            )
        
        # Single comparison without swap (not recommended for production)
        client = self._create_client(provider, model)
        system_prompt = create_depth_comparison_prompt()
        user_prompt = create_depth_user_prompt(query, report_a, report_b)
        return await self._compare_single_judge(client, system_prompt, user_prompt)
    
    async def _compare_single_judge(self, client, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Compare reports using a single judge call."""
        try:
            result = await client.generate_with_schema_async(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                schema=DEPTH_COMPARISON_SCHEMA
            )
            
            return {
                'winner': result.get('winner', 'tie'),
                'scores': result.get('scores', {}),
                'justification': result.get('justification', ''),
                'major_flaws': result.get('major_flaws', {'A': [], 'B': []})
            }
            
        except Exception as e:
            logger.error(f"Error in depth comparison: {e}")
            raise
    
    async def _compare_with_position_swap(self, query: str, report_a: str, report_b: str,
                                         provider: str, model: str) -> Dict[str, Any]:
        """
        Compare two reports with position-swap to mitigate position bias.
        
        Performs two comparisons:
        1. (A vs B): A in position 1, B in position 2 → get winner1
        2. (B vs A): B in position 1, A in position 2 → get winner2
        
        Returns both winner decisions separately.
        The batch evaluator will count wins/losses/ties from both comparisons
        to calculate the final win rate.
        
        
        Args:
            query: Research query
            report_a: First report (typically the evaluated model)
            report_b: Second report (typically the reference)
            provider: AI provider
            model: Specific model to use
            
        Returns:
            Dictionary with both comparison results (winner_comparison_1, winner_comparison_2)
        """
        import asyncio
        
        client = self._create_client(provider, model)
        system_prompt = create_depth_comparison_prompt()
        
        # Comparison 1: A vs B (A in position 1)
        user_prompt_1 = create_depth_user_prompt(query, report_a, report_b)
        
        # Comparison 2: B vs A (B in position 1, A in position 2)
        user_prompt_2 = create_depth_user_prompt(query, report_b, report_a)
        
        # Run both comparisons in parallel
        logger.info("Running position-swap comparisons (A vs B, then B vs A)...")
        
        result_1, result_2 = await asyncio.gather(
            self._compare_single_judge(client, system_prompt, user_prompt_1),
            self._compare_single_judge(client, system_prompt, user_prompt_2)
        )
        
        # Extract winners from each comparison
        winner_1 = result_1['winner']  # Winner from (A vs B)
        winner_2 = result_2['winner']  # Winner from (B vs A), but positions are swapped
        
        # Map winner_2 back to original positions
        # In comparison 2, B was in position A and A was in position B
        # So if winner_2 says 'A', it means B won (since B was in A's position)
        # And if winner_2 says 'B', it means A won (since A was in B's position)
        if winner_2 == 'A':
            winner_2_mapped = 'B'  # B was in position A
        elif winner_2 == 'B':
            winner_2_mapped = 'A'  # A was in position B
        else:
            winner_2_mapped = 'tie'
        
        # Don't aggregate winners here - return both decisions
        # The batch evaluator will count wins/losses/ties from both comparisons
        
        # Map winners to semantic names for clarity
        def map_winner_to_semantic(winner_ab):
            if winner_ab == 'A':
                return 'evaluated_model'
            elif winner_ab == 'B':
                return 'reference_model'
            else:
                return 'tie'
        
        winner_1_semantic = map_winner_to_semantic(winner_1)
        winner_2_semantic = map_winner_to_semantic(winner_2_mapped)
        
        # Combine justifications (for debugging/reference)
        justification = (
            f"Position-swap result. "
            f"Comparison 1 winner: {winner_1_semantic}. "
            f"Comparison 2 winner: {winner_2_semantic}. "
            f"Details in raw_comparison_1 and raw_comparison_2."
        )
        
        return {
            'winner_comparison_1': winner_1_semantic,  # 'evaluated_model', 'reference_model', or 'tie'
            'winner_comparison_2': winner_2_semantic,  # 'evaluated_model', 'reference_model', or 'tie'
            'justification': justification,
            'position_swap_used': True,
            'raw_comparison_1': result_1,  # Full details from first comparison (positions: A=evaluated, B=reference)
            'raw_comparison_2': result_2   # Full details from second comparison (positions: A=reference, B=evaluated)
        }
    
    async def grade_async(self, query: str, report_content: str, criterion: str,
                         provider: str = "gemini", model: str = None,
                         current_date: str = None, **kwargs) -> Dict[str, Any]:
        """
        Grade using pairwise comparison (requires two reports).
        
        Args:
            query: Original research query
            report_content: First report (report_a)
            criterion: Criterion name ('depth')
            provider: AI provider
            model: Specific model to use
            current_date: Current date (unused for pairwise)
            **kwargs: Must include 'report_b', optionally 'swap_positions' (default True)
            
        Returns:
            Dictionary with comparison results
        """
        if criterion.lower() != 'depth':
            raise ValueError(f"Unknown pairwise criterion: {criterion}")
        
        report_b = kwargs.get('report_b')
        if not report_b:
            raise ValueError("report_b required for pairwise depth comparison")
        
        swap_positions = kwargs.get('swap_positions', True)
        
        return await self.compare_depth_async(
            query, report_content, report_b, provider, model, swap_positions
        )

