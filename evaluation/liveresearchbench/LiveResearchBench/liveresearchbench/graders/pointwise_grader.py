"""Pointwise grader for additive/error-counting evaluation."""

import logging
from typing import Dict, Any, Optional

from .base_grader import BaseGrader
from liveresearchbench.criteria.consistency import (
    CONSISTENCY_EVALUATION_SCHEMA,
    create_consistency_prompt
)
from liveresearchbench.criteria.citation import (
    CITATION_EVALUATION_SCHEMA,
    create_citation_prompt
)

logger = logging.getLogger(__name__)


class PointwiseGrader(BaseGrader):
    """Grader for pointwise/additive evaluation (Consistency & Citation)."""
    
    def __init__(self):
        """Initialize the pointwise grader."""
        super().__init__()
    
    async def grade_consistency_async(self, query: str, report_content: str,
                                     provider: str = "gemini", model: str = None,
                                     current_date: str = None) -> Dict[str, Any]:
        """
        Grade a report on factual & logical consistency (count errors, score 10-100).
        
        Args:
            query: Original research query
            report_content: Report to evaluate
            provider: AI provider (gemini or openai)
            model: Specific model to use
            current_date: Current date for temporal context
            
        Returns:
            Dictionary with evaluation results
        """
        current_date = self._get_current_date(current_date)
        client = self._create_client(provider, model)
        
        # Create prompts
        system_prompt, user_prompt = create_consistency_prompt(query, report_content, current_date)
        
        try:
            # Get evaluation from AI
            result = await client.generate_with_schema_async(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                schema=CONSISTENCY_EVALUATION_SCHEMA
            )
            
            return {
                'specific_issues': result.get('specific_issues', []),
                'total_issues': result.get('total_issues', 0),
                'score': result.get('score', 0),
                'justification': result.get('justification', '')
            }
            
        except Exception as e:
            logger.error(f"Error in consistency grading: {e}")
            raise
    
    async def grade_citation_async(self, query: str, report_content: str,
                                  provider: str = "gemini", model: str = None,
                                  current_date: str = None) -> Dict[str, Any]:
        """
        Grade a report on citation association (count missing citations, score 10-100).
        
        Args:
            query: Original research query
            report_content: Report to evaluate
            provider: AI provider (gemini or openai)
            model: Specific model to use
            current_date: Current date for temporal context
            
        Returns:
            Dictionary with evaluation results
        """
        current_date = self._get_current_date(current_date)
        client = self._create_client(provider, model)
        
        # Create prompts
        system_prompt, user_prompt = create_citation_prompt(query, report_content, current_date)
        
        try:
            # Get evaluation from AI
            result = await client.generate_with_schema_async(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                schema=CITATION_EVALUATION_SCHEMA
            )
            
            return {
                'specific_issues': result.get('specific_issues', []),
                'total_issues': result.get('total_issues', 0),
                'score': result.get('score', 0),
                'justification': result.get('justification', '')
            }
            
        except Exception as e:
            logger.error(f"Error in citation grading: {e}")
            raise
    
    async def grade_async(self, query: str, report_content: str, criterion: str,
                         provider: str = "gemini", model: str = None,
                         current_date: str = None, **kwargs) -> Dict[str, Any]:
        """
        Grade a report on a pointwise criterion.
        
        Args:
            query: Original research query
            report_content: Report to evaluate
            criterion: Criterion name ('consistency' or 'citation')
            provider: AI provider
            model: Specific model to use
            current_date: Current date
            **kwargs: Additional arguments (unused)
            
        Returns:
            Dictionary with evaluation results
        """
        if criterion.lower() == 'consistency':
            return await self.grade_consistency_async(
                query, report_content, provider, model, current_date
            )
        elif criterion.lower() == 'citation':
            return await self.grade_citation_async(
                query, report_content, provider, model, current_date
            )
        else:
            raise ValueError(f"Unknown pointwise criterion: {criterion}")

