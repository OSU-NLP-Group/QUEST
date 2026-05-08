"""Checklist grader for binary (0/1) evaluation of checklist items."""

import logging
from typing import Dict, Any, Optional

from .base_grader import BaseGrader
from liveresearchbench.criteria.presentation import (
    PRESENTATION_QUESTIONS, 
    PRESENTATION_EVALUATION_SCHEMA,
    create_presentation_prompt
)
from liveresearchbench.criteria.coverage import (
    COVERAGE_EVALUATION_SCHEMA,
    create_coverage_prompt
)

logger = logging.getLogger(__name__)


class ChecklistGrader(BaseGrader):
    """Grader for checklist-based evaluation (Presentation & Coverage)."""
    
    def __init__(self):
        """Initialize the checklist grader."""
        super().__init__()
    
    async def grade_presentation_async(self, query: str, report_content: str, 
                                      provider: str = "gemini", model: str = None,
                                      current_date: str = None) -> Dict[str, Any]:
        """
        Grade a report on presentation quality (10 binary questions).
        
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
        system_prompt, user_prompt = create_presentation_prompt(query, report_content, current_date)
        
        try:
            # Get evaluation from AI
            result = await client.generate_with_schema_async(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                schema=PRESENTATION_EVALUATION_SCHEMA
            )
            
            evaluations = result.get('evaluations', {})
            
            # Calculate summary statistics
            total_criteria = len(PRESENTATION_QUESTIONS)
            total_score = sum(eval_data.get('score', 0) for eval_data in evaluations.values())
            passed_count = total_score
            failed_count = total_criteria - passed_count
            average_pass_rate = (passed_count / total_criteria * 100) if total_criteria > 0 else 0
            
            return {
                'evaluations': evaluations,
                'summary': {
                    'total_criteria': total_criteria,
                    'total_score': total_score,
                    'passed_count': passed_count,
                    'failed_count': failed_count,
                    'average_pass_rate': average_pass_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Error in presentation grading: {e}")
            raise
    
    async def grade_coverage_async(self, query: str, report_content: str,
                                  checklists, provider: str = "gemini",
                                  model: str = None, current_date: str = None) -> Dict[str, Any]:
        """
        Grade a report on coverage/comprehensiveness using custom checklists.
        
        Args:
            query: Original research query
            report_content: Report to evaluate
            checklists: List of checklist items (strings) or Dict of {checklist_id: checklist_text}
            provider: AI provider (gemini or openai)
            model: Specific model to use
            current_date: Current date for temporal context
            
        Returns:
            Dictionary with evaluation results
        """
        # Convert dict to list if needed (for backward compatibility)
        if isinstance(checklists, dict):
            checklists = list(checklists.values())
        current_date = self._get_current_date(current_date)
        client = self._create_client(provider, model)
        
        # Create prompts
        system_prompt, user_prompt = create_coverage_prompt(query, report_content, checklists, current_date)
        
        try:
            # Get evaluation from AI
            result = await client.generate_with_schema_async(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                schema=COVERAGE_EVALUATION_SCHEMA
            )
            
            evaluations = result.get('evaluations', {})
            
            # Calculate summary statistics
            total_checklists = len(checklists)
            total_score = sum(eval_data.get('score', 0) for eval_data in evaluations.values())
            addressed_count = total_score
            not_addressed_count = total_checklists - addressed_count
            percentage_addressed = (addressed_count / total_checklists * 100) if total_checklists > 0 else 0
            
            return {
                'evaluations': evaluations,
                'summary': {
                    'total_checklists': total_checklists,
                    'total_score': total_score,
                    'addressed_count': addressed_count,
                    'not_addressed_count': not_addressed_count,
                    'percentage_addressed': percentage_addressed
                }
            }
            
        except Exception as e:
            logger.error(f"Error in coverage grading: {e}")
            raise
    
    async def grade_async(self, query: str, report_content: str, criterion: str,
                         provider: str = "gemini", model: str = None,
                         current_date: str = None, **kwargs) -> Dict[str, Any]:
        """
        Grade a report on a checklist-based criterion.
        
        Args:
            query: Original research query
            report_content: Report to evaluate
            criterion: Criterion name ('presentation' or 'coverage')
            provider: AI provider
            model: Specific model to use
            current_date: Current date
            **kwargs: Additional arguments (checklists for coverage)
            
        Returns:
            Dictionary with evaluation results
        """
        if criterion.lower() == 'presentation':
            return await self.grade_presentation_async(
                query, report_content, provider, model, current_date
            )
        elif criterion.lower() == 'coverage':
            checklists = kwargs.get('checklists', {})
            if not checklists:
                raise ValueError("Checklists required for coverage evaluation")
            return await self.grade_coverage_async(
                query, report_content, checklists, provider, model, current_date
            )
        else:
            raise ValueError(f"Unknown checklist criterion: {criterion}")

