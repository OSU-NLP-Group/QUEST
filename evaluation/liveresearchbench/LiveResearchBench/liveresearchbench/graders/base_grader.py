"""Base grader class providing common functionality for all graders."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

from liveresearchbench.common.model_clients import create_ai_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseGrader(ABC):
    """Abstract base class for all graders."""
    
    def __init__(self):
        """Initialize the base grader."""
        pass
    
    @abstractmethod
    async def grade_async(self, query: str, report_content: str, provider: str = "gemini",
                         model: str = None, current_date: str = None, **kwargs) -> Dict[str, Any]:
        """
        Grade a report asynchronously.
        
        Args:
            query: Original research query
            report_content: Report content to evaluate
            provider: AI provider (gemini or openai)
            model: Specific model to use
            current_date: Current date for temporal context
            **kwargs: Additional criterion-specific arguments
            
        Returns:
            Dictionary containing grading results
        """
        pass
    
    def _get_current_date(self, current_date: Optional[str] = None) -> str:
        """Get current date string for evaluation context."""
        if not current_date:
            return datetime.now().strftime("%B %d, %Y")
        return current_date
    
    def _create_client(self, provider: str, model: Optional[str] = None):
        """Create an AI client with optional model specification."""
        kwargs = {}
        if model:
            kwargs['model'] = model
        return create_ai_client(provider, **kwargs)
    
    def _format_result(self, provider: str, model: str, score: Any, 
                      justification: str = "", **additional_fields) -> Dict[str, Any]:
        """
        Format grading result with standard fields.
        
        Args:
            provider: AI provider used
            model: Model used
            score: Score value
            justification: Justification text
            **additional_fields: Any additional fields to include
            
        Returns:
            Formatted result dictionary
        """
        result = {
            'provider': provider,
            'model': model or 'default',
            'graded_at': datetime.now().isoformat(),
            'score': score,
            'justification': justification
        }
        result.update(additional_fields)
        return result

