import asyncio
import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import AggregationStrategy
from mind2web2.llm_client.base_client import LLMClient
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "overleaf_template"
TASK_DESCRIPTION = """
I want to find three Overleaf templates from Overleaf's official template list for my CV. I hope three templates can already display my photo without further customized modification, with the photo placed in the top left corner of my CV. Please return the links to these templates.
"""

EVALUATION_INSTRUCTION = """
The photo doesn't need to be strictly located in the top left corner. Just roughly in the top left corner is acceptable.
"""



# --------------------------------------------------------------------------- #
# Data model for extracted URLs                                               #
# --------------------------------------------------------------------------- #
class TemplateUrlGroup(BaseModel):
    main_url: Optional[str] = None
    related_urls: List[str] = Field(default_factory=list)


class TemplateUrlsExtraction(BaseModel):
    templates: List[TemplateUrlGroup] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_template_urls() -> str:
    return """
    Extract all Overleaf CV template URLs mentioned in the answer, grouping them by template.

    For each template mentioned:
    1. Identify the main template URL (the primary link to the template)
    2. Identify any related URLs that provide additional but direct information about the same template (if any). Notice, related URLs should be directly relevant to the same template, such as links to example documents or official descriptions of that specific template. It should not be just a general link to Overleaf or a general category or gallery page.

    Group these together so we know which URLs refer to the same template.
    If no template URLs are mentioned, return an empty list.

    Note: Only extract URLs that point to Overleaf templates or related content. Ignore any other URLs that might be present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification for each template                                              #
# --------------------------------------------------------------------------- #
async def verify_template(
        evaluator: Evaluator,
        parent_node,
        template: TemplateUrlGroup,
        template_index: int,
) -> None:
    """
    Verify a single template against all requirements using all related URLs.
    """
    # Create a parent node for this template
    template_node = evaluator.add_parallel(
        id=f"template_{template_index}",
        desc=f"Template {template_index + 1} meets all requirements",
        parent=parent_node,
        critical=False,
    )

    # Add critical existence check
    evaluator.add_custom_node(
        result=bool(template.main_url),
        id=f"template_{template_index}_exists",
        desc=f"Template {template_index + 1} URL exists",
        parent=template_node,
        critical=True
    )

    # Add verification node
    verification_node = evaluator.add_leaf(
        id=f"template_{template_index}_verification",
        desc=f"Verify template {template_index + 1} requirements",
        parent=template_node,
        critical=True
    )

    # Prepare all URLs for this template
    all_urls = []
    if template.main_url:
        all_urls.append(template.main_url)
    all_urls.extend(template.related_urls)

    # Verify all requirements
    template_claim = f"""
    The template at {template.main_url}:
    1. Is an official Overleaf template from Overleaf's official template gallery
    2. Already supports displaying a photo without requiring customized modification
    3. Places the photo roughly in the top left corner area of the CV
    """

    await evaluator.verify(
        claim=template_claim,
        node=verification_node,
        sources=all_urls,
        additional_instruction=EVALUATION_INSTRUCTION,
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Initialize evaluator ----------------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
        strategy=AggregationStrategy.PARALLEL
    )

    # -------- 2. Extract template URLs from the answer ----------------- #
    template_urls_data = await evaluator.extract(
        prompt=prompt_extract_template_urls(),
        template_class=TemplateUrlsExtraction,
        extraction_name="template_urls"
    )

    # Get the first 3 templates, pad with empty ones if needed
    templates = list(template_urls_data.templates[:3])
    while len(templates) < 3:
        templates.append(TemplateUrlGroup())

    # -------- 3. Build verification tree -------------------------------- #
    # Verify each of the 3 templates
    for i, template in enumerate(templates):
        await verify_template(evaluator, root, template, i)

    # -------- 4. Aggregate score and prepare result -------------------- #
    final_score = evaluator.score()

    # Add custom information to the summary
    evaluator.add_custom_info(
        {"templates_provided": len([t for t in templates if t.main_url])},
        "template_count"
    )

    return evaluator.get_summary()