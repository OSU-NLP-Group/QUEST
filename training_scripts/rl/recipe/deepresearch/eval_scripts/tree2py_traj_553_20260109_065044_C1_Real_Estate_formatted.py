import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "travis_homestead_exemption"
TASK_DESCRIPTION = (
    "I recently purchased a home in Travis County, Texas, and I want to apply for a homestead exemption to reduce my property taxes. "
    "Please provide the official website for the Travis County Appraisal District where I can find the application form, and tell me "
    "the standard deadline for filing a homestead exemption application."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HomesteadInfo(BaseModel):
    official_website_url: Optional[str] = None
    deadline_text: Optional[str] = None
    deadline_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_homestead_info() -> str:
    return """
    Extract the following information exactly as presented in the answer:
    1) official_website_url: The official website URL of the Travis Central Appraisal District (TCAD), i.e., the appraisal district serving Travis County, Texas, where homestead exemption information or forms are available. Return exactly one URL if the answer clearly provides it. If multiple URLs are present, prefer the main homepage or a clearly official TCAD page.
    2) deadline_text: The standard deadline stated in the answer for filing a residence homestead exemption application (e.g., 'April 30' or 'by April 30'). Extract it as a short phrase or date exactly as written in the answer.
    3) deadline_urls: All URLs that the answer cites to support the deadline information (e.g., the TCAD website page or the Texas Comptroller page). If the answer does not provide any specific source for the deadline, return an empty list.
    
    Rules:
    - Extract only URLs explicitly provided in the answer text (including markdown links).
    - Do not invent or infer any URLs.
    - If any required field is missing in the answer, return null for that field (or an empty list for deadline_urls).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_homestead_info(evaluator: Evaluator, parent_node, extracted: HomesteadInfo) -> None:
    # Existence checks as separate critical leaf nodes to gate subsequent verifications
    website_provided = evaluator.add_custom_node(
        result=bool(extracted.official_website_url and extracted.official_website_url.strip()),
        id="Official_Website_Provided",
        desc="Official website URL is provided in the answer",
        parent=parent_node,
        critical=True
    )

    deadline_provided = evaluator.add_custom_node(
        result=bool(extracted.deadline_text and extracted.deadline_text.strip()),
        id="Application_Deadline_Provided",
        desc="Application deadline is stated in the answer",
        parent=parent_node,
        critical=True
    )

    # Leaf: Official Website verification
    official_node = evaluator.add_leaf(
        id="Official_Website",
        desc="Provide the official Travis Central Appraisal District website URL",
        parent=parent_node,
        critical=True
    )
    official_claim = (
        "This webpage belongs to the official website of the Travis Central Appraisal District (TCAD), "
        "the appraisal district serving Travis County, Texas."
    )
    await evaluator.verify(
        claim=official_claim,
        node=official_node,
        sources=extracted.official_website_url,
        additional_instruction=(
            "Verify that the page explicitly identifies itself as the Travis Central Appraisal District (TCAD) "
            "or the appraisal district for Travis County, Texas (e.g., in the header, footer, logo, about text, or contact info). "
            "Pages from unrelated government offices or third-party sites are not acceptable."
        ),
        extra_prerequisites=[website_provided]
    )

    # Leaf: Application Deadline verification
    deadline_node = evaluator.add_leaf(
        id="Application_Deadline",
        desc="State the standard deadline for filing a homestead exemption application in Travis County",
        parent=parent_node,
        critical=True
    )

    # Build source list for deadline verification
    deadline_sources: List[str] = list(extracted.deadline_urls or [])
    if extracted.official_website_url:
        if extracted.official_website_url not in deadline_sources:
            deadline_sources.append(extracted.official_website_url)

    deadline_text_display = extracted.deadline_text or ""
    deadline_claim = (
        f"The standard deadline for filing a residence homestead exemption application in Travis County, Texas is {deadline_text_display}."
    )
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_node,
        sources=deadline_sources if deadline_sources else None,
        additional_instruction=(
            "Confirm the standard filing deadline as stated on the provided official page(s). "
            "If the source states a filing window like 'between January 1 and April 30', treat the standard deadline as 'April 30'. "
            "Ignore exceptions or late-filing provisions (e.g., filing after the deadline or up to two years after the delinquency date); "
            "focus on the normal/standard annual deadline referenced for homestead exemption applications."
        ),
        extra_prerequisites=[deadline_provided]
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator with a parallel root, matching the rubric root aggregation
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Provide the official website and application deadline for Travis County, Texas homestead exemption",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extraction
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_homestead_info(),
        template_class=HomesteadInfo,
        extraction_name="homestead_info"
    )

    # Verification
    await verify_homestead_info(evaluator, root, extracted_info)

    # Summary
    return evaluator.get_summary()