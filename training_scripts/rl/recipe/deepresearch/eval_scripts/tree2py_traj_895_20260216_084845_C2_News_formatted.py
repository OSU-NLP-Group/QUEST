import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "powell_fed_terms"
TASK_DESCRIPTION = """
Regarding Jerome Powell's role at the Federal Reserve, provide the following information:
(1) The end date of his current term as Chair of the Board of Governors,
(2) The end date of his term as a member of the Board of Governors,
(3) An explanation of how a person can simultaneously hold both a Chair position and a Board Governor position with different term end dates.
For items 1 and 2, include supporting URL references from official Federal Reserve sources or authoritative sources.
""".strip()

# Ground truth expectations (used for verification claims)
CHAIR_END_DATE_EXPECTED = "May 23, 2026"
GOVERNOR_END_DATE_EXPECTED = "January 31, 2028"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PowellTermsExtraction(BaseModel):
    chair_term_end: Optional[str] = None
    chair_sources: List[str] = Field(default_factory=list)
    governor_term_end: Optional[str] = None
    governor_sources: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_powell_terms() -> str:
    return """
    Extract from the provided answer the following fields about Jerome Powell's Federal Reserve roles:

    1) chair_term_end: The end date stated for his current term as Chair of the Board of Governors (return as a single string exactly as written in the answer, e.g., "May 23, 2026" or "05/23/2026").
    2) chair_sources: A list of all URL(s) provided in the answer that specifically support the Chair term end date. Only include valid URLs explicitly present in the answer.
    3) governor_term_end: The end date stated for his term as a member of the Board of Governors (return as a single string exactly as written in the answer, e.g., "January 31, 2028" or "01/31/2028").
    4) governor_sources: A list of all URL(s) provided in the answer that specifically support the Board Governor term end date. Only include valid URLs explicitly present in the answer.
    5) explanation: The explanation text (as a single string) that describes how a person can simultaneously hold the Chair position and a Board Governor position with different term end dates. Return null if not provided.

    Rules:
    - Do not infer or invent URLs. Only extract URLs that appear in the answer content (including in markdown link format).
    - If a field is not present, return null (or empty list for sources).
    """.strip()


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_chair_term_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: PowellTermsExtraction,
) -> None:
    """
    Build and verify the subtree for the Chair position term.
    Critical node: All children must be critical and pass.
    """
    chair_node = evaluator.add_parallel(
        id="Chair_Position_Term",
        desc="Provides accurate information about Jerome Powell's term as Chair of the Federal Reserve",
        parent=parent_node,
        critical=True
    )

    # Existence check for sources (to enforce source-grounding)
    evaluator.add_custom_node(
        result=bool(extracted.chair_sources),
        id="Chair_Source_Provided",
        desc="At least one Chair term source URL is provided in the answer",
        parent=chair_node,
        critical=True
    )

    # Leaf: URL reference support & authority
    chair_ref_leaf = evaluator.add_leaf(
        id="Chair_Reference_URL",
        desc="Provides a valid URL reference from an official Federal Reserve source or authoritative news source documenting Powell's Chair term end date",
        parent=chair_node,
        critical=True
    )
    chair_ref_claim = (
        f"This webpage is an official Federal Reserve source (federalreserve.gov) or a highly authoritative source "
        f"(e.g., a .gov site, Congress/White House, or a major reputable news outlet) that explicitly states Jerome "
        f"H. Powell's current term as Chair of the Board of Governors ends on {CHAIR_END_DATE_EXPECTED}."
    )
    await evaluator.verify(
        claim=chair_ref_claim,
        node=chair_ref_leaf,
        sources=extracted.chair_sources if extracted.chair_sources else None,
        additional_instruction=(
            "First, confirm the domain is an official Federal Reserve page (federalreserve.gov) or a highly "
            "authoritative site (.gov, White House, or a major reputable news outlet). Then verify the page "
            f"explicitly includes the Chair term end date {CHAIR_END_DATE_EXPECTED}. Accept minor formatting "
            "differences for the date (e.g., '05/23/2026'), but the exact day, month, and year must match."
        )
    )

    # Leaf: The answer states the correct Chair end date
    chair_date_leaf = evaluator.add_leaf(
        id="Chair_Term_End_Date",
        desc=f"States that Jerome Powell's current term as Chair of the Board of Governors ends on {CHAIR_END_DATE_EXPECTED}",
        parent=chair_node,
        critical=True
    )
    chair_date_claim = (
        f"In the provided answer, it is explicitly stated that Jerome Powell's current term as Chair of the Board of "
        f"Governors ends on {CHAIR_END_DATE_EXPECTED}."
    )
    await evaluator.verify(
        claim=chair_date_claim,
        node=chair_date_leaf,
        sources=None,
        additional_instruction=(
            "Check the answer text to see if it clearly states the end date exactly as "
            f"'{CHAIR_END_DATE_EXPECTED}' (allowing trivial punctuation or preposition differences such as "
            "'term ends on' vs 'term expires on')."
        )
    )


async def build_governor_term_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: PowellTermsExtraction,
) -> None:
    """
    Build and verify the subtree for the Board Governor term.
    Critical node: All children must be critical and pass.
    """
    gov_node = evaluator.add_parallel(
        id="Board_Governor_Term",
        desc="Provides accurate information about Jerome Powell's term as a Board Governor",
        parent=parent_node,
        critical=True
    )

    # Existence check for sources (to enforce source-grounding)
    evaluator.add_custom_node(
        result=bool(extracted.governor_sources),
        id="Governor_Source_Provided",
        desc="At least one Governor term source URL is provided in the answer",
        parent=gov_node,
        critical=True
    )

    # Leaf: URL reference support & official Fed
    gov_ref_leaf = evaluator.add_leaf(
        id="Governor_Reference_URL",
        desc="Provides a valid URL reference from an official Federal Reserve source documenting Powell's Board Governor term end date",
        parent=gov_node,
        critical=True
    )
    gov_ref_claim = (
        f"This webpage is an official Federal Reserve page (domain federalreserve.gov) that explicitly states that "
        f"Jerome H. Powell's term as a member of the Board of Governors ends on {GOVERNOR_END_DATE_EXPECTED}."
    )
    await evaluator.verify(
        claim=gov_ref_claim,
        node=gov_ref_leaf,
        sources=extracted.governor_sources if extracted.governor_sources else None,
        additional_instruction=(
            "Verify the URL is hosted on the official Federal Reserve domain (federalreserve.gov). Then confirm the page "
            f"explicitly includes the Board Governor term end date {GOVERNOR_END_DATE_EXPECTED}. Accept minor date "
            "format variations (e.g., '01/31/2028'), but the exact day, month, and year must match."
        )
    )

    # Leaf: The answer states the correct Governor end date
    gov_date_leaf = evaluator.add_leaf(
        id="Governor_Term_End_Date",
        desc=f"States that Jerome Powell's term as a member of the Board of Governors ends on {GOVERNOR_END_DATE_EXPECTED}",
        parent=gov_node,
        critical=True
    )
    gov_date_claim = (
        f"In the provided answer, it is explicitly stated that Jerome Powell's term as a member of the Board of Governors "
        f"ends on {GOVERNOR_END_DATE_EXPECTED}."
    )
    await evaluator.verify(
        claim=gov_date_claim,
        node=gov_date_leaf,
        sources=None,
        additional_instruction=(
            "Check the answer text to see if it clearly states the end date exactly as "
            f"'{GOVERNOR_END_DATE_EXPECTED}' (allowing trivial punctuation or preposition differences)."
        )
    )


async def build_explanation_leaf(
    evaluator: Evaluator,
    parent_node,
    extracted: PowellTermsExtraction
) -> None:
    """
    Build and verify the explanation leaf about how the Chair role and Board Governor role can have different end dates.
    """
    explanation_leaf = evaluator.add_leaf(
        id="Term_Structure_Explanation",
        desc="Explains that Powell's Chair position and Board Governor membership are two separate terms that can run concurrently",
        parent=parent_node,
        critical=True
    )
    explanation_claim = (
        "The answer explains that the Chair position is a separate 4‑year leadership designation among the Board of "
        "Governors, distinct from the underlying Governor term (generally much longer, e.g., 14 years), so the two "
        "appointments can run concurrently and therefore have different end dates."
    )
    await evaluator.verify(
        claim=explanation_claim,
        node=explanation_leaf,
        sources=None,
        additional_instruction=(
            "Verify based on the answer text that it conveys the key ideas: (1) the Chair is a separate appointment from "
            "Board membership; (2) the Chair term is a fixed shorter term (commonly 4 years); (3) the Governor term is longer; "
            "(4) hence they can overlap/run concurrently and end on different dates. Minor wording variations are acceptable."
        )
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
    """
    Evaluate an answer for Jerome Powell's Federal Reserve terms.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_powell_terms(),
        template_class=PowellTermsExtraction,
        extraction_name="powell_terms_extraction"
    )

    # Add ground truth info for traceability
    evaluator.add_ground_truth(
        {
            "expected_chair_term_end": CHAIR_END_DATE_EXPECTED,
            "expected_governor_term_end": GOVERNOR_END_DATE_EXPECTED,
            "notes": "Chair term is a separate 4-year appointment; Governor term is a longer appointment; end dates may differ."
        },
        gt_type="ground_truth"
    )

    # Build critical top-level node mirroring the rubric
    top_node = evaluator.add_parallel(
        id="Powell_Federal_Reserve_Terms",
        desc="Provides complete and accurate information about Jerome Powell's two distinct terms at the Federal Reserve",
        parent=root,
        critical=True
    )

    # Chair subtree
    await build_chair_term_subtree(evaluator, top_node, extracted)

    # Governor subtree
    await build_governor_term_subtree(evaluator, top_node, extracted)

    # Explanation leaf
    await build_explanation_leaf(evaluator, top_node, extracted)

    # Return summary
    return evaluator.get_summary()