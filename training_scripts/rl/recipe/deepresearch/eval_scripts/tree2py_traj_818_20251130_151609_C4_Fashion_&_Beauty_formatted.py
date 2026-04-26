import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "loreal_men_expert_ambassador_may_2025"
TASK_DESCRIPTION = (
    "Evaluate whether the answer identifies the May 2025 L’Oréal Men Expert global brand ambassador "
    "(a professional racing driver) and provides all requested attributes with at least one supporting source URL."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AmbassadorInfo(BaseModel):
    """Structured extraction of the ambassador-related information from the answer."""
    full_name: Optional[str] = None
    announcement_date: Optional[str] = None  # Expecting format 'Month DD, YYYY' in May 2025
    nationality: Optional[str] = None
    racing_series: Optional[str] = None
    loreal_hq_country: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_ambassador_info() -> str:
    return """
    Extract the following information exactly as stated in the answer. Do not invent or infer any content that is not explicitly in the answer.

    Required fields:
    1. full_name: The ambassador’s full name as provided in the answer text.
    2. announcement_date: The exact date of the announcement, in the format "Month DD, YYYY". Extract it exactly as written in the answer. If the answer does not state the date, return null.
    3. nationality: The ambassador’s nationality or country of origin as given in the answer. If not provided, return null.
    4. racing_series: The specific racing series or championship the ambassador competes in professionally (e.g., "Formula 1", "FIA World Endurance Championship", "IndyCar", etc.). If not provided, return null.
    5. loreal_hq_country: The country where L'Oréal’s corporate headquarters are located, as provided in the answer. If not provided, return null.
    6. source_urls: A list of all source URLs explicitly included in the answer that support the identification/announcement. Extract only actual URLs (including markdown links); do not infer or add any new URLs. If none are present, return an empty list.

    Notes:
    - For the announcement_date, return the date string exactly as written in the answer. Do not reformat.
    - For source_urls, include only valid URLs that appear in the answer. If a URL is missing protocol, prepend 'http://'.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_full_name(name: Optional[str]) -> bool:
    """Basic heuristic: a full name should be non-empty and contain at least a space."""
    if not name:
        return False
    s = name.strip()
    if not s or " " not in s:
        return False
    # Avoid names with digits
    if any(ch.isdigit() for ch in s):
        return False
    return True


def _is_valid_may_2025_date(date_str: Optional[str]) -> bool:
    """
    Check that the date is present, formatted as 'Month DD, YYYY', and corresponds to May 2025.
    """
    if not date_str:
        return False
    try:
        dt = datetime.strptime(date_str.strip(), "%B %d, %Y")
        return dt.year == 2025 and dt.month == 5
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Verification building                                                       #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    extracted: AmbassadorInfo
) -> None:
    """
    Build the verification tree and perform required checks according to the rubric.
    """
    # Top-level critical node as per rubric
    main_node = evaluator.add_parallel(
        id="LOreal_Men_Expert_Global_Ambassador_May_2025",
        desc="Evaluate whether the answer identifies the May 2025 L’Oréal Men Expert global brand ambassador (a professional racing driver) and provides all requested attributes with at least one supporting source URL.",
        parent=root_node,
        critical=True
    )

    # 1) Context_Correct: Check the answer states this is L’Oréal Men Expert's new global brand ambassador
    #    and that the person is a professional racing driver.
    context_node = evaluator.add_leaf(
        id="Context_Correct",
        desc="Answer states this is the new global brand ambassador announcement for L’Oréal Men Expert (men’s skincare line) and that the ambassador is a professional racing driver.",
        parent=main_node,
        critical=True
    )
    context_claim = (
        "The answer explicitly identifies the person as the new global brand ambassador for L’Oréal Men Expert "
        "(a men's skincare line) and explicitly states that the person is a professional racing driver."
    )
    await evaluator.verify(
        claim=context_claim,
        node=context_node,
        additional_instruction="Focus on the presence of these statements in the answer itself. Minor wording variations are acceptable if the meaning is the same."
    )

    # 2) Ambassador_Full_Name_Provided: existence check
    evaluator.add_custom_node(
        result=_is_full_name(extracted.full_name),
        id="Ambassador_Full_Name_Provided",
        desc="Answer provides the ambassador’s full name (the identified person).",
        parent=main_node,
        critical=True
    )

    # 3) Announcement_Date_Provided_And_Formatted: check presence and format + that it's in May 2025
    evaluator.add_custom_node(
        result=_is_valid_may_2025_date(extracted.announcement_date),
        id="Announcement_Date_Provided_And_Formatted",
        desc='Answer provides the exact announcement date, formatted as "Month DD, YYYY", and the date is in May 2025.',
        parent=main_node,
        critical=True
    )

    # 4) Nationality_Provided: existence check
    evaluator.add_custom_node(
        result=bool(extracted.nationality and extracted.nationality.strip()),
        id="Nationality_Provided",
        desc="Answer provides the ambassador’s nationality or country of origin.",
        parent=main_node,
        critical=True
    )

    # 5) Racing_Series_Provided: existence check
    evaluator.add_custom_node(
        result=bool(extracted.racing_series and extracted.racing_series.strip()),
        id="Racing_Series_Provided",
        desc="Answer provides the specific racing series or championship in which the ambassador competes professionally.",
        parent=main_node,
        critical=True
    )

    # 6) LOreal_HQ_Country_Provided: existence check
    evaluator.add_custom_node(
        result=bool(extracted.loreal_hq_country and extracted.loreal_hq_country.strip()),
        id="LOreal_HQ_Country_Provided",
        desc="Answer provides the country where L’Oréal’s corporate headquarters are located.",
        parent=main_node,
        critical=True
    )

    # 7) Source_URL_Provided_And_Relevant: verify by URLs that at least one provided source explicitly supports
    #    the identification/announcement of the ambassador for L’Oréal Men Expert.
    source_node = evaluator.add_leaf(
        id="Source_URL_Provided_And_Relevant",
        desc="Answer includes at least one reliable source URL that supports the identification of the ambassador and the announcement (e.g., announcement/press/social post or reputable news/official page).",
        parent=main_node,
        critical=True
    )

    display_name = extracted.full_name or "the ambassador"
    source_claim = (
        f"{display_name} is the new global brand ambassador for L'Oréal Men Expert (men's skincare line). "
        f"At least one of the provided URLs explicitly announces or reports this."
    )
    await evaluator.verify(
        claim=source_claim,
        node=source_node,
        sources=extracted.source_urls,  # Pass the list; will fail if empty
        additional_instruction=(
            "Treat as reliable sources: official L’Oréal/L’Oréal Men Expert press releases or corporate pages, "
            "official verified social posts (e.g., brand's verified accounts), and reputable news outlets. "
            "Reject irrelevant, dead, or unrelated pages. Minor wording variations like 'global ambassador' vs. 'brand ambassador' "
            "are acceptable when clearly referring to L’Oréal Men Expert."
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
    Evaluate the answer for the L’Oréal Men Expert May 2025 global brand ambassador task.
    """
    # Initialize evaluator (root node is non-critical; we add a critical main node under it)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
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
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_ambassador_info(),
        template_class=AmbassadorInfo,
        extraction_name="ambassador_info"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted_info)

    # Return standardized evaluation summary
    return evaluator.get_summary()