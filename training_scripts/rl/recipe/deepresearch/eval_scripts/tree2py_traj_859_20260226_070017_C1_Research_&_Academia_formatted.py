import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "africa_university_identification"
TASK_DESCRIPTION = (
    "Identify the name of the African university that satisfies all of the following criteria as of the 2025-2026 academic rankings: "
    "(1) ranked #1 in Africa according to the U.S. News 2025-2026 Best Global Universities in Africa rankings, "
    "(2) ranked 164th globally in the Times Higher Education World University Rankings 2026, and "
    "(3) located in South Africa."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityExtraction(BaseModel):
    """
    Extract the identified university name and categorized source URLs cited in the answer.
    """
    university_name: Optional[str] = None
    usnews_urls: List[str] = Field(default_factory=list)
    the_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_info() -> str:
    return (
        "From the provided answer, extract the single university name that is claimed to satisfy all three criteria. "
        "Also extract and categorize all cited URLs that support each criterion as follows:\n"
        "- university_name: The institution explicitly identified as the final answer. If multiple institutions are mentioned, select the one asserted to meet all criteria.\n"
        "- usnews_urls: All URLs that point to U.S. News pages relevant to the 'Best Global Universities in Africa' rankings, especially for the 2025-2026 cycle. "
        "Include only actual URLs mentioned in the answer (plain URLs or markdown links). Prefer domains under 'usnews.com'.\n"
        "- the_urls: All URLs that point to Times Higher Education pages relevant to 'World University Rankings 2026' for the identified institution or ranking list. "
        "Include only actual URLs mentioned in the answer. Prefer domains under 'timeshighereducation.com'.\n"
        "- location_urls: All URLs that can support the claim that the institution is located in South Africa, such as "
        "the university’s official website, Wikipedia, U.S. News profile page, THE profile page, or other credible sources. "
        "Include only actual URLs mentioned in the answer.\n\n"
        "General rules:\n"
        "1) Extract only URLs explicitly present in the answer; do not invent URLs.\n"
        "2) Return valid full URLs; if a URL is missing protocol, prepend 'http://'.\n"
        "3) If a category has no URLs mentioned, return an empty array for that field.\n"
        "4) If no university name is mentioned, set 'university_name' to null."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def union_sources(info: UniversityExtraction) -> List[str]:
    """
    Create a deduplicated union of all available sources across categories.
    Useful for location verification, which may be supported by any credible cited page.
    """
    combined = list(dict.fromkeys((info.location_urls or []) + (info.usnews_urls or []) + (info.the_urls or [])))
    return combined


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def add_usnews_verification(
    evaluator: Evaluator,
    parent_node,
    info: UniversityExtraction,
) -> None:
    """
    Build the US News Africa ranking verification subtree:
    - Ensure US News sources are provided (critical).
    - Verify the claim that the identified university is ranked #1 in Africa in U.S. News 2025-2026.
    """
    usnews_node = evaluator.add_sequential(
        id="US_News_Africa_Ranking",
        desc="The university is ranked #1 in Africa according to the U.S. News 2025-2026 Best Global Universities in Africa rankings.",
        parent=parent_node,
        critical=True
    )

    # Existence of US News sources (Critical)
    evaluator.add_custom_node(
        result=bool(info.usnews_urls),
        id="US_News_sources_provided",
        desc="U.S. News source URLs are provided in the answer",
        parent=usnews_node,
        critical=True
    )

    # Claim verification (Critical)
    usnews_leaf = evaluator.add_leaf(
        id="US_News_Africa_Ranking_check",
        desc="Verify U.S. News 2025-2026 Africa #1 ranking for the identified university",
        parent=usnews_node,
        critical=True
    )

    university = info.university_name or ""
    claim = (
        f"In the U.S. News 2025-2026 Best Global Universities in Africa rankings, {university} is ranked #1 in Africa."
    )
    await evaluator.verify(
        claim=claim,
        node=usnews_leaf,
        sources=info.usnews_urls,
        additional_instruction=(
            "Confirm that the provided U.S. News page(s) correspond to the 2025-2026 Best Global Universities in Africa rankings "
            "or an equivalent official page clearly indicating regional ranking (Africa). "
            "Verify that the identified university is explicitly ranked #1 within Africa on the page. "
            "If the URL(s) are irrelevant, inaccessible, or do not show the #1 Africa rank for the named university, mark as not supported."
        ),
    )


async def add_the_verification(
    evaluator: Evaluator,
    parent_node,
    info: UniversityExtraction,
) -> None:
    """
    Build the THE Global Ranking verification subtree:
    - Ensure THE sources are provided (critical).
    - Verify the claim that the identified university is ranked 164th globally in THE WUR 2026.
    """
    the_node = evaluator.add_sequential(
        id="THE_Global_Ranking_2026",
        desc="The university is ranked 164th globally in the Times Higher Education World University Rankings 2026.",
        parent=parent_node,
        critical=True
    )

    # Existence of THE sources (Critical)
    evaluator.add_custom_node(
        result=bool(info.the_urls),
        id="THE_sources_provided",
        desc="Times Higher Education (THE) source URLs are provided in the answer",
        parent=the_node,
        critical=True
    )

    # Claim verification (Critical)
    the_leaf = evaluator.add_leaf(
        id="THE_Global_Ranking_2026_check",
        desc="Verify THE 2026 global rank 164 for the identified university",
        parent=the_node,
        critical=True
    )

    university = info.university_name or ""
    claim = (
        f"In the Times Higher Education World University Rankings 2026, {university} is ranked 164th globally."
    )
    await evaluator.verify(
        claim=claim,
        node=the_leaf,
        sources=info.the_urls,
        additional_instruction=(
            "Confirm that the provided THE page(s) explicitly refer to 'World University Rankings 2026' and show the identified university's world rank as 164. "
            "Institution profile pages or ranking list pages are acceptable if they clearly display 'World University Rankings 2026' with rank 164 for the institution. "
            "If the year is not 2026 or the rank is not 164, mark as not supported."
        ),
    )


async def add_location_verification(
    evaluator: Evaluator,
    parent_node,
    info: UniversityExtraction,
) -> None:
    """
    Build the geographic location verification subtree:
    - Ensure at least one credible source is provided (critical).
    - Verify the claim that the identified university is located in South Africa.
    """
    loc_node = evaluator.add_sequential(
        id="Geographic_Location",
        desc="The university is located in South Africa.",
        parent=parent_node,
        critical=True
    )

    # Existence of any credible sources for location (Critical)
    location_sources = union_sources(info)
    evaluator.add_custom_node(
        result=bool(location_sources),
        id="Location_sources_provided",
        desc="At least one credible source URL is provided to support the South Africa location claim",
        parent=loc_node,
        critical=True
    )

    # Claim verification (Critical)
    loc_leaf = evaluator.add_leaf(
        id="Geographic_Location_check",
        desc="Verify the institution is located in South Africa",
        parent=loc_node,
        critical=True
    )

    university = info.university_name or ""
    claim = f"{university} is located in South Africa."
    await evaluator.verify(
        claim=claim,
        node=loc_leaf,
        sources=location_sources,
        additional_instruction=(
            "Check the provided page(s) for the institution's location. "
            "Any credible page (official university site, Wikipedia, U.S. News profile, THE profile) that explicitly shows the university is in South Africa suffices. "
            "Minor formatting or naming variations (e.g., city names, abbreviations) are acceptable as long as 'South Africa' is clearly indicated."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the African university identification task.
    Builds a critical parallel node 'University_Identification' with three critical sequential subtrees.
    """
    # Initialize evaluator and root
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
        default_model=model,
    )

    # Extract university and sources from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_university_info(),
        template_class=UniversityExtraction,
        extraction_name="university_extraction"
    )

    # Build the main critical node
    uni_node = evaluator.add_parallel(
        id="University_Identification",
        desc="The identified university must satisfy all three specified criteria: U.S. News Africa ranking, THE global ranking, and geographic location.",
        parent=root,
        critical=True
    )

    # Add three critical verification branches
    await add_usnews_verification(evaluator, uni_node, extracted_info)
    await add_the_verification(evaluator, uni_node, extracted_info)
    await add_location_verification(evaluator, uni_node, extracted_info)

    # Return structured summary with verification tree
    return evaluator.get_summary()