import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "public_universities_career_services_spring2026"
TASK_DESCRIPTION = """
Identify four public universities in the United States that meet all of the following criteria:

1. The university must offer one-on-one career counseling appointments to both current students and recent alumni.
2. The university must explicitly state a time limit for alumni eligibility for career counseling services (such as "within one year of graduation" or "recent graduates").
3. The university must host at least one career fair during Spring 2026 (between January 2026 and May 2026).

For each of the four universities, provide:
- The university name and confirmation that it is a public (state-funded) institution
- A link to the official university website or about page
- A link to the career center page that shows one-on-one career counseling is available to students and alumni, including the stated time limit for alumni eligibility
- The specific date (month and day) of at least one Spring 2026 career fair
- The specific location or venue name where the career fair will be held
- A link to the career fair information page or events calendar
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityRecord(BaseModel):
    """Flattened per-university info extracted from the answer."""
    name: Optional[str] = None

    # Public university verification
    official_university_url: Optional[str] = None
    public_status_urls: List[str] = Field(default_factory=list)

    # Career counseling eligibility (students + alumni, time limit)
    career_center_url: Optional[str] = None
    one_on_one_text: Optional[str] = None
    alumni_eligible_text: Optional[str] = None
    alumni_time_limit_text: Optional[str] = None

    # Spring 2026 career fair information
    career_fair_url: Optional[str] = None
    fair_date_text: Optional[str] = None  # e.g., "March 5, 2026"
    fair_location: Optional[str] = None   # venue/building name, etc.


class UniversitiesExtraction(BaseModel):
    """Top-level list of university records extracted from the answer."""
    universities: List[UniversityRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four (4) universities listed in the answer that are claimed to meet all criteria.
    For each university, extract the following fields as a JSON object. Use only information explicitly present in the answer text. Do not invent URLs or facts.

    For each university object, extract:
    - name: The full university name (string)
    - official_university_url: URL to the official university website or about page (string or null)
    - public_status_urls: Array of additional URLs (if any) that the answer cites to support that it is a public (state-funded) university. If none, return an empty array.
    - career_center_url: URL to the career center page that mentions one-on-one career counseling/advising (string or null)
    - one_on_one_text: Short quoted phrase from the answer that indicates one-on-one/individual career counseling/advising appointments are available (string or null)
    - alumni_eligible_text: Short quoted phrase from the answer that indicates alumni are eligible for career counseling/advising (string or null)
    - alumni_time_limit_text: The exact phrase from the answer that states the alumni eligibility time limit (e.g., "within 1 year of graduation", "first-year alumni", or "recent graduates") (string or null)
    - career_fair_url: URL to the career fair information page or events calendar (string or null)
    - fair_date_text: The specific date (month and day, optionally including year) of a Spring 2026 career fair mentioned in the answer (string or null). Examples: "February 12, 2026" or "Mar 5, 2026".
    - fair_location: The specific venue or location name for the cited career fair (string or null)

    Return a JSON object with one field:
    - universities: an array of up to four of these university objects, ordered as they appear in the answer.

    Rules:
    - If a field is missing or not explicitly provided in the answer, set it to null (or an empty array for public_status_urls).
    - Do not infer or add any URLs that are not explicitly present in the answer.
    - Keep string values as they are written in the answer (minor normalization is okay).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(url: Optional[str]) -> bool:
    return bool(url and isinstance(url, str) and url.strip() != "")


def _combine_sources(*sources: List[Optional[str] | List[str] | None]) -> List[str]:
    """Flatten and deduplicate URLs."""
    out: List[str] = []
    for s in sources:
        if not s:
            continue
        if isinstance(s, list):
            for u in s:
                if _non_empty(u) and u not in out:
                    out.append(u)
        elif isinstance(s, str):
            if _non_empty(s) and s not in out:
                out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityRecord,
    index: int,
) -> None:
    """Build verification subtree for a single university."""

    # Create the sequential wrapper for this university (non-critical to allow partial credit across universities)
    uni_node = evaluator.add_sequential(
        id=f"university_{index}",
        desc=f"{['First','Second','Third','Fourth'][index-1]} qualifying university with complete information",
        parent=parent_node,
        critical=False,
    )

    # ------------------- Public University Verification (Critical) ------------------- #
    public_node = evaluator.add_parallel(
        id=f"public_university_verification_{index}",
        desc="Verification that the institution is a public U.S. university",
        parent=uni_node,
        critical=True,
    )

    # Existence of official university URL (critical presence check)
    evaluator.add_custom_node(
        result=_non_empty(uni.official_university_url),
        id=f"official_university_url_{index}",
        desc="Provides URL to official university website or about page",
        parent=public_node,
        critical=True,
    )

    combined_pub_sources = _combine_sources(uni.official_university_url, uni.public_status_urls)

    # Public Institution Status (critical)
    leaf_public_status = evaluator.add_leaf(
        id=f"public_institution_status_{index}",
        desc="Confirms the university is a public institution (state-funded, not private)",
        parent=public_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The institution '{uni.name or 'the university'}' is a public (state-funded) university.",
        node=leaf_public_status,
        sources=combined_pub_sources,
        additional_instruction=(
            "Accept descriptions like 'public university', 'public research university', 'state university', "
            "or membership in a public state system (e.g., CSU, SUNY, UC, etc.). "
            "Prefer explicit statements from the official site or authoritative pages cited in the answer."
        ),
    )

    # U.S. Location (critical)
    leaf_us_loc = evaluator.add_leaf(
        id=f"us_location_{index}",
        desc="Confirms the university is located in the United States",
        parent=public_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The institution '{uni.name or 'the university'}' is located in the United States.",
        node=leaf_us_loc,
        sources=combined_pub_sources,
        additional_instruction=(
            "Verify the campus location is within the United States. Accept state or city references in the U.S."
        ),
    )

    # ------------------- Career Counseling Eligibility (Critical) ------------------- #
    career_node = evaluator.add_parallel(
        id=f"career_counseling_eligibility_{index}",
        desc="Verification of one-on-one career counseling availability for students and alumni",
        parent=uni_node,
        critical=True,
    )

    # Existence of career center URL (critical presence check)
    evaluator.add_custom_node(
        result=_non_empty(uni.career_center_url),
        id=f"career_center_url_{index}",
        desc="Provides URL to career center page with counseling information",
        parent=career_node,
        critical=True,
    )

    # One-on-one counseling available (critical)
    leaf_one_on_one = evaluator.add_leaf(
        id=f"one_on_one_counseling_{index}",
        desc="Confirms availability of individual/one-on-one career counseling appointments",
        parent=career_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The cited career center page indicates that individual or one-on-one career counseling/advising appointments "
            "are available to current students."
        ),
        node=leaf_one_on_one,
        sources=uni.career_center_url,
        additional_instruction=(
            "Look for phrases like 'one-on-one', 'individual appointment', 'career coaching', or similar. "
            "Group workshops alone do NOT satisfy this requirement unless the page also clearly offers individual sessions."
        ),
    )

    # Alumni eligibility (critical)
    leaf_alumni_elig = evaluator.add_leaf(
        id=f"alumni_eligibility_{index}",
        desc="Confirms that recent alumni are eligible for career counseling services",
        parent=career_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The cited career center page indicates that alumni (recent graduates) are eligible for career counseling/advising appointments.",
        node=leaf_alumni_elig,
        sources=uni.career_center_url,
        additional_instruction=(
            "Accept synonyms like 'alumni advising', 'services for recent graduates', 'first-year alumni', etc."
        ),
    )

    # Alumni time limit explicitly stated (critical)
    leaf_alumni_limit = evaluator.add_leaf(
        id=f"alumni_time_limit_{index}",
        desc="Specifies the time limit for alumni eligibility (e.g., within 1 year of graduation)",
        parent=career_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The cited career center page explicitly states a time limit for alumni eligibility for career counseling services "
            "(e.g., 'within one year of graduation', 'first-year alumni', 'alumni up to 12 months after graduation', or 'recent graduates')."
        ),
        node=leaf_alumni_limit,
        sources=uni.career_center_url,
        additional_instruction=(
            "The page should explicitly communicate a time-bound window for alumni counseling eligibility. "
            "Accept phrases like 'recent alumni' or 'recent graduates' when used as a defined eligibility category, "
            "even if the exact number of months is not specified."
        ),
    )

    # ------------------- Spring 2026 Career Fair (Critical) ------------------- #
    fair_node = evaluator.add_parallel(
        id=f"spring_2026_career_fair_{index}",
        desc="Verification of at least one career fair in Spring 2026",
        parent=uni_node,
        critical=True,
    )

    # Existence of career fair URL (critical presence check)
    evaluator.add_custom_node(
        result=_non_empty(uni.career_fair_url),
        id=f"career_fair_url_{index}",
        desc="Provides URL to career fair page or events calendar",
        parent=fair_node,
        critical=True,
    )

    # Specific fair date in Jan–May 2026 (critical)
    leaf_fair_date = evaluator.add_leaf(
        id=f"specific_fair_date_{index}",
        desc="Provides specific date (month and day) for the career fair and confirms it falls between January and May 2026",
        parent=fair_node,
        critical=True,
    )
    date_text_display = uni.fair_date_text or "the cited date"
    await evaluator.verify(
        claim=(
            f"The cited page shows at least one career fair scheduled on {date_text_display}, "
            "and this date falls between January 1, 2026 and May 31, 2026."
        ),
        node=leaf_fair_date,
        sources=uni.career_fair_url,
        additional_instruction=(
            "If the page lists multiple events, it's sufficient that at least one event is a career fair during Jan–May 2026. "
            "Confirm the exact date text appears on the page."
        ),
    )

    # Fair location/venue (critical)
    leaf_fair_loc = evaluator.add_leaf(
        id=f"fair_location_{index}",
        desc="Specifies the physical location or venue name for the career fair",
        parent=fair_node,
        critical=True,
    )
    loc_display = uni.fair_location or "the cited venue/location"
    await evaluator.verify(
        claim=f"The cited page shows the venue/location for the fair as '{loc_display}'.",
        node=leaf_fair_loc,
        sources=uni.career_fair_url,
        additional_instruction="Look for venue names, buildings, halls, ballrooms, student centers, etc.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the 'public universities with counseling + Spring 2026 career fair' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Four universities evaluated independently
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

    # Extract candidate universities and their associated fields from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Record useful config info
    evaluator.add_custom_info(
        info={
            "spring_window_start": "2026-01-01",
            "spring_window_end": "2026-05-31",
            "required_universities": 4,
        },
        info_type="policy",
        info_name="spring_2026_window_and_requirements",
    )

    # Normalize to exactly four slots (pad with empty records if needed)
    universities: List[UniversityRecord] = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityRecord())

    # Build verification tree for each of the four universities
    for i in range(4):
        await verify_university(
            evaluator=evaluator,
            parent_node=root,
            uni=universities[i],
            index=i + 1,
        )

    # Return evaluation summary
    return evaluator.get_summary()