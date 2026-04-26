import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "acm_doctoral_2022_award_advisor"
TASK_DESCRIPTION = (
    "Identify the 2022 ACM Doctoral Dissertation Award (main award) recipient and provide complete, "
    "verifiable information about that recipient’s PhD advisor: full name, current faculty university, "
    "current academic title/position, and primary research areas."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AwardInfo(BaseModel):
    recipient_name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class AdvisorInfo(BaseModel):
    name: Optional[str] = None
    relationship_source_urls: List[str] = Field(default_factory=list)
    profile_urls: List[str] = Field(default_factory=list)

    current_university: Optional[str] = None
    university_source_urls: List[str] = Field(default_factory=list)

    current_title: Optional[str] = None
    title_source_urls: List[str] = Field(default_factory=list)

    primary_research_areas: List[str] = Field(default_factory=list)
    research_source_urls: List[str] = Field(default_factory=list)


class AwardAdvisorExtraction(BaseModel):
    award: Optional[AwardInfo] = None
    advisor: Optional[AdvisorInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_award_and_advisor() -> str:
    return """
    Extract structured information from the answer about the 2022 ACM Doctoral Dissertation Award (main award) recipient and their PhD advisor. Return a single JSON object with the following schema:

    award:
      - recipient_name: The full name of the main award recipient (NOT honorable mentions). If multiple names are listed, select the one clearly identified as the main award recipient.
      - source_urls: All URLs explicitly provided in the answer that support the identification of the 2022 ACM Doctoral Dissertation Award recipient. Include only valid URLs mentioned in the answer.

    advisor:
      - name: The advisor’s full name (as presented in the answer).
      - relationship_source_urls: URLs explicitly provided in the answer that support the advisor-advisee relationship between the advisor and the award recipient (e.g., academic profiles, official pages, CVs).
      - profile_urls: URLs to the advisor’s official university or lab profile pages, if provided in the answer.

      - current_university: The university where the advisor currently holds a primary faculty appointment (as stated in the answer).
      - university_source_urls: URLs that support the advisor’s current faculty appointment at that university (official university pages preferred).

      - current_title: The advisor’s current academic title/position (e.g., Professor, Associate Professor, Endowed Chair) as stated in the answer.
      - title_source_urls: URLs that support the advisor’s current title/position at the university (official university pages preferred).

      - primary_research_areas: A list of the advisor’s primary research area(s) in computer science as presented in the answer. Keep each area as a concise string (e.g., "distributed systems", "machine learning", "PL").
      - research_source_urls: URLs that support the stated research area(s) (advisor’s official profile, publications page, Google Scholar, etc.), explicitly mentioned in the answer.

    Rules:
    - Only include URLs that are explicitly present in the answer. Do not invent or infer URLs.
    - If any field is missing in the answer, set it to null (for single value fields) or an empty list (for URL lists or research areas).
    - Ensure that "recipient_name" corresponds to the main award recipient (not an honorable mention).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s) and bool(s.strip())


def _merge_sources(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        for url in lst:
            if url and url not in merged:
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_award_verification(
    evaluator: Evaluator,
    parent_node,
    extracted: AwardAdvisorExtraction,
) -> None:
    """
    Build and run verification for the award winner identification.
    """
    award_desc = (
        "Correctly identifies the recipient of the 2022 ACM Doctoral Dissertation Award "
        "(main award, not an honorable mention), consistent with official ACM award records."
    )
    award_node = evaluator.add_parallel(
        id="award_winner_identified",
        desc=award_desc,
        parent=parent_node,
        critical=True,
    )

    recipient_name = extracted.award.recipient_name if extracted.award else None
    award_sources = (extracted.award.source_urls if extracted.award else []) or []

    # Existence check: recipient name provided
    evaluator.add_custom_node(
        result=_non_empty_str(recipient_name),
        id="award_winner_name_provided",
        desc="Award recipient name is provided for the 2022 ACM Doctoral Dissertation Award (main award).",
        parent=award_node,
        critical=True,
    )

    # Source-backed verification: recipient matches ACM records
    award_verify_leaf = evaluator.add_leaf(
        id="award_winner_supported_by_sources",
        desc="The identified 2022 ACM Doctoral Dissertation Award recipient is supported by cited sources.",
        parent=award_node,
        critical=True,
    )
    award_claim = (
        f"The recipient of the 2022 ACM Doctoral Dissertation Award (main award) is {recipient_name}."
        if _non_empty_str(recipient_name) else
        "The recipient of the 2022 ACM Doctoral Dissertation Award (main award) is correctly identified."
    )
    await evaluator.verify(
        claim=award_claim,
        node=award_verify_leaf,
        sources=award_sources,  # multi-URL verification if multiple, single if one, none -> simple
        additional_instruction=(
            "Confirm this is the MAIN award recipient for 2022, not an honorable mention. "
            "Rely on official ACM award pages or authoritative sources. If the provided webpage is irrelevant or "
            "inaccessible, treat the claim as not supported."
        ),
    )


async def build_advisor_relationship_verification(
    evaluator: Evaluator,
    parent_node,
    extracted: AwardAdvisorExtraction,
) -> None:
    """
    Build and run verification for identifying the PhD advisor of the award recipient.
    """
    advisor_rel_desc = (
        "Correctly identifies the PhD advisor of the award recipient, with the advisor-advisee "
        "relationship verifiable via academic records or official university sources."
    )
    advisor_rel_node = evaluator.add_parallel(
        id="phd_advisor_identified",
        desc=advisor_rel_desc,
        parent=parent_node,
        critical=True,
    )

    recipient_name = extracted.award.recipient_name if extracted.award else None
    advisor_name = extracted.advisor.name if extracted.advisor else None
    rel_sources = _merge_sources(
        extracted.advisor.relationship_source_urls if extracted.advisor else [],
        extracted.advisor.profile_urls if extracted.advisor else [],
    )

    # Existence check: advisor name provided
    evaluator.add_custom_node(
        result=_non_empty_str(advisor_name),
        id="phd_advisor_name_provided",
        desc="PhD advisor name for the award recipient is provided.",
        parent=advisor_rel_node,
        critical=True,
    )

    # Source-backed verification: advisor-advisee relationship
    advisor_rel_leaf = evaluator.add_leaf(
        id="phd_advisor_relationship_supported",
        desc="Advisor-advisee relationship between the identified advisor and the award recipient is supported by cited sources.",
        parent=advisor_rel_node,
        critical=True,
    )
    rel_claim = (
        f"{advisor_name} is the PhD advisor of {recipient_name}."
        if _non_empty_str(advisor_name) and _non_empty_str(recipient_name) else
        "The identified PhD advisor-advisee relationship is correct."
    )
    await evaluator.verify(
        claim=rel_claim,
        node=advisor_rel_leaf,
        sources=rel_sources,
        additional_instruction=(
            "Verify the doctoral advisor relationship via official university profiles, CVs, or authoritative records. "
            "Allow minor variations in name formatting. If co-advisors exist, the claim should explicitly match one of them."
        ),
    )


async def build_advisor_details_verification(
    evaluator: Evaluator,
    parent_node,
    extracted: AwardAdvisorExtraction,
) -> None:
    """
    Build and run verification for advisor details (name, current university, current title, research areas).
    """
    details_desc = "Provides the required current, verifiable advisor details (all fields)."
    details_node = evaluator.add_parallel(
        id="advisor_information_provided",
        desc=details_desc,
        parent=parent_node,
        critical=True,
    )

    advisor = extracted.advisor or AdvisorInfo()

    # Common sources: profile + any category-specific URLs
    common_sources = _merge_sources(advisor.profile_urls)

    # 1) Advisor Full Name
    full_name_group = evaluator.add_parallel(
        id="advisor_full_name",
        desc="Provides the advisor’s full name accurately.",
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_str(advisor.name),
        id="advisor_full_name_provided",
        desc="Advisor full name is provided.",
        parent=full_name_group,
        critical=True,
    )
    full_name_leaf = evaluator.add_leaf(
        id="advisor_full_name_accurate",
        desc="Advisor full name is accurately supported by cited sources.",
        parent=full_name_group,
        critical=True,
    )
    await evaluator.verify(
        claim=(f"The advisor’s full name is {advisor.name}." if _non_empty_str(advisor.name)
               else "The advisor’s full name is correctly provided."),
        node=full_name_leaf,
        sources=_merge_sources(common_sources, advisor.relationship_source_urls),
        additional_instruction=(
            "Verify the advisor’s name via official university profiles or other authoritative sources. "
            "Allow minor variations (middle initials, diacritics, hyphenation)."
        ),
    )

    # 2) Advisor Current University (primary appointment)
    university_group = evaluator.add_parallel(
        id="advisor_current_university_primary_appointment",
        desc="Provides the university where the advisor currently holds their primary faculty appointment accurately.",
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_str(advisor.current_university),
        id="advisor_current_university_provided",
        desc="Advisor current (primary) faculty university is provided.",
        parent=university_group,
        critical=True,
    )
    university_leaf = evaluator.add_leaf(
        id="advisor_current_university_supported",
        desc="Advisor’s current (primary) faculty university is supported by cited sources.",
        parent=university_group,
        critical=True,
    )
    uni_claim = (
        f"The advisor currently holds a primary faculty appointment at {advisor.current_university}."
        if _non_empty_str(advisor.current_university) else
        "The advisor’s current faculty university is correctly identified."
    )
    await evaluator.verify(
        claim=uni_claim,
        node=university_leaf,
        sources=_merge_sources(common_sources, advisor.university_source_urls),
        additional_instruction=(
            "Use official university sources (faculty profile, department listing) to verify the current appointment. "
            "If multiple affiliations exist, verify the primary/tenure-track appointment."
        ),
    )

    # 3) Advisor Current Title/Position
    title_group = evaluator.add_parallel(
        id="advisor_current_title_faculty_position",
        desc="Provides the advisor’s current academic title/position accurately.",
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty_str(advisor.current_title),
        id="advisor_current_title_provided",
        desc="Advisor current academic title/position is provided.",
        parent=title_group,
        critical=True,
    )
    title_leaf = evaluator.add_leaf(
        id="advisor_current_title_supported",
        desc="Advisor’s current academic title/position is supported by cited sources.",
        parent=title_group,
        critical=True,
    )
    title_claim = (
        f"The advisor’s current academic title/position is '{advisor.current_title}'"
        + (f" at {advisor.current_university}." if _non_empty_str(advisor.current_university) else ".")
        if _non_empty_str(advisor.current_title) else
        "The advisor’s current academic title/position is correctly identified."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=_merge_sources(common_sources, advisor.title_source_urls, advisor.university_source_urls),
        additional_instruction=(
            "Verify title/position via official university pages (faculty profile, endowed chair listing). "
            "Titles may include variations like Professor, Associate Professor, Assistant Professor, or named chairs."
        ),
    )

    # 4) Advisor Primary Research Areas
    research_group = evaluator.add_parallel(
        id="advisor_primary_research_areas",
        desc="Provides the advisor’s primary research area(s) accurately.",
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(advisor.primary_research_areas),
        id="advisor_research_areas_provided",
        desc="Advisor primary research area(s) are provided.",
        parent=research_group,
        critical=True,
    )
    research_leaf = evaluator.add_leaf(
        id="advisor_research_areas_supported",
        desc="Advisor’s primary research area(s) are supported by cited sources.",
        parent=research_group,
        critical=True,
    )
    areas_text = ", ".join(advisor.primary_research_areas) if advisor.primary_research_areas else ""
    research_claim = (
        f"The advisor’s primary research areas include: {areas_text}."
        if areas_text else
        "The advisor’s primary research areas are correctly identified."
    )
    await evaluator.verify(
        claim=research_claim,
        node=research_leaf,
        sources=_merge_sources(common_sources, advisor.research_source_urls),
        additional_instruction=(
            "Verify via official profiles and publication summaries. Allow reasonable synonyms and closely related "
            "terminology (e.g., 'NLP' vs 'natural language processing')."
        ),
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
    Entry point: Build the verification tree and run extraction and verification according to the rubric.
    """
    # Initialize evaluator with SEQUENTIAL root to respect task ordering.
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_award_and_advisor(),
        template_class=AwardAdvisorExtraction,
        extraction_name="award_advisor_extraction",
    )

    # Build and run verifications in the specified order
    await build_award_verification(evaluator, root, extraction)
    await build_advisor_relationship_verification(evaluator, root, extraction)
    await build_advisor_details_verification(evaluator, root, extraction)

    # Return the standard summary
    return evaluator.get_summary()