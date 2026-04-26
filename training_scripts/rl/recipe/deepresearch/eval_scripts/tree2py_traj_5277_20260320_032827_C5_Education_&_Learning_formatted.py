import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "passhe_closest_philly"
TASK_DESCRIPTION = """
Among Pennsylvania's State System of Higher Education (PASSHE) universities, identify the university that is closest to Philadelphia. Provide the following information about this university: its approximate driving distance from Philadelphia in miles, its campus size in acres, its founding year, any special historical designations it holds, and its complete mailing address.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityInfoExtraction(BaseModel):
    # Core identification
    university_name: Optional[str] = None

    # Required information values (keep as free-form strings to be robust)
    distance_miles_text: Optional[str] = None
    campus_size_acres_text: Optional[str] = None
    founding_year_text: Optional[str] = None
    historical_designation_text: Optional[str] = None
    mailing_address_text: Optional[str] = None

    # Source URLs grouped by claim/topic
    passhe_membership_source_urls: List[str] = Field(default_factory=list)
    closest_comparison_source_urls: List[str] = Field(default_factory=list)
    official_name_source_urls: List[str] = Field(default_factory=list)

    distance_source_urls: List[str] = Field(default_factory=list)
    campus_size_source_urls: List[str] = Field(default_factory=list)
    founding_year_source_urls: List[str] = Field(default_factory=list)
    historical_designation_source_urls: List[str] = Field(default_factory=list)
    address_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_info() -> str:
    return """
    Extract the single PASSHE university that the answer identifies as closest to Philadelphia, along with all requested attributes and the exact URLs cited in the answer that support each item.

    Return a JSON object with these fields:
    - university_name: string or null
    - distance_miles_text: string or null              (e.g., "25", "about 27", "≈30")
    - campus_size_acres_text: string or null           (e.g., "406", "about 400")
    - founding_year_text: string or null               (e.g., "1869")
    - historical_designation_text: string or null      (e.g., "First HBCU in the U.S.", "National Historic Landmark")
    - mailing_address_text: string or null             (complete street, city, state, ZIP as presented)

    Also extract the URLs the answer cites to support each item. Only include URLs explicitly present in the answer. Accept plain URLs or markdown links:
    - passhe_membership_source_urls: array of URLs confirming PASSHE membership
    - closest_comparison_source_urls: array of URLs supporting the claim that this is the closest PASSHE university to Philadelphia (e.g., comparison pages, Google Maps comparisons, articles listing distances)
    - official_name_source_urls: array of URLs that show the official name (e.g., university homepage, PASSHE list)
    - distance_source_urls: array of URLs supporting the driving distance (e.g., Google Maps route link)
    - campus_size_source_urls: array of URLs verifying campus size
    - founding_year_source_urls: array of URLs verifying founding year
    - historical_designation_source_urls: array of URLs verifying the historical designation
    - address_source_urls: array of URLs verifying the complete mailing address

    Important:
    - Only extract information that is explicitly stated in the provided answer.
    - If an item is missing in the answer, set the corresponding value to null (and the corresponding URLs array to an empty array).
    - For URL fields, extract only valid URLs that appear in the answer (do not invent).
    - Keep numbers as strings exactly as written in the answer (including qualifiers like "about", "~", "approx.").
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: Optional[List[str]]) -> List[str]:
    """Combine multiple URL lists, preserving order and removing duplicates and falsy entries."""
    combined: List[str] = []
    seen = set()
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url or not isinstance(url, str):
                continue
            if url not in seen:
                combined.append(url)
                seen.add(url)
    return combined


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_university_identification(
    evaluator: Evaluator,
    parent_node,
    info: UniversityInfoExtraction
) -> None:
    """
    Build and verify the 'university_identification' subtree.

    Note: We mark this node and all its leaves as critical to reflect the rubric intent
    and to ensure sequential gating for subsequent sections.
    """
    node = evaluator.add_parallel(
        id="university_identification",
        desc="Correctly identify which PASSHE university is closest to Philadelphia",
        parent=parent_node,
        critical=True
    )

    name = info.university_name or ""

    # 1) PASSHE membership verification (critical)
    membership_leaf = evaluator.add_leaf(
        id="passhe_membership",
        desc="Verify the university is a member of Pennsylvania's State System of Higher Education",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{name}' is a member of Pennsylvania's State System of Higher Education (PASSHE).",
        node=membership_leaf,
        sources=info.passhe_membership_source_urls,
        additional_instruction="Use authoritative sources such as PASSHE's official site or other credible references that list PASSHE member universities."
    )

    # 2) 'Closest to Philadelphia' verification (critical)
    closest_leaf = evaluator.add_leaf(
        id="closest_to_philadelphia",
        desc="Verify it is the PASSHE university with the shortest driving distance to Philadelphia",
        parent=node,
        critical=True
    )
    closest_sources = _combine_sources(
        info.closest_comparison_source_urls,
        info.distance_source_urls,
        info.passhe_membership_source_urls
    )
    await evaluator.verify(
        claim=f"Among PASSHE universities, '{name}' has the shortest driving distance to Philadelphia.",
        node=closest_leaf,
        sources=closest_sources,
        additional_instruction="Interpret 'closest' as the shortest typical driving distance from central Philadelphia (e.g., City Hall) to the university's main campus. Accept minor ties or small variations if the sources clearly support that this university is the closest or among the closest with the shortest distance."
    )

    # 3) Correct official name verification (critical)
    name_leaf = evaluator.add_leaf(
        id="correct_name",
        desc="Provide the correct official name",
        parent=node,
        critical=True
    )
    name_sources = _combine_sources(
        info.official_name_source_urls,
        info.passhe_membership_source_urls,
        info.address_source_urls
    )
    await evaluator.verify(
        claim=f"The official name of the institution is '{name}'.",
        node=name_leaf,
        sources=name_sources,
        additional_instruction="Allow minor formatting or capitalization variations; verify against the university's official website or the official PASSHE member listing."
    )

    # 4) Presence of a PASSHE membership URL (critical per source-grounding policy)
    evaluator.add_custom_node(
        result=bool(info.passhe_membership_source_urls),
        id="passhe_source_url",
        desc="Provide URL confirming PASSHE membership",
        parent=node,
        critical=True
    )


async def build_required_information(
    evaluator: Evaluator,
    parent_node,
    info: UniversityInfoExtraction
) -> None:
    """
    Build and verify the 'required_information' subtree and each of its sub-items.
    Marked as critical with critical children to reflect rubric requirements.
    """
    req_node = evaluator.add_parallel(
        id="required_information",
        desc="Provide all required information about the identified university",
        parent=parent_node,
        critical=True
    )

    uni_name = info.university_name or "the university"

    # -------------------- Distance Information -------------------- #
    dist_node = evaluator.add_parallel(
        id="distance_information",
        desc="Provide the driving distance from Philadelphia",
        parent=req_node,
        critical=True
    )

    # Presence of a distance URL (critical)
    evaluator.add_custom_node(
        result=bool(info.distance_source_urls),
        id="distance_source_url",
        desc="Provide URL supporting the distance",
        parent=dist_node,
        critical=True
    )

    # Value verification (critical)
    dist_value_leaf = evaluator.add_leaf(
        id="distance_value",
        desc="State the approximate driving distance in miles",
        parent=dist_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The driving distance from Philadelphia to {uni_name} is approximately {info.distance_miles_text or ''} miles.",
        node=dist_value_leaf,
        sources=info.distance_source_urls,
        additional_instruction="Verify that at least one provided source (e.g., Google Maps route) shows a driving distance within a reasonable tolerance (≈±15%) of the stated miles. Prefer the shortest typical route to the main campus."
    )

    # -------------------- Campus Size Information -------------------- #
    campus_node = evaluator.add_parallel(
        id="campus_size_information",
        desc="Provide the campus size",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.campus_size_source_urls),
        id="campus_size_source_url",
        desc="Provide URL verifying campus size",
        parent=campus_node,
        critical=True
    )

    campus_value_leaf = evaluator.add_leaf(
        id="campus_size_value",
        desc="State campus size in acres",
        parent=campus_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The campus size of {uni_name} is approximately {info.campus_size_acres_text or ''} acres.",
        node=campus_value_leaf,
        sources=info.campus_size_source_urls,
        additional_instruction="Allow approximate language (e.g., 'about', '~'). If multiple campuses or sites exist, the source should clearly pertain to the university's main campus size stated in the answer."
    )

    # -------------------- Founding Year Information -------------------- #
    founding_node = evaluator.add_parallel(
        id="founding_year_information",
        desc="Provide the founding year",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.founding_year_source_urls),
        id="founding_year_source_url",
        desc="Provide URL verifying founding year",
        parent=founding_node,
        critical=True
    )

    founding_value_leaf = evaluator.add_leaf(
        id="founding_year_value",
        desc="State the founding year",
        parent=founding_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} was founded in {info.founding_year_text or ''}.",
        node=founding_value_leaf,
        sources=info.founding_year_source_urls,
        additional_instruction="Confirm the founding year (a four-digit year) as stated on official or authoritative sources (university 'About' page, PASSHE, reputable references)."
    )

    # -------------------- Historical Designation Information -------------------- #
    hist_node = evaluator.add_parallel(
        id="historical_designation_information",
        desc="Provide special historical designation",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.historical_designation_source_urls),
        id="historical_designation_source_url",
        desc="Provide URL verifying the historical designation",
        parent=hist_node,
        critical=True
    )

    hist_value_leaf = evaluator.add_leaf(
        id="historical_designation_description",
        desc="Describe any special historical designation",
        parent=hist_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{uni_name} holds the following special historical designation: {info.historical_designation_text or ''}",
        node=hist_value_leaf,
        sources=info.historical_designation_source_urls,
        additional_instruction="Confirm the designation as described (e.g., first HBCU, National Historic Landmark). Allow close paraphrases, but the substance must match the source."
    )

    # -------------------- Mailing Address Information -------------------- #
    address_node = evaluator.add_parallel(
        id="mailing_address_information",
        desc="Provide the complete mailing address",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.address_source_urls),
        id="address_source_url",
        desc="Provide URL verifying the address",
        parent=address_node,
        critical=True
    )

    address_leaf = evaluator.add_leaf(
        id="full_address",
        desc="Complete address with street, city, state, ZIP",
        parent=address_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The complete official mailing address of {uni_name} is: {info.mailing_address_text or ''}",
        node=address_leaf,
        sources=info.address_source_urls,
        additional_instruction="Permit minor formatting differences (e.g., 'PA' vs 'Pennsylvania', ZIP vs ZIP+4). The source should be an official contact or address page or another authoritative listing."
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
    Evaluate an answer for the PASSHE-closest-to-Philadelphia task using the Mind2Web2 framework.
    """
    # Initialize evaluator with a sequential root to enforce order dependency
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
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_university_info(),
        template_class=UniversityInfoExtraction,
        extraction_name="university_info_extraction"
    )

    # Build verification tree according to rubric
    await build_university_identification(evaluator, root, extracted)
    await build_required_information(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()