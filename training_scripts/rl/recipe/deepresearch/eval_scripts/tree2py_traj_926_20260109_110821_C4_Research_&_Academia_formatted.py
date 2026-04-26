import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs_conf_2025_top_tier"
TASK_DESCRIPTION = """Identify one top-tier computer science conference in 2025 that meets all of the following criteria:
1. The conference must be held in the United States.
2. The conference dates must fall entirely within the period from April 1, 2025 to August 31, 2025.
3. The conference must be ranked as A* in the CORE rankings OR have a Research.com impact score above 40.
4. The conference's acceptance rate for 2025 must be below 25%.
5. The conference must be held at a major convention center, and you must provide the complete street address of the venue.
6. The conference must be a premier computer science conference focused on Artificial Intelligence, Machine Learning, or Computer Vision.

For the identified conference, provide:
- The full conference name
- The exact dates (start and end date)
- The city and state where it is held
- The complete street address of the venue
- The official conference ranking (CORE ranking and/or Research.com impact score)
- The 2025 acceptance rate
- A reference URL to the official conference website
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RankingInfo(BaseModel):
    core_ranking: Optional[str] = None
    core_source_urls: List[str] = Field(default_factory=list)
    research_com_impact_score: Optional[str] = None
    research_com_source_urls: List[str] = Field(default_factory=list)


class VenueInfo(BaseModel):
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    venue_url: Optional[str] = None


class ConferenceInfo(BaseModel):
    conference_name: Optional[str] = None
    official_url: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    field_focus: Optional[str] = None  # e.g., "Artificial Intelligence", "Machine Learning", "Computer Vision"
    acceptance_rate_2025: Optional[str] = None
    acceptance_rate_source_urls: List[str] = Field(default_factory=list)
    ranking: RankingInfo = Field(default_factory=RankingInfo)
    venue: VenueInfo = Field(default_factory=VenueInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conference() -> str:
    return """
    Extract details for exactly ONE conference mentioned in the answer that is intended to satisfy all constraints.
    If multiple conferences are mentioned, select the first one that appears to meet the constraints and extract its details.

    Return a JSON object with the following fields:
    - conference_name: Full official name of the conference (string).
    - official_url: The official conference website URL (string). Must be an actual URL present in the answer.
    - city: The city where the conference is held (string).
    - state: The U.S. state where the conference is held (string). Use standard abbreviations if the answer uses them.
    - country: The country where the conference is held (string). If not explicitly stated, infer from context; otherwise null.
    - start_date: The exact start date of the 2025 conference (string as provided).
    - end_date: The exact end date of the 2025 conference (string as provided).
    - field_focus: A short phrase indicating the field focus, e.g., "Artificial Intelligence", "Machine Learning", or "Computer Vision".
    - acceptance_rate_2025: The acceptance rate for the 2025 edition, as a percentage or ratio string (e.g., "22%", "0.22"). If not provided, set null.
    - acceptance_rate_source_urls: An array of URLs that explicitly support the acceptance rate information. If none are provided in the answer, return an empty array.

    - ranking: Object containing:
        - core_ranking: The CORE ranking for the conference (e.g., "A*", "A"). If not provided, set null.
        - core_source_urls: An array of URLs that show the CORE ranking for the conference (e.g., CORE ranking list page or conference entry). If none are provided, return an empty array.
        - research_com_impact_score: The Research.com impact score (string). If not provided, set null.
        - research_com_source_urls: An array of URLs from Research.com that show the impact score for the conference. If none are provided, return an empty array.

    - venue: Object containing:
        - venue_name: Name of the venue (e.g., "Boston Convention and Exhibition Center").
        - venue_address: Complete street address, including street number/name, city, state, and ZIP code if applicable (string).
        - venue_url: An official venue page URL (e.g., venue's own site or a page on the conference site). If not provided, set null.

    IMPORTANT:
    - Only extract URLs explicitly present in the answer (plain URLs or markdown links). Do not invent URLs.
    - If any field is missing in the answer, set it to null (or empty array for URLs).
    - Preserve the exact formatting of dates and names as presented.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _non_empty_urls(*url_or_lists: Optional[Any]) -> List[str]:
    urls: List[str] = []
    for item in url_or_lists:
        if item is None:
            continue
        if isinstance(item, list):
            for u in item:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
        elif isinstance(item, str) and item.strip():
            urls.append(item.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    if not url:
        return False
    # Basic URL pattern check
    return bool(re.match(r"^(https?://)[^\s]+$", url))


def is_complete_us_address(addr: Optional[str]) -> bool:
    """
    Heuristic check for a complete US street address:
    - Contains a street number (at least one digit)
    - Contains a comma-separated city/state portion
    - Contains a US ZIP code pattern (12345 or 12345-6789)
    """
    if not addr or not isinstance(addr, str):
        return False
    s = addr.strip()
    if len(s) < 10:
        return False
    has_number = bool(re.search(r"\d+", s))
    has_city_state = ("," in s) and bool(re.search(r",[ ]*[A-Za-z].*[A-Za-z]", s))
    has_zip = bool(re.search(r"\b\d{5}(-\d{4})?\b", s))
    return has_number and has_city_state and has_zip


# --------------------------------------------------------------------------- #
# Verification tree construction & checks                                     #
# --------------------------------------------------------------------------- #
async def build_conference_verification(
    evaluator: Evaluator,
    parent_node,
    conf: ConferenceInfo,
) -> None:
    """
    Build the verification tree under the provided parent node and run verifications.
    All nodes here are CRITICAL to match the rubric's strict requirements.
    """

    # Top-level node (critical, parallel aggregation)
    main_node = evaluator.add_parallel(
        id="Conference_Response",
        desc="Identify one 2025 top-tier CS conference that satisfies all stated constraints and provide all requested details",
        parent=parent_node,
        critical=True,
    )

    # 1. Conference_Name_Provided (existence check)
    evaluator.add_custom_node(
        result=bool(conf.conference_name and conf.conference_name.strip()),
        id="Conference_Name_Provided",
        desc="Provide the full conference name",
        parent=main_node,
        critical=True,
    )

    # 2. City_And_State_Provided (existence check)
    evaluator.add_custom_node(
        result=bool(conf.city and conf.city.strip() and conf.state and conf.state.strip()),
        id="City_And_State_Provided",
        desc="Provide the city and state where the conference is held",
        parent=main_node,
        critical=True,
    )

    # 3. Official_Conference_Website_URL_Provided (existence + validity)
    evaluator.add_custom_node(
        result=is_valid_url(conf.official_url),
        id="Official_Conference_Website_URL_Provided",
        desc="Provide a reference URL to the official conference website",
        parent=main_node,
        critical=True,
    )

    # 4. Held_In_United_States (verification by official site)
    held_us_node = evaluator.add_leaf(
        id="Held_In_United_States",
        desc="Verify the conference is held in the United States",
        parent=main_node,
        critical=True,
    )
    claim_us = f"The conference is held in the United States; the location is {conf.city}, {conf.state}, USA."
    await evaluator.verify(
        claim=claim_us,
        node=held_us_node,
        sources=conf.official_url,
        additional_instruction="Use the official website to confirm the location. If the page shows a U.S. city/state and/or 'USA', treat it as being held in the United States.",
    )

    # 5. Start_And_End_Dates_Provided (existence check)
    evaluator.add_custom_node(
        result=bool(conf.start_date and conf.start_date.strip() and conf.end_date and conf.end_date.strip()),
        id="Start_And_End_Dates_Provided",
        desc="Provide the exact start date and end date of the conference",
        parent=main_node,
        critical=True,
    )

    # 6. Dates_Fall_Entirely_Within_Range (verification using official site and range rule)
    dates_range_node = evaluator.add_leaf(
        id="Dates_Fall_Entirely_Within_Range",
        desc="Verify the conference dates fall entirely within Apr 1, 2025 to Aug 31, 2025",
        parent=main_node,
        critical=True,
    )
    claim_dates = (
        f"The conference runs from {conf.start_date} to {conf.end_date}, "
        "and both dates fall between April 1, 2025 and August 31, 2025 inclusive."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=dates_range_node,
        sources=conf.official_url,
        additional_instruction="Confirm the exact 2025 dates on the official website, then check that both start and end dates lie within 2025-04-01 to 2025-08-31 inclusive.",
    )

    # 7. Ranking_Info_Provided (existence of either CORE ranking or Research.com impact score)
    ranking_info_provided = bool(
        (conf.ranking.core_ranking and conf.ranking.core_ranking.strip())
        or (conf.ranking.research_com_impact_score and conf.ranking.research_com_impact_score.strip())
    )
    evaluator.add_custom_node(
        result=ranking_info_provided,
        id="Ranking_Info_Provided",
        desc="Provide the official conference ranking information (CORE ranking and/or Research.com impact score)",
        parent=main_node,
        critical=True,
    )

    # 8. Ranking_Qualifies (CORE A* OR Research.com impact score > 40), verify by URLs
    ranking_urls = _non_empty_urls(
        conf.ranking.core_source_urls,
        conf.ranking.research_com_source_urls,
        conf.official_url  # fallback if the answer claims ranking on the official site
    )
    ranking_qual_node = evaluator.add_leaf(
        id="Ranking_Qualifies",
        desc="Verify the conference is CORE A* OR has a Research.com impact score above 40",
        parent=main_node,
        critical=True,
    )
    claim_ranking = (
        "This conference satisfies at least one of the following: "
        "it is ranked A* in the CORE rankings OR it has a Research.com impact score above 40."
    )
    await evaluator.verify(
        claim=claim_ranking,
        node=ranking_qual_node,
        sources=ranking_urls if ranking_urls else None,
        additional_instruction=(
            "Check provided ranking sources (CORE or Research.com). "
            "If the CORE page shows A*, that suffices. "
            "Alternatively, if Research.com shows an impact score > 40, that suffices. "
            "Treat 'A*' as equivalent to 'A-star'."
        ),
    )

    # 9. Acceptance_Rate_2025_Provided (existence check)
    evaluator.add_custom_node(
        result=bool(conf.acceptance_rate_2025 and conf.acceptance_rate_2025.strip()),
        id="Acceptance_Rate_2025_Provided",
        desc="Provide the 2025 acceptance rate",
        parent=main_node,
        critical=True,
    )

    # 10. Acceptance_Rate_Below_25 (verify by URLs)
    accept_urls = _non_empty_urls(conf.acceptance_rate_source_urls, conf.official_url)
    acceptance_node = evaluator.add_leaf(
        id="Acceptance_Rate_Below_25",
        desc="Verify the 2025 acceptance rate is below 25%",
        parent=main_node,
        critical=True,
    )
    claim_accept = (
        f"The 2025 acceptance rate for this conference is below 25% (reported as '{conf.acceptance_rate_2025}')."
    )
    await evaluator.verify(
        claim=claim_accept,
        node=acceptance_node,
        sources=accept_urls if accept_urls else None,
        additional_instruction=(
            "Use the provided sources to confirm the acceptance rate for the 2025 edition. "
            "If a percentage is given, ensure it is strictly less than 25%. "
            "If a fraction or ratio is given, convert appropriately."
        ),
    )

    # 11. Venue_Requirements (critical, parallel)
    venue_node = evaluator.add_parallel(
        id="Venue_Requirements",
        desc="Provide venue information meeting the stated venue constraints",
        parent=main_node,
        critical=True,
    )

    # 11.1 Venue_Complete_Street_Address_Provided (existence with heuristics)
    evaluator.add_custom_node(
        result=is_complete_us_address(conf.venue.venue_address),
        id="Venue_Complete_Street_Address_Provided",
        desc="Provide the complete street address of the venue (street address, city, state, and ZIP/postal code if applicable)",
        parent=venue_node,
        critical=True,
    )

    # 11.2 Venue_Is_Major_Convention_Center (verify by URLs)
    venue_urls = _non_empty_urls(conf.venue.venue_url, conf.official_url)
    venue_major_node = evaluator.add_leaf(
        id="Venue_Is_Major_Convention_Center",
        desc="Verify the venue is a major convention center (with appropriate evidence/citation)",
        parent=venue_node,
        critical=True,
    )
    claim_venue = f"The venue '{conf.venue.venue_name}' is a major convention center."
    await evaluator.verify(
        claim=claim_venue,
        node=venue_major_node,
        sources=venue_urls if venue_urls else None,
        additional_instruction=(
            "Confirm that the venue is a recognized convention center (e.g., named 'Convention Center', "
            "or described by the official venue site as a large event/conference facility)."
        ),
    )

    # 12. Field_Focus_AI_ML_CV (verify by official site)
    field_focus_node = evaluator.add_leaf(
        id="Field_Focus_AI_ML_CV",
        desc="Verify the conference focuses on Artificial Intelligence, Machine Learning, or Computer Vision",
        parent=main_node,
        critical=True,
    )
    claim_focus = (
        "This conference is a premier computer science conference focused on Artificial Intelligence, "
        "Machine Learning, or Computer Vision."
    )
    await evaluator.verify(
        claim=claim_focus,
        node=field_focus_node,
        sources=conf.official_url,
        additional_instruction=(
            "Check the conference 'About', 'Call for Papers', 'Scope', or 'Tracks' pages to confirm "
            "a primary focus on AI, ML, or CV."
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
    Evaluate the answer for the 2025 top-tier CS conference identification task.
    """
    # Initialize evaluator (root is non-critical by design; we will add a critical top-level node)
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

    # Extract structured conference info from the answer
    conf_info = await evaluator.extract(
        prompt=prompt_extract_conference(),
        template_class=ConferenceInfo,
        extraction_name="conference_info",
    )

    # Build verification tree and run checks
    await build_conference_verification(evaluator, root, conf_info)

    # Return structured evaluation summary
    return evaluator.get_summary()