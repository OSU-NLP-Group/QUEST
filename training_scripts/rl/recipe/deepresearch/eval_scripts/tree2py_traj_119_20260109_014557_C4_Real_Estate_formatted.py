import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "leed_energy_star_ca_office_one_building"
TASK_DESCRIPTION = (
    "Identify one commercial office building located in California that meets all of the following requirements: "
    "(1) has achieved LEED Gold certification (60-79 points) under the LEED BD+C (Building Design and Construction) rating system, "
    "(2) received this LEED Gold certification between 2020 and 2025, (3) has an ENERGY STAR score of 75 or higher, "
    "(4) is used as a commercial office building, and (5) has at least 1,000 square feet of gross floor area. "
    "Provide the building's name, complete address, and reference URL from official LEED or ENERGY STAR databases."
)

TIMEFRAME_START = 2020
TIMEFRAME_END = 2025

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class BuildingCandidate(BaseModel):
    building_name: Optional[str] = None
    full_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None

    # Claimed facts (as stated by the answer, if present)
    leed_cert_level: Optional[str] = None
    leed_rating_system: Optional[str] = None  # e.g., "LEED BD+C", "LEED BD+C: New Construction v4"
    leed_points: Optional[str] = None         # keep string to be lenient (e.g., "65", "~65")
    leed_certification_date: Optional[str] = None  # e.g., "2021-05-12" or "May 2021"

    energy_star_score: Optional[str] = None   # keep string to be lenient

    # URLs
    leed_urls: List[str] = Field(default_factory=list)          # Official USGBC / LEED project directory URLs
    energy_star_urls: List[str] = Field(default_factory=list)   # Official energystar.gov URLs
    reference_urls: List[str] = Field(default_factory=list)     # Union of all URLs referenced for this building


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_building_candidate() -> str:
    return """
    Extract exactly one building (the first complete one described) that the answer claims satisfies the task.
    Return a JSON object with these fields (use null for any missing field):

    - building_name: The building's name as stated in the answer.
    - full_address: The complete street address as a single line (include street number, street name, city, state, and postal code if available).
    - city: City name if given.
    - state: State (use two-letter code like CA if provided; otherwise the full state name).
    - postal_code: ZIP/postal code if provided.

    - leed_cert_level: The LEED certification level (e.g., "LEED Gold") if mentioned.
    - leed_rating_system: The LEED rating system (e.g., "LEED BD+C", "LEED BD+C: New Construction v4") if mentioned.
    - leed_points: The number of LEED points (as text) if mentioned.
    - leed_certification_date: The certification date or year (as text) if mentioned.

    - energy_star_score: ENERGY STAR score (as text) if mentioned.

    - leed_urls: Array of official LEED/USGBC project directory URLs explicitly present in the answer (e.g., domains including usgbc.org or leed.usgbc.org).
    - energy_star_urls: Array of official ENERGY STAR building/profile URLs explicitly present in the answer (domain energystar.gov).
    - reference_urls: Array of all official database URLs (the union of leed_urls and energy_star_urls). If both are present, include both.

    Rules:
    - Do not invent any data. Extract only what appears explicitly in the answer.
    - For URLs, only include valid URLs that are explicitly present in the answer text (including markdown links).
    - If multiple buildings are listed, only extract the first complete one.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_urls(lst: Optional[List[str]]) -> List[str]:
    if not lst:
        return []
    return [u for u in lst if isinstance(u, str) and u.strip()]


def _prefer(primary: List[str], fallback: List[str]) -> List[str]:
    return primary if primary else fallback


async def verify_building(
    evaluator: Evaluator,
    parent_node,
    b: BuildingCandidate,
) -> None:
    """
    Build the verification subtree for the single identified building.
    Follows the provided rubric tree and ensures each leaf is a binary check.
    """

    # Create the main aggregation node (critical, parallel)
    identified_node = evaluator.add_parallel(
        id="Identified_Building",
        desc="Identify one commercial office building in California that meets all specified sustainability certification and property requirements, and provide its name, complete address, and reference URL",
        parent=parent_node,
        critical=True
    )

    # Prepare URL lists
    all_refs = _safe_urls(b.reference_urls)
    leed_refs = _prefer(_safe_urls(b.leed_urls), all_refs)
    es_refs = _prefer(_safe_urls(b.energy_star_urls), all_refs)

    # 1) Building_Name_Provided (critical existence check)
    evaluator.add_custom_node(
        result=bool(b.building_name and b.building_name.strip()),
        id="Building_Name_Provided",
        desc="The building's name is provided in the answer",
        parent=identified_node,
        critical=True
    )

    # 2) Complete_Address_Provided (critical existence check)
    evaluator.add_custom_node(
        result=bool(b.full_address and b.full_address.strip()),
        id="Complete_Address_Provided",
        desc="The building's complete address is provided in the answer",
        parent=identified_node,
        critical=True
    )

    # 3) Reference_URL_Provided (critical; verify at least one official db URL is provided)
    # We check officialness by verifying each provided url is an official LEED or ENERGY STAR database page.
    ref_url_node = evaluator.add_leaf(
        id="Reference_URL_Provided",
        desc="A reference URL from official LEED or ENERGY STAR databases is provided in the answer",
        parent=identified_node,
        critical=True
    )
    ref_claim = (
        "This webpage is an official LEED (usgbc.org or leed.usgbc.org) or ENERGY STAR (energystar.gov) database page for a building."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_url_node,
        sources=all_refs,  # Multi-URL verification; passes if any provided URL is official
        additional_instruction=(
            "Judge officialness by the domain. Accept only usgbc.org (including usgbc.org/projects/..., leed.usgbc.org) or energystar.gov. "
            "If no URL is provided, or if all provided URLs are non-official domains, this should be considered NOT supported."
        )
    )

    # 4) Location_California
    loc_node = evaluator.add_leaf(
        id="Location_California",
        desc="The building is located in California, United States",
        parent=identified_node,
        critical=True
    )
    await evaluator.verify(
        claim="The building is located in California, United States.",
        node=loc_node,
        sources=all_refs,
        additional_instruction=(
            "Look on the referenced official page for location details (state, city, address). "
            "Accept abbreviations like 'CA' for California. If the page shows a different state, mark as NOT supported."
        )
    )

    # 5) LEED_Gold_Certification
    leed_gold_node = evaluator.add_leaf(
        id="LEED_Gold_Certification",
        desc="The building has achieved LEED Gold certification (60-79 points) under the LEED BD+C or equivalent rating system",
        parent=identified_node,
        critical=True
    )
    await evaluator.verify(
        claim="This building has achieved LEED Gold certification under the LEED BD+C (Building Design and Construction) rating system.",
        node=leed_gold_node,
        sources=leed_refs,
        additional_instruction=(
            "Use the official USGBC/LEED project directory page(s). Accept sub-variants like 'LEED BD+C: New Construction' or 'LEED v4/v4.1 BD+C'. "
            "Do NOT accept other rating systems like O+M (Operations and Maintenance) or ID+C unless the page clearly indicates BD+C. "
            "If the level shown is not Gold, consider NOT supported. If points are shown (60–79) that corroborate Gold, that supports the claim."
        )
    )

    # 6) Certification_Timeframe (2020–2025 inclusive)
    timeframe_node = evaluator.add_leaf(
        id="Certification_Timeframe",
        desc="The building received its LEED Gold certification between 2020 and 2025",
        parent=identified_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LEED Gold certification was received between {TIMEFRAME_START} and {TIMEFRAME_END}, inclusive.",
        node=timeframe_node,
        sources=leed_refs,
        additional_instruction=(
            "Check the 'Certification Date' or equivalent on the official LEED page. "
            f"The year must be {TIMEFRAME_START}, {TIMEFRAME_START+1}, {TIMEFRAME_START+2}, {TIMEFRAME_START+3}, {TIMEFRAME_START+4}, or {TIMEFRAME_END}. "
            "Registration or design completion dates are not sufficient; it must be the certification date."
        )
    )

    # 7) ENERGY_STAR_Score >= 75
    es_score_node = evaluator.add_leaf(
        id="ENERGY_STAR_Score",
        desc="The building has an ENERGY STAR score of 75 or higher",
        parent=identified_node,
        critical=True
    )
    await evaluator.verify(
        claim="The building has an ENERGY STAR score of 75 or higher.",
        node=es_score_node,
        sources=es_refs,
        additional_instruction=(
            "Use the official ENERGY STAR building/profile page. Look for a 1–100 score. "
            "A score of 75 exactly should be considered meeting the threshold. "
            "If the page shows no score or a score below 75, mark as NOT supported."
        )
    )

    # 8) Building_Type is commercial office
    btype_node = evaluator.add_leaf(
        id="Building_Type",
        desc="The building is a commercial office building",
        parent=identified_node,
        critical=True
    )
    await evaluator.verify(
        claim="The building is used as a commercial office building.",
        node=btype_node,
        sources=all_refs,
        additional_instruction=(
            "Look for property type or building type on the official page. "
            "Accept 'Office' or equivalent wording for a commercial office building. "
            "Do not accept residential, mixed-use dominated by non-office, or unrelated property types."
        )
    )

    # 9) Minimum_Size >= 1,000 sqft
    min_size_node = evaluator.add_leaf(
        id="Minimum_Size",
        desc="The building has at least 1,000 square feet of gross floor area",
        parent=identified_node,
        critical=True
    )
    await evaluator.verify(
        claim="The building's gross floor area is at least 1,000 square feet (>= 92.9 square meters).",
        node=min_size_node,
        sources=all_refs,
        additional_instruction=(
            "Check 'Gross floor area', 'Total floor area', or similar. "
            "If the size is only provided in square meters, convert using 1 m^2 ≈ 10.7639 ft^2. "
            "If the area is missing or clearly below the threshold, mark as NOT supported."
        )
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the single-building LEED/ENERGY STAR California office task.
    """
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

    # Extract the building candidate from the answer
    building = await evaluator.extract(
        prompt=prompt_extract_building_candidate(),
        template_class=BuildingCandidate,
        extraction_name="building_candidate"
    )

    # Record timeframe as custom info for transparency
    evaluator.add_custom_info(
        {"required_leed_timeframe": f"{TIMEFRAME_START}-{TIMEFRAME_END}", "min_energy_star_score": 75, "min_gross_area_sqft": 1000},
        info_type="constraints",
        info_name="task_constraints"
    )

    # Build verification nodes per rubric and verify
    await verify_building(evaluator, root, building)

    # Return the standardized summary
    return evaluator.get_summary()