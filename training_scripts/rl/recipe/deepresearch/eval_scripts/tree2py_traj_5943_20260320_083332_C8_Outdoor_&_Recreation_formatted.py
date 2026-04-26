import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "breeze_west_parks_touchless_2026"
TASK_DESCRIPTION = """
I'm planning a spring 2026 outdoor adventure trip and want to visit national parks in the western United States that feature dramatic mountain or canyon landscapes. I prefer to fly with Breeze Airways and would like to use airports that have the new TSA PreCheck Touchless ID feature for faster security screening.

Please identify at least 3 distinct national parks that meet all of the following criteria:

1. The park must be listed on Breeze Airways' official national parks webpage (https://www.flybreeze.com/shopping/en-us/national-parks)
2. The park must be accessible from a Breeze Airways destination airport
3. The departure airport must have TSA PreCheck Touchless ID available (as documented in sources from January 2026 or later)
4. The travel time from the Breeze Airways airport to the national park must be 2 hours or less (as stated on Breeze's national parks page)
5. The park must feature mountain or canyon terrain, specifically categorized under "Tall Alpine Mountains" or "Rocky Canyons" on Breeze's national parks page
6. The park must be located in a western United States state (west of the 100th meridian - roughly west of Kansas/Nebraska/Dakotas)

For each park you identify, please provide:
- The national park name
- The specific Breeze Airways airport code (e.g., DEN, LAS, SFO) serving that destination
- The exact travel time from that airport to the park (as stated on Breeze's page)
- The terrain category classification ("Tall Alpine Mountains" or "Rocky Canyons")
- The state where the park is located
- Reference URLs supporting your answer (both the Breeze national parks page and documentation of TSA Touchless ID availability at that airport)
"""

BREEZE_NP_URL = "https://www.flybreeze.com/shopping/en-us/national-parks"
ALLOWED_TERRAIN_CATEGORIES = {"Tall Alpine Mountains", "Rocky Canyons"}
REQUIRED_MIN_TSA_DATE_TEXT = "January 2026"  # For instruction text to judge date freshness


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkItem(BaseModel):
    park_name: Optional[str] = None
    airport_code: Optional[str] = None
    travel_time: Optional[str] = None
    terrain_category: Optional[str] = None
    state: Optional[str] = None
    breeze_url: Optional[str] = None
    tsa_url: Optional[str] = None


class ParksExtraction(BaseModel):
    parks: List[ParkItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return f"""
    From the provided answer, extract a list of national parks (at least three if available) that the answer claims meet the user's criteria.

    For each park, extract the following EXACTLY as written in the answer:
    - park_name: The full name of the national park.
    - airport_code: The specific Breeze Airways destination airport code (3-letter IATA, e.g., LAS). If multiple airports are mentioned, pick the one paired with this park in the answer.
    - travel_time: The exact travel time string from that Breeze airport to the park as stated on Breeze's national parks page (e.g., "1.5 hours", "90 minutes", "2 hours", etc.). Keep the original phrasing used in the answer.
    - terrain_category: The terrain classification string as stated on Breeze's page. It must be either "Tall Alpine Mountains" or "Rocky Canyons" if the answer claims so. If the answer provides a variant or similar text, extract exactly what the answer provided.
    - state: The U.S. state in which the park is located, as stated in the answer.
    - breeze_url: A URL to Breeze Airways’ national parks page (ideally the official page: {BREEZE_NP_URL}) cited by the answer. If the answer gives a different format (e.g., same page but different tracking params), extract it. If NO Breeze national parks URL is provided in the answer, set to null.
    - tsa_url: A URL the answer cites that documents TSA PreCheck Touchless ID (CAT-2 / Digital ID / Touchless ID) availability at the specified airport (the source date must be January 2026 or later per the task requirement). If the answer provides multiple URLs, pick the most specific one for the airport and prefer a page dated Jan 2026 or later. If none provided, set to null.

    RULES:
    - Only extract what is explicitly present in the answer text. Do not infer, invent, or normalize values.
    - Preserve formatting (e.g., "1 hr 45 min" stays that way).
    - For airport_code, if it is missing or not clearly a 3-letter IATA code, set it to null.
    - For terrain_category, if the answer states a value, return it as-is; otherwise set to null.
    - If any field is missing in the answer for a park, set that field to null.
    - Return all parks mentioned in the answer in order of appearance; we will later evaluate only the first three.

    Return a JSON object with a single field:
    - "parks": an array of the extracted park objects.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_iata(code: Optional[str]) -> bool:
    if not code:
        return False
    code = code.strip().upper()
    return len(code) == 3 and code.isalpha()


def _norm_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return " ".join(name.split()).strip().lower()


# --------------------------------------------------------------------------- #
# Verification logic per park                                                 #
# --------------------------------------------------------------------------- #
async def verify_one_park(
    evaluator: Evaluator,
    parent_node,
    park: ParkItem,
    park_index: int,
    seen_names: Set[str],
) -> None:
    """
    Build the verification sub-tree and run checks for a single park.
    Tree mirrors the rubric structure for Park_{i}.
    """
    park_id = f"Park_{park_index}"
    park_title = f"{'First' if park_index == 1 else ('Second' if park_index == 2 else 'Third')} national park meeting all criteria"

    # Create Park node (non-critical aggregator, parallel per rubric)
    park_node = evaluator.add_parallel(
        id=park_id,
        desc=park_title,
        parent=parent_node,
        critical=False
    )

    # --------------------- Identification (Critical, Parallel) --------------------- #
    ident_node = evaluator.add_parallel(
        id=f"{park_id}_Identification",
        desc=f"Valid identification of the {['first','second','third'][park_index-1]} national park",
        parent=park_node,
        critical=True
    )

    # Valid Name (treat as existence/format check to avoid brittle world-knowledge dependency)
    evaluator.add_custom_node(
        result=bool(park.park_name and park.park_name.strip()),
        id=f"{park_id}_Valid_Name",
        desc="The park name provided is a valid national park (name is present and non-empty)",
        parent=ident_node,
        critical=True
    )

    # Listed on Breeze (URL-verified against Breeze national parks page)
    listed_leaf = evaluator.add_leaf(
        id=f"{park_id}_Listed_On_Breeze",
        desc="The park appears on Breeze Airways' official national parks page",
        parent=ident_node,
        critical=True
    )
    breeze_source = park.breeze_url or BREEZE_NP_URL
    listed_claim = f"The Breeze Airways national parks page lists the park named '{park.park_name}'."
    await evaluator.verify(
        claim=listed_claim,
        node=listed_leaf,
        sources=breeze_source,
        additional_instruction=(
            "Search the Breeze national parks page for the given park name (case-insensitive, allow minor variations). "
            "If the page is long, scan all sections and categories."
        )
    )

    # --------------------- Airport Access (Critical, Parallel) --------------------- #
    access_node = evaluator.add_parallel(
        id=f"{park_id}_Airport_Access",
        desc="Verification of airport accessibility and TSA Touchless ID availability",
        parent=park_node,
        critical=True
    )

    # Breeze serves (via NP page recommending a Breeze destination airport)
    serves_leaf = evaluator.add_leaf(
        id=f"{park_id}_Breeze_Serves",
        desc="Breeze Airways serves a destination airport near this park",
        parent=access_node,
        critical=True
    )
    serves_claim = (
        f"On Breeze's national parks page for '{park.park_name}', the recommended Breeze destination airport is "
        f"'{park.airport_code}'."
    )
    await evaluator.verify(
        claim=serves_claim,
        node=serves_leaf,
        sources=breeze_source,
        additional_instruction=(
            "Verify that the page explicitly recommends or names the Breeze destination airport to use for this park, "
            "typically in the form 'Fly into <Airport Name> (<IATA>)' or similar."
        )
    )

    # Airport Code provided (existence/format)
    evaluator.add_custom_node(
        result=_is_valid_iata(park.airport_code),
        id=f"{park_id}_Airport_Code",
        desc="The specific Breeze Airways airport code is provided (valid 3‑letter IATA)",
        parent=access_node,
        critical=True
    )

    # TSA Touchless ID availability as of Jan 2026+
    if park.tsa_url and park.tsa_url.strip():
        tsa_leaf = evaluator.add_leaf(
            id=f"{park_id}_TSA_Touchless_ID",
            desc="The identified airport has TSA PreCheck Touchless ID available as of January 2026",
            parent=access_node,
            critical=True
        )
        tsa_claim = (
            f"As of January 2026 or later, TSA PreCheck Touchless ID (also referred as CAT-2, Digital ID, or "
            f"Touchless ID for PreCheck) is available at airport '{park.airport_code}'."
        )
        await evaluator.verify(
            claim=tsa_claim,
            node=tsa_leaf,
            sources=park.tsa_url,
            additional_instruction=(
                "Confirm the page specifically documents TSA PreCheck Touchless ID / CAT-2 / Digital ID availability "
                f"at the specified airport and that the page clearly shows a publication/update date of {REQUIRED_MIN_TSA_DATE_TEXT} or later. "
                "If no clear date or the date is earlier than 2026-01, treat as NOT supported."
            )
        )
    else:
        # Fail this critical leaf due to missing source
        evaluator.add_custom_node(
            result=False,
            id=f"{park_id}_TSA_Touchless_ID",
            desc="The identified airport has TSA PreCheck Touchless ID available as of January 2026 (missing or invalid TSA source)",
            parent=access_node,
            critical=True
        )

    # --------------------- Travel Time (Critical, Parallel) --------------------- #
    travel_node = evaluator.add_parallel(
        id=f"{park_id}_Travel_Time",
        desc="Travel time from airport to park meets the 2-hour constraint",
        parent=park_node,
        critical=True
    )

    # Within 2 hours (logic check from provided string)
    within_leaf = evaluator.add_leaf(
        id=f"{park_id}_Within_2_Hours",
        desc="The travel time from the Breeze airport to the park is 2 hours or less",
        parent=travel_node,
        critical=True
    )
    within_claim = (
        f"The travel time string '{park.travel_time}' indicates a total travel time of 2 hours or less."
    )
    await evaluator.verify(
        claim=within_claim,
        node=within_leaf,
        additional_instruction=(
            "Interpret common time formats (e.g., '1.5 hours', '90 minutes', '1 hr 45 min', '2 hours'). "
            "Treat '2 hours' as acceptable (<= 2 hours). Values like '2.5 hours' should be considered > 2 hours."
        )
    )

    # Exact time stated on Breeze page
    time_leaf = evaluator.add_leaf(
        id=f"{park_id}_Time_Stated",
        desc="The exact travel time is stated",
        parent=travel_node,
        critical=True
    )
    time_claim = (
        f"On Breeze's national parks page for '{park.park_name}', the travel time from airport '{park.airport_code}' "
        f"to the park is stated as '{park.travel_time}'."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        sources=breeze_source,
        additional_instruction=(
            "Match the quoted travel time string as closely as possible. Allow minor formatting variations "
            "(e.g., '1 hr 30 min' vs '1h 30m' or similar), but the numeric value must be the same."
        )
    )

    # --------------------- Terrain Type (Critical, Parallel) --------------------- #
    terrain_node = evaluator.add_parallel(
        id=f"{park_id}_Terrain_Type",
        desc="Verification of terrain classification",
        parent=park_node,
        critical=True
    )

    # Category stated (existence)
    evaluator.add_custom_node(
        result=bool(park.terrain_category and park.terrain_category.strip()),
        id=f"{park_id}_Category_Stated",
        desc="The terrain category classification is stated",
        parent=terrain_node,
        critical=True
    )

    # Mountain or Canyon (URL-verified against Breeze page with allowed categories)
    terrain_leaf = evaluator.add_leaf(
        id=f"{park_id}_Mountain_Or_Canyon",
        desc="The park features mountain or canyon terrain (classified under 'Tall Alpine Mountains' or 'Rocky Canyons' on Breeze's page)",
        parent=terrain_node,
        critical=True
    )
    terrain_claim = (
        f"On Breeze's national parks page, the park '{park.park_name}' is classified under the terrain category "
        f"'{park.terrain_category}', and this category is either 'Tall Alpine Mountains' or 'Rocky Canyons'."
    )
    await evaluator.verify(
        claim=terrain_claim,
        node=terrain_leaf,
        sources=breeze_source,
        additional_instruction=(
            "Locate the park within the page and confirm the category heading or label that the park is listed under. "
            "Accept minor punctuation/casing variants but the category must correspond to one of the two allowed groups."
        )
    )

    # --------------------- Western Location (Critical, Parallel) --------------------- #
    west_node = evaluator.add_parallel(
        id=f"{park_id}_Western_Location",
        desc="Geographic location verification",
        parent=park_node,
        critical=True
    )

    # State stated (existence)
    evaluator.add_custom_node(
        result=bool(park.state and park.state.strip()),
        id=f"{park_id}_State_Stated",
        desc="The state where the park is located is stated",
        parent=west_node,
        critical=True
    )

    # Western US (logic/knowledge check)
    western_leaf = evaluator.add_leaf(
        id=f"{park_id}_Western_US",
        desc="The park is located in a western United States state (west of the 100th meridian)",
        parent=west_node,
        critical=True
    )
    western_claim = (
        f"The state '{park.state}' is located in the western United States (i.e., west of the 100th meridian; "
        "roughly west of Kansas/Nebraska/Dakotas)."
    )
    await evaluator.verify(
        claim=western_claim,
        node=western_leaf,
        additional_instruction=(
            "Consider the following states as west of the 100th meridian: WA, OR, CA, NV, ID, MT, WY, UT, CO, AZ, NM, AK, HI. "
            "Borderline states that are mostly east (e.g., KS, NE, ND, SD, OK, TX) should NOT be considered western for this task."
        )
    )

    # --------------------- Distinctness (Critical, single leaf) --------------------- #
    name_norm = _norm_name(park.park_name)
    is_distinct = bool(name_norm) and (name_norm not in seen_names)
    evaluator.add_custom_node(
        result=is_distinct,
        id=f"{park_id}_Distinctness",
        desc="The park is distinct from other parks in the answer",
        parent=park_node,
        critical=True
    )
    if is_distinct and name_norm:
        seen_names.add(name_norm)

    # --------------------- References (Critical, Parallel) --------------------- #
    refs_node = evaluator.add_parallel(
        id=f"{park_id}_References",
        desc=f"Reference URLs supporting the {['first','second','third'][park_index-1]} park identification and verification",
        parent=park_node,
        critical=True
    )

    # Breeze reference URL provided (must reference the NP page)
    breeze_ref_ok = bool(park.breeze_url and "flybreeze.com" in park.breeze_url and "national-parks" in park.breeze_url)
    evaluator.add_custom_node(
        result=breeze_ref_ok,
        id=f"{park_id}_Breeze_Reference_URL",
        desc="A reference URL to the Breeze Airways national parks page is provided",
        parent=refs_node,
        critical=True
    )

    # TSA Touchless ID reference URL provided
    tsa_ref_ok = bool(park.tsa_url and park.tsa_url.startswith(("http://", "https://")))
    evaluator.add_custom_node(
        result=tsa_ref_ok,
        id=f"{park_id}_TSA_Reference_URL",
        desc="A reference URL documenting TSA Touchless ID availability is provided",
        parent=refs_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Breeze western national parks with TSA Touchless ID (2026) task.
    """
    # Initialize evaluator (root parallel as rubric Root is parallel)
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction",
    )

    # Keep first 3 parks; pad if fewer
    parks: List[ParkItem] = list(extracted.parks[:3])
    while len(parks) < 3:
        parks.append(ParkItem())

    # Add helpful ground truth/context info (not used for scoring)
    evaluator.add_ground_truth({
        "required_breeze_page": BREEZE_NP_URL,
        "allowed_terrain_categories": list(ALLOWED_TERRAIN_CATEGORIES),
        "tsa_date_requirement": "January 2026 or later",
        "items_expected": 3
    })

    # Build verification subtrees for each of the three parks
    seen_names: Set[str] = set()
    for idx, park in enumerate(parks, start=1):
        await verify_one_park(
            evaluator=evaluator,
            parent_node=root,
            park=park,
            park_index=idx,
            seen_names=seen_names,
        )

    # Return summary
    return evaluator.get_summary()