import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "caribbean_family_resorts_bos"
TASK_DESCRIPTION = """
I am planning a family vacation from Boston and need to find all-inclusive resort options in the Caribbean that work well for families with young children. Specifically, I need resorts that meet ALL of the following requirements:

1. The destination must have direct (non-stop) flights from Boston Logan International Airport (BOS)
2. The direct flight time must be 4 hours and 30 minutes or less
3. The resort must offer all-inclusive packages (including meals, drinks, and activities)
4. The resort's kids club must accept children starting at age 3 or younger (not requiring children to be 4 years old)
5. The resort must offer family suites, multi-bedroom units, or interconnecting rooms that can accommodate at least 2 adults and 2 children
6. The resort must have a dedicated children's pool, splash pad, or water play area

Please identify three (3) different all-inclusive resorts in the Caribbean that meet ALL of these criteria. For each resort, provide:
- The resort name and specific location (island/country)
- The destination airport code and direct flight time from Boston
- Confirmation of the all-inclusive package availability
- Confirmation of kids club age requirements (must accept age 3 or younger)
- Description of family accommodation options
- Description of children's pool facilities
- A reference URL to the resort's official website or a verified booking platform page for this resort
"""


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class ResortEntry(BaseModel):
    # Identification/location
    name: Optional[str] = None
    location: Optional[str] = None  # island/country
    destination_name: Optional[str] = None  # city/island destination name
    destination_airport_code: Optional[str] = None  # IATA code like NAS, CUN, PLS

    # Flight info
    direct_flight_time: Optional[str] = None  # e.g., "3h 45m", "4 hours 25 minutes"
    flight_info_urls: List[str] = Field(default_factory=list)  # URLs that support direct flight + duration

    # All-inclusive
    all_inclusive: Optional[str] = None  # a phrase/claim that it’s all-inclusive
    all_inclusive_urls: List[str] = Field(default_factory=list)

    # Kids club
    kids_club_min_age: Optional[str] = None  # e.g., "age 3+", "starts at 2", "from 3 years"
    kids_club_urls: List[str] = Field(default_factory=list)

    # Family accommodations
    family_accommodation: Optional[str] = None  # description of family suites/interconnecting/multi-bedroom
    family_accommodation_urls: List[str] = Field(default_factory=list)

    # Children pool/splash
    childrens_pool: Optional[str] = None  # description
    childrens_pool_urls: List[str] = Field(default_factory=list)

    # General reference URLs (official site or verified booking platform)
    reference_urls: List[str] = Field(default_factory=list)


class ResortsExtraction(BaseModel):
    resorts: List[ResortEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_resorts() -> str:
    return """
    Extract up to five (5) resort entries exactly as presented in the answer. For each resort mentioned, return:
    - name: Resort name (string)
    - location: Specific island/country (string)
    - destination_name: Destination city/island name if given (string or null)
    - destination_airport_code: IATA airport code for the destination (string like "NAS", "CUN", "PLS"; or null if not explicitly provided)
    - direct_flight_time: The direct (non-stop) flight time from Boston (string, as written)
    - flight_info_urls: All URLs cited to support direct (non-stop) flight availability and/or flight duration claims for Boston (array, may be empty)
    - all_inclusive: A phrase/statement indicating all-inclusive availability (string or null)
    - all_inclusive_urls: All URLs cited to support the all-inclusive package claim (array, may be empty)
    - kids_club_min_age: The stated minimum age accepted at the kids club (string such as "age 3+", "from 2 years", or null if not stated)
    - kids_club_urls: All URLs cited to support the kids club age policy (array, may be empty)
    - family_accommodation: Description of family suites, multi-bedroom, or interconnecting rooms (string or null)
    - family_accommodation_urls: All URLs cited to support these accommodation options (array, may be empty)
    - childrens_pool: Description of dedicated children's pool, splash pad, or water play area (string or null)
    - childrens_pool_urls: All URLs cited to support the children's pool/splash feature (array, may be empty)
    - reference_urls: A list of URLs to the resort's official website and/or verified booking platform page(s) (array, may be empty)

    Rules:
    - Only extract information that is explicitly present in the answer.
    - For URL fields, include every URL the answer associates with that specific aspect. Do not invent any URLs.
    - If a field is missing in the answer, set it to null (for strings) or [] (for arrays).
    - The "reference_urls" should only include URLs shown in the answer that clearly point to the resort’s official site or well-known booking platforms (e.g., Expedia, Booking.com, Marriott/Hyatt brand pages, etc.).
    - Keep all strings as they appear; do not normalize or re-interpret times or ages.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                url = u.strip()
                if url and url not in seen:
                    merged.append(url)
                    seen.add(url)
    return merged


def _first_url(urls: List[str]) -> Optional[str]:
    return urls[0] if urls else None


def _ordinal(idx: int) -> str:
    mapping = {0: "First", 1: "Second", 2: "Third"}
    return mapping.get(idx, f"#{idx + 1}")


def _dest_label(resort: ResortEntry) -> str:
    # Build a readable destination label for flight claims
    code = (resort.destination_airport_code or "").strip()
    name = (resort.destination_name or "").strip()
    loc = (resort.location or "").strip()

    if code and name:
        return f"{name} (airport code {code})"
    if code and loc:
        return f"{loc} (airport code {code})"
    if code:
        return f"airport code {code}"
    if name:
        return name
    if loc:
        return loc
    return "the resort's destination"


def _resort_display_name(resort: ResortEntry, fallback: str) -> str:
    return resort.name.strip() if resort.name else fallback


def _must_use_urls_instruction(extra: str = "") -> str:
    base = (
        "Base your judgment ONLY on the content of the provided URL(s). "
        "If the URL(s) are missing, irrelevant, inaccessible, or do not clearly show the fact, "
        "you must conclude the claim is Not supported/Incorrect. "
    )
    return base + (extra or "")


# --------------------------------------------------------------------------- #
# Verification for one resort                                                 #
# --------------------------------------------------------------------------- #
async def verify_one_resort(
    evaluator: Evaluator,
    parent_node,
    resort: ResortEntry,
    idx: int,
) -> None:
    # Create the resort-level node (non-critical under root to allow partial credit across resorts)
    resort_node = evaluator.add_parallel(
        id=f"resort_{idx + 1}",
        desc=f"{_ordinal(idx)} qualifying resort with complete and accurate information",
        parent=parent_node,
        critical=False,
    )

    resort_name_disp = _resort_display_name(resort, f"Resort #{idx + 1}")

    # Group: Destination and Flight (critical sub-node: all three must hold)
    df_node = evaluator.add_parallel(
        id=f"resort_{idx + 1}_destination_flight",
        desc=f"Destination and flight information for {_ordinal(idx)} resort",
        parent=resort_node,
        critical=True
    )

    # 1) Direct flight from BOS
    df_leaf = evaluator.add_leaf(
        id=f"resort_{idx + 1}_direct_flight",
        desc="Verify the resort's destination has direct (non-stop) flights from Boston Logan Airport (BOS)",
        parent=df_node,
        critical=True
    )

    dest_lbl = _dest_label(resort)
    direct_flight_claim = (
        f"There is at least one direct (non-stop) commercial flight route from Boston Logan International Airport (BOS) "
        f"to {dest_lbl}."
    )
    await evaluator.verify(
        claim=direct_flight_claim,
        node=df_leaf,
        sources=resort.flight_info_urls,
        additional_instruction=_must_use_urls_instruction(
            "Seasonal or limited nonstop service counts as long as it is (or was recently) offered."
        ),
    )

    # 2) Flight duration <= 4h 30m
    fd_leaf = evaluator.add_leaf(
        id=f"resort_{idx + 1}_flight_duration",
        desc="Verify the direct flight time from Boston to this destination is 4 hours and 30 minutes or less",
        parent=df_node,
        critical=True
    )
    flight_duration_claim = (
        f"A typical direct (non-stop) flight from Boston Logan (BOS) to {dest_lbl} takes 4 hours 30 minutes or less."
    )
    await evaluator.verify(
        claim=flight_duration_claim,
        node=fd_leaf,
        sources=resort.flight_info_urls,
        additional_instruction=_must_use_urls_instruction(
            "Use published block times/durations on the cited page(s). Allow small reasonable schedule variations; "
            "if typical durations exceed 4h30m, mark Not supported."
        ),
    )

    # 3) Caribbean location
    car_leaf = evaluator.add_leaf(
        id=f"resort_{idx + 1}_caribbean_location",
        desc="Verify the resort is located in the Caribbean region",
        parent=df_node,
        critical=True
    )
    caribbean_claim = (
        f"{resort_name_disp} is located in the Caribbean region."
    )
    caribbean_sources = _merge_urls(resort.reference_urls)
    await evaluator.verify(
        claim=caribbean_claim,
        node=car_leaf,
        sources=caribbean_sources,
        additional_instruction=_must_use_urls_instruction(
            "Consider as Caribbean: Bahamas, Turks & Caicos, Dominican Republic, Puerto Rico, Jamaica, Cayman Islands, "
            "Aruba, Curaçao, Bonaire, St. Lucia, Antigua & Barbuda, Barbados, Grenada, St. Kitts & Nevis, "
            "St. Vincent & the Grenadines, USVI/BVI, Martinique, Guadeloupe, etc."
        ),
    )

    # All-inclusive packages
    ai_leaf = evaluator.add_leaf(
        id=f"resort_{idx + 1}_all_inclusive",
        desc="Verify the resort offers all-inclusive packages including meals, drinks, and activities",
        parent=resort_node,
        critical=True
    )
    ai_sources = _merge_urls(resort.all_inclusive_urls, resort.reference_urls)
    ai_claim = (
        f"{resort_name_disp} offers an all-inclusive plan/package that includes meals, drinks, and activities "
        f"(or equivalent inclusions)."
    )
    await evaluator.verify(
        claim=ai_claim,
        node=ai_leaf,
        sources=ai_sources,
        additional_instruction=_must_use_urls_instruction(
            "All-inclusive may be standard or an optional plan; confirm inclusions on the cited page(s)."
        ),
    )

    # Kids club accepts age 3 or younger
    kc_leaf = evaluator.add_leaf(
        id=f"resort_{idx + 1}_kids_club_age",
        desc="Verify the resort's kids club accepts children at age 3 or younger (not requiring age 4+)",
        parent=resort_node,
        critical=True
    )
    kc_sources = _merge_urls(resort.kids_club_urls, resort.reference_urls)
    kc_claim = (
        f"The kids club at {resort_name_disp} accepts children aged 3 years or younger (minimum age is 3 or lower)."
    )
    await evaluator.verify(
        claim=kc_claim,
        node=kc_leaf,
        sources=kc_sources,
        additional_instruction=_must_use_urls_instruction(
            "Look for minimum age policy; '3 years and up' qualifies. If '4 years and up' is required, it fails."
        ),
    )

    # Family suites / multi-bedroom / interconnecting rooms for 2 adults + 2 children
    fs_leaf = evaluator.add_leaf(
        id=f"resort_{idx + 1}_family_suite",
        desc="Verify the resort offers family suites, multi-bedroom units, or interconnecting rooms accommodating at least 2 adults and 2 children",
        parent=resort_node,
        critical=True
    )
    fs_sources = _merge_urls(resort.family_accommodation_urls, resort.reference_urls)
    fs_claim = (
        f"{resort_name_disp} offers family suites, multi-bedroom units, or interconnecting rooms that can "
        f"accommodate at least 2 adults and 2 children."
    )
    await evaluator.verify(
        claim=fs_claim,
        node=fs_leaf,
        sources=fs_sources,
        additional_instruction=_must_use_urls_instruction(
            "Evidence can be room type names (e.g., family suite, 2-bedroom), interconnecting availability, or "
            "occupancy/sleeping capacity statements clearly covering 2 adults + 2 children (>=4 guests)."
        ),
    )

    # Children's pool / splash pad / water play area
    cp_leaf = evaluator.add_leaf(
        id=f"resort_{idx + 1}_children_pool",
        desc="Verify the resort has a dedicated children's pool, splash pad, or water play area",
        parent=resort_node,
        critical=True
    )
    cp_sources = _merge_urls(resort.childrens_pool_urls, resort.reference_urls)
    cp_claim = (
        f"{resort_name_disp} has a dedicated children's pool, splash pad, or water play area designed for kids."
    )
    await evaluator.verify(
        claim=cp_claim,
        node=cp_leaf,
        sources=cp_sources,
        additional_instruction=_must_use_urls_instruction(
            "The feature must be explicitly for children (e.g., 'kids pool', 'splash pad', 'water playground'). "
            "A generic main pool is not sufficient."
        ),
    )

    # Reference URL is valid and about the resort (official site or verified booking platform)
    ref_leaf = evaluator.add_leaf(
        id=f"resort_{idx + 1}_reference_url",
        desc="Provide a valid reference URL to the resort's official website or verified booking platform",
        parent=resort_node,
        critical=True
    )
    ref_url = _first_url(resort.reference_urls)
    ref_claim = (
        f"This URL is an official website or a recognized booking platform page for {resort_name_disp}."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=ref_url,
        additional_instruction=_must_use_urls_instruction(
            "Recognized booking platforms include major OTAs (e.g., Expedia, Booking.com) or brand/chain sites "
            "(e.g., Marriott/Hyatt). The page must clearly be about this resort."
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
    Evaluate an answer for the 'Caribbean family resorts with BOS direct flights under 4.5h' task.
    Returns a standardized summary dictionary from the evaluator.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent resorts allow partial credit
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

    # Extract structured resort data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_resorts(),
        template_class=ResortsExtraction,
        extraction_name="resorts_extraction",
    )

    # Normalize to exactly 3 resort entries: take first 3 non-empty, then pad with empty
    resorts: List[ResortEntry] = []
    for r in (extracted.resorts or []):
        resorts.append(r)
        if len(resorts) >= 3:
            break
    while len(resorts) < 3:
        resorts.append(ResortEntry())

    # Build verification tree for each of the three resorts
    for i in range(3):
        await verify_one_resort(evaluator, root, resorts[i], i)

    # Optionally record custom info
    evaluator.add_custom_info(
        {
            "total_resorts_extracted": len(extracted.resorts) if extracted and extracted.resorts else 0,
            "evaluated_resorts": 3,
        },
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    return evaluator.get_summary()