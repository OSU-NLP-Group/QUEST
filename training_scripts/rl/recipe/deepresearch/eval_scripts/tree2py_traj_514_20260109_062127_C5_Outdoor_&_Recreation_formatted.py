import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "co_national_parks_2025"
TASK_DESCRIPTION = (
    "Identify two distinct Colorado national parks that meet the following criteria: "
    "(1) one park must have a visitor center located above 10,000 feet elevation, and you must specify which visitor center and provide its exact elevation; "
    "(2) the other park must require advance online reservations for accessing its primary attractions during summer 2025 (May through September), and you must specify the reservation system used. "
    "For each park, provide the park name, the requested specific details, and an official reference URL supporting your information."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class HighElevationPark(BaseModel):
    park_name: Optional[str] = None
    visitor_center_name: Optional[str] = None
    elevation_feet: Optional[str] = None  # keep as raw string from answer (e.g., "11,796 ft" or "11796 feet")
    reference_urls: List[str] = Field(default_factory=list)


class ReservationPark(BaseModel):
    park_name: Optional[str] = None
    reservation_system: Optional[str] = None  # e.g., "Recreation.gov timed entry", "NPS Timed Entry"
    summer_2025_applicability: Optional[str] = None  # raw text describing May–September 2025 applicability
    reference_urls: List[str] = Field(default_factory=list)


class ParksExtraction(BaseModel):
    park_high_elevation: Optional[HighElevationPark] = None
    park_reservation: Optional[ReservationPark] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
Extract the following structured information from the answer for exactly two Colorado national parks that satisfy the two distinct criteria below. If the answer lists multiple candidates for a criterion, choose the first well-supported one mentioned. If any field is missing, return null for that field (or an empty array for URLs).

1) Park with high-elevation visitor center (Park A):
   - park_name: The Colorado national park name (e.g., "Rocky Mountain National Park").
   - visitor_center_name: The specific visitor center within the park (e.g., "Alpine Visitor Center").
   - elevation_feet: The elevation of that visitor center in feet as presented in the answer (e.g., "11,796 ft", "11796 feet"); do not convert units; extract exactly as written.
   - reference_urls: All official URLs cited in the answer that support the elevation claim for that visitor center (e.g., pages on nps.gov, recreation.gov, or other official .gov domains). If none are cited, return an empty array.

2) Park requiring advance reservations in summer 2025 (Park B):
   - park_name: The Colorado national park name.
   - reservation_system: The online reservation platform/system named in the answer (e.g., "Recreation.gov", "Timed Entry via Recreation.gov").
   - summer_2025_applicability: The statement from the answer that indicates the reservation requirement is in effect during summer 2025 (May–September 2025); extract the relevant text phrase or date range if present.
   - reference_urls: All official URLs cited in the answer that support the reservation requirement and the platform used (prefer nps.gov or recreation.gov). If none are cited, return an empty array.

Return the result as a JSON object with fields:
{
  "park_high_elevation": { ... },
  "park_reservation": { ... }
}
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _parse_feet_value(elev_str: Optional[str]) -> Optional[float]:
    """
    Parse a numeric feet value from a free-form elevation string.
    Accepts formats like "11,796 ft", "11796 feet", "11,796", etc.
    Returns None if not parseable.
    """
    if not elev_str:
        return None
    # Keep digits and dot only; remove commas and non-digit separators first
    cleaned = elev_str.replace(",", " ")
    # Capture the first number (possibly decimal)
    m = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _normalize_park_name(name: Optional[str]) -> str:
    """
    Normalize park names for equality comparison:
    - Lowercase
    - Remove 'national park' or 'np'
    - Remove punctuation and whitespace
    """
    if not name:
        return ""
    n = name.lower()
    # Remove common suffix words
    n = re.sub(r"\bnational\s+park\b", "", n)
    n = re.sub(r"\bnp\b", "", n)
    # Remove punctuation and spaces
    n = re.sub(r"[^a-z0-9]", "", n)
    return n


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_high_elevation_subtree(
    evaluator: Evaluator,
    parent,
    data: Optional[HighElevationPark],
) -> None:
    """
    Build verification nodes for:
    - Park with a visitor center above 10,000 ft, including exact elevation and official reference.
    """
    node = evaluator.add_parallel(
        id="park_with_high_elev_vc",
        desc="Colorado national park with a visitor center located above 10,000 feet elevation, including the specific visitor center and its exact elevation",
        parent=parent,
        critical=True
    )

    park_name = data.park_name if data else None
    vc_name = data.visitor_center_name if data else None
    elev_str = data.elevation_feet if data else None
    elev_val = _parse_feet_value(elev_str)
    urls = data.reference_urls if data else []

    # Leaf: Park_Name_Provided (existence)
    evaluator.add_custom_node(
        result=_is_nonempty(park_name),
        id="park_a_name_provided",
        desc="A specific Colorado national park is named (Park A)",
        parent=node,
        critical=True
    )

    # Leaf: Visitor_Center_Named (existence)
    evaluator.add_custom_node(
        result=_is_nonempty(vc_name),
        id="visitor_center_named",
        desc="The specific visitor center within Park A is identified by name",
        parent=node,
        critical=True
    )

    # Leaf: Exact_Elevation_Provided (must be numeric feet)
    evaluator.add_custom_node(
        result=(elev_val is not None),
        id="exact_elevation_provided",
        desc="The exact elevation of the visitor center is provided as a specific numeric value in feet above sea level",
        parent=node,
        critical=True
    )

    # Leaf: Elevation_Above_10000_Feet (numeric check)
    evaluator.add_custom_node(
        result=(elev_val is not None and elev_val > 10000.0),
        id="elevation_above_10000",
        desc="The provided elevation value is above 10,000 feet above sea level",
        parent=node,
        critical=True
    )

    # Leaf: Reference_URL (verify support with official URL(s))
    ref_leaf = evaluator.add_leaf(
        id="high_elev_reference_url",
        desc="An official reference URL supporting the visitor center elevation claim is provided",
        parent=node,
        critical=True
    )

    # Build claim using available data; allow for minor rounding tolerance
    vc_display = vc_name or "the specified visitor center"
    park_display = park_name or "the specified park"
    elev_display = f"{int(elev_val)}" if (elev_val is not None and elev_val.is_integer()) else (str(elev_val) if elev_val is not None else (elev_str or "an elevation above 10,000"))

    claim = (
        f"The official webpage confirms that {vc_display} in {park_display} is at an elevation of approximately {elev_display} feet (above sea level). "
        f"Minor rounding differences are acceptable."
    )

    add_ins = (
        "Judge only based on the provided webpage(s). Consider the claim supported only if the page is an official source (e.g., nps.gov, recreation.gov, or another .gov official site) "
        "and it explicitly mentions the visitor center by name and its elevation in feet. Allow minor rounding differences (e.g., 11,796 vs. 11,800). "
        "If the URL is missing, irrelevant, or not official, mark as not supported."
    )

    await evaluator.verify(
        claim=claim,
        node=ref_leaf,
        sources=urls if urls else None,
        additional_instruction=add_ins
    )


async def build_reservation_subtree(
    evaluator: Evaluator,
    parent,
    data: Optional[ReservationPark],
) -> None:
    """
    Build verification nodes for:
    - Park requiring advance online reservations during summer 2025 (May–September), specifying the reservation system and an official reference URL.
    """
    node = evaluator.add_parallel(
        id="park_requiring_advance_reservations",
        desc="Colorado national park requiring advance online reservations for accessing its primary attractions during summer 2025 (May–September), including the reservation system used",
        parent=parent,
        critical=True
    )

    park_name = data.park_name if data else None
    system = data.reservation_system if data else None
    applicability_text = data.summer_2025_applicability if data else None
    urls = data.reference_urls if data else []

    # Leaf: Park_Name_Provided (existence)
    evaluator.add_custom_node(
        result=_is_nonempty(park_name),
        id="park_b_name_provided",
        desc="A specific Colorado national park is named (Park B)",
        parent=node,
        critical=True
    )

    # Leaf: Reference_URL (existence of at least one URL)
    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="reservation_reference_url",
        desc="An official reference URL supporting the reservation requirement and system is provided",
        parent=node,
        critical=True
    )

    # Leaf: Online_Reservation_System (verify platform used)
    sys_leaf = evaluator.add_leaf(
        id="online_reservation_system",
        desc="The online reservation system/platform used is identified",
        parent=node,
        critical=True
    )

    sys_claim = (
        f"The official webpage for {park_name or 'the specified park'} indicates that advance reservations are managed via "
        f"{system or 'the specified platform'} (e.g., Recreation.gov), for accessing primary attractions or timed-entry areas."
    )

    sys_add_ins = (
        "Focus on whether the page identifies the reservation platform/system used (e.g., Recreation.gov). "
        "Do not judge the date window in this verification; only the system/platform. "
        "Consider the claim supported only if the page is an official source (e.g., nps.gov or recreation.gov) and clearly mentions the system used."
    )

    await evaluator.verify(
        claim=sys_claim,
        node=sys_leaf,
        sources=urls if urls else None,
        additional_instruction=sys_add_ins
    )

    # Leaf: Summer_2025_Applicability (verify timeframe for May–September 2025)
    app_leaf = evaluator.add_leaf(
        id="summer_2025_applicability",
        desc="The reservation requirement is documented as applicable during summer 2025 (May through September 2025)",
        parent=node,
        critical=True
    )

    app_claim = (
        f"The official webpage states that advance online reservations are required to access primary attractions in "
        f"summer 2025 (May through September 2025) for {park_name or 'the specified park'}."
    )

    app_add_ins = (
        "Consider the claim supported if the page clearly indicates the requirement applies during summer 2025 and covers a period spanning May through September 2025. "
        "Allow reasonable phrasing like 'May to September 2025', 'May–Sept 2025', specific date ranges within those months, or equivalent. "
        "If the page only discusses other years or lacks dates, mark as not supported. The page should be official (nps.gov or recreation.gov)."
    )

    await evaluator.verify(
        claim=app_claim,
        node=app_leaf,
        sources=urls if urls else None,
        additional_instruction=app_add_ins
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Colorado national parks criteria task.
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction"
    )

    # Build two main critical branches in parallel
    await build_high_elevation_subtree(evaluator, root, extracted.park_high_elevation)
    await build_reservation_subtree(evaluator, root, extracted.park_reservation)

    # Distinct parks check (critical)
    park_a_name = extracted.park_high_elevation.park_name if extracted.park_high_elevation else None
    park_b_name = extracted.park_reservation.park_name if extracted.park_reservation else None

    norm_a = _normalize_park_name(park_a_name)
    norm_b = _normalize_park_name(park_b_name)
    distinct = bool(norm_a and norm_b and norm_a != norm_b)

    evaluator.add_custom_node(
        result=distinct,
        id="distinct_parks",
        desc="Park A and Park B are different parks (the two provided park names are not the same)",
        parent=root,
        critical=True
    )

    # Optional: record some custom info for debugging
    evaluator.add_custom_info(
        {
            "park_a_raw_name": park_a_name,
            "park_b_raw_name": park_b_name,
            "park_a_normalized": norm_a,
            "park_b_normalized": norm_b,
            "park_a_elevation_str": (extracted.park_high_elevation.elevation_feet if extracted.park_high_elevation else None),
            "park_a_elevation_numeric": _parse_feet_value(extracted.park_high_elevation.elevation_feet) if extracted.park_high_elevation else None,
            "park_a_urls": (extracted.park_high_elevation.reference_urls if extracted.park_high_elevation else []),
            "park_b_system": (extracted.park_reservation.reservation_system if extracted.park_reservation else None),
            "park_b_applicability": (extracted.park_reservation.summer_2025_applicability if extracted.park_reservation else None),
            "park_b_urls": (extracted.park_reservation.reference_urls if extracted.park_reservation else []),
        },
        info_type="debug_info",
        info_name="extracted_debug_info"
    )

    return evaluator.get_summary()