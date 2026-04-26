import asyncio
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ada_trail_top3_2024"
TASK_DESCRIPTION = """
Identify an officially designated ADA-accessible hiking trail in one of the top 3 most visited United States national parks in 2024. The trail must meet all of the following requirements:
(1) The national park must rank within the top 3 for visitor numbers in 2024 according to National Park Service statistics;
(2) The trail must be officially listed as ADA-accessible or wheelchair accessible by the National Park Service;
(3) The trail must have a paved or firm, stable surface suitable for wheelchairs;
(4) The trail must meet ADA width requirements (minimum 36 inches clear width throughout, or 32 inches for sections no longer than 24 inches);
(5) The trail must have an average grade of 5% or less;
(6) The trail length must be between 0.5 and 4 miles (specify whether round trip or one-way);
(7) Using Shenandoah National Park's hiking difficulty formula (calculated as the square root of [Elevation Gain in feet × 2 × Distance in miles]), the trail must have a difficulty rating of less than 50, placing it in the 'Easiest' category.
Provide the name of the national park, the specific trail name, and all relevant specifications including trail length, elevation gain, surface type, width confirmation, and grade information, with reference URLs from official sources confirming each requirement.
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class TrailSpecs(BaseModel):
    # Length
    length_text: Optional[str] = None
    length_miles_numeric: Optional[str] = None  # Use string to be lenient; we'll parse
    length_label: Optional[str] = None          # one-way | round trip | loop | out-and-back (treated as round trip)
    length_source_urls: List[str] = Field(default_factory=list)

    # Elevation gain
    elevation_gain_ft_text: Optional[str] = None
    elevation_gain_ft_numeric: Optional[str] = None
    elevation_gain_source_urls: List[str] = Field(default_factory=list)

    # Surface
    surface_type: Optional[str] = None  # e.g., paved, asphalt, concrete, boardwalk, compacted gravel (firm/stable)
    surface_source_urls: List[str] = Field(default_factory=list)

    # Width (ADA)
    width_info_text: Optional[str] = None
    width_source_urls: List[str] = Field(default_factory=list)

    # Average grade
    grade_percent_text: Optional[str] = None
    grade_percent_numeric: Optional[str] = None
    grade_source_urls: List[str] = Field(default_factory=list)


class ADAAccessibleTrailInfo(BaseModel):
    # Park selection
    park_name: Optional[str] = None
    park_top3_nps_urls: List[str] = Field(default_factory=list)  # Official NPS visitation/ranking URLs

    # Trail selection
    trail_name: Optional[str] = None
    trail_nps_access_urls: List[str] = Field(default_factory=list)  # Official NPS park or NPS.gov page that lists ADA/wheelchair-accessible status

    # Required specifications
    specs: Optional[TrailSpecs] = None

    # Shenandoah difficulty formula and computed difficulty (if the answer computed it)
    shenandoah_formula_urls: List[str] = Field(default_factory=list)
    computed_difficulty_text: Optional[str] = None
    computed_difficulty_numeric: Optional[str] = None

    # Optional supporting amenities
    amenities_accessible_parking_urls: List[str] = Field(default_factory=list)
    amenities_accessible_restrooms_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trail_info() -> str:
    return """
    Extract the single final choice of park and trail that the answer claims satisfy ALL requirements. If the answer mentions multiple candidates, pick the one it ultimately presents as satisfying the constraints (otherwise pick the first complete one).

    Return a JSON object with:
    - park_name: string or null
    - park_top3_nps_urls: array of URLs explicitly included in the answer that support the park being in the top 3 most visited in 2024 (NPS official statistics pages required)
    - trail_name: string or null
    - trail_nps_access_urls: array of URLs (from NPS.gov or the official park site under NPS.gov) explicitly included in the answer, that list/describe the trail as ADA-accessible or wheelchair accessible
    - specs: object with the following fields (strings can be free-form text copied from answer; numeric fields should be numeric strings when available):
        - length_text: the trail length phrase exactly as stated (e.g., "1.2-mile loop (round trip)")
        - length_miles_numeric: numeric miles if given (e.g., "1.2"); else null
        - length_label: one of "one-way", "round trip", "loop", "out-and-back", or null if not explicit
        - length_source_urls: array of URLs that support the length and labeling (prefer official sources)
        - elevation_gain_ft_text: the elevation gain phrase from the answer (e.g., "120 feet")
        - elevation_gain_ft_numeric: numeric elevation gain in feet if given (e.g., "120"); else null
        - elevation_gain_source_urls: array of URLs that support elevation gain
        - surface_type: surface label from the answer (e.g., "paved", "asphalt", "boardwalk", "compacted gravel (firm/stable)")
        - surface_source_urls: array of URLs that support surface type and its suitability for wheelchairs
        - width_info_text: width/ADA width compliance text from the answer (e.g., "meets 36-inch minimum width")
        - width_source_urls: array of URLs that support width/compliance basis
        - grade_percent_text: grade phrase from the answer (e.g., "average grade 3%")
        - grade_percent_numeric: numeric grade percent if given (e.g., "3"); else null
        - grade_source_urls: array of URLs that support grade info (average grade)
    - shenandoah_formula_urls: array of URLs (prefer an official NPS Shenandoah page) included in the answer that state the difficulty formula √[Elevation Gain (ft) × 2 × Distance (miles)] and the category thresholds
    - computed_difficulty_text: the computed difficulty phrase exactly as presented in the answer (if the answer computed it), else null
    - computed_difficulty_numeric: numeric difficulty value from the answer if present (string), else null
    - amenities_accessible_parking_urls: array of URLs that the answer cites for accessible parking near the trailhead/route (optional)
    - amenities_accessible_restrooms_urls: array of URLs that the answer cites for accessible restrooms near the trailhead/route (optional)

    CRITICAL RULES:
    - Only extract URLs that are explicitly present in the answer (plain links or markdown links). Do not invent URLs.
    - Prefer official NPS URLs (nps.gov, irma.nps.gov, data.nps.gov) where required by the prompt. If the answer provides multiple URLs, include them all.
    - If any field is missing in the answer, set it to null or an empty array accordingly.
    """


# --------------------------------------------------------------------------- #
# Helpers: parsing and computation                                            #
# --------------------------------------------------------------------------- #
def _first_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    s = text.strip()
    s = s.replace(",", "")
    m = re.search(r"([-+]?\d*\.?\d+)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_miles(value_str: Optional[str], fallback_text: Optional[str]) -> Optional[float]:
    # Priority: explicit numeric string
    v = _first_number(value_str)
    if v is not None:
        return v
    # Fallback: scan the text for "X mile(s)"
    if fallback_text:
        txt = fallback_text.replace(",", "").lower()
        m = re.search(r"(\d*\.?\d+)\s*mi(le)?s?\b", txt)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
        # Also catch hyphenated like "1.2-mile"
        m2 = re.search(r"(\d*\.?\d+)\s*-\s*mi(le)?s?|\b(\d*\.?\d+)-mile", txt)
        if m2:
            # groups may vary; extract the first floating number
            m3 = re.search(r"(\d*\.?\d+)", m2.group(0))
            if m3:
                try:
                    return float(m3.group(1))
                except Exception:
                    pass
    return None


def parse_feet(value_str: Optional[str], fallback_text: Optional[str]) -> Optional[float]:
    v = _first_number(value_str)
    if v is not None:
        return v
    if fallback_text:
        txt = fallback_text.replace(",", "").lower()
        m = re.search(r"(\d*\.?\d+)\s*(ft|feet|foot)\b", txt)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def parse_percent(value_str: Optional[str], fallback_text: Optional[str]) -> Optional[float]:
    v = _first_number(value_str)
    if v is not None:
        return v
    if fallback_text:
        txt = fallback_text.lower()
        m = re.search(r"(\d*\.?\d+)\s*%", txt)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def derive_length_label(label: Optional[str], text: Optional[str]) -> Optional[str]:
    if label:
        return label.strip().lower()
    if not text:
        return None
    t = text.lower()
    if "round trip" in t or "round-trip" in t or "loop" in t or "circuit" in t:
        return "round trip"
    if "one way" in t or "one-way" in t:
        return "one-way"
    if "out and back" in t or "out-and-back" in t:
        return "round trip"
    return None


def compute_shenandoah_difficulty(elev_gain_ft: Optional[float], distance_miles: Optional[float]) -> Optional[float]:
    if elev_gain_ft is None or distance_miles is None:
        return None
    try:
        return math.sqrt(elev_gain_ft * 2 * distance_miles)
    except Exception:
        return None


def combine_sources(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for urls in url_lists:
        if not urls:
            continue
        for u in urls:
            u = (u or "").strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_park_selection_subtree(evaluator: Evaluator, parent, data: ADAAccessibleTrailInfo):
    node = evaluator.add_parallel(
        id="park_selection",
        desc="Select and name a U.S. national park that is in the top 3 most visited in 2024 per official NPS statistics.",
        parent=parent,
        critical=True,  # Children must all be critical to satisfy JSON consistency
    )

    # Leaf: park_name_provided (existence)
    evaluator.add_custom_node(
        result=bool(data and data.park_name and data.park_name.strip()),
        id="park_name_provided",
        desc="The national park name is provided.",
        parent=node,
        critical=True
    )

    # Leaf: park_top3_2024_nps_verified (verify with NPS URL(s))
    park_rank_node = evaluator.add_leaf(
        id="park_top3_2024_nps_verified",
        desc="A reference URL to an official NPS resource supports that the selected park ranks within the top 3 most visited in 2024.",
        parent=node,
        critical=True
    )

    park_name = (data.park_name or "").strip()
    claim = f"The selected park '{park_name}' ranks within the top 3 U.S. national parks by recreation visits in 2024 according to the National Park Service."
    await evaluator.verify(
        claim=claim,
        node=park_rank_node,
        sources=data.park_top3_nps_urls,
        additional_instruction="Only accept official NPS sources (e.g., nps.gov, irma.nps.gov, data.nps.gov). Confirm that the 2024 ranking places the chosen park within the top 3."
    )


async def build_trail_selection_subtree(evaluator: Evaluator, parent, data: ADAAccessibleTrailInfo):
    node = evaluator.add_parallel(
        id="trail_selection",
        desc="Identify a specific trail in the selected park and verify it is officially listed by NPS as ADA-accessible or wheelchair accessible.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data and data.trail_name and data.trail_name.strip()),
        id="trail_name_provided",
        desc="The specific trail name is provided.",
        parent=node,
        critical=True
    )

    verify_access_node = evaluator.add_leaf(
        id="nps_accessibility_listing_verified",
        desc="NPS page explicitly lists/describes the trail as ADA-accessible/ADA-compliant or wheelchair accessible.",
        parent=node,
        critical=True
    )

    trail_name = (data.trail_name or "").strip()
    claim = f"The trail '{trail_name}' is explicitly listed by the National Park Service as ADA-accessible (or wheelchair accessible)."
    await evaluator.verify(
        claim=claim,
        node=verify_access_node,
        sources=data.trail_nps_access_urls,
        additional_instruction="Look for explicit phrases such as 'ADA accessible', 'ADA-compliant', 'wheelchair accessible', or equivalent on an official NPS.gov page for this specific trail."
    )


async def build_required_specs_subtree(evaluator: Evaluator, parent, data: ADAAccessibleTrailInfo,
                                       parsed_distance_mi: Optional[float],
                                       parsed_elev_gain_ft: Optional[float],
                                       parsed_grade_pct: Optional[float],
                                       length_label: Optional[str]):
    node = evaluator.add_parallel(
        id="required_trail_specs_with_sources",
        desc="Provide required specs (distance, elevation gain, surface, width, grade) with supporting reference URLs.",
        parent=parent,
        critical=True
    )

    specs = data.specs or TrailSpecs()

    # Length requirement
    length_sources = combine_sources(specs.length_source_urls, data.trail_nps_access_urls)
    length_leaf = evaluator.add_leaf(
        id="length_requirement_verified",
        desc="Trail length is stated, labeled one-way/round trip, between 0.5 and 4 miles inclusive, supported by a reference URL.",
        parent=node,
        critical=True
    )

    length_text = specs.length_text or ""
    label_display = length_label or "unspecified"
    if parsed_distance_mi is not None:
        range_text = f"which is between 0.5 and 4 miles inclusive: {0.5 <= parsed_distance_mi <= 4}"
        dist_phrase = f"approximately {parsed_distance_mi:.2f} miles"
    else:
        range_text = "and the stated distance lies between 0.5 and 4 miles inclusive"
        dist_phrase = "a distance in miles"

    claim = (
        f"The cited source(s) state that the trail length is {length_text} ({dist_phrase}), "
        f"and the labeling is clear as '{label_display}' (one-way/round trip/loop/out-and-back). "
        f"Also verify the value is between 0.5 and 4 miles inclusive."
    )
    await evaluator.verify(
        claim=claim,
        node=length_leaf,
        sources=length_sources,
        additional_instruction="Confirm that the page explicitly indicates the distance and whether it is one-way, round-trip, loop, or out-and-back. Treat 'loop' as total trip distance. The numeric value must be within [0.5, 4]."
    )

    # Elevation gain requirement
    elev_sources = combine_sources(specs.elevation_gain_source_urls, data.trail_nps_access_urls)
    elev_leaf = evaluator.add_leaf(
        id="elevation_gain_verified",
        desc="Elevation gain (in feet) is provided and supported by a reference URL.",
        parent=node,
        critical=True
    )

    elev_phrase = f"about {parsed_elev_gain_ft:.0f} feet" if parsed_elev_gain_ft is not None else (specs.elevation_gain_ft_text or "an elevation gain")
    claim = f"The cited source(s) indicate that the trail's elevation gain is {elev_phrase}."
    await evaluator.verify(
        claim=claim,
        node=elev_leaf,
        sources=elev_sources,
        additional_instruction="Accept small rounding differences (e.g., 95 vs 100 ft). 'Elevation change' or cumulative gain figures are acceptable if clearly describing overall ascent."
    )

    # Surface requirement
    surface_sources = combine_sources(specs.surface_source_urls, data.trail_nps_access_urls)
    surface_leaf = evaluator.add_leaf(
        id="surface_requirement_verified",
        desc="Surface type is stated (paved or firm/stable suitable for wheelchairs) and supported by a reference URL.",
        parent=node,
        critical=True
    )

    surface_label = (specs.surface_type or "").strip().lower()
    claim = (
        f"The cited source(s) confirm the trail surface is '{surface_label}', and that it is paved or firm/stable and suitable for wheelchairs."
    )
    await evaluator.verify(
        claim=claim,
        node=surface_leaf,
        sources=surface_sources,
        additional_instruction="Accept surfaces such as asphalt, concrete, boardwalk, or compacted/crushed stone explicitly described as 'firm and stable'. Do not accept soft sand or loose, unstable surfaces."
    )

    # Width requirement (ADA)
    width_sources = combine_sources(specs.width_source_urls, data.trail_nps_access_urls)
    width_leaf = evaluator.add_leaf(
        id="width_requirement_verified",
        desc="Information supports ADA width compliance (≥36\" clear throughout, or ≥32\" for short ≤24\" segments), with supporting URL(s).",
        parent=node,
        critical=True
    )
    width_text = specs.width_info_text or ""
    claim = (
        "The cited source(s) confirm that the trail meets ADA width requirements: a minimum 36 inches clear width throughout, "
        "or at least 32 inches for sections no longer than 24 inches. "
        f"The provided information states: '{width_text}'."
    )
    await evaluator.verify(
        claim=claim,
        node=width_leaf,
        sources=width_sources,
        additional_instruction=(
            "Prefer explicit width statements. If an official NPS accessibility page explicitly designates the trail as ADA- or wheelchair-accessible and "
            "provides compliance details or references built-to-standards infrastructure (e.g., boardwalk with ADA compliance), that may be acceptable."
        )
    )

    # Average grade requirement (<= 5%)
    grade_sources = combine_sources(specs.grade_source_urls, data.trail_nps_access_urls)
    grade_leaf = evaluator.add_leaf(
        id="average_grade_requirement_verified",
        desc="Average grade is ≤ 5% and supported by a reference URL.",
        parent=node,
        critical=True
    )

    if parsed_grade_pct is not None:
        grade_phrase = f"an average grade of about {parsed_grade_pct:.1f}% (≤ 5% is required)"
    else:
        grade_phrase = "an average grade that is ≤ 5%"

    claim = f"The cited source(s) indicate the trail has {grade_phrase}."
    await evaluator.verify(
        claim=claim,
        node=grade_leaf,
        sources=grade_sources,
        additional_instruction="Focus on average/typical grade. If only a max grade is provided, it is insufficient alone unless the page also states average/typical grade ≤ 5%."
    )


async def build_difficulty_subtree(evaluator: Evaluator, parent, data: ADAAccessibleTrailInfo,
                                   distance_mi: Optional[float],
                                   elev_gain_ft: Optional[float],
                                   computed_from_code: Optional[float]):
    node = evaluator.add_sequential(
        id="difficulty_check",
        desc="Verify Shenandoah difficulty requirement (<50, 'Easiest') using the specified formula.",
        parent=parent,
        critical=True
    )

    # Shenandoah formula cited
    cite_leaf = evaluator.add_leaf(
        id="shenandoah_formula_cited",
        desc="A reference URL (NPS Shenandoah or equivalent official) states the formula √[Elevation Gain (ft) × 2 × Distance (miles)].",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Shenandoah National Park (NPS) defines hiking difficulty as the square root of [Elevation Gain in feet × 2 × Distance in miles].",
        node=cite_leaf,
        sources=data.shenandoah_formula_urls,
        additional_instruction="Look for the exact or near-exact formula text on an official Shenandoah/NPS page."
    )

    # Difficulty computed and < 50
    diff_leaf = evaluator.add_leaf(
        id="difficulty_computed_and_below_50",
        desc="The answer computes the difficulty using the formula and the computed rating is < 50.",
        parent=node,
        critical=True
    )

    # We'll compute here using parsed numbers; if unavailable, still phrase a verification.
    if computed_from_code is not None:
        calc_phrase = f"{computed_from_code:.1f}"
        lt50 = computed_from_code < 50.0
    else:
        calc_phrase = "a value less than 50 using the formula"
        lt50 = False  # unknown

    # Use Shenandoah page(s) again as sources to ground the formula reference
    claim = (
        f"Using the Shenandoah formula sqrt(Elevation Gain (ft) × 2 × Distance (miles)) "
        f"with the provided elevation gain and distance, the computed difficulty is {calc_phrase} and is less than 50."
    )
    await evaluator.verify(
        claim=claim,
        node=diff_leaf,
        sources=data.shenandoah_formula_urls,
        additional_instruction="Recalculate the difficulty using the verified elevation gain (ft) and distance (miles). Accept standard rounding; result must be strictly < 50."
    )

    # Easiest category confirmed (< 50)
    easiest_leaf = evaluator.add_leaf(
        id="easiest_category_confirmed",
        desc="The answer states that difficulty < 50 corresponds to 'Easiest' category per Shenandoah.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Per Shenandoah National Park guidance, a difficulty rating below 50 is categorized as 'Easiest'.",
        node=easiest_leaf,
        sources=data.shenandoah_formula_urls,
        additional_instruction="Check the same Shenandoah/NPS difficulty guidance for category thresholds that define 'Easiest' as < 50."
    )


async def build_optional_amenities_subtree(evaluator: Evaluator, parent, data: ADAAccessibleTrailInfo):
    node = evaluator.add_parallel(
        id="supporting_amenities_optional",
        desc="Optional accessible amenities near the trailhead/route.",
        parent=parent,
        critical=False
    )

    # Accessible parking (optional)
    park_leaf = evaluator.add_leaf(
        id="accessible_parking_mentioned_with_support",
        desc="Accessible parking availability is mentioned and supported by a reference URL.",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="Accessible parking is available at or near the trailhead/route.",
        node=park_leaf,
        sources=data.amenities_accessible_parking_urls,
        additional_instruction="Verify that the cited page mentions accessible parking specifically for this trail or its immediate area."
    )

    # Accessible restrooms (optional)
    rest_leaf = evaluator.add_leaf(
        id="accessible_restrooms_mentioned_with_support",
        desc="Accessible restroom availability is mentioned and supported by a reference URL.",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="Accessible restrooms are available at or near the trailhead/route.",
        node=rest_leaf,
        sources=data.amenities_accessible_restrooms_urls,
        additional_instruction="Verify that the cited page mentions accessible restrooms specifically for this trail or its immediate area."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the ADA-accessible trail in a top-3 (2024) national park task.
    """
    # Initialize evaluator (root: sequential to short-circuit on early failure)
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

    # Extract structured info from the answer
    extracted: ADAAccessibleTrailInfo = await evaluator.extract(
        prompt=prompt_extract_trail_info(),
        template_class=ADAAccessibleTrailInfo,
        extraction_name="ada_trail_choice",
    )

    # Pre-parse numeric values for internal checks/computation
    specs = extracted.specs or TrailSpecs()
    parsed_distance_mi = parse_miles(specs.length_miles_numeric, specs.length_text)
    parsed_elev_gain_ft = parse_feet(specs.elevation_gain_ft_numeric, specs.elevation_gain_ft_text)
    parsed_grade_pct = parse_percent(specs.grade_percent_numeric, specs.grade_percent_text)
    length_label = derive_length_label(specs.length_label, specs.length_text)
    computed_difficulty = compute_shenandoah_difficulty(parsed_elev_gain_ft, parsed_distance_mi)

    # Record computed info (for transparency/debugging)
    evaluator.add_custom_info(
        info={
            "park_name": extracted.park_name,
            "trail_name": extracted.trail_name,
            "parsed_distance_mi": parsed_distance_mi,
            "parsed_elevation_gain_ft": parsed_elev_gain_ft,
            "parsed_grade_percent": parsed_grade_pct,
            "derived_length_label": length_label,
            "computed_difficulty": computed_difficulty,
        },
        info_type="computed_values",
        info_name="computed_summary"
    )

    # Build verification tree per rubric
    # 1) Park selection
    await build_park_selection_subtree(evaluator, root, extracted)

    # 2) Trail selection
    await build_trail_selection_subtree(evaluator, root, extracted)

    # 3) Required trail specs with sources
    await build_required_specs_subtree(
        evaluator, root, extracted, parsed_distance_mi, parsed_elev_gain_ft, parsed_grade_pct, length_label
    )

    # 4) Difficulty check
    await build_difficulty_subtree(evaluator, root, extracted, parsed_distance_mi, parsed_elev_gain_ft, computed_difficulty)

    # 5) Optional supporting amenities
    await build_optional_amenities_subtree(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()