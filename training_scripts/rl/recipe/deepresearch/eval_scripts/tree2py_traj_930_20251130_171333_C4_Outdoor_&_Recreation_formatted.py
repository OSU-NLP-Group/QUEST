import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "december_parks_western_us"
TASK_DESCRIPTION = """
Identify at least four national parks in the western United States that are recommended for December visits. For each park, provide: (1) specific reasons why December is a good time to visit, (2) December temperature ranges and general weather conditions, (3) winter road accessibility and operational status for December 2025, and (4) any current safety alerts or closures that may affect December visitors. Ensure geographic diversity among your selections. Provide reference URLs from National Park Service websites and reputable travel or weather sources to support your answer.
"""

# Define Western US states (US Census West: Pacific + Mountain Divisions)
WEST_STATE_NAMES = {
    "Alaska", "Arizona", "California", "Colorado", "Hawaii", "Idaho",
    "Montana", "Nevada", "New Mexico", "Oregon", "Utah", "Washington", "Wyoming"
}
WEST_STATE_ABBR = {
    "AK": "Alaska", "AZ": "Arizona", "CA": "California", "CO": "Colorado",
    "HI": "Hawaii", "ID": "Idaho", "MT": "Montana", "NV": "Nevada",
    "NM": "New Mexico", "OR": "Oregon", "UT": "Utah", "WA": "Washington", "WY": "Wyoming"
}
WEST_STATES_ALL = set(WEST_STATE_NAMES)  # Canonical names set


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkItem(BaseModel):
    park_name: Optional[str] = None
    state_or_region: Optional[str] = None

    # NPS official unit page (designation and location)
    nps_unit_url: Optional[str] = None

    # Travel recommendation sources explicitly mentioning December (URLs)
    travel_recommendation_urls: List[str] = Field(default_factory=list)

    # Reasons why December is good (from sources)
    december_reasons: List[str] = Field(default_factory=list)

    # December temperatures and weather (from reputable weather/climate sources)
    dec_temp_range: Optional[str] = None  # e.g., "30–50°F"
    dec_weather_summary: Optional[str] = None
    weather_urls: List[str] = Field(default_factory=list)

    # Winter road accessibility and operations for Dec 2025 (from NPS sources)
    dec_2025_ops_summary: Optional[str] = None
    nps_ops_urls: List[str] = Field(default_factory=list)

    # Current safety alerts or closures (from NPS alerts/conditions URLs)
    alerts_summary: Optional[str] = None
    nps_alerts_urls: List[str] = Field(default_factory=list)


class ParksExtraction(BaseModel):
    parks: List[ParkItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
    Extract the proposed western U.S. National Parks recommended for December visits from the answer.
    Return an array 'parks' with up to 5 items. For each park item, extract the following fields strictly from the answer:

    1) park_name: The full park name (e.g., "Zion National Park").
    2) state_or_region: The state name(s) or region(s) explicitly mentioned for the park (e.g., "Utah", "Southern Utah").
    3) nps_unit_url: The official National Park Service URL for the park (e.g., "https://www.nps.gov/zion/index.htm"). If absent, return null.
    4) travel_recommendation_urls: URLs for reliable travel sources that explicitly recommend this park for visits in December.
       - Extract only URLs explicitly present in the answer (plain or markdown links).
       - If none provided, return an empty list.
    5) december_reasons: A list of specific reasons the answer gives for visiting in December (e.g., "fewer crowds", "milder weather", "open scenic drives").
       - Extract as short phrases or clauses as they appear in the answer.
    6) dec_temp_range: The December temperature range mentioned (e.g., "30–50°F"); if not provided, return null.
    7) dec_weather_summary: The general December weather conditions mentioned (e.g., "dry with occasional snow"); if not provided, return null.
    8) weather_urls: URLs from reputable weather/climate sources supporting December temperatures/weather (e.g., NOAA, Weather.gov, WeatherSpark).
       - Extract only URLs explicitly present in the answer. If none, return an empty list.
    9) dec_2025_ops_summary: A summary statement about winter road accessibility/operations for December 2025 as presented in the answer; if not provided, return null.
    10) nps_ops_urls: NPS URLs supporting winter road accessibility/operations (e.g., "conditions", "road closures", "plan your visit").
        - Extract only URLs explicitly present in the answer. If none, return an empty list.
    11) alerts_summary: A summary of current safety alerts/closures (or "none found" if the answer explicitly says no alerts); if not provided, return null.
    12) nps_alerts_urls: NPS alerts/conditions URLs supporting the alerts/closures statement.
        - Extract only URLs explicitly present in the answer. If none, return an empty list.

    IMPORTANT URL RULES:
    - Extract only actual URLs present in the answer. Do not invent or infer URLs.
    - For markdown links, return the underlying URL.
    - If a URL lacks a protocol, prepend "http://".
    - If a field is missing in the answer, return null (for single value) or an empty list (for arrays).

    Limit to the first 5 parks mentioned. If fewer than 4 are provided, extract whatever is present.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_state_tokens(text: Optional[str]) -> List[str]:
    """Extract and normalize state names from a free-form location string."""
    if not text:
        return []
    tokens = [t.strip() for t in text.replace(",", " ").replace("/", " ").split() if t.strip()]
    found_states = set()

    # Direct full-name matches
    for name in WEST_STATE_NAMES:
        if name.lower() in text.lower():
            found_states.add(name)

    # Abbreviation matches
    for abbr, fullname in WEST_STATE_ABBR.items():
        # Match exact token like "UT" or "(UT)"
        if any(tok.upper() == abbr for tok in tokens):
            found_states.add(fullname)

    # Heuristics for phrases like "southern Utah", "eastern Washington"
    for name in WEST_STATE_NAMES:
        lowered = text.lower()
        if any(k in lowered for k in [f"southern {name.lower()}", f"northern {name.lower()}",
                                      f"eastern {name.lower()}", f"western {name.lower()}"]):
            found_states.add(name)

    return sorted(found_states)


def _combine_sources(*lists: List[str]) -> List[str]:
    """Combine lists of URLs and de-duplicate while preserving order."""
    seen = set()
    combined = []
    for lst in lists:
        for url in lst or []:
            if not url:
                continue
            if url not in seen:
                combined.append(url)
                seen.add(url)
    return combined


def _park_node_passed_all_critical(park_node: VerificationNode) -> bool:
    """Check whether all critical leaf children of a per-park node have passed."""
    if not park_node.children:
        return False
    critical_leaves = [c for c in park_node.children if c.critical and not c.children]
    if not critical_leaves:
        return False
    return all(c.status == "passed" and c.score == 1.0 for c in critical_leaves)


# --------------------------------------------------------------------------- #
# Verification for a single park                                              #
# --------------------------------------------------------------------------- #
async def verify_single_park(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    park: ParkItem,
    park_index: int,
) -> VerificationNode:
    """
    Build verification subtree for a single park item and run all verifications.
    """
    # Determine node id and description based on index (Park_5 is optional)
    is_optional = (park_index == 4)
    node_id = f"Park_{5}_optional" if is_optional else f"Park_{park_index + 1}"
    node_desc = (
        "Optional fifth park (only evaluated if provided; used for partial credit without exceeding the 5-item limit)."
        if is_optional else
        ["First proposed park (evaluated as one candidate item).",
         "Second proposed park (evaluated as one candidate item).",
         "Third proposed park (evaluated as one candidate item).",
         "Fourth proposed park (evaluated as one candidate item)."][park_index]
    )

    park_node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=False  # per-park node is non-critical; its children are critical
    )

    park_name = park.park_name or "the park"
    state_tokens = _normalize_state_tokens(park.state_or_region)
    state_display = ", ".join(state_tokens) if state_tokens else (park.state_or_region or "unknown location")

    # 1) Is_National_Park
    node_np = evaluator.add_leaf(
        id=f"{node_id}_Is_National_Park",
        desc="Park is an NPS-designated National Park.",
        parent=park_node,
        critical=True,
    )
    claim_np = f"The National Park Service page indicates that {park_name} is a National Park or National Park & Preserve."
    await evaluator.verify(
        claim=claim_np,
        node=node_np,
        sources=park.nps_unit_url,
        additional_instruction=(
            "Use the NPS official unit page. Pass only if the designation is 'National Park' or 'National Park & Preserve'. "
            "If the unit is a Monument, Recreation Area, Seashore, etc., fail."
        ),
    )

    # 2) Is_Located_In_Western_US
    node_west = evaluator.add_leaf(
        id=f"{node_id}_Is_Located_In_Western_US",
        desc="Park is located in the western United States.",
        parent=park_node,
        critical=True,
    )
    allowed_states_str = ", ".join(sorted(WEST_STATES_ALL))
    claim_west = f"{park_name} is located in {state_display}, which is in the western United States."
    await evaluator.verify(
        claim=claim_west,
        node=node_west,
        sources=park.nps_unit_url,
        additional_instruction=(
            f"Confirm via the NPS page that the park's state(s) are among: {allowed_states_str}. "
            "Multi-state parks count if any listed state is within this set. If not within these, fail."
        ),
    )

    # 3) Explicit_December_Recommendation_Source
    node_rec = evaluator.add_leaf(
        id=f"{node_id}_Explicit_December_Recommendation_Source",
        desc="A reliable travel source explicitly recommends this park for December visits (URL provided).",
        parent=park_node,
        critical=True,
    )
    claim_rec = f"At least one cited travel source explicitly recommends visiting {park_name} in December."
    await evaluator.verify(
        claim=claim_rec,
        node=node_rec,
        sources=park.travel_recommendation_urls,
        additional_instruction=(
            "Check the provided travel source(s). Pass only if the page clearly recommends December (specifically) for this park. "
            "If URLs are missing or the page only suggests generic winter without mentioning December, fail."
        ),
    )

    # 4) December_Visit_Reasons
    node_reasons = evaluator.add_leaf(
        id=f"{node_id}_December_Visit_Reasons",
        desc="Provides specific reasons December is a good time to visit (e.g., weather, crowds, activities, accessibility), consistent with cited sources.",
        parent=park_node,
        critical=True,
    )
    reasons_text = "; ".join(park.december_reasons) if park.december_reasons else "No specific reasons provided"
    claim_reasons = f"The cited source(s) support these December visit reasons for {park_name}: {reasons_text}."
    combined_reason_sources = _combine_sources(park.travel_recommendation_urls, [park.nps_unit_url] if park.nps_unit_url else [])
    await evaluator.verify(
        claim=claim_reasons,
        node=node_reasons,
        sources=combined_reason_sources,
        additional_instruction=(
            "Look for the stated reasons on the source pages (lower crowds, mild weather, specific seasonal activities, accessibility, etc.). "
            "Pass only if at least one source explicitly supports the listed reasons. If no URLs are provided, fail."
        ),
    )

    # 5) December_Temperatures_And_Weather
    node_weather = evaluator.add_leaf(
        id=f"{node_id}_December_Temperatures_And_Weather",
        desc="Provides December temperature ranges AND general December weather conditions, supported by a reputable weather/climate source (URL provided).",
        parent=park_node,
        critical=True,
    )
    temp_text = park.dec_temp_range or "no temperature range provided"
    wx_text = park.dec_weather_summary or "no weather summary provided"
    claim_weather = f"In December, typical temperatures at or near {park_name} are {temp_text}, and weather conditions are {wx_text}."
    await evaluator.verify(
        claim=claim_weather,
        node=node_weather,
        sources=park.weather_urls,
        additional_instruction=(
            "Use reputable weather/climate sources (e.g., NOAA, Weather.gov, WeatherSpark, Climate-Data). "
            "Pass only if BOTH the temperature range and the general weather conditions are supported. "
            "Gateway towns are acceptable if clearly applicable to the park's area. If no weather URLs are provided, fail."
        ),
    )

    # 6) Dec_2025_Road_Access_And_Operations
    node_ops = evaluator.add_leaf(
        id=f"{node_id}_Dec_2025_Road_Access_And_Operations",
        desc="Provides winter road accessibility and operational status information for December 2025, supported by an official NPS source (URL provided).",
        parent=park_node,
        critical=True,
    )
    ops_text = park.dec_2025_ops_summary or "no December 2025 operations summary provided"
    claim_ops = f"For December 2025 at {park_name}, winter road accessibility and operational status is: {ops_text}."
    await evaluator.verify(
        claim=claim_ops,
        node=node_ops,
        sources=park.nps_ops_urls,
        additional_instruction=(
            "Verify using NPS 'conditions', 'alerts', 'road closures', or official 'plan your visit' pages. "
            "Seasonal closures that apply annually can support December 2025 conditions if clearly stated for winter months. "
            "If no NPS operations URLs are provided, fail."
        ),
    )

    # 7) Safety_Alerts_Or_Closures
    node_alerts = evaluator.add_leaf(
        id=f"{node_id}_Safety_Alerts_Or_Closures",
        desc="Documents any current safety alerts/closures/hazards that may affect December visitors (or explicitly states none found), supported by NPS alerts/conditions source(s) (URL provided).",
        parent=park_node,
        critical=True,
    )
    alerts_text = park.alerts_summary or "no alerts summary provided"
    claim_alerts = f"Current safety alerts or closures relevant to December visitors at {park_name}: {alerts_text}."
    alerts_sources = _combine_sources(park.nps_alerts_urls, park.nps_ops_urls)  # prefer alerts URLs; ops pages can also list closures
    await evaluator.verify(
        claim=claim_alerts,
        node=node_alerts,
        sources=alerts_sources,
        additional_instruction=(
            "Check NPS alerts/conditions pages. If the answer claims 'none', confirm the page indicates no active alerts/closures. "
            "If URLs are missing or the pages contradict the claim, fail."
        ),
    )

    return park_node


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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Western US National Parks December recommendations task.
    """
    # Initialize evaluator
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

    # Extract parks data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction",
    )

    # Build main task node (non-critical, with critical gating children added later)
    main_node = evaluator.add_parallel(
        id="December_National_Parks_Western_US",
        desc="Identify ≥4 western U.S. National Parks recommended for December visits and provide required December-specific details with appropriate sources.",
        parent=root,
        critical=False
    )

    # Limit to first 5 parks
    parks = (extraction.parks or [])[:5]
    # If fewer than 5 extracted, pad with empty ParkItem to stabilize tree
    while len(parks) < 5:
        parks.append(ParkItem())

    # Verify each park (Park_1..Park_4, Park_5_optional)
    per_park_nodes: List[VerificationNode] = []
    for i, park in enumerate(parks):
        node = await verify_single_park(evaluator, main_node, park, i)
        per_park_nodes.append(node)

    # Compute how many parks fully satisfied all their per-park critical checks
    good_park_indices = [idx for idx, n in enumerate(per_park_nodes[:5]) if _park_node_passed_all_critical(n)]
    good_count = len(good_park_indices)

    # Critical gate: Minimum_Park_Count (≥4 fully passing parks)
    min_count_node = evaluator.add_custom_node(
        result=(good_count >= 4),
        id="Minimum_Park_Count",
        desc="At least four provided park items satisfy all of their per-park critical checks.",
        parent=main_node,
        critical=True
    )

    # Compute geographic diversity among the fully passing parks
    def _states_for_index(idx: int) -> List[str]:
        item = parks[idx]
        states = _normalize_state_tokens(item.state_or_region)
        # If not found in text, try scraping from URLs indirectly (not available here), so just return parsed states
        return states

    distinct_states = set()
    for idx in good_park_indices:
        for st in _states_for_index(idx):
            if st in WEST_STATES_ALL:
                distinct_states.add(st)

    # Heuristic: require ≥3 distinct states among the passing parks to demonstrate diversity
    geo_diverse = (len(distinct_states) >= 3 and good_count >= 4)

    geo_node = evaluator.add_custom_node(
        result=geo_diverse,
        id="Geographic_Diversity",
        desc="Selections demonstrate geographic diversity within the western United States (the response provides enough location context to verify diversity).",
        parent=main_node,
        critical=True
    )

    # Record custom info for transparency
    evaluator.add_custom_info(
        info={
            "total_parks_evaluated": len(parks),
            "fully_passing_parks_indices": good_park_indices,
            "fully_passing_parks_count": good_count,
            "distinct_states_among_passing": sorted(list(distinct_states)),
            "geographic_diversity_passed": geo_diverse
        },
        info_type="metrics",
        info_name="park_evaluation_metrics"
    )

    # Return evaluation summary
    return evaluator.get_summary()