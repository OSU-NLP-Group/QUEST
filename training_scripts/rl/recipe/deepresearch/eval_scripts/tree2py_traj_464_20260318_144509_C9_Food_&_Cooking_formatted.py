import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "restaurant_chains_holiday_2024_2025"
TASK_DESCRIPTION = (
    "A food industry consultant is preparing a comprehensive analysis of restaurant chain operations during the 2024-2025 holiday season, "
    "with particular focus on operational resilience and geographic distribution patterns across the United States. Identify four distinct "
    "American restaurant chains, where each chain meets the specific criteria outlined below. For each chain, provide the chain name, total US "
    "location count (as of December 2024), and supporting evidence.\n\n"
    "Chain A Criteria:\n"
    "- Operates primarily in 25 or fewer US states\n"
    "- Has its highest concentration of locations in a southeastern US state\n"
    "- Has 400 or more locations in that southeastern state\n"
    "- Is known for 24/7 operations and a policy of rarely closing (even during emergencies)\n\n"
    "Chain B Criteria:\n"
    "- Has 300 or more locations in California\n"
    "- Serves breakfast items as a significant part of its menu (such as all-day breakfast)\n"
    "- Had most of its locations open on Christmas Day 2024\n\n"
    "Chain C Criteria:\n"
    "- Has 10,000 or more total locations across the United States\n"
    "- Is a fast food or quick service restaurant\n"
    "- Had most of its locations open on Christmas Day 2024\n\n"
    "Chain D Criteria:\n"
    "- Has 100 or more locations in Texas\n"
    "- Serves traditional American comfort food in a sit-down restaurant format\n"
    "- Was closed on Christmas Day 2024 at all or most locations\n\n"
    "For each of the four chains, provide:\n"
    "1. The chain name\n"
    "2. The total number of US locations as of December 2024\n"
    "3. For Chain A: the southeastern state with the most locations and the exact count in that state\n"
    "4. For Chain B: the exact number of California locations\n"
    "5. For Chain C: confirmation of the restaurant type (fast food/quick service)\n"
    "6. For Chain D: the exact number of Texas locations\n"
    "7. URL references that verify each of the above claims"
)

SOUTHEAST_STATES_ABBR = {
    "AL", "AR", "FL", "GA", "KY", "LA", "MS", "NC", "SC", "TN", "VA", "WV"
}
SOUTHEAST_STATES_FULL = {
    "ALABAMA", "ARKANSAS", "FLORIDA", "GEORGIA", "KENTUCKY", "LOUISIANA", "MISSISSIPPI",
    "NORTH CAROLINA", "SOUTH CAROLINA", "TENNESSEE", "VIRGINIA", "WEST VIRGINIA"
}


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class ChainA(BaseModel):
    name: Optional[str] = None
    existence_urls: List[str] = Field(default_factory=list)

    total_us_locations: Optional[str] = None
    total_locations_urls: List[str] = Field(default_factory=list)

    operating_states_count: Optional[str] = None
    operating_states_urls: List[str] = Field(default_factory=list)

    primary_state: Optional[str] = None
    primary_state_urls: List[str] = Field(default_factory=list)

    primary_state_location_count: Optional[str] = None
    primary_state_count_urls: List[str] = Field(default_factory=list)

    ops_247_urls: List[str] = Field(default_factory=list)


class ChainB(BaseModel):
    name: Optional[str] = None
    existence_urls: List[str] = Field(default_factory=list)

    total_us_locations: Optional[str] = None
    total_locations_urls: List[str] = Field(default_factory=list)

    california_locations: Optional[str] = None
    california_urls: List[str] = Field(default_factory=list)

    breakfast_menu_urls: List[str] = Field(default_factory=list)

    christmas_2024_urls: List[str] = Field(default_factory=list)


class ChainC(BaseModel):
    name: Optional[str] = None
    existence_urls: List[str] = Field(default_factory=list)

    total_us_locations: Optional[str] = None
    total_locations_urls: List[str] = Field(default_factory=list)

    restaurant_type: Optional[str] = None
    type_urls: List[str] = Field(default_factory=list)

    christmas_2024_urls: List[str] = Field(default_factory=list)


class ChainD(BaseModel):
    name: Optional[str] = None
    existence_urls: List[str] = Field(default_factory=list)

    total_us_locations: Optional[str] = None
    total_locations_urls: List[str] = Field(default_factory=list)

    texas_locations: Optional[str] = None
    texas_urls: List[str] = Field(default_factory=list)

    restaurant_style: Optional[str] = None
    style_urls: List[str] = Field(default_factory=list)

    christmas_2024_urls: List[str] = Field(default_factory=list)


class ChainsExtraction(BaseModel):
    chain_a: Optional[ChainA] = None
    chain_b: Optional[ChainB] = None
    chain_c: Optional[ChainC] = None
    chain_d: Optional[ChainD] = None


# -----------------------------------------------------------------------------
# Extraction prompt builder
# -----------------------------------------------------------------------------
def prompt_extract_chains() -> str:
    return """
You must extract structured information for four distinct U.S. restaurant chains labeled Chain A, Chain B, Chain C, and Chain D, exactly as stated in the answer. Return null for any field that is not present in the answer. Extract only URLs explicitly provided in the answer (plain URL or markdown link).

For each chain, extract:

Chain A (≤25 states; SE primary state with ≥400 locations; 24/7 rarely closes):
- name: The chain name
- existence_urls: URLs that confirm this is a real U.S. restaurant chain (brand/company site, Wikipedia, reputable articles)
- total_us_locations: The total number of U.S. locations as of December 2024 (string as written, e.g., "2,900+" or "about 2,800")
- total_locations_urls: URLs supporting the total U.S. locations
- operating_states_count: The number of U.S. states where the chain operates (string exactly as written)
- operating_states_urls: URLs supporting the number of states
- primary_state: The southeastern U.S. state with the most locations (as written; can be full name or abbreviation)
- primary_state_urls: URLs that indicate this is the state with the most locations
- primary_state_location_count: The number of locations in that primary state (string as written)
- primary_state_count_urls: URLs supporting the primary state location count
- ops_247_urls: URLs indicating 24/7 operations and/or a policy of rarely closing (e.g., Waffle House Index, policy pages, press)

Chain B (≥300 in California; breakfast significant; open Christmas Day 2024 at most locations):
- name
- existence_urls
- total_us_locations
- total_locations_urls
- california_locations: California location count (string as written)
- california_urls: URLs supporting the California location count
- breakfast_menu_urls: URLs that show breakfast is a significant part of the menu (e.g., all-day breakfast, dedicated breakfast menu)
- christmas_2024_urls: URLs about Christmas Day 2024 hours showing most locations were open

Chain C (≥10,000 U.S. locations; fast food/quick service; open Christmas Day 2024 at most locations):
- name
- existence_urls
- total_us_locations
- total_locations_urls
- restaurant_type: Type description as written (e.g., "fast food", "quick service", "QSR")
- type_urls: URLs confirming fast food/quick service classification
- christmas_2024_urls: URLs about Christmas Day 2024 hours showing most locations were open

Chain D (≥100 in Texas; sit-down American comfort food; closed Christmas Day 2024 at most or all locations):
- name
- existence_urls
- total_us_locations
- total_locations_urls
- texas_locations: Texas location count (string as written)
- texas_urls: URLs supporting the Texas location count
- restaurant_style: Type/cuisine description as written (e.g., "sit-down", "traditional American comfort food")
- style_urls: URLs confirming sit-down American comfort food positioning
- christmas_2024_urls: URLs about Christmas Day 2024 hours showing the chain was closed (most or all locations)

Return a single JSON object with keys: chain_a, chain_b, chain_c, chain_d (each an object with the fields listed above).
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _first_int_or_none(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    import re
    m = re.search(r"\d[\d,]*", text)
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except Exception:
        return None


def _is_southeastern(state_text: Optional[str]) -> bool:
    if not state_text:
        return False
    s = state_text.strip().upper()
    # Normalize common punctuation
    s = s.replace(".", "")
    if s in SOUTHEAST_STATES_ABBR or s in SOUTHEAST_STATES_FULL:
        return True
    # Handle alternate forms like "North Carolina (NC)"
    for abbr in SOUTHEAST_STATES_ABBR:
        if f"({abbr})" in s:
            return True
    for full in SOUTHEAST_STATES_FULL:
        if full in s:
            return True
    return False


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def _merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for ul in url_lists:
        for u in ul:
            if u and u not in seen:
                seen.add(u)
                result.append(u)
    return result


NUMERIC_TOLERANCE_INSTRUCTION = (
    "When checking counts, allow small rounding or formatting differences (e.g., 'about 400', '400+', 'approximately 300'). "
    "Focus on whether the threshold condition is satisfied (e.g., ≥ 300, ≥ 400, ≥ 10,000) rather than exact formatting. "
    "Prefer evidence dated near December 2024; if not exactly dated, accept the closest available 2024 data."
)

CHRISTMAS_2024_INSTRUCTION = (
    "Verify specifically for Christmas Day 2024. If a corporate page or credible news confirms most or all locations were open/closed on that date, "
    "accept it even if phrased as 'many locations' or 'most locations', consistent with the claim. Ignore unrelated years."
)


# -----------------------------------------------------------------------------
# Verification builders per chain
# -----------------------------------------------------------------------------
async def build_chain_distinctness(
    evaluator: Evaluator,
    parent,
    a: Optional[ChainA],
    b: Optional[ChainB],
    c: Optional[ChainC],
    d: Optional[ChainD],
):
    node = evaluator.add_sequential(
        id="Chain_Distinctness",
        desc="All four identified chains are distinct (different brands, not duplicates)",
        parent=parent,
        critical=True,
    )

    # Leaf 1: Distinctness_Check (custom)
    names = [
        (a.name if a else None),
        (b.name if b else None),
        (c.name if c else None),
        (d.name if d else None),
    ]
    unique_nonempty = len([n for n in names if _non_empty(n)]) == len(set([n.strip().lower() for n in names if _non_empty(n)])) if all(
        _non_empty(n) for n in names
    ) else False
    evaluator.add_custom_node(
        result=unique_nonempty,
        id="Distinctness_Check",
        desc="Chain A, Chain B, Chain C, and Chain D are four different restaurant brands",
        parent=node,
        critical=True,
    )

    # Leaf 2: Distinctness_URL (verify – collective statement; requires sources)
    distinct_urls = _merge_urls(
        (a.existence_urls if a else []),
        (b.existence_urls if b else []),
        (c.existence_urls if c else []),
        (d.existence_urls if d else []),
    )
    leaf = evaluator.add_leaf(
        id="Distinctness_URL",
        desc="URL references confirm the four chains are separate brands",
        parent=node,
        critical=True,
    )
    claim = f"The following are four different U.S. restaurant brands: {a.name if a else 'Chain A'}, {b.name if b else 'Chain B'}, {c.name if c else 'Chain C'}, {d.name if d else 'Chain D'}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=distinct_urls,
        additional_instruction="The provided URLs should collectively confirm that the listed entities are distinct restaurant brands (not the same brand). Consider each source carefully; if no single page verifies all four together, but collectively they establish distinct brands, judge as supported.",
    )


async def verify_chain_a(evaluator: Evaluator, parent, data: ChainA):
    chain_node = evaluator.add_parallel(
        id="Chain_A",
        desc="Chain operating primarily in ≤25 states, with 400+ locations in a southeastern state, known for 24/7 operations",
        parent=parent,
        critical=False,
    )

    # Identification (critical, sequential)
    ident_node = evaluator.add_sequential(
        id="Chain_A_Identification",
        desc="Valid chain name provided that corresponds to a real US restaurant chain",
        parent=chain_node,
        critical=True,
    )
    # Name provided
    evaluator.add_custom_node(
        result=_non_empty(data.name),
        id="Chain_A_Name",
        desc="Chain name is provided",
        parent=ident_node,
        critical=True,
    )
    # Existence URL verification
    leaf = evaluator.add_leaf(
        id="Chain_A_Name_URL",
        desc="URL reference confirming chain existence",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} is a real U.S. restaurant chain brand with U.S. locations.",
        node=leaf,
        sources=data.existence_urls,
        additional_instruction="Accept official websites, Wikipedia, or credible trade press that clearly identify the entity as a U.S. restaurant chain.",
    )

    # Geographic (critical, parallel)
    geo_node = evaluator.add_parallel(
        id="Chain_A_Geographic",
        desc="Geographic distribution meets criteria: ≤25 states, 400+ locations in southeastern state",
        parent=chain_node,
        critical=True,
    )

    # State count (critical, sequential)
    state_count_node = evaluator.add_sequential(
        id="Chain_A_State_Count",
        desc="Chain operates in 25 or fewer US states",
        parent=geo_node,
        critical=True,
    )
    states_int = _first_int_or_none(data.operating_states_count)
    evaluator.add_custom_node(
        result=(states_int is not None and states_int <= 25),
        id="Chain_A_State_Count_Value",
        desc="Number of operating states provided and is ≤25",
        parent=state_count_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_A_State_Count_URL",
        desc="URL reference for state count",
        parent=state_count_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} operates in {data.operating_states_count} U.S. states (which is ≤ 25).",
        node=leaf,
        sources=data.operating_states_urls,
        additional_instruction=NUMERIC_TOLERANCE_INSTRUCTION,
    )

    # Primary state (critical, sequential)
    primary_state_node = evaluator.add_sequential(
        id="Chain_A_Primary_State",
        desc="Primary state (with most locations) is in southeastern US",
        parent=geo_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_is_southeastern(data.primary_state),
        id="Chain_A_Primary_State_Name",
        desc="State name provided and is in southeastern US (e.g., GA, FL, SC, NC, AL, TN, etc.)",
        parent=primary_state_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_A_Primary_State_URL",
        desc="URL reference for primary state identification",
        parent=primary_state_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.primary_state} is the U.S. state with the most {data.name} locations.",
        node=leaf,
        sources=data.primary_state_urls,
        additional_instruction="The source should clearly indicate that this state has the highest number of the chain's locations in the U.S.",
    )

    # Location count in primary state (critical, sequential)
    ps_count_node = evaluator.add_sequential(
        id="Chain_A_Location_Count",
        desc="Chain has 400+ locations in its primary state",
        parent=geo_node,
        critical=True,
    )
    ps_count_int = _first_int_or_none(data.primary_state_location_count)
    evaluator.add_custom_node(
        result=(ps_count_int is not None and ps_count_int >= 400),
        id="Chain_A_Location_Count_Value",
        desc="Location count in primary state provided and is ≥400",
        parent=ps_count_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_A_Location_Count_URL",
        desc="URL reference for location count in primary state",
        parent=ps_count_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"There are {data.primary_state_location_count} {data.name} locations in {data.primary_state} (≥ 400).",
        node=leaf,
        sources=data.primary_state_count_urls,
        additional_instruction=NUMERIC_TOLERANCE_INSTRUCTION,
    )

    # Operations 24/7 (critical, sequential)
    ops_node = evaluator.add_sequential(
        id="Chain_A_Operations",
        desc="Chain is known for 24/7 operations and rarely closing",
        parent=chain_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_urls(data.ops_247_urls),
        id="Chain_A_247_Evidence",
        desc="Evidence provided that chain operates 24/7 or has policy of rarely closing",
        parent=ops_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_A_247_URL",
        desc="URL reference for 24/7 operations policy",
        parent=ops_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} is known for 24/7 operations and a policy of rarely closing, including during emergencies.",
        node=leaf,
        sources=data.ops_247_urls,
        additional_instruction="Accept credible references such as official policy pages, widely cited 'rarely closes' references, and reputable coverage (e.g., 'Waffle House Index').",
    )

    # Total U.S. locations (critical, sequential)
    total_node = evaluator.add_sequential(
        id="Chain_A_Total_Locations",
        desc="Total US location count provided as requested",
        parent=chain_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(data.total_us_locations),
        id="Chain_A_Total_Count",
        desc="Total US location count stated",
        parent=total_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_A_Total_URL",
        desc="URL reference for total location count",
        parent=total_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} has {data.total_us_locations} locations in the United States as of December 2024.",
        node=leaf,
        sources=data.total_locations_urls,
        additional_instruction=NUMERIC_TOLERANCE_INSTRUCTION,
    )


async def verify_chain_b(evaluator: Evaluator, parent, data: ChainB):
    chain_node = evaluator.add_parallel(
        id="Chain_B",
        desc="Chain with 300+ California locations, serves breakfast items, open on Christmas Day 2024",
        parent=parent,
        critical=False,
    )

    # Identification
    ident_node = evaluator.add_sequential(
        id="Chain_B_Identification",
        desc="Valid chain name provided that corresponds to a real US restaurant chain",
        parent=chain_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(data.name),
        id="Chain_B_Name",
        desc="Chain name is provided",
        parent=ident_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_B_Name_URL",
        desc="URL reference confirming chain existence",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} is a real U.S. restaurant chain brand with U.S. locations.",
        node=leaf,
        sources=data.existence_urls,
        additional_instruction="Accept official websites, Wikipedia, or credible trade press that clearly identify the entity as a U.S. restaurant chain.",
    )

    # California count
    ca_node = evaluator.add_sequential(
        id="Chain_B_California",
        desc="Chain has 300+ locations in California",
        parent=chain_node,
        critical=True,
    )
    ca_int = _first_int_or_none(data.california_locations)
    evaluator.add_custom_node(
        result=(ca_int is not None and ca_int >= 300),
        id="Chain_B_CA_Count",
        desc="California location count provided and is ≥300",
        parent=ca_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_B_CA_URL",
        desc="URL reference for California location count",
        parent=ca_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"There are {data.california_locations} {data.name} locations in California (≥ 300).",
        node=leaf,
        sources=data.california_urls,
        additional_instruction=NUMERIC_TOLERANCE_INSTRUCTION,
    )

    # Breakfast menu significance
    menu_node = evaluator.add_sequential(
        id="Chain_B_Menu",
        desc="Chain serves breakfast items as significant part of menu",
        parent=chain_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_urls(data.breakfast_menu_urls),
        id="Chain_B_Breakfast_Evidence",
        desc="Evidence that chain offers breakfast items or all-day breakfast",
        parent=menu_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_B_Menu_URL",
        desc="URL reference for breakfast menu offerings",
        parent=menu_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} offers breakfast items as a significant part of its menu (e.g., all-day breakfast or a dedicated breakfast menu).",
        node=leaf,
        sources=data.breakfast_menu_urls,
        additional_instruction="Accept official menu pages or credible references that clearly establish breakfast as a key offering.",
    )

    # Christmas 2024 open
    xmas_node = evaluator.add_sequential(
        id="Chain_B_Christmas",
        desc="Most locations were open on Christmas Day 2024",
        parent=chain_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_urls(data.christmas_2024_urls),
        id="Chain_B_Christmas_Status",
        desc="Evidence that most/many locations were open on Christmas Day 2024",
        parent=xmas_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_B_Christmas_URL",
        desc="URL reference for Christmas Day 2024 hours",
        parent=xmas_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Most {data.name} locations were open on Christmas Day 2024.",
        node=leaf,
        sources=data.christmas_2024_urls,
        additional_instruction=CHRISTMAS_2024_INSTRUCTION,
    )

    # Total locations
    total_node = evaluator.add_sequential(
        id="Chain_B_Total_Locations",
        desc="Total US location count provided as requested",
        parent=chain_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(data.total_us_locations),
        id="Chain_B_Total_Count",
        desc="Total US location count stated",
        parent=total_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_B_Total_URL",
        desc="URL reference for total location count",
        parent=total_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} has {data.total_us_locations} locations in the United States as of December 2024.",
        node=leaf,
        sources=data.total_locations_urls,
        additional_instruction=NUMERIC_TOLERANCE_INSTRUCTION,
    )


async def verify_chain_c(evaluator: Evaluator, parent, data: ChainC):
    chain_node = evaluator.add_parallel(
        id="Chain_C",
        desc="Fast food chain with 10,000+ US locations, most open on Christmas Day 2024",
        parent=parent,
        critical=False,
    )

    # Identification
    ident_node = evaluator.add_sequential(
        id="Chain_C_Identification",
        desc="Valid chain name provided that corresponds to a real US restaurant chain",
        parent=chain_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(data.name),
        id="Chain_C_Name",
        desc="Chain name is provided",
        parent=ident_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_C_Name_URL",
        desc="URL reference confirming chain existence",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} is a real U.S. restaurant chain brand with U.S. locations.",
        node=leaf,
        sources=data.existence_urls,
        additional_instruction="Accept official websites, Wikipedia, or credible trade press that clearly identify the entity as a U.S. restaurant chain.",
    )

    # Size ≥ 10,000
    size_node = evaluator.add_sequential(
        id="Chain_C_Size",
        desc="Chain has 10,000+ total US locations",
        parent=chain_node,
        critical=True,
    )
    total_int = _first_int_or_none(data.total_us_locations)
    evaluator.add_custom_node(
        result=(total_int is not None and total_int >= 10_000),
        id="Chain_C_Location_Count",
        desc="Total US location count provided and is ≥10,000",
        parent=size_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_C_Size_URL",
        desc="URL reference for total location count",
        parent=size_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} has {data.total_us_locations} U.S. locations (≥ 10,000) as of December 2024.",
        node=leaf,
        sources=data.total_locations_urls,
        additional_instruction=NUMERIC_TOLERANCE_INSTRUCTION,
    )

    # Type fast food / quick service
    type_node = evaluator.add_sequential(
        id="Chain_C_Type",
        desc="Chain is fast food/quick service restaurant",
        parent=chain_node,
        critical=True,
    )
    # Evidence (custom from the extracted text label)
    type_text = (data.restaurant_type or "").lower()
    evaluator.add_custom_node(
        result=any(k in type_text for k in ["fast food", "quick service", "qsr"]),
        id="Chain_C_QSR_Evidence",
        desc="Evidence that chain is fast food or quick service",
        parent=type_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_C_Type_URL",
        desc="URL reference confirming restaurant type",
        parent=type_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} is a fast food or quick service (QSR) restaurant chain.",
        node=leaf,
        sources=(data.type_urls if _has_urls(data.type_urls) else data.existence_urls),
        additional_instruction="Accept classification from official descriptions, Wikipedia infobox/lead, or credible trade press explicitly calling it fast food or quick service.",
    )

    # Christmas 2024 open
    xmas_node = evaluator.add_sequential(
        id="Chain_C_Christmas",
        desc="Most locations were open on Christmas Day 2024",
        parent=chain_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_urls(data.christmas_2024_urls),
        id="Chain_C_Christmas_Status",
        desc="Evidence that most locations were open on Christmas Day 2024",
        parent=xmas_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_C_Christmas_URL",
        desc="URL reference for Christmas Day 2024 hours",
        parent=xmas_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Most {data.name} locations were open on Christmas Day 2024.",
        node=leaf,
        sources=data.christmas_2024_urls,
        additional_instruction=CHRISTMAS_2024_INSTRUCTION,
    )


async def verify_chain_d(evaluator: Evaluator, parent, data: ChainD):
    chain_node = evaluator.add_parallel(
        id="Chain_D",
        desc="Sit-down restaurant chain closed on Christmas Day 2024, with 100+ Texas locations",
        parent=parent,
        critical=False,
    )

    # Identification
    ident_node = evaluator.add_sequential(
        id="Chain_D_Identification",
        desc="Valid chain name provided that corresponds to a real US restaurant chain",
        parent=chain_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(data.name),
        id="Chain_D_Name",
        desc="Chain name is provided",
        parent=ident_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_D_Name_URL",
        desc="URL reference confirming chain existence",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} is a real U.S. restaurant chain brand with U.S. locations.",
        node=leaf,
        sources=data.existence_urls,
        additional_instruction="Accept official websites, Wikipedia, or credible trade press that clearly identify the entity as a U.S. restaurant chain.",
    )

    # Texas count
    tx_node = evaluator.add_sequential(
        id="Chain_D_Texas",
        desc="Chain has 100+ locations in Texas",
        parent=chain_node,
        critical=True,
    )
    tx_int = _first_int_or_none(data.texas_locations)
    evaluator.add_custom_node(
        result=(tx_int is not None and tx_int >= 100),
        id="Chain_D_TX_Count",
        desc="Texas location count provided and is ≥100",
        parent=tx_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_D_TX_URL",
        desc="URL reference for Texas location count",
        parent=tx_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"There are {data.texas_locations} {data.name} locations in Texas (≥ 100).",
        node=leaf,
        sources=data.texas_urls,
        additional_instruction=NUMERIC_TOLERANCE_INSTRUCTION,
    )

    # Style: sit-down, traditional American comfort food
    style_node = evaluator.add_sequential(
        id="Chain_D_Style",
        desc="Chain serves sit-down, traditional American cuisine",
        parent=chain_node,
        critical=True,
    )
    style_text = (data.restaurant_style or "").lower()
    evaluator.add_custom_node(
        result=any(k in style_text for k in ["sit-down", "table service", "full-service"]) and any(
            k in style_text for k in ["american", "comfort food", "home-style", "homestyle"]
        ),
        id="Chain_D_Cuisine_Evidence",
        desc="Evidence that chain offers traditional American comfort food in sit-down format",
        parent=style_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_D_Style_URL",
        desc="URL reference for restaurant style and cuisine",
        parent=style_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} is a sit-down (table-service) restaurant known for traditional American comfort food.",
        node=leaf,
        sources=(data.style_urls if _has_urls(data.style_urls) else data.existence_urls),
        additional_instruction="Accept official 'About', menu descriptions, Wikipedia, or credible press that clearly indicate sit-down/table service and American comfort food positioning.",
    )

    # Christmas 2024 closed
    xmas_node = evaluator.add_sequential(
        id="Chain_D_Christmas",
        desc="Chain was closed on Christmas Day 2024",
        parent=chain_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_urls(data.christmas_2024_urls),
        id="Chain_D_Closed_Status",
        desc="Evidence that chain was closed on Christmas Day 2024",
        parent=xmas_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_D_Christmas_URL",
        desc="URL reference confirming closure on Christmas Day 2024",
        parent=xmas_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} locations were closed on Christmas Day 2024 (at all or most locations).",
        node=leaf,
        sources=data.christmas_2024_urls,
        additional_instruction=CHRISTMAS_2024_INSTRUCTION,
    )

    # Total locations
    total_node = evaluator.add_sequential(
        id="Chain_D_Total_Locations",
        desc="Total US location count provided as requested",
        parent=chain_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(data.total_us_locations),
        id="Chain_D_Total_Count",
        desc="Total US location count stated",
        parent=total_node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="Chain_D_Total_URL",
        desc="URL reference for total location count",
        parent=total_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{data.name} has {data.total_us_locations} locations in the United States as of December 2024.",
        node=leaf,
        sources=data.total_locations_urls,
        additional_instruction=NUMERIC_TOLERANCE_INSTRUCTION,
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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

    # Extract structured information
    extracted: ChainsExtraction = await evaluator.extract(
        prompt=prompt_extract_chains(),
        template_class=ChainsExtraction,
        extraction_name="chains_extraction",
    )

    # Build verification tree according to rubric
    # 1) Chain distinctness (critical sequential under root)
    await build_chain_distinctness(
        evaluator,
        root,
        extracted.chain_a or ChainA(),
        extracted.chain_b or ChainB(),
        extracted.chain_c or ChainC(),
        extracted.chain_d or ChainD(),
    )

    # 2) Chain A subtree
    await verify_chain_a(evaluator, root, extracted.chain_a or ChainA())

    # 3) Chain B subtree
    await verify_chain_b(evaluator, root, extracted.chain_b or ChainB())

    # 4) Chain C subtree
    await verify_chain_c(evaluator, root, extracted.chain_c or ChainC())

    # 5) Chain D subtree
    await verify_chain_d(evaluator, root, extracted.chain_d or ChainD())

    # Add a small custom info block for transparency
    evaluator.add_custom_info(
        info={
            "southeast_states_abbr": sorted(list(SOUTHEAST_STATES_ABBR)),
            "southeast_states_full": sorted(list(SOUTHEAST_STATES_FULL)),
        },
        info_type="region_reference",
        info_name="southeast_region_reference",
    )

    return evaluator.get_summary()