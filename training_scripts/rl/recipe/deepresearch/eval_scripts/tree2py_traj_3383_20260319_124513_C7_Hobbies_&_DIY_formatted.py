import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "beginners_holiday_hobby_guide"
TASK_DESCRIPTION = """
You are creating a comprehensive 'Beginner's Holiday Hobby Guide' for a local community center that will offer four different craft workshops: freshwater aquarium setup, needle felting, cookie decorating, and handmade card making. For each workshop, you must specify the essential material requirements and technical specifications appropriate for absolute beginners with no prior experience. Your guide must include:

For the Aquarium Setup Workshop:
1. The minimum tank size (in gallons) recommended for keeping fancy goldfish
2. The filter capacity guideline expressed as a multiple of tank volume
3. The optimal water temperature range (in Fahrenheit) for fancy goldfish

For the Needle Felting Workshop:
4. The three main needle gauge categories (coarse, medium, and fine) with their corresponding gauge numbers
5. The two primary types of wool roving needed: one for building structure/bulk and one for color/detail work
6. The expected project completion time range (in minutes) for small beginner-level projects

For the Cookie Decorating Workshop:
7. The two main royal icing consistency types used for decorating and their specific purposes
8. The typical yield (in dozens) expected from a standard cookie recipe batch
9. The recommended maximum storage duration (in weeks) for baked cookies at room temperature

For the Card Making Workshop:
10. The standard card size dimensions (in inches) when folded, known as A2 size
11. The most popular cardstock weight (in lb or GSM) recommended for card bases

Additionally, as bird watching may be offered as a supplementary outdoor activity:
12. The optimal binocular magnification range recommended for general bird watching

Provide all specifications with their supporting reference URLs from reliable sources.
"""

# Ground truth / expectations we will check compliance against (recorded for transparency)
EXPECTED_SPECS = {
    "aquarium": {
        "tank_min_gallons": "20–30 gallons recommended for a single fancy goldfish (at least 20 gallons minimum)",
        "filter_capacity_multiple": "3–4× tank volume (GPH) turnover",
        "temperature_f": "68–74°F optimal range",
    },
    "felting": {
        "needle_gauges": "36 (coarse), 38 (medium/all‑purpose), 40–42 (fine)",
        "wool_types": "Core wool (for structure/bulk) and Merino wool (for color/detail)",
        "project_time_minutes": "About 15–90 minutes for small beginner projects",
    },
    "cookie": {
        "icing": "Two consistencies: piping (thicker, holds shape) and flooding (thinner, ~7–15 second)",
        "yield_dozens": "About 2–4 dozen per standard batch",
        "storage_weeks": "Up to ~2–3 weeks at room temperature (airtight)",
    },
    "card": {
        "a2_size_inches": "4.25 × 5.5 inches when folded (A2)",
        "cardstock_weight": "80 lb (≈216 gsm) recommended/popular for card bases",
    },
    "birding": {
        "magnification": "7×–8× is optimal for general bird watching",
    }
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AquariumInfo(BaseModel):
    tank_min_gallons: Optional[str] = None
    tank_min_sources: List[str] = Field(default_factory=list)
    filter_capacity_multiple: Optional[str] = None
    filter_sources: List[str] = Field(default_factory=list)
    temperature_range_f: Optional[str] = None
    temperature_sources: List[str] = Field(default_factory=list)


class FeltingInfo(BaseModel):
    needle_coarse_gauge: Optional[str] = None
    needle_medium_gauge: Optional[str] = None
    needle_fine_gauge: Optional[str] = None
    needle_sources: List[str] = Field(default_factory=list)

    wool_core_type: Optional[str] = None
    wool_detail_type: Optional[str] = None
    wool_sources: List[str] = Field(default_factory=list)

    project_time_minutes: Optional[str] = None
    project_time_sources: List[str] = Field(default_factory=list)


class CookieInfo(BaseModel):
    piping_desc: Optional[str] = None
    flooding_desc: Optional[str] = None
    icing_sources: List[str] = Field(default_factory=list)

    yield_dozens: Optional[str] = None
    yield_sources: List[str] = Field(default_factory=list)

    storage_weeks: Optional[str] = None
    storage_sources: List[str] = Field(default_factory=list)


class CardInfo(BaseModel):
    a2_size_inches: Optional[str] = None
    size_sources: List[str] = Field(default_factory=list)

    cardstock_weight: Optional[str] = None
    cardstock_sources: List[str] = Field(default_factory=list)


class BirdingInfo(BaseModel):
    magnification_range: Optional[str] = None
    magnification_sources: List[str] = Field(default_factory=list)


class GuideExtraction(BaseModel):
    aquarium: Optional[AquariumInfo] = None
    felting: Optional[FeltingInfo] = None
    cookie: Optional[CookieInfo] = None
    card: Optional[CardInfo] = None
    birding: Optional[BirdingInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_guide() -> str:
    return """
Extract from the answer all requested beginner workshop specifications and the exact supporting URLs cited for each item. Return the results in the provided JSON schema.

For each field below, follow these rules:
- Copy the value text exactly as written in the answer (preserve units, ranges, and phrasing).
- For each "*_sources" field, extract ONLY the explicit URLs associated with that specific spec in the answer.
- Accept URLs written as plain links or markdown links [text](url); always return the actual URL.
- If multiple URLs are provided for one spec, include all of them in the list.
- If a required value is missing, set it to null; if no URL is given for that spec, return an empty list.

Fields to extract:

1) Aquarium (freshwater, fancy goldfish context)
- aquarium.tank_min_gallons: string (e.g., "20–30 gallons", "at least 20 gallons")
- aquarium.tank_min_sources: string[] (URLs only)
- aquarium.filter_capacity_multiple: string (e.g., "3–4× tank volume (GPH)")
- aquarium.filter_sources: string[] (URLs only)
- aquarium.temperature_range_f: string (e.g., "68–74°F")
- aquarium.temperature_sources: string[] (URLs only)

2) Needle Felting
- felting.needle_coarse_gauge: string (e.g., "36")
- felting.needle_medium_gauge: string (e.g., "38")
- felting.needle_fine_gauge: string (e.g., "40–42")
- felting.needle_sources: string[] (URLs only)
- felting.wool_core_type: string (e.g., "core wool", "batting")
- felting.wool_detail_type: string (e.g., "merino wool")
- felting.wool_sources: string[] (URLs only)
- felting.project_time_minutes: string (e.g., "15–90 minutes", "about 30–60 minutes")
- felting.project_time_sources: string[] (URLs only)

3) Cookie Decorating
- cookie.piping_desc: string (piping consistency purpose/description)
- cookie.flooding_desc: string (flooding consistency purpose/description, possibly with "~7–15 second" or similar)
- cookie.icing_sources: string[] (URLs only)
- cookie.yield_dozens: string (e.g., "2–4 dozen", "24–48 cookies")
- cookie.yield_sources: string[] (URLs only)
- cookie.storage_weeks: string (e.g., "2–3 weeks", "up to two weeks")
- cookie.storage_sources: string[] (URLs only)

4) Card Making
- card.a2_size_inches: string (e.g., "4.25 × 5.5 inches", "4-1/4 in × 5-1/2 in")
- card.size_sources: string[] (URLs only)
- card.cardstock_weight: string (e.g., "80 lb", "≈216 gsm")
- card.cardstock_sources: string[] (URLs only)

5) Bird Watching (supplement)
- birding.magnification_range: string (e.g., "7×–8×", "8x")
- birding.magnification_sources: string[] (URLs only)
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_value_and_sources(value: Optional[str], sources: Optional[List[str]]) -> bool:
    return bool(value and str(value).strip()) and bool(sources and len(sources) > 0)


def _list_or_empty(sources: Optional[List[str]]) -> List[str]:
    return sources if sources else []


# --------------------------------------------------------------------------- #
# Per-spec verification builders                                              #
# Each spec node is a sequential critical aggregator with 3 steps:
#   1) existence + sources (custom binary)
#   2) compliance with expected requirement (simple verify)
#   3) source support for the claim (verify by urls)
# --------------------------------------------------------------------------- #
async def build_aquarium_tank_size(e: Evaluator, parent, data: GuideExtraction):
    node = e.add_sequential(
        id="Aquarium_Tank_Size",
        desc="The guide specifies a minimum tank size of 20-30 gallons for fancy goldfish",
        parent=parent,
        critical=True,
    )
    val = (data.aquarium.tank_min_gallons if data and data.aquarium else None)
    srcs = _list_or_empty(data.aquarium.tank_min_sources if data and data.aquarium else [])
    e.add_custom_node(
        result=_has_value_and_sources(val, srcs),
        id="Aquarium_Tank_Size_exists",
        desc="Aquarium tank size value and sources are provided",
        parent=node,
        critical=True,
    )
    comp_leaf = e.add_leaf(
        id="Aquarium_Tank_Size_compliance",
        desc="Tank size statement matches required '20–30 gallons (or at least 20 gallons)' for a single fancy goldfish",
        parent=node,
        critical=True,
    )
    comp_claim = f"The stated minimum tank size '{val}' is compliant with a beginner recommendation of 20–30 gallons (or at least 20 gallons) for a single fancy goldfish. Accept phrasing like '20 gallons minimum' or '20–30 gallons recommended'."
    await e.verify(claim=comp_claim, node=comp_leaf)

    support_leaf = e.add_leaf(
        id="Aquarium_Tank_Size_source",
        desc="Cited sources support the tank size recommendation",
        parent=node,
        critical=True,
    )
    support_claim = "For a single fancy goldfish, a beginner-appropriate minimum tank size is at least 20 gallons, and many guides recommend about 20–30 gallons."
    await e.verify(
        claim=support_claim,
        node=support_leaf,
        sources=srcs,
        additional_instruction="Accept sources that state 'at least 20 gallons' and/or recommend a 20–30 gallon tank for a single fancy goldfish."
    )


async def build_aquarium_filter_capacity(e: Evaluator, parent, data: GuideExtraction):
    node = e.add_sequential(
        id="Aquarium_Filter_Capacity",
        desc="The guide states that filter capacity should be 3-4 times the tank volume (GPH rating)",
        parent=parent,
        critical=True,
    )
    val = (data.aquarium.filter_capacity_multiple if data and data.aquarium else None)
    srcs = _list_or_empty(data.aquarium.filter_sources if data and data.aquarium else [])
    e.add_custom_node(
        result=_has_value_and_sources(val, srcs),
        id="Aquarium_Filter_Capacity_exists",
        desc="Filter capacity guideline and sources are provided",
        parent=node,
        critical=True,
    )
    comp_leaf = e.add_leaf(
        id="Aquarium_Filter_Capacity_compliance",
        desc="Filter capacity matches '3–4× tank volume (GPH)' guideline",
        parent=node,
        critical=True,
    )
    comp_claim = f"The stated filter guideline '{val}' indicates a turnover of roughly 3–4× the tank volume per hour (GPH). Treat 'turnover per hour' phrasing as equivalent."
    await e.verify(claim=comp_claim, node=comp_leaf)

    support_leaf = e.add_leaf(
        id="Aquarium_Filter_Capacity_source",
        desc="Cited sources support the 3–4× turnover guideline",
        parent=node,
        critical=True,
    )
    support_claim = "A filter capacity around 3–4 times the tank volume per hour (GPH) is appropriate guidance for goldfish beginners."
    await e.verify(
        claim=support_claim,
        node=support_leaf,
        sources=srcs,
        additional_instruction="Look for 'turnover per hour' or 'GPH relative to tank volume' statements that specifically indicate 3–4× (or an equivalent wording that includes this range)."
    )


async def build_aquarium_temperature(e: Evaluator, parent, data: GuideExtraction):
    node = e.add_sequential(
        id="Aquarium_Temperature_Range",
        desc="The guide provides the optimal water temperature range of 68-74°F for fancy goldfish",
        parent=parent,
        critical=True,
    )
    val = (data.aquarium.temperature_range_f if data and data.aquarium else None)
    srcs = _list_or_empty(data.aquarium.temperature_sources if data and data.aquarium else [])
    e.add_custom_node(
        result=_has_value_and_sources(val, srcs),
        id="Aquarium_Temperature_Range_exists",
        desc="Temperature range and sources are provided",
        parent=node,
        critical=True,
    )
    comp_leaf = e.add_leaf(
        id="Aquarium_Temperature_Range_compliance",
        desc="Temperature statement matches '68–74°F' optimal range",
        parent=node,
        critical=True,
    )
    comp_claim = f"The stated temperature '{val}' indicates an optimal range including 68–74°F (≈20–23°C) for fancy goldfish."
    await e.verify(claim=comp_claim, node=comp_leaf)

    support_leaf = e.add_leaf(
        id="Aquarium_Temperature_Range_source",
        desc="Cited sources support the 68–74°F range",
        parent=node,
        critical=True,
    )
    support_claim = "The optimal water temperature range for fancy goldfish is about 68–74°F (roughly 20–23°C)."
    await e.verify(
        claim=support_claim,
        node=support_leaf,
        sources=srcs,
        additional_instruction="Allow close variants that explicitly include the 68–74°F span."
    )


async def build_felting_needle_gauges(e: Evaluator, parent, data: GuideExtraction):
    node = e.add_sequential(
        id="Felting_Needle_Gauges",
        desc="The guide identifies three needle gauge categories: 36 (coarse), 38 (medium/all-purpose), and 40-42 (fine)",
        parent=parent,
        critical=True,
    )
    coarse = (data.felting.needle_coarse_gauge if data and data.felting else None)
    medium = (data.felting.needle_medium_gauge if data and data.felting else None)
    fine = (data.felting.needle_fine_gauge if data and data.felting else None)
    srcs = _list_or_empty(data.felting.needle_sources if data and data.felting else [])
    e.add_custom_node(
        result=_has_value_and_sources(coarse, srcs) and bool(medium) and bool(fine),
        id="Felting_Needle_Gauges_exists",
        desc="Needle gauge categories and sources are provided",
        parent=node,
        critical=True,
    )
    comp_leaf = e.add_leaf(
        id="Felting_Needle_Gauges_compliance",
        desc="Needle gauges match 36 (coarse), 38 (medium), 40–42 (fine)",
        parent=node,
        critical=True,
    )
    comp_claim = f"The stated gauges (coarse: '{coarse}', medium: '{medium}', fine: '{fine}') align with 36 (coarse), 38 (medium/all-purpose), and 40–42 (fine). Permit equivalent phrasing and ranges containing these numbers."
    await e.verify(claim=comp_claim, node=comp_leaf)

    support_leaf = e.add_leaf(
        id="Felting_Needle_Gauges_source",
        desc="Cited sources support the gauge mapping",
        parent=node,
        critical=True,
    )
    support_claim = "Common needle felting sizes are 36 (coarse), 38 (medium/all‑purpose), and 40–42 (fine)."
    await e.verify(
        claim=support_claim,
        node=support_leaf,
        sources=srcs,
        additional_instruction="Accept sources that explicitly map these gauge numbers to coarse/medium/fine usage."
    )


async def build_felting_wool_types(e: Evaluator, parent, data: GuideExtraction):
    node = e.add_sequential(
        id="Felting_Wool_Types",
        desc="The guide specifies two primary wool types: core wool for structure/bulk and merino wool for color/detail work",
        parent=parent,
        critical=True,
    )
    core = (data.felting.wool_core_type if data and data.felting else None)
    detail = (data.felting.wool_detail_type if data and data.felting else None)
    srcs = _list_or_empty(data.felting.wool_sources if data and data.felting else [])
    e.add_custom_node(
        result=_has_value_and_sources(core, srcs) and bool(detail),
        id="Felting_Wool_Types_exists",
        desc="Wool types and sources are provided",
        parent=node,
        critical=True,
    )
    comp_leaf = e.add_leaf(
        id="Felting_Wool_Types_compliance",
        desc="Wool types match 'core for structure' and 'merino for color/detail'",
        parent=node,
        critical=True,
    )
    comp_claim = f"The stated wool choices (structure/bulk: '{core}', color/detail: '{detail}') correspond to 'core wool' (e.g., batting) for structure and 'merino wool' for color/detail. Accept synonyms like 'batting' for core."
    await e.verify(claim=comp_claim, node=comp_leaf)

    support_leaf = e.add_leaf(
        id="Felting_Wool_Types_source",
        desc="Cited sources support the core/merino usage",
        parent=node,
        critical=True,
    )
    support_claim = "For needle felting, core wool (e.g., batting) is used to build structure/bulk, while merino wool top/roving is commonly used for color/detail layers."
    await e.verify(
        claim=support_claim,
        node=support_leaf,
        sources=srcs,
        additional_instruction="Look for explicit guidance distinguishing 'core' wool for building shape vs. merino/top for surface detail/color."
    )


async def build_felting_project_time(e: Evaluator, parent, data: GuideExtraction):
    node = e.add_sequential(
        id="Felting_Project_Time",
        desc="The guide indicates that small beginner projects take 15-90 minutes to complete",
        parent=parent,
        critical=True,
    )
    val = (data.felting.project_time_minutes if data and data.felting else None)
    srcs = _list_or_empty(data.felting.project_time_sources if data and data.felting else [])
    e.add_custom_node(
        result=_has_value_and_sources(val, srcs),
        id="Felting_Project_Time_exists",
        desc="Project time range and sources are provided",
        parent=node,
        critical=True,
    )
    comp_leaf = e.add_leaf(
        id="Felting_Project_Time_compliance",
        desc="Project time matches 'about 15–90 minutes' for small beginner projects",
        parent=node,
        critical=True,
    )
    comp_claim = f"The stated time '{val}' is consistent with a small beginner needle-felting project typically taking about 15–90 minutes."
    await e.verify(claim=comp_claim, node=comp_leaf)

    support_leaf = e.add_leaf(
        id="Felting_Project_Time_source",
        desc="Cited sources support the 15–90 minute range",
        parent=node,
        critical=True,
    )
    support_claim = "Small, beginner-friendly needle felting projects commonly take roughly 15–90 minutes to complete."
    await e.verify(
        claim=support_claim,
        node=support_leaf,
        sources=srcs,
        additional_instruction="Accept sources that provide a range falling within or clearly overlapping ~15–90 minutes for simple beginner projects."
    )


async def build_cookie_icing(e: Evaluator, parent, data: GuideExtraction):
    node = e.add_sequential(
        id="Cookie_Icing_Consistencies",
        desc="The guide describes two royal icing consistency types: piping (thicker, holds shape) and flooding (thinner, 7-15 second settling)",
        parent=parent,
        critical=True,
    )
    piping = (data.cookie.piping_desc if data and data.cookie else None)
    flooding = (data.cookie.flooding_desc if data and data.cookie else None)
    srcs = _list_or_empty(data.cookie.icing_sources if data and data.cookie else [])
    e.add_custom_node(
        result=_has_value_and_sources(piping, srcs) and bool(flooding),
        id="Cookie_Icing_Consistencies_exists",
        desc="Royal icing consistency descriptions and sources are provided",
        parent=node,
        critical=True,
    )
    comp_leaf = e.add_leaf(
        id="Cookie_Icing_Consistencies_compliance",
        desc="Two consistencies correctly described: piping (thick/holds shape) and flooding (~7–15s thin)",
        parent=node,
        critical=True,
    )
    comp_claim = f"The provided descriptions (piping: '{piping}', flooding: '{flooding}') align with two main royal icing consistencies: piping (thicker, holds shape/outline) and flooding (thinner, levels in ~7–15 seconds)."
    await e.verify(claim=comp_claim, node=comp_leaf)

    support_leaf = e.add_leaf(
        id="Cookie_Icing_Consistencies_source",
        desc="Cited sources support the piping vs flooding definitions",
        parent=node,
        critical=True,
    )
    support_claim = "For cookie decorating, royal icing commonly uses two consistencies: piping (thicker to hold outlines/details) and flooding (thinner to self-level in ~7–15 seconds)."
    await e.verify(
        claim=support_claim,
        node=support_leaf,
        sources=srcs,
        additional_instruction="Allow small timing variations around the ~7–15 second flooding benchmark."
    )


async def build_cookie_yield(e: Evaluator, parent, data: GuideExtraction):
    node = e.add_sequential(
        id="Cookie_Recipe_Yield",
        desc="The guide states that standard cookie recipes yield 2-4 dozen per batch",
        parent=parent,
        critical=True,
    )
    val = (data.cookie.yield_dozens if data and data.cookie else None)
    srcs = _list_or_empty(data.cookie.yield_sources if data and data.cookie else [])
    e.add_custom_node(
        result=_has_value_and_sources(val, srcs),
        id="Cookie_Recipe_Yield_exists",
        desc="Cookie yield and sources are provided",
        parent=node,
        critical=True,
    )
    comp_leaf = e.add_leaf(
        id="Cookie_Recipe_Yield_compliance",
        desc="Yield matches 'about 2–4 dozen' per standard batch",
        parent=node,
        critical=True,
    )
    comp_claim = f"The stated yield '{val}' is consistent with a standard cookie recipe producing about 2–4 dozen cookies per batch (depending on cutter size)."
    await e.verify(claim=comp_claim, node=comp_leaf)

    support_leaf = e.add_leaf(
        id="Cookie_Recipe_Yield_source",
        desc="Cited sources support the 2–4 dozen yield",
        parent=node,
        critical=True,
    )
    support_claim = "A typical roll-out cookie recipe yields roughly 2–4 dozen cookies per batch."
    await e.verify(
        claim=support_claim,
        node=support_leaf,
        sources=srcs,
        additional_instruction="Accept sources indicating yields within about 24–48 cookies per batch."
    )


async def build_cookie_storage(e: Evaluator, parent, data: GuideExtraction):
    node = e.add_sequential(
        id="Cookie_Storage_Duration",
        desc="The guide provides the maximum storage duration of 2-3 weeks for baked cookies at room temperature",
        parent=parent,
        critical=True,
    )
    val = (data.cookie.storage_weeks if data and data.cookie else None)
    srcs = _list_or_empty(data.cookie.storage_sources if data and data.cookie else [])
    e.add_custom_node(
        result=_has_value_and_sources(val, srcs),
        id="Cookie_Storage_Duration_exists",
        desc="Cookie storage duration and sources are provided",
        parent=node,
        critical=True,
    )
    comp_leaf = e.add_leaf(
        id="Cookie_Storage_Duration_compliance",
        desc="Storage duration matches 'up to ~2–3 weeks' at room temp (airtight)",
        parent=node,
        critical=True,
    )
    comp_claim = f"The stated storage duration '{val}' indicates a maximum around 2–3 weeks at room temperature when stored airtight."
    await e.verify(claim=comp_claim, node=comp_leaf)

    support_leaf = e.add_leaf(
        id="Cookie_Storage_Duration_source",
        desc="Cited sources support the 2–3 week room-temp storage",
        parent=node,
        critical=True,
    )
    support_claim = "Baked cookies can keep at room temperature for up to about 2–3 weeks if stored in an airtight container."
    await e.verify(
        claim=support_claim,
        node=support_leaf,
        sources=srcs,
        additional_instruction="Look for general guidance on shelf life of baked cookies at room temperature."
    )


async def build_card_size(e: Evaluator, parent, data: GuideExtraction):
    node = e.add_sequential(
        id="Card_Size_Dimensions",
        desc="The guide specifies the A2 card size as 4.25 inches × 5.5 inches when folded",
        parent=parent,
        critical=True,
    )
    val = (data.card.a2_size_inches if data and data.card else None)
    srcs = _list_or_empty(data.card.size_sources if data and data.card else [])
    e.add_custom_node(
        result=_has_value_and_sources(val, srcs),
        id="Card_Size_Dimensions_exists",
        desc="A2 folded size and sources are provided",
        parent=node,
        critical=True,
    )
    comp_leaf = e.add_leaf(
        id="Card_Size_Dimensions_compliance",
        desc="A2 folded size equals 4.25 × 5.5 inches (allow 4-1/4 × 5-1/2 notation)",
        parent=node,
        critical=True,
    )
    comp_claim = f"The stated A2 card size '{val}' matches 4.25 × 5.5 inches when folded (equivalently 4-1/4 × 5-1/2 in)."
    await e.verify(claim=comp_claim, node=comp_leaf)

    support_leaf = e.add_leaf(
        id="Card_Size_Dimensions_source",
        desc="Cited sources support A2 = 4.25 × 5.5 in folded",
        parent=node,
        critical=True,
    )
    support_claim = "The A2 greeting card size (folded) is 4.25 × 5.5 inches (also written as 4-1/4 × 5-1/2 inches)."
    await e.verify(
        claim=support_claim,
        node=support_leaf,
        sources=srcs,
        additional_instruction="Accept reputable craft/paper size charts that list A2 folded dimensions."
    )


async def build_cardstock_weight(e: Evaluator, parent, data: GuideExtraction):
    node = e.add_sequential(
        id="Cardstock_Weight",
        desc="The guide recommends 80 lb (approximately 216 GSM) cardstock weight for card bases",
        parent=parent,
        critical=True,
    )
    val = (data.card.cardstock_weight if data and data.card else None)
    srcs = _list_or_empty(data.card.cardstock_sources if data and data.card else [])
    e.add_custom_node(
        result=_has_value_and_sources(val, srcs),
        id="Cardstock_Weight_exists",
        desc="Cardstock weight recommendation and sources are provided",
        parent=node,
        critical=True,
    )
    comp_leaf = e.add_leaf(
        id="Cardstock_Weight_compliance",
        desc="Recommendation includes '80 lb' or ≈216 gsm as popular for card bases",
        parent=node,
        critical=True,
    )
    comp_claim = f"The stated cardstock weight '{val}' includes or is equivalent to 80 lb (approximately 216 gsm) as a popular/recommended base weight."
    await e.verify(claim=comp_claim, node=comp_leaf)

    support_leaf = e.add_leaf(
        id="Cardstock_Weight_source",
        desc="Cited sources support 80 lb (≈216 gsm) as a recommended/popular base weight",
        parent=node,
        critical=True,
    )
    support_claim = "For handmade card bases, 80 lb cover stock (approximately 216 gsm) is a popular and commonly recommended weight."
    await e.verify(
        claim=support_claim,
        node=support_leaf,
        sources=srcs,
        additional_instruction="Accept reputable crafting/paper sources that recommend ~80 lb (≈216 gsm) for card bases."
    )


async def build_birding_magnification(e: Evaluator, parent, data: GuideExtraction):
    node = e.add_sequential(
        id="Binocular_Magnification",
        desc="The guide recommends 7x-8x magnification as optimal for general bird watching",
        parent=parent,
        critical=True,
    )
    val = (data.birding.magnification_range if data and data.birding else None)
    srcs = _list_or_empty(data.birding.magnification_sources if data and data.birding else [])
    e.add_custom_node(
        result=_has_value_and_sources(val, srcs),
        id="Binocular_Magnification_exists",
        desc="Birding magnification and sources are provided",
        parent=node,
        critical=True,
    )
    comp_leaf = e.add_leaf(
        id="Binocular_Magnification_compliance",
        desc="Magnification matches '7×–8×' as optimal for general birdwatching",
        parent=node,
        critical=True,
    )
    comp_claim = f"The stated magnification '{val}' matches the commonly recommended 7×–8× range for general bird watching."
    await e.verify(claim=comp_claim, node=comp_leaf)

    support_leaf = e.add_leaf(
        id="Binocular_Magnification_source",
        desc="Cited sources support 7×–8× as optimal for general birding",
        parent=node,
        critical=True,
    )
    support_claim = "For general bird watching, 7× to 8× binocular magnification is widely recommended as optimal."
    await e.verify(
        claim=support_claim,
        node=support_leaf,
        sources=srcs,
        additional_instruction="Accept reputable birding optics guides (e.g., Audubon, Cornell, similar) that recommend 7×–8×."
    )


# Cookie remaining items: storage done, yield done, icing done.

# --------------------------------------------------------------------------- #
# Orchestration for building the full tree                                    #
# --------------------------------------------------------------------------- #
async def build_verification_tree(e: Evaluator, root, data: GuideExtraction):
    # Aquarium (3)
    await build_aquarium_tank_size(e, root, data)
    await build_aquarium_filter_capacity(e, root, data)
    await build_aquarium_temperature(e, root, data)

    # Felting (3)
    await build_felting_needle_gauges(e, root, data)
    await build_felting_wool_types(e, root, data)
    await build_felting_project_time(e, root, data)

    # Cookie (3)
    await build_cookie_icing(e, root, data)
    await build_cookie_yield(e, root, data)
    await build_cookie_storage(e, root, data)

    # Card making (2)
    await build_card_size(e, root, data)
    await build_cardstock_weight(e, root, data)

    # Birding (1)
    await build_birding_magnification(e, root, data)


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
    Evaluate an answer for the Beginner's Holiday Hobby Guide task.
    """
    # Initialize evaluator (root is parallel: each spec is independent)
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

    # Record expected specs (ground truth intent)
    evaluator.add_ground_truth(
        {"expected_requirements": EXPECTED_SPECS},
        gt_type="expected_specs"
    )

    # Extract structured information from the answer
    extracted: GuideExtraction = await evaluator.extract(
        prompt=prompt_extract_guide(),
        template_class=GuideExtraction,
        extraction_name="guide_extraction",
    )

    # Build the verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # Return the summary
    return evaluator.get_summary()