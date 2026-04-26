import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_gaming_conventions_2026"
TASK_DESCRIPTION = """
Identify three major gaming conventions in the United States scheduled to take place in 2026 that would be suitable for an indie game development studio planning to exhibit their game. For each convention, you must provide:

1. The convention name
2. The exact dates (including year)
3. The venue name and city/state location
4. The venue's total exhibit/expo space size in square feet
5. The nearest major commercial airport and its distance from the venue
6. Available multi-day ticket or pass options and their prices
7. Confirmation that the convention offers opportunities for indie game developers or exhibitors to showcase games
8. Types of competitive gaming tournaments or esports competitions featured at the convention
9. Reference URLs supporting each piece of information

Requirements:
- Each convention must be located in the United States
- Each convention must be scheduled for 2026
- Each venue must have at least 400,000 square feet of exhibit or expo space
- Each convention must be within 15 miles of a major commercial airport
- Each convention must offer indie game developer or exhibitor showcase opportunities
- Each convention must offer 3-day or 4-day attendance passes or badges
- Each convention must feature competitive gaming tournaments or esports competitions
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PassOption(BaseModel):
    option_name: Optional[str] = None  # e.g., "3-Day Pass", "4-Day Badge", etc.
    duration: Optional[str] = None     # e.g., "3-day", "4-day", "3 days"
    price: Optional[str] = None        # keep as string to allow ranges like "$99–$129"
    urls: List[str] = Field(default_factory=list)


class ConventionExtraction(BaseModel):
    name: Optional[str] = None
    dates: Optional[str] = None                 # e.g., "June 12–15, 2026"
    city: Optional[str] = None                  # e.g., "Seattle"
    state: Optional[str] = None                 # e.g., "WA"
    basic_info_urls: List[str] = Field(default_factory=list)

    venue_name: Optional[str] = None
    venue_exhibit_space_sqft: Optional[str] = None  # keep textual, e.g., "1.2 million sq ft" or "1,200,000 sq ft"
    venue_urls: List[str] = Field(default_factory=list)

    nearest_airport: Optional[str] = None
    airport_distance_miles: Optional[str] = None    # keep textual, e.g., "8 miles", "10.5 mi"
    accessibility_urls: List[str] = Field(default_factory=list)

    pass_options: List[PassOption] = Field(default_factory=list)
    pricing_urls: List[str] = Field(default_factory=list)

    indie_opportunities: Optional[str] = None       # textual description if provided
    tournaments_esports: Optional[str] = None       # textual description if provided
    features_urls: List[str] = Field(default_factory=list)


class ConventionsExtraction(BaseModel):
    conventions: List[ConventionExtraction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conventions() -> str:
    return """
Extract up to 5 gaming conventions mentioned in the answer. For each convention, extract the following fields exactly as presented in the answer text:
- name: The convention's name.
- dates: The exact 2026 dates for the convention (include month, day(s), and year if provided).
- city: City where the convention takes place.
- state: U.S. state abbreviation or full name where the convention takes place.
- basic_info_urls: URLs that support the convention’s name, dates, and location (prefer the official event site if listed).

- venue_name: Name of the venue or convention center.
- venue_exhibit_space_sqft: The venue's total exhibit/expo space in square feet as stated (keep the exact text, e.g., "1,200,000 sq ft", "1.2 million square feet").
- venue_urls: URLs that support the venue name and exhibit/expo space size.

- nearest_airport: The nearest major commercial airport serving the venue.
- airport_distance_miles: The distance from the airport to the venue (keep the textual format, e.g., "8 miles", "12.3 mi").
- accessibility_urls: URLs that support the airport identification and/or distance to the venue (e.g., event travel page, airport page, or map).

- pass_options: A list of multi-day pass/ticket options (especially 3-day or 4-day), each with:
  - option_name
  - duration (e.g., "3-day", "4-day", "3 days")
  - price (keep exact text, e.g., "$99", "$99–$129", "from $120")
  - urls: any URL(s) cited for this option
- pricing_urls: Additional URLs that support ticket/pass options and pricing.

- indie_opportunities: Text that indicates indie developer or exhibitor showcase opportunities (if provided).
- tournaments_esports: Text that indicates competitive gaming tournaments or esports competitions (if provided).
- features_urls: URLs supporting indie opportunities and tournaments/esports features.

General rules:
- Only extract URLs explicitly present in the answer (plain URLs or markdown links). Do not invent URLs.
- If any field is missing in the answer, set it to null (for string fields) or [] (for URL lists).
- Do not normalize or change numeric values; keep the exact strings as written in the answer.
Return a JSON object with a single key "conventions", which is an array of convention objects with the fields above.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _uniq(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if not x:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def all_sources_for(conv: ConventionExtraction) -> List[str]:
    return _uniq(
        (conv.basic_info_urls or [])
        + (conv.venue_urls or [])
        + (conv.accessibility_urls or [])
        + (conv.pricing_urls or [])
        + (conv.features_urls or [])
        + sum([opt.urls or [] for opt in (conv.pass_options or [])], [])
    )


def pick_sources(preferred_lists: List[List[str]], fallback: List[str]) -> List[str]:
    for lst in preferred_lists:
        if lst:
            return _uniq(lst)
    return _uniq(fallback)


def has_three_or_four_day(pass_options: List[PassOption]) -> bool:
    """
    Check whether any pass mentions 3-day or 4-day duration.
    """
    keys = ["3-day", "3 day", "3 days", "three-day", "4-day", "4 day", "4 days", "four-day"]
    for opt in pass_options or []:
        dur_text = (opt.duration or "") + " " + (opt.option_name or "")
        dur_text = dur_text.lower()
        if any(k in dur_text for k in keys):
            return True
    return False


def summarize_multi_day_passes(pass_options: List[PassOption]) -> str:
    """
    Create a concise textual summary of multi-day pass options and prices to embed in a claim.
    """
    items = []
    for opt in pass_options or []:
        label = opt.option_name or (opt.duration or "multi-day pass")
        price = opt.price or "price not specified"
        items.append(f"{label}: {price}")
    if not items:
        return "No multi-day pass options were extracted."
    return "; ".join(items)


# --------------------------------------------------------------------------- #
# Verification logic per convention                                           #
# --------------------------------------------------------------------------- #
async def verify_convention(
    evaluator: Evaluator,
    parent_node,
    conv: ConventionExtraction,
    index_zero_based: int,
) -> None:
    idx = index_zero_based + 1

    # Parent node for this convention
    conv_node = evaluator.add_parallel(
        id=f"Convention_{idx}",
        desc=f"Convention #{idx} verification - meets all specified criteria",
        parent=parent_node,
        critical=False
    )

    # ---------------------- Basic Information ---------------------------- #
    basic_node = evaluator.add_parallel(
        id=f"Conv{idx}_Basic_Information",
        desc="Basic convention identification and scheduling information",
        parent=conv_node,
        critical=True
    )

    # Name (leaf)
    name_leaf = evaluator.add_leaf(
        id=f"Conv{idx}_Name",
        desc="Convention name is provided",
        parent=basic_node,
        critical=True
    )
    name_sources = pick_sources([conv.basic_info_urls], all_sources_for(conv))
    await evaluator.verify(
        claim=f"The convention's official name is '{conv.name}'.",
        node=name_leaf,
        sources=name_sources,
        additional_instruction="Verify that the referenced page(s) clearly indicate the event's name. Allow stylistic variations (e.g., 'Expo' vs 'Exposition')."
    )

    # Dates in 2026 (leaf)
    dates_leaf = evaluator.add_leaf(
        id=f"Conv{idx}_Dates",
        desc="Specific dates in 2026 are provided",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In 2026, the convention '{conv.name}' is scheduled for {conv.dates}. The year is 2026.",
        node=dates_leaf,
        sources=name_sources,
        additional_instruction="Confirm that the dates explicitly reference the year 2026 on the page. Minor format differences are acceptable."
    )

    # US Location (leaf)
    us_loc_leaf = evaluator.add_leaf(
        id=f"Conv{idx}_US_Location",
        desc="Convention is located in the United States",
        parent=basic_node,
        critical=True
    )
    city_part = conv.city or ""
    state_part = conv.state or ""
    await evaluator.verify(
        claim=f"The convention '{conv.name}' takes place in {city_part}, {state_part}, United States.",
        node=us_loc_leaf,
        sources=name_sources,
        additional_instruction="Verify that the page shows the city and state and that the location is in the U.S."
    )

    # ---------------------- Venue Requirements --------------------------- #
    venue_node = evaluator.add_parallel(
        id=f"Conv{idx}_Venue_Requirements",
        desc="Venue specifications and capacity information",
        parent=conv_node,
        critical=True
    )

    # Venue reference existence (critical custom leaf)
    venue_ref_leaf = evaluator.add_custom_node(
        result=bool(conv.venue_urls),
        id=f"Conv{idx}_Venue_Reference",
        desc="Reference URL supporting venue specifications",
        parent=venue_node,
        critical=True
    )

    # Venue name and city (leaf)
    venue_name_leaf = evaluator.add_leaf(
        id=f"Conv{idx}_Venue_Name",
        desc="Venue name and city are provided",
        parent=venue_node,
        critical=True
    )
    venue_name_sources = pick_sources([conv.venue_urls, conv.basic_info_urls], all_sources_for(conv))
    await evaluator.verify(
        claim=f"The event venue is {conv.venue_name} in {city_part}, {state_part}.",
        node=venue_name_leaf,
        sources=venue_name_sources,
        additional_instruction="Verify that the page indicates the venue name and the city/state it is located in.",
        extra_prerequisites=[venue_ref_leaf]
    )

    # Exhibit space >= 400,000 sq ft (leaf)
    exhibit_space_leaf = evaluator.add_leaf(
        id=f"Conv{idx}_Exhibit_Space",
        desc="Venue has at least 400,000 square feet of exhibit/expo space",
        parent=venue_node,
        critical=True
    )
    space_text = conv.venue_exhibit_space_sqft or "an exhibit space meeting or exceeding 400,000 square feet"
    await evaluator.verify(
        claim=f"The venue {conv.venue_name} has total exhibit/expo space of {space_text}, which is at least 400,000 square feet.",
        node=exhibit_space_leaf,
        sources=venue_name_sources,
        additional_instruction="Use the venue or official facility specifications page. Accept synonymous terms like 'exhibit space', 'exhibition space', or 'gross exhibit space'.",
        extra_prerequisites=[venue_ref_leaf]
    )

    # ---------------------- Accessibility -------------------------------- #
    access_node = evaluator.add_parallel(
        id=f"Conv{idx}_Accessibility",
        desc="Airport proximity and travel accessibility",
        parent=conv_node,
        critical=True
    )

    # Accessibility reference existence (critical custom leaf)
    access_ref_leaf = evaluator.add_custom_node(
        result=bool(conv.accessibility_urls),
        id=f"Conv{idx}_Accessibility_Reference",
        desc="Reference URL supporting airport distance information",
        parent=access_node,
        critical=True
    )

    # Nearest airport (leaf)
    airport_leaf = evaluator.add_leaf(
        id=f"Conv{idx}_Airport",
        desc="Nearest major airport is identified",
        parent=access_node,
        critical=True
    )
    access_sources = pick_sources([conv.accessibility_urls], all_sources_for(conv))
    await evaluator.verify(
        claim=f"The nearest major commercial airport to {conv.venue_name} is {conv.nearest_airport}.",
        node=airport_leaf,
        sources=access_sources,
        additional_instruction="Accept official event travel pages, airport pages, or authoritative travel guidance indicating the primary/nearest commercial airport.",
        extra_prerequisites=[access_ref_leaf]
    )

    # Airport within 15 miles (leaf)
    airport_dist_leaf = evaluator.add_leaf(
        id=f"Conv{idx}_Airport_Distance",
        desc="Airport is within 15 miles of the convention venue",
        parent=access_node,
        critical=True
    )
    approx_dist = conv.airport_distance_miles or "within 15 miles"
    await evaluator.verify(
        claim=f"The distance from {conv.nearest_airport} to {conv.venue_name} is within 15 miles (approximately {approx_dist}).",
        node=airport_dist_leaf,
        sources=access_sources,
        additional_instruction="If a map or travel page indicates distance/time, consider <= 15 miles acceptable. Allow small rounding differences.",
        extra_prerequisites=[access_ref_leaf]
    )

    # ---------------------- Cost Information ------------------------------ #
    cost_node = evaluator.add_parallel(
        id=f"Conv{idx}_Cost_Information",
        desc="Ticket pricing and multi-day attendance options",
        parent=conv_node,
        critical=True
    )

    # Pricing reference existence (critical custom leaf)
    pricing_ref_leaf = evaluator.add_custom_node(
        result=bool(conv.pricing_urls) or any((opt.urls for opt in conv.pass_options or [])),
        id=f"Conv{idx}_Pricing_Reference",
        desc="Reference URL supporting ticket pricing information",
        parent=cost_node,
        critical=True
    )

    # Multiday options (leaf)
    multiday_leaf = evaluator.add_leaf(
        id=f"Conv{idx}_Multiday_Options",
        desc="Convention offers 3-day or 4-day attendance passes/badges",
        parent=cost_node,
        critical=True
    )
    pricing_sources = pick_sources(
        [conv.pricing_urls, sum([opt.urls or [] for opt in (conv.pass_options or [])], [])],
        all_sources_for(conv)
    )
    await evaluator.verify(
        claim="The convention offers a 3-day or 4-day pass or badge among its attendance options.",
        node=multiday_leaf,
        sources=pricing_sources,
        additional_instruction="Check registration/tickets pages. Any multi-day option explicitly labeled 3-day or 4-day qualifies.",
        extra_prerequisites=[pricing_ref_leaf]
    )

    # Pricing provided (leaf)
    pricing_leaf = evaluator.add_leaf(
        id=f"Conv{idx}_Pricing",
        desc="Ticket/pass pricing information is provided",
        parent=cost_node,
        critical=True
    )
    passes_summary = summarize_multi_day_passes(conv.pass_options)
    await evaluator.verify(
        claim=f"The following multi-day ticket or pass options and prices are available for the convention: {passes_summary}",
        node=pricing_leaf,
        sources=pricing_sources,
        additional_instruction="Accept reasonable variations (e.g., early-bird vs. standard). Ensure that at least one multi-day option and its price is shown on the cited page(s).",
        extra_prerequisites=[pricing_ref_leaf]
    )

    # ---------------------- Gaming Features ------------------------------- #
    features_node = evaluator.add_parallel(
        id=f"Conv{idx}_Gaming_Features",
        desc="Gaming-specific features and opportunities",
        parent=conv_node,
        critical=True
    )

    # Features reference existence (critical custom leaf)
    features_ref_leaf = evaluator.add_custom_node(
        result=bool(conv.features_urls),
        id=f"Conv{idx}_Features_Reference",
        desc="Reference URL supporting gaming features information",
        parent=features_node,
        critical=True
    )

    # Indie opportunities (leaf)
    indie_leaf = evaluator.add_leaf(
        id=f"Conv{idx}_Indie_Opportunities",
        desc="Convention offers indie game developer or exhibitor showcase opportunities",
        parent=features_node,
        critical=True
    )
    features_sources = pick_sources([conv.features_urls], all_sources_for(conv))
    indie_text = conv.indie_opportunities or "indie developer or exhibitor showcase opportunities are available"
    await evaluator.verify(
        claim=f"The convention offers opportunities for indie game developers or exhibitors to showcase games (e.g., an indie showcase or exhibitor booths). Specifically: {indie_text}",
        node=indie_leaf,
        sources=features_sources,
        additional_instruction="Look for terms like 'Indie Showcase', 'Indie Zone', 'Exhibitor application', 'Indie pavilion', or similar.",
        extra_prerequisites=[features_ref_leaf]
    )

    # Tournaments/esports (leaf)
    tourney_leaf = evaluator.add_leaf(
        id=f"Conv{idx}_Tournaments",
        desc="Convention features competitive gaming tournaments or esports competitions",
        parent=features_node,
        critical=True
    )
    tourney_text = conv.tournaments_esports or "competitive gaming tournaments or esports competitions"
    await evaluator.verify(
        claim=f"The convention features competitive gaming tournaments or esports competitions. Specifically: {tourney_text}",
        node=tourney_leaf,
        sources=features_sources,
        additional_instruction="Look for 'tournament', 'competition', 'esports', 'league', 'brackets', or similar cues on the cited page(s).",
        extra_prerequisites=[features_ref_leaf]
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 2026 U.S. gaming conventions task.
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

    # Root node for Gaming_Convention_Selection (parallel aggregation)
    selection_root = evaluator.add_parallel(
        id="Gaming_Convention_Selection",
        desc="Task requires identifying three major gaming conventions in the United States scheduled for 2026, each meeting specific criteria for venue size, accessibility, developer opportunities, and gaming features",
        parent=root,
        critical=False
    )

    # Extract structured conventions info
    extracted = await evaluator.extract(
        prompt=prompt_extract_conventions(),
        template_class=ConventionsExtraction,
        extraction_name="conventions_extraction"
    )

    conventions: List[ConventionExtraction] = extracted.conventions[:3] if extracted and extracted.conventions else []
    # Pad to exactly three conventions to build a consistent verification tree
    while len(conventions) < 3:
        conventions.append(ConventionExtraction())

    # Build verification subtrees for each of the three conventions
    for i in range(3):
        await verify_convention(
            evaluator=evaluator,
            parent_node=selection_root,
            conv=conventions[i],
            index_zero_based=i
        )

    return evaluator.get_summary()