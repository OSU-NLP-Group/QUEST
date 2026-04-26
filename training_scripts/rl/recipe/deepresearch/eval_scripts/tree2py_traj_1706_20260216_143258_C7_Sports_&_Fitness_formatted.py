import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nfl_stadiums_2026_lactation_parking_capacity"
TASK_DESCRIPTION = (
    "Identify four NFL stadiums in the United States that meet all of the following criteria:\n"
    "1) The stadium must currently serve as a home stadium for an NFL team during the 2026 season.\n"
    "2) The stadium must have a seating capacity of at least 70,000.\n"
    "3) The stadium must provide nursing rooms or designated lactation spaces.\n"
    "4) The stadium must offer on-site parking for game attendees.\n"
    "For each stadium, provide the official name, seating capacity, confirmation of nursing/lactation spaces, "
    "confirmation of on-site parking, and the complete physical address (street, city, state)."
)

# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class StadiumItem(BaseModel):
    official_name: Optional[str] = None
    seating_capacity: Optional[str] = None
    has_nursing_or_lactation_space: Optional[bool] = None
    has_on_site_parking: Optional[bool] = None
    complete_address: Optional[str] = None
    home_teams_2026: List[str] = Field(default_factory=list)

    # Additional amenities (non-critical in evaluation)
    ada_wheelchair_accessible_seating: Optional[bool] = None
    clear_bag_policy: Optional[bool] = None
    public_tours: Optional[bool] = None
    baby_changing_stations: Optional[bool] = None
    wifi: Optional[bool] = None
    mobile_app_navigation: Optional[bool] = None
    accessible_public_transport_nearby: Optional[bool] = None

    # Evidence URLs as cited in the answer for this stadium
    source_urls: List[str] = Field(default_factory=list)


class StadiumsExtraction(BaseModel):
    stadiums: List[StadiumItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadiums() -> str:
    return """
    Extract all stadium entries mentioned in the answer that pertain to NFL stadiums. For each stadium, extract the following fields exactly as they appear in the answer:

    Required fields:
    - official_name: The official name of the stadium (string).
    - seating_capacity: The seating capacity value as written in the answer (string; do not convert to number; keep commas or formatting if present).
    - has_nursing_or_lactation_space: Whether the answer explicitly states the stadium provides nursing rooms or lactation spaces (boolean true/false). If not stated, return null.
    - has_on_site_parking: Whether the answer explicitly states the stadium offers on-site parking for game attendees (boolean true/false). If not stated, return null.
    - complete_address: The complete physical address including street address, city, and state (one-line string). If the address is incomplete or missing, return null.
    - home_teams_2026: A list of NFL team names (strings) that the answer claims use this stadium as a home stadium during the 2026 season. If not explicitly stated, return an empty list.

    Additional (optional) amenities (set to true/false if explicitly stated; otherwise null):
    - ada_wheelchair_accessible_seating
    - clear_bag_policy
    - public_tours
    - baby_changing_stations
    - wifi
    - mobile_app_navigation
    - accessible_public_transport_nearby

    Evidence URLs:
    - source_urls: Collect all URLs in the answer that are explicitly tied to this stadium (including official team sites, stadium A–Z guides, parking pages, accessibility pages, policy pages, etc.). 
      SPECIAL RULES FOR URL EXTRACTION:
      • Only extract URLs that are explicitly present in the answer text (including markdown links).
      • Do not invent or infer any URL.
      • Always include full URLs; if a URL is missing the protocol, prepend http://.
      • If there are no URLs for a stadium, return an empty array.

    Return a JSON object with a single field:
    {
      "stadiums": [ ... up to all stadiums found in the answer ... ]
    }

    If any field is missing for a stadium, follow the rules above (null for booleans when unspecified; null for strings when missing).
    """


# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(value: Optional[str]) -> str:
    return value or ""

def _has_sources(urls: List[str]) -> bool:
    return bool(urls and len(urls) > 0)

def _web_grounding_instruction(base: str, has_sources: bool) -> str:
    suffix = ""
    # Enforce source-grounding: if no sources provided, instruct judge to mark incorrect
    if not has_sources:
        suffix = (
            "\nImportant: The answer did not provide any URLs to verify this claim. "
            "Treat the claim as not supported by evidence and judge it as Incorrect."
        )
    return base + suffix


# --------------------------------------------------------------------------- #
# Verification Logic for One Stadium                                          #
# --------------------------------------------------------------------------- #
async def verify_stadium(
    evaluator: Evaluator,
    parent_node,
    stadium: StadiumItem,
    idx: int
) -> None:
    """
    Build and verify the subtree for a single stadium.
    """
    disp_idx = idx + 1
    stadium_node = evaluator.add_parallel(
        id=f"stadium_{disp_idx}",
        desc=f"Stadium #{disp_idx} satisfies all constraints and required fields are provided.",
        parent=parent_node,
        critical=False  # stadium nodes contribute partial credit at root
    )

    # Critical sibling: official name provided (gates other checks under this stadium)
    name_provided = bool(stadium.official_name and stadium.official_name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id=f"stadium_{disp_idx}_official_name_provided",
        desc=f"The official name of Stadium #{disp_idx} is provided.",
        parent=stadium_node,
        critical=True
    )

    # 1) US location
    us_loc_leaf = evaluator.add_leaf(
        id=f"stadium_{disp_idx}_us_location",
        desc=f"Stadium #{disp_idx} is located in the United States.",
        parent=stadium_node,
        critical=True
    )
    claim_us = (
        f"{_safe(stadium.official_name)} is located in the United States."
        if name_provided else "This stadium is located in the United States."
    )
    add_ins_us = _web_grounding_instruction(
        "Verify via the cited page(s) that the stadium is in a US city/state. "
        "Accept if the page shows a US address or explicitly mentions a US state/city.",
        _has_sources(stadium.source_urls)
    )
    await evaluator.verify(
        claim=claim_us,
        node=us_loc_leaf,
        sources=stadium.source_urls
    ,   additional_instruction=add_ins_us)

    # 2) Active NFL home in 2026
    home_leaf = evaluator.add_leaf(
        id=f"stadium_{disp_idx}_active_nfl_home_2026",
        desc=f"Stadium #{disp_idx} currently serves as a home stadium for an NFL team during the 2026 season.",
        parent=stadium_node,
        critical=True
    )
    teams_txt = ", ".join(stadium.home_teams_2026) if stadium.home_teams_2026 else "an NFL team"
    claim_home = (
        f"During the 2026 NFL season, {_safe(stadium.official_name)} serves as a home stadium for {teams_txt}."
        if name_provided else
        f"During the 2026 NFL season, this stadium serves as a home stadium for {teams_txt}."
    )
    add_ins_home = _web_grounding_instruction(
        "Accept official team or stadium pages that show the stadium as the team's home. "
        "If pages are dated 2024–2026 and clearly establish the home stadium status, that is acceptable for 2026.",
        _has_sources(stadium.source_urls)
    )
    await evaluator.verify(
        claim=claim_home,
        node=home_leaf,
        sources=stadium.source_urls,
        additional_instruction=add_ins_home
    )

    # 3) Seating capacity provided and >= 70,000
    cap_leaf = evaluator.add_leaf(
        id=f"stadium_{disp_idx}_capacity_provided_and_ge_70000",
        desc=f"Stadium #{disp_idx} seating capacity is provided and is at least 70,000.",
        parent=stadium_node,
        critical=True
    )
    cap_txt = _safe(stadium.seating_capacity)
    if cap_txt:
        claim_cap = (
            f"The seating capacity of {_safe(stadium.official_name)} is {cap_txt} and this is at least 70,000."
            if name_provided else
            f"The stadium's seating capacity is {cap_txt} and this is at least 70,000."
        )
    else:
        claim_cap = (
            f"The seating capacity of {_safe(stadium.official_name)} is at least 70,000."
            if name_provided else
            "The stadium's seating capacity is at least 70,000."
        )
    add_ins_cap = _web_grounding_instruction(
        "Confirm the seating capacity from an official or reputable page. "
        "Consider minor rounding acceptable (e.g., 70,000 vs. 70k).",
        _has_sources(stadium.source_urls)
    )
    await evaluator.verify(
        claim=claim_cap,
        node=cap_leaf,
        sources=stadium.source_urls,
        additional_instruction=add_ins_cap
    )

    # 4) Nursing/lactation spaces available
    lact_leaf = evaluator.add_leaf(
        id=f"stadium_{disp_idx}_nursing_or_lactation_spaces",
        desc=f"Stadium #{disp_idx} provides designated nursing rooms or lactation spaces for breastfeeding mothers.",
        parent=stadium_node,
        critical=True
    )
    claim_lact = (
        f"{_safe(stadium.official_name)} provides nursing rooms or designated lactation spaces for breastfeeding mothers."
        if name_provided else
        "The stadium provides nursing rooms or designated lactation spaces for breastfeeding mothers."
    )
    add_ins_lact = _web_grounding_instruction(
        "Look for terms such as 'nursing room', 'lactation room', 'mother's room', 'Mamava', or 'family room' in the A–Z guide or amenities page.",
        _has_sources(stadium.source_urls)
    )
    await evaluator.verify(
        claim=claim_lact,
        node=lact_leaf,
        sources=stadium.source_urls,
        additional_instruction=add_ins_lact
    )

    # 5) On-site parking available
    park_leaf = evaluator.add_leaf(
        id=f"stadium_{disp_idx}_on_site_parking",
        desc=f"Stadium #{disp_idx} offers on-site parking facilities available for game attendees.",
        parent=stadium_node,
        critical=True
    )
    claim_parking = (
        f"On-site parking is available at {_safe(stadium.official_name)} for fans attending games."
        if name_provided else
        "On-site parking is available at the stadium for fans attending games."
    )
    add_ins_parking = _web_grounding_instruction(
        "Use parking information pages or A–Z guides. Accept 'on-site' lots or garages that are part of the stadium campus.",
        _has_sources(stadium.source_urls)
    )
    await evaluator.verify(
        claim=claim_parking,
        node=park_leaf,
        sources=stadium.source_urls,
        additional_instruction=add_ins_parking
    )

    # Additional amenities (non-critical)
    # ADA seating
    ada_leaf = evaluator.add_leaf(
        id=f"stadium_{disp_idx}_ada_wheelchair_accessible_seating",
        desc=f"Stadium #{disp_idx} offers wheelchair-accessible seating with ADA compliance.",
        parent=stadium_node,
        critical=False
    )
    claim_ada = (
        f"{_safe(stadium.official_name)} offers wheelchair-accessible seating that is ADA compliant."
        if name_provided else
        "The stadium offers wheelchair-accessible seating that is ADA compliant."
    )
    await evaluator.verify(
        claim=claim_ada,
        node=ada_leaf,
        sources=stadium.source_urls,
        additional_instruction=_web_grounding_instruction(
            "Look for accessibility or ADA information pages mentioning accessible/companion seating.",
            _has_sources(stadium.source_urls)
        )
    )

    # Clear bag policy
    bag_leaf = evaluator.add_leaf(
        id=f"stadium_{disp_idx}_clear_bag_policy",
        desc=f"Stadium #{disp_idx} follows the NFL clear bag policy.",
        parent=stadium_node,
        critical=False
    )
    claim_bag = (
        f"{_safe(stadium.official_name)} follows the NFL clear bag policy."
        if name_provided else
        "The stadium follows the NFL clear bag policy."
    )
    await evaluator.verify(
        claim=claim_bag,
        node=bag_leaf,
        sources=stadium.source_urls,
        additional_instruction=_web_grounding_instruction(
            "Look for 'NFL Clear Bag Policy' or equivalent bag policy language on official pages.",
            _has_sources(stadium.source_urls)
        )
    )

    # Public tours
    tours_leaf = evaluator.add_leaf(
        id=f"stadium_{disp_idx}_public_tours",
        desc=f"Stadium #{disp_idx} offers public stadium tours.",
        parent=stadium_node,
        critical=False
    )
    claim_tours = (
        f"{_safe(stadium.official_name)} offers public stadium tours."
        if name_provided else
        "The stadium offers public stadium tours."
    )
    await evaluator.verify(
        claim=claim_tours,
        node=tours_leaf,
        sources=stadium.source_urls,
        additional_instruction=_web_grounding_instruction(
            "Accept evidence of public or guided tours offered to visitors on non-game days.",
            _has_sources(stadium.source_urls)
        )
    )

    # Baby changing stations
    baby_leaf = evaluator.add_leaf(
        id=f"stadium_{disp_idx}_baby_changing_stations",
        desc=f"Stadium #{disp_idx} has family-friendly amenities including baby changing stations.",
        parent=stadium_node,
        critical=False
    )
    claim_baby = (
        f"Baby changing stations are available at {_safe(stadium.official_name)}."
        if name_provided else
        "Baby changing stations are available at the stadium."
    )
    await evaluator.verify(
        claim=claim_baby,
        node=baby_leaf,
        sources=stadium.source_urls,
        additional_instruction=_web_grounding_instruction(
            "Look for 'changing tables', 'family restrooms', or 'baby changing stations' in A–Z or amenities guides.",
            _has_sources(stadium.source_urls)
        )
    )

    # Wi-Fi
    wifi_leaf = evaluator.add_leaf(
        id=f"stadium_{disp_idx}_wifi",
        desc=f"Stadium #{disp_idx} provides Wi-Fi connectivity for fans.",
        parent=stadium_node,
        critical=False
    )
    claim_wifi = (
        f"Wi‑Fi is available for fans at {_safe(stadium.official_name)}."
        if name_provided else
        "Wi‑Fi is available for fans at the stadium."
    )
    await evaluator.verify(
        claim=claim_wifi,
        node=wifi_leaf,
        sources=stadium.source_urls,
        additional_instruction=_web_grounding_instruction(
            "Accept mentions of stadium Wi‑Fi availability, SSID info, or connectivity guides.",
            _has_sources(stadium.source_urls)
        )
    )

    # Mobile app navigation
    app_leaf = evaluator.add_leaf(
        id=f"stadium_{disp_idx}_mobile_app_navigation",
        desc=f"Stadium #{disp_idx} has a dedicated mobile app with stadium navigation features.",
        parent=stadium_node,
        critical=False
    )
    claim_app = (
        f"The stadium or its team provides a mobile app with in‑stadium navigation or interactive maps for {_safe(stadium.official_name)}."
        if name_provided else
        "The stadium or its team provides a mobile app with in‑stadium navigation or interactive maps."
    )
    await evaluator.verify(
        claim=claim_app,
        node=app_leaf,
        sources=stadium.source_urls,
        additional_instruction=_web_grounding_instruction(
            "Look for official app pages or A–Z guides that mention interactive maps, wayfinding, or navigation features.",
            _has_sources(stadium.source_urls)
        )
    )

    # Accessible public transportation nearby
    transit_leaf = evaluator.add_leaf(
        id=f"stadium_{disp_idx}_accessible_public_transport_nearby",
        desc=f"Stadium #{disp_idx} has accessible public transportation options within reasonable distance.",
        parent=stadium_node,
        critical=False
    )
    claim_transit = (
        f"There are accessible public transportation options within reasonable distance of {_safe(stadium.official_name)}."
        if name_provided else
        "There are accessible public transportation options within reasonable distance of the stadium."
    )
    await evaluator.verify(
        claim=claim_transit,
        node=transit_leaf,
        sources=stadium.source_urls,
        additional_instruction=_web_grounding_instruction(
            "Accept official mentions of bus, rail, or transit stops serving the stadium area; look for accessibility references where possible.",
            _has_sources(stadium.source_urls)
        )
    )

    # Complete address (street, city, state)
    addr_leaf = evaluator.add_leaf(
        id=f"stadium_{disp_idx}_complete_address",
        desc=f"Stadium #{disp_idx} complete physical address is provided (street address, city, state).",
        parent=stadium_node,
        critical=True
    )
    address_txt = _safe(stadium.complete_address)
    if address_txt:
        claim_addr = (
            f"The complete street address of {_safe(stadium.official_name)} is '{address_txt}'."
            if name_provided else
            f"The complete street address of the stadium is '{address_txt}'."
        )
    else:
        claim_addr = (
            f"The complete street address (street, city, state) of {_safe(stadium.official_name)} is provided as stated."
            if name_provided else
            "The complete street address (street, city, state) of the stadium is provided as stated."
        )
    add_ins_addr = _web_grounding_instruction(
        "Verify the full physical address from an official or reputable page. Allow minor formatting differences "
        "(e.g., 'Ave' vs 'Avenue', punctuation).",
        _has_sources(stadium.source_urls)
    )
    await evaluator.verify(
        claim=claim_addr,
        node=addr_leaf,
        sources=stadium.source_urls,
        additional_instruction=add_ins_addr
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
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
    Evaluate the answer for the 'four NFL stadiums with 2026 home status, >=70,000 capacity, lactation spaces, and on-site parking' task.
    """
    # 1) Initialize evaluator (root is non-critical to allow partial credit aggregation)
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

    # 2) Extract structured stadium information
    extraction = await evaluator.extract(
        prompt=prompt_extract_stadiums(),
        template_class=StadiumsExtraction,
        extraction_name="stadiums_extraction"
    )

    # 3) Select exactly four stadiums for evaluation (pad with placeholders if fewer)
    selected: List[StadiumItem] = []
    for item in extraction.stadiums:
        if len(selected) >= 4:
            break
        selected.append(item)
    while len(selected) < 4:
        selected.append(StadiumItem())

    # 4) Add distinctness check (critical under root to gate all stadium verifications)
    #    We consider the four selected items as the "exactly four" that will be graded.
    names = [(_safe(s.official_name).strip().lower()) for s in selected if s.official_name and s.official_name.strip()]
    distinct_ok = (len(names) == 4) and (len(set(names)) == 4)
    evaluator.add_custom_node(
        result=distinct_ok,
        id="four_distinct_stadiums_provided",
        desc="Exactly four distinct stadiums are identified (no duplicates).",
        parent=root,
        critical=True
    )

    # Optional: Record which stadiums were selected for evaluation
    evaluator.add_custom_info(
        info={"selected_stadium_names": [s.official_name for s in selected]},
        info_type="debug",
        info_name="selected_stadiums"
    )

    # 5) Build verification subtree for each stadium
    for i, stadium in enumerate(selected):
        await verify_stadium(evaluator, root, stadium, i)

    # 6) Return evaluation summary
    return evaluator.get_summary()