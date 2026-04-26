import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_theaters_3"
TASK_DESCRIPTION = """Identify 3 professional performing arts theaters in the United States that meet the following comprehensive requirements:

Venue Specifications:
- Total seating capacity between 800 and 1,500 seats
- Proscenium stage configuration with a minimum proscenium opening width of 30 feet
- Stage depth of at least 25 feet from the proscenium line to the back wall

Accessibility & Operations:
- At least 4 wheelchair-accessible seating spaces complying with ADA requirements
- Active box office operations or online ticketing system
- Currently operational and hosting live performances

Technical & Backstage Facilities:
- Loading dock or dedicated loading area for equipment delivery
- Dedicated dressing room facilities for performers
- Professional-grade technical infrastructure (rigging, sound, and lighting systems)

Additional Requirements:
- Parking availability information (on-site, nearby, or public parking options)
- Primary use for live performing arts (theater, dance, concerts, or musicals)
- Suitable for hosting professional touring productions

For each theater, provide:
1. Official theater name
2. City and state location
3. Exact seating capacity
4. Specific proscenium width and stage depth measurements
5. Number of wheelchair-accessible spaces
6. Description of loading facilities
7. Description of backstage/dressing room facilities
8. Parking information
9. Technical capabilities overview
10. Reference URLs confirming all specifications (official theater website, technical specification documents, or credible venue information sources)

Each theater must be verified as a professional-grade performing arts venue with comprehensive facilities suitable for major theatrical productions.
"""


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class TheaterItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    seating_capacity: Optional[str] = None
    stage_configuration: Optional[str] = None  # e.g., "proscenium"
    proscenium_width: Optional[str] = None     # e.g., "32 ft", "9.8 m", "32'-0\""
    stage_depth: Optional[str] = None          # e.g., "28 ft", "8.5 m", "28'-0\""

    wheelchair_spaces: Optional[str] = None    # number as string in answer
    ada_space_dimensions: Optional[str] = None # e.g., "36 in x 48 in", or textual description

    box_office_or_online_ticketing: Optional[str] = None
    operational_status: Optional[str] = None   # e.g., "operational", "hosting live shows"

    loading_facilities: Optional[str] = None
    backstage_facilities: Optional[str] = None

    parking_info: Optional[str] = None

    rigging_capability: Optional[str] = None   # description or yes/no text
    sound_system: Optional[str] = None
    lighting_system: Optional[str] = None
    technical_overview: Optional[str] = None

    reference_urls: List[str] = Field(default_factory=list)


class TheatersExtraction(BaseModel):
    theaters: List[TheaterItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theaters() -> str:
    return """
    Extract up to three U.S. professional performing arts theaters mentioned in the answer, capturing the following fields exactly as stated in the answer (do not invent information that is not present):
    For each theater, return an object with keys:
    - name: Official theater name (string)
    - city: City (string)
    - state: U.S. state (string; use the two-letter or full name as it appears)
    - seating_capacity: Exact total seating capacity as stated (string; keep formatting, commas allowed)
    - stage_configuration: Stage type/configuration (e.g., "proscenium") (string)
    - proscenium_width: Proscenium opening width measurement (string with units if present; e.g., "32 ft", "9.8 m", "32'-0\"")
    - stage_depth: Stage depth from proscenium line to back wall (string with units if present)
    - wheelchair_spaces: Number of wheelchair-accessible seating spaces as stated (string; extract the number mentioned)
    - ada_space_dimensions: Dimensions text for each wheelchair space if stated (e.g., "36 inches wide and 48 inches deep") (string)
    - box_office_or_online_ticketing: Text indicating box office or online ticketing (string)
    - operational_status: Text indicating current operational status and hosting of live performances (string)
    - loading_facilities: Description of loading dock/area (string)
    - backstage_facilities: Description of dressing rooms/backstage facilities (string)
    - parking_info: Parking availability information (string)
    - rigging_capability: Text indicating professional rigging capability/fly system (string)
    - sound_system: Text indicating professional-grade sound system (string)
    - lighting_system: Text indicating professional-grade lighting systems (string)
    - technical_overview: Overall technical capabilities overview text (string)
    - reference_urls: List of URLs (strings) that the answer cites for this theater; include only actual URLs present in the answer (official site, tech specs, credible venue info)

    Return JSON with a single key:
    { "theaters": [ { ... }, { ... }, { ... } ] }

    Rules:
    - Only include theaters explicitly mentioned in the answer.
    - If a field is not present in the answer, set it to null (or empty list for reference_urls).
    - Ensure URLs are valid and complete (include protocol).
    - If more than three theaters are mentioned, include the first three in the answer order.
    """


# --------------------------------------------------------------------------- #
# Helper parsing utilities                                                    #
# --------------------------------------------------------------------------- #
def _safe_str(s: Optional[str]) -> str:
    return s or ""

def parse_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    t = text.strip()
    t = t.replace(",", "")
    m = re.search(r"(\d{1,6})", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def parse_feet_inches(text: str) -> Optional[float]:
    """
    Parse patterns like 31'-6" or 31' 6" -> return feet as float.
    """
    if not text:
        return None
    # Feet and inches pattern
    m = re.search(r"(\d+)\s*['′]\s*(\d+)\s*(?:\"|in|inches)?", text)
    if m:
        feet = float(m.group(1))
        inches = float(m.group(2))
        return feet + inches / 12.0
    return None

def parse_length_ft(text: Optional[str]) -> Optional[float]:
    """
    Parse a length string and return value in feet.
    Supports feet (ft/feet/foot/'), inches ("/in/inches), meters (m, meter, meters).
    """
    if not text:
        return None
    t = text.lower().strip()

    # Feet+inches like 31'-6"
    val = parse_feet_inches(text)
    if val is not None:
        return val

    # Feet only
    m_ft = re.search(r"(\d+(?:\.\d+)?)\s*(?:ft|feet|foot)\b", t)
    if m_ft:
        try:
            return float(m_ft.group(1))
        except Exception:
            pass

    # Apostrophe feet only like 32'
    m_ap = re.search(r"(\d+(?:\.\d+)?)\s*['′]\b", t)
    if m_ap:
        try:
            return float(m_ap.group(1))
        except Exception:
            pass

    # Inches
    m_in = re.search(r"(\d+(?:\.\d+)?)\s*(?:in|inch|inches|\"|”)\b", t)
    if m_in:
        try:
            inches = float(m_in.group(1))
            return inches / 12.0
        except Exception:
            pass

    # Meters
    m_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters)\b", t)
    if m_m:
        try:
            meters = float(m_m.group(1))
            return meters * 3.28084
        except Exception:
            pass

    # Fallback: raw number (assume feet)
    m_num = re.search(r"(\d+(?:\.\d+)?)", t)
    if m_num:
        try:
            return float(m_num.group(1))
        except Exception:
            pass

    return None

def parse_ada_dimensions_in_inches(text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Attempt to parse ADA wheelchair space dimensions width/depth in inches from free text.
    Return (width_in, depth_in) if both available or some; else (None, None) or partially None.
    """
    if not text:
        return None, None
    t = text.lower()

    # Patterns like "36 inches wide and 48 inches deep"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:in|inch|inches)\s*wide.*?(\d+(?:\.\d+)?)\s*(?:in|inch|inches)\s*deep", t)
    if m:
        try:
            w = float(m.group(1))
            d = float(m.group(2))
            return w, d
        except Exception:
            pass

    # Generic two numbers in inches like "36 in x 48 in"
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*(?:in|inch|inches)[^\d]+(\d+(?:\.\d+)?)\s*(?:in|inch|inches)", t)
    if m2:
        try:
            a = float(m2.group(1))
            b = float(m2.group(2))
            # Heuristic: assume first is width, second depth
            return a, b
        except Exception:
            pass

    # Feet/inch textual like 3'-0" x 4'-0"
    m3 = re.findall(r"(\d+)\s*['′]\s*(\d+)\s*(?:\"|in|inches)?", text)
    if m3 and len(m3) >= 2:
        try:
            w_ft = float(m3[0][0]) + float(m3[0][1]) / 12.0
            d_ft = float(m3[1][0]) + float(m3[1][1]) / 12.0
            return w_ft * 12.0, d_ft * 12.0
        except Exception:
            pass

    return None, None


# --------------------------------------------------------------------------- #
# Verification for one theater                                                #
# --------------------------------------------------------------------------- #
async def verify_theater(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    item: TheaterItem,
) -> None:
    """
    Build the verification subtree for a single theater as per rubric.
    """
    theater_num = idx + 1
    theater_node = evaluator.add_parallel(
        id=f"theater_{theater_num}",
        desc=f"Theater #{theater_num} (one qualifying venue)",
        parent=parent_node,
        critical=False  # Allow partial credit across theaters
    )

    # Sources node (critical sibling to gate all verifications needing evidence)
    sources_node = evaluator.add_parallel(
        id=f"sources_{theater_num}",
        desc="Verification sources",
        parent=theater_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item.reference_urls),
        id=f"reference_urls_{theater_num}",
        desc="Provide reference URL(s) that collectively verify all required specifications and claims for this theater (official site, tech specs, or other credible sources)",
        parent=sources_node,
        critical=True
    )

    # Identification
    ident_node = evaluator.add_parallel(
        id=f"identification_{theater_num}",
        desc="Provide official theater identification and US location",
        parent=theater_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_str(item.name).strip()),
        id=f"name_{theater_num}",
        desc="Provide the official theater name",
        parent=ident_node,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id=f"location_us_{theater_num}",
        desc="Provide city and U.S. state location (must be in the United States)",
        parent=ident_node,
        critical=True
    )
    location_claim = f"The theater '{_safe_str(item.name)}' is located in {_safe_str(item.city)}, {_safe_str(item.state)}, United States."
    await evaluator.verify(
        claim=location_claim,
        node=loc_leaf,
        sources=item.reference_urls,
        additional_instruction="Verify the theater location (city, state) is in the United States based on the provided official or credible venue sources."
    )

    # Capacity
    cap_node = evaluator.add_parallel(
        id=f"capacity_{theater_num}",
        desc="Seating capacity provided and within required range",
        parent=theater_node,
        critical=True
    )
    cap_exact_leaf = evaluator.add_leaf(
        id=f"capacity_exact_{theater_num}",
        desc="Provide the exact total seating capacity as a specific number",
        parent=cap_node,
        critical=True
    )
    cap_claim = f"The seating capacity of '{_safe_str(item.name)}' is {_safe_str(item.seating_capacity)} seats."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_exact_leaf,
        sources=item.reference_urls,
        additional_instruction="Verify the exact total seating capacity from official spec pages or credible venue information."
    )
    cap_val = parse_int(item.seating_capacity)
    evaluator.add_custom_node(
        result=(cap_val is not None and 800 <= cap_val <= 1500),
        id=f"capacity_range_{theater_num}",
        desc="Seating capacity is between 800 and 1,500 seats (inclusive)",
        parent=cap_node,
        critical=True
    )

    # Stage configuration and dimensions
    stage_node = evaluator.add_parallel(
        id=f"stage_{theater_num}",
        desc="Stage configuration and dimensions provided and meet minima",
        parent=theater_node,
        critical=True
    )

    pros_config_leaf = evaluator.add_leaf(
        id=f"proscenium_config_{theater_num}",
        desc="Theater has a proscenium stage configuration",
        parent=stage_node,
        critical=True
    )
    pros_config_claim = f"The stage configuration of '{_safe_str(item.name)}' is proscenium."
    await evaluator.verify(
        claim=pros_config_claim,
        node=pros_config_leaf,
        sources=item.reference_urls,
        additional_instruction="Confirm that the venue uses a proscenium stage configuration (as opposed to thrust, arena, etc.)."
    )

    pros_width_leaf = evaluator.add_leaf(
        id=f"proscenium_width_value_{theater_num}",
        desc="Provide a specific proscenium opening width measurement (in feet or equivalent)",
        parent=stage_node,
        critical=True
    )
    pros_width_claim = f"The proscenium opening width at '{_safe_str(item.name)}' is {_safe_str(item.proscenium_width)}."
    await evaluator.verify(
        claim=pros_width_claim,
        node=pros_width_leaf,
        sources=item.reference_urls,
        additional_instruction="Verify the proscenium opening width from technical specs or credible sources (allow ft/inches/meters)."
    )
    width_ft = parse_length_ft(item.proscenium_width)
    evaluator.add_custom_node(
        result=(width_ft is not None and width_ft >= 30.0),
        id=f"proscenium_width_min_{theater_num}",
        desc="Proscenium opening width is at least 30 feet",
        parent=stage_node,
        critical=True
    )

    stage_depth_leaf = evaluator.add_leaf(
        id=f"stage_depth_value_{theater_num}",
        desc="Provide a specific stage depth measurement from the proscenium line to the back wall (in feet or equivalent)",
        parent=stage_node,
        critical=True
    )
    stage_depth_claim = f"The stage depth (from proscenium line to back wall) at '{_safe_str(item.name)}' is {_safe_str(item.stage_depth)}."
    await evaluator.verify(
        claim=stage_depth_claim,
        node=stage_depth_leaf,
        sources=item.reference_urls,
        additional_instruction="Verify the stage depth from technical specs or credible sources (allow ft/inches/meters)."
    )
    depth_ft = parse_length_ft(item.stage_depth)
    evaluator.add_custom_node(
        result=(depth_ft is not None and depth_ft >= 25.0),
        id=f"stage_depth_min_{theater_num}",
        desc="Stage depth is at least 25 feet from the proscenium line to the back wall",
        parent=stage_node,
        critical=True
    )

    # Accessibility
    access_node = evaluator.add_parallel(
        id=f"accessibility_{theater_num}",
        desc="Wheelchair accessibility requirements",
        parent=theater_node,
        critical=True
    )
    wc_count_leaf = evaluator.add_leaf(
        id=f"wheelchair_count_value_{theater_num}",
        desc="Provide the number of wheelchair-accessible seating spaces as a specific number",
        parent=access_node,
        critical=True
    )
    wc_claim = f"The number of wheelchair-accessible seating spaces at '{_safe_str(item.name)}' is {_safe_str(item.wheelchair_spaces)}."
    await evaluator.verify(
        claim=wc_claim,
        node=wc_count_leaf,
        sources=item.reference_urls,
        additional_instruction="Confirm the quantity of wheelchair-accessible seating locations from official seating charts or venue accessibility info."
    )
    wc_val = parse_int(item.wheelchair_spaces)
    evaluator.add_custom_node(
        result=(wc_val is not None and wc_val >= 4),
        id=f"wheelchair_min_{theater_num}",
        desc="At least 4 wheelchair-accessible seating spaces are available",
        parent=access_node,
        critical=True
    )
    ada_dims_leaf = evaluator.add_leaf(
        id=f"ada_dimensions_{theater_num}",
        desc="Each wheelchair-accessible space meets ADA minimum dimensions of at least 36 inches wide and 48 inches deep",
        parent=access_node,
        critical=True
    )
    ada_claim = "Each wheelchair-accessible space meets ADA minimum dimensions of at least 36 inches wide and 48 inches deep."
    await evaluator.verify(
        claim=ada_claim,
        node=ada_dims_leaf,
        sources=item.reference_urls,
        additional_instruction="Verify wheelchair seating platform dimensions (or compliance statements) meet or exceed 36\" width and 48\" depth."
    )

    # Ticketing
    ticket_node = evaluator.add_parallel(
        id=f"ticketing_{theater_num}",
        desc="Box office operations or online ticketing is active",
        parent=theater_node,
        critical=True
    )
    ticket_leaf = evaluator.add_leaf(
        id=f"active_ticketing_or_boxoffice_{theater_num}",
        desc="Theater has active box office operations or an online ticketing system",
        parent=ticket_node,
        critical=True
    )
    ticket_claim = "The theater has active box office operations or an online ticketing system for purchasing tickets."
    await evaluator.verify(
        claim=ticket_claim,
        node=ticket_leaf,
        sources=item.reference_urls,
        additional_instruction="Confirm ticket purchase pathways (box office hours or online ticketing link) from official site or trusted ticketing platforms."
    )

    # Operational status
    op_node = evaluator.add_parallel(
        id=f"operational_status_{theater_num}",
        desc="Currently operational and hosting live performances",
        parent=theater_node,
        critical=True
    )
    op_leaf = evaluator.add_leaf(
        id=f"operational_and_hosting_{theater_num}",
        desc="Venue is currently operational and hosting live performances",
        parent=op_node,
        critical=True
    )
    op_claim = "The venue is currently operational and hosts live performances (e.g., active season calendar, upcoming shows)."
    await evaluator.verify(
        claim=op_claim,
        node=op_leaf,
        sources=item.reference_urls,
        additional_instruction="Check season calendars, event listings, or 'about' pages to confirm current operations and live events."
    )

    # Loading facilities
    load_node = evaluator.add_parallel(
        id=f"loading_{theater_num}",
        desc="Loading facilities",
        parent=theater_node,
        critical=True
    )
    load_present_leaf = evaluator.add_leaf(
        id=f"loading_present_{theater_num}",
        desc="Theater has a loading dock or dedicated loading area for equipment delivery",
        parent=load_node,
        critical=True
    )
    load_present_claim = "The theater has a loading dock or dedicated loading area suitable for production equipment deliveries."
    await evaluator.verify(
        claim=load_present_claim,
        node=load_present_leaf,
        sources=item.reference_urls,
        additional_instruction="Verify venue specs, rental/tech info, or backstage documents describing loading dock/area."
    )
    evaluator.add_custom_node(
        result=bool(_safe_str(item.loading_facilities).strip()),
        id=f"loading_description_{theater_num}",
        desc="Provide a description of loading facilities",
        parent=load_node,
        critical=True
    )

    # Backstage/dressing facilities
    back_node = evaluator.add_parallel(
        id=f"backstage_{theater_num}",
        desc="Backstage/dressing facilities",
        parent=theater_node,
        critical=True
    )
    dressing_leaf = evaluator.add_leaf(
        id=f"dressing_present_{theater_num}",
        desc="Venue provides dedicated dressing room facilities for performers",
        parent=back_node,
        critical=True
    )
    dressing_claim = "The venue provides dedicated dressing room facilities for performers."
    await evaluator.verify(
        claim=dressing_claim,
        node=dressing_leaf,
        sources=item.reference_urls,
        additional_instruction="Verify backstage/rental specs or technical documents indicating dressing rooms."
    )
    evaluator.add_custom_node(
        result=bool(_safe_str(item.backstage_facilities).strip()),
        id=f"backstage_description_{theater_num}",
        desc="Provide a description of backstage/dressing room facilities",
        parent=back_node,
        critical=True
    )

    # Technical infrastructure
    tech_node = evaluator.add_parallel(
        id=f"technical_{theater_num}",
        desc="Professional-grade technical infrastructure",
        parent=theater_node,
        critical=True
    )
    rig_leaf = evaluator.add_leaf(
        id=f"rigging_{theater_num}",
        desc="Venue has professional rigging capability (rigging/fly/overhead rigging suitable for productions)",
        parent=tech_node,
        critical=True
    )
    rig_claim = "The venue has professional rigging capability (e.g., fly system or overhead rigging) suitable for productions."
    await evaluator.verify(
        claim=rig_claim,
        node=rig_leaf,
        sources=item.reference_urls,
        additional_instruction="Confirm mention of fly system, rigging grid, line sets, or equivalent from technical specs."
    )

    sound_leaf = evaluator.add_leaf(
        id=f"sound_{theater_num}",
        desc="Venue has a professional-grade sound system",
        parent=tech_node,
        critical=True
    )
    sound_claim = "The venue has a professional-grade sound system suitable for major theatrical productions."
    await evaluator.verify(
        claim=sound_claim,
        node=sound_leaf,
        sources=item.reference_urls,
        additional_instruction="Look for audio system descriptions, mixing consoles, speaker arrays, or venue specs describing pro-grade sound."
    )

    light_leaf = evaluator.add_leaf(
        id=f"lighting_{theater_num}",
        desc="Venue has professional-grade lighting systems",
        parent=tech_node,
        critical=True
    )
    light_claim = "The venue has professional-grade theatrical lighting systems suitable for major productions."
    await evaluator.verify(
        claim=light_claim,
        node=light_leaf,
        sources=item.reference_urls,
        additional_instruction="Verify lighting inventory/controls (dimmers, fixtures, consoles) in tech specs or credible sources."
    )

    evaluator.add_custom_node(
        result=bool(_safe_str(item.technical_overview).strip()),
        id=f"technical_overview_{theater_num}",
        desc="Provide a technical capabilities overview (rigging, sound, lighting) as requested",
        parent=tech_node,
        critical=True
    )

    # Parking
    park_node = evaluator.add_parallel(
        id=f"parking_{theater_num}",
        desc="Parking availability information",
        parent=theater_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_str(item.parking_info).strip()),
        id=f"parking_info_{theater_num}",
        desc="Provide parking availability information (on-site, nearby, or public parking options)",
        parent=park_node,
        critical=True
    )

    # Use and touring suitability
    use_node = evaluator.add_parallel(
        id=f"use_and_touring_{theater_num}",
        desc="Primary use and suitability for touring productions",
        parent=theater_node,
        critical=True
    )
    primary_use_leaf = evaluator.add_leaf(
        id=f"primary_use_performing_arts_{theater_num}",
        desc="Venue’s primary use is live performing arts (theater/dance/concerts/musicals), not exclusively cinema/other",
        parent=use_node,
        critical=True
    )
    primary_use_claim = "The venue’s primary use is live performing arts (theater, dance, concerts, or musicals), not exclusively cinema or other non-performing uses."
    await evaluator.verify(
        claim=primary_use_claim,
        node=primary_use_leaf,
        sources=item.reference_urls,
        additional_instruction="Verify mission/season programming or venue description indicating primary performing arts use."
    )

    touring_leaf = evaluator.add_leaf(
        id=f"touring_suitable_{theater_num}",
        desc="Venue is suitable for hosting professional touring productions (professional standards/comprehensive facilities)",
        parent=use_node,
        critical=True
    )
    touring_claim = "The venue is suitable for hosting professional touring productions (meets professional standards and has comprehensive facilities)."
    await evaluator.verify(
        claim=touring_claim,
        node=touring_leaf,
        sources=item.reference_urls,
        additional_instruction="Look for rental/touring info, prior touring shows hosted, or specs indicating suitability for touring productions."
    )

    # Record some parsed numeric info for transparency
    evaluator.add_custom_info(
        info={
            "theater_index": theater_num,
            "name": item.name,
            "parsed_capacity_int": cap_val,
            "parsed_proscenium_width_ft": width_ft,
            "parsed_stage_depth_ft": depth_ft,
            "parsed_wheelchair_spaces_int": wc_val
        },
        info_type="parsed_numbers",
        info_name=f"parsed_numbers_theater_{theater_num}"
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the agent's answer for the U.S. theaters task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # The three theaters are independent
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

    # Extract theaters data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_theaters(),
        template_class=TheatersExtraction,
        extraction_name="theaters_extraction",
    )

    # Prepare up to 3 theaters; pad if fewer
    theaters: List[TheaterItem] = list(extracted.theaters[:3])
    while len(theaters) < 3:
        theaters.append(TheaterItem())

    # Build verification subtrees for each theater
    for i in range(3):
        await verify_theater(evaluator, root, i, theaters[i])

    # Return the summary (includes verification tree and recorded info)
    return evaluator.get_summary()