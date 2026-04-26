import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "spring_2026_ca_music_festivals"
TASK_DESCRIPTION = """
Identify three distinct music festivals taking place in California during spring 2026 (March 1 - May 31, 2026) that meet all of the following requirements:

1. Duration: The festival must span at least 3 consecutive days
2. Location: The festival must be held in California, and the venue must have a complete physical street address (including street number, street name, city, state, and ZIP code) publicly available
3. Ticket Options: The festival must offer at least two different ticket tier options (such as General Admission and VIP, or similar variations) with clearly different access levels or benefits
4. Musical Focus: The event must be primarily marketed and organized as a music festival (not a food festival, cultural event, or other non-music-focused event)
5. Multiple Stages: The festival must feature at least 3 stages or performance areas
6. Official Documentation: The festival must have an official website containing detailed event information

For each of the three festivals you identify, provide:
- Festival name
- Exact dates (start and end dates)
- Complete venue address (venue name, street address, city, state, ZIP code)
- Names of at least two ticket tiers offered
- Number of stages or performance areas
- Official website URL
- Reference URLs documenting each piece of information (dates, venue address, ticket tiers, stage count)
"""

SPRING_2026_WINDOW_TEXT = "between March 1 and May 31, 2026 (inclusive)"


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueAddress(BaseModel):
    venue_name: Optional[str] = None
    street_address: Optional[str] = None  # Expecting a full street like "123 Main St"
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None


class FestivalItem(BaseModel):
    name: Optional[str] = None
    start_date: Optional[str] = None   # Keep as string for robustness (e.g., "Apr 10-12, 2026")
    end_date: Optional[str] = None
    address: Optional[VenueAddress] = None
    ticket_tiers: List[str] = Field(default_factory=list)  # e.g., ["GA", "VIP"]
    stage_count: Optional[str] = None  # Keep as string; we will verify with URLs
    official_website_url: Optional[str] = None

    # Reference URLs (extracted from the answer) documenting specific pieces
    date_urls: List[str] = Field(default_factory=list)     # URLs supporting dates
    address_urls: List[str] = Field(default_factory=list)  # URLs supporting venue/address
    ticket_urls: List[str] = Field(default_factory=list)   # URLs supporting ticket tiers
    stages_urls: List[str] = Field(default_factory=list)   # URLs supporting stage count


class FestivalsExtraction(BaseModel):
    festivals: List[FestivalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_festivals() -> str:
    return """
    Extract up to the first three music festivals mentioned in the answer.
    For each festival, extract the following fields strictly from the answer text (do not invent):
    - name: Festival name
    - start_date: Exact start date (include month, day, and year if given)
    - end_date: Exact end date (include month, day, and year if given)
    - address: An object with:
        - venue_name
        - street_address
        - city
        - state
        - zip_code
    - ticket_tiers: An array with the names of ticket tiers (e.g., ["General Admission", "VIP"]); include at least two if available
    - stage_count: Stated number of stages or performance areas (string)
    - official_website_url: The official website URL for the festival

    Also extract reference URLs (only those explicitly present in the answer) documenting each specific piece:
    - date_urls: URLs that show or confirm the festival dates
    - address_urls: URLs that show or confirm the venue and full street address
    - ticket_urls: URLs that show or confirm ticket tier options and benefits
    - stages_urls: URLs that show or confirm the number of stages/performance areas

    Return a JSON object:
    {
      "festivals": [
        {
          "name": ...,
          "start_date": ...,
          "end_date": ...,
          "address": {
            "venue_name": ...,
            "street_address": ...,
            "city": ...,
            "state": ...,
            "zip_code": ...
          },
          "ticket_tiers": [...],
          "stage_count": ...,
          "official_website_url": ...,
          "date_urls": [...],
          "address_urls": [...],
          "ticket_urls": [...],
          "stages_urls": [...]
        },
        ...
      ]
    }

    Rules:
    - Extract ONLY what is explicitly present in the answer text. If a field is missing, set it to null (or [] for lists).
    - For URLs, extract only valid URLs that appear in the answer (including markdown links); do not infer.
    - Preserve the original date strings as written in the answer (do not normalize).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(*url_groups: Optional[List[str] | str]) -> List[str]:
    """Merge multiple URL groups into a unique list; ignore None/empty."""
    merged: List[str] = []
    seen = set()
    for group in url_groups:
        if not group:
            continue
        if isinstance(group, str):
            candidates = [group]
        else:
            candidates = group
        for u in candidates:
            if u and isinstance(u, str) and u.strip() and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _full_address_string(addr: Optional[VenueAddress]) -> str:
    if not addr:
        return ""
    parts = []
    if addr.street_address:
        parts.append(addr.street_address.strip())
    city_state_zip = " ".join([p.strip() for p in [addr.city or "", f"{addr.state or ''}", f"{addr.zip_code or ''}"] if p.strip()])
    if city_state_zip:
        # Prefer "City, State ZIP"
        if addr.city and addr.state:
            csz = f"{addr.city.strip()}, {addr.state.strip()}"
            if addr.zip_code:
                csz = f"{csz} {addr.zip_code.strip()}"
            parts.append(csz)
        else:
            parts.append(city_state_zip)
    return ", ".join(parts)


def _address_is_complete(addr: Optional[VenueAddress]) -> bool:
    if not addr:
        return False
    return all([
        isinstance(addr.street_address, str) and addr.street_address.strip() != "",
        isinstance(addr.city, str) and addr.city.strip() != "",
        isinstance(addr.state, str) and addr.state.strip() != "",
        isinstance(addr.zip_code, str) and addr.zip_code.strip() != "",
    ])


def _first_two_tiers(tiers: List[str]) -> List[str]:
    return [t for t in tiers if isinstance(t, str) and t.strip()][:2]


# --------------------------------------------------------------------------- #
# Verification logic for one festival                                         #
# --------------------------------------------------------------------------- #
async def verify_one_festival(
    evaluator: Evaluator,
    parent_node,
    fest: FestivalItem,
    fest_index: int,  # 1-based index for readability in node IDs
) -> None:
    """
    Build and verify the full rubric tree for one festival.
    """

    # Top-level node for this festival
    festival_node = evaluator.add_parallel(
        id=f"Festival_{fest_index}",
        desc=f"{['First','Second','Third'][fest_index-1]} qualifying festival meets all requirements",
        parent=parent_node,
        critical=False
    )

    # --------------------- Temporal Requirements (CRITICAL) ------------------ #
    temporal_node = evaluator.add_parallel(
        id=f"Festival_{fest_index}_Temporal_Requirements",
        desc="Festival occurs during spring 2026 period and meets duration requirements",
        parent=festival_node,
        critical=True
    )

    # Spring 2026 Dates (CRITICAL)
    dates_node = evaluator.add_parallel(
        id=f"Festival_{fest_index}_Spring_2026_Dates",
        desc="Festival takes place between March 1 and May 31, 2026",
        parent=temporal_node,
        critical=True
    )

    # Start date within window (leaf)
    start_leaf = evaluator.add_leaf(
        id=f"Festival_{fest_index}_Start_Date_Verification",
        desc="Festival start date is within March 1 - May 31, 2026",
        parent=dates_node,
        critical=True
    )
    start_claim = (
        f"The festival start date '{fest.start_date or ''}' is {SPRING_2026_WINDOW_TEXT}."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        additional_instruction="Interpret common date formats. Treat the boundary dates as inclusive. If the provided date is missing or not in 2026 spring window, mark as Incorrect."
    )

    # End date within window (leaf)
    end_leaf = evaluator.add_leaf(
        id=f"Festival_{fest_index}_End_Date_Verification",
        desc="Festival end date is within March 1 - May 31, 2026",
        parent=dates_node,
        critical=True
    )
    end_claim = (
        f"The festival end date '{fest.end_date or ''}' is {SPRING_2026_WINDOW_TEXT}."
    )
    await evaluator.verify(
        claim=end_claim,
        node=end_leaf,
        additional_instruction="Interpret common date formats. Treat the boundary dates as inclusive. If the provided date is missing or not in 2026 spring window, mark as Incorrect."
    )

    # Date sources existence (leaf as custom existence check)
    date_src_exists = evaluator.add_custom_node(
        result=bool(fest.date_urls and len(fest.date_urls) > 0),
        id=f"Festival_{fest_index}_Date_Source_URL",
        desc="Provide URL source documenting the festival dates",
        parent=dates_node,
        critical=True
    )

    # Multi-day Duration (CRITICAL)
    duration_node = evaluator.add_parallel(
        id=f"Festival_{fest_index}_Multi_Day_Duration",
        desc="Festival spans at least 3 consecutive days",
        parent=temporal_node,
        critical=True
    )

    # Duration count (leaf)
    duration_leaf = evaluator.add_leaf(
        id=f"Festival_{fest_index}_Duration_Count",
        desc="Festival duration is 3 or more consecutive days",
        parent=duration_node,
        critical=True
    )
    duration_claim = (
        f"Based on the provided dates ('{fest.start_date or ''}' to '{fest.end_date or ''}'), "
        f"the festival spans at least 3 consecutive days (inclusive of both start and end dates)."
    )
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        additional_instruction="Assume inclusive date span. If dates are missing or the span is < 3 consecutive days, mark as Incorrect."
    )

    # Duration is supported by sources (leaf)
    duration_src_leaf = evaluator.add_leaf(
        id=f"Festival_{fest_index}_Duration_Source_URL",
        desc="Provide URL source documenting the festival duration",
        parent=duration_node,
        critical=True
    )
    duration_src_claim = (
        f"The cited source(s) confirm the festival runs for at least 3 consecutive days between '{fest.start_date or ''}' and '{fest.end_date or ''}'."
    )
    await evaluator.verify(
        claim=duration_src_claim,
        node=duration_src_leaf,
        sources=_safe_urls(fest.date_urls),
        additional_instruction="Verify from the page(s) that the event clearly spans 3 or more consecutive days."
    )

    # --------------------- Location Requirements (CRITICAL) ------------------ #
    location_node = evaluator.add_parallel(
        id=f"Festival_{fest_index}_Location_Requirements",
        desc="Festival is located in California with complete venue documentation",
        parent=festival_node,
        critical=True
    )

    # California Location (CRITICAL)
    ca_loc_node = evaluator.add_parallel(
        id=f"Festival_{fest_index}_California_Location",
        desc="Festival takes place in the state of California, USA",
        parent=location_node,
        critical=True
    )
    state_leaf = evaluator.add_leaf(
        id=f"Festival_{fest_index}_State_Verification",
        desc="Venue is confirmed to be in California",
        parent=ca_loc_node,
        critical=True
    )
    city_part = fest.address.city if fest.address and fest.address.city else ""
    state_claim = (
        f"The festival takes place in {city_part+', ' if city_part else ''}California, USA."
    )
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=_safe_urls(fest.address_urls, fest.date_urls, fest.official_website_url),
        additional_instruction="Confirm that the venue/event location is in the state of California."
    )

    loc_src_exists = evaluator.add_custom_node(
        result=bool(fest.address_urls and len(fest.address_urls) > 0),
        id=f"Festival_{fest_index}_Location_Source_URL",
        desc="Provide URL source confirming California location",
        parent=ca_loc_node,
        critical=True
    )

    # Complete Venue Address (CRITICAL)
    full_addr_node = evaluator.add_parallel(
        id=f"Festival_{fest_index}_Complete_Venue_Address",
        desc="Festival venue has a complete physical street address publicly available",
        parent=location_node,
        critical=True
    )

    # Street address completeness check (leaf as custom)
    street_complete = evaluator.add_custom_node(
        result=_address_is_complete(fest.address),
        id=f"Festival_{fest_index}_Street_Address",
        desc="Complete street address is provided (street number, street name, city, state, ZIP)",
        parent=full_addr_node,
        critical=True
    )

    # Venue name verification (leaf with URLs)
    venue_leaf = evaluator.add_leaf(
        id=f"Festival_{fest_index}_Venue_Name",
        desc="Official venue name is provided",
        parent=full_addr_node,
        critical=True
    )
    venue_claim = f"The official venue name is '{fest.address.venue_name if fest.address and fest.address.venue_name else ''}'."
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        sources=_safe_urls(fest.address_urls, fest.official_website_url),
        additional_instruction="Verify the venue's official name from the cited page(s)."
    )

    addr_src_exists = evaluator.add_custom_node(
        result=bool(fest.address_urls and len(fest.address_urls) > 0),
        id=f"Festival_{fest_index}_Address_Source_URL",
        desc="Provide URL source documenting the complete venue address",
        parent=full_addr_node,
        critical=True
    )

    # --------------------- Ticket Structure (CRITICAL) ---------------------- #
    tickets_node = evaluator.add_parallel(
        id=f"Festival_{fest_index}_Ticket_Structure",
        desc="Festival offers multiple ticket tier options",
        parent=festival_node,
        critical=True
    )

    multi_tier_node = evaluator.add_parallel(
        id=f"Festival_{fest_index}_Multiple_Ticket_Tiers",
        desc="Festival provides at least two different ticket tier options",
        parent=tickets_node,
        critical=True
    )

    # Tier Count (leaf, verify with URLs)
    tiers = _first_two_tiers(fest.ticket_tiers)
    tier_a = tiers[0] if len(tiers) > 0 else ""
    tier_b = tiers[1] if len(tiers) > 1 else ""
    tier_count_leaf = evaluator.add_leaf(
        id=f"Festival_{fest_index}_Tier_Count",
        desc="At least two distinct ticket tiers are available (e.g., GA and VIP)",
        parent=multi_tier_node,
        critical=True
    )
    tier_count_claim = (
        f"The festival offers at least two distinct ticket tiers such as '{tier_a}' and '{tier_b}'."
        if tier_a or tier_b else
        "The festival offers at least two distinct ticket tiers (e.g., General Admission and VIP)."
    )
    await evaluator.verify(
        claim=tier_count_claim,
        node=tier_count_leaf,
        sources=_safe_urls(fest.ticket_urls, fest.official_website_url),
        additional_instruction="Confirm there are 2 or more tiers. Allow synonyms (e.g., GA vs General Admission)."
    )

    # Tier Differentiation (leaf)
    tier_diff_leaf = evaluator.add_leaf(
        id=f"Festival_{fest_index}_Tier_Differentiation",
        desc="Ticket tiers have clearly different access levels or benefits",
        parent=multi_tier_node,
        critical=True
    )
    tier_diff_claim = (
        f"The '{tier_a}' and '{tier_b}' ticket tiers provide clearly different access levels or benefits (e.g., VIP perks vs General Admission)."
        if tier_a and tier_b else
        "The different ticket tiers provide clearly different access levels or benefits (e.g., VIP perks vs General Admission)."
    )
    await evaluator.verify(
        claim=tier_diff_claim,
        node=tier_diff_leaf,
        sources=_safe_urls(fest.ticket_urls, fest.official_website_url),
        additional_instruction="Look for explicit differences in benefits, access areas, amenities, or privileges across tiers."
    )

    ticket_src_exists = evaluator.add_custom_node(
        result=bool(fest.ticket_urls and len(fest.ticket_urls) > 0),
        id=f"Festival_{fest_index}_Ticket_Source_URL",
        desc="Provide URL source documenting the available ticket tiers",
        parent=multi_tier_node,
        critical=True
    )

    # ---------------- Event Characteristics (CRITICAL) ---------------------- #
    characteristics_node = evaluator.add_parallel(
        id=f"Festival_{fest_index}_Event_Characteristics",
        desc="Festival is a music-focused event with multiple performance areas",
        parent=festival_node,
        critical=True
    )

    # Music primary focus (CRITICAL)
    music_focus_node = evaluator.add_parallel(
        id=f"Festival_{fest_index}_Music_Primary_Focus",
        desc="Event is primarily a music festival, not another type of event",
        parent=characteristics_node,
        critical=True
    )

    music_focus_leaf = evaluator.add_leaf(
        id=f"Festival_{fest_index}_Music_Focus_Verification",
        desc="Festival is marketed and described as a music festival",
        parent=music_focus_node,
        critical=True
    )
    music_focus_claim = "This event is primarily marketed and described as a music festival (not just a food or cultural fair)."
    await evaluator.verify(
        claim=music_focus_claim,
        node=music_focus_leaf,
        sources=_safe_urls(fest.official_website_url, fest.date_urls),
        additional_instruction="Rely on how the event self-describes on the official site. Accept if the page emphasizes live music performances across multiple days as a primary focus."
    )

    focus_src_leaf = evaluator.add_leaf(
        id=f"Festival_{fest_index}_Focus_Source_URL",
        desc="Provide URL source confirming music festival classification",
        parent=music_focus_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cited source explicitly indicates that this event is a music festival.",
        node=focus_src_leaf,
        sources=_safe_urls(fest.official_website_url, fest.date_urls),
        additional_instruction="Look for explicit language like 'music festival', 'festival of music', etc."
    )

    # Multiple stages (CRITICAL)
    stages_node = evaluator.add_parallel(
        id=f"Festival_{fest_index}_Multiple_Stages",
        desc="Festival features at least 3 stages or performance areas",
        parent=characteristics_node,
        critical=True
    )

    stage_count_leaf = evaluator.add_leaf(
        id=f"Festival_{fest_index}_Stage_Count",
        desc="Festival has 3 or more stages/performance areas",
        parent=stages_node,
        critical=True
    )
    stage_count_claim = (
        f"The festival features at least 3 stages or performance areas (stated stage count: '{fest.stage_count or ''}')."
    )
    await evaluator.verify(
        claim=stage_count_claim,
        node=stage_count_leaf,
        sources=_safe_urls(fest.stages_urls, fest.official_website_url),
        additional_instruction="Confirm that the event lists 3 or more named stages/performance areas."
    )

    stage_src_exists = evaluator.add_custom_node(
        result=bool(fest.stages_urls and len(fest.stages_urls) > 0),
        id=f"Festival_{fest_index}_Stage_Source_URL",
        desc="Provide URL source documenting the number of stages",
        parent=stages_node,
        critical=True
    )

    # ---------------- Official Documentation (CRITICAL) --------------------- #
    official_doc_node = evaluator.add_parallel(
        id=f"Festival_{fest_index}_Official_Documentation",
        desc="Festival has an official website with detailed event information",
        parent=festival_node,
        critical=True
    )

    # Official website URL provided (leaf as existence)
    official_site_exists = evaluator.add_custom_node(
        result=bool(fest.official_website_url and fest.official_website_url.strip()),
        id=f"Festival_{fest_index}_Official_Website",
        desc="Festival has an official website URL",
        parent=official_doc_node,
        critical=True
    )

    website_content_leaf = evaluator.add_leaf(
        id=f"Festival_{fest_index}_Website_Content",
        desc="Official website contains detailed festival information (dates, location, tickets)",
        parent=official_doc_node,
        critical=True
    )
    website_content_claim = (
        "The official website contains detailed event information including festival dates, location/venue address, and ticket options."
    )
    await evaluator.verify(
        claim=website_content_claim,
        node=website_content_leaf,
        sources=_safe_urls(fest.official_website_url),
        additional_instruction="It's acceptable if details are spread across subpages on the official site (e.g., /tickets, /info, /lineup)."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'Three Spring 2026 California Music Festivals' task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # festivals evaluated independently
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

    # Optional named top-level node to mirror rubric
    top_node = evaluator.add_parallel(
        id="Three_Spring_2026_California_Music_Festivals",
        desc="Evaluation of three distinct multi-day spring 2026 California music festivals meeting all specified criteria",
        parent=root,
        critical=False
    )

    # Extract structured festival info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_festivals(),
        template_class=FestivalsExtraction,
        extraction_name="festivals_extraction"
    )

    # Normalize to exactly 3 entries
    festivals = list(extracted.festivals)[:3] if extracted and extracted.festivals else []
    while len(festivals) < 3:
        festivals.append(FestivalItem())

    # Record a small custom info block with the extracted festival names (for debugging)
    evaluator.add_custom_info(
        info={"extracted_festival_names": [f.name for f in festivals]},
        info_type="extraction_overview",
        info_name="extracted_overview"
    )

    # Build and verify the rubric tree for each of the three festivals
    for idx in range(3):
        await verify_one_festival(
            evaluator=evaluator,
            parent_node=top_node,
            fest=festivals[idx],
            fest_index=idx + 1
        )

    # Return structured evaluation summary
    return evaluator.get_summary()