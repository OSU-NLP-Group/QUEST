import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hdwgh_world_tour_2026"
TASK_DESCRIPTION = """
A solo artist who was formerly a member of One Direction announced a 2026 world arena tour on October 1, 2025. The tour, titled "How Did We Get Here? World Tour," includes concert dates across Europe, the UK, and North America.

One venue on this tour is a naturally-occurring amphitheatre located in Colorado, United States. This venue is formed by geological rock formations, has a seating capacity between 9,000 and 10,000 people, and is situated at an elevation above 6,000 feet above sea level. The concert at this venue is scheduled for a date in June 2026.

The opening act performing at this Colorado venue is a four-member band. This band originated from Utah, was formed during the period of 2008-2012, and originally performed under a different band name before adopting their current name.

Based on this information, provide the following:

1. Venue Elevation: What is the exact elevation of this Colorado venue in feet above sea level?
2. Venue Address: What is the complete street address of this venue, including street number, street name, city, state, and ZIP code?
3. Support Act's Original Name: What was the original band name that the support act performed under before their current name?
4. Support Act's Record Label: What record label is this support act currently signed to?
5. Support Act's Debut Album: What is the title of this support act's debut full-length studio album?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TourInfo(BaseModel):
    artist_name: Optional[str] = None
    tour_name: Optional[str] = None
    announcement_date: Optional[str] = None
    tour_year: Optional[str] = None
    regions: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class VenueInfo(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    capacity_text: Optional[str] = None
    elevation_feet: Optional[str] = None
    elevation_text: Optional[str] = None
    street_number: Optional[str] = None
    street_name: Optional[str] = None
    zip_code: Optional[str] = None
    street_address: Optional[str] = None
    concert_date: Optional[str] = None
    venue_urls: List[str] = Field(default_factory=list)


class SupportActInfo(BaseModel):
    band_name: Optional[str] = None
    member_count: Optional[str] = None
    origin_city: Optional[str] = None
    origin_state: Optional[str] = None
    formation_year: Optional[str] = None
    formation_period: Optional[str] = None
    original_name: Optional[str] = None
    record_label: Optional[str] = None
    debut_album: Optional[str] = None
    support_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_tour() -> str:
    return """
    Identify and extract the specific tour referenced in the answer that matches the following constraints:
    - Artist: a solo artist formerly a member of One Direction
    - Tour title: "How Did We Get Here? World Tour"
    - Announcement date: October 1, 2025
    - Tour year: 2026
    - Regions: includes dates in Europe, the UK, and North America

    Extract the following fields exactly as presented in the answer text:
    - artist_name: the solo artist's full name
    - tour_name: the tour title
    - announcement_date: the official announcement date (e.g., "October 1, 2025")
    - tour_year: the year the tour takes place (e.g., "2026")
    - regions: an array of regions explicitly mentioned (e.g., ["Europe", "UK", "North America"])
    - source_urls: an array of URL(s) cited in the answer that support the tour details (include official announcement pages, press releases, or news articles; only extract explicit URLs)

    If any field is missing, return null for that field; if no URLs are cited, return an empty array for source_urls.
    """


def prompt_extract_venue() -> str:
    return """
    Identify the Colorado venue on the tour and extract its attributes and event details from the answer text.
    The venue must be a naturally-occurring amphitheatre formed by geological rock formations, located in Colorado, with capacity between 9,000–10,000, elevation above 6,000 feet, and the concert is scheduled for a date in June 2026.

    Extract the following fields exactly as presented:
    - venue_name: the venue’s name (e.g., "Red Rocks Amphitheatre")
    - city: venue city
    - state: venue state (e.g., "Colorado")
    - country: venue country (e.g., "United States")
    - capacity_text: the capacity as stated (string; e.g., "9,525" or "approximately 9,500")
    - elevation_feet: the exact elevation in feet above sea level as a number string (e.g., "6450"); if a range or multiple values, choose the most authoritative or precise foot figure mentioned
    - elevation_text: the elevation as stated in the answer (string; may include units)
    - street_number: street number (e.g., "18300")
    - street_name: street name (e.g., "W Alameda Pkwy")
    - zip_code: ZIP code (e.g., "80465")
    - street_address: complete address string if provided in one line (e.g., "18300 W Alameda Pkwy, Morrison, CO 80465")
    - concert_date: the specific concert date at this venue (e.g., "June 12, 2026")
    - venue_urls: array of URL(s) cited that support the venue details and/or the specific event date at this venue

    If any field is missing in the answer, return null for that field. If no URLs are cited, return an empty array for venue_urls.
    """


def prompt_extract_support_act() -> str:
    return """
    Identify the opening/support act performing at the Colorado venue on the specified tour date and extract the band's attributes from the answer.

    Extract the following fields exactly as presented:
    - band_name: the support act's current band name
    - member_count: the number of band members (string or number; e.g., "4")
    - origin_city: the band’s origin city if mentioned
    - origin_state: the band’s origin state (e.g., "Utah")
    - formation_year: the specific year the band formed if a single year is cited (e.g., "2011")
    - formation_period: a period string if the formation is stated as a range (e.g., "2008–2012")
    - original_name: the original band name they performed under before adopting their current name
    - record_label: the record label the band is currently signed to
    - debut_album: the title of the band’s debut full-length studio album
    - support_urls: array of URL(s) cited that support the above band details and/or confirm they are the opening act for the Colorado tour date

    If any field is missing, return null for that field. If no URLs are cited, return an empty array for support_urls.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def full_address_from_parts(venue: VenueInfo) -> Optional[str]:
    if venue.street_address and venue.street_address.strip():
        return venue.street_address.strip()
    parts = []
    if venue.street_number and venue.street_number.strip():
        parts.append(venue.street_number.strip())
    if venue.street_name and venue.street_name.strip():
        parts.append(venue.street_name.strip())
    street = " ".join(parts) if parts else None
    city = venue.city.strip() if venue.city else None
    state = venue.state.strip() if venue.state else None
    zipc = venue.zip_code.strip() if venue.zip_code else None
    address_parts = []
    if street:
        address_parts.append(street)
    locality = ", ".join([p for p in [city, state] if p])
    if locality:
        address_parts.append(locality)
    if zipc:
        address_parts.append(zipc)
    return ", ".join(address_parts) if address_parts else None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_tour_identification(
    evaluator: Evaluator,
    parent_node,
    tour: TourInfo,
) -> None:
    tour_node = evaluator.add_parallel(
        id="Tour_Identification",
        desc="Identify the specific tour that matches the given tour constraints.",
        parent=parent_node,
        critical=True,
    )

    # Tour Artist Verification
    artist_leaf = evaluator.add_leaf(
        id="Tour_Artist_Verification",
        desc="The tour must be by a solo artist who was formerly a member of One Direction.",
        parent=tour_node,
        critical=True,
    )
    artist_claim = f"The artist '{tour.artist_name or ''}' is a solo artist who was formerly a member of One Direction."
    await evaluator.verify(
        claim=artist_claim,
        node=artist_leaf,
        sources=tour.source_urls,
        additional_instruction="Confirm the artist was a One Direction member and now tours as a solo act. Allow minor name variant handling.",
    )

    # Tour Year and Scope
    year_scope_leaf = evaluator.add_leaf(
        id="Tour_Year_and_Scope",
        desc="The tour must be a 2026 world tour.",
        parent=tour_node,
        critical=True,
    )
    year_scope_claim = "This tour is scheduled for 2026 and is a world tour (arena-scale)."
    await evaluator.verify(
        claim=year_scope_claim,
        node=year_scope_leaf,
        sources=tour.source_urls,
        additional_instruction="Verify that sources indicate a 2026 schedule and world/arena tour scope.",
    )

    # Tour Name
    name_leaf = evaluator.add_leaf(
        id="Tour_Name",
        desc="The tour must be titled 'How Did We Get Here? World Tour'.",
        parent=tour_node,
        critical=True,
    )
    name_claim = f"The tour is titled 'How Did We Get Here? World Tour'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=tour.source_urls,
        additional_instruction="Match the exact tour title; allow minor punctuation/casing variations.",
    )

    # Tour Announcement Date
    announce_leaf = evaluator.add_leaf(
        id="Tour_Announcement_Date",
        desc="The tour must have been officially announced on October 1, 2025.",
        parent=tour_node,
        critical=True,
    )
    announce_claim = "The tour was officially announced on October 1, 2025."
    await evaluator.verify(
        claim=announce_claim,
        node=announce_leaf,
        sources=tour.source_urls,
        additional_instruction="Check the announcement article or official post for the date Oct 1, 2025.",
    )

    # Tour Regions
    regions_leaf = evaluator.add_leaf(
        id="Tour_Regions",
        desc="The tour must include dates in Europe, the UK, and North America.",
        parent=tour_node,
        critical=True,
    )
    regions_claim = "The tour itinerary includes dates in Europe, the UK, and North America."
    await evaluator.verify(
        claim=regions_claim,
        node=regions_leaf,
        sources=tour.source_urls,
        additional_instruction="Verify that the itinerary or press materials list stops spanning Europe, the UK, and North America (UK can be listed separately).",
    )


async def verify_venue_investigation(
    evaluator: Evaluator,
    parent_node,
    venue: VenueInfo,
    tour: TourInfo,
) -> None:
    venue_node = evaluator.add_sequential(
        id="Venue_Investigation",
        desc="Identify the Colorado venue on the tour and provide the required venue attributes.",
        parent=parent_node,
        critical=True,
    )

    # Venue identification constraints (parallel critical)
    venue_ident = evaluator.add_parallel(
        id="Venue_Identification",
        desc="Identify the venue that matches all venue constraints.",
        parent=venue_node,
        critical=True,
    )

    # Location Colorado
    location_leaf = evaluator.add_leaf(
        id="Venue_Location_Colorado",
        desc="The venue must be located in Colorado, United States.",
        parent=venue_ident,
        critical=True,
    )
    loc_claim = f"The venue '{venue.venue_name or ''}' is located in Colorado, United States."
    await evaluator.verify(
        claim=loc_claim,
        node=location_leaf,
        sources=venue.venue_urls,
        additional_instruction="Confirm the venue geo-location is in Colorado, USA (e.g., Morrison, CO).",
    )

    # Type and Formation
    type_leaf = evaluator.add_leaf(
        id="Venue_Type_and_Formation",
        desc="The venue must be a naturally-occurring amphitheatre formed by geological rock formations.",
        parent=venue_ident,
        critical=True,
    )
    type_claim = f"The venue '{venue.venue_name or ''}' is a naturally-occurring amphitheatre formed by geological rock formations."
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=venue.venue_urls,
        additional_instruction="Accept phrasing like 'open-air amphitheatre built into/among rock formations'.",
    )

    # Capacity Range
    capacity_leaf = evaluator.add_leaf(
        id="Venue_Capacity_Range",
        desc="The venue must have a seating capacity between 9,000 and 10,000 people.",
        parent=venue_ident,
        critical=True,
    )
    capacity_claim = f"The venue '{venue.venue_name or ''}' has a seating capacity between 9,000 and 10,000 people."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=venue.venue_urls,
        additional_instruction="If capacity is listed as ~9,525 or ~9,500, it satisfies the 9k–10k range.",
    )

    # Elevation Constraint
    elevation_constraint_leaf = evaluator.add_leaf(
        id="Venue_Elevation_Constraint",
        desc="The venue must be located at an elevation above 6,000 feet above sea level.",
        parent=venue_ident,
        critical=True,
    )
    elev_constraint_claim = f"The venue '{venue.venue_name or ''}' is situated at an elevation above 6,000 feet above sea level."
    await evaluator.verify(
        claim=elev_constraint_claim,
        node=elevation_constraint_leaf,
        sources=venue.venue_urls,
        additional_instruction="Confirm elevation in feet exceeds 6,000; allow minor rounding.",
    )

    # Concert date June 2026
    date_constraint_leaf = evaluator.add_leaf(
        id="Venue_Concert_Date_Constraint",
        desc="The concert at this venue must be scheduled for a date in June 2026.",
        parent=venue_ident,
        critical=True,
    )
    date_phrase = venue.concert_date or "June 2026"
    date_claim = f"The concert at '{venue.venue_name or ''}' on the 'How Did We Get Here? World Tour' is scheduled for {date_phrase}, which is in June 2026."
    await evaluator.verify(
        claim=date_claim,
        node=date_constraint_leaf,
        sources=merge_urls(venue.venue_urls, tour.source_urls),
        additional_instruction="Verify the event listing or tour schedule explicitly shows a June 2026 date at the specified venue.",
    )

    # Venue required outputs (parallel critical)
    venue_outputs = evaluator.add_parallel(
        id="Venue_Required_Outputs",
        desc="Provide the required venue outputs requested by the question.",
        parent=venue_node,
        critical=True,
    )

    # Exact Elevation Output
    elevation_output_leaf = evaluator.add_leaf(
        id="Venue_Exact_Elevation_Output",
        desc="Provide the exact elevation of the venue in feet above sea level.",
        parent=venue_outputs,
        critical=True,
    )
    elevation_value = venue.elevation_feet or ""
    elevation_claim = f"The exact elevation of '{venue.venue_name or ''}' is {elevation_value} feet above sea level."
    await evaluator.verify(
        claim=elevation_claim,
        node=elevation_output_leaf,
        sources=venue.venue_urls,
        additional_instruction="Verify the precise elevation (feet). Allow minor variations if multiple sources list slightly different values.",
    )

    # Complete Address Output
    address_output_leaf = evaluator.add_leaf(
        id="Venue_Complete_Address_Output",
        desc="Provide the complete street address of the venue including street number, street name, city, state, and ZIP code.",
        parent=venue_outputs,
        critical=True,
    )
    full_addr = full_address_from_parts(venue) or ""
    address_claim = f"The complete street address of '{venue.venue_name or ''}' is: {full_addr}."
    await evaluator.verify(
        claim=address_claim,
        node=address_output_leaf,
        sources=venue.venue_urls,
        additional_instruction="Verify the full street address. Allow abbreviations (e.g., Pkwy vs Parkway) and minor formatting variations.",
    )


async def verify_support_act_investigation(
    evaluator: Evaluator,
    parent_node,
    support: SupportActInfo,
    venue: VenueInfo,
    tour: TourInfo,
) -> None:
    support_node = evaluator.add_sequential(
        id="Support_Act_Investigation",
        desc="Identify the opening/support act at the Colorado venue date and provide the required band attributes.",
        parent=parent_node,
        critical=True,
    )

    # Identification and constraints (parallel critical)
    support_ident = evaluator.add_parallel(
        id="Support_Act_Identification",
        desc="Identify the opening/support act performing at the Colorado venue on the specified tour date and verify required band constraints.",
        parent=support_node,
        critical=True,
    )

    # Identity
    identity_leaf = evaluator.add_leaf(
        id="Support_Act_Identity",
        desc="Must identify the opening/support act performing at the Colorado venue on this specific tour date.",
        parent=support_ident,
        critical=True,
    )
    identity_claim = (
        f"The opening/support act at '{venue.venue_name or ''}' on {venue.concert_date or 'June 2026'} "
        f"for the '{tour.tour_name or 'How Did We Get Here? World Tour'}' is '{support.band_name or ''}'."
    )
    await evaluator.verify(
        claim=identity_claim,
        node=identity_leaf,
        sources=merge_urls(support.support_urls, venue.venue_urls, tour.source_urls),
        additional_instruction="Confirm the bill/listing shows this band as the opening/support act for the specified venue/date.",
    )

    # Four members
    four_leaf = evaluator.add_leaf(
        id="Support_Act_Four_Members",
        desc="The support act must be a band with exactly four members.",
        parent=support_ident,
        critical=True,
    )
    four_claim = f"The band '{support.band_name or ''}' has exactly four members."
    await evaluator.verify(
        claim=four_claim,
        node=four_leaf,
        sources=support.support_urls,
        additional_instruction="Confirm member count equals 4; accepting synonyms like 'quartet'.",
    )

    # Utah origin
    origin_leaf = evaluator.add_leaf(
        id="Support_Act_Utah_Origin",
        desc="The support act must originate from Utah, United States.",
        parent=support_ident,
        critical=True,
    )
    origin_claim = f"The band '{support.band_name or ''}' originates from Utah, United States."
    await evaluator.verify(
        claim=origin_claim,
        node=origin_leaf,
        sources=support.support_urls,
        additional_instruction="Accept city-specific origins within Utah (e.g., Provo, Orem).",
    )

    # Formation period
    formation_leaf = evaluator.add_leaf(
        id="Support_Act_Formation_Period",
        desc="The support act must have been formed during the period 2008–2012.",
        parent=support_ident,
        critical=True,
    )
    formation_phrase = support.formation_year or support.formation_period or "between 2008 and 2012"
    formation_claim = f"The band '{support.band_name or ''}' was formed {formation_phrase}, which falls within 2008–2012."
    await evaluator.verify(
        claim=formation_claim,
        node=formation_leaf,
        sources=support.support_urls,
        additional_instruction="If a single formation year (e.g., 2011) is shown, it satisfies the 2008–2012 window.",
    )

    # Original name output
    original_name_leaf = evaluator.add_leaf(
        id="Support_Act_Original_Name_Output",
        desc="Must identify the original band name the support act performed under before adopting their current name.",
        parent=support_ident,
        critical=True,
    )
    original_name_claim = f"The band '{support.band_name or ''}' originally performed under the name '{support.original_name or ''}'."
    await evaluator.verify(
        claim=original_name_claim,
        node=original_name_leaf,
        sources=support.support_urls,
        additional_instruction="Verify historical references or early materials indicating the band's original name.",
    )

    # Required outputs (parallel critical)
    support_outputs = evaluator.add_parallel(
        id="Support_Act_Required_Outputs",
        desc="Provide the additional support-act outputs requested by the question.",
        parent=support_node,
        critical=True,
    )

    # Record label
    label_leaf = evaluator.add_leaf(
        id="Support_Act_Record_Label_Output",
        desc="Must identify the record label the support act is currently signed to.",
        parent=support_outputs,
        critical=True,
    )
    label_claim = f"The band '{support.band_name or ''}' is currently signed to '{support.record_label or ''}'."
    await evaluator.verify(
        claim=label_claim,
        node=label_leaf,
        sources=support.support_urls,
        additional_instruction="Confirm the current record label. Accept official biographies, label roster pages, or reliable discography references.",
    )

    # Debut album
    debut_leaf = evaluator.add_leaf(
        id="Support_Act_Debut_Album_Output",
        desc="Must identify the title of the support act's debut full-length studio album.",
        parent=support_outputs,
        critical=True,
    )
    debut_claim = f"The debut full-length studio album of '{support.band_name or ''}' is titled '{support.debut_album or ''}'."
    await evaluator.verify(
        claim=debut_claim,
        node=debut_leaf,
        sources=support.support_urls,
        additional_instruction="Verify using reliable sources (e.g., official discography, Wikipedia). Allow minor punctuation/casing variations.",
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
    Evaluate the agent's answer for the 'How Did We Get Here? World Tour' concert investigation.
    """
    evaluator = Evaluator()
    concert_root = evaluator.initialize(
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

    # Add the top-level node for this rubric tree (as child of evaluator.root)
    root_node = evaluator.add_sequential(
        id="Concert_Investigation",
        desc="Identify the correct tour, venue, and support act that match the given constraints, and provide the requested venue/support-act attributes.",
        parent=concert_root,
        critical=False,
    )

    # --------------------- Extraction --------------------- #
    tour_info = await evaluator.extract(
        prompt=prompt_extract_tour(),
        template_class=TourInfo,
        extraction_name="tour_info",
    )

    venue_info = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueInfo,
        extraction_name="venue_info",
    )

    support_info = await evaluator.extract(
        prompt=prompt_extract_support_act(),
        template_class=SupportActInfo,
        extraction_name="support_act_info",
    )

    # --------------------- Verification Tree --------------------- #
    # 1) Tour Identification (parallel, critical)
    await verify_tour_identification(evaluator, root_node, tour_info)

    # 2) Venue Investigation (sequential, critical)
    await verify_venue_investigation(evaluator, root_node, venue_info, tour_info)

    # 3) Support Act Investigation (sequential, critical)
    await verify_support_act_investigation(evaluator, root_node, support_info, venue_info, tour_info)

    # --------------------- Return Summary --------------------- #
    return evaluator.get_summary()