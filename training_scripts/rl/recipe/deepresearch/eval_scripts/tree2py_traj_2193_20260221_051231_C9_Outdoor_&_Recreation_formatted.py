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
TASK_ID = "dark_sky_campgrounds_4"
TASK_DESCRIPTION = """
Identify 4 different developed campgrounds in the United States that meet ALL of the following requirements:

General Requirements (all 4 campgrounds):
- Located within certified International Dark Sky Parks or International Dark Sky Reserves
- Have developed campground facilities with designated campsites, picnic tables, and fire rings
- Accept reservations through an online booking system
- Offer astronomy programs or stargazing events during summer months (Memorial Day through Labor Day period)

Specific Requirements:
- Campground 1: Any qualifying campground meeting the general requirements
- Campground 2: Must be located in a park with Gold Tier International Dark Sky Park certification (the highest darkness rating)
- Campground 3: Must be located at an elevation above 5,000 feet
- Campground 4: Must be in a park that offers ranger-led telescope viewing programs during summer months, and the campground must have flush toilet facilities

For each campground, provide:
1. Campground name and the national/state park where it is located
2. Dark sky certification status (type and tier if applicable)
3. Campground facilities details: restroom type, maximum group capacity per site, RV/trailer length restrictions
4. Astronomy program information: type of program and schedule/frequency
5. Reservation system: booking platform name and advance booking window
6. Reference URLs supporting each piece of information
"""

SUMMER_MONTHS_GUIDANCE = (
    "Summer months here refer to the period around Memorial Day through Labor Day (roughly late May through early September). "
    "Programs described as 'summer', 'June–August', 'Memorial Day to Labor Day', or specific summer schedules should be considered compliant."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Certification(BaseModel):
    certification_type: Optional[str] = None  # e.g., "International Dark Sky Park", "International Dark Sky Reserve"
    tier: Optional[str] = None  # e.g., "Gold", "Silver", "Bronze"
    cert_urls: List[str] = Field(default_factory=list)


class Facilities(BaseModel):
    has_picnic_tables_and_fire_rings: Optional[str] = None  # free-text confirmation (e.g., "Yes, at all sites")
    toilet_type: Optional[str] = None  # e.g., "flush toilets", "vault toilets"
    group_capacity_per_site: Optional[str] = None  # e.g., "6 people", "8 persons", "two vehicles"
    rv_trailer_length_restrictions: Optional[str] = None  # e.g., "Max 30 ft", "Up to 25 ft"
    facilities_urls: List[str] = Field(default_factory=list)


class Astronomy(BaseModel):
    program_type: Optional[str] = None  # e.g., "Ranger-led stargazing", "Telescope viewing", "Night sky programs"
    schedule: Optional[str] = None  # e.g., "Fridays in summer", "Weekly June–August"
    astronomy_urls: List[str] = Field(default_factory=list)


class Reservation(BaseModel):
    platform_name: Optional[str] = None  # e.g., "Recreation.gov", "ReserveCalifornia", "State Parks booking"
    advance_booking_window: Optional[str] = None  # e.g., "6 months in advance", "up to 90 days ahead"
    reservation_urls: List[str] = Field(default_factory=list)


class ElevationInfo(BaseModel):
    elevation_ft: Optional[str] = None  # Keep as string; answer may include ranges or approximations
    elevation_urls: List[str] = Field(default_factory=list)


class Campground(BaseModel):
    name: Optional[str] = None
    park: Optional[str] = None  # National or State park name
    certification: Certification = Field(default_factory=Certification)
    facilities: Facilities = Field(default_factory=Facilities)
    astronomy: Astronomy = Field(default_factory=Astronomy)
    reservation: Reservation = Field(default_factory=Reservation)
    elevation: ElevationInfo = Field(default_factory=ElevationInfo)  # Used especially for Campground 3


class CampgroundsExtraction(BaseModel):
    campgrounds: List[Campground] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
Extract up to 4 campgrounds described in the answer that meet the requested criteria. Return a JSON object with a field 'campgrounds' which is an array of at most 4 campground objects. If the answer lists more than 4 campgrounds, extract only the first 4. If fewer are provided, extract those available.

For each campground, extract the following fields as shown. Use strings for values when uncertain; do NOT invent information that is not in the answer text.

Campground object fields:
- name: The official campground name
- park: The national or state park in which the campground is located
- certification:
  - certification_type: e.g., "International Dark Sky Park" or "International Dark Sky Reserve"
  - tier: the tier level if stated (e.g., "Gold", "Silver", "Bronze"); set to null if not mentioned
  - cert_urls: list of URLs cited in the answer that confirm the park’s dark sky certification
- facilities:
  - has_picnic_tables_and_fire_rings: a short string confirming these amenities if stated; set to null if not mentioned
  - toilet_type: e.g., "flush toilets", "vault toilets"; set to null if not mentioned
  - group_capacity_per_site: e.g., "6 people", "8 persons"; set to null if not mentioned
  - rv_trailer_length_restrictions: any RV/trailer length limits; set to null if not mentioned
  - facilities_urls: list of URLs cited in the answer that support the facilities information
- astronomy:
  - program_type: what kind of astronomy program exists (e.g., "ranger-led telescope viewing", "night sky talks")
  - schedule: summary of schedule or frequency (e.g., "weekly in summer", "June–August")
  - astronomy_urls: list of URLs cited in the answer that support the astronomy program information
- reservation:
  - platform_name: the booking platform used (e.g., "Recreation.gov", state reservation system)
  - advance_booking_window: how far in advance users can book (e.g., "6 months", "up to 90 days")
  - reservation_urls: list of URLs cited in the answer for reservation information
- elevation:
  - elevation_ft: the campground elevation in feet if mentioned (e.g., "5200 ft", "~6000 feet"); use string and preserve any qualifiers; set to null if not mentioned
  - elevation_urls: list of URLs cited in the answer that support the elevation

SPECIAL RULES FOR URL EXTRACTION:
- Only include URLs that are explicitly present in the answer as evidence for that field (or category).
- If an URL appears relevant to multiple fields (e.g., a park page that includes both facilities and astronomy info), include it in both URL lists as appropriate.
- Do not invent or infer URLs not present in the answer text.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def all_sources_for_identification(cg: Campground) -> List[str]:
    # Combine all available URLs to help verify name/park identification
    urls = []
    urls += cg.certification.cert_urls
    urls += cg.facilities.facilities_urls
    urls += cg.astronomy.astronomy_urls
    urls += cg.reservation.reservation_urls
    urls += cg.elevation.elevation_urls
    return _dedup(urls)


# --------------------------------------------------------------------------- #
# Subtree builders per campground                                             #
# --------------------------------------------------------------------------- #
async def build_identification_subtree(
    evaluator: Evaluator,
    parent,
    prefix: str,
    cg: Campground,
):
    ident_node = evaluator.add_parallel(
        id=f"{prefix}_identification",
        desc="Provide the campground name and the national/state park where it is located",
        parent=parent,
        critical=True,
    )

    # Leaf: campground name (verify with any available URLs)
    name_leaf = evaluator.add_leaf(
        id=f"{prefix}_campground_name",
        desc="Specify the campground name",
        parent=ident_node,
        critical=True,
    )
    name_claim = f"The official campground name is '{cg.name or ''}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=all_sources_for_identification(cg),
        additional_instruction="Verify that the cited sources mention this campground by this name or a clearly equivalent variant. Allow minor formatting differences (e.g., with or without the word 'Campground').",
    )

    # Leaf: park location (verify)
    park_leaf = evaluator.add_leaf(
        id=f"{prefix}_park_location",
        desc="Specify the national/state park location",
        parent=ident_node,
        critical=True,
    )
    park_claim = f"This campground is located within '{cg.park or ''}'."
    await evaluator.verify(
        claim=park_claim,
        node=park_leaf,
        sources=all_sources_for_identification(cg),
        additional_instruction="Verify that the sources indicate this campground lies within the named national or state park. Allow minor variations in park naming (e.g., abbreviations).",
    )


async def build_certification_subtree_general(
    evaluator: Evaluator,
    parent,
    prefix: str,
    cg: Campground,
):
    cert_node = evaluator.add_parallel(
        id=f"{prefix}_dark_sky_certification",
        desc="The park containing this campground must be certified as an International Dark Sky Park (Gold, Silver, or Bronze tier) or be part of an International Dark Sky Reserve",
        parent=parent,
        critical=True,
    )

    # Leaf: cert details (type and tier if any)
    cert_details_leaf = evaluator.add_leaf(
        id=f"{prefix}_cert_details",
        desc="Specify the certification type and tier level",
        parent=cert_node,
        critical=True,
    )
    cert_type = cg.certification.certification_type or ""
    tier = cg.certification.tier or ""
    cert_claim = f"The park is recognized as '{cert_type}' with tier '{tier}'. If no tier is publicly specified, the page may omit it."
    await evaluator.verify(
        claim=cert_claim,
        node=cert_details_leaf,
        sources=cg.certification.cert_urls,
        additional_instruction="Confirm that the provided sources show that the park is an International Dark Sky Park or part of a Dark Sky Reserve. If a tier is claimed, confirm the tier (Gold/Silver/Bronze). If no tier is given, the claim should not assert a specific tier.",
    )

    # Leaf: cert URL existence
    cert_url_exist = evaluator.add_custom_node(
        result=bool(cg.certification.cert_urls),
        id=f"{prefix}_cert_url",
        desc="Provide URL reference confirming the dark sky certification",
        parent=cert_node,
        critical=True,
    )


async def build_certification_subtree_gold(
    evaluator: Evaluator,
    parent,
    prefix: str,
    cg: Campground,
):
    gold_node = evaluator.add_parallel(
        id=f"{prefix}_gold_tier_requirement",
        desc="This campground must be located in a park with Gold Tier International Dark Sky Park certification (the highest darkness rating)",
        parent=parent,
        critical=True,
    )

    gold_leaf = evaluator.add_leaf(
        id=f"{prefix}_gold_confirmation",
        desc="Confirm Gold Tier certification status",
        parent=gold_node,
        critical=True,
    )
    gold_claim = "This park holds an International Dark Sky Park certification at the Gold Tier (the highest darkness rating)."
    await evaluator.verify(
        claim=gold_claim,
        node=gold_leaf,
        sources=cg.certification.cert_urls,
        additional_instruction="Verify that the sources explicitly indicate 'Gold Tier' for the park’s International Dark Sky Park designation. Allow synonyms like 'Gold-tier' or 'Gold'.",
    )

    gold_url_exist = evaluator.add_custom_node(
        result=bool(cg.certification.cert_urls),
        id=f"{prefix}_gold_url",
        desc="Provide URL reference confirming Gold Tier designation",
        parent=gold_node,
        critical=True,
    )


async def build_facilities_subtree_generic(
    evaluator: Evaluator,
    parent,
    prefix: str,
    cg: Campground,
    require_flush_toilets_explicit: bool = False,
):
    fac_node = evaluator.add_parallel(
        id=f"{prefix}_campground_facilities",
        desc="The campground must have developed facilities including designated campsites with picnic tables and fire rings",
        parent=parent,
        critical=True,
    )

    # Basic amenities (picnic tables and fire rings)
    basic_leaf = evaluator.add_leaf(
        id=f"{prefix}_basic_amenities",
        desc="Confirm the campground has picnic tables and fire rings at campsites",
        parent=fac_node,
        critical=True,
    )
    basic_claim = "Campsites at this campground have picnic tables and fire rings (or fire pits)."
    await evaluator.verify(
        claim=basic_claim,
        node=basic_leaf,
        sources=cg.facilities.facilities_urls,
        additional_instruction="Verify that the facilities description explicitly lists both picnic tables and fire rings (or fire pits) at individual campsites.",
    )

    if require_flush_toilets_explicit:
        flush_leaf = evaluator.add_leaf(
            id=f"{prefix}_flush_toilets",
            desc="Confirm that flush toilet facilities are available at the campground",
            parent=fac_node,
            critical=True,
        )
        flush_claim = "Flush toilet facilities are available at this campground."
        await evaluator.verify(
            claim=flush_claim,
            node=flush_leaf,
            sources=cg.facilities.facilities_urls,
            additional_instruction="Confirm that the campground facility information explicitly mentions 'flush toilets'. If only vault/pit toilets are mentioned, this should be considered not supported.",
        )
    else:
        toilet_leaf = evaluator.add_leaf(
            id=f"{prefix}_toilet_type",
            desc="Specify the restroom facilities type (flush toilets or vault toilets)",
            parent=fac_node,
            critical=True,
        )
        toilet_claim = f"The restroom facilities type for this campground is described as '{cg.facilities.toilet_type or ''}'."
        await evaluator.verify(
            claim=toilet_claim,
            node=toilet_leaf,
            sources=cg.facilities.facilities_urls,
            additional_instruction="Verify the restroom/toilet type on the official campground page or reservation page (e.g., flush toilets vs. vault toilets).",
        )

    # Group capacity
    group_leaf = evaluator.add_leaf(
        id=f"{prefix}_group_capacity",
        desc="Specify the maximum number of people allowed per standard campsite",
        parent=fac_node,
        critical=True,
    )
    group_claim = f"The maximum number of people allowed per standard campsite is '{cg.facilities.group_capacity_per_site or ''}'."
    await evaluator.verify(
        claim=group_claim,
        node=group_leaf,
        sources=cg.facilities.facilities_urls,
        additional_instruction="Verify the stated standard campsite occupancy limit. Allow minor formatting differences (e.g., 'six people' vs '6 people').",
    )

    # RV/trailer length restrictions
    rv_leaf = evaluator.add_leaf(
        id=f"{prefix}_rv_restrictions",
        desc="Document any RV or trailer length restrictions for the campground",
        parent=fac_node,
        critical=True,
    )
    rv_claim = f"RV/trailer length restriction information for this campground is: '{cg.facilities.rv_trailer_length_restrictions or ''}'."
    await evaluator.verify(
        claim=rv_claim,
        node=rv_leaf,
        sources=cg.facilities.facilities_urls,
        additional_instruction="Verify any stated maximum vehicle/RV/trailer lengths for campsites or loops, allowing ranges (e.g., 'up to 25–30 ft') and per-loop differences.",
    )

    # Facilities URLs existence
    fac_url_exist = evaluator.add_custom_node(
        result=bool(cg.facilities.facilities_urls),
        id=f"{prefix}_facilities_url",
        desc="Provide URL reference for campground facilities information",
        parent=fac_node,
        critical=True,
    )


async def build_astronomy_subtree(
    evaluator: Evaluator,
    parent,
    prefix: str,
    cg: Campground,
):
    astro_node = evaluator.add_parallel(
        id=f"{prefix}_astronomy_programs",
        desc="The park must offer astronomy programs or stargazing events during summer months (Memorial Day through Labor Day)",
        parent=parent,
        critical=True,
    )

    # Program type
    prog_type_leaf = evaluator.add_leaf(
        id=f"{prefix}_program_type",
        desc="Describe the type of astronomy program offered (ranger-led talks, telescope viewing, etc.)",
        parent=astro_node,
        critical=True,
    )
    prog_type_claim = f"The astronomy/night-sky program type is described as '{cg.astronomy.program_type or ''}'."
    await evaluator.verify(
        claim=prog_type_claim,
        node=prog_type_leaf,
        sources=cg.astronomy.astronomy_urls,
        additional_instruction="Verify the nature of the program (e.g., ranger-led talks, star parties, telescope viewing) as described on the cited source(s).",
    )

    # Program schedule (emphasize summer months)
    prog_sched_leaf = evaluator.add_leaf(
        id=f"{prefix}_program_schedule",
        desc="Specify when these programs are typically offered (days of week, frequency)",
        parent=astro_node,
        critical=True,
    )
    prog_sched_claim = f"The astronomy program schedule indicates summer-month availability: '{cg.astronomy.schedule or ''}'."
    await evaluator.verify(
        claim=prog_sched_claim,
        node=prog_sched_leaf,
        sources=cg.astronomy.astronomy_urls,
        additional_instruction=(
            "Confirm that the programs occur during summer months (Memorial Day through Labor Day), even if phrased as 'summer' or given as typical months such as June–August. "
            + SUMMER_MONTHS_GUIDANCE
        ),
    )

    # Astronomy URLs presence
    astro_url_exist = evaluator.add_custom_node(
        result=bool(cg.astronomy.astronomy_urls),
        id=f"{prefix}_program_url",
        desc="Provide URL reference for astronomy program information",
        parent=astro_node,
        critical=True,
    )


async def build_reservation_subtree(
    evaluator: Evaluator,
    parent,
    prefix: str,
    cg: Campground,
):
    res_node = evaluator.add_parallel(
        id=f"{prefix}_reservation_system",
        desc="The campground must accept reservations through an online booking system",
        parent=parent,
        critical=True,
    )

    platform_leaf = evaluator.add_leaf(
        id=f"{prefix}_booking_platform",
        desc="Identify the reservation platform used (Recreation.gov, state park system, etc.)",
        parent=res_node,
        critical=True,
    )
    platform_claim = f"Reservations for this campground are made via '{cg.reservation.platform_name or ''}'."
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=cg.reservation.reservation_urls,
        additional_instruction="Verify the named reservation/booking platform as shown on the provided URLs (e.g., Recreation.gov, state reservation system).",
    )

    window_leaf = evaluator.add_leaf(
        id=f"{prefix}_advance_window",
        desc="Specify how far in advance reservations can be made",
        parent=res_node,
        critical=True,
    )
    window_claim = f"Reservations can be made '{cg.reservation.advance_booking_window or ''}' in advance."
    await evaluator.verify(
        claim=window_claim,
        node=window_leaf,
        sources=cg.reservation.reservation_urls,
        additional_instruction="Confirm the stated advance booking window if available (e.g., '6 months ahead', 'up to 90 days'). Allow equivalent phrasing.",
    )

    res_url_exist = evaluator.add_custom_node(
        result=bool(cg.reservation.reservation_urls),
        id=f"{prefix}_reservation_url",
        desc="Provide URL reference for reservation information",
        parent=res_node,
        critical=True,
    )


async def build_elevation_subtree(
    evaluator: Evaluator,
    parent,
    prefix: str,
    cg: Campground,
):
    elev_node = evaluator.add_parallel(
        id=f"{prefix}_elevation_requirement",
        desc="This campground must be located at an elevation above 5,000 feet",
        parent=parent,
        critical=True,
    )

    elev_value_leaf = evaluator.add_leaf(
        id=f"{prefix}_elevation_value",
        desc="Specify the campground elevation in feet",
        parent=elev_node,
        critical=True,
    )
    # We verify the threshold requirement explicitly in this leaf (above 5,000 ft),
    # while referencing the extracted elevation string for clarity.
    elev_str = cg.elevation.elevation_ft or ""
    elev_value_claim = f"The campground's elevation is above 5,000 feet (reported as '{elev_str}')."
    await evaluator.verify(
        claim=elev_value_claim,
        node=elev_value_leaf,
        sources=cg.elevation.elevation_urls,
        additional_instruction="Verify that the provided sources indicate an elevation exceeding 5,000 ft for the campground area. If an approximate elevation is given (e.g., '~5200 ft'), that counts as above 5,000 ft.",
    )

    elev_url_exist = evaluator.add_custom_node(
        result=bool(cg.elevation.elevation_urls),
        id=f"{prefix}_elevation_url",
        desc="Provide URL reference confirming the elevation",
        parent=elev_node,
        critical=True,
    )


async def build_telescope_requirement_subtree(
    evaluator: Evaluator,
    parent,
    prefix: str,
    cg: Campground,
):
    tel_node = evaluator.add_parallel(
        id=f"{prefix}_telescope_requirement",
        desc="This campground must be in a park that offers ranger-led telescope viewing programs during summer months",
        parent=parent,
        critical=True,
    )

    tel_leaf = evaluator.add_leaf(
        id=f"{prefix}_telescope_confirmation",
        desc="Confirm that ranger-led telescope viewing programs are offered",
        parent=tel_node,
        critical=True,
    )
    tel_claim = "The park offers ranger-led telescope viewing programs during the summer months."
    await evaluator.verify(
        claim=tel_claim,
        node=tel_leaf,
        sources=cg.astronomy.astronomy_urls,
        additional_instruction="Confirm that the sources explicitly mention ranger-led telescope viewing programs (or equivalent) occurring in summer months. "
                              + SUMMER_MONTHS_GUIDANCE,
    )

    tel_url_exist = evaluator.add_custom_node(
        result=bool(cg.astronomy.astronomy_urls),
        id=f"{prefix}_telescope_url",
        desc="Provide URL reference for telescope viewing program information",
        parent=tel_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Campground builders (per specific requirement)                              #
# --------------------------------------------------------------------------- #
async def build_campground_1(
    evaluator: Evaluator,
    root,
    cg: Campground,
):
    node = evaluator.add_parallel(
        id="campground_1",
        desc="First qualifying dark sky campground with all required attributes",
        parent=root,
        critical=False,
    )
    await build_identification_subtree(evaluator, node, "c1", cg)
    await build_certification_subtree_general(evaluator, node, "c1", cg)
    await build_facilities_subtree_generic(evaluator, node, "c1", cg, require_flush_toilets_explicit=False)
    await build_astronomy_subtree(evaluator, node, "c1", cg)
    await build_reservation_subtree(evaluator, node, "c1", cg)


async def build_campground_2_gold(
    evaluator: Evaluator,
    root,
    cg: Campground,
):
    node = evaluator.add_parallel(
        id="campground_2",
        desc="Second qualifying dark sky campground meeting the Gold Tier requirement",
        parent=root,
        critical=False,
    )
    await build_identification_subtree(evaluator, node, "c2", cg)
    await build_certification_subtree_gold(evaluator, node, "c2", cg)
    await build_facilities_subtree_generic(evaluator, node, "c2", cg, require_flush_toilets_explicit=False)
    await build_astronomy_subtree(evaluator, node, "c2", cg)
    await build_reservation_subtree(evaluator, node, "c2", cg)


async def build_campground_3_elevation(
    evaluator: Evaluator,
    root,
    cg: Campground,
):
    node = evaluator.add_parallel(
        id="campground_3",
        desc="Third qualifying dark sky campground with high elevation requirement",
        parent=root,
        critical=False,
    )
    await build_identification_subtree(evaluator, node, "c3", cg)
    await build_elevation_subtree(evaluator, node, "c3", cg)
    await build_certification_subtree_general(evaluator, node, "c3", cg)
    await build_facilities_subtree_generic(evaluator, node, "c3", cg, require_flush_toilets_explicit=False)
    await build_astronomy_subtree(evaluator, node, "c3", cg)
    await build_reservation_subtree(evaluator, node, "c3", cg)


async def build_campground_4_telescope_flush(
    evaluator: Evaluator,
    root,
    cg: Campground,
):
    node = evaluator.add_parallel(
        id="campground_4",
        desc="Fourth qualifying dark sky campground with telescope viewing programs and flush toilets",
        parent=root,
        critical=False,
    )
    await build_identification_subtree(evaluator, node, "c4", cg)
    await build_telescope_requirement_subtree(evaluator, node, "c4", cg)  # telescope requirement
    await build_certification_subtree_general(evaluator, node, "c4", cg)
    await build_facilities_subtree_generic(evaluator, node, "c4", cg, require_flush_toilets_explicit=True)
    await build_astronomy_subtree(evaluator, node, "c4", cg)
    await build_reservation_subtree(evaluator, node, "c4", cg)


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
    Evaluate an answer for the dark sky campgrounds task using the Mind2Web2 framework.
    """

    # Initialize evaluator (root is non-critical to allow partial credit across campgrounds)
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

    # Extract up to 4 campgrounds
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds_structured",
    )

    # Normalize to exactly 4 items (pad with empty placeholders if fewer)
    campgrounds: List[Campground] = list(extracted.campgrounds[:4])
    while len(campgrounds) < 4:
        campgrounds.append(Campground())

    evaluator.add_custom_info(
        info={
            "extracted_count": len(extracted.campgrounds),
            "evaluated_count": 4,
        },
        info_type="meta",
        info_name="extraction_counts",
    )

    # Build and verify each campground subtree according to specific requirements
    await build_campground_1(evaluator, root, campgrounds[0])
    await build_campground_2_gold(evaluator, root, campgrounds[1])
    await build_campground_3_elevation(evaluator, root, campgrounds[2])
    await build_campground_4_telescope_flush(evaluator, root, campgrounds[3])

    # Return evaluation summary
    return evaluator.get_summary()