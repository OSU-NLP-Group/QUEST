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
TASK_ID = "az_trip_april_2026"
TASK_DESCRIPTION = """You are an international visitor (non-U.S. resident, aged 16 or older) planning an outdoor recreation trip to Arizona from April 5-15, 2026. You will be traveling by private vehicle and plan to visit three specific national park service sites: Grand Canyon National Park (South Rim), Saguaro National Park (Tucson Mountain District/West), and Organ Pipe Cactus National Monument.

For each of these three parks/monuments, provide the following information:

1. Visitor Center Name: What is the name of the main visitor center you should visit at each location?

2. Operating Hours: What are the daily operating hours of each visitor center during your visit dates (April 5-15, 2026)?

3. Entrance Fee: What is the total entrance fee you will need to pay at each park/monument as a non-U.S. resident adult entering by private vehicle? (Include all applicable fees and surcharges that apply starting January 1, 2026.)

4. Additional Information: For each park/monument, provide one additional relevant piece of information:
   - For Grand Canyon: Name at least one specific service or facility available at the visitor center
   - For Saguaro: Explain the park's district structure (how many districts and their names)
   - For Organ Pipe Cactus: Indicate whether the monument grounds themselves are accessible 24 hours a day or have restricted access hours

Format your answer clearly with all information for each park/monument, and include reference URLs that support your findings.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkBase(BaseModel):
    visitor_center_name: Optional[str] = None
    visitor_center_hours_april_5_15_2026: Optional[str] = None
    entrance_fee_total: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class GrandCanyonInfo(ParkBase):
    visitor_services: List[str] = Field(default_factory=list)


class SaguaroInfo(ParkBase):
    district_count: Optional[str] = None
    district_names: List[str] = Field(default_factory=list)


class OrganPipeInfo(ParkBase):
    monument_access_description: Optional[str] = None
    monument_open_24_hours: Optional[str] = None  # e.g., "yes" / "no" / "unknown"


class ArizonaTripExtraction(BaseModel):
    grand_canyon: Optional[GrandCanyonInfo] = None
    saguaro: Optional[SaguaroInfo] = None
    organ_pipe: Optional[OrganPipeInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks_info() -> str:
    return """
Extract structured information from the answer for the three specified National Park Service locations. For each location, strictly extract only what is explicitly stated in the answer. If any field is missing, set it to null (or an empty list for list fields). Also extract all reference URLs that the answer cites for the given location.

Return a JSON object with the following shape:

{
  "grand_canyon": {
    "visitor_center_name": string | null,
    "visitor_center_hours_april_5_15_2026": string | null,
    "entrance_fee_total": string | null,
    "visitor_services": string[]   // e.g., "restrooms", "bookstore", "parking", "film theater", etc.
    "source_urls": string[]        // all URLs cited in the answer for Grand Canyon
  },
  "saguaro": {
    "visitor_center_name": string | null,   // Tucson Mountain District (West) visitor center name
    "visitor_center_hours_april_5_15_2026": string | null,
    "entrance_fee_total": string | null,    // a number or a range as written in the answer
    "district_count": string | null,        // e.g., "2"
    "district_names": string[],             // e.g., ["Tucson Mountain District (West)", "Rincon Mountain District (East)"]
    "source_urls": string[]                 // all URLs cited in the answer for Saguaro
  },
  "organ_pipe": {
    "visitor_center_name": string | null,
    "visitor_center_hours_april_5_15_2026": string | null,
    "entrance_fee_total": string | null,
    "monument_access_description": string | null,  // free-text description of access hours (e.g., "grounds open 24 hours", or "day-use hours only")
    "monument_open_24_hours": string | null,       // "yes" if the answer claims grounds are open 24 hours; "no" if restricted; "unknown" otherwise
    "source_urls": string[]                        // all URLs cited in the answer for Organ Pipe Cactus
  }
}

Guidelines:
- Interpret "visitor center name" as the main public visitor center relevant to the specified area (e.g., South Rim for Grand Canyon).
- For the "visitor_center_hours_april_5_15_2026", extract the hours exactly as written in the answer for that date window (or the applicable April schedule that covers those dates).
- For "entrance_fee_total", extract the total amount or range (as a string) that the answer claims applies to a non-U.S. resident adult arriving by private vehicle, including any surcharges effective January 1, 2026 (as claimed).
- For Grand Canyon, list at least one specific service/facility if provided.
- For Saguaro, extract how many districts and the names of both districts if provided.
- For Organ Pipe Cactus, extract whether the monument grounds are open 24 hours a day or have restricted hours as claimed ("monument_open_24_hours": "yes"/"no"/"unknown") and include a concise free-text description in "monument_access_description" if the answer provides one.
- For all "source_urls", include only valid URLs that are explicitly mentioned in the answer. Do not invent URLs. Deduplicate them.
    """


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def _bool_from_24h_text(text: Optional[str]) -> Optional[bool]:
    if not text:
        return None
    low = text.lower()
    tokens = ["24/7", "24-7", "24 hours", "open 24", "open 24hrs", "open 24 hr", "open all day", "open all hours"]
    negatives = ["not 24", "no 24", "closed at night", "day-use only", "day use only", "gate closes", "gates close"]
    if any(t in low for t in tokens):
        return True
    if any(n in low for n in negatives):
        return False
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_grand_canyon(evaluator: Evaluator, parent_node, info: Optional[GrandCanyonInfo]) -> None:
    node = evaluator.add_parallel(
        id="grand_canyon_national_park",
        desc="Provide visitor center name, operating hours, entrance fee, and available services for Grand Canyon National Park",
        parent=parent_node,
        critical=False
    )

    # Critical: at least one source URL provided to support claims
    sources = info.source_urls if info else []
    evaluator.add_custom_node(
        result=bool(sources),
        id="grand_canyon_sources_present",
        desc="Grand Canyon: Reference URLs are provided",
        parent=node,
        critical=True
    )

    # Leaf: Visitor Center Name (critical)
    vc_name_node = evaluator.add_leaf(
        id="grand_canyon_visitor_center_name",
        desc="Provide the name of the main visitor center at the South Rim of Grand Canyon National Park",
        parent=node,
        critical=True
    )
    vc_name = info.visitor_center_name if info else ""
    claim_vc_name = f"The main visitor center at the South Rim of Grand Canyon National Park is called '{vc_name}'."
    # Leaf: Hours (critical)
    hours_node = evaluator.add_leaf(
        id="grand_canyon_visitor_center_hours_april",
        desc="Provide the daily operating hours of the South Rim visitor center during April 5-15, 2026",
        parent=node,
        critical=True
    )
    hours = info.visitor_center_hours_april_5_15_2026 if info else ""
    claim_hours = f"During April 5–15, 2026, the South Rim visitor center is open {hours} each day."

    # Leaf: Entrance Fee (critical)
    fee_node = evaluator.add_leaf(
        id="grand_canyon_nonresident_entrance_fee",
        desc="Provide the total entrance fee (including all surcharges) for one non-U.S. resident adult aged 16+ entering by private vehicle",
        parent=node,
        critical=True
    )
    fee = info.entrance_fee_total if info else ""
    claim_fee = (
        f"The total entrance fee for a non-U.S. resident adult aged 16+ entering by private vehicle is {fee}, "
        f"inclusive of any surcharges effective January 1, 2026."
    )

    # Leaf: Visitor Services (non-critical)
    services_node = evaluator.add_leaf(
        id="grand_canyon_visitor_services",
        desc="Identify at least one specific service or facility available at the Grand Canyon Visitor Center",
        parent=node,
        critical=False
    )
    example_service = (info.visitor_services[0] if info and info.visitor_services else "").strip()
    service_claim = (
        f"The Grand Canyon Visitor Center offers the following specific service or facility: {example_service}."
        if example_service else
        "The Grand Canyon Visitor Center offers at least one specific visitor service or facility."
    )

    # Run verifications (parallel)
    await evaluator.batch_verify([
        (
            claim_vc_name,
            sources,
            vc_name_node,
            "Check official NPS pages or authoritative park sources. Allow minor naming variants (e.g., inclusion of 'South Rim')."
        ),
        (
            claim_hours,
            sources,
            hours_node,
            "Verify the visitor center hours that apply to early-to-mid April. If a page lists 'April' or 'spring' hours covering April 5–15, treat as applicable."
        ),
        (
            claim_fee,
            sources,
            fee_node,
            "Verify the entrance fee applicable to private vehicles (typically per-vehicle, valid for several days). "
            "If the source does not distinguish residents vs non-residents, accept the general private-vehicle fee."
        ),
        (
            service_claim,
            sources,
            services_node,
            "Confirm that at least one specific visitor service/facility (e.g., restrooms, bookstore, parking, theater, information desk) is offered at the visitor center."
        ),
    ])


async def verify_saguaro(evaluator: Evaluator, parent_node, info: Optional[SaguaroInfo]) -> None:
    node = evaluator.add_parallel(
        id="saguaro_national_park",
        desc="Provide visitor center name, operating hours, entrance fee, and district information for Saguaro National Park West District",
        parent=parent_node,
        critical=False
    )

    # Critical: sources present
    sources = info.source_urls if info else []
    evaluator.add_custom_node(
        result=bool(sources),
        id="saguaro_sources_present",
        desc="Saguaro: Reference URLs are provided",
        parent=node,
        critical=True
    )

    # Leaf: Visitor Center Name (critical) - West/Tucson Mountain District
    vc_name_node = evaluator.add_leaf(
        id="saguaro_west_visitor_center_name",
        desc="Provide the name of the visitor center in Saguaro National Park's Tucson Mountain District (West)",
        parent=node,
        critical=True
    )
    vc_name = info.visitor_center_name if info else ""
    claim_vc_name = f"The visitor center for Saguaro National Park's Tucson Mountain District (West) is called '{vc_name}'."

    # Leaf: Hours (critical)
    hours_node = evaluator.add_leaf(
        id="saguaro_visitor_center_hours_april",
        desc="Provide the daily operating hours of the West District visitor center during April 5-15, 2026",
        parent=node,
        critical=True
    )
    hours = info.visitor_center_hours_april_5_15_2026 if info else ""
    claim_hours = f"During April 5–15, 2026, the Saguaro West (Tucson Mountain District) visitor center is open {hours} each day."

    # Leaf: Entrance Fee (critical)
    fee_node = evaluator.add_leaf(
        id="saguaro_nonresident_entrance_fee",
        desc="Provide the total entrance fee range (including all surcharges) for one non-U.S. resident adult aged 16+",
        parent=node,
        critical=True
    )
    fee = info.entrance_fee_total if info else ""
    claim_fee = (
        f"For a non-U.S. resident adult aged 16+ arriving by private vehicle, the entrance fee is {fee}, "
        f"inclusive of any applicable surcharges effective January 1, 2026."
    )

    # Leaf: District Structure (non-critical)
    districts_node = evaluator.add_leaf(
        id="saguaro_district_structure",
        desc="Identify that Saguaro National Park consists of two separate districts and name both districts",
        parent=node,
        critical=False
    )
    count = (info.district_count.strip() if info and info.district_count else "")
    names = info.district_names if info else []
    if len(names) >= 2:
        claim_districts = (
            f"Saguaro National Park consists of {count or 'two'} districts named {names[0]} and {names[1]}."
        )
    else:
        claim_districts = (
            "Saguaro National Park consists of two districts: the Tucson Mountain District (West) and the Rincon Mountain District (East)."
        )

    await evaluator.batch_verify([
        (
            claim_vc_name,
            sources,
            vc_name_node,
            "Confirm the official name of the visitor center for the West/Tucson Mountain District. Allow minor naming variants."
        ),
        (
            claim_hours,
            sources,
            hours_node,
            "Verify hours applicable to early-to-mid April. Accept 'April' or 'spring' hour schedules that cover April 5–15."
        ),
        (
            claim_fee,
            sources,
            fee_node,
            "Verify the entrance fee applicable for private vehicles. If a range is given (e.g., motorcycle vs private vehicle), ensure the stated figure/range covers private vehicles."
        ),
        (
            claim_districts,
            sources,
            districts_node,
            "Verify that Saguaro NP has two districts and confirm their names (Tucson Mountain/West and Rincon Mountain/East). Allow minor naming variants."
        ),
    ])


async def verify_organ_pipe(evaluator: Evaluator, parent_node, info: Optional[OrganPipeInfo]) -> None:
    node = evaluator.add_parallel(
        id="organ_pipe_cactus_nm",
        desc="Provide visitor center name, operating hours, entrance fee, and access information for Organ Pipe Cactus National Monument",
        parent=parent_node,
        critical=False
    )

    # Critical: sources present
    sources = info.source_urls if info else []
    evaluator.add_custom_node(
        result=bool(sources),
        id="organ_pipe_sources_present",
        desc="Organ Pipe Cactus NM: Reference URLs are provided",
        parent=node,
        critical=True
    )

    # Leaf: Visitor Center Name (critical)
    vc_name_node = evaluator.add_leaf(
        id="organ_pipe_visitor_center_name",
        desc="Provide the full name of the visitor center at Organ Pipe Cactus National Monument",
        parent=node,
        critical=True
    )
    vc_name = info.visitor_center_name if info else ""
    claim_vc_name = f"The visitor center at Organ Pipe Cactus National Monument is called '{vc_name}'."

    # Leaf: Hours (critical)
    hours_node = evaluator.add_leaf(
        id="organ_pipe_visitor_center_hours_april",
        desc="Provide the daily operating hours of the visitor center during April 5-15, 2026",
        parent=node,
        critical=True
    )
    hours = info.visitor_center_hours_april_5_15_2026 if info else ""
    claim_hours = f"During April 5–15, 2026, the Organ Pipe Cactus NM visitor center is open {hours} each day."

    # Leaf: Entrance Fee (critical)
    fee_node = evaluator.add_leaf(
        id="organ_pipe_nonresident_entrance_fee",
        desc="Provide the total entrance fee (including all surcharges) for one non-U.S. resident adult aged 16+",
        parent=node,
        critical=True
    )
    fee = info.entrance_fee_total if info else ""
    claim_fee = (
        f"The entrance fee for a non-U.S. resident adult aged 16+ entering by private vehicle is {fee}, "
        f"including any surcharges effective January 1, 2026."
    )

    # Leaf: Monument Access Hours (non-critical)
    access_node = evaluator.add_leaf(
        id="organ_pipe_monument_access_hours",
        desc="Indicate whether the monument grounds are open 24 hours or have scheduled access hours",
        parent=node,
        critical=False
    )
    access_txt = info.monument_access_description if info else ""
    is_24h = _bool_from_24h_text(info.monument_open_24_hours if info else None)
    if is_24h is True:
        claim_access = "The monument grounds are open 24 hours a day."
    elif is_24h is False:
        claim_access = "The monument grounds do not remain open 24 hours a day; access hours are restricted."
    else:
        claim_access = f"The monument access policy is: {access_txt}".strip() or \
                       "The monument has a defined access schedule (not necessarily 24 hours)."

    await evaluator.batch_verify([
        (
            claim_vc_name,
            sources,
            vc_name_node,
            "Verify the official visitor center name; accept minor variants."
        ),
        (
            claim_hours,
            sources,
            hours_node,
            "Verify the visitor center hours that apply to early-to-mid April. Accept schedules labeled 'April' or similar."
        ),
        (
            claim_fee,
            sources,
            fee_node,
            "Verify the entrance fee for private vehicles (often per-vehicle for multiple days). Accept if the page does not distinguish residents."
        ),
        (
            claim_access,
            sources,
            access_node,
            "Check whether the monument grounds are 24/7 accessible or have specific access hours. Accept concise equivalences."
        ),
    ])


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    # Initialize evaluator (root node is non-critical parallel aggregator)
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

    # Extract structured info
    parks_info = await evaluator.extract(
        prompt=prompt_extract_parks_info(),
        template_class=ArizonaTripExtraction,
        extraction_name="parks_info"
    )

    # Optional: record trip window info
    evaluator.add_custom_info(
        info={"visit_dates": "2026-04-05 to 2026-04-15", "transport": "private vehicle", "visitor": "non-U.S. resident, 16+"},
        info_type="trip_context"
    )

    # Build verification for each park in parallel
    tasks = [
        verify_grand_canyon(evaluator, root, parks_info.grand_canyon),
        verify_saguaro(evaluator, root, parks_info.saguaro),
        verify_organ_pipe(evaluator, root, parks_info.organ_pipe),
    ]
    await asyncio.gather(*tasks)

    # Return evaluation summary
    return evaluator.get_summary()