import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "weekend_recreation_plan_sd_2026"
TASK_DESCRIPTION = (
    "A family of 6 (including 2 children and 1 person who uses a wheelchair) is planning a 3-day outdoor recreation "
    "weekend in San Diego County during early June 2026. They need your help identifying appropriate facilities and "
    "activities. Specifically, find: (1) A San Diego County campground that has at least 100 total campsites, offers "
    "fishing as an amenity, and provides ADA-accessible facilities. (2) A hiking trail at Torrey Pines State Natural "
    "Reserve that is between 2 and 3 miles in length and is rated as moderate difficulty. (3) An indoor recreation "
    "facility in San Diego that is larger than 30,000 square feet, offers badminton, and is open on Saturday mornings. "
    "(4) Information about the 2026 America the Beautiful annual pass for U.S. residents, including its price, whether "
    "it is available as a digital pass, and where the digital pass can be purchased. For each item, provide the specific "
    "name, relevant details confirming it meets the requirements, and at least one reference URL supporting your answer."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampgroundInfo(BaseModel):
    name: Optional[str] = None
    total_campsites_text: Optional[str] = None
    fishing_amenity_text: Optional[str] = None
    ada_access_text: Optional[str] = None
    sd_county_indicator_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class TrailInfo(BaseModel):
    name: Optional[str] = None
    location_text: Optional[str] = None
    distance_text: Optional[str] = None
    difficulty_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class IndoorFacilityInfo(BaseModel):
    name: Optional[str] = None
    location_text: Optional[str] = None
    size_text: Optional[str] = None
    badminton_text: Optional[str] = None
    saturday_hours_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class PassInfo(BaseModel):
    pass_type_text: Optional[str] = None
    price_text: Optional[str] = None
    digital_available_text: Optional[str] = None
    purchase_platform_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class WeekendPlanExtraction(BaseModel):
    campground: Optional[CampgroundInfo] = None
    trail: Optional[TrailInfo] = None
    indoor_facility: Optional[IndoorFacilityInfo] = None
    pass_info: Optional[PassInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_weekend_plan() -> str:
    return """
    Extract the structured information for the four requested items from the provided answer text. Return a single JSON object with the following structure and field names exactly:

    {
      "campground": {
        "name": string or null,
        "total_campsites_text": string or null,        // any mention of total number of campsites, as written
        "fishing_amenity_text": string or null,        // any mention confirming fishing is offered
        "ada_access_text": string or null,             // any mention that ADA-accessible facilities are available
        "sd_county_indicator_text": string or null,    // any mention indicating it's a San Diego County (County of San Diego Parks & Recreation) campground
        "source_urls": [array of URLs explicitly listed for the campground in the answer]
      },
      "trail": {
        "name": string or null,
        "location_text": string or null,               // any mention confirming it is at Torrey Pines State Natural Reserve
        "distance_text": string or null,               // the distance/length as stated (e.g., "2.3 miles", "2-3 miles")
        "difficulty_text": string or null,             // the difficulty rating as stated (e.g., "moderate")
        "source_urls": [array of URLs explicitly listed for the trail in the answer]
      },
      "indoor_facility": {
        "name": string or null,
        "location_text": string or null,               // any mention confirming it is in San Diego
        "size_text": string or null,                   // size as described (e.g., "40,000 sq ft")
        "badminton_text": string or null,              // any mention confirming badminton is offered
        "saturday_hours_text": string or null,         // hours text that includes Saturday morning availability if provided
        "source_urls": [array of URLs explicitly listed for the indoor facility in the answer]
      },
      "pass_info": {
        "pass_type_text": string or null,              // e.g., "2026 America the Beautiful annual pass"
        "price_text": string or null,                  // the stated price as text (e.g., "$80")
        "digital_available_text": string or null,      // text indicating digital availability (e.g., "available as a digital pass", "not available")
        "purchase_platform_text": string or null,      // where to buy the digital pass (e.g., "Recreation.gov", "USGS Store")
        "source_urls": [array of URLs explicitly listed for the pass information in the answer]
      }
    }

    Rules:
    - Extract only what is explicitly present in the answer text; do not invent or infer.
    - If a field is not present in the answer, set it to null (or [] for arrays).
    - For URLs, extract only valid, explicit URLs mentioned in the answer for the corresponding item. If none are provided, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if not (s.startswith("http://") or s.startswith("https://")):
            s = "http://" + s
        if s not in out:
            out.append(s)
    return out


def _truthy(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    t = text.strip().lower()
    if not t:
        return None
    # Basic heuristic for yes/no
    positives = ["yes", "available", "is available", "digital", "y", "true"]
    negatives = ["no", "not available", "unavailable", "n", "false"]
    if any(p in t for p in positives) and not any(n in t for n in negatives):
        return True
    if any(n in t for n in negatives) and not any(p in t for p in positives):
        return False
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_campground(evaluator: Evaluator, parent_node, cg: Optional[CampgroundInfo]) -> None:
    node = evaluator.add_parallel(
        id="Item_1_Campground",
        desc="Campground meeting specified requirements",
        parent=parent_node,
        critical=False,
    )

    name_exists = bool(cg and cg.name and cg.name.strip())
    srcs = _clean_urls(cg.source_urls if cg else [])

    evaluator.add_custom_node(
        result=name_exists,
        id="Campground_Name",
        desc="Provides the specific campground name",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id="Campground_Reference",
        desc="Provides at least one valid reference URL supporting campground claims",
        parent=node,
        critical=True,
    )

    # The following factual checks are gated by the above critical siblings automatically.
    # 1) Is a San Diego County park campground
    n1 = evaluator.add_leaf(
        id="Campground_Is_SD_County_Park_Campground",
        desc="Verifies the campground is a San Diego County park campground",
        parent=node,
        critical=True,
    )
    camp_name = cg.name if cg and cg.name else ""
    claim1 = (
        f"The campground named '{camp_name}' is part of the County of San Diego Parks and Recreation system (i.e., a San Diego County park campground)."
    )
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=srcs,
        additional_instruction="Confirm that the page indicates the campground is operated by or belongs to County of San Diego Parks and Recreation.",
    )

    # 2) Capacity >= 100 sites
    n2 = evaluator.add_leaf(
        id="Campground_Capacity",
        desc="Verifies campground has at least 100 total campsites",
        parent=node,
        critical=True,
    )
    claim2 = f"The campground '{camp_name}' has at least 100 total campsites."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=srcs,
        additional_instruction="Verify the total number of campsites shown. If the number is 100 or greater, consider the claim supported.",
    )

    # 3) Fishing amenity
    n3 = evaluator.add_leaf(
        id="Campground_Fishing",
        desc="Verifies campground offers fishing as an amenity",
        parent=node,
        critical=True,
    )
    claim3 = f"The campground '{camp_name}' offers fishing as an amenity (on-site or adjacent)."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=srcs,
        additional_instruction="Look for mentions of 'fishing' as an amenity or nearby activity on the page.",
    )

    # 4) ADA-accessible facilities
    n4 = evaluator.add_leaf(
        id="Campground_ADA_Access",
        desc="Verifies campground provides ADA-accessible facilities",
        parent=node,
        critical=True,
    )
    claim4 = f"The campground '{camp_name}' provides ADA-accessible facilities or accessible campsites."
    await evaluator.verify(
        claim=claim4,
        node=n4,
        sources=srcs,
        additional_instruction="Evidence may include terms like 'ADA', 'accessible', 'wheelchair-accessible', or icons/labels indicating accessibility.",
    )


async def verify_trail(evaluator: Evaluator, parent_node, tr: Optional[TrailInfo]) -> None:
    node = evaluator.add_parallel(
        id="Item_2_Hiking_Trail",
        desc="Hiking trail meeting specified requirements",
        parent=parent_node,
        critical=False,
    )

    name_exists = bool(tr and tr.name and tr.name.strip())
    srcs = _clean_urls(tr.source_urls if tr else [])

    evaluator.add_custom_node(
        result=name_exists,
        id="Trail_Identity",
        desc="Identifies a specific trail by name",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id="Trail_Reference",
        desc="Provides at least one valid reference URL supporting trail claims",
        parent=node,
        critical=True,
    )

    trail_name = tr.name if tr and tr.name else ""

    # 1) Located at Torrey Pines State Natural Reserve
    n1 = evaluator.add_leaf(
        id="Trail_Location",
        desc="Verifies trail is located at Torrey Pines State Natural Reserve",
        parent=node,
        critical=True,
    )
    claim1 = f"The trail '{trail_name}' is located within Torrey Pines State Natural Reserve in San Diego, California."
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=srcs,
        additional_instruction="The page should clearly indicate it's a trail of Torrey Pines State Natural Reserve (TPSNR).",
    )

    # 2) Distance between 2 and 3 miles
    n2 = evaluator.add_leaf(
        id="Trail_Distance",
        desc="Verifies trail length is between 2 and 3 miles",
        parent=node,
        critical=True,
    )
    claim2 = f"The trail '{trail_name}' has a total length between 2 and 3 miles (inclusive)."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=srcs,
        additional_instruction="If the page shows a distance in miles, check whether it's >= 2.0 and <= 3.0 miles. Loops and combined routes also qualify if stated.",
    )

    # 3) Moderate difficulty
    n3 = evaluator.add_leaf(
        id="Trail_Difficulty",
        desc="Verifies trail is rated as moderate difficulty",
        parent=node,
        critical=True,
    )
    claim3 = f"The trail '{trail_name}' is rated as moderate difficulty (or equivalent)."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=srcs,
        additional_instruction="Accept synonyms like 'moderate', 'intermediate', or similar language indicating a moderate level.",
    )


async def verify_indoor_facility(evaluator: Evaluator, parent_node, fac: Optional[IndoorFacilityInfo]) -> None:
    node = evaluator.add_parallel(
        id="Item_3_Indoor_Facility",
        desc="Indoor recreation facility meeting specified requirements",
        parent=parent_node,
        critical=False,
    )

    name_exists = bool(fac and fac.name and fac.name.strip())
    srcs = _clean_urls(fac.source_urls if fac else [])

    evaluator.add_custom_node(
        result=name_exists,
        id="Facility_Identity",
        desc="Identifies a specific indoor recreation facility by name",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id="Facility_Reference",
        desc="Provides at least one valid reference URL supporting facility claims",
        parent=node,
        critical=True,
    )

    facility_name = fac.name if fac and fac.name else ""

    # 1) Located in San Diego
    n1 = evaluator.add_leaf(
        id="Facility_Location",
        desc="Verifies the indoor facility is located in San Diego",
        parent=node,
        critical=True,
    )
    claim1 = f"The indoor recreation facility '{facility_name}' is located in San Diego, California."
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=srcs,
        additional_instruction="Confirm the address or city listing indicates 'San Diego, CA' (city, not just the county).",
    )

    # 2) Larger than 30,000 sq ft
    n2 = evaluator.add_leaf(
        id="Facility_Size",
        desc="Verifies facility is larger than 30,000 square feet",
        parent=node,
        critical=True,
    )
    claim2 = f"The indoor recreation facility '{facility_name}' has a size greater than 30,000 square feet."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=srcs,
        additional_instruction="Look for stated square footage; if a range or multiple buildings are given, consider total usable indoor recreation space.",
    )

    # 3) Badminton offered
    n3 = evaluator.add_leaf(
        id="Facility_Badminton",
        desc="Verifies facility offers badminton",
        parent=node,
        critical=True,
    )
    claim3 = f"The indoor recreation facility '{facility_name}' offers badminton (courts, programs, or drop-in)."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=srcs,
        additional_instruction="Accept mentions of 'badminton courts', 'badminton program', 'open play badminton', etc.",
    )

    # 4) Open on Saturday mornings
    n4 = evaluator.add_leaf(
        id="Facility_Saturday_Hours",
        desc="Verifies facility is open on Saturday mornings",
        parent=node,
        critical=True,
    )
    claim4 = f"The indoor recreation facility '{facility_name}' is open on Saturday mornings (open at or before 12:00 PM on Saturdays)."
    await evaluator.verify(
        claim=claim4,
        node=n4,
        sources=srcs,
        additional_instruction="Check operating hours or schedule pages to confirm Saturday morning availability.",
    )


async def verify_pass_info(evaluator: Evaluator, parent_node, ps: Optional[PassInfo]) -> None:
    node = evaluator.add_parallel(
        id="Item_4_Pass_Information",
        desc="America the Beautiful annual pass information for U.S. residents in 2026 meeting specified requirements",
        parent=parent_node,
        critical=False,
    )

    pass_type_exists = bool(ps and ps.pass_type_text and ps.pass_type_text.strip())
    srcs = _clean_urls(ps.source_urls if ps else [])

    evaluator.add_custom_node(
        result=pass_type_exists,
        id="Pass_Type",
        desc="Identifies the relevant 2026 America the Beautiful annual pass for U.S. residents being described",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id="Pass_Reference",
        desc="Provides at least one valid reference URL supporting pass claims",
        parent=node,
        critical=True,
    )

    pass_label = ps.pass_type_text if ps and ps.pass_type_text else "the 2026 America the Beautiful annual pass"

    # 1) Price (verify the specific stated price if provided; otherwise, still check that a price is stated)
    n1 = evaluator.add_leaf(
        id="Pass_Price",
        desc="Provides the price of the 2026 America the Beautiful annual pass for U.S. residents",
        parent=node,
        critical=True,
    )
    if ps and ps.price_text and ps.price_text.strip():
        claim1 = f"The price of {pass_label} is {ps.price_text.strip()}."
        add_ins1 = "Verify the page explicitly states this price for the 2026 pass (standard adult U.S. resident pass). Minor formatting differences (e.g., $80.00 vs $80) are acceptable."
    else:
        claim1 = f"The page states the price of {pass_label}."
        add_ins1 = "Verify that the page explicitly lists a price for the 2026 America the Beautiful annual pass."
    await evaluator.verify(
        claim=claim1,
        node=n1,
        sources=srcs,
        additional_instruction=add_ins1,
    )

    # 2) Digital availability
    n2 = evaluator.add_leaf(
        id="Pass_Digital_Availability",
        desc="States whether a digital version of the pass is available",
        parent=node,
        critical=True,
    )
    dig = _truthy(ps.digital_available_text if ps else None)
    if dig is True:
        claim2 = f"{pass_label} is available as a digital pass."
    elif dig is False:
        claim2 = f"{pass_label} is not available as a digital pass."
    else:
        # If unclear from extraction, still phrase as availability claim (the sources should settle it)
        claim2 = f"A digital version of {pass_label} is available."
    await evaluator.verify(
        claim=claim2,
        node=n2,
        sources=srcs,
        additional_instruction="Confirm whether the 2026 pass can be obtained in a digital form (e.g., mobile/digital).",
    )

    # 3) Purchase platform (where the digital pass can be purchased)
    n3 = evaluator.add_leaf(
        id="Pass_Purchase_Platform",
        desc="Identifies where the digital pass can be purchased (website/platform name)",
        parent=node,
        critical=True,
    )
    if ps and ps.purchase_platform_text and ps.purchase_platform_text.strip():
        claim3 = f"The digital version of {pass_label} can be purchased at {ps.purchase_platform_text.strip()}."
    else:
        claim3 = f"The page indicates where the digital version of {pass_label} can be purchased (e.g., Recreation.gov or USGS Store)."
    await evaluator.verify(
        claim=claim3,
        node=n3,
        sources=srcs,
        additional_instruction="Accept official platforms such as Recreation.gov or the USGS Store; verify that the page clearly identifies the purchase platform.",
    )


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the given answer for the San Diego weekend recreation plan task.
    """
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_weekend_plan(),
        template_class=WeekendPlanExtraction,
        extraction_name="weekend_plan_extraction",
    )

    # Build top-level node per rubric root
    plan_node = evaluator.add_parallel(
        id="Weekend_Recreation_Plan",
        desc="Comprehensive weekend plan meeting all stated item requirements",
        parent=root,
        critical=False,
    )

    # Verify each item sub-tree
    await verify_campground(evaluator, plan_node, extracted.campground)
    await verify_trail(evaluator, plan_node, extracted.trail)
    await verify_indoor_facility(evaluator, plan_node, extracted.indoor_facility)
    await verify_pass_info(evaluator, plan_node, extracted.pass_info)

    # Return summary
    return evaluator.get_summary()