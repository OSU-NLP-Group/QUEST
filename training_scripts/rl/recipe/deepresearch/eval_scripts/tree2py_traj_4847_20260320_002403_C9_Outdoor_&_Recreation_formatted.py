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
TASK_ID = "accessible_youth_outdoor_program"
TASK_DESCRIPTION = """
Design a comprehensive 7-day accessible outdoor recreation program in the United States for a youth group (with participants under 18 years old) that must satisfy ALL of the following requirements:

Destination & Transportation:
- The program must be based at a U.S. national park that is served by Breeze Airways
- The nearest Breeze Airways airport must be within 1 hour of the national park

Camping Accommodation:
- Include an accessible campground that can be booked through Recreation.gov
- The campground must follow the standard 6-month advance booking window
- The campground must have wheelchair-accessible campsites available
- The campground must provide accessible restrooms and shower facilities

Beach Activity:
- Include a beach resort visit with ADA-compliant beach access
- The beach access route must have a minimum clear width of 60 inches (as required by ADA standards)
- Beach wheelchairs must be available for guest use
- The resort must provide ADA-compliant guest rooms

Winter Activity:
- Include a ski resort that has a mandatory helmet policy for all participants under 18 years old
- The ski resort must require helmets that meet ASTM F2040, CE EN 1077, or CSA certification standards
- The ski resort must offer ski lessons for youth participants

Safety Certification:
- The program must employ adventure guides who hold valid Wilderness First Responder (WFR) certification
- The WFR certification must be from an 80-hour professional-level course
- Verify that WFR certification requires no prerequisite certification

For your answer, provide:
1. The specific national park name and nearest Breeze Airways airport
2. The specific campground name and its Recreation.gov booking details
3. The specific beach resort name and its ADA accessibility features
4. The specific ski resort name and its helmet policy details
5. The specific adventure tourism company or certification provider name and guide certification details
6. Reference URLs for each facility to verify all requirements
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DestinationInfo(BaseModel):
    national_park_name: Optional[str] = None
    breeze_airways_airport_name: Optional[str] = None
    airport_proximity_time_text: Optional[str] = None  # e.g., "45 minutes"
    destination_urls: List[str] = Field(default_factory=list)  # park, maps, NPS, etc.
    breeze_airways_related_urls: List[str] = Field(default_factory=list)  # Breeze route/airport pages


class CampgroundInfo(BaseModel):
    campground_name: Optional[str] = None
    recreation_gov_url: Optional[str] = None
    booking_window_text: Optional[str] = None  # e.g., "6 months in advance"
    wheelchair_accessible_sites_text: Optional[str] = None
    accessible_restrooms_showers_text: Optional[str] = None
    camping_urls: List[str] = Field(default_factory=list)  # additional refs


class BeachResortInfo(BaseModel):
    resort_name: Optional[str] = None
    resort_urls: List[str] = Field(default_factory=list)
    beach_access_route_width_text: Optional[str] = None  # e.g., "60 inches"
    beach_wheelchair_availability_text: Optional[str] = None
    ada_compliant_rooms_text: Optional[str] = None
    access_route_spec_urls: List[str] = Field(default_factory=list)  # ADA standard/spec pages


class SkiResortInfo(BaseModel):
    ski_resort_name: Optional[str] = None
    ski_resort_urls: List[str] = Field(default_factory=list)
    helmet_policy_text: Optional[str] = None
    mandatory_under_18_text: Optional[str] = None
    helmet_certification_standards_text: Optional[str] = None  # e.g., "ASTM F2040, CE EN 1077, or CSA"
    ski_lessons_youth_text: Optional[str] = None
    helmet_policy_urls: List[str] = Field(default_factory=list)


class SafetyCertificationInfo(BaseModel):
    company_or_provider_name: Optional[str] = None  # adventure company or WFR provider
    guide_wfr_employment_text: Optional[str] = None  # "Guides hold WFR" (from answer)
    company_urls: List[str] = Field(default_factory=list)  # company/provider website(s)
    wfr_course_hours_text: Optional[str] = None  # e.g., "80-hour"
    wfr_prerequisite_text: Optional[str] = None  # "no prerequisites"
    wfr_professional_level_text: Optional[str] = None  # "professional-level"
    wfr_standard_urls: List[str] = Field(default_factory=list)  # provider course pages/standards


class ProgramExtraction(BaseModel):
    destination: Optional[DestinationInfo] = None
    campground: Optional[CampgroundInfo] = None
    beach: Optional[BeachResortInfo] = None
    ski: Optional[SkiResortInfo] = None
    safety: Optional[SafetyCertificationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program() -> str:
    return """
Extract the structured details the answer provides for the requested 7-day accessible outdoor program. Return a JSON with these top-level fields: destination, campground, beach, ski, safety. For each section, extract EXACTLY what is written in the answer (do not infer or invent). If any field is not present, set it to null (for strings) or [] (for URL lists).

destination:
- national_park_name: string
- breeze_airways_airport_name: string
- airport_proximity_time_text: string (e.g., "45 minutes", "about 1 hour")
- destination_urls: array of URLs specifically cited for the national park, maps, or proximity
- breeze_airways_related_urls: array of URLs cited for Breeze Airways route/airport info

campground:
- campground_name: string
- recreation_gov_url: string (the booking URL on Recreation.gov if provided)
- booking_window_text: string describing booking window (e.g., "6 months in advance")
- wheelchair_accessible_sites_text: string describing availability of accessible campsites
- accessible_restrooms_showers_text: string describing accessible restrooms/showers
- camping_urls: array of any additional URLs cited for the campground

beach:
- resort_name: string
- resort_urls: array of URLs cited for the beach resort (accessibility, rooms, amenities)
- beach_access_route_width_text: string describing the access route clear width (e.g., "60 inches" or "5 feet")
- beach_wheelchair_availability_text: string about beach wheelchairs availability
- ada_compliant_rooms_text: string about ADA/accessible guest rooms
- access_route_spec_urls: array of URLs for ADA/specification pages cited to justify route width

ski:
- ski_resort_name: string
- ski_resort_urls: array of URLs cited for the ski resort (policies, lessons)
- helmet_policy_text: string summarizing the helmet policy
- mandatory_under_18_text: string confirming helmets are mandatory for under 18
- helmet_certification_standards_text: string listing required standards (e.g., ASTM F2040, CE EN 1077, CSA)
- ski_lessons_youth_text: string confirming youth lessons
- helmet_policy_urls: array of URLs cited for the helmet policy

safety:
- company_or_provider_name: string (adventure tourism company or certification provider)
- guide_wfr_employment_text: string stating guides have WFR certification (if present)
- company_urls: array of URLs cited for the company/provider
- wfr_course_hours_text: string confirming WFR is an 80-hour course (or equivalent language)
- wfr_prerequisite_text: string confirming no prerequisite certification is required
- wfr_professional_level_text: string confirming WFR is professional-level training
- wfr_standard_urls: array of URLs cited for WFR certification standards/course page

Strict rules:
- Only extract URLs explicitly present in the answer. Do not invent or infer URLs.
- Keep all text exactly as in the answer (do not normalize numbers/units).
- If a requested field is not in the answer, return null (or [] for URL lists).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str) and u.strip() and u not in merged:
                merged.append(u.strip())
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_destination(evaluator: Evaluator, parent_node, dest: Optional[DestinationInfo]) -> None:
    node = evaluator.add_parallel(
        id="Destination_and_Transportation",
        desc="National park destination accessible via Breeze Airways",
        parent=parent_node,
        critical=False
    )

    park_name = dest.national_park_name if dest else None
    airport_name = dest.breeze_airways_airport_name if dest else None
    park_urls = dest.destination_urls if dest else []
    breeze_urls = dest.breeze_airways_related_urls if dest else []
    all_dest_urls = merge_urls(park_urls, breeze_urls)

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(park_name and park_name.strip()),
        id="Park_Name_Provided",
        desc="Specific national park name is identified",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(airport_name and airport_name.strip()),
        id="Airport_Name_Provided",
        desc="Specific Breeze Airways airport is identified",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(park_urls) and bool(breeze_urls),
        id="Reference_URLs_Destination",
        desc="Reference URLs provided for destination verification",
        parent=node,
        critical=True
    )

    # Leaf: National park is served by Breeze Airways (via the specified airport)
    park_served_leaf = evaluator.add_leaf(
        id="National_Park_Selection",
        desc="Selected national park is served by Breeze Airways",
        parent=node,
        critical=True
    )
    claim_served = (
        f"Breeze Airways operates service to {airport_name}, which serves as the gateway for {park_name}. "
        f"Therefore, {park_name} is served by Breeze Airways via {airport_name}."
    )
    await evaluator.verify(
        claim=claim_served,
        node=park_served_leaf,
        sources=merge_urls(breeze_urls, park_urls),
        additional_instruction="Verify that Breeze Airways serves the named airport and that this airport is presented as a gateway for the named national park."
    )

    # Leaf: Breeze airport is within 1 hour of the national park
    proximity_leaf = evaluator.add_leaf(
        id="Airport_Proximity",
        desc="Breeze Airways airport is within 1 hour of the national park",
        parent=node,
        critical=True
    )
    claim_proximity = (
        f"The driving time from {airport_name} to {park_name} is 60 minutes or less (i.e., within 1 hour)."
    )
    await evaluator.verify(
        claim=claim_proximity,
        node=proximity_leaf,
        sources=merge_urls(park_urls),
        additional_instruction="Treat '1 hour' as inclusive (<= 60 minutes). Accept minor phrasing variants like 'about 1 hr' or '~55 min'."
    )


async def verify_camping(evaluator: Evaluator, parent_node, camp: Optional[CampgroundInfo]) -> None:
    node = evaluator.add_parallel(
        id="Accessible_Camping_Accommodation",
        desc="Accessible campground meeting booking and accessibility requirements",
        parent=parent_node,
        critical=False
    )

    name = camp.campground_name if camp else None
    recgov_url = camp.recreation_gov_url if camp else None
    extra_urls = camp.camping_urls if camp else []
    urls_all = merge_urls([recgov_url] if recgov_url else [], extra_urls)

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id="Campground_Name_Provided",
        desc="Specific campground name is identified",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(recgov_url or extra_urls),
        id="Reference_URLs_Camping",
        desc="Reference URLs provided for camping facility verification",
        parent=node,
        critical=True
    )

    # Recreation.gov bookable
    bookable_leaf = evaluator.add_leaf(
        id="Recreation_gov_Bookable",
        desc="Campground is bookable through Recreation.gov",
        parent=node,
        critical=True
    )
    claim_bookable = f"Reservations for {name} are made via Recreation.gov."
    await evaluator.verify(
        claim=claim_bookable,
        node=bookable_leaf,
        sources=[recgov_url] if recgov_url else urls_all,
        additional_instruction="Confirm that the official reservation/booking for this campground is handled on Recreation.gov (domain recreation.gov)."
    )

    # 6-month advance booking window
    six_month_leaf = evaluator.add_leaf(
        id="Six_Month_Booking_Window",
        desc="Campground follows 6-month advance booking window",
        parent=node,
        critical=True
    )
    claim_six_month = f"The campground {name} follows a 6-month advance reservation window."
    await evaluator.verify(
        claim=claim_six_month,
        node=six_month_leaf,
        sources=[recgov_url] if recgov_url else urls_all,
        additional_instruction="Look for phrasing such as 'available 6 months in advance' or a reservation window of 6 months on Recreation.gov or the official policy page."
    )

    # Wheelchair-accessible campsites
    accessible_sites_leaf = evaluator.add_leaf(
        id="Wheelchair_Accessible_Sites",
        desc="Campground has wheelchair-accessible campsites available",
        parent=node,
        critical=True
    )
    claim_sites = f"The campground {name} offers wheelchair-accessible campsites."
    await evaluator.verify(
        claim=claim_sites,
        node=accessible_sites_leaf,
        sources=[recgov_url] if recgov_url else urls_all,
        additional_instruction="Confirm presence of accessible/ADA-designated campsites or equivalent wording."
    )

    # Accessible restrooms and showers
    accessible_rest_leaf = evaluator.add_leaf(
        id="Accessible_Restrooms",
        desc="Campground provides accessible restrooms and shower facilities",
        parent=node,
        critical=True
    )
    claim_rest = f"The campground {name} provides accessible restrooms and accessible shower facilities."
    await evaluator.verify(
        claim=claim_rest,
        node=accessible_rest_leaf,
        sources=[recgov_url] if recgov_url else urls_all,
        additional_instruction="Confirm both accessible restrooms and accessible showers are available; accept equivalent phrasing like ADA-accessible bathrooms/showers."
    )


async def verify_beach(evaluator: Evaluator, parent_node, beach: Optional[BeachResortInfo]) -> None:
    node = evaluator.add_parallel(
        id="Beach_Activity_Facility",
        desc="Beach resort with ADA-compliant accessibility features",
        parent=parent_node,
        critical=False
    )

    name = beach.resort_name if beach else None
    resort_urls = beach.resort_urls if beach else []
    spec_urls = beach.access_route_spec_urls if beach else []

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id="Beach_Resort_Name_Provided",
        desc="Specific beach resort name is identified",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(resort_urls),
        id="Reference_URLs_Beach_Resort",
        desc="Reference URLs provided for beach resort verification",
        parent=node,
        critical=True
    )

    # ADA Beach Access Route (subnode)
    access_route_node = evaluator.add_parallel(
        id="ADA_Beach_Access_Route",
        desc="Beach access route meets ADA minimum width requirement",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(spec_urls),
        id="Reference_URLs_Access_Route",
        desc="Reference URLs for beach access route specifications",
        parent=access_route_node,
        critical=True
    )

    min_width_leaf = evaluator.add_leaf(
        id="Minimum_60_Inch_Width",
        desc="Beach access route has minimum clear width of 60 inches",
        parent=access_route_node,
        critical=True
    )
    claim_width = "ADA standards require a minimum 60-inch (5 feet, approximately 1525 mm) clear width for beach access routes."
    await evaluator.verify(
        claim=claim_width,
        node=min_width_leaf,
        sources=spec_urls,
        additional_instruction="Verify against ADA standards or authoritative guidance. Accept '60 in', '5 ft', or '1525 mm' as equivalent."
    )

    # Beach wheelchairs available
    wheelchairs_leaf = evaluator.add_leaf(
        id="Beach_Wheelchair_Availability",
        desc="Beach wheelchairs are available for guest use",
        parent=node,
        critical=True
    )
    claim_wheelchairs = f"The resort {name} provides beach wheelchairs for guest use (loan or rental)."
    await evaluator.verify(
        claim=claim_wheelchairs,
        node=wheelchairs_leaf,
        sources=resort_urls,
        additional_instruction="Look for terms like 'beach wheelchair', 'sand wheelchair', or 'accessible beach chair' available to guests."
    )

    # ADA-compliant rooms
    ada_rooms_leaf = evaluator.add_leaf(
        id="ADA_Compliant_Rooms",
        desc="Resort provides ADA-compliant guest rooms with accessibility features",
        parent=node,
        critical=True
    )
    claim_rooms = f"The resort {name} offers ADA-compliant accessible guest rooms (mobility-accessible)."
    await evaluator.verify(
        claim=claim_rooms,
        node=ada_rooms_leaf,
        sources=resort_urls,
        additional_instruction="Confirm the presence of ADA/accessible rooms with mobility features; synonyms like 'mobility accessible' are acceptable."
    )


async def verify_winter(evaluator: Evaluator, parent_node, ski: Optional[SkiResortInfo]) -> None:
    node = evaluator.add_parallel(
        id="Winter_Activity_Facility",
        desc="Ski resort with mandatory helmet requirements for youth participants",
        parent=parent_node,
        critical=False
    )

    name = ski.ski_resort_name if ski else None
    resort_urls = ski.ski_resort_urls if ski else []
    policy_urls = ski.helmet_policy_urls if ski else []
    policy_refs = merge_urls(policy_urls, resort_urls)

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id="Ski_Resort_Name_Provided",
        desc="Specific ski resort name is identified",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(resort_urls),
        id="Reference_URLs_Ski_Resort",
        desc="Reference URLs provided for ski resort verification",
        parent=node,
        critical=True
    )

    # Helmet policy (subnode)
    helmet_node = evaluator.add_parallel(
        id="Helmet_Requirement_Policy",
        desc="Ski resort has helmet requirements for participants under 18",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(policy_urls) or bool(resort_urls),
        id="Reference_URLs_Helmet_Policy",
        desc="Reference URLs for helmet policy verification",
        parent=helmet_node,
        critical=True
    )

    under18_leaf = evaluator.add_leaf(
        id="Mandatory_Under_18",
        desc="Helmets are mandatory for all participants under 18 years old",
        parent=helmet_node,
        critical=True
    )
    claim_under18 = f"The ski resort {name} requires helmets for all participants under 18 years old."
    await evaluator.verify(
        claim=claim_under18,
        node=under18_leaf,
        sources=policy_refs,
        additional_instruction="Confirm the policy explicitly states helmets are required/mandatory for minors/under 18."
    )

    certs_leaf = evaluator.add_leaf(
        id="Certification_Standards",
        desc="Helmets must meet ASTM F2040, CE EN 1077, or CSA certification standards",
        parent=helmet_node,
        critical=True
    )
    claim_certs = (
        f"The ski resort {name}'s helmet policy requires helmets that meet at least one of: ASTM F2040, CE EN 1077, or CSA."
    )
    await evaluator.verify(
        claim=claim_certs,
        node=certs_leaf,
        sources=policy_refs,
        additional_instruction="Look for mention of ASTM F2040, CE EN 1077, or CSA as accepted certifications for helmets."
    )

    # Ski lessons for youth
    lessons_leaf = evaluator.add_leaf(
        id="Ski_Lessons_Available",
        desc="Ski resort offers lessons for youth participants",
        parent=node,
        critical=True
    )
    claim_lessons = f"The ski resort {name} offers ski lessons for youth/children."
    await evaluator.verify(
        claim=claim_lessons,
        node=lessons_leaf,
        sources=resort_urls,
        additional_instruction="Confirm dedicated youth/child lessons, ski school for kids, or similar offerings."
    )


async def verify_safety(evaluator: Evaluator, parent_node, safety: Optional[SafetyCertificationInfo]) -> None:
    node = evaluator.add_parallel(
        id="Safety_Certification_Requirements",
        desc="Adventure guides with proper wilderness medical certification",
        parent=parent_node,
        critical=False
    )

    company_name = safety.company_or_provider_name if safety else None
    company_urls = safety.company_urls if safety else []
    wfr_urls = safety.wfr_standard_urls if safety else []
    all_wfr_urls = merge_urls(wfr_urls, company_urls)

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(company_name and company_name.strip()),
        id="Adventure_Company_Name_Provided",
        desc="Specific adventure tourism company is identified",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(company_urls),
        id="Reference_URLs_Guide_Company",
        desc="Reference URLs provided for adventure company verification",
        parent=node,
        critical=True
    )

    # WFR Certification standards (subnode)
    wfr_node = evaluator.add_parallel(
        id="Wilderness_First_Responder_Certification",
        desc="Adventure company employs guides with WFR certification",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(wfr_urls),
        id="Reference_URLs_WFR_Standards",
        desc="Reference URLs for WFR certification standards",
        parent=wfr_node,
        critical=True
    )

    # 80-hour course
    hours_leaf = evaluator.add_leaf(
        id="80_Hour_Course_Completion",
        desc="WFR certification requires completion of 80-hour course",
        parent=wfr_node,
        critical=True
    )
    claim_hours = "A Wilderness First Responder (WFR) certification course is approximately 80 hours in length."
    await evaluator.verify(
        claim=claim_hours,
        node=hours_leaf,
        sources=all_wfr_urls,
        additional_instruction="Allow phrasing like '80-hour', 'approximately 80 hours', or '70–80 hours' if clearly WFR (not WFA)."
    )

    # No prerequisite certification
    prereq_leaf = evaluator.add_leaf(
        id="No_Prerequisite_Required",
        desc="WFR course has no prerequisite certification requirement",
        parent=wfr_node,
        critical=True
    )
    claim_prereq = "A Wilderness First Responder (WFR) course requires no prerequisite certification; CPR/AED may be required separately or provided concurrently."
    await evaluator.verify(
        claim=claim_prereq,
        node=prereq_leaf,
        sources=all_wfr_urls,
        additional_instruction="Confirm that WFR has no formal prerequisite certifications. It's okay if providers recommend fitness or CPR; but no prerequisite should be required."
    )

    # Professional-level certification
    pro_level_leaf = evaluator.add_leaf(
        id="Professional_Level_Certification",
        desc="WFR is recognized as professional-level wilderness medical training",
        parent=wfr_node,
        critical=True
    )
    claim_pro = "The Wilderness First Responder (WFR) is considered professional-level wilderness medicine training appropriate for guides."
    await evaluator.verify(
        claim=claim_pro,
        node=pro_level_leaf,
        sources=all_wfr_urls,
        additional_instruction="Look for language like 'professional-level', 'industry standard for guides', 'guide-level certification', or equivalent."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Accessible Youth Outdoor Program task.
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
        default_model=model
    )

    # Program-level grouping node (non-critical to allow partial credit across categories)
    program_node = evaluator.add_parallel(
        id="Accessible_Youth_Outdoor_Program",
        desc="A comprehensive 7-day outdoor recreation program that meets accessibility, safety, and logistical requirements for youth participants",
        parent=root,
        critical=False  # Keep non-critical to satisfy framework constraints on critical parent/child consistency
    )

    # Extraction
    extracted: ProgramExtraction = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction"
    )

    # Build subtrees
    await verify_destination(evaluator, program_node, extracted.destination)
    await verify_camping(evaluator, program_node, extracted.campground)
    await verify_beach(evaluator, program_node, extracted.beach)
    await verify_winter(evaluator, program_node, extracted.ski)
    await verify_safety(evaluator, program_node, extracted.safety)

    # Return summary
    return evaluator.get_summary()