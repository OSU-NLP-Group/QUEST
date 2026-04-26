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
TASK_ID = "houston_uil6a_stadiums_2024_26"
TASK_DESCRIPTION = (
    "Identify four Texas high school football programs in the Greater Houston metropolitan area that meet all of the "
    "following criteria for the 2024-26 UIL realignment cycle: (1) The school must be classified as UIL Conference 6A "
    "(enrollment of 2,275 or above) according to the official 2024-26 UIL realignment data. (2) The school must have "
    "access to a home stadium facility with a seating capacity of at least 10,000 that is regularly used for varsity "
    "football games. (3) The school must be located in the Greater Houston area (Harris County, Fort Bend County, "
    "Montgomery County, or Brazoria County) and assigned to UIL Region III for 6A football. (4) The school must be part "
    "of a school district that operates multiple high schools and maintains at least one stadium with 10,000+ seating "
    "capacity. For each school, provide: (a) the school name, (b) the 2024-26 UIL enrollment number, (c) the stadium name "
    "and its seating capacity, (d) the county location, and (e) the school district name. Include reference URLs that "
    "verify each piece of information."
)

ALLOWED_COUNTIES = {"Harris", "Fort Bend", "Montgomery", "Brazoria"}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SchoolEntry(BaseModel):
    school_name: Optional[str] = None
    county: Optional[str] = None
    district_name: Optional[str] = None

    uil_enrollment: Optional[str] = None
    uil_conference: Optional[str] = None
    uil_region: Optional[str] = None

    stadium_name: Optional[str] = None
    stadium_capacity: Optional[str] = None
    stadium_usage_desc: Optional[str] = None

    district_multi_hs_desc: Optional[str] = None
    district_stadium_desc: Optional[str] = None

    sources_enrollment: List[str] = Field(default_factory=list)
    sources_conference_region: List[str] = Field(default_factory=list)
    sources_stadium_capacity: List[str] = Field(default_factory=list)
    sources_stadium_usage: List[str] = Field(default_factory=list)
    sources_county: List[str] = Field(default_factory=list)
    sources_district_multi_hs: List[str] = Field(default_factory=list)
    sources_district_stadium: List[str] = Field(default_factory=list)
    sources_general: List[str] = Field(default_factory=list)


class FourSchoolsExtraction(BaseModel):
    schools: List[SchoolEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_schools() -> str:
    return (
        "From the answer, extract details for up to four (4) Texas high school football programs in the Greater Houston "
        "area that the answer claims meet the 2024-26 UIL 6A criteria. For each school, extract these fields exactly as "
        "stated in the answer:\n"
        "- school_name\n"
        "- county (e.g., Harris, Fort Bend, Montgomery, Brazoria)\n"
        "- district_name\n"
        "- uil_enrollment (the 2024-26 UIL enrollment number as presented)\n"
        "- uil_conference (e.g., 6A)\n"
        "- uil_region (e.g., Region III)\n"
        "- stadium_name\n"
        "- stadium_capacity (as stated; keep formatting like '10,000' or '12,700')\n"
        "- stadium_usage_desc (short text about home/varsity usage if present)\n"
        "- district_multi_hs_desc (short text indicating the district operates multiple high schools if present)\n"
        "- district_stadium_desc (short text indicating district has at least one 10,000+ stadium if present)\n\n"
        "Also extract URL sources explicitly mentioned in the answer for each verification category as arrays of URLs:\n"
        "- sources_enrollment: URLs supporting the UIL enrollment number (prefer official UIL realignment docs)\n"
        "- sources_conference_region: URLs supporting 6A assignment and Region III for 2024-26\n"
        "- sources_stadium_capacity: URLs supporting stated stadium capacity\n"
        "- sources_stadium_usage: URLs supporting that the school uses the stadium for varsity home games\n"
        "- sources_county: URLs supporting the county location of the school\n"
        "- sources_district_multi_hs: URLs supporting that the district operates multiple high schools\n"
        "- sources_district_stadium: URLs supporting that the district operates a 10,000+ capacity stadium\n"
        "- sources_general: any other relevant URLs tied to this school entry\n\n"
        "Return a JSON object: { 'schools': [ SchoolEntry, SchoolEntry, ... ] }. If a field or URL is missing in the "
        "answer, set it to null or an empty list accordingly. Extract URLs only if explicitly present (plain or markdown)."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(v: Optional[str]) -> str:
    return v or ""

def _merge_sources(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            u = (url or "").strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# School verification tree construction                                       #
# --------------------------------------------------------------------------- #
async def verify_school(
    evaluator: Evaluator,
    parent_node,
    school: SchoolEntry,
    idx: int,
) -> None:
    # Create school-level node (parallel, non-critical to allow partial credit across schools)
    school_node = evaluator.add_parallel(
        id=f"School_{idx+1}",
        desc=f"{['First','Second','Third','Fourth'][idx]} qualifying high school football program",
        parent=parent_node,
        critical=False
    )

    name = _safe(school.school_name)
    county = _safe(school.county)
    district = _safe(school.district_name)
    enrollment = _safe(school.uil_enrollment)
    conference = _safe(school.uil_conference)
    region = _safe(school.uil_region)
    stadium = _safe(school.stadium_name)
    capacity = _safe(school.stadium_capacity)

    # UIL Classification group
    uil_class_node = evaluator.add_parallel(
        id=f"UIL_Classification_{idx+1}",
        desc="School meets UIL 6A classification requirements for 2024-26",
        parent=school_node,
        critical=True
    )

    # Enrollment threshold subgroup
    enroll_thr_node = evaluator.add_parallel(
        id=f"Enrollment_Threshold_{idx+1}",
        desc="School enrollment meets minimum threshold",
        parent=uil_class_node,
        critical=True
    )

    # Enrollment >= 2,275 check (value)
    enrollment_value_leaf = evaluator.add_leaf(
        id=f"Enrollment_Value_{idx+1}",
        desc="School enrollment is 2,275 or above according to 2024-26 UIL realignment data",
        parent=enroll_thr_node,
        critical=True
    )
    enrollment_value_claim = (
        f"According to official UIL 2024-26 realignment data, the UIL enrollment for {name} is at least 2,275."
    )
    await evaluator.verify(
        claim=enrollment_value_claim,
        node=enrollment_value_leaf,
        sources=_merge_sources(school.sources_enrollment, school.sources_conference_region),
        additional_instruction=(
            "Rely on the cited official UIL 2024-26 alignment documentation (or district/UIL sources that include the "
            "alignment table). Confirm that the enrollment listed for the 2024-26 cycle is >= 2,275. If the URL is "
            "irrelevant or does not provide the enrollment number, conclude not supported."
        ),
    )

    # Enrollment exact number reference check
    enrollment_ref_leaf = evaluator.add_leaf(
        id=f"Enrollment_Reference_{idx+1}",
        desc="Enrollment number is supported by official UIL documentation",
        parent=enroll_thr_node,
        critical=True
    )
    enrollment_ref_claim = (
        f"The 2024-26 UIL enrollment number for {name} is '{enrollment}'."
        if enrollment else f"The 2024-26 UIL enrollment number for {name} is explicitly stated in the cited source."
    )
    await evaluator.verify(
        claim=enrollment_ref_claim,
        node=enrollment_ref_leaf,
        sources=_merge_sources(school.sources_enrollment, school.sources_conference_region),
        additional_instruction=(
            "Verify that the cited official UIL source (or a district source reproducing the UIL table) explicitly shows "
            "the 2024-26 UIL enrollment value for the school. Allow minor formatting differences (commas, spacing)."
        ),
    )

    # Conference assignment subgroup
    conf_node = evaluator.add_parallel(
        id=f"Conference_Assignment_{idx+1}",
        desc="School is assigned to correct UIL conference",
        parent=uil_class_node,
        critical=True
    )

    conf_val_leaf = evaluator.add_leaf(
        id=f"Conference_Value_{idx+1}",
        desc="School is officially assigned to UIL Conference 6A for 2024-26",
        parent=conf_node,
        critical=True
    )
    conf_val_claim = f"{name} is assigned to UIL Conference 6A for the 2024-26 cycle."
    await evaluator.verify(
        claim=conf_val_claim,
        node=conf_val_leaf,
        sources=_merge_sources(school.sources_conference_region, school.sources_enrollment),
        additional_instruction=(
            "Use official UIL alignment data or credible district/region listings for 2024-26. Confirm that the school is "
            "in Conference 6A. If the page lists a different conference or no conference, fail."
        ),
    )

    conf_ref_leaf = evaluator.add_leaf(
        id=f"Conference_Reference_{idx+1}",
        desc="Conference assignment is supported by official UIL documentation",
        parent=conf_node,
        critical=True
    )
    conf_ref_claim = (
        f"The official UIL documentation shows {name} in Conference 6A for 2024-26."
    )
    await evaluator.verify(
        claim=conf_ref_claim,
        node=conf_ref_leaf,
        sources=_merge_sources(school.sources_conference_region, school.sources_enrollment),
        additional_instruction=(
            "Find explicit evidence on the provided source that the school is placed in Conference 6A for the 2024-26 cycle."
        ),
    )

    # Stadium facility group
    stadium_node = evaluator.add_parallel(
        id=f"Stadium_Facility_{idx+1}",
        desc="School has access to a stadium meeting capacity requirements",
        parent=school_node,
        critical=True
    )

    # Stadium capacity subgroup
    cap_node = evaluator.add_parallel(
        id=f"Stadium_Capacity_{idx+1}",
        desc="Stadium meets seating capacity threshold",
        parent=stadium_node,
        critical=True
    )

    cap_val_leaf = evaluator.add_leaf(
        id=f"Capacity_Value_{idx+1}",
        desc="The stadium used by the school has a seating capacity of at least 10,000",
        parent=cap_node,
        critical=True
    )
    cap_val_claim = (
        f"The stadium '{stadium}' used by {name} has a seating capacity of at least 10,000."
        if stadium else "The school's home stadium has a seating capacity of at least 10,000."
    )
    await evaluator.verify(
        claim=cap_val_claim,
        node=cap_val_leaf,
        sources=_merge_sources(school.sources_stadium_capacity, school.sources_general),
        additional_instruction=(
            "Verify the seating capacity stated on the cited stadium/district source. If the number shown is below "
            "10,000 or the page does not provide capacity, conclude not supported."
        ),
    )

    cap_ref_leaf = evaluator.add_leaf(
        id=f"Capacity_Reference_{idx+1}",
        desc="Stadium capacity is supported by verifiable sources",
        parent=cap_node,
        critical=True
    )
    cap_ref_claim = (
        f"The seating capacity of '{stadium}' is '{capacity}'."
        if stadium and capacity else "The cited source explicitly states the stadium seating capacity."
    )
    await evaluator.verify(
        claim=cap_ref_claim,
        node=cap_ref_leaf,
        sources=_merge_sources(school.sources_stadium_capacity, school.sources_general),
        additional_instruction=(
            "Confirm the specific capacity figure or assertion from the cited source (allow reasonable formatting like commas)."
        ),
    )

    # Stadium usage subgroup
    usage_node = evaluator.add_parallel(
        id=f"Stadium_Usage_{idx+1}",
        desc="Stadium is used for school's games",
        parent=stadium_node,
        critical=True
    )

    usage_val_leaf = evaluator.add_leaf(
        id=f"Usage_Value_{idx+1}",
        desc="The school regularly uses this stadium for varsity home football games",
        parent=usage_node,
        critical=True
    )
    usage_val_claim = (
        f"{name} regularly uses '{stadium}' for varsity home football games."
        if stadium else f"{name} regularly uses the cited stadium for varsity home football games."
    )
    await evaluator.verify(
        claim=usage_val_claim,
        node=usage_val_leaf,
        sources=_merge_sources(school.sources_stadium_usage, school.sources_general),
        additional_instruction=(
            "Look for explicit statements, schedules, or district pages indicating home varsity football usage of the stadium."
        ),
    )

    usage_ref_leaf = evaluator.add_leaf(
        id=f"Usage_Reference_{idx+1}",
        desc="Stadium usage is supported by verifiable sources",
        parent=usage_node,
        critical=True
    )
    usage_ref_claim = (
        f"The provided source confirms that '{stadium}' is used by {name} for varsity football home games."
        if stadium else "The provided source confirms that the school uses the cited stadium for varsity football home games."
    )
    await evaluator.verify(
        claim=usage_ref_claim,
        node=usage_ref_leaf,
        sources=_merge_sources(school.sources_stadium_usage, school.sources_general),
        additional_instruction=(
            "Confirm usage via schedules, district facility pages, or news releases. If usage is not shown, fail."
        ),
    )

    # Geographic location group
    geo_node = evaluator.add_parallel(
        id=f"Geographic_Location_{idx+1}",
        desc="School is located in the Greater Houston metropolitan area",
        parent=school_node,
        critical=True
    )

    # County location subgroup
    county_node = evaluator.add_parallel(
        id=f"County_Location_{idx+1}",
        desc="School is in correct county",
        parent=geo_node,
        critical=True
    )

    county_val_leaf = evaluator.add_leaf(
        id=f"County_Value_{idx+1}",
        desc="School is located in Harris, Fort Bend, Montgomery, or Brazoria County",
        parent=county_node,
        critical=True
    )
    county_val_claim = (
        f"{name} is located in {county} County, Texas, which is one of Harris, Fort Bend, Montgomery, or Brazoria."
        if county else f"The school is located in one of Harris, Fort Bend, Montgomery, or Brazoria County."
    )
    await evaluator.verify(
        claim=county_val_claim,
        node=county_val_leaf,
        sources=_merge_sources(school.sources_county, school.sources_general),
        additional_instruction=(
            "From the provided source(s), confirm the school's county and ensure it is one of Harris, Fort Bend, "
            "Montgomery, or Brazoria. If the county differs or is not shown, conclude not supported."
        ),
    )

    county_ref_leaf = evaluator.add_leaf(
        id=f"County_Reference_{idx+1}",
        desc="County location is supported by verifiable sources",
        parent=county_node,
        critical=True
    )
    county_ref_claim = (
        f"The cited source explicitly shows '{name}' is located in {county} County, Texas."
        if county else f"The cited source explicitly shows the school's county location."
    )
    await evaluator.verify(
        claim=county_ref_claim,
        node=county_ref_leaf,
        sources=_merge_sources(school.sources_county, school.sources_general),
        additional_instruction=(
            "Look for address, district boundaries, or county listings confirming the county of the school."
        ),
    )

    # Region assignment subgroup
    region_node = evaluator.add_parallel(
        id=f"Region_Assignment_{idx+1}",
        desc="School is in correct UIL region",
        parent=geo_node,
        critical=True
    )

    region_val_leaf = evaluator.add_leaf(
        id=f"Region_Value_{idx+1}",
        desc="School is part of UIL Region III (Greater Houston region) for 6A football",
        parent=region_node,
        critical=True
    )
    region_val_claim = f"{name} is assigned to UIL Region III for 6A football in the 2024-26 cycle."
    await evaluator.verify(
        claim=region_val_claim,
        node=region_val_leaf,
        sources=_merge_sources(school.sources_conference_region, school.sources_enrollment),
        additional_instruction=(
            "Use the official UIL alignment data or credible district/region listings showing the school assigned to Region III for 6A football."
        ),
    )

    region_ref_leaf = evaluator.add_leaf(
        id=f"Region_Reference_{idx+1}",
        desc="Region assignment is supported by verifiable sources",
        parent=region_node,
        critical=True
    )
    region_ref_claim = f"The official documentation confirms {name} is in UIL Region III for 6A football (2024-26)."
    await evaluator.verify(
        claim=region_ref_claim,
        node=region_ref_leaf,
        sources=_merge_sources(school.sources_conference_region, school.sources_enrollment),
        additional_instruction="Find explicit region assignment (Region III) for 6A football on the provided source.",
    )

    # School district group
    district_node = evaluator.add_parallel(
        id=f"School_District_{idx+1}",
        desc="School is part of a qualifying school district",
        parent=school_node,
        critical=True
    )

    # Multiple high schools subgroup
    multi_node = evaluator.add_parallel(
        id=f"Multiple_High_Schools_{idx+1}",
        desc="District operates multiple schools",
        parent=district_node,
        critical=True
    )

    multi_val_leaf = evaluator.add_leaf(
        id=f"Multiple_Schools_Value_{idx+1}",
        desc="The school district operates multiple high schools",
        parent=multi_node,
        critical=True
    )
    multi_val_claim = (
        f"The school district '{district}' operates multiple high schools."
        if district else "The school district operates multiple high schools."
    )
    await evaluator.verify(
        claim=multi_val_claim,
        node=multi_val_leaf,
        sources=_merge_sources(school.sources_district_multi_hs, school.sources_general),
        additional_instruction=(
            "Confirm via district website (campus list), UIL listings, or other credible sources that the district has "
            "two or more high schools."
        ),
    )

    multi_ref_leaf = evaluator.add_leaf(
        id=f"Multiple_Schools_Reference_{idx+1}",
        desc="Multiple schools information is supported by verifiable sources",
        parent=multi_node,
        critical=True
    )
    multi_ref_claim = (
        f"The cited source explicitly shows that {district} ISD operates multiple high schools."
        if district else "The cited source explicitly shows the district operates multiple high schools."
    )
    await evaluator.verify(
        claim=multi_ref_claim,
        node=multi_ref_leaf,
        sources=_merge_sources(school.sources_district_multi_hs, school.sources_general),
        additional_instruction="Look for an explicit listing of multiple high school campuses operated by the district.",
    )

    # District stadium subgroup
    dist_stad_node = evaluator.add_parallel(
        id=f"District_Stadium_{idx+1}",
        desc="District has qualifying stadium",
        parent=district_node,
        critical=True
    )

    dist_stad_val_leaf = evaluator.add_leaf(
        id=f"District_Stadium_Value_{idx+1}",
        desc="The school district operates at least one stadium with 10,000+ seating capacity",
        parent=dist_stad_node,
        critical=True
    )
    dist_stad_val_claim = (
        f"{district} operates at least one football stadium with seating capacity of 10,000 or more."
        if district else "The school district operates at least one football stadium with seating capacity of 10,000 or more."
    )
    await evaluator.verify(
        claim=dist_stad_val_claim,
        node=dist_stad_val_leaf,
        sources=_merge_sources(school.sources_district_stadium, school.sources_stadium_capacity, school.sources_general),
        additional_instruction=(
            "Confirm that at least one district-operated stadium has capacity >= 10,000. Use district facilities pages, "
            "official press materials, or credible sources with capacity numbers."
        ),
    )

    dist_stad_ref_leaf = evaluator.add_leaf(
        id=f"District_Stadium_Reference_{idx+1}",
        desc="District stadium information is supported by verifiable sources",
        parent=dist_stad_node,
        critical=True
    )
    dist_stad_ref_claim = (
        f"The cited source explicitly indicates a district stadium with capacity at least 10,000."
    )
    await evaluator.verify(
        claim=dist_stad_ref_claim,
        node=dist_stad_ref_leaf,
        sources=_merge_sources(school.sources_district_stadium, school.sources_stadium_capacity, school.sources_general),
        additional_instruction="Verify the capacity figure or assertion in the cited source meets or exceeds 10,000.",
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation across schools
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify four Texas UIL 6A high school football programs in the Greater Houston area that have access to stadiums with at least 10,000 seating capacity for the 2024-26 cycle",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_schools(),
        template_class=FourSchoolsExtraction,
        extraction_name="schools_extraction",
    )

    # Restrict to first 4 schools; pad if fewer
    schools: List[SchoolEntry] = list(extracted.schools[:4])
    while len(schools) < 4:
        schools.append(SchoolEntry())

    # Build verification tree per school
    verify_tasks = []
    for idx, school in enumerate(schools):
        verify_tasks.append(verify_school(evaluator, root, school, idx))

    await asyncio.gather(*verify_tasks)

    # Add custom info (allowed counties set)
    evaluator.add_custom_info(
        info={"allowed_counties": sorted(list(ALLOWED_COUNTIES))},
        info_type="constraints",
        info_name="allowed_counties"
    )

    # Return evaluation summary
    return evaluator.get_summary()