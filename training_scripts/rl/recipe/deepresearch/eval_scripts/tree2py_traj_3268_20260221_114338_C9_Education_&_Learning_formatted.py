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
TASK_ID = "k12_msche_task"
TASK_DESCRIPTION = """
Identify three large public school districts and two universities that meet the following criteria:

School Districts:
- Each district must have current enrollment of at least 125,000 students
- All three districts must be located in either North Carolina or Florida
- For each district, provide: (1) the district name, (2) the state where it is located, (3) the current enrollment number, (4) the full name of the current superintendent, (5) confirmation that the district includes elementary, middle, and high schools, and (6) confirmation that the district has a school board to which the superintendent reports

Universities:
- Each university must be accredited by the Middle States Commission on Higher Education (MSCHE)
- Both universities must be located in either New York or Washington, D.C.
- For each university, provide: (1) the university name, (2) the city where the main campus is located, (3) the state or district where it is located, (4) confirmation of MSCHE accreditation, (5) the number of campuses the university operates, and (6) confirmation that the university is eligible to participate in federal financial aid programs (Title IV) due to its regional accreditation

All information must be supported by verifiable URL references from official or authoritative sources.
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class DistrictInfo(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    enrollment: Optional[str] = None
    superintendent: Optional[str] = None
    has_elementary_middle_high: Optional[str] = None  # e.g., "Yes", "Confirmed", or description
    has_school_board_superintendent_reports: Optional[str] = None  # e.g., "Yes", "Confirmed", or description
    source_urls: List[str] = Field(default_factory=list)


class UniversityInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state_or_district: Optional[str] = None
    msche_accreditation_confirm: Optional[str] = None  # e.g., "Accredited by MSCHE", "Yes"
    campuses_count: Optional[str] = None
    title_iv_eligibility_confirm: Optional[str] = None  # e.g., "Eligible for Title IV", "Participates in Title IV"
    source_urls: List[str] = Field(default_factory=list)


class AnswerExtraction(BaseModel):
    districts: List[DistrictInfo] = Field(default_factory=list)
    universities: List[UniversityInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_entities() -> str:
    return """
    Extract structured information for up to three public school districts and up to two universities exactly as stated in the answer.

    For each school district, extract these fields:
    - name: Official district name, as written in the answer.
    - state: The state (e.g., "North Carolina" or "Florida").
    - enrollment: The current enrollment number as a string (e.g., "130,000", "about 125k").
    - superintendent: Full name of the current superintendent.
    - has_elementary_middle_high: A confirmation string indicating the district includes elementary, middle, and high schools (e.g., "Yes", "K-12", or a descriptive sentence).
    - has_school_board_superintendent_reports: A confirmation string indicating the district has a school board to which the superintendent reports (e.g., "Yes", "Reports to Board", or a descriptive sentence).
    - source_urls: All URLs cited in the answer that support any of the above fields for this district. Include official district websites, state DOE pages, or other authoritative sources. If no URLs are cited, return an empty list.

    For each university, extract these fields:
    - name: Official university name, as written in the answer.
    - city: City of the main campus (e.g., "New York", "Washington").
    - state_or_district: "New York" or "Washington, D.C." (or equivalent phrasing like "NY", "Washington DC").
    - msche_accreditation_confirm: A confirmation string indicating MSCHE accreditation (e.g., "Accredited by MSCHE", "Yes", or a descriptive sentence).
    - campuses_count: The number of campuses operated by the university as a string (e.g., "3", "multiple", "5+").
    - title_iv_eligibility_confirm: A confirmation string indicating eligibility to participate in Title IV federal financial aid programs (e.g., "Eligible for Title IV", "Participates in Title IV", or descriptive text).
    - source_urls: All URLs cited in the answer that support any of the above fields for this university (e.g., MSCHE directory, university website, US Department of Education listings). If no URLs are cited, return an empty list.

    Output format:
    {
      "districts": [DistrictInfo, DistrictInfo, DistrictInfo],
      "universities": [UniversityInfo, UniversityInfo]
    }

    Rules:
    - Do not invent information. Only extract what appears in the answer.
    - If a field is missing, set it to null (or empty list for source_urls).
    - Normalize URLs from markdown links to plain URLs when possible; include full protocol (http/https).
    - If the answer lists more than three districts or more than two universities, extract the first three districts and the first two universities only.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n + 1}"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_district(evaluator: Evaluator, parent_node, district: DistrictInfo, index: int) -> None:
    """
    Build verification sub-tree for a single district.
    """
    item_title = f"{_ordinal(index)} large public school district meeting all specified criteria"
    district_node = evaluator.add_parallel(
        id=f"district_{index + 1}",
        desc=item_title,
        parent=parent_node,
        critical=False  # Parent non-critical to allow partial credit across fields
    )

    # Global sources existence gate for this district item
    sources_gate = evaluator.add_custom_node(
        result=(len(district.source_urls) > 0),
        id=f"district_{index + 1}_sources_present",
        desc="At least one authoritative URL source is provided for this district",
        parent=district_node,
        critical=True
    )

    # 1) Name + reference (sequential)
    name_main = evaluator.add_sequential(
        id=f"district_{index + 1}_name",
        desc="Provide the name of the school district",
        parent=district_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(district.name),
        id=f"district_{index + 1}_name_provided",
        desc="District name is provided in the answer",
        parent=name_main,
        critical=True
    )
    name_ref_leaf = evaluator.add_leaf(
        id=f"district_{index + 1}_name_reference",
        desc="URL reference confirming the district's name",
        parent=name_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name of this school district is '{district.name}'.",
        node=name_ref_leaf,
        sources=district.source_urls,
        additional_instruction="Verify the district's official name from authoritative sources (district website, state DOE). Allow minor stylistic variants like 'Public Schools' vs 'School District'."
    )

    # 2) Location (allowed state + reference)
    loc_main = evaluator.add_sequential(
        id=f"district_{index + 1}_location",
        desc="District must be located in either North Carolina or Florida",
        parent=district_node,
        critical=False
    )
    loc_allowed_leaf = evaluator.add_leaf(
        id=f"district_{index + 1}_location_allowed",
        desc="Provided state is either North Carolina or Florida",
        parent=loc_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided state '{district.state}' is either 'North Carolina' or 'Florida'.",
        node=loc_allowed_leaf,
        sources=None,
        additional_instruction="This is a simple logical check on the provided state text. Accept common abbreviations (NC, FL) or full names with minor formatting."
    )
    loc_ref_leaf = evaluator.add_leaf(
        id=f"district_{index + 1}_location_reference",
        desc="URL reference confirming the district's state location",
        parent=loc_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"This school district is located in the state of {district.state}.",
        node=loc_ref_leaf,
        sources=district.source_urls,
        additional_instruction="Confirm the district's state via authoritative pages (district site, state DOE)."
    )

    # 3) Enrollment (threshold + reference number)
    enr_main = evaluator.add_sequential(
        id=f"district_{index + 1}_enrollment",
        desc="District must have enrollment of at least 125,000 students",
        parent=district_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(district.enrollment),
        id=f"district_{index + 1}_enrollment_provided",
        desc="Enrollment figure is provided in the answer",
        parent=enr_main,
        critical=True
    )
    enr_thresh_leaf = evaluator.add_leaf(
        id=f"district_{index + 1}_enrollment_at_least_125k",
        desc="Enrollment is at least 125,000 students",
        parent=enr_main,
        critical=True
    )
    await evaluator.verify(
        claim="The district currently enrolls at least 125,000 students.",
        node=enr_thresh_leaf,
        sources=district.source_urls,
        additional_instruction="Check official or authoritative sources for the latest enrollment; accept statements like 'over 125,000', 'approximately 130,000'."
    )
    enr_ref_leaf = evaluator.add_leaf(
        id=f"district_{index + 1}_enrollment_reference",
        desc="URL reference confirming current enrollment figures",
        parent=enr_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"The district's current enrollment is reported as '{district.enrollment}'.",
        node=enr_ref_leaf,
        sources=district.source_urls,
        additional_instruction="Verify the enrollment figure textually; allow reasonable rounding or formatting differences (commas, 'k')."
    )

    # 4) Superintendent (presence + reference)
    sup_main = evaluator.add_sequential(
        id=f"district_{index + 1}_superintendent",
        desc="Provide the current superintendent's full name",
        parent=district_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(district.superintendent),
        id=f"district_{index + 1}_superintendent_provided",
        desc="Superintendent name is provided in the answer",
        parent=sup_main,
        critical=True
    )
    sup_ref_leaf = evaluator.add_leaf(
        id=f"district_{index + 1}_superintendent_reference",
        desc="URL reference confirming the superintendent's identity",
        parent=sup_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"The current superintendent of {district.name} is {district.superintendent}.",
        node=sup_ref_leaf,
        sources=district.source_urls,
        additional_instruction="Use official leadership/superintendent pages or press releases; allow honorifics (Dr., Mr./Ms.) and middle initials."
    )

    # 5) Structure (presence + reference)
    struct_main = evaluator.add_sequential(
        id=f"district_{index + 1}_structure",
        desc="District must include elementary, middle, and high schools",
        parent=district_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(district.has_elementary_middle_high),
        id=f"district_{index + 1}_structure_provided",
        desc="Confirmation of elementary, middle, and high schools is provided",
        parent=struct_main,
        critical=True
    )
    struct_ref_leaf = evaluator.add_leaf(
        id=f"district_{index + 1}_structure_reference",
        desc="URL reference confirming the district's school composition",
        parent=struct_main,
        critical=True
    )
    await evaluator.verify(
        claim="This district includes elementary, middle, and high schools (K-12).",
        node=struct_ref_leaf,
        sources=district.source_urls,
        additional_instruction="Check 'Schools' listings or district overview pages; 'K-12' or equivalent phrasing counts as confirmation."
    )

    # 6) Governance (presence + reference)
    gov_main = evaluator.add_sequential(
        id=f"district_{index + 1}_governance",
        desc="District must have a school board to which the superintendent reports",
        parent=district_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(district.has_school_board_superintendent_reports),
        id=f"district_{index + 1}_governance_provided",
        desc="Confirmation of governance and reporting relationship is provided",
        parent=gov_main,
        critical=True
    )
    gov_ref_leaf = evaluator.add_leaf(
        id=f"district_{index + 1}_governance_reference",
        desc="URL reference confirming governance structure",
        parent=gov_main,
        critical=True
    )
    await evaluator.verify(
        claim="The superintendent reports to a school board (Board of Education).",
        node=gov_ref_leaf,
        sources=district.source_urls,
        additional_instruction="Look for governance descriptions indicating the superintendent is accountable to or reports to the Board of Education."
    )


async def verify_university(evaluator: Evaluator, parent_node, university: UniversityInfo, index: int) -> None:
    """
    Build verification sub-tree for a single university.
    """
    item_title = f"{_ordinal(index)} university meeting all specified criteria"
    uni_node = evaluator.add_parallel(
        id=f"university_{index + 1}",
        desc=item_title,
        parent=parent_node,
        critical=False
    )

    # Global sources existence gate
    sources_gate = evaluator.add_custom_node(
        result=(len(university.source_urls) > 0),
        id=f"university_{index + 1}_sources_present",
        desc="At least one authoritative URL source is provided for this university",
        parent=uni_node,
        critical=True
    )

    # 1) Name (presence + reference)
    name_main = evaluator.add_sequential(
        id=f"university_{index + 1}_name",
        desc="Provide the name of the university",
        parent=uni_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(university.name),
        id=f"university_{index + 1}_name_provided",
        desc="University name is provided in the answer",
        parent=name_main,
        critical=True
    )
    name_ref_leaf = evaluator.add_leaf(
        id=f"university_{index + 1}_name_reference",
        desc="URL reference confirming the university's name",
        parent=name_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name of the university is '{university.name}'.",
        node=name_ref_leaf,
        sources=university.source_urls,
        additional_instruction="Confirm the official name via the university website or authoritative listings."
    )

    # 2) City (presence + reference)
    city_main = evaluator.add_sequential(
        id=f"university_{index + 1}_city",
        desc="Provide the city where the university's main campus is located",
        parent=uni_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(university.city),
        id=f"university_{index + 1}_city_provided",
        desc="City of the main campus is provided",
        parent=city_main,
        critical=True
    )
    city_ref_leaf = evaluator.add_leaf(
        id=f"university_{index + 1}_city_reference",
        desc="URL reference confirming the main campus city location",
        parent=city_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"The main campus of {university.name} is located in {university.city}.",
        node=city_ref_leaf,
        sources=university.source_urls,
        additional_instruction="Verify main campus location via official 'About' or 'Campuses' pages."
    )

    # 3) State/District (allowed + reference)
    state_main = evaluator.add_sequential(
        id=f"university_{index + 1}_state",
        desc="University must be located in either New York or Washington, D.C.",
        parent=uni_node,
        critical=False
    )
    state_allowed_leaf = evaluator.add_leaf(
        id=f"university_{index + 1}_state_allowed",
        desc="Provided state/district is either New York or Washington, D.C.",
        parent=state_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided state/district '{university.state_or_district}' is either 'New York' or 'Washington, D.C.'.",
        node=state_allowed_leaf,
        sources=None,
        additional_instruction="Simple logical check; accept variants like 'NY', 'New York State', 'Washington DC', 'District of Columbia'."
    )
    state_ref_leaf = evaluator.add_leaf(
        id=f"university_{index + 1}_state_reference",
        desc="URL reference confirming the state/district location",
        parent=state_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"The university is located in {university.state_or_district}.",
        node=state_ref_leaf,
        sources=university.source_urls,
        additional_instruction="Confirm location via authoritative sources (university site, MSCHE listing, government data)."
    )

    # 4) MSCHE accreditation (presence + reference)
    accred_main = evaluator.add_sequential(
        id=f"university_{index + 1}_accreditor",
        desc="University must be accredited by the Middle States Commission on Higher Education (MSCHE)",
        parent=uni_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(university.msche_accreditation_confirm),
        id=f"university_{index + 1}_accreditor_provided",
        desc="MSCHE accreditation confirmation is provided",
        parent=accred_main,
        critical=True
    )
    accred_ref_leaf = evaluator.add_leaf(
        id=f"university_{index + 1}_accreditor_reference",
        desc="URL reference confirming MSCHE accreditation status",
        parent=accred_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"{university.name} is accredited by the Middle States Commission on Higher Education (MSCHE).",
        node=accred_ref_leaf,
        sources=university.source_urls,
        additional_instruction="Prefer MSCHE directory entries or official accreditation statements; ensure MSCHE is named explicitly."
    )

    # 5) Campuses count (presence + reference)
    campuses_main = evaluator.add_sequential(
        id=f"university_{index + 1}_campuses",
        desc="Provide the number of campuses operated by the university",
        parent=uni_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(university.campuses_count),
        id=f"university_{index + 1}_campuses_provided",
        desc="Number of campuses is provided",
        parent=campuses_main,
        critical=True
    )
    campuses_ref_leaf = evaluator.add_leaf(
        id=f"university_{index + 1}_campuses_reference",
        desc="URL reference confirming the campus count and locations",
        parent=campuses_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"The university operates '{university.campuses_count}' campuses.",
        node=campuses_ref_leaf,
        sources=university.source_urls,
        additional_instruction="Check official 'Campuses' or 'Locations' pages; allow reasonable phrasing ('multiple campuses', enumerations)."
    )

    # 6) Title IV eligibility (presence + reference)
    aid_main = evaluator.add_sequential(
        id=f"university_{index + 1}_federal_aid",
        desc="University must be eligible to participate in federal financial aid programs (Title IV)",
        parent=uni_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(university.title_iv_eligibility_confirm),
        id=f"university_{index + 1}_federal_aid_provided",
        desc="Title IV eligibility confirmation is provided",
        parent=aid_main,
        critical=True
    )
    aid_ref_leaf = evaluator.add_leaf(
        id=f"university_{index + 1}_federal_aid_reference",
        desc="URL reference confirming federal aid eligibility through regional accreditation",
        parent=aid_main,
        critical=True
    )
    await evaluator.verify(
        claim=f"{university.name} is eligible to participate in Title IV federal financial aid programs due to its regional accreditation.",
        node=aid_ref_leaf,
        sources=university.source_urls,
        additional_instruction="Confirm via authoritative sources (US Dept. of Education listings, institutional disclosures). Accept explicit statements like 'participates in Title IV' or 'eligible for federal financial aid'."
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
    Evaluate an answer for the K-12 districts and MSCHE universities task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Use parallel to avoid skipping independent university checks if district section is partial
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

    # Extract structured data
    extraction = await evaluator.extract(
        prompt=prompt_extract_entities(),
        template_class=AnswerExtraction,
        extraction_name="entities_extraction",
    )

    # Build top-level nodes
    districts_node = evaluator.add_parallel(
        id="large_districts_identification",
        desc="Identify three distinct public school districts, each with enrollment of at least 125,000 students",
        parent=root,
        critical=False
    )
    universities_node = evaluator.add_parallel(
        id="msche_universities_identification",
        desc="Identify two distinct universities accredited by the Middle States Commission on Higher Education",
        parent=root,
        critical=False
    )

    # Prepare items (filter/pad)
    districts = list(extraction.districts[:3])
    while len(districts) < 3:
        districts.append(DistrictInfo())

    universities = list(extraction.universities[:2])
    while len(universities) < 2:
        universities.append(UniversityInfo())

    # Verify districts
    for idx, dist in enumerate(districts):
        await verify_district(evaluator, districts_node, dist, idx)

    # Verify universities
    for idx, uni in enumerate(universities):
        await verify_university(evaluator, universities_node, uni, idx)

    # Custom info: counts
    evaluator.add_custom_info(
        {"extracted_districts": len(extraction.districts), "used_districts": 3,
         "extracted_universities": len(extraction.universities), "used_universities": 2},
        info_type="extraction_counts",
    )

    # Return summary
    return evaluator.get_summary()