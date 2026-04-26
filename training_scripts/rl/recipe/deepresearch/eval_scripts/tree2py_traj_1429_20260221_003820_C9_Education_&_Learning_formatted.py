import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "western_states_transfer_consortium_universities"
TASK_DESCRIPTION = (
    "A Western States Community College Transfer Consortium is establishing a regional partnership program to create "
    "streamlined transfer pathways for community college students across four western states. The consortium requires "
    "identification of one public university from each of the following states: California, Oregon, New Mexico, and Nevada.\n\n"
    "Each selected university must meet ALL of the following requirements:\n\n"
    "INSTITUTIONAL REQUIREMENTS:\n"
    "1. Must be a public university located in the specified state\n"
    "2. Must be part of a multi-campus state university system (not a standalone institution)\n"
    "3. Must hold current regional accreditation from the appropriate regional accrediting body for its geographic region\n"
    "4. The accrediting body must be one of the six federally recognized regional accreditors\n\n"
    "ACADEMIC REQUIREMENTS:\n"
    "5. Must offer at least 80 distinct undergraduate degree programs (bachelor's degrees, counting majors but not minors, certificates, or credentials)\n"
    "6. Must offer graduate degree programs (master's or doctoral level)\n"
    "7. Must have a total enrollment (undergraduate and graduate combined) of at least 12,000 students for the 2024-2025 or 2025-2026 academic year\n\n"
    "TRANSFER CREDIT REQUIREMENTS:\n"
    "8. Must have a clearly documented, publicly available transfer credit policy on its official website\n"
    "9. Must explicitly accept a minimum of 60 semester units (or 90 quarter units equivalent) from regionally accredited community colleges toward a bachelor's degree\n"
    "10. The transfer credit policy must state the specific maximum number of community college credits that can be applied toward a bachelor's degree\n\n"
    "OPERATIONAL REQUIREMENTS:\n"
    "11. Must be currently operational and accepting student applications for the Fall 2026 term\n"
    "12. Must provide clear accreditation information on its official website, including the name of the regional accrediting body\n\n"
    "For each of the four states (California, Oregon, New Mexico, and Nevada), identify ONE public university that meets all twelve requirements listed above. "
    "For each university, provide:\n"
    "- The complete official name of the university\n"
    "- The name of the state university system to which it belongs\n"
    "- The regional accrediting body name\n"
    "- The documented number of undergraduate degree programs offered\n"
    "- The total student enrollment figure\n"
    "- The maximum number of community college semester credits (or quarter credit equivalent) accepted for transfer\n"
    "- Reference URLs supporting each piece of information"
)

# Expected regional accreditors by state
STATE_ACCREDITORS = {
    "California": "WASC Senior College and University Commission (WSCUC)",
    "Oregon": "Northwest Commission on Colleges and Universities (NWCCU)",
    "Nevada": "Northwest Commission on Colleges and Universities (NWCCU)",
    "New Mexico": "Higher Learning Commission (HLC)",
}

RECOGNIZED_REGIONAL_ACCREDITORS = [
    "WASC Senior College and University Commission (WSCUC)",
    "Northwest Commission on Colleges and Universities (NWCCU)",
    "Higher Learning Commission (HLC)",
    "New England Commission of Higher Education (NECHE)",
    "Middle States Commission on Higher Education (MSCHE)",
    "Southern Association of Colleges and Schools Commission on Colleges (SACSCOC)",
]


class UniversityInfo(BaseModel):
    official_name: Optional[str] = None
    system_name: Optional[str] = None
    accreditor_name: Optional[str] = None
    undergrad_program_count: Optional[str] = None
    total_enrollment: Optional[str] = None

    min_transfer_semester_units: Optional[str] = None
    min_transfer_quarter_units: Optional[str] = None
    max_transfer_semester_units: Optional[str] = None
    max_transfer_quarter_units: Optional[str] = None

    # URL sources per requirement
    location_urls: List[str] = Field(default_factory=list)
    public_status_urls: List[str] = Field(default_factory=list)
    system_membership_urls: List[str] = Field(default_factory=list)
    accreditation_status_urls: List[str] = Field(default_factory=list)
    accreditation_info_urls: List[str] = Field(default_factory=list)

    undergraduate_programs_urls: List[str] = Field(default_factory=list)
    graduate_programs_urls: List[str] = Field(default_factory=list)
    enrollment_urls: List[str] = Field(default_factory=list)

    transfer_policy_urls: List[str] = Field(default_factory=list)
    minimum_credits_urls: List[str] = Field(default_factory=list)
    maximum_credits_urls: List[str] = Field(default_factory=list)

    operational_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    california: Optional[UniversityInfo] = None
    oregon: Optional[UniversityInfo] = None
    new_mexico: Optional[UniversityInfo] = None
    nevada: Optional[UniversityInfo] = None


def prompt_extract_universities() -> str:
    return (
        "Extract information for exactly one public university in each of the following states: California, Oregon, New Mexico, and Nevada. "
        "If the answer lists multiple options for a state, extract only the first one mentioned for that state. "
        "For each state, return an object with the following fields:\n"
        "- official_name: the complete official name of the university\n"
        "- system_name: the name of the multi-campus state university system the university belongs to\n"
        "- accreditor_name: the regional accrediting body name (e.g., WSCUC, NWCCU, HLC)\n"
        "- undergrad_program_count: the documented number of undergraduate degree programs/majors offered (as stated)\n"
        "- total_enrollment: the total student enrollment figure (UG + grad combined), preferably for 2024-2025 or 2025-2026\n"
        "- min_transfer_semester_units: the minimum semester units explicitly accepted from community colleges\n"
        "- min_transfer_quarter_units: the minimum quarter units equivalent accepted (if stated)\n"
        "- max_transfer_semester_units: the maximum semester units that can be applied toward a bachelor's degree\n"
        "- max_transfer_quarter_units: the maximum quarter units equivalent (if stated)\n"
        "For each requirement, also extract the supporting URLs mentioned in the answer text:\n"
        "- location_urls: URLs proving the university is located in the specified state (e.g., About pages, facts pages).\n"
        "- public_status_urls: URLs confirming the institution is public (e.g., About, Facts).\n"
        "- system_membership_urls: URLs confirming the university's membership in the state university system.\n"
        "- accreditation_status_urls: URLs confirming current accreditation and naming the accreditor.\n"
        "- accreditation_info_urls: URLs to the university's accreditation info page listing the accreditor.\n"
        "- undergraduate_programs_urls: URLs showing majors or counts of undergraduate degree programs.\n"
        "- graduate_programs_urls: URLs showing graduate degrees/programs.\n"
        "- enrollment_urls: URLs showing total enrollment figures.\n"
        "- transfer_policy_urls: URLs to the official transfer credit policy page.\n"
        "- minimum_credits_urls: URLs where the minimum accepted CC credits are stated.\n"
        "- maximum_credits_urls: URLs where the maximum accepted CC credits are stated.\n"
        "- operational_urls: URLs indicating applications are being accepted for Fall 2026.\n\n"
        "Return a JSON object with keys: california, oregon, new_mexico, nevada. "
        "Each key maps to the described object. If any field or URL is not present in the answer, set it to null or an empty list as appropriate. "
        "Extract only information explicitly present in the answer; do not infer or add anything not stated."
    )


def _has_any_url(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str) and len(u.strip()) > 0:
            return True
    return False


def _safe_name(name: Optional[str], fallback: str) -> str:
    return name.strip() if name else fallback


async def _verify_institutional_requirements(
    evaluator: Evaluator,
    parent,
    state_label: str,
    uni: UniversityInfo,
    expected_accreditor: str,
) -> None:
    inst_node = evaluator.add_parallel(
        id=f"{state_label}_Institutional_Requirements",
        desc=f"Verify the {state_label} university meets all institutional requirements including location, public status, system membership, and accreditation",
        parent=parent,
        critical=True,
    )

    uni_name = _safe_name(uni.official_name, f"the selected {state_label} university")

    # Location
    loc_node = evaluator.add_parallel(
        id=f"{state_label}_State_Location",
        desc=f"Verify the university is located in {state_label}",
        parent=inst_node,
        critical=True,
    )
    loc_leaf = evaluator.add_leaf(
        id=f"{state_label}_State_Located",
        desc=f"The university is physically located in the state of {state_label}",
        parent=loc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni_name} is located in {state_label}.",
        node=loc_leaf,
        sources=uni.location_urls,
        additional_instruction="Use official pages (About, Facts, Contact, campus information) to confirm the institution's state location."
    )
    evaluator.add_custom_node(
        result=_has_any_url(uni.location_urls),
        id=f"{state_label}_State_URL",
        desc=f"Reference URL provided documenting the university's {state_label} location",
        parent=loc_node,
        critical=True,
    )

    # Public status
    pub_node = evaluator.add_parallel(
        id=f"{state_label}_Public_Status",
        desc="Verify the university is a public institution",
        parent=inst_node,
        critical=True,
    )
    pub_leaf = evaluator.add_leaf(
        id=f"{state_label}_Public_Institution",
        desc="The university is a public institution, not a private university",
        parent=pub_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni_name} is a public university.",
        node=pub_leaf,
        sources=uni.public_status_urls,
        additional_instruction="Confirm that the institution is public (state-supported). Use official About/Facts pages or state system sites."
    )
    evaluator.add_custom_node(
        result=_has_any_url(uni.public_status_urls),
        id=f"{state_label}_Public_URL",
        desc="Reference URL provided documenting the university's public status",
        parent=pub_node,
        critical=True,
    )

    # System membership
    sys_node = evaluator.add_parallel(
        id=f"{state_label}_System_Membership",
        desc="Verify the university is part of a multi-campus state university system",
        parent=inst_node,
        critical=True,
    )
    sys_leaf = evaluator.add_leaf(
        id=f"{state_label}_Multi_Campus_System",
        desc="The university is part of a multi-campus state university system",
        parent=sys_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{uni_name} is part of the {uni.system_name} multi-campus state university system.",
        node=sys_leaf,
        sources=uni.system_membership_urls,
        additional_instruction="The system must be a multi-campus public system (e.g., CSU, UC, NSHE, Oregon public university network). Verify official membership pages."
    )
    evaluator.add_custom_node(
        result=_has_any_url(uni.system_membership_urls),
        id=f"{state_label}_System_URL",
        desc="Reference URL provided documenting the university's system membership",
        parent=sys_node,
        critical=True,
    )

    # Regional accreditation
    acc_node = evaluator.add_parallel(
        id=f"{state_label}_Regional_Accreditation",
        desc="Verify the university holds appropriate regional accreditation",
        parent=inst_node,
        critical=True,
    )
    acc_leaf = evaluator.add_leaf(
        id=f"{state_label}_Accredited_By_{expected_accreditor.split()[0]}",
        desc=f"The university holds current regional accreditation from {expected_accreditor}",
        parent=acc_node,
        critical=True,
    )
    combined_acc_urls = (uni.accreditation_status_urls or []) + (uni.accreditation_info_urls or [])
    await evaluator.verify(
        claim=f"{uni_name} holds current regional accreditation from {expected_accreditor}.",
        node=acc_leaf,
        sources=combined_acc_urls,
        additional_instruction="Confirm the specific regional accreditor name on the official accreditation page or accreditor's directory. The accreditor must be one of the six recognized U.S. regional accreditors."
    )
    evaluator.add_custom_node(
        result=_has_any_url(uni.accreditation_status_urls),
        id=f"{state_label}_Accreditation_URL",
        desc=f"Reference URL provided documenting the university's accreditation status with {expected_accreditor}",
        parent=acc_node,
        critical=True,
    )


async def _verify_academic_requirements(
    evaluator: Evaluator,
    parent,
    state_label: str,
    uni: UniversityInfo,
) -> None:
    acad_node = evaluator.add_parallel(
        id=f"{state_label}_Academic_Requirements",
        desc=f"Verify the {state_label} university meets all academic program and enrollment requirements",
        parent=parent,
        critical=True,
    )

    # Undergrad programs
    ug_node = evaluator.add_parallel(
        id=f"{state_label}_Undergraduate_Programs",
        desc="Verify the university offers sufficient undergraduate degree programs",
        parent=acad_node,
        critical=True,
    )
    ug_leaf = evaluator.add_leaf(
        id=f"{state_label}_Program_Count_Minimum",
        desc="The university offers at least 80 distinct undergraduate degree programs",
        parent=ug_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_safe_name(uni.official_name, 'The university')} offers at least 80 distinct undergraduate degree programs (majors; excluding minors/certificates).",
        node=ug_leaf,
        sources=uni.undergraduate_programs_urls,
        additional_instruction="Check official majors/programs pages or fact books. The count must be 80+ majors; do not count minors, certificates, or credentials."
    )
    evaluator.add_custom_node(
        result=_has_any_url(uni.undergraduate_programs_urls),
        id=f"{state_label}_Program_Count_URL",
        desc="Reference URL provided documenting the number of undergraduate programs offered",
        parent=ug_node,
        critical=True,
    )

    # Graduate programs
    grad_node = evaluator.add_parallel(
        id=f"{state_label}_Graduate_Programs",
        desc="Verify the university offers graduate degree programs",
        parent=acad_node,
        critical=True,
    )
    grad_leaf = evaluator.add_leaf(
        id=f"{state_label}_Graduate_Programs_Exist",
        desc="The university offers graduate degree programs at the master's or doctoral level",
        parent=grad_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_safe_name(uni.official_name, 'The university')} offers graduate degree programs (master's and/or doctoral).",
        node=grad_leaf,
        sources=uni.graduate_programs_urls,
        additional_instruction="Use official graduate school or program pages listing master's/doctoral degrees."
    )
    evaluator.add_custom_node(
        result=_has_any_url(uni.graduate_programs_urls),
        id=f"{state_label}_Graduate_Programs_URL",
        desc="Reference URL provided documenting graduate program offerings",
        parent=grad_node,
        critical=True,
    )

    # Enrollment minimum
    enr_node = evaluator.add_parallel(
        id=f"{state_label}_Enrollment",
        desc="Verify the university meets minimum enrollment requirements",
        parent=acad_node,
        critical=True,
    )
    enr_leaf = evaluator.add_leaf(
        id=f"{state_label}_Enrollment_Minimum",
        desc="The university has a total enrollment of at least 12,000 students for the 2024-2025 or 2025-2026 academic year",
        parent=enr_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_safe_name(uni.official_name, 'The university')} has total enrollment (UG+grad) of at least 12,000 students in the 2024-2025 or 2025-2026 academic year.",
        node=enr_leaf,
        sources=uni.enrollment_urls,
        additional_instruction="Use official facts, common data set, institutional research, or enrollment pages. Prefer 2024-2025 or 2025-2026; slight seasonal timing variances acceptable if the figure clearly exceeds 12,000."
    )
    evaluator.add_custom_node(
        result=_has_any_url(uni.enrollment_urls),
        id=f"{state_label}_Enrollment_URL",
        desc="Reference URL provided documenting the university's total enrollment figures",
        parent=enr_node,
        critical=True,
    )


async def _verify_transfer_requirements(
    evaluator: Evaluator,
    parent,
    state_label: str,
    uni: UniversityInfo,
) -> None:
    transfer_node = evaluator.add_parallel(
        id=f"{state_label}_Transfer_Credit_Requirements",
        desc=f"Verify the {state_label} university meets all transfer credit policy requirements",
        parent=parent,
        critical=True,
    )

    # Policy documentation
    pol_node = evaluator.add_parallel(
        id=f"{state_label}_Transfer_Policy_Documentation",
        desc="Verify the university has a documented, publicly available transfer credit policy",
        parent=transfer_node,
        critical=True,
    )
    pol_leaf = evaluator.add_leaf(
        id=f"{state_label}_Policy_Exists_Public",
        desc="The university has a clearly documented transfer credit policy that is publicly accessible",
        parent=pol_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_safe_name(uni.official_name, 'The university')} has a publicly accessible transfer credit policy on its official website.",
        node=pol_leaf,
        sources=uni.transfer_policy_urls,
        additional_instruction="Confirm that the policy page is official and clearly outlines transfer credit rules (not a third-party site)."
    )
    evaluator.add_custom_node(
        result=_has_any_url(uni.transfer_policy_urls),
        id=f"{state_label}_Policy_URL",
        desc="Reference URL provided for the university's transfer credit policy documentation",
        parent=pol_node,
        critical=True,
    )

    # Minimum credits
    min_node = evaluator.add_parallel(
        id=f"{state_label}_Minimum_Transfer_Credits",
        desc="Verify the university accepts sufficient community college transfer credits",
        parent=transfer_node,
        critical=True,
    )
    min_leaf = evaluator.add_leaf(
        id=f"{state_label}_Accepts_Minimum_Credits",
        desc="The university explicitly accepts a minimum of 60 semester units or 90 quarter units from regionally accredited community colleges",
        parent=min_node,
        critical=True,
    )
    min_sources = (uni.minimum_credits_urls or []) + (uni.transfer_policy_urls or [])
    await evaluator.verify(
        claim=f"{_safe_name(uni.official_name, 'The university')} explicitly accepts at least 60 semester units or 90 quarter units from regionally accredited community colleges toward a bachelor's degree.",
        node=min_leaf,
        sources=min_sources,
        additional_instruction="Look for explicit minimum transfer limits (e.g., 'up to 60 semester units' or '90 quarter units'). The policy must state this minimum."
    )
    evaluator.add_custom_node(
        result=_has_any_url(uni.minimum_credits_urls),
        id=f"{state_label}_Minimum_Credits_URL",
        desc="Reference URL provided documenting the minimum community college credits accepted",
        parent=min_node,
        critical=True,
    )

    # Maximum credits
    max_node = evaluator.add_parallel(
        id=f"{state_label}_Maximum_Transfer_Credits",
        desc="Verify the transfer credit policy states the maximum credits accepted",
        parent=transfer_node,
        critical=True,
    )
    max_leaf = evaluator.add_leaf(
        id=f"{state_label}_Maximum_Stated",
        desc="The transfer credit policy explicitly states the specific maximum number of community college credits that can be applied toward a bachelor's degree",
        parent=max_node,
        critical=True,
    )
    max_sources = (uni.maximum_credits_urls or []) + (uni.transfer_policy_urls or [])
    await evaluator.verify(
        claim=f"The transfer credit policy for {_safe_name(uni.official_name, 'the university')} explicitly states the maximum number of community college credits that can be applied toward a bachelor's degree.",
        node=max_leaf,
        sources=max_sources,
        additional_instruction="Find language such as 'maximum transferable credits' or 'no more than X semester/quarter credits may be applied.'"
    )
    evaluator.add_custom_node(
        result=_has_any_url(uni.maximum_credits_urls),
        id=f"{state_label}_Maximum_Credits_URL",
        desc="Reference URL provided documenting the maximum community college credits accepted",
        parent=max_node,
        critical=True,
    )


async def _verify_operational_requirements(
    evaluator: Evaluator,
    parent,
    state_label: str,
    uni: UniversityInfo,
    expected_accreditor: str,
) -> None:
    op_node = evaluator.add_parallel(
        id=f"{state_label}_Operational_Requirements",
        desc=f"Verify the {state_label} university meets all operational requirements",
        parent=parent,
        critical=True,
    )

    # Currently operational and accepting Fall 2026
    cur_node = evaluator.add_parallel(
        id=f"{state_label}_Currently_Operational",
        desc="Verify the university is currently operational and accepting applications",
        parent=op_node,
        critical=True,
    )
    cur_leaf = evaluator.add_leaf(
        id=f"{state_label}_Operational_Fall_2026",
        desc="The university is currently operational and accepting student applications for the Fall 2026 term",
        parent=cur_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_safe_name(uni.official_name, 'The university')} is currently operational and accepting student applications for Fall 2026.",
        node=cur_leaf,
        sources=uni.operational_urls,
        additional_instruction="Admissions or academic calendars should explicitly reference Fall 2026 application periods, deadlines, or availability."
    )
    evaluator.add_custom_node(
        result=_has_any_url(uni.operational_urls),
        id=f"{state_label}_Operational_URL",
        desc="Reference URL provided documenting the university's operational status and Fall 2026 application acceptance",
        parent=cur_node,
        critical=True,
    )

    # Accreditation information clarity on website
    ai_node = evaluator.add_parallel(
        id=f"{state_label}_Accreditation_Information",
        desc="Verify accreditation information is clearly available on the website",
        parent=op_node,
        critical=True,
    )
    ai_leaf = evaluator.add_leaf(
        id=f"{state_label}_Accreditation_Info_Clear",
        desc="The university's official website provides clear accreditation information including the name of the regional accrediting body",
        parent=ai_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official website of {_safe_name(uni.official_name, 'the university')} clearly provides accreditation information including the name of the regional accrediting body ({expected_accreditor}).",
        node=ai_leaf,
        sources=uni.accreditation_info_urls,
        additional_instruction="Look for an 'Accreditation' page or comparable official page clearly naming the regional accreditor and stating current accreditation."
    )
    evaluator.add_custom_node(
        result=_has_any_url(uni.accreditation_info_urls),
        id=f"{state_label}_Accreditation_Info_URL",
        desc="Reference URL provided for the university's public accreditation information page",
        parent=ai_node,
        critical=True,
    )


async def _verify_state_university(
    evaluator: Evaluator,
    parent,
    state_key: str,
    state_label: str,
    uni: Optional[UniversityInfo],
    expected_accreditor: str,
) -> None:
    # Create the state node; to comply with critical-parent constraint (main node critical),
    # we set the state node as critical too so failing any requirement fails the entire task.
    state_node = evaluator.add_parallel(
        id=f"{state_label}_University",
        desc=f"Identify one public university in {state_label} that meets all specified requirements",
        parent=parent,
        critical=True,
    )

    # Existence check for university name and system presence (basic sanity)
    evaluator.add_custom_node(
        result=bool(uni and uni.official_name and uni.system_name),
        id=f"{state_label}_University_Exists",
        desc=f"A university is named for {state_label} with a stated system membership",
        parent=state_node,
        critical=True,
    )

    if not uni:
        # If no data extracted, create failing leaves quickly to reflect missing info
        await _verify_institutional_requirements(evaluator, state_node, state_label, UniversityInfo(), expected_accreditor)
        await _verify_academic_requirements(evaluator, state_node, state_label, UniversityInfo())
        await _verify_transfer_requirements(evaluator, state_node, state_label, UniversityInfo())
        await _verify_operational_requirements(evaluator, state_node, state_label, UniversityInfo(), expected_accreditor)
        return

    await _verify_institutional_requirements(evaluator, state_node, state_label, uni, expected_accreditor)
    await _verify_academic_requirements(evaluator, state_node, state_label, uni)
    await _verify_transfer_requirements(evaluator, state_node, state_label, uni)
    await _verify_operational_requirements(evaluator, state_node, state_label, uni, expected_accreditor)


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

    extraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    evaluator.add_ground_truth({
        "expected_accreditors_by_state": STATE_ACCREDITORS,
        "recognized_regional_accreditors": RECOGNIZED_REGIONAL_ACCREDITORS,
        "states": ["California", "Oregon", "New Mexico", "Nevada"],
        "requirements_total": 12
    })

    # Main verification node mirroring the rubric root; set critical to True
    main = evaluator.add_parallel(
        id="Western_States_Transfer_Consortium_University_Identification",
        desc="Identify one public university from each of four western states (California, Oregon, New Mexico, Nevada) that meets comprehensive institutional, academic, transfer credit, and operational requirements",
        parent=root,
        critical=True,
    )

    # Verify each state
    await _verify_state_university(
        evaluator,
        main,
        "california",
        "California",
        extraction.california,
        STATE_ACCREDITORS["California"],
    )

    await _verify_state_university(
        evaluator,
        main,
        "oregon",
        "Oregon",
        extraction.oregon,
        STATE_ACCREDITORS["Oregon"],
    )

    await _verify_state_university(
        evaluator,
        main,
        "new_mexico",
        "New Mexico",
        extraction.new_mexico,
        STATE_ACCREDITORS["New Mexico"],
    )

    await _verify_state_university(
        evaluator,
        main,
        "nevada",
        "Nevada",
        extraction.nevada,
        STATE_ACCREDITORS["Nevada"],
    )

    return evaluator.get_summary()