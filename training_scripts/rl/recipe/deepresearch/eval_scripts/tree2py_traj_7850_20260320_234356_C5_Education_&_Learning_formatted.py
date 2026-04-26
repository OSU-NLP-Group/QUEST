import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "belmont_transfer_plan_fall2026"
TASK_DESCRIPTION = (
    "I am a transfer student from an Ohio community college who has earned 50 semester credit hours and I am planning "
    "to apply to Belmont University in Nashville, Tennessee for Fall 2026 enrollment. I need to create a comprehensive "
    "enrollment preparation plan that addresses all critical requirements and deadlines. Please provide the following "
    "information: (1) Application Process: All available application deadline options for Fall 2026 with specific dates, "
    "when admission decisions are released for each option, which application platforms Belmont accepts, information about "
    "scholarship eligibility based on application timing, and a reference URL to Belmont's official application information. "
    "(2) Housing Requirements: The general on-campus living requirement for undergraduate students, whether transfer students "
    "are exempt from this requirement, other exemption criteria that might apply, when housing information is sent after paying "
    "the deposit, how the priority deposit deadline affects housing and orientation selection, and a reference URL to Belmont's "
    "housing policy page. (3) Enrollment Deposits: The required enrollment deposit amount, the priority enrollment deposit deadline "
    "and what benefits it provides, the final enrollment deposit deadline, whether the deposit is refundable, and a reference URL "
    "to Belmont's official deposit information. (4) Academic Calendar: The typical start date for the fall semester (using Fall 2025 "
    "as a reference for Fall 2026 planning), when summer orientation typically occurs for fall enrollees, major academic breaks in the "
    "calendar, and a reference URL to Belmont's academic calendar. (5) Transfer Credit Information: How Ohio Transfer 36 works and its "
    "benefits for transferring general education credits, minimum grade requirements for transfer credits, any special benefits for "
    "students with associate degrees from Ohio public institutions, and a reference URL to relevant transfer credit information. "
    "For each section, include direct links to official Belmont University pages where the information can be verified."
)


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class ApplicationDeadlines(BaseModel):
    ea1_date: Optional[str] = None
    ea2_date: Optional[str] = None
    rd_date: Optional[str] = None


class ApplicationDecisions(BaseModel):
    ea1_decision_by: Optional[str] = None
    ea2_decision_by: Optional[str] = None
    rd_decision_timeline: Optional[str] = None  # e.g., "Jan 2026" or "January 2026"


class ApplicationSection(BaseModel):
    deadlines: ApplicationDeadlines = Field(default_factory=ApplicationDeadlines)
    decisions: ApplicationDecisions = Field(default_factory=ApplicationDecisions)
    platforms: List[str] = Field(default_factory=list)
    scholarship_timing: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HousingSection(BaseModel):
    general_requirement: Optional[str] = None
    transfer_exempt: Optional[bool] = None
    other_exemptions: List[str] = Field(default_factory=list)
    info_sent_timeline: Optional[str] = None
    priority_deposit_impact: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class DepositSection(BaseModel):
    amount: Optional[str] = None
    priority_deadline: Optional[str] = None
    priority_benefits: List[str] = Field(default_factory=list)
    final_deadline: Optional[str] = None
    refundable: Optional[str] = None  # Use the exact phrasing from the answer, e.g., "non-refundable"
    urls: List[str] = Field(default_factory=list)


class AcademicCalendarSection(BaseModel):
    fall_start_2025: Optional[str] = None
    orientation_timing: Optional[str] = None
    major_breaks: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class TransferCreditSection(BaseModel):
    ohio_transfer_36: Optional[str] = None
    min_grade: Optional[str] = None
    assoc_degree_benefit: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PlanExtraction(BaseModel):
    application: ApplicationSection = Field(default_factory=ApplicationSection)
    housing: HousingSection = Field(default_factory=HousingSection)
    deposit: DepositSection = Field(default_factory=DepositSection)
    academic: AcademicCalendarSection = Field(default_factory=AcademicCalendarSection)
    transfer: TransferCreditSection = Field(default_factory=TransferCreditSection)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
Extract a structured plan from the answer for a transfer student applying to Belmont University for Fall 2026.

For each section, pull ONLY what is explicitly stated in the answer text. Do not infer or invent. Capture dates exactly as written (e.g., "Nov 1, 2025" vs "November 1, 2025"), and keep phrasing for policy/benefit statements faithful to the answer.

1) Application section:
   - deadlines.ea1_date: Early Action I deadline date (Fall 2026 cycle), e.g., "Nov 1, 2025"
   - deadlines.ea2_date: Early Action II deadline date (Fall 2026 cycle), e.g., "Dec 1, 2025"
   - deadlines.rd_date: Regular Decision deadline date (Fall 2026 cycle), e.g., "Mar 1, 2026"
   - decisions.ea1_decision_by: EA I decision release timing, e.g., "by Nov 20, 2025"
   - decisions.ea2_decision_by: EA II decision release timing, e.g., "by Dec 21, 2025"
   - decisions.rd_decision_timeline: Regular Decision release timing, e.g., "Jan 2026" or "January 2026"
   - platforms: list all named application platforms accepted (e.g., "Common App", "Belmont application")
   - scholarship_timing: statement describing how scholarship eligibility depends on application timing
   - urls: list all URLs the answer cites for Belmont application info (include ONLY URLs explicitly shown in the answer)

2) Housing section:
   - general_requirement: the general on-campus living requirement as written in the answer
   - transfer_exempt: true/false if the answer explicitly states transfer students are exempt; null if not stated
   - other_exemptions: list other exemption criteria as enumerated (e.g., "21+ by first day of classes", "married/has children", "living with parents/guardians/..."); use the answer's phrasing
   - info_sent_timeline: timing for when housing information is sent after the deposit is processed (verbatim)
   - priority_deposit_impact: how meeting the priority deposit deadline affects housing/orientation selection
   - urls: list all URLs the answer cites for housing/residence life policy information (Belmont official pages)

3) Deposit section:
   - amount: the enrollment deposit amount as stated (e.g., "$400")
   - priority_deadline: the priority deposit deadline date (e.g., "Mar 1, 2026")
   - priority_benefits: list the benefits tied to meeting the priority deadline (verbatim phrases from the answer)
   - final_deadline: the final deposit deadline date (e.g., "May 1, 2026")
   - refundable: whether the deposit is refundable/non-refundable; capture the exact wording from the answer
   - urls: list all URLs the answer cites for Belmont deposit information

4) Academic section:
   - fall_start_2025: the Fall 2025 start date used as a reference (e.g., "Aug 20, 2025")
   - orientation_timing: typical month/timing of summer orientation for fall enrollees (e.g., "June")
   - major_breaks: list of major academic breaks with names and dates/date ranges as stated (verbatim)
   - urls: list all URLs the answer cites for Belmont academic calendar

5) Transfer credit section:
   - ohio_transfer_36: explanation/benefit of Ohio Transfer 36 exactly as stated in the answer
   - min_grade: minimum grade requirement for transfer credits (verbatim, e.g., "C or higher")
   - assoc_degree_benefit: benefit for students with associate degrees from Ohio public institutions (verbatim)
   - urls: list all URLs the answer cites related to transfer credit evaluation/policies (include Belmont official links if present)

SPECIAL RULES FOR URL FIELDS:
- Only extract URLs explicitly shown in the answer. Do not infer.
- Include full URLs (add http:// if protocol missing).
- Prefer Belmont official pages when available in the answer (belmont.edu), but still extract all URLs given.

Return a single JSON object following the PlanExtraction schema described above.
    """.strip()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _val_or_missing(v: Optional[str]) -> str:
    return v.strip() if isinstance(v, str) and v.strip() else "MISSING"


def _list_or_missing(items: Optional[List[str]]) -> str:
    if not items:
        return "MISSING"
    cleaned = [s.strip() for s in items if isinstance(s, str) and s.strip()]
    return ", ".join(cleaned) if cleaned else "MISSING"


def _has_belmont_url(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str) and "belmont.edu" in u.lower():
            return True
    return False


def _belmont_only(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and "belmont.edu" in u.lower()]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_application_section(evaluator: Evaluator, parent, data: ApplicationSection) -> None:
    section = evaluator.add_parallel(
        id="Application_Process_Information",
        desc="Application deadlines/options, decision timing, platforms, scholarship timing, and official verification link(s).",
        parent=parent,
        critical=True,
    )

    # Official Belmont URL presence (critical)
    app_belmont_url_node = evaluator.add_custom_node(
        result=_has_belmont_url(data.urls),
        id="Application_Official_Belmont_URL",
        desc="Provides at least one direct URL to an official Belmont University page where the application information can be verified.",
        parent=section,
        critical=True,
    )

    belmont_sources = _belmont_only(data.urls)

    # Deadlines - exact and complete set
    deadlines_leaf = evaluator.add_leaf(
        id="Application_Deadline_Options_and_Dates",
        desc="States ALL Fall 2026 application deadline options and dates: Early Action I, Early Action II, Regular Decision.",
        parent=section,
        critical=True,
    )
    d_ea1 = _val_or_missing(data.deadlines.ea1_date)
    d_ea2 = _val_or_missing(data.deadlines.ea2_date)
    d_rd = _val_or_missing(data.deadlines.rd_date)
    claim_deadlines = (
        f"Belmont University's Fall 2026 undergraduate application deadlines are exactly and completely: "
        f"Early Action I — {d_ea1}; Early Action II — {d_ea2}; Regular Decision — {d_rd}. "
        f"There are no additional Fall 2026 application deadline options."
    )
    await evaluator.verify(
        claim=claim_deadlines,
        node=deadlines_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Judge strictly for completeness and date accuracy. Accept minor month formatting variants (e.g., Nov vs November; Mar vs March). "
                               "If the webpage lists any additional options or different dates, mark as not supported.",
    )

    # Decision release timeline by option
    decisions_leaf = evaluator.add_leaf(
        id="Decision_Release_Timeline_by_Option",
        desc="States decision release timing for each option: EA I, EA II, Regular Decision rolling beginning Jan 2026.",
        parent=section,
        critical=True,
    )
    t_ea1 = _val_or_missing(data.decisions.ea1_decision_by)
    t_ea2 = _val_or_missing(data.decisions.ea2_decision_by)
    t_rd = _val_or_missing(data.decisions.rd_decision_timeline)
    claim_decisions = (
        f"Admission decision release timing for Fall 2026 is exactly: "
        f"Early Action I decisions by {t_ea1}; Early Action II decisions by {t_ea2}; "
        f"Regular Decision notifications released on a rolling basis beginning {t_rd}."
    )
    await evaluator.verify(
        claim=claim_decisions,
        node=decisions_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Treat phrasing variants like 'by' vs 'no later than' as acceptable if equivalent. For 'rolling beginning', the month/year must match.",
    )

    # Application platforms accepted (exact list)
    platforms_leaf = evaluator.add_leaf(
        id="Application_Platforms_Accepted",
        desc="Identifies which application platform(s) Belmont accepts (names the platform(s)).",
        parent=section,
        critical=True,
    )
    platforms_str = _list_or_missing(data.platforms)
    claim_platforms = (
        f"Belmont University accepts undergraduate transfer applications via exactly the following platform(s): {platforms_str}. No other platforms are accepted."
    )
    await evaluator.verify(
        claim=claim_platforms,
        node=platforms_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Allow reasonable naming variants (e.g., 'Common App' vs 'The Common Application'). If the official page lists any other platforms not in the claim, mark as not supported.",
    )

    # Scholarship eligibility by timing
    scholarships_leaf = evaluator.add_leaf(
        id="Scholarship_Eligibility_by_Timing",
        desc="Explains how scholarship consideration/eligibility varies based on application timing/deadline choice.",
        parent=section,
        critical=True,
    )
    schol_text = _val_or_missing(data.scholarship_timing)
    claim_scholar = f"Scholarship consideration/eligibility by application timing is described as: {schol_text}"
    await evaluator.verify(
        claim=claim_scholar,
        node=scholarships_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Verify that the official page(s) explicitly support the description of how scholarship consideration depends on the deadline chosen.",
    )


async def verify_housing_section(evaluator: Evaluator, parent, data: HousingSection) -> None:
    section = evaluator.add_parallel(
        id="Housing_Requirements_and_Policies",
        desc="On-campus living requirement, transfer exemption, other exemptions, timeline after deposit, priority deposit impact, and official verification link(s).",
        parent=parent,
        critical=True,
    )

    # Official Belmont URL presence
    housing_belmont_url_node = evaluator.add_custom_node(
        result=_has_belmont_url(data.urls),
        id="Housing_Official_Belmont_URL",
        desc="Provides at least one direct URL to an official Belmont University housing/residential life policy page.",
        parent=section,
        critical=True,
    )
    belmont_sources = _belmont_only(data.urls)

    # General on-campus living requirement
    general_req_leaf = evaluator.add_leaf(
        id="General_On_Campus_Living_Requirement",
        desc="States the general housing requirement: full-time undergraduates with fewer than 60 credit hours at start of fall semester must live on campus.",
        parent=section,
        critical=True,
    )
    claim_general = f"Belmont's general undergraduate on-campus living requirement is: {_val_or_missing(data.general_requirement)}"
    await evaluator.verify(
        claim=claim_general,
        node=general_req_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Confirm that the policy states that full-time undergraduates with fewer than 60 earned credit hours at the start of the fall term must live on campus. If the threshold or condition differs or is missing, mark as not supported.",
    )

    # Transfer student exemption status
    transfer_exempt_leaf = evaluator.add_leaf(
        id="Transfer_Student_Exemption_Status",
        desc="States that transfer students are explicitly exempt from Belmont's on-campus living requirement.",
        parent=section,
        critical=True,
    )
    if data.transfer_exempt is None:
        transfer_statement = "MISSING"
    else:
        transfer_statement = (
            "Transfer students are explicitly exempt from the on-campus living requirement."
            if data.transfer_exempt else
            "Transfer students are not exempt from the on-campus living requirement."
        )
    await evaluator.verify(
        claim=transfer_statement,
        node=transfer_exempt_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Verify the policy specifically addresses transfer students' housing requirement (exempt vs not exempt).",
    )

    # Other housing exemptions
    other_exempt_leaf = evaluator.add_leaf(
        id="Other_Housing_Exemptions",
        desc="Lists other exemption criteria (21+, married/with children, living with parents/legal guardians/grandparents/siblings over 25).",
        parent=section,
        critical=True,
    )
    other_list = _list_or_missing(data.other_exemptions)
    claim_other = f"Other housing exemptions include: {other_list}"
    await evaluator.verify(
        claim=claim_other,
        node=other_exempt_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Check the official policy lists these exemption categories. Allow minor wording variants but the categories must align.",
    )

    # Housing info sent timeline after deposit
    info_timeline_leaf = evaluator.add_leaf(
        id="Housing_Info_Sent_Timeline",
        desc="States housing information is sent within one week after the enrollment deposit is processed.",
        parent=section,
        critical=True,
    )
    claim_info_timeline = f"Housing information is sent {_val_or_missing(data.info_sent_timeline)} after the enrollment deposit is processed."
    await evaluator.verify(
        claim=claim_info_timeline,
        node=info_timeline_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Confirm the timeline as stated (e.g., 'within one week' if that's the claim).",
    )

    # Priority deposit impact on housing/orientation selection
    priority_impact_leaf = evaluator.add_leaf(
        id="Priority_Deposit_Impact_on_Housing_and_Orientation",
        desc="Explains how meeting the priority deposit deadline affects housing selection and orientation date selection.",
        parent=section,
        critical=True,
    )
    claim_priority_impact = f"Meeting the priority deposit deadline affects access/priority for housing selection and orientation as follows: {_val_or_missing(data.priority_deposit_impact)}"
    await evaluator.verify(
        claim=claim_priority_impact,
        node=priority_impact_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Verify the official page explains earlier access/priority for housing and orientation tied to the priority deposit.",
    )


async def verify_deposit_section(evaluator: Evaluator, parent, data: DepositSection) -> None:
    section = evaluator.add_parallel(
        id="Enrollment_Deposit_Requirements",
        desc="Deposit amount, deadlines, benefits, refundability, and official verification link(s).",
        parent=parent,
        critical=True,
    )

    # Official Belmont URL presence
    deposit_belmont_url_node = evaluator.add_custom_node(
        result=_has_belmont_url(data.urls),
        id="Deposit_Official_Belmont_URL",
        desc="Provides at least one direct URL to an official Belmont University page where deposit information can be verified.",
        parent=section,
        critical=True,
    )
    belmont_sources = _belmont_only(data.urls)

    # Deposit amount
    amount_leaf = evaluator.add_leaf(
        id="Enrollment_Deposit_Amount",
        desc="States the enrollment deposit amount is $400.",
        parent=section,
        critical=True,
    )
    claim_amount = f"The required enrollment deposit amount is {_val_or_missing(data.amount)}."
    await evaluator.verify(
        claim=claim_amount,
        node=amount_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Accept '$400' vs '400 USD' as equivalent if clearly the same amount.",
    )

    # Priority deposit deadline and benefits
    priority_leaf = evaluator.add_leaf(
        id="Priority_Deposit_Deadline_and_Benefits",
        desc="States the priority enrollment deposit deadline and specifies its benefits (e.g., first-choice room type + preferred orientation date).",
        parent=section,
        critical=True,
    )
    benefits_str = _list_or_missing(data.priority_benefits)
    claim_priority = f"The priority enrollment deposit deadline is {_val_or_missing(data.priority_deadline)}, and its benefits are: {benefits_str}."
    await evaluator.verify(
        claim=claim_priority,
        node=priority_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Verify both the date and the list of benefits are supported by the official page(s).",
    )

    # Final deposit deadline
    final_deadline_leaf = evaluator.add_leaf(
        id="Final_Deposit_Deadline",
        desc="States the final enrollment deposit deadline (e.g., May 1, 2026).",
        parent=section,
        critical=True,
    )
    claim_final = f"The final enrollment deposit deadline is {_val_or_missing(data.final_deadline)}."
    await evaluator.verify(
        claim=claim_final,
        node=final_deadline_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Dates like 'May 1' should be accepted whether written as 'May 1' or 'May 1st' if equivalent.",
    )

    # Deposit refundability
    refundable_leaf = evaluator.add_leaf(
        id="Deposit_Refundability",
        desc="States the enrollment deposit is non-refundable (or refundable, as claimed).",
        parent=section,
        critical=True,
    )
    claim_refund = f"The enrollment deposit refundability policy is: {_val_or_missing(data.refundable)}."
    await evaluator.verify(
        claim=claim_refund,
        node=refundable_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Confirm whether the policy is 'non-refundable' or otherwise as explicitly stated.",
    )


async def verify_academic_section(evaluator: Evaluator, parent, data: AcademicCalendarSection) -> None:
    section = evaluator.add_parallel(
        id="Academic_Calendar_and_Orientation",
        desc="Fall start reference, orientation timing, major breaks, and official verification link(s).",
        parent=parent,
        critical=True,
    )

    # Official Belmont URL presence
    calendar_belmont_url_node = evaluator.add_custom_node(
        result=_has_belmont_url(data.urls),
        id="Academic_Calendar_Official_Belmont_URL",
        desc="Provides at least one direct URL to an official Belmont University academic calendar page.",
        parent=section,
        critical=True,
    )
    belmont_sources = _belmont_only(data.urls)

    # Fall start date reference (from Fall 2025)
    start_ref_leaf = evaluator.add_leaf(
        id="Fall_Start_Date_Reference_From_Fall_2025",
        desc="Uses Fall 2025 as reference for planning and states the fall term start date.",
        parent=section,
        critical=True,
    )
    claim_start = f"The Fall 2025 semester start date (used as a planning reference) is {_val_or_missing(data.fall_start_2025)}."
    await evaluator.verify(
        claim=claim_start,
        node=start_ref_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="The date should match what's listed on the official academic calendar for Fall 2025.",
    )

    # Typical summer orientation timing
    orientation_leaf = evaluator.add_leaf(
        id="Typical_Summer_Orientation_Timing",
        desc="States summer orientation for fall enrollees typically occurs in June.",
        parent=section,
        critical=True,
    )
    claim_orient = f"Summer orientation for fall enrollees typically occurs in {_val_or_missing(data.orientation_timing)}."
    await evaluator.verify(
        claim=claim_orient,
        node=orientation_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="If the page indicates orientation sessions primarily in June, consider that supported; minor variations are acceptable.",
    )

    # Major academic breaks
    breaks_leaf = evaluator.add_leaf(
        id="Major_Academic_Breaks_Listed",
        desc="Lists the major academic breaks in the referenced academic calendar (break names with corresponding dates/date ranges).",
        parent=section,
        critical=True,
    )
    breaks_str = _list_or_missing(data.major_breaks)
    claim_breaks = f"The academic calendar lists the following major breaks (names and dates): {breaks_str}"
    await evaluator.verify(
        claim=claim_breaks,
        node=breaks_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Verify each listed break (name + dates/date ranges) can be found on the official calendar page.",
    )


async def verify_transfer_section(evaluator: Evaluator, parent, data: TransferCreditSection) -> None:
    section = evaluator.add_parallel(
        id="Transfer_Credit_Considerations",
        desc="Ohio Transfer 36 explanation/benefits, minimum grades, associate-degree-related benefit, and official Belmont verification link(s).",
        parent=parent,
        critical=True,
    )

    # Official Belmont URL presence (at least one Belmont page for transfer credit/policies)
    transfer_belmont_url_node = evaluator.add_custom_node(
        result=_has_belmont_url(data.urls),
        id="Transfer_Credit_Official_Belmont_URL",
        desc="Provides at least one direct URL to an official Belmont University page related to transfer credit evaluation/policies.",
        parent=section,
        critical=True,
    )
    belmont_sources = _belmont_only(data.urls)

    # Ohio Transfer 36 explanation/benefit
    ot36_leaf = evaluator.add_leaf(
        id="Ohio_Transfer_36_How_It_Works_and_Benefit",
        desc="Explains Ohio Transfer 36 and includes the stated benefit: it guarantees transfer of general education credits among Ohio public colleges/universities.",
        parent=section,
        critical=True,
    )
    claim_ot36 = f"Ohio Transfer 36 explanation and benefit: {_val_or_missing(data.ohio_transfer_36)}"
    await evaluator.verify(
        claim=claim_ot36,
        node=ot36_leaf,
        sources=data.urls,  # May include Ohio state official pages if cited by the answer
        additional_instruction="Verify that the claim accurately reflects the Ohio Transfer 36 framework and its guarantee across Ohio public institutions.",
    )

    # Minimum grade requirement
    min_grade_leaf = evaluator.add_leaf(
        id="Minimum_Grade_For_Transfer_Credit",
        desc="States the minimum grade requirement for transfer credits (e.g., C or higher).",
        parent=section,
        critical=True,
    )
    claim_min = f"The minimum grade requirement for transfer credits is stated as: {_val_or_missing(data.min_grade)}."
    await evaluator.verify(
        claim=claim_min,
        node=min_grade_leaf,
        sources=belmont_sources if belmont_sources else data.urls,
        additional_instruction="Confirm the threshold as stated (e.g., C or higher) on Belmont's transfer credit policy page when available.",
    )

    # Associate degree benefit
    assoc_benefit_leaf = evaluator.add_leaf(
        id="Associate_Degree_Special_Benefit",
        desc="States the special benefit for graduates with associate degrees from Ohio public institutions with completed Ohio Transfer 36.",
        parent=section,
        critical=True,
    )
    claim_assoc = f"Special benefit for students with associate degrees from Ohio public institutions (with completed Ohio Transfer 36): {_val_or_missing(data.assoc_degree_benefit)}"
    await evaluator.verify(
        claim=claim_assoc,
        node=assoc_benefit_leaf,
        sources=data.urls,
        additional_instruction="Verify the benefit exactly as claimed using the provided official sources (Ohio or institutional).",
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
    model: str = "o4-mini",
) -> Dict:
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

    # 1) Extract structured plan
    plan: PlanExtraction = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="transfer_enrollment_plan",
    )

    # 2) Build verification tree according to rubric
    root.desc = "Comprehensive enrollment preparation plan for a transfer student applying to Belmont University for Fall 2026, covering all requested sections and required verifiable links."
    root.critical = True  # Root is critical; all children must be critical

    # Sections (all critical under a critical root)
    await verify_application_section(evaluator, root, plan.application)
    await verify_housing_section(evaluator, root, plan.housing)
    await verify_deposit_section(evaluator, root, plan.deposit)
    await verify_academic_section(evaluator, root, plan.academic)
    await verify_transfer_section(evaluator, root, plan.transfer)

    # 3) Return summary
    return evaluator.get_summary()