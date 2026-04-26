import asyncio
import logging
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# ------------------------------------------------------------------------------------
# Task constants
# ------------------------------------------------------------------------------------
TASK_ID = "mi_realid_passport_coordination_2026"
TASK_DESCRIPTION = (
    "A Michigan resident is planning to coordinate two government services in early 2026: obtaining a REAL ID from the "
    "Michigan Secretary of State and renewing their U.S. passport online. They need to visit a federal courthouse in "
    "Michigan in March 2026 for jury duty.\n\n"
    "Provide a comprehensive coordination plan that includes:\n\n"
    "1. REAL ID Application Requirements: List all four categories of documents required to apply for a Michigan REAL ID, "
    "including: (a) citizenship/identity documentation, (b) Michigan residency proof, (c) current Michigan identification, "
    "and (d) Social Security number verification.\n\n"
    "2. Online Passport Renewal Eligibility: List all five eligibility criteria that must be met to renew a U.S. passport "
    "online, including citizenship status, age requirements, passport issue date range, location requirements, and personal "
    "information stability.\n\n"
    "3. Passport Processing Timeline: Identify the processing duration for both routine and expedited passport renewal "
    "services (in weeks), the additional fee for expedited service (in dollars), and the potential additional time that "
    "mailing may add to the total process.\n\n"
    "4. Federal Holiday Impact: Identify all federal holidays in January and February 2026 (including specific dates and "
    "day of the week) when government offices will be closed, as these affect service availability.\n\n"
    "5. Federal Courthouse Access: Identify the REAL ID requirement for accessing federal facilities, including the specific "
    "date when this requirement took effect for adults 18 and older.\n\n"
    "Your response must cite specific URLs from official government sources (.gov domains) to support each category of requirements."
)


# ------------------------------------------------------------------------------------
# Data models (extraction targets)
# ------------------------------------------------------------------------------------
class RealIDRequirements(BaseModel):
    citizenship_identity: Optional[str] = None
    residency_proof: Optional[str] = None
    residency_requires_two_docs: Optional[bool] = None
    current_michigan_id: Optional[str] = None
    ssn_verification: Optional[str] = None
    citations: List[str] = Field(default_factory=list)


class OnlineRenewalEligibility(BaseModel):
    citizenship_or_national: Optional[str] = None
    age_requirement: Optional[str] = None
    issue_date_range: Optional[str] = None
    location_requirement: Optional[str] = None
    personal_info_stability: Optional[str] = None
    citations: List[str] = Field(default_factory=list)


class ProcessingTimeline(BaseModel):
    routine_weeks: Optional[str] = None
    expedited_weeks: Optional[str] = None
    expedited_fee: Optional[str] = None
    mailing_additional_time: Optional[str] = None
    citations: List[str] = Field(default_factory=list)


class HolidayItem(BaseModel):
    name: Optional[str] = None
    date: Optional[str] = None
    day_of_week: Optional[str] = None


class FederalHolidaysJanFeb2026(BaseModel):
    january: List[HolidayItem] = Field(default_factory=list)
    february: List[HolidayItem] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)


class FederalFacilitiesRealIDEffDate(BaseModel):
    requirement_text: Optional[str] = None
    effective_date: Optional[str] = None
    citations: List[str] = Field(default_factory=list)


class CourthouseIDRequirement(BaseModel):
    requirement_text: Optional[str] = None
    citations: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------------
# Extraction prompts
# ------------------------------------------------------------------------------------
def prompt_extract_real_id_requirements() -> str:
    return """
Extract the Michigan REAL ID application document requirements as explicitly presented in the answer.

Return the following fields:
- citizenship_identity: The exact text the answer uses for the citizenship/identity documentation category.
- residency_proof: The exact text for Michigan residency proof.
- residency_requires_two_docs: A boolean indicating whether the answer explicitly states that TWO (2) documents are required for Michigan residency proof. True only if this is clearly stated in the answer; otherwise False.
- current_michigan_id: The exact text for the current Michigan driver's license or state ID category.
- ssn_verification: The exact text for Social Security number verification (e.g., SSN card or acceptable proof).
- citations: An array of all URLs in the answer that support the Michigan REAL ID requirements (extract all .gov and non-.gov URLs; do not invent any).

Rules:
- Extract exactly what the answer says. Do not infer.
- If a field is not present in the answer, set it to null. For the boolean, set to false if not clearly stated.
- For citations, include every URL mentioned for REAL ID requirements in the answer (e.g., michigan.gov/sos, dhs.gov/real-id).
"""


def prompt_extract_online_renewal_eligibility() -> str:
    return """
Extract the online U.S. passport renewal eligibility criteria as listed in the answer.

Return the following fields (as strings, exactly as written in the answer):
- citizenship_or_national
- age_requirement
- issue_date_range
- location_requirement
- personal_info_stability
- citations: Array of all URLs the answer cites for the online passport renewal eligibility (include all .gov and non-.gov URLs mentioned for this part).

Rules:
- Do not invent.
- If an item is missing in the answer, set it to null.
- Extract URLs exactly as shown in the answer text (full URLs).
"""


def prompt_extract_processing_timeline() -> str:
    return """
Extract the passport renewal processing timeline details as given in the answer.

Return the following fields (strings, exactly as stated in the answer text):
- routine_weeks: Routine processing time in weeks (e.g., "6-8 weeks").
- expedited_weeks: Expedited processing time in weeks (e.g., "2-3 weeks").
- expedited_fee: The additional dollar amount for expedited service (e.g., "$60").
- mailing_additional_time: Any description of additional time due to mailing (e.g., "mailing can add up to 2 weeks each way").
- citations: Array of all URLs the answer cites for these processing times/fees/mailing notes (include all .gov and non-.gov URLs mentioned for this part).

If a field is not present in the answer, set it to null. Do not infer or calculate.
"""


def prompt_extract_holidays_2026() -> str:
    return """
From the answer, extract the federal holidays in January and February 2026, including the specific date and the day of the week as written.

Return:
- january: array of {name, date, day_of_week} objects for January 2026 (as stated in the answer).
- february: array of {name, date, day_of_week} objects for February 2026 (as stated in the answer).
- citations: array of URLs cited for these holidays (e.g., OPM federal holidays page). Include all URLs mentioned for this section.

Rules:
- Do not add, omit, or change names/dates/days. Use exactly what the answer states.
- If a value is missing for an item, set the missing field to null.
"""


def prompt_extract_federal_facilities_realid() -> str:
    return """
From the answer, extract the REAL ID requirement for accessing federal facilities and the effective date as stated.

Return:
- requirement_text: The exact text the answer uses indicating REAL ID is required for accessing most federal facilities for adults 18+.
- effective_date: The specific date when this requirement took effect (as written in the answer).
- citations: Array of all URLs cited to support this federal facility REAL ID requirement (e.g., dhs.gov/real-id or tsa.gov/real-id).

If any field is missing in the answer, set it to null.
"""


def prompt_extract_courthouse_id_requirement() -> str:
    return """
From the answer, extract the federal courthouse visitor photo ID requirement as given.

Return:
- requirement_text: Exact text from the answer stating that federal courthouse visitors must present a valid photo ID issued by a federal or state government agency.
- citations: Array of all URLs the answer cites to support this courthouse visitor ID rule (e.g., uscourts.gov or a district court .gov page).

If requirement_text is missing, set it to null.
"""


# ------------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------------
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _any_gov_url(urls: List[str]) -> bool:
    for u in urls:
        try:
            netloc = urlparse(u).netloc.lower()
        except Exception:
            continue
        if netloc.endswith(".gov"):
            return True
    return False


def _holidays_content_ok(hol: FederalHolidaysJanFeb2026) -> bool:
    # Must include Jan: New Year's Day and MLK Day; Feb: Washington's Birthday/Presidents Day
    jan_names = [((h.name or "").lower()) for h in hol.january]
    feb_names = [((h.name or "").lower()) for h in hol.february]

    has_new_year = any(("new" in n and "year" in n) for n in jan_names)
    has_mlk = any(("martin" in n and "king" in n) for n in jan_names)
    has_presidents = any(("washington" in n) or ("president" in n) for n in feb_names)

    # Each listed holiday item must have date and day_of_week
    all_have_fields = True
    for month in (hol.january + hol.february):
        if not (_nonempty(month.date) and _nonempty(month.day_of_week) and _nonempty(month.name)):
            all_have_fields = False
            break

    return has_new_year and has_mlk and has_presidents and all_have_fields


# ------------------------------------------------------------------------------------
# Verification builders
# ------------------------------------------------------------------------------------
async def build_real_id_section(evaluator: Evaluator, parent, data: RealIDRequirements):
    sec = evaluator.add_parallel(
        id="REAL_ID_Application_Requirements",
        desc="Provide Michigan REAL ID application document requirements.",
        parent=parent,
        critical=True
    )

    # Leaf: REAL_ID_Document_Categories (content completeness in the answer)
    all_present = (
        _nonempty(data.citizenship_identity)
        and _nonempty(data.residency_proof)
        and bool(data.residency_requires_two_docs)
        and _nonempty(data.current_michigan_id)
        and _nonempty(data.ssn_verification)
    )
    evaluator.add_custom_node(
        result=all_present,
        id="REAL_ID_Document_Categories",
        desc="Lists all four REAL ID document categories: (a) citizenship/identity documentation, (b) Michigan residency proof (including that two documents are required), (c) current Michigan driver's license/ID, and (d) Social Security number verification.",
        parent=sec,
        critical=True
    )

    # Leaf: REAL_ID_Citations (.gov and content support)
    # If no citations, fail directly
    if not data.citations:
        evaluator.add_custom_node(
            result=False,
            id="REAL_ID_Citations",
            desc="Provides official .gov URL citation(s) supporting the REAL ID document requirements section.",
            parent=sec,
            critical=True
        )
    else:
        cit_leaf = evaluator.add_leaf(
            id="REAL_ID_Citations",
            desc="Provides official .gov URL citation(s) supporting the REAL ID document requirements section.",
            parent=sec,
            critical=True
        )
        claim = ("These official government (.gov) pages support Michigan REAL ID document requirements, including "
                 "citizenship/identity documentation, two documents proving Michigan residency, a current Michigan "
                 "driver's license/ID, and Social Security number verification.")
        await evaluator.verify(
            claim=claim,
            node=cit_leaf,
            sources=data.citations,
            additional_instruction=(
                "Only accept if the URL domain ends with .gov and the content discusses Michigan REAL ID document "
                "requirements categories (prefer michigan.gov/sos, dhs.gov/real-id). If none of the URLs are .gov or "
                "the content is irrelevant, mark Incorrect."
            )
        )


async def build_online_renewal_section(evaluator: Evaluator, parent, data: OnlineRenewalEligibility):
    sec = evaluator.add_parallel(
        id="Online_Passport_Renewal_Eligibility",
        desc="Provide online U.S. passport renewal eligibility requirements.",
        parent=parent,
        critical=True
    )

    # Leaf: Online_Renewal_Eligibility_Criteria (content completeness in the answer)
    all_present = (
        _nonempty(data.citizenship_or_national)
        and _nonempty(data.age_requirement)
        and _nonempty(data.issue_date_range)
        and _nonempty(data.location_requirement)
        and _nonempty(data.personal_info_stability)
    )
    evaluator.add_custom_node(
        result=all_present,
        id="Online_Renewal_Eligibility_Criteria",
        desc="Lists all five online renewal eligibility criteria: citizenship/national status, age-at-issuance requirement, passport issue date range, location requirement, and no personal-info changes (full name/sex).",
        parent=sec,
        critical=True
    )

    # Leaf: Online_Renewal_Citations
    if not data.citations:
        evaluator.add_custom_node(
            result=False,
            id="Online_Renewal_Citations",
            desc="Provides official .gov URL citation(s) supporting the online passport renewal eligibility section.",
            parent=sec,
            critical=True
        )
    else:
        cit_leaf = evaluator.add_leaf(
            id="Online_Renewal_Citations",
            desc="Provides official .gov URL citation(s) supporting the online passport renewal eligibility section.",
            parent=sec,
            critical=True
        )
        claim = ("These official government (.gov) pages describe the online U.S. passport renewal eligibility criteria "
                 "(citizenship/national status, age-at-issuance, passport issue date range, location requirement, and "
                 "no personal-info changes).")
        await evaluator.verify(
            claim=claim,
            node=cit_leaf,
            sources=data.citations,
            additional_instruction=(
                "Only accept .gov domains (prefer travel.state.gov). If none are .gov or the content is irrelevant to "
                "online passport renewal eligibility, mark Incorrect."
            )
        )


async def build_processing_timeline_section(evaluator: Evaluator, parent, data: ProcessingTimeline):
    sec = evaluator.add_parallel(
        id="Passport_Processing_Timeline",
        desc="Provide passport renewal processing times and related timing/cost details.",
        parent=parent,
        critical=True
    )

    # Leaf: Processing_Durations_Fees_Mailing (content completeness in the answer)
    all_present = (
        _nonempty(data.routine_weeks)
        and _nonempty(data.expedited_weeks)
        and _nonempty(data.expedited_fee)
        and _nonempty(data.mailing_additional_time)
    )
    evaluator.add_custom_node(
        result=all_present,
        id="Processing_Durations_Fees_Mailing",
        desc="Provides routine and expedited processing durations (in weeks), the expedited additional fee (in dollars), and the potential additional time mailing may add to the total process.",
        parent=sec,
        critical=True
    )

    # Leaf: Processing_Timeline_Citations
    if not data.citations:
        evaluator.add_custom_node(
            result=False,
            id="Processing_Timeline_Citations",
            desc="Provides official .gov URL citation(s) supporting the passport processing timeline section.",
            parent=sec,
            critical=True
        )
    else:
        cit_leaf = evaluator.add_leaf(
            id="Processing_Timeline_Citations",
            desc="Provides official .gov URL citation(s) supporting the passport processing timeline section.",
            parent=sec,
            critical=True
        )
        claim = ("These official government (.gov) pages support the passport renewal processing durations (routine and expedited), "
                 "the additional expedited fee, and the potential additional time from mailing.")
        await evaluator.verify(
            claim=claim,
            node=cit_leaf,
            sources=data.citations,
            additional_instruction=(
                "Only accept .gov domains (prefer travel.state.gov pages for processing times and fees). "
                "If none are .gov or they do not discuss processing durations/fees/mailing times, mark Incorrect."
            )
        )


async def build_holidays_section(evaluator: Evaluator, parent, data: FederalHolidaysJanFeb2026):
    sec = evaluator.add_parallel(
        id="Federal_Holidays_Jan_Feb_2026",
        desc="Provide the federal holiday closures that affect government office availability.",
        parent=parent,
        critical=True
    )

    # Leaf: Holidays_List_With_Dates_And_Days (content completeness in the answer)
    holidays_ok = _holidays_content_ok(data)
    evaluator.add_custom_node(
        result=holidays_ok,
        id="Holidays_List_With_Dates_And_Days",
        desc="Identifies all federal holidays in January and February 2026, including specific date and day-of-week for each.",
        parent=sec,
        critical=True
    )

    # Leaf: Holiday_Citations
    if not data.citations:
        evaluator.add_custom_node(
            result=False,
            id="Holiday_Citations",
            desc="Provides official .gov URL citation(s) supporting the Jan/Feb 2026 federal holidays list.",
            parent=sec,
            critical=True
        )
    else:
        cit_leaf = evaluator.add_leaf(
            id="Holiday_Citations",
            desc="Provides official .gov URL citation(s) supporting the Jan/Feb 2026 federal holidays list.",
            parent=sec,
            critical=True
        )
        claim = "These official government (.gov) pages list U.S. federal holidays and support the Jan/Feb 2026 holidays with dates and days of the week."
        await evaluator.verify(
            claim=claim,
            node=cit_leaf,
            sources=data.citations,
            additional_instruction=(
                "Only accept .gov domains (e.g., opm.gov federal holidays page). If none are .gov or the page does not "
                "list federal holidays/dates/days, mark Incorrect."
            )
        )


async def build_federal_facilities_section(evaluator: Evaluator, parent, data: FederalFacilitiesRealIDEffDate):
    sec = evaluator.add_parallel(
        id="Federal_Facilities_REAL_ID_Effective_Date",
        desc="Provide the REAL ID requirement for accessing federal facilities.",
        parent=parent,
        critical=True
    )

    # Leaf: Federal_Facilities_Requirement_And_Effective_Date (content completeness in the answer)
    has_req = _nonempty(data.requirement_text) and ("18" in (data.requirement_text or ""))
    has_date = _nonempty(data.effective_date)
    evaluator.add_custom_node(
        result=(has_req and has_date),
        id="Federal_Facilities_Requirement_And_Effective_Date",
        desc="States the REAL ID requirement for accessing most federal facilities for adults 18+ and includes the effective date when it took effect.",
        parent=sec,
        critical=True
    )

    # Leaf: Federal_Facilities_Citations
    if not data.citations:
        evaluator.add_custom_node(
            result=False,
            id="Federal_Facilities_Citations",
            desc="Provides official .gov URL citation(s) supporting the federal facilities REAL ID requirement and effective date.",
            parent=sec,
            critical=True
        )
    else:
        cit_leaf = evaluator.add_leaf(
            id="Federal_Facilities_Citations",
            desc="Provides official .gov URL citation(s) supporting the federal facilities REAL ID requirement and effective date.",
            parent=sec,
            critical=True
        )
        claim = "These official (.gov) pages confirm that adults 18+ need REAL ID-compliant identification to access most federal facilities and state the effective date."
        await evaluator.verify(
            claim=claim,
            node=cit_leaf,
            sources=data.citations,
            additional_instruction=(
                "Only accept .gov domains (e.g., dhs.gov/real-id, tsa.gov/real-id). If none are .gov or the content does "
                "not mention the federal facilities access requirement and the effective date, mark Incorrect."
            )
        )


async def build_courthouse_id_section(evaluator: Evaluator, parent, data: CourthouseIDRequirement):
    sec = evaluator.add_parallel(
        id="Federal_Courthouse_Visitor_Photo_ID_Requirement",
        desc="Provide the federal courthouse visitor ID requirement included in the constraints.",
        parent=parent,
        critical=True
    )

    # Leaf: Courthouse_Photo_ID_Requirement (content completeness in the answer)
    # Require that answer explicitly mentions a valid photo ID issued by a federal or state government agency
    rt = (data.requirement_text or "").lower()
    mentions_photo = "photo" in rt and "id" in rt
    mentions_gov_issuer = ("federal" in rt or "state" in rt) and ("government" in rt)
    evaluator.add_custom_node(
        result=(_nonempty(data.requirement_text) and mentions_photo and mentions_gov_issuer),
        id="Courthouse_Photo_ID_Requirement",
        desc="States that federal courthouse visitors must present a valid photo ID issued by a federal or state government agency.",
        parent=sec,
        critical=True
    )

    # Leaf: Courthouse_ID_Citations
    if not data.citations:
        evaluator.add_custom_node(
            result=False,
            id="Courthouse_ID_Citations",
            desc="Provides official .gov URL citation(s) supporting the federal courthouse visitor photo ID requirement.",
            parent=sec,
            critical=True
        )
    else:
        cit_leaf = evaluator.add_leaf(
            id="Courthouse_ID_Citations",
            desc="Provides official .gov URL citation(s) supporting the federal courthouse visitor photo ID requirement.",
            parent=sec,
            critical=True
        )
        claim = "These official (.gov) pages state that visitors must present a valid government-issued photo ID to enter a federal courthouse."
        await evaluator.verify(
            claim=claim,
            node=cit_leaf,
            sources=data.citations,
            additional_instruction=(
                "Only accept .gov domains (e.g., uscourts.gov or specific district court .gov sites). If none are .gov "
                "or the content does not mention valid government-issued photo ID for courthouse entry, mark Incorrect."
            )
        )


# ------------------------------------------------------------------------------------
# Main evaluation entry point
# ------------------------------------------------------------------------------------
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
    Build and execute the verification tree for the Michigan REAL ID + Online Passport Renewal + Federal Facilities/Courthouse requirements task.
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

    # Build a critical top-level aggregation node under evaluator.root
    gov_coord = evaluator.add_parallel(
        id="Government_Services_Coordination",
        desc="Provide the requested coordination information for Michigan REAL ID + online passport renewal + federal courthouse/federal facility access, with official .gov citations as required.",
        parent=root,
        critical=True
    )

    # Parallelize extractions
    (
        realid_ext,
        online_ext,
        timeline_ext,
        holidays_ext,
        facilities_ext,
        courthouse_ext
    ) = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_real_id_requirements(),
            template_class=RealIDRequirements,
            extraction_name="real_id_requirements"
        ),
        evaluator.extract(
            prompt=prompt_extract_online_renewal_eligibility(),
            template_class=OnlineRenewalEligibility,
            extraction_name="online_passport_renewal_eligibility"
        ),
        evaluator.extract(
            prompt=prompt_extract_processing_timeline(),
            template_class=ProcessingTimeline,
            extraction_name="passport_processing_timeline"
        ),
        evaluator.extract(
            prompt=prompt_extract_holidays_2026(),
            template_class=FederalHolidaysJanFeb2026,
            extraction_name="federal_holidays_jan_feb_2026"
        ),
        evaluator.extract(
            prompt=prompt_extract_federal_facilities_realid(),
            template_class=FederalFacilitiesRealIDEffDate,
            extraction_name="federal_facilities_real_id"
        ),
        evaluator.extract(
            prompt=prompt_extract_courthouse_id_requirement(),
            template_class=CourthouseIDRequirement,
            extraction_name="federal_courthouse_visitor_id_requirement"
        ),
    )

    # Build verification subtrees
    await asyncio.gather(
        build_real_id_section(evaluator, gov_coord, realid_ext),
        build_online_renewal_section(evaluator, gov_coord, online_ext),
        build_processing_timeline_section(evaluator, gov_coord, timeline_ext),
        build_holidays_section(evaluator, gov_coord, holidays_ext),
        build_federal_facilities_section(evaluator, gov_coord, facilities_ext),
        build_courthouse_id_section(evaluator, gov_coord, courthouse_ext),
    )

    # Return evaluation summary
    return evaluator.get_summary()