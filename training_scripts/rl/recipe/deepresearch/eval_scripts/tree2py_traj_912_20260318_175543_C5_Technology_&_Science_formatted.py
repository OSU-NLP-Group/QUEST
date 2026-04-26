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
TASK_ID = "verizon_outage_2026_fcc_compilation"
TASK_DESCRIPTION = (
    "In January 2026, Verizon experienced a major nationwide network outage that prompted a Federal Communications "
    "Commission (FCC) investigation. I'm an affected customer considering submitting feedback to the FCC and need to "
    "compile comprehensive information about this incident.\n\n"
    "Please provide the following details with supporting reference URLs:\n\n"
    "1. Outage Specifics: On what date did the outage occur? How long did it last? What did Verizon identify as the "
    "technical cause? What geographic scope did it cover (nationwide or specific regions)?\n\n"
    "2. FCC Submission Information: What is the deadline for submitting public comments to the FCC investigation? "
    "What is the designated email address for submissions? Is there an alternative submission method?\n\n"
    "3. Customer Impact: What specific behavior did customer phones exhibit during the outage? What compensation amount "
    "did Verizon offer to affected customers? Provide examples of at least two states that were affected.\n\n"
    "4. FCC Investigation Focus: What are the key areas the FCC is investigating? Include information about: the impact "
    "on emergency 911 calling; public safety concerns; effects on critical services like hospitals; assessment of "
    "Verizon's communication during the incident.\n\n"
    "For each major category of information, please include at least one reference URL from a reliable source that supports your findings."
)

# Expected constraints per rubric
EXPECTED_OUTAGE_DATE = "January 14, 2026"
EXPECTED_OUTAGE_DURATION_APPROX_HOURS = 10
EXPECTED_TECH_CAUSE_KEYWORD = "software"
EXPECTED_SCOPE_NATIONWIDE = True

FCC_SUBMISSION_DEADLINE = "March 16, 2026"
FCC_SUBMISSION_EMAIL = "VerizonOutage2026@fcc.gov"
FCC_ALTERNATIVE_METHOD_KEYWORD = "ECFS"

AFFECTED_STATES_WHITELIST = {"Texas", "Georgia", "New York", "California"}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BasicOutageDetailsExtraction(BaseModel):
    outage_date: Optional[str] = None
    outage_duration: Optional[str] = None
    technical_cause: Optional[str] = None
    geographic_scope: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)


class FCCSubmissionExtraction(BaseModel):
    submission_deadline: Optional[str] = None
    submission_email: Optional[str] = None
    alternative_submission_method: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)


class CustomerImpactExtraction(BaseModel):
    phone_behavior: Optional[str] = None
    compensation_amount: Optional[str] = None
    affected_states: List[str] = Field(default_factory=list)
    supporting_urls: List[str] = Field(default_factory=list)


class InvestigationFocusExtraction(BaseModel):
    investigating_bureau: Optional[str] = None
    focus_911_and_harm: Optional[str] = None
    focus_public_safety: Optional[str] = None
    focus_businesses_and_critical_services: Optional[str] = None
    focus_number_affected_and_duration: Optional[str] = None
    focus_communication: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_basic_outage_details() -> str:
    return """
    Extract the basic outage details as explicitly stated in the answer (do not infer):
    - outage_date: the specific calendar date stated for the outage (e.g., "January 14, 2026", "Jan 14, 2026", "1/14/2026").
    - outage_duration: the duration as stated (free text, e.g., "~10 hours", "about 10 hours", "10 hours").
    - technical_cause: the cause as stated (e.g., "software issue", "software update problem").
    - geographic_scope: the scope as stated (e.g., "nationwide", "across the U.S.", or specific regions if that's what the answer says).
    - supporting_urls: list all URLs the answer cites for these basic details (date/duration/cause/scope). Return an empty list if none.
    """


def prompt_extract_fcc_submission_info() -> str:
    return """
    Extract the FCC public comment submission information as explicitly stated in the answer:
    - submission_deadline: the exact deadline date for public comments (as written, e.g., "March 16, 2026", "3/16/2026").
    - submission_email: the stated email address (as written).
    - alternative_submission_method: the alternative method named (e.g., "ECFS", "Electronic Comment Filing System").
    - supporting_urls: list all URLs cited for the FCC submission information (deadline/email/ECFS). Return an empty list if none.
    """


def prompt_extract_customer_impact() -> str:
    return """
    Extract the customer impact details as explicitly stated in the answer:
    - phone_behavior: what affected phones displayed or did (e.g., "SOS-only mode", "SOS", etc.).
    - compensation_amount: the compensation amount Verizon offered (as written, e.g., "$20 credit", "20-dollar credit").
    - affected_states: list of example states mentioned as affected (return each state name as a separate string).
    - supporting_urls: list all URLs cited for the customer impact information. Return an empty list if none.
    """


def prompt_extract_investigation_focus() -> str:
    return """
    Extract the FCC investigation focus as explicitly stated in the answer:
    - investigating_bureau: which FCC bureau launched or is running the investigation (as written).
    - focus_911_and_harm: the statement about 911 impacts and whether harm or injury resulted from inability to reach 911 (as written).
    - focus_public_safety: the statement mentioning public safety concerns (as written).
    - focus_businesses_and_critical_services: the statement mentioning impacts on businesses and critical services such as hospitals (as written).
    - focus_number_affected_and_duration: the statement mentioning number of customers affected and duration of service loss (as written).
    - focus_communication: the statement assessing appropriateness and timeliness of Verizon's communications during the incident (as written).
    - supporting_urls: list all URLs cited for the FCC investigation focus points. Return an empty list if none.
    """


# --------------------------------------------------------------------------- #
# Verification helper functions                                               #
# --------------------------------------------------------------------------- #
async def verify_basic_outage_details(
    evaluator: Evaluator,
    parent_node,
    data: BasicOutageDetailsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="BasicOutageDetails",
        desc="Constrained facts about the Verizon network outage (with supporting URL).",
        parent=parent_node,
        critical=True,
    )

    # 1) Outage Date = January 14, 2026
    leaf_date = evaluator.add_leaf(
        id="OutageDate_Jan14_2026",
        desc="States that the outage occurred on January 14, 2026.",
        parent=node,
        critical=True,
    )
    stated_date = data.outage_date or "None"
    claim_date = (
        f"The answer's outage date field is '{stated_date}'. "
        f"This explicitly indicates that the outage occurred on {EXPECTED_OUTAGE_DATE}. "
        "Return Correct only if the stated date clearly expresses January 14, 2026 (allow common formats like "
        "'Jan 14, 2026', '1/14/2026', '14 January 2026', or '2026-01-14')."
    )
    await evaluator.verify(claim=claim_date, node=leaf_date)

    # 2) Outage Duration ≈ 10 hours
    leaf_duration = evaluator.add_leaf(
        id="OutageDuration_Approx10Hours",
        desc="States that the outage lasted approximately 10 hours.",
        parent=node,
        critical=True,
    )
    stated_duration = data.outage_duration or "None"
    claim_duration = (
        f"The answer's stated duration is '{stated_duration}'. "
        f"This indicates approximately {EXPECTED_OUTAGE_DURATION_APPROX_HOURS} hours. "
        "Return Correct if the stated duration clearly indicates around 10 hours (accept ~8–12 hours or phrases like "
        "'about 10 hours', 'roughly 10 hours')."
    )
    await evaluator.verify(claim=claim_duration, node=leaf_duration)

    # 3) Technical Cause = software issue
    leaf_cause = evaluator.add_leaf(
        id="TechnicalCause_SoftwareIssue",
        desc="States that Verizon attributed the outage to a software issue.",
        parent=node,
        critical=True,
    )
    stated_cause = data.technical_cause or "None"
    claim_cause = (
        f"The answer's stated cause is '{stated_cause}'. "
        "Return Correct only if it clearly attributes the outage to a software-related issue "
        "(e.g., software bug, software update/configuration problem)."
    )
    await evaluator.verify(claim=claim_cause, node=leaf_cause)

    # 4) Geographic Scope = nationwide
    leaf_scope = evaluator.add_leaf(
        id="GeographicScope_Nationwide",
        desc="States that the outage was nationwide (not limited to a single local region).",
        parent=node,
        critical=True,
    )
    stated_scope = data.geographic_scope or "None"
    claim_scope = (
        f"The answer's geographic scope is stated as '{stated_scope}'. "
        "Return Correct only if this clearly means the outage was nationwide across the United States "
        "(e.g., 'nationwide', 'across the U.S.', 'national', or similar). Mentions limited to specific single local "
        "regions without indicating nationwide coverage should be marked Incorrect."
    )
    await evaluator.verify(claim=claim_scope, node=leaf_scope)

    # 5) Supporting URL(s) for basic details
    leaf_support = evaluator.add_leaf(
        id="SupportingURLBasicDetails",
        desc="Provides at least one reliable reference URL supporting the basic outage details (date, duration, cause, and nationwide scope).",
        parent=node,
        critical=True,
    )
    urls = data.supporting_urls or []
    claim_support = (
        "This page is a reliable source about the January 2026 Verizon outage and supports at least one of the "
        "following basic facts: the outage date (January 14, 2026), duration (~10 hours), that the cause was a software "
        "issue, or that the scope was nationwide. Return Supported if the page clearly contains at least one of these facts."
    )
    await evaluator.verify(
        claim=claim_support,
        node=leaf_support,
        sources=urls,
        additional_instruction="News from reputable outlets, Verizon official statements, or FCC materials are considered reliable. Any one of the listed facts is sufficient for support.",
    )


async def verify_fcc_submission_requirements(
    evaluator: Evaluator,
    parent_node,
    data: FCCSubmissionExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="FCCSubmissionRequirements",
        desc="Constrained information needed to submit comments to the FCC investigation (with supporting URL).",
        parent=parent_node,
        critical=True,
    )

    # 1) Deadline = March 16, 2026
    leaf_deadline = evaluator.add_leaf(
        id="SubmissionDeadline_March16_2026",
        desc="States that the FCC public comment deadline is March 16, 2026.",
        parent=node,
        critical=True,
    )
    stated_deadline = data.submission_deadline or "None"
    claim_deadline = (
        f"The answer's submission deadline is '{stated_deadline}'. "
        "Return Correct only if it clearly expresses March 16, 2026 (accept formats like 'March 16, 2026', '3/16/2026', '2026-03-16')."
    )
    await evaluator.verify(claim=claim_deadline, node=leaf_deadline)

    # 2) Submission Email = VerizonOutage2026@fcc.gov
    leaf_email = evaluator.add_leaf(
        id="SubmissionEmail_VerizonOutage2026_at_fcc_gov",
        desc="States that submissions can be sent to VerizonOutage2026@fcc.gov.",
        parent=node,
        critical=True,
    )
    stated_email = data.submission_email or "None"
    claim_email = (
        f"The answer's submission email is '{stated_email}'. "
        f"Return Correct only if this equals '{FCC_SUBMISSION_EMAIL}' (case-insensitive, allow trivial punctuation/formatting variants)."
    )
    await evaluator.verify(claim=claim_email, node=leaf_email)

    # 3) Alternative submission method = ECFS
    leaf_alt = evaluator.add_leaf(
        id="AlternativeSubmissionMethod_ECFS",
        desc="States that an alternative submission method is the FCC’s ECFS system.",
        parent=node,
        critical=True,
    )
    stated_alt = data.alternative_submission_method or "None"
    claim_alt = (
        f"The answer's alternative submission method is stated as '{stated_alt}'. "
        "Return Correct only if it clearly mentions the FCC's ECFS (Electronic Comment Filing System)."
    )
    await evaluator.verify(claim=claim_alt, node=leaf_alt)

    # 4) Supporting URL(s) for submission details
    leaf_support = evaluator.add_leaf(
        id="SupportingURLSubmission",
        desc="Provides at least one reliable reference URL supporting the FCC submission information (deadline, email, and ECFS option).",
        parent=node,
        critical=True,
    )
    urls = data.supporting_urls or []
    claim_support = (
        "This page provides official FCC or equivalent authoritative information about the investigation submission "
        "process for the January 2026 Verizon outage, and it includes at least one of: the March 16, 2026 deadline, "
        "the email address VerizonOutage2026@fcc.gov, or the ability to submit via ECFS."
    )
    await evaluator.verify(
        claim=claim_support,
        node=leaf_support,
        sources=urls,
        additional_instruction="At least one of the three submission details must be present on the page. FCC pages preferred.",
    )


async def verify_customer_impact(
    evaluator: Evaluator,
    parent_node,
    data: CustomerImpactExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="CustomerImpactInformation",
        desc="Constrained details about customer impact (with supporting URL).",
        parent=parent_node,
        critical=True,
    )

    # 1) Phone behavior = SOS-only mode
    leaf_phone = evaluator.add_leaf(
        id="PhoneBehavior_SOSOnlyMode",
        desc="States that affected customers’ phones displayed SOS-only mode during the outage.",
        parent=node,
        critical=True,
    )
    stated_behavior = data.phone_behavior or "None"
    claim_phone = (
        f"The answer's phone behavior is '{stated_behavior}'. "
        "Return Correct only if it clearly indicates 'SOS-only' or similar SOS display during the outage."
    )
    await evaluator.verify(claim=claim_phone, node=leaf_phone)

    # 2) Compensation = $20 credit
    leaf_comp = evaluator.add_leaf(
        id="CompensationAmount_20DollarCredit",
        desc="States that Verizon offered a $20 credit to affected customers.",
        parent=node,
        critical=True,
    )
    stated_comp = data.compensation_amount or "None"
    claim_comp = (
        f"The answer's compensation is '{stated_comp}'. "
        "Return Correct only if it clearly states a $20 credit (allow variants like '$20 bill credit', '20-dollar credit')."
    )
    await evaluator.verify(claim=claim_comp, node=leaf_comp)

    # 3) Affected states include at least two from the whitelist
    #    Use a custom node: check the answer's provided examples
    states_list = [s.strip() for s in (data.affected_states or []) if isinstance(s, str)]
    match_count = sum(1 for s in states_list if s.casefold() in {x.casefold() for x in AFFECTED_STATES_WHITELIST})
    evaluator.add_custom_node(
        result=match_count >= 2,
        id="AffectedStates_AtLeastTwoFromListedExamples",
        desc="Provides at least two affected-state examples, and they must be among: Texas, Georgia, New York, California.",
        parent=node,
        critical=True,
    )

    # 4) Supporting URL(s) for customer impact
    leaf_support = evaluator.add_leaf(
        id="SupportingURLCustomerImpact",
        desc="Provides at least one reliable reference URL supporting the customer impact information (SOS-only behavior, $20 credit, and affected states).",
        parent=node,
        critical=True,
    )
    urls = data.supporting_urls or []
    claim_support = (
        "This page reports customer impact from the January 2026 Verizon outage and clearly supports at least one of: "
        "phones showing 'SOS-only', Verizon offering a $20 credit, or examples of affected U.S. states."
    )
    await evaluator.verify(
        claim=claim_support,
        node=leaf_support,
        sources=urls,
        additional_instruction="Any one of the three impact items present on the page is sufficient for support.",
    )


async def verify_investigation_focus(
    evaluator: Evaluator,
    parent_node,
    data: InvestigationFocusExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="FCCInvestigationFocus",
        desc="Constrained key areas the FCC is investigating (with supporting URL).",
        parent=parent_node,
        critical=True,
    )

    # 1) Bureau = Safety and Homeland Security Bureau
    leaf_bureau = evaluator.add_leaf(
        id="InvestigatingBureau_SafetyAndHomelandSecurityBureau",
        desc="Identifies the FCC Safety and Homeland Security Bureau as the bureau that launched the investigation.",
        parent=node,
        critical=True,
    )
    stated_bureau = data.investigating_bureau or "None"
    claim_bureau = (
        f"The answer identifies the investigating bureau as '{stated_bureau}'. "
        "Return Correct only if it clearly names the FCC Safety and Homeland Security Bureau."
    )
    await evaluator.verify(claim=claim_bureau, node=leaf_bureau)

    # 2) Focus: 911 calling and harm/injury
    leaf_911 = evaluator.add_leaf(
        id="Focus_911CallingAndHarmOrInjury",
        desc="Includes that the FCC is investigating impact on 911 calling, including whether harm or injury resulted from inability to reach 911.",
        parent=node,
        critical=True,
    )
    stated_911 = data.focus_911_and_harm or "None"
    claim_911 = (
        f"The answer states: '{stated_911}'. "
        "Return Correct only if it clearly mentions investigation into 911 calling and whether harm or injury resulted from inability to reach 911."
    )
    await evaluator.verify(claim=claim_911, node=leaf_911)

    # 3) Focus: Public safety
    leaf_ps = evaluator.add_leaf(
        id="Focus_PublicSafety",
        desc="Includes that the FCC is investigating public safety concerns.",
        parent=node,
        critical=True,
    )
    stated_ps = data.focus_public_safety or "None"
    claim_ps = (
        f"The answer states: '{stated_ps}'. "
        "Return Correct only if it clearly mentions public safety concerns as an investigation focus."
    )
    await evaluator.verify(claim=claim_ps, node=leaf_ps)

    # 4) Focus: Businesses and critical services (hospitals)
    leaf_hosp = evaluator.add_leaf(
        id="Focus_BusinessesAndCriticalServices_Hospitals",
        desc="Includes that the FCC is investigating impacts on businesses and critical services such as hospitals.",
        parent=node,
        critical=True,
    )
    stated_hosp = data.focus_businesses_and_critical_services or "None"
    claim_hosp = (
        f"The answer states: '{stated_hosp}'. "
        "Return Correct only if it clearly mentions impacts on businesses and critical services such as hospitals."
    )
    await evaluator.verify(claim=claim_hosp, node=leaf_hosp)

    # 5) Focus: Number affected and duration of service loss
    leaf_numdur = evaluator.add_leaf(
        id="Focus_NumberAffectedAndDurationOfServiceLoss",
        desc="Includes that the FCC is investigating the number of customers affected and the duration of service loss.",
        parent=node,
        critical=True,
    )
    stated_numdur = data.focus_number_affected_and_duration or "None"
    claim_numdur = (
        f"The answer states: '{stated_numdur}'. "
        "Return Correct only if it clearly mentions investigating the number of affected customers and the duration of service loss."
    )
    await evaluator.verify(claim=claim_numdur, node=leaf_numdur)

    # 6) Focus: Communication appropriateness and timeliness
    leaf_comm = evaluator.add_leaf(
        id="Focus_Communication_AppropriatenessAndTimeliness",
        desc="Includes that the FCC is assessing the appropriateness and timeliness of Verizon’s communication during the incident.",
        parent=node,
        critical=True,
    )
    stated_comm = data.focus_communication or "None"
    claim_comm = (
        f"The answer states: '{stated_comm}'. "
        "Return Correct only if it clearly mentions assessing the appropriateness and timeliness of Verizon's communication during the incident."
    )
    await evaluator.verify(claim=claim_comm, node=leaf_comm)

    # 7) Supporting URL(s) for investigation focus
    leaf_support = evaluator.add_leaf(
        id="SupportingURLInvestigation",
        desc="Provides at least one reliable reference URL supporting the FCC investigation focus areas listed above.",
        parent=node,
        critical=True,
    )
    urls = data.supporting_urls or []
    claim_support = (
        "This page outlines the FCC's investigation of the January 2026 Verizon outage and clearly supports at least one "
        "of these focus points: 911 calling and potential harm/injury; public safety; impacts on businesses/critical "
        "services (e.g., hospitals); number affected and duration; or the appropriateness/timeliness of Verizon's communications."
    )
    await evaluator.verify(
        claim=claim_support,
        node=leaf_support,
        sources=urls,
        additional_instruction="FCC Public Notices or official statements are preferred; at least one focus area present is sufficient.",
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator
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

    # Create top-level critical node per rubric
    main_node = evaluator.add_parallel(
        id="VerizonOutageInformationCompilation",
        desc="Complete information package about the January 2026 Verizon outage and FCC investigation requirements, matching all stated constraints and including supporting URLs per category.",
        parent=root,
        critical=True,
    )

    # Extract all parts (in parallel)
    basic_task = evaluator.extract(
        prompt=prompt_extract_basic_outage_details(),
        template_class=BasicOutageDetailsExtraction,
        extraction_name="basic_outage_details",
    )
    fcc_task = evaluator.extract(
        prompt=prompt_extract_fcc_submission_info(),
        template_class=FCCSubmissionExtraction,
        extraction_name="fcc_submission_info",
    )
    impact_task = evaluator.extract(
        prompt=prompt_extract_customer_impact(),
        template_class=CustomerImpactExtraction,
        extraction_name="customer_impact",
    )
    focus_task = evaluator.extract(
        prompt=prompt_extract_investigation_focus(),
        template_class=InvestigationFocusExtraction,
        extraction_name="investigation_focus",
    )

    basic, fcc_info, impact, focus = await asyncio.gather(basic_task, fcc_task, impact_task, focus_task)

    # Add ground truth/expectations for transparency
    evaluator.add_ground_truth({
        "expected_outage_date": EXPECTED_OUTAGE_DATE,
        "expected_duration_hours_approx": EXPECTED_OUTAGE_DURATION_APPROX_HOURS,
        "expected_cause_contains": EXPECTED_TECH_CAUSE_KEYWORD,
        "expected_scope_nationwide": EXPECTED_SCOPE_NATIONWIDE,
        "fcc_deadline": FCC_SUBMISSION_DEADLINE,
        "fcc_email": FCC_SUBMISSION_EMAIL,
        "fcc_alt_method_keyword": FCC_ALTERNATIVE_METHOD_KEYWORD,
        "affected_states_must_include_at_least_two_of": sorted(list(AFFECTED_STATES_WHITELIST)),
    }, gt_type="expected_constraints")

    # Build verification subtrees
    await verify_basic_outage_details(evaluator, main_node, basic)
    await verify_fcc_submission_requirements(evaluator, main_node, fcc_info)
    await verify_customer_impact(evaluator, main_node, impact)
    await verify_investigation_focus(evaluator, main_node, focus)

    # Return evaluation summary
    return evaluator.get_summary()