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
TASK_ID = "us_gov_actions_2025_2026"
TASK_DESCRIPTION = """
Between December 2025 and March 2026, the United States government undertook several major actions spanning military operations, personnel changes, benefit programs, and constitutional obligations. Identify and provide comprehensive details about the following four distinct government actions:

1. Military Operation (January 2026): A U.S. military operation that resulted in the capture of a foreign head of state and their spouse. For this operation, provide: (a) the exact date it occurred, (b) the target country, (c) the names of the captured individuals, (d) which U.S. President announced it, (e) a senior military officer who publicly addressed it, (f) the U.S. Secretary of State involved, (g) where the captured individuals were arraigned and on what charges, (h) who assumed power in the target country afterward, (i) the type of U.S. military forces used, and (j) information about prior military buildup. Include reference URLs for all information.

2. Cabinet Nomination (March 2026): A cabinet-level nomination announced in March 2026 to replace an official who was ousted. For this nomination, provide: (a) the nominee's full name, (b) the specific cabinet position, (c) the announcement date, (d) the name of the predecessor being replaced, (e) the nominee's current government position at the time of nomination, (f) which U.S. state they represent, (g) their total congressional service duration, (h) relevant Senate committee assignments, (i) any tribal nation affiliation, and (j) whether Senate confirmation is required. Include reference URLs for all information.

3. Military Benefit Program (December 2025): A one-time monetary benefit program for U.S. military service members announced in December 2025. For this program, provide: (a) the program's official or common name, (b) the exact dollar amount per service member, (c) the announcement date, (d) which U.S. official announced it, (e) the tax status of the payment, (f) how the payment is classified in the military pay system, (g) the total budget allocated, (h) the approximate total number of eligible recipients, (i) pay grade eligibility for active-duty members, (j) eligibility requirements for reserve members, (k) the status determination date for eligibility, (l) the payment deadline, and (m) the disbursement method. Include reference URLs for all information.

4. State of the Union (2026): The constitutionally-mandated State of the Union address delivered in 2026. For this address, provide: (a) the exact date it was delivered, (b) which President delivered it, (c) the specific constitutional provision (Article and Section) mandating it, (d) where it is delivered (to which body/bodies of Congress), (e) the Congressional Record volume and issue number, (f) other official publication series documenting it, and (g) any time/duration records it broke. Include reference URLs for all information.

For each of the four actions, ensure all required details are grounded in verifiable sources and include appropriate reference URLs to support your findings.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OperationExtraction(BaseModel):
    # values
    date: Optional[str] = None
    target_country: Optional[str] = None
    captured_head_of_state: Optional[str] = None
    captured_spouse: Optional[str] = None
    announced_by_president: Optional[str] = None
    senior_military_officer: Optional[str] = None
    secretary_of_state: Optional[str] = None
    arraignment_location: Optional[str] = None
    arraignment_charges: Optional[str] = None
    successor_in_country: Optional[str] = None
    forces_used: Optional[str] = None
    prior_buildup_summary: Optional[str] = None
    # per-field URLs explicitly cited in the answer
    urls_date: List[str] = Field(default_factory=list)
    urls_target_country: List[str] = Field(default_factory=list)
    urls_captured_individuals: List[str] = Field(default_factory=list)
    urls_announced_by: List[str] = Field(default_factory=list)
    urls_senior_officer: List[str] = Field(default_factory=list)
    urls_secretary_of_state: List[str] = Field(default_factory=list)
    urls_arraignment: List[str] = Field(default_factory=list)
    urls_successor: List[str] = Field(default_factory=list)
    urls_forces_used: List[str] = Field(default_factory=list)
    urls_prior_buildup: List[str] = Field(default_factory=list)


class NominationExtraction(BaseModel):
    nomination_date: Optional[str] = None
    nominee_name: Optional[str] = None
    cabinet_position: Optional[str] = None
    predecessor_name: Optional[str] = None
    predecessor_status: Optional[str] = None  # e.g., "ousted", "removed", etc.
    nominee_current_position: Optional[str] = None  # e.g., "U.S. Senator"
    nominee_state: Optional[str] = None
    congressional_service_timeline: Optional[str] = None  # e.g., "in Congress since 2013; 10 yrs House; Senator in 2023"
    committee_assignments: List[str] = Field(default_factory=list)
    tribal_affiliation: Optional[str] = None
    senate_confirmation_required: Optional[str] = None  # e.g., "Yes"/"Requires Senate confirmation"
    # per-field URLs explicitly cited
    urls_nomination_date: List[str] = Field(default_factory=list)
    urls_nominee_name: List[str] = Field(default_factory=list)
    urls_cabinet_position: List[str] = Field(default_factory=list)
    urls_predecessor_name: List[str] = Field(default_factory=list)
    urls_predecessor_status: List[str] = Field(default_factory=list)
    urls_nominee_current_position: List[str] = Field(default_factory=list)
    urls_nominee_state: List[str] = Field(default_factory=list)
    urls_congressional_service: List[str] = Field(default_factory=list)
    urls_committees: List[str] = Field(default_factory=list)
    urls_tribal_affiliation: List[str] = Field(default_factory=list)
    urls_confirmation_required: List[str] = Field(default_factory=list)


class BenefitProgramExtraction(BaseModel):
    program_name: Optional[str] = None  # e.g., "Warrior Dividend"
    amount_per_member: Optional[str] = None  # e.g., "$1,776"
    announcement_date: Optional[str] = None
    additional_details_date: Optional[str] = None
    announced_by: Optional[str] = None
    tax_status: Optional[str] = None  # e.g., "non-taxable", "tax-free"
    classification: Optional[str] = None  # e.g., "one-time BAH supplement"
    total_budget: Optional[str] = None  # e.g., "$2.6 billion"
    eligible_total: Optional[str] = None  # e.g., "approximately 1.45 million"
    eligible_active_duty: Optional[str] = None  # e.g., "1.28 million"
    eligible_reserve: Optional[str] = None  # e.g., "174,000"
    eligibility_status_date: Optional[str] = None  # e.g., "November 30, 2025"
    active_duty_paygrade_eligibility: Optional[str] = None  # e.g., "O-6 and below"
    reserve_eligibility_requirement: Optional[str] = None  # e.g., "orders of 31+ days"
    payment_deadline: Optional[str] = None  # e.g., "before December 20, 2025"
    disbursement_method: Optional[str] = None  # e.g., "standard military pay disbursing system"
    # per-field URLs explicitly cited
    urls_program_name: List[str] = Field(default_factory=list)
    urls_amount_per_member: List[str] = Field(default_factory=list)
    urls_announcement_date: List[str] = Field(default_factory=list)
    urls_additional_details_date: List[str] = Field(default_factory=list)
    urls_announced_by: List[str] = Field(default_factory=list)
    urls_tax_status: List[str] = Field(default_factory=list)
    urls_classification: List[str] = Field(default_factory=list)
    urls_total_budget: List[str] = Field(default_factory=list)
    urls_eligible_total_and_breakdown: List[str] = Field(default_factory=list)
    urls_eligibility_status_date: List[str] = Field(default_factory=list)
    urls_active_duty_paygrade_eligibility: List[str] = Field(default_factory=list)
    urls_reserve_eligibility_requirement: List[str] = Field(default_factory=list)
    urls_payment_deadline: List[str] = Field(default_factory=list)
    urls_disbursement_method: List[str] = Field(default_factory=list)


class SOTUExtraction(BaseModel):
    sotu_date: Optional[str] = None
    delivered_by: Optional[str] = None
    constitutional_provision: Optional[str] = None
    delivered_before: Optional[str] = None  # e.g., "a joint session of the House and Senate"
    congressional_record_citation: Optional[str] = None  # e.g., "Vol. 172, Issue 36"
    other_official_series: Optional[str] = None  # list or combined series names as a string
    duration_record: Optional[str] = None
    # per-field URLs explicitly cited
    urls_sotu_date: List[str] = Field(default_factory=list)
    urls_delivered_by: List[str] = Field(default_factory=list)
    urls_constitutional_provision: List[str] = Field(default_factory=list)
    urls_delivered_before: List[str] = Field(default_factory=list)
    urls_congressional_record_citation: List[str] = Field(default_factory=list)
    urls_other_official_series: List[str] = Field(default_factory=list)
    urls_duration_record: List[str] = Field(default_factory=list)


class AllExtraction(BaseModel):
    operation: Optional[OperationExtraction] = None
    nomination: Optional[NominationExtraction] = None
    benefit: Optional[BenefitProgramExtraction] = None
    sotu: Optional[SOTUExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_operation() -> str:
    return """
    Extract details for the January 2026 U.S. military operation as presented in the answer. For each field, extract the exact value stated and, separately, collect only the reference URLs explicitly cited in the answer that support that specific field. If a URL is provided via markdown link, extract the actual URL target.

    Fields to extract (values):
    - date
    - target_country
    - captured_head_of_state
    - captured_spouse
    - announced_by_president
    - senior_military_officer
    - secretary_of_state
    - arraignment_location
    - arraignment_charges
    - successor_in_country
    - forces_used
    - prior_buildup_summary

    For each field also extract corresponding url list (if any in the answer):
    - urls_date
    - urls_target_country
    - urls_captured_individuals
    - urls_announced_by
    - urls_senior_officer
    - urls_secretary_of_state
    - urls_arraignment
    - urls_successor
    - urls_forces_used
    - urls_prior_buildup

    Rules:
    - Only extract information and URLs explicitly present in the answer. Do not infer or add new URLs.
    - If a field value is missing, set it to null.
    - If no supporting URLs are given for a field, return an empty list for that field's urls_... entry.
    """


def prompt_extract_nomination() -> str:
    return """
    Extract details for the March 2026 cabinet nomination as presented in the answer. For each field, extract the exact value and the supporting URLs explicitly cited.

    Fields (values):
    - nomination_date
    - nominee_name
    - cabinet_position
    - predecessor_name
    - predecessor_status
    - nominee_current_position
    - nominee_state
    - congressional_service_timeline
    - committee_assignments (as an array of committee names)
    - tribal_affiliation
    - senate_confirmation_required

    Per-field URL lists:
    - urls_nomination_date
    - urls_nominee_name
    - urls_cabinet_position
    - urls_predecessor_name
    - urls_predecessor_status
    - urls_nominee_current_position
    - urls_nominee_state
    - urls_congressional_service
    - urls_committees
    - urls_tribal_affiliation
    - urls_confirmation_required

    Rules:
    - Only extract what is explicitly present in the answer; use the actual URLs printed in the answer.
    - Missing values -> null; missing URLs -> empty list.
    """


def prompt_extract_benefit_program() -> str:
    return """
    Extract details for the December 2025 one-time monetary benefit program as presented in the answer, along with the supporting URLs explicitly cited for each field.

    Fields (values):
    - program_name
    - amount_per_member
    - announcement_date
    - additional_details_date
    - announced_by
    - tax_status
    - classification
    - total_budget
    - eligible_total
    - eligible_active_duty
    - eligible_reserve
    - eligibility_status_date
    - active_duty_paygrade_eligibility
    - reserve_eligibility_requirement
    - payment_deadline
    - disbursement_method

    Per-field URL lists:
    - urls_program_name
    - urls_amount_per_member
    - urls_announcement_date
    - urls_additional_details_date
    - urls_announced_by
    - urls_tax_status
    - urls_classification
    - urls_total_budget
    - urls_eligible_total_and_breakdown
    - urls_eligibility_status_date
    - urls_active_duty_paygrade_eligibility
    - urls_reserve_eligibility_requirement
    - urls_payment_deadline
    - urls_disbursement_method

    Rules:
    - Only include URLs explicitly cited in the answer for each field.
    - Use exact strings and preserve formatting in values; missing -> null; URLs missing -> empty list.
    """


def prompt_extract_sotu() -> str:
    return """
    Extract details for the 2026 State of the Union as presented in the answer, and collect supporting URLs explicitly cited for each field.

    Fields (values):
    - sotu_date
    - delivered_by
    - constitutional_provision
    - delivered_before
    - congressional_record_citation
    - other_official_series
    - duration_record

    Per-field URL lists:
    - urls_sotu_date
    - urls_delivered_by
    - urls_constitutional_provision
    - urls_delivered_before
    - urls_congressional_record_citation
    - urls_other_official_series
    - urls_duration_record

    Rules:
    - Extract only what the answer explicitly states and only the URLs it explicitly cites.
    - Missing values -> null; missing URLs -> empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: Optional[List[str] | str]) -> List[str]:
    if urls is None:
        return []
    if isinstance(urls, str):
        urls_list = [urls.strip()]
    else:
        urls_list = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    # de-duplicate while preserving order
    seen = set()
    out = []
    for u in urls_list:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _fail_leaf_due_to_missing_sources(node_desc: str, logger: logging.Logger):
    logger.info(f"Marking node as failed due to missing sources: {node_desc}")


def _mk_leaf_and_maybe_task(
    evaluator: Evaluator,
    parent_node,
    *,
    node_id: str,
    desc: str,
    claim: str,
    urls: Optional[List[str] | str],
    critical: bool = True,
    add_ins: Optional[str] = None,
    tasks: List = None,
):
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )
    srcs = _normalize_urls(urls)
    if not srcs:
        # Enforce source-grounding: if no URLs, fail this leaf immediately
        node.score = 0.0
        node.status = "failed"
        _fail_leaf_due_to_missing_sources(desc, evaluator.verifier.logger if evaluator.verifier else logging.getLogger(__name__))
    else:
        assert tasks is not None
        tasks.append((claim, srcs, node, add_ins or "None"))
    return node


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_military_operation(evaluator: Evaluator, root):
    op_node = evaluator.add_parallel(
        id="military_operation_jan_2026",
        desc="Military operation (January 2026) resulting in capture of a foreign head of state and spouse; all required details with reference URLs.",
        parent=root,
        critical=False
    )

    # Extraction (from the answer)
    op: OperationExtraction = await evaluator.extract(
        prompt=prompt_extract_operation(),
        template_class=OperationExtraction,
        extraction_name="operation_extraction"
    )

    tasks: List = []

    # Leaves (critical) with URL-backed verification
    _mk_leaf_and_maybe_task(
        evaluator, op_node,
        node_id="op_date_is_jan_3_2026",
        desc="States the operation occurred on January 3, 2026, with a supporting reference URL.",
        claim="The U.S. military operation occurred on January 3, 2026.",
        urls=op.urls_date,
        add_ins="Allow date formatting variants (e.g., Jan. 3, 2026). Reject if the page does not clearly indicate the operation date.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, op_node,
        node_id="op_target_country_venezuela",
        desc="Identifies Venezuela as the target country, with a supporting reference URL.",
        claim="The operation targeted Venezuela.",
        urls=op.urls_target_country,
        add_ins="The page should explicitly indicate the operation took place in or targeted Venezuela.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, op_node,
        node_id="op_captured_individuals_maduro_and_cilia_flores",
        desc="Names the captured individuals as Nicolás Maduro and Cilia Flores, with a supporting reference URL.",
        claim="The captured individuals were Nicolás Maduro and his spouse Cilia Flores.",
        urls=op.urls_captured_individuals,
        add_ins="Allow minor spelling/diacritic variations for names. The page should clearly indicate both individuals were captured.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, op_node,
        node_id="op_announced_by_trump",
        desc="Identifies President Donald J. Trump as the announcer, with a supporting reference URL.",
        claim="The operation was announced by U.S. President Donald J. Trump.",
        urls=op.urls_announced_by,
        add_ins="The source should show Trump publicly announced or formally disclosed the operation.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, op_node,
        node_id="op_public_military_officer_dan_caine",
        desc="Identifies Chairman of the Joint Chiefs of Staff General Dan Caine as a senior military officer who publicly addressed it, with a supporting reference URL.",
        claim="General Dan Caine, the Chairman of the Joint Chiefs of Staff, publicly addressed the operation.",
        urls=op.urls_senior_officer,
        add_ins="Look for quotes, press briefings, or public remarks attributable to Gen. Dan Caine in his role as Chairman of the Joint Chiefs.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, op_node,
        node_id="op_secretary_of_state_marco_rubio",
        desc="Identifies Secretary of State Marco Rubio as involved in communications about the operation, with a supporting reference URL.",
        claim="U.S. Secretary of State Marco Rubio was involved in communications about the operation.",
        urls=op.urls_secretary_of_state,
        add_ins="The source should indicate Secretary Rubio's role or public statements tied to the operation.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, op_node,
        node_id="op_arraigned_in_ny_on_drug_trafficking",
        desc="States the captured individuals were arraigned in New York on drug trafficking charges, with a supporting reference URL.",
        claim="The captured individuals were arraigned in New York on drug trafficking charges.",
        urls=op.urls_arraignment,
        add_ins="It should explicitly indicate arraignment in New York and that the charges include drug trafficking.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, op_node,
        node_id="op_succession_delcy_rodriguez",
        desc="States Vice President Delcy Rodríguez assumed power afterward, with a supporting reference URL.",
        claim="Following the capture, Venezuelan Vice President Delcy Rodríguez assumed power.",
        urls=op.urls_successor,
        add_ins="The page should clearly indicate Delcy Rodríguez assumed leadership or the presidency afterward.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, op_node,
        node_id="op_forces_used_special_operations",
        desc="States U.S. special operations forces were used, with a supporting reference URL.",
        claim="U.S. special operations forces were used in the operation.",
        urls=op.urls_forces_used,
        add_ins="The page should explicitly describe the force type as special operations or equivalent.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, op_node,
        node_id="op_prior_buildup_months_caribbean",
        desc="Describes months of U.S. military buildup in the Caribbean prior to the operation, with a supporting reference URL.",
        claim="There had been months of U.S. military buildup in the Caribbean prior to the operation.",
        urls=op.urls_prior_buildup,
        add_ins="Look for reporting of sustained U.S. military presence or buildup in the Caribbean in the preceding months.",
        tasks=tasks
    )

    if tasks:
        await evaluator.batch_verify(tasks)


async def verify_cabinet_nomination(evaluator: Evaluator, root):
    nom_node = evaluator.add_parallel(
        id="cabinet_nomination_mar_2026",
        desc="Cabinet nomination (March 2026) to replace an ousted official; all required details with reference URLs.",
        parent=root,
        critical=False
    )

    nom: NominationExtraction = await evaluator.extract(
        prompt=prompt_extract_nomination(),
        template_class=NominationExtraction,
        extraction_name="nomination_extraction"
    )

    tasks: List = []

    _mk_leaf_and_maybe_task(
        evaluator, nom_node,
        node_id="nomination_date_mar_5_2026",
        desc="States the nomination was announced on March 5, 2026, with a supporting reference URL.",
        claim="The cabinet nomination was announced on March 5, 2026.",
        urls=nom.urls_nomination_date,
        add_ins="Allow date formatting variants. The source should clearly mark March 5, 2026 as the announcement date.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, nom_node,
        node_id="nominee_markwayne_mullin",
        desc="States the nominee is Senator Markwayne Mullin, with a supporting reference URL.",
        claim="The nominee was Senator Markwayne Mullin.",
        urls=nom.urls_nominee_name,
        add_ins="The source should explicitly identify Markwayne Mullin as the nominee.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, nom_node,
        node_id="position_dhs_secretary",
        desc="States the nominated cabinet position is Secretary of Homeland Security, with a supporting reference URL.",
        claim="The nomination was for the position of Secretary of Homeland Security.",
        urls=nom.urls_cabinet_position,
        add_ins="Look for specific reference to the Department of Homeland Security Secretary role.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, nom_node,
        node_id="predecessor_name_kristi_noem",
        desc="States the predecessor being replaced is Kristi Noem, with a supporting reference URL.",
        claim="The predecessor being replaced was Kristi Noem.",
        urls=nom.urls_predecessor_name,
        add_ins="The source should link Kristi Noem to the role being replaced.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, nom_node,
        node_id="predecessor_was_ousted",
        desc="States the replaced official was ousted, with a supporting reference URL.",
        claim="The replaced official, Kristi Noem, was ousted from the position.",
        urls=nom.urls_predecessor_status,
        add_ins="The page should indicate ouster, removal, or forced departure of the predecessor.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, nom_node,
        node_id="nominee_current_position_us_senator",
        desc="States the nominee's current government position at the time of nomination was U.S. Senator, with a supporting reference URL.",
        claim="At the time of nomination, the nominee was serving as a U.S. Senator.",
        urls=nom.urls_nominee_current_position,
        add_ins="The source should identify the nominee's current office as U.S. Senator at that time.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, nom_node,
        node_id="nominee_state_oklahoma",
        desc="States the nominee represents Oklahoma, with a supporting reference URL.",
        claim="The nominee represents the state of Oklahoma.",
        urls=nom.urls_nominee_state,
        add_ins="The source should indicate the Senator's represented state is Oklahoma.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, nom_node,
        node_id="congressional_service_duration_and_timeline",
        desc="States the nominee's congressional service duration/timeline (in Congress since 2013; 10 years in the House; became a Senator in 2023), with a supporting reference URL.",
        claim="Markwayne Mullin has served in Congress since 2013, served 10 years in the House of Representatives, and became a U.S. Senator in 2023.",
        urls=nom.urls_congressional_service,
        add_ins="The page should substantiate the timeline (2013 House start, ~10 years in House, Senator from 2023). Minor rounding tolerances are acceptable.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, nom_node,
        node_id="committee_assignments_listed",
        desc="Lists the nominee's committee assignments as: Senate Armed Services, Appropriations, Health Education Labor and Pensions, and Indian Affairs, with a supporting reference URL.",
        claim="The nominee's Senate committee assignments include Armed Services, Appropriations, Health, Education, Labor and Pensions (HELP), and Indian Affairs.",
        urls=nom.urls_committees,
        add_ins="The source should show committee assignments covering the four listed committees (minor naming variations like HELP are fine).",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, nom_node,
        node_id="tribal_affiliation_cherokee_nation",
        desc="States the nominee is an enrolled member of the Cherokee Nation, with a supporting reference URL.",
        claim="The nominee is an enrolled member of the Cherokee Nation.",
        urls=nom.urls_tribal_affiliation,
        add_ins="The page should explicitly note Cherokee Nation enrollment or membership.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, nom_node,
        node_id="senate_confirmation_required",
        desc="States Senate confirmation is required for the DHS Secretary position, with a supporting reference URL.",
        claim="The position of Secretary of Homeland Security requires Senate confirmation.",
        urls=nom.urls_confirmation_required,
        add_ins="The source should indicate that the DHS Secretary is a Senate-confirmed position (e.g., statute or official documentation).",
        tasks=tasks
    )

    if tasks:
        await evaluator.batch_verify(tasks)


async def verify_benefit_program(evaluator: Evaluator, root):
    ben_node = evaluator.add_parallel(
        id="military_benefit_program_dec_2025",
        desc="One-time monetary benefit program for U.S. service members (December 2025); all required details with reference URLs.",
        parent=root,
        critical=False
    )

    ben: BenefitProgramExtraction = await evaluator.extract(
        prompt=prompt_extract_benefit_program(),
        template_class=BenefitProgramExtraction,
        extraction_name="benefit_program_extraction"
    )

    tasks: List = []

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="program_name_warrior_dividend",
        desc="States the program is called the 'Warrior Dividend', with a supporting reference URL.",
        claim="The program is called the 'Warrior Dividend'.",
        urls=ben.urls_program_name,
        add_ins="The page should explicitly refer to the program by the name 'Warrior Dividend'.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="amount_1776",
        desc="States the payment amount is exactly $1,776 per service member, with a supporting reference URL.",
        claim="The one-time payment amount is exactly $1,776 per eligible service member.",
        urls=ben.urls_amount_per_member,
        add_ins="Ensure the exact figure $1,776 is stated; allow standard currency formatting variants.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="announcement_date_dec_17_2025",
        desc="States the program was announced on December 17, 2025, with a supporting reference URL.",
        claim="The program was announced on December 17, 2025.",
        urls=ben.urls_announcement_date,
        add_ins="Allow date formatting variants; must clearly link to the program's announcement.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="additional_details_date_dec_18_2025",
        desc="States additional details were provided on December 18, 2025, with a supporting reference URL.",
        claim="Additional program details were provided on December 18, 2025.",
        urls=ben.urls_additional_details_date,
        add_ins="The source should indicate a follow-up or detailed guidance on Dec 18, 2025.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="announced_by_trump",
        desc="States President Donald J. Trump announced the program, with a supporting reference URL.",
        claim="President Donald J. Trump announced the Warrior Dividend program.",
        urls=ben.urls_announced_by,
        add_ins="The page should clearly attribute the announcement to President Trump.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="tax_status_non_taxable",
        desc="States the payment is tax-free (non-taxable), with a supporting reference URL.",
        claim="The Warrior Dividend payment is tax-free (non-taxable).",
        urls=ben.urls_tax_status,
        add_ins="The source should explicitly state the non-taxable (tax-free) status of the disbursement.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="classified_as_one_time_bah_supplement",
        desc="States the payment is classified as a one-time Basic Allowance for Housing (BAH) supplement, with a supporting reference URL.",
        claim="The payment is classified as a one-time Basic Allowance for Housing (BAH) supplement.",
        urls=ben.urls_classification,
        add_ins="Look for explicit classification as a one-time BAH supplement in official DoD/Service guidance.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="total_budget_2_6_billion",
        desc="States the total program payout/budget was $2.6 billion, with a supporting reference URL.",
        claim="The program's total payout/budget was $2.6 billion.",
        urls=ben.urls_total_budget,
        add_ins="The source should clearly indicate the total budget figure ($2.6B).",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="eligible_total_and_breakdown",
        desc="States approximately 1.45 million are eligible, including 1.28 million active-duty and 174,000 reserve, with a supporting reference URL.",
        claim="Approximately 1.45 million are eligible for the payment, including about 1.28 million active-duty members and 174,000 reserve-component members.",
        urls=ben.urls_eligible_total_and_breakdown,
        add_ins="The source should provide the total eligibility and the active/reserve breakdown figures (minor rounding acceptable).",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="eligibility_status_date_nov_30_2025",
        desc="States eligibility is determined by status as of November 30, 2025, with a supporting reference URL.",
        claim="Eligibility is determined by a member's status as of November 30, 2025.",
        urls=ben.urls_eligibility_status_date,
        add_ins="The page should explicitly specify Nov 30, 2025 as the status cutoff date for eligibility.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="active_duty_paygrade_eligibility_o6_and_below",
        desc="States active-duty eligibility includes pay grades O-6 and below, with a supporting reference URL.",
        claim="Active-duty eligibility includes pay grades O-6 and below.",
        urls=ben.urls_active_duty_paygrade_eligibility,
        add_ins="The source should state the pay-grade eligibility threshold (O-6 and below) for active-duty.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="reserve_eligibility_orders_31_plus_days",
        desc="States reserve-component eligibility includes being on active-duty orders of 31+ days, with a supporting reference URL.",
        claim="Reserve-component eligibility includes members on active-duty orders of 31 days or more.",
        urls=ben.urls_reserve_eligibility_requirement,
        add_ins="The source should describe reserve eligibility in terms of active-duty orders length (31+ days).",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="payment_deadline_before_dec_20_2025",
        desc="States payments were issued before December 20, 2025, with a supporting reference URL.",
        claim="Payments were issued before December 20, 2025.",
        urls=ben.urls_payment_deadline,
        add_ins="Look for a stated issuance/payment deadline before Dec 20, 2025.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, ben_node,
        node_id="disbursement_standard_military_pay_system",
        desc="States payments were delivered through the standard military pay disbursing system, with a supporting reference URL.",
        claim="Payments were delivered through the standard military pay disbursing system.",
        urls=ben.urls_disbursement_method,
        add_ins="The page should mention disbursement via the standard military pay system (e.g., DFAS/Service pay systems).",
        tasks=tasks
    )

    if tasks:
        await evaluator.batch_verify(tasks)


async def verify_sotu(evaluator: Evaluator, root):
    sotu_node = evaluator.add_parallel(
        id="state_of_union_2026",
        desc="2026 constitutionally mandated State of the Union; all required details with reference URLs.",
        parent=root,
        critical=False
    )

    so: SOTUExtraction = await evaluator.extract(
        prompt=prompt_extract_sotu(),
        template_class=SOTUExtraction,
        extraction_name="sotu_extraction"
    )

    tasks: List = []

    _mk_leaf_and_maybe_task(
        evaluator, sotu_node,
        node_id="sotu_date_feb_24_2026",
        desc="States the address was delivered on February 24, 2026, with a supporting reference URL.",
        claim="The 2026 State of the Union address was delivered on February 24, 2026.",
        urls=so.urls_sotu_date,
        add_ins="Allow date formatting variants; the page should clearly denote Feb 24, 2026 as the delivery date.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, sotu_node,
        node_id="sotu_delivered_by_trump",
        desc="States President Donald J. Trump delivered it, with a supporting reference URL.",
        claim="President Donald J. Trump delivered the 2026 State of the Union address.",
        urls=so.urls_delivered_by,
        add_ins="The page should clearly attribute the 2026 State of the Union to President Trump.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, sotu_node,
        node_id="constitutional_provision_article_ii_section_3",
        desc="Cites Article II, Section 3 as the constitutional provision mandating it, with a supporting reference URL.",
        claim="The State of the Union obligation is mandated by Article II, Section 3 of the U.S. Constitution.",
        urls=so.urls_constitutional_provision,
        add_ins="The source should explicitly cite Article II, Section 3 as the basis for the State of the Union.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, sotu_node,
        node_id="delivered_before_joint_session",
        desc="States it was delivered before a joint session of the House and Senate, with a supporting reference URL.",
        claim="The address was delivered before a joint session of the House of Representatives and the Senate.",
        urls=so.urls_delivered_before,
        add_ins="The page should explicitly mention a joint session of both chambers.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, sotu_node,
        node_id="congressional_record_vol_172_issue_36",
        desc="States the transcript appears in Congressional Record Volume 172, Issue 36, with a supporting reference URL.",
        claim="The transcript appears in Congressional Record Volume 172, Issue 36.",
        urls=so.urls_congressional_record_citation,
        add_ins="The source should include a citation or metadata matching Vol. 172, Issue 36 for the 2026 SOTU.",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, sotu_node,
        node_id="documented_in_other_official_series",
        desc="States it is documented in the Compilation of Presidential Documents, Congressional Record Bound Editions, and Public Papers of the Presidents, with a supporting reference URL.",
        claim="The 2026 State of the Union is documented in the Compilation of Presidential Documents, the Congressional Record Bound Editions, and the Public Papers of the Presidents.",
        urls=so.urls_other_official_series,
        add_ins="The page should explicitly indicate that the SOTU appears in those three official series (minor naming variations acceptable).",
        tasks=tasks
    )

    _mk_leaf_and_maybe_task(
        evaluator, sotu_node,
        node_id="duration_record_over_1h40",
        desc="States it broke previous time records by running over 1 hour 40 minutes, with a supporting reference URL.",
        claim="The 2026 State of the Union broke prior duration records by lasting over 1 hour and 40 minutes.",
        urls=so.urls_duration_record,
        add_ins="The source should compare duration to past records and indicate it exceeded 1 hour 40 minutes.",
        tasks=tasks
    )

    if tasks:
        await evaluator.batch_verify(tasks)


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

    # Add (hypothesized) ground-truth expectations for transparency (these are used for reporting only)
    evaluator.add_ground_truth({
        "military_operation_expected": {
            "date": "January 3, 2026",
            "target_country": "Venezuela",
            "captured_individuals": ["Nicolás Maduro", "Cilia Flores"],
            "announced_by": "President Donald J. Trump",
            "senior_officer": "General Dan Caine (Chairman of the Joint Chiefs of Staff)",
            "secretary_of_state": "Marco Rubio",
            "arraignment": "New York; drug trafficking charges",
            "succession": "Delcy Rodríguez assumed power",
            "forces_used": "U.S. special operations forces",
            "prior_buildup": "Months of U.S. military buildup in the Caribbean"
        },
        "cabinet_nomination_expected": {
            "date": "March 5, 2026",
            "nominee": "Senator Markwayne Mullin",
            "position": "Secretary of Homeland Security",
            "predecessor": "Kristi Noem (ousted)",
            "nominee_current_position": "U.S. Senator from Oklahoma",
            "congressional_timeline": "In Congress since 2013; ~10 years in the House; Senator since 2023",
            "committees": ["Armed Services", "Appropriations", "HELP", "Indian Affairs"],
            "tribal": "Cherokee Nation",
            "confirmation_required": "Yes"
        },
        "benefit_program_expected": {
            "program_name": "Warrior Dividend",
            "amount": "$1,776",
            "announcement_date": "December 17, 2025",
            "additional_details_date": "December 18, 2025",
            "announced_by": "President Donald J. Trump",
            "tax_status": "Non-taxable",
            "classification": "One-time BAH supplement",
            "total_budget": "$2.6 billion",
            "eligibility_counts": "Approx. 1.45M total; 1.28M active-duty; 174k reserve",
            "eligibility_status_date": "November 30, 2025",
            "active_duty_paygrade": "O-6 and below",
            "reserve_requirement": "On active-duty orders of 31+ days",
            "payment_deadline": "Before December 20, 2025",
            "disbursement": "Standard military pay disbursing system"
        },
        "sotu_expected": {
            "date": "February 24, 2026",
            "delivered_by": "President Donald J. Trump",
            "constitution": "Article II, Section 3",
            "delivered_before": "Joint session of House and Senate",
            "congressional_record": "Vol. 172, Issue 36",
            "other_series": [
                "Compilation of Presidential Documents",
                "Congressional Record Bound Editions",
                "Public Papers of the Presidents"
            ],
            "duration_record": "Over 1 hour 40 minutes"
        }
    })

    # Build subtrees (parallel at root)
    await verify_military_operation(evaluator, root)
    await verify_cabinet_nomination(evaluator, root)
    await verify_benefit_program(evaluator, root)
    await verify_sotu(evaluator, root)

    return evaluator.get_summary()