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
TASK_ID = "us_services_march_2026"
TASK_DESCRIPTION = """
In March 2026, a U.S. lawful permanent resident who is a dual national of Brazil and Canada needs to navigate several U.S. government services. They plan to apply for the Global Entry trusted traveler program and need to obtain a U.S. passport for urgent international travel scheduled in 10 days. Additionally, they want to understand their status regarding the immigrant visa processing pause announced in January 2026, and they need to know the constitutional requirements if Congress were to override a presidential veto of legislation related to government programs. Based on official U.S. government policies and procedures as of March 2026, provide: (1) Immigrant Visa Pause Status: Determine whether Brazil is on the list of 75 countries subject to the immigrant visa processing pause that took effect January 21, 2026, and explain what exemption applies to dual nationals who hold a Canadian passport. (2) Global Entry Requirements: Identify all requirements for applying for Global Entry, including the application fee amount, membership validity period, confirmation that U.S. lawful permanent residents are eligible, and the required vetting processes. (3) Urgent Passport Service: Determine the appropriate type of expedited passport service for someone with international travel in 10 days, explain the appointment process, and specify what proof of travel is required. (4) DHS Shutdown Context: Provide information about the February-March 2026 partial DHS shutdown's impact on trusted traveler programs, including the date Global Entry was suspended, when it was restored, and what happened with TSA PreCheck. (5) Congressional Override Process: Explain the constitutional requirements for Congress to override a presidential veto, including the specific vote threshold required in both chambers and the type of vote that must be conducted. All responses must be supported by official U.S. government source URLs.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VisaPauseExtraction(BaseModel):
    brazil_on_list_statement: Optional[str] = None
    brazil_support_urls: List[str] = Field(default_factory=list)
    exemption_statement: Optional[str] = None
    exemption_support_urls: List[str] = Field(default_factory=list)
    policy_reference_url: Optional[str] = None


class GlobalEntryExtraction(BaseModel):
    application_fee: Optional[str] = None
    membership_duration: Optional[str] = None
    lpr_eligible_statement: Optional[str] = None
    vetting_process_statement: Optional[str] = None
    ge_support_urls: List[str] = Field(default_factory=list)
    reference_url: Optional[str] = None


class PassportUrgentExtraction(BaseModel):
    service_type_statement: Optional[str] = None
    appointment_statement: Optional[str] = None
    proof_requirement_statement: Optional[str] = None
    passport_support_urls: List[str] = Field(default_factory=list)
    reference_url: Optional[str] = None


class DHSShutdownExtraction(BaseModel):
    ge_suspension_dates_statement: Optional[str] = None
    tsa_precheck_statement: Optional[str] = None
    shutdown_support_urls: List[str] = Field(default_factory=list)
    reference_url: Optional[str] = None


class OverrideExtraction(BaseModel):
    vote_threshold_statement: Optional[str] = None
    both_chambers_statement: Optional[str] = None
    vote_type_statement: Optional[str] = None
    override_support_urls: List[str] = Field(default_factory=list)
    reference_url: Optional[str] = None


class FullExtraction(BaseModel):
    visa_pause: Optional[VisaPauseExtraction] = None
    global_entry: Optional[GlobalEntryExtraction] = None
    passport: Optional[PassportUrgentExtraction] = None
    dhs_shutdown: Optional[DHSShutdownExtraction] = None
    override: Optional[OverrideExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_full() -> str:
    return """
    Extract from the provided answer ONLY the explicit statements and the source URLs the answer cites for each required topic. Return null for any missing scalar field and an empty array for any missing URLs list. Do not fabricate or infer anything not present in the answer text. Always return complete absolute URLs as they appear in the answer.

    Structure your JSON as follows:

    {
      "visa_pause": {
        "brazil_on_list_statement": string or null,
        "brazil_support_urls": [urls...],
        "exemption_statement": string or null,
        "exemption_support_urls": [urls...],
        "policy_reference_url": string or null
      },
      "global_entry": {
        "application_fee": string or null,
        "membership_duration": string or null,
        "lpr_eligible_statement": string or null,
        "vetting_process_statement": string or null,
        "ge_support_urls": [urls...],
        "reference_url": string or null
      },
      "passport": {
        "service_type_statement": string or null,
        "appointment_statement": string or null,
        "proof_requirement_statement": string or null,
        "passport_support_urls": [urls...],
        "reference_url": string or null
      },
      "dhs_shutdown": {
        "ge_suspension_dates_statement": string or null,
        "tsa_precheck_statement": string or null,
        "shutdown_support_urls": [urls...],
        "reference_url": string or null
      },
      "override": {
        "vote_threshold_statement": string or null,
        "both_chambers_statement": string or null,
        "vote_type_statement": string or null,
        "override_support_urls": [urls...],
        "reference_url": string or null
      }
    }

    Special URL extraction rules:
    - Extract only URLs that are explicitly present in the answer (including within markdown links).
    - Do not add or infer any URLs not shown in the answer.
    - Prefer official U.S. government sources (domains ending in .gov, .mil) when present in the answer.
    - For the DHS shutdown context, credible major news outlets are acceptable if cited in the answer.

    Notes on values:
    - application_fee should reflect the numeric amount as stated (e.g., "$120").
    - membership_duration should reflect the validity text (e.g., "5 years").
    - For vetting_process_statement, capture any wording that mentions both background check and an in-person interview if present.
    - For phone numbers or appointment methods in appointment_statement, capture the exact statement from the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def collect_sources(*groups: Optional[List[str] | str]) -> List[str]:
    """Collect and de-duplicate URL sources from mixed inputs while preserving order."""
    seen = set()
    ordered: List[str] = []
    for g in groups:
        if not g:
            continue
        if isinstance(g, list):
            for u in g:
                if isinstance(u, str) and u.strip() and u not in seen:
                    seen.add(u)
                    ordered.append(u.strip())
        elif isinstance(g, str):
            u = g.strip()
            if u and u not in seen:
                seen.add(u)
                ordered.append(u)
    return ordered


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_visa_pause_checks(evaluator: Evaluator, parent, data: VisaPauseExtraction) -> None:
    # Top-level: immigrant_visa_pause_status (critical, parallel)
    top = evaluator.add_parallel(
        id="immigrant_visa_pause_status",
        desc="Determine the applicant's status under the immigrant visa processing pause that took effect January 21, 2026",
        parent=parent,
        critical=True
    )

    # Sequential content checks (critical)
    content = evaluator.add_sequential(
        id="visa_pause_content",
        desc="Verify Brazil's status and dual national exemption",
        parent=top,
        critical=True
    )

    # Leaf: brazil_on_list (critical)
    node_brazil = evaluator.add_leaf(
        id="brazil_on_list",
        desc="Verify that Brazil is on the list of 75 countries subject to the immigrant visa pause",
        parent=content,
        critical=True
    )
    claim_brazil = "Brazil is included among the 75 countries subject to the immigrant visa processing pause that took effect on January 21, 2026."
    await evaluator.verify(
        claim=claim_brazil,
        node=node_brazil,
        sources=collect_sources(data.brazil_support_urls, data.policy_reference_url),
        additional_instruction="Use the cited Department of State or other official policy page(s). Look for an explicit country list that includes Brazil. Treat as unsupported if the provided page(s) do not show Brazil on the list."
    )

    # Leaf: exemption_applies (critical)
    node_exempt = evaluator.add_leaf(
        id="exemption_applies",
        desc="Confirm that dual nationals using a passport from a non-listed country (Canada) are exempt from the pause",
        parent=content,
        critical=True
    )
    claim_exempt = "Dual nationals who present a passport from a country not on the pause list—such as Canada—are exempt from the immigrant visa processing pause."
    await evaluator.verify(
        claim=claim_exempt,
        node=node_exempt,
        sources=collect_sources(data.exemption_support_urls, data.policy_reference_url),
        additional_instruction="Verify that the official policy explicitly states an exemption for individuals who present a passport from a non-listed country (e.g., Canada)."
    )

    # Leaf: reference_url_visa (critical)
    node_ref = evaluator.add_leaf(
        id="reference_url_visa",
        desc="Provide official U.S. Department of State URL confirming the visa pause policy and exemptions",
        parent=top,
        critical=True
    )
    claim_ref = "This webpage is an official U.S. Department of State page (state.gov) that describes the immigrant visa processing pause effective January 21, 2026 and includes the relevant exemptions."
    await evaluator.verify(
        claim=claim_ref,
        node=node_ref,
        sources=collect_sources(data.policy_reference_url, data.brazil_support_urls, data.exemption_support_urls),
        additional_instruction="Confirm the domain is state.gov and that the page discusses the pause and exemptions. If the URL is not from state.gov or does not cover the policy, mark as unsupported."
    )


async def build_global_entry_checks(evaluator: Evaluator, parent, data: GlobalEntryExtraction) -> None:
    # Top-level: global_entry_requirements (critical, parallel)
    ge = evaluator.add_parallel(
        id="global_entry_requirements",
        desc="Identify all requirements for Global Entry application as of March 2026",
        parent=parent,
        critical=True
    )
    sources = collect_sources(data.reference_url, data.ge_support_urls)

    # application_fee (critical)
    node_fee = evaluator.add_leaf(
        id="application_fee",
        desc="State the non-refundable application fee amount ($120)",
        parent=ge,
        critical=True
    )
    claim_fee = "The Global Entry non-refundable application fee is $120."
    await evaluator.verify(
        claim=claim_fee,
        node=node_fee,
        sources=sources,
        additional_instruction="Use the official CBP Global Entry program page or another official DHS/CBP source that states the current application fee."
    )

    # membership_duration (critical)
    node_duration = evaluator.add_leaf(
        id="membership_duration",
        desc="State the membership validity period (5 years)",
        parent=ge,
        critical=True
    )
    claim_duration = "Global Entry membership is valid for five (5) years."
    await evaluator.verify(
        claim=claim_duration,
        node=node_duration,
        sources=sources,
        additional_instruction="Verify on the CBP Global Entry page or equivalent official source that membership lasts 5 years."
    )

    # eligibility_status (critical)
    node_elig = evaluator.add_leaf(
        id="eligibility_status",
        desc="Confirm that U.S. lawful permanent residents are eligible to apply",
        parent=ge,
        critical=True
    )
    claim_elig = "U.S. lawful permanent residents (green card holders) are eligible to apply for Global Entry."
    await evaluator.verify(
        claim=claim_elig,
        node=node_elig,
        sources=sources,
        additional_instruction="Confirm on an official CBP/DHS source that LPRs are eligible applicants for Global Entry."
    )

    # required_process (critical)
    node_process = evaluator.add_leaf(
        id="required_process",
        desc="Identify that both a background check and in-person interview are required",
        parent=ge,
        critical=True
    )
    claim_process = "Global Entry requires both a background check and an in-person interview prior to approval."
    await evaluator.verify(
        claim=claim_process,
        node=node_process,
        sources=sources,
        additional_instruction="Confirm that the program requires a background check and an in-person interview (e.g., at an enrollment center or via Enrollment on Arrival)."
    )

    # reference_url_ge (critical)
    node_ref = evaluator.add_leaf(
        id="reference_url_ge",
        desc="Provide official CBP or DHS URL for Global Entry program requirements",
        parent=ge,
        critical=True
    )
    claim_ref = "This webpage is an official U.S. Customs and Border Protection (CBP) or DHS page that describes Global Entry requirements."
    await evaluator.verify(
        claim=claim_ref,
        node=node_ref,
        sources=collect_sources(data.reference_url),
        additional_instruction="Confirm that the domain is cbp.gov or dhs.gov and that the page covers Global Entry requirements."
    )


async def build_passport_urgent_checks(evaluator: Evaluator, parent, data: PassportUrgentExtraction) -> None:
    # Top-level: urgent_passport_service (critical, parallel)
    up = evaluator.add_parallel(
        id="urgent_passport_service",
        desc="Determine the appropriate passport service type and requirements for travel in 10 days",
        parent=parent,
        critical=True
    )

    # Sequential content checks (critical)
    content = evaluator.add_sequential(
        id="passport_service_content",
        desc="Identify service type, appointment process, and proof requirements",
        parent=up,
        critical=True
    )
    sources = collect_sources(data.reference_url, data.passport_support_urls)

    # service_type (critical)
    node_service = evaluator.add_leaf(
        id="service_type",
        desc="Identify that Urgent Travel Service (not Life-or-Death Emergency) applies for international travel within 14 days",
        parent=content,
        critical=True
    )
    claim_service = "For international travel within 14 days, the correct option is Urgent Travel Service (not Life-or-Death Emergency) at a passport agency/center."
    await evaluator.verify(
        claim=claim_service,
        node=node_service,
        sources=sources,
        additional_instruction="Verify on the official Travel.State.Gov pages that Urgent Travel Service applies within 14 days and is distinct from Life-or-Death Emergency Service."
    )

    # appointment_required (critical)
    node_appt = evaluator.add_leaf(
        id="appointment_required",
        desc="Confirm that an appointment must be made by calling 1-877-487-2778 and walk-ins are not allowed",
        parent=content,
        critical=True
    )
    claim_appt = "An appointment for Urgent Travel Service must be made by calling 1-877-487-2778; walk-ins are not permitted."
    await evaluator.verify(
        claim=claim_appt,
        node=node_appt,
        sources=sources,
        additional_instruction="Confirm the appointment process on Travel.State.Gov, including the phone number and that walk-ins are not accepted."
    )

    # proof_requirement (critical)
    node_proof = evaluator.add_leaf(
        id="proof_requirement",
        desc="State that proof of urgent travel within 14 days must be provided",
        parent=content,
        critical=True
    )
    claim_proof = "Applicants must provide proof of international travel within 14 days to use Urgent Travel Service."
    await evaluator.verify(
        claim=claim_proof,
        node=node_proof,
        sources=sources,
        additional_instruction="Verify that Travel.State.Gov requires evidence of travel within 14 days (e.g., flight itinerary) for Urgent Travel Service."
    )

    # reference_url_passport (critical)
    node_ref = evaluator.add_leaf(
        id="reference_url_passport",
        desc="Provide official U.S. Department of State Travel.State.Gov URL for urgent passport services",
        parent=up,
        critical=True
    )
    claim_ref = "This webpage is an official U.S. Department of State Travel.State.Gov page that describes Urgent Travel Service and its appointment/proof requirements."
    await evaluator.verify(
        claim=claim_ref,
        node=node_ref,
        sources=collect_sources(data.reference_url),
        additional_instruction="Confirm the domain is travel.state.gov and the subject is Urgent Travel or expedited passport service."
    )


async def build_dhs_shutdown_checks(evaluator: Evaluator, parent, data: DHSShutdownExtraction) -> None:
    # Top-level: dhs_shutdown_context (critical, parallel)
    dhs = evaluator.add_parallel(
        id="dhs_shutdown_context",
        desc="Provide information about the February-March 2026 DHS shutdown's impact on trusted traveler programs",
        parent=parent,
        critical=True
    )
    sources = collect_sources(data.reference_url, data.shutdown_support_urls)

    # global_entry_suspension (critical)
    node_ge = evaluator.add_leaf(
        id="global_entry_suspension",
        desc="State the date Global Entry was suspended (February 22, 2026) and when it was restored (March 11, 2026)",
        parent=dhs,
        critical=True
    )
    claim_ge = "Global Entry was suspended on February 22, 2026, and restored on March 11, 2026."
    await evaluator.verify(
        claim=claim_ge,
        node=node_ge,
        sources=sources,
        additional_instruction="Use official DHS/CBP releases or credible national news coverage that reported both the suspension date (Feb 22, 2026) and restoration date (Mar 11, 2026)."
    )

    # tsa_precheck_reversal (critical)
    node_tsa = evaluator.add_leaf(
        id="tsa_precheck_reversal",
        desc="Identify that TSA PreCheck was initially suspended but then the suspension was reversed",
        parent=dhs,
        critical=True
    )
    claim_tsa = "TSA PreCheck was initially suspended during the DHS funding lapse but the suspension decision was subsequently reversed (restored)."
    await evaluator.verify(
        claim=claim_tsa,
        node=node_tsa,
        sources=sources,
        additional_instruction="Confirm that sources explicitly note both an initial suspension affecting TSA PreCheck and a reversal/restoration."
    )

    # reference_url_shutdown (critical)
    node_ref = evaluator.add_leaf(
        id="reference_url_shutdown",
        desc="Provide credible news or government URL documenting the DHS shutdown and program suspensions",
        parent=dhs,
        critical=True
    )
    claim_ref = "This webpage is either an official U.S. government page or a credible national news outlet reporting on the DHS shutdown and the status of trusted traveler programs (e.g., Global Entry, TSA PreCheck) in Feb–Mar 2026."
    await evaluator.verify(
        claim=claim_ref,
        node=node_ref,
        sources=collect_sources(data.reference_url),
        additional_instruction="Accept .gov domains or major reputable outlets (e.g., AP, Reuters, NPR, major national newspapers). Reject personal blogs or non-credible sites."
    )


async def build_override_checks(evaluator: Evaluator, parent, data: OverrideExtraction) -> None:
    # Top-level: congressional_override_process (critical, parallel)
    ov = evaluator.add_parallel(
        id="congressional_override_process",
        desc="Explain the constitutional requirements for Congress to override a presidential veto of legislation restoring suspended programs",
        parent=parent,
        critical=True
    )
    sources = collect_sources(data.reference_url, data.override_support_urls)

    # vote_threshold (critical)
    node_thresh = evaluator.add_leaf(
        id="vote_threshold",
        desc="State that a two-thirds vote of Members voting (with quorum present) is required in both chambers",
        parent=ov,
        critical=True
    )
    claim_thresh = "Overriding a presidential veto requires a two-thirds vote of Members present and voting, assuming a quorum is present."
    await evaluator.verify(
        claim=claim_thresh,
        node=node_thresh,
        sources=sources,
        additional_instruction="Verify via an official U.S. government source (e.g., Congress.gov, Archives.gov) that the constitutional override threshold is two-thirds of Members present and voting with a quorum."
    )

    # both_chambers (critical)
    node_both = evaluator.add_leaf(
        id="both_chambers",
        desc="Confirm that both the House and Senate must successfully override the veto",
        parent=ov,
        critical=True
    )
    claim_both = "Both the House of Representatives and the Senate must each achieve the two-thirds vote to override a presidential veto."
    await evaluator.verify(
        claim=claim_both,
        node=node_both,
        sources=sources,
        additional_instruction="Confirm official guidance that both chambers must independently vote to override by two-thirds."
    )

    # vote_type (critical)
    node_type = evaluator.add_leaf(
        id="vote_type",
        desc="Specify that the override vote must be a recorded/roll call vote",
        parent=ov,
        critical=True
    )
    claim_type = "A veto override must be conducted as a recorded (roll call) vote in each chamber."
    await evaluator.verify(
        claim=claim_type,
        node=node_type,
        sources=sources,
        additional_instruction="Use an official source describing veto override procedure and confirm that the vote is taken by roll call/recorded vote."
    )

    # reference_url_override (critical)
    node_ref = evaluator.add_leaf(
        id="reference_url_override",
        desc="Provide official government URL (Congress.gov, Archives.gov, or similar) documenting the veto override process",
        parent=ov,
        critical=True
    )
    claim_ref = "This webpage is an official U.S. government source (e.g., Congress.gov or Archives.gov) that explains the presidential veto override process and requirements."
    await evaluator.verify(
        claim=claim_ref,
        node=node_ref,
        sources=collect_sources(data.reference_url),
        additional_instruction="Confirm the domain is a U.S. government site such as congress.gov, archives.gov, house.gov, or senate.gov and the page explains veto overrides."
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
    Evaluate an answer for the 'us_services_march_2026' task using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates major topics independently
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_full(),
        template_class=FullExtraction,
        extraction_name="full_extraction"
    )

    # Normalize extracted sections to non-None objects for downstream logic
    visa_data = extracted.visa_pause or VisaPauseExtraction()
    ge_data = extracted.global_entry or GlobalEntryExtraction()
    passport_data = extracted.passport or PassportUrgentExtraction()
    dhs_data = extracted.dhs_shutdown or DHSShutdownExtraction()
    override_data = extracted.override or OverrideExtraction()

    # Build verification subtrees according to the rubric
    await build_visa_pause_checks(evaluator, root, visa_data)
    await build_global_entry_checks(evaluator, root, ge_data)
    await build_passport_urgent_checks(evaluator, root, passport_data)
    await build_dhs_shutdown_checks(evaluator, root, dhs_data)
    await build_override_checks(evaluator, root, override_data)

    # Return evaluation summary
    return evaluator.get_summary()