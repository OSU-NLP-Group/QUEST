import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "outbreak_prep_plan"
TASK_DESCRIPTION = """
Your hospital's Infection Prevention and Control Committee has tasked you with developing a comprehensive outbreak preparedness plan document to address two priority infectious disease threats: measles and Candida auris. The plan must be compliant with the Joint Commission infection prevention and control standards that became effective July 1, 2024, and CDC guidelines.

The outbreak preparedness plan document must include the following components:

For Measles Response:
1. Specify the duration of airborne precautions for measles patients (including how to count days from rash onset)
2. Define the minimum air changes per hour (ACH) required for Airborne Infection Isolation Rooms (AIIR)
3. Identify the required respiratory protection for healthcare personnel entering measles patient rooms
4. Describe the immunity documentation requirements for healthcare personnel who care for measles patients
5. Specify the work exclusion period for exposed healthcare personnel who lack documented immunity
6. Provide a reference URL to an authoritative source (CDC or equivalent) for measles infection control guidelines

For Candida auris Response:
1. Specify the type of transmission-based precautions required for patients colonized or infected with C. auris
2. Describe the preferred room assignment for C. auris patients
3. Define the environmental cleaning and disinfection protocol requirements
4. Specify the type of disinfectant that must be used (including regulatory approval)
5. Describe when admission screening for C. auris should be considered
6. Provide a reference URL to an authoritative source (CDC or equivalent) for C. auris infection control guidelines

For General Infection Control Requirements:
1. State the baseline precautions that apply to all patient care
2. Identify the personnel qualification requirement for infection prevention oversight
3. Describe staff training requirements for infection control
4. Specify the personal protective equipment that must be available

For Regulatory Compliance:
1. Reference the applicable Joint Commission standards (with effective date)
2. Reference CDC's core infection prevention and control practices

Develop this outbreak preparedness plan document ensuring all specified components are included with accurate, evidence-based information grounded in current CDC guidelines and Joint Commission standards.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MeaslesSection(BaseModel):
    airborne_precautions_statement: Optional[str] = None
    aiir_ach: Optional[str] = None
    respirator_requirement: Optional[str] = None
    hcp_immunity_requirement: Optional[str] = None
    exposed_hcp_exclusion: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CAurisSection(BaseModel):
    precautions_type: Optional[str] = None
    room_assignment: Optional[str] = None
    environmental_cleaning: Optional[str] = None
    disinfectant_type: Optional[str] = None
    admission_screening: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class GeneralSection(BaseModel):
    baseline_precautions: Optional[str] = None
    infection_preventionist_requirement: Optional[str] = None
    staff_training: Optional[str] = None
    ppe_availability: Optional[str] = None


class RegulatorySection(BaseModel):
    joint_commission_reference: Optional[str] = None
    effective_date: Optional[str] = None
    jc_urls: List[str] = Field(default_factory=list)
    cdc_core_practices_reference: Optional[str] = None
    cdc_core_urls: List[str] = Field(default_factory=list)


class PlanExtraction(BaseModel):
    measles: Optional[MeaslesSection] = None
    candida_auris: Optional[CAurisSection] = None
    general: Optional[GeneralSection] = None
    regulatory: Optional[RegulatorySection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
Extract the outbreak preparedness plan content exactly as stated in the provided answer text. Populate the following JSON fields with direct quotations or close paraphrases from the answer (do not invent or normalize beyond what is present). If a field is not explicitly present, return null (or an empty list for URL arrays).

Sections and fields to extract:

1) measles:
- airborne_precautions_statement: The plan's stated duration policy for airborne precautions for measles AND how to count days (e.g., "4 days after rash onset; rash onset is Day 0").
- aiir_ach: The plan's stated minimum air changes per hour (ACH) for AIIRs used for measles (e.g., "12 ACH minimum"; include any qualifiers such as 'new construction 12 ACH; existing 6 ACH').
- respirator_requirement: The plan's stated respiratory protection for personnel entering measles rooms (e.g., "fit-tested N95 or higher-level respirator").
- hcp_immunity_requirement: The plan's documentation requirement for HCP immunity to measles (e.g., "2 documented MMR doses or laboratory evidence").
- exposed_hcp_exclusion: The plan's stated work exclusion period for exposed HCP who lack documented immunity (e.g., "exclude from day 5 through day 21 after last exposure").
- reference_urls: All URLs in the answer cited for measles infection control guidance (prefer CDC; include any authoritative public health URLs actually listed).

2) candida_auris:
- precautions_type: The plan's transmission-based precautions for C. auris (e.g., "Contact Precautions").
- room_assignment: The plan's preferred placement (e.g., "single-patient room"; cohort if needed).
- environmental_cleaning: The plan's cleaning/disinfection requirements for rooms of C. auris patients (daily/terminal cleaning, high-touch surfaces, etc.).
- disinfectant_type: The plan's required disinfectant type including regulatory qualifier (e.g., "EPA-registered hospital-grade disinfectant effective against C. auris (EPA List P)").
- admission_screening: The plan's criteria for when to consider admission screening for C. auris (e.g., transfers from facilities with known C. auris).
- reference_urls: All URLs in the answer cited for C. auris infection control guidance (prefer CDC/EPA authoritative pages).

3) general:
- baseline_precautions: The plan's statement that Standard Precautions apply to all patient care.
- infection_preventionist_requirement: The plan's requirement to designate a qualified infection preventionist for oversight (as stated).
- staff_training: The plan's staff training requirements for infection control (e.g., onboarding and periodic refreshers).
- ppe_availability: The plan's statement on PPE availability (e.g., gloves, gowns, masks/respirators, eye protection at point of care).

4) regulatory:
- joint_commission_reference: Any explicit reference to the Joint Commission infection prevention and control standards (as cited in the plan text).
- effective_date: The effective date cited for the Joint Commission standards if present (e.g., "July 1, 2024").
- jc_urls: Any URLs pointing to Joint Commission standards or official JC pages explicitly included in the answer.
- cdc_core_practices_reference: Any explicit reference to CDC's Core Infection Prevention and Control Practices.
- cdc_core_urls: Any URLs to CDC's Core Practices or CDC infection control framework pages mentioned in the answer.

Return a JSON object with structure:
{
  "measles": {...},
  "candida_auris": {...},
  "general": {...},
  "regulatory": {...}
}
Only extract what the answer explicitly provides. For all URL arrays, include only valid URLs actually shown in the answer (plain link or markdown). If none are present, return an empty list.
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_list(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    cleaned = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    return cleaned if cleaned else None


def _safe(s: Optional[str]) -> str:
    return s if isinstance(s, str) else ""


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_measles_protocol(
    evaluator: Evaluator,
    parent_node,
    measles: Optional[MeaslesSection],
) -> None:
    measles_node = evaluator.add_parallel(
        id="Measles_Response_Protocol",
        desc="Complete measles outbreak response protocol including patient isolation, healthcare worker protection, and facility requirements",
        parent=parent_node,
        critical=True,
    )
    measles_sources = _nonempty_list(measles.reference_urls) if measles else None

    # 1) Airborne precautions duration (4 days after rash onset; Day 0 = rash onset)
    node_airborne = evaluator.add_leaf(
        id="Measles_Airborne_Precautions_Duration",
        desc="Specifies that patients with measles remain in airborne precautions for 4 days after rash onset, with rash onset designated as Day 0",
        parent=measles_node,
        critical=True,
    )
    claim_airborne = (
        f"The outbreak preparedness plan includes the following measles airborne-precautions duration policy: "
        f"'{_safe(measles.airborne_precautions_statement) if measles else ''}'. "
        f"This correctly specifies that patients with measles remain in airborne precautions for 4 days after rash onset, "
        f"with rash onset counted as Day 0."
    )
    await evaluator.verify(
        claim=claim_airborne,
        node=node_airborne,
        sources=measles_sources,
        additional_instruction=(
            "Check two things: (A) The Answer text explicitly contains this or an equivalent statement "
            "about 4 days after rash onset with rash onset as Day 0 (accept minor wording variations such as "
            "'through day 4 after rash onset'). (B) The cited source URL(s) (preferably CDC) support this rule. "
            "If either condition is not met, mark as Incorrect."
        ),
    )

    # 2) AIIR ventilation ≥12 ACH
    node_aiir = evaluator.add_leaf(
        id="Measles_AIIR_Ventilation",
        desc="Specifies that Airborne Infection Isolation Rooms (AIIR) must provide minimum 12 air changes per hour (ACH)",
        parent=measles_node,
        critical=True,
    )
    claim_aiir = (
        f"The plan specifies the AIIR ventilation requirement as: '{_safe(measles.aiir_ach) if measles else ''}'. "
        f"This correctly requires Airborne Infection Isolation Rooms to provide a minimum of 12 air changes per hour (ACH)."
    )
    await evaluator.verify(
        claim=claim_aiir,
        node=node_aiir,
        sources=measles_sources,  # If none provided, the verifier will fall back to simple verification.
        additional_instruction=(
            "Accept formulations like '≥12 ACH', 'at least 12 ACH', or '12 ACH minimum'. "
            "If the plan distinguishes new/renovated (≥12 ACH) vs. existing (≥6 ACH), that is acceptable. "
            "Verify that the Answer text actually states an AIIR ACH standard and that supplied sources align."
        ),
    )

    # 3) N95 respirator (or higher) required for HCP entering room
    node_n95 = evaluator.add_leaf(
        id="Measles_N95_Respirator_Requirement",
        desc="Requires N95 respirators (or equivalent) for healthcare personnel entering rooms of patients with suspected or confirmed measles",
        parent=measles_node,
        critical=True,
    )
    claim_n95 = (
        f"The plan requires respiratory protection for personnel entering measles rooms stated as: "
        f"'{_safe(measles.respirator_requirement) if measles else ''}'. "
        f"This correctly requires a fit-tested N95 or higher-level respirator (e.g., PAPR) for entry."
    )
    await evaluator.verify(
        claim=claim_n95,
        node=node_n95,
        sources=measles_sources,
        additional_instruction=(
            "Confirm the Answer explicitly requires N95 (or equivalent/higher, e.g., PAPR) for suspected/confirmed measles rooms. "
            "Minor wording variants are acceptable. Validate with cited authoritative source(s)."
        ),
    )

    # 4) HCP immunity documentation
    node_immunity = evaluator.add_leaf(
        id="Measles_HCP_Immunity_Documentation",
        desc="Requires healthcare personnel to have documented presumptive evidence of measles immunity (2 doses MMR vaccine or laboratory evidence)",
        parent=measles_node,
        critical=True,
    )
    claim_immunity = (
        f"The plan states HCP immunity documentation as: '{_safe(measles.hcp_immunity_requirement) if measles else ''}'. "
        f"This correctly requires presumptive evidence of immunity such as 2 documented MMR doses or laboratory evidence of immunity."
    )
    await evaluator.verify(
        claim=claim_immunity,
        node=node_immunity,
        sources=measles_sources,
        additional_instruction=(
            "Accept phrasing that clearly requires documented MMR vaccination (2 doses) or laboratory evidence of immunity for HCP. "
            "Do not require 'birth before 1957' as sufficient for HCP. Validate alignment with CDC guidance using provided URLs."
        ),
    )

    # 5) Exposed HCP without immunity: work exclusion day 5–21 after last exposure
    node_excl = evaluator.add_leaf(
        id="Measles_Exposed_HCP_Exclusion",
        desc="Specifies that exposed healthcare personnel without immunity must be excluded from work from day 5 through day 21 after last exposure",
        parent=measles_node,
        critical=True,
    )
    claim_excl = (
        f"The plan states the work exclusion period for exposed HCP without documented immunity as: "
        f"'{_safe(measles.exposed_hcp_exclusion) if measles else ''}'. "
        f"This correctly specifies exclusion from day 5 through day 21 after the last exposure."
    )
    await evaluator.verify(
        claim=claim_excl,
        node=node_excl,
        sources=measles_sources,
        additional_instruction=(
            "Verify the Answer explicitly states exclusion from day 5 through day 21 after last exposure for HCP without immunity. "
            "Validate with the cited CDC source(s)."
        ),
    )

    # 6) Reference URL to authoritative measles guidance
    node_m_ref = evaluator.add_leaf(
        id="Measles_Reference_URL",
        desc="Provides reference URL to CDC or authoritative source for measles infection control guidelines",
        parent=measles_node,
        critical=True,
    )
    first_measles_url = measles_sources[0] if measles_sources else ""
    claim_m_ref = (
        f"The plan includes at least one reference URL to an authoritative measles infection control guideline: "
        f"{measles_sources if measles_sources else []}. "
        f"The URL '{first_measles_url}' is an authoritative measles infection control guidance page."
    )
    await evaluator.verify(
        claim=claim_m_ref,
        node=node_m_ref,
        sources=measles_sources or None,
        additional_instruction=(
            "Use the Answer to confirm that at least one URL is actually included. "
            "The URL should be authoritative (preferably cdc.gov). "
            "Mark Incorrect if no URL is present in the Answer or if the provided page is irrelevant."
        ),
    )


async def verify_cauris_protocol(
    evaluator: Evaluator,
    parent_node,
    cauris: Optional[CAurisSection],
) -> None:
    ca_node = evaluator.add_parallel(
        id="Candida_Auris_Response_Protocol",
        desc="Complete Candida auris response protocol including isolation measures, environmental controls, and screening procedures",
        parent=parent_node,
        critical=True,
    )
    ca_sources = _nonempty_list(cauris.reference_urls) if cauris else None

    # 1) Contact precautions
    node_cp = evaluator.add_leaf(
        id="C_Auris_Contact_Precautions",
        desc="Specifies that contact precautions are required for patients colonized or infected with Candida auris",
        parent=ca_node,
        critical=True,
    )
    claim_cp = (
        f"The plan's transmission-based precautions for C. auris are: '{_safe(cauris.precautions_type) if cauris else ''}'. "
        f"This correctly requires Contact Precautions for colonized or infected patients in acute care."
    )
    await evaluator.verify(
        claim=claim_cp,
        node=node_cp,
        sources=ca_sources,
        additional_instruction=(
            "Confirm the Answer explicitly requires Contact Precautions (note: long-term care may use Enhanced Barrier Precautions, "
            "but for acute care Contact Precautions are appropriate). Validate alignment with CDC guidance using the provided URLs."
        ),
    )

    # 2) Single room placement
    node_room = evaluator.add_leaf(
        id="C_Auris_Single_Room",
        desc="Describes the preferred room assignment for C. auris patients (single-patient room placement when possible)",
        parent=ca_node,
        critical=True,
    )
    claim_room = (
        f"The plan's room assignment for C. auris is: '{_safe(cauris.room_assignment) if cauris else ''}'. "
        f"This correctly prefers single-patient room placement when feasible (or cohorting with other C. auris patients if necessary)."
    )
    await evaluator.verify(
        claim=claim_room,
        node=node_room,
        sources=ca_sources,
        additional_instruction=(
            "Verify the Answer explicitly states preference for single-patient rooms (with dedicated restroom if possible), "
            "and cohorting only when necessary. Validate using the cited CDC pages."
        ),
    )

    # 3) Enhanced environmental cleaning and disinfection
    node_clean = evaluator.add_leaf(
        id="C_Auris_Environmental_Cleaning",
        desc="Specifies enhanced environmental cleaning and disinfection protocols for rooms of C. auris patients",
        parent=ca_node,
        critical=True,
    )
    claim_clean = (
        f"The plan's environmental cleaning protocol for C. auris rooms is: "
        f"'{_safe(cauris.environmental_cleaning) if cauris else ''}'. "
        f"This correctly describes enhanced cleaning (focus on high-touch surfaces) with daily and terminal cleaning."
    )
    await evaluator.verify(
        claim=claim_clean,
        node=node_clean,
        sources=ca_sources,
        additional_instruction=(
            "Confirm the Answer specifies enhanced environmental cleaning (daily and terminal) emphasizing high-touch surfaces. "
            "Validate consistency with CDC guidance using the provided URLs."
        ),
    )

    # 4) EPA-registered disinfectant effective against C. auris
    node_dis = evaluator.add_leaf(
        id="C_Auris_EPA_Disinfectant",
        desc="Requires use of EPA-registered hospital-grade disinfectant effective against Candida auris",
        parent=ca_node,
        critical=True,
    )
    claim_dis = (
        f"The plan's disinfectant requirement is: '{_safe(cauris.disinfectant_type) if cauris else ''}'. "
        f"This correctly requires an EPA-registered hospital-grade disinfectant effective against Candida auris (e.g., EPA List P)."
    )
    await evaluator.verify(
        claim=claim_dis,
        node=node_dis,
        sources=ca_sources,
        additional_instruction=(
            "Accept statements that require an EPA-registered hospital-grade disinfectant with proven activity against C. auris "
            "(EPA List P is acceptable evidence). Do not accept references to EPA List K (C. difficile) as sufficient. "
            "Validate with CDC/EPA authoritative sources."
        ),
    )

    # 5) Admission screening when to consider
    node_screen = evaluator.add_leaf(
        id="C_Auris_Admission_Screening",
        desc="Describes when admission screening for C. auris should be considered (e.g., patients at high risk including transfers from facilities with known C. auris)",
        parent=ca_node,
        critical=True,
    )
    claim_screen = (
        f"The plan's admission screening criteria for C. auris are stated as: "
        f"'{_safe(cauris.admission_screening) if cauris else ''}'. "
        f"This correctly indicates screening should be considered for higher-risk admissions (e.g., transfers from facilities with known C. auris)."
    )
    await evaluator.verify(
        claim=claim_screen,
        node=node_screen,
        sources=ca_sources,
        additional_instruction=(
            "Check the Answer for explicit criteria such as recent hospitalization in facilities with known C. auris, "
            "mechanical ventilation/tracheostomy, prior colonization/infection, or outbreak involvement. Validate with CDC pages."
        ),
    )

    # 6) Reference URL leaf
    node_c_ref = evaluator.add_leaf(
        id="C_Auris_Reference_URL",
        desc="Provides reference URL to CDC or authoritative source for Candida auris infection control guidelines",
        parent=ca_node,
        critical=True,
    )
    first_c_url = ca_sources[0] if ca_sources else ""
    claim_c_ref = (
        f"The plan includes at least one reference URL to an authoritative C. auris infection control guidance page: "
        f"{ca_sources if ca_sources else []}. "
        f"The URL '{first_c_url}' is an authoritative C. auris guidance page."
    )
    await evaluator.verify(
        claim=claim_c_ref,
        node=node_c_ref,
        sources=ca_sources or None,
        additional_instruction=(
            "Use the Answer to confirm a URL is actually included. The linked page must be authoritative (preferably CDC or EPA). "
            "Mark Incorrect if no URL is present or if the linked content is irrelevant."
        ),
    )


async def verify_general_requirements(
    evaluator: Evaluator,
    parent_node,
    general: Optional[GeneralSection],
    regulatory: Optional[RegulatorySection],
) -> None:
    gen_node = evaluator.add_parallel(
        id="General_Infection_Control_Requirements",
        desc="Core infection prevention and control requirements applicable to all infectious disease responses",
        parent=parent_node,
        critical=True,
    )
    cdc_core_sources = _nonempty_list(regulatory.cdc_core_urls) if regulatory else None
    jc_sources = _nonempty_list(regulatory.jc_urls) if regulatory else None

    # 1) Standard Precautions baseline
    node_std = evaluator.add_leaf(
        id="Standard_Precautions",
        desc="States that standard precautions must be applied to all patient care activities",
        parent=gen_node,
        critical=True,
    )
    claim_std = (
        f"The plan states baseline precautions as: '{_safe(general.baseline_precautions) if general else ''}'. "
        f"This correctly affirms that Standard Precautions apply to all patient care."
    )
    await evaluator.verify(
        claim=claim_std,
        node=node_std,
        sources=cdc_core_sources or None,
        additional_instruction=(
            "Confirm the Answer explicitly states that Standard Precautions apply to all patient care activities. "
            "Validate alignment with CDC Core Infection Prevention and Control Practices if URLs are provided."
        ),
    )

    # 2) Qualified Infection Preventionist oversight
    node_ip = evaluator.add_leaf(
        id="Qualified_Infection_Preventionist",
        desc="Identifies that the facility has or will designate a qualified infection preventionist",
        parent=gen_node,
        critical=True,
    )
    claim_ip = (
        f"The plan's oversight personnel requirement is: '{_safe(general.infection_preventionist_requirement) if general else ''}'. "
        f"This correctly identifies that the facility designates a qualified infection preventionist to oversee the program."
    )
    await evaluator.verify(
        claim=claim_ip,
        node=node_ip,
        sources=jc_sources or cdc_core_sources or None,
        additional_instruction=(
            "Confirm the Answer explicitly designates a qualified infection preventionist (e.g., with appropriate training/certification) "
            "to oversee infection prevention. Validate with Joint Commission standards or CDC Core Practices if URLs are present."
        ),
    )

    # 3) Staff training requirements
    node_train = evaluator.add_leaf(
        id="Staff_Training_Requirement",
        desc="Describes requirement for staff training on infection control protocols and outbreak response procedures",
        parent=gen_node,
        critical=True,
    )
    claim_train = (
        f"The plan's staff training requirements are: '{_safe(general.staff_training) if general else ''}'. "
        f"This correctly requires training on infection control policies and outbreak response."
    )
    await evaluator.verify(
        claim=claim_train,
        node=node_train,
        sources=cdc_core_sources or jc_sources or None,
        additional_instruction=(
            "Confirm the Answer explicitly requires training (e.g., at onboarding and periodically) on infection prevention protocols "
            "and outbreak response procedures. Validate with CDC Core Practices or Joint Commission if URLs are provided."
        ),
    )

    # 4) PPE availability
    node_ppe = evaluator.add_leaf(
        id="PPE_Availability",
        desc="Ensures availability of appropriate personal protective equipment (gloves, gowns, masks, respirators, eye protection)",
        parent=gen_node,
        critical=True,
    )
    claim_ppe = (
        f"The plan's PPE availability statement is: '{_safe(general.ppe_availability) if general else ''}'. "
        f"This ensures appropriate PPE (gloves, gowns, masks/respirators, eye protection) is available for staff."
    )
    await evaluator.verify(
        claim=claim_ppe,
        node=node_ppe,
        sources=cdc_core_sources or None,
        additional_instruction=(
            "Confirm the Answer explicitly lists required PPE availability (at minimum: gloves, gowns, masks/respirators, eye protection) "
            "and ensures access at point of care. Validate with CDC Core Practices if URLs are present."
        ),
    )


async def verify_regulatory_compliance(
    evaluator: Evaluator,
    parent_node,
    regulatory: Optional[RegulatorySection],
) -> None:
    reg_node = evaluator.add_parallel(
        id="Regulatory_Compliance",
        desc="Demonstrates alignment with current regulatory standards and accreditation requirements",
        parent=parent_node,
        critical=True,
    )
    jc_sources = _nonempty_list(regulatory.jc_urls) if regulatory else None
    cdc_core_sources = _nonempty_list(regulatory.cdc_core_urls) if regulatory else None

    # 1) Joint Commission 2024 standards (effective July 1, 2024)
    node_jc = evaluator.add_leaf(
        id="Joint_Commission_2024_Standards",
        desc="References compliance with Joint Commission infection prevention and control standards effective July 1, 2024",
        parent=reg_node,
        critical=True,
    )
    claim_jc = (
        f"The plan references Joint Commission infection prevention and control standards as: "
        f"'{_safe(regulatory.joint_commission_reference) if regulatory else ''}', "
        f"with effective date noted as '{_safe(regulatory.effective_date) if regulatory else ''}'. "
        f"This correctly references compliance with JC standards effective July 1, 2024."
    )
    await evaluator.verify(
        claim=claim_jc,
        node=node_jc,
        sources=jc_sources or None,
        additional_instruction=(
            "Check that the Answer explicitly references Joint Commission infection prevention and control standards and includes "
            "the effective date July 1, 2024. Validate with an official Joint Commission page if a URL is present. "
            "Mark Incorrect if the Answer lacks the reference or the effective date."
        ),
    )

    # 2) CDC Core Infection Prevention and Control Practices
    node_core = evaluator.add_leaf(
        id="CDC_Core_Practices",
        desc="References CDC's Core Infection Prevention and Control Practices for safe healthcare delivery",
        parent=reg_node,
        critical=True,
    )
    claim_core = (
        f"The plan references CDC's Core Infection Prevention and Control Practices as: "
        f"'{_safe(regulatory.cdc_core_practices_reference) if regulatory else ''}'. "
        f"This correctly cites CDC's Core Practices."
    )
    await evaluator.verify(
        claim=claim_core,
        node=node_core,
        sources=cdc_core_sources or None,
        additional_instruction=(
            "Confirm the Answer explicitly references CDC's Core Infection Prevention and Control Practices. "
            "Validate with a CDC Core Practices URL if provided."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an agent's outbreak preparedness plan answer for measles and C. auris readiness,
    general infection control requirements, and regulatory compliance.
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
    extracted_plan = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction",
    )

    # Build the rubric tree root (critical)
    plan_root = evaluator.add_parallel(
        id="Outbreak_Preparedness_Plan",
        desc="A comprehensive hospital outbreak preparedness plan addressing measles and Candida auris response protocols, including isolation measures, healthcare worker protection, environmental controls, and regulatory compliance",
        parent=root,
        critical=True,
    )

    # Measles protocol subtree
    await verify_measles_protocol(
        evaluator=evaluator,
        parent_node=plan_root,
        measles=extracted_plan.measles,
    )

    # Candida auris protocol subtree
    await verify_cauris_protocol(
        evaluator=evaluator,
        parent_node=plan_root,
        cauris=extracted_plan.candida_auris,
    )

    # General requirements subtree
    await verify_general_requirements(
        evaluator=evaluator,
        parent_node=plan_root,
        general=extracted_plan.general,
        regulatory=extracted_plan.regulatory,
    )

    # Regulatory compliance subtree
    await verify_regulatory_compliance(
        evaluator=evaluator,
        parent_node=plan_root,
        regulatory=extracted_plan.regulatory,
    )

    # Return summary
    return evaluator.get_summary()