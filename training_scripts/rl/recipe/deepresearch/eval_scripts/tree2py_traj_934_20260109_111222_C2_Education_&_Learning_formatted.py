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
TASK_ID = "cmhc_online_cacrep_eval"
TASK_DESCRIPTION = (
    "Identify one online master's degree program in Clinical Mental Health Counseling that meets all of the following requirements: "
    "(1) The program must be accredited by the Council for Accreditation of Counseling and Related Educational Programs (CACREP); "
    "(2) The institution offering the program must be accredited by one of the four U.S. regional accrediting organizations recognized by "
    "the Council for Higher Education Accreditation (CHEA); "
    "(3) The program must require between 30 and 60 credit hours for degree completion; "
    "(4) The program must require a minimum of 700 supervised clinical hours, consistent with CACREP standards for practicum and internship; "
    "(5) The program must be available fully online; "
    "(6) The program must have a minimum undergraduate GPA requirement of 3.0 or lower for admission; and "
    "(7) The program must not require GRE scores for admission. "
    "Provide the name of the institution, the specific program name, the total credit hours required, confirmation of the clinical hours requirement, "
    "the minimum GPA requirement, confirmation that GRE is not required, and direct URL references to both the program's official CACREP accreditation status "
    "and the program's admission requirements page."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramExtraction(BaseModel):
    # Identity
    institution_name: Optional[str] = None
    program_name: Optional[str] = None
    program_page_url: Optional[str] = None

    # Accreditation
    cacrep_status_url: Optional[str] = None
    institutional_accreditation_url: Optional[str] = None
    regional_accreditor_name: Optional[str] = None

    # Structure & delivery
    total_credit_hours: Optional[str] = None
    supervised_clinical_hours: Optional[str] = None
    practicum_hours: Optional[str] = None
    internship_hours: Optional[str] = None
    delivery_mode_statement: Optional[str] = None  # e.g., "fully online", "100% online"

    # Admissions
    admissions_requirements_url: Optional[str] = None
    min_undergrad_gpa_requirement: Optional[str] = None
    gre_policy_statement: Optional[str] = None  # e.g., "No GRE required", "GRE not required", "GRE optional"

    # Other URLs present in the answer
    other_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program() -> str:
    return """
    Extract structured information for one online master's program in Clinical Mental Health Counseling as presented in the answer.

    Required fields to extract:

    Identity:
    - institution_name: The name of the institution offering the program.
    - program_name: The specific program name (e.g., "M.A./M.S. in Clinical Mental Health Counseling").
    - program_page_url: The official program page URL, if explicitly provided.

    Accreditation:
    - cacrep_status_url: A direct URL to the program’s CACREP accreditation status page (e.g., CACREP directory entry). Extract only if explicitly present.
    - institutional_accreditation_url: A direct URL that states the institution’s regional accreditation, if explicitly present (e.g., HLC, SACSCOC, NWCCU, ACCJC).
    - regional_accreditor_name: The accrediting body name mentioned in the answer (e.g., "HLC", "SACSCOC"), if explicitly present.

    Structure & delivery:
    - total_credit_hours: The total credit hours required for degree completion (as written in the answer; may be a number or text).
    - supervised_clinical_hours: The total supervised clinical hours mentioned (e.g., "700+"), if explicitly present.
    - practicum_hours: Practicum hours (e.g., "100 hours"), if explicitly present.
    - internship_hours: Internship hours (e.g., "600 hours"), if explicitly present.
    - delivery_mode_statement: The statement in the answer about online availability (e.g., "fully online", "100% online"), if explicitly present.

    Admissions:
    - admissions_requirements_url: A direct URL to the program’s admissions requirements page that shows GPA and GRE policy, if explicitly present.
    - min_undergrad_gpa_requirement: The stated minimum undergraduate GPA requirement for admission (as written in the answer).
    - gre_policy_statement: The stated GRE policy (e.g., "GRE not required", "GRE optional").

    Other:
    - other_urls: List ALL other URLs explicitly mentioned in the answer that are relevant to this program.

    RULES:
    - Do not invent or infer any URLs; extract only explicit URLs present in the answer.
    - For missing fields, return null (or empty list for other_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm_str(s: Optional[str]) -> Optional[str]:
    return s.strip() if isinstance(s, str) and s.strip() else None


def _dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _program_sources(extracted: ProgramExtraction) -> List[str]:
    urls = []
    if extracted.program_page_url:
        urls.append(extracted.program_page_url)
    if extracted.admissions_requirements_url:
        urls.append(extracted.admissions_requirements_url)
    urls.extend(extracted.other_urls or [])
    return _dedupe_preserve_order(urls)


def _admissions_sources(extracted: ProgramExtraction) -> List[str]:
    urls = []
    if extracted.admissions_requirements_url:
        urls.append(extracted.admissions_requirements_url)
    urls.extend(extracted.other_urls or [])
    # Also include program page if admissions not explicit; it sometimes lists admissions info
    if extracted.program_page_url:
        urls.append(extracted.program_page_url)
    return _dedupe_preserve_order(urls)


def _institution_accreditation_sources(extracted: ProgramExtraction) -> List[str]:
    urls = []
    if extracted.institutional_accreditation_url:
        urls.append(extracted.institutional_accreditation_url)
    # Fall back to admissions/program pages which sometimes note institutional accreditation
    if extracted.admissions_requirements_url:
        urls.append(extracted.admissions_requirements_url)
    if extracted.program_page_url:
        urls.append(extracted.program_page_url)
    urls.extend(extracted.other_urls or [])
    return _dedupe_preserve_order(urls)


# --------------------------------------------------------------------------- #
# Verification tree construction & checks                                     #
# --------------------------------------------------------------------------- #
async def build_and_verify_cmhc_program(
    evaluator: Evaluator,
    root_node,
    extracted: ProgramExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run verification checks.
    All critical nodes are enforced to satisfy mandatory constraints.
    """

    # Top-level critical parallel node
    program_root = evaluator.add_parallel(
        id="Program_Identification_and_Validation",
        desc="Identify one fully online master's program in Clinical Mental Health Counseling and provide the required details, ensuring all stated constraints are satisfied.",
        parent=root_node,
        critical=True,
    )

    # -------------------- Program Identity Provided -------------------- #
    identity_node = evaluator.add_parallel(
        id="Program_Identity_Provided",
        desc="Response provides the required program identity fields.",
        parent=program_root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_norm_str(extracted.institution_name) is not None,
        id="Institution_Name_Provided",
        desc="Provide the name of the institution offering the program.",
        parent=identity_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_norm_str(extracted.program_name) is not None,
        id="Program_Name_Provided",
        desc="Provide the specific program name (master’s in Clinical Mental Health Counseling).",
        parent=identity_node,
        critical=True,
    )

    # -------------------- Accreditation Requirements ------------------- #
    accred_node = evaluator.add_parallel(
        id="Accreditation_Requirements",
        desc="Program and institution satisfy accreditation constraints and required CACREP status link is provided.",
        parent=program_root,
        critical=True,
    )

    # CACREP status URL must be provided (critical)
    evaluator.add_custom_node(
        result=_norm_str(extracted.cacrep_status_url) is not None,
        id="CACREP_Status_URL_Provided",
        desc="Provide a direct URL reference to the program’s official CACREP accreditation status (e.g., CACREP directory entry).",
        parent=accred_node,
        critical=True,
    )

    # CACREP accredited check via CACREP URL
    cacrep_leaf = evaluator.add_leaf(
        id="CACREP_Accredited",
        desc="The program is accredited by CACREP.",
        parent=accred_node,
        critical=True,
    )
    inst_name = _norm_str(extracted.institution_name) or "the institution"
    prog_name = _norm_str(extracted.program_name) or "the Clinical Mental Health Counseling program"
    cacrep_claim = (
        f"The program '{prog_name}' at '{inst_name}' is accredited by CACREP, as shown on the CACREP directory/status page."
    )
    await evaluator.verify(
        claim=cacrep_claim,
        node=cacrep_leaf,
        sources=_norm_str(extracted.cacrep_status_url),
        additional_instruction=(
            "Check the CACREP directory/status page to confirm the program is listed and accredited. "
            "Allow reasonable naming variants (e.g., MA/MS in Clinical Mental Health Counseling). "
            "If the page shows the program name and institution with accredited status, consider it supported."
        ),
    )

    # Institution regional accreditation by one of the four recognized bodies
    inst_reg_leaf = evaluator.add_leaf(
        id="Institution_Regional_Accreditation",
        desc="The institution is accredited by one of: HLC, SACSCOC, NWCCU, or ACCJC.",
        parent=accred_node,
        critical=True,
    )
    accreditor = _norm_str(extracted.regional_accreditor_name)
    if accreditor:
        inst_reg_claim = (
            f"The institution '{inst_name}' is accredited by {accreditor}, which is one of HLC, SACSCOC, NWCCU, or ACCJC."
        )
    else:
        inst_reg_claim = (
            f"The institution '{inst_name}' is accredited by one of HLC, SACSCOC, NWCCU, or ACCJC."
        )
    await evaluator.verify(
        claim=inst_reg_claim,
        node=inst_reg_leaf,
        sources=_institution_accreditation_sources(extracted),
        additional_instruction=(
            "Verify the page indicates institutional accreditation by HLC (Higher Learning Commission), "
            "SACSCOC (Southern Association of Colleges and Schools Commission on Colleges), "
            "NWCCU (Northwest Commission on Colleges and Universities), or "
            "ACCJC (Accrediting Commission for Community and Junior Colleges). "
            "Accept official institutional pages or accreditor listings. Allow common abbreviations."
        ),
    )

    # -------------------- Program Structure and Delivery ---------------- #
    struct_node = evaluator.add_parallel(
        id="Program_Structure_and_Delivery",
        desc="Program meets structural and delivery constraints and the required values/confirmations are provided.",
        parent=program_root,
        critical=True,
    )

    # Break "Credit_Hours_Stated_and_In_Range" into two critical leaves under a critical sub-node
    credit_node = evaluator.add_parallel(
        id="Credit_Hours_Stated_and_In_Range",
        desc="State the total credit hours required, and it is between 30 and 60 credit hours.",
        parent=struct_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_norm_str(extracted.total_credit_hours) is not None,
        id="Credit_Hours_Provided",
        desc="Total credit hours value is stated in the answer.",
        parent=credit_node,
        critical=True,
    )
    credit_range_leaf = evaluator.add_leaf(
        id="Credit_Hours_In_Range_Verified",
        desc="Total credit hours is supported by sources and falls between 30 and 60 (inclusive).",
        parent=credit_node,
        critical=True,
    )
    credit_val = _norm_str(extracted.total_credit_hours) or "unknown"
    credit_claim = (
        f"The program requires {credit_val} total credit hours for degree completion, and this value is between 30 and 60 (inclusive)."
    )
    await evaluator.verify(
        claim=credit_claim,
        node=credit_range_leaf,
        sources=_program_sources(extracted),
        additional_instruction=(
            "Find the total credit hours on the referenced page(s); confirm the specific number matches the claim and "
            "judge whether it falls within 30–60 (inclusive). If a range is given, ensure it lies entirely within 30–60."
        ),
    )

    # Break "Clinical_Hours_Stated_and_Meet_Minimum" into two critical leaves under a critical sub-node
    clinical_node = evaluator.add_parallel(
        id="Clinical_Hours_Stated_and_Meet_Minimum",
        desc="Confirm the supervised clinical hours requirement, and it is at least 700 hours (100 practicum + 600 internship) consistent with CACREP standards.",
        parent=struct_node,
        critical=True,
    )
    clinical_exists = bool(_norm_str(extracted.supervised_clinical_hours)) or (
        bool(_norm_str(extracted.practicum_hours)) and bool(_norm_str(extracted.internship_hours))
    )
    evaluator.add_custom_node(
        result=clinical_exists,
        id="Clinical_Hours_Provided",
        desc="Clinical hours requirement is stated (either total supervised hours or breakdown of practicum + internship).",
        parent=clinical_node,
        critical=True,
    )
    clinical_verify_leaf = evaluator.add_leaf(
        id="Clinical_Hours_Minimum_Verified",
        desc="Supervised clinical hours meet minimum of 700 (≥100 practicum + ≥600 internship).",
        parent=clinical_node,
        critical=True,
    )
    clinical_claim = (
        "The program requires at least 700 supervised clinical hours, consisting of at least 100 practicum hours and at least 600 internship hours (consistent with CACREP standards)."
    )
    await evaluator.verify(
        claim=clinical_claim,
        node=clinical_verify_leaf,
        sources=_program_sources(extracted),
        additional_instruction=(
            "Check the program/handbook/admissions page for practicum and internship hour requirements. "
            "Accept equivalent phrasing (e.g., 'minimum 100 practicum hours' and 'minimum 600 internship hours'). "
            "If total ≥700 is explicitly stated, that also qualifies."
        ),
    )

    # Fully Online check
    online_leaf = evaluator.add_leaf(
        id="Fully_Online",
        desc="Program is available fully online.",
        parent=struct_node,
        critical=True,
    )
    online_claim = "The program is available fully online (100% online)."
    await evaluator.verify(
        claim=online_claim,
        node=online_leaf,
        sources=_program_sources(extracted),
        additional_instruction=(
            "Confirm the referenced page indicates the program is fully online (e.g., '100% online', 'fully online'). "
            "If the page states hybrid or campus-residency is required, this should not pass."
        ),
    )

    # -------------------- Admissions Requirements & References ---------- #
    adm_node = evaluator.add_parallel(
        id="Admissions_Requirements_and_References",
        desc="Admissions constraints are satisfied and the admissions requirements URL is provided.",
        parent=program_root,
        critical=True,
    )

    # Admissions requirements URL must be provided (critical)
    evaluator.add_custom_node(
        result=_norm_str(extracted.admissions_requirements_url) is not None,
        id="Admissions_Requirements_URL_Provided",
        desc="Provide a direct URL reference to the program’s official admissions requirements page showing GPA and GRE policy.",
        parent=adm_node,
        critical=True,
    )

    # GPA: break into stated & eligible leaves under a critical sub-node
    gpa_node = evaluator.add_parallel(
        id="Minimum_GPA_Stated_and_Eligible",
        desc="State the minimum undergraduate GPA requirement, and it is 3.0 or lower.",
        parent=adm_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_norm_str(extracted.min_undergrad_gpa_requirement) is not None,
        id="Minimum_GPA_Provided",
        desc="Minimum undergraduate GPA requirement is stated in the answer.",
        parent=gpa_node,
        critical=True,
    )
    gpa_leaf = evaluator.add_leaf(
        id="Minimum_GPA_Eligible_Verified",
        desc="Admissions page indicates minimum undergraduate GPA is 3.0 or lower.",
        parent=gpa_node,
        critical=True,
    )
    gpa_claim = "The program’s admissions page indicates the minimum undergraduate GPA required for admission is 3.0 or lower."
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_leaf,
        sources=_admissions_sources(extracted),
        additional_instruction=(
            "Check the admissions page for the minimum undergraduate GPA requirement. "
            "Treat 'minimum GPA 3.0' as eligible. Any threshold above 3.0 (e.g., 3.2, 3.5) should not pass."
        ),
    )

    # GRE not required (critical leaf)
    gre_leaf = evaluator.add_leaf(
        id="GRE_Not_Required_Confirmed",
        desc="Confirm that GRE scores are not required for admission.",
        parent=adm_node,
        critical=True,
    )
    gre_claim = "GRE scores are not required for admission to this program."
    await evaluator.verify(
        claim=gre_claim,
        node=gre_leaf,
        sources=_admissions_sources(extracted),
        additional_instruction=(
            "Look for policy text such as 'GRE not required' or 'GRE optional'. "
            "If GRE is optional (not mandatory), consider 'not required' satisfied."
        ),
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the CMHC online CACREP program identification task.
    """
    # Initialize evaluator with a parallel root (we will add our own critical root under it)
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

    # Extract program information from the answer
    extracted_program = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_cmhc_program(evaluator, root, extracted_program)

    # Return structured summary
    return evaluator.get_summary()