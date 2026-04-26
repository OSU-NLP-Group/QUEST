import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sacs_r1_2025_single_university"
TASK_DESCRIPTION = (
    "Identify one university located in a state within the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC) accreditation region that holds Carnegie R1 classification (Research 1: Very High Spending and Doctorate Production) as of the 2025 Carnegie Classifications.\n\n"
    "For this university, provide the following information:\n\n"
    "1. Institution Name and State: The full official name of the university and the U.S. state in which it is located.\n"
    "2. Carnegie R1 Verification: Confirm that the institution holds R1 classification and meets the threshold criteria (at least $50 million in annual R&D expenditures and at least 70 research doctorates awarded per year). Include a reference URL from the official Carnegie Classifications website.\n"
    "3. SACSCOC Accreditation: Verify that the institution is currently accredited by SACSCOC. Include a reference URL from the SACSCOC website or the institution's official accreditation page.\n"
    "4. Doctoral Program Credit Requirements: Document the institution's doctoral degree credit requirements, specifically the minimum total credits required for a doctoral degree. Include a reference URL from the institution's official graduate school or doctoral program policy page.\n"
    "5. NSF HERD FY 2024 Data: Report the institution's FY 2024 total research and development expenditures in thousands of dollars, as recorded in the NSF HERD survey database. Include a reference URL from the NSF NCSES HERD database or the institution's research office.\n\n"
    "All information must be supported by official, verifiable sources with direct URL references."
)

# SACSCOC region states: abbreviations and full names for robust matching
SACSCOC_STATE_ABBRS = ["AL", "FL", "GA", "KY", "LA", "MS", "NC", "SC", "TN", "TX", "VA"]
SACSCOC_STATE_FULL = [
    "Alabama", "Florida", "Georgia", "Kentucky", "Louisiana", "Mississippi",
    "North Carolina", "South Carolina", "Tennessee", "Texas", "Virginia"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class InstitutionExtraction(BaseModel):
    """
    Extracted information from the agent's answer for a single institution.
    """
    institution_name: Optional[str] = None
    institution_state: Optional[str] = None

    # Carnegie Classifications
    carnegie_r1_url: Optional[str] = None  # Official Carnegie page for the institution
    carnegie_methodology_url: Optional[str] = None  # Optional: Methodology page describing R1 thresholds

    # SACSCOC accreditation
    sacscoc_url: Optional[str] = None  # SACSCOC site URL or institution's official accreditation page

    # Doctoral program requirements
    doctoral_policy_url: Optional[str] = None
    min_total_credits: Optional[str] = None  # As stated in the answer (string)
    min_course_credits: Optional[str] = None  # Optional (string)
    min_thesis_credits: Optional[str] = None  # Optional (string)

    # NSF HERD FY 2024
    herd_url: Optional[str] = None
    fy2024_expenditure_text: Optional[str] = None  # As stated in answer; keep textual to handle units/format


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_institution() -> str:
    return """
    Extract the following fields for the single university identified in the answer. Return null for any missing field.

    Required fields:
    1. institution_name: The full official name of the university provided in the answer.
    2. institution_state: The U.S. state (either full name like "Georgia" or standard abbreviation like "GA") where the university is located, as provided in the answer.

    Carnegie Classifications:
    3. carnegie_r1_url: A URL to the official Carnegie Classifications page for this institution confirming R1 status. Must be an explicit URL in the answer.
    4. carnegie_methodology_url: (Optional) A URL to the official Carnegie Classifications methodology or definitions page that describes the R1 thresholds (>= $50M R&D and >= 70 research doctorates per year). If present in the answer, extract it; otherwise, return null.

    SACSCOC accreditation:
    5. sacscoc_url: A URL from sacscoc.org (preferred) or the institution's official accreditation page confirming SACSCOC accreditation.

    Doctoral program requirements:
    6. doctoral_policy_url: A URL to the institution's official graduate school or doctoral program policy page that documents credit requirements.
    7. min_total_credits: The minimum total credits required for a doctoral degree at this institution, as stated in the answer (string, do not convert units; e.g., "48 credits", "minimum 60 credits").
    8. min_course_credits: If the answer specifies, extract the minimum graduate-level course credits (string), else null.
    9. min_thesis_credits: If the answer specifies, extract the minimum doctoral thesis/dissertation credits (string), else null.

    NSF HERD FY 2024:
    10. herd_url: A URL either from the NSF NCSES HERD database (ncses.nsf.gov) or the institution's research office confirming the FY 2024 total R&D expenditures.
    11. fy2024_expenditure_text: The FY 2024 total R&D expenditures figure for the institution exactly as stated in the answer (string; keep the units and formatting as presented, e.g., "1,234,567 (thousand dollars)" or "$1.24 billion").

    Rules:
    - Extract URLs only if they are explicitly present in the answer (including markdown links).
    - Do not invent or infer any URL.
    - If any field is missing or not clearly stated, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def non_empty_urls(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip()]


def sacscoc_states_text() -> str:
    ab = ", ".join(SACSCOC_STATE_ABBRS)
    full = ", ".join(SACSCOC_STATE_FULL)
    return f"Abbreviations: {ab}. Full names: {full}."


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_institution_identification(
    evaluator: Evaluator,
    parent_node,
    ex: InstitutionExtraction,
) -> None:
    """
    Build and verify the institution identification subtree:
    - Institution name correctness (grounded by official URLs)
    - Institution state correctness and membership in SACSCOC region
    """
    ident_node = evaluator.add_parallel(
        id="institution_identification",
        desc="Verify that the institution name and state are correctly provided.",
        parent=parent_node,
        critical=True,
    )

    # Institution name existence gate
    name_exists_node = evaluator.add_custom_node(
        result=bool(ex.institution_name and ex.institution_name.strip()),
        id="institution_name_exists",
        desc="Institution name is provided in the answer.",
        parent=ident_node,
        critical=True,
    )

    # Institution name verification (use any strong official URL)
    name_sources = non_empty_urls(ex.carnegie_r1_url, ex.sacscoc_url, ex.doctoral_policy_url)
    name_leaf = evaluator.add_leaf(
        id="institution_name",
        desc="The full official name of the university is correctly provided.",
        parent=ident_node,
        critical=True,
    )
    claim_name = f"The page shows the institution's official name as '{ex.institution_name or ''}'."
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        sources=name_sources if name_sources else None,
        additional_instruction=(
            "Confirm that the page clearly displays the official institutional name in the header or profile section. "
            "Allow minor formatting variants (e.g., presence/absence of 'The', punctuation, or abbreviations) as long as it unambiguously refers to the same institution."
        ),
        extra_prerequisites=[name_exists_node],
    )

    # Institution state subtree (split into existence, match, region membership)
    state_node = evaluator.add_sequential(
        id="institution_state",
        desc="The U.S. state in which the institution is located is correctly identified and is within the SACSCOC region (AL, FL, GA, KY, LA, MS, NC, SC, TN, TX, or VA).",
        parent=ident_node,
        critical=True,
    )

    state_exists_node = evaluator.add_custom_node(
        result=bool(ex.institution_state and ex.institution_state.strip()),
        id="institution_state_exists",
        desc="Institution state is provided in the answer.",
        parent=state_node,
        critical=True,
    )

    state_sources = non_empty_urls(ex.sacscoc_url, ex.carnegie_r1_url)
    state_match_leaf = evaluator.add_leaf(
        id="institution_state_match",
        desc="The institution's location state matches the provided state.",
        parent=state_node,
        critical=True,
    )
    claim_state = f"The institution is located in the state of '{ex.institution_state or ''}'."
    await evaluator.verify(
        claim=claim_state,
        node=state_match_leaf,
        sources=state_sources if state_sources else None,
        additional_instruction=(
            "Look for the institution's address or location details on the page (e.g., 'City, ST' or full state name). "
            "Accept either the state abbreviation or full state name as a match."
        ),
        extra_prerequisites=[state_exists_node],
    )

    # Is the state in SACSCOC region?
    state_region_leaf = evaluator.add_leaf(
        id="institution_state_in_region",
        desc="The provided state is within the SACSCOC region.",
        parent=state_node,
        critical=True,
    )
    claim_region = (
        f"The state '{ex.institution_state or ''}' is within the SACSCOC region "
        f"(AL, FL, GA, KY, LA, MS, NC, SC, TN, TX, VA)."
    )
    await evaluator.verify(
        claim=claim_region,
        node=state_region_leaf,
        sources=None,  # Logical/membership check; no URL needed
        additional_instruction=(
            "This is a simple membership check. Accept either state abbreviations or full names. "
            f"Reference set: {sacscoc_states_text()}"
        ),
        extra_prerequisites=[state_exists_node],
    )


async def verify_r1_classification(
    evaluator: Evaluator,
    parent_node,
    ex: InstitutionExtraction,
) -> None:
    """
    Build and verify the Carnegie R1 classification subtree:
    - Confirm R1 status via official Carnegie page
    - Confirm thresholds logically (>= $50M R&D and >= 70 doctorates) given R1 classification
    """
    r1_node = evaluator.add_sequential(
        id="carnegie_r1_verification",
        desc="Verify that the identified institution holds Carnegie R1 classification (Research 1: Very High Spending and Doctorate Production) as of the 2025 Carnegie Classifications.",
        parent=parent_node,
        critical=True,
    )

    compliance_node = evaluator.add_parallel(
        id="r1_threshold_compliance",
        desc="Confirm that the institution meets the R1 threshold criteria: at least $50 million in annual research & development expenditures and at least 70 research doctorates awarded per year.",
        parent=r1_node,
        critical=True,
    )

    # Existence of Carnegie URL gate
    r1_url_exists = evaluator.add_custom_node(
        result=bool(ex.carnegie_r1_url and ex.carnegie_r1_url.strip()),
        id="carnegie_r1_url_exists",
        desc="A Carnegie Classifications URL confirming R1 status is provided.",
        parent=compliance_node,
        critical=True,
    )

    # R1 classification confirmation using Carnegie page
    r1_status_leaf = evaluator.add_leaf(
        id="carnegie_classification_url",
        desc="Provide a reference URL from the official Carnegie Classifications website confirming the institution's R1 status.",
        parent=compliance_node,
        critical=True,
    )
    claim_r1 = (
        f"The official Carnegie Classifications page for '{ex.institution_name or 'the institution'}' confirms "
        "R1 classification (Research 1: Very High Spending and Doctorate Production) for the 2025 classifications."
    )
    await evaluator.verify(
        claim=claim_r1,
        node=r1_status_leaf,
        sources=ex.carnegie_r1_url,
        additional_instruction=(
            "Verify that the page explicitly indicates the institution's classification as 'R1' or "
            "equivalent wording such as 'R1: Very High Spending and Doctorate Production'. "
            "Minor formatting variations are acceptable."
        ),
        extra_prerequisites=[r1_url_exists],
    )

    # Expenditure threshold (logical inference given R1 status)
    r1_exp_leaf = evaluator.add_leaf(
        id="research_expenditure_threshold",
        desc="The institution spends at least $50 million annually on research & development.",
        parent=compliance_node,
        critical=True,
    )
    claim_exp = (
        "Given the institution is classified as R1 under the 2025 Carnegie Classifications, "
        "it therefore meets the >= $50 million annual R&D expenditure threshold required for R1."
    )
    await evaluator.verify(
        claim=claim_exp,
        node=r1_exp_leaf,
        sources=None,  # Logical verification; rely on R1 status as precondition
        additional_instruction=(
            "Treat this as a logical verification: the R1 category explicitly requires >= $50M R&D expenditures. "
            "Since R1 status was confirmed in a preceding check, this threshold is necessarily satisfied. "
            "You do not need to find the numeric amount on the page."
        ),
        extra_prerequisites=[r1_status_leaf, r1_url_exists],
    )

    # Doctorate production threshold (logical inference given R1 status)
    r1_doc_leaf = evaluator.add_leaf(
        id="doctorate_production_threshold",
        desc="The institution awards at least 70 research doctorates per year.",
        parent=compliance_node,
        critical=True,
    )
    claim_doc = (
        "Given the institution is classified as R1 under the 2025 Carnegie Classifications, "
        "it therefore meets the >= 70 research doctorates per year threshold required for R1."
    )
    await evaluator.verify(
        claim=claim_doc,
        node=r1_doc_leaf,
        sources=None,  # Logical verification; rely on R1 status as precondition
        additional_instruction=(
            "Treat this as a logical verification: the R1 category explicitly requires >= 70 research doctorates per year. "
            "Since R1 status was confirmed in a preceding check, this threshold is necessarily satisfied."
        ),
        extra_prerequisites=[r1_status_leaf, r1_url_exists],
    )


async def verify_sacscoc_accreditation(
    evaluator: Evaluator,
    parent_node,
    ex: InstitutionExtraction,
) -> None:
    """
    Build and verify the SACSCOC accreditation subtree:
    - Confirm current SACSCOC accreditation
    - Confirm the URL is from sacscoc.org or official institutional accreditation page
    """
    accred_node = evaluator.add_parallel(
        id="regional_accreditation_verification",
        desc="Verify that the institution is accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC).",
        parent=parent_node,
        critical=True,
    )

    sacscoc_url_exists = evaluator.add_custom_node(
        result=bool(ex.sacscoc_url and ex.sacscoc_url.strip()),
        id="sacscoc_url_exists",
        desc="A SACSCOC or official institution accreditation URL is provided.",
        parent=accred_node,
        critical=True,
    )

    status_leaf = evaluator.add_leaf(
        id="sacscoc_accreditation_status",
        desc="The institution holds current accreditation from SACSCOC.",
        parent=accred_node,
        critical=True,
    )
    claim_status = (
        f"The institution '{ex.institution_name or ''}' is accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC) "
        "and the page indicates current accreditation status."
    )
    await evaluator.verify(
        claim=claim_status,
        node=status_leaf,
        sources=ex.sacscoc_url,
        additional_instruction=(
            "Accept either: (1) a listing or profile page on sacscoc.org confirming accreditation status, or "
            "(2) the institution's official accreditation page that explicitly states accreditation by SACSCOC."
        ),
        extra_prerequisites=[sacscoc_url_exists],
    )

    refurl_leaf = evaluator.add_leaf(
        id="sacscoc_reference_url",
        desc="Provide a reference URL from the SACSCOC website or the institution's official accreditation page confirming SACSCOC accreditation.",
        parent=accred_node,
        critical=True,
    )
    claim_refurl = (
        "This accreditation reference URL is either on the sacscoc.org domain or on the institution's official domain "
        "and constitutes an official accreditation page."
    )
    await evaluator.verify(
        claim=claim_refurl,
        node=refurl_leaf,
        sources=ex.sacscoc_url,
        additional_instruction=(
            "Check the URL shown in the prompt: "
            "- If it is sacscoc.org, accept. "
            "- If it is the institution's official domain (e.g., .edu) and the page is explicitly an accreditation page, accept. "
            "- Otherwise, do not accept."
        ),
        extra_prerequisites=[sacscoc_url_exists],
    )


async def verify_doctoral_requirements(
    evaluator: Evaluator,
    parent_node,
    ex: InstitutionExtraction,
) -> None:
    """
    Build and verify the doctoral program requirements subtree:
    - Minimum credit requirement check (48 total credits with at least 24 course + 24 thesis/dissertation credits, or documented exception)
    - Confirm the URL is a proper graduate school or doctoral policy page
    """
    doc_node = evaluator.add_parallel(
        id="doctoral_program_requirements",
        desc="Verify the institution's doctoral degree credit requirements structure.",
        parent=parent_node,
        critical=True,
    )

    doctoral_url_exists = evaluator.add_custom_node(
        result=bool(ex.doctoral_policy_url and ex.doctoral_policy_url.strip()),
        id="doctoral_policy_url_exists",
        desc="A doctoral/graduate policy URL is provided.",
        parent=doc_node,
        critical=True,
    )

    min_credit_leaf = evaluator.add_leaf(
        id="minimum_credit_requirement",
        desc="The institution's doctoral programs require a minimum of 48 total credits (consisting of at least 24 graduate-level course credits and at least 24 doctoral thesis credits), or a documented exception to this requirement has been granted.",
        parent=doc_node,
        critical=True,
    )
    claim_min_credit = (
        "The institution's official doctoral/graduate policies document that doctoral degrees require at least 48 total credits, "
        "including at least 24 course credits and at least 24 thesis/dissertation credits; "
        "alternatively, the page explicitly documents an exception to this minimum requirement."
    )
    await evaluator.verify(
        claim=claim_min_credit,
        node=min_credit_leaf,
        sources=ex.doctoral_policy_url,
        additional_instruction=(
            "Look for explicit minimum credit requirements and distribution across coursework and thesis/dissertation credits. "
            "Accept synonyms like 'dissertation research' or 'thesis research'. "
            "If an exception policy is clearly documented, that also satisfies this requirement."
        ),
        extra_prerequisites=[doctoral_url_exists],
    )

    docurl_leaf = evaluator.add_leaf(
        id="doctoral_program_url",
        desc="Provide a reference URL from the institution's official graduate school or doctoral program policy page documenting the credit requirements.",
        parent=doc_node,
        critical=True,
    )
    claim_docurl = (
        "This URL is the institution's official graduate school or doctoral program policy page that documents credit requirements."
    )
    await evaluator.verify(
        claim=claim_docurl,
        node=docurl_leaf,
        sources=ex.doctoral_policy_url,
        additional_instruction=(
            "Confirm that this page belongs to the institution's official domain (e.g., .edu) "
            "and explicitly discusses doctoral credit requirements/policies."
        ),
        extra_prerequisites=[doctoral_url_exists],
    )


async def verify_herd_data(
    evaluator: Evaluator,
    parent_node,
    ex: InstitutionExtraction,
) -> None:
    """
    Build and verify the HERD expenditure subtree:
    - Verify FY 2024 total R&D expenditures figure (as stated in the answer)
    - Confirm the URL source is acceptable (NSF NCSES HERD database or institution's research office)
    """
    herd_node = evaluator.add_parallel(
        id="herd_expenditure_data",
        desc="Extract and verify the institution's FY 2024 total research and development expenditures from the NSF HERD survey database.",
        parent=parent_node,
        critical=True,
    )

    herd_url_exists = evaluator.add_custom_node(
        result=bool(ex.herd_url and ex.herd_url.strip()),
        id="herd_url_exists",
        desc="An NSF NCSES HERD database URL or official institution research office URL is provided.",
        parent=herd_node,
        critical=True,
    )

    amount_leaf = evaluator.add_leaf(
        id="fy2024_expenditure_amount",
        desc="The institution's FY 2024 total R&D expenditures (in thousands of dollars) are correctly reported from the NSF HERD survey data.",
        parent=herd_node,
        critical=True,
    )
    claim_amount = (
        f"The FY 2024 total R&D expenditures for '{ex.institution_name or ''}' are reported as "
        f"'{ex.fy2024_expenditure_text or ''}' on this page."
    )
    await evaluator.verify(
        claim=claim_amount,
        node=amount_leaf,
        sources=ex.herd_url,
        additional_instruction=(
            "Confirm the FY 2024 total R&D expenditures figure for the institution as shown on the page. "
            "The answer's figure may be expressed in thousands, millions, or full dollars; "
            "consider reasonable unit conversions or rounding as equivalent."
        ),
        extra_prerequisites=[herd_url_exists],
    )

    herdurl_leaf = evaluator.add_leaf(
        id="herd_data_url",
        desc="Provide a reference URL from the NSF NCSES HERD database or the institution's research office confirming the FY 2024 expenditure figure.",
        parent=herd_node,
        critical=True,
    )
    claim_herdurl = (
        "This URL is either from the NSF NCSES HERD database (ncses.nsf.gov) or from the institution's official research office "
        "and it confirms the FY 2024 expenditure figure."
    )
    await evaluator.verify(
        claim=claim_herdurl,
        node=herdurl_leaf,
        sources=ex.herd_url,
        additional_instruction=(
            "Check the URL shown in the prompt: "
            "- If it is ncses.nsf.gov, accept. "
            "- If it is the institution's official domain (e.g., .edu) and clearly confirms the FY 2024 HERD figure, accept. "
            "- Otherwise, do not accept."
        ),
        extra_prerequisites=[herd_url_exists],
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
    """
    Evaluate an answer for the SACSCOC-region R1 university verification task.
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

    # Create a critical task root under the evaluator's non-critical root to enforce strict gating.
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Identify a U.S. university located in a SACSCOC-accredited region (AL, FL, GA, KY, LA, MS, NC, SC, TN, TX, or VA) that holds Carnegie R1 classification and verify all required information about the institution.",
        parent=root,
        critical=True,
    )

    # Extraction
    ex = await evaluator.extract(
        prompt=prompt_extract_institution(),
        template_class=InstitutionExtraction,
        extraction_name="institution_extraction",
    )

    # Build verification subtrees
    await verify_institution_identification(evaluator, task_root, ex)
    await verify_r1_classification(evaluator, task_root, ex)
    await verify_sacscoc_accreditation(evaluator, task_root, ex)
    await verify_doctoral_requirements(evaluator, task_root, ex)
    await verify_herd_data(evaluator, task_root, ex)

    # Return summary
    return evaluator.get_summary()