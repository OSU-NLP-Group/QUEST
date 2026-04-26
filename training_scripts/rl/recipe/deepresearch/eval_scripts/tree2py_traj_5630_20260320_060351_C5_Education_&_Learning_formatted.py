import asyncio
import logging
import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ks_dual_credit_psychology_pd"
TASK_DESCRIPTION = (
    "A Kansas high school teacher with a master's degree in Education wants to become qualified to teach dual credit "
    "Psychology courses at their school. According to Higher Learning Commission (HLC) standards adopted by Kansas "
    "institutions, what is the minimum number of graduate credit hours in Psychology required for someone with a master's "
    "degree in a different field to qualify as a dual credit instructor? Identify one specific Kansas public university "
    "that offers an 18-credit hour graduate certificate program designed for dual credit instructor credentialing in "
    "Psychology or Psychology of Learning. Calculate how many professional development (PD) points this 18-credit hour "
    "program would generate using the Kansas conversion standard (1 college credit = 20 PD points), and determine whether "
    "this number of PD points would satisfy the Kansas teaching license renewal requirement for teachers who hold a "
    "graduate degree (which requires 120 PD points). Provide reference URLs from both the Kansas Board of Regents dual "
    "credit faculty qualifications page and the identified university's program page."
)

ALLOWED_KS_PUBLIC_UNIS = [
    "Emporia State University",
    "Fort Hays State University",
    "Pittsburg State University",
]

PD_POINTS_PER_CREDIT = 20
KS_RENEWAL_PD_REQUIREMENT_WITH_GRAD = 120


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CredentialingExtraction(BaseModel):
    # HLC / KBOR
    hlc_standard_statement: Optional[str] = None  # e.g., "master's in discipline OR master's + 18 graduate hours"
    min_hours: Optional[str] = None               # e.g., "18"
    kbor_url: Optional[str] = None                # URL to KBOR dual credit (concurrent) faculty qualifications page

    # University program
    university_name: Optional[str] = None         # e.g., "Fort Hays State University"
    program_name: Optional[str] = None            # e.g., "Graduate Certificate in Psychology (Dual Credit Credentialing)"
    program_url: Optional[str] = None             # program page URL
    program_credits: Optional[str] = None         # e.g., "18", "18 credit hours", "18 hours"
    program_credential_type: Optional[str] = None # e.g., "Graduate Certificate"
    program_discipline: Optional[str] = None      # e.g., "Psychology", "Psychology of Learning"
    dual_credit_language: Optional[str] = None    # any text indicating "dual credit" / "concurrent enrollment" intent

    # PD calculation & renewal conclusion as stated by the answer (if provided)
    pd_points: Optional[str] = None               # e.g., "360"
    renewal_requirement_points: Optional[str] = None  # e.g., "120"
    renewal_conclusion: Optional[str] = None      # e.g., "Yes, it meets the requirement"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_credentialing() -> str:
    return """
    Extract the following fields exactly as stated in the answer. Use null when missing.

    Required fields:
    - hlc_standard_statement: The policy-language or paraphrase the answer gives for the HLC faculty qualification standard for dual/concurrent enrollment (e.g., "master's in the discipline OR master's + 18 graduate hours in the discipline").
    - min_hours: The minimum number of graduate credit hours in the discipline (string, usually "18").
    - kbor_url: The URL for the Kansas Board of Regents page about dual credit/concurrent enrollment faculty qualifications. Must be a valid URL explicitly present in the answer.

    University/program fields (for one specific Kansas public university: Emporia State University, Fort Hays State University, or Pittsburg State University):
    - university_name: The identified university's name as written in the answer (string).
    - program_name: The exact name of the graduate certificate program (string).
    - program_url: The URL for that program page (string URL).
    - program_credits: How many credit hours the certificate requires (string, e.g., "18", "18 hours", "18 credit hours").
    - program_credential_type: The credential type (string, e.g., "Graduate Certificate").
    - program_discipline: The discipline of the program (string, e.g., "Psychology" or "Psychology of Learning").
    - dual_credit_language: Any phrase from the answer that indicates the program is intended for dual credit/concurrent enrollment instructor credentialing (string; null if not provided).

    PD calculation fields as stated by the answer (if present):
    - pd_points: The total PD points computed in the answer (string, like "360").
    - renewal_requirement_points: The renewal PD requirement mentioned in the answer for teachers with a graduate degree (string, like "120").
    - renewal_conclusion: Whether the computed PD points meet the requirement (string like "yes", "meets", or "no").

    Notes:
    - Extract ONLY what appears explicitly in the answer.
    - Do not infer or fabricate any URLs or numbers.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def first_int_in_text(text: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if not text:
        return default
    m = re.search(r"\d+", text)
    if not m:
        return default
    try:
        return int(m.group(0))
    except Exception:
        return default


def normalize_boolish(text: Optional[str]) -> Optional[bool]:
    if text is None:
        return None
    t = text.strip().lower()
    if t in {"yes", "true", "meets", "satisfies", "y"}:
        return True
    if t in {"no", "false", "does not meet", "n"}:
        return False
    return None


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_dual_credit_requirements(
    evaluator: Evaluator,
    parent,
    ex: CredentialingExtraction
):
    node = evaluator.add_parallel(
        id="Dual_Credit_Qualification_Requirements",
        desc="Identification of the credential standard required for teaching dual credit courses in Kansas",
        parent=parent,
        critical=False,
    )

    # HLC standard citation — verify against KBOR page
    hlc_node = evaluator.add_leaf(
        id="HLC_Standard_Citation",
        desc="HLC requires a master's in the teaching discipline OR a master's in any field plus 18 graduate credit hours in the discipline",
        parent=node,
        critical=True,
    )
    if not ex.kbor_url:
        hlc_node.score = 0.0
        hlc_node.status = "failed"
    else:
        claim = (
            "The Kansas Board of Regents (KBOR) faculty qualifications page states the Higher Learning Commission (HLC) "
            "standard for dual/concurrent enrollment instructors as: either a master's degree in the teaching discipline "
            "OR any master's degree plus at least 18 graduate credit hours in the teaching discipline."
        )
        await evaluator.verify(
            claim=claim,
            node=hlc_node,
            sources=ex.kbor_url,
            additional_instruction="Look for HLC language about master's in discipline or master's + 18 graduate hours (concurrent enrollment/dual credit).",
        )

    # Minimum graduate credit hours — verify against KBOR page
    min_hours_node = evaluator.add_leaf(
        id="Minimum_Graduate_Credit_Hours",
        desc="States that 18 graduate credit hours are required in the discipline when master's is in a different field",
        parent=node,
        critical=True,
    )
    if not ex.kbor_url:
        min_hours_node.score = 0.0
        min_hours_node.status = "failed"
    else:
        claim = "The minimum number of graduate credit hours in the teaching discipline required by the HLC standard is 18."
        await evaluator.verify(
            claim=claim,
            node=min_hours_node,
            sources=ex.kbor_url,
            additional_instruction="Confirm the number '18' graduate credit hours appears as the threshold.",
        )


async def build_graduate_program_selection(
    evaluator: Evaluator,
    parent,
    ex: CredentialingExtraction
):
    node = evaluator.add_parallel(
        id="Graduate_Program_Selection",
        desc="Identification of a specific Kansas public university offering an appropriate graduate certificate program",
        parent=parent,
        critical=False,
    )

    # Kansas public university check (simple verification against allowed list)
    uni_name = ex.university_name or ""
    allowed_list = "; ".join(ALLOWED_KS_PUBLIC_UNIS)
    ks_uni_node = evaluator.add_leaf(
        id="Kansas_Public_University",
        desc="Identifies a specific Kansas public university (ESU, FHSU, or PSU)",
        parent=node,
        critical=True,
    )
    ks_uni_claim = (
        f"The identified university name '{uni_name}' refers to one of the following Kansas public universities: "
        f"{allowed_list}."
    )
    await evaluator.verify(
        claim=ks_uni_claim,
        node=ks_uni_node,
        additional_instruction="Allow common abbreviations or shortened forms (e.g., 'Emporia State', 'Fort Hays', 'Pitt State').",
    )

    # 18-hour graduate certificate program verification by URL
    cert18_node = evaluator.add_leaf(
        id="Eighteen_Hour_Certificate_Program",
        desc="Program is an 18-credit hour graduate certificate designed for dual credit instructor qualification",
        parent=node,
        critical=True,
    )
    if not ex.program_url:
        cert18_node.score = 0.0
        cert18_node.status = "failed"
    else:
        claim = (
            "This university page describes a graduate certificate that totals 18 credit hours and is intended for dual "
            "credit or concurrent enrollment instructor credentialing."
        )
        await evaluator.verify(
            claim=claim,
            node=cert18_node,
            sources=ex.program_url,
            additional_instruction="On the page, look for 'Graduate Certificate', '18 credit hours' (or '18 hours'), and language about dual credit/concurrent enrollment credentialing.",
        )

    # Discipline specification (Psychology or Psychology of Learning) by URL
    discipline_node = evaluator.add_leaf(
        id="Discipline_Specification",
        desc="Program is in Psychology or Psychology of Learning for dual credit Psychology instruction",
        parent=node,
        critical=True,
    )
    if not ex.program_url:
        discipline_node.score = 0.0
        discipline_node.status = "failed"
    else:
        claim = (
            "This program is in the field of Psychology (including Psychology of Learning/Educational Psychology), "
            "suitable preparation for teaching dual credit Psychology."
        )
        await evaluator.verify(
            claim=claim,
            node=discipline_node,
            sources=ex.program_url,
            additional_instruction="Confirm the academic area is Psychology or Psychology of Learning/Educational Psychology.",
        )


async def build_license_renewal_impact(
    evaluator: Evaluator,
    parent,
    ex: CredentialingExtraction
):
    node = evaluator.add_parallel(
        id="License_Renewal_Impact_Calculation",
        desc="Calculation of how the graduate coursework counts toward Kansas teaching license renewal requirements",
        parent=parent,
        critical=False,
    )

    # Compute PD points from 18 credits using Kansas standard 1 credit = 20 PD points
    credits_int = first_int_in_text(ex.program_credits, default=18)
    if credits_int is None or credits_int <= 0:
        credits_int = 18  # default fallback as per task assumption
    computed_pd_points = credits_int * PD_POINTS_PER_CREDIT

    # Record calculation details for transparency
    evaluator.add_custom_info(
        info={
            "credits_used_for_pd_calc": credits_int,
            "pd_points_per_credit": PD_POINTS_PER_CREDIT,
            "computed_pd_points": computed_pd_points,
            "renewal_requirement_points_with_grad_degree": KS_RENEWAL_PD_REQUIREMENT_WITH_GRAD,
        },
        info_type="calculation",
        info_name="pd_points_calculation",
    )

    # PD points generated
    pd_points_node = evaluator.add_leaf(
        id="PD_Points_Generated",
        desc="18-credit hour program generates 360 PD points (18 × 20)",
        parent=node,
        critical=True,
    )
    pd_claim = (
        f"With the Kansas conversion standard of 1 college credit = {PD_POINTS_PER_CREDIT} professional development points, "
        f"an 18-credit hour program generates {18 * PD_POINTS_PER_CREDIT} PD points."
    )
    await evaluator.verify(
        claim=pd_claim,
        node=pd_points_node,
        additional_instruction="This is a straightforward arithmetic check; verify 18 × 20 = 360.",
    )

    # Renewal threshold analysis
    renewal_node = evaluator.add_leaf(
        id="Renewal_Threshold_Analysis",
        desc="360 PD points exceeds the 120-point requirement for Kansas license renewal (graduate degree holders)",
        parent=node,
        critical=True,
    )
    renewal_claim = (
        f"{18 * PD_POINTS_PER_CREDIT} PD points meets or exceeds the {KS_RENEWAL_PD_REQUIREMENT_WITH_GRAD} PD points "
        "required for Kansas teaching license renewal for teachers who hold a graduate degree."
    )
    await evaluator.verify(
        claim=renewal_claim,
        node=renewal_node,
        additional_instruction="Simple comparison check: 360 >= 120.",
    )


async def build_supporting_documentation(
    evaluator: Evaluator,
    parent,
    ex: CredentialingExtraction
):
    node = evaluator.add_parallel(
        id="Supporting_Documentation",
        desc="Provision of reference URLs supporting the analysis",
        parent=parent,
        critical=False,
    )

    # KBOR URL presence & relevance
    kbor_node = evaluator.add_leaf(
        id="KBOR_Dual_Credit_URL",
        desc="KBOR dual credit/concurrent enrollment faculty qualifications webpage is provided",
        parent=node,
        critical=True,
    )
    if not ex.kbor_url:
        kbor_node.score = 0.0
        kbor_node.status = "failed"
    else:
        claim = (
            "This webpage is from the Kansas Board of Regents and describes the faculty qualifications for dual credit/"
            "concurrent enrollment instructors (adopting HLC standards)."
        )
        await evaluator.verify(
            claim=claim,
            node=kbor_node,
            sources=ex.kbor_url,
            additional_instruction="Verify that the page is on a KBOR domain and addresses dual/concurrent enrollment faculty qualifications.",
        )

    # University program URL presence & relevance
    uni_prog_node = evaluator.add_leaf(
        id="University_Program_URL",
        desc="A Kansas public university program page confirms an 18-hour graduate certificate for dual credit credentialing in Psychology/Psychology of Learning",
        parent=node,
        critical=True,
    )
    if not ex.program_url:
        uni_prog_node.score = 0.0
        uni_prog_node.status = "failed"
    else:
        claim = (
            "This university webpage describes an 18-credit hour Graduate Certificate in Psychology or Psychology of "
            "Learning that is intended to help instructors meet dual credit/concurrent enrollment credentialing requirements."
        )
        await evaluator.verify(
            claim=claim,
            node=uni_prog_node,
            sources=ex.program_url,
            additional_instruction="Confirm the page is on the identified Kansas public university site and shows a graduate certificate with 18 hours in Psychology-related content for dual/concurrent enrollment credentialing.",
        )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    # Initialize evaluator (root sequential as rubric specifies; keep root non-critical to allow mixed children)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_credentialing(),
        template_class=CredentialingExtraction,
        extraction_name="extracted_credentialing_info",
    )

    # Add ground truth/reference constants used for evaluation context
    evaluator.add_ground_truth(
        {
            "hlc_min_hours_expected": 18,
            "allowed_kansas_public_universities": ALLOWED_KS_PUBLIC_UNIS,
            "kansas_pd_points_per_credit": PD_POINTS_PER_CREDIT,
            "kansas_license_renewal_pd_requirement_with_grad_degree": KS_RENEWAL_PD_REQUIREMENT_WITH_GRAD,
        },
        gt_type="reference_expectations",
    )

    # Build tree per rubric (sequential root; children parallel groups with critical leaves)
    await build_dual_credit_requirements(evaluator, root, extraction)
    await build_graduate_program_selection(evaluator, root, extraction)
    await build_license_renewal_impact(evaluator, root, extraction)
    await build_supporting_documentation(evaluator, root, extraction)

    # Return summary
    return evaluator.get_summary()