import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ga_fbs_requirements"
TASK_DESCRIPTION = (
    "You are planning to apply for Graduate Assistant football coaching positions at NCAA Division I FBS programs in the United States. "
    "Compile a comprehensive reference document that lists all the mandatory requirements, qualifications, certifications, and application materials typically needed to be eligible for and competitive in applying for these positions. "
    "Include educational requirements, required certifications, experience qualifications, necessary application materials, and typical salary expectations."
)


# --------------------------------------------------------------------------- #
# Data models for extracting structured info from the answer                  #
# --------------------------------------------------------------------------- #
class ReferenceDocExtraction(BaseModel):
    # Scope and framing
    scope_ncaafbs_us: Optional[bool] = None
    framed_as_reference_doc: Optional[bool] = None

    # Educational requirements
    bachelors_degree_required: Optional[bool] = None
    preferred_degree_fields: List[str] = Field(default_factory=list)  # e.g., kinesiology, sports management, etc.

    # Certifications & compliance
    cpr_cert_current_required: Optional[bool] = None
    aed_cert_current_required: Optional[bool] = None
    first_aid_cert_often_required: Optional[bool] = None
    ncaa_rules_recruiting_cert_required: Optional[bool] = None

    # Experience & other eligibility
    playing_or_coaching_experience_required: Optional[bool] = None
    criminal_background_check_required: Optional[bool] = None
    valid_drivers_license_often_required: Optional[bool] = None

    # Application materials
    resume_required: Optional[bool] = None
    professional_references_required: Optional[bool] = None
    references_count_typical: Optional[str] = None  # e.g., "3–5", "three to five"
    cover_letter_typically_required: Optional[bool] = None
    coaching_philosophy_statement_often_required: Optional[bool] = None

    # Salary
    typical_salary_range_text: Optional[str] = None  # e.g., "$35,000–$40,000 annually"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_reference_doc() -> str:
    return """
    You will extract whether the answer includes each of the following elements explicitly.
    For each boolean field, set it to true if the answer clearly includes or states the item; false if the answer clearly excludes or contradicts; and null if unclear or not mentioned.
    For list fields, extract exact phrases mentioned. For textual fields, return the phrase exactly as stated in the answer.

    Fields to extract:
    - scope_ncaafbs_us: Does the document explicitly scope to NCAA Division I FBS programs in the United States?
    - framed_as_reference_doc: Does the document frame itself as a reference document of typical GA football coaching application/eligibility requirements?

    Educational requirements:
    - bachelors_degree_required: Does it state a Bachelor's degree is required?
    - preferred_degree_fields: List any preferred/related degree fields explicitly mentioned (e.g., kinesiology, sports management, physical education, exercise science, or similar).

    Certifications & NCAA compliance:
    - cpr_cert_current_required: Does it include requirement for current CPR certification?
    - aed_cert_current_required: Does it include requirement for current AED certification?
    - first_aid_cert_often_required: Does it include that First Aid certification is often required?
    - ncaa_rules_recruiting_cert_required: Does it include requirement to complete NCAA rules education and recruiting certification (or equivalent NCAA compliance training for recruiting)?

    Experience & other eligibility:
    - playing_or_coaching_experience_required: Does it state playing or coaching experience in college or professional football is required?
    - criminal_background_check_required: Does it state successful completion of a criminal background check is required?
    - valid_drivers_license_often_required: Does it include that a valid driver's license is often required for recruiting travel?

    Application materials:
    - resume_required: Does it require submission of a professional coaching resume?
    - professional_references_required: Does it require professional references?
    - references_count_typical: If references are required, extract the typical count phrasing (e.g., "3–5", "three to five"). Otherwise, null.
    - cover_letter_typically_required: Does it include that a cover letter is typically required?
    - coaching_philosophy_statement_often_required: Does it include that a written coaching philosophy statement is often required?

    Salary:
    - typical_salary_range_text: Extract any typical salary range or expectation wording (e.g., "$35,000–$40,000 annually"). If none, return null.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def add_leaf_and_claim(
    evaluator: Evaluator,
    parent,
    node_id: str,
    node_desc: str,
    claim_text: str,
    additional_instruction: Optional[str] = None,
    critical: bool = True,
):
    """Create a leaf node and schedule verification with a simple claim."""
    node = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=critical,
    )
    return (claim_text, None, node, additional_instruction or "None")


# --------------------------------------------------------------------------- #
# Build verification tree and schedule checks                                 #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    extraction: ReferenceDocExtraction,
):
    # Create a critical top-level node to represent the reference document coverage
    ga_node = evaluator.add_parallel(
        id="Graduate_Assistant_Position_Requirements",
        desc=(
            "Reference document covers typical mandatory requirements/qualifications/certifications and application materials "
            "for NCAA Division I FBS Graduate Assistant football coaching positions in the United States, including typical salary expectations."
        ),
        parent=root_node,
        critical=True,
    )

    claims: List[tuple] = []

    # Scope and Context (leaf)
    claims.append(add_leaf_and_claim(
        evaluator,
        ga_node,
        "Scope_and_Context",
        "Scopes the document to NCAA Division I FBS programs in the United States and frames the output as a reference document of typical GA football coaching application/eligibility requirements.",
        claim_text=(
            "The document is scoped to NCAA Division I FBS programs in the United States and is framed as a reference document "
            "covering typical GA football coaching eligibility and application requirements."
        ),
        additional_instruction=(
            "Pass if the answer clearly indicates NCAA Division I FBS (not just NCAA broadly) and explicitly references the U.S. context. "
            "Also confirm it presents a generalized/typical reference overview rather than only a single school's posting."
        ),
        critical=True,
    ))

    # Educational Requirements (group)
    edu_node = evaluator.add_parallel(
        id="Educational_Requirements",
        desc="Covers educational requirements specified in constraints.",
        parent=ga_node,
        critical=True,
    )

    claims.append(add_leaf_and_claim(
        evaluator,
        edu_node,
        "Bachelors_Degree_Required",
        "States that a Bachelor's degree is required.",
        claim_text="The document states that a Bachelor's (BA/BS) degree is required for GA football coaching eligibility.",
        additional_instruction="Accept synonyms such as BA/BS or 'undergraduate degree'; reject vague 'preferred' if not 'required'.",
        critical=True,
    ))

    claims.append(add_leaf_and_claim(
        evaluator,
        edu_node,
        "Preferred_Degree_Fields_Listed",
        "Lists preferred/related degree fields (kinesiology, sports management, physical education, exercise science, or related fields).",
        claim_text=(
            "The document lists preferred or related degree fields such as kinesiology, sports management, physical education, "
            "exercise science, or explicitly notes 'related fields'."
        ),
        additional_instruction=(
            "Pass if at least some of these example fields are explicitly listed or if the answer clearly states related fields in the same vein."
        ),
        critical=True,
    ))

    # Certifications and NCAA Compliance (group)
    cert_node = evaluator.add_parallel(
        id="Certifications_and_NCAA_Compliance",
        desc="Covers certifications and NCAA compliance items specified in constraints.",
        parent=ga_node,
        critical=True,
    )

    claims.append(add_leaf_and_claim(
        evaluator,
        cert_node,
        "CPR_Certification_Current",
        "Includes requirement for current CPR certification.",
        claim_text="The document includes a requirement for current CPR certification.",
        additional_instruction="Accept 'CPR certification' and 'current CPR' phrasing.",
        critical=True,
    ))

    claims.append(add_leaf_and_claim(
        evaluator,
        cert_node,
        "AED_Certification_Current",
        "Includes requirement for current AED certification.",
        claim_text="The document includes a requirement for current AED certification.",
        additional_instruction="Accept 'AED certification' and 'current AED' phrasing.",
        critical=True,
    ))

    claims.append(add_leaf_and_claim(
        evaluator,
        cert_node,
        "First_Aid_Certification_Often",
        "Includes that First Aid certification is often required.",
        claim_text="The document includes that First Aid certification is often required.",
        additional_instruction="Accept 'First Aid certification' or equivalent wording such as 'basic first aid' and allow 'often required'.",
        critical=True,
    ))

    claims.append(add_leaf_and_claim(
        evaluator,
        cert_node,
        "NCAA_Rules_and_Recruiting_Certification",
        "Includes requirement to complete NCAA rules education and recruiting certification.",
        claim_text=(
            "The document includes a requirement to complete NCAA rules education and the NCAA recruiting certification (or equivalent NCAA compliance training needed for recruiting)."
        ),
        additional_instruction=(
            "Pass if the answer mentions NCAA rules education/training and certification enabling recruiting activities under NCAA regulations."
        ),
        critical=True,
    ))

    # Experience and Other Eligibility Requirements (group)
    exp_node = evaluator.add_parallel(
        id="Experience_and_Other_Eligibility_Requirements",
        desc="Covers non-certification eligibility/qualification requirements specified in constraints.",
        parent=ga_node,
        critical=True,
    )

    claims.append(add_leaf_and_claim(
        evaluator,
        exp_node,
        "Football_Playing_or_Coaching_Experience",
        "States that playing or coaching experience in college or professional football is required.",
        claim_text="The document states that playing or coaching experience in college or professional football is required.",
        additional_instruction="Reject if the answer only says 'preferred' or 'recommended' without 'required'.",
        critical=True,
    ))

    claims.append(add_leaf_and_claim(
        evaluator,
        exp_node,
        "Criminal_Background_Check",
        "States that successful completion of a criminal background check is required.",
        claim_text="The document states that successful completion of a criminal background check is required.",
        additional_instruction="Accept common phrasing such as 'background check required' or 'clear background check needed'.",
        critical=True,
    ))

    claims.append(add_leaf_and_claim(
        evaluator,
        exp_node,
        "Valid_Drivers_License_Often",
        "Includes that a valid driver's license is often required for recruiting travel.",
        claim_text="The document includes that a valid driver's license is often required for recruiting travel.",
        additional_instruction="Accept wording like 'must have valid driver's license' or 'often required for travel/recruiting'.",
        critical=True,
    ))

    # Application Materials (group)
    app_node = evaluator.add_parallel(
        id="Application_Materials",
        desc="Covers required/typical application materials specified in constraints.",
        parent=ga_node,
        critical=True,
    )

    claims.append(add_leaf_and_claim(
        evaluator,
        app_node,
        "Resume_Submitted",
        "Requires submission of a professional coaching resume.",
        claim_text="The document requires submission of a professional coaching resume.",
        additional_instruction="Accept synonyms like 'coaching résumé' or 'updated resume'.",
        critical=True,
    ))

    claims.append(add_leaf_and_claim(
        evaluator,
        app_node,
        "Professional_References_3_to_5",
        "Requires professional references, typically 3–5.",
        claim_text="The document requires professional references, typically 3–5.",
        additional_instruction=(
            "Pass if the answer indicates references are required and gives a typical count like 3–5 (accept '3-5' or 'three to five')."
        ),
        critical=True,
    ))

    claims.append(add_leaf_and_claim(
        evaluator,
        app_node,
        "Cover_Letter_Typically",
        "Includes that a cover letter is typically required.",
        claim_text="The document includes that a cover letter is typically required.",
        additional_instruction="Accept phrasing like 'cover letter typically required' or 'usually required'.",
        critical=True,
    ))

    claims.append(add_leaf_and_claim(
        evaluator,
        app_node,
        "Coaching_Philosophy_Statement_Often",
        "Includes that a written coaching philosophy statement is often required as part of the coaching portfolio.",
        claim_text="The document includes that a written coaching philosophy statement is often required as part of the coaching portfolio.",
        additional_instruction="Accept 'coaching philosophy statement' or equivalent 'statement of coaching philosophy'.",
        critical=True,
    ))

    # Typical Salary Expectations (leaf)
    claims.append(add_leaf_and_claim(
        evaluator,
        ga_node,
        "Typical_Salary_Expectations",
        "Includes typical salary range of approximately $35,000–$40,000 annually.",
        claim_text="The document includes a typical salary range of approximately $35,000–$40,000 annually.",
        additional_instruction=(
            "Pass if the answer states a typical range around $35k–$40k annually (accept '$35,000-$40,000', '$35k-$40k', or similar phrasing denoting approximately that range)."
        ),
        critical=True,
    ))

    # Execute verifications in parallel
    await evaluator.batch_verify(claims)


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
    Evaluate an answer for the NCAA Division I FBS Graduate Assistant football coaching requirements reference document.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured summary (recorded in the final summary for transparency)
    extraction = await evaluator.extract(
        prompt=prompt_extract_reference_doc(),
        template_class=ReferenceDocExtraction,
        extraction_name="reference_doc_extraction",
    )

    # Add ground truth info (rubric-driven expectations)
    evaluator.add_ground_truth({
        "required_sections": [
            "Scope_and_Context",
            "Educational_Requirements",
            "Certifications_and_NCAA_Compliance",
            "Experience_and_Other_Eligibility_Requirements",
            "Application_Materials",
            "Typical_Salary_Expectations",
        ],
        "key_items": {
            "education": ["Bachelor's degree required", "preferred fields listed (e.g., kinesiology, sports management, PE, exercise science)"],
            "certifications": ["CPR current", "AED current", "First Aid often", "NCAA rules education & recruiting certification"],
            "experience": ["playing/coaching experience required", "criminal background check required", "valid driver's license often required"],
            "materials": ["resume", "professional references (typically 3–5)", "cover letter typically required", "coaching philosophy statement often required"],
            "salary": "approximately $35,000–$40,000 annually",
        },
    })

    # Build verification tree and run all checks
    await build_verification_tree(evaluator, root, extraction)

    # Return final summary
    return evaluator.get_summary()