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
TASK_ID = "us_ttt_requirements"
TASK_DESCRIPTION = """
I am a PhD candidate planning to apply for tenure-track assistant professor positions at research universities in the United States for the first time. What are the standard required application materials I need to prepare, and what are the key qualifications required? Please provide specific information about the degree qualification needed, the number of letters of recommendation typically required, whether research and teaching statements are needed (including typical length expectations), and any other standard required documents for a complete application package.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ApplicationMaterialsExtraction(BaseModel):
    # Degree and key qualifications
    degree_required_statement: Optional[str] = None
    degree_required: Optional[bool] = None
    key_qualifications_beyond_degree: List[str] = Field(default_factory=list)
    describes_beyond_degree: Optional[bool] = None  # True if research + teaching qualifications are discussed
    
    # Letters of recommendation
    letters_statement: Optional[str] = None
    typical_letters_count: Optional[str] = None  # e.g., "3", "2–4", "3 to 5"
    
    # Research statement
    research_statement_required: Optional[bool] = None
    research_statement_length: Optional[str] = None  # e.g., "2–4 pages", "about 3 pages"
    
    # Teaching statement
    teaching_statement_required: Optional[bool] = None
    teaching_statement_length: Optional[str] = None  # e.g., "1–2 pages", "varies by field"
    
    # CV
    cv_included: Optional[bool] = None
    
    # Cover letter
    cover_letter_required: Optional[bool] = None
    cover_letter_length: Optional[str] = None  # e.g., "1–3 pages"
    
    # Diversity statement (optional)
    diversity_statement_common: Optional[bool] = None
    diversity_statement_length: Optional[str] = None  # e.g., "up to 2 pages"
    
    # Other documents mention (optional)
    other_documents_mentioned: Optional[bool] = None
    other_documents_examples: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_application_requirements() -> str:
    return """
    Extract from the answer the applicant-facing requirements and qualifications for U.S. research-university tenure-track assistant professor applications.

    Return a JSON object with the following fields, using only information explicitly present in the answer:

    degree_required_statement: short quote/phrase indicating the required degree, if present (e.g., "PhD (or equivalent terminal degree) required by start date")
    degree_required: true/false if the answer states that a PhD (or equivalent terminal degree) is required (including "by start date" variants)

    key_qualifications_beyond_degree: list of phrases (e.g., "evidence of research potential", "publication record", "teaching effectiveness", "mentoring experience")
    describes_beyond_degree: true/false if the answer clearly discusses qualifications beyond the degree and includes BOTH research-related and teaching-related qualifications

    letters_statement: verbatim or paraphrased phrase about letters (e.g., "three letters of recommendation")
    typical_letters_count: the typical number or range as a short string (e.g., "3", "2–4", "3 to 5")

    research_statement_required: true/false if the answer states a research statement is required
    research_statement_length: short phrase for typical length if present (e.g., "2–4 pages", "about 3 pages")

    teaching_statement_required: true/false if the answer states a teaching statement is required
    teaching_statement_length: short phrase for typical length or "varies by field" if stated (e.g., "1–2 pages", "varies by field")

    cv_included: true/false if the answer includes that a CV is required

    cover_letter_required: true/false if the answer states a cover letter is required
    cover_letter_length: short phrase for typical length if present (e.g., "1–3 pages")

    diversity_statement_common: true/false if the answer notes that diversity statements are commonly/often required (even if not always)
    diversity_statement_length: short phrase for typical length when required (e.g., "no more than 2 pages")

    other_documents_mentioned: true/false if the answer mentions that additional documents may be requested depending on field/institution
    other_documents_examples: list of any examples mentioned (e.g., "job market paper", "teaching evaluations", "sample syllabi")

    Rules:
    - Do not invent content. If a field is not clearly stated, set it to null (or false for booleans).
    - For lengths and counts, keep them as short strings exactly as written in the answer (numbers/ranges, like "2–4 pages").
    - Recognize common synonyms: research statement ≈ research plan; teaching statement ≈ teaching philosophy; cover letter ≈ letter of application; diversity statement ≈ DEI statement.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_requirements_tree(
    evaluator: Evaluator,
    parent_node,
) -> None:
    """
    Build the verification tree per rubric and verify each leaf against the answer text.
    This verification checks whether the answer contains the requested information,
    not whether the information is universally true.
    """

    # Main container node (non-critical to permit optional sub-criteria under it)
    req_node = evaluator.add_parallel(
        id="Standard_Academic_Application_Requirements",
        desc="Identifies standard required application materials and key qualifications for tenure-track assistant professor positions at U.S. research universities, consistent with the question.",
        parent=parent_node,
        critical=False
    )

    # 1) Degree Qualification (Critical)
    degree_leaf = evaluator.add_leaf(
        id="Degree_Qualification",
        desc="States that a PhD or equivalent terminal degree in the relevant field is required.",
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, it is stated that a PhD (or equivalent terminal/doctoral degree) is required for U.S. research-university tenure-track assistant professor roles (including formulations like 'by start date' or 'expected by start date').",
        node=degree_leaf,
        additional_instruction="Accept synonyms like Ph.D., doctoral degree, doctorate, or specific terminal degrees in the field (e.g., EdD, MD, MFA where appropriate)."
    )

    # 2) Key Qualifications Beyond Degree (Critical)
    kq_leaf = evaluator.add_leaf(
        id="Key_Qualifications_Beyond_Degree",
        desc="Describes key qualifications beyond the degree (e.g., evidence of research potential/track record and teaching ability), not just a list of documents.",
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, beyond stating the degree requirement, it describes qualifications such as research potential/track record and teaching ability/effectiveness (i.e., both research and teaching aspects are discussed, not just a document checklist).",
        node=kq_leaf,
        additional_instruction="Look for both research-related (e.g., publications, research vision, grants) and teaching-related (e.g., teaching experience, evaluations) qualifiers."
    )

    # 3) Letters of Recommendation (Critical)
    letters_leaf = evaluator.add_leaf(
        id="Letters_of_Recommendation",
        desc="States that 3 letters of recommendation are typically required (may also note a broader range such as 2–6).",
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, it is stated that applicants typically need 3 letters of recommendation; stating a range that includes 3 (e.g., 2–6, 2–4, 3–5) also satisfies this.",
        node=letters_leaf,
        additional_instruction="Accept synonyms such as 'reference letters'. The key is that 3 is the standard or that a range including 3 is mentioned."
    )

    # 4) Research Statement (Critical, presence + length)
    rs_parent = evaluator.add_sequential(
        id="Research_Statement",
        desc="States that a research statement is required and gives a typical length expectation of 2–4 pages.",
        parent=req_node,
        critical=True
    )
    rs_req_leaf = evaluator.add_leaf(
        id="Research_Statement_required",
        desc="The answer states that a research statement is required.",
        parent=rs_parent,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, it is stated that a research statement (aka research plan) is required.",
        node=rs_req_leaf,
        additional_instruction="Accept synonyms like 'research plan', 'statement of research'."
    )
    rs_len_leaf = evaluator.add_leaf(
        id="Research_Statement_length",
        desc="The answer gives a typical length expectation of about 2–4 pages for the research statement.",
        parent=rs_parent,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, it provides a typical length expectation for the research statement of about 2–4 pages.",
        node=rs_len_leaf,
        additional_instruction="Minor reasonable variants around this band (e.g., 2–3, 3–4, ~3 pages) are acceptable as representing 'about 2–4 pages'."
    )

    # 5) Teaching Statement (Critical, presence + some length expectation or note on variance)
    ts_parent = evaluator.add_sequential(
        id="Teaching_Statement",
        desc="States that a teaching statement is required AND provides a typical length expectation (may note that norms vary by field; no specific page count required unless stated).",
        parent=req_node,
        critical=True
    )
    ts_req_leaf = evaluator.add_leaf(
        id="Teaching_Statement_required",
        desc="The answer states that a teaching statement is required.",
        parent=ts_parent,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, it is stated that a teaching statement (aka teaching philosophy) is required.",
        node=ts_req_leaf,
        additional_instruction="Accept synonyms like 'teaching philosophy statement'."
    )
    ts_len_leaf = evaluator.add_leaf(
        id="Teaching_Statement_length",
        desc="The answer provides a typical length expectation for the teaching statement or explicitly notes that norms vary by field.",
        parent=ts_parent,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, it provides a typical length expectation for the teaching statement (e.g., ~1–2 pages) OR explicitly notes that length norms vary by field; an exact page count is not required if variance is clearly stated.",
        node=ts_len_leaf,
        additional_instruction="Either a concrete typical length (like 1–2 pages) or a clear statement that expectations vary by field should be present."
    )

    # 6) Curriculum Vitae (Critical)
    cv_leaf = evaluator.add_leaf(
        id="Curriculum_Vitae",
        desc="Includes that a Curriculum Vitae (CV) is a required document for academic position applications.",
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, it indicates that a Curriculum Vitae (CV) is required in the application package.",
        node=cv_leaf,
        additional_instruction="CV is a standard required document; accept 'academic CV' phrasing."
    )

    # 7) Cover Letter (Critical, presence + length)
    cl_parent = evaluator.add_sequential(
        id="Cover_Letter",
        desc="Includes that a cover letter is required and provides a typical length expectation of about 1–3 pages.",
        parent=req_node,
        critical=True
    )
    cl_req_leaf = evaluator.add_leaf(
        id="Cover_Letter_required",
        desc="The answer states that a cover letter is required.",
        parent=cl_parent,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, it indicates that a cover letter (letter of application) is required.",
        node=cl_req_leaf,
        additional_instruction="Accept synonyms like 'letter of application' or 'cover letter'."
    )
    cl_len_leaf = evaluator.add_leaf(
        id="Cover_Letter_length",
        desc="The answer provides a typical cover letter length of about 1–3 pages (field-dependent acceptable).",
        parent=cl_parent,
        critical=True
    )
    await evaluator.verify(
        claim="In the answer, it provides a typical cover letter length of about 1–3 pages.",
        node=cl_len_leaf,
        additional_instruction="A typical guidance around 1 page is also acceptable within the 'about 1–3 pages' envelope."
    )

    # 8) Diversity Statement Information (Non-Critical, presence + length)
    ds_parent = evaluator.add_sequential(
        id="Diversity_Statement_Information",
        desc="Notes that diversity statements are commonly required and, when required, are typically no more than 2 pages.",
        parent=req_node,
        critical=False
    )
    ds_req_leaf = evaluator.add_leaf(
        id="Diversity_Statement_common",
        desc="The answer notes that diversity statements are commonly required (even if not universal).",
        parent=ds_parent,
        critical=False
    )
    await evaluator.verify(
        claim="In the answer, it notes that diversity statements (DEI statements) are commonly or often required.",
        node=ds_req_leaf,
        additional_instruction="Accept synonyms such as 'DEI statement', 'statement on diversity, equity, and inclusion'."
    )
    ds_len_leaf = evaluator.add_leaf(
        id="Diversity_Statement_length",
        desc="The answer indicates that, when required, a diversity statement is typically no more than 2 pages.",
        parent=ds_parent,
        critical=False
    )
    await evaluator.verify(
        claim="In the answer, it indicates that when a diversity statement is required, it is typically no more than 2 pages.",
        node=ds_len_leaf,
        additional_instruction="Phrasings like 'about 1–2 pages' or '≤ 2 pages' qualify."
    )

    # 9) Other Standard Documents Mention (Non-Critical)
    other_leaf = evaluator.add_leaf(
        id="Other_Standard_Documents_Mention",
        desc="Mentions that additional documents may be requested depending on field/institution (without requiring any specific extra document).",
        parent=req_node,
        critical=False
    )
    await evaluator.verify(
        claim="In the answer, it mentions that additional documents may be requested depending on the field or institution (e.g., teaching evaluations, sample syllabi, job market paper, research samples).",
        node=other_leaf,
        additional_instruction="Any reasonable mention that extra documents may be requested depending on context qualifies; no specific extra document is required."
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
    Evaluate an answer for the U.S. tenure-track assistant professor application requirements task.
    """
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

    # Extract structured info from the answer (for summary/traceability)
    extracted = await evaluator.extract(
        prompt=prompt_extract_application_requirements(),
        template_class=ApplicationMaterialsExtraction,
        extraction_name="application_requirements_extraction",
    )

    # Ground truth expectations (for transparency)
    evaluator.add_ground_truth({
        "essential_requirements": {
            "degree": "PhD or equivalent terminal degree (often by start date)",
            "letters": "Typically 3 letters of recommendation (ranges like 2–6 appear)",
            "research_statement": "Required; typical length about 2–4 pages",
            "teaching_statement": "Required; typical length often ~1–2 pages or note that norms vary by field",
            "cv": "Required",
            "cover_letter": "Required; typical length about 1–3 pages",
        },
        "common_optional": {
            "diversity_statement": "Commonly required; typical when required is ≤ 2 pages",
            "other_documents": "May be requested depending on field/institution (e.g., job market paper, teaching evaluations, syllabi)"
        }
    })

    # Build verification tree and run checks
    await build_and_verify_requirements_tree(evaluator, root)

    # Return structured result
    return evaluator.get_summary()