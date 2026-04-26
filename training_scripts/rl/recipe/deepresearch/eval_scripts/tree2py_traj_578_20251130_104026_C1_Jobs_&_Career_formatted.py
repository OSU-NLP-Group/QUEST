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
TASK_ID = "yale_coach_undergrad"
TASK_DESCRIPTION = "What college did the current head football coach at Yale University attend for his undergraduate degree?"

# Ground truth context (for debugging/summary purposes only)
GROUND_TRUTH = {
    "coach_name": "Tony Reno",
    "coach_title": "Joel E. Smilow '54 Head Coach of Football",
    "undergrad_institution_aliases": [
        "Worcester State College",
        "Worcester State University",
        "Worcester State"
    ],
    "undergrad_grad_year": "1997"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CoachEducationExtraction(BaseModel):
    coach_name: Optional[str] = None
    coach_title: Optional[str] = None
    undergrad_institution: Optional[str] = None
    undergrad_grad_year: Optional[str] = None
    sources_identity_title: List[str] = Field(default_factory=list)
    sources_education: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_info() -> str:
    return """
    Extract the following information exactly as it appears in the provided answer text.

    Required fields:
    - coach_name: The person the answer identifies as the current Yale University head football coach (name only).
    - coach_title: The official title for the Yale head coach position if stated (e.g., Joel E. Smilow '54 Head Coach of Football). If not stated, return null.
    - undergrad_institution: The undergraduate institution named for the coach (e.g., Worcester State College or Worcester State University). If not stated, return null.
    - undergrad_grad_year: The undergraduate graduation year as a 4-digit string if stated (e.g., "1997"). If the answer uses shorthand like '97 or 97, normalize it to "1997" when unambiguous; otherwise return the exact text or null if unclear.
    - sources_identity_title: A list of all URLs explicitly cited in the answer that could support the coach's identity and/or official title. Only include URLs explicitly present in the answer (including markdown links). If none are provided, return an empty list.
    - sources_education: A list of all URLs explicitly cited in the answer that could support the undergraduate institution and/or graduation year. Only include URLs explicitly present in the answer (including markdown links). If none are provided, return an empty list.

    Notes:
    - Do not invent or infer any URLs; only return URLs explicitly present in the answer.
    - If a URL is missing a protocol, prepend http:// as needed.
    - For undergrad_institution, keep the exact wording from the answer (e.g., "Worcester State College", "Worcester State University", or "Worcester State").
    - For undergrad_grad_year, prefer a 4-digit year if clearly stated; otherwise return what's stated or null if absent.
    """


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: CoachEducationExtraction) -> None:
    """
    Build the verification tree according to the rubric and run all checks.
    """
    # Critical top-level group (must pass all children)
    main_node = evaluator.add_parallel(
        id="yale_head_coach_education",
        desc="Evaluates whether the answer identifies the current Yale head football coach (as of 2025) and correctly states his undergraduate education, with verifiable sourcing.",
        parent=evaluator.root,
        critical=True
    )

    # ------------------------------------------------------------------- #
    # Leaf checks: The answer text states the required information         #
    # ------------------------------------------------------------------- #

    # 1) Coach identity as of 2025 -> Answer states Tony Reno
    node_identity_answer = evaluator.add_leaf(
        id="coach_identity_as_of_2025",
        desc="Answer identifies the current Yale University head football coach (as of 2025) as Tony Reno.",
        parent=main_node,
        critical=True
    )
    claim_identity_answer = (
        "In the answer text, the current Yale University head football coach (as of 2025) is identified as 'Tony Reno'. "
        "Judge only whether the answer text states this; do not rely on outside knowledge."
    )
    await evaluator.verify(
        claim=claim_identity_answer,
        node=node_identity_answer,
        additional_instruction="Check the answer content. Allow minor name variants (e.g., Anthony/Tony Reno). If the answer names a different person or omits the name, mark incorrect."
    )

    # 2) Coach title -> Answer states the endowed title (if present)
    node_title_answer = evaluator.add_leaf(
        id="coach_official_title",
        desc="Answer states Tony Reno holds the Joel E. Smilow '54 Head Coach of Football position.",
        parent=main_node,
        critical=True
    )
    claim_title_answer = (
        "In the answer text, the coach's official title is given as the 'Joel E. Smilow '54 Head Coach of Football' (or a clearly equivalent phrasing). "
        "Judge only based on what the answer explicitly states."
    )
    await evaluator.verify(
        claim=claim_title_answer,
        node=node_title_answer,
        additional_instruction="Allow minor punctuation or formatting variations, and allow presence/absence of the ’54 text. If the answer does not state any title or states a different title, mark incorrect."
    )

    # 3) Undergrad institution -> Answer states Worcester State College/University
    node_undergrad_inst_answer = evaluator.add_leaf(
        id="undergrad_institution",
        desc="Answer states the coach’s undergraduate institution is Worcester State College (now Worcester State University).",
        parent=main_node,
        critical=True
    )
    claim_undergrad_inst_answer = (
        "In the answer text, the coach’s undergraduate institution is stated as Worcester State College or Worcester State University (treat these as equivalent). "
        "Judge only based on the answer text."
    )
    await evaluator.verify(
        claim=claim_undergrad_inst_answer,
        node=node_undergrad_inst_answer,
        additional_instruction="Accept 'Worcester State College', 'Worcester State University', or 'Worcester State' as equivalent phrasings."
    )

    # 4) Undergrad graduation year -> Answer states 1997
    node_undergrad_year_answer = evaluator.add_leaf(
        id="undergrad_graduation_year",
        desc="Answer states the coach graduated in 1997.",
        parent=main_node,
        critical=True
    )
    claim_undergrad_year_answer = (
        "In the answer text, the coach's undergraduate graduation year is stated as 1997 (accept variants like ’97 or 97 when clearly meaning 1997). "
        "Judge only based on the answer text."
    )
    await evaluator.verify(
        claim=claim_undergrad_year_answer,
        node=node_undergrad_year_answer,
        additional_instruction="Accept '1997', '’97', or '97' if clearly used in the context of the undergraduate graduation year."
    )

    # ------------------------------------------------------------------- #
    # Verifiable sourcing checks (evidence-based)                          #
    # ------------------------------------------------------------------- #
    source_group = evaluator.add_parallel(
        id="verifiable_sourcing_for_claims",
        desc="Answer provides verifiable citation(s) from official Yale athletics/athletic department sources or other reliable biographical sources that support the coach identity/title and undergraduate education claims.",
        parent=main_node,
        critical=True
    )

    # Subgroup: Identity & Title sourcing
    id_title_group = evaluator.add_parallel(
        id="identity_title_sourcing",
        desc="Sourcing for coach identity and official title claims.",
        parent=source_group,
        critical=True
    )

    # Sources present for identity/title
    id_title_sources_present = evaluator.add_custom_node(
        result=bool(extracted.sources_identity_title),
        id="sources_identity_title_present",
        desc="At least one source URL is provided in the answer for coach identity/title.",
        parent=id_title_group,
        critical=True
    )

    # Identity supported by provided sources
    node_identity_supported = evaluator.add_leaf(
        id="identity_supported_by_sources",
        desc="Provided sources support the claim about the current Yale head coach identity.",
        parent=id_title_group,
        critical=True
    )
    name_for_claim = extracted.coach_name or "Tony Reno"
    claim_identity_supported = (
        f"According to at least one of these pages, {name_for_claim} is the head football coach of Yale University."
    )
    await evaluator.verify(
        claim=claim_identity_supported,
        node=node_identity_supported,
        sources=extracted.sources_identity_title,
        additional_instruction="Treat as supported if the page indicates the person is the Yale head football coach (allowing phrasing like 'Head Coach of Football'). Prefer official Yale athletics pages; if not available, other widely recognized reliable sources are acceptable. If no page explicitly supports this, mark as not supported."
    )

    # Title supported by provided sources
    node_title_supported = evaluator.add_leaf(
        id="title_supported_by_sources",
        desc="Provided sources support the claim about the official endowed title.",
        parent=id_title_group,
        critical=True
    )
    title_for_claim = extracted.coach_title or "Joel E. Smilow '54 Head Coach of Football"
    claim_title_supported = (
        f"According to at least one of these pages, {name_for_claim} holds the endowed title '{title_for_claim}' at Yale."
    )
    await evaluator.verify(
        claim=claim_title_supported,
        node=node_title_supported,
        sources=extracted.sources_identity_title,
        additional_instruction="Look for the endowed title on the page (allow minor punctuation or formatting variants, and presence/absence of the ’54 text). If no page shows this endowed title (or clear equivalent), mark as not supported."
    )

    # Subgroup: Education sourcing
    edu_group = evaluator.add_parallel(
        id="education_sourcing",
        desc="Sourcing for undergraduate institution and graduation year claims.",
        parent=source_group,
        critical=True
    )

    # Sources present for education
    edu_sources_present = evaluator.add_custom_node(
        result=bool(extracted.sources_education),
        id="sources_education_present",
        desc="At least one source URL is provided in the answer for undergraduate education.",
        parent=edu_group,
        critical=True
    )

    # Undergrad institution supported by provided sources
    node_undergrad_inst_supported = evaluator.add_leaf(
        id="undergrad_institution_supported",
        desc="Provided sources support the undergraduate institution claim.",
        parent=edu_group,
        critical=True
    )
    inst_for_claim = extracted.undergrad_institution or "Worcester State College"
    claim_undergrad_inst_supported = (
        f"According to at least one of these pages, {name_for_claim}'s undergraduate institution is {inst_for_claim}. "
        "Treat 'Worcester State College' and 'Worcester State University' as equivalent names for the same institution."
    )
    await evaluator.verify(
        claim=claim_undergrad_inst_supported,
        node=node_undergrad_inst_supported,
        sources=extracted.sources_education,
        additional_instruction="Support the claim only if the page explicitly indicates the coach's undergraduate institution as stated. Accept Worcester State College and Worcester State University as equivalent phrasings."
    )

    # Undergrad graduation year supported by provided sources
    node_undergrad_year_supported = evaluator.add_leaf(
        id="undergrad_year_supported",
        desc="Provided sources support the undergraduate graduation year claim.",
        parent=edu_group,
        critical=True
    )
    year_for_claim = extracted.undergrad_grad_year or "1997"
    claim_undergrad_year_supported = (
        f"According to at least one of these pages, {name_for_claim} graduated in {year_for_claim}."
    )
    await evaluator.verify(
        claim=claim_undergrad_year_supported,
        node=node_undergrad_year_supported,
        sources=extracted.sources_education,
        additional_instruction="Accept if the page clearly indicates the undergraduate graduation year as claimed (e.g., 'graduated in 1997'). Reject if the page contradicts the claimed year or is ambiguous."
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
    Evaluate an answer for the Yale head coach undergraduate education task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_coach_info(),
        template_class=CoachEducationExtraction,
        extraction_name="coach_education_extraction"
    )

    # Optional ground truth info for transparency
    evaluator.add_ground_truth(
        {
            "expected_coach_name": GROUND_TRUTH["coach_name"],
            "expected_coach_title": GROUND_TRUTH["coach_title"],
            "expected_undergrad_institution_aliases": GROUND_TRUTH["undergrad_institution_aliases"],
            "expected_undergrad_grad_year": GROUND_TRUTH["undergrad_grad_year"],
        },
        gt_type="ground_truth"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()