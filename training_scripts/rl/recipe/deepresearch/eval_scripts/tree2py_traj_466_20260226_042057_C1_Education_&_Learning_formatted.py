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
TASK_ID = "uf_spring_2026_start"
TASK_DESCRIPTION = "When does the Spring 2026 semester begin at the University of Florida?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UFStartDateExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer regarding UF Spring 2026 start date.
    - institution: The institution mentioned in the answer (expect 'University of Florida' or 'UF').
    - term: The academic term mentioned (expect 'Spring 2026' or reasonable variant, e.g., '2026 Spring semester').
    - start_date_text: The date string the answer claims is when classes begin for Spring 2026 at UF.
    - start_date_iso: The same date normalized to ISO format YYYY-MM-DD if possible; otherwise null.
    - sources: All URLs cited in the answer relevant to this claim (UF academic calendar pages, registrar pages, etc.).
    """
    institution: Optional[str] = None
    term: Optional[str] = None
    start_date_text: Optional[str] = None
    start_date_iso: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_uf_start_date_info() -> str:
    return """
    Extract the information the answer provides about when the Spring 2026 semester begins at the University of Florida.

    You must extract the following fields strictly from the answer text:

    1) institution:
       - The institution explicitly named in the answer for which the start date is provided.
       - Expect 'University of Florida' or the abbreviation 'UF'.
       - If multiple institutions are mentioned, extract the one tied to the start date claim.
       - If not mentioned, set to null.

    2) term:
       - The academic term explicitly mentioned in the answer (e.g., 'Spring 2026', '2026 Spring semester', 'Spring term 2026').
       - If not mentioned, set to null.

    3) start_date_text:
       - The exact date string the answer claims is the official "classes begin" / "first day of classes" date for Spring 2026 at UF.
       - Quote the date exactly as presented in the answer (e.g., 'January 6, 2026', '1/6/2026', 'Mon, Jan 6, 2026').
       - If no such date is provided, set to null.

    4) start_date_iso:
       - Normalize the 'start_date_text' to ISO format 'YYYY-MM-DD' if possible.
       - If normalization is not possible or the date is missing, set to null.

    5) sources:
       - Extract all URLs cited in the answer that relate to this start-date claim (UF academic calendar, registrar dates page, etc.).
       - Include URLs presented in plain form or within markdown links.
       - Do not invent URLs. If none are provided, return an empty list.

    Important guidance:
    - The date of interest is the official 'Classes Begin' / 'First day of classes' date for Spring 2026 at UF, not orientation, move-in, registration, or add/drop.
    - Return null for any missing field. Do not invent information.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree_and_verify(
    evaluator: Evaluator,
    root: Any,
    extracted: UFStartDateExtraction,
) -> None:
    """
    Build the rubric tree and execute verifications based on the extracted data.
    Tree per rubric:
    - Spring_2026_Start_Date_at_UF (critical, parallel)
      • Term_and_Institution_Specified (critical, leaf)
      • Correct_Official_Start_Date (critical, leaf)
    """
    # Top-level node (critical, parallel) mirroring rubric's root
    top_node = evaluator.add_parallel(
        id="Spring_2026_Start_Date_at_UF",
        desc="State when the Spring 2026 semester begins at the University of Florida (UF).",
        parent=root,
        critical=True
    )

    # Child 1: Term and Institution specified (critical, leaf)
    term_inst_leaf = evaluator.add_leaf(
        id="Term_and_Institution_Specified",
        desc="Answer clearly identifies the institution as the University of Florida (UF) and the term as Spring 2026.",
        parent=top_node,
        critical=True
    )

    inst = extracted.institution or ""
    term = extracted.term or ""
    term_inst_claim = (
        f"The answer explicitly identifies the institution as the University of Florida (UF) and the term as Spring 2026. "
        f"Extracted institution='{inst}', extracted term='{term}'. Both must be present and refer to UF and Spring 2026."
    )
    await evaluator.verify(
        claim=term_inst_claim,
        node=term_inst_leaf,
        additional_instruction=(
            "Accept reasonable variants and abbreviations: 'UF' == 'University of Florida'; "
            "'Spring 2026' == '2026 Spring semester' == 'Spring term 2026'. "
            "If either institution or term is missing, vague, or refers to a different school/term, judge incorrect."
        )
    )

    # Child 2: Correct official start date (critical, leaf)
    start_date_leaf = evaluator.add_leaf(
        id="Correct_Official_Start_Date",
        desc="Provides the correct official start date for the Spring 2026 semester at UF (i.e., the academic-calendar date when classes begin).",
        parent=top_node,
        critical=True
    )

    # Prefer the exact text the answer provided; fall back to ISO if needed
    date_text = (extracted.start_date_text or extracted.start_date_iso or "").strip()
    sources_list = extracted.sources if extracted and extracted.sources else []

    start_date_claim = (
        f"The official University of Florida (UF) Spring 2026 'Classes Begin' (first day of classes) date is {date_text}."
    )
    await evaluator.verify(
        claim=start_date_claim,
        node=start_date_leaf,
        sources=sources_list,  # If empty, falls back to simple verification; otherwise uses URL(s)
        additional_instruction=(
            "Verify this claim against the cited webpage(s). Look specifically for UF registrar/academic calendar pages. "
            "The target label is 'Classes Begin' or 'First day of classes' for Spring 2026. "
            "Ignore orientation, move-in, registration, or add/drop dates. "
            "Allow minor formatting differences (e.g., day of week present/absent, abbreviated months). "
            "If URLs are irrelevant/invalid or do not support the date, judge not supported."
        )
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
    Evaluate the agent's answer for: 'When does the Spring 2026 semester begin at the University of Florida?'
    Returns a structured summary with the verification tree and final score.
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
        default_model=model
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_uf_start_date_info(),
        template_class=UFStartDateExtraction,
        extraction_name="uf_spring_2026_start_date_info"
    )

    # Build tree and verify per rubric
    await build_verification_tree_and_verify(evaluator, root, extracted)

    # Return evaluator's summary
    return evaluator.get_summary()