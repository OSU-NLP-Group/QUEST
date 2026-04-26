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
TASK_ID = "coursera_colorado_pba_feb2026"
TASK_DESCRIPTION = (
    "Identify the name of an online master's degree program offered through Coursera by a university in Colorado that meets all of the following criteria: "
    "the program must be in Computer Science, Artificial Intelligence, or Data Science; it must offer performance-based admission (allowing enrollment without a traditional application "
    "or without requiring a bachelor's degree upfront); and it must have an application deadline in February 2026."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramExtraction(BaseModel):
    """
    Extracted information about the program as presented in the agent's answer.
    Prefer strings for flexibility; URLs explicitly mentioned in the answer should be captured.
    """
    program_name: Optional[str] = None
    university_name: Optional[str] = None
    field_of_study: Optional[str] = None
    location_state: Optional[str] = None

    coursera_url: Optional[str] = None                 # Coursera degree page URL if provided
    university_program_url: Optional[str] = None       # Official university program page URL if provided
    sources: List[str] = Field(default_factory=list)   # Any other URLs the answer cites as supporting sources

    performance_based_admission_text: Optional[str] = None  # Any text the answer quotes/claims about PBA
    application_deadline_text: Optional[str] = None         # Any text the answer quotes/claims about deadlines


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_program() -> str:
    return """
    From the answer text, extract the single Coursera-offered online master's degree program that the answer identifies, including the metadata and all cited URLs.

    Required fields:
    - program_name: The exact name/title of the master's program mentioned in the answer.
    - university_name: The university offering the identified program (as written in the answer).
    - field_of_study: The field or specialization stated for the program (e.g., "Data Science", "Computer Science", "Artificial Intelligence" or close variants).
    - location_state: The U.S. state associated with the university, if explicitly mentioned (e.g., "Colorado"). If not explicitly stated, return null.
    - coursera_url: The URL to the Coursera degree page for this program if provided. If missing, return null.
    - university_program_url: The official university program page URL if provided. If missing, return null.
    - sources: An array of all additional URLs the answer cites for this program (beyond the coursera_url and university_program_url). Include only URLs explicitly present in the answer. If none, return an empty array.
    - performance_based_admission_text: Any sentence or phrase in the answer that asserts or describes performance-based admission, such as 'no traditional application required', 'no bachelor's required upfront', 'complete for-credit pathway courses to qualify', or 'performance-based admissions'. If nothing is provided, return null.
    - application_deadline_text: Any sentence or phrase in the answer that states or implies the application/enrollment deadline (e.g., 'apply by Feb 15, 2026', 'enroll by Feb 2026'). If nothing is provided, return null.

    Rules for URLs:
    - Only include URLs explicitly present in the answer. Do not invent URLs.
    - If a URL lacks protocol, prepend http://
    - Accept plain URLs or markdown links; extract the actual URL string.

    If a field is not present in the answer, set it to null (or [] for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_and_dedupe_urls(*url_lists: List[Optional[str] | List[str] | None]) -> List[str]:
    """Merge multiple URL inputs into a single de-duplicated list while preserving order."""
    seen = set()
    merged: List[str] = []
    for item in url_lists:
        if item is None:
            continue
        if isinstance(item, list):
            for u in item:
                if not u:
                    continue
                us = u.strip()
                if us and us not in seen:
                    seen.add(us)
                    merged.append(us)
        elif isinstance(item, str):
            us = item.strip()
            if us and us not in seen:
                seen.add(us)
                merged.append(us)
        else:
            # Ignore unknown types silently
            continue
    return merged


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: ProgramExtraction) -> None:
    """
    Build the rubric verification tree and run verifications based on extracted information.
    """
    # Top-level rubric node: Program_Identification (critical, parallel)
    program_node = evaluator.add_parallel(
        id="Program_Identification",
        desc="Response identifies a Coursera-offered online master's program from a Colorado university that meets all listed constraints.",
        parent=evaluator.root,
        critical=True,
    )

    # Child 1: Provides_Program_Name (critical leaf via custom check)
    has_program_name = bool(extraction.program_name and extraction.program_name.strip())
    evaluator.add_custom_node(
        result=has_program_name,
        id="Provides_Program_Name",
        desc="Response provides the name/title of the program being identified.",
        parent=program_node,
        critical=True
    )

    # Child 2: Meets_All_Constraints (critical, parallel)
    constraints_node = evaluator.add_parallel(
        id="Meets_All_Constraints",
        desc="The identified program satisfies every stated constraint.",
        parent=program_node,
        critical=True
    )

    # Consolidate all sources provided in the answer
    all_sources = _merge_and_dedupe_urls(
        [extraction.coursera_url] if extraction.coursera_url else [],
        [extraction.university_program_url] if extraction.university_program_url else [],
        extraction.sources if extraction.sources else []
    )

    # 2.1 Online_Masters_Degree_Program
    online_node = evaluator.add_leaf(
        id="Online_Masters_Degree_Program",
        desc="The program is an online master's degree program.",
        parent=constraints_node,
        critical=True
    )
    online_claim = (
        "This program is an online master's degree program (fully online delivery; degree level is Master's such as MS/MSc/MA/MEng)."
    )
    await evaluator.verify(
        claim=online_claim,
        node=online_node,
        sources=all_sources,
        additional_instruction="Check the degree level (e.g., MS, MSc, MA, MEng) and that the program is delivered online."
    )

    # 2.2 Offered_Through_Coursera
    coursera_node = evaluator.add_leaf(
        id="Offered_Through_Coursera",
        desc="The program is offered through Coursera.",
        parent=constraints_node,
        critical=True
    )
    if extraction.coursera_url:
        coursera_claim = (
            f"This program is offered through Coursera; a degree program page exists on coursera.org at {extraction.coursera_url}."
        )
        coursera_sources = _merge_and_dedupe_urls([extraction.coursera_url], all_sources)
    else:
        coursera_claim = "This program is offered through Coursera (it is a Coursera degree)."
        coursera_sources = all_sources
    await evaluator.verify(
        claim=coursera_claim,
        node=coursera_node,
        sources=coursera_sources,
        additional_instruction="Confirm the program appears on coursera.org as a degree, not just a certificate."
    )

    # 2.3 Field_CS_AI_or_DS
    field_node = evaluator.add_leaf(
        id="Field_CS_AI_or_DS",
        desc="The program is in Computer Science, Artificial Intelligence, or Data Science.",
        parent=constraints_node,
        critical=True
    )
    field_text = extraction.field_of_study or "Computer Science / Artificial Intelligence / Data Science"
    field_claim = (
        f"The program's domain is '{field_text}', which belongs to Computer Science, Artificial Intelligence, or Data Science."
    )
    await evaluator.verify(
        claim=field_claim,
        node=field_node,
        sources=all_sources,
        additional_instruction="Look for explicit mentions like 'Computer Science', 'Data Science', 'Artificial Intelligence', or close synonyms (e.g., 'machine learning' under AI)."
    )

    # 2.4 Performance_Based_Admission
    pba_node = evaluator.add_leaf(
        id="Performance_Based_Admission",
        desc="The program offers performance-based admission (e.g., allows enrollment without a traditional application or without requiring a bachelor's degree upfront).",
        parent=constraints_node,
        critical=True
    )
    pba_claim = (
        "This program offers performance-based admission, such as allowing enrollment without a traditional application or without requiring a bachelor's degree upfront; "
        "it may allow non-degree pathway/gateway courses to qualify for degree admission."
    )
    await evaluator.verify(
        claim=pba_claim,
        node=pba_node,
        sources=all_sources,
        additional_instruction="Look for phrases like 'performance-based admission', 'no application required', 'no bachelor's required upfront', "
                              "'complete for-credit pathway/gateway courses to qualify', or similar policy on the official or Coursera page."
    )

    # 2.5 University_in_Colorado
    colorado_node = evaluator.add_leaf(
        id="University_in_Colorado",
        desc="The university offering the program is located in Colorado.",
        parent=constraints_node,
        critical=True
    )
    if extraction.university_name:
        colorado_claim = f"The university '{extraction.university_name}' is located in the U.S. state of Colorado."
    else:
        colorado_claim = "The university offering this program is located in the U.S. state of Colorado."
    await evaluator.verify(
        claim=colorado_claim,
        node=colorado_node,
        sources=all_sources,
        additional_instruction="Accept evidence like 'University of Colorado Boulder/UCCS/Denver' or explicit addresses in Colorado. "
                              "If the page explicitly shows the university name that clearly indicates Colorado, that counts."
    )

    # 2.6 Application_Deadline_Feb_2026
    deadline_node = evaluator.add_leaf(
        id="Application_Deadline_Feb_2026",
        desc="The program has an application deadline in February 2026.",
        parent=constraints_node,
        critical=True
    )
    deadline_claim = (
        "This program has at least one application or enrollment deadline in February 2026 (any day in Feb 2026)."
    )
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_node,
        sources=all_sources,
        additional_instruction="Confirm any 'apply by', 'enroll by', 'priority deadline', or 'application close date' falling in February 2026. "
                              "Focus on the month and year (February 2026); specific day is not required."
    )

    # Record some helpful custom info
    evaluator.add_custom_info(
        info={
            "extracted_program_name": extraction.program_name,
            "extracted_university": extraction.university_name,
            "extracted_field": extraction.field_of_study,
            "extracted_state": extraction.location_state,
            "total_sources": len(all_sources),
            "sources": all_sources,
            "notes": "Additional info recorded to aid debugging of verification."
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
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
    Entry point for evaluating an answer for the Coursera Colorado program identification task.
    """
    # 1. Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root can be parallel; the key gate is the Program_Identification node
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

    # 2. Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction"
    )

    # 3. Build verification tree and run checks
    await build_verification_tree(evaluator, extraction)

    # 4. Return structured evaluation summary
    return evaluator.get_summary()