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
TASK_ID = "operation_absolute_resolve_info"
TASK_DESCRIPTION = "On what date did Operation Absolute Resolve take place, and who became Venezuela's acting president following the operation?"

EXPECTED_OPERATION_DATE = "January 3, 2026"
EXPECTED_ACTING_PRESIDENT = "Delcy Rodríguez"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OperationInfoExtraction(BaseModel):
    """
    Extracted structured information from the agent's answer about:
    - Operation Absolute Resolve date
    - Acting president following the operation
    - URL sources cited for each claim (and general sources if not clearly tied)
    """
    operation_date: Optional[str] = None
    acting_president: Optional[str] = None
    operation_date_sources: List[str] = Field(default_factory=list)
    acting_president_sources: List[str] = Field(default_factory=list)
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_operation_info() -> str:
    return """
    Extract the information provided in the answer about "Operation Absolute Resolve".

    Return a JSON object with the following fields:
    - operation_date: The date explicitly stated in the answer for when Operation Absolute Resolve took place (string). If multiple formats are present (e.g., "January 3, 2026" vs "2026-01-03"), pick the most explicit one. If not mentioned, return null.
    - acting_president: The person stated in the answer as who became Venezuela's acting president following the operation (string). If not mentioned, return null.

    Also extract URLs cited in the answer:
    - operation_date_sources: All URLs explicitly associated with (or immediately following) the operation date statement.
    - acting_president_sources: All URLs explicitly associated with (or immediately following) the acting president statement.
    - general_sources: Any other URLs mentioned in the answer that are not clearly tied to one of the above claims.

    Notes and rules:
    - Do not invent any URLs. Extract only URLs actually present in the answer (including plain URLs, markdown links, or footnote-style listings).
    - If a URL lacks a protocol, prepend "http://" as needed.
    - If no URLs are present for a category, return an empty array for that category.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _pick_sources(primary: List[str], fallback: List[str]) -> List[str]:
    seq = primary if primary else (fallback if fallback else [])
    seen = set()
    out = []
    for u in seq:
        if not _non_empty_str(u):
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_operation_date(
    evaluator: Evaluator,
    parent_node,
    extracted: OperationInfoExtraction,
) -> None:
    """
    Build and verify the 'Operation_Date' subtree:
    - Check date provided
    - Check it matches the expected date (format-insensitive)
    - Check sources provided
    - Verify sources support the claimed date
    """
    op_date_node = evaluator.add_sequential(
        id="Operation_Date",
        desc="Correctly identify the date when Operation Absolute Resolve took place as January 3, 2026",
        parent=parent_node,
        critical=True,
    )

    # 1) Existence of date in the answer (critical)
    date_provided = evaluator.add_custom_node(
        result=_non_empty_str(extracted.operation_date),
        id="Operation_Date_Provided",
        desc="Operation date is provided in the answer",
        parent=op_date_node,
        critical=True,
    )

    # 2) Value matches expected date (critical)
    date_match_node = evaluator.add_leaf(
        id="Operation_Date_Value_Match",
        desc=f"The provided operation date matches {EXPECTED_OPERATION_DATE}",
        parent=op_date_node,
        critical=True,
    )
    provided_date = extracted.operation_date or ""
    date_equivalence_claim = (
        f"The date '{provided_date}' refers to the same calendar date as '{EXPECTED_OPERATION_DATE}'. "
        f"Treat equivalent formats (e.g., '3 January 2026', '2026-01-03', 'Jan 3, 2026') as the same date."
    )
    await evaluator.verify(
        claim=date_equivalence_claim,
        node=date_match_node,
        additional_instruction=(
            "Judge equivalence of dates ignoring formatting, language variants, or presence/absence of commas. "
            "Focus only on whether both strings denote the same calendar date."
        ),
    )

    # 3) Sources provided (critical, to enforce source-grounding)
    date_sources = _pick_sources(extracted.operation_date_sources, extracted.general_sources)
    date_sources_provided_node = evaluator.add_custom_node(
        result=len(date_sources) > 0,
        id="Operation_Date_Sources_Provided",
        desc="Sources are provided for the operation date claim",
        parent=op_date_node,
        critical=True,
    )

    # 4) Sources support the claim (critical)
    date_source_support_node = evaluator.add_leaf(
        id="Operation_Date_Source_Support",
        desc="Sources support the stated operation date",
        parent=op_date_node,
        critical=True,
    )
    support_claim = f"Operation Absolute Resolve took place on {provided_date}."
    await evaluator.verify(
        claim=support_claim,
        node=date_source_support_node,
        sources=date_sources,
        additional_instruction=(
            "Verify that the webpage explicitly supports the date of Operation Absolute Resolve. "
            "The page should clearly indicate the operation occurred on the stated date. "
            "If the page is irrelevant or does not support the date, mark as not supported."
        ),
    )


async def verify_acting_president(
    evaluator: Evaluator,
    parent_node,
    extracted: OperationInfoExtraction,
) -> None:
    """
    Build and verify the 'Acting_President_Identity' subtree:
    - Check name provided
    - Check it matches expected (allow diacritics and name variants)
    - Check sources provided
    - Verify sources support the acting president claim
    """
    acting_node = evaluator.add_sequential(
        id="Acting_President_Identity",
        desc="Correctly identify Delcy Rodríguez as the person who became Venezuela's acting president following the operation",
        parent=parent_node,
        critical=True,
    )

    # 1) Existence of acting president in the answer (critical)
    pres_provided = evaluator.add_custom_node(
        result=_non_empty_str(extracted.acting_president),
        id="Acting_President_Provided",
        desc="Acting president is provided in the answer",
        parent=acting_node,
        critical=True,
    )

    # 2) Value matches expected identity (critical)
    pres_match_node = evaluator.add_leaf(
        id="Acting_President_Value_Match",
        desc=f"The provided acting president matches {EXPECTED_ACTING_PRESIDENT}",
        parent=acting_node,
        critical=True,
    )
    provided_pres = extracted.acting_president or ""
    name_equivalence_claim = (
        f"The person '{provided_pres}' refers to the same individual as '{EXPECTED_ACTING_PRESIDENT}'. "
        f"Allow diacritics, full names (e.g., 'Delcy Eloína Rodríguez'), case differences, or minor spelling variants."
    )
    await evaluator.verify(
        claim=name_equivalence_claim,
        node=pres_match_node,
        additional_instruction=(
            "Judge whether the two names refer to the same person. "
            "Allow diacritics, middle names, and capitalization differences."
        ),
    )

    # 3) Sources provided (critical, to enforce source-grounding)
    acting_sources = _pick_sources(extracted.acting_president_sources, extracted.general_sources)
    pres_sources_provided_node = evaluator.add_custom_node(
        result=len(acting_sources) > 0,
        id="Acting_President_Sources_Provided",
        desc="Sources are provided for the acting president claim",
        parent=acting_node,
        critical=True,
    )

    # 4) Sources support the claim (critical)
    pres_source_support_node = evaluator.add_leaf(
        id="Acting_President_Source_Support",
        desc="Sources support the stated acting president following the operation",
        parent=acting_node,
        critical=True,
    )
    support_pres_claim = (
        f"Following Operation Absolute Resolve, {provided_pres} became Venezuela's acting president."
    )
    await evaluator.verify(
        claim=support_pres_claim,
        node=pres_source_support_node,
        sources=acting_sources,
        additional_instruction=(
            "Verify that the webpage clearly states that this person became Venezuela's acting (or interim) president "
            "following Operation Absolute Resolve. If the page is irrelevant or does not support this claim, mark as not supported."
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
) -> Dict:
    """
    Evaluate an answer for the Operation Absolute Resolve task.
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

    # Extract information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_operation_info(),
        template_class=OperationInfoExtraction,
        extraction_name="operation_info_extraction",
    )

    # Add ground truth to summary
    evaluator.add_ground_truth({
        "expected_operation_date": EXPECTED_OPERATION_DATE,
        "expected_acting_president": EXPECTED_ACTING_PRESIDENT,
    })

    # Parent node reflecting the rubric's top-level item
    main_node = evaluator.add_parallel(
        id="Operation_Absolute_Resolve_Information",
        desc="Provide accurate information about Operation Absolute Resolve, including the date of the operation and the identity of Venezuela's acting president following the operation",
        parent=root,
        critical=False,
    )

    # Build subtrees
    await verify_operation_date(evaluator, main_node, extracted)
    await verify_acting_president(evaluator, main_node, extracted)

    # Add small custom info about source counts
    evaluator.add_custom_info(
        {
            "operation_date_sources_count": len(extracted.operation_date_sources or []),
            "acting_president_sources_count": len(extracted.acting_president_sources or []),
            "general_sources_count": len(extracted.general_sources or []),
        },
        info_type="stats",
        info_name="source_counts"
    )

    # Return structured summary
    return evaluator.get_summary()