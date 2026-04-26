import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "maduro_hearing_date_2026"
TASK_DESCRIPTION = (
    "According to publicly available information about the January 2026 US military operation that captured "
    "Venezuelan President Nicolás Maduro, what date did US District Judge Alvin Hellerstein set for Maduro's next court hearing "
    "following his initial arraignment in Manhattan federal court?"
)

EXPECTED_OPERATION_DATE = "January 3, 2026"
EXPECTED_ARRAIGNMENT_DATE = "January 5, 2026"
EXPECTED_COURT_LOCATION = "Manhattan federal court (Southern District of New York)"
EXPECTED_JUDGE_NAME = "US District Judge Alvin Hellerstein"
EXPECTED_HEARING_DATE = "March 17, 2026"


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class OperationInfo(BaseModel):
    date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ArraignmentInfo(BaseModel):
    date: Optional[str] = None
    location: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class JudgeInfo(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OrderInfo(BaseModel):
    statement: Optional[str] = None  # e.g., "Judge Hellerstein ordered Maduro to be held pending further proceedings"
    sources: List[str] = Field(default_factory=list)


class NextHearingInfo(BaseModel):
    date: Optional[str] = None  # e.g., "March 17" or "March 17, 2026"
    sources: List[str] = Field(default_factory=list)


class MaduroCaseExtraction(BaseModel):
    operation: Optional[OperationInfo] = None
    arraignment: Optional[ArraignmentInfo] = None
    judge: Optional[JudgeInfo] = None
    order: Optional[OrderInfo] = None
    next_hearing: Optional[NextHearingInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_maduro_case() -> str:
    return """
    Extract from the answer the following structured information related to Nicolás Maduro's capture operation and subsequent court proceedings.
    Return a single JSON object with these nested objects and fields:

    - operation:
        - date: The stated calendar date of the US military operation in Venezuela that resulted in Maduro's capture, as written in the answer (e.g., "January 3, 2026"). If not stated, null.
        - sources: All URLs in the answer that support identifying this operation and/or its date. Return as an array of URLs (can be markdown links); if none, return [].

    - arraignment:
        - date: The stated date of Maduro's initial arraignment as written in the answer (e.g., "January 5, 2026"). If not stated, null.
        - location: The stated location of the arraignment as written (e.g., "Manhattan federal court" or "Southern District of New York"). If not stated, null.
        - sources: All URLs in the answer supporting the arraignment date/location. Return as an array; if none, [].

    - judge:
        - name: The stated judge who presided over the initial proceedings (e.g., "US District Judge Alvin Hellerstein"). If not stated, null.
        - sources: All URLs in the answer supporting the judge identification. If none, [].

    - order:
        - statement: The answer's statement summarizing any detention order (e.g., "Judge Hellerstein ordered Maduro to be held pending further proceedings"). If none, null.
        - sources: All URLs in the answer supporting the detention order. If none, [].

    - next_hearing:
        - date: The stated date of the next scheduled hearing following the initial arraignment (e.g., "March 17" or "March 17, 2026"). If not stated, null.
        - sources: All URLs in the answer supporting the next-hearing date. If none, [].

    IMPORTANT:
    - Extract exactly what the answer explicitly states; do not infer or invent text.
    - For URLs, include only actual URLs present in the answer (including markdown links). If a URL is missing the protocol, prepend "http://".
    - If a given field is not present, set it to null (for strings) or [] (for URLs).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_operation_group(evaluator: Evaluator, parent_node, info: Optional[OperationInfo]) -> None:
    node = evaluator.add_parallel(
        id="operation_identification",
        desc="Correctly identify the US military operation in Venezuela that resulted in Maduro's capture",
        parent=parent_node,
        critical=True
    )

    # Leaf: operation_date (answer-level statement check)
    leaf_date = evaluator.add_leaf(
        id="operation_date",
        desc="State that the operation occurred on January 3, 2026",
        parent=node,
        critical=True
    )
    claim_date = "According to the answer, the operation occurred on January 3, 2026."
    await evaluator.verify(
        claim=claim_date,
        node=leaf_date,
        additional_instruction="Look only at the provided answer context to verify if it explicitly states Jan 3, 2026 as the operation date."
    )

    # Leaf: operation_reference (source-backed verification)
    leaf_ref = evaluator.add_leaf(
        id="operation_reference",
        desc="Provide URL reference supporting the operation identification",
        parent=node,
        critical=True
    )
    op_sources = info.sources if info and info.sources else []
    if len(op_sources) == 0:
        leaf_ref.score = 0.0
        leaf_ref.status = "failed"
    else:
        ref_claim = "Public sources at the provided URLs report that the operation that captured Nicolás Maduro occurred on January 3, 2026."
        await evaluator.verify(
            claim=ref_claim,
            node=leaf_ref,
            sources=op_sources,
            additional_instruction="Confirm that at least one provided URL explicitly mentions the operation and the date January 3, 2026."
        )


async def verify_arraignment_group(evaluator: Evaluator, parent_node, info: Optional[ArraignmentInfo]) -> None:
    node = evaluator.add_parallel(
        id="arraignment_identification",
        desc="Correctly identify when and where Maduro's initial arraignment took place",
        parent=parent_node,
        critical=True
    )

    # Leaf: arraignment_date (answer-level)
    leaf_date = evaluator.add_leaf(
        id="arraignment_date",
        desc="State that the arraignment occurred on January 5, 2026",
        parent=node,
        critical=True
    )
    claim_adate = "According to the answer, Nicolás Maduro's initial arraignment occurred on January 5, 2026."
    await evaluator.verify(
        claim=claim_adate,
        node=leaf_date,
        additional_instruction="Verify within the answer text only."
    )

    # Leaf: court_location (answer-level)
    leaf_loc = evaluator.add_leaf(
        id="court_location",
        desc="State that the arraignment took place in Manhattan federal court (Southern District of New York)",
        parent=node,
        critical=True
    )
    claim_loc = "According to the answer, the arraignment took place in Manhattan federal court (Southern District of New York)."
    await evaluator.verify(
        claim=claim_loc,
        node=leaf_loc,
        additional_instruction="Allow synonymous phrasing like 'Manhattan federal court', 'SDNY', or 'Southern District of New York'."
    )

    # Leaf: arraignment_reference (source-backed)
    leaf_ref = evaluator.add_leaf(
        id="arraignment_reference",
        desc="Provide URL reference supporting the arraignment information",
        parent=node,
        critical=True
    )
    arr_sources = info.sources if info and info.sources else []
    if len(arr_sources) == 0:
        leaf_ref.score = 0.0
        leaf_ref.status = "failed"
    else:
        ref_claim = "The provided sources state that Nicolás Maduro's initial arraignment occurred on January 5, 2026 in Manhattan federal court (Southern District of New York)."
        await evaluator.verify(
            claim=ref_claim,
            node=leaf_ref,
            sources=arr_sources,
            additional_instruction="Confirm both the date (Jan 5, 2026) and the SDNY/Manhattan federal court location are supported."
        )


async def verify_judge_group(evaluator: Evaluator, parent_node, info: Optional[JudgeInfo]) -> None:
    node = evaluator.add_parallel(
        id="judge_identification",
        desc="Correctly identify the federal judge who presided over the initial court proceedings",
        parent=parent_node,
        critical=True
    )

    # Leaf: judge_name (answer-level)
    leaf_jname = evaluator.add_leaf(
        id="judge_name",
        desc="State that US District Judge Alvin Hellerstein presided over the hearing",
        parent=node,
        critical=True
    )
    claim_jname = "According to the answer, US District Judge Alvin Hellerstein presided over the hearing."
    await evaluator.verify(
        claim=claim_jname,
        node=leaf_jname,
        additional_instruction="Allow minor variants such as 'Alvin K. Hellerstein' or different punctuation/casing."
    )

    # Leaf: judge_reference (source-backed)
    leaf_ref = evaluator.add_leaf(
        id="judge_reference",
        desc="Provide URL reference supporting the judge identification",
        parent=node,
        critical=True
    )
    j_sources = info.sources if info and info.sources else []
    if len(j_sources) == 0:
        leaf_ref.score = 0.0
        leaf_ref.status = "failed"
    else:
        ref_claim = "The provided sources confirm that US District Judge Alvin Hellerstein presided over Nicolás Maduro's initial court proceedings."
        await evaluator.verify(
            claim=ref_claim,
            node=leaf_ref,
            sources=j_sources,
            additional_instruction="Treat 'Alvin K. Hellerstein' as the same judge; confirm presiding role over the initial arraignment/hearing."
        )


async def verify_order_group(evaluator: Evaluator, parent_node, info: Optional[OrderInfo]) -> None:
    node = evaluator.add_parallel(
        id="judicial_order_identification",
        desc="Correctly identify the judicial order regarding Maduro's detention",
        parent=parent_node,
        critical=True
    )

    # Leaf: detention_order (answer-level)
    leaf_order = evaluator.add_leaf(
        id="detention_order",
        desc="State that Judge Hellerstein ordered Maduro to be held pending further proceedings",
        parent=node,
        critical=True
    )
    claim_order = "According to the answer, Judge Alvin Hellerstein ordered Nicolás Maduro to be held pending further proceedings."
    await evaluator.verify(
        claim=claim_order,
        node=leaf_order,
        additional_instruction="Focus on whether the answer states detention pending further proceedings; allow phrasing like 'remanded' or 'detained'."
    )

    # Leaf: order_reference (source-backed)
    leaf_ref = evaluator.add_leaf(
        id="order_reference",
        desc="Provide URL reference supporting the detention order information",
        parent=node,
        critical=True
    )
    o_sources = info.sources if info and info.sources else []
    if len(o_sources) == 0:
        leaf_ref.score = 0.0
        leaf_ref.status = "failed"
    else:
        ref_claim = "The provided sources confirm that Judge Alvin Hellerstein ordered Nicolás Maduro held pending further proceedings."
        await evaluator.verify(
            claim=ref_claim,
            node=leaf_ref,
            sources=o_sources,
            additional_instruction="Accept synonymous language like 'remanded to custody', 'detained pending proceedings', or 'held without bail'."
        )


async def verify_hearing_group(evaluator: Evaluator, parent_node, info: Optional[NextHearingInfo]) -> None:
    node = evaluator.add_parallel(
        id="next_hearing_date",
        desc="Correctly state that the next scheduled hearing was set for March 17",
        parent=parent_node,
        critical=True
    )

    # Leaf: hearing_date_value (answer-level)
    leaf_value = evaluator.add_leaf(
        id="hearing_date_value",
        desc="Provide the specific date: March 17 (with the year 2026 understood from context)",
        parent=node,
        critical=True
    )
    claim_hdate = "According to the answer, the next scheduled hearing was set for March 17, 2026."
    await evaluator.verify(
        claim=claim_hdate,
        node=leaf_value,
        additional_instruction="Treat 'March 17' without the year as correct if context clearly concerns 2026."
    )

    # Leaf: hearing_date_reference (source-backed)
    leaf_ref = evaluator.add_leaf(
        id="hearing_date_reference",
        desc="Provide URL reference supporting the March 17 hearing date",
        parent=node,
        critical=True
    )
    h_sources = info.sources if info and info.sources else []
    if len(h_sources) == 0:
        leaf_ref.score = 0.0
        leaf_ref.status = "failed"
    else:
        ref_claim = "The provided sources confirm that Judge Alvin Hellerstein set Nicolás Maduro's next court hearing for March 17, 2026."
        await evaluator.verify(
            claim=ref_claim,
            node=leaf_ref,
            sources=h_sources,
            additional_instruction="Confirm that the date is March 17 (year 2026 in context), set by Judge Hellerstein following the initial arraignment."
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
        default_model=model
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_maduro_case(),
        template_class=MaduroCaseExtraction,
        extraction_name="maduro_case_extraction"
    )

    # Add ground-truth context for transparency (not used for direct scoring)
    evaluator.add_ground_truth({
        "expected_operation_date": EXPECTED_OPERATION_DATE,
        "expected_arraignment_date": EXPECTED_ARRAIGNMENT_DATE,
        "expected_court_location": EXPECTED_COURT_LOCATION,
        "expected_judge_name": EXPECTED_JUDGE_NAME,
        "expected_hearing_date": EXPECTED_HEARING_DATE
    }, gt_type="expected_facts")

    # Build verification tree according to rubric (root sequential)
    # 1) Operation identification
    await verify_operation_group(evaluator, root, extraction.operation if extraction else None)

    # 2) Arraignment identification
    await verify_arraignment_group(evaluator, root, extraction.arraignment if extraction else None)

    # 3) Judge identification
    await verify_judge_group(evaluator, root, extraction.judge if extraction else None)

    # 4) Judicial order identification
    await verify_order_group(evaluator, root, extraction.order if extraction else None)

    # 5) Next hearing date
    await verify_hearing_group(evaluator, root, extraction.next_hearing if extraction else None)

    return evaluator.get_summary()