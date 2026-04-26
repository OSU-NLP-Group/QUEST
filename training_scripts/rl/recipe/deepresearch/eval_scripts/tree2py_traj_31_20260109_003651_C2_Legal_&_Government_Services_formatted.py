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
TASK_ID = "la_residential_ce_renewal"
TASK_DESCRIPTION = (
    "James Rodriguez holds both a Louisiana Residential Building Contractor license and a Louisiana commercial "
    "contractor license in Building Construction. His residential license is due for renewal in 45 days. "
    "Does James need to complete continuing education hours before renewing his residential license? If so, "
    "how many hours are required? Provide the regulatory source for your answer."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CEAnswerExtraction(BaseModel):
    """Structured extraction from the agent's answer regarding CE requirement and sources."""
    ce_required: Optional[bool] = None
    ce_hours: Optional[str] = None
    reason: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)
    source_texts: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_ce() -> str:
    return (
        "From the answer text, extract the following fields about whether continuing education (CE) is needed before "
        "renewing the Louisiana Residential Building Contractor license in the given scenario and the supporting source(s):\n"
        "1) ce_required: A boolean (true/false). Set to true only if the answer explicitly states CE hours are required "
        "before renewal; set to false only if the answer explicitly states CE is not required or that the requirement is "
        "satisfied/exempted. If unclear, set to null.\n"
        "2) ce_hours: The number of hours required as stated by the answer (use a string, exactly as written, such as '0', "
        "'0 hours', 'six (6)', etc.). If the answer implies no hours required, use '0' or '0 hours' as it appears; if not mentioned, set to null.\n"
        "3) reason: A brief sentence summarizing the stated reason, if any (e.g., that a qualifying Louisiana commercial "
        "contractor license in Building Construction satisfies/exempts the residential CE requirement). If not provided, set to null.\n"
        "4) source_urls: An array of all URLs the answer provides as regulatory/official sources supporting the CE outcome. "
        "Extract only actual URLs present in the answer (including URLs inside markdown links). If none, return an empty array.\n"
        "5) source_texts: An array of non-URL citations the answer provides that appear to be regulatory/official "
        "(e.g., Louisiana Administrative Code citations, statute numbers, official board rule or guidance titles) where no URL was given. "
        "If none, return an empty array.\n"
        "Return a single JSON object with these fields. Do not invent information."
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: CEAnswerExtraction,
) -> None:
    """
    Build the verification tree and run the checks according to the rubric.
    """

    # Root: CE requirement determination with regulatory support (Critical, Parallel)
    root_node = evaluator.add_parallel(
        id="CE_Requirement_Before_Renewal",
        desc="Determines whether James must complete CE before renewing his Louisiana Residential Building Contractor license, "
             "states the required hours for this scenario, and provides a regulatory/official source.",
        parent=evaluator.root,
        critical=True
    )

    # Sub-node: CE need and hours stated correctly (Critical, Parallel)
    need_hours_node = evaluator.add_parallel(
        id="CE_Need_And_Hours",
        desc="Correctly states whether CE is required before renewal in this scenario and how many hours are required.",
        parent=root_node,
        critical=True
    )

    # Existence check for CE fields (Critical)
    ce_fields_present = evaluator.add_custom_node(
        result=(extracted.ce_required is not None and bool(extracted.ce_hours and extracted.ce_hours.strip())),
        id="CE_Fields_Present",
        desc="The answer explicitly states whether CE is required and provides the number of hours.",
        parent=need_hours_node,
        critical=True
    )

    # Leaf: Answer states CE is NOT required (Critical)
    ce_need_is_no = evaluator.add_leaf(
        id="CE_Not_Required",
        desc="The answer states that CE is not required before renewing the residential license in this scenario.",
        parent=need_hours_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that continuing education is not required before renewing the Louisiana Residential Building Contractor license in this scenario.",
        node=ce_need_is_no,
        additional_instruction="Check the answer carefully for an explicit 'no CE required' or equivalent statement (e.g., 'exempt from CE'). Allow synonyms."
    )

    # Leaf: Answer states zero hours (Critical)
    ce_hours_zero = evaluator.add_leaf(
        id="CE_Hours_Zero",
        desc="The answer states that zero continuing education hours (0 hours) are required for the residential renewal.",
        parent=need_hours_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the required continuing education hours for the residential renewal are zero (0).",
        node=ce_hours_zero,
        additional_instruction="Accept equivalent phrasing such as '0', '0 hours', 'no CE hours needed'."
    )

    # Leaf: Reason references commercial Building Construction license satisfying/exempting residential CE (Critical)
    ce_reason_commercial = evaluator.add_leaf(
        id="CE_Reason_Commercial_Satisfies",
        desc="The answer explains that the qualifying Louisiana commercial contractor license in Building Construction satisfies/exempts the residential CE requirement.",
        parent=need_hours_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explains that because James holds a current Louisiana commercial contractor license in the Building Construction classification, the residential continuing education requirement is satisfied or exempted.",
        node=ce_reason_commercial,
        additional_instruction="Allow equivalent language like 'waives', 'fulfills', 'satisfies', or 'exempts the residential CE requirement'."
    )

    # Sub-node: Regulatory source support (Critical, Parallel)
    source_node = evaluator.add_parallel(
        id="Regulatory_Source",
        desc="Provides a verifiable regulatory/official source supporting the stated CE requirement outcome.",
        parent=root_node,
        critical=True
    )

    # Existence check: at least one source provided (URL or textual citation) (Critical)
    has_any_source = evaluator.add_custom_node(
        result=(len(extracted.source_urls) > 0 or len(extracted.source_texts) > 0),
        id="Source_Provided",
        desc="At least one regulatory/official source is provided (URL or specific citation).",
        parent=source_node,
        critical=True
    )

    # Leaf: Source is regulatory/official (Critical)
    source_is_official = evaluator.add_leaf(
        id="Source_Is_Official",
        desc="The provided source(s) are regulatory/official (statute, administrative code, or official licensing-board regulation/guidance).",
        parent=source_node,
        critical=True,
    )
    # If URLs exist, verify with URLs; otherwise verify that the answer includes official citation text
    official_claim = "This source is regulatory/official, such as a statute, administrative code provision, or official licensing-board regulation/guidance."
    if extracted.source_urls:
        await evaluator.verify(
            claim=official_claim,
            node=source_is_official,
            sources=extracted.source_urls,
            additional_instruction="Judge based on the content: government code, administrative rules, or official board/regulatory guidance qualifies as official."
        )
    else:
        await evaluator.verify(
            claim="The answer includes specific regulatory/official citation text (e.g., Louisiana Administrative Code section, statute number, or official board guidance title).",
            node=source_is_official,
            additional_instruction="Check the answer text for explicit official citations even if no URL is provided."
        )

    # Leaf: Source supports exemption rule (Critical)
    source_supports_exemption = evaluator.add_leaf(
        id="Source_Supports_Exemption",
        desc="The source confirms that a qualifying current Louisiana commercial Building Construction license satisfies/exempts the residential CE requirement.",
        parent=source_node,
        critical=True,
    )
    await evaluator.verify(
        claim="A qualifying, current Louisiana commercial contractor license in the Building Construction classification satisfies or exempts the continuing education requirement for the Louisiana Residential Building Contractor license renewal.",
        node=source_supports_exemption,
        sources=extracted.source_urls if extracted.source_urls else None,
        additional_instruction="Look for explicit language stating the residential CE obligation is satisfied/exempted if the individual holds a current commercial Building Construction license. Accept synonyms like 'waived', 'exempt', or 'fulfills requirement'."
    )

    # Leaf: Source supports zero hours required (Critical)
    source_supports_zero = evaluator.add_leaf(
        id="Source_Supports_Zero_Hours",
        desc="The source confirms that in this scenario, zero (0) CE hours are required for residential renewal.",
        parent=source_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Under the rule that the commercial Building Construction license satisfies/exempts the residential CE requirement, zero (0) continuing education hours are required for residential renewal in this scenario.",
        node=source_supports_zero,
        sources=extracted.source_urls if extracted.source_urls else None,
        additional_instruction="It is acceptable if the source states the exemption/satisfaction rather than '0 hours' verbatim, provided it logically implies no CE hours are required."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Louisiana Residential CE renewal scenario.
    """
    # Initialize evaluator (root parallel since the rubric criteria are independent checks under a critical root)
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_ce(),
        template_class=CEAnswerExtraction,
        extraction_name="ce_requirement_extraction",
    )

    # Optional: Add ground truth guidance for transparency (not used in scoring)
    evaluator.add_ground_truth({
        "expected_outcome_summary": "When an individual holds a current Louisiana commercial contractor license with the Building Construction classification, "
                                    "the residential continuing education requirement is satisfied/exempted, resulting in 0 hours required before residential renewal.",
        "notes": "Evaluation requires the answer to state that CE is not required (0 hours) and provide a regulatory/official source supporting that exemption rule."
    })

    # Add custom info for debugging
    evaluator.add_custom_info(
        info={
            "source_urls_count": len(extracted.source_urls),
            "source_texts_count": len(extracted.source_texts)
        },
        info_type="source_counts",
        info_name="provided_source_counts"
    )

    # Build and verify the tree
    await build_and_verify_tree(evaluator, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()