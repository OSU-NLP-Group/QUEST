import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "fincen_residential_real_estate_rule_start_date"
TASK_DESCRIPTION = (
    "What is the effective start date when reporting persons must begin filing reports to FinCEN under the "
    "Residential Real Estate Rule for non-financed transfers of residential real estate to legal entities or trusts? "
    "Provide the specific date and include a reference URL from an official FinCEN source."
)

EXPECTED_START_DATE = "March 1, 2026"


class RuleAnswerExtraction(BaseModel):
    effective_start_date_text: Optional[str] = None
    fincen_urls: List[str] = Field(default_factory=list)


def prompt_extract_rule_answer() -> str:
    return """
    Extract from the answer the following:
    1) effective_start_date_text: The specific date text the answer claims as the effective reporting start date under FinCEN's Residential Real Estate Rule (e.g., "March 1, 2026"). If the answer does not provide a date, return null.
    2) fincen_urls: A list of all official FinCEN URLs explicitly shown in the answer. Only include URLs whose domain contains "fincen.gov" (case-insensitive). Do not include non-FinCEN domains. Extract actual URLs (including markdown links). Deduplicate and trim whitespace/punctuation.

    If any required field is missing, return null for that field or an empty list for the URLs.
    """


async def verify_fincen_rule(
    evaluator: Evaluator,
    parent_node,
    extraction: RuleAnswerExtraction,
) -> None:
    # Create the critical main node for this evaluation
    main_node = evaluator.add_parallel(
        id="FinCEN_Residential_Real_Estate_Reporting_Start_Date",
        desc="Evaluate whether the answer satisfies all stated constraints about the Residential Real Estate Rule reporting start date and required citation.",
        parent=parent_node,
        critical=True,
    )

    # 1) FinCEN official reference URL existence (critical, custom node)
    has_fincen_url = any(isinstance(u, str) and ("fincen.gov" in u.lower()) for u in (extraction.fincen_urls or []))

    ref_node = evaluator.add_custom_node(
        result=has_fincen_url,
        id="FinCEN_Official_Reference_URL",
        desc="Provides at least one supporting reference URL from FinCEN.gov.",
        parent=main_node,
        critical=True,
    )

    # Sources for subsequent URL-based verifications
    sources_list = extraction.fincen_urls if extraction and extraction.fincen_urls else []

    # 2) Effective start date check (critical, verify by FinCEN URLs)
    effective_node = evaluator.add_leaf(
        id="Effective_Start_Date",
        desc=f"States the effective reporting start date as {EXPECTED_START_DATE}.",
        parent=main_node,
        critical=True,
    )
    effective_claim = (
        f"Under FinCEN's Residential Real Estate Rule for non-financed transfers of residential real estate, "
        f"reporting persons must begin filing reports on {EXPECTED_START_DATE}."
    )
    await evaluator.verify(
        claim=effective_claim,
        node=effective_node,
        sources=sources_list,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Verify on the provided FinCEN page(s) whether the effective reporting start date for reporting persons "
            f"is explicitly stated as {EXPECTED_START_DATE}. Accept synonymous phrasing like 'effective on', "
            "'start date', or 'begin filing'. If the page is not an official FinCEN page or does not explicitly support "
            "the date, mark as not supported."
        ),
    )

    # 3) Applicability to professionals involved in closings/settlements (critical, verify by URLs)
    professionals_node = evaluator.add_leaf(
        id="Applicability_Professionals",
        desc="Indicates the reporting requirement applies to certain professionals involved in real estate closings and settlements.",
        parent=main_node,
        critical=True,
    )
    professionals_claim = (
        "The Residential Real Estate Rule identifies 'reporting persons' as certain professionals involved in "
        "real estate closings and settlements (e.g., settlement agents, closing agents, title companies, or escrow agents)."
    )
    await evaluator.verify(
        claim=professionals_claim,
        node=professionals_node,
        sources=sources_list,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Check the FinCEN page(s) for descriptions of 'reporting persons' tied to professionals handling real estate "
            "closings/settlements. Accept reasonable synonyms (e.g., settlement/closing agents, title companies, escrow agents). "
            "If unclear or not mentioned on the page, mark as not supported."
        ),
    )

    # 4) Scope: non-financed transfers to legal entities or trusts (critical, verify by URLs)
    scope_node = evaluator.add_leaf(
        id="Scope_NonFinanced_Transfers_To_Entities_Or_Trusts",
        desc="Indicates the reports cover non-financed transfers of residential real estate to legal entities or trusts.",
        parent=main_node,
        critical=True,
    )
    scope_claim = (
        "The Residential Real Estate Rule covers non-financed transfers of residential real estate to legal entities or trusts."
    )
    await evaluator.verify(
        claim=scope_claim,
        node=scope_node,
        sources=sources_list,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            "Verify that the FinCEN page(s) explicitly state the scope as non-financed transfers of residential real estate "
            "to legal entities or trusts. If the scope is described differently or is not clearly stated, mark as not supported."
        ),
    )

    # 5) Exemption for transfers closing before the effective date (critical, verify by URLs)
    exemption_node = evaluator.add_leaf(
        id="Pre_Effective_Date_Exemption",
        desc=f"States that transfers closing before {EXPECTED_START_DATE} are exempt from the reporting requirement.",
        parent=main_node,
        critical=True,
    )
    exemption_claim = (
        f"Transfers that close before {EXPECTED_START_DATE} are exempt from the Residential Real Estate Rule's reporting requirement."
    )
    await evaluator.verify(
        claim=exemption_claim,
        node=exemption_node,
        sources=sources_list,
        extra_prerequisites=[ref_node],
        additional_instruction=(
            f"Check the FinCEN page(s) for an explicit exemption stating that transfers closing before {EXPECTED_START_DATE} "
            "are not subject to the reporting requirement. Accept equivalent phrases like 'prior to the effective date' or "
            "'completed before the effective date are not reportable.' If the page does not confirm this, mark as not supported."
        ),
    )


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

    extraction = await evaluator.extract(
        prompt=prompt_extract_rule_answer(),
        template_class=RuleAnswerExtraction,
        extraction_name="rule_answer_extraction",
    )

    evaluator.add_ground_truth(
        {
            "expected_effective_start_date": EXPECTED_START_DATE,
            "required_source_domain": "fincen.gov",
        },
        gt_type="ground_truth",
    )

    evaluator.add_custom_info(
        {
            "provided_fincen_urls": extraction.fincen_urls,
            "fincen_url_count": len(extraction.fincen_urls),
            "extracted_effective_date_text": extraction.effective_start_date_text,
        },
        info_type="extraction_info",
        info_name="parsed_answer_fields",
    )

    await verify_fincen_rule(evaluator, root, extraction)

    return evaluator.get_summary()