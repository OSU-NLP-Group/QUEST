import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "anthropic_bfi_partnership_2025"
TASK_DESCRIPTION = (
    "In 2025, Anthropic announced a research partnership with the University of Chicago's Becker Friedman Institute for Economics "
    "to study AI's impact on labor markets and the economy. Please provide the following information about this partnership: "
    "(1) the exact date when the partnership was publicly announced, and (2) the specific academic year for which this partnership was established."
)


class PartnershipExtraction(BaseModel):
    partner_private: Optional[str] = None
    partner_academic: Optional[str] = None
    announcement_date: Optional[str] = None
    academic_year: Optional[str] = None
    scope_summary: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


def prompt_extract_partnership_fields() -> str:
    return (
        "Extract the following fields explicitly as they appear in the provided answer text about the Anthropic–University of Chicago partnership:\n"
        "1) partner_private: The private organization partner name (e.g., 'Anthropic').\n"
        "2) partner_academic: The academic partner entity name (e.g., 'University of Chicago's Becker Friedman Institute for Economics', 'BFI at UChicago').\n"
        "3) announcement_date: The exact public announcement date for the partnership (keep the original formatting as stated in the answer).\n"
        "4) academic_year: The specific academic year the partnership was established for (e.g., '2025–26', '2025-2026', 'AY 2025-26').\n"
        "5) scope_summary: A brief phrase summarizing the stated scope (e.g., 'study AI’s impact on labor markets and the economy').\n"
        "6) sources: A list of all URLs cited in the answer that are intended to support the announcement date and/or the academic year. Only include actual URLs explicitly present in the answer (plain or markdown links). If none are provided, return an empty list.\n"
        "Return these in a single JSON object. Do not invent or infer missing information; use null for any field not present in the answer."
    )


def filter_official_sources(urls: List[str]) -> List[str]:
    official_domains = ["anthropic.com", "uchicago.edu", "bfi.uchicago.edu"]
    filtered = []
    for u in urls:
        if not isinstance(u, str):
            continue
        if any(dom in u for dom in official_domains):
            filtered.append(u)
    return filtered


async def build_verification_tree(evaluator: Evaluator, extracted: PartnershipExtraction) -> None:
    main_node = evaluator.add_parallel(
        id="Anthropic_University_Chicago_Partnership",
        desc="Provides the announcement date and the academic year for Anthropic's research partnership with the University of Chicago's Becker Friedman Institute for Economics, with official-source support.",
        parent=evaluator.root,
        critical=True
    )

    all_sources = extracted.sources or []
    official_sources = filter_official_sources(all_sources)

    official_node = evaluator.add_custom_node(
        result=bool(official_sources),
        id="Official_Source_Verifiability",
        desc="Includes citation(s) or reference link(s) to official source(s) that verify the stated announcement date and academic year.",
        parent=main_node,
        critical=True
    )

    scope_node = evaluator.add_leaf(
        id="Partnership_Scope_Match",
        desc="Information provided pertains specifically to the research partnership between Anthropic and the University of Chicago's Becker Friedman Institute for Economics (not a different partnership/entity).",
        parent=main_node,
        critical=True
    )
    scope_claim = (
        "The cited source(s) explicitly describe a research partnership between Anthropic and the University of Chicago's Becker Friedman Institute for Economics (BFI) "
        "to study AI's impact on labor markets and the economy."
    )
    await evaluator.verify(
        claim=scope_claim,
        node=scope_node,
        sources=official_sources if official_sources else all_sources,
        additional_instruction=(
            "Confirm the page(s) clearly indicate a partnership between Anthropic and BFI at UChicago. "
            "Allow reasonable naming variants such as 'Becker Friedman Institute', 'BFI', or 'UChicago'. "
            "The page(s) should reference research on AI's impact on labor markets/the economy. "
            "If no URL is provided, rely strictly on the claim and answer text; do not use outside knowledge."
        )
    )

    announcement_node = evaluator.add_leaf(
        id="Announcement_Date_Provided",
        desc="States the exact public announcement date of the partnership.",
        parent=main_node,
        critical=True
    )
    announcement_date_text = extracted.announcement_date or ""
    announcement_claim = (
        f"The public announcement date of the partnership between Anthropic and the University of Chicago's Becker Friedman Institute for Economics is {announcement_date_text}."
    )
    await evaluator.verify(
        claim=announcement_claim,
        node=announcement_node,
        sources=official_sources if official_sources else all_sources,
        additional_instruction=(
            "Verify the announcement date based on the page's published/announcement date (in header, metadata, or body). "
            "Treat date formats like 'Jan 8, 2025' and 'January 8, 2025' as equivalent. "
            "If the claim contains an empty or placeholder date (e.g., '', 'null', 'None'), consider it unsupported. "
            "Prefer official Anthropic or UChicago/BFI pages where available."
        )
    )

    academic_year_node = evaluator.add_leaf(
        id="Academic_Year_Provided",
        desc="States the specific academic year for which the partnership was established.",
        parent=main_node,
        critical=True
    )
    academic_year_text = extracted.academic_year or ""
    academic_year_claim = (
        f"The partnership was established for the {academic_year_text} academic year."
    )
    await evaluator.verify(
        claim=academic_year_claim,
        node=academic_year_node,
        sources=official_sources if official_sources else all_sources,
        additional_instruction=(
            "Confirm that the page(s) explicitly state the academic year of the partnership. "
            "Accept equivalent notations such as '2025–26', '2025-26', '2025-2026', or 'AY 2025-26'. "
            "If the claim contains an empty or placeholder year (e.g., '', 'null', 'None'), consider it unsupported. "
            "Prefer official Anthropic or UChicago/BFI pages where available."
        )
    )


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
        default_model=model
    )

    extracted = await evaluator.extract(
        prompt=prompt_extract_partnership_fields(),
        template_class=PartnershipExtraction,
        extraction_name="partnership_info"
    )

    evaluator.add_ground_truth({
        "expected_partners": [
            "Anthropic",
            "University of Chicago's Becker Friedman Institute for Economics (BFI)"
        ],
        "expected_scope": "Study AI's impact on labor markets and the economy"
    })

    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()