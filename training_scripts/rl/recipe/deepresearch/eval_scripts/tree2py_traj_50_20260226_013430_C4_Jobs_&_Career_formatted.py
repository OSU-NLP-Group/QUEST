import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_university_coop_eval"
TASK_DESCRIPTION = (
    "I am researching universities with strong cooperative education (co-op) programs to help guide my college decision. "
    "Identify a university in the United States that meets all of the following criteria:\n\n"
    "1. The co-op program achieves a post-graduation success rate of at least 90%, meaning that at least 90% of graduates are employed full-time or enrolled in graduate school within six months of graduation.\n\n"
    "2. The university facilitates at least 3,000 paid cooperative education experiences for students annually.\n\n"
    "3. Students participating in the co-op program collectively earn at least $50 million per year through their co-op placements.\n\n"
    "4. The university maintains partnerships with at least 1,000 employers for co-op placements, research positions, or project-based work.\n\n"
    "5. The co-op experiences are structured as full-time, multi-month professional work experiences, not short-term internships or part-time positions.\n\n"
    "6. The university is located in the United States.\n\n"
    "7. The employment outcomes and program statistics are based on current data from the 2023-2026 period.\n\n"
    "Provide the name of the university and include reference URLs that verify each of these requirements."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CoOpEvaluationExtraction(BaseModel):
    """
    Extracted information from the agent's answer regarding the target university and supporting references.
    All numeric/statements are kept as strings to maximize compatibility with varied answer formats.
    """
    university_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)

    # Free-text fields capturing what the answer claims (optional, may be null)
    success_rate_text: Optional[str] = None
    annual_experiences_text: Optional[str] = None
    annual_earnings_text: Optional[str] = None
    employer_network_text: Optional[str] = None
    structure_text: Optional[str] = None
    location_text: Optional[str] = None
    data_currency_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_and_urls() -> str:
    return (
        "Extract the identified university and all reference URLs provided in the answer that are intended to verify "
        "the co-op program claims. Also extract any textual claims for each requirement if present.\n\n"
        "Return a JSON with the following fields:\n"
        "- university_name: The name of the university identified in the answer. If multiple universities are mentioned, "
        "  select the primary one clearly recommended to meet the criteria.\n"
        "- reference_urls: An array of all URLs explicitly included in the answer that are used as references/evidence. "
        "  Include both official pages and third-party sources. Extract actual URLs from any markdown links.\n"
        "- success_rate_text: The exact text or number related to post-graduation success rate, if mentioned.\n"
        "- annual_experiences_text: The exact text or number related to annual co-op experiences/facilitated positions, if mentioned.\n"
        "- annual_earnings_text: The exact text or number related to collective student earnings from co-ops per year, if mentioned.\n"
        "- employer_network_text: The exact text or number related to the number of employer partnerships, if mentioned.\n"
        "- structure_text: Any text describing co-op structure (e.g., full-time, multi-month), if mentioned.\n"
        "- location_text: Any text indicating the university is located in the United States, if mentioned.\n"
        "- data_currency_text: Any text indicating the data is current (e.g., references to years 2023, 2024, 2025, 2026), if mentioned.\n\n"
        "If any field is not present in the answer, set it to null. For URLs, extract only valid URLs exactly as shown."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str]) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else "the university"


def _normalize_sources(urls: Optional[List[str]]) -> Optional[List[str]]:
    """Return URLs if non-empty; otherwise None."""
    if urls and len(urls) > 0:
        return urls
    return None


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    parent_node,
    info: CoOpEvaluationExtraction
) -> None:
    """
    Build the verification subtree under a critical parallel node and run all checks.
    """

    # Create the main critical parallel aggregation node to match rubric
    main_node = evaluator.add_parallel(
        id="University_Co-op_Program_Evaluation",
        desc=(
            "Evaluate whether the identified university's cooperative education program meets all specified requirements "
            "for post-graduation success, program scale, student earnings, employer partnerships, program structure, "
            "geographic location, data currency, and whether the response includes the university name and supporting "
            "reference URLs."
        ),
        parent=parent_node,
        critical=True,
    )

    # Existence checks (custom leaf nodes, critical gating)
    name_exists_node = evaluator.add_custom_node(
        result=bool(info.university_name and info.university_name.strip()),
        id="University_Name_Provided",
        desc="The response provides the name of the university being identified.",
        parent=main_node,
        critical=True
    )

    urls_exist_node = evaluator.add_custom_node(
        result=bool(info.reference_urls and len(info.reference_urls) > 0),
        id="Reference_URLs_Provided",
        desc="The response includes reference URLs that support and verify the claims made about the university's co-op program requirements.",
        parent=main_node,
        critical=True
    )

    # Prepare common data
    uni_name = _safe_name(info.university_name)
    sources = _normalize_sources(info.reference_urls)

    # 1) Post-graduation success rate >= 90% within six months
    pg_success_node = evaluator.add_leaf(
        id="Post_Graduation_Success_Rate",
        desc="The university reports a post-graduation success rate of at least 90% within six months of graduation.",
        parent=main_node,
        critical=True
    )
    claim_pg_success = (
        f"{uni_name} reports a post-graduation success rate of at least 90% for graduates "
        f"within six months (employed full-time or enrolled in graduate school)."
    )
    await evaluator.verify(
        claim=claim_pg_success,
        node=pg_success_node,
        sources=sources,
        additional_instruction=(
            "Check the referenced pages for an explicit statement showing ≥90% success rate within six months, "
            "including employment or graduate school outcomes. Allow phrasing variations like 'career outcomes rate' "
            "or 'knowledge rate' if clearly defined to include full-time employment or graduate enrollment."
        )
    )

    # 2) Annual co-op participation scale >= 3,000 paid experiences
    scale_node = evaluator.add_leaf(
        id="Annual_Co-op_Participation_Scale",
        desc="The university facilitates at least 3,000 paid cooperative education experiences annually for its students.",
        parent=main_node,
        critical=True
    )
    claim_scale = (
        f"{uni_name} facilitates at least 3,000 paid cooperative education experiences for students each year."
    )
    await evaluator.verify(
        claim=claim_scale,
        node=scale_node,
        sources=sources,
        additional_instruction=(
            "Verify that the referenced materials explicitly state a scale of ≥3,000 co-op placements or experiences per year. "
            "Equivalent wording such as 'co-op positions per year' or 'annual co-op assignments' is acceptable."
        )
    )

    # 3) Collective student earnings >= $50 million per year
    earnings_node = evaluator.add_leaf(
        id="Collective_Student_Earnings",
        desc="Students participating in the co-op program collectively earn at least $50 million annually through their co-op placements.",
        parent=main_node,
        critical=True
    )
    claim_earnings = (
        f"Students at {uni_name} collectively earn at least $50 million per year through co-op placements."
    )
    await evaluator.verify(
        claim=claim_earnings,
        node=earnings_node,
        sources=sources,
        additional_instruction=(
            "Confirm an explicit figure or statement on referenced pages indicating total annual student co-op earnings "
            "of at least $50 million. Allow reasonable rounding or phrasing like 'over $50 million'."
        )
    )

    # 4) Employer partnerships network >= 1,000
    partners_node = evaluator.add_leaf(
        id="Employer_Partnership_Network",
        desc="The university maintains active partnerships with at least 1,000 employers for co-op placements, research positions, or project-based work.",
        parent=main_node,
        critical=True
    )
    claim_partners = (
        f"{uni_name} maintains partnerships with at least 1,000 employers for co-op placements, research, or projects."
    )
    await evaluator.verify(
        claim=claim_partners,
        node=partners_node,
        sources=sources,
        additional_instruction=(
            "Look for statements about the size of the employer network or number of employer partners being ≥1,000. "
            "Mentions like 'over 1,000 employers' should be considered sufficient."
        )
    )

    # 5) Co-op experiences are full-time, multi-month professional work (not internships/part-time)
    structure_node = evaluator.add_leaf(
        id="Full_Time_Professional_Experience",
        desc="The co-op experiences offered by the university are full-time, multi-month professional work experiences, not short-term internships or part-time positions.",
        parent=main_node,
        critical=True
    )
    claim_structure = (
        f"The co-op program at {uni_name} offers full-time, multi-month professional work experiences (not short-term internships or part-time roles)."
    )
    await evaluator.verify(
        claim=claim_structure,
        node=structure_node,
        sources=sources,
        additional_instruction=(
            "Check referenced pages for explicit descriptions stating that co-ops are full-time and last multiple months. "
            "Phrasing like 'six months', 'three to six months', or 'full-time co-op' should count; internship-only "
            "or part-time descriptions should not."
        )
    )

    # 6) University is located in the United States
    location_node = evaluator.add_leaf(
        id="United_States_Location",
        desc="The university is located in the United States.",
        parent=main_node,
        critical=True
    )
    claim_location = (
        f"{uni_name} is located in the United States."
    )
    await evaluator.verify(
        claim=claim_location,
        node=location_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the referenced pages clearly indicate the university is in the United States, "
            "including references to US states or explicit mention of 'United States'."
        )
    )

    # 7) Data currency within 2023-2026
    currency_node = evaluator.add_leaf(
        id="Data_Currency_2023_2026",
        desc="The employment outcomes and program statistics provided are based on data from the 2023-2026 time period.",
        parent=main_node,
        critical=True
    )
    claim_currency = (
        f"The employment outcomes and co-op program statistics for {uni_name} are based on data from 2023, 2024, 2025, or 2026."
    )
    await evaluator.verify(
        claim=claim_currency,
        node=currency_node,
        sources=sources,
        additional_instruction=(
            "Verify on referenced pages that metrics are tied to recent years in the range 2023–2026. "
            "Accept metadata or report titles containing those years. If only older data (e.g., 2021/2022) is shown without "
            "an update in 2023–2026, consider it not supported."
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
    Evaluate the agent's answer for the US university co-op program criteria.
    """
    # Initialize evaluator with a non-critical root; we will add a critical aggregation node under it
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

    # Extraction step
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_university_and_urls(),
        template_class=CoOpEvaluationExtraction,
        extraction_name="coop_university_info"
    )

    # Optional: record a summary of extracted info for debugging
    evaluator.add_custom_info(
        info={
            "university_name": extracted_info.university_name,
            "reference_url_count": len(extracted_info.reference_urls),
            "first_3_urls": extracted_info.reference_urls[:3],
            "claimed_texts": {
                "success_rate_text": extracted_info.success_rate_text,
                "annual_experiences_text": extracted_info.annual_experiences_text,
                "annual_earnings_text": extracted_info.annual_earnings_text,
                "employer_network_text": extracted_info.employer_network_text,
                "structure_text": extracted_info.structure_text,
                "location_text": extracted_info.location_text,
                "data_currency_text": extracted_info.data_currency_text,
            }
        },
        info_type="extraction_summary",
        info_name="extraction_overview"
    )

    # Build verification tree and run checks
    await build_and_verify(evaluator, root, extracted_info)

    # Return evaluator summary
    return evaluator.get_summary()