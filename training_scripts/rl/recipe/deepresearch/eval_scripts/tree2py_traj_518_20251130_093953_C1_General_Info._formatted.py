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
TASK_ID = "stranger_things_hawkins_incorp_year"
TASK_DESCRIPTION = (
    "What year was the city where the downtown Hawkins scenes of the Netflix series Stranger Things were primarily filmed incorporated?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AnswerExtraction(BaseModel):
    """
    Structured extraction from the agent's answer:
    - city_name: The specific city that the answer identifies as the primary filming location for downtown Hawkins scenes.
    - incorporation_year: The year (preferably a 4-digit YYYY) the identified city was incorporated.
    - filming_sources: URLs cited in the answer that support the filming location claims.
    - incorporation_sources: URLs cited in the answer that support the incorporation year (ideally official sources).
    """
    city_name: Optional[str] = None
    incorporation_year: Optional[str] = None
    filming_sources: List[str] = Field(default_factory=list)
    incorporation_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return """
    Extract the following items exactly as presented in the answer:

    1) city_name:
       - The specific city that the answer claims was primarily used for the downtown Hawkins scenes in Stranger Things.
       - It must be a city name (not just a state or region). If not explicitly stated, return null.

    2) incorporation_year:
       - The incorporation year for the above city (ideally a 4-digit YYYY).
       - If not explicitly stated, return null. Do not infer.

    3) filming_sources:
       - A list of all URLs in the answer that directly support the filming location claims
         (e.g., pages confirming the city is the primary filming location for downtown Hawkins or that the city's downtown square was used).
       - Only include URLs explicitly shown in the answer. If none, return [].

    4) incorporation_sources:
       - A list of all URLs in the answer that directly support the incorporation year for the identified city.
       - Prefer official sources (municipal/county/state government domains or official civic registries) if presented in the answer.
       - Only include URLs explicitly shown in the answer. If none, return [].

    Important:
    - Only extract information explicitly present in the answer.
    - For URLs, include full URLs exactly as shown. Accept plain links or markdown-form links, but return the actual URL strings.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _combine_sources(a: List[str], b: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for url in (a or []) + (b or []):
        if not url:
            continue
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: AnswerExtraction) -> None:
    """
    Build the verification tree based on the rubric and run all verifications.
    """
    # Top-level critical node aggregating all checks
    answer_eval_node = evaluator.add_parallel(
        id="Answer_Evaluation",
        desc="Evaluates whether the answer identifies the correct Georgia city used for downtown Hawkins filming and states that city's incorporation year with proper verification/format.",
        parent=evaluator.root,
        critical=True,
    )

    # -------------------- Filming Location City --------------------------- #
    filming_node = evaluator.add_parallel(
        id="Filming_Location_City",
        desc="The answer identifies the correct city that was primarily used for downtown Hawkins scenes.",
        parent=answer_eval_node,
        critical=True,
    )

    # City_Is_Specified (existence/format check) - Critical
    city_specified_node = evaluator.add_custom_node(
        result=_non_empty_str(extracted.city_name),
        id="City_Is_Specified",
        desc="Provides a specific city name (not just a state/region).",
        parent=filming_node,
        critical=True,
    )

    # City_Is_Primary_Downtown_Hawkins_Filming_Location - Critical, verify by URLs
    city_primary_node = evaluator.add_leaf(
        id="City_Is_Primary_Downtown_Hawkins_Filming_Location",
        desc="The named city is in fact the primary filming location for the downtown Hawkins scenes in Stranger Things.",
        parent=filming_node,
        critical=True,
    )
    claim_primary = (
        f"The city '{extracted.city_name or ''}' was the primary filming location for the downtown Hawkins scenes "
        f"in the Netflix series Stranger Things."
    )
    await evaluator.verify(
        claim=claim_primary,
        node=city_primary_node,
        sources=extracted.filming_sources,
        additional_instruction=(
            "Focus on whether the page explicitly supports that this city was used for downtown Hawkins. "
            "Look for terms such as 'Hawkins', 'downtown', 'town square', or mentions of 'Stranger Things' main street scenes. "
            "If no URLs are provided, conclude Incorrect."
        ),
        extra_prerequisites=[city_specified_node],
    )

    # City_Located_In_Georgia_USA - Critical, verify by URLs
    city_in_ga_node = evaluator.add_leaf(
        id="City_Located_In_Georgia_USA",
        desc="The named city is located in Georgia, United States.",
        parent=filming_node,
        critical=True,
    )
    ga_sources = _combine_sources(extracted.filming_sources, extracted.incorporation_sources)
    claim_ga = f"The city '{extracted.city_name or ''}' is located in the U.S. state of Georgia."
    await evaluator.verify(
        claim=claim_ga,
        node=city_in_ga_node,
        sources=ga_sources,
        additional_instruction=(
            "Verify that the page indicates the city is in Georgia (GA). "
            "Accept mentions like 'Jackson, Georgia (GA)' etc. If no URLs are provided, conclude Incorrect."
        ),
        extra_prerequisites=[city_specified_node],
    )

    # Downtown_Square_Used_For_Main_Town_Square_Scenes - Critical, verify by URLs
    downtown_square_node = evaluator.add_leaf(
        id="Downtown_Square_Used_For_Main_Town_Square_Scenes",
        desc="The city's downtown square was used for the main town square scenes in Stranger Things.",
        parent=filming_node,
        critical=True,
    )
    claim_square = (
        f"The downtown square of '{extracted.city_name or ''}' was used for the main town square scenes in Stranger Things."
    )
    await evaluator.verify(
        claim=claim_square,
        node=downtown_square_node,
        sources=extracted.filming_sources,
        additional_instruction=(
            "Check if the page states that the city's downtown square (courthouse square or town square) "
            "served as the Hawkins town square filming location. If no URLs are provided, conclude Incorrect."
        ),
        extra_prerequisites=[city_specified_node],
    )

    # -------------------- Year of Incorporation --------------------------- #
    year_node = evaluator.add_parallel(
        id="Year_of_Incorporation",
        desc="The answer provides the incorporation year for the identified city and satisfies sourcing/format constraints.",
        parent=answer_eval_node,
        critical=True,
    )

    # Year_Is_4_Digit - Critical (format check)
    year_is_4_digit = _non_empty_str(extracted.incorporation_year) and (
        extracted.incorporation_year.strip().isdigit() and len(extracted.incorporation_year.strip()) == 4
    )
    year_format_node = evaluator.add_custom_node(
        result=year_is_4_digit,
        id="Year_Is_4_Digit",
        desc="The incorporation year is presented as a specific 4-digit year (YYYY).",
        parent=year_node,
        critical=True,
    )

    # Incorporation_Year_Matches_Identified_City - Critical, verify by URLs
    year_matches_node = evaluator.add_leaf(
        id="Incorporation_Year_Matches_Identified_City",
        desc="States the year the identified city was incorporated (and it corresponds to that same city).",
        parent=year_node,
        critical=True,
    )
    claim_year = (
        f"The city '{extracted.city_name or ''}' was incorporated in {extracted.incorporation_year or ''}."
    )
    await evaluator.verify(
        claim=claim_year,
        node=year_matches_node,
        sources=extracted.incorporation_sources,
        additional_instruction=(
            "Verify the exact incorporation year for the specified city. "
            "Ensure the page is about the same city and that the stated year matches. "
            "If no URLs are provided, conclude Incorrect."
        ),
        extra_prerequisites=[city_specified_node, year_format_node],
    )

    # Official_Source_Verification - Critical, verify that at least one official source confirms the year
    official_source_node = evaluator.add_leaf(
        id="Official_Source_Verification",
        desc="Includes support from at least one reliable official source (e.g., municipal/county/state government record or official civic registry) that verifies the stated incorporation year.",
        parent=year_node,
        critical=True,
    )
    claim_official = (
        f"This page is an official government or civic registry source and explicitly verifies that "
        f"'{extracted.city_name or ''}' was incorporated in {extracted.incorporation_year or ''}."
    )
    await evaluator.verify(
        claim=claim_official,
        node=official_source_node,
        sources=extracted.incorporation_sources,
        additional_instruction=(
            "Treat government/civic registry sites as official (e.g., domains ending with .gov, .ga.gov, "
            "or clearly official city/county/state websites). The page must also confirm the incorporation year. "
            "If no URLs are provided, conclude Incorrect."
        ),
        extra_prerequisites=[city_specified_node, year_format_node],
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
    Evaluate the agent's answer for the Stranger Things downtown Hawkins incorporation year question.
    """
    # Initialize evaluator with a parallel root
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
        prompt=prompt_extract_core(),
        template_class=AnswerExtraction,
        extraction_name="core_extraction",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()