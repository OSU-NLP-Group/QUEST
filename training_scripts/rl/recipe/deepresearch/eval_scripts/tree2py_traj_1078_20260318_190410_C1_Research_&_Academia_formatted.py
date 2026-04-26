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
TASK_ID = "tx_lunar_eclipse_2026_best_view_city"
TASK_DESCRIPTION = """
Which city in Texas offers the best viewing conditions, in terms of lowest average cloud cover, for observing the total lunar eclipse on March 3, 2026? Provide the city name and its average cloud cover percentage.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BestCityExtraction(BaseModel):
    """
    Structured extraction from the agent's answer.
    """
    city_name: Optional[str] = None  # Texas city identified as best for the event
    average_cloud_cover: Optional[str] = None  # The stated average cloud cover percentage (keep as string, e.g., "27%")
    source_urls: List[str] = Field(default_factory=list)  # All URLs cited as evidence in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_best_city() -> str:
    return """
    From the answer, extract:
    - city_name: The single Texas city the answer claims has the best viewing conditions (i.e., the lowest average cloud cover) for the March 3, 2026 total lunar eclipse. If multiple cities are mentioned, choose the one explicitly identified as "best" or with the lowest stated cloud cover. If ambiguous, select the first city that is claimed to be best.
    - average_cloud_cover: The specific average cloud cover percentage stated for that city in the answer (return it as it appears, e.g., "23%", "23–25%", or "about 23%").
    - source_urls: A list of all explicit URLs cited in the answer that support the cloud cover/visibility data for that city for the March 3, 2026 total lunar eclipse. Include any links like timeanddate.com, nasa.gov, or similar. Extract actual URLs even if in markdown.
    If any field is missing, return null for that field or an empty list for source_urls.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def build_urls_inline(urls: List[str]) -> str:
    if not urls:
        return "(no URLs provided)"
    return "; ".join(urls)


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root, extracted: BestCityExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """

    # ---------------- Reference URL Node (Non-critical criterion) ---------------- #
    ref_node = evaluator.add_parallel(
        id="Reference_URL",
        desc="The answer includes a reference URL from an authoritative source documenting Texas cloud cover data for the March 3, 2026 eclipse",
        parent=root,
        critical=False
    )

    # 1) Existence of at least one reference URL (critical within this sub-criterion)
    has_ref_url_node = evaluator.add_custom_node(
        result=bool(extracted.source_urls),
        id="has_reference_url",
        desc="At least one reference URL is provided in the answer",
        parent=ref_node,
        critical=True
    )

    # 2) At least one URL is authoritative (non-critical within this sub-criterion)
    authoritative_url_leaf = evaluator.add_leaf(
        id="authoritative_reference_url",
        desc="At least one provided URL is from an authoritative domain (e.g., timeanddate.com, nasa.gov, noaa.gov, .gov, .edu, or a major meteorological service)",
        parent=ref_node,
        critical=False
    )
    # We present the URL list within the claim text for simple verification
    urls_inline = build_urls_inline(extracted.source_urls)
    authoritative_claim = (
        "Among the following provided URLs, at least one is from an authoritative source domain for eclipse "
        "or meteorological climatology (e.g., timeanddate.com, nasa.gov, noaa.gov, metoffice.gov.uk, met.no, "
        "ecmwf.int, or any .gov/.edu domain): "
        f"{urls_inline}"
    )
    await evaluator.verify(
        claim=authoritative_claim,
        node=authoritative_url_leaf,
        additional_instruction="Judge domain reputability based on well-known authoritative sites. Do not require the page itself to declare 'authoritative'; focus on domain-level reputation."
    )

    # ---------------- Optimal City Selection (Critical) ---------------- #
    optimal_node = evaluator.add_parallel(
        id="Optimal_City_Selection",
        desc="The identified Texas city is documented for the March 3, 2026 total lunar eclipse and has the lowest average cloud cover among documented Texas cities",
        parent=root,
        critical=True
    )

    city = extracted.city_name or ""
    cloud = extracted.average_cloud_cover or ""
    sources = extracted.source_urls if extracted.source_urls else None

    # A) City is listed in the provided source(s) for this event with cloud cover data
    city_listed_leaf = evaluator.add_leaf(
        id="city_listed_with_data",
        desc="The chosen city is a Texas location listed in the source(s) for the March 3, 2026 total lunar eclipse and includes cloud cover data",
        parent=optimal_node,
        critical=True
    )
    city_listed_claim = (
        f"The provided source page(s) explicitly list {city} (Texas) as a location for the March 3, 2026 total lunar eclipse "
        f"and include average cloud cover data for that city."
    )
    await evaluator.verify(
        claim=city_listed_claim,
        node=city_listed_leaf,
        sources=sources,
        extra_prerequisites=[has_ref_url_node],
        additional_instruction="Allow date representations like 'March 3–4, 2026' due to time zones. Accept terms like 'average cloud cover', 'mean cloudiness', or equivalent. Verify that the city is indeed in Texas on the page."
    )

    # B) City has the lowest (or tied-lowest) average cloud cover among all documented Texas cities on the source page(s)
    lowest_leaf = evaluator.add_leaf(
        id="city_lowest_cloud_cover_in_texas",
        desc="Among listed Texas cities on the source page(s), the chosen city has the lowest average cloud cover",
        parent=optimal_node,
        critical=True
    )
    lowest_claim = (
        f"Among all Texas cities listed on the provided source page(s) for the March 3, 2026 total lunar eclipse, "
        f"{city} has the lowest average cloud cover percentage."
    )
    await evaluator.verify(
        claim=lowest_claim,
        node=lowest_leaf,
        sources=sources,
        extra_prerequisites=[has_ref_url_node],
        additional_instruction="Confirm the city is the minimum among Texas entries. If multiple cities are tied for the lowest value, consider the claim satisfied."
    )

    # ---------------- Cloud Cover Data Provision (Critical) ---------------- #
    cloud_node = evaluator.add_parallel(
        id="Cloud_Cover_Data_Provision",
        desc="The answer provides the stated average cloud cover percentage for the identified city and it matches the source(s)",
        parent=root,
        critical=True
    )

    # A) Existence of the cloud cover percentage in the answer
    cloud_provided_node = evaluator.add_custom_node(
        result=bool(extracted.average_cloud_cover and extracted.average_cloud_cover.strip()),
        id="cloud_cover_provided",
        desc="The answer provides a specific average cloud cover percentage for the chosen city",
        parent=cloud_node,
        critical=True
    )

    # B) The provided cloud cover value matches what the source(s) state for the event
    cloud_match_leaf = evaluator.add_leaf(
        id="cloud_cover_matches_source",
        desc="The provided average cloud cover percentage matches the value shown in the source(s) for the city and event",
        parent=cloud_node,
        critical=True
    )
    cloud_match_claim = (
        f"For {city}, Texas, the source page(s) for the March 3, 2026 total lunar eclipse report an average cloud cover "
        f"that matches the stated value in the answer: '{cloud}'."
    )
    await evaluator.verify(
        claim=cloud_match_claim,
        node=cloud_match_leaf,
        sources=sources,
        extra_prerequisites=[has_ref_url_node, cloud_provided_node],
        additional_instruction="Allow minor rounding differences and formatting variants (e.g., '23%' vs 'about 23%' or small ±1% rounding). Ensure the value corresponds specifically to the March 3, 2026 total lunar eclipse context."
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
    Entry point for evaluating the agent's answer for the Texas best-viewing city
    for the March 3, 2026 total lunar eclipse (lowest average cloud cover).
    """
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

    # 1) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_best_city(),
        template_class=BestCityExtraction,
        extraction_name="best_city_extraction",
    )

    # 2) Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted)

    # 3) Return structured summary
    return evaluator.get_summary()