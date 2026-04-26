import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "australian_open_2026_info"
TASK_DESCRIPTION = (
    "I am planning to attend the Australian Open 2026 tennis tournament. Please provide the following information: "
    "(1) The exact dates when the tournament will take place, "
    "(2) The name and location of the venue where the tournament will be held, "
    "(3) The starting price for an adult Ground Pass ticket for week one of the tournament. "
    "For each piece of information, include a reference URL from an official or reliable source."
)

# Ground truth expectations expressed as human-readable strings
EXPECTED_DATES_TEXT = "January 18 to February 1, 2026"
EXPECTED_VENUE_NAME = "Melbourne Park"
EXPECTED_VENUE_LOCATION = "Melbourne, Australia"
EXPECTED_GROUND_PASS_PRICE = "$59"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AO2026Extraction(BaseModel):
    # Event dates
    event_dates_text: Optional[str] = None
    event_dates_urls: List[str] = Field(default_factory=list)

    # Venue and location
    venue_name: Optional[str] = None
    venue_location: Optional[str] = None
    venue_urls: List[str] = Field(default_factory=list)

    # Ground Pass pricing (adult, week one, starting price)
    ground_pass_week_one_price_text: Optional[str] = None
    pricing_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ao2026() -> str:
    return """
    Extract the specific information about the Australian Open 2026 as explicitly stated in the answer. Return the following fields:
    1) event_dates_text: The exact date range (as a single string) the answer states the tournament will take place (e.g., "January 18 to February 1, 2026", "18 Jan – 1 Feb 2026", etc.).
    2) event_dates_urls: A list of all reference URLs that the answer cites for the event dates (extract actual URLs only; include all if multiple).
    3) venue_name: The venue name as stated in the answer (e.g., "Melbourne Park").
    4) venue_location: The location (city and country) of the venue as stated in the answer (e.g., "Melbourne, Australia").
    5) venue_urls: A list of all reference URLs that the answer cites for the venue/location (extract actual URLs only; include all if multiple).
    6) ground_pass_week_one_price_text: The starting price for an adult Ground Pass ticket for week one, exactly as stated in the answer (e.g., "$59", "A$59", "59 AUD").
    7) pricing_urls: A list of all reference URLs that the answer cites for the ticket pricing (extract actual URLs only; include all if multiple).

    Rules:
    - Extract only what is explicitly present in the answer. Do not infer or invent values.
    - If a field is missing, set it to null (for strings) or an empty array (for URLs).
    - URLs may appear as plain links or markdown links; always return the actual URL.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_sources(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Keep only non-empty strings; Evaluator/Verifier will handle validity via fetch
    return [u for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_event_dates_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: AO2026Extraction,
) -> None:
    """
    Build verification for:
    - Content: The event dates must be January 18 to February 1, 2026 (compare to what the answer claimed).
    - URL: At least one official/reliable reference is provided and supports the dates.
    """
    group = evaluator.add_parallel(
        id="Event_Dates_Information",
        desc="Verify the event dates and reference URL",
        parent=parent_node,
        critical=True,
    )

    # Content check (compare the answer's stated dates to the expected window)
    content_leaf = evaluator.add_leaf(
        id="Event_Dates_Content",
        desc="The event dates must be January 18 to February 1, 2026",
        parent=group,
        critical=True,
    )

    stated_dates = extracted.event_dates_text or ""
    content_claim = (
        f"The event dates stated in the answer ('{stated_dates}') refer to the same time window as "
        f"'{EXPECTED_DATES_TEXT}'. Consider minor format variations equivalent (e.g., '18 Jan – 1 Feb 2026')."
    )
    await evaluator.verify(
        claim=content_claim,
        node=content_leaf,
        additional_instruction=(
            "Judge if the stated date range is effectively the same as January 18 to February 1, 2026. "
            "Allow variations like abbreviations (Jan/Feb), day–month reordering (18 January), different dashes, "
            "and optional year repetition. If the stated dates are missing or clearly not equivalent, mark incorrect."
        ),
    )

    # URL support check (the referenced URL(s) support the dates)
    url_leaf = evaluator.add_leaf(
        id="Event_Dates_URL",
        desc="A reference URL from an official or reliable source must be provided for the event dates",
        parent=group,
        critical=True,
    )

    dates_urls = _normalize_sources(extracted.event_dates_urls)
    # If the answer provided dates text, verify that those dates are supported by the URLs.
    if stated_dates.strip():
        url_claim = (
            f"The provided webpage(s) explicitly confirm that the Australian Open 2026 runs on {stated_dates}."
        )
    else:
        # Fallback: verify the expected official window if the answer didn't state dates (content check would fail anyway).
        url_claim = (
            f"The provided webpage(s) explicitly confirm that the Australian Open 2026 runs on {EXPECTED_DATES_TEXT}."
        )

    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=dates_urls,
        additional_instruction=(
            "Pass only if the page(s) clearly state the Australian Open 2026 tournament dates. "
            "Prefer official sources (e.g., ausopen.com, Tennis Australia). If URLs are missing, irrelevant, or do not "
            "state the dates, mark incorrect."
        ),
    )


async def build_venue_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: AO2026Extraction,
) -> None:
    """
    Build verification for:
    - Content: The venue must be Melbourne Park in Melbourne, Australia (compare to what the answer claimed).
    - URL: At least one official/reliable reference is provided and supports the venue/location.
    """
    group = evaluator.add_parallel(
        id="Venue_Location_Information",
        desc="Verify the venue location and reference URL",
        parent=parent_node,
        critical=True,
    )

    # Content check
    content_leaf = evaluator.add_leaf(
        id="Venue_Location_Content",
        desc="The venue must be Melbourne Park in Melbourne, Australia",
        parent=group,
        critical=True,
    )

    stated_venue = extracted.venue_name or ""
    stated_location = extracted.venue_location or ""
    venue_content_claim = (
        f"The venue and location stated in the answer ('{stated_venue}', '{stated_location}') are equivalent to "
        f"'{EXPECTED_VENUE_NAME}' in '{EXPECTED_VENUE_LOCATION}'. Allow minor formatting variations (e.g., "
        f"'Melbourne, VIC, Australia' vs 'Melbourne, Australia')."
    )
    await evaluator.verify(
        claim=venue_content_claim,
        node=content_leaf,
        additional_instruction=(
            "Judge string equivalence robustly: accept reasonable variations in punctuation, abbreviations, and word order. "
            "If the answer's venue or location is missing or refers to a different place, mark incorrect."
        ),
    )

    # URL support check
    url_leaf = evaluator.add_leaf(
        id="Venue_Location_URL",
        desc="A reference URL from an official or reliable source must be provided for the venue location",
        parent=group,
        critical=True,
    )

    venue_urls = _normalize_sources(extracted.venue_urls)
    if stated_venue.strip() and stated_location.strip():
        url_claim = (
            f"The provided webpage(s) explicitly confirm that the Australian Open 2026 is held at "
            f"{stated_venue} in {stated_location}."
        )
    else:
        url_claim = (
            f"The provided webpage(s) explicitly confirm that the Australian Open 2026 is held at "
            f"{EXPECTED_VENUE_NAME} in {EXPECTED_VENUE_LOCATION}."
        )

    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=venue_urls,
        additional_instruction=(
            "Pass only if the page(s) clearly state the tournament venue and its location. "
            "Prefer official sources (e.g., ausopen.com). If URLs are missing, irrelevant, or do not confirm the venue "
            "and location, mark incorrect."
        ),
    )


async def build_pricing_subtree(
    evaluator: Evaluator,
    parent_node,
    extracted: AO2026Extraction,
) -> None:
    """
    Build verification for:
    - Content: The starting price for an adult Ground Pass ticket for week one must be $59 (compare to answer).
    - URL: At least one official/reliable reference is provided and supports the pricing.
    """
    group = evaluator.add_parallel(
        id="Ground_Pass_Pricing_Information",
        desc="Verify the Ground Pass pricing and reference URL",
        parent=parent_node,
        critical=True,
    )

    # Content check
    content_leaf = evaluator.add_leaf(
        id="Ground_Pass_Pricing_Content",
        desc="Adult Ground Pass ticket pricing for week one must start from $59",
        parent=group,
        critical=True,
    )

    stated_price = extracted.ground_pass_week_one_price_text or ""
    price_content_claim = (
        f"The starting price for an adult Ground Pass ticket for week one stated in the answer ('{stated_price}') "
        f"is equivalent to '{EXPECTED_GROUND_PASS_PRICE}'. Allow currency format variants like 'A$59', '59 AUD', or '$59.00'."
    )
    await evaluator.verify(
        claim=price_content_claim,
        node=content_leaf,
        additional_instruction=(
            "Judge equivalence robustly for currency formatting and symbols. If the extracted price is missing or not "
            "equivalent to $59, mark incorrect."
        ),
    )

    # URL support check
    url_leaf = evaluator.add_leaf(
        id="Ground_Pass_Pricing_URL",
        desc="A reference URL from an official or reliable source must be provided for the ticket pricing",
        parent=group,
        critical=True,
    )

    pricing_urls = _normalize_sources(extracted.pricing_urls)
    if stated_price.strip():
        url_claim = (
            f"The provided webpage(s) explicitly confirm that the starting price for an adult Ground Pass ticket for "
            f"week one is {stated_price}."
        )
    else:
        url_claim = (
            f"The provided webpage(s) explicitly confirm that the starting price for an adult Ground Pass ticket for "
            f"week one is {EXPECTED_GROUND_PASS_PRICE}."
        )

    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=pricing_urls,
        additional_instruction=(
            "Pass only if the page(s) clearly state the pricing for an adult Ground Pass in week one and the starting "
            "price matches the claim. Prefer official sources (e.g., ausopen.com ticketing). If URLs are missing, "
            "irrelevant, or do not confirm the price, mark incorrect."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Australian Open 2026 information task.
    """
    # Initialize evaluator (root is a non-critical wrapper; we add the task node as critical)
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_ao2026(),
        template_class=AO2026Extraction,
        extraction_name="ao2026_extraction",
    )

    # Add ground truth expectations as GT info (for transparency)
    evaluator.add_ground_truth(
        {
            "expected_dates": EXPECTED_DATES_TEXT,
            "expected_venue_name": EXPECTED_VENUE_NAME,
            "expected_venue_location": EXPECTED_VENUE_LOCATION,
            "expected_ground_pass_start_price_week_one": EXPECTED_GROUND_PASS_PRICE,
        },
        gt_type="expected_values",
    )

    # Build the critical task node and all subtrees
    ao_task_node = evaluator.add_parallel(
        id="Australian_Open_2026_Information",
        desc="Verify information about the Australian Open 2026 tennis tournament",
        parent=root,
        critical=True,
    )

    await build_event_dates_subtree(evaluator, ao_task_node, extracted)
    await build_venue_subtree(evaluator, ao_task_node, extracted)
    await build_pricing_subtree(evaluator, ao_task_node, extracted)

    # Return the structured evaluation summary
    return evaluator.get_summary()