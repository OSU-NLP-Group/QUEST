import asyncio
import logging
import re
from typing import List, Optional, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "aims_bleaching_2024_southern_gbr"
TASK_DESCRIPTION = (
    "The Australian Institute of Marine Science conducted comprehensive monitoring of the Great Barrier Reef following "
    "the 2024 mass coral bleaching event. Identify the specific research publication released between August 2024 and "
    "August 2025 that documents the impacts of this bleaching event on the Southern Great Barrier Reef region. For this "
    "publication, provide the following information: (1) The exact title of the publication, (2) The specific release date, "
    "(3) The baseline percentage of hard coral cover in the Southern Great Barrier Reef in 2024 (before the bleaching impact "
    "assessment), (4) The percentage of hard coral cover in the Southern Great Barrier Reef after the 2024 bleaching event, "
    "(5) The overall percentage decline in coral cover for the Southern region, (6) The total number of reefs surveyed during "
    "the data collection period, (7) The time span during which data was collected (start month/year to end month/year), and "
    "(8) A verifiable URL reference to the publication from AIMS or an authoritative related source. All values must be "
    "extracted directly from the identified publication and must specifically pertain to the Southern Great Barrier Reef region."
)


# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class AIMSGBRPublicationExtraction(BaseModel):
    # Publication metadata
    publication_title: Optional[str] = None
    release_date: Optional[str] = None  # Keep as free text (e.g., "15 April 2025")
    publication_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)

    # Southern GBR quantitative metrics (as stated in the answer, verbatim)
    baseline_hard_coral_cvr_pct_southern_2024: Optional[str] = None
    post_bleaching_hard_coral_cvr_pct_southern_2024: Optional[str] = None
    decline_pct_southern: Optional[str] = None
    total_reefs_surveyed_southern: Optional[str] = None
    timespan_start: Optional[str] = None  # e.g., "October 2023" or "Dec 2023"
    timespan_end: Optional[str] = None    # e.g., "May 2024"


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_aims_publication() -> str:
    return """
    Extract the single AIMS-related publication identified in the answer that documents the impacts of the 2024 mass coral bleaching event
    on the Southern Great Barrier Reef region. Return the following fields exactly as written in the answer:

    Publication metadata (all from the answer text; do not invent):
    - publication_title: The exact title string of the identified publication as provided in the answer.
    - release_date: The specific release or publication date as provided (e.g., "15 April 2025"). Keep original formatting.
    - publication_url: The primary verifiable URL to the publication (prefer aims.gov.au or an authoritative partner like gbrmpa.gov.au).
    - additional_urls: Any additional URLs mentioned in the answer that point to the same publication or authoritative supporting pages
      (include only URLs explicitly present in the answer).

    Southern GBR metrics (verbatim as in the answer; do not normalize formatting):
    - baseline_hard_coral_cvr_pct_southern_2024: The baseline percentage of hard coral cover in the Southern GBR in 2024 before
      the bleaching impact assessment (e.g., "28%", "about 28%", "27–29%", etc.).
    - post_bleaching_hard_coral_cvr_pct_southern_2024: The percentage of hard coral cover in the Southern GBR after the 2024 bleaching event.
    - decline_pct_southern: The overall percentage decline in coral cover for the Southern region as reported in the answer.
    - total_reefs_surveyed_southern: The total number of reefs surveyed for the Southern GBR region during the data collection period.
    - timespan_start: The start month/year for data collection specific to the Southern GBR (e.g., "October 2023", "Dec 2023").
    - timespan_end: The end month/year for data collection specific to the Southern GBR (e.g., "May 2024", "Apr 2024").

    Important:
    - Extract exactly what the answer states (including symbols like % and ranges if applicable). Do not infer or compute new values.
    - Only include URLs explicitly present in the answer; do not invent or guess URLs.
    - If any field is missing from the answer, return null for that field (or an empty list for additional_urls).
    """


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _collect_all_urls(extracted: AIMSGBRPublicationExtraction) -> List[str]:
    urls: List[str] = []
    if extracted.publication_url and isinstance(extracted.publication_url, str) and extracted.publication_url.strip():
        urls.append(extracted.publication_url.strip())
    if extracted.additional_urls:
        for u in extracted.additional_urls:
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def _parse_percentage_value(value: Optional[str]) -> Optional[float]:
    """
    Extract a numeric percentage value from a string.
    - If a range like "27–29%" or "27-29%" appears, return the midpoint.
    - Otherwise return the first numeric value found.
    - Return None if no numeric token can be confidently parsed.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()

    # Normalize dashes
    text = text.replace("–", "-").replace("—", "-")

    # Look for range pattern first: e.g., "27-29", "27 - 29"
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", text)
    if range_match:
        a = float(range_match.group(1))
        b = float(range_match.group(2))
        return (a + b) / 2.0

    # Otherwise pick the first numeric token
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _format_percent_one_decimal(x: float) -> str:
    return f"{round(x, 1):.1f}%"


# -----------------------------------------------------------------------------
# Verification subtrees
# -----------------------------------------------------------------------------
async def verify_identification(
    evaluator: Evaluator,
    parent_node,
    extracted: AIMSGBRPublicationExtraction
) -> None:
    """
    Build and verify the 'Qualifying_Publication_Identification' parallel critical subtree.
    """
    node = evaluator.add_parallel(
        id="Qualifying_Publication_Identification",
        desc="Identify the publication meeting the stated qualification constraints and provide its identifying metadata.",
        parent=parent_node,
        critical=True
    )

    all_urls = _collect_all_urls(extracted)

    # Leaf: Exact title
    leaf_title = evaluator.add_leaf(
        id="Exact_Publication_Title_Provided",
        desc="Provides the exact title of the identified publication.",
        parent=node,
        critical=True
    )
    title_claim = f"The exact title of the publication is '{extracted.publication_title}'."
    await evaluator.verify(
        claim=title_claim,
        node=leaf_title,
        sources=all_urls,
        additional_instruction=(
            "Confirm on the provided publication page the exact publication title string equals the one quoted. "
            "Be strict about exactness: do not accept paraphrases or near matches. "
            "If the answer did not provide a title or no valid URL is supplied, mark this as unsupported."
        )
    )

    # Leaf: Release date provided and within range (Aug 2024–Aug 2025 inclusive)
    leaf_date = evaluator.add_leaf(
        id="Release_Date_Provided_And_In_Range",
        desc="Provides the publication's specific release date, and it falls between August 2024 and August 2025 (inclusive).",
        parent=node,
        critical=True
    )
    date_claim = (
        f"The publication's specific release date is '{extracted.release_date}', "
        "and this date falls between Aug 1, 2024 and Aug 31, 2025 inclusive."
    )
    await evaluator.verify(
        claim=date_claim,
        node=leaf_date,
        sources=all_urls,
        additional_instruction=(
            "Verify the stated release/publication date on the page and ensure it lies within the inclusive window "
            "Aug 1, 2024 to Aug 31, 2025. Accept common formats (e.g., '15 April 2025', '2025-04-15'). "
            "If no date is provided in the answer or it is outside the range, mark as unsupported."
        )
    )

    # Leaf: AIMS involvement confirmed
    leaf_aims = evaluator.add_leaf(
        id="AIMS_Involvement_Confirmed",
        desc="The publication explicitly indicates AIMS conducted the research or was involved as an institution/partner.",
        parent=node,
        critical=True
    )
    aims_claim = (
        "The publication indicates that the Australian Institute of Marine Science (AIMS) conducted the research or "
        "was involved as an institution/partner."
    )
    await evaluator.verify(
        claim=aims_claim,
        node=leaf_aims,
        sources=all_urls,
        additional_instruction=(
            "Look for 'AIMS' or 'Australian Institute of Marine Science' in author/affiliation/credits or in-page acknowledgements. "
            "Co-publications (e.g., with GBRMPA) also count if AIMS is explicitly credited."
        )
    )

    # Leaf: Documents 2024 mass bleaching event
    leaf_bleaching = evaluator.add_leaf(
        id="Documents_2024_Mass_Bleaching_Event",
        desc="The publication documents impacts of the 2024 mass coral bleaching event on the Great Barrier Reef.",
        parent=node,
        critical=True
    )
    bleaching_claim = (
        "The publication documents the impacts of the 2024 mass coral bleaching event on the Great Barrier Reef."
    )
    await evaluator.verify(
        claim=bleaching_claim,
        node=leaf_bleaching,
        sources=all_urls,
        additional_instruction=(
            "Confirm explicit reference to the 2024 (or 2023–24 summer) mass coral bleaching event and its impacts."
        )
    )

    # Leaf: Pertains specifically to Southern GBR region
    leaf_southern = evaluator.add_leaf(
        id="Pertains_To_Southern_GBR_Region",
        desc="The publication's reported impacts/analysis explicitly pertain to the Southern Great Barrier Reef region.",
        parent=node,
        critical=True
    )
    southern_claim = (
        "The publication explicitly reports impacts or analysis that pertain to the Southern Great Barrier Reef region."
    )
    await evaluator.verify(
        claim=southern_claim,
        node=leaf_southern,
        sources=all_urls,
        additional_instruction=(
            "Look for explicit mentions of the 'Southern Great Barrier Reef' region or equivalent regional grouping from the publication. "
            "Do not accept generic GBR-wide statements without a specific Southern-region breakdown."
        )
    )

    # Leaf: Verifiable publication URL present and authoritative
    leaf_url = evaluator.add_leaf(
        id="Verifiable_Publication_URL",
        desc="Provides a verifiable URL to the publication from AIMS or an authoritative related source.",
        parent=node,
        critical=True
    )
    url_claim = (
        "At least one provided URL is the actual publication page from AIMS (aims.gov.au) or an authoritative partner "
        "(e.g., gbrmpa.gov.au or a co-published 'Reef Snapshot' page), not a generic homepage or unrelated news item."
    )
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=all_urls,
        additional_instruction=(
            "Check domain and page content to ensure it is the publication itself (or its official landing page), "
            "not just a general site page. If no URL was provided in the answer, mark this as not supported."
        )
    )


async def verify_southern_data(
    evaluator: Evaluator,
    parent_node,
    extracted: AIMSGBRPublicationExtraction
) -> None:
    """
    Build and verify the 'Southern_GBR_Data_Extraction' parallel critical subtree.
    """
    node = evaluator.add_parallel(
        id="Southern_GBR_Data_Extraction",
        desc="Extract all required quantitative values directly from the identified publication, ensuring each value pertains specifically to the Southern GBR region (as required by the question).",
        parent=parent_node,
        critical=True
    )

    all_urls = _collect_all_urls(extracted)

    # Baseline percentage (2024, before bleaching assessment)
    baseline_leaf = evaluator.add_leaf(
        id="Baseline_Hard_Coral_Cover_2024_Southern",
        desc="Extracts the baseline percentage of hard coral cover in the Southern GBR in 2024 (before the bleaching impact assessment), as stated in the publication.",
        parent=node,
        critical=True
    )
    baseline_claim = (
        f"The publication states that before assessing the 2024 bleaching impact, the baseline percentage of hard coral "
        f"cover in the Southern Great Barrier Reef in 2024 was {extracted.baseline_hard_coral_cvr_pct_southern_2024}."
    )
    await evaluator.verify(
        claim=baseline_claim,
        node=baseline_leaf,
        sources=all_urls,
        additional_instruction=(
            "Verify the value refers specifically to the Southern GBR and to the pre-impact (baseline) 2024 condition. "
            "Values taken from tables/figures are acceptable. If the answer did not provide this specific value, mark as unsupported."
        )
    )

    # Post-bleaching percentage (after 2024 event)
    post_leaf = evaluator.add_leaf(
        id="Post_Bleaching_Hard_Coral_Cover_Southern",
        desc="Extracts the percentage of hard coral cover in the Southern GBR after the 2024 bleaching event, as stated in the publication.",
        parent=node,
        critical=True
    )
    post_claim = (
        f"The publication states that after the 2024 bleaching event, the percentage of hard coral cover in the "
        f"Southern Great Barrier Reef was {extracted.post_bleaching_hard_coral_cvr_pct_southern_2024}."
    )
    await evaluator.verify(
        claim=post_claim,
        node=post_leaf,
        sources=all_urls,
        additional_instruction=(
            "Confirm the value is explicitly post-bleaching (2024 event) and specific to the Southern GBR region. "
            "Numbers from charts/figures are acceptable."
        )
    )

    # Total number of reefs surveyed (Southern region)
    reefs_leaf = evaluator.add_leaf(
        id="Total_Number_of_Reefs_Surveyed_Southern",
        desc="Extracts the total number of reefs surveyed for the Southern GBR region during the data collection period, as stated in the publication.",
        parent=node,
        critical=True
    )
    reefs_claim = (
        f"The publication states that the total number of reefs surveyed in the Southern Great Barrier Reef during the "
        f"data collection period is {extracted.total_reefs_surveyed_southern}."
    )
    await evaluator.verify(
        claim=reefs_claim,
        node=reefs_leaf,
        sources=all_urls,
        additional_instruction=(
            "Verify the count pertains to the Southern region subset during the relevant monitoring period."
        )
    )

    # Data collection timespan (start to end, Southern region)
    timespan_leaf = evaluator.add_leaf(
        id="Data_Collection_Timespan_Start_to_End_Southern",
        desc="Extracts the data-collection timespan (start month/year to end month/year) specific to the Southern GBR region, as stated in the publication.",
        parent=node,
        critical=True
    )
    timespan_claim = (
        f"The publication states that data for the Southern Great Barrier Reef were collected from "
        f"{extracted.timespan_start} to {extracted.timespan_end}."
    )
    await evaluator.verify(
        claim=timespan_claim,
        node=timespan_leaf,
        sources=all_urls,
        additional_instruction=(
            "Confirm the start and end month/year time window is explicitly tied to the Southern GBR monitoring covered by the publication."
        )
    )


async def verify_decline_computation(
    evaluator: Evaluator,
    parent_node,
    extracted: AIMSGBRPublicationExtraction
) -> None:
    """
    Build and verify the 'Decline_Percentage_Computation' parallel critical subtree.
    This is a pure arithmetic/logical check based on the extracted before/after Southern GBR values.
    """
    node = evaluator.add_parallel(
        id="Decline_Percentage_Computation",
        desc="Provide the overall percentage decline in Southern hard coral cover computed from the extracted before/after values.",
        parent=parent_node,
        critical=True
    )

    # Compute expected decline from extracted strings (may be None if unparsable)
    baseline_num = _parse_percentage_value(extracted.baseline_hard_coral_cvr_pct_southern_2024)
    post_num = _parse_percentage_value(extracted.post_bleaching_hard_coral_cvr_pct_southern_2024)

    computed_decline_str: Optional[str] = None
    if baseline_num is not None and post_num is not None and baseline_num > 0:
        decline_val = ((baseline_num - post_num) / baseline_num) * 100.0
        computed_decline_str = _format_percent_one_decimal(decline_val)

    leaf = evaluator.add_leaf(
        id="Decline_Percentage_Computed_From_Before_After_And_Correct",
        desc="Computes the overall percentage decline using the extracted baseline and post-bleaching Southern GBR coral cover values, and the arithmetic is correct (i.e., the reported decline is consistent with the before/after values as a percentage decline).",
        parent=node,
        critical=True
    )

    if computed_decline_str is not None:
        decline_claim = (
            f"Using the answer's stated Southern GBR values (baseline {extracted.baseline_hard_coral_cvr_pct_southern_2024} in 2024 "
            f"and post-bleaching {extracted.post_bleaching_hard_coral_cvr_pct_southern_2024}), the overall percentage decline in hard coral "
            f"cover is about {computed_decline_str} (rounded to one decimal place). Any decline percentage stated in the answer should "
            f"match this arithmetic result."
        )
    else:
        decline_claim = (
            "The answer does not provide sufficiently numeric baseline and post-bleaching Southern GBR values to compute a valid decline "
            "percentage, so any claim that a correct computed decline is provided should be considered incorrect."
        )

    # Pure logical/arithmetic check; no URL evidence required
    await evaluator.verify(
        claim=decline_claim,
        node=leaf,
        additional_instruction=(
            "Treat minor rounding as acceptable (±0.1). If either input value is missing or non-numeric, consider the computation invalid."
        )
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the AIMS 2024 GBR Southern region bleaching publication task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,   # Enforce sequential gating across major stages
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

    # Extraction
    extracted: AIMSGBRPublicationExtraction = await evaluator.extract(
        prompt=prompt_extract_aims_publication(),
        template_class=AIMSGBRPublicationExtraction,
        extraction_name="aims_publication_southern_metrics"
    )

    # Build main (critical) sequential node mirroring rubric root
    main_node = evaluator.add_sequential(
        id="Complete_Analysis_2024_GBR_Coral_Bleaching_Research",
        desc="Identify the qualifying AIMS-related publication (Aug 2024–Aug 2025) documenting 2024 mass bleaching impacts on the Southern GBR and extract all required Southern-region metrics, including a correctly computed decline percentage, with a verifiable URL.",
        parent=root,
        critical=True
    )

    # Subtree 1: Qualifying publication identification
    await verify_identification(evaluator, main_node, extracted)

    # Subtree 2: Southern region quantitative data
    await verify_southern_data(evaluator, main_node, extracted)

    # Subtree 3: Decline computation correctness
    await verify_decline_computation(evaluator, main_node, extracted)

    # Add custom info for transparency (computed decline)
    baseline_num = _parse_percentage_value(extracted.baseline_hard_coral_cvr_pct_southern_2024)
    post_num = _parse_percentage_value(extracted.post_bleaching_hard_coral_cvr_pct_southern_2024)
    computed_decline = None
    if baseline_num is not None and post_num is not None and baseline_num > 0:
        computed_decline = _format_percent_one_decimal(((baseline_num - post_num) / baseline_num) * 100.0)

    evaluator.add_custom_info(
        info={
            "extracted_publication_title": extracted.publication_title,
            "extracted_release_date": extracted.release_date,
            "publication_url": extracted.publication_url,
            "additional_urls": extracted.additional_urls,
            "southern_baseline_pct_2024": extracted.baseline_hard_coral_cvr_pct_southern_2024,
            "southern_post_bleaching_pct_2024": extracted.post_bleaching_hard_coral_cvr_pct_southern_2024,
            "southern_decline_reported_in_answer": extracted.decline_pct_southern,
            "southern_total_reefs_surveyed": extracted.total_reefs_surveyed_southern,
            "southern_timespan": {
                "start": extracted.timespan_start,
                "end": extracted.timespan_end
            },
            "computed_decline_from_baseline_post": computed_decline
        },
        info_type="extraction_summary",
        info_name="computed_and_extracted_summary"
    )

    return evaluator.get_summary()