import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants and expected ground truth                                    #
# --------------------------------------------------------------------------- #
TASK_ID = "theater_architect_research"
TASK_DESCRIPTION = (
    "Identify the architect who designed the Broadway theater currently hosting "
    "Stranger Things: The First Shadow as of November 2025. In your answer, provide: "
    "(1) the theater's name, (2) the architect's full name, (3) the year the theater "
    "officially opened, (4) the theater's seating capacity, and (5) confirmation that it "
    "meets the minimum seating requirement to be classified as a Broadway theater (500 or more seats)."
)

EXPECTED_VENUE_NAME = "Marquis Theatre"
EXPECTED_ARCHITECT_FULL_NAME = "John C. Portman Jr."
EXPECTED_OPENING_YEAR = "1986"
EXPECTED_SEATING_CAPACITY = "1,611"  # allow variants like 1611 via instructions


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SupportSources(BaseModel):
    venue_hosting: List[str] = Field(default_factory=list)
    architect: List[str] = Field(default_factory=list)
    opening_year: List[str] = Field(default_factory=list)
    seating_capacity: List[str] = Field(default_factory=list)
    broadway_requirement: List[str] = Field(default_factory=list)


class TheaterInfoExtraction(BaseModel):
    theater_name: Optional[str] = None
    architect_full_name: Optional[str] = None
    opening_year: Optional[str] = None
    seating_capacity: Optional[str] = None
    broadway_minimum_confirmation: Optional[str] = None  # free-form confirmation text in the answer
    sources: SupportSources = Field(default_factory=SupportSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theater_info() -> str:
    return """
    Extract the information the answer presents about the Broadway theater hosting “Stranger Things: The First Shadow” (as of November 2025).
    Return a JSON object with the following fields:
    - theater_name: The theater’s name as stated in the answer (string).
    - architect_full_name: The architect’s full name as stated in the answer (string).
    - opening_year: The year the theater officially opened as stated in the answer (string; keep exactly as written, e.g., "1986").
    - seating_capacity: The seating capacity as stated in the answer (string; keep punctuation like commas if present, e.g., "1,611").
    - broadway_minimum_confirmation: The exact phrase/sentence in the answer that confirms the theater meets the Broadway minimum seating requirement (500+ seats). If not explicitly stated, return null.
    - sources: An object containing arrays of URLs explicitly cited in the answer to support each fact. Only include URLs that actually appear in the answer text (plain links or markdown links):
        * venue_hosting: URLs that support the venue identification/that Marquis Theatre is hosting Stranger Things: The First Shadow (as of Nov 2025).
        * architect: URLs that support the architect’s identity.
        * opening_year: URLs that support the opening year.
        * seating_capacity: URLs that support the seating capacity.
        * broadway_requirement: URLs that support the Broadway 500+ seating requirement and/or the basis of your confirmation that the theater meets that requirement.

    IMPORTANT:
    - Do not invent URLs. Extract only URLs truly present in the answer.
    - If a field is missing from the answer, return null for that field (or an empty list for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        u_norm = (u or "").strip()
        if not u_norm:
            continue
        if u_norm not in seen:
            seen.add(u_norm)
            out.append(u_norm)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_content_correctness(
    evaluator: Evaluator,
    parent_node,
) -> None:
    """
    Build the 'Content_Correctness' parallel node and its critical leaf checks.
    These checks verify that the answer provides the exact required fields/values.
    """
    content_node = evaluator.add_parallel(
        id="Content_Correctness",
        desc="All required factual fields are provided and match the stated constraints.",
        parent=parent_node,
        critical=True,
    )

    # 1) Venue identification as of Nov 2025 => Marquis Theatre
    venue_leaf = evaluator.add_leaf(
        id="Venue_Identification_AsOfDate",
        desc="Correctly identifies the hosting venue as of November 2025 as the Marquis Theatre (i.e., the named theater is the one currently hosting Stranger Things: The First Shadow at that time).",
        parent=content_node,
        critical=True,
    )
    venue_claim = (
        f"The answer identifies the hosting venue as of November 2025 for 'Stranger Things: The First Shadow' "
        f"as the {EXPECTED_VENUE_NAME}."
    )

    # 2) Architect full name => John C. Portman Jr.
    architect_leaf = evaluator.add_leaf(
        id="Architect_Full_Name",
        desc="Provide the architect's full name as John C. Portman Jr.",
        parent=content_node,
        critical=True,
    )
    architect_claim = (
        f"The answer states that the architect who designed the {EXPECTED_VENUE_NAME} is {EXPECTED_ARCHITECT_FULL_NAME}."
    )

    # 3) Opening year => 1986
    opening_leaf = evaluator.add_leaf(
        id="Opening_Year",
        desc="State that the theater officially opened in 1986.",
        parent=content_node,
        critical=True,
    )
    opening_claim = "The answer states that the theater officially opened in 1986."

    # 4) Seating capacity => 1,611 seats
    capacity_leaf = evaluator.add_leaf(
        id="Seating_Capacity",
        desc="Provide the theater's seating capacity as 1,611 seats.",
        parent=content_node,
        critical=True,
    )
    capacity_claim = (
        "The answer states that the theater's seating capacity is 1,611 seats."
    )

    # 5) Broadway minimum seating confirmation (>= 500)
    min_bway_leaf = evaluator.add_leaf(
        id="Broadway_Minimum_Seating_Confirmation",
        desc="Confirm the theater meets the Broadway minimum seating requirement (500 or more seats).",
        parent=content_node,
        critical=True,
    )
    min_bway_claim = (
        "The answer explicitly confirms that the theater meets the Broadway minimum seating requirement (500 or more seats)."
    )

    # Batch verify content nodes (simple checks against the answer text)
    await evaluator.batch_verify([
        (
            venue_claim,
            None,
            venue_leaf,
            "Allow minor wording variations for the venue line. Focus on whether the answer identifies Marquis Theatre as the hosting venue as of November 2025."
        ),
        (
            architect_claim,
            None,
            architect_leaf,
            "Allow minor formatting variants in the name (e.g., punctuation or spacing), but the architect must be John C. Portman Jr."
        ),
        (
            opening_claim,
            None,
            opening_leaf,
            "Treat 'opened in 1986' as correct even if phrased as 'officially opened (1986)'."
        ),
        (
            capacity_claim,
            None,
            capacity_leaf,
            "Accept minor formatting variants like '1611', '1,611', or '1 611', and with/without the word 'seats'."
        ),
        (
            min_bway_claim,
            None,
            min_bway_leaf,
            "The answer must contain an explicit confirmation statement, not just an implied one."
        ),
    ])


async def build_source_urls_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: TheaterInfoExtraction
) -> None:
    """
    Build the 'Source_URLs' parallel node and its critical leaf checks.
    Each leaf verifies that at least one of the provided URLs actually supports the stated claim.
    If no URLs are provided for a leaf, the check should fail.
    """
    sources_node = evaluator.add_parallel(
        id="Source_URLs",
        desc="Provide reference URL(s) from gathered sources that support each required factual claim.",
        parent=parent_node,
        critical=True,
    )

    # Collect URLs (deduped). If no URLs provided, we'll still run verification with instruction to fail.
    urls_venue = _dedupe_urls(extracted.sources.venue_hosting if extracted and extracted.sources else [])
    urls_arch = _dedupe_urls(extracted.sources.architect if extracted and extracted.sources else [])
    urls_open = _dedupe_urls(extracted.sources.opening_year if extracted and extracted.sources else [])
    urls_cap = _dedupe_urls(extracted.sources.seating_capacity if extracted and extracted.sources else [])
    # For the Broadway requirement, also allow capacity URLs as supportive evidence
    urls_bway_req = _dedupe_urls((extracted.sources.broadway_requirement if extracted and extracted.sources else []) + urls_cap)

    # 1) Sources for venue and hosting claim
    venue_src_leaf = evaluator.add_leaf(
        id="Sources_For_Venue_And_Hosting",
        desc="At least one URL supports the venue identification/hosting-as-of-Nov-2025 claim.",
        parent=sources_node,
        critical=True,
    )
    venue_src_claim = (
        f"As of November 2025, 'Stranger Things: The First Shadow' is hosted at the {EXPECTED_VENUE_NAME}."
    )
    await evaluator.verify(
        claim=venue_src_claim,
        node=venue_src_leaf,
        sources=urls_venue if urls_venue else None,
        additional_instruction=(
            "Use only the provided URL(s). Confirm the page shows the show 'Stranger Things: The First Shadow' "
            f"at the {EXPECTED_VENUE_NAME} (or an unambiguous schedule/venue listing around that time). "
            "If no URL is provided, you must return 'Incorrect'."
        )
    )

    # 2) Sources for architect claim
    arch_src_leaf = evaluator.add_leaf(
        id="Sources_For_Architect",
        desc="At least one URL supports the architect claim.",
        parent=sources_node,
        critical=True,
    )
    arch_src_claim = (
        f"The {EXPECTED_VENUE_NAME} was designed by {EXPECTED_ARCHITECT_FULL_NAME}."
    )
    await evaluator.verify(
        claim=arch_src_claim,
        node=arch_src_leaf,
        sources=urls_arch if urls_arch else None,
        additional_instruction=(
            "Use only the provided URL(s). Allow minor name formatting variants (e.g., punctuation). "
            "If no URL is provided, you must return 'Incorrect'."
        )
    )

    # 3) Sources for opening year claim
    open_src_leaf = evaluator.add_leaf(
        id="Sources_For_Opening_Year",
        desc="At least one URL supports the opening year claim.",
        parent=sources_node,
        critical=True,
    )
    open_src_claim = f"The {EXPECTED_VENUE_NAME} officially opened in {EXPECTED_OPENING_YEAR}."
    await evaluator.verify(
        claim=open_src_claim,
        node=open_src_leaf,
        sources=urls_open if urls_open else None,
        additional_instruction=(
            "Use only the provided URL(s). Accept wording like 'Opened: 1986' or 'officially opened in 1986'. "
            "If no URL is provided, you must return 'Incorrect'."
        )
    )

    # 4) Sources for seating capacity claim
    cap_src_leaf = evaluator.add_leaf(
        id="Sources_For_Seating_Capacity",
        desc="At least one URL supports the seating capacity claim.",
        parent=sources_node,
        critical=True,
    )
    cap_src_claim = f"The seating capacity of the {EXPECTED_VENUE_NAME} is {EXPECTED_SEATING_CAPACITY}."
    await evaluator.verify(
        claim=cap_src_claim,
        node=cap_src_leaf,
        sources=urls_cap if urls_cap else None,
        additional_instruction=(
            "Use only the provided URL(s). Accept minor formatting variants like '1611' vs '1,611'. "
            "If no URL is provided, you must return 'Incorrect'."
        )
    )

    # 5) Sources for Broadway 500+ requirement and/or confirmation basis
    bway_req_leaf = evaluator.add_leaf(
        id="Sources_For_Broadway_500_Plus_Requirement",
        desc="At least one URL supports the Broadway-theater 500+ seats requirement and/or the classification basis used for the confirmation.",
        parent=sources_node,
        critical=True,
    )
    bway_req_claim = (
        f"The {EXPECTED_VENUE_NAME} meets the Broadway minimum seating requirement (500 or more seats): "
        "either because Broadway theaters are defined as having 500+ seats or because the Marquis Theatre "
        "has at least 500 seats (1,611)."
    )
    await evaluator.verify(
        claim=bway_req_claim,
        node=bway_req_leaf,
        sources=urls_bway_req if urls_bway_req else None,
        additional_instruction=(
            "Use only the provided URL(s). It is sufficient if a page explicitly states that Broadway theaters "
            "are defined as 500+ seats (general rule) OR it explicitly confirms that the Marquis Theatre "
            "has at least 500 seats (capacity evidence). If no URL is provided, you must return 'Incorrect'."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the theatre architect research task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root stays non-critical container
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
        prompt=prompt_extract_theater_info(),
        template_class=TheaterInfoExtraction,
        extraction_name="theater_info_extraction",
    )

    # Add ground truth for transparency in the summary
    evaluator.add_ground_truth(
        {
            "expected_theater_name": EXPECTED_VENUE_NAME,
            "expected_architect_full_name": EXPECTED_ARCHITECT_FULL_NAME,
            "expected_opening_year": EXPECTED_OPENING_YEAR,
            "expected_seating_capacity": EXPECTED_SEATING_CAPACITY,
            "broadway_minimum_requirement": ">= 500 seats",
        },
        gt_type="expected_values",
    )

    # Build the main research node (critical, sequential)
    research_node = evaluator.add_sequential(
        id="Theater_Architect_Research",
        desc="Identify the Broadway theater hosting Stranger Things: The First Shadow (as of Nov 2025) and provide the requested architect and theater specifications, with supporting URLs.",
        parent=root,
        critical=True,
    )

    # Child 1: Content correctness (critical, parallel)
    await build_content_correctness(evaluator, research_node)

    # Child 2: Source URLs (critical, parallel) – auto-skipped if Content fails (due to sequential parent)
    await build_source_urls_checks(evaluator, research_node, extracted)

    # Return the final structured evaluation summary
    return evaluator.get_summary()