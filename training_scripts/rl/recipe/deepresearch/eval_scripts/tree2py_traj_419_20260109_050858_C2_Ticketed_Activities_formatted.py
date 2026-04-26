import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ikon_co_highest_summit_and_price"
TASK_DESCRIPTION = (
    "Among all Colorado ski resorts that are accessible with an Ikon Pass, identify the one with the highest "
    "summit elevation. Then, provide the adult window rate (walk-up price) for an all-mountain lift ticket at that resort."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResortSelectionExtraction(BaseModel):
    """
    Information about the selected resort and supporting sources as explicitly stated in the answer.
    All URLs must be extracted exactly as they appear in the answer. Do not invent URLs.
    """
    resort_name: Optional[str] = None

    # Eligibility evidence
    eligibility_colorado_sources: List[str] = Field(default_factory=list)
    eligibility_ikon_sources: List[str] = Field(default_factory=list)

    # Elevation information for the selected resort
    summit_elevation: Optional[str] = None  # Keep as free-form text (e.g., "13,050 ft", "3,978 m")
    elevation_sources: List[str] = Field(default_factory=list)

    # Sources that support the "highest among eligible CO Ikon resorts" claim
    highest_claim_sources: List[str] = Field(default_factory=list)

    # Pricing information
    adult_window_rate: Optional[str] = None  # Free-form (e.g., "$199", "about $200", "varies by day: walk-up $219+")
    price_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_resort_selection() -> str:
    return """
    From the provided answer, extract the following fields about the identified resort and its supporting evidence.

    Fields to extract:
    1) resort_name: The single resort chosen as the answer (string).
    2) eligibility_colorado_sources: List of URLs explicitly cited in the answer that support the resort being located in Colorado.
    3) eligibility_ikon_sources: List of URLs explicitly cited in the answer that support that this resort is accessible with an Ikon Pass.
    4) summit_elevation: The stated summit elevation for the selected resort, exactly as written in the answer (string, keep units and formatting).
    5) elevation_sources: List of URLs that directly support the summit elevation of the selected resort (e.g., the resort page, a stats page).
    6) highest_claim_sources: List of URLs supporting the comparative claim that, among Colorado ski resorts accessible with an Ikon Pass, the selected resort has the highest summit elevation (e.g., authoritative comparison/list pages).
    7) adult_window_rate: The adult window (walk-up) rate for an all-mountain single-day lift ticket at the selected resort, exactly as written in the answer (string).
    8) price_sources: List of URLs that support the adult window (walk-up) rate (ideally the resort's ticketing page showing window pricing).

    Rules:
    - Extract only what is explicitly present in the answer.
    - If any field is missing in the answer, set it to null (for strings) or an empty list (for URL lists).
    - For URL fields, include only valid URLs cited in the answer text (plain URL or markdown link). Do not infer or fabricate URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_nonempty(urls: List[str]) -> List[str]:
    """Deduplicate URLs and drop empty/None-like entries."""
    seen = set()
    result: List[str] = []
    for u in urls or []:
        if not u:
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu not in seen:
            seen.add(uu)
            result.append(uu)
    return result


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extraction: ResortSelectionExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    Tree structure (matching the provided rubric):

    Root (created in initialize) ->
      Colorado_Ski_Resort_Task (critical, sequential)
        ├─ Resort_Identification (critical, parallel)
        │    ├─ Eligibility_Criteria (critical, parallel)
        │    │    ├─ Colorado_Location (critical, leaf)
        │    │    └─ Ikon_Pass_Access (critical, leaf)
        │    └─ Highest_Summit_Elevation_Among_Eligible (critical, leaf)
        └─ Ticket_Price_Information (critical, leaf)
    """
    # Create the critical main node to reflect the rubric's top-level "CRITICAL sequential"
    main_node = evaluator.add_sequential(
        id="Colorado_Ski_Resort_Task",
        desc="Identify the Colorado ski resort accessible with an Ikon Pass that has the highest summit elevation, then provide the adult window (walk-up) rate for an all-mountain lift ticket at that resort.",
        parent=evaluator.root,
        critical=True,
    )

    # Child 1: Resort_Identification (critical, parallel)
    resort_identification = evaluator.add_parallel(
        id="Resort_Identification",
        desc="Identify the correct resort by verifying eligibility and that it has the maximum summit elevation among eligible resorts.",
        parent=main_node,
        critical=True,
    )

    # 1a) Eligibility_Criteria (critical, parallel)
    eligibility = evaluator.add_parallel(
        id="Eligibility_Criteria",
        desc="The identified resort meets the eligibility constraints.",
        parent=resort_identification,
        critical=True,
    )

    # Leaves for eligibility
    colorado_node = evaluator.add_leaf(
        id="Colorado_Location",
        desc="The identified resort is located in Colorado.",
        parent=eligibility,
        critical=True,
    )
    ikon_access_node = evaluator.add_leaf(
        id="Ikon_Pass_Access",
        desc="The identified resort is accessible with an Ikon Pass.",
        parent=eligibility,
        critical=True,
    )

    resort_name = (extraction.resort_name or "").strip()

    colorado_sources = _unique_nonempty(extraction.eligibility_colorado_sources)
    ikon_sources = _unique_nonempty(extraction.eligibility_ikon_sources)

    # Verify eligibility leaves (can be parallelized)
    claims_and_sources = [
        (
            f"The resort named '{resort_name}' is located in the state of Colorado, United States.",
            colorado_sources if colorado_sources else None,
            colorado_node,
            "Look for explicit mention of Colorado (CO), the resort's address within Colorado, or clear location statements on the provided pages."
        ),
        (
            f"The resort named '{resort_name}' is accessible with an Ikon Pass.",
            ikon_sources if ikon_sources else None,
            ikon_access_node,
            "Accept if the provided evidence clearly indicates Ikon Pass access for this resort (e.g., Ikon’s official site or resort page listing Ikon partnership/acceptance)."
        ),
    ]
    await evaluator.batch_verify(claims_and_sources)

    # 1b) Highest summit elevation among eligible (critical, leaf)
    highest_leaf = evaluator.add_leaf(
        id="Highest_Summit_Elevation_Among_Eligible",
        desc="Among all ski resorts satisfying the eligibility criteria above, the identified resort has the highest summit elevation.",
        parent=resort_identification,
        critical=True,
    )

    summit_elevation_text = (extraction.summit_elevation or "").strip()
    elevation_sources = _unique_nonempty(extraction.elevation_sources)
    highest_sources = _unique_nonempty(extraction.highest_claim_sources)
    all_highest_sources = _unique_nonempty(elevation_sources + highest_sources + ikon_sources + colorado_sources)

    highest_claim = (
        f"Among all Colorado ski resorts accessible with an Ikon Pass, the resort with the highest summit elevation is '{resort_name}'"
        + (f", with a summit elevation of {summit_elevation_text}." if summit_elevation_text else ".")
    )

    await evaluator.verify(
        claim=highest_claim,
        node=highest_leaf,
        sources=all_highest_sources if all_highest_sources else None,
        additional_instruction=(
            "Verify the comparative claim considering only Colorado resorts that accept the Ikon Pass. "
            "Support should come from authoritative or comprehensive sources (e.g., Ikon pages, resort stats pages, "
            "or reputable listings that compare elevations). If there is a tie for highest, accept if the identified resort "
            "is among those tied for the highest summit elevation. Minor unit differences (ft vs m) and rounding differences are acceptable."
        ),
    )

    # Child 2: Ticket_Price_Information (critical, leaf)
    price_leaf = evaluator.add_leaf(
        id="Ticket_Price_Information",
        desc="Provides the adult window rate (walk-up price) for an all-mountain lift ticket at the identified resort.",
        parent=main_node,
        critical=True,
    )

    price_text = (extraction.adult_window_rate or "").strip()
    price_sources = _unique_nonempty(extraction.price_sources)

    price_claim = (
        f"The adult window (walk-up) rate for an all-mountain single-day lift ticket at '{resort_name}' is {price_text}."
        if price_text else
        f"The answer provides a concrete adult window (walk-up) rate for an all-mountain single-day lift ticket at '{resort_name}'."
    )

    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=price_sources if price_sources else None,
        additional_instruction=(
            "Specifically verify 'window' or 'walk-up' pricing for an adult, for an all-mountain single-day lift ticket. "
            "Do not accept advance-purchase-only or online-discount pricing unless the page clearly labels it as the window/walk-up rate. "
            "If the evidence shows dynamic pricing with a clearly labeled window rate that matches the answer (within reasonable rounding), consider it supported."
        ),
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
    Evaluate an answer for the 'Ikon Colorado highest summit + adult window price' task.
    Returns a structured summary with the verification tree and final score.
    """
    # Initialize evaluator (root is always non-critical by framework design)
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root organizes the two main steps sequentially
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
    selection = await evaluator.extract(
        prompt=prompt_extract_resort_selection(),
        template_class=ResortSelectionExtraction,
        extraction_name="resort_selection",
    )

    # Build rubric tree and run verification
    await build_and_verify_tree(evaluator, selection)

    # Return standardized summary
    return evaluator.get_summary()