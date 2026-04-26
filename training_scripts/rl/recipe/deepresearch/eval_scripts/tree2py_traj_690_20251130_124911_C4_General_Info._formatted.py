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
TASK_ID = "entertainment_facts_2024_2026"
TASK_DESCRIPTION = (
    "I'm compiling a timeline of major entertainment events and releases from late 2024 through early 2026 for a pop culture database. "
    "Please provide the following specific information:\n\n"
    "1. What is the runtime (in minutes) of Episode 4 of Stranger Things Season 5?\n"
    "2. What is the premiere date of Billy Bob Thornton's television show 'Landman' on Paramount+?\n"
    "3. What is the opening night date of Lea Michele's Broadway show 'Chess' at the Imperial Theatre?\n"
    "4. Who was the main/headlining performer at the Detroit Lions Thanksgiving halftime show in 2024?\n"
    "5. What is the scheduled date for the Dancing with the Stars 2026 tour stop in Richmond, Virginia at the Altria Theater?\n\n"
    "For each answer, please provide the specific date (in MM/DD/YYYY format where applicable) or runtime (in minutes), along with a reference URL that confirms this information."
)

# Ground truth / expected values for evaluation according to rubric
EXPECTED_VALUES = {
    "stranger_things_s5e4_runtime_minutes": "83",  # minutes
    "landman_premiere_date_mmddyyyy": "11/17/2024",
    "chess_opening_night_mmddyyyy": "11/16/2025",
    "detroit_lions_thx_2024_headliner": "Jack White",
    "dwts_2026_richmond_mmddyyyy": "03/13/2026",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FactWithSources(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class EntertainmentFactsExtraction(BaseModel):
    stranger_things_s5e4_runtime: Optional[FactWithSources] = None
    landman_premiere_date: Optional[FactWithSources] = None
    lea_michele_chess_opening_date: Optional[FactWithSources] = None
    detroit_lions_thanksgiving_2024_headliner: Optional[FactWithSources] = None
    dwts_2026_richmond_date: Optional[FactWithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_entertainment_facts() -> str:
    return """
    Extract exactly the five requested facts from the answer. For each fact, return:
    - value: the value as stated in the answer (a string; do not invent anything).
    - urls: an array of one or more URLs that the answer explicitly cites as supporting this specific fact.
    
    Facts to extract (map each fact to the specified JSON fields):
    1) Stranger Things S5 Episode 4 runtime (in minutes)
       -> stranger_things_s5e4_runtime.value
       -> stranger_things_s5e4_runtime.urls
       Rules:
       - value should be the numeric minutes (e.g., "83" or "83 minutes"; if the answer uses hours+minutes like "1h 23m", return the exact string as written).

    2) 'Landman' (Billy Bob Thornton) premiere date on Paramount+
       -> landman_premiere_date.value
       -> landman_premiere_date.urls
       Rules:
       - value should be the date string as written in the answer (e.g., "11/17/2024" or "November 17, 2024" or "2024-11-17").

    3) Opening night date for Lea Michele's Broadway show 'Chess' at the Imperial Theatre
       -> lea_michele_chess_opening_date.value
       -> lea_michele_chess_opening_date.urls
       Rules:
       - value should be the date string as written in the answer.

    4) Main/headlining performer at the Detroit Lions Thanksgiving halftime show in 2024
       -> detroit_lions_thanksgiving_2024_headliner.value
       -> detroit_lions_thanksgiving_2024_headliner.urls
       Rules:
       - value should be the performer name exactly as written in the answer.

    5) Scheduled date for the Dancing with the Stars 2026 tour stop in Richmond, VA at the Altria Theater
       -> dwts_2026_richmond_date.value
       -> dwts_2026_richmond_date.urls
       Rules:
       - value should be the date string as written in the answer.

    URL extraction rules:
    - Only include URLs explicitly present in the answer (including markdown links).
    - If a general sources section exists, associate URLs with the specific fact(s) they support if reasonably clear. If unclear, include the most relevant ones.
    - If no URL is present for a fact, return an empty array for that fact's urls.

    If any fact is missing from the answer, set value to null and urls to an empty array for that fact.
    """


# --------------------------------------------------------------------------- #
# Helper: add item nodes                                                      #
# --------------------------------------------------------------------------- #
async def verify_stranger_things_runtime(
    evaluator: Evaluator,
    parent,
    extracted: EntertainmentFactsExtraction
):
    # Parent item node
    item_node = evaluator.add_parallel(
        id="StrangerThingsSeason5Episode4Item",
        desc="Runtime for Stranger Things Season 5 Episode 4, plus a reference URL.",
        parent=parent,
        critical=False
    )

    # Leaf: runtime equals 83 minutes (critical)
    runtime_leaf = evaluator.add_leaf(
        id="RuntimeEquals83Minutes",
        desc="States the runtime as 83 minutes.",
        parent=item_node,
        critical=True,
    )
    # We judge against the answer text; allow equivalent expressions like "1h 23m"
    claim = (
        "In the provided answer, the runtime of Episode 4 of Stranger Things Season 5 equals 83 minutes. "
        "Treat '83', '83 minutes', or equivalent time expressions like '1h 23m' as the same duration."
    )
    await evaluator.verify(
        claim=claim,
        node=runtime_leaf,
        additional_instruction="Focus on whether the answer communicates a duration equivalent to 83 minutes."
    )

    # Leaf: reference URL present (critical) — use a custom node for presence check
    runtime_urls = []
    if extracted and extracted.stranger_things_s5e4_runtime:
        runtime_urls = extracted.stranger_things_s5e4_runtime.urls or []
    evaluator.add_custom_node(
        result=bool(runtime_urls),
        id="RuntimeReferenceURLPresent",
        desc="Provides at least one reference URL for the runtime claim.",
        parent=item_node,
        critical=True
    )


async def verify_landman_premiere(
    evaluator: Evaluator,
    parent,
    extracted: EntertainmentFactsExtraction
):
    item_node = evaluator.add_parallel(
        id="LandmanPremiereDateItem",
        desc="Premiere date for 'Landman' on Paramount+, plus a reference URL.",
        parent=parent,
        critical=False
    )

    date_leaf = evaluator.add_leaf(
        id="PremiereDateEquals11172024",
        desc="States the premiere date as 11/17/2024.",
        parent=item_node,
        critical=True
    )
    claim = (
        "In the provided answer, the premiere date of Billy Bob Thornton's television show 'Landman' on Paramount+ "
        "is November 17, 2024 (11/17/2024). Accept equivalent formats like 'November 17, 2024' or '2024-11-17' as the same date."
    )
    await evaluator.verify(
        claim=claim,
        node=date_leaf,
        additional_instruction="Judge true if the answer communicates the same calendar date, regardless of formatting."
    )

    landman_urls = []
    if extracted and extracted.landman_premiere_date:
        landman_urls = extracted.landman_premiere_date.urls or []
    evaluator.add_custom_node(
        result=bool(landman_urls),
        id="PremiereDateReferenceURLPresent",
        desc="Provides at least one reference URL for the premiere date claim.",
        parent=item_node,
        critical=True
    )


async def verify_lea_michele_chess_opening(
    evaluator: Evaluator,
    parent,
    extracted: EntertainmentFactsExtraction
):
    item_node = evaluator.add_parallel(
        id="LeaMicheleChessOpeningItem",
        desc="Opening night date for Lea Michele's Broadway show 'Chess' at the Imperial Theatre, plus a reference URL.",
        parent=parent,
        critical=False
    )

    opening_leaf = evaluator.add_leaf(
        id="OpeningNightDateEquals11162025",
        desc="States the opening night date as 11/16/2025.",
        parent=item_node,
        critical=True
    )
    claim = (
        "In the provided answer, the opening night date for Lea Michele's Broadway show 'Chess' at the Imperial Theatre "
        "is November 16, 2025 (11/16/2025). Accept equivalent date formats that indicate the same date."
    )
    await evaluator.verify(
        claim=claim,
        node=opening_leaf,
        additional_instruction="Judge true if the date in the answer is clearly the same calendar date, regardless of formatting."
    )

    chess_urls = []
    if extracted and extracted.lea_michele_chess_opening_date:
        chess_urls = extracted.lea_michele_chess_opening_date.urls or []
    evaluator.add_custom_node(
        result=bool(chess_urls),
        id="OpeningNightReferenceURLPresent",
        desc="Provides at least one reference URL for the opening night date claim.",
        parent=item_node,
        critical=True
    )


async def verify_detroit_lions_halftime(
    evaluator: Evaluator,
    parent,
    extracted: EntertainmentFactsExtraction
):
    item_node = evaluator.add_parallel(
        id="DetroitLionsHalftimePerformerItem",
        desc="Main/headlining performer at the Detroit Lions Thanksgiving halftime show in 2024, plus a reference URL.",
        parent=parent,
        critical=False
    )

    performer_leaf = evaluator.add_leaf(
        id="HeadliningPerformerEqualsJackWhite",
        desc="States the headlining performer as Jack White.",
        parent=item_node,
        critical=True
    )
    claim = (
        "In the provided answer, the main/headlining performer at the Detroit Lions Thanksgiving halftime show in 2024 is Jack White. "
        "Allow minor variations like 'Jack White (of The White Stripes)' as equivalent."
    )
    await evaluator.verify(
        claim=claim,
        node=performer_leaf,
        additional_instruction="Focus on whether the answer identifies Jack White as the headliner, allowing minor naming variants."
    )

    lions_urls = []
    if extracted and extracted.detroit_lions_thanksgiving_2024_headliner:
        lions_urls = extracted.detroit_lions_thanksgiving_2024_headliner.urls or []
    evaluator.add_custom_node(
        result=bool(lions_urls),
        id="PerformerReferenceURLPresent",
        desc="Provides at least one reference URL for the performer claim.",
        parent=item_node,
        critical=True
    )


async def verify_dwts_richmond_date(
    evaluator: Evaluator,
    parent,
    extracted: EntertainmentFactsExtraction
):
    item_node = evaluator.add_parallel(
        id="DWTSTourRichmondDateItem",
        desc="Scheduled date for the Dancing with the Stars 2026 tour stop in Richmond, VA at the Altria Theater, plus a reference URL.",
        parent=parent,
        critical=False
    )

    date_leaf = evaluator.add_leaf(
        id="TourStopDateEquals03132026",
        desc="States the Richmond, VA (Altria Theater) tour stop date as 03/13/2026.",
        parent=item_node,
        critical=True
    )
    claim = (
        "In the provided answer, the scheduled date for the Dancing with the Stars 2026 tour stop in Richmond, Virginia at the Altria Theater "
        "is March 13, 2026 (03/13/2026). Accept equivalent date formats that indicate the same date."
    )
    await evaluator.verify(
        claim=claim,
        node=date_leaf,
        additional_instruction="Judge true if the date matches the same calendar day, regardless of formatting."
    )

    dwts_urls = []
    if extracted and extracted.dwts_2026_richmond_date:
        dwts_urls = extracted.dwts_2026_richmond_date.urls or []
    evaluator.add_custom_node(
        result=bool(dwts_urls),
        id="TourStopReferenceURLPresent",
        desc="Provides at least one reference URL for the tour stop date claim.",
        parent=item_node,
        critical=True
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
    Evaluate an answer for the Entertainment Facts Collection task.
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
        default_model=model
    )

    # Optional: Add a grouping node to mirror rubric top-level collection
    collection_node = evaluator.add_parallel(
        id="EntertainmentFactsCollection",
        desc="Provide the 5 requested entertainment facts, each with the correct value per constraints and at least one reference URL.",
        parent=root,
        critical=False
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_entertainment_facts(),
        template_class=EntertainmentFactsExtraction,
        extraction_name="entertainment_facts_extraction"
    )

    # Record expected values as ground truth info
    evaluator.add_ground_truth({
        "expected_values": EXPECTED_VALUES
    }, gt_type="ground_truth_facts")

    # Build and verify each item sub-tree
    await asyncio.gather(
        verify_stranger_things_runtime(evaluator, collection_node, extracted),
        verify_landman_premiere(evaluator, collection_node, extracted),
        verify_lea_michele_chess_opening(evaluator, collection_node, extracted),
        verify_detroit_lions_halftime(evaluator, collection_node, extracted),
        verify_dwts_richmond_date(evaluator, collection_node, extracted),
    )

    return evaluator.get_summary()