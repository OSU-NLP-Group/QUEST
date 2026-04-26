import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nba_all_star_2026_venue"
TASK_DESCRIPTION = """
What is the name and complete street address of the venue that hosted the 2026 NBA All-Star Game? Please provide a reference URL that confirms this information.
"""

EXPECTED = {
    "event_name": "2026 NBA All-Star Game",
    "edition": "75th",
    "event_date": "February 15, 2026",
    "venue_name": "Intuit Dome",
    "venue_address": "3930 W Century Blvd, Inglewood, CA 90303",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueAnswerExtraction(BaseModel):
    event_reference: Optional[str] = None         # e.g., "2026 NBA All-Star Game"
    edition: Optional[str] = None                 # e.g., "75th" or "75"
    event_date: Optional[str] = None              # e.g., "February 15, 2026"
    venue_name: Optional[str] = None              # e.g., "Intuit Dome"
    venue_address: Optional[str] = None           # e.g., "3930 W Century Blvd, Inglewood, CA 90303"
    reference_urls: List[str] = Field(default_factory=list)  # all URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_answer() -> str:
    return """
    Extract the specific details the answer provides about the 2026 NBA All-Star Game and its venue. Return a JSON with these fields:
    - event_reference: The explicit event mention (e.g., "2026 NBA All-Star Game"). If not stated, return null.
    - edition: The edition stated for the event (e.g., "75th" or "75"). If not stated, return null.
    - event_date: The event date stated in the answer (e.g., "February 15, 2026" or "Feb 15, 2026"). If not stated, return null.
    - venue_name: The venue name provided (e.g., "Intuit Dome"). If not provided, return null.
    - venue_address: The complete street address provided for the venue (e.g., "3930 W Century Blvd, Inglewood, CA 90303"). If not provided, return null.
    - reference_urls: All URLs explicitly included in the answer, especially those intended as references corroborating the venue information. Extract actual URLs (including from markdown).
    Only extract what is explicitly present in the answer text. Do not infer or add information.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_venue_answer(evaluator: Evaluator, parent_node, extracted: VenueAnswerExtraction) -> None:
    # Create the main critical parallel node (as per rubric)
    main_node = evaluator.add_parallel(
        id="2026_NBA_All_Star_Game_Venue_Answer",
        desc="Verify the response provides the correct venue name and complete street address for the venue that hosted the 2026 NBA All-Star Game, and includes a reference URL corroborating the information.",
        parent=parent_node,
        critical=True
    )

    # 1) Correct_Event_And_Edition_Specified (critical leaf)
    node_event_edition = evaluator.add_leaf(
        id="Correct_Event_And_Edition_Specified",
        desc="The answer makes clear it is about the 2026 NBA All-Star Game and identifies it as the 75th edition.",
        parent=main_node,
        critical=True
    )
    claim_event_edition = (
        "The answer explicitly focuses on the 2026 NBA All-Star Game and states that it is the 75th edition."
    )
    await evaluator.verify(
        claim=claim_event_edition,
        node=node_event_edition,
        additional_instruction="Accept minor phrasing variants like '75th NBA All-Star Game', 'the 75th edition', or '75th'."
    )

    # 2) Correct_Event_Date_Stated (critical leaf)
    node_event_date = evaluator.add_leaf(
        id="Correct_Event_Date_Stated",
        desc="The answer states the event date as February 15, 2026.",
        parent=main_node,
        critical=True
    )
    claim_event_date = "The answer states the event date as February 15, 2026."
    await evaluator.verify(
        claim=claim_event_date,
        node=node_event_date,
        additional_instruction="Treat 'Feb 15, 2026' and 'February 15, 2026' as equivalent; punctuation variations are acceptable."
    )

    # 3) Venue_Name_Matches_Constraint (critical leaf)
    node_venue_name = evaluator.add_leaf(
        id="Venue_Name_Matches_Constraint",
        desc="The venue name is given as Intuit Dome.",
        parent=main_node,
        critical=True
    )
    claim_venue_name = "The answer provides the venue name as 'Intuit Dome'."
    await evaluator.verify(
        claim=claim_venue_name,
        node=node_venue_name,
        additional_instruction="Allow minor case differences and surrounding words (e.g., 'the Intuit Dome'). Focus on whether 'Intuit Dome' is the stated venue."
    )

    # 4) Venue_Address_Matches_Constraint (critical leaf)
    node_venue_address = evaluator.add_leaf(
        id="Venue_Address_Matches_Constraint",
        desc="The complete street address is given as 3930 W Century Blvd, Inglewood, CA 90303.",
        parent=main_node,
        critical=True
    )
    claim_venue_address = "The answer provides the complete street address as '3930 W Century Blvd, Inglewood, CA 90303'."
    await evaluator.verify(
        claim=claim_venue_address,
        node=node_venue_address,
        additional_instruction="Accept equivalent formatting (e.g., 'W.' vs 'West', optional commas/periods) if it clearly refers to the same complete address."
    )

    # 5) Reference_URL_Provided_And_Corroborates (critical leaf)
    node_reference = evaluator.add_leaf(
        id="Reference_URL_Provided_And_Corroborates",
        desc="At least one reference URL is provided that corroborates the venue name and address in connection with hosting the 2026 NBA All-Star Game.",
        parent=main_node,
        critical=True
    )
    # Sources: use extracted reference URLs (may be empty; failure is expected in that case)
    sources = extracted.reference_urls if extracted and extracted.reference_urls else []
    claim_reference = (
        "This webpage confirms that the 2026 NBA All-Star Game was hosted (or will be hosted) at Intuit Dome and "
        "also shows the venue's address as 3930 W Century Blvd, Inglewood, CA 90303. "
        "Both facts must be present on the same page to count as corroboration."
    )
    await evaluator.verify(
        claim=claim_reference,
        node=node_reference,
        sources=sources,
        additional_instruction=(
            "Pass if the page clearly indicates Intuit Dome as the host venue for the 2026 NBA All-Star Game "
            "and, somewhere on the same page, lists the venue's street address as 3930 W Century Blvd, Inglewood, CA 90303. "
            "The two pieces of evidence can appear in different sections of the same page. "
            "Allow minor address formatting variations (e.g., 'W.' vs 'West'). If no single provided URL contains both facts, mark as not supported."
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
    # Initialize evaluator
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

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth({
        "expected_event": EXPECTED["event_name"],
        "expected_edition": EXPECTED["edition"],
        "expected_event_date": EXPECTED["event_date"],
        "expected_venue_name": EXPECTED["venue_name"],
        "expected_venue_address": EXPECTED["venue_address"]
    })

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_answer(),
        template_class=VenueAnswerExtraction,
        extraction_name="venue_answer_extraction"
    )

    # Build tree and verify according to rubric
    await verify_venue_answer(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()