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
TASK_ID = "gaten_broadway_career"
TASK_DESCRIPTION = """
Gaten Matarazzo, best known for playing Dustin Henderson in Netflix's Stranger Things, began his professional acting career on Broadway before achieving television fame. Trace his complete Broadway career by identifying all of his Broadway shows in chronological order. For each Broadway show he appeared in, provide: (1) The show title, (2) The name of the Broadway theatre where it was performed, (3) The complete street address of that theatre in New York City, (4) The specific role/character name he played, and (5) Key production dates (opening date, closing date, and/or his final performance date as applicable). Present your answer in chronological order, starting with his first Broadway appearance. Ensure all information is verifiable with reference URLs from reliable sources.
"""

# Ground-truth expectations (for reference recording and claim construction)
EXPECTED_SHOWS = [
    {
        "ordinal_desc": "1st (earliest) Broadway show listed in the answer",
        "title": "Priscilla, Queen of the Desert",
        "theatre": "Palace Theatre",
        "address": "160 W 47th St, New York, NY 10036",
        "role": "Benji",
        "role_context": "Alternate/Replacement",
        "key_dates_desc": "Key production dates include opening date (March 20, 2011) and closing date (June 24, 2012)",
        "key_dates_claim": "The Broadway production opened on March 20, 2011 and closed on June 24, 2012."
    },
    {
        "ordinal_desc": "2nd Broadway show listed in the answer",
        "title": "Les Misérables",
        "theatre": "Imperial Theatre",
        "address": "249 W 45th St, New York, NY 10036",
        "role": "Gavroche",
        "role_context": "Alternate",
        "key_dates_desc": "Key production dates include opening date (March 23, 2014) and closing date (September 4, 2016)",
        "key_dates_claim": "The 2014 Broadway revival opened on March 23, 2014 and closed on September 4, 2016."
    },
    {
        "ordinal_desc": "3rd Broadway show listed in the answer",
        "title": "Dear Evan Hansen",
        "theatre": "Music Box Theatre",
        "address": "239 W 45th St, New York, NY 10036",
        "role": "Jared Kleinman",
        "role_context": "Replacement",
        "key_dates_desc": "Key dates include that he joined in 2022 and the show closed September 18, 2022",
        "key_dates_claim": "Gaten Matarazzo joined the Broadway production in 2022 and the show closed on September 18, 2022."
    },
    {
        "ordinal_desc": "4th (latest) Broadway show listed in the answer",
        "title": "Sweeney Todd",
        "theatre": "Lunt-Fontanne Theatre",
        "address": "205 W 46th St, New York, NY 10036",
        "role": "Tobias",
        "role_context": "Original cast",
        "key_dates_desc": "Key dates include opening date (March 26, 2023) and his final performance date (November 5, 2023)",
        "key_dates_claim": "The Broadway revival opened on March 26, 2023 and Gaten Matarazzo's final performance was on November 5, 2023."
    }
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ShowInfo(BaseModel):
    """Information for one Broadway show."""
    title: Optional[str] = None
    theatre_name: Optional[str] = None
    theatre_address: Optional[str] = None
    role: Optional[str] = None
    opening_date: Optional[str] = None
    closing_date: Optional[str] = None
    final_performance_date: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ShowsExtraction(BaseModel):
    """All Broadway shows listed in the answer (in chronological order)."""
    shows: List[ShowInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_broadway_shows() -> str:
    return """
    Extract Gaten Matarazzo's complete Broadway career from the answer, in chronological order.
    Return a JSON object { "shows": [...] } where each element has the following fields:
    - title: The Broadway show title as stated in the answer.
    - theatre_name: The name of the Broadway theatre where it was performed.
    - theatre_address: The complete NYC street address for that theatre.
    - role: The specific role/character name he played (e.g., Benji, Gavroche, Jared Kleinman, Tobias).
    - opening_date: The production opening date (if given).
    - closing_date: The production closing date (if given).
    - final_performance_date: His final performance date (if applicable and given).
    - reference_urls: An array of all reference URLs cited in the answer specifically for that show; include any relevant IBDB/Playbill/Broadway League/BroadwayWorld/etc. Only include valid URLs explicitly present in the answer.

    Rules:
    1. Only include Broadway shows (New York City Broadway theatres).
    2. Keep chronological order (earliest to latest) as in the answer.
    3. If a field is missing for a show, set it to null (or an empty array for reference_urls).
    4. If the answer lists more than 4 shows, include all; we will only evaluate the first 4 later.
    5. Accept reasonable date formats (e.g., 'March 20, 2011', '2011-03-20').
    6. For 'reference_urls', extract only actual URLs (plain or markdown links). If none are provided, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal_label(idx: int) -> str:
    mapping = {0: "1st (earliest)", 1: "2nd", 2: "3rd", 3: "4th (latest)"}
    return mapping.get(idx, f"{idx + 1}th")


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_show(
        evaluator: Evaluator,
        parent_node,
        show_index: int,
        show: ShowInfo,
        expected: Dict[str, str],
        total_shows_count: int
) -> None:
    """
    Build verification sub-tree and perform checks for a single show.
    All nodes under the show are critical (because the root is critical).
    """

    # Create show-level node (critical because root is critical)
    show_node = evaluator.add_parallel(
        id=f"Show_{show_index + 1}_Chronological",
        desc=expected["ordinal_desc"],
        parent=parent_node,
        critical=True
    )

    # Reference URLs existence (gate subsequent URL verifications)
    has_sources = bool(show.reference_urls) and len(show.reference_urls) > 0
    evaluator.add_custom_node(
        result=has_sources,
        id=f"Show_{show_index + 1}_Reference_URLs",
        desc="Provides at least one reference URL supporting the show/theatre/role/dates information",
        parent=show_node,
        critical=True
    )

    # Show Title (simple check against expected – allow minor title variations)
    title_leaf = evaluator.add_leaf(
        id=f"Show_{show_index + 1}_Show_Title",
        desc=f"Show title is '{expected['title']}'",
        parent=show_node,
        critical=True
    )
    show_title_in_answer = show.title or ""
    title_claim = f"The listed show title for the {ordinal_label(show_index)} Broadway entry is equivalent to '{expected['title']}'."
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        additional_instruction=(
            "Judge equivalence based on the answer text. Allow common musical title variants, "
            "such as including or omitting articles/subtitles/punctuation (e.g., "
            "'Priscilla Queen of the Desert The Musical' vs 'Priscilla, Queen of the Desert'; "
            "'Sweeney Todd: The Demon Barber of Fleet Street' vs 'Sweeney Todd')."
        )
    )

    # Theatre Name (verify with URLs)
    theatre_leaf = evaluator.add_leaf(
        id=f"Show_{show_index + 1}_Theatre_Name",
        desc=f"Broadway theatre name is '{expected['theatre']}'",
        parent=show_node,
        critical=True
    )
    theatre_claim = (
        f"The Broadway production of '{expected['title']}' played at '{expected['theatre']}'."
    )
    await evaluator.verify(
        claim=theatre_claim,
        node=theatre_leaf,
        sources=show.reference_urls,
        additional_instruction=(
            "Use the provided reference URLs to confirm the Broadway theatre where this production ran. "
            "Focus on the Broadway venue, not tours or regional productions."
        )
    )

    # Theatre Street Address (verify with URLs)
    address_leaf = evaluator.add_leaf(
        id=f"Show_{show_index + 1}_Theatre_Street_Address",
        desc=f"Theatre street address is '{expected['address']}'",
        parent=show_node,
        critical=True
    )
    address_claim = (
        f"The street address of the Broadway theatre '{expected['theatre']}' is '{expected['address']}'."
    )
    await evaluator.verify(
        claim=address_claim,
        node=address_leaf,
        sources=show.reference_urls,
        additional_instruction=(
            "Confirm the NYC street address of the theatre using reliable theatre or production pages. "
            "Minor formatting differences (e.g., 'W' vs 'West') are acceptable if equivalent."
        )
    )

    # Role / Character (verify with URLs)
    role_leaf = evaluator.add_leaf(
        id=f"Show_{show_index + 1}_Role_Character",
        desc=f"Role/character is '{expected['role']}' ({expected['role_context']})",
        parent=show_node,
        critical=True
    )
    role_claim = (
        f"Gaten Matarazzo performed in '{expected['title']}' on Broadway as '{expected['role']}', "
        f"with status '{expected['role_context']}' (e.g., alternate/replacement/original as applicable)."
    )
    await evaluator.verify(
        claim=role_claim,
        node=role_leaf,
        sources=show.reference_urls,
        additional_instruction=(
            "Verify the specific role name and casting status (alternate/replacement/original) "
            "from cast lists, production announcements, or reputable theatre databases."
        )
    )

    # Key production dates (verify with URLs)
    dates_leaf = evaluator.add_leaf(
        id=f"Show_{show_index + 1}_Key_Dates",
        desc=expected["key_dates_desc"],
        parent=show_node,
        critical=True
    )
    dates_claim = f"For '{expected['title']}', {expected['key_dates_claim']}"
    await evaluator.verify(
        claim=dates_claim,
        node=dates_leaf,
        sources=show.reference_urls,
        additional_instruction=(
            "Confirm opening/closing dates and/or his final performance date from the provided sources. "
            "Accept minor formatting differences, but the dates must match substantively."
        )
    )

    # For the latest show: ensure exactly four shows total (no additional shows listed)
    if show_index == 3:
        evaluator.add_custom_node(
            result=(total_shows_count == 4),
            id=f"Show_{show_index + 1}_No_Additional_Shows_Listed",
            desc="No additional Broadway shows are listed after this 4th entry (i.e., exactly four shows total)",
            parent=show_node,
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
    Evaluate an answer for Gaten Matarazzo's Broadway career.
    """
    # Initialize evaluator (root is critical parallel per rubric; thus all children must be critical)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluate the complete Broadway career output for Gaten Matarazzo: "
                         "the four shows in chronological order, each with required attributes and at least one supporting reference URL.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )
    # Make root critical by wrapping with a parallel critical node to satisfy framework constraints
    # Since VerificationNode root created by initialize is non-critical by default, we simulate root critical behaviour
    # by adding a single critical child aggregator node "Broadway_Career_Complete_Analysis".
    root_agg = evaluator.add_parallel(
        id="Broadway_Career_Complete_Analysis",
        desc="Evaluate the complete Broadway career output for Gaten Matarazzo: the four shows in chronological order, each with required attributes and at least one supporting reference URL.",
        parent=root,
        critical=True
    )

    # Extract shows from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_broadway_shows(),
        template_class=ShowsExtraction,
        extraction_name="broadway_career"
    )

    # Record ground-truth guidance for transparency
    evaluator.add_ground_truth({
        "expected_sequence": [
            {
                "title": s["title"],
                "theatre": s["theatre"],
                "address": s["address"],
                "role": s["role"],
                "role_context": s["role_context"],
                "key_dates_summary": s["key_dates_desc"]
            }
            for s in EXPECTED_SHOWS
        ]
    })

    # Prepare shows: use first 4; pad if fewer
    shows = list(extracted.shows or [])
    total_shows_count = len(shows)
    if len(shows) < 4:
        shows.extend([ShowInfo() for _ in range(4 - len(shows))])
    else:
        shows = shows[:4]

    # Build verification subtrees for each of the four shows
    for idx in range(4):
        await verify_single_show(
            evaluator=evaluator,
            parent_node=root_agg,
            show_index=idx,
            show=shows[idx],
            expected=EXPECTED_SHOWS[idx],
            total_shows_count=total_shows_count if idx == 3 else len(shows)  # check total count on last
        )

    # Return summary
    return evaluator.get_summary()