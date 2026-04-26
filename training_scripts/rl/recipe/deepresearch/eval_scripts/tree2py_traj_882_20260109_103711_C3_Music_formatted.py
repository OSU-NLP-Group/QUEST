import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "pulitzer_music_2024"
TASK_DESCRIPTION = """
Identify the composer and title of the work that won the 2024 Pulitzer Prize for Music. Then determine the specific venue and date where this work received its world premiere. Finally, identify which organization(s) commissioned this work and specify the name of the broader commissioning program initiative under which this commission was made.
"""


class PulitzerMusicExtraction(BaseModel):
    composer: Optional[str] = None
    work_title: Optional[str] = None
    winner_sources: List[str] = Field(default_factory=list)

    world_premiere_venue: Optional[str] = None
    world_premiere_date: Optional[str] = None
    premiere_sources: List[str] = Field(default_factory=list)

    commissioning_organizations: List[str] = Field(default_factory=list)
    commissioning_program_initiative: Optional[str] = None
    commission_sources: List[str] = Field(default_factory=list)


def prompt_extract_pulitzer_music() -> str:
    return """
    Extract the following information strictly from the provided answer text about the 2024 Pulitzer Prize for Music:

    1) composer: The name of the composer who won the 2024 Pulitzer Prize for Music (as stated in the answer).
    2) work_title: The full official title of the winning work (as stated in the answer). Preserve punctuation and capitalization where possible.
    3) winner_sources: All URLs cited in the answer that directly support the identification of the 2024 Pulitzer Prize for Music winner and the winning work. Include plain URLs or Markdown links; return actual URL strings.

    4) world_premiere_venue: The specific venue where the winning work received its world premiere (not a later or U.S. premiere).
    5) world_premiere_date: The date of the world premiere (use the format as presented in the answer, e.g., 'May 6, 2023' or '2023-05-06').
    6) premiere_sources: All URLs cited in the answer that support the world premiere venue and date.

    7) commissioning_organizations: A list of all organizations that commissioned the work, as explicitly stated in the answer. Include each organization as a separate string entry.
    8) commissioning_program_initiative: The name of the broader commissioning program initiative under which the commission was made, if provided in the answer.
    9) commission_sources: All URLs cited in the answer that support the commissioning organizations and the commissioning program initiative.

    Rules:
    - Extract only what the answer explicitly provides; do not infer or invent.
    - For URL fields (winner_sources, premiere_sources, commission_sources), include only valid URLs explicitly present in the answer. If a URL lacks http/https, prepend http://.
    - If any field is missing in the answer, return null for that field (or an empty list for list fields).
    """


def _natural_join(items: List[str]) -> str:
    items = [s.strip() for s in items if s and s.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


async def verify_pulitzer_winner(
    evaluator: Evaluator,
    parent_node,
    info: PulitzerMusicExtraction,
) -> None:
    winner_node = evaluator.add_parallel(
        id="Pulitzer_Winner_Identification",
        desc="Correctly identify the 2024 Pulitzer Prize for Music winning work.",
        parent=parent_node,
        critical=True,
    )

    composer_leaf = evaluator.add_leaf(
        id="Composer_Correct",
        desc="Composer is correctly identified as the 2024 Pulitzer Prize for Music winner.",
        parent=winner_node,
        critical=True,
    )
    composer_val = info.composer or ""
    composer_claim = (
        f"The composer of the 2024 Pulitzer Prize for Music winning work is {composer_val}."
    )
    await evaluator.verify(
        claim=composer_claim,
        node=composer_leaf,
        sources=info.winner_sources,
        additional_instruction=(
            "Verify against the cited sources that for the 2024 Pulitzer Prize for Music, "
            f"the named composer '{composer_val}' is indeed the winner (typically phrased as 'awarded to' or 'won by'). "
            "Minor variations in name formatting are acceptable (e.g., middle initials)."
        ),
    )

    title_leaf = evaluator.add_leaf(
        id="Title_Correct",
        desc="Title of the winning work is correctly stated (full official title).",
        parent=winner_node,
        critical=True,
    )
    work_title_val = info.work_title or ""
    title_claim = (
        f"The title of the 2024 Pulitzer Prize for Music winning work is '{work_title_val}'."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=info.winner_sources,
        additional_instruction=(
            "Check the cited sources to confirm the exact (or trivially equivalent) official title of the 2024 Pulitzer Prize for Music winning work. "
            "Allow minor punctuation/capitalization variants."
        ),
    )


async def verify_world_premiere(
    evaluator: Evaluator,
    parent_node,
    info: PulitzerMusicExtraction,
) -> None:
    premiere_node = evaluator.add_parallel(
        id="World_Premiere_Details",
        desc="Correctly identify where and when the winning work received its world premiere.",
        parent=parent_node,
        critical=True,
    )

    venue_leaf = evaluator.add_leaf(
        id="World_Premiere_Venue_Correct",
        desc="World premiere venue is correctly identified (must be the world premiere, not a later/U.S. premiere).",
        parent=premiere_node,
        critical=True,
    )
    work_ref = info.work_title or "the 2024 Pulitzer Prize for Music winning work"
    venue_val = info.world_premiere_venue or ""
    venue_claim = f"The world premiere of '{work_ref}' took place at {venue_val}."
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        sources=info.premiere_sources,
        additional_instruction=(
            "Verify that the cited sources explicitly indicate a 'world premiere' at the specified venue. "
            "Do not confuse U.S. premiere or later performances with the world premiere."
        ),
    )

    date_leaf = evaluator.add_leaf(
        id="World_Premiere_Date_Correct",
        desc="World premiere date is correctly identified (must correspond to the world premiere).",
        parent=premiere_node,
        critical=True,
    )
    date_val = info.world_premiere_date or ""
    date_claim = f"The world premiere date of '{work_ref}' was {date_val}."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=info.premiere_sources,
        additional_instruction=(
            "Confirm that the date provided corresponds to the world premiere (as stated in the sources). "
            "Allow reasonable date format variants such as 'May 6, 2023' vs '2023-05-06'."
        ),
    )


async def verify_commissioning(
    evaluator: Evaluator,
    parent_node,
    info: PulitzerMusicExtraction,
) -> None:
    commission_node = evaluator.add_parallel(
        id="Commissioning_Information",
        desc="Correctly identify commissioning organizations and the broader commissioning program initiative for the work.",
        parent=parent_node,
        critical=True,
    )

    orgs_leaf = evaluator.add_leaf(
        id="Commissioning_Organizations_All",
        desc="All commissioning organizations for the work are identified (none missing).",
        parent=commission_node,
        critical=True,
    )
    orgs_text = _natural_join(info.commissioning_organizations)
    work_ref = info.work_title or "the 2024 Pulitzer Prize for Music winning work"
    orgs_claim = (
        f"The work '{work_ref}' was commissioned by the following organizations: {orgs_text}."
        if orgs_text
        else f"The work '{work_ref}' is stated to have commissioning organizations listed as none."
    )
    await evaluator.verify(
        claim=orgs_claim,
        node=orgs_leaf,
        sources=info.commission_sources,
        additional_instruction=(
            "Check the cited sources to confirm that the listed commissioning organizations match exactly those presented. "
            "If the source lists additional commissioners not in the claim, consider the claim incorrect."
        ),
    )

    program_leaf = evaluator.add_leaf(
        id="Commissioning_Program_Initiative_Correct",
        desc="The name of the broader commissioning program initiative under which the commission was made is correctly stated.",
        parent=commission_node,
        critical=True,
    )
    program_val = info.commissioning_program_initiative or ""
    program_claim = (
        f"The commission for '{work_ref}' was made under the broader commissioning program initiative named '{program_val}'."
        if program_val
        else f"No broader commissioning program initiative is named for '{work_ref}'."
    )
    await evaluator.verify(
        claim=program_claim,
        node=program_leaf,
        sources=info.commission_sources,
        additional_instruction=(
            "Verify whether the cited sources explicitly name a commissioning program/initiative (e.g., a named program umbrella) "
            "under which the commission was made; confirm it matches the provided name."
        ),
    )


async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Honor rubric's critical root requirement by setting root critical before adding children
    root.critical = True

    extraction = await evaluator.extract(
        prompt=prompt_extract_pulitzer_music(),
        template_class=PulitzerMusicExtraction,
        extraction_name="pulitzer_music_extraction",
    )

    await verify_pulitzer_winner(evaluator, root, extraction)
    await verify_world_premiere(evaluator, root, extraction)
    await verify_commissioning(evaluator, root, extraction)

    return evaluator.get_summary()