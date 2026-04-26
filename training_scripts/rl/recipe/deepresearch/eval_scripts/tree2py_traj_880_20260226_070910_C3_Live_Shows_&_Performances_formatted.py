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
TASK_ID = "broadway_performer_show_parade_2025"
TASK_DESCRIPTION = (
    "A Broadway performer currently starring at a thrust-stage theater in Midtown Manhattan performed at the 2025 Macy's "
    "Thanksgiving Day Parade on November 27, 2025. The Broadway show is a biographical musical about a singer who died at a young age. "
    "Identify: (1) The performer's full name, (2) The date of the performer's announced final performance in this show, "
    "(3) The name of the theater, (4) The theater's seating capacity (provide the most recent documented figure), "
    "(5) The theater's street address, and (6) The show's official opening date."
)

# For a specific rubric check
EXPECTED_SINGER_NAME = "Bobby Darin"

# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class PerformerInfo(BaseModel):
    full_name: Optional[str] = None
    parade_urls: List[str] = Field(default_factory=list)
    current_starring_urls: List[str] = Field(default_factory=list)
    final_performance_date: Optional[str] = None
    final_performance_urls: List[str] = Field(default_factory=list)


class ShowInfo(BaseModel):
    name: Optional[str] = None
    singer_subject_name: Optional[str] = None
    official_opening_date: Optional[str] = None
    overview_urls: List[str] = Field(default_factory=list)
    opening_date_urls: List[str] = Field(default_factory=list)


class TheaterInfo(BaseModel):
    name: Optional[str] = None
    seating_capacity: Optional[str] = None
    street_address: Optional[str] = None
    thrust_stage_urls: List[str] = Field(default_factory=list)
    midtown_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)
    address_urls: List[str] = Field(default_factory=list)


class BroadwayExtraction(BaseModel):
    performer: Optional[PerformerInfo] = None
    show: Optional[ShowInfo] = None
    theater: Optional[TheaterInfo] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_broadway_info() -> str:
    return """
    From the provided answer, extract the following information about the Broadway performer, show, and theater. 
    This must reflect exactly what the answer states. If any item is not present in the answer, set it to null (for strings) or an empty array (for URLs).

    Fields to extract (grouped):

    performer:
      - full_name: The performer's full name.
      - parade_urls: A list of URLs cited in the answer that specifically support that the performer appeared at the 2025 Macy's Thanksgiving Day Parade on November 27, 2025.
      - current_starring_urls: A list of URLs cited in the answer that support that the performer is currently starring in the show as of February 2026.
      - final_performance_date: The announced final performance date for this performer in this show, as written in the answer.
      - final_performance_urls: A list of URLs cited in the answer that support that final performance date.

    show:
      - name: The show's name.
      - singer_subject_name: The singer the show is about (if explicitly mentioned).
      - official_opening_date: The show's official opening date (as provided in the answer).
      - overview_urls: A list of URLs cited in the answer that describe the show (e.g., official site, Playbill/BroadwayWorld, Wikipedia).
      - opening_date_urls: A list of URLs cited in the answer that specifically support the official opening date.
    
    theater:
      - name: The theater name.
      - seating_capacity: The theater’s seating capacity figure stated in the answer (use the most recent figure the answer claims).
      - street_address: The street address of the theater (as provided in the answer).
      - thrust_stage_urls: A list of URLs cited in the answer that support the theater having a thrust stage configuration.
      - midtown_urls: A list of URLs cited in the answer that support that the theater is located in Midtown Manhattan.
      - capacity_urls: A list of URLs cited in the answer that support the capacity figure.
      - address_urls: A list of URLs cited in the answer that support the street address.

    Important:
    - Only extract URLs that are explicitly present in the answer (including those embedded in markdown links).
    - Do NOT invent or infer any information or URLs beyond what the answer provides.
    - If multiple performers/shows/theaters are mentioned, select the primary one that the answer associates with the parade, the thrust-stage theater in Midtown Manhattan, and the biographical musical constraints.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


def _non_empty(text: Optional[str]) -> bool:
    return isinstance(text, str) and text.strip() != ""


def _val_or(text: Optional[str], fallback: str) -> str:
    return text.strip() if _non_empty(text) else fallback


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_performer_subtree(evaluator: Evaluator, parent_node, ex: BroadwayExtraction):
    performer = ex.performer or PerformerInfo()
    show = ex.show or ShowInfo()

    performer_node = evaluator.add_parallel(
        id="Performer",
        desc="Performer is identified and satisfies all performer-related constraints.",
        parent=parent_node,
        critical=True
    )

    # Existence check for full name (as required by rubric)
    evaluator.add_custom_node(
        result=_non_empty(performer.full_name),
        id="Performer_Full_Name_Provided",
        desc="The performer’s full name is provided.",
        parent=performer_node,
        critical=True
    )

    # Parade appearance leaf
    parade_leaf = evaluator.add_leaf(
        id="Parade_Appearance",
        desc="Performer appeared at the 2025 Macy's Thanksgiving Day Parade on November 27, 2025.",
        parent=performer_node,
        critical=True
    )
    parade_claim = f"On November 27, 2025, {_val_or(performer.full_name, 'the performer')} appeared at the 2025 Macy's Thanksgiving Day Parade."
    await evaluator.verify(
        claim=parade_claim,
        node=parade_leaf,
        sources=_safe_list(performer.parade_urls),
        additional_instruction="Verify that the source explicitly states this performer appeared or performed at the 2025 Macy's Thanksgiving Day Parade on November 27, 2025. "
                               "Accept wording like 'performed', 'appeared', 'featured', or participation as part of the show's cast performance."
    )

    # Currently starring as of Feb 2026 leaf
    current_leaf = evaluator.add_leaf(
        id="Currently_Starring_Feb_2026",
        desc="Performer is currently starring in the Broadway show as of February 2026.",
        parent=performer_node,
        critical=True
    )
    current_claim = f"As of February 2026, {_val_or(performer.full_name, 'the performer')} is starring in {_val_or(show.name, 'the Broadway show')}."
    await evaluator.verify(
        claim=current_claim,
        node=current_leaf,
        sources=_safe_list(performer.current_starring_urls),
        additional_instruction="Confirm that the page supports the performer being a current star of the show around February 2026. "
                               "Accept phrases like 'currently starring', 'now starring', 'through February 2026', or listings that clearly indicate active lead status in that timeframe."
    )


async def build_show_subtree(evaluator: Evaluator, parent_node, ex: BroadwayExtraction):
    show = ex.show or ShowInfo()

    show_node = evaluator.add_parallel(
        id="Show",
        desc="The Broadway show satisfies all show-related constraints, including opening date.",
        parent=parent_node,
        critical=True
    )

    # Biographical musical about a singer who died young
    bio_leaf = evaluator.add_leaf(
        id="Biographical_Musical_About_Singer_Died_Young",
        desc="Show is a biographical musical about a singer who died at a young age.",
        parent=show_node,
        critical=True
    )
    singer_fragment = f"about singer {_val_or(show.singer_subject_name, 'who died at a young age')}"
    bio_claim = f"'{_val_or(show.name, 'the show')}' is a biographical musical {singer_fragment} who died at a young age."
    await evaluator.verify(
        claim=bio_claim,
        node=bio_leaf,
        sources=_safe_list(show.overview_urls),
        additional_instruction="Check that the show is a biographical musical and that its subject is a singer who died young. "
                               "The page should indicate music biography and the subject being a singer with an early death (generally under ~40)."
    )

    # Show is about Bobby Darin
    about_leaf = evaluator.add_leaf(
        id="Show_Is_About_Bobby_Darin",
        desc="Show is about Bobby Darin.",
        parent=show_node,
        critical=True
    )
    about_claim = f"'{_val_or(show.name, 'the show')}' is about {EXPECTED_SINGER_NAME}."
    await evaluator.verify(
        claim=about_claim,
        node=about_leaf,
        sources=_safe_list(show.overview_urls),
        additional_instruction=f"Verify that the show is centered on {EXPECTED_SINGER_NAME} (the subject of the musical)."
    )

    # Official opening date is provided (existence check required by rubric)
    evaluator.add_custom_node(
        result=_non_empty(show.official_opening_date),
        id="Official_Opening_Date",
        desc="Show’s official opening date is provided.",
        parent=show_node,
        critical=True
    )


async def build_theater_subtree(evaluator: Evaluator, parent_node, ex: BroadwayExtraction):
    theater = ex.theater or TheaterInfo()

    theater_node = evaluator.add_parallel(
        id="Theater",
        desc="The theater is identified and satisfies all theater-related constraints, including requested details.",
        parent=parent_node,
        critical=True
    )

    # Theater name provided (existence check)
    evaluator.add_custom_node(
        result=_non_empty(theater.name),
        id="Theater_Name",
        desc="The theater name is provided.",
        parent=theater_node,
        critical=True
    )

    # Thrust stage configuration
    thrust_leaf = evaluator.add_leaf(
        id="Thrust_Stage",
        desc="The theater has a thrust stage configuration.",
        parent=theater_node,
        critical=True
    )
    thrust_claim = f"{_val_or(theater.name, 'The theater')} has a thrust stage configuration."
    await evaluator.verify(
        claim=thrust_claim,
        node=thrust_leaf,
        sources=_safe_list(theater.thrust_stage_urls),
        additional_instruction="Verify that the page explicitly mentions 'thrust stage' or equivalent descriptions indicating a stage that extends into the audience on multiple sides."
    )

    # Midtown Manhattan location
    midtown_leaf = evaluator.add_leaf(
        id="Midtown_Manhattan",
        desc="The theater is located in Midtown Manhattan.",
        parent=theater_node,
        critical=True
    )
    midtown_claim = f"{_val_or(theater.name, 'The theater')} is located in Midtown Manhattan."
    await evaluator.verify(
        claim=midtown_claim,
        node=midtown_leaf,
        sources=_safe_list(theater.midtown_urls),
        additional_instruction="Confirm that the page states the theater is in Midtown Manhattan (accept 'Midtown' or explicit neighborhood descriptions that clearly indicate Midtown Manhattan)."
    )

    # Seating capacity (most recent documented figure)
    capacity_leaf = evaluator.add_leaf(
        id="Seating_Capacity_Most_Recent_Documented",
        desc="The theater’s seating capacity is provided using the most recent documented figure.",
        parent=theater_node,
        critical=True
    )
    capacity_claim = f"The seating capacity of {_val_or(theater.name, 'the theater')} is {_val_or(theater.seating_capacity, '[missing figure]')}."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=_safe_list(theater.capacity_urls),
        additional_instruction="Verify that the cited capacity matches the page as the most recent documented figure. "
                               "If multiple numbers are mentioned, prefer the most recent or clearly labeled current capacity; "
                               "accept phrases like 'about X seats' if consistent."
    )

    # Street address
    address_leaf = evaluator.add_leaf(
        id="Street_Address",
        desc="The theater’s street address is provided.",
        parent=theater_node,
        critical=True
    )
    address_claim = f"The street address of {_val_or(theater.name, 'the theater')} is {_val_or(theater.street_address, '[missing address]')}."
    await evaluator.verify(
        claim=address_claim,
        node=address_leaf,
        sources=_safe_list(theater.address_urls),
        additional_instruction="Verify the exact street address (street number and street name). "
                               "It's acceptable if the page includes city/state/ZIP as well."
    )


async def add_final_performance_leaf(evaluator: Evaluator, parent_node, ex: BroadwayExtraction):
    performer = ex.performer or PerformerInfo()
    show = ex.show or ShowInfo()

    final_leaf = evaluator.add_leaf(
        id="Announced_Final_Performance_Date",
        desc="A specific announced final performance date for the performer in this show is provided.",
        parent=parent_node,
        critical=True
    )
    final_claim = f"{_val_or(performer.full_name, 'The performer')}'s announced final performance in '{_val_or(show.name, 'the show')}' is on {_val_or(performer.final_performance_date, '[missing date]')}."
    await evaluator.verify(
        claim=final_claim,
        node=final_leaf,
        sources=_safe_list(performer.final_performance_urls),
        additional_instruction="Verify that the source explicitly states the performer's announced final performance date for this specific show. "
                               "Accept official announcements, reputable theater press, or the show's official channels."
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Function                                                    #
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_broadway_info(),
        template_class=BroadwayExtraction,
        extraction_name="broadway_extraction"
    )

    # Build main critical node per rubric
    main_node = evaluator.add_parallel(
        id="Broadway_Performer_And_Show_Identification",
        desc="Answer provides the performer, theater, and show details meeting all stated constraints, including required dates and venue details.",
        parent=root,
        critical=True
    )

    # Subtrees
    await build_performer_subtree(evaluator, main_node, extracted)
    await build_show_subtree(evaluator, main_node, extracted)
    await build_theater_subtree(evaluator, main_node, extracted)
    await add_final_performance_leaf(evaluator, main_node, extracted)

    return evaluator.get_summary()