import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "concert_venues_and_awards_2026"
TASK_DESCRIPTION = (
    "Identify at least two music concert venues in the United States that meet the following criteria: "
    "(1) Each venue must have a seating capacity between 2,500 and 10,000 people (inclusive); "
    "(2) Each venue must be suitable for hosting live music concerts or performances; "
    "(3) The venues must be located in at least two different U.S. states; "
    "(4) Provide the venue name, location (city and state), and exact capacity for each venue. "
    "Additionally, identify at least one musical work (song or score) that meets the following criteria: "
    "(1) The work must have won an award in the music categories at the 2026 Golden Globes ceremony (held January 11, 2026) - "
    "specifically either \"Best Original Song - Motion Picture\" or \"Best Original Score - Motion Picture\"; "
    "(2) The same work must have also won an award in the music categories at the 2026 Grammy Awards ceremony (held February 1, 2026) - "
    "specifically either \"Best Song Written for Visual Media\" or \"Best Score Soundtrack for Visual Media\"; "
    "(3) Provide the title of the work, the film or visual media it was created for, and the specific award categories won at both ceremonies. "
    "For all answers, provide reference URLs that verify each piece of information."
)

GOLDEN_GLOBES_ALLOWED_CATEGORIES = {
    "Best Original Song - Motion Picture",
    "Best Original Score - Motion Picture",
}
GRAMMYS_ALLOWED_CATEGORIES = {
    "Best Song Written for Visual Media",
    "Best Score Soundtrack for Visual Media",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueRefs(BaseModel):
    name_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)
    type_urls: List[str] = Field(default_factory=list)


class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # Keep as text to be robust (e.g., "5,000", "approx. 6,500")
    type_desc: Optional[str] = None  # e.g., "amphitheater", "music hall", "concert venue"
    refs: VenueRefs = Field(default_factory=VenueRefs)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


class WorkRefs(BaseModel):
    identity_urls: List[str] = Field(default_factory=list)
    gg_urls: List[str] = Field(default_factory=list)
    grammy_urls: List[str] = Field(default_factory=list)


class AwardWork(BaseModel):
    title: Optional[str] = None
    film_or_media: Optional[str] = None

    gg_category: Optional[str] = None
    gg_winner: Optional[bool] = None
    gg_year: Optional[str] = None  # e.g., "2026", "January 11, 2026"

    grammy_category: Optional[str] = None
    grammy_winner: Optional[bool] = None
    grammy_year: Optional[str] = None  # e.g., "2026", "February 1, 2026"

    refs: WorkRefs = Field(default_factory=WorkRefs)


class AwardsExtraction(BaseModel):
    works: List[AwardWork] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to the first 3 venues mentioned in the answer that are intended as music concert venues in the United States.

    For each venue, extract the following fields as they appear in the answer:
    - name: The official or commonly used name of the venue.
    - city: The city or town where the venue is located.
    - state: The U.S. state where the venue is located.
    - capacity: The stated seating capacity number for the venue (keep the exact text as presented, e.g., "5,000" or "approx. 6,500").
    - type_desc: A short descriptor indicating the venue type/usage (e.g., "amphitheater", "concert venue", "music hall", "arena").
    - refs.name_urls: All URLs specifically cited to verify the venue name.
    - refs.location_urls: All URLs specifically cited to verify the venue location (city & state).
    - refs.capacity_urls: All URLs specifically cited to verify the venue capacity.
    - refs.type_urls: All URLs specifically cited to verify the venue type/usage.

    Rules:
    - Extract only what is explicitly present in the answer.
    - Include only valid URLs; if an answer mentions a site without a URL, do not invent one.
    - If any field is missing for a venue, set it to null (or an empty list for URLs).
    - Return a JSON object with a 'venues' array of objects in the exact schema provided.
    """


def prompt_extract_award_work() -> str:
    return """
    Extract up to the first 2 musical works (songs or scores) mentioned in the answer that are intended to satisfy BOTH:
      • Won a music-category award at the 2026 Golden Globes (Jan 11, 2026): either "Best Original Song - Motion Picture" or "Best Original Score - Motion Picture".
      • Won a music-category award at the 2026 Grammy Awards (Feb 1, 2026): either "Best Song Written for Visual Media" or "Best Score Soundtrack for Visual Media".

    For each work, extract the following fields exactly as presented in the answer:
    - title: The title of the musical work.
    - film_or_media: The film or visual media that the work was created for.
    - gg_category: The Golden Globes category name as text.
    - gg_winner: Boolean indicating if the work is stated as the "winner" (not just a nominee) at the Golden Globes.
    - gg_year: The year/date mentioned for the Golden Globes recognition (e.g., "2026" or "January 11, 2026").
    - grammy_category: The Grammys category name as text.
    - grammy_winner: Boolean indicating if the work is stated as the "winner" (not just a nominee) at the Grammys.
    - grammy_year: The year/date mentioned for the Grammys recognition (e.g., "2026" or "February 1, 2026").
    - refs.identity_urls: URLs cited to verify the identity (title & film/media association).
    - refs.gg_urls: URLs cited to verify the Golden Globes category and winner status.
    - refs.grammy_urls: URLs cited to verify the Grammys category and winner status.

    Rules:
    - Extract only what is explicitly present in the answer.
    - Include only valid URLs; do not invent any URLs.
    - If any field is missing, set it to null (or an empty list for URLs).
    - Return a JSON object with a 'works' array of objects in the exact schema provided.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    """Return a clean list of URLs."""
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if isinstance(u, str) and u.strip():
            # Prepend http:// if missing protocol (Extractor may already do this)
            if not re.match(r"^https?://", u.strip(), flags=re.I):
                cleaned.append("http://" + u.strip())
            else:
                cleaned.append(u.strip())
    return cleaned


def parse_capacity_number(capacity_text: Optional[str]) -> Optional[int]:
    """Attempt to parse a capacity number from a text like '5,000' or 'approx. 6,500'."""
    if not capacity_text:
        return None
    # Find the first reasonable number (with optional commas)
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d{3,5})", capacity_text.replace("\u00A0", " "))
    if not m:
        return None
    num_str = m.group(1).replace(",", "")
    try:
        return int(num_str)
    except Exception:
        return None


def states_distinct(state1: Optional[str], state2: Optional[str]) -> bool:
    """Check if two states appear distinct (string-wise)."""
    if not state1 or not state2:
        return False
    s1 = state1.strip().lower()
    s2 = state2.strip().lower()
    return s1 != s2


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index: int,
) -> None:
    """
    Verify all aspects of a single venue according to the rubric.
    """
    idx = index + 1
    venue_node = evaluator.add_parallel(
        id=f"Venue_{idx}",
        desc=f"{'First' if idx == 1 else 'Second'} identified venue meets all requirements",
        parent=parent_node,
        critical=True  # Parent is critical, so children must be critical in this framework
    )

    # --- Name ---
    name_node = evaluator.add_parallel(
        id=f"Venue_{idx}_Name",
        desc="Provide the official name of the venue",
        parent=venue_node,
        critical=True
    )
    name_provided = evaluator.add_custom_node(
        result=bool(venue.name and venue.name.strip()),
        id=f"Venue_{idx}_Name_Provided",
        desc="Official venue name is stated",
        parent=name_node,
        critical=True
    )
    # We treat "Reference" as a factual verification using the cited URLs.
    name_ref_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Name_Reference",
        desc="URL reference provided for venue name verification",
        parent=name_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name of the venue is '{venue.name or ''}'.",
        node=name_ref_leaf,
        sources=sanitize_urls(venue.refs.name_urls),
        additional_instruction="Verify on the cited page(s) that the venue is known by this official name. Allow minor variations like sponsor prefixes or stylistic punctuation."
    )

    # --- Location ---
    loc_node = evaluator.add_parallel(
        id=f"Venue_{idx}_Location",
        desc="Venue is located in a city or town in the United States with specific city and state provided",
        parent=venue_node,
        critical=True
    )
    loc_details = evaluator.add_custom_node(
        result=bool(venue.city and venue.city.strip() and venue.state and venue.state.strip()),
        id=f"Venue_{idx}_Location_Details",
        desc="Specific city and state are provided for the venue",
        parent=loc_node,
        critical=True
    )
    loc_ref_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Location_Reference",
        desc="URL reference provided for venue location",
        parent=loc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{venue.name or ''}' is located in {venue.city or ''}, {venue.state or ''}, United States.",
        node=loc_ref_leaf,
        sources=sanitize_urls(venue.refs.location_urls),
        additional_instruction="Verify the city and state of the venue on the referenced page(s). Accept common abbreviations (e.g., 'CA' for California)."
    )

    # --- Capacity ---
    cap_node = evaluator.add_parallel(
        id=f"Venue_{idx}_Capacity",
        desc="Venue has a stated capacity between 2,500 and 10,000 people (inclusive)",
        parent=venue_node,
        critical=True
    )
    cap_verify_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Capacity_Verification",
        desc="Specific capacity number is provided and falls within the 2,500-10,000 range",
        parent=cap_node,
        critical=True
    )
    capacity_num = parse_capacity_number(venue.capacity)
    cap_claim = (
        f"The seating capacity of '{venue.name or ''}' is {venue.capacity or ''}, "
        f"and this capacity lies between 2,500 and 10,000 (inclusive)."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_verify_leaf,
        sources=sanitize_urls(venue.refs.capacity_urls),
        additional_instruction=(
            "Check the stated capacity for the venue on the referenced page(s). "
            "If the capacity number extracted is approximate or formatted with commas, treat it as the actual capacity value."
        )
    )
    cap_ref_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Capacity_Reference",
        desc="URL reference provided for venue capacity",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page explicitly states the seating capacity of '{venue.name or ''}' as {venue.capacity or ''}.",
        node=cap_ref_leaf,
        sources=sanitize_urls(venue.refs.capacity_urls),
        additional_instruction="Confirm that the page mentions the specific capacity figure (or an equivalent capacity statement)."
    )

    # --- Type / suitability ---
    type_node = evaluator.add_parallel(
        id=f"Venue_{idx}_Type",
        desc="Venue is suitable for hosting live music concerts or performances",
        parent=venue_node,
        critical=True
    )
    type_verify_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Type_Verification",
        desc="Venue is described as a concert venue, amphitheater, music hall, or similar music performance space",
        parent=type_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The venue '{venue.name or ''}' is suitable for hosting live music concerts or performances "
            f"(e.g., it is a concert venue, amphitheater, music hall, or similar)."
        ),
        node=type_verify_leaf,
        sources=sanitize_urls(venue.refs.type_urls),
        additional_instruction=(
            "Confirm on the referenced page(s) that the venue is used for live music concerts/performances, "
            "or is described as a concert venue, amphitheater, music hall, or similar."
        )
    )
    type_ref_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Type_Reference",
        desc="URL reference provided confirming venue type",
        parent=type_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The referenced page(s) confirm that '{venue.name or ''}' is a venue type suitable for live music performances."
        ),
        node=type_ref_leaf,
        sources=sanitize_urls(venue.refs.type_urls),
        additional_instruction="Check that the page conveys the venue's suitability for live music (e.g., mentions concerts, performances, gigs, etc.)."
    )


async def verify_geographic_diversity(
    evaluator: Evaluator,
    parent_node,
    venue1: VenueItem,
    venue2: VenueItem
) -> None:
    """
    Verify that the identified venues represent at least two different U.S. states.
    """
    geo_leaf = evaluator.add_leaf(
        id="Geographic_Diversity",
        desc="The identified venues represent at least two different U.S. states",
        parent=parent_node,
        critical=True
    )

    s1 = (venue1.state or "").strip()
    s2 = (venue2.state or "").strip()
    sources = sanitize_urls(venue1.refs.location_urls) + sanitize_urls(venue2.refs.location_urls)
    claim = (
        f"The two venues are located in different U.S. states: '{s1}' and '{s2}'. "
        f"Both venues are within the United States."
    )

    await evaluator.verify(
        claim=claim,
        node=geo_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the two venues are in distinct U.S. states based on the cited pages. "
            "Accept common abbreviations (e.g., 'CA' vs 'California'). If the states are identical or unclear, mark as Incorrect."
        )
    )


async def verify_award_work(
    evaluator: Evaluator,
    parent_node,
    work: AwardWork,
    index: int
) -> None:
    """
    Verify the musical work meets the award requirements at both Golden Globes 2026 and Grammys 2026.
    """
    idx = index + 1
    work_node = evaluator.add_parallel(
        id=f"Work_{idx}",
        desc="First identified work meets all award requirements" if idx == 1 else f"Work #{idx} meets all award requirements",
        parent=parent_node,
        critical=True
    )

    # Identity
    identity_node = evaluator.add_parallel(
        id=f"Work_{idx}_Identity",
        desc="Provide the title of the work and the film/media it was created for",
        parent=work_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(work.title and work.title.strip()),
        id=f"Work_{idx}_Title",
        desc="Title of the musical work is provided",
        parent=identity_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(work.film_or_media and work.film_or_media.strip()),
        id=f"Work_{idx}_Film",
        desc="Name of the film or visual media is provided",
        parent=identity_node,
        critical=True
    )
    identity_ref_leaf = evaluator.add_leaf(
        id=f"Work_{idx}_Identity_Reference",
        desc="URL reference provided for work identification",
        parent=identity_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The musical work titled '{work.title or ''}' was created for '{work.film_or_media or ''}'.",
        node=identity_ref_leaf,
        sources=sanitize_urls(work.refs.identity_urls),
        additional_instruction="Confirm on the referenced page(s) the association between the work title and the film/visual media."
    )

    # Golden Globes 2026
    gg_node = evaluator.add_parallel(
        id=f"Work_{idx}_Golden_Globe",
        desc="Work won a Golden Globe award at the 2026 ceremony (January 11, 2026) in a music category",
        parent=work_node,
        critical=True
    )

    gg_cat_node = evaluator.add_parallel(
        id=f"Work_{idx}_GG_Category",
        desc="Award was in the category Best Original Song - Motion Picture or Best Original Score - Motion Picture",
        parent=gg_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(work.gg_category and work.gg_category.strip() in GOLDEN_GLOBES_ALLOWED_CATEGORIES),
        id=f"Work_{idx}_GG_Category_Stated",
        desc="Specific Golden Globe category is stated",
        parent=gg_cat_node,
        critical=True
    )
    gg_cat_ref_leaf = evaluator.add_leaf(
        id=f"Work_{idx}_GG_Category_Reference",
        desc="URL reference provided for Golden Globe award category",
        parent=gg_cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The Golden Globes category for '{work.title or ''}' is '{work.gg_category or ''}', "
            f"and it was conferred at the 2026 ceremony."
        ),
        node=gg_cat_ref_leaf,
        sources=sanitize_urls(work.refs.gg_urls),
        additional_instruction=(
            "Confirm the specific Golden Globes category on the cited page(s). "
            "Valid categories: Best Original Song - Motion Picture OR Best Original Score - Motion Picture."
        )
    )

    gg_winner_node = evaluator.add_parallel(
        id=f"Work_{idx}_GG_Winner",
        desc="Work is confirmed as the winner (not just a nominee) in the specified category",
        parent=gg_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(work.gg_winner is True),
        id=f"Work_{idx}_GG_Winner_Status",
        desc="Winner status is explicitly stated",
        parent=gg_winner_node,
        critical=True
    )
    gg_winner_ref_leaf = evaluator.add_leaf(
        id=f"Work_{idx}_GG_Winner_Reference",
        desc="URL reference provided confirming Golden Globe win",
        parent=gg_winner_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The work '{work.title or ''}' is the winner (not just a nominee) of the Golden Globes category "
            f"'{work.gg_category or ''}' at the 2026 ceremony."
        ),
        node=gg_winner_ref_leaf,
        sources=sanitize_urls(work.refs.gg_urls),
        additional_instruction="Confirm on the referenced page(s) that the work is the WINNER in the specified Golden Globes category."
    )

    gg_year_leaf = evaluator.add_leaf(
        id=f"Work_{idx}_GG_Year",
        desc="Award was presented at the 2026 Golden Globes ceremony",
        parent=gg_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Golden Globe award for this work was presented at the 2026 Golden Globes ceremony held January 11, 2026.",
        node=gg_year_leaf,
        sources=sanitize_urls(work.refs.gg_urls),
        additional_instruction="Verify the year/cermony context (2026 Golden Globes)."
    )

    # Grammys 2026
    grammy_node = evaluator.add_parallel(
        id=f"Work_{idx}_Grammy",
        desc="Work won a Grammy award at the 2026 ceremony (February 1, 2026) in a music category",
        parent=work_node,
        critical=True
    )

    grammy_cat_node = evaluator.add_parallel(
        id=f"Work_{idx}_Grammy_Category",
        desc="Award was in the category Best Song Written for Visual Media or Best Score Soundtrack for Visual Media",
        parent=grammy_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(work.grammy_category and work.grammy_category.strip() in GRAMMYS_ALLOWED_CATEGORIES),
        id=f"Work_{idx}_Grammy_Category_Stated",
        desc="Specific Grammy category is stated",
        parent=grammy_cat_node,
        critical=True
    )
    grammy_cat_ref_leaf = evaluator.add_leaf(
        id=f"Work_{idx}_Grammy_Category_Reference",
        desc="URL reference provided for Grammy award category",
        parent=grammy_cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The Grammy category for '{work.title or ''}' is '{work.grammy_category or ''}', "
            f"and it was conferred at the 2026 ceremony."
        ),
        node=grammy_cat_ref_leaf,
        sources=sanitize_urls(work.refs.grammy_urls),
        additional_instruction=(
            "Confirm the specific Grammy category on the cited page(s). "
            "Valid categories: Best Song Written for Visual Media OR Best Score Soundtrack for Visual Media."
        )
    )

    grammy_winner_node = evaluator.add_parallel(
        id=f"Work_{idx}_Grammy_Winner",
        desc="Work is confirmed as the winner (not just a nominee) in the specified category",
        parent=grammy_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(work.grammy_winner is True),
        id=f"Work_{idx}_Grammy_Winner_Status",
        desc="Winner status is explicitly stated",
        parent=grammy_winner_node,
        critical=True
    )
    grammy_winner_ref_leaf = evaluator.add_leaf(
        id=f"Work_{idx}_Grammy_Winner_Reference",
        desc="URL reference provided confirming Grammy win",
        parent=grammy_winner_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The work '{work.title or ''}' is the winner (not just a nominee) of the Grammy category "
            f"'{work.grammy_category or ''}' at the 2026 ceremony."
        ),
        node=grammy_winner_ref_leaf,
        sources=sanitize_urls(work.refs.grammy_urls),
        additional_instruction="Confirm on the referenced page(s) that the work is the WINNER in the specified Grammy category."
    )

    grammy_year_leaf = evaluator.add_leaf(
        id=f"Work_{idx}_Grammy_Year",
        desc="Award was presented at the 2026 Grammy Awards ceremony",
        parent=grammy_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Grammy award for this work was presented at the 2026 Grammy Awards ceremony held February 1, 2026.",
        node=grammy_year_leaf,
        sources=sanitize_urls(work.refs.grammy_urls),
        additional_instruction="Verify the year/cermony context (2026 Grammys)."
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
    Evaluate the provided answer against the rubric for venues and award-winning work (2026).
    """
    # Initialize evaluator (framework root is non-critical by design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall tasks can be evaluated independently
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

    # Add Task_Completion node (critical)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc=("Identify at least two music concert venues in the United States with capacity between 2,500 and 10,000, "
              "located in at least two different states, and identify at least one musical work that won awards in both "
              "the 2026 Golden Globes music categories and the 2026 Grammy music categories."),
        parent=root,
        critical=True
    )

    # Create category nodes (must be critical since parent is critical)
    venue_main = evaluator.add_parallel(
        id="Venue_Identification",
        desc="Identify at least two music concert venues meeting all specified criteria",
        parent=task_node,
        critical=True
    )
    award_main = evaluator.add_parallel(
        id="Award_Winning_Work_Identification",
        desc="Identify at least one musical work that won awards in both 2026 Golden Globes and 2026 Grammy Awards",
        parent=task_node,
        critical=True
    )

    # Extract information
    venues_task = evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )
    awards_task = evaluator.extract(
        prompt=prompt_extract_award_work(),
        template_class=AwardsExtraction,
        extraction_name="awards_extraction"
    )

    venues_extracted, awards_extracted = await asyncio.gather(venues_task, awards_task)

    # Process and limit to required counts
    venues: List[VenueItem] = list(venues_extracted.venues[:2])
    while len(venues) < 2:
        venues.append(VenueItem())

    works: List[AwardWork] = list(awards_extracted.works[:1])
    while len(works) < 1:
        works.append(AwardWork())

    # Verify venues
    for i, v in enumerate(venues[:2]):
        await verify_venue(evaluator, venue_main, v, i)

    # Geographic diversity check (depends logically on location info of both venues)
    await verify_geographic_diversity(evaluator, venue_main, venues[0], venues[1])

    # Verify award work(s) — at least one required
    for i, w in enumerate(works[:1]):
        await verify_award_work(evaluator, award_main, w, i)

    # Return structured summary
    return evaluator.get_summary()