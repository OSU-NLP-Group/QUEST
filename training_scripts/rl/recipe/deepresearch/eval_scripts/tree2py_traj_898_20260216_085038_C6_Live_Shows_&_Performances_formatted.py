import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_late_night_family"
TASK_DESCRIPTION = (
    "A family of four is planning a trip to New York City in April 2026 and would like to attend live tapings of late-night talk shows. "
    "The family consists of two adults (ages 42 and 39) and two teenagers (ages 17 and 16).\n\n"
    "Identify TWO different late-night talk shows that regularly tape with live audiences in New York City and that ALL four family members are eligible to attend together based on the shows' age requirements.\n\n"
    "For each of the two shows you identify, provide the following information:\n\n"
    "1. Show name: The full, official name of the show\n"
    "2. Age requirement: The minimum age requirement for audience members, including any special conditions (e.g., requirements for parental accompaniment)\n"
    "3. Eligibility confirmation: Explicit confirmation that all four family members (ages 42, 39, 17, and 16) meet the age requirements to attend\n"
    "4. Venue name: The name of the theater or studio where the show tapes\n"
    "5. Full address: The complete street address of the venue\n"
    "6. Ticketing platform: The website or platform used to request/reserve free audience tickets\n"
    "7. Standby ticket availability: Whether standby tickets are available for the show, and if so, where and when they are distributed\n\n"
    "For each piece of information, include a reference URL from an official source (such as the show's official website, the network's website, or the ticketing platform) that verifies the information you provide."
)

# Acceptable NYC late-night talk shows for this task
ESTABLISHED_NYC_SHOWS = [
    "The Tonight Show Starring Jimmy Fallon",
    "Late Night with Seth Meyers",
    "The Late Show with Stephen Colbert",
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ShowInfo(BaseModel):
    show_name: Optional[str] = None

    # Identity/reference URL for the show (official page or ticketing page that confirms show identity)
    show_identity_url: Optional[str] = None

    # Age requirement text and specific minimum age value (as strings for flexibility)
    age_requirement_text: Optional[str] = None
    min_age_value: Optional[str] = None
    age_requirement_url: Optional[str] = None

    # Eligibility confirmation text explicitly stated in the answer
    eligibility_confirmation_text: Optional[str] = None

    # Location details and reference URL
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    location_reference_url: Optional[str] = None

    # Ticketing platform and reference URL
    ticketing_platform_name: Optional[str] = None
    ticketing_platform_url: Optional[str] = None

    # Standby ticket policy and reference URL
    standby_availability_text: Optional[str] = None
    standby_reference_url: Optional[str] = None


class ShowsExtraction(BaseModel):
    shows: List[ShowInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shows() -> str:
    return """
    You must extract up to TWO late-night talk shows mentioned in the answer that tape with live audiences in New York City.
    For each identified show, extract the following fields EXACTLY as stated in the answer:

    1) show_name: Full official name of the show.
    2) show_identity_url: A reference URL (official show/network page or official ticketing page) that identifies the show and is relevant to attending/taping.
    3) age_requirement_text: The exact phrasing of the minimum age requirement (including any conditions like parental accompaniment).
    4) min_age_value: The specific minimum age value stated (e.g., "16+", "18 and older"). Keep it as a string.
    5) age_requirement_url: A reference URL from an official source that states the age requirement.
    6) eligibility_confirmation_text: The explicit statement in the answer confirming that the family members aged 42, 39, 17, and 16 can attend together.
    7) venue_name: Name of the theater or studio where the show tapes.
    8) venue_address: The full street address of the venue.
    9) location_reference_url: A reference URL from an official source that lists the venue and/or address.
    10) ticketing_platform_name: The website/platform used to request free audience tickets (e.g., 1iota).
    11) ticketing_platform_url: The URL to the official ticketing platform page for this show.
    12) standby_availability_text: The stated standby ticket availability policy (e.g., "Standby tickets are available at XYZ at 9am day-of", or "No standby tickets").
    13) standby_reference_url: A reference URL from an official source for the standby policy.

    RULES:
    - Extract only URLs explicitly present in the answer. Do not invent or infer URLs.
    - If any field is missing in the answer, set it to null.
    - Return an array 'shows' with up to two show objects. If more than two are mentioned, include only the first two.
    - If fewer than two are mentioned, include what is available (one or zero), and set missing fields to null.

    Output JSON must match the provided template exactly.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_show(
    evaluator: Evaluator,
    parent_node,
    show: ShowInfo,
    idx: int,
) -> None:
    """
    Build verification sub-tree for one show (idx is 1-based).
    """
    # Top-level node for this show's selection (parallel aggregation, non-critical to allow partial credit per show)
    show_node = evaluator.add_parallel(
        id=f"show_selection_{idx}",
        desc=f"{'First' if idx == 1 else 'Second'} late-night show selection and complete information provided",
        parent=parent_node,
        critical=False,
    )

    # 1) Show identity block (sequential)
    identity_node = evaluator.add_sequential(
        id=f"show_identity_{idx}",
        desc=f"{'First' if idx == 1 else 'Second'} show correctly identified as an NYC late-night talk show taping",
        parent=show_node,
        critical=True,
    )

    # 1.a) Valid show name matches established NYC late-night talk shows
    valid_name_leaf = evaluator.add_leaf(
        id=f"valid_show_name_{idx}",
        desc=(
            f"Show name matches one of the established NYC late-night talk shows "
            f"({', '.join(ESTABLISHED_NYC_SHOWS)})"
        ),
        parent=identity_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The provided show name '{_safe(show.show_name)}' is one of the following established NYC late-night talk shows: "
            f"{', '.join(ESTABLISHED_NYC_SHOWS)}."
        ),
        node=valid_name_leaf,
        additional_instruction=(
            "Consider reasonable formatting variations or minor name variants acceptable if they clearly refer to the same show."
        ),
    )

    # 1.b) Reference URL provided for identification
    identity_url_exists = evaluator.add_custom_node(
        result=bool(show.show_identity_url and show.show_identity_url.strip()),
        id=f"show_reference_url_{idx}",
        desc=f"Reference URL provided for {'first' if idx == 1 else 'second'} show identification",
        parent=identity_node,
        critical=True,
    )

    # 2) Age eligibility block (sequential)
    age_node = evaluator.add_sequential(
        id=f"age_eligibility_{idx}",
        desc=f"Age requirement analysis for {'first' if idx == 1 else 'second'} show demonstrates all family members can attend",
        parent=show_node,
        critical=True,
    )

    # 2.a) Age requirement stated (parallel)
    age_stated_node = evaluator.add_parallel(
        id=f"age_requirement_stated_{idx}",
        desc=f"Minimum age requirement for {'first' if idx == 1 else 'second'} show correctly stated",
        parent=age_node,
        critical=True,
    )

    # 2.a.i) Specific minimum age value matches official requirement
    specific_age_leaf = evaluator.add_leaf(
        id=f"specific_age_value_{idx}",
        desc=f"Specific minimum age value provided matches official requirement for the identified show",
        parent=age_stated_node,
        critical=True,
    )
    age_claim = (
        f"For the show '{_safe(show.show_name)}', the minimum audience age requirement is '{_safe(show.min_age_value)}'. "
        f"Stated policy: '{_safe(show.age_requirement_text)}'."
    )
    await evaluator.verify(
        claim=age_claim,
        node=specific_age_leaf,
        sources=show.age_requirement_url,
        additional_instruction=(
            "Verify on the provided official source that the minimum audience age matches the stated value/text. "
            "Allow reasonable phrasing variations such as '16 years or older' vs '16+'. "
            "If minors must be accompanied by a parent/legal guardian, that should be reflected in the policy text."
        ),
    )

    # 2.a.ii) Age requirement reference URL provided (existence check)
    age_url_exists = evaluator.add_custom_node(
        result=bool(show.age_requirement_url and show.age_requirement_url.strip()),
        id=f"age_requirement_url_{idx}",
        desc=f"Reference URL provided for age requirement of {'first' if idx == 1 else 'second'} show",
        parent=age_stated_node,
        critical=True,
    )

    # 2.b) Family eligibility verified
    family_elig_leaf = evaluator.add_leaf(
        id=f"family_eligibility_verified_{idx}",
        desc=(
            "Explicit verification that all four family members (ages 42, 39, 17, 16) meet the age requirement "
            f"for {'first' if idx == 1 else 'second'} show"
        ),
        parent=age_node,
        critical=True,
    )
    family_claim = (
        f"Based on the age requirement '{_safe(show.min_age_value)}' for '{_safe(show.show_name)}' "
        f"('{_safe(show.age_requirement_text)}'), the family (ages 42, 39, 17, and 16) is eligible to attend together. "
        f"The two minors (17 and 16) will be accompanied by their parents (42 and 39), satisfying any parental "
        f"accompaniment conditions."
    )
    await evaluator.verify(
        claim=family_claim,
        node=family_elig_leaf,
        sources=show.age_requirement_url,
        additional_instruction=(
            "Judge logically using the stated minimum age policy and any parental accompaniment rules from the official source. "
            "If minimum age is 16+, then ages 17 and 16 should be acceptable (with parental accompaniment if required). "
            "If minimum age is 18+, the 16-year-old would not meet the requirement."
        ),
    )

    # 3) Location information (parallel)
    location_node = evaluator.add_parallel(
        id=f"location_information_{idx}",
        desc=f"Complete and accurate location details for {'first' if idx == 1 else 'second'} show",
        parent=show_node,
        critical=True,
    )

    # 3.a) Venue name correctly identified
    venue_leaf = evaluator.add_leaf(
        id=f"venue_name_{idx}",
        desc=f"Venue name correctly identified for {'first' if idx == 1 else 'second'} show",
        parent=location_node,
        critical=True,
    )
    venue_claim = (
        f"The show '{_safe(show.show_name)}' tapes at '{_safe(show.venue_name)}' in New York City."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        sources=show.location_reference_url,
        additional_instruction=(
            "Confirm on the official venue/show page that the stated venue name is correct. "
            "Allow reasonable variants (e.g., 'Studio 6B at 30 Rockefeller Plaza' vs 'NBC Studio 6B')."
        ),
    )

    # 3.b) Full address provided
    address_leaf = evaluator.add_leaf(
        id=f"full_address_{idx}",
        desc=f"Complete street address provided for {'first' if idx == 1 else 'second'} show venue",
        parent=location_node,
        critical=True,
    )
    address_claim = (
        f"The full address of the venue for '{_safe(show.show_name)}' is '{_safe(show.venue_address)}'."
    )
    await evaluator.verify(
        claim=address_claim,
        node=address_leaf,
        sources=show.location_reference_url,
        additional_instruction=(
            "Verify that the provided address string matches the official venue/show page."
        ),
    )

    # 3.c) Location reference URL provided (existence check)
    location_url_exists = evaluator.add_custom_node(
        result=bool(show.location_reference_url and show.location_reference_url.strip()),
        id=f"location_reference_url_{idx}",
        desc=f"Reference URL provided for location information of {'first' if idx == 1 else 'second'} show",
        parent=location_node,
        critical=True,
    )

    # 4) Ticketing platform (parallel)
    ticketing_node = evaluator.add_parallel(
        id=f"ticketing_platform_{idx}",
        desc=f"Correct ticketing platform and process information for {'first' if idx == 1 else 'second'} show",
        parent=show_node,
        critical=True,
    )

    # 4.a) Platform identified correctly
    platform_leaf = evaluator.add_leaf(
        id=f"platform_identified_{idx}",
        desc=f"Ticketing website/platform correctly identified for {'first' if idx == 1 else 'second'} show",
        parent=ticketing_node,
        critical=True,
    )
    platform_claim = (
        f"Audience tickets for '{_safe(show.show_name)}' are requested or reserved via '{_safe(show.ticketing_platform_name)}'."
    )
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=show.ticketing_platform_url,
        additional_instruction=(
            "Confirm on the official ticketing platform page (e.g., 1iota or the network's ticketing site) that this show uses this platform."
        ),
    )

    # 4.b) Ticketing reference URL provided (existence check)
    ticketing_url_exists = evaluator.add_custom_node(
        result=bool(show.ticketing_platform_url and show.ticketing_platform_url.strip()),
        id=f"ticketing_reference_url_{idx}",
        desc=f"Reference URL provided for ticketing information of {'first' if idx == 1 else 'second'} show",
        parent=ticketing_node,
        critical=True,
    )

    # 5) Standby information (parallel; per rubric it's non-critical, keep non-critical here)
    standby_node = evaluator.add_parallel(
        id=f"standby_information_{idx}",
        desc=f"Accurate information about standby ticket availability for {'first' if idx == 1 else 'second'} show",
        parent=show_node,
        critical=False,
    )

    # 5.a) Standby availability status stated
    standby_leaf = evaluator.add_leaf(
        id=f"standby_availability_{idx}",
        desc=f"Standby ticket availability status correctly stated for {'first' if idx == 1 else 'second'} show",
        parent=standby_node,
        critical=False,
    )
    standby_claim = (
        f"Standby ticket policy for '{_safe(show.show_name)}': '{_safe(show.standby_availability_text)}'."
    )
    await evaluator.verify(
        claim=standby_claim,
        node=standby_leaf,
        sources=show.standby_reference_url,
        additional_instruction=(
            "Verify on the official source whether standby tickets exist for this show and, if so, where/when they are distributed. "
            "If the policy states no standby, confirm that explicitly."
        ),
    )

    # 5.b) Standby reference URL provided (existence check)
    standby_url_exists = evaluator.add_custom_node(
        result=bool(show.standby_reference_url and show.standby_reference_url.strip()),
        id=f"standby_reference_url_{idx}",
        desc=f"Reference URL provided for standby information of {'first' if idx == 1 else 'second'} show",
        parent=standby_node,
        critical=False,
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
    Evaluate an answer for the NYC late-night show family eligibility task.
    """
    # Initialize evaluator. Note: root node is non-critical to allow partial credit overall.
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

    # Extract up to two shows from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_shows(),
        template_class=ShowsExtraction,
        extraction_name="shows_extraction",
    )

    # Normalize to exactly two shows (pad with empty if fewer)
    shows: List[ShowInfo] = list(extraction.shows[:2])
    while len(shows) < 2:
        shows.append(ShowInfo())

    # Build verification subtrees for each show
    await verify_single_show(evaluator, root, shows[0], idx=1)
    await verify_single_show(evaluator, root, shows[1], idx=2)

    # Final check: the two identified shows are different
    different_leaf = evaluator.add_leaf(
        id="shows_different",
        desc="The two identified shows are different from each other",
        parent=root,
        critical=True,
    )
    show1_name = _safe(shows[0].show_name)
    show2_name = _safe(shows[1].show_name)
    await evaluator.verify(
        claim=(
            f"The two identified shows '{show1_name}' and '{show2_name}' are different (i.e., not the same show)."
        ),
        node=different_leaf,
        additional_instruction=(
            "Allow reasonable formatting/name variants; judge whether they refer to distinct shows. "
            "If either name is missing or both names are identical, this should be considered incorrect."
        ),
    )

    # Add custom info helpful for debugging
    evaluator.add_custom_info(
        info={
            "established_nyc_shows": ESTABLISHED_NYC_SHOWS,
            "family_ages": [42, 39, 17, 16],
        },
        info_type="context",
        info_name="task_context",
    )

    return evaluator.get_summary()