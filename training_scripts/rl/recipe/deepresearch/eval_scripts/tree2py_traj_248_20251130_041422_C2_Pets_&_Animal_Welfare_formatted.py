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
TASK_ID = "national_dog_show_2025"
TASK_DESCRIPTION = (
    "I missed the 2025 National Dog Show that aired on Thanksgiving Day and would like to learn about the Best in Show winner. "
    "Please provide the following information: (1) the name and breed of the dog that won Best in Show, "
    "(2) which AKC group category this breed competes in, "
    "(3) key physical characteristics of this breed according to AKC standards (such as coat color, build, and size), "
    "(4) the name of the handler who presented the winning dog, "
    "(5) the specific venue name and location (city and state) where the live competition was held, and "
    "(6) the name of the kennel club that organizes this annual event."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DogShowExtraction(BaseModel):
    # Winner basics
    winner_name: Optional[str] = None
    winner_breed: Optional[str] = None

    # AKC classification
    akc_group: Optional[str] = None

    # Breed characteristics (AKC standards)
    breed_characteristics: List[str] = Field(default_factory=list)

    # Handler info
    handler_name: Optional[str] = None

    # Event details
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    # Organizer
    organizing_body: Optional[str] = None

    # Sources: general and per-claim (only extract URLs explicitly present in the answer)
    sources: List[str] = Field(default_factory=list)
    sources_winner: List[str] = Field(default_factory=list)
    sources_group: List[str] = Field(default_factory=list)
    sources_breed: List[str] = Field(default_factory=list)
    sources_handler: List[str] = Field(default_factory=list)
    sources_venue: List[str] = Field(default_factory=list)
    sources_organizer: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_dog_show_info() -> str:
    return (
        "Extract structured information from the answer about the 2025 National Dog Show Best in Show winner and event details.\n"
        "Return a JSON object matching the template with the following fields:\n"
        "1) winner_name: The dog's name that won Best in Show.\n"
        "2) winner_breed: The breed of the Best in Show winner.\n"
        "3) akc_group: The AKC group category this breed competes in (e.g., Hound, Herding, Sporting, Non-Sporting, Working, Toy, Terrier).\n"
        "4) breed_characteristics: A list of key physical characteristics according to AKC standards (e.g., coat texture/color, overall build/proportions, height/weight range).\n"
        "5) handler_name: The full name of the handler who presented the winning dog.\n"
        "6) venue_name: The specific venue name where the live competition was held.\n"
        "7) city: The city where the venue is located.\n"
        "8) state: The U.S. state for the venue location.\n"
        "9) organizing_body: The kennel club that organizes the annual National Dog Show.\n\n"
        "Additionally, extract explicit source URLs mentioned in the answer:\n"
        "- sources: All URLs cited in the answer (including AKC pages, official show site, press releases, and news pages).\n"
        "- sources_winner: URLs specifically supporting the winner's name and breed.\n"
        "- sources_group: URLs specifically supporting the AKC group classification for the breed.\n"
        "- sources_breed: URLs specifically supporting AKC breed standard characteristics.\n"
        "- sources_handler: URLs specifically supporting the handler's identity for the winning dog.\n"
        "- sources_venue: URLs specifically supporting the venue name and location (city/state).\n"
        "- sources_organizer: URLs specifically supporting the organizing kennel club.\n\n"
        "Rules:\n"
        "- Extract only information explicitly present in the provided answer text; do not infer missing details.\n"
        "- For any missing field, return null (or empty list for breed_characteristics).\n"
        "- For URLs, extract the actual URLs shown in the answer (plain links or markdown links). If a URL lacks protocol, prepend http://.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def pick_sources(ex: DogShowExtraction, preferred_keys: List[str]) -> List[str]:
    """Pick best available sources from preferred per-claim lists; fallback to general sources."""
    combined: List[str] = []
    for key in preferred_keys:
        lst = getattr(ex, key, [])
        if lst:
            combined.extend(lst)
    if not combined and ex.sources:
        combined.extend(ex.sources)
    return _dedup_urls(combined)


def characteristics_text(ex: DogShowExtraction) -> str:
    if not ex.breed_characteristics:
        return ""
    # Join as a readable list
    return "; ".join([c.strip() for c in ex.breed_characteristics if isinstance(c, str) and c.strip()])


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extraction: DogShowExtraction) -> None:
    # Top-level critical node
    research_main = evaluator.add_parallel(
        id="2025_National_Dog_Show_Research",
        desc="Complete and accurate research about the 2025 National Dog Show Best in Show winner, handler, and event details",
        parent=evaluator.root,
        critical=True,
    )

    # Best in Show winner information (critical)
    winner_info = evaluator.add_parallel(
        id="Best_in_Show_Winner_Information",
        desc="Accurate identification and details of the Best in Show winner",
        parent=research_main,
        critical=True,
    )

    # Existence gates for winner info
    evaluator.add_custom_node(
        result=bool(extraction.winner_name and extraction.winner_name.strip() and extraction.winner_breed and extraction.winner_breed.strip()),
        id="Winner_Identity_Provided",
        desc="Winner identity provided (name and breed present)",
        parent=winner_info,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(extraction.akc_group and extraction.akc_group.strip()),
        id="Winner_Group_Provided",
        desc="AKC group info provided for the winning breed",
        parent=winner_info,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(extraction.breed_characteristics and len(extraction.breed_characteristics) > 0),
        id="Breed_Description_Provided",
        desc="Breed characteristics provided (AKC standards)",
        parent=winner_info,
        critical=True,
    )

    # Leaf: Winner Identity
    winner_identity_leaf = evaluator.add_leaf(
        id="Winner_Identity",
        desc="Correct dog name and breed of the Best in Show winner",
        parent=winner_info,
        critical=True,
    )
    identity_claim = (
        f"The dog that won Best in Show at the 2025 National Dog Show was {extraction.winner_name}, "
        f"a {extraction.winner_breed}."
    )
    identity_sources = pick_sources(extraction, ["sources_winner"])
    await evaluator.verify(
        claim=identity_claim,
        node=winner_identity_leaf,
        sources=identity_sources,
        additional_instruction=(
            "Confirm that the cited page(s) explicitly state the 2025 National Dog Show Best in Show winner, "
            "including both the dog's name (call name or registered name variants are acceptable) and the breed."
        ),
    )

    # Leaf: Winner Group
    winner_group_leaf = evaluator.add_leaf(
        id="Winner_Group",
        desc="Correct AKC group category that the winner competed in",
        parent=winner_info,
        critical=True,
    )
    group_claim = (
        f"The breed {extraction.winner_breed} competes in the {extraction.akc_group} group in the American Kennel Club (AKC) classification."
    )
    group_sources = pick_sources(extraction, ["sources_group", "sources_breed"])
    await evaluator.verify(
        claim=group_claim,
        node=winner_group_leaf,
        sources=group_sources,
        additional_instruction=(
            "Verify via an AKC breed page or authoritative source that the stated breed belongs to the given AKC group. "
            "Minor wording variations are acceptable (e.g., 'Non-Sporting group' vs 'Non-Sporting')."
        ),
    )

    # Leaf: Breed Description
    breed_desc_leaf = evaluator.add_leaf(
        id="Breed_Description",
        desc="Key physical characteristics of the winning breed as recognized by AKC standards",
        parent=winner_info,
        critical=True,
    )
    breed_chars_text = characteristics_text(extraction)
    breed_desc_claim = (
        f"According to AKC standards, key physical characteristics of the {extraction.winner_breed} include: {breed_chars_text}"
        if breed_chars_text
        else f"According to AKC standards, key physical characteristics of the {extraction.winner_breed} are as stated."
    )
    breed_sources = pick_sources(extraction, ["sources_breed", "sources_group"])
    await evaluator.verify(
        claim=breed_desc_claim,
        node=breed_desc_leaf,
        sources=breed_sources,
        additional_instruction=(
            "Check the AKC breed standard or AKC breed page to confirm the listed physical characteristics "
            "(e.g., coat, build, typical size ranges). Minor phrasing differences are fine, but the characteristics "
            "must be consistent with AKC descriptions."
        ),
    )

    # Handler information (critical)
    handler_info = evaluator.add_parallel(
        id="Handler_Information",
        desc="Correct identification of the Best in Show winner's handler",
        parent=research_main,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(extraction.handler_name and extraction.handler_name.strip()),
        id="Handler_Name_Provided",
        desc="Handler name provided",
        parent=handler_info,
        critical=True,
    )

    handler_leaf = evaluator.add_leaf(
        id="Handler_Name",
        desc="Full name of the handler who presented the Best in Show winner",
        parent=handler_info,
        critical=True,
    )
    handler_claim = f"The handler who presented the Best in Show winner was {extraction.handler_name}."
    handler_sources = pick_sources(extraction, ["sources_handler", "sources_winner"])
    await evaluator.verify(
        claim=handler_claim,
        node=handler_leaf,
        sources=handler_sources,
        additional_instruction=(
            "Confirm the named handler is explicitly identified as the presenter of the Best in Show winner "
            "for the 2025 National Dog Show."
        ),
    )

    # Event details (critical)
    event_details = evaluator.add_parallel(
        id="Event_Details",
        desc="Accurate details about the event organization and location",
        parent=research_main,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(extraction.venue_name and extraction.venue_name.strip() and extraction.city and extraction.city.strip() and extraction.state and extraction.state.strip()),
        id="Venue_and_Location_Provided",
        desc="Venue name and city/state provided",
        parent=event_details,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(extraction.organizing_body and extraction.organizing_body.strip()),
        id="Organizing_Body_Provided",
        desc="Organizing kennel club provided",
        parent=event_details,
        critical=True,
    )

    venue_leaf = evaluator.add_leaf(
        id="Venue_and_Location",
        desc="Correct venue name and city/state where the live event was held",
        parent=event_details,
        critical=True,
    )
    venue_claim = (
        f"The 2025 National Dog Show live competition was held at {extraction.venue_name} "
        f"in {extraction.city}, {extraction.state}."
    )
    venue_sources = pick_sources(extraction, ["sources_venue"])
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        sources=venue_sources,
        additional_instruction=(
            "Confirm the venue name and its city/state location for the live competition of the 2025 National Dog Show. "
            "Minor variations in naming (e.g., 'Greater Philadelphia Expo Center, Oaks, PA') are acceptable if equivalent."
        ),
    )

    organizer_leaf = evaluator.add_leaf(
        id="Organizing_Body",
        desc="Correct name of the kennel club that organized the show",
        parent=event_details,
        critical=True,
    )
    organizer_claim = f"The annual National Dog Show is organized by the {extraction.organizing_body}."
    organizer_sources = pick_sources(extraction, ["sources_organizer", "sources_venue"])
    await evaluator.verify(
        claim=organizer_claim,
        node=organizer_leaf,
        sources=organizer_sources,
        additional_instruction=(
            "Confirm that the stated kennel club is the organizing body of the National Dog Show (annual host). "
            "Accept minor naming variants such as inclusion of 'The' in the club name."
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
    # Initialize evaluator with a parallel root
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract all required fields and sources from the answer
    extraction: DogShowExtraction = await evaluator.extract(
        prompt=prompt_extract_dog_show_info(),
        template_class=DogShowExtraction,
        extraction_name="national_dog_show_2025_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extraction)

    # Return standard summary with verification tree and score
    return evaluator.get_summary()