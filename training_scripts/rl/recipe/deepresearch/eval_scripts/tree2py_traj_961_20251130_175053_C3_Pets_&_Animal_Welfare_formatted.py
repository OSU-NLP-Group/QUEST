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
TASK_ID = "national_dog_show_2025_bis"
TASK_DESCRIPTION = (
    "A Belgian Sheepdog won Best in Show at the 2025 National Dog Show held on Thanksgiving Day (November 27, 2025) "
    "in Pennsylvania. Identify this winning dog's call name and confirm the breed. Based on the dog's registered name, "
    "identify the kennel name that appears in it, and provide the city and state where this kennel is located. Finally, "
    "provide two historical milestones for the Belgian Sheepdog breed: the year when Belgian Sheepdogs were first "
    "registered in the AKC Stud Book, and the year when the Belgian Sheepdog Club of America was formed."
)

EVENT_CONTEXT = {
    "event_name": "National Dog Show",
    "event_year": 2025,
    "event_date": "November 27, 2025",
    "event_location_state": "Pennsylvania",
    "event_day": "Thanksgiving Day"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WinnerCall(BaseModel):
    call_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class WinnerBreed(BaseModel):
    breed: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RegisteredNameInfo(BaseModel):
    registered_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class KennelInfo(BaseModel):
    kennel_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Milestones(BaseModel):
    akc_stud_book_first_registration_year: Optional[str] = None
    akc_sources: List[str] = Field(default_factory=list)
    bsca_formation_year: Optional[str] = None
    bsca_sources: List[str] = Field(default_factory=list)


class DogShowExtraction(BaseModel):
    winner_call: Optional[WinnerCall] = None
    winner_breed: Optional[WinnerBreed] = None
    registered_name: Optional[RegisteredNameInfo] = None
    kennel_info: Optional[KennelInfo] = None
    milestones: Optional[Milestones] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return (
        "Extract structured information about the 2025 National Dog Show Best in Show (BIS) winner (a Belgian Sheepdog) "
        "and breed milestones, exactly as presented in the answer.\n"
        "Required fields:\n"
        "1) winner_call:\n"
        "   - call_name: The dog's call name (everyday name), not the registered name.\n"
        "   - sources: Array of URL strings explicitly cited in the answer that support the BIS result or the dog's call name.\n"
        "2) winner_breed:\n"
        "   - breed: The breed for the BIS winner as stated in the answer.\n"
        "   - sources: Array of URL strings supporting the BIS winner's breed at the 2025 National Dog Show.\n"
        "3) registered_name:\n"
        "   - registered_name: The dog's registered name as stated in the answer.\n"
        "   - sources: Array of URL strings supporting the dog's registered name.\n"
        "4) kennel_info:\n"
        "   - kennel_name: The kennel name that appears within the dog's registered name (e.g., a prefix/suffix/group of words indicating the kennel).\n"
        "   - city: The city where this kennel is located.\n"
        "   - state: The state where this kennel is located (use the two-letter abbreviation if that is how it appears).\n"
        "   - sources: Array of URL strings supporting the kennel identity and location.\n"
        "5) milestones:\n"
        "   - akc_stud_book_first_registration_year: The year Belgian Sheepdogs were first registered in the AKC Stud Book.\n"
        "   - akc_sources: Array of URL strings supporting the AKC Stud Book first registration year.\n"
        "   - bsca_formation_year: The year the Belgian Sheepdog Club of America (BSCA) was formed.\n"
        "   - bsca_sources: Array of URL strings supporting the BSCA formation year.\n\n"
        "URL extraction rules:\n"
        "- Extract only URLs explicitly present in the answer (plain links or markdown links). Do not invent URLs.\n"
        "- Include full URLs. If a URL is missing the scheme, prepend http://.\n"
        "- If a required field is missing from the answer, set it to null; for sources, return an empty array when none are given.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(*source_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in source_lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                result.append(url)
    return result


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_winner_call_subtree(evaluator: Evaluator, parent_node, data: DogShowExtraction) -> None:
    node = evaluator.add_sequential(
        id="WinnerCallName",
        desc="Provide the Best in Show winner's call name",
        parent=parent_node,
        critical=True
    )

    call = data.winner_call.call_name if data.winner_call else None
    call_sources = data.winner_call.sources if data.winner_call else []

    exists_node = evaluator.add_custom_node(
        result=bool(call and call.strip()) and bool(call_sources),
        id="WinnerCallName_exists",
        desc="Winner call name is provided with sources",
        parent=node,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="WinnerCallName_supported",
        desc="The call name for the 2025 National Dog Show BIS winner is correctly stated",
        parent=node,
        critical=True
    )

    claim = (
        f"The Best in Show winner at the 2025 National Dog Show has the call name '{call}'."
        if call else "The Best in Show winner at the 2025 National Dog Show has a specific call name."
    )
    add_ins = (
        "Confirm the page refers to the 2025 National Dog Show BIS winner (Thanksgiving Day, November 27, 2025, in Pennsylvania), "
        "and that the stated call name matches the dog's name reported by official results or credible news releases."
    )
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=call_sources,
        additional_instruction=add_ins
    )


async def build_winner_breed_subtree(evaluator: Evaluator, parent_node, data: DogShowExtraction) -> None:
    node = evaluator.add_sequential(
        id="WinnerBreed",
        desc="Confirm the Best in Show winner's breed is Belgian Sheepdog",
        parent=parent_node,
        critical=True
    )

    breed = data.winner_breed.breed if data.winner_breed else None
    breed_sources = data.winner_breed.sources if data.winner_breed else []

    exists_node = evaluator.add_custom_node(
        result=bool(breed and breed.strip()) and bool(breed_sources),
        id="WinnerBreed_exists",
        desc="Winner breed is provided with sources",
        parent=node,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="WinnerBreed_belgian",
        desc="The BIS winner's breed is Belgian Sheepdog",
        parent=node,
        critical=True
    )

    claim = "The Best in Show winner at the 2025 National Dog Show is a Belgian Sheepdog."
    add_ins = (
        "Ensure the source refers to the 2025 National Dog Show BIS winner and identifies the breed as Belgian Sheepdog. "
        "Minor naming variations like 'Belgian Sheepdog (Groenendael)' should be treated as equivalent to Belgian Sheepdog."
    )
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=breed_sources,
        additional_instruction=add_ins
    )


async def build_kennel_name_in_registered_subtree(evaluator: Evaluator, parent_node, data: DogShowExtraction) -> None:
    node = evaluator.add_sequential(
        id="KennelNameInRegisteredName",
        desc="From the dog's registered name, identify the kennel name that appears in it",
        parent=parent_node,
        critical=True
    )

    reg_name = data.registered_name.registered_name if data.registered_name else None
    reg_sources = data.registered_name.sources if data.registered_name else []
    kennel_name = data.kennel_info.kennel_name if data.kennel_info else None
    kennel_sources = data.kennel_info.sources if data.kennel_info else []
    combined_sources = merge_sources(reg_sources, kennel_sources)

    exists_node = evaluator.add_custom_node(
        result=bool(reg_name and reg_name.strip()) and bool(kennel_name and kennel_name.strip()) and bool(combined_sources),
        id="KennelNameInRegisteredName_exists",
        desc="Registered name and kennel name are provided with sources",
        parent=node,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="KennelNameInRegisteredName_supported",
        desc="The identified kennel name appears within the dog's registered name",
        parent=node,
        critical=True
    )

    claim = (
        f"The dog's registered name is '{reg_name}', and it includes the kennel name '{kennel_name}'."
        if reg_name and kennel_name else
        "The dog's registered name includes a specific kennel name."
    )
    add_ins = (
        "Verify that the cited registered name string contains the stated kennel name or kennel identifier. "
        "Allow case-insensitive matching and common kennel naming forms (e.g., 'of', 'van', 'vom', 'de'). "
        "Use the provided sources that explicitly show the registered name."
    )
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=combined_sources,
        additional_instruction=add_ins
    )


async def build_kennel_location_subtree(evaluator: Evaluator, parent_node, data: DogShowExtraction) -> None:
    node = evaluator.add_sequential(
        id="KennelLocationCityState",
        desc="Provide the city and state where the identified kennel is located",
        parent=parent_node,
        critical=True
    )

    kennel_name = data.kennel_info.kennel_name if data.kennel_info else None
    city = data.kennel_info.city if data.kennel_info else None
    state = data.kennel_info.state if data.kennel_info else None
    sources = data.kennel_info.sources if data.kennel_info else []

    exists_node = evaluator.add_custom_node(
        result=bool(kennel_name and kennel_name.strip()) and bool(city and city.strip()) and bool(state and state.strip()) and bool(sources),
        id="KennelLocationCityState_exists",
        desc="Kennel location (city/state) is provided with sources",
        parent=node,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="KennelLocationCityState_supported",
        desc="The kennel's location (city and state) is correctly stated",
        parent=node,
        critical=True
    )

    claim = (
        f"The kennel '{kennel_name}' is located in {city}, {state}."
        if kennel_name and city and state else
        "The kennel is located in a specific city and state."
    )
    add_ins = (
        "Verify the kennel's location details (city and state) using the provided sources (official kennel website, AKC breeder listing, or credible directories). "
        "Treat standard US state abbreviations as equivalent to full names."
    )
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=sources,
        additional_instruction=add_ins
    )


async def build_milestones_subtree(evaluator: Evaluator, parent_node, data: DogShowExtraction) -> None:
    milestone_root = evaluator.add_parallel(
        id="BreedMilestones",
        desc="Provide the two requested historical milestones for the Belgian Sheepdog breed",
        parent=parent_node,
        critical=True
    )

    # AKC Stud Book first registration year
    akc_node = evaluator.add_sequential(
        id="AKCStudBookFirstRegistrationYear",
        desc="Provide the year Belgian Sheepdogs were first registered in the AKC Stud Book",
        parent=milestone_root,
        critical=True
    )
    akc_year = data.milestones.akc_stud_book_first_registration_year if data.milestones else None
    akc_sources = data.milestones.akc_sources if data.milestones else []
    evaluator.add_custom_node(
        result=bool(akc_year and akc_year.strip()) and bool(akc_sources),
        id="AKCStudBookFirstRegistrationYear_exists",
        desc="AKC Stud Book first registration year is provided with sources",
        parent=akc_node,
        critical=True
    )
    akc_verify = evaluator.add_leaf(
        id="AKCStudBookFirstRegistrationYear_supported",
        desc="The AKC Stud Book first registration year for Belgian Sheepdogs is correctly stated",
        parent=akc_node,
        critical=True
    )
    akc_claim = (
        f"Belgian Sheepdogs were first registered in the AKC Stud Book in {akc_year}."
        if akc_year else
        "Belgian Sheepdogs were first registered in the AKC Stud Book in a specific year."
    )
    akc_add_ins = (
        "Confirm the 'first registered' year specifically (not recognition or later events). "
        "Prefer official AKC breed history or stud book references; if a credible secondary source is used, ensure it clearly states the first registration year."
    )
    await evaluator.verify(
        claim=akc_claim,
        node=akc_verify,
        sources=akc_sources,
        additional_instruction=akc_add_ins
    )

    # BSCA formation year
    bsca_node = evaluator.add_sequential(
        id="BSCAFormationYear",
        desc="Provide the year the Belgian Sheepdog Club of America was formed",
        parent=milestone_root,
        critical=True
    )
    bsca_year = data.milestones.bsca_formation_year if data.milestones else None
    bsca_sources = data.milestones.bsca_sources if data.milestones else []
    evaluator.add_custom_node(
        result=bool(bsca_year and bsca_year.strip()) and bool(bsca_sources),
        id="BSCAFormationYear_exists",
        desc="BSCA formation year is provided with sources",
        parent=bsca_node,
        critical=True
    )
    bsca_verify = evaluator.add_leaf(
        id="BSCAFormationYear_supported",
        desc="The formation year of the Belgian Sheepdog Club of America is correctly stated",
        parent=bsca_node,
        critical=True
    )
    bsca_claim = (
        f"The Belgian Sheepdog Club of America was formed in {bsca_year}."
        if bsca_year else
        "The Belgian Sheepdog Club of America was formed in a specific year."
    )
    bsca_add_ins = (
        "Prefer official BSCA or AKC historical sources. Verify that the year refers to the club's formation, not incorporation or later milestones."
    )
    await evaluator.verify(
        claim=bsca_claim,
        node=bsca_verify,
        sources=bsca_sources,
        additional_instruction=bsca_add_ins
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
    Evaluate the answer for the 2025 National Dog Show Belgian Sheepdog BIS and breed milestones.
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

    # Extract structured information
    extraction = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=DogShowExtraction,
        extraction_name="nds_2025_bis_extraction"
    )

    # Record event context for transparency
    evaluator.add_custom_info(
        info={
            "event_name": EVENT_CONTEXT["event_name"],
            "event_year": EVENT_CONTEXT["event_year"],
            "event_date": EVENT_CONTEXT["event_date"],
            "event_day": EVENT_CONTEXT["event_day"],
            "event_location_state": EVENT_CONTEXT["event_location_state"]
        },
        info_type="event_context",
        info_name="national_dog_show_2025_context"
    )

    # Build verification tree according to rubric
    # All top-level parts are critical under a parallel root
    await build_winner_call_subtree(evaluator, root, extraction)
    await build_winner_breed_subtree(evaluator, root, extraction)
    await build_kennel_name_in_registered_subtree(evaluator, root, extraction)
    await build_kennel_location_subtree(evaluator, root, extraction)
    await build_milestones_subtree(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()