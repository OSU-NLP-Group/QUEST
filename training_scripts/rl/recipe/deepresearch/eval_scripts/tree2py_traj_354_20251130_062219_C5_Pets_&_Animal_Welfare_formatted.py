import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# ----------------------------- Task Constants ----------------------------- #
TASK_ID = "akc_2025_best_in_show"
TASK_DESCRIPTION = (
    "Research the Best in Show winners from the two major AKC-sanctioned dog shows that took place in 2025: "
    "the Westminster Kennel Club Dog Show and the National Dog Show Presented by Purina. For each show, provide "
    "the breed of the Best in Show winner, the handler, the AKC group, the relevant date (award date for Westminster; "
    "broadcast date for National Dog Show), the venue (Westminster only), and a reference URL supporting the data."
)

ALLOWED_AKC_GROUPS = [
    "Sporting", "Hound", "Working", "Terrier", "Toy", "Non-Sporting", "Non Sporting", "Herding"
]


# ----------------------------- Data Models -------------------------------- #
class WestminsterInfo(BaseModel):
    breed: Optional[str] = None
    handler: Optional[str] = None
    group: Optional[str] = None
    award_date: Optional[str] = None  # Should be a February 2025 date
    venue: Optional[str] = None
    annual_number: Optional[str] = None  # e.g., "149th" or "149"
    references: List[str] = Field(default_factory=list)
    winner_count: Optional[int] = None
    winner_names: List[str] = Field(default_factory=list)


class NationalDogShowInfo(BaseModel):
    breed: Optional[str] = None
    handler: Optional[str] = None
    group: Optional[str] = None
    broadcast_date: Optional[str] = None  # Should be a November 2025 date
    references: List[str] = Field(default_factory=list)
    winner_count: Optional[int] = None
    winner_names: List[str] = Field(default_factory=list)


class DogShowsExtraction(BaseModel):
    westminster_2025: Optional[WestminsterInfo] = None
    national_dog_show_2025: Optional[NationalDogShowInfo] = None


# --------------------------- Extraction Prompt ---------------------------- #
def prompt_extract_dog_shows() -> str:
    return (
        "Extract structured information mentioned in the answer for the two shows:\n"
        "1) Westminster Kennel Club Dog Show 2025 (149th annual)\n"
        "2) National Dog Show Presented by Purina 2025\n\n"
        "For Westminster 2025, extract the following fields:\n"
        "- breed: the breed of the Best in Show winner\n"
        "- handler: the handler who presented the winning dog\n"
        "- group: the AKC group the winner represented\n"
        "- award_date: the date the Best in Show was awarded (should be in February 2025 if provided)\n"
        "- venue: the named venue where Best in Show judging took place\n"
        "- annual_number: the annual number of the show (e.g., 149 or 149th) if present\n"
        "- references: all reference URLs provided that support any of these fields (must be actual URLs; include all that apply)\n"
        "- winner_names: a list of the unique Best in Show winner names explicitly mentioned for Westminster (use registered name if present; otherwise, any unique identifier such as a call name)\n"
        "- winner_count: the number of unique Best in Show winner entities mentioned for Westminster (count the unique items in winner_names)\n\n"
        "For National Dog Show 2025, extract the following fields:\n"
        "- breed: the breed of the Best in Show winner\n"
        "- handler: the handler who presented the winning dog\n"
        "- group: the AKC group the winner represented\n"
        "- broadcast_date: the date the show was broadcast on television (should be in November 2025 if provided)\n"
        "- references: all reference URLs provided that support any of these fields (must be actual URLs; include all that apply)\n"
        "- winner_names: a list of the unique Best in Show winner names explicitly mentioned for the National Dog Show (use registered name if present; otherwise, any unique identifier)\n"
        "- winner_count: the number of unique Best in Show winner entities mentioned for the National Dog Show (count the unique items in winner_names)\n\n"
        "Important extraction rules:\n"
        "- Only extract information that is explicitly present in the answer text.\n"
        "- If a field is not mentioned, return null for scalars or an empty list for arrays.\n"
        "- For 'references', include all URLs in the answer that correspond to the specific show; valid formats include raw links and markdown links.\n"
        "- For 'annual_number', return a numeric string if possible (e.g., 149), otherwise keep the original provided token (e.g., '149th').\n"
        "- winner_count must equal the length of winner_names when both are provided; if winner_names is empty and the text clearly implies exactly one winner without a name, set winner_count to 1.\n"
        "- Do not invent URLs or infer missing values."
    )


# -------------------------- Verification Helpers -------------------------- #
def estimate_winner_count(names: Optional[List[str]], count_val: Optional[int]) -> Optional[int]:
    if count_val is not None:
        return count_val
    if names is not None:
        return len([n for n in names if isinstance(n, str) and n.strip() != ""])
    return None


def ensure_list(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# ---------------------------- Show Verifiers ------------------------------ #
async def verify_westminster(evaluator: Evaluator, parent_node, west: Optional[WestminsterInfo]) -> None:
    # Create show node
    show_node = evaluator.add_parallel(
        id="Westminster_2025",
        desc="Complete, verifiable information for the 2025 (149th annual) Westminster Kennel Club Dog Show Best in Show winner.",
        parent=parent_node,
        critical=False
    )

    refs = ensure_list(west.references if west else None)

    # Exactly one winner
    count_est = estimate_winner_count(west.winner_names if west else None, west.winner_count if west else None)
    evaluator.add_custom_node(
        result=(count_est == 1),
        id="Westminster_Exactly_One_Winner",
        desc="Documents exactly one Best in Show winner for Westminster 2025 (no ambiguity/multiple winners).",
        parent=show_node,
        critical=True
    )

    # 149th annual
    node_149 = evaluator.add_leaf(
        id="Westminster_149th_Annual",
        desc="States/reflects that Westminster 2025 is the 149th annual show.",
        parent=show_node,
        critical=True
    )
    claim_149 = "The 2025 Westminster Kennel Club Dog Show is the 149th annual show."
    await evaluator.verify(
        claim=claim_149,
        node=node_149,
        sources=refs,
        additional_instruction="Verify that at least one provided source explicitly mentions '149' or '149th' in connection with the 2025 Westminster Kennel Club Dog Show."
    )

    # Winner breed
    node_breed = evaluator.add_leaf(
        id="Westminster_Winner_Breed",
        desc="Identifies the breed of the Best in Show winner.",
        parent=show_node,
        critical=True
    )
    claim_breed = f"The Best in Show winner's breed at the 2025 Westminster Kennel Club Dog Show is '{(west.breed if west and west.breed else '')}'."
    await evaluator.verify(
        claim=claim_breed,
        node=node_breed,
        sources=refs,
        additional_instruction="Verify that the provided sources explicitly state the Best in Show winner's breed for the 2025 event."
    )

    # Breed is AKC-recognized
    node_akc = evaluator.add_leaf(
        id="Westminster_Breed_Is_AKC_Recognized",
        desc="The stated winning breed is an AKC-recognized breed (verifiable).",
        parent=show_node,
        critical=True
    )
    claim_akc = f"The breed '{(west.breed if west and west.breed else '')}' is recognized by the American Kennel Club."
    await evaluator.verify(
        claim=claim_akc,
        node=node_akc,
        sources=refs,
        additional_instruction="Prefer AKC or official show sources to confirm AKC recognition. If the breed is not mentioned or the pages do not show AKC recognition, judge as not supported."
    )

    # Handler name
    node_handler = evaluator.add_leaf(
        id="Westminster_Handler_Name",
        desc="Provides the handler name who presented the winning dog.",
        parent=show_node,
        critical=True
    )
    claim_handler = f"The handler of the 2025 Westminster Best in Show winner was '{(west.handler if west and west.handler else '')}'."
    await evaluator.verify(
        claim=claim_handler,
        node=node_handler,
        sources=refs,
        additional_instruction="Confirm the handler's name from the official show page, AKC, or major news coverage explicitly tied to the 2025 Westminster BIS result."
    )

    # Winner group
    node_group = evaluator.add_leaf(
        id="Westminster_Winner_Group",
        desc="Identifies the AKC group the winner represented, and it is one of: Sporting, Hound, Working, Terrier, Toy, Non-Sporting, Herding.",
        parent=show_node,
        critical=True
    )
    group_val = (west.group if west and west.group else "")
    claim_group = (
        f"The 2025 Westminster Best in Show winner represented the AKC '{group_val}' Group, "
        f"which must be one of {ALLOWED_AKC_GROUPS}."
    )
    await evaluator.verify(
        claim=claim_group,
        node=node_group,
        sources=refs,
        additional_instruction="Verify both that the page states the group for the BIS winner and that it is one of the seven AKC groups listed."
    )

    # Best in Show award date
    node_date = evaluator.add_leaf(
        id="Westminster_Best_In_Show_Date",
        desc="Provides the date Best in Show was awarded, and the date is in February 2025.",
        parent=show_node,
        critical=True
    )
    claim_date = (
        f"The Best in Show was awarded on '{(west.award_date if west and west.award_date else '')}', "
        "and this date falls in February 2025."
    )
    await evaluator.verify(
        claim=claim_date,
        node=node_date,
        sources=refs,
        additional_instruction="Confirm the exact award date from the page; also verify that the month is February and the year is 2025."
    )

    # Venue name
    node_venue = evaluator.add_leaf(
        id="Westminster_Venue_Name",
        desc="Specifies the named venue where Best in Show judging took place.",
        parent=show_node,
        critical=True
    )
    claim_venue = f"The Best in Show judging took place at '{(west.venue if west and west.venue else '')}'."
    await evaluator.verify(
        claim=claim_venue,
        node=node_venue,
        sources=refs,
        additional_instruction="Verify that the page explicitly names the venue where the BIS judging occurred for 2025."
    )

    # Reference URLs checks: existence, acceptability, and support for fields
    refs_parent = evaluator.add_parallel(
        id="Westminster_Reference_URLs_Acceptable_And_Supporting",
        desc="Provides reference URL(s) from official dog show sources, AKC sources, or major news outlets, and the URL(s) support the stated Westminster fields (breed, handler, group, award date, venue).",
        parent=show_node,
        critical=True
    )

    # Existence of at least one URL
    evaluator.add_custom_node(
        result=(len(refs) > 0),
        id="Westminster_URLs_Exist",
        desc="At least one reference URL is provided for Westminster.",
        parent=refs_parent,
        critical=True
    )

    # Acceptability of at least one URL
    node_urls_ok = evaluator.add_leaf(
        id="Westminster_URLs_Acceptable",
        desc="At least one Westminster reference URL is from an official show source, AKC, or a major news outlet.",
        parent=refs_parent,
        critical=True
    )
    claim_urls_ok = (
        "This page is an acceptable reference source for Westminster 2025 Best in Show information: "
        "it is either an official dog show source (e.g., westminsterkennelclub.org), "
        "an AKC source (akc.org), or a major news outlet (AP, Reuters, NBC, ABC, CBS, CNN, NYTimes, Washington Post, etc.)."
    )
    await evaluator.verify(
        claim=claim_urls_ok,
        node=node_urls_ok,
        sources=refs,
        additional_instruction="Judge a URL acceptable if it is from the official Westminster site, AKC site, or a widely recognized major news organization."
    )

    # Field-specific support leaves (pass if any URL supports the claim)
    node_refs_breed = evaluator.add_leaf(
        id="Westminster_URLs_Support_Breed",
        desc="The references support the breed field for Westminster 2025.",
        parent=refs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms the Best in Show winner's breed is '{(west.breed if west and west.breed else '')}' for Westminster 2025.",
        node=node_refs_breed,
        sources=refs,
        additional_instruction="Verify that the page explicitly mentions the breed of the 2025 Westminster BIS winner."
    )

    node_refs_handler = evaluator.add_leaf(
        id="Westminster_URLs_Support_Handler",
        desc="The references support the handler field for Westminster 2025.",
        parent=refs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms the handler for the 2025 Westminster BIS winner is '{(west.handler if west and west.handler else '')}'.",
        node=node_refs_handler,
        sources=refs,
        additional_instruction="Verify that the page explicitly mentions the handler associated with the 2025 Westminster BIS winner."
    )

    node_refs_group = evaluator.add_leaf(
        id="Westminster_URLs_Support_Group",
        desc="The references support the AKC group field for Westminster 2025.",
        parent=refs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms the BIS winner represented the '{group_val}' Group at Westminster 2025.",
        node=node_refs_group,
        sources=refs,
        additional_instruction="Verify that the page explicitly mentions the group for the Westminster 2025 BIS winner."
    )

    node_refs_date = evaluator.add_leaf(
        id="Westminster_URLs_Support_Date",
        desc="The references support the award date field for Westminster 2025.",
        parent=refs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms that the Best in Show award date was '{(west.award_date if west and west.award_date else '')}' (February 2025).",
        node=node_refs_date,
        sources=refs,
        additional_instruction="Verify that the page explicitly mentions the BIS award date for 2025 and that it is in February 2025."
    )

    node_refs_venue = evaluator.add_leaf(
        id="Westminster_URLs_Support_Venue",
        desc="The references support the venue field for Westminster 2025.",
        parent=refs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms the Best in Show judging venue was '{(west.venue if west and west.venue else '')}'.",
        node=node_refs_venue,
        sources=refs,
        additional_instruction="Verify that the page explicitly names the venue where BIS was judged in 2025."
    )


async def verify_national(evaluator: Evaluator, parent_node, nat: Optional[NationalDogShowInfo]) -> None:
    # Create show node
    show_node = evaluator.add_parallel(
        id="National_Dog_Show_2025",
        desc="Complete, verifiable information for the 2025 National Dog Show Presented by Purina Best in Show winner.",
        parent=parent_node,
        critical=False
    )

    refs = ensure_list(nat.references if nat else None)

    # Exactly one winner
    count_est = estimate_winner_count(nat.winner_names if nat else None, nat.winner_count if nat else None)
    evaluator.add_custom_node(
        result=(count_est == 1),
        id="National_Exactly_One_Winner",
        desc="Documents exactly one Best in Show winner for National Dog Show 2025 (no ambiguity/multiple winners).",
        parent=show_node,
        critical=True
    )

    # Winner breed
    node_breed = evaluator.add_leaf(
        id="National_Winner_Breed",
        desc="Identifies the breed of the Best in Show winner.",
        parent=show_node,
        critical=True
    )
    claim_breed = f"The 2025 National Dog Show Best in Show winner's breed is '{(nat.breed if nat and nat.breed else '')}'."
    await evaluator.verify(
        claim=claim_breed,
        node=node_breed,
        sources=refs,
        additional_instruction="Verify that the page explicitly states the BIS winner's breed for the 2025 National Dog Show Presented by Purina."
    )

    # Breed is AKC-recognized
    node_akc = evaluator.add_leaf(
        id="National_Breed_Is_AKC_Recognized",
        desc="The stated winning breed is an AKC-recognized breed (verifiable).",
        parent=show_node,
        critical=True
    )
    claim_akc = f"The breed '{(nat.breed if nat and nat.breed else '')}' is recognized by the American Kennel Club."
    await evaluator.verify(
        claim=claim_akc,
        node=node_akc,
        sources=refs,
        additional_instruction="Prefer AKC or official show sources to confirm AKC recognition. If the breed cannot be confirmed as AKC-recognized from the sources, judge as not supported."
    )

    # Handler name
    node_handler = evaluator.add_leaf(
        id="National_Handler_Name",
        desc="Provides the handler name who presented the winning dog.",
        parent=show_node,
        critical=True
    )
    claim_handler = f"The handler of the 2025 National Dog Show BIS winner was '{(nat.handler if nat and nat.handler else '')}'."
    await evaluator.verify(
        claim=claim_handler,
        node=node_handler,
        sources=refs,
        additional_instruction="Verify that the page explicitly mentions the handler associated with the 2025 National Dog Show BIS winner."
    )

    # Winner group
    node_group = evaluator.add_leaf(
        id="National_Winner_Group",
        desc="Identifies the AKC group the winner represented, and it is one of: Sporting, Hound, Working, Terrier, Toy, Non-Sporting, Herding.",
        parent=show_node,
        critical=True
    )
    group_val = (nat.group if nat and nat.group else "")
    claim_group = (
        f"The 2025 National Dog Show Best in Show winner represented the AKC '{group_val}' Group, "
        f"which must be one of {ALLOWED_AKC_GROUPS}."
    )
    await evaluator.verify(
        claim=claim_group,
        node=node_group,
        sources=refs,
        additional_instruction="Verify both that the page states the group for the BIS winner and that it is one of the seven AKC groups listed."
    )

    # Broadcast date in November 2025
    node_date = evaluator.add_leaf(
        id="National_Broadcast_Date",
        desc="Provides the date the show was broadcast on television, and the date is in November 2025.",
        parent=show_node,
        critical=True
    )
    claim_date = (
        f"The 2025 National Dog Show broadcast date was '{(nat.broadcast_date if nat and nat.broadcast_date else '')}', "
        "and this date falls in November 2025."
    )
    await evaluator.verify(
        claim=claim_date,
        node=node_date,
        sources=refs,
        additional_instruction="Confirm the broadcast date on the page and that it is within November 2025."
    )

    # Thanksgiving Day airing
    node_thanks = evaluator.add_leaf(
        id="National_Thanksgiving_Day_Airing",
        desc="The stated broadcast date corresponds to U.S. Thanksgiving Day (consistent with the constraint that it traditionally airs on Thanksgiving Day).",
        parent=show_node,
        critical=True
    )
    claim_thanksgiving = (
        f"The date '{(nat.broadcast_date if nat and nat.broadcast_date else '')}' falls on U.S. Thanksgiving Day in 2025 "
        "(the 4th Thursday of November)."
    )
    await evaluator.verify(
        claim=claim_thanksgiving,
        node=node_thanks,
        additional_instruction="Compute the 4th Thursday of November 2025 (which is November 27, 2025). "
                              "Judge as correct only if the given broadcast date equals that day."
    )

    # Reference URLs checks: existence, acceptability, and support for fields (breed, handler, group, broadcast date)
    refs_parent = evaluator.add_parallel(
        id="National_Reference_URLs_Acceptable_And_Supporting",
        desc="Provides reference URL(s) from official dog show sources, AKC sources, or major news outlets, and the URL(s) support the stated National Dog Show fields (breed, handler, group, broadcast date).",
        parent=show_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(refs) > 0),
        id="National_URLs_Exist",
        desc="At least one reference URL is provided for the National Dog Show.",
        parent=refs_parent,
        critical=True
    )

    node_urls_ok = evaluator.add_leaf(
        id="National_URLs_Acceptable",
        desc="At least one National Dog Show reference URL is from an official show source, AKC, or a major news outlet.",
        parent=refs_parent,
        critical=True
    )
    claim_urls_ok = (
        "This page is an acceptable reference source for the 2025 National Dog Show Best in Show information: "
        "it is either an official National Dog Show/Purina/NBC page, an AKC source, or a major news outlet "
        "(AP, Reuters, NBC, ABC, CBS, CNN, NYTimes, Washington Post, etc.)."
    )
    await evaluator.verify(
        claim=claim_urls_ok,
        node=node_urls_ok,
        sources=refs,
        additional_instruction="Judge a URL acceptable if it is from an official show page (nationaldogshow.com, purina.com, nbc.com), AKC (akc.org), or a widely recognized major news organization."
    )

    node_refs_breed = evaluator.add_leaf(
        id="National_URLs_Support_Breed",
        desc="The references support the breed field for the National Dog Show 2025.",
        parent=refs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms the BIS winner's breed is '{(nat.breed if nat and nat.breed else '')}' for the 2025 National Dog Show.",
        node=node_refs_breed,
        sources=refs,
        additional_instruction="Verify that the page explicitly mentions the winner's breed for the National Dog Show 2025."
    )

    node_refs_handler = evaluator.add_leaf(
        id="National_URLs_Support_Handler",
        desc="The references support the handler field for the National Dog Show 2025.",
        parent=refs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms the handler is '{(nat.handler if nat and nat.handler else '')}' for the 2025 National Dog Show BIS winner.",
        node=node_refs_handler,
        sources=refs,
        additional_instruction="Verify that the page explicitly mentions the handler associated with the 2025 National Dog Show BIS winner."
    )

    node_refs_group = evaluator.add_leaf(
        id="National_URLs_Support_Group",
        desc="The references support the group field for the National Dog Show 2025.",
        parent=refs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms the BIS winner represented the '{group_val}' Group at the National Dog Show 2025.",
        node=node_refs_group,
        sources=refs,
        additional_instruction="Verify that the page explicitly mentions the group for the 2025 National Dog Show BIS winner."
    )

    node_refs_date = evaluator.add_leaf(
        id="National_URLs_Support_Broadcast_Date",
        desc="The references support the broadcast date field for the National Dog Show 2025.",
        parent=refs_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms the National Dog Show 2025 broadcast date was '{(nat.broadcast_date if nat and nat.broadcast_date else '')}' (in November 2025).",
        node=node_refs_date,
        sources=refs,
        additional_instruction="Verify that the page explicitly mentions the TV broadcast date and it is in November 2025."
    )


# --------------------------- Main Evaluation ------------------------------ #
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
        strategy=AggregationStrategy.PARALLEL,  # Westminster and National are independent
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
        prompt=prompt_extract_dog_shows(),
        template_class=DogShowsExtraction,
        extraction_name="dog_shows_extraction"
    )

    # Root node (set non-critical to enable partial scoring even if one show fails)
    main_node = evaluator.add_parallel(
        id="Major_2025_Dog_Shows",
        desc="Best in Show winner information for the two specified major AKC-sanctioned 2025 dog shows (Westminster and National Dog Show Presented by Purina).",
        parent=root,
        critical=False
    )

    # Verify Westminster
    await verify_westminster(
        evaluator=evaluator,
        parent_node=main_node,
        west=extraction.westminster_2025
    )

    # Verify National Dog Show
    await verify_national(
        evaluator=evaluator,
        parent_node=main_node,
        nat=extraction.national_dog_show_2025
    )

    # Optional: add custom info to summary
    evaluator.add_custom_info(
        info={
            "allowed_akc_groups": ALLOWED_AKC_GROUPS,
            "notes": "All field verifications prefer official show, AKC, or major news sources."
        },
        info_type="config",
        info_name="evaluation_policy"
    )

    return evaluator.get_summary()