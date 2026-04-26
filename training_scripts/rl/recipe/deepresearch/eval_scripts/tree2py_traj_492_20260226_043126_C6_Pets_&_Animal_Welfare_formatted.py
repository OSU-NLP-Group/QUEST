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
TASK_ID = "dog_show_winners_2025"
TASK_DESCRIPTION = (
    "Identify the Best in Show winners from three major dog show competitions held in 2025: "
    "the Westminster Kennel Club Dog Show, the National Dog Show (Thanksgiving Day), and the AKC National Championship. "
    "For each winning dog, provide the following information: (1) The dog's official registered name and call name (if available), "
    "(2) The breed of the dog, (3) The name(s) of the dog's owner(s), (4) The name of the dog's breeder (if available), "
    "(5) The date or time period when the competition took place, (6) Confirmation that the dog won Best in Show, "
    "(7) The name of the handler (if available), and (8) A reference URL from an official source (such as the competition's official website "
    "or verified news outlet) documenting the win. All three competitions should be from 2025, and each should be a distinct dog show event."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WinnerItem(BaseModel):
    registered_name: Optional[str] = None
    call_name: Optional[str] = None
    breed: Optional[str] = None
    owners: List[str] = Field(default_factory=list)
    breeder: Optional[str] = None
    competition_date: Optional[str] = None
    handler: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class DogShowWinnersExtraction(BaseModel):
    westminster: Optional[WinnerItem] = None
    national_dog_show: Optional[WinnerItem] = None
    akc_national_championship: Optional[WinnerItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_winners() -> str:
    return """
    Extract the Best in Show winners for three separate 2025 dog show competitions, if mentioned in the answer:
    1) Westminster Kennel Club Dog Show (2025),
    2) National Dog Show (Thanksgiving Day, 2025),
    3) AKC National Championship (2025).
    
    For each event, if the answer provides the information, extract the following fields exactly as stated in the answer:
    - registered_name: The dog's official registered name (e.g., AKC registered name). If not stated, return null.
    - call_name: The dog's call name or nickname (if given). If not stated, return null.
    - breed: The dog's breed (e.g., "Sealyham Terrier"). If not stated, return null.
    - owners: An array of the owner names (strings) exactly as listed. If none are listed, return an empty array.
    - breeder: The breeder's name, if provided. If not stated, return null.
    - competition_date: The date or time period when the competition took place (e.g., "February 10–11, 2025", "Thanksgiving Day 2025"). If not stated, return null.
    - handler: The handler's name, if provided. If not stated, return null.
    - reference_urls: An array of explicit URLs (not just domain mentions) that document the win. Extract only URLs explicitly present in the answer, including markdown links; if none are present, return an empty array. Prefer official sources or verified news outlets when available.
    
    Return a JSON object with these top-level keys:
    {
      "westminster": WinnerItem | null,
      "national_dog_show": WinnerItem | null,
      "akc_national_championship": WinnerItem | null
    }
    If the answer does not mention one of the three competitions, set that field to null.
    Do not invent or infer any information not explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def safe_name_for_win(item: WinnerItem) -> str:
    """Choose a display name for win confirmation claim."""
    if item and item.registered_name and item.registered_name.strip():
        return item.registered_name.strip()
    if item and item.call_name and item.call_name.strip():
        return item.call_name.strip()
    return "the winning dog"


def owners_text(owners: List[str]) -> str:
    if not owners:
        return ""
    return ", ".join([o.strip() for o in owners if o and o.strip()])


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_westminster(evaluator: Evaluator, parent_node, item: Optional[WinnerItem]) -> None:
    show_node = evaluator.add_parallel(
        id="Westminster_2025_Winner",
        desc="Best in Show winner from the 2025 Westminster Kennel Club Dog Show",
        parent=parent_node,
        critical=False
    )

    # Reference URL presence (Critical for gating downstream verifications)
    ref_present = bool(item and item.reference_urls and len(item.reference_urls) > 0)
    evaluator.add_custom_node(
        result=ref_present,
        id="Reference_URL_Westminster",
        desc="A URL to the official Westminster Kennel Club website or verified news source documenting the win is provided",
        parent=show_node,
        critical=True
    )

    # Dog Identity
    dog_identity = evaluator.add_parallel(
        id="Dog_Identity",
        desc="Identity information about the winning dog",
        parent=show_node,
        critical=False
    )

    name_info = evaluator.add_parallel(
        id="Name_Information",
        desc="The dog's registered and call names",
        parent=dog_identity,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(item and item.registered_name and item.registered_name.strip()),
        id="Registered_Name",
        desc="The dog's official AKC registered name is provided",
        parent=name_info,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item and item.call_name and item.call_name.strip()),
        id="Call_Name",
        desc="The dog's call name (nickname) is provided",
        parent=name_info,
        critical=False
    )

    breed_leaf = evaluator.add_leaf(
        id="Breed_Name",
        desc="The specific breed of the winning dog is correctly identified",
        parent=dog_identity,
        critical=True
    )
    breed_claim = f"The winning dog's breed is '{(item.breed or '').strip()}'." if item else "The winning dog's breed is ''."
    await evaluator.verify(
        claim=breed_claim,
        node=breed_leaf,
        sources=(item.reference_urls if item else []),
        additional_instruction="Verify that the cited official/verified page explicitly specifies the dog's breed. Allow minor naming variants (e.g., 'Poodle, Miniature' vs 'Miniature Poodle')."
    )

    # Ownership Details
    ownership = evaluator.add_parallel(
        id="Ownership_Details",
        desc="Information about the dog's ownership and breeding",
        parent=show_node,
        critical=False
    )

    owners_leaf = evaluator.add_leaf(
        id="Owner_Names",
        desc="The name(s) of the dog's owner(s) are provided",
        parent=ownership,
        critical=True
    )
    owners_claim = (
        f"The owner(s) of the winning dog include {owners_text(item.owners)}."
        if item else "The owner(s) of the winning dog include ."
    )
    await evaluator.verify(
        claim=owners_claim,
        node=owners_leaf,
        sources=(item.reference_urls if item else []),
        additional_instruction="Verify the owner names on the source page. Treat as supported if the page lists owners that match or reasonably align with the provided names, allowing minor variations."
    )

    evaluator.add_custom_node(
        result=bool(item and item.breeder and item.breeder.strip()),
        id="Breeder_Name",
        desc="The name of the dog's breeder is provided",
        parent=ownership,
        critical=False
    )

    # Competition Performance
    perf = evaluator.add_parallel(
        id="Competition_Performance",
        desc="Details about the dog's win at Westminster 2025",
        parent=show_node,
        critical=False
    )

    win_info = evaluator.add_parallel(
        id="Win_Information",
        desc="Specific details about the competition win",
        parent=perf,
        critical=False
    )

    date_leaf = evaluator.add_leaf(
        id="Competition_Date",
        desc="The date or time period when the competition took place is provided",
        parent=win_info,
        critical=True
    )
    date_claim = "This competition took place in 2025."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=(item.reference_urls if item else []),
        additional_instruction="Confirm from the cited page that the event date/time period is in the year 2025 (e.g., explicit dates or references to the 2025 show)."
    )

    win_title_leaf = evaluator.add_leaf(
        id="Win_Title",
        desc="The specific title won (Best in Show) is confirmed",
        parent=win_info,
        critical=True
    )
    display_name = safe_name_for_win(item or WinnerItem())
    win_claim = f"The dog {display_name} won Best in Show at the 2025 Westminster Kennel Club Dog Show."
    await evaluator.verify(
        claim=win_claim,
        node=win_title_leaf,
        sources=(item.reference_urls if item else []),
        additional_instruction="Verify the page clearly states the dog won 'Best in Show' at the 2025 Westminster Kennel Club Dog Show."
    )

    evaluator.add_custom_node(
        result=bool(item and item.handler and item.handler.strip()),
        id="Handler_Name",
        desc="The name of the handler is provided",
        parent=perf,
        critical=False
    )


async def verify_nds(evaluator: Evaluator, parent_node, item: Optional[WinnerItem]) -> None:
    show_node = evaluator.add_parallel(
        id="National_Dog_Show_2025_Winner",
        desc="Best in Show winner from the 2025 National Dog Show (Thanksgiving Day)",
        parent=parent_node,
        critical=False
    )

    ref_present = bool(item and item.reference_urls and len(item.reference_urls) > 0)
    evaluator.add_custom_node(
        result=ref_present,
        id="Reference_URL_NDS",
        desc="A URL to an official or verified news source documenting the win is provided",
        parent=show_node,
        critical=True
    )

    dog_identity = evaluator.add_parallel(
        id="Dog_Identity_NDS",
        desc="Identity information about the winning dog",
        parent=show_node,
        critical=False
    )

    name_info = evaluator.add_parallel(
        id="Name_Information_NDS",
        desc="The dog's registered and call names",
        parent=dog_identity,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(item and item.registered_name and item.registered_name.strip()),
        id="Registered_Name_NDS",
        desc="The dog's official registered name is provided",
        parent=name_info,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item and item.call_name and item.call_name.strip()),
        id="Call_Name_NDS",
        desc="The dog's call name is provided",
        parent=name_info,
        critical=False
    )

    breed_leaf = evaluator.add_leaf(
        id="Breed_Name_NDS",
        desc="The specific breed of the winning dog is correctly identified",
        parent=dog_identity,
        critical=True
    )
    breed_claim = f"The winning dog's breed is '{(item.breed or '').strip()}'." if item else "The winning dog's breed is ''."
    await evaluator.verify(
        claim=breed_claim,
        node=breed_leaf,
        sources=(item.reference_urls if item else []),
        additional_instruction="Verify the dog's breed from the official/verified source for the 2025 National Dog Show. Allow minor naming variants."
    )

    ownership = evaluator.add_parallel(
        id="Ownership_Details_NDS",
        desc="Information about the dog's ownership",
        parent=show_node,
        critical=False
    )

    owners_leaf = evaluator.add_leaf(
        id="Owner_Names_NDS",
        desc="The name(s) of the dog's owner(s) are provided",
        parent=ownership,
        critical=True
    )
    owners_claim = (
        f"The owner(s) of the winning dog include {owners_text(item.owners)}."
        if item else "The owner(s) of the winning dog include ."
    )
    await evaluator.verify(
        claim=owners_claim,
        node=owners_leaf,
        sources=(item.reference_urls if item else []),
        additional_instruction="Verify the owner names from the cited source for the National Dog Show. Accept reasonable variants."
    )

    evaluator.add_custom_node(
        result=bool(item and item.breeder and item.breeder.strip()),
        id="Breeder_Name_NDS",
        desc="The name of the dog's breeder is provided",
        parent=ownership,
        critical=False
    )

    perf = evaluator.add_parallel(
        id="Competition_Performance_NDS",
        desc="Details about the dog's win at National Dog Show 2025",
        parent=show_node,
        critical=False
    )

    win_info = evaluator.add_parallel(
        id="Win_Information_NDS",
        desc="Specific details about the competition win",
        parent=perf,
        critical=False
    )

    date_leaf = evaluator.add_leaf(
        id="Competition_Date_NDS",
        desc="The date when the competition took place is provided",
        parent=win_info,
        critical=True
    )
    date_claim = "This competition took place in 2025."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=(item.reference_urls if item else []),
        additional_instruction="Confirm that the National Dog Show event referenced is the 2025 Thanksgiving broadcast/show."
    )

    win_title_leaf = evaluator.add_leaf(
        id="Win_Title_NDS",
        desc="The specific title won (Best in Show) is confirmed",
        parent=win_info,
        critical=True
    )
    display_name = safe_name_for_win(item or WinnerItem())
    win_claim = f"The dog {display_name} won Best in Show at the 2025 National Dog Show."
    await evaluator.verify(
        claim=win_claim,
        node=win_title_leaf,
        sources=(item.reference_urls if item else []),
        additional_instruction="Verify that the page explicitly confirms the dog won Best in Show at the 2025 National Dog Show."
    )

    evaluator.add_custom_node(
        result=bool(item and item.handler and item.handler.strip()),
        id="Handler_Name_NDS",
        desc="The name of the handler is provided",
        parent=perf,
        critical=False
    )


async def verify_akc(evaluator: Evaluator, parent_node, item: Optional[WinnerItem]) -> None:
    show_node = evaluator.add_parallel(
        id="AKC_National_Championship_2025_Winner",
        desc="Best in Show winner from the 2025 AKC National Championship",
        parent=parent_node,
        critical=False
    )

    ref_present = bool(item and item.reference_urls and len(item.reference_urls) > 0)
    evaluator.add_custom_node(
        result=ref_present,
        id="Reference_URL_AKC",
        desc="A URL to the official AKC website or verified news source documenting the win is provided",
        parent=show_node,
        critical=True
    )

    dog_identity = evaluator.add_parallel(
        id="Dog_Identity_AKC",
        desc="Identity information about the winning dog",
        parent=show_node,
        critical=False
    )

    name_info = evaluator.add_parallel(
        id="Name_Information_AKC",
        desc="The dog's registered and call names",
        parent=dog_identity,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(item and item.registered_name and item.registered_name.strip()),
        id="Registered_Name_AKC",
        desc="The dog's official AKC registered name is provided",
        parent=name_info,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item and item.call_name and item.call_name.strip()),
        id="Call_Name_AKC",
        desc="The dog's call name is provided",
        parent=name_info,
        critical=False
    )

    breed_leaf = evaluator.add_leaf(
        id="Breed_Name_AKC",
        desc="The specific breed of the winning dog is correctly identified",
        parent=dog_identity,
        critical=True
    )
    breed_claim = f"The winning dog's breed is '{(item.breed or '').strip()}'." if item else "The winning dog's breed is ''."
    await evaluator.verify(
        claim=breed_claim,
        node=breed_leaf,
        sources=(item.reference_urls if item else []),
        additional_instruction="Verify the breed from the official/verified AKC National Championship source. Allow minor naming variants."
    )

    ownership = evaluator.add_parallel(
        id="Ownership_Details_AKC",
        desc="Information about the dog's ownership and breeding",
        parent=show_node,
        critical=False
    )

    owners_leaf = evaluator.add_leaf(
        id="Owner_Names_AKC",
        desc="The name(s) of the dog's owner(s) or breeder are provided",
        parent=ownership,
        critical=True
    )
    owners_claim = (
        f"The owner(s) of the winning dog include {owners_text(item.owners)}."
        if item else "The owner(s) of the winning dog include ."
    )
    await evaluator.verify(
        claim=owners_claim,
        node=owners_leaf,
        sources=(item.reference_urls if item else []),
        additional_instruction="Verify the listed owner(s) on the cited AKC source. Accept reasonable variants or ordering differences."
    )

    evaluator.add_custom_node(
        result=bool(item and item.breeder and item.breeder.strip()),
        id="Breeder_Name_AKC",
        desc="The name of the dog's breeder is provided",
        parent=ownership,
        critical=False
    )

    perf = evaluator.add_parallel(
        id="Competition_Performance_AKC",
        desc="Details about the dog's win at AKC National Championship 2025",
        parent=show_node,
        critical=False
    )

    win_info = evaluator.add_parallel(
        id="Win_Information_AKC",
        desc="Specific details about the competition win",
        parent=perf,
        critical=False
    )

    date_leaf = evaluator.add_leaf(
        id="Competition_Date_AKC",
        desc="The date or time period when the competition took place is provided",
        parent=win_info,
        critical=True
    )
    date_claim = "This competition took place in 2025."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=(item.reference_urls if item else []),
        additional_instruction="Confirm from the cited page that the AKC National Championship event referenced is in the year 2025."
    )

    win_title_leaf = evaluator.add_leaf(
        id="Win_Title_AKC",
        desc="The specific title won (Best in Show) is confirmed",
        parent=win_info,
        critical=True
    )
    display_name = safe_name_for_win(item or WinnerItem())
    win_claim = f"The dog {display_name} won Best in Show at the 2025 AKC National Championship."
    await evaluator.verify(
        claim=win_claim,
        node=win_title_leaf,
        sources=(item.reference_urls if item else []),
        additional_instruction="Verify that the page clearly states the dog won 'Best in Show' at the 2025 AKC National Championship."
    )

    evaluator.add_custom_node(
        result=bool(item and item.handler and item.handler.strip()),
        id="Handler_Name_AKC",
        desc="The name of the handler is provided",
        parent=perf,
        critical=False
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 2025 Dog Show Best in Show winners task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation across the three shows
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_winners(),
        template_class=DogShowWinnersExtraction,
        extraction_name="winners_2025_extraction"
    )

    # Build verification subtrees for each show (always create nodes; missing info will naturally fail or be partial)
    await verify_westminster(evaluator, root, extraction.westminster)
    await verify_nds(evaluator, root, extraction.national_dog_show)
    await verify_akc(evaluator, root, extraction.akc_national_championship)

    # Optional: record custom info
    evaluator.add_custom_info(
        {"script_version": "1.0", "task_id": TASK_ID},
        info_type="meta",
        info_name="evaluation_metadata"
    )

    return evaluator.get_summary()