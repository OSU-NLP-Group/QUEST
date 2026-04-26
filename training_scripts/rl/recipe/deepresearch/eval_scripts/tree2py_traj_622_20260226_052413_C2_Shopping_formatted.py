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
TASK_ID = "atl_24hr_pharmacy_christmas"
TASK_DESCRIPTION = """
I need to find two 24-hour pharmacy locations in the Atlanta, Georgia metropolitan area that will be open and providing pharmacy services on Christmas Day (December 25, 2025). The pharmacies must be part of major national chains (CVS or Walgreens). For each pharmacy, provide the complete street address and a direct link to the pharmacy's store locator page from the chain's official website (CVS.com or Walgreens.com) that confirms the location details.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Pharmacy(BaseModel):
    chain: Optional[str] = None  # e.g., "CVS" or "Walgreens"
    name: Optional[str] = None   # store name or descriptor if present
    address: Optional[str] = None  # full street address as a single line if possible
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    url: Optional[str] = None  # official store locator URL on cvs.com or walgreens.com
    hours_text: Optional[str] = None  # any hours text quoted from the answer
    pharmacy_hours_text: Optional[str] = None  # quoted text for pharmacy hours if present
    is_24_hour: Optional[str] = None  # textual signal indicating 24 hours (e.g., "open 24 hours")
    open_on_christmas_2025: Optional[str] = None  # explicit claim or wording if present in answer
    notes: Optional[str] = None  # any extra details from answer


class PharmaciesExtraction(BaseModel):
    pharmacies: List[Pharmacy] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pharmacies() -> str:
    return """
    From the provided answer, extract up to two pharmacy locations that the answer claims meet the following requirements:
    - Chain: Must be CVS or Walgreens (major national chains).
    - Geography: Located in the Atlanta, Georgia metropolitan area.
    - Availability: Pharmacy is 24-hour and will be open/providing pharmacy services on Christmas Day 2025 (December 25, 2025).
    - URL: Provide a direct link to the official store locator page on CVS.com or Walgreens.com that shows the store location details.

    For each extracted pharmacy, return the following fields:
    - chain: "CVS" or "Walgreens" if stated; otherwise null.
    - name: The store name or descriptor if present; otherwise null.
    - address: A complete one-line street address if available in the answer (street number and name, city, state, zip). If parts are missing, include what is present from the answer only (do not invent).
    - city: City name if explicitly stated in the answer; otherwise null.
    - state: State (e.g., "GA") if explicitly stated; otherwise null.
    - zip: ZIP code if explicitly stated; otherwise null.
    - url: The direct store locator page URL on cvs.com or walgreens.com, if provided in the answer. If the answer does not include a valid official URL, set to null. Do not fabricate URLs.
    - hours_text: Any hours text quoted exactly from the answer that indicates general hours.
    - pharmacy_hours_text: Any hours text quoted exactly from the answer that specifically indicates pharmacy hours.
    - is_24_hour: Textual signal (e.g., "open 24 hours", "24/7") if present in the answer; otherwise null.
    - open_on_christmas_2025: Any explicit mention in the answer that this pharmacy is open on Christmas 2025; otherwise null.
    - notes: Any other helpful detail quoted or summarized from the answer.

    IMPORTANT:
    - Extract only what is explicitly present in the answer. Do not infer new details.
    - If the answer lists more than two candidates, extract only the first two mentioned.
    - For URLs, only include those on cvs.com or walgreens.com that appear to be store locator location pages; otherwise set url to null.
    - If a field is not present in the answer, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
ATLANTA_METRO_HINT = (
    "Consider a location to be in the Atlanta metropolitan area if the page shows an address with a city commonly "
    "recognized as part of Greater Atlanta (examples include: Atlanta, Sandy Springs, Roswell, Alpharetta, Johns Creek, "
    "Marietta, Smyrna, Decatur, Dunwoody, Brookhaven, East Point, College Park, Forest Park, Doraville, Chamblee, Tucker, "
    "Stone Mountain, Norcross, Duluth, Lawrenceville, Suwanee, Buford, Sugar Hill, Snellville, Lilburn, Mableton, Austell, "
    "Kennesaw, Woodstock, Acworth, Canton, Cumming, Stockbridge, McDonough, Conyers, Covington, Douglasville, Fayetteville, "
    "Peachtree City, Newnan, Jonesboro, Union City, Fairburn, Riverdale). Focus on what the page explicitly shows for the address."
)

CHRISTMAS_OPEN_HINT = (
    "Determine whether the PHARMACY (not just the retail store) will be open on December 25, 2025 (Christmas Day). "
    "If the page explicitly shows 'Pharmacy open 24 hours', '24/7 pharmacy', or pharmacy hours indicating 24 hours every day, "
    "then it is reasonable to conclude the pharmacy is open on that date. If the page only shows 'store open 24 hours' but "
    "the pharmacy hours differ or show closures on holidays, treat this as NOT supported. Prefer cues labeled 'Pharmacy hours'."
)

REFERENCE_URL_HINT = (
    "The URL must be an official chain site store-locator page that shows location details. Acceptable examples include: "
    "CVS: https://www.cvs.com/store-locator/... ; Walgreens: https://www.walgreens.com/locator/... . "
    "Reject URLs that are not on cvs.com or walgreens.com or that do not show location details (address/hours) for a specific store."
)


async def verify_one_pharmacy(
    evaluator: Evaluator,
    parent_node,
    item: Pharmacy,
    idx: int
) -> None:
    """
    Build and evaluate the verification subtree for a single pharmacy.
    """
    # Create a parallel node for this pharmacy (non-critical to allow partial scoring across pharmacies)
    label = "First" if idx == 0 else "Second"
    ph_node = evaluator.add_parallel(
        id=f"{label}_Pharmacy",
        desc=f"The {label.lower()} pharmacy meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # 1) Reference_URL (critical)
    ref_leaf = evaluator.add_leaf(
        id=f"{label}_Reference_URL",
        desc="A direct link to the pharmacy's store locator page from CVS.com or Walgreens.com showing the location details",
        parent=ph_node,
        critical=True
    )
    ref_claim = (
        f"The provided URL is on cvs.com or walgreens.com and is a store-locator page that displays this location's details "
        f"(address and hours). URL: {item.url if item and item.url else 'None provided'}"
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=item.url if item and item.url else None,
        additional_instruction=REFERENCE_URL_HINT
    )

    # 2) Chain_Membership (critical)
    chain_leaf = evaluator.add_leaf(
        id=f"{label}_Chain_Membership",
        desc="The pharmacy must be part of a major national chain (CVS or Walgreens)",
        parent=ph_node,
        critical=True
    )
    chain_name = (item.chain or "").strip() if item else ""
    if not chain_name:
        chain_claim = "This location is part of either CVS Pharmacy or Walgreens (a major national chain)."
    else:
        chain_claim = f"This location is a {chain_name} pharmacy (CVS Pharmacy or Walgreens Pharmacy)."
    await evaluator.verify(
        claim=chain_claim,
        node=chain_leaf,
        sources=item.url if item and item.url else None,
        additional_instruction="Confirm brand identity from the page (logos, headings, or labels)."
    )

    # 3) Location_Atlanta_Metro (critical)
    loc_leaf = evaluator.add_leaf(
        id=f"{label}_Location_Atlanta_Metro",
        desc="The pharmacy must be located in the Atlanta, Georgia metropolitan area with a verifiable street address",
        parent=ph_node,
        critical=True
    )
    addr_display = ""
    if item:
        parts = [p for p in [item.address, item.city, item.state, item.zip] if p]
        addr_display = ", ".join(parts)
    if addr_display:
        loc_claim = (
            f"The page shows the complete street address '{addr_display}', and this address is in the Atlanta, Georgia "
            f"metropolitan area."
        )
    else:
        loc_claim = (
            "The page shows a complete street address (street, city, state, and ZIP) for this location, and that address is "
            "in the Atlanta, Georgia metropolitan area."
        )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=item.url if item and item.url else None,
        additional_instruction=(
            "Verify that the page explicitly displays the full mailing address and that it indicates a city in Georgia that "
            "belongs to the Atlanta metropolitan area. " + ATLANTA_METRO_HINT
        )
    )

    # 4) 24Hour_Open_Christmas (critical)
    open_leaf = evaluator.add_leaf(
        id=f"{label}_24Hour_Open_Christmas",
        desc="The pharmacy must be a 24-hour pharmacy location that is open and providing pharmacy services on Christmas Day (December 25, 2025)",
        parent=ph_node,
        critical=True
    )
    # Provide what the answer said for transparency (not required, but helpful)
    hours_bits = []
    if item and item.is_24_hour:
        hours_bits.append(f"24h-indicator: {item.is_24_hour}")
    if item and item.pharmacy_hours_text:
        hours_bits.append(f"pharmacy_hours_text: {item.pharmacy_hours_text}")
    if item and item.hours_text:
        hours_bits.append(f"hours_text: {item.hours_text}")
    if item and item.open_on_christmas_2025:
        hours_bits.append(f"christmas_note: {item.open_on_christmas_2025}")
    hours_context = "; ".join(hours_bits) if hours_bits else "no explicit hours text captured from the answer"

    open_claim = (
        f"The pharmacy is open 24 hours and will be providing pharmacy services on December 25, 2025 (Christmas Day). "
        f"(Answer-provided hours context: {hours_context})"
    )
    await evaluator.verify(
        claim=open_claim,
        node=open_leaf,
        sources=item.url if item and item.url else None,
        additional_instruction=CHRISTMAS_OPEN_HINT
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
) -> Dict:
    """
    Evaluate an answer for the Atlanta 24-hour pharmacies open on Christmas 2025 task.
    """
    # Initialize evaluator with a PARALLEL root aggregation (two pharmacies can independently satisfy the task)
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

    # Create the top-level node mirroring the rubric root
    task_node = evaluator.add_parallel(
        id="Find_Two_24Hour_Pharmacies_Open_Christmas",
        desc="Find two 24-hour pharmacy locations from major chains (CVS or Walgreens) in the Atlanta, Georgia metropolitan area that are open for pharmacy services on Christmas Day 2025",
        parent=root,
        critical=False
    )

    # Extract pharmacies from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_pharmacies(),
        template_class=PharmaciesExtraction,
        extraction_name="pharmacies_extraction"
    )

    # Select up to two pharmacies (pad with empty placeholders if fewer)
    items: List[Pharmacy] = list(extracted.pharmacies[:2])
    while len(items) < 2:
        items.append(Pharmacy())

    # Build verification subtrees for the first two pharmacies
    await verify_one_pharmacy(evaluator, task_node, items[0], idx=0)
    await verify_one_pharmacy(evaluator, task_node, items[1], idx=1)

    # Return the evaluation summary
    return evaluator.get_summary()