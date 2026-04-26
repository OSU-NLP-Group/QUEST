import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "aruba_outdoor_daytrip"
TASK_DESCRIPTION = (
    "I am planning a beginner-friendly outdoor recreation day trip in Aruba that includes both hiking and snorkeling. "
    "I need help gathering the following planning information:\n\n"
    "1. Identify one specific hiking trail within Arikok National Park that is documented as easy or suitable for beginners, "
    "and provide the trail's name and approximate duration or distance.\n\n"
    "2. Identify one specific snorkeling location in Aruba that is documented as accessible directly from shore (no boat required), "
    "and provide the location's name.\n\n"
    "3. Identify a snorkel gear rental service or provider in Aruba where the daily rental cost is $15 or less per set, "
    "including the provider's name and the exact daily rental price per set.\n\n"
    "4. Provide the current adult entrance fee for Arikok National Park.\n\n"
    "For each piece of information (hiking trail, snorkeling location, and snorkel rental service), include a URL reference "
    "to a webpage that supports the information provided."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ArubaPlanExtraction(BaseModel):
    # Hiking
    hiking_trail_name: Optional[str] = None
    hiking_trail_duration_or_distance: Optional[str] = None
    hiking_urls: List[str] = Field(default_factory=list)

    # Snorkeling
    snorkeling_location_name: Optional[str] = None
    snorkeling_urls: List[str] = Field(default_factory=list)

    # Rental
    rental_provider_name: Optional[str] = None
    rental_daily_price_per_set: Optional[str] = None
    rental_urls: List[str] = Field(default_factory=list)

    # Arikok entrance fee
    arikok_adult_entrance_fee: Optional[str] = None
    arikok_fee_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_aruba_plan() -> str:
    return """
    Extract the first (or primary) instance of each required planning item exactly as stated in the answer text. 
    Do not invent information. If an item is not present, set it to null (or empty list for URLs).
    Required fields:
    - hiking_trail_name: The specific named hiking trail inside Arikok National Park that is documented as easy or beginner-friendly.
    - hiking_trail_duration_or_distance: The approximate duration (e.g., "1 hour", "45 minutes") or distance (e.g., "2 km", "1.5 miles") for that trail.
    - hiking_urls: All URL(s) in the answer that support the hiking trail details (name and beginner/easy characterization). Return as an array of URLs.

    - snorkeling_location_name: The specific named snorkeling spot in Aruba that is explicitly stated as being accessible from shore (no boat required).
    - snorkeling_urls: All URL(s) in the answer that support the snorkeling location details (name and shore accessibility). Return as an array of URLs.

    - rental_provider_name: The name of a snorkel gear rental provider in Aruba.
    - rental_daily_price_per_set: The exact stated per-set daily rental price (e.g., "$10", "AWG 15") from the answer text.
    - rental_urls: All URL(s) in the answer that support the rental provider and its pricing. Return as an array of URLs.

    - arikok_adult_entrance_fee: The current adult entrance fee for Arikok National Park as stated in the answer text (e.g., "$15", "AWG 15").
    - arikok_fee_urls: All URL(s) in the answer that support the entrance fee information (if any are provided). Return as an array of URLs.

    Notes:
    - If multiple items are listed, select the first clearly identified one for each category.
    - Extract URLs exactly as written in the answer (including markdown links). Only include valid-looking URLs.
    - If a URL is missing a protocol, prepend http:// as needed.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_valid_url(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str):
            s = u.strip()
            if s.startswith("http://") or s.startswith("https://"):
                return True
    return False


def _filter_valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    res = []
    for u in urls:
        if isinstance(u, str):
            s = u.strip()
            if s.startswith("http://") or s.startswith("https://"):
                res.append(s)
    return res


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def _build_and_verify_tree(evaluator: Evaluator, extraction: ArubaPlanExtraction) -> None:
    # Top-level critical node representing the overall planning task
    plan_node = evaluator.add_parallel(
        id="Outdoor_Recreation_Planning",
        desc="Provide complete planning information for a beginner-friendly hiking and snorkeling day trip in Aruba",
        parent=evaluator.root,
        critical=True
    )

    # Create grouped critical sub-nodes to avoid cross-category gating side-effects
    hiking_group = evaluator.add_parallel(
        id="Hiking_Group",
        desc="Hiking trail information verification (within Arikok NP, easy/beginner-friendly, duration/distance, URL)",
        parent=plan_node,
        critical=True
    )
    snorkeling_group = evaluator.add_parallel(
        id="Snorkeling_Group",
        desc="Snorkeling location information verification (shore-accessible, URL)",
        parent=plan_node,
        critical=True
    )
    rental_group = evaluator.add_parallel(
        id="Rental_Group",
        desc="Snorkel rental provider information verification (<= $15 per set daily, URL)",
        parent=plan_node,
        critical=True
    )
    fee_group = evaluator.add_parallel(
        id="Fee_Group",
        desc="Arikok National Park adult entrance fee verification",
        parent=plan_node,
        critical=True
    )

    # ---------------------- Hiking checks ---------------------- #
    hiking_urls_valid = _has_valid_url(extraction.hiking_urls)
    # Hiking reference URL presence (critical)
    evaluator.add_custom_node(
        result=hiking_urls_valid,
        id="Hiking_Reference_URL",
        desc="Provide a valid URL reference supporting the hiking trail information (name and difficulty level)",
        parent=hiking_group,
        critical=True
    )

    # Verify trail name and beginner/easy claim using URLs
    node_hike_name = evaluator.add_leaf(
        id="Easy_Hiking_Trail_Name",
        desc="Identify a specific named hiking trail in Arikok National Park that is documented as easy or suitable for beginners",
        parent=hiking_group,
        critical=True
    )
    hike_name = (extraction.hiking_trail_name or "").strip()
    hike_claim = (
        f"The cited page(s) indicate that a hiking trail named '{hike_name}' is located within Arikok National Park "
        f"and is described as easy or suitable for beginners."
    )
    await evaluator.verify(
        claim=hike_claim,
        node=node_hike_name,
        sources=_filter_valid_urls(extraction.hiking_urls),
        additional_instruction=(
            "Confirm both the trail name and that its difficulty is described as 'easy', 'beginner-friendly', "
            "'family-friendly', or similar phrasing indicating low difficulty."
        ),
    )

    # Verify approximate duration or distance using URLs
    node_hike_duration = evaluator.add_leaf(
        id="Hiking_Trail_Duration",
        desc="Provide the approximate duration or distance for the identified hiking trail",
        parent=hiking_group,
        critical=True
    )
    hike_duration = (extraction.hiking_trail_duration_or_distance or "").strip()
    duration_claim = (
        f"The cited page(s) provide the approximate duration or distance for the trail '{hike_name}' "
        f"as '{hike_duration}'."
    )
    await evaluator.verify(
        claim=duration_claim,
        node=node_hike_duration,
        sources=_filter_valid_urls(extraction.hiking_urls),
        additional_instruction=(
            "Allow approximate phrasing, rounding, and reasonable unit variations (e.g., minutes vs. hours, miles vs. km). "
            "The page should explicitly or clearly imply this duration/distance for the named trail."
        ),
    )

    # ------------------- Snorkeling checks --------------------- #
    snorkeling_urls_valid = _has_valid_url(extraction.snorkeling_urls)
    evaluator.add_custom_node(
        result=snorkeling_urls_valid,
        id="Snorkeling_Reference_URL",
        desc="Provide a valid URL reference supporting the snorkeling location information (name and shore accessibility)",
        parent=snorkeling_group,
        critical=True
    )

    node_snorkel_spot = evaluator.add_leaf(
        id="Shore_Accessible_Snorkeling_Location",
        desc="Identify a specific named snorkeling location in Aruba that is documented as accessible from shore without requiring a boat",
        parent=snorkeling_group,
        critical=True
    )
    snorkel_name = (extraction.snorkeling_location_name or "").strip()
    snorkel_claim = (
        f"The cited page(s) indicate that the snorkeling location '{snorkel_name}' in Aruba "
        f"is accessible directly from shore (no boat required)."
    )
    await evaluator.verify(
        claim=snorkel_claim,
        node=node_snorkel_spot,
        sources=_filter_valid_urls(extraction.snorkeling_urls),
        additional_instruction=(
            "Look for wording like 'shore entry', 'beach entry', 'accessible from shore', or explicitly 'no boat required'."
        ),
    )

    # -------------------- Rental checks ------------------------ #
    rental_urls_valid = _has_valid_url(extraction.rental_urls)
    evaluator.add_custom_node(
        result=rental_urls_valid,
        id="Rental_Reference_URL",
        desc="Provide a valid URL reference supporting the snorkel rental pricing information",
        parent=rental_group,
        critical=True
    )

    node_rental = evaluator.add_leaf(
        id="Affordable_Snorkel_Rental_Option",
        desc="Identify a snorkel gear rental service or provider in Aruba with daily rental cost of $15 or less per set, including both the provider name and the exact rental price per set",
        parent=rental_group,
        critical=True
    )
    provider = (extraction.rental_provider_name or "").strip()
    price_text = (extraction.rental_daily_price_per_set or "").strip()
    rental_claim = (
        f"The cited page(s) state that the provider '{provider}' offers snorkel gear rental at '{price_text}' per set per day, "
        f"and this price is $15 or less per set per day."
    )
    await evaluator.verify(
        claim=rental_claim,
        node=node_rental,
        sources=_filter_valid_urls(extraction.rental_urls),
        additional_instruction=(
            "Confirm the provider name and the exact daily price per set for snorkel gear. "
            "Sets typically include mask, snorkel, and fins; minor variations are acceptable. "
            "The price must be at most $15 per set per day. If currency is not USD, focus on the stated value; "
            "do not perform currency conversion—just confirm the stated price is ≤ $15 if USD or explicitly listed as $X."
        ),
    )

    # ---------------------- Fee checks ------------------------- #
    node_fee = evaluator.add_leaf(
        id="Arikok_Adult_Entrance_Fee",
        desc="Provide the current adult entrance fee for Arikok National Park",
        parent=fee_group,
        critical=True
    )
    fee_text = (extraction.arikok_adult_entrance_fee or "").strip()
    fee_claim = (
        f"The cited page(s) state that the current adult entrance fee for Arikok National Park is '{fee_text}'."
    )
    await evaluator.verify(
        claim=fee_claim,
        node=node_fee,
        sources=_filter_valid_urls(extraction.arikok_fee_urls),
        additional_instruction=(
            "Confirm the adult entrance fee (per person) for Arikok National Park. "
            "Accept either USD or AWG as shown on the page. Minor formatting differences (e.g., 'per adult', 'per person') are acceptable."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Aruba beginner-friendly day trip planning task.
    """
    # Initialize evaluator with a parallel root; the critical logic will be inside the tree
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
        default_model=model
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_aruba_plan(),
        template_class=ArubaPlanExtraction,
        extraction_name="aruba_plan_extraction"
    )

    # Build verification tree and run checks
    await _build_and_verify_tree(evaluator, extraction)

    # Return summary
    return evaluator.get_summary()