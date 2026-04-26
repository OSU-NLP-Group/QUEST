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
TASK_ID = "park_sleep_cruise_fll"
TASK_DESCRIPTION = """
I'm planning a 7-night Caribbean cruise departing from Port Everglades Terminal 4 in Fort Lauderdale in July 2026. I need to find three different hotels in the Fort Lauderdale area that offer park-sleep-cruise packages suitable for my trip.

For each of the three hotels, please provide:
1. The hotel name and complete address
2. The distance from Port Everglades Terminal 4 (Disney Cruise Line terminal)
3. Confirmation that shuttle service to the cruise terminal is available
4. Verification that parking is included for at least 7 days (to cover my 7-night cruise)
5. The total package price for one night's stay with 7-day parking included
6. A reference URL where I can verify these package details

Please find three different hotels that meet all these requirements.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    """Structured info for one hotel as claimed by the answer."""
    name: Optional[str] = None
    address: Optional[str] = None
    distance_to_terminal_4: Optional[str] = None  # keep as free text (e.g., "2 miles", "approx. 1.8 mi")
    shuttle_to_terminal: Optional[str] = None     # free-text confirmation or description if present in the answer
    parking_days_included: Optional[str] = None   # free-text like "7 days", "up to 14 days", etc.
    package_price_1n_7d: Optional[str] = None     # free-text price string (e.g., "$199 + tax")
    reference_url: Optional[str] = None           # the URL to verify package details


class HotelsExtraction(BaseModel):
    """List of hotels mentioned in the answer."""
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    From the provided answer, extract all hotels in the Fort Lauderdale area that are claimed to offer a park-sleep-cruise (or park & cruise / stay, park & cruise) package for a Port Everglades cruise.

    For each hotel, return a JSON object with the following fields:
    - name: The hotel's name as stated in the answer
    - address: The complete street address as stated in the answer (including city/state/ZIP if available)
    - distance_to_terminal_4: The stated distance from Port Everglades Terminal 4 (as free text, e.g., "2 miles")
    - shuttle_to_terminal: Any statement indicating shuttle service to the Port Everglades cruise terminal (free text)
    - parking_days_included: The stated number of parking days included (free text, e.g., "7 days", "up to 14 days")
    - package_price_1n_7d: The total package price for one night with 7-day parking included (if stated; free text, e.g., "$199 + tax")
    - reference_url: A single reference URL where the hotel’s package details can be verified. Extract the actual URL explicitly present in the answer text.

    Rules:
    1) Extract only information explicitly present in the answer; do not infer or add details.
    2) For any missing field, set it to null.
    3) For reference_url, only include valid URLs explicitly present in the answer (plain URL or markdown link).
    4) Return a JSON object with a 'hotels' array containing all extracted hotels. We will only evaluate the first three hotels if more are provided.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _normalize_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    u = url.strip()
    if not u:
        return None
    if not (u.startswith("http://") or u.startswith("https://")):
        # lightweight normalization rule as per toolkit's special rules
        return "http://" + u
    return u


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelItem,
    index: int,
) -> None:
    """
    Build and verify the subtree for a single hotel.

    We use a sequential aggregation to gate downstream checks on basic existence
    (name, address, and a reference URL). Concrete factual checks are leaves
    verified against the provided reference URL.
    """
    hotel_id = f"hotel_{index + 1}"
    hotel_title = f"Hotel #{index + 1} verification"

    # Create the hotel node as SEQUENTIAL to enable gating by existence checks
    hotel_node = evaluator.add_sequential(
        id=hotel_id,
        desc=hotel_title,
        parent=parent_node,
        critical=False  # allow partial credit per hotel
    )

    # ---------- Existence gate: require name, address, and reference URL ----------
    ref_url = _normalize_url(hotel.reference_url)
    required_info_ok = bool(hotel.name and hotel.name.strip()) and \
                       bool(hotel.address and hotel.address.strip()) and \
                       bool(ref_url)

    evaluator.add_custom_node(
        result=required_info_ok,
        id=f"{hotel_id}_required_info",
        desc=f"{hotel_title} – required info present (name, address, reference URL)",
        parent=hotel_node,
        critical=True  # mandatory for meaningful verification
    )

    # ---------- Name & Address verification (parallel, both critical) ----------
    name_addr_node = evaluator.add_parallel(
        id=f"{hotel_id}_name_location",
        desc="Provide hotel name and complete address (both must be correct)",
        parent=hotel_node,
        critical=True  # both child leaves are critical
    )

    # Name accuracy
    name_leaf = evaluator.add_leaf(
        id=f"{hotel_id}_name_accurate",
        desc="Hotel name matches the referenced page",
        parent=name_addr_node,
        critical=True
    )
    name_claim = f"The hotel's name is '{hotel.name}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=ref_url,
        additional_instruction=(
            "Verify that the referenced page is about the same hotel property and the displayed hotel name "
            "matches the claimed name. Allow minor variations, casing differences, or inclusion of brand/chain "
            "suffixes (e.g., 'Fort Lauderdale Airport/Cruise Port')."
        ),
    )

    # Address accuracy
    address_leaf = evaluator.add_leaf(
        id=f"{hotel_id}_address_accurate",
        desc="Hotel address matches the referenced page",
        parent=name_addr_node,
        critical=True
    )
    address_claim = f"The hotel's address is '{hotel.address}'."
    await evaluator.verify(
        claim=address_claim,
        node=address_leaf,
        sources=ref_url,
        additional_instruction=(
            "Verify that the page lists the same street address. Accept minor formatting differences such as "
            "abbreviations (e.g., 'Ave' vs 'Avenue'), punctuation, inclusion/exclusion of ZIP code, or capitalization."
        ),
    )

    # ---------- Distance verification ----------
    # Require that the answer provides a distance string before verifying
    distance_provided = bool(hotel.distance_to_terminal_4 and hotel.distance_to_terminal_4.strip())
    evaluator.add_custom_node(
        result=distance_provided,
        id=f"{hotel_id}_distance_provided",
        desc=f"{hotel_title} – distance value provided in the answer",
        parent=hotel_node,
        critical=True
    )

    distance_leaf = evaluator.add_leaf(
        id=f"{hotel_id}_distance",
        desc="Distance from Port Everglades Terminal 4 is supported by the referenced page",
        parent=hotel_node,
        critical=True
    )
    distance_claim = (
        f"The hotel's page indicates it is approximately '{hotel.distance_to_terminal_4}' "
        f"from Port Everglades cruise terminal (Terminal 4)."
    )
    await evaluator.verify(
        claim=distance_claim,
        node=distance_leaf,
        sources=ref_url,
        additional_instruction=(
            "Confirm that the page states the distance to Port Everglades (or Fort Lauderdale cruise port). "
            "Terminal 4 is within Port Everglades; consider statements about distance to 'Port Everglades' or "
            "'cruise port' as acceptable. Allow reasonable approximation (e.g., rounding)."
        ),
    )

    # ---------- Shuttle availability verification ----------
    shuttle_leaf = evaluator.add_leaf(
        id=f"{hotel_id}_shuttle",
        desc="Shuttle service to the Port Everglades cruise terminal is available",
        parent=hotel_node,
        critical=True
    )
    shuttle_claim = (
        "The hotel's package includes shuttle service to Port Everglades (Fort Lauderdale cruise terminal)."
    )
    await evaluator.verify(
        claim=shuttle_claim,
        node=shuttle_leaf,
        sources=ref_url,
        additional_instruction=(
            "Verify that the package mentions a shuttle to the cruise port (Port Everglades). "
            "Accept equivalent phrasing such as 'shuttle to cruise terminal' or 'transportation to port'. "
            "It may be complimentary or paid; availability is the key."
        ),
    )

    # ---------- Parking duration verification ----------
    parking_leaf = evaluator.add_leaf(
        id=f"{hotel_id}_parking_duration",
        desc="Parking included for a minimum of 7 days is supported",
        parent=hotel_node,
        critical=True
    )
    parking_claim = (
        "The hotel's park-sleep-cruise (or park & cruise) package includes parking for at least 7 days."
    )
    await evaluator.verify(
        claim=parking_claim,
        node=parking_leaf,
        sources=ref_url,
        additional_instruction=(
            "Check the page for included parking duration. Consider the requirement satisfied if the page "
            "explicitly states 7 days or more (e.g., 'up to 14 days', '8 days')."
        ),
    )

    # ---------- Price verification ----------
    price_provided = bool(hotel.package_price_1n_7d and hotel.package_price_1n_7d.strip())
    evaluator.add_custom_node(
        result=price_provided,
        id=f"{hotel_id}_price_provided",
        desc=f"{hotel_title} – package price value provided in the answer",
        parent=hotel_node,
        critical=True
    )

    price_leaf = evaluator.add_leaf(
        id=f"{hotel_id}_price",
        desc="Total package price for 1-night stay with 7-day parking is supported",
        parent=hotel_node,
        critical=True
    )
    price_claim = (
        f"The total package price for one night's stay with 7-day parking included is '{hotel.package_price_1n_7d}'."
    )
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=ref_url,
        additional_instruction=(
            "Verify the stated price on the referenced page. Allow minor variations due to taxes/fees or date-dependent "
            "pricing; the figure should be explicitly present or clearly indicated as the package rate."
        ),
    )

    # ---------- Reference URL relevance verification ----------
    reference_leaf = evaluator.add_leaf(
        id=f"{hotel_id}_reference",
        desc="Reference URL confirms park-sleep-cruise / park & cruise package details",
        parent=hotel_node,
        critical=True
    )
    reference_claim = (
        f"This URL is a valid page describing a park-sleep-cruise or park & cruise package for '{hotel.name}' "
        f"in the Fort Lauderdale area."
    )
    await evaluator.verify(
        claim=reference_claim,
        node=reference_leaf,
        sources=ref_url,
        additional_instruction=(
            "Confirm the page is relevant: it should be about the hotel's park & cruise / stay, park & cruise / "
            "park-sleep-cruise package (or equivalent phrasing), and pertain to the Fort Lauderdale/Port Everglades area."
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
    Evaluate an answer for the Fort Lauderdale park-sleep-cruise hotel task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # hotels evaluated independently
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

    # Extract hotels from the answer
    extracted_hotels = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    # Record trip context
    evaluator.add_custom_info(
        {
            "departure_port": "Port Everglades Terminal 4 (Fort Lauderdale, FL)",
            "target_month_year": "July 2026",
            "stay_parking_requirement": "1-night stay + ≥7 days parking",
        },
        info_type="trip_context"
    )

    # Prepare the first three hotels (pad if fewer)
    hotels = extracted_hotels.hotels[:3]
    while len(hotels) < 3:
        hotels.append(HotelItem())

    # Add critical distinct hotels check under root
    names_clean = [h.name.strip() for h in hotels if h.name and h.name.strip()]
    distinct_ok = len(names_clean) == 3 and len(set(names_clean)) == 3
    evaluator.add_custom_node(
        result=distinct_ok,
        id="distinct_hotels",
        desc="All three hotels are different (no duplicates among the first three listed)",
        parent=root,
        critical=True
    )
    evaluator.add_custom_info(
        {"hotel_names_first_three": names_clean},
        info_type="extraction_summary",
        info_name="first_three_hotel_names"
    )

    # Build verification subtrees for each hotel
    for i, hotel in enumerate(hotels, start=1):
        await verify_hotel(evaluator, root, hotel, i - 1)

    # Return structured summary
    return evaluator.get_summary()