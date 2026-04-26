import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "heb_sa_store_2025"
TASK_DESCRIPTION = (
    "Identify the H-E-B grocery store in San Antonio, Texas that opened on January 7, 2025. "
    "This store is approximately 120,000 square feet in size and features a True Texas BBQ restaurant with drive-thru service, "
    "a full-service H-E-B Pharmacy with drive-thru service, an on-site fuel center, and a Texas Backyard area for outdoor and garden products. "
    "The store has over 600 parking spaces and offers both curbside pickup and delivery services. Provide the store's complete street address."
)


class AddressInfo(BaseModel):
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None


class StoreExtraction(BaseModel):
    store_name: Optional[str] = None
    address: Optional[AddressInfo] = None
    opening_date_text: Optional[str] = None
    size_text: Optional[str] = None
    parking_spaces_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


def prompt_extract_store_info() -> str:
    return """
    Extract the specific H-E-B store details as presented in the answer. We are evaluating whether the answer correctly identifies
    the San Antonio, Texas H-E-B store that opened on January 7, 2025 and provides its complete address and supporting sources.

    Return a single JSON object with these fields:
    - store_name: The specific store name/identifier mentioned (e.g., "H-E-B [Area]" or "H-E-B at [Street]"), or null if not specified.
    - address: An object with:
        - street_address: The full street address string (must include street number and street name), or null if missing.
        - city: The city, or null if missing.
        - state: The state (e.g., "Texas" or "TX"), or null if missing.
        - zip_code: The ZIP code, or null if missing.
    - opening_date_text: The opening date as a string exactly as written in the answer (e.g., "January 7, 2025"), or null if missing.
    - size_text: The store size as text (e.g., "approximately 120,000 square feet"), or null if missing.
    - parking_spaces_text: The parking spaces information as text (e.g., "over 600 parking spaces"), or null if missing.
    - source_urls: An array of all URLs explicitly cited in the answer as sources for this store (include any press releases, store pages, or news articles).
      If a URL is missing a protocol, prepend http://. If no URLs are provided, return an empty array.

    Do not fabricate any information. Extract only what is explicitly stated in the answer.
    """


def _is_complete_street_address(addr: Optional[AddressInfo]) -> bool:
    if not addr:
        return False
    if not (addr.street_address and addr.city and addr.state and addr.zip_code):
        return False
    street = addr.street_address.strip()
    city = addr.city.strip()
    state = addr.state.strip()
    zipc = addr.zip_code.strip()
    if not street or not city or not state or not zipc:
        return False
    # Heuristic: require at least one digit in street_address to indicate a street number
    has_number = any(ch.isdigit() for ch in street)
    return has_number


async def build_and_verify_store_nodes(
    evaluator: Evaluator,
    parent_node,
    extracted: StoreExtraction,
) -> None:
    # Create critical parent node for store identification and all checks
    store_node = evaluator.add_parallel(
        id="Store_Identification",
        desc="Evaluate whether the identified H-E-B store meets all specified criteria and the response provides the required address",
        parent=parent_node,
        critical=True,
    )

    # Street address provided (existence and completeness check)
    address_provided_node = evaluator.add_custom_node(
        result=_is_complete_street_address(extracted.address),
        id="Street_Address_Provided",
        desc="The response provides the store’s complete street address (street number + street name, city, state, and ZIP code)",
        parent=store_node,
        critical=True,
    )

    # Prepare sources (may be empty -> will fall back to simple verification)
    sources_list = extracted.source_urls or []

    # Create leaf nodes for each criterion
    location_node = evaluator.add_leaf(
        id="Location",
        desc="The store is located in San Antonio, Texas",
        parent=store_node,
        critical=True,
    )
    opening_date_node = evaluator.add_leaf(
        id="Opening_Date",
        desc="The store opened on January 7, 2025",
        parent=store_node,
        critical=True,
    )
    size_node = evaluator.add_leaf(
        id="Store_Size",
        desc="The store is approximately 120,000 square feet in size",
        parent=store_node,
        critical=True,
    )
    bbq_node = evaluator.add_leaf(
        id="True_Texas_BBQ",
        desc="The store features a True Texas BBQ restaurant with drive-thru service",
        parent=store_node,
        critical=True,
    )
    pharmacy_node = evaluator.add_leaf(
        id="Pharmacy_Service",
        desc="The store has a full-service H-E-B Pharmacy with drive-thru service",
        parent=store_node,
        critical=True,
    )
    fuel_node = evaluator.add_leaf(
        id="Fuel_Center",
        desc="The store has an on-site fuel center/gas station",
        parent=store_node,
        critical=True,
    )
    backyard_node = evaluator.add_leaf(
        id="Texas_Backyard",
        desc="The store has a Texas Backyard area for outdoor and garden products",
        parent=store_node,
        critical=True,
    )
    parking_node = evaluator.add_leaf(
        id="Parking_Spaces",
        desc="The store has over 600 parking spaces",
        parent=store_node,
        critical=True,
    )
    curbside_delivery_node = evaluator.add_leaf(
        id="Curbside_and_Delivery",
        desc="The store offers both curbside pickup and delivery services",
        parent=store_node,
        critical=True,
    )

    # Build claims
    claims_and_sources = [
        (
            "The identified H-E-B store is located in San Antonio, Texas (TX).",
            sources_list,
            location_node,
            "Verify city and state for the specific store referenced by the provided source(s). Allow 'TX' as equivalent to 'Texas'.",
        ),
        (
            "The store opened on January 7, 2025.",
            sources_list,
            opening_date_node,
            "Check the opening or grand opening date on the page. Accept reasonable phrasing variants like 'opens Jan. 7, 2025' or 'grand opening January 7, 2025'.",
        ),
        (
            "The store is approximately 120,000 square feet in size.",
            sources_list,
            size_node,
            "Confirm the store size is stated as approximately or around 120,000 sq ft. Allow minor numerical variants if explicitly described as approximate.",
        ),
        (
            "The store features a True Texas BBQ restaurant with drive-thru service.",
            sources_list,
            bbq_node,
            "Look for 'True Texas BBQ' at this store and whether 'drive-thru' is available. Allow 'drive through' spelling variants.",
        ),
        (
            "The store has a full-service H-E-B Pharmacy with drive-thru service.",
            sources_list,
            pharmacy_node,
            "Confirm a full-service H-E-B Pharmacy exists at the store and that a drive-thru is available.",
        ),
        (
            "The store has an on-site fuel center (gas station).",
            sources_list,
            fuel_node,
            "Verify the presence of an on-site fuel center or gas station associated with this store.",
        ),
        (
            "The store has a Texas Backyard area for outdoor and garden products.",
            sources_list,
            backyard_node,
            "Confirm the presence of 'Texas Backyard' or a clearly equivalent outdoor and garden products area at this store.",
        ),
        (
            "The store has over 600 parking spaces.",
            sources_list,
            parking_node,
            "Check that the store's parking capacity is stated as over 600 spaces. Allow equivalent phrasing like 'more than 600'.",
        ),
        (
            "The store offers both curbside pickup and delivery services.",
            sources_list,
            curbside_delivery_node,
            "Confirm both services are available. Accept equivalent phrasing such as 'Curbside' and 'Home Delivery'.",
        ),
    ]

    # Run verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_store_info(),
        template_class=StoreExtraction,
        extraction_name="store_extraction",
    )

    # Record some helpful custom info
    addr = extracted.address or AddressInfo()
    evaluator.add_custom_info(
        {
            "store_name": extracted.store_name,
            "street_address": addr.street_address,
            "city": addr.city,
            "state": addr.state,
            "zip_code": addr.zip_code,
            "opening_date_text": extracted.opening_date_text,
            "size_text": extracted.size_text,
            "parking_spaces_text": extracted.parking_spaces_text,
            "source_urls_count": len(extracted.source_urls or []),
        },
        info_type="extraction_summary",
    )

    await build_and_verify_store_nodes(evaluator, root, extracted)

    return evaluator.get_summary()