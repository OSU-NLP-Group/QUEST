import asyncio
import logging
import math
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "denver_xmas_retail_2025"
TASK_DESCRIPTION = (
    "I need to find three retail locations in Denver, Colorado that will be open for business on Christmas Day 2025 "
    "(December 25, 2025) for emergency shopping needs. Specifically, I need to locate:\n\n"
    "1. One Walgreens pharmacy store\n"
    "2. One CVS pharmacy store\n"
    "3. One 7-Eleven convenience store\n\n"
    "All three locations must be within 3 miles of downtown Denver (centered at coordinates 39.7392° N, 104.9903° W).\n\n"
    "For each of the three stores, please provide:\n"
    "- The complete street address (including street, city, state, and ZIP code)\n"
    "- The specific operating hours on Christmas Day 2025 (December 25, 2025)\n"
    "- A reference URL from an official source (such as the retailer's official website, store locator page, or "
    "corporate holiday hours announcement) that confirms the Christmas Day 2025 operating hours for that specific location"
)

DOWNTOWN_DENVER_LAT = 39.7392
DOWNTOWN_DENVER_LON = -104.9903
MAX_DISTANCE_MI = 3.0


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class StoreInfo(BaseModel):
    brand: Optional[str] = None
    address: Optional[str] = None
    christmas_hours: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    map_urls: List[str] = Field(default_factory=list)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: Optional[str] = None


class StoresExtraction(BaseModel):
    walgreens: Optional[StoreInfo] = None
    cvs: Optional[StoreInfo] = None
    seven_eleven: Optional[StoreInfo] = None


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def is_address_complete(addr: Optional[str]) -> bool:
    if not addr or not addr.strip():
        return False
    s = addr.strip()
    # Heuristics for a complete US street address in Denver, CO with ZIP
    has_number_street = bool(re.search(r"\b\d+\s+\S+", s))
    has_city_state = bool(re.search(r"Denver[, ]+CO\b", s, flags=re.IGNORECASE))
    has_zip = bool(re.search(r"\b\d{5}(?:-\d{4})?\b", s))
    return has_number_street and has_city_state and has_zip


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.7613  # Earth radius in miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def combine_sources(store: Optional[StoreInfo]) -> List[str]:
    urls: List[str] = []
    if not store:
        return urls
    for u in (store.reference_urls or []):
        if u and u.strip():
            urls.append(u.strip())
    for u in (store.map_urls or []):
        if u and u.strip():
            if u.strip() not in urls:
                urls.append(u.strip())
    return urls


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_stores() -> str:
    return """
    Extract structured information for exactly three Denver, CO locations mentioned in the answer text:
    - One Walgreens pharmacy store
    - One CVS pharmacy store
    - One 7-Eleven convenience store

    For each brand, extract the following fields ONLY if explicitly present in the answer text (do not infer):
    - brand: The retailer brand name ("Walgreens", "CVS", or "7-Eleven")
    - address: The complete street address string (should include street, Denver, CO, and ZIP if present)
    - christmas_hours: The specific operating hours on Christmas Day 2025 (the text should clearly associate the hours with Dec 25, 2025 or "Christmas Day")
    - reference_urls: A list of URL(s) that the answer cites as evidence for the Christmas Day 2025 hours for that specific location (e.g., official store page, store locator, corporate holiday page). Extract only actual URLs that appear in the answer.
    - map_urls: A list of URL(s) to mapping pages (e.g., Google Maps, Apple Maps, retailer's embedded map) for that specific location, if present in the answer.
    - latitude: If the answer explicitly contains numeric latitude for the location (including inside a URL like @lat,lon), extract it as a number; otherwise null.
    - longitude: If the answer explicitly contains numeric longitude for the location (including inside a URL like @lat,lon), extract it as a number; otherwise null.
    - notes: Any short clarifying note present in the answer about this store's Christmas hours (optional).

    Organize the result into three objects:
    - walgreens: info for the Walgreens store (or null if not present)
    - cvs: info for the CVS store (or null if not present)
    - seven_eleven: info for the 7-Eleven store (or null if not present)

    Rules:
    - Do not invent or infer any URL or hours. Only extract what appears in the answer text.
    - If multiple candidate locations per brand are listed, extract the FIRST one per brand.
    - If any field is missing for a brand, set it to null or an empty list appropriately.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_store(
    evaluator: Evaluator,
    parent_node,
    store: Optional[StoreInfo],
    store_key: str,
    brand_pretty: str,
) -> None:
    """
    Build verification subtree for a single store.
    """
    # Parent node for this store (non-critical, allows partial credit among brands)
    store_node = evaluator.add_parallel(
        id=f"{store_key}",
        desc=f"Identify one {brand_pretty} location in Denver that is open on Christmas Day 2025",
        parent=parent_node,
        critical=False
    )

    # 1) Address group (Critical): presence + completeness/format
    addr_group = evaluator.add_parallel(
        id=f"{store_key}_address_group",
        desc=f"Provide the complete street address of the {brand_pretty} location including street, city, state, and ZIP code",
        parent=store_node,
        critical=True
    )

    addr_present = evaluator.add_custom_node(
        result=bool(store and store.address and store.address.strip()),
        id=f"{store_key}_address_present",
        desc=f"{brand_pretty} address is provided",
        parent=addr_group,
        critical=True
    )

    addr_complete = evaluator.add_custom_node(
        result=is_address_complete(store.address if store else None),
        id=f"{store_key}_address_complete",
        desc=f"{brand_pretty} address appears complete (includes street, 'Denver, CO', and ZIP code)",
        parent=addr_group,
        critical=True
    )

    # 2) Hours group (Critical): hours string present + ref URL present + source supports hours
    hours_group = evaluator.add_parallel(
        id=f"{store_key}_christmas_hours_main",
        desc=f"Provide and verify specific operating hours on Christmas Day 2025 for this {brand_pretty} with an official reference URL",
        parent=store_node,
        critical=True
    )

    hours_provided = evaluator.add_custom_node(
        result=bool(store and store.christmas_hours and store.christmas_hours.strip()),
        id=f"{store_key}_hours_provided",
        desc=f"Christmas Day 2025 hours string is provided for {brand_pretty}",
        parent=hours_group,
        critical=True
    )

    ref_present = evaluator.add_custom_node(
        result=bool(store and store.reference_urls and len(store.reference_urls) > 0),
        id=f"{store_key}_ref_url_present",
        desc=f"At least one reference URL is provided that purportedly confirms Christmas Day 2025 hours for this {brand_pretty}",
        parent=hours_group,
        critical=True
    )

    hours_supported_leaf = evaluator.add_leaf(
        id=f"{store_key}_hours_supported_by_url",
        desc=f"Christmas Day 2025 hours for this {brand_pretty} are supported by the cited URL(s)",
        parent=hours_group,
        critical=True
    )
    all_urls = combine_sources(store)
    address_str = (store.address or "").strip() if store else ""
    hours_str = (store.christmas_hours or "").strip() if store else ""
    hours_claim = (
        f"For the {brand_pretty} located at {address_str}, the operating hours on Christmas Day 2025 "
        f"(December 25, 2025) are '{hours_str}' or an equivalent phrasing indicating the same hours "
        f"(e.g., 'Open 24 hours' if appropriate)."
    )
    await evaluator.verify(
        claim=hours_claim,
        node=hours_supported_leaf,
        sources=all_urls,  # If empty, verify() will fall back; but critical sibling ref_present gates this node
        additional_instruction=(
            "Use only the provided URL(s) to check whether this specific location is open on Dec 25, 2025 and "
            "that the hours match. If the page only states generic information like 'hours may vary' or suggests "
            "calling the store without confirming hours for Dec 25, treat as not supported. "
            "Allow minor formatting differences, such as '8:00 AM–5:00 PM' vs '8am-5pm', or 'Open 24 hours'."
        ),
    )

    # 3) Geographic constraint (Critical): within 3 miles of downtown Denver center
    # Prefer a coordinate-based calculation if lat/lon are provided; otherwise verify via cited URLs (maps/store pages).
    if store and store.latitude is not None and store.longitude is not None:
        distance_mi = haversine_miles(DOWNTOWN_DENVER_LAT, DOWNTOWN_DENVER_LON, store.latitude, store.longitude)
        within_range = distance_mi <= MAX_DISTANCE_MI + 1e-6
        evaluator.add_custom_node(
            result=within_range,
            id=f"{store_key}_within_geo_range",
            desc=f"Verify this {brand_pretty} location is within 3 miles of downtown Denver "
                 f"(computed distance ≈ {distance_mi:.2f} mi)",
            parent=store_node,
            critical=True
        )
    else:
        geo_leaf = evaluator.add_leaf(
            id=f"{store_key}_within_geo_range",
            desc=f"Verify this {brand_pretty} location is within 3 miles of downtown Denver (39.7392, -104.9903)",
            parent=store_node,
            critical=True
        )
        await evaluator.verify(
            claim=(
                f"The {brand_pretty} at address '{address_str}' is within 3 miles (straight-line or equivalent) "
                f"of downtown Denver's center at coordinates 39.7392, -104.9903."
            ),
            node=geo_leaf,
            sources=all_urls,  # Prefer map/store locator URLs if present
            additional_instruction=(
                "Use any provided map URL(s) or official locator pages to judge proximity. "
                "If a driving distance is provided and is ≤3 miles, accept it. "
                "If no credible evidence on the page allows estimating the distance, treat as not supported."
            ),
            extra_prerequisites=[addr_present]  # Skip if no address
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
    Evaluate an answer for the Denver Christmas Day 2025 emergency shopping task.
    """
    # Initialize evaluator and root node
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

    # Extract structured info from the provided answer
    extracted: StoresExtraction = await evaluator.extract(
        prompt=prompt_extract_stores(),
        template_class=StoresExtraction,
        extraction_name="stores_extraction",
    )

    # Record ground-truth constraints/context (not used for scoring, only for transparency)
    evaluator.add_ground_truth({
        "required_brands": ["Walgreens", "CVS", "7-Eleven"],
        "downtown_center_coords": {"lat": DOWNTOWN_DENVER_LAT, "lon": DOWNTOWN_DENVER_LON},
        "max_distance_miles": MAX_DISTANCE_MI,
        "target_holiday": "Christmas Day 2025 (December 25, 2025)"
    }, gt_type="task_constraints")

    # Build three store verification subtrees (parallel, partial credit allowed across brands)
    await verify_single_store(
        evaluator=evaluator,
        parent_node=root,
        store=extracted.walgreens,
        store_key="walgreens_store",
        brand_pretty="Walgreens pharmacy store"
    )
    await verify_single_store(
        evaluator=evaluator,
        parent_node=root,
        store=extracted.cvs,
        store_key="cvs_store",
        brand_pretty="CVS pharmacy store"
    )
    await verify_single_store(
        evaluator=evaluator,
        parent_node=root,
        store=extracted.seven_eleven,
        store_key="7eleven_store",
        brand_pretty="7-Eleven convenience store"
    )

    # Return standardized evaluation summary
    return evaluator.get_summary()