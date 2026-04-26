import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chi_business_hotels_loop"
TASK_DESCRIPTION = """
I am organizing a corporate business meeting in Chicago for 50 attendees, some of whom will be traveling with service dogs. I need to identify 4 suitable hotels in downtown Chicago that can accommodate our needs. Each hotel must meet ALL of the following requirements:

1. Be located in downtown Chicago's business district (The Loop area or nearby)
2. Belong to one of the major hotel chains: Marriott International, Hilton, Hyatt, or IHG (InterContinental Hotels Group)
3. Have on-site meeting or conference facilities that can accommodate at least 40 people
4. Provide a business center with computers, printing capabilities, and internet access
5. Offer on-site dining options (either a full-service restaurant or complimentary breakfast service)
6. Have an on-site fitness center or gym
7. Provide parking facilities (either on-site parking garage or valet parking service)
8. Offer 24-hour front desk service
9. Allow dogs as pets (with standard pet policies)
10. Have ADA-compliant accessible rooms available
11. Have a swimming pool (indoor or outdoor)
12. Provide complimentary high-speed WiFi throughout the property

For each of the 4 hotels, provide the following information:
- Hotel name
- Specific address
- Chain affiliation (Marriott, Hilton, Hyatt, or IHG brand)
- Official website URL or booking page URL where the hotel's amenities and policies can be verified
- Brief confirmation that each of the 12 requirements listed above is met, with reference URLs supporting each major amenity or policy
"""

ALLOWED_CHAINS = ["Marriott", "Hilton", "Hyatt", "IHG"]  # Accept brand-level (e.g., Hyatt Regency under Hyatt)
DOWNTOWN_NEIGHBORHOODS = [
    "The Loop", "Loop", "River North", "Streeterville", "West Loop",
    "Near North Side", "Magnificent Mile", "South Loop"
]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelInfo(BaseModel):
    # Required identifying info
    name: Optional[str] = None
    address: Optional[str] = None
    chain_affiliation: Optional[str] = None  # Accept brand name if clearly under one of the four chains
    official_url: Optional[str] = None

    # Requirement-specific supporting URLs (as explicitly provided in the answer)
    urls_location: List[str] = Field(default_factory=list)
    urls_chain: List[str] = Field(default_factory=list)
    urls_meeting: List[str] = Field(default_factory=list)
    urls_business_center: List[str] = Field(default_factory=list)
    urls_dining: List[str] = Field(default_factory=list)
    urls_fitness: List[str] = Field(default_factory=list)
    urls_parking: List[str] = Field(default_factory=list)
    urls_front_desk: List[str] = Field(default_factory=list)
    urls_pet_policy: List[str] = Field(default_factory=list)
    urls_ada: List[str] = Field(default_factory=list)
    urls_pool: List[str] = Field(default_factory=list)
    urls_wifi: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    hotels: List[HotelInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract all hotels mentioned in the answer that are proposed for the corporate meeting in Chicago.
    For each hotel, extract exactly the following fields:
    1) name: The hotel's name (string). If missing, null.
    2) address: The specific street address (string). If missing, null.
    3) chain_affiliation: The hotel brand or chain (string), as explicitly stated in the answer (e.g., 'Hyatt Regency', 'Hilton', 'Marriott', 'IHG' brand). If missing, null.
    4) official_url: The official hotel website URL or booking page URL suitable for verifying amenities/policies. If missing, null.

    Also extract the supporting URLs explicitly provided for each of the 12 requirements. Only include actual URLs explicitly present in the answer (plain URLs or markdown links). If the answer does not provide any URL for a requirement, return an empty list for that field.

    Requirement-specific URL arrays:
    - urls_location: URLs supporting location downtown (Loop or nearby)
    - urls_chain: URLs supporting chain affiliation (Marriott/Hilton/Hyatt/IHG)
    - urls_meeting: URLs supporting on-site meeting/conference facilities that can accommodate at least 40 people
    - urls_business_center: URLs supporting business center with computers, printing, and internet
    - urls_dining: URLs supporting on-site dining (full-service restaurant OR complimentary breakfast)
    - urls_fitness: URLs supporting fitness center/gym
    - urls_parking: URLs supporting parking facilities (on-site garage OR valet)
    - urls_front_desk: URLs supporting 24-hour front desk service
    - urls_pet_policy: URLs supporting dog-friendly pet policy
    - urls_ada: URLs supporting ADA-compliant accessible rooms
    - urls_pool: URLs supporting indoor/outdoor swimming pool
    - urls_wifi: URLs supporting complimentary high-speed WiFi throughout

    Return a JSON object with a single field:
    { "hotels": [ HotelInfo, HotelInfo, ... ] }
    where HotelInfo matches the schema described above.

    IMPORTANT:
    - Extract only information explicitly present in the answer text.
    - If a field is missing, use null for strings and [] for URL arrays.
    - Include all hotels mentioned by the answer (do not limit to 4 in extraction).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def gather_sources(h: HotelInfo, url_field: str) -> List[str]:
    """Gather requirement-specific URLs; if none, fall back to official_url."""
    urls = getattr(h, url_field) or []
    # Deduplicate while preserving order
    if h.official_url:
        if not urls:
            urls = [h.official_url]
        elif h.official_url not in urls:
            urls.append(h.official_url)
    return urls

def chain_allowed(chain_str: Optional[str]) -> bool:
    if not chain_str:
        return False
    s = chain_str.lower()
    # Accept brand-level under the four chains
    return any(c.lower() in s for c in ALLOWED_CHAINS)

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelInfo,
    idx: int,
) -> None:
    """
    Build verification sub-tree for a single hotel and run verifications.
    """

    # Top-level node for this hotel (non-critical to allow partial scoring per hotel)
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{idx+1}",
        desc=f"Hotel #{idx+1} details and requirement verification",
        parent=parent_node,
        critical=False
    )

    # 1) Required identifying information (critical, parallel)
    required_node = evaluator.add_parallel(
        id=f"hotel_{idx+1}_required_details",
        desc=f"Provides the required identifying information for Hotel #{idx+1}",
        parent=hotel_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hotel.name and hotel.name.strip()),
        id=f"hotel_{idx+1}_name",
        desc="Hotel name is provided.",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel.address and hotel.address.strip()),
        id=f"hotel_{idx+1}_address",
        desc="Specific street address is provided.",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel.chain_affiliation and hotel.chain_affiliation.strip()),
        id=f"hotel_{idx+1}_chain_affiliation_field",
        desc="Chain affiliation is explicitly stated as Marriott, Hilton, Hyatt, or IHG (brand acceptable if clearly under one).",
        parent=required_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel.official_url and hotel.official_url.strip()),
        id=f"hotel_{idx+1}_official_website_url",
        desc="Provides an official hotel website URL or booking page URL suitable for verifying amenities/policies.",
        parent=required_node,
        critical=True
    )

    # 2) Meets all 12 requirements (critical, parallel)
    reqs_node = evaluator.add_parallel(
        id=f"hotel_{idx+1}_meets_all_12",
        desc="Briefly confirms each of the 12 requirements is met and provides supporting reference URL(s) for each major amenity/policy.",
        parent=hotel_node,
        critical=True
    )

    # Create leaf nodes for each requirement, then batch verify for parallel execution
    claims_and_nodes: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # 2.1 Location - Downtown Loop or nearby
    loc_node = evaluator.add_leaf(
        id=f"hotel_{idx+1}_location_downtown_loop",
        desc="Hotel is in downtown Chicago business district (The Loop area or nearby) and includes a supporting URL or other verifiable reference.",
        parent=reqs_node,
        critical=True
    )
    loc_sources = gather_sources(hotel, "urls_location")
    loc_claim = (
        f"The hotel '{hotel.name or ''}' is located in downtown Chicago's business district "
        f"(The Loop area or nearby). The address '{hotel.address or ''}' and/or official page should "
        f"support this (e.g., Loop, River North, Streeterville, West Loop, Near North Side, or Magnificent Mile)."
    )
    loc_ins = (
        "Downtown Chicago includes The Loop and adjacent neighborhoods like River North, Streeterville, West Loop, "
        "Near North Side, and the Magnificent Mile. Consider explicit neighborhood mentions or well-known downtown "
        "landmarks on the official page as evidence."
    )
    claims_and_nodes.append((loc_claim, loc_sources, loc_node, loc_ins))

    # 2.2 Chain constraint
    chain_node = evaluator.add_leaf(
        id=f"hotel_{idx+1}_chain_constraint",
        desc="Hotel belongs to Marriott International, Hilton, Hyatt, or IHG, with a supporting URL/reference.",
        parent=reqs_node,
        critical=True
    )
    chain_sources = gather_sources(hotel, "urls_chain")
    chain_claim = (
        f"The hotel '{hotel.name or ''}' is affiliated with '{hotel.chain_affiliation or ''}', "
        f"which is under Marriott International, Hilton, Hyatt, or IHG."
    )
    chain_ins = (
        "Brand-level evidence is acceptable (e.g., Hyatt Regency implies Hyatt; Hilton Garden Inn implies Hilton; "
        "Marriott Marquis implies Marriott; InterContinental or Holiday Inn implies IHG). Confirm with the official page."
    )
    claims_and_nodes.append((chain_claim, chain_sources, chain_node, chain_ins))

    # 2.3 Meeting facilities >= 40 ppl
    meet_node = evaluator.add_leaf(
        id=f"hotel_{idx+1}_meeting_40ppl",
        desc="Hotel has on-site meeting/conference facilities that can accommodate at least 40 people, with supporting URL/reference.",
        parent=reqs_node,
        critical=True
    )
    meet_sources = gather_sources(hotel, "urls_meeting")
    meet_claim = (
        "The hotel provides on-site meeting or conference facilities that can accommodate at least 40 people."
    )
    meet_ins = (
        "Look for event/meetings pages that explicitly list capacities ≥ 40 for a room or combined spaces."
    )
    claims_and_nodes.append((meet_claim, meet_sources, meet_node, meet_ins))

    # 2.4 Business center with computers, printing, internet
    bc_node = evaluator.add_leaf(
        id=f"hotel_{idx+1}_business_center",
        desc="Hotel provides a business center with computers, printing capabilities, and internet access, with supporting URL/reference.",
        parent=reqs_node,
        critical=True
    )
    bc_sources = gather_sources(hotel, "urls_business_center")
    bc_claim = (
        "The hotel offers a business center that provides computers, printing capabilities, and internet access."
    )
    bc_ins = (
        "Accept explicit 'business center' amenities mentioning computers/workstations, printing, and internet/Wi-Fi access."
    )
    claims_and_nodes.append((bc_claim, bc_sources, bc_node, bc_ins))

    # 2.5 On-site dining options
    dining_node = evaluator.add_leaf(
        id=f"hotel_{idx+1}_onsite_dining",
        desc="Hotel offers on-site dining options (full-service restaurant OR complimentary breakfast service), with supporting URL/reference.",
        parent=reqs_node,
        critical=True
    )
    dining_sources = gather_sources(hotel, "urls_dining")
    dining_claim = (
        "The hotel offers on-site dining options, either a full-service restaurant or complimentary breakfast service."
    )
    dining_ins = (
        "On-site dining may include on-property restaurant(s), bar/café, or clearly stated complimentary breakfast."
    )
    claims_and_nodes.append((dining_claim, dining_sources, dining_node, dining_ins))

    # 2.6 Fitness center/gym
    fit_node = evaluator.add_leaf(
        id=f"hotel_{idx+1}_fitness_center",
        desc="Hotel has an on-site fitness center/gym, with supporting URL/reference.",
        parent=reqs_node,
        critical=True
    )
    fit_sources = gather_sources(hotel, "urls_fitness")
    fit_claim = "The hotel has an on-site fitness center or gym."
    fit_ins = "Check amenities pages for 'fitness center', 'gym', or similar wording."
    claims_and_nodes.append((fit_claim, fit_sources, fit_node, fit_ins))

    # 2.7 Parking facilities
    park_node = evaluator.add_leaf(
        id=f"hotel_{idx+1}_parking",
        desc="Hotel provides parking facilities (on-site garage OR valet parking), with supporting URL/reference.",
        parent=reqs_node,
        critical=True
    )
    park_sources = gather_sources(hotel, "urls_parking")
    park_claim = "The hotel provides parking facilities—either an on-site parking garage or valet parking service."
    park_ins = "Look for 'parking', 'valet', or 'garage' details on the official page."
    claims_and_nodes.append((park_claim, park_sources, park_node, park_ins))

    # 2.8 24-hour front desk
    fd_node = evaluator.add_leaf(
        id=f"hotel_{idx+1}_front_desk_24hr",
        desc="Hotel offers 24-hour front desk service, with supporting URL/reference.",
        parent=reqs_node,
        critical=True
    )
    fd_sources = gather_sources(hotel, "urls_front_desk")
    fd_claim = "The hotel offers 24-hour front desk service."
    fd_ins = "Confirm front desk hours or '24-hour front desk' statement on amenities/policies."
    claims_and_nodes.append((fd_claim, fd_sources, fd_node, fd_ins))

    # 2.9 Allows dogs (pet policy)
    pet_node = evaluator.add_leaf(
        id=f"hotel_{idx+1}_allows_dogs",
        desc="Hotel allows dogs as pets (standard pet policy), with supporting URL/reference.",
        parent=reqs_node,
        critical=True
    )
    pet_sources = gather_sources(hotel, "urls_pet_policy")
    pet_claim = "The hotel allows dogs as pets according to its standard pet policy."
    pet_ins = "Look for 'pet-friendly', 'dogs allowed', or detailed pet policy on the official page."
    claims_and_nodes.append((pet_claim, pet_sources, pet_node, pet_ins))

    # 2.10 ADA accessible rooms
    ada_node = evaluator.add_leaf(
        id=f"hotel_{idx+1}_ada_accessible_rooms",
        desc="Hotel has ADA-compliant accessible rooms available, with supporting URL/reference.",
        parent=reqs_node,
        critical=True
    )
    ada_sources = gather_sources(hotel, "urls_ada")
    ada_claim = "The hotel has ADA-compliant accessible rooms available."
    ada_ins = "Check accessibility statements, ADA compliance, or 'accessible rooms' details on the official page."
    claims_and_nodes.append((ada_claim, ada_sources, ada_node, ada_ins))

    # 2.11 Swimming pool
    pool_node = evaluator.add_leaf(
        id=f"hotel_{idx+1}_swimming_pool",
        desc="Hotel has an indoor or outdoor swimming pool, with supporting URL/reference.",
        parent=reqs_node,
        critical=True
    )
    pool_sources = gather_sources(hotel, "urls_pool")
    pool_claim = "The hotel has a swimming pool (indoor or outdoor)."
    pool_ins = "Look for amenities pages indicating 'pool' along with details."
    claims_and_nodes.append((pool_claim, pool_sources, pool_node, pool_ins))

    # 2.12 Complimentary high-speed WiFi
    wifi_node = evaluator.add_leaf(
        id=f"hotel_{idx+1}_complimentary_wifi",
        desc="Hotel provides complimentary high-speed WiFi throughout the property, with supporting URL/reference.",
        parent=reqs_node,
        critical=True
    )
    wifi_sources = gather_sources(hotel, "urls_wifi")
    wifi_claim = "The hotel provides complimentary high-speed WiFi throughout the property."
    wifi_ins = "Look for 'complimentary WiFi', 'free high-speed internet', or similar wording on amenities/policies."
    claims_and_nodes.append((wifi_claim, wifi_sources, wifi_node, wifi_ins))

    # Batch verify the 12 requirement claims (parallel)
    await evaluator.batch_verify(claims_and_nodes)


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
    Evaluate an answer for the Chicago downtown business hotels task.
    """
    # Initialize evaluator (root node is non-critical to allow partial credit aggregation)
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

    # Extract hotel info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="extracted_hotels"
    )

    # Record ground-truth criteria for transparency
    evaluator.add_ground_truth({
        "required_count": 4,
        "allowed_chains": ALLOWED_CHAINS,
        "downtown_chicago_neighborhoods": DOWNTOWN_NEIGHBORHOODS,
        "requirements": [
            "Downtown Loop/nearby",
            "Chain: Marriott/Hilton/Hyatt/IHG",
            "Meeting facilities >= 40 ppl",
            "Business center with computers/printing/internet",
            "On-site dining or complimentary breakfast",
            "Fitness center/gym",
            "Parking (garage or valet)",
            "24-hour front desk",
            "Dogs allowed (pet policy)",
            "ADA accessible rooms",
            "Swimming pool",
            "Complimentary high-speed WiFi"
        ]
    })

    # Count and uniqueness check (critical)
    all_hotels = extraction.hotels or []
    non_empty_hotels = [h for h in all_hotels if h.name and h.name.strip()]
    names_normalized = [normalize_name(h.name) for h in non_empty_hotels]
    unique_names = set(names_normalized)
    count_unique_ok = (len(non_empty_hotels) == 4 and len(unique_names) == 4)

    evaluator.add_custom_info(
        info={
            "total_hotels_mentioned": len(all_hotels),
            "non_empty_hotels": len(non_empty_hotels),
            "unique_names_count": len(unique_names),
            "unique_names": list(unique_names),
        },
        info_type="stats",
        info_name="hotel_count_uniqueness_stats"
    )

    evaluator.add_custom_node(
        result=count_unique_ok,
        id="hotel_count_and_uniqueness",
        desc="Response provides exactly 4 distinct hotels (no duplicates).",
        parent=root,
        critical=True
    )

    # Choose up to 4 hotels to evaluate (first 4 unique by normalized name)
    chosen: List[HotelInfo] = []
    seen: set = set()
    for h in non_empty_hotels:
        key = normalize_name(h.name)
        if key and key not in seen:
            chosen.append(h)
            seen.add(key)
        if len(chosen) == 4:
            break

    # Pad to exactly 4 for consistent tree shape (placeholders will likely fail critical checks)
    while len(chosen) < 4:
        chosen.append(HotelInfo())

    # Build verification subtrees for each chosen hotel
    hotel_tasks = []
    for i, hotel in enumerate(chosen, start=1):
        hotel_tasks.append(verify_hotel(evaluator, root, hotel, i - 1))
    await asyncio.gather(*hotel_tasks)

    # Return structured summary
    return evaluator.get_summary()