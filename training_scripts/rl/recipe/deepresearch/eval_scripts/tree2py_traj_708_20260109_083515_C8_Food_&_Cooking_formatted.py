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
TASK_ID = "us_michelin_green_stars"
TASK_DESCRIPTION = """
Identify three restaurants in the United States that have earned both a Michelin star for culinary excellence and a Michelin Green Star for their commitment to sustainability. Each restaurant must be located in a different US state. For each restaurant, provide:

1. The full restaurant name
2. The complete physical address (including street address, city, state, and ZIP code)
3. A reference URL from the Michelin Guide or the restaurant's official website
4. The name of the head chef or executive chef
5. The chef's formal culinary training background, including the name of the culinary institution(s) where they trained, or if self-taught, documentation of their apprenticeship or professional culinary development
6. A reference URL confirming the chef's credentials and training background
7. At least two specific sustainability practices employed by the restaurant (such as on-site farming, composting programs, local sourcing partnerships, renewable energy use, zero-waste initiatives, or regenerative agriculture), with verifiable details

All information must be traceable to authoritative sources such as the Michelin Guide, official restaurant websites, or verified news articles.
""".strip()


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #

# Basic mappings for US states (full name -> 2-letter code) and validation helpers
US_STATE_TO_ABBR = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY", "DISTRICT OF COLUMBIA": "DC", "WASHINGTON DC": "DC",
    "WASHINGTON D.C.": "DC", "D.C.": "DC", "DC": "DC",
}
US_ABBRS = set(US_STATE_TO_ABBR.values())


def normalize_state(state_str: Optional[str]) -> Optional[str]:
    if not state_str:
        return None
    s = state_str.strip().upper()
    # Direct match for 2-letter abbr
    if s in US_ABBRS:
        return s
    # Try map from full name
    return US_STATE_TO_ABBR.get(s, None)


def safe_str(x: Optional[str]) -> str:
    return (x or "").strip()


def choose_award_sources(restaurant: "Restaurant") -> List[str]:
    # Prefer explicit award URLs if given; else fallback to the main restaurant ref URL
    urls: List[str] = []
    if restaurant.award_urls:
        urls.extend([u for u in restaurant.award_urls if safe_str(u)])
    if safe_str(restaurant.restaurant_ref_url):
        urls.append(restaurant.restaurant_ref_url)  # include as fallback
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def practice_sources(pr: "SustainabilityPractice", restaurant: "Restaurant") -> List[str]:
    urls: List[str] = []
    if pr and pr.urls:
        urls.extend([u for u in pr.urls if safe_str(u)])
    if not urls:
        urls = choose_award_sources(restaurant)
    # Deduplicate
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SustainabilityPractice(BaseModel):
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ChefInfo(BaseModel):
    name: Optional[str] = None
    training_background: Optional[str] = None  # e.g., institution names or "self-taught/apprenticeship"
    institutions: List[str] = Field(default_factory=list)
    ref_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class Restaurant(BaseModel):
    name: Optional[str] = None

    # Address components
    address_street: Optional[str] = None
    address_city: Optional[str] = None
    address_state: Optional[str] = None
    address_zip: Optional[str] = None
    address_full: Optional[str] = None

    # Primary reference URL (Michelin Guide or official site)
    restaurant_ref_url: Optional[str] = None

    # Award-related URLs (e.g., Michelin Guide page confirming star/green star)
    award_urls: List[str] = Field(default_factory=list)

    # Chef info
    chef: Optional[ChefInfo] = None

    # Sustainability practices (at least two expected)
    sustainability_practices: List[SustainabilityPractice] = Field(default_factory=list)


class RestaurantsExtraction(BaseModel):
    restaurants: List[Restaurant] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurants() -> str:
    return """
    Extract up to 5 restaurants from the answer that are stated to have BOTH:
    – at least one MICHELIN Star (culinary excellence)
    – a MICHELIN Green Star (sustainability)

    For each restaurant, extract the following structured fields:

    1) name: The full restaurant name as written in the answer.

    2) address_street: The street address line (e.g., "123 Main St").
    3) address_city: City.
    4) address_state: U.S. state (prefer 2-letter abbreviation if available).
    5) address_zip: ZIP code (e.g., "94118" or "10013-1234").
    6) address_full: The full postal address string if provided.

    7) restaurant_ref_url: A single primary reference URL that is either a Michelin Guide page or the restaurant's official website.

    8) award_urls: An array of URLs (if any) that confirm the MICHELIN Star and/or MICHELIN Green Star status (prefer Michelin Guide links).

    9) chef: An object with:
       - name: Head or Executive Chef name, exactly as in the answer.
       - training_background: A textual description of the chef's formal culinary training, institutions attended, or "self-taught" with apprenticeship/experience details if applicable.
       - institutions: An array listing the culinary schools/institutions if mentioned (each as strings).
       - ref_url: A URL that specifically confirms the chef's credentials/training.
       - additional_urls: Any extra URLs relevant to the chef's credentials.

    10) sustainability_practices: An array with at least two objects (if provided), each with:
        - description: A specific sustainability practice (e.g., "on-site farm", "composting program", "renewable energy", "zero-waste initiative", "regenerative agriculture", etc.).
        - urls: An array of URLs that support or describe this specific practice. If no dedicated URLs for practices are given, leave this array empty.

    IMPORTANT:
    - Do NOT invent any information. Only extract fields explicitly present in the answer.
    - For any missing field, set it to null (or empty array where appropriate).
    - Keep URLs exactly as written; if URLs are given in Markdown, extract the actual URL target.
    - Prefer 2-letter state abbreviations for 'address_state' when possible.
    - Maintain the order in which restaurants appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_restaurant(
    evaluator: Evaluator,
    parent_node,
    restaurant: Restaurant,
    index: int,
    previous_states: List[str],
):
    """
    Construct verification sub-tree for a single restaurant and run checks.
    index: 0-based (0 -> Restaurant_1, 1 -> Restaurant_2, 2 -> Restaurant_3)
    previous_states: list of normalized (2-letter) state codes from earlier restaurants
    """
    rid = index + 1

    # Parent node for this restaurant
    r_node = evaluator.add_parallel(
        id=f"Restaurant_{rid}",
        desc=("First restaurant meeting all criteria" if rid == 1 else
              ("Second restaurant meeting all criteria in a different state than Restaurant 1" if rid == 2 else
               "Third restaurant meeting all criteria in a different state than Restaurants 1 and 2")),
        parent=parent_node,
        critical=False
    )

    # Prepare sources
    award_srcs = choose_award_sources(restaurant)
    main_src = restaurant.restaurant_ref_url if safe_str(restaurant.restaurant_ref_url) else None

    # 1) Michelin Star presence (critical)
    star_leaf = evaluator.add_leaf(
        id=f"r{rid}_Michelin_Star",
        desc="Restaurant holds at least one Michelin star for culinary excellence",
        parent=r_node,
        critical=True
    )
    star_claim = (
        f"The provided page(s) confirm that the restaurant "
        f"{f'\"{restaurant.name}\" ' if safe_str(restaurant.name) else ''}"
        f"holds at least one MICHELIN Star."
    )
    await evaluator.verify(
        claim=star_claim,
        node=star_leaf,
        sources=award_srcs or main_src,
        additional_instruction=(
            "Look for explicit indicators such as 'One MICHELIN Star', 'Two MICHELIN Stars', "
            "'Three MICHELIN Stars', star icons, or equivalent language on the Michelin Guide page or "
            "another authoritative confirmation page."
        )
    )

    # 2) Michelin Green Star presence (critical)
    green_leaf = evaluator.add_leaf(
        id=f"r{rid}_Green_Star",
        desc="Restaurant holds a Michelin Green Star for sustainability",
        parent=r_node,
        critical=True
    )
    green_claim = (
        f"The provided page(s) confirm that the restaurant "
        f"{f'\"{restaurant.name}\" ' if safe_str(restaurant.name) else ''}"
        f"holds a MICHELIN Green Star (award for sustainability)."
    )
    await evaluator.verify(
        claim=green_claim,
        node=green_leaf,
        sources=award_srcs or main_src,
        additional_instruction=(
            "Look for 'MICHELIN Green Star' label/text and/or the green leaf icon, or comparable phrasing "
            "on Michelin Guide or other authoritative confirmation pages."
        )
    )

    # 3) US Location (critical)
    us_loc_leaf = evaluator.add_leaf(
        id=f"r{rid}_US_Location",
        desc="Restaurant is located in the United States",
        parent=r_node,
        critical=True
    )
    us_loc_claim = (
        f"The referenced page confirms that the restaurant "
        f"{f'\"{restaurant.name}\" ' if safe_str(restaurant.name) else ''}"
        f"is located in the United States."
    )
    await evaluator.verify(
        claim=us_loc_claim,
        node=us_loc_leaf,
        sources=main_src or (award_srcs if award_srcs else None),
        additional_instruction=(
            "Confirm the address shows a U.S. city and state (e.g., CA/NY/TX or full state name) and/or 'USA'/'United States'. "
            "If the page clearly indicates a U.S. address, mark as supported."
        )
    )

    # 4) Different State constraints for restaurant 2 and 3 (critical)
    norm_state = normalize_state(restaurant.address_state)
    if rid == 2:
        diff2 = norm_state is not None and norm_state not in previous_states
        evaluator.add_custom_node(
            result=diff2,
            id=f"r{rid}_Different_State",
            desc="Restaurant is located in a different US state than Restaurant 1",
            parent=r_node,
            critical=True
        )
    elif rid == 3:
        # must be distinct from both previous states
        diff3 = norm_state is not None and all(norm_state != s for s in previous_states)
        evaluator.add_custom_node(
            result=diff3,
            id=f"r{rid}_Different_State",
            desc="Restaurant is located in a different US state than Restaurants 1 and 2",
            parent=r_node,
            critical=True
        )

    # 5) Restaurant Basic Info (critical aggregate)
    basic_node = evaluator.add_parallel(
        id=f"r{rid}_Restaurant_Basic_Info",
        desc="Basic restaurant information is provided",
        parent=r_node,
        critical=True
    )

    # 5.1) Restaurant name provided (critical)
    name_exists = bool(safe_str(restaurant.name))
    evaluator.add_custom_node(
        result=name_exists,
        id=f"r{rid}_Restaurant_Name",
        desc="Full restaurant name is provided",
        parent=basic_node,
        critical=True
    )

    # 5.2) Complete address provided (street, city, state, zip) (critical)
    full_addr_present = all([
        bool(safe_str(restaurant.address_street)),
        bool(safe_str(restaurant.address_city)),
        bool(safe_str(restaurant.address_state)),
        bool(safe_str(restaurant.address_zip)),
    ])
    evaluator.add_custom_node(
        result=full_addr_present,
        id=f"r{rid}_Complete_Address",
        desc="Complete physical address including street, city, state, and ZIP code is provided",
        parent=basic_node,
        critical=True
    )

    # 5.3) Restaurant reference URL is authoritative (Michelin Guide or official site) (critical)
    ref_leaf = evaluator.add_leaf(
        id=f"r{rid}_Restaurant_Reference_URL",
        desc="Reference URL from Michelin Guide or official restaurant website confirming the information",
        parent=basic_node,
        critical=True
    )
    ref_claim = (
        f"The URL is an authoritative source for "
        f"{f'\"{restaurant.name}\"' if safe_str(restaurant.name) else 'the restaurant'}, "
        f"specifically either a Michelin Guide page (guide.michelin.com) or the restaurant's official website."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=restaurant.restaurant_ref_url,
        additional_instruction=(
            "Check if the domain and content indicate that the page is either: "
            "a) an official Michelin Guide page for the restaurant (e.g., guide.michelin.com), or "
            "b) the restaurant's own website. The page should clearly be about this restaurant."
        )
    )

    # 6) Chef Information (critical aggregate)
    chef_node = evaluator.add_parallel(
        id=f"r{rid}_Chef_Information",
        desc="Head chef or executive chef information is provided",
        parent=r_node,
        critical=True
    )

    chef = restaurant.chef or ChefInfo()

    # 6.1) Chef name provided (critical)
    chef_name_exists = bool(safe_str(chef.name))
    evaluator.add_custom_node(
        result=chef_name_exists,
        id=f"r{rid}_Chef_Name",
        desc="Name of the head chef or executive chef is provided",
        parent=chef_node,
        critical=True
    )

    # 6.2) Culinary training background provided (critical)
    training_exists = bool(safe_str(chef.training_background)) or (chef.institutions and any(safe_str(x) for x in chef.institutions))
    evaluator.add_custom_node(
        result=training_exists,
        id=f"r{rid}_Culinary_Training",
        desc="Chef's culinary training background or professional development through apprenticeships is documented",
        parent=chef_node,
        critical=True
    )

    # 6.3) Chef reference URL confirms credentials/training (critical)
    chef_ref_leaf = evaluator.add_leaf(
        id=f"r{rid}_Chef_Reference_URL",
        desc="Reference URL confirming the chef's credentials and training background",
        parent=chef_node,
        critical=True
    )
    chef_urls: List[str] = []
    if safe_str(chef.ref_url):
        chef_urls.append(chef.ref_url)  # primary
    if chef.additional_urls:
        chef_urls.extend([u for u in chef.additional_urls if safe_str(u)])
    # include restaurant page too, in case chef info is there
    if safe_str(restaurant.restaurant_ref_url):
        chef_urls.append(restaurant.restaurant_ref_url)
    # dedupe
    seen = set()
    chef_urls_deduped = []
    for u in chef_urls:
        if u not in seen:
            chef_urls_deduped.append(u)
            seen.add(u)

    chef_training_text = safe_str(chef.training_background)
    inst_text = ", ".join([x for x in chef.institutions if safe_str(x)]) if chef.institutions else ""
    both_training = chef_training_text or inst_text

    chef_claim = (
        f"The page confirms that {f'{chef.name} ' if safe_str(chef.name) else 'the named chef '} "
        f"is the head/executive chef of {f'\"{restaurant.name}\" ' if safe_str(restaurant.name) else 'the restaurant '} "
        f"and documents their culinary training background"
        f"{f' (e.g., {chef_training_text})' if chef_training_text else ''}"
        f"{f' including institution(s): {inst_text}' if inst_text else ''}."
    )
    await evaluator.verify(
        claim=chef_claim,
        node=chef_ref_leaf,
        sources=chef_urls_deduped or None,
        additional_instruction=(
            "Accept equivalent wording. Either the chef's official bio, Michelin Guide chef blurb, or reputable news profile "
            "must clearly indicate both the chef's role and the described training background or institutions."
        )
    )

    # 7) Sustainability Practices (critical aggregate)
    sust_node = evaluator.add_parallel(
        id=f"r{rid}_Sustainability_Practices",
        desc="At least two specific sustainability practices are documented",
        parent=r_node,
        critical=True
    )

    practices = restaurant.sustainability_practices or []
    # Ensure we evaluate two practice leaves even if missing (they will fail if not provided)
    for pidx in range(2):
        pr = practices[pidx] if pidx < len(practices) else SustainabilityPractice()
        pr_leaf = evaluator.add_leaf(
            id=f"r{rid}_Practice_{pidx+1}",
            desc=("First specific sustainability practice is described with verifiable details" if pidx == 0
                  else "Second specific sustainability practice is described with verifiable details"),
            parent=sust_node,
            critical=True
        )
        prac_desc = safe_str(pr.description)
        pr_claim = (
            f"The provided source(s) confirm that the restaurant "
            f"{f'\"{restaurant.name}\" ' if safe_str(restaurant.name) else ''}"
            f"employs the sustainability practice: {prac_desc if prac_desc else '[practice unspecified by answer]'}."
        )
        await evaluator.verify(
            claim=pr_claim,
            node=pr_leaf,
            sources=practice_sources(pr, restaurant) or None,
            additional_instruction=(
                "Look for explicit mention of concrete sustainability actions (e.g., on-site farm/garden, composting, "
                "local sourcing partnerships, renewable energy, zero-waste, regenerative agriculture). "
                "There should be clear, verifiable details on the page indicating the practice exists."
            )
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
    Evaluate an answer for the 'US restaurants with MICHELIN Star + Green Star' task.
    Returns a structured summary with the verification tree and scores.
    """
    # Initialize evaluator with a parallel root as per rubric
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

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_restaurants(),
        template_class=RestaurantsExtraction,
        extraction_name="restaurants_extraction"
    )

    # Select up to 3 restaurants (padding if fewer to keep fixed structure)
    restaurants = list(extraction.restaurants[:3])
    while len(restaurants) < 3:
        restaurants.append(Restaurant())

    # Build subtrees for each restaurant, maintaining state-difference constraints
    normalized_states: List[str] = []
    for idx in range(3):
        r = restaurants[idx]
        # Track previous normalized states for "Different_State" constraints
        await verify_restaurant(
            evaluator=evaluator,
            parent_node=root,
            restaurant=r,
            index=idx,
            previous_states=normalized_states.copy()
        )
        # Update state list for subsequent comparisons
        ns = normalize_state(r.address_state)
        if ns:
            normalized_states.append(ns)
        else:
            normalized_states.append("")  # placeholder to preserve indexing

    # Optionally record some debug info
    evaluator.add_custom_info(
        info={
            "extracted_states": [safe_str(r.address_state) for r in restaurants],
            "normalized_states": normalized_states
        },
        info_type="debug",
        info_name="state_info"
    )

    return evaluator.get_summary()