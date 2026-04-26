import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_metro_michelin_winespec_private_outdoor_ada_valet_4"
TASK_DESCRIPTION = """
Identify 4 restaurants in California (specifically in the Los Angeles, San Francisco, or San Diego metropolitan areas) that meet ALL of the following criteria as of 2024-2025:

1. The restaurant must hold at least one Michelin star (one, two, or three stars) according to the 2024 or 2025 Michelin Guide California
2. The restaurant must have received a Wine Spectator Restaurant Award (Award of Excellence, Best of Award of Excellence, or Grand Award) for 2024 or 2025
3. The restaurant must offer a private dining room or designated private dining space
4. The restaurant must provide outdoor seating options (such as a patio, terrace, or outdoor dining area)
5. The restaurant must be wheelchair accessible with ADA-compliant features, including an accessible entrance, accessible seating areas, and accessible restrooms
6. The restaurant must offer valet parking service
7. The restaurant must be currently operating

For each restaurant, provide:
- The restaurant's name
- A brief description explaining how it meets each criterion
- The official website URL or authoritative source URL verifying the information
"""

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        key = uu.lower()
        if key not in seen:
            seen.add(key)
            out.append(uu)
    return out


def _gather_all_sources(r: "RestaurantEntry") -> List[str]:
    urls = []
    if r.website_url:
        urls.append(r.website_url)
    urls.extend(r.general_sources or [])
    urls.extend(r.location_sources or [])
    urls.extend(r.michelin_sources or [])
    urls.extend(r.wine_spectator_sources or [])
    urls.extend(r.private_dining_sources or [])
    urls.extend(r.outdoor_seating_sources or [])
    urls.extend(r.wheelchair_access_sources or [])
    urls.extend(r.valet_parking_sources or [])
    urls.extend(r.operating_sources or [])
    return _dedup_urls(urls)


def _combine_for_claim(preferred: List[str], fallback: List[str]) -> List[str]:
    if preferred:
        return _dedup_urls(preferred)
    return _dedup_urls(fallback)


def _canon_name(name: Optional[str]) -> str:
    return (name or "").strip().lower()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RestaurantEntry(BaseModel):
    # Display / identity
    name: Optional[str] = None
    explanation: Optional[str] = None  # brief description explaining how criteria are met (as provided in answer)
    website_url: Optional[str] = None

    # General sources mentioned for this restaurant in the answer
    general_sources: List[str] = Field(default_factory=list)

    # Location
    location_area: Optional[str] = None  # e.g., "Los Angeles", "San Francisco", "San Diego" (as stated by the answer)
    location_city_state: Optional[str] = None  # e.g., "San Francisco, CA" if present
    location_sources: List[str] = Field(default_factory=list)

    # Michelin
    michelin_status: Optional[str] = None  # e.g., "1 star", "two Michelin stars", etc.
    michelin_sources: List[str] = Field(default_factory=list)

    # Wine Spectator
    wine_spectator_status: Optional[str] = None  # e.g., "2024 Best of Award of Excellence"
    wine_spectator_sources: List[str] = Field(default_factory=list)

    # Private dining
    private_dining_details: Optional[str] = None  # e.g., "private dining rooms available"
    private_dining_sources: List[str] = Field(default_factory=list)

    # Outdoor seating
    outdoor_seating_details: Optional[str] = None  # e.g., "outdoor patio"
    outdoor_seating_sources: List[str] = Field(default_factory=list)

    # Wheelchair accessibility
    wheelchair_access_details: Optional[str] = None  # e.g., "ADA compliant"
    wheelchair_access_sources: List[str] = Field(default_factory=list)

    # Valet parking
    valet_parking_details: Optional[str] = None  # e.g., "valet parking available"
    valet_parking_sources: List[str] = Field(default_factory=list)

    # Operating status
    operating_status: Optional[str] = None  # e.g., "Open and taking reservations"
    operating_sources: List[str] = Field(default_factory=list)


class RestaurantsExtraction(BaseModel):
    restaurants: List[RestaurantEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurants() -> str:
    return """
    Extract up to 6 restaurant entries from the answer that are proposed to meet the task criteria.
    For each restaurant, extract the following fields exactly as stated in the answer:

    1) name: Restaurant name (string)
    2) explanation: A brief explanation (prose or bullets) about how the criteria are met (string)
    3) website_url: The official website URL if provided (string or null)
    4) general_sources: All additional URLs cited for this restaurant that are not clearly tied to a single criterion (array of strings)

    5) location_area: The metro area claimed for the restaurant (one of: "Los Angeles", "San Francisco", "San Diego") if stated (string or null)
    6) location_city_state: City and state if provided (e.g., "San Francisco, CA") (string or null)
    7) location_sources: URLs cited that support the location/metro area (array of strings)

    8) michelin_status: The claim about Michelin status (e.g., "1 star", "two Michelin stars", "Michelin-starred") (string or null)
    9) michelin_sources: URLs cited that support the Michelin status (array of strings)

    10) wine_spectator_status: The claimed Wine Spectator award and year if present (e.g., "2024 Award of Excellence", "2025 Grand Award") (string or null)
    11) wine_spectator_sources: URLs cited that support the Wine Spectator award (array of strings)

    12) private_dining_details: The claim related to private dining (e.g., "private dining room available", "private event space") (string or null)
    13) private_dining_sources: URLs cited that support private dining availability (array of strings)

    14) outdoor_seating_details: The claim regarding outdoor seating (e.g., "patio", "terrace", "outdoor dining") (string or null)
    15) outdoor_seating_sources: URLs cited that support outdoor seating (array of strings)

    16) wheelchair_access_details: The claim regarding ADA/wheelchair accessibility (e.g., "ADA compliant", "wheelchair accessible restroom") (string or null)
    17) wheelchair_access_sources: URLs cited that support wheelchair accessibility (array of strings)

    18) valet_parking_details: The claim about valet parking (e.g., "valet parking available") (string or null)
    19) valet_parking_sources: URLs cited that support valet parking (array of strings)

    20) operating_status: The claim that the restaurant is currently operating (e.g., "currently open", "now accepting reservations") (string or null)
    21) operating_sources: URLs cited that support current operation (array of strings)

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer text (including markdown links). Do not invent URLs.
    - Normalize to full URLs including protocol (prepend http:// if protocol missing).
    - If a criterion is mentioned but has no explicit supporting URL in the answer, return an empty array for that criterion's sources.
    - For general_sources, include URLs that the answer appears to use as general references for the restaurant but are not tied to a single criterion.

    Return a JSON object with one field:
    {
      "restaurants": [ ... up to 6 RestaurantEntry objects ... ]
    }

    If fewer than 4 are present in the answer, return only those available.
    """


# --------------------------------------------------------------------------- #
# Verification for a single restaurant                                        #
# --------------------------------------------------------------------------- #
async def verify_restaurant(
    evaluator: Evaluator,
    parent_node,
    rest: RestaurantEntry,
    idx: int,
) -> None:
    """
    Build verification subtree for one restaurant.
    This node is critical, and all children are critical to enforce that each selected restaurant must meet all criteria.
    """

    rnode = evaluator.add_parallel(
        id=f"restaurant_{idx+1}",
        desc=f"Restaurant #{idx+1} (one of the four) and its required attributes/evidence.",
        parent=parent_node,
        critical=True,  # Make this restaurant critical under the strict parent
    )

    # Existence checks (critical)
    name_ok = bool(rest.name and rest.name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id=f"R{idx+1}_name",
        desc="Restaurant name is provided.",
        parent=rnode,
        critical=True,
    )

    explanation_ok = bool(rest.explanation and rest.explanation.strip())
    evaluator.add_custom_node(
        result=explanation_ok,
        id=f"R{idx+1}_explanation",
        desc="Includes a brief description that explicitly explains how the restaurant meets each listed criterion (can be prose or bullet points).",
        parent=rnode,
        critical=True,
    )

    # Reference URL(s) existence (critical). At least one URL (website or any cited source) must exist.
    all_sources = _gather_all_sources(rest)
    has_any_source = len(all_sources) > 0
    evaluator.add_custom_node(
        result=has_any_source,
        id=f"R{idx+1}_reference_url",
        desc="Provides at least one official website URL or authoritative source URL for verification.",
        parent=rnode,
        critical=True,
    )

    # Helper: sources per-claim with fallback to all known URLs for this restaurant
    def s_for(preferred: List[str]) -> List[str]:
        return _combine_for_claim(preferred, all_sources)

    # Leaf checks (all critical)
    claims_and_nodes: List[Tuple[str, List[str], Any, Optional[str]]] = []

    # 1) Location in allowed metro areas
    loc_leaf = evaluator.add_leaf(
        id=f"R{idx+1}_location",
        desc="Located in California in the Los Angeles, San Francisco, or San Diego metropolitan area.",
        parent=rnode,
        critical=True,
    )
    location_instruction = (
        "Verify the restaurant's address/location on the provided page(s). "
        "Accept 'Los Angeles', 'LA', 'Beverly Hills', 'Santa Monica', 'West Hollywood', 'Pasadena', "
        "'San Francisco', 'SF', 'Bay Area' (including Oakland, Berkeley, etc.), "
        "or 'San Diego' (including La Jolla, Del Mar, etc.) as qualifying metro areas in California. "
        "The page should clearly show a location/address that falls within one of these metro regions."
    )
    loc_sources = s_for(rest.location_sources or [])
    loc_name = rest.name or "the restaurant"
    loc_claim = f"The restaurant '{rest.name}' is located in California within the Los Angeles, San Francisco (Bay Area), or San Diego metropolitan area."
    claims_and_nodes.append((loc_claim, loc_sources, loc_leaf, location_instruction))

    # 2) Michelin star (2024 or 2025 Michelin Guide California)
    mic_leaf = evaluator.add_leaf(
        id=f"R{idx+1}_michelin",
        desc="Holds at least one Michelin star per the 2024 or 2025 Michelin Guide California.",
        parent=rnode,
        critical=True,
    )
    mic_instruction = (
        "Confirm that the restaurant holds at least one Michelin star (1, 2, or 3) in the 2024 or 2025 Michelin Guide California. "
        "Accept explicit Michelin pages or credible references indicating 'Michelin star' with the 2024 or 2025 California edition. "
        "Do not accept 'Michelin recommended' or 'Bib Gourmand' alone."
    )
    mic_sources = s_for(rest.michelin_sources or [])
    mic_claim = f"The restaurant '{rest.name}' holds at least one Michelin star in the 2024 or 2025 Michelin Guide California."
    claims_and_nodes.append((mic_claim, mic_sources, mic_leaf, mic_instruction))

    # 3) Wine Spectator award (2024 or 2025)
    ws_leaf = evaluator.add_leaf(
        id=f"R{idx+1}_wine_spectator",
        desc="Received a Wine Spectator Restaurant Award (Award of Excellence, Best of Award of Excellence, or Grand Award) for 2024 or 2025.",
        parent=rnode,
        critical=True,
    )
    ws_instruction = (
        "Confirm the restaurant received one of Wine Spectator's Restaurant Awards in 2024 or 2025: "
        "Award of Excellence, Best of Award of Excellence, or Grand Award. "
        "Look for official Wine Spectator listings or the restaurant's page citing the award with the correct year (2024 or 2025)."
    )
    ws_sources = s_for(rest.wine_spectator_sources or [])
    ws_claim = f"The restaurant '{rest.name}' received a Wine Spectator Restaurant Award in 2024 or 2025 (Award of Excellence, Best of Award of Excellence, or Grand Award)."
    claims_and_nodes.append((ws_claim, ws_sources, ws_leaf, ws_instruction))

    # 4) Private dining
    pd_leaf = evaluator.add_leaf(
        id=f"R{idx+1}_private_dining",
        desc="Offers a private dining room or designated private dining space.",
        parent=rnode,
        critical=True,
    )
    pd_instruction = (
        "Verify that the restaurant offers private dining (private room, designated private dining space, or private event space). "
        "Accept terms like 'private dining', 'private room', 'private event spaces', or 'buyouts'."
    )
    pd_sources = s_for(rest.private_dining_sources or [])
    pd_claim = f"The restaurant '{rest.name}' offers a private dining room or a designated private dining space."
    claims_and_nodes.append((pd_claim, pd_sources, pd_leaf, pd_instruction))

    # 5) Outdoor seating
    out_leaf = evaluator.add_leaf(
        id=f"R{idx+1}_outdoor_seating",
        desc="Provides outdoor seating options (e.g., patio, terrace, outdoor dining area).",
        parent=rnode,
        critical=True,
    )
    out_instruction = (
        "Verify the restaurant has outdoor seating such as a patio, terrace, outdoor dining, or alfresco seating."
    )
    out_sources = s_for(rest.outdoor_seating_sources or [])
    out_claim = f"The restaurant '{rest.name}' provides outdoor seating options (e.g., patio, terrace, or outdoor dining)."
    claims_and_nodes.append((out_claim, out_sources, out_leaf, out_instruction))

    # 6) Wheelchair accessibility / ADA-compliant features
    ada_leaf = evaluator.add_leaf(
        id=f"R{idx+1}_wheelchair_access",
        desc="Wheelchair accessible with ADA-compliant features including accessible entrance, accessible seating areas, and accessible restrooms.",
        parent=rnode,
        critical=True,
    )
    ada_instruction = (
        "Verify that the restaurant is wheelchair accessible with ADA-compliant features. "
        "Accept explicit statements like 'ADA compliant', 'wheelchair accessible', 'accessible restroom', "
        "or pages detailing accessible entrance and seating. If multiple sources collectively demonstrate accessibility, that is acceptable."
    )
    ada_sources = s_for(rest.wheelchair_access_sources or [])
    ada_claim = f"The restaurant '{rest.name}' is wheelchair accessible with ADA-compliant features (accessible entrance, accessible seating, and accessible restrooms)."
    claims_and_nodes.append((ada_claim, ada_sources, ada_leaf, ada_instruction))

    # 7) Valet parking
    valet_leaf = evaluator.add_leaf(
        id=f"R{idx+1}_valet",
        desc="Offers valet parking service.",
        parent=rnode,
        critical=True,
    )
    valet_instruction = (
        "Verify that the restaurant offers valet parking. Accept 'valet parking', 'valet available', or 'complimentary valet'."
    )
    valet_sources = s_for(rest.valet_parking_sources or [])
    valet_claim = f"The restaurant '{rest.name}' offers valet parking service."
    claims_and_nodes.append((valet_claim, valet_sources, valet_leaf, valet_instruction))

    # 8) Currently operating
    op_leaf = evaluator.add_leaf(
        id=f"R{idx+1}_operating",
        desc="Currently operating as of 2024–2025.",
        parent=rnode,
        critical=True,
    )
    op_instruction = (
        "Confirm the restaurant is currently operating (open to the public) as of 2024–2025. "
        "Accept evidence like active reservation links, current hours/menus, or recent official updates indicating it is open. "
        "If the page indicates 'permanently closed' or similar, this fails."
    )
    op_sources = s_for(rest.operating_sources or [])
    op_claim = f"The restaurant '{rest.name}' is currently operating (open to the public) as of 2024–2025."
    claims_and_nodes.append((op_claim, op_sources, op_leaf, op_instruction))

    # Perform verifications in parallel for this restaurant
    await evaluator.batch_verify(claims_and_nodes)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
# --------------------------------------------------------------------------- #
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
    """
    Evaluate an answer for the '4 CA metro Michelin + Wine Spectator + Private + Outdoor + ADA + Valet + Operating' task.
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
        default_model=model,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_restaurants(),
        template_class=RestaurantsExtraction,
        extraction_name="restaurants_extraction",
    )

    # Select exactly 4 (pad with empty entries if needed)
    selected: List[RestaurantEntry] = list(extracted.restaurants[:4])
    while len(selected) < 4:
        selected.append(RestaurantEntry())

    # Build a strict, critical parent node under root to enforce all-or-nothing for the four restaurants + count/distinctness
    strict_root = evaluator.add_parallel(
        id="all_requirements",
        desc="Identify exactly 4 distinct currently operating restaurants in the Los Angeles, San Francisco, or San Diego metro areas (California) that satisfy all listed 2024–2025 criteria, and provide required writeups and source URL(s) per restaurant.",
        parent=root,
        critical=True,
    )

    # Count and distinctness (critical)
    names = [(_canon_name(r.name)) for r in selected]
    non_empty_names = [n for n in names if n]
    count_and_distinct = (len(non_empty_names) == 4) and (len(set(non_empty_names)) == 4)
    evaluator.add_custom_node(
        result=count_and_distinct,
        id="Count_and_distinctness",
        desc="Response provides exactly 4 distinct restaurants (no duplicates).",
        parent=strict_root,
        critical=True,
    )

    # Add each restaurant subtree (critical under strict_root)
    for i, r in enumerate(selected):
        await verify_restaurant(evaluator, strict_root, r, i)

    # Return summary
    return evaluator.get_summary()