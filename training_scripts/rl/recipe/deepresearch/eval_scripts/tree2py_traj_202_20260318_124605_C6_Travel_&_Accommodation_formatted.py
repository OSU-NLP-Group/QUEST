import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "travel_rewards_cities_jetblue_hotels_recreation_2026"
TASK_DESCRIPTION = """
A travel blogger is creating a comprehensive guide for families who want to maximize their travel rewards and access. Identify four U.S. cities that meet ALL of the following criteria:

1. The city must be served by JetBlue Airlines as either their main hub or one of their designated focus cities (as of 2026).

2. The city must have at least one hotel property that participates in one of these four major loyalty programs: Marriott Bonvoy, Hilton Honors, World of Hyatt, or IHG One Rewards. The hotel must offer elite status benefits including at least two of the following: room upgrades, bonus points, late checkout, or complimentary breakfast.

3. The city must provide access to either:
   - A federal recreation area (such as a national park, national forest, or other federal land) that accepts the America the Beautiful Annual Pass, OR
   - A major commercial theme park

For each of the four cities, provide:
- The city name and state
- Confirmation of JetBlue hub/focus city status
- The specific hotel chain from the four major loyalty programs
- The specific recreation facility or theme park name
- At least one reference URL supporting each piece of information
"""

ALLOWED_CHAINS = {"Marriott Bonvoy", "Hilton Honors", "World of Hyatt", "IHG One Rewards"}
ALLOWED_BENEFITS = {"room upgrades", "bonus points", "late checkout", "complimentary breakfast"}
# As referenced in the rubric description for 2026:
JETBLUE_HUB_FOCUS_CODES = {"hub": {"JFK"}, "focus": {"BOS", "FLL", "LAX", "MCO", "SJU"}}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class JetBlueInfo(BaseModel):
    status: Optional[str] = None  # "hub" or "focus"
    airport_code: Optional[str] = None  # e.g., "JFK", "BOS", "FLL", "LAX", "MCO", "SJU"
    urls: List[str] = Field(default_factory=list)  # At least one URL supporting hub/focus status


class HotelInfo(BaseModel):
    property_name: Optional[str] = None  # Specific hotel property name
    chain: Optional[str] = None  # One of ALLOWED_CHAINS
    claimed_benefits: List[str] = Field(default_factory=list)  # Subset of ALLOWED_BENEFITS
    benefit_urls: List[str] = Field(default_factory=list)  # URLs supporting the elite benefit claims


class RecreationInfo(BaseModel):
    name: Optional[str] = None  # Specific facility or theme park name
    type: Optional[str] = None  # "federal_area" or "theme_park"
    urls: List[str] = Field(default_factory=list)  # URLs supporting the facility and its access type


class CityItem(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    jetblue: Optional[JetBlueInfo] = None
    hotel: Optional[HotelInfo] = None
    recreation: Optional[RecreationInfo] = None


class CitiesExtraction(BaseModel):
    cities: List[CityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cities() -> str:
    return """
    Extract up to four (4) U.S. cities from the answer that meet the aviation (JetBlue hub/focus), accommodation (major loyalty chain + elite benefits), and recreation (federal area accepting the America the Beautiful pass OR major theme park) requirements.

    Return a JSON object with:
    {
      "cities": [
        {
          "city": string | null,
          "state": string | null,
          "jetblue": {
            "status": "hub" | "focus" | null,
            "airport_code": "JFK" | "BOS" | "FLL" | "LAX" | "MCO" | "SJU" | null,
            "urls": string[]    // URLs that explicitly support the hub/focus status; include at least one if provided
          } | null,
          "hotel": {
            "property_name": string | null,   // The specific hotel property name mentioned
            "chain": "Marriott Bonvoy" | "Hilton Honors" | "World of Hyatt" | "IHG One Rewards" | null,
            "claimed_benefits": string[],     // Zero or more from: "room upgrades", "bonus points", "late checkout", "complimentary breakfast"
            "benefit_urls": string[]          // URLs that explicitly support elite benefits (brand/program page or property page); include at least one if provided
          } | null,
          "recreation": {
            "name": string | null,            // Specific federal facility or theme park name
            "type": "federal_area" | "theme_park" | null,
            "urls": string[]                  // URLs that explicitly support either federal area acceptance of the America the Beautiful pass or that it is a major theme park
          } | null
        }
      ]
    }

    Rules:
    - Do NOT invent information; extract exactly what the answer provides.
    - If an item (city) in the answer is incomplete or missing information, fill available fields and use null for the rest.
    - Keep URLs exactly as written (plain URLs or from markdown links).
    - For "claimed_benefits", only include any of these four if explicitly claimed: room upgrades, bonus points, late checkout, complimentary breakfast.
    - Return at most four cities in the order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"][n - 1] if 1 <= n <= 6 else f"#{n}"


def non_empty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


def benefits_list_or_default(benefits: Optional[List[str]]) -> List[str]:
    if not benefits:
        return []
    return [b for b in benefits if isinstance(b, str) and b.strip()]


# --------------------------------------------------------------------------- #
# Verification for a single city                                              #
# --------------------------------------------------------------------------- #
async def verify_one_city(
    evaluator: Evaluator,
    parent_node,
    city_idx: int,
    item: CityItem,
) -> None:
    parent_desc = f"{ordinal(city_idx)} qualifying city meets all aviation, accommodation, and recreation requirements"
    city_node = evaluator.add_parallel(
        id=f"City_{city_idx}",
        desc=parent_desc,
        parent=parent_node,
        critical=False  # each city contributes partial credit independently
    )

    # 1) City_Name_and_State (existence)
    city_name_ok = non_empty_str(item.city) and non_empty_str(item.state)
    evaluator.add_custom_node(
        result=city_name_ok,
        id=f"City_{city_idx}_City_Name_and_State",
        desc="Provides the city name and state location",
        parent=city_node,
        critical=True
    )

    # 2) JetBlue_Reference_URL (existence of supporting URL)
    jb_urls = item.jetblue.urls if (item.jetblue and item.jetblue.urls) else []
    evaluator.add_custom_node(
        result=has_any_url(jb_urls),
        id=f"City_{city_idx}_JetBlue_Reference_URL",
        desc="Provides URL reference confirming JetBlue hub or focus city status",
        parent=city_node,
        critical=True
    )

    # 3) JetBlue_Hub_Focus_Status (verify with URLs)
    jb_status_leaf = evaluator.add_leaf(
        id=f"City_{city_idx}_JetBlue_Hub_Focus_Status",
        desc="City's airport is JetBlue's main hub (JFK) or one of the five focus cities (BOS, FLL, LAX, MCO, SJU)",
        parent=city_node,
        critical=True
    )

    city_label = f"{item.city}, {item.state}" if city_name_ok else "the target city"
    jb_status_str = (item.jetblue.status or "").strip().lower() if item.jetblue else ""
    jb_code = (item.jetblue.airport_code or "").strip().upper() if item.jetblue else ""
    if jb_status_str in {"hub", "focus"}:
        claim_jb = f"{city_label} is JetBlue's {jb_status_str} city" + (f" (airport code {jb_code})" if jb_code else "") + "."
    else:
        claim_jb = f"{city_label} is one of JetBlue's hub or focus cities."

    await evaluator.verify(
        claim=claim_jb,
        node=jb_status_leaf,
        sources=jb_urls,
        additional_instruction=(
            "Verify using the provided sources that the city is JetBlue's main hub (JFK) or one of its focus cities "
            "(BOS, FLL, LAX, MCO, SJU). Accept reputable sources (e.g., JetBlue official site, Wikipedia with citations, "
            "or credible news/industry pages). If the source explicitly confirms hub/focus status, mark as supported."
        ),
    )

    # 4) Hotel_Reference_URL (existence)
    hotel = item.hotel or HotelInfo()
    benefit_urls = hotel.benefit_urls or []
    evaluator.add_custom_node(
        result=has_any_url(benefit_urls),
        id=f"City_{city_idx}_Hotel_Reference_URL",
        desc="Provides URL reference for hotel loyalty program benefits information",
        parent=city_node,
        critical=True
    )

    # 5) Hotel_Chain_Identification (existence + chain in allowed set)
    chain_ok = non_empty_str(hotel.chain) and (hotel.chain in ALLOWED_CHAINS)
    property_ok = non_empty_str(hotel.property_name)
    evaluator.add_custom_node(
        result=(chain_ok and property_ok),
        id=f"City_{city_idx}_Hotel_Chain_Identification",
        desc="Identifies a specific hotel from one of the four major loyalty programs: Marriott Bonvoy, Hilton Honors, World of Hyatt, or IHG One Rewards",
        parent=city_node,
        critical=True
    )

    # 6) Hotel_Elite_Benefits (verify: at least two among the four benefits)
    hotel_benefits_leaf = evaluator.add_leaf(
        id=f"City_{city_idx}_Hotel_Elite_Benefits",
        desc="Hotel offers elite status benefits including at least two of: room upgrades, bonus points, late checkout, or complimentary breakfast",
        parent=city_node,
        critical=True
    )
    claimed_benefits = benefits_list_or_default(hotel.claimed_benefits)
    # Prefer using claimed list if present; otherwise assert generic "at least two" from the canonical four.
    if claimed_benefits:
        listed = ", ".join(claimed_benefits)
        claim_hotel = (
            f"According to the provided source(s), elite members for {hotel.chain or 'the cited loyalty program'} "
            f"(staying at {hotel.property_name or 'the cited property'}) receive at least two of these benefits: "
            f"{listed}. At least two of these four canonical benefits are supported: room upgrades, bonus points, late checkout, complimentary breakfast."
        )
    else:
        claim_hotel = (
            f"According to the provided source(s), elite members for {hotel.chain or 'the cited loyalty program'} "
            f"(staying at {hotel.property_name or 'the cited property'}) receive at least two of the following benefits: "
            f"room upgrades, bonus points, late checkout, complimentary breakfast."
        )

    await evaluator.verify(
        claim=claim_hotel,
        node=hotel_benefits_leaf,
        sources=benefit_urls,
        additional_instruction=(
            "Use the loyalty program benefit page and/or the specific hotel's page to confirm elite benefits. "
            "It's sufficient if the program generally grants the benefits at the chain/brand level (property-level exceptions may apply). "
            "Count as supported if at least two of the four canonical benefits are clearly offered to elite members."
        ),
    )

    # 7) Recreation_Reference_URL (existence)
    rec = item.recreation or RecreationInfo()
    rec_urls = rec.urls or []
    evaluator.add_custom_node(
        result=has_any_url(rec_urls),
        id=f"City_{city_idx}_Recreation_Reference_URL",
        desc="Provides URL reference for recreation facility information",
        parent=city_node,
        critical=True
    )

    # 8) Recreation_Facility_Name (existence)
    evaluator.add_custom_node(
        result=non_empty_str(rec.name),
        id=f"City_{city_idx}_Recreation_Facility_Name",
        desc="Provides the specific name of the recreation facility (federal area or theme park)",
        parent=city_node,
        critical=True
    )

    # 9) Recreation_Access_Type (verify)
    rec_type = (rec.type or "").strip().lower()
    rec_leaf = evaluator.add_leaf(
        id=f"City_{city_idx}_Recreation_Access_Type",
        desc="Identifies whether the facility is a federal recreation area accepting America the Beautiful pass OR a major commercial theme park",
        parent=city_node,
        critical=True
    )
    if rec_type == "federal_area":
        claim_rec = (
            f"{rec.name or 'The cited facility'} is a federal recreation area and it accepts the America the Beautiful Annual Pass "
            f"(also known as the Interagency Annual Pass)."
        )
        add_ins = (
            "Look for explicit acceptance of the America the Beautiful/Interagency Annual Pass. "
            "Examples include National Park Service, US Forest Service, Bureau of Land Management, etc. "
            "If acceptance is clearly stated for entrance fees (or equivalent), mark as supported."
        )
    else:
        # Default to theme_park requirement if unspecified; still verify as 'major commercial theme park'
        claim_rec = f"{rec.name or 'The cited facility'} is a major commercial theme park."
        add_ins = (
            "Verify that the facility is a well-known commercial theme park (e.g., Disney, Universal, SeaWorld, Six Flags, Cedar Fair, LEGOLAND, etc.), "
            "featuring rides/attractions and operating as a commercial park."
        )

    await evaluator.verify(
        claim=claim_rec,
        node=rec_leaf,
        sources=rec_urls,
        additional_instruction=add_ins
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
    Evaluate an answer for the 'four U.S. cities with JetBlue hub/focus + hotel elite benefits + recreation access' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Cities are evaluated independently
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

    # Extract structured data about cities
    extracted = await evaluator.extract(
        prompt=prompt_extract_cities(),
        template_class=CitiesExtraction,
        extraction_name="cities_extraction"
    )

    # Normalize to exactly 4 cities (pad with empty if fewer)
    items: List[CityItem] = list(extracted.cities[:4])
    while len(items) < 4:
        items.append(CityItem())

    # Build top-level "plan" node (non-critical to allow partial credit)
    plan_node = evaluator.add_parallel(
        id="Root_Complete_Travel_Plan",
        desc="Identify four U.S. cities that each meet specific aviation, accommodation, and recreation criteria for a family vacation",
        parent=root,
        critical=False
    )

    # Verify each city
    for idx in range(1, 5):
        await verify_one_city(evaluator, plan_node, idx, items[idx - 1])

    return evaluator.get_summary()