import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "michelin_us_top4_2025"
TASK_DESCRIPTION = """Find four Michelin-starred restaurants in the United States that meet ALL of the following criteria. The restaurants must be located in at least three different US states.

For each restaurant, provide:

Basic Qualifications:
1. Restaurant name and location (city and state)
2. Current Michelin star rating (must be at least 2 stars according to the 2025 Michelin Guide)
3. Head chef or executive chef name and their James Beard Award(s) won (specify award category and year)

Dining Experience:
4. Tasting menu details: number of courses (minimum 6) and price per person
5. Wine program recognition: Wine Spectator Restaurant Award level (Award of Excellence, Best of Award of Excellence, or Grand Award)
6. Sustainability practices: Either Michelin Green Star status OR specific documented sustainable sourcing practices (e.g., farm partnerships, sustainable seafood certifications)

Practical Information:
7. Reservation platform used (must be OpenTable, Resy, or Tock)
8. Private dining capacity: Confirm availability for groups of at least 8 guests
9. Operating schedule: Days of the week dinner service is offered (must be at least 5 days)
10. Dietary accommodations: At least two types from vegetarian/vegan options, gluten-free options, or allergy accommodations
11. Dress code policy

References:
12. Provide reference URLs supporting the information for each restaurant (minimum one URL per major category: Basic Qualifications, Dining Experience, and Practical Information)
"""

ALLOWED_WINE_SPECTATOR_LEVELS = [
    "Award of Excellence",
    "Best of Award of Excellence",
    "Grand Award",
]
ALLOWED_RESERVATION_PLATFORMS = ["OpenTable", "Resy", "Tock"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class JBAward(BaseModel):
    category: Optional[str] = None
    year: Optional[str] = None
    note: Optional[str] = None  # e.g., "Winner", "Won"


class TastingMenuInfo(BaseModel):
    courses_text: Optional[str] = None  # e.g., "8" or "8-10" or "at least 6"
    courses_min: Optional[int] = None   # best guess minimal integer if available
    price_per_person: Optional[str] = None  # keep as text to allow ranges or symbols


class SustainabilityInfo(BaseModel):
    michelin_green_star: Optional[bool] = None
    practices: Optional[str] = None  # brief description of sustainable sourcing practices (if any)


class PracticalInfo(BaseModel):
    reservation_platform: Optional[str] = None  # expected to be one of OpenTable, Resy, Tock
    private_dining_min_group: Optional[str] = None  # numeric or text (e.g., "8", "8+")
    dinner_days: List[str] = Field(default_factory=list)  # days like ["Tue","Wed","Thu","Fri","Sat"]
    dietary_accommodations: List[str] = Field(default_factory=list)  # e.g., ["vegetarian", "gluten-free"]
    dress_code: Optional[str] = None


class RestaurantItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # US state (full name or 2-letter code)
    michelin_stars_2025: Optional[str] = None  # textual (e.g., "2", "Two stars", "3")
    chef_name: Optional[str] = None
    chef_jb_awards: List[JBAward] = Field(default_factory=list)  # only winner awards extracted
    tasting_menu: Optional[TastingMenuInfo] = None
    wine_spectator_level: Optional[str] = None
    sustainability: Optional[SustainabilityInfo] = None
    practical: Optional[PracticalInfo] = None
    sources_basic: List[str] = Field(default_factory=list)
    sources_dining: List[str] = Field(default_factory=list)
    sources_practical: List[str] = Field(default_factory=list)


class RestaurantsExtraction(BaseModel):
    restaurants: List[RestaurantItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurants() -> str:
    return """
Extract up to the first FOUR (4) Michelin-starred restaurants in the United States mentioned in the answer, along with the required details for each. Do not invent information. Use exactly what is in the answer text. If something is missing, set it to null or an empty list as appropriate.

For each restaurant, extract the following fields into an object:

- name: Restaurant name
- city: City
- state: US state (full name or 2-letter code)
- michelin_stars_2025: The number of Michelin stars in the 2025 Michelin Guide (text as presented, e.g., "2", "3", "Two stars")
- chef_name: Head or Executive Chef name
- chef_jb_awards: An array of James Beard Awards where the chef is stated as a winner (not just a nominee). Each item:
  - category: e.g., "Best Chef: Northeast"
  - year: e.g., "2022"
  - note: e.g., "Winner" (use "Winner" if stated as won; omit nominee/finalist entries)

- tasting_menu: Information about the tasting menu:
  - courses_text: number of courses as presented (can be a range or phrase like "at least 6")
  - courses_min: minimum number of courses as an integer if it can be reasonably determined; else null
  - price_per_person: price per person as text as presented (e.g., "$295", "$295–$325")

- wine_spectator_level: Award level as presented; expected values include "Award of Excellence", "Best of Award of Excellence", or "Grand Award" (if different, still record text exactly)

- sustainability:
  - michelin_green_star: true if explicitly stated the restaurant holds a Michelin Green Star; false if explicitly stated it does not; null if not mentioned
  - practices: brief text describing specific sustainable sourcing practices if mentioned (e.g., farm partnerships, sustainable seafood certifications); else null

- practical:
  - reservation_platform: the platform used if mentioned (e.g., "OpenTable", "Resy", "Tock"; if another platform is mentioned, record it exactly as text)
  - private_dining_min_group: the minimum group size for private dining if provided (text, e.g., "8", "8+", "10")
  - dinner_days: list the days of the week with dinner service offered (e.g., ["Tue","Wed","Thu","Fri","Sat"]); expand ranges if provided (e.g., "Tue–Sat" -> ["Tue","Wed","Thu","Fri","Sat"])
  - dietary_accommodations: list explicit types mentioned from among: "vegetarian", "vegan", "gluten-free", "allergy"
  - dress_code: the dress code policy text if provided

- sources_basic: array of URLs that support the Basic Qualifications (name/location, 2025 Michelin stars, chef and James Beard awards)
- sources_dining: array of URLs that support the Dining Experience (tasting menu courses and price, Wine Spectator award, sustainability)
- sources_practical: array of URLs that support the Practical Information (reservation platform, private dining capacity, dinner days, dietary accommodations, dress code)

Rules:
- Extract only URLs explicitly present in the answer. If URLs are in markdown, extract the URL part.
- Do not fabricate URLs. If no URL is provided for a category, return an empty array for that sources_* field.
- If more than four restaurants are present, extract only the first four in the order they appear.
- If fewer than four restaurants are present, return the ones available.

Return a JSON with one field:
- restaurants: an array of at most 4 RestaurantItem objects as specified above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def safe_list(x: Optional[List[str]]) -> List[str]:
    return x if isinstance(x, list) else []


def combine_sources(*args: List[str]) -> List[str]:
    # Unique-preserve order
    seen = set()
    result: List[str] = []
    for lst in args:
        for u in lst:
            if u and u not in seen:
                seen.add(u)
                result.append(u)
    return result


def text_or_empty(x: Optional[str]) -> str:
    return x or ""


def int_from_text(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        # Try plain integer first
        return int(value)
    except Exception:
        # Try to parse something like "8+" or "8-10"
        digits = ""
        for ch in value:
            if ch.isdigit():
                digits += ch
            elif digits:
                break
        try:
            return int(digits) if digits else None
        except Exception:
            return None


# --------------------------------------------------------------------------- #
# Verification of a single restaurant                                         #
# --------------------------------------------------------------------------- #
async def verify_restaurant(
    evaluator: Evaluator,
    parent_node,
    item: RestaurantItem,
    index: int,
) -> None:
    """
    Build the verification subtree for a single restaurant.
    index is 0-based; displayed nodes will use 1-based numbering to match rubric IDs.
    """
    rn = index + 1  # 1..4

    # Restaurant node (non-critical, parallel aggregation)
    restaurant_node = evaluator.add_parallel(
        id=f"Restaurant_{rn}",
        desc=f"{rn}th restaurant (qualifying item)" if rn > 1 else "1st restaurant (qualifying item)",
        parent=parent_node,
        critical=False,
    )

    # --------------------- Basic Qualifications -------------------------- #
    basic_node = evaluator.add_parallel(
        id=f"R{rn}_Basic_Qualifications",
        desc="Meets basic qualification constraints and provides required basic details",
        parent=restaurant_node,
        critical=True,  # all children must pass
    )

    # Rn_Name_And_Location (existence check)
    name_loc_exists = (
        bool(text_or_empty(item.name).strip())
        and bool(text_or_empty(item.city).strip())
        and bool(text_or_empty(item.state).strip())
    )
    evaluator.add_custom_node(
        result=name_loc_exists,
        id=f"R{rn}_Name_And_Location",
        desc="Restaurant name, city, and US state are provided",
        parent=basic_node,
        critical=True,
    )

    # Rn_Michelin_2025_Stars (verify at least 2 stars)
    n_michelin = evaluator.add_leaf(
        id=f"R{rn}_Michelin_2025_Stars",
        desc="Restaurant has at least 2 Michelin stars according to the 2025 Michelin Guide",
        parent=basic_node,
        critical=True,
    )
    claim_michelin = (
        f"According to the 2025 Michelin Guide, the restaurant '{text_or_empty(item.name)}' "
        f"holds at least two Michelin stars (2 or 3 stars)."
    )
    await evaluator.verify(
        claim=claim_michelin,
        node=n_michelin,
        sources=item.sources_basic,
        additional_instruction="Verify explicitly from the cited pages that in the 2025 Michelin Guide the restaurant has 2 or 3 stars. Accept phrases like 'two-star' or 'three-star' as equivalent.",
    )

    # Rn_Chef_And_JamesBeard (verify winner award)
    n_jb = evaluator.add_leaf(
        id=f"R{rn}_Chef_And_JamesBeard",
        desc="Head chef or executive chef name is provided AND at least one James Beard Award is listed with award category and year (chef is a winner)",
        parent=basic_node,
        critical=True,
    )
    # Build a claim from first award if available; otherwise a generic "winner" claim (likely to fail)
    if item.chef_name and item.chef_jb_awards:
        first_award = item.chef_jb_awards[0]
        award_cat = text_or_empty(first_award.category)
        award_year = text_or_empty(first_award.year)
        claim_jb = (
            f"Chef {text_or_empty(item.chef_name)} won a James Beard Award"
            f"{' in ' + award_year if award_year else ''}"
            f"{' for ' + award_cat if award_cat else ''}."
        )
    else:
        claim_jb = f"Chef {text_or_empty(item.chef_name)} is a James Beard Award winner."

    await evaluator.verify(
        claim=claim_jb,
        node=n_jb,
        sources=item.sources_basic,
        additional_instruction="Confirm that the chef is a James Beard Award WINNER (not merely a nominee or finalist). If the page only shows nominations, this claim is not supported.",
    )

    # --------------------- Dining Experience ----------------------------- #
    dining_node = evaluator.add_parallel(
        id=f"R{rn}_Dining_Experience",
        desc="Meets dining experience constraints and provides required dining details",
        parent=restaurant_node,
        critical=True,
    )

    # Rn_Tasting_Menu_Courses (verify at least 6 courses)
    n_courses = evaluator.add_leaf(
        id=f"R{rn}_Tasting_Menu_Courses",
        desc="Tasting menu course count is provided and is at least 6",
        parent=dining_node,
        critical=True,
    )
    # Build a robust claim: at least 6 courses
    claim_courses = "The tasting menu has at least 6 courses."
    await evaluator.verify(
        claim=claim_courses,
        node=n_courses,
        sources=item.sources_dining,
        additional_instruction="Look for the stated number of courses on the menu page; accept ranges like '8–10 courses' or wording like 'at least six'.",
    )

    # Rn_Tasting_Menu_Price (existence only per rubric)
    price_provided = bool(item.tasting_menu and text_or_empty(item.tasting_menu.price_per_person).strip())
    evaluator.add_custom_node(
        result=price_provided,
        id=f"R{rn}_Tasting_Menu_Price",
        desc="Tasting menu price per person is provided",
        parent=dining_node,
        critical=True,
    )

    # Rn_Wine_Spectator_Level (verify and ensure it is one of allowed)
    n_wine = evaluator.add_leaf(
        id=f"R{rn}_Wine_Spectator_Level",
        desc="Wine Spectator Restaurant Award level is provided AND is one of: Award of Excellence, Best of Award of Excellence, Grand Award",
        parent=dining_node,
        critical=True,
    )
    claim_wine = (
        f"The restaurant holds the Wine Spectator Restaurant Award level "
        f"'{text_or_empty(item.wine_spectator_level)}', which must be one of: "
        f"Award of Excellence, Best of Award of Excellence, or Grand Award."
    )
    await evaluator.verify(
        claim=claim_wine,
        node=n_wine,
        sources=item.sources_dining,
        additional_instruction="Confirm the exact Wine Spectator award level. Treat 'Best of Award of Excellence' and 'Award of Excellence' exactly. If the page shows a different award or no Wine Spectator award, this claim fails.",
    )

    # Rn_Sustainability (verify: either Green Star OR documented practices)
    n_sust = evaluator.add_leaf(
        id=f"R{rn}_Sustainability",
        desc="Either Michelin Green Star status is stated OR specific documented sustainable sourcing practices are described",
        parent=dining_node,
        critical=True,
    )
    if item.sustainability and item.sustainability.michelin_green_star:
        claim_sust = "The restaurant holds a Michelin Green Star."
    elif item.sustainability and text_or_empty(item.sustainability.practices).strip():
        claim_sust = (
            f"The restaurant explicitly documents sustainable sourcing practices such as: "
            f"{text_or_empty(item.sustainability.practices)}."
        )
    else:
        claim_sust = "The restaurant holds a Michelin Green Star or explicitly documents sustainable sourcing practices."

    await evaluator.verify(
        claim=claim_sust,
        node=n_sust,
        sources=combine_sources(item.sources_dining, item.sources_basic),
        additional_instruction="Support either explicit Michelin Green Star recognition OR specific, concrete sustainable practices (e.g., farm partnerships, sustainable seafood certifications).",
    )

    # --------------------- Practical Information ------------------------- #
    practical_node = evaluator.add_parallel(
        id=f"R{rn}_Practical_Information",
        desc="Meets practical constraints and provides required practical details",
        parent=restaurant_node,
        critical=True,
    )

    # Rn_Reservation_Platform (verify and ensure allowed)
    n_res = evaluator.add_leaf(
        id=f"R{rn}_Reservation_Platform",
        desc="Reservation platform is specified AND is OpenTable, Resy, or Tock",
        parent=practical_node,
        critical=True,
    )
    claim_res = (
        f"Reservations for the restaurant are made via {text_or_empty(item.practical.reservation_platform) if item.practical else ''}, "
        f"which must be one of OpenTable, Resy, or Tock."
    )
    await evaluator.verify(
        claim=claim_res,
        node=n_res,
        sources=item.sources_practical,
        additional_instruction="Verify the platform link or text. If reservations are by phone/email or on a different platform (e.g., SevenRooms), then this claim should fail.",
    )

    # Rn_Private_Dining_8plus (verify >= 8 guests)
    n_priv = evaluator.add_leaf(
        id=f"R{rn}_Private_Dining_8plus",
        desc="Private dining availability is confirmed for groups of at least 8 guests",
        parent=practical_node,
        critical=True,
    )
    claim_priv = "Private dining is available for groups of at least 8 guests."
    await evaluator.verify(
        claim=claim_priv,
        node=n_priv,
        sources=item.sources_practical,
        additional_instruction="Confirm that private dining minimum group size is 8 or more; if the minimum is less than 8 and no larger capacity is specified, this claim should fail.",
    )

    # Rn_Dinner_Days_Listed_5plus (existence and threshold via extraction)
    dinner_days = item.practical.dinner_days if (item.practical and item.practical.dinner_days) else []
    dinner_5plus = len(dinner_days) >= 5
    evaluator.add_custom_node(
        result=dinner_5plus,
        id=f"R{rn}_Dinner_Days_Listed_5plus",
        desc="Days of the week dinner service is offered are listed AND total dinner-service days are at least 5",
        parent=practical_node,
        critical=True,
    )

    # Rn_Dietary_Accommodations_2plus (verify at least two types)
    n_diet = evaluator.add_leaf(
        id=f"R{rn}_Dietary_Accommodations_2plus",
        desc="At least two dietary accommodation types are explicitly stated from: vegetarian/vegan options, gluten-free options, allergy accommodations",
        parent=practical_node,
        critical=True,
    )
    listed_types = (item.practical.dietary_accommodations if item.practical else []) or []
    claim_diet = (
        f"The restaurant offers at least two of the following dietary accommodations: "
        f"vegetarian/vegan options, gluten-free options, allergy accommodations. "
        f"Listed: {listed_types}."
    )
    await evaluator.verify(
        claim=claim_diet,
        node=n_diet,
        sources=item.sources_practical,
        additional_instruction="Confirm at least two distinct types among vegetarian/vegan, gluten-free, and allergy accommodations. Accept reasonable synonyms.",
    )

    # Rn_Dress_Code (verify dress code policy)
    n_dress = evaluator.add_leaf(
        id=f"R{rn}_Dress_Code",
        desc="Dress code policy is stated",
        parent=practical_node,
        critical=True,
    )
    claim_dress = f"The restaurant states a dress code policy: {text_or_empty(item.practical.dress_code) if item.practical else ''}."
    await evaluator.verify(
        claim=claim_dress,
        node=n_dress,
        sources=item.sources_practical,
        additional_instruction="Look for any dress code mention (e.g., 'smart casual', 'jacket required', or 'no formal dress code'). If no dress code is stated, this claim should fail.",
    )

    # --------------------- References (existence per category) ------------ #
    refs_node = evaluator.add_parallel(
        id=f"R{rn}_References",
        desc="Provides required reference URLs (at least one per major category)",
        parent=restaurant_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(safe_list(item.sources_basic)) >= 1,
        id=f"R{rn}_Basic_Qualifications_URL",
        desc="At least one URL is provided that supports the Basic Qualifications information for this restaurant",
        parent=refs_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(safe_list(item.sources_dining)) >= 1,
        id=f"R{rn}_Dining_Experience_URL",
        desc="At least one URL is provided that supports the Dining Experience information for this restaurant",
        parent=refs_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(safe_list(item.sources_practical)) >= 1,
        id=f"R{rn}_Practical_Information_URL",
        desc="At least one URL is provided that supports the Practical Information for this restaurant",
        parent=refs_node,
        critical=True,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Michelin 2025 US restaurants task.
    """
    # Initialize evaluator (root parallel; leave root non-critical for framework consistency)
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_restaurants(),
        template_class=RestaurantsExtraction,
        extraction_name="restaurants_extraction",
    )

    # Keep only first 4 restaurants; pad if fewer
    restaurants: List[RestaurantItem] = list(extracted.restaurants[:4])
    while len(restaurants) < 4:
        restaurants.append(RestaurantItem())

    # Build per-restaurant verification subtrees
    for i in range(4):
        await verify_restaurant(evaluator, root, restaurants[i], i)

    # Geographic diversity: at least 3 different states among the 4 restaurants
    states = []
    for r in restaurants:
        s = text_or_empty(r.state).strip()
        if s:
            states.append(s.upper())
    unique_states = len(set(states))
    evaluator.add_custom_node(
        result=unique_states >= 3,
        id="Geographic_Diversity",
        desc="The 4 restaurants are located in at least 3 different US states",
        parent=root,
        critical=True,
    )

    # Optional: record allowed lists for transparency
    evaluator.add_custom_info(
        info={
            "allowed_wine_spectator_levels": ALLOWED_WINE_SPECTATOR_LEVELS,
            "allowed_reservation_platforms": ALLOWED_RESERVATION_PLATFORMS,
            "unique_states_count": unique_states,
            "states_seen": list(set(states)),
        },
        info_type="meta",
        info_name="evaluation_parameters",
    )

    # Return summary (tree + score)
    return evaluator.get_summary()