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
TASK_ID = "culinary_best_practices_4_restaurants"
TASK_DESCRIPTION = (
    "You are consulting for a prestigious culinary investment group planning to document best practices from exceptional fine dining establishments across the United States. "
    "Your task is to identify exactly 4 fine dining restaurants, each located in a different US state, that collectively demonstrate excellence across multiple operational dimensions.\n\n"
    "The 4 restaurants must meet the following specific criteria:\n\n"
    "Restaurant 1:\n"
    "- Won the James Beard Award for Best New Restaurant in 2024\n"
    "- Provide the restaurant's total seating capacity in the main dining room\n"
    "- Identify which online reservation platform the restaurant uses (OpenTable, Resy, Tock, or other)\n\n"
    "Restaurant 2:\n"
    "- Has three Michelin stars in the current (2025) or most recent (2024) Michelin Guide\n"
    "- Must be located in a different US state than Restaurant 1\n"
    "- Has a private dining room or private event space (provide the capacity)\n"
    "- Has a wine program led by a certified sommelier (provide certification level if available)\n\n"
    "Restaurant 3:\n"
    "- Has a recognized sustainability certification (such as Green Restaurant Association, LEED, MSC/ASC seafood certification, or similar environmental certification)\n"
    "- Must be located in a different US state than Restaurants 1 and 2\n"
    "- Offers a multi-course tasting menu (provide the number of courses and price per person)\n"
    "- Has a stated dress code policy (casual, business casual, smart casual, formal, or jacket required)\n\n"
    "Restaurant 4:\n"
    "- Is ADA-compliant with documented wheelchair-accessible entrance and accessibility features\n"
    "- Must be located in a different US state than Restaurants 1, 2, and 3\n"
    "- Operates dinner service (provide dinner service hours)\n"
    "- Has received at least one Michelin star, James Beard Award (winner or semifinalist in any year), or other recognized culinary award\n\n"
    "For each restaurant, provide:\n"
    "1. Restaurant name\n"
    "2. US state location\n"
    "3. All requested specific information for that restaurant category\n"
    "4. URL references supporting each claim\n\n"
    "All four restaurants must be in different US states. Provide complete documentation with URL references for all claims."
)

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class R1Fields(BaseModel):
    jbf_best_new_2024_urls: List[str] = Field(default_factory=list)
    main_dining_capacity: Optional[str] = None
    main_dining_capacity_urls: List[str] = Field(default_factory=list)
    reservation_platform: Optional[str] = None
    reservation_platform_urls: List[str] = Field(default_factory=list)


class R2Fields(BaseModel):
    michelin_stars: Optional[str] = None  # Use string for flexibility (e.g., "3", "three")
    michelin_year: Optional[str] = None   # "2024" or "2025"
    michelin_urls: List[str] = Field(default_factory=list)
    private_dining_capacity: Optional[str] = None
    private_dining_urls: List[str] = Field(default_factory=list)
    sommelier_led: Optional[str] = None   # "yes"/"no"/None
    sommelier_cert_level: Optional[str] = None  # e.g., "MS", "Advanced", "WSET Diploma"
    sommelier_urls: List[str] = Field(default_factory=list)


class R3Fields(BaseModel):
    sustainability_certification: Optional[str] = None
    sustainability_urls: List[str] = Field(default_factory=list)
    tasting_menu_courses: Optional[str] = None
    tasting_menu_price_per_person: Optional[str] = None
    tasting_menu_urls: List[str] = Field(default_factory=list)
    dress_code: Optional[str] = None
    dress_code_urls: List[str] = Field(default_factory=list)


class R4Fields(BaseModel):
    ada_compliant_note: Optional[str] = None  # free text, e.g. "Wheelchair accessible entrance"
    ada_urls: List[str] = Field(default_factory=list)
    dinner_hours: Optional[str] = None
    dinner_hours_urls: List[str] = Field(default_factory=list)
    award_note: Optional[str] = None  # e.g., "Michelin 1 star 2024" / "JBF semifinalist 2023"
    award_urls: List[str] = Field(default_factory=list)


class RestaurantItem(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    category: Optional[int] = None  # 1..4 indicating which restaurant role
    r1: Optional[R1Fields] = None
    r2: Optional[R2Fields] = None
    r3: Optional[R3Fields] = None
    r4: Optional[R4Fields] = None


class RestaurantsExtraction(BaseModel):
    restaurants: List[RestaurantItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurants() -> str:
    return """
    Extract up to four fine dining restaurants and their details exactly as presented in the answer. 
    Map each to the required category roles 1 through 4 (Restaurant 1..4). If the answer lists more than four, select the first four. 
    If fewer than four are present, return only those found (other entries can be omitted or returned with null fields).

    For each restaurant, extract:
    - name: Restaurant name as given
    - state: The US state as given (full name or abbreviation as presented)
    - location_urls: Array of URL(s) explicitly cited in the answer that support the location/address/state
    - category: Integer 1..4 indicating which category the restaurant is intended to satisfy (Restaurant 1, 2, 3, or 4)
    
    For Restaurant 1 fields (use object 'r1'):
    - jbf_best_new_2024_urls: Array of URL(s) supporting the claim that it won the James Beard Award for Best New Restaurant in 2024
    - main_dining_capacity: String with the total seating capacity of the main dining room as stated in the answer (keep text exactly as presented)
    - main_dining_capacity_urls: Array of URL(s) supporting the capacity
    - reservation_platform: String indicating reservation platform (OpenTable, Resy, Tock, or other) as stated
    - reservation_platform_urls: Array of URL(s) supporting the platform used

    For Restaurant 2 fields (use object 'r2'):
    - michelin_stars: String for the star count as presented (e.g., "3", "three")
    - michelin_year: String "2024" or "2025" if the answer specifies the guide year; else null
    - michelin_urls: Array of URL(s) supporting the 3-star rating and guide year
    - private_dining_capacity: String capacity for private dining/event space as presented
    - private_dining_urls: Array of URL(s) supporting existence and capacity
    - sommelier_led: String "yes" if the wine program is led by a certified sommelier, "no" otherwise, or null if not specified
    - sommelier_cert_level: String certification level if provided (e.g., "Master Sommelier (MS)", "Advanced Sommelier", "WSET Diploma"), else null
    - sommelier_urls: Array of URL(s) supporting the sommelier leadership and/or certification

    For Restaurant 3 fields (use object 'r3'):
    - sustainability_certification: String name/acronym of the certification (e.g., "GRA", "LEED", "MSC", "ASC")
    - sustainability_urls: Array of URL(s) supporting the certification
    - tasting_menu_courses: String for number of courses as presented (e.g., "8 courses")
    - tasting_menu_price_per_person: String for price per person as presented (e.g., "$175")
    - tasting_menu_urls: Array of URL(s) supporting both the number of courses and price
    - dress_code: String for stated dress code (e.g., casual, business casual, smart casual, formal, jacket required) as presented
    - dress_code_urls: Array of URL(s) supporting the dress code

    For Restaurant 4 fields (use object 'r4'):
    - ada_compliant_note: String noting wheelchair-accessible entrance and ADA accessibility features as presented
    - ada_urls: Array of URL(s) supporting ADA accessibility documentation
    - dinner_hours: String for dinner service hours as presented
    - dinner_hours_urls: Array of URL(s) supporting dinner hours
    - award_note: String describing at least one recognized culinary award (e.g., Michelin star, James Beard Award winner or semifinalist, etc.)
    - award_urls: Array of URL(s) supporting the award recognition

    Output JSON:
    {
      "restaurants": [
        {
          "name": ...,
          "state": ...,
          "location_urls": [...],
          "category": 1|2|3|4,
          "r1": { ... } or null,
          "r2": { ... } or null,
          "r3": { ... } or null,
          "r4": { ... } or null
        },
        ...
      ]
    }

    Rules:
    - Do not invent any values. Only extract information explicitly mentioned in the answer.
    - For any required URL lists, include only URLs explicitly present in the answer (plain or markdown-format).
    - If a field is missing, set it to null (for primitive) or [] (for URL arrays).
    - Prefer state strings exactly as presented (e.g., "CA" or "California").
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_restaurant_1(evaluator: Evaluator, parent_node, item: RestaurantItem) -> None:
    rnode = evaluator.add_parallel(
        id="Restaurant_1",
        desc="Restaurant 1 meets all Restaurant 1-specific constraints and provides supporting URLs.",
        parent=parent_node,
        critical=False
    )
    # Name & State with URL (existence + verification)
    exists_name_state = evaluator.add_custom_node(
        result=_non_empty_str(item.name) and _non_empty_str(item.state) and _has_urls(item.location_urls),
        id="R1_Name_State_URL_Provided",
        desc="Restaurant 1 name/state and a supporting location URL are provided.",
        parent=rnode,
        critical=True
    )
    name_state_leaf = evaluator.add_leaf(
        id="R1_Name_And_State_With_URL",
        desc="Provide Restaurant 1 name and US state, with a URL supporting the stated location.",
        parent=rnode,
        critical=True
    )
    claim_loc = f"The restaurant '{item.name}' is located in the US state of {item.state}."
    await evaluator.verify(
        claim=claim_loc,
        node=name_state_leaf,
        sources=item.location_urls,
        additional_instruction="Confirm that the provided URL(s) show the restaurant's location in the stated US state. Accept official site pages, authoritative listings with address. Allow state abbreviations or full names.",
        extra_prerequisites=[exists_name_state]
    )

    # James Beard Award Best New Restaurant 2024
    jbf_exist = evaluator.add_custom_node(
        result=_has_urls(item.r1.jbf_best_new_2024_urls) if item.r1 else False,
        id="R1_JBF_URL_Provided",
        desc="URLs supporting James Beard Best New Restaurant 2024 are provided for Restaurant 1.",
        parent=rnode,
        critical=True
    )
    jbf_leaf = evaluator.add_leaf(
        id="R1_JamesBeard_BestNewRestaurant_2024_With_URL",
        desc="Restaurant 1 won the James Beard Award for Best New Restaurant in 2024, with a URL supporting this claim.",
        parent=rnode,
        critical=True
    )
    claim_jbf = f"The restaurant '{item.name}' won the James Beard Award for Best New Restaurant in 2024."
    await evaluator.verify(
        claim=claim_jbf,
        node=jbf_leaf,
        sources=(item.r1.jbf_best_new_2024_urls if item.r1 else []),
        additional_instruction="Verify that the URL(s) explicitly identify the restaurant as the James Beard Foundation Best New Restaurant winner for 2024 (not nominee/semifinalist). Official JBF pages or authoritative press are acceptable.",
        extra_prerequisites=[jbf_exist, exists_name_state]
    )

    # Main dining room seating capacity
    cap_exist = evaluator.add_custom_node(
        result=(item.r1 is not None and _non_empty_str(item.r1.main_dining_capacity) and _has_urls(item.r1.main_dining_capacity_urls)),
        id="R1_MainDining_Capacity_Info_Provided",
        desc="Main dining room capacity and supporting URL are provided for Restaurant 1.",
        parent=rnode,
        critical=True
    )
    cap_leaf = evaluator.add_leaf(
        id="R1_MainDining_SeatingCapacity_With_URL",
        desc="Provide total seating capacity in the main dining room for Restaurant 1, with a URL supporting the capacity.",
        parent=rnode,
        critical=True
    )
    claim_cap = f"The main dining room seating capacity of '{item.name}' is {item.r1.main_dining_capacity if item.r1 else ''}."
    await evaluator.verify(
        claim=claim_cap,
        node=cap_leaf,
        sources=(item.r1.main_dining_capacity_urls if item.r1 else []),
        additional_instruction="Confirm that the URL(s) explicitly state the total seating capacity for the main dining room. Minor formatting or wording differences are acceptable if the capacity matches.",
        extra_prerequisites=[cap_exist, exists_name_state]
    )

    # Reservation platform
    res_exist = evaluator.add_custom_node(
        result=(item.r1 is not None and _non_empty_str(item.r1.reservation_platform) and _has_urls(item.r1.reservation_platform_urls)),
        id="R1_ReservationPlatform_Info_Provided",
        desc="Reservation platform and supporting URL are provided for Restaurant 1.",
        parent=rnode,
        critical=True
    )
    res_leaf = evaluator.add_leaf(
        id="R1_ReservationPlatform_With_URL",
        desc="Identify Restaurant 1's online reservation platform (OpenTable, Resy, Tock, or other), with a URL supporting the platform used.",
        parent=rnode,
        critical=True
    )
    claim_res = f"The restaurant '{item.name}' uses {item.r1.reservation_platform if item.r1 else ''} for online reservations."
    await evaluator.verify(
        claim=claim_res,
        node=res_leaf,
        sources=(item.r1.reservation_platform_urls if item.r1 else []),
        additional_instruction="Confirm the platform used (OpenTable, Resy, Tock, or others) from the URL(s). Accept official booking pages, platform listing pages, or the restaurant's reservation page referencing the platform.",
        extra_prerequisites=[res_exist, exists_name_state]
    )


async def verify_restaurant_2(evaluator: Evaluator, parent_node, item: RestaurantItem) -> None:
    rnode = evaluator.add_parallel(
        id="Restaurant_2",
        desc="Restaurant 2 meets all Restaurant 2-specific constraints and provides supporting URLs.",
        parent=parent_node,
        critical=False
    )
    # Name & State
    exists_name_state = evaluator.add_custom_node(
        result=_non_empty_str(item.name) and _non_empty_str(item.state) and _has_urls(item.location_urls),
        id="R2_Name_State_URL_Provided",
        desc="Restaurant 2 name/state and a supporting location URL are provided.",
        parent=rnode,
        critical=True
    )
    name_state_leaf = evaluator.add_leaf(
        id="R2_Name_And_State_With_URL",
        desc="Provide Restaurant 2 name and US state, with a URL supporting the stated location.",
        parent=rnode,
        critical=True
    )
    claim_loc = f"The restaurant '{item.name}' is located in the US state of {item.state}."
    await evaluator.verify(
        claim=claim_loc,
        node=name_state_leaf,
        sources=item.location_urls,
        additional_instruction="Confirm that the provided URL(s) show the restaurant's location in the stated US state. Allow state abbreviations or full names.",
        extra_prerequisites=[exists_name_state]
    )

    # Michelin three stars (2024 or 2025)
    mic_exist = evaluator.add_custom_node(
        result=(item.r2 is not None and _has_urls(item.r2.michelin_urls)),
        id="R2_Michelin_URL_Provided",
        desc="URLs supporting three Michelin stars (2024/2025) are provided for Restaurant 2.",
        parent=rnode,
        critical=True
    )
    mic_leaf = evaluator.add_leaf(
        id="R2_ThreeMichelinStars_2024or2025_With_URL",
        desc="Restaurant 2 has exactly three Michelin stars in the current (2025) or most recent (2024) Michelin Guide, with a URL supporting the star rating and guide year.",
        parent=rnode,
        critical=True
    )
    year_text = item.r2.michelin_year if (item.r2 and _non_empty_str(item.r2.michelin_year)) else "2024 or 2025"
    claim_mic = f"The restaurant '{item.name}' has exactly three Michelin stars in the {year_text} Michelin Guide."
    await evaluator.verify(
        claim=claim_mic,
        node=mic_leaf,
        sources=(item.r2.michelin_urls if item.r2 else []),
        additional_instruction="Confirm on Michelin's official site or authoritative sources that the restaurant holds three stars in either the 2024 or 2025 guide.",
        extra_prerequisites=[mic_exist, exists_name_state]
    )

    # Private dining room capacity
    pdr_exist = evaluator.add_custom_node(
        result=(item.r2 is not None and _non_empty_str(item.r2.private_dining_capacity) and _has_urls(item.r2.private_dining_urls)),
        id="R2_PrivateDining_Info_Provided",
        desc="Private dining room/event space capacity and supporting URL are provided for Restaurant 2.",
        parent=rnode,
        critical=True
    )
    pdr_leaf = evaluator.add_leaf(
        id="R2_PrivateDining_And_Capacity_With_URL",
        desc="Restaurant 2 has a private dining room or private event space and provides its capacity, with a URL supporting both existence and capacity.",
        parent=rnode,
        critical=True
    )
    claim_pdr = f"The restaurant '{item.name}' has a private dining/event space with capacity {item.r2.private_dining_capacity if item.r2 else ''}."
    await evaluator.verify(
        claim=claim_pdr,
        node=pdr_leaf,
        sources=(item.r2.private_dining_urls if item.r2 else []),
        additional_instruction="Confirm both existence of private dining/event space and the capacity number from the provided URL(s).",
        extra_prerequisites=[pdr_exist, exists_name_state]
    )

    # Wine program led by a certified sommelier
    somm_exist = evaluator.add_custom_node(
        result=(item.r2 is not None and _non_empty_str(item.r2.sommelier_led) and item.r2.sommelier_led.strip().lower() == "yes" and _has_urls(item.r2.sommelier_urls)),
        id="R2_SommelierLed_Info_Provided",
        desc="Certified sommelier-led wine program claim and supporting URL are provided for Restaurant 2.",
        parent=rnode,
        critical=True
    )
    somm_leaf = evaluator.add_leaf(
        id="R2_CertifiedSommelierLedWineProgram_With_URL",
        desc="Restaurant 2 has a wine program led by a certified sommelier, with a URL supporting this claim.",
        parent=rnode,
        critical=True
    )
    claim_somm = f"The wine program at '{item.name}' is led by a certified sommelier."
    await evaluator.verify(
        claim=claim_somm,
        node=somm_leaf,
        sources=(item.r2.sommelier_urls if item.r2 else []),
        additional_instruction="Look for explicit references to certifications (CMS: Certified/Advanced/Master Sommelier, WSET Diploma, etc.) or titles verifying certified leadership of the wine program.",
        extra_prerequisites=[somm_exist, exists_name_state]
    )

    # Optional: Sommelier certification level if available (non-critical)
    somm_level_result = (item.r2 is not None and _non_empty_str(item.r2.sommelier_cert_level) and _has_urls(item.r2.sommelier_urls))
    evaluator.add_custom_node(
        result=somm_level_result,
        id="R2_Sommelier_CertificationLevel_IfAvailable",
        desc="Provide the sommelier certification level if available (as stated in the question).",
        parent=rnode,
        critical=False
    )


async def verify_restaurant_3(evaluator: Evaluator, parent_node, item: RestaurantItem) -> None:
    rnode = evaluator.add_parallel(
        id="Restaurant_3",
        desc="Restaurant 3 meets all Restaurant 3-specific constraints and provides supporting URLs.",
        parent=parent_node,
        critical=False
    )
    # Name & State
    exists_name_state = evaluator.add_custom_node(
        result=_non_empty_str(item.name) and _non_empty_str(item.state) and _has_urls(item.location_urls),
        id="R3_Name_State_URL_Provided",
        desc="Restaurant 3 name/state and a supporting location URL are provided.",
        parent=rnode,
        critical=True
    )
    name_state_leaf = evaluator.add_leaf(
        id="R3_Name_And_State_With_URL",
        desc="Provide Restaurant 3 name and US state, with a URL supporting the stated location.",
        parent=rnode,
        critical=True
    )
    claim_loc = f"The restaurant '{item.name}' is located in the US state of {item.state}."
    await evaluator.verify(
        claim=claim_loc,
        node=name_state_leaf,
        sources=item.location_urls,
        additional_instruction="Confirm that the provided URL(s) show the restaurant's location in the stated US state. Allow state abbreviations or full names.",
        extra_prerequisites=[exists_name_state]
    )

    # Sustainability certification
    sust_exist = evaluator.add_custom_node(
        result=(item.r3 is not None and _non_empty_str(item.r3.sustainability_certification) and _has_urls(item.r3.sustainability_urls)),
        id="R3_Sustainability_Info_Provided",
        desc="Sustainability certification and supporting URL are provided for Restaurant 3.",
        parent=rnode,
        critical=True
    )
    sust_leaf = evaluator.add_leaf(
        id="R3_SustainabilityCertification_With_URL",
        desc="Restaurant 3 has a recognized sustainability certification (e.g., GRA, LEED, MSC/ASC, or similar), with a URL supporting the certification.",
        parent=rnode,
        critical=True
    )
    claim_sust = f"The restaurant '{item.name}' has a recognized sustainability certification: {item.r3.sustainability_certification if item.r3 else ''}."
    await evaluator.verify(
        claim=claim_sust,
        node=sust_leaf,
        sources=(item.r3.sustainability_urls if item.r3 else []),
        additional_instruction="Confirm the certification (e.g., Green Restaurant Association, LEED, MSC/ASC) explicitly from the URL(s).",
        extra_prerequisites=[sust_exist, exists_name_state]
    )

    # Tasting menu - courses and price
    tm_exist = evaluator.add_custom_node(
        result=(item.r3 is not None and _non_empty_str(item.r3.tasting_menu_courses) and _non_empty_str(item.r3.tasting_menu_price_per_person) and _has_urls(item.r3.tasting_menu_urls)),
        id="R3_TastingMenu_Info_Provided",
        desc="Tasting menu courses/price and supporting URL are provided for Restaurant 3.",
        parent=rnode,
        critical=True
    )
    tm_leaf = evaluator.add_leaf(
        id="R3_TastingMenu_Courses_And_Price_With_URL",
        desc="Restaurant 3 offers a multi-course tasting menu and provides number of courses and price per person, with a URL supporting these details.",
        parent=rnode,
        critical=True
    )
    claim_tm = f"The restaurant '{item.name}' offers a multi-course tasting menu of {item.r3.tasting_menu_courses if item.r3 else ''}, priced at {item.r3.tasting_menu_price_per_person if item.r3 else ''} per person."
    await evaluator.verify(
        claim=claim_tm,
        node=tm_leaf,
        sources=(item.r3.tasting_menu_urls if item.r3 else []),
        additional_instruction="Confirm both number of courses and price per person from the URL(s). Accept reasonable variants such as ranges or 'starting at' if clearly stated.",
        extra_prerequisites=[tm_exist, exists_name_state]
    )

    # Dress code
    dc_exist = evaluator.add_custom_node(
        result=(item.r3 is not None and _non_empty_str(item.r3.dress_code) and _has_urls(item.r3.dress_code_urls)),
        id="R3_DressCode_Info_Provided",
        desc="Dress code and supporting URL are provided for Restaurant 3.",
        parent=rnode,
        critical=True
    )
    dc_leaf = evaluator.add_leaf(
        id="R3_DressCode_With_URL",
        desc="Restaurant 3 has a stated dress code policy, with a URL supporting the dress code.",
        parent=rnode,
        critical=True
    )
    claim_dc = f"The restaurant '{item.name}' has a stated dress code: {item.r3.dress_code if item.r3 else ''}."
    await evaluator.verify(
        claim=claim_dc,
        node=dc_leaf,
        sources=(item.r3.dress_code_urls if item.r3 else []),
        additional_instruction="Confirm the dress code category (casual, business casual, smart casual, formal, jacket required) from the URL(s). Accept close synonyms if they clearly correspond.",
        extra_prerequisites=[dc_exist, exists_name_state]
    )


async def verify_restaurant_4(evaluator: Evaluator, parent_node, item: RestaurantItem) -> None:
    rnode = evaluator.add_parallel(
        id="Restaurant_4",
        desc="Restaurant 4 meets all Restaurant 4-specific constraints and provides supporting URLs.",
        parent=parent_node,
        critical=False
    )
    # Name & State
    exists_name_state = evaluator.add_custom_node(
        result=_non_empty_str(item.name) and _non_empty_str(item.state) and _has_urls(item.location_urls),
        id="R4_Name_State_URL_Provided",
        desc="Restaurant 4 name/state and a supporting location URL are provided.",
        parent=rnode,
        critical=True
    )
    name_state_leaf = evaluator.add_leaf(
        id="R4_Name_And_State_With_URL",
        desc="Provide Restaurant 4 name and US state, with a URL supporting the stated location.",
        parent=rnode,
        critical=True
    )
    claim_loc = f"The restaurant '{item.name}' is located in the US state of {item.state}."
    await evaluator.verify(
        claim=claim_loc,
        node=name_state_leaf,
        sources=item.location_urls,
        additional_instruction="Confirm that the provided URL(s) show the restaurant's location in the stated US state. Allow state abbreviations or full names.",
        extra_prerequisites=[exists_name_state]
    )

    # ADA compliance: documented wheelchair-accessible entrance and features
    ada_exist = evaluator.add_custom_node(
        result=(item.r4 is not None and _has_urls(item.r4.ada_urls)),
        id="R4_ADA_URL_Provided",
        desc="ADA accessibility documentation URL(s) are provided for Restaurant 4.",
        parent=rnode,
        critical=True
    )
    ada_leaf = evaluator.add_leaf(
        id="R4_ADA_Compliance_AccessibleEntrance_And_Features_With_URL",
        desc="Restaurant 4 is ADA-compliant with documented wheelchair-accessible entrance and accessibility features, with a URL supporting these accessibility claims.",
        parent=rnode,
        critical=True
    )
    claim_ada = f"The restaurant '{item.name}' has a wheelchair-accessible entrance and ADA accessibility features."
    await evaluator.verify(
        claim=claim_ada,
        node=ada_leaf,
        sources=(item.r4.ada_urls if item.r4 else []),
        additional_instruction="Verify that the URL(s) explicitly mention a wheelchair-accessible entrance and other ADA accessibility features (e.g., accessible restrooms, seating, ramps).",
        extra_prerequisites=[ada_exist, exists_name_state]
    )

    # Dinner service hours
    dh_exist = evaluator.add_custom_node(
        result=(item.r4 is not None and _non_empty_str(item.r4.dinner_hours) and _has_urls(item.r4.dinner_hours_urls)),
        id="R4_DinnerHours_Info_Provided",
        desc="Dinner service hours and supporting URL are provided for Restaurant 4.",
        parent=rnode,
        critical=True
    )
    dh_leaf = evaluator.add_leaf(
        id="R4_DinnerServiceHours_With_URL",
        desc="Restaurant 4 operates dinner service and provides dinner service hours, with a URL supporting the hours.",
        parent=rnode,
        critical=True
    )
    claim_dh = f"The dinner service hours for '{item.name}' are {item.r4.dinner_hours if item.r4 else ''}."
    await evaluator.verify(
        claim=claim_dh,
        node=dh_leaf,
        sources=(item.r4.dinner_hours_urls if item.r4 else []),
        additional_instruction="Confirm the dinner hours from the URL(s). Accept minor formatting differences or ranges.",
        extra_prerequisites=[dh_exist, exists_name_state]
    )

    # Culinary award (Michelin star, JBF award, or other recognized award)
    aw_exist = evaluator.add_custom_node(
        result=(item.r4 is not None and _non_empty_str(item.r4.award_note) and _has_urls(item.r4.award_urls)),
        id="R4_Award_Info_Provided",
        desc="Recognized culinary award and supporting URL are provided for Restaurant 4.",
        parent=rnode,
        critical=True
    )
    aw_leaf = evaluator.add_leaf(
        id="R4_CulinaryAward_With_URL",
        desc="Restaurant 4 has received at least one Michelin star, James Beard Award (winner or semifinalist), or other recognized culinary award, with a URL supporting the recognition.",
        parent=rnode,
        critical=True
    )
    claim_aw = f"The restaurant '{item.name}' has received a recognized culinary award: {item.r4.award_note if item.r4 else ''}."
    await evaluator.verify(
        claim=claim_aw,
        node=aw_leaf,
        sources=(item.r4.award_urls if item.r4 else []),
        additional_instruction="Confirm at least one recognized award (e.g., Michelin star(s), James Beard Award winner/semifinalist) explicitly from the URL(s).",
        extra_prerequisites=[aw_exist, exists_name_state]
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
    Evaluate an answer for the 4-restaurant best practices task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root node aggregates parallel checks
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

    # Extract structured restaurant data
    extraction = await evaluator.extract(
        prompt=prompt_extract_restaurants(),
        template_class=RestaurantsExtraction,
        extraction_name="restaurants_extraction",
    )

    # Select up to 4 restaurants, in order
    selected: List[RestaurantItem] = extraction.restaurants[:4]
    # Pad to 4 if fewer are provided (to keep structure consistent)
    while len(selected) < 4:
        selected.append(RestaurantItem())

    # Top-level critical checks
    # 1) Exactly 4 restaurants provided (allow answers listing more, but we evaluate first 4)
    #    We consider this passed if the answer provided at least 4 identifiable restaurants (name+state).
    valid_count_in_answer = sum(1 for r in extraction.restaurants if _non_empty_str(r.name) and _non_empty_str(r.state))
    evaluator.add_custom_node(
        result=(valid_count_in_answer >= 4),
        id="Restaurant_Count",
        desc="Exactly 4 restaurants are provided (Restaurant 1–4).",
        parent=root,
        critical=True
    )

    # 2) State uniqueness across selected four
    states = [r.state.strip().lower() for r in selected if _non_empty_str(r.state)]
    state_unique = (len(states) == 4 and len(set(states)) == 4)
    evaluator.add_custom_node(
        result=state_unique,
        id="State_Uniqueness",
        desc="All four restaurants are located in four different US states.",
        parent=root,
        critical=True
    )

    # Build per-restaurant verification subtrees
    # Restaurant 1
    await verify_restaurant_1(evaluator, root, selected[0])

    # Restaurant 2
    await verify_restaurant_2(evaluator, root, selected[1])

    # Restaurant 3
    await verify_restaurant_3(evaluator, root, selected[2])

    # Restaurant 4
    await verify_restaurant_4(evaluator, root, selected[3])

    # Add custom info about selection for transparency
    evaluator.add_custom_info(
        info={
            "selected_restaurants_snapshot": [
                {
                    "name": r.name,
                    "state": r.state,
                    "category": r.category
                } for r in selected
            ],
            "total_extracted": len(extraction.restaurants),
            "valid_count_in_answer": valid_count_in_answer,
            "states_unique": state_unique
        },
        info_type="selection_info"
    )

    # Return evaluation summary
    return evaluator.get_summary()