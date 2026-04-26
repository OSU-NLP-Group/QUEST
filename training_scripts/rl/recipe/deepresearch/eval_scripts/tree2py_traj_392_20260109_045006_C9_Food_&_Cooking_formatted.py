import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "jbf_2024_restaurants_four"
TASK_DESCRIPTION = """
Identify four distinct restaurants that won specific 2024 James Beard Foundation Restaurant and Chef Awards. For each restaurant, provide the following information:

1. Restaurant that won Outstanding Restaurant (2024):
   - Restaurant name and complete address (street, city, state)
   - Seating capacity with supporting URL reference
   - Chef/owner name and information about their culinary background
   - Primary cuisine type and dining format
   - URL reference confirming the award

2. Restaurant that won Best New Restaurant (2024):
   - Restaurant name and complete address (street, city, state)
   - Chef/owner name with information about their cultural heritage or culinary inspiration, supported by a URL reference
   - Primary cuisine style and menu format description
   - URL reference confirming the award

3. Restaurant of the chef who won Outstanding Chef (2024):
   - Chef's full name and restaurant name
   - City and neighborhood/district location
   - Year the restaurant opened with supporting URL reference
   - Primary cuisine or cultural influence and notable cooking technique
   - URL reference confirming the award

4. Restaurant of a chef who won Best Chef in any regional category (2024):
   - Chef's full name, restaurant name, and the specific James Beard regional category won
   - City and state location
   - Primary cuisine style with supporting URL reference
   - Any additional 2024 awards or recognition (if applicable) with URL reference
   - URL reference confirming the Best Chef award

All four restaurants must be different establishments, and all information must be verifiable through the provided URL references.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class R1OutstandingRestaurant(BaseModel):
    restaurant_name: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    seating_capacity: Optional[str] = None
    seating_capacity_urls: List[str] = Field(default_factory=list)

    chef_owner_name: Optional[str] = None
    culinary_background: Optional[str] = None

    primary_cuisine: Optional[str] = None
    dining_format: Optional[str] = None

    award_urls: List[str] = Field(default_factory=list)


class R2BestNewRestaurant(BaseModel):
    restaurant_name: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    chef_owner_name: Optional[str] = None
    heritage_or_inspiration: Optional[str] = None
    heritage_urls: List[str] = Field(default_factory=list)

    primary_cuisine_style: Optional[str] = None
    menu_format: Optional[str] = None

    award_urls: List[str] = Field(default_factory=list)


class R3OutstandingChefRestaurant(BaseModel):
    chef_full_name: Optional[str] = None
    restaurant_name: Optional[str] = None
    city: Optional[str] = None
    neighborhood: Optional[str] = None

    opening_year: Optional[str] = None
    opening_year_urls: List[str] = Field(default_factory=list)

    primary_cuisine_or_influence: Optional[str] = None
    notable_technique: Optional[str] = None

    award_urls: List[str] = Field(default_factory=list)


class R4BestChefRegionalRestaurant(BaseModel):
    chef_full_name: Optional[str] = None
    restaurant_name: Optional[str] = None
    regional_category: Optional[str] = None

    city: Optional[str] = None
    state: Optional[str] = None

    cuisine_style: Optional[str] = None
    cuisine_urls: List[str] = Field(default_factory=list)

    additional_2024_recognition: Optional[str] = None
    additional_2024_recognition_urls: List[str] = Field(default_factory=list)

    award_urls: List[str] = Field(default_factory=list)


class JBF2024Extraction(BaseModel):
    outstanding_restaurant: Optional[R1OutstandingRestaurant] = None
    best_new_restaurant: Optional[R2BestNewRestaurant] = None
    outstanding_chef_restaurant: Optional[R3OutstandingChefRestaurant] = None
    best_chef_regional_restaurant: Optional[R4BestChefRegionalRestaurant] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_jbf_2024() -> str:
    return """
    Extract exactly four entries from the answer, each corresponding to these fixed categories:
    1) outstanding_restaurant (the 2024 Outstanding Restaurant winner),
    2) best_new_restaurant (the 2024 Best New Restaurant winner),
    3) outstanding_chef_restaurant (the restaurant of the chef who won 2024 Outstanding Chef),
    4) best_chef_regional_restaurant (the restaurant of a chef who won a 2024 Best Chef regional category).

    For each category, extract the following fields as strings (URLs as arrays of strings). If a field is missing, set it to null; if a URL list is missing, return an empty array.

    For outstanding_restaurant:
    - restaurant_name
    - street
    - city
    - state
    - seating_capacity
    - seating_capacity_urls (array of URLs specifically supporting the capacity)
    - chef_owner_name
    - culinary_background
    - primary_cuisine
    - dining_format
    - award_urls (array of URLs confirming the Outstanding Restaurant (2024) win)

    For best_new_restaurant:
    - restaurant_name
    - street
    - city
    - state
    - chef_owner_name
    - heritage_or_inspiration
    - heritage_urls (array of URLs supporting the heritage/inspiration claim)
    - primary_cuisine_style
    - menu_format
    - award_urls (array of URLs confirming the Best New Restaurant (2024) win)

    For outstanding_chef_restaurant:
    - chef_full_name
    - restaurant_name
    - city
    - neighborhood
    - opening_year
    - opening_year_urls (array of URLs supporting the opening year)
    - primary_cuisine_or_influence
    - notable_technique
    - award_urls (array of URLs confirming the chef won Outstanding Chef in 2024)

    For best_chef_regional_restaurant:
    - chef_full_name
    - restaurant_name
    - regional_category (e.g., "Best Chef: Texas")
    - city
    - state
    - cuisine_style
    - cuisine_urls (array of URLs supporting the cuisine style)
    - additional_2024_recognition (if any other 2024 awards/recognition are explicitly claimed in the answer, summarize them here; else null)
    - additional_2024_recognition_urls (array of URLs supporting the additional recognition if claimed; else empty)
    - award_urls (array of URLs confirming the chef won the stated Best Chef regional award in 2024)

    SPECIAL RULES FOR URL FIELDS:
    - Extract only URLs explicitly present in the answer (including markdown links).
    - If a URL is missing a protocol, prepend http://.
    - Do not fabricate URLs.

    Return a single JSON object with four top-level keys:
    - outstanding_restaurant
    - best_new_restaurant
    - outstanding_chef_restaurant
    - best_chef_regional_restaurant
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_digits(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r"\d", text))


def is_four_digit_year(text: Optional[str]) -> bool:
    if not text:
        return False
    m = re.search(r"\b(19|20)\d{2}\b", text.strip())
    return m is not None


def normalize_restaurant_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    s = re.sub(r"\s+", " ", s)  # collapse spaces
    return s


def filter_valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if not re.match(r"^https?://", u):
            u = "http://" + u
        out.append(u)
    return out


async def verify_with_required_urls(
    evaluator: Evaluator,
    node,
    claim: str,
    urls: Optional[List[str]],
    additional_instruction: str,
) -> bool:
    url_list = filter_valid_urls(urls)
    if len(url_list) == 0:
        node.score = 0.0
        node.status = "failed"
        return False
    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=url_list,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_global_constraints(
    evaluator: Evaluator,
    parent,
    data: JBF2024Extraction,
) -> None:
    gnode = evaluator.add_parallel(
        id="global_requirements",
        desc="Global constraints across all four restaurant entries",
        parent=parent,
        critical=True,
    )

    # exactly_four_required_categories_present
    r1_present = bool(
        data.outstanding_restaurant
        and data.outstanding_restaurant.restaurant_name
        and data.outstanding_restaurant.restaurant_name.strip()
    )
    r2_present = bool(
        data.best_new_restaurant
        and data.best_new_restaurant.restaurant_name
        and data.best_new_restaurant.restaurant_name.strip()
    )
    r3_present = bool(
        data.outstanding_chef_restaurant
        and data.outstanding_chef_restaurant.chef_full_name
        and data.outstanding_chef_restaurant.chef_full_name.strip()
        and data.outstanding_chef_restaurant.restaurant_name
        and data.outstanding_chef_restaurant.restaurant_name.strip()
    )
    r4_present = bool(
        data.best_chef_regional_restaurant
        and data.best_chef_regional_restaurant.chef_full_name
        and data.best_chef_regional_restaurant.chef_full_name.strip()
        and data.best_chef_regional_restaurant.restaurant_name
        and data.best_chef_regional_restaurant.restaurant_name.strip()
        and data.best_chef_regional_restaurant.regional_category
        and data.best_chef_regional_restaurant.regional_category.strip()
    )
    evaluator.add_custom_node(
        result=(r1_present and r2_present and r3_present and r4_present),
        id="exactly_four_required_categories_present",
        desc="Response includes exactly four restaurant entries corresponding to: Outstanding Restaurant (2024), Best New Restaurant (2024), restaurant of Outstanding Chef (2024), and restaurant of a Best Chef regional winner (2024)",
        parent=gnode,
        critical=True,
    )

    # all_restaurants_distinct
    names = [
        (data.outstanding_restaurant.restaurant_name if data.outstanding_restaurant else None),
        (data.best_new_restaurant.restaurant_name if data.best_new_restaurant else None),
        (data.outstanding_chef_restaurant.restaurant_name if data.outstanding_chef_restaurant else None),
        (data.best_chef_regional_restaurant.restaurant_name if data.best_chef_regional_restaurant else None),
    ]
    norm_names = [normalize_restaurant_name(n) for n in names]
    all_present = all(n is not None and n.strip() for n in names)
    unique = len({n for n in norm_names if n}) == 4
    evaluator.add_custom_node(
        result=(all_present and unique),
        id="all_restaurants_distinct",
        desc="All four restaurants are different establishments (no restaurant appears in multiple categories)",
        parent=gnode,
        critical=True,
    )


async def build_r1_outstanding_restaurant(
    evaluator: Evaluator,
    parent,
    r1: Optional[R1OutstandingRestaurant],
) -> None:
    node = evaluator.add_parallel(
        id="restaurant_1_outstanding_restaurant",
        desc="Outstanding Restaurant (2024) winner restaurant entry",
        parent=parent,
        critical=False,
    )

    # r1_name
    evaluator.add_custom_node(
        result=bool(r1 and r1.restaurant_name and r1.restaurant_name.strip()),
        id="r1_name",
        desc="Restaurant name provided",
        parent=node,
        critical=True,
    )

    # r1_complete_address
    evaluator.add_custom_node(
        result=bool(
            r1
            and r1.street and r1.street.strip()
            and r1.city and r1.city.strip()
            and r1.state and r1.state.strip()
        ),
        id="r1_complete_address",
        desc="Complete address provided (street number/name, city, state)",
        parent=node,
        critical=True,
    )

    # r1_seating_capacity
    cap_node = evaluator.add_parallel(
        id="r1_seating_capacity",
        desc="Numeric seating capacity provided with supporting URL reference",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(r1 and r1.seating_capacity and has_digits(r1.seating_capacity)),
        id="r1_capacity_value",
        desc="Numeric seating capacity is stated",
        parent=cap_node,
        critical=True,
    )

    cap_url_leaf = evaluator.add_leaf(
        id="r1_capacity_url",
        desc="URL reference supports the seating capacity value",
        parent=cap_node,
        critical=True,
    )
    if r1:
        await verify_with_required_urls(
            evaluator,
            cap_url_leaf,
            claim=f"The seating capacity of {r1.restaurant_name or 'the restaurant'} is stated as '{r1.seating_capacity}'.",
            urls=r1.seating_capacity_urls,
            additional_instruction="Verify the page(s) explicitly state the seating capacity (allow minor wording variations, ranges, or 'seats about N'). If no URL is provided, the result should be False.",
        )

    # r1_chef_owner_and_background
    evaluator.add_custom_node(
        result=bool(
            r1
            and r1.chef_owner_name and r1.chef_owner_name.strip()
            and r1.culinary_background and r1.culinary_background.strip()
        ),
        id="r1_chef_owner_and_background",
        desc="Chef/owner name and culinary background/training are provided",
        parent=node,
        critical=True,
    )

    # r1_cuisine_and_format
    evaluator.add_custom_node(
        result=bool(
            r1
            and r1.primary_cuisine and r1.primary_cuisine.strip()
            and r1.dining_format and r1.dining_format.strip()
        ),
        id="r1_cuisine_and_format",
        desc="Primary cuisine type and dining format are described",
        parent=node,
        critical=True,
    )

    # r1_award_confirmation_url
    award_leaf = evaluator.add_leaf(
        id="r1_award_confirmation_url",
        desc="URL reference confirms the restaurant won Outstanding Restaurant (2024)",
        parent=node,
        critical=True,
    )
    if r1:
        await verify_with_required_urls(
            evaluator,
            award_leaf,
            claim=f"This page confirms that {r1.restaurant_name or 'the restaurant'} won the James Beard Foundation Outstanding Restaurant award in 2024.",
            urls=r1.award_urls,
            additional_instruction="Confirm the page explicitly lists the restaurant as the 2024 Outstanding Restaurant winner (allow minor wording variations like 'Outstanding Restaurant Winner' or 'JBF Awards 2024'). If unrelated or no URL, return False.",
        )


async def build_r2_best_new_restaurant(
    evaluator: Evaluator,
    parent,
    r2: Optional[R2BestNewRestaurant],
) -> None:
    node = evaluator.add_parallel(
        id="restaurant_2_best_new_restaurant",
        desc="Best New Restaurant (2024) winner restaurant entry",
        parent=parent,
        critical=False,
    )

    # r2_name
    evaluator.add_custom_node(
        result=bool(r2 and r2.restaurant_name and r2.restaurant_name.strip()),
        id="r2_name",
        desc="Restaurant name provided",
        parent=node,
        critical=True,
    )

    # r2_complete_address
    evaluator.add_custom_node(
        result=bool(
            r2
            and r2.street and r2.street.strip()
            and r2.city and r2.city.strip()
            and r2.state and r2.state.strip()
        ),
        id="r2_complete_address",
        desc="Complete address provided (street, city, state)",
        parent=node,
        critical=True,
    )

    # r2_chef_heritage_or_inspiration (parallel)
    ch_node = evaluator.add_parallel(
        id="r2_chef_heritage_or_inspiration",
        desc="Chef/owner name plus cultural heritage or culinary inspiration, supported by a URL reference",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(r2 and r2.chef_owner_name and r2.chef_owner_name.strip()),
        id="r2_chef_name",
        desc="Chef/owner name is provided",
        parent=ch_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(r2 and r2.heritage_or_inspiration and r2.heritage_or_inspiration.strip()),
        id="r2_heritage_or_inspiration",
        desc="Cultural heritage or culinary inspiration is described",
        parent=ch_node,
        critical=True,
    )
    heritage_leaf = evaluator.add_leaf(
        id="r2_supporting_url",
        desc="URL reference supports the heritage/inspiration claim",
        parent=ch_node,
        critical=True,
    )
    if r2:
        await verify_with_required_urls(
            evaluator,
            heritage_leaf,
            claim=f"The source states the chef's cultural heritage or culinary inspiration as: '{r2.heritage_or_inspiration}'.",
            urls=r2.heritage_urls,
            additional_instruction="Verify the page explicitly mentions the chef's cultural heritage or culinary inspiration consistent with the provided description. If no relevant URL, return False.",
        )

    # r2_cuisine_and_menu_format
    evaluator.add_custom_node(
        result=bool(
            r2
            and r2.primary_cuisine_style and r2.primary_cuisine_style.strip()
            and r2.menu_format and r2.menu_format.strip()
        ),
        id="r2_cuisine_and_menu_format",
        desc="Primary cuisine style and menu format are described",
        parent=node,
        critical=True,
    )

    # r2_award_confirmation_url
    award_leaf = evaluator.add_leaf(
        id="r2_award_confirmation_url",
        desc="URL reference confirms the restaurant won Best New Restaurant (2024)",
        parent=node,
        critical=True,
    )
    if r2:
        await verify_with_required_urls(
            evaluator,
            award_leaf,
            claim=f"This page confirms that {r2.restaurant_name or 'the restaurant'} won the James Beard Foundation Best New Restaurant award in 2024.",
            urls=r2.award_urls,
            additional_instruction="Confirm the page explicitly lists the restaurant as the 2024 Best New Restaurant winner (wording variations acceptable). If no URL, return False.",
        )


async def build_r3_outstanding_chef_restaurant(
    evaluator: Evaluator,
    parent,
    r3: Optional[R3OutstandingChefRestaurant],
) -> None:
    node = evaluator.add_parallel(
        id="restaurant_3_outstanding_chef_restaurant",
        desc="Restaurant of the Outstanding Chef (2024) winner",
        parent=parent,
        critical=False,
    )

    # r3_chef_and_restaurant (parallel)
    cr_node = evaluator.add_parallel(
        id="r3_chef_and_restaurant",
        desc="Chef full name and restaurant name are provided",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(r3 and r3.chef_full_name and r3.chef_full_name.strip()),
        id="r3_chef_name",
        desc="Chef full name is provided",
        parent=cr_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(r3 and r3.restaurant_name and r3.restaurant_name.strip()),
        id="r3_restaurant_name",
        desc="Restaurant name is provided",
        parent=cr_node,
        critical=True,
    )

    # r3_city_and_neighborhood
    evaluator.add_custom_node(
        result=bool(
            r3
            and r3.city and r3.city.strip()
            and r3.neighborhood and r3.neighborhood.strip()
        ),
        id="r3_city_and_neighborhood",
        desc="City and neighborhood/district location are provided",
        parent=node,
        critical=True,
    )

    # r3_opening_year_with_url (parallel)
    oy_node = evaluator.add_parallel(
        id="r3_opening_year_with_url",
        desc="Year the restaurant opened is provided with a supporting URL reference",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(r3 and r3.opening_year and is_four_digit_year(r3.opening_year)),
        id="r3_opening_year",
        desc="Opening year is stated",
        parent=oy_node,
        critical=True,
    )
    oy_leaf = evaluator.add_leaf(
        id="r3_opening_year_url",
        desc="URL reference supports the opening year",
        parent=oy_node,
        critical=True,
    )
    if r3:
        await verify_with_required_urls(
            evaluator,
            oy_leaf,
            claim=f"The restaurant {r3.restaurant_name or 'the restaurant'} opened in {r3.opening_year}.",
            urls=r3.opening_year_urls,
            additional_instruction="Verify the page explicitly states the opening year (allow wording like 'opened in 2014', 'since 2014'). If no URL, return False.",
        )

    # r3_cuisine_and_technique
    evaluator.add_custom_node(
        result=bool(
            r3
            and r3.primary_cuisine_or_influence and r3.primary_cuisine_or_influence.strip()
            and r3.notable_technique and r3.notable_technique.strip()
        ),
        id="r3_cuisine_and_technique",
        desc="Primary cuisine/cultural influence and a notable cooking technique/method are described",
        parent=node,
        critical=True,
    )

    # r3_award_confirmation_url
    award_leaf = evaluator.add_leaf(
        id="r3_award_confirmation_url",
        desc="URL reference confirms the chef won Outstanding Chef (2024)",
        parent=node,
        critical=True,
    )
    if r3:
        await verify_with_required_urls(
            evaluator,
            award_leaf,
            claim=f"This page confirms that {r3.chef_full_name or 'the chef'} won the James Beard Foundation Outstanding Chef award in 2024.",
            urls=r3.award_urls,
            additional_instruction="Confirm the page lists the person as the 2024 Outstanding Chef winner (allow minor wording variations). If no URL, return False.",
        )


async def build_r4_best_chef_regional_restaurant(
    evaluator: Evaluator,
    parent,
    r4: Optional[R4BestChefRegionalRestaurant],
) -> None:
    node = evaluator.add_parallel(
        id="restaurant_4_best_chef_regional_restaurant",
        desc="Restaurant of a Best Chef regional category winner (2024)",
        parent=parent,
        critical=False,
    )

    # r4_chef_restaurant_and_category
    evaluator.add_custom_node(
        result=bool(
            r4
            and r4.chef_full_name and r4.chef_full_name.strip()
            and r4.restaurant_name and r4.restaurant_name.strip()
            and r4.regional_category and r4.regional_category.strip()
        ),
        id="r4_chef_restaurant_and_category",
        desc="Chef full name, restaurant name, and the specific Best Chef regional category are provided",
        parent=node,
        critical=True,
    )

    # r4_city_and_state
    evaluator.add_custom_node(
        result=bool(
            r4
            and r4.city and r4.city.strip()
            and r4.state and r4.state.strip()
        ),
        id="r4_city_and_state",
        desc="City and state location are provided",
        parent=node,
        critical=True,
    )

    # r4_cuisine_with_url (parallel)
    cu_node = evaluator.add_parallel(
        id="r4_cuisine_with_url",
        desc="Primary cuisine style is described with a supporting URL reference",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(r4 and r4.cuisine_style and r4.cuisine_style.strip()),
        id="r4_cuisine_description",
        desc="Cuisine style is described",
        parent=cu_node,
        critical=True,
    )
    cuisine_leaf = evaluator.add_leaf(
        id="r4_cuisine_url",
        desc="URL reference supports the cuisine description",
        parent=cu_node,
        critical=True,
    )
    if r4:
        await verify_with_required_urls(
            evaluator,
            cuisine_leaf,
            claim=f"The primary cuisine style of {r4.restaurant_name or 'the restaurant'} is described as '{r4.cuisine_style}'.",
            urls=r4.cuisine_urls,
            additional_instruction="Verify the page describes the restaurant's cuisine style consistent with the provided description. Allow synonyms (e.g., 'New American' vs 'American, New'). If no URL, return False.",
        )

    # r4_additional_2024_recognition_if_applicable (non-critical)
    # Pass if no claim; If claimed, require at least one URL (content verification is not required by rubric).
    add_rec_ok = True
    if r4 and r4.additional_2024_recognition:
        add_rec_ok = len(filter_valid_urls(r4.additional_2024_recognition_urls)) > 0
    evaluator.add_custom_node(
        result=bool(add_rec_ok),
        id="r4_additional_2024_recognition_if_applicable",
        desc="If additional 2024 awards/recognition are claimed, provide a URL reference",
        parent=node,
        critical=False,
    )

    # r4_award_confirmation_url
    award_leaf = evaluator.add_leaf(
        id="r4_award_confirmation_url",
        desc="URL reference confirms the chef won the stated Best Chef regional award (2024)",
        parent=node,
        critical=True,
    )
    if r4:
        await verify_with_required_urls(
            evaluator,
            award_leaf,
            claim=f"This page confirms that {r4.chef_full_name or 'the chef'} won the 2024 James Beard Foundation Best Chef award for the region/category '{r4.regional_category}'.",
            urls=r4.award_urls,
            additional_instruction="Confirm the page lists the chef as a 2024 Best Chef winner for the specified regional category (e.g., 'Best Chef: Texas'). If no URL, return False.",
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
    Evaluate an answer for the JBF 2024 four-restaurants awards task and return a structured result.
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

    # Extract structured information from the answer
    extracted: JBF2024Extraction = await evaluator.extract(
        prompt=prompt_extract_jbf_2024(),
        template_class=JBF2024Extraction,
        extraction_name="jbf_2024_extraction",
    )

    # Build tree according to rubric
    # 1) Global constraints (critical)
    await build_global_constraints(evaluator, root, extracted)

    # 2) Category: Outstanding Restaurant (2024)
    await build_r1_outstanding_restaurant(evaluator, root, extracted.outstanding_restaurant)

    # 3) Category: Best New Restaurant (2024)
    await build_r2_best_new_restaurant(evaluator, root, extracted.best_new_restaurant)

    # 4) Category: Outstanding Chef (2024) - Restaurant of the chef
    await build_r3_outstanding_chef_restaurant(evaluator, root, extracted.outstanding_chef_restaurant)

    # 5) Category: Best Chef (regional) - Restaurant of a regional winner
    await build_r4_best_chef_regional_restaurant(evaluator, root, extracted.best_chef_regional_restaurant)

    # Return structured evaluation summary
    return evaluator.get_summary()