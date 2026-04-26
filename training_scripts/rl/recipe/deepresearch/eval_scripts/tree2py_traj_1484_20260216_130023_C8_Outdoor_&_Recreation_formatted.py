import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cancun_family_vacation_mlk_2026"
TASK_DESCRIPTION = """
My family of four (two parents, a 4-year-old, and a 10-year-old) wants to plan a vacation to Cancun, Mexico during Martin Luther King Jr. Day weekend 2026 (January 17-19, arriving Friday and departing Monday). We'll be flying from Bangor, Maine. Please help us plan this trip by providing: (1) Resort Recommendation: Identify one all-inclusive resort in the Cancun/Riviera Maya area that meets ALL of the following requirements: has an on-site water park with water slides, offers a supervised kids club that accepts children starting at age 4 and serves children up to at least age 10, and provides family-friendly accommodations and dining. (2) Flight Information: Confirm that Breeze Airways operates routes that would allow us to travel from Bangor, ME to Cancun for these dates, and provide relevant details about the flight options (including whether connections are required). (3) Activities: Briefly describe what age-appropriate water park activities and kids club programs would be available for both our 4-year-old and 10-year-old at the resort you recommend. For each major component (resort and flights), please provide reference URLs that support your recommendations.
"""

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ResortExtraction(BaseModel):
    resort_name: Optional[str] = None
    resort_urls: List[str] = Field(default_factory=list)


class FlightExtraction(BaseModel):
    flight_urls: List[str] = Field(default_factory=list)
    dates_text: Optional[str] = None
    connection_info_text: Optional[str] = None
    baggage_policy_text: Optional[str] = None
    airline_mentioned: Optional[str] = None


class ActivitiesExtraction(BaseModel):
    water_park_for_4yo_desc: Optional[str] = None
    water_park_for_10yo_desc: Optional[str] = None
    kids_club_schedule_desc: Optional[str] = None
    additional_resort_activities: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_resort_info() -> str:
    return """
    Extract the resort recommendation details from the answer.

    Required fields:
    - resort_name: The specific resort name recommended (e.g., "Hyatt Ziva Cancun", "Iberostar Paraiso Lindo").
    - resort_urls: A list of all URLs cited in the answer that specifically refer to the recommended resort
      (official site, brand page, or authoritative listings like major OTAs). Only include URLs explicitly present in the answer.

    If the resort name is not given, set resort_name to null.
    If no URLs are provided, return an empty array for resort_urls.
    """


def prompt_extract_flight_info() -> str:
    return """
    Extract the flight information and references from the answer.

    Required fields:
    - flight_urls: A list of all URLs cited in the answer that support the flight information
      (e.g., Breeze Airways route map or airport pages, booking pages, airline announcements, or credible travel sites).
    - dates_text: The exact text snippet stating the proposed travel dates (e.g., “January 17–19, 2026” or
      “arrive Friday, Jan 17 and depart Monday, Jan 19”).
    - connection_info_text: The exact text snippet where the answer explains whether the flight is direct or requires connections.
      If not mentioned, set to null.
    - baggage_policy_text: The exact text snippet where the answer mentions relevant baggage policy details (personal item,
      carry-on, checked bags). If not mentioned, set to null.
    - airline_mentioned: The airline named for the flights (e.g., “Breeze Airways”). If not specified, set to null.

    Only extract URLs explicitly present in the answer.
    """


def prompt_extract_activities_info() -> str:
    return """
    Extract the activity planning details from the answer.

    Required fields:
    - water_park_for_4yo_desc: A concise snippet describing age-appropriate water park activities for a 4-year-old at the recommended resort.
      If not provided, set to null.
    - water_park_for_10yo_desc: A concise snippet describing age-appropriate water park activities for a 10-year-old at the recommended resort.
      If not provided, set to null.
    - kids_club_schedule_desc: A snippet mentioning kids club hours or programming (e.g., age ranges, schedules, themes).
      If not mentioned, set to null.
    - additional_resort_activities: A list of other family activities mentioned at the resort (e.g., beach, pools, sports, entertainment).
      If none are mentioned, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_resort(
    evaluator: Evaluator,
    parent_node,
    resort: ResortExtraction,
) -> None:
    """
    Build and verify the Resort Identification subtree.
    Note: To satisfy framework constraints (critical parent cannot have non-critical children),
    this parent node is non-critical, while individual essential checks are marked critical.
    """
    resort_node = evaluator.add_parallel(
        id="Resort_Identification",
        desc="Identify a Cancun all-inclusive resort that meets all family requirements",
        parent=parent_node,
        critical=False,
    )

    # Existence checks (custom) to gate URL-based verifications
    name_provided_node = evaluator.add_custom_node(
        result=bool(resort.resort_name and resort.resort_name.strip()),
        id="Resort_Name_Provided",
        desc="A specific resort name is provided",
        parent=resort_node,
        critical=True,
    )
    urls_provided_node = evaluator.add_custom_node(
        result=bool(resort.resort_urls and len(resort.resort_urls) > 0),
        id="Resort_URLs_Provided",
        desc="At least one resort reference URL is provided",
        parent=resort_node,
        critical=True,
    )

    # Validate the reference URL(s) correspond to the resort
    reference_leaf = evaluator.add_leaf(
        id="Resort_Reference_URL",
        desc="Valid reference URL provided for the identified resort",
        parent=resort_node,
        critical=True,
    )
    if resort.resort_name:
        await evaluator.verify(
            claim=f"This webpage is about the resort named '{resort.resort_name}'.",
            node=reference_leaf,
            sources=resort.resort_urls,
            additional_instruction="Accept official resort sites, brand pages, or credible OTA pages clearly about the named resort.",
        )
    else:
        # If no resort name, skip verification by marking dependency failure through existing checks
        reference_leaf.score = 0.0
        reference_leaf.status = "skipped"

    # Prepare convenience variables
    resort_name = resort.resort_name or "the resort"
    resort_sources = resort.resort_urls

    # All-inclusive status (critical)
    all_inclusive_leaf = evaluator.add_leaf(
        id="All_Inclusive_Status",
        desc="Resort is confirmed as all-inclusive",
        parent=resort_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The resort {resort_name} is an all-inclusive resort (meals and drinks included).",
        node=all_inclusive_leaf,
        sources=resort_sources,
        additional_instruction="Look for explicit terms like 'all-inclusive', 'all inclusive', 'AI', or a meal plan including food and drinks.",
    )

    # On-site water park with slides (critical)
    water_park_leaf = evaluator.add_leaf(
        id="Water_Park_Presence",
        desc="Resort has an on-site water park with slides",
        parent=resort_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The resort {resort_name} has an on-site water park with water slides (also called an aquapark).",
        node=water_park_leaf,
        sources=resort_sources,
        additional_instruction="Look for 'water park', 'aqua park', 'splash park', and presence of slides on official or credible pages.",
    )

    # Optional features (non-critical)
    lazy_river_leaf = evaluator.add_leaf(
        id="Lazy_River_Feature",
        desc="Water park includes a lazy river",
        parent=resort_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The water park at {resort_name} includes a lazy river.",
        node=lazy_river_leaf,
        sources=resort_sources,
        additional_instruction="Confirm the presence of a 'lazy river' feature if available; otherwise this item can fail without affecting critical checks.",
    )

    splash_zone_leaf = evaluator.add_leaf(
        id="Kids_Splash_Zone",
        desc="Dedicated splash zone or water playground for young children",
        parent=resort_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The resort {resort_name} offers a splash zone or water playground suitable for young children.",
        node=splash_zone_leaf,
        sources=resort_sources,
        additional_instruction="Accept terms like 'splash pad', 'splash zone', 'kids water playground', or 'spray park'.",
    )

    # Kids club availability (critical)
    kids_club_leaf = evaluator.add_leaf(
        id="Kids_Club_Available",
        desc="Resort offers a supervised kids club program",
        parent=resort_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The resort {resort_name} offers a supervised kids club program.",
        node=kids_club_leaf,
        sources=resort_sources,
        additional_instruction="Look for terms like 'kids club', 'children's club', 'supervised program', or similar.",
    )

    # Kids club minimum age 4 (critical)
    age4_leaf = evaluator.add_leaf(
        id="Kids_Club_Age_4_Minimum",
        desc="Kids club accepts children starting at age 4",
        parent=resort_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The kids club at {resort_name} accepts children starting at age 4 (or younger ages are supervised differently but 4-year-olds are accepted).",
        node=age4_leaf,
        sources=resort_sources,
        additional_instruction="Accept phrasing like 'ages 4-12', '4 to 12', or 'from age 4'. If minimum age is 5 or higher, this should fail.",
    )

    # Kids club serves up to ≥10 (critical)
    age_upper_leaf = evaluator.add_leaf(
        id="Kids_Club_Age_Range_Upper",
        desc="Kids club serves children up to at least age 10",
        parent=resort_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The kids club at {resort_name} serves children up to at least age 10.",
        node=age_upper_leaf,
        sources=resort_sources,
        additional_instruction="Accept ranges that include 10 (e.g., '4-12', '5-12'). If the maximum is below 10, fail.",
    )

    # More non-critical resort amenities
    dining_leaf = evaluator.add_leaf(
        id="Multiple_Dining_Options",
        desc="Resort offers multiple restaurant options",
        parent=resort_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The resort {resort_name} has multiple restaurants (two or more).",
        node=dining_leaf,
        sources=resort_sources,
        additional_instruction="Look for a restaurant count or mention of several a la carte and buffet venues.",
    )

    beach_access_leaf = evaluator.add_leaf(
        id="Beach_Access",
        desc="Resort provides direct beach access",
        parent=resort_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The resort {resort_name} is beachfront with direct beach access.",
        node=beach_access_leaf,
        sources=resort_sources,
        additional_instruction="Accept 'beachfront', 'on the beach', or equivalent phrasing.",
    )

    non_motor_leaf = evaluator.add_leaf(
        id="Non_Motorized_Water_Sports",
        desc="Non-motorized water sports (kayaking, snorkeling, etc.) included",
        parent=resort_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"Non-motorized water sports are included for guests at {resort_name}.",
        node=non_motor_leaf,
        sources=resort_sources,
        additional_instruction="Accept mentions like kayaks, paddle boards, snorkeling gear included as part of the stay or all-inclusive plan.",
    )

    family_pools_leaf = evaluator.add_leaf(
        id="Family_Friendly_Pools",
        desc="Multiple swimming pools suitable for families",
        parent=resort_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The resort {resort_name} has multiple family-friendly swimming pools.",
        node=family_pools_leaf,
        sources=resort_sources,
        additional_instruction="Look for 'family pool', 'multiple pools', or similar descriptions.",
    )

    # Location in Cancun/Riviera Maya (critical)
    location_leaf = evaluator.add_leaf(
        id="Resort_Location_Cancun_Area",
        desc="Resort is located in Cancun or Riviera Maya area",
        parent=resort_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The resort {resort_name} is located in the Cancun or Riviera Maya area of Quintana Roo, Mexico.",
        node=location_leaf,
        sources=resort_sources,
        additional_instruction="Accept 'Cancun', 'Hotel Zone', 'Riviera Maya', 'Playa del Carmen', 'Puerto Morelos', or similar within Quintana Roo.",
    )


async def verify_flights(
    evaluator: Evaluator,
    parent_node,
    flights: FlightExtraction,
) -> None:
    """
    Build and verify the Flight Arrangements subtree.
    This parent is non-critical to satisfy framework constraints; essential checks within are marked critical.
    """
    flights_node = evaluator.add_parallel(
        id="Flight_Arrangements",
        desc="Verify flight availability and provide booking details for MLK weekend 2026",
        parent=parent_node,
        critical=False,
    )

    # Existence of URLs (gate URL verifications)
    flight_urls_exist = evaluator.add_custom_node(
        result=bool(flights.flight_urls and len(flights.flight_urls) > 0),
        id="Flight_URLs_Provided",
        desc="At least one flight reference URL is provided",
        parent=flights_node,
        critical=True,
    )

    # At least one flight reference URL is relevant
    flight_ref_leaf = evaluator.add_leaf(
        id="Flight_Reference_URL",
        desc="Valid reference URL provided for flight information",
        parent=flights_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This webpage provides flight or airline route information relevant to traveling between Bangor (BGR) and Cancun (CUN) or Breeze Airways' network.",
        node=flight_ref_leaf,
        sources=flights.flight_urls,
        additional_instruction="Accept airline route maps, airport pages, booking pages, or credible travel sites clearly referencing Breeze, BGR, or CUN.",
    )

    # Breeze serves Bangor (critical)
    breeze_from_bgr = evaluator.add_leaf(
        id="Breeze_Airways_Route_Confirmed",
        desc="Confirm Breeze Airways flies from Bangor, ME",
        parent=flights_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Breeze Airways operates service from Bangor, Maine (BGR).",
        node=breeze_from_bgr,
        sources=flights.flight_urls,
        additional_instruction="Look for Breeze route maps, airport service lists, or announcements that explicitly include BGR.",
    )

    # Cancun available in Breeze network (critical)
    breeze_to_cun = evaluator.add_leaf(
        id="Cancun_Destination_Available",
        desc="Cancun is available as a destination from Breeze Airways network",
        parent=flights_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Breeze Airways operates flights to Cancun (CUN), Mexico.",
        node=breeze_to_cun,
        sources=flights.flight_urls,
        additional_instruction="Verify destination lists, route maps, or pages indicating Cancun (CUN) service by Breeze.",
    )

    # Dates alignment with MLK weekend (critical per rubric, but we check via simple content verification)
    dates_leaf = evaluator.add_leaf(
        id="MLK_Weekend_Travel_Dates",
        desc="Flight dates align with MLK weekend 2026 (January 17-19)",
        parent=flights_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer aligns travel dates with MLK weekend 2026: arriving Friday, January 17, 2026 and departing Monday, January 19, 2026.",
        node=dates_leaf,
        sources=None,
        additional_instruction="Check the answer text for these exact dates or an equivalent clear statement (e.g., 'Jan 17–19, 2026').",
    )

    # Non-critical: whether the answer mentions direct vs connections
    connections_leaf = evaluator.add_custom_node(
        result=bool(flights.connection_info_text and flights.connection_info_text.strip()),
        id="Connection_Information",
        desc="Information provided about whether flight is direct or requires connections",
        parent=flights_node,
        critical=False,
    )

    # Non-critical: baggage policy mentioned
    baggage_leaf = evaluator.add_custom_node(
        result=bool(flights.baggage_policy_text and flights.baggage_policy_text.strip()),
        id="Baggage_Policy_Mentioned",
        desc="Relevant baggage policy information (personal item, carry-on, checked bags) mentioned for family travel",
        parent=flights_node,
        critical=False,
    )


async def verify_activities(
    evaluator: Evaluator,
    parent_node,
    acts: ActivitiesExtraction,
) -> None:
    """
    Build and verify the Activity Planning subtree.
    All items here are treated as non-critical, focusing on presence/quality in the answer text.
    """
    activities_node = evaluator.add_parallel(
        id="Activity_Planning",
        desc="Plan age-appropriate activities for both children during the vacation",
        parent=parent_node,
        critical=False,
    )

    # 4-year-old water park activities
    water4_leaf = evaluator.add_leaf(
        id="Water_Park_Activities_4yr",
        desc="Identify suitable water park activities for 4-year-old",
        parent=activities_node,
        critical=False,
    )
    await evaluator.verify(
        claim="The answer includes age-appropriate water park activities for a 4-year-old at the recommended resort.",
        node=water4_leaf,
        additional_instruction="Look for mentions like splash pad/zone, shallow pools, gentle/small slides, or similar toddler-friendly features.",
        sources=None,
    )

    # 10-year-old water park activities
    water10_leaf = evaluator.add_leaf(
        id="Water_Park_Activities_10yr",
        desc="Identify suitable water park activities for 10-year-old",
        parent=activities_node,
        critical=False,
    )
    await evaluator.verify(
        claim="The answer includes age-appropriate water park activities for a 10-year-old at the recommended resort.",
        node=water10_leaf,
        additional_instruction="Look for bigger slides, multi-lane slides, water playgrounds for older kids, wave pools, or similar age-appropriate items.",
        sources=None,
    )

    # Kids club schedule/programming info
    kids_schedule_leaf = evaluator.add_leaf(
        id="Kids_Club_Schedule_Info",
        desc="Information about kids club hours or programming",
        parent=activities_node,
        critical=False,
    )
    await evaluator.verify(
        claim="The answer mentions kids club hours or programming schedule/details.",
        node=kids_schedule_leaf,
        additional_instruction="Check for operating hours, age ranges, themed activities, or program descriptions.",
        sources=None,
    )

    # Additional resort family activities
    addl_activities_leaf = evaluator.add_leaf(
        id="Additional_Resort_Activities",
        desc="Mention other family activities available at resort (beach, sports, entertainment)",
        parent=activities_node,
        critical=False,
    )
    await evaluator.verify(
        claim="The answer mentions other family activities at the resort such as beach time, sports, shows, or entertainment.",
        node=addl_activities_leaf,
        additional_instruction="Look for examples like beach activities, pools, mini-golf, tennis, kids shows, or family entertainment.",
        sources=None,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate a single answer for the Cancun family vacation planning task.
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

    # Run extractions (in parallel)
    resort_extraction_task = evaluator.extract(
        prompt=prompt_extract_resort_info(),
        template_class=ResortExtraction,
        extraction_name="resort_extraction",
    )
    flight_extraction_task = evaluator.extract(
        prompt=prompt_extract_flight_info(),
        template_class=FlightExtraction,
        extraction_name="flight_extraction",
    )
    activities_extraction_task = evaluator.extract(
        prompt=prompt_extract_activities_info(),
        template_class=ActivitiesExtraction,
        extraction_name="activities_extraction",
    )

    resort_info, flight_info, activities_info = await asyncio.gather(
        resort_extraction_task, flight_extraction_task, activities_extraction_task
    )

    # Build verification subtrees
    await verify_resort(evaluator, root, resort_info)
    await verify_flights(evaluator, root, flight_info)
    await verify_activities(evaluator, root, activities_info)

    return evaluator.get_summary()