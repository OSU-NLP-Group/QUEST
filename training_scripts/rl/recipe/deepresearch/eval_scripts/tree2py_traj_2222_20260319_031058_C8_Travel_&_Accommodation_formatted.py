import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "ca_trip_senior_pass_planning"
TASK_DESCRIPTION = (
    "A retired couple (both age 65) from Bangor, Maine is planning a week-long California vacation in March 2026. "
    "They want to spend several days in San Diego and visit a nearby national park that charges entrance fees. "
    "They're considering purchasing an America the Beautiful Senior Pass but want to verify it would be cost-effective for their trip.\n\n"
    "Please provide comprehensive planning information for their trip, including:\n\n"
    "1. National Park Selection: Identify a national park located within 200 miles of San Diego that charges entrance fees and is covered by the America the Beautiful Pass.\n"
    "2. Cost Analysis: Calculate whether purchasing a Senior Lifetime Pass ($80) would provide cost savings compared to paying individual entrance fees if they plan to visit the park twice during their week-long stay.\n"
    "3. Accommodation: Recommend a hotel located in or near the San Diego Harbor district.\n"
    "4. Travel Logistics: Confirm that direct flights are available from Bangor International Airport to reach the San Diego area.\n"
    "5. March Conditions: Verify that March weather conditions in San Diego (average temperatures and sunset times) are suitable for outdoor park activities and evening harbor visits.\n\n"
    "Your response should include: the specific national park name, entrance fee details, distance from San Diego, Senior Pass cost-benefit analysis, hotel recommendation with location details, flight availability confirmation, and March weather/sunset time verification."
)


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class ParkDetails(BaseModel):
    park_name: Optional[str] = None
    distance_text: Optional[str] = None
    fee_details: Optional[str] = None
    pass_coverage_statement: Optional[str] = None
    park_urls: List[str] = Field(default_factory=list)
    fee_urls: List[str] = Field(default_factory=list)
    pass_urls: List[str] = Field(default_factory=list)
    distance_urls: List[str] = Field(default_factory=list)


class SeniorPassCostAnalysis(BaseModel):
    senior_pass_price_text: Optional[str] = None
    two_visit_fee_total_text: Optional[str] = None
    cost_effectiveness_conclusion_text: Optional[str] = None
    pass_info_urls: List[str] = Field(default_factory=list)
    calc_urls: List[str] = Field(default_factory=list)


class AccommodationInfo(BaseModel):
    hotel_name: Optional[str] = None
    hotel_location_details: Optional[str] = None
    hotel_urls: List[str] = Field(default_factory=list)


class FlightInfo(BaseModel):
    direct_flight_statement: Optional[str] = None  # Expect an explicit yes/no statement in the answer
    flight_urls: List[str] = Field(default_factory=list)


class MarchConditions(BaseModel):
    march_high_avg_text: Optional[str] = None
    march_low_avg_text: Optional[str] = None
    march_sunset_times_text: Optional[str] = None
    suitability_statement_text: Optional[str] = None
    weather_urls: List[str] = Field(default_factory=list)
    sunset_urls: List[str] = Field(default_factory=list)


class TripPlanExtraction(BaseModel):
    park: Optional[ParkDetails] = None
    cost: Optional[SeniorPassCostAnalysis] = None
    accommodation: Optional[AccommodationInfo] = None
    flight: Optional[FlightInfo] = None
    march: Optional[MarchConditions] = None


# -----------------------------------------------------------------------------
# Extraction prompts
# -----------------------------------------------------------------------------
def prompt_extract_trip_plan() -> str:
    return """
Extract the following structured information exactly as presented in the answer. Do not invent information.

1) Park details (a national park within 200 miles of San Diego that charges an entrance fee and is covered by the America the Beautiful Pass):
- park_name: the specific national park named in the answer (must be a U.S. National Park with 'National Park' designation, not other NPS unit types)
- distance_text: the distance from San Diego to the selected park (as stated; include units if present)
- fee_details: the entrance fee details as stated (amount and basis such as per-vehicle 7-day pass, per-person, etc.)
- pass_coverage_statement: the statement that the park's entrance fees are covered by the America the Beautiful Pass (if present in the answer)
- park_urls: list of any URLs specifically about the park used as sources in the answer
- fee_urls: list of any URLs specifically supporting the park’s entrance fees (can overlap with park_urls if applicable)
- pass_urls: list of any URLs supporting America the Beautiful Pass coverage at the park (can be NPS/USGS pass page or the park page)
- distance_urls: list of any URLs used to support distance claims (e.g., map links)

2) Senior pass cost analysis:
- senior_pass_price_text: the stated price for the Senior Lifetime Pass (should be $80 if correctly stated)
- two_visit_fee_total_text: the stated total cost of paying individual entrance fees for two visits for the couple (2 people), based on the selected park’s fee structure and validity periods
- cost_effectiveness_conclusion_text: the explicit conclusion whether buying the pass costs less, the same, or more than paying individual fees for two visits (e.g., 'the pass saves money', 'about the same', 'not cost-effective')
- pass_info_urls: list of URLs supporting the Senior Pass details/pricing (official NPS/USGS preferred)
- calc_urls: list of URLs used to support the entrance fee basis used in the calculation (can overlap with fee_urls)

3) Accommodation:
- hotel_name: a specific hotel recommended
- hotel_location_details: the location info as stated (e.g., address/neighborhood/area), demonstrating proximity to San Diego Harbor/Embarcadero/waterfront
- hotel_urls: list of URLs that support the hotel and location details (official site or map/listing pages preferred)

4) Flight availability:
- direct_flight_statement: an explicit 'yes' or 'no' statement on whether nonstop/direct flights exist from Bangor International Airport (BGR) to the San Diego area airport(s) (e.g., SAN). Do not include general discussion of connections; extract the explicit determination text.
- flight_urls: list of URLs used to support the direct/nonstop availability determination (airline or aggregator pages)

5) March conditions:
- march_high_avg_text: stated average high temperature for March in San Diego (text as given; include units if provided)
- march_low_avg_text: stated average low temperature for March in San Diego
- march_sunset_times_text: stated typical sunset time info/range in March (e.g., 'around 6:00–7:00 PM')
- suitability_statement_text: explicit conclusion in the answer that (a) outdoor park activities and (b) evening harbor visits are suitable in March, and that references the given temperatures and sunset times as justification
- weather_urls: list of URLs supporting March temperature data (e.g., NOAA, WeatherSpark, Climate normals, etc.)
- sunset_urls: list of URLs supporting March sunset times (e.g., timeanddate.com)

If any field is missing in the answer, return null for that field (empty list for URL lists). Do not infer or add information.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def non_empty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def combine_unique_urls(*lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for u in lst:
            if isinstance(u, str):
                url = u.strip()
                if url and url not in seen:
                    combined.append(url)
                    seen.add(url)
    return combined


def normalize_yes_no(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.strip().lower()
    if t in {"yes", "y", "true", "available", "exists"}:
        return "yes"
    if t in {"no", "n", "false", "not available", "does not exist", "not exist", "none"}:
        return "no"
    # try to catch phrases like "no direct flights", "there are no nonstops"
    if "no direct" in t or "no nonstop" in t or "without nonstop" in t:
        return "no"
    if "direct flight" in t or "nonstop flight" in t:
        # ambiguous, but leaning to yes if not prefixed by 'no'
        if "no " not in t and "not " not in t:
            return "yes"
    return s  # return original if cannot normalize


def mark_failed(node, reason: str, logger: Optional[logging.Logger] = None):
    node.score = 0.0
    node.status = "failed"
    if logger:
        logger.debug(f"Marking node {node.id} failed due to: {reason}")


# -----------------------------------------------------------------------------
# Verification subtrees
# -----------------------------------------------------------------------------
async def verify_national_park_and_cost_analysis(
    evaluator: Evaluator,
    parent_node,
    park: Optional[ParkDetails],
    cost: Optional[SeniorPassCostAnalysis],
    logger: logging.Logger,
):
    # national_park_and_cost_analysis (sequential, critical)
    np_and_cost_node = evaluator.add_sequential(
        id="national_park_and_cost_analysis",
        desc="Provide a qualifying national park near San Diego and analyze Senior Pass cost-effectiveness for two visits",
        parent=parent_node,
        critical=True,
    )

    # ---- 1) National park details (parallel, critical) ----
    np_details_node = evaluator.add_parallel(
        id="national_park_details",
        desc="Provide required details for a national park within 200 miles of San Diego that charges an entrance fee and is covered by the America the Beautiful Pass",
        parent=np_and_cost_node,
        critical=True,
    )

    # park_name leaf
    park_name_leaf = evaluator.add_leaf(
        id="park_name",
        desc="State the specific national park name (must be an actual U.S. National Park unit designated as a National Park)",
        parent=np_details_node,
        critical=True,
    )
    if not park or not non_empty(park.park_name):
        mark_failed(park_name_leaf, "Missing park_name", logger)
    else:
        claim = f"'{park.park_name}' is an official U.S. National Park (designation 'National Park') within the National Park System."
        sources = combine_unique_urls(park.park_urls)
        await evaluator.verify(
            claim=claim,
            node=park_name_leaf,
            sources=sources if sources else None,
            additional_instruction="Verify the park is a U.S. National Park (designation exactly 'National Park', not monument, seashore, etc.). Prefer NPS official pages.",
        )

    # distance_from_san_diego leaf
    dist_leaf = evaluator.add_leaf(
        id="distance_from_san_diego",
        desc="Provide the distance from San Diego to the selected park and ensure it is within 200 miles",
        parent=np_details_node,
        critical=True,
    )
    if not park or not non_empty(park.park_name) or not non_empty(park.distance_text):
        mark_failed(dist_leaf, "Missing distance_text or park_name", logger)
    else:
        claim = (
            f"The selected park '{park.park_name}' is within 200 miles of San Diego, California. "
            f"The answer states the distance as '{park.distance_text}'."
        )
        sources = combine_unique_urls(park.distance_urls, park.park_urls)
        await evaluator.verify(
            claim=claim,
            node=dist_leaf,
            sources=sources if sources else None,
            additional_instruction=(
                "Verify the stated distance is within 200 miles using the provided sources (maps or official directions). "
                "Driving distance or straight-line distance are acceptable if clearly supported. "
                "Allow reasonable approximations and rounding."
            ),
        )

    # entrance_fee_details leaf
    fee_leaf = evaluator.add_leaf(
        id="entrance_fee_details",
        desc="State the park's entrance fee details (fee amount and basis such as per-vehicle/per-person), demonstrating that it charges entrance fees",
        parent=np_details_node,
        critical=True,
    )
    if not park or not non_empty(park.fee_details):
        mark_failed(fee_leaf, "Missing fee_details", logger)
    else:
        claim = f"The park charges entrance fees as follows: {park.fee_details}"
        sources = combine_unique_urls(park.fee_urls, park.park_urls)
        await evaluator.verify(
            claim=claim,
            node=fee_leaf,
            sources=sources if sources else None,
            additional_instruction=(
                "Confirm the entrance fee amount and basis (e.g., per-vehicle 7-day pass, per-person) from the provided sources. "
                "If no entrance fee is charged, the claim should be marked incorrect."
            ),
        )

    # pass_coverage_confirmation leaf
    pass_cov_leaf = evaluator.add_leaf(
        id="pass_coverage_confirmation",
        desc="Confirm the park's entrance fees are covered by the America the Beautiful Pass",
        parent=np_details_node,
        critical=True,
    )
    if not park or not non_empty(park.pass_coverage_statement):
        # Still can verify the fact if sources provided; but rubric expects the confirmation to be provided.
        # If the answer didn't include it, fail explicitly.
        mark_failed(pass_cov_leaf, "Missing pass_coverage_statement in the answer", logger)
    else:
        claim = (
            "The park's entrance fees are covered by the America the Beautiful Interagency Pass, including the Senior Pass."
        )
        sources = combine_unique_urls(park.pass_urls, park.park_urls)
        await evaluator.verify(
            claim=claim,
            node=pass_cov_leaf,
            sources=sources if sources else None,
            additional_instruction=(
                "Verify from the park or NPS sources that America the Beautiful (Interagency) Passes cover the park's entrance fees. "
                "If sources show passes are accepted for entrance fees, mark supported."
            ),
        )

    # ---- 2) Senior pass cost analysis (parallel, critical) ----
    cost_node = evaluator.add_parallel(
        id="senior_pass_cost_analysis",
        desc="Compute and compare costs for two park visits vs buying the $80 Senior Lifetime Pass",
        parent=np_and_cost_node,
        critical=True,
    )

    # senior_pass_price leaf
    pass_price_leaf = evaluator.add_leaf(
        id="senior_pass_price",
        desc="Correctly state the Senior Lifetime Pass price is $80",
        parent=cost_node,
        critical=True,
    )
    if not cost or not non_empty(cost.senior_pass_price_text):
        mark_failed(pass_price_leaf, "Missing senior_pass_price_text", logger)
    else:
        claim = "The price of the America the Beautiful Senior Lifetime Pass is $80."
        sources = combine_unique_urls(cost.pass_info_urls)
        await evaluator.verify(
            claim=claim,
            node=pass_price_leaf,
            sources=sources if sources else None,
            additional_instruction="Ensure the claim refers to the LIFETIME Senior Pass (not the annual $20 pass). Prefer USGS/NPS official sources.",
        )

    # two_visit_fee_total leaf
    two_visit_leaf = evaluator.add_leaf(
        id="two_visit_fee_total",
        desc="Calculate the total cost of paying individual entrance fees for two visits for the couple (2 people), using the selected park's fee structure",
        parent=cost_node,
        critical=True,
    )
    if not cost or not non_empty(cost.two_visit_fee_total_text) or not park or not non_empty(park.fee_details):
        mark_failed(two_visit_leaf, "Missing two_visit_fee_total_text or fee_details", logger)
    else:
        claim = (
            f"For two visits during one week, the total cost for the couple paying individual entrance fees is "
            f"{cost.two_visit_fee_total_text}, based on the fee structure: {park.fee_details}"
        )
        sources = combine_unique_urls(park.fee_urls, park.park_urls, cost.calc_urls)
        await evaluator.verify(
            claim=claim,
            node=two_visit_leaf,
            sources=sources if sources else None,
            additional_instruction=(
                "Check whether the stated total correctly applies the fee basis and validity window (e.g., a per-vehicle 7‑day pass "
                "may cover multiple entries within a week, so two visits might require only one fee). Verify the math and assumptions "
                "from the provided fee sources."
            ),
        )

    # cost_effectiveness_conclusion leaf
    cost_effect_leaf = evaluator.add_leaf(
        id="cost_effectiveness_conclusion",
        desc="Provide a clear conclusion on whether buying the pass costs less, the same, or more than paying individual entrance fees for the two planned visits, and show the comparison using the computed totals",
        parent=cost_node,
        critical=True,
    )
    if not cost or not non_empty(cost.cost_effectiveness_conclusion_text) or not non_empty(cost.two_visit_fee_total_text):
        mark_failed(cost_effect_leaf, "Missing cost_effectiveness_conclusion_text or two_visit_fee_total_text", logger)
    else:
        claim = (
            f"The answer clearly concludes: {cost.cost_effectiveness_conclusion_text}; "
            f"and this conclusion correctly compares $80 (Senior Lifetime Pass) with the stated two-visit total "
            f"({cost.two_visit_fee_total_text}) for the couple."
        )
        await evaluator.verify(
            claim=claim,
            node=cost_effect_leaf,
            sources=None,  # Logical consistency check against provided totals
            additional_instruction=(
                "Judge internal logical consistency: given the two-visit total and the $80 pass price, is the stated conclusion "
                "('less'/'same'/'more') correct? Ignore external facts; focus on the comparison as presented."
            ),
        )


async def verify_accommodation(
    evaluator: Evaluator,
    parent_node,
    acc: Optional[AccommodationInfo],
    logger: logging.Logger,
):
    acc_node = evaluator.add_parallel(
        id="accommodation",
        desc="Recommend a hotel in or near the San Diego Harbor district with location details",
        parent=parent_node,
        critical=True,
    )

    hotel_rec_leaf = evaluator.add_leaf(
        id="hotel_recommendation",
        desc="Recommend at least one specific hotel in or near the San Diego Harbor district",
        parent=acc_node,
        critical=True,
    )
    if not acc or not non_empty(acc.hotel_name):
        mark_failed(hotel_rec_leaf, "Missing hotel_name", logger)
    else:
        claim = f"The answer recommends a specific hotel named '{acc.hotel_name}' in San Diego."
        sources = combine_unique_urls(acc.hotel_urls)
        await evaluator.verify(
            claim=claim,
            node=hotel_rec_leaf,
            sources=sources if sources else None,
            additional_instruction="Verify that the named property is an actual hotel in San Diego (from the provided hotel or map/listing link).",
        )

    hotel_loc_leaf = evaluator.add_leaf(
        id="hotel_location_details",
        desc="Provide location details for the recommended hotel (e.g., neighborhood/area and/or address) demonstrating harbor proximity",
        parent=acc_node,
        critical=True,
    )
    if not acc or not non_empty(acc.hotel_location_details) or not non_empty(acc.hotel_name):
        mark_failed(hotel_loc_leaf, "Missing hotel_location_details or hotel_name", logger)
    else:
        claim = (
            f"The hotel '{acc.hotel_name}' is located in or near the San Diego Harbor/Embarcadero waterfront area "
            f"(location details in the answer: '{acc.hotel_location_details}')."
        )
        sources = combine_unique_urls(acc.hotel_urls)
        await evaluator.verify(
            claim=claim,
            node=hotel_loc_leaf,
            sources=sources if sources else None,
            additional_instruction=(
                "Confirm proximity to the San Diego Harbor/Embarcadero/waterfront along San Diego Bay (e.g., references to Embarcadero, "
                "San Diego Bay, USS Midway area). Location/address on official or map/listing page is acceptable."
            ),
        )


async def verify_flight_availability(
    evaluator: Evaluator,
    parent_node,
    flight: Optional[FlightInfo],
    logger: logging.Logger,
):
    flight_node = evaluator.add_parallel(
        id="flight_availability",
        desc="Address availability of direct flights from Bangor (BGR) to the San Diego area",
        parent=parent_node,
        critical=True,
    )

    direct_leaf = evaluator.add_leaf(
        id="direct_flight_check",
        desc="Provide an explicit yes/no statement on whether nonstop/direct flights from BGR to the San Diego area airport(s) (e.g., SAN) are available (a clear determination is required, not just a discussion of connecting flights)",
        parent=flight_node,
        critical=True,
    )
    if not flight or not non_empty(flight.direct_flight_statement):
        mark_failed(direct_leaf, "Missing direct_flight_statement", logger)
    else:
        yn = normalize_yes_no(flight.direct_flight_statement)
        if yn in {"yes", "no"}:
            polarity = "are available" if yn == "yes" else "are NOT available"
            claim = (
                f"Nonstop/direct flights {polarity} from Bangor International Airport (BGR) to a San Diego area airport "
                f"(acceptable: SAN or CLD/CRQ)."
            )
        else:
            # If ambiguous, still construct claim using the raw statement
            claim = (
                f"The answer provides a clear determination regarding availability of nonstop/direct flights from BGR "
                f"to the San Diego area (SAN or CLD/CRQ): '{flight.direct_flight_statement}'."
            )
        sources = combine_unique_urls(flight.flight_urls)
        await evaluator.verify(
            claim=claim,
            node=direct_leaf,
            sources=sources if sources else None,
            additional_instruction=(
                "Use airline/aggregator sources to verify whether any nonstop/direct BGR→SAN or BGR→CLD/CRQ service exists. "
                "Do not consider LAX/SNA; only San Diego area airports (SAN, CRQ/CLD). If no page shows such nonstops, treat as not available."
            ),
        )


async def verify_march_conditions(
    evaluator: Evaluator,
    parent_node,
    march: Optional[MarchConditions],
    logger: logging.Logger,
):
    march_node = evaluator.add_parallel(
        id="march_conditions",
        desc="Verify March conditions in San Diego (temps and sunset times) for outdoor and evening harbor activities",
        parent=parent_node,
        critical=True,
    )

    temps_leaf = evaluator.add_leaf(
        id="march_temperatures",
        desc="Provide San Diego average March temperatures (at least average high and low)",
        parent=march_node,
        critical=True,
    )
    if not march or not non_empty(march.march_high_avg_text) or not non_empty(march.march_low_avg_text):
        mark_failed(temps_leaf, "Missing March average high/low text", logger)
    else:
        claim = (
            f"In March, San Diego average temperatures are approximately: high {march.march_high_avg_text} "
            f"and low {march.march_low_avg_text}."
        )
        sources = combine_unique_urls(march.weather_urls)
        await evaluator.verify(
            claim=claim,
            node=temps_leaf,
            sources=sources if sources else None,
            additional_instruction=(
                "Verify March climate normals/averages for San Diego from the provided sources. "
                "Allow minor variations and rounding."
            ),
        )

    sunset_leaf = evaluator.add_leaf(
        id="march_sunset_times",
        desc="Provide typical sunset time information for San Diego in March (e.g., approximate range across the month)",
        parent=march_node,
        critical=True,
    )
    if not march or not non_empty(march.march_sunset_times_text):
        mark_failed(sunset_leaf, "Missing March sunset times text", logger)
    else:
        claim = f"In March, typical sunset times in San Diego are {march.march_sunset_times_text} (approximate range)."
        sources = combine_unique_urls(march.sunset_urls)
        await evaluator.verify(
            claim=claim,
            node=sunset_leaf,
            sources=sources if sources else None,
            additional_instruction="Verify with reputable almanac/time sources (e.g., timeanddate.com). Allow approximate ranges.",
        )

    suitability_leaf = evaluator.add_leaf(
        id="suitability_statement",
        desc="Include an explicit suitability conclusion for (a) outdoor park activities and (b) evening harbor visits, and explicitly reference the provided March temperature and sunset data as justification (presence-and-grounding check, not subjective correctness of the conclusion)",
        parent=march_node,
        critical=True,
    )
    if not march or not non_empty(march.suitability_statement_text):
        mark_failed(suitability_leaf, "Missing suitability statement text", logger)
    else:
        claim = (
            f"The answer includes a clear suitability statement covering (a) outdoor park activities and (b) evening harbor visits, "
            f"and it references the March temperatures and sunset times provided (text: '{march.suitability_statement_text}')."
        )
        await evaluator.verify(
            claim=claim,
            node=suitability_leaf,
            sources=None,  # Presence/grounding within the answer itself
            additional_instruction=(
                "Check the answer content for an explicit conclusion about suitability that cites or references the presented March "
                "average temperatures and sunset times as justification. This is a presence-and-grounding check within the answer text."
            ),
        )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel per rubric
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
    # Set root critical to match rubric; ensure all children we add are also critical
    if root:
        root.critical = True

    # Extraction
    trip_extraction = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction",
    )

    # Optional: record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "park_name": getattr(trip_extraction.park or ParkDetails(), "park_name", None),
            "hotel_name": getattr(trip_extraction.accommodation or AccommodationInfo(), "hotel_name", None),
            "direct_flight_statement": getattr(trip_extraction.flight or FlightInfo(), "direct_flight_statement", None),
        },
        info_type="extraction_preview",
        info_name="extraction_preview",
    )

    # Build and verify subtrees
    # All direct children under root must be critical=True to satisfy critical parent constraint
    await verify_national_park_and_cost_analysis(
        evaluator,
        parent_node=root,
        park=trip_extraction.park,
        cost=trip_extraction.cost,
        logger=logger,
    )

    await verify_accommodation(
        evaluator,
        parent_node=root,
        acc=trip_extraction.accommodation,
        logger=logger,
    )

    await verify_flight_availability(
        evaluator,
        parent_node=root,
        flight=trip_extraction.flight,
        logger=logger,
    )

    await verify_march_conditions(
        evaluator,
        parent_node=root,
        march=trip_extraction.march,
        logger=logger,
    )

    # Return structured summary
    return evaluator.get_summary()