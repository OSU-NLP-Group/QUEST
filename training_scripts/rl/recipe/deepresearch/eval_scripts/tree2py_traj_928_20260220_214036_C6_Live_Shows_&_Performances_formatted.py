import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tour_venues_2026_ne_us"
TASK_DESCRIPTION = """A mid-level touring comedian is planning a regional tour segment through the Northeastern United States in spring 2026. Identify suitable comedy venues for a 4-city tour with the following requirements:

1. Cleveland, Ohio - A venue that can accommodate 2-3 consecutive nights during February 27 - March 1, 2026, with a capacity between 200-400 seats

2. Syracuse, New York - A venue that can accommodate 2 consecutive nights during March 27-28, 2026, with a capacity between 200-400 seats

3. Arlington, Virginia - A venue available for a single night on April 30, 2026, with a capacity between 200-400 seats

4. East Providence, Rhode Island - A venue that can accommodate 2 consecutive nights during April 11-12, 2026, with a capacity between 200-400 seats

For each venue, provide:
- The venue name and type (comedy club, theater, or entertainment venue)
- The seating capacity
- Confirmation of availability for the specified dates
- The venue's age restriction policy
- Whether the venue has a drink minimum requirement
- Reference URLs supporting each piece of information
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueBasics(BaseModel):
    name: Optional[str] = None
    name_urls: List[str] = Field(default_factory=list)
    venue_type: Optional[str] = None  # expected one of: comedy club, theater, entertainment venue
    type_urls: List[str] = Field(default_factory=list)
    city: Optional[str] = None
    state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    region_urls: List[str] = Field(default_factory=list)  # URLs supporting "Northeastern US" classification


class VenueSuitability(BaseModel):
    standup_statement: Optional[str] = None  # e.g., "hosts stand-up comedy", "comedy shows"
    standup_urls: List[str] = Field(default_factory=list)


class VenueCapacity(BaseModel):
    capacity: Optional[str] = None  # Keep as string to allow ranges like "250-300", "about 300"
    capacity_urls: List[str] = Field(default_factory=list)


class VenueDates(BaseModel):
    availability_statement: Optional[str] = None  # Free-form summary the answer claims, if any
    date_urls: List[str] = Field(default_factory=list)


class VenuePolicies(BaseModel):
    age_policy: Optional[str] = None  # e.g., "21+", "18+", "all ages with adult"
    age_urls: List[str] = Field(default_factory=list)
    drink_minimum_policy: Optional[str] = None  # e.g., "2-drink minimum", "no drink minimum"
    drink_urls: List[str] = Field(default_factory=list)


class CityVenue(BaseModel):
    basics: Optional[VenueBasics] = None
    suitability: Optional[VenueSuitability] = None
    capacity_info: Optional[VenueCapacity] = None
    dates: Optional[VenueDates] = None
    policies: Optional[VenuePolicies] = None


class TourVenuesExtraction(BaseModel):
    cleveland: Optional[CityVenue] = None
    syracuse: Optional[CityVenue] = None
    arlington: Optional[CityVenue] = None
    east_providence: Optional[CityVenue] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract the selected venues for the 4 specified cities from the answer and the supporting URLs for each required piece of information.

For each of the following city slots in the answer (even if some are missing, still include the slot with nulls):
- cleveland (Cleveland, Ohio)
- syracuse (Syracuse, New York)
- arlington (Arlington, Virginia)
- east_providence (East Providence, Rhode Island)

For each city slot, extract a JSON object with the following nested structure:

{
  "basics": {
    "name": string or null,
    "name_urls": [list of URLs that support the venue name],
    "venue_type": string or null (should be one of: "comedy club", "theater", or "entertainment venue"),
    "type_urls": [list of URLs that support the venue type classification],
    "city": string or null (city of the venue),
    "state": string or null (state of the venue),
    "location_urls": [list of URLs that explicitly show the venue’s city/state],
    "region_urls": [list of URLs that support that the city is in the Northeastern United States under some cited definition]
  },
  "suitability": {
    "standup_statement": string or null (short phrase indicating stand-up suitability, if present),
    "standup_urls": [list of URLs that show the venue hosts stand-up comedy shows]
  },
  "capacity_info": {
    "capacity": string or null (exact or approximate; do NOT coerce to a number),
    "capacity_urls": [list of URLs that support this capacity figure]
  },
  "dates": {
    "availability_statement": string or null (the answer’s claim about availability for the specified dates),
    "date_urls": [list of URLs that support the stated date availability or booking feasibility]
  },
  "policies": {
    "age_policy": string or null,
    "age_urls": [list of URLs that support the age restriction policy],
    "drink_minimum_policy": string or null,
    "drink_urls": [list of URLs that support the drink minimum policy (or explicitly state none)]
  }
}

Rules:
- Extract only what the answer explicitly claims and the URLs it cites. If a field is not present in the answer, set it to null (or an empty list for URLs).
- For URL fields, return only fully qualified URLs, not plain site names. If a markdown link is used, extract its URL target.
- Do not invent or infer information or sources not present in the answer.
- Keep capacity as a free-form string to accommodate ranges or approximations.
- Return the JSON object with top-level keys: "cleveland", "syracuse", "arlington", "east_providence", each mapping to the structure described above.
    """


# --------------------------------------------------------------------------- #
# City specifications                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class CitySpec:
    key: str                    # extraction key: cleveland, syracuse, arlington, east_providence
    top_id: str                 # top node id for the city branch
    city: str
    state: str
    # Availability claims
    date_node_desc: str         # parent Date_Availability description
    date_leaf_id: str           # id for the meets-availability leaf
    date_leaf_desc: str         # description for the meets-availability leaf
    date_claim: str             # factual claim to verify
    # Capacity range
    cap_min: int = 200
    cap_max: int = 400


CITY_SPECS = [
    CitySpec(
        key="cleveland",
        top_id="Venue_1_Cleveland_OH",
        city="Cleveland",
        state="Ohio",
        date_node_desc="Verify availability for 2–3 consecutive nights within Feb 27–Mar 1, 2026.",
        date_leaf_id="Meets_2_to_3_Consecutive_Nights_In_Window",
        date_leaf_desc="Confirm the venue can accommodate 2–3 consecutive nights during Feb 27–Mar 1, 2026.",
        date_claim="The venue can accommodate 2–3 consecutive nights during Feb 27–Mar 1, 2026."
    ),
    CitySpec(
        key="syracuse",
        top_id="Venue_2_Syracuse_NY",
        city="Syracuse",
        state="New York",
        date_node_desc="Verify availability for 2 consecutive nights on Mar 27–28, 2026.",
        date_leaf_id="Meets_2_Consecutive_Nights_Mar_27_28",
        date_leaf_desc="Confirm the venue can accommodate 2 consecutive nights during Mar 27–28, 2026.",
        date_claim="The venue can accommodate 2 consecutive nights during Mar 27–28, 2026."
    ),
    CitySpec(
        key="arlington",
        top_id="Venue_3_Arlington_VA",
        city="Arlington",
        state="Virginia",
        date_node_desc="Verify availability for a single night on Apr 30, 2026.",
        date_leaf_id="Available_April_30_2026",
        date_leaf_desc="Confirm the venue can accommodate a performance on Apr 30, 2026.",
        date_claim="The venue can accommodate a performance on Apr 30, 2026."
    ),
    CitySpec(
        key="east_providence",
        top_id="Venue_4_East_Providence_RI",
        city="East Providence",
        state="Rhode Island",
        date_node_desc="Verify availability for 2 consecutive nights on Apr 11–12, 2026.",
        date_leaf_id="Meets_2_Consecutive_Nights_Apr_11_12",
        date_leaf_desc="Confirm the venue can accommodate 2 consecutive nights during Apr 11–12, 2026.",
        date_claim="The venue can accommodate 2 consecutive nights during Apr 11–12, 2026."
    ),
]


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _urls_or_empty(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


def _non_empty_str(s: Optional[str]) -> bool:
    return s is not None and isinstance(s, str) and s.strip() != ""


def _city_state_str(city: Optional[str], state: Optional[str]) -> str:
    c = city or ""
    st = state or ""
    if c and st:
        return f"{c}, {st}"
    return (c or st).strip()


# --------------------------------------------------------------------------- #
# Verification per-city                                                       #
# --------------------------------------------------------------------------- #
async def verify_city_venue(
    evaluator: Evaluator,
    parent_node,
    spec: CitySpec,
    venue: Optional[CityVenue],
):
    """
    Build the verification subtree for a single city based on the rubric.
    """
    # Top-level city node
    city_node = evaluator.add_parallel(
        id=spec.top_id,
        desc=f"{spec.city}, {spec.state} venue meeting capacity, date, policy, suitability, region, and sourcing requirements.",
        parent=parent_node,
        critical=False,  # city branches are non-critical at root to allow partial success across cities
    )

    basics = venue.basics if venue else None
    suitability = venue.suitability if venue else None
    capacity_info = venue.capacity_info if venue else None
    dates = venue.dates if venue else None
    policies = venue.policies if venue else None

    # ---------------- Venue_Basics (critical, parallel) ------------------- #
    basics_node = evaluator.add_parallel(
        id=f"{spec.top_id}_Venue_Basics",
        desc="Provide and verify the venue’s identity, type, and location (including regional constraint).",
        parent=city_node,
        critical=True,
    )

    # Venue_Name (existence)
    evaluator.add_custom_node(
        result=_non_empty_str(basics.name) if basics else False,
        id=f"{spec.top_id}_Venue_Name",
        desc="Provide the venue name.",
        parent=basics_node,
        critical=True
    )

    # Venue_Name_Reference_URLs (verify name with cited pages)
    vn_leaf = evaluator.add_leaf(
        id=f"{spec.top_id}_Venue_Name_Reference_URLs",
        desc="Provide reference URL(s) supporting the venue name.",
        parent=basics_node,
        critical=True
    )
    name_claim = f"The venue's name is '{(basics.name if basics else '')}'."
    await evaluator.verify(
        claim=name_claim,
        node=vn_leaf,
        sources=_urls_or_empty(basics.name_urls if basics else None),
        additional_instruction="Check that the cited page clearly shows the exact venue name, allowing minor formatting differences or abbreviations."
    )

    # Venue_Type_Allowed (simple check)
    vtype_allowed = evaluator.add_leaf(
        id=f"{spec.top_id}_Venue_Type_Allowed",
        desc="Confirm the venue type is one of: comedy club, theater, or entertainment venue.",
        parent=basics_node,
        critical=True
    )
    allowed_claim = f"The venue type '{(basics.venue_type if basics else '')}' is one of: comedy club, theater, or entertainment venue."
    await evaluator.verify(
        claim=allowed_claim,
        node=vtype_allowed,
        additional_instruction="Treat case-insensitive matches as valid. Accept 'entertainment venue' or closely equivalent phrasing."
    )

    # Venue_Type_Reference_URLs (verify type classification on cited pages)
    vtype_ref_leaf = evaluator.add_leaf(
        id=f"{spec.top_id}_Venue_Type_Reference_URLs",
        desc="Provide reference URL(s) supporting the venue type claim.",
        parent=basics_node,
        critical=True
    )
    type_claim = f"The venue is a '{(basics.venue_type if basics else '')}'."
    await evaluator.verify(
        claim=type_claim,
        node=vtype_ref_leaf,
        sources=_urls_or_empty(basics.type_urls if basics else None),
        additional_instruction="Look for text on the page that classifies the venue as a comedy club, theater, or entertainment venue (or clearly equivalent phrasing)."
    )

    # Venue_Location_City_State (verify location with URLs)
    vloc_leaf = evaluator.add_leaf(
        id=f"{spec.top_id}_Venue_Location_City_State",
        desc=f"Confirm the venue is located in {spec.city}, {spec.state}.",
        parent=basics_node,
        critical=True
    )
    vname = basics.name if basics and basics.name else "the venue"
    location_claim = f"{vname} is located in {spec.city}, {spec.state}."
    # Prefer location_urls; if empty, the verification will likely fail — matching the rubric’s sourcing requirement
    await evaluator.verify(
        claim=location_claim,
        node=vloc_leaf,
        sources=_urls_or_empty(basics.location_urls if basics else None),
        additional_instruction="The cited page should explicitly show the venue’s city and state. Minor address formatting differences are acceptable."
    )

    # Venue_Location_Reference_URLs (existence of location URLs)
    evaluator.add_custom_node(
        result=bool(basics and basics.location_urls and len(basics.location_urls) > 0),
        id=f"{spec.top_id}_Venue_Location_Reference_URLs",
        desc="Provide reference URL(s) supporting the venue location (city/state).",
        parent=basics_node,
        critical=True
    )

    # City_In_Northeastern_US (verify regional classification with URLs)
    region_leaf = evaluator.add_leaf(
        id=f"{spec.top_id}_City_In_Northeastern_US",
        desc=f"Confirm {spec.city}, {spec.state} is in the Northeastern United States region (using a cited definition/source for the region).",
        parent=basics_node,
        critical=True
    )
    region_claim = f"{spec.city}, {spec.state} is considered part of the Northeastern United States region according to the cited source."
    await evaluator.verify(
        claim=region_claim,
        node=region_leaf,
        sources=_urls_or_empty(basics.region_urls if basics else None),
        additional_instruction=(
            "Rely on the cited source’s own regional definition. If the cited source defines 'Northeast' to include "
            "the Mid-Atlantic (e.g., parts of VA), that is acceptable. Your judgment must be grounded in the provided URL(s)."
        )
    )

    # Regional_Classification_Reference_URLs (existence of regional URLs)
    evaluator.add_custom_node(
        result=bool(basics and basics.region_urls and len(basics.region_urls) > 0),
        id=f"{spec.top_id}_Regional_Classification_Reference_URLs",
        desc=f"Provide reference URL(s) supporting the Northeastern-region classification claim for {spec.city}, {spec.state}.",
        parent=basics_node,
        critical=True
    )

    # ---------------- Standup_Suitability (critical leaf) ----------------- #
    standup_leaf = evaluator.add_leaf(
        id=f"{spec.top_id}_Standup_Suitability",
        desc="Verify the venue is suitable for stand-up comedy performances.",
        parent=city_node,
        critical=True
    )
    standup_claim = f"{vname} hosts stand-up comedy performances."
    await evaluator.verify(
        claim=standup_claim,
        node=standup_leaf,
        sources=_urls_or_empty(suitability.standup_urls if suitability else None),
        additional_instruction="Look for evidence of stand-up comedy shows, stand-up headliners, or a calendar clearly showing stand-up events."
    )

    # Standup_Suitability_Reference_URLs (existence of URLs)
    evaluator.add_custom_node(
        result=bool(suitability and suitability.standup_urls and len(suitability.standup_urls) > 0),
        id=f"{spec.top_id}_Standup_Suitability_Reference_URLs",
        desc="Provide reference URL(s) supporting stand-up suitability.",
        parent=city_node,
        critical=True
    )

    # ---------------- Capacity (critical, parallel) ----------------------- #
    capacity_node = evaluator.add_parallel(
        id=f"{spec.top_id}_Capacity",
        desc="Provide and verify seating capacity meets the required range.",
        parent=city_node,
        critical=True
    )

    # Capacity_Value_Provided
    evaluator.add_custom_node(
        result=_non_empty_str(capacity_info.capacity) if capacity_info else False,
        id=f"{spec.top_id}_Capacity_Value_Provided",
        desc="Provide the seating capacity value.",
        parent=capacity_node,
        critical=True
    )

    # Capacity_In_Range_200_400 (verify by URLs)
    cap_range_leaf = evaluator.add_leaf(
        id=f"{spec.top_id}_Capacity_In_Range_200_400",
        desc="Confirm seating capacity is between 200 and 400 seats (inclusive).",
        parent=capacity_node,
        critical=True
    )
    cap_range_claim = f"The seating capacity for {vname} is between {spec.cap_min} and {spec.cap_max} seats (inclusive)."
    await evaluator.verify(
        claim=cap_range_claim,
        node=cap_range_leaf,
        sources=_urls_or_empty(capacity_info.capacity_urls if capacity_info else None),
        additional_instruction=(
            "Use the cited page(s) to determine the seating capacity. If a range is given, confirm that it falls entirely within 200–400. "
            "Allow reasonable approximations and minor rounding."
        )
    )

    # Capacity_Reference_URLs (existence)
    evaluator.add_custom_node(
        result=bool(capacity_info and capacity_info.capacity_urls and len(capacity_info.capacity_urls) > 0),
        id=f"{spec.top_id}_Capacity_Reference_URLs",
        desc="Provide reference URL(s) supporting the seating capacity claim.",
        parent=capacity_node,
        critical=True
    )

    # ---------------- Date_Availability (critical, parallel) -------------- #
    dates_node = evaluator.add_parallel(
        id=f"{spec.top_id}_Date_Availability",
        desc=spec.date_node_desc,
        parent=city_node,
        critical=True
    )

    dates_leaf = evaluator.add_leaf(
        id=f"{spec.top_id}_{spec.date_leaf_id}",
        desc=spec.date_leaf_desc,
        parent=dates_node,
        critical=True
    )
    await evaluator.verify(
        claim=spec.date_claim,
        node=dates_leaf,
        sources=_urls_or_empty(dates.date_urls if dates else None),
        additional_instruction=(
            "Look for booking calendars, availability pages, or booking/contact policies that credibly indicate "
            "availability for the specified date(s)/window. If the cited page is clearly unrelated or does not support the claim, fail."
        )
    )

    evaluator.add_custom_node(
        result=bool(dates and dates.date_urls and len(dates.date_urls) > 0),
        id=f"{spec.top_id}_Date_Availability_Reference_URLs",
        desc="Provide reference URL(s) supporting the stated date availability.",
        parent=dates_node,
        critical=True
    )

    # ---------------- Venue_Policies (critical, parallel) ----------------- #
    policies_node = evaluator.add_parallel(
        id=f"{spec.top_id}_Venue_Policies",
        desc="Provide required venue policies and sources.",
        parent=city_node,
        critical=True
    )

    # Age_Restriction_Policy (verify by URLs)
    age_leaf = evaluator.add_leaf(
        id=f"{spec.top_id}_Age_Restriction_Policy",
        desc="Identify the venue's age restriction policy.",
        parent=policies_node,
        critical=True
    )
    age_claim = f"The venue's age restriction policy is: {(policies.age_policy if policies and policies.age_policy else '')}."
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=_urls_or_empty(policies.age_urls if policies else None),
        additional_instruction="Accept reasonable equivalences such as '21+', '21 and over', 'All ages with adult', etc."
    )

    evaluator.add_custom_node(
        result=bool(policies and policies.age_urls and len(policies.age_urls) > 0),
        id=f"{spec.top_id}_Age_Restriction_Reference_URLs",
        desc="Provide reference URL(s) supporting the age restriction policy.",
        parent=policies_node,
        critical=True
    )

    # Drink_Minimum_Policy (verify by URLs)
    drink_leaf = evaluator.add_leaf(
        id=f"{spec.top_id}_Drink_Minimum_Policy",
        desc="Identify whether there is a drink minimum requirement (including explicitly stating none if none).",
        parent=policies_node,
        critical=True
    )
    drink_claim = f"The venue's drink minimum policy is: {(policies.drink_minimum_policy if policies and policies.drink_minimum_policy else '')}."
    await evaluator.verify(
        claim=drink_claim,
        node=drink_leaf,
        sources=_urls_or_empty(policies.drink_urls if policies else None),
        additional_instruction="Verify if the page explicitly mentions a drink minimum (e.g., 2-drink minimum) or explicitly states no drink minimum."
    )

    evaluator.add_custom_node(
        result=bool(policies and policies.drink_urls and len(policies.drink_urls) > 0),
        id=f"{spec.top_id}_Drink_Minimum_Reference_URLs",
        desc="Provide reference URL(s) supporting the drink minimum policy claim.",
        parent=policies_node,
        critical=True
    )


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
    Evaluate an answer for the 2026 Northeastern US 4-city tour venues task.
    """
    evaluator = Evaluator()
    # Root is parallel and non-critical to allow partial scoring across cities
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

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=TourVenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Add a ground-truth-like configuration context (constraints) for transparency
    evaluator.add_ground_truth({
        "cities": [
            {"city": "Cleveland", "state": "Ohio", "dates": "Feb 27–Mar 1, 2026", "nights": "2–3", "capacity_range": "200–400"},
            {"city": "Syracuse", "state": "New York", "dates": "Mar 27–28, 2026", "nights": "2", "capacity_range": "200–400"},
            {"city": "Arlington", "state": "Virginia", "dates": "Apr 30, 2026", "nights": "1", "capacity_range": "200–400"},
            {"city": "East Providence", "state": "Rhode Island", "dates": "Apr 11–12, 2026", "nights": "2", "capacity_range": "200–400"}
        ],
        "allowed_types": ["comedy club", "theater", "entertainment venue"]
    }, gt_type="constraints")

    # Build city subtrees
    # Map extraction keys to the extracted CityVenue objects
    city_map: Dict[str, Optional[CityVenue]] = {
        "cleveland": extraction.cleveland,
        "syracuse": extraction.syracuse,
        "arlington": extraction.arlington,
        "east_providence": extraction.east_providence,
    }

    # Top-level container node (non-critical) described by rubric
    tour_node = evaluator.add_parallel(
        id="Tour_Venue_Selection",
        desc="Select 4 suitable stand-up comedy venues (one per specified city) for the specified spring 2026 dates, meeting all stated constraints and providing verifiable sources.",
        parent=root,
        critical=False  # Adjusted to comply with framework constraints and allow partial credit across cities
    )

    # Verify each city
    for spec in CITY_SPECS:
        venue = city_map.get(spec.key)
        await verify_city_venue(evaluator, tour_node, spec, venue)

    # Final summary
    return evaluator.get_summary()