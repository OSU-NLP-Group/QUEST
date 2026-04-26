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
TASK_ID = "multi_activity_vacation_2026"
TASK_DESCRIPTION = """Plan a multi-activity outdoor vacation for late spring 2026 with the following requirements:

You want to start with a challenging day hike in the San Diego area before your cruise departure. The trail should be approximately 3 miles round trip with around 900 feet of elevation gain and classified as moderate difficulty.

After the hiking day, you will embark on an Alaska cruise. The cruise must meet these specifications:
- Operated by Princess Cruises on the Ruby Princess ship
- Depart from and return to San Francisco (round-trip)
- Depart in May 2026, allowing at least one day after your San Diego hike for travel to San Francisco
- Duration of 10-11 days
- Itinerary must include stops at Juneau, Ketchikan, and Glacier Bay National Park

After completing the cruise, you plan to travel to Whistler, British Columbia for mountain biking at Whistler Mountain Bike Park. The bike park must be open for the summer season during your post-cruise visit in late May 2026, and should offer trails suitable for intermediate (blue-level) riders.

Provide:
1. The name and specifications of the San Diego hiking trail
2. The specific departure date of the Princess Cruises Ruby Princess Alaska cruise from San Francisco in May 2026
3. Confirmation that Whistler Mountain Bike Park will be open during your late May 2026 visit
4. URL references supporting each component of your plan
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TrailSpec(BaseModel):
    name: Optional[str] = None
    distance_roundtrip_miles: Optional[str] = None
    elevation_gain_feet: Optional[str] = None
    difficulty: Optional[str] = None
    location_area: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CruiseSpec(BaseModel):
    operator: Optional[str] = None            # e.g., "Princess Cruises"
    ship: Optional[str] = None                # e.g., "Ruby Princess"
    departure_port_city: Optional[str] = None # e.g., "San Francisco"
    return_port_city: Optional[str] = None    # e.g., "San Francisco"
    departure_month_year: Optional[str] = None # e.g., "May 2026"
    departure_date: Optional[str] = None       # e.g., "May 23, 2026"
    duration_days: Optional[str] = None        # e.g., "10 days"
    itinerary_ports: List[str] = Field(default_factory=list)  # e.g., ["Juneau", "Ketchikan", "Glacier Bay National Park"]
    urls: List[str] = Field(default_factory=list)


class BikeParkSpec(BaseModel):
    name: Optional[str] = None                # e.g., "Whistler Mountain Bike Park"
    location_city_province: Optional[str] = None # e.g., "Whistler, British Columbia"
    opening_dates_info: Optional[str] = None  # e.g., "Opens mid-May 2026"
    operating_late_may_confirmation: Optional[str] = None  # explicit statement from answer
    intermediate_trails_info: Optional[str] = None # description confirming blue-level trails
    blue_trail_examples: List[str] = Field(default_factory=list) # optional specific trail names
    urls: List[str] = Field(default_factory=list)


class TransportSDtoSF(BaseModel):
    flight_urls: List[str] = Field(default_factory=list)
    budget_airlines: List[str] = Field(default_factory=list)  # e.g., ["Southwest", "Frontier"]


class TransportSFtoWhistler(BaseModel):
    connection_route_via: Optional[str] = None  # e.g., "via Vancouver" or "via Seattle"
    connection_urls: List[str] = Field(default_factory=list)
    feasibility_notes: Optional[str] = None     # statement confirming reasonable timeframe


class PlanExtraction(BaseModel):
    trail: Optional[TrailSpec] = None
    cruise: Optional[CruiseSpec] = None
    bike_park: Optional[BikeParkSpec] = None
    transport_sd_to_sf: Optional[TransportSDtoSF] = None
    transport_sf_to_whistler: Optional[TransportSFtoWhistler] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract the structured information for the proposed multi-activity vacation plan from the answer.

    Return a JSON object with these fields:

    trail:
      - name: The hiking trail name in the San Diego area
      - distance_roundtrip_miles: The round-trip distance (as text, include units or qualifiers if present; e.g., "3 miles")
      - elevation_gain_feet: The elevation gain (as text; e.g., "900 ft")
      - difficulty: The stated difficulty (e.g., "moderate")
      - location_area: The general location, e.g., "San Diego area"
      - urls: Array of all URLs provided that describe or verify the trail and its specs

    cruise:
      - operator: The cruise line (e.g., "Princess Cruises")
      - ship: The ship name (e.g., "Ruby Princess")
      - departure_port_city: The departure city (e.g., "San Francisco")
      - return_port_city: The return city (e.g., "San Francisco")
      - departure_month_year: The month and year for departure (e.g., "May 2026")
      - departure_date: The specific departure date (e.g., "May 23, 2026")
      - duration_days: The cruise duration (as text; e.g., "10 days" or "11 days")
      - itinerary_ports: List of ports/areas included in the itinerary (e.g., ["Juneau","Ketchikan","Glacier Bay National Park"])
      - urls: Array of all URLs that confirm the cruise details, itinerary, timing, duration, and ship

    bike_park:
      - name: The bike park name (e.g., "Whistler Mountain Bike Park")
      - location_city_province: The location (e.g., "Whistler, British Columbia")
      - opening_dates_info: Any mention of opening dates/season start for summer 2026 (text)
      - operating_late_may_confirmation: Text confirming operation in late May 2026 if provided
      - intermediate_trails_info: Text confirming suitable trails for intermediate/blue-level riders
      - blue_trail_examples: Array of any specific blue trail names mentioned (if any)
      - urls: Array of URLs that confirm the park identity, location, opening/operating dates for 2026, and trail difficulty offerings

    transport_sd_to_sf:
      - flight_urls: Array of URLs that show flight availability from San Diego to San Francisco
      - budget_airlines: Array of budget airlines mentioned for this route (e.g., "Southwest", "Frontier"). Use the names exactly as they appear in the answer.

    transport_sf_to_whistler:
      - connection_route_via: Text describing the suggested route (e.g., "via Vancouver" or "via Seattle")
      - connection_urls: Array of URLs that show the feasibility (e.g., airline routes or schedules)
      - feasibility_notes: Text confirming feasibility within a reasonable timeframe (if the answer provides it)

    IMPORTANT:
    - Extract exactly what appears in the answer. Do not infer or create data not present.
    - For any missing field, return null (or empty array for URLs/lists).
    - Include all URLs cited in the answer relevant to each component.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_hiking(evaluator: Evaluator, parent_node, trail: Optional[TrailSpec]) -> None:
    # Pre_Cruise_Hiking (critical)
    hiking_node = evaluator.add_parallel(
        id="Pre_Cruise_Hiking",
        desc="Hiking activity completed before cruise departure",
        parent=parent_node,
        critical=True
    )

    # Trail_Selection (critical)
    trail_sel_node = evaluator.add_parallel(
        id="Trail_Selection",
        desc="Identified a hiking trail in San Diego meeting specified distance and elevation criteria",
        parent=hiking_node,
        critical=True
    )

    # Existence gate
    exists = trail is not None and bool(trail.name) and bool(trail.urls)
    evaluator.add_custom_node(
        result=exists,
        id="Trail_Provided",
        desc="Trail information and at least one reference URL are provided",
        parent=trail_sel_node,
        critical=True
    )

    # Trail_Distance
    n_distance = evaluator.add_leaf(
        id="Trail_Distance",
        desc="Trail is approximately 3 miles round trip",
        parent=trail_sel_node,
        critical=True
    )
    dist_claim = "The trail's round-trip distance is approximately 3 miles (e.g., between about 2.7 and 3.3 miles)."
    await evaluator.verify(
        claim=dist_claim,
        node=n_distance,
        sources=trail.urls if trail else [],
        additional_instruction="Allow reasonable approximation around 3 miles round trip; minor variants (e.g., 2.8-3.2 mi) count."
    )

    # Trail_Elevation
    n_elev = evaluator.add_leaf(
        id="Trail_Elevation",
        desc="Trail has approximately 900 feet of elevation gain",
        parent=trail_sel_node,
        critical=True
    )
    elev_claim = "The trail's elevation gain is approximately 900 feet (e.g., roughly 800–1000 ft)."
    await evaluator.verify(
        claim=elev_claim,
        node=n_elev,
        sources=trail.urls if trail else [],
        additional_instruction="Treat elevation around ~900 ft as acceptable; allow modest variance (±100–150 ft)."
    )

    # Trail_Difficulty
    n_diff = evaluator.add_leaf(
        id="Trail_Difficulty",
        desc="Trail is classified as moderate difficulty",
        parent=trail_sel_node,
        critical=True
    )
    diff_claim = "This trail is classified as moderate difficulty."
    await evaluator.verify(
        claim=diff_claim,
        node=n_diff,
        sources=trail.urls if trail else [],
        additional_instruction="Allow wording like 'moderate' or 'moderate to strenuous' to count as moderate classification."
    )

    # Trail_Location
    n_loc = evaluator.add_leaf(
        id="Trail_Location",
        desc="Trail is located in San Diego area",
        parent=trail_sel_node,
        critical=True
    )
    loc_claim = "This trail is in the San Diego area (broader metro region is acceptable)."
    await evaluator.verify(
        claim=loc_claim,
        node=n_loc,
        sources=trail.urls if trail else [],
        additional_instruction="Accept broader 'San Diego area' or specific nearby locale (e.g., La Jolla, Poway) within SD County."
    )

    # Trail_Reference
    n_ref = evaluator.add_leaf(
        id="Trail_Reference",
        desc="Provide URL reference verifying trail specifications",
        parent=trail_sel_node,
        critical=True
    )
    ref_claim = "The provided URLs are legitimate references that describe this trail and include its distance, elevation gain, and difficulty."
    await evaluator.verify(
        claim=ref_claim,
        node=n_ref,
        sources=trail.urls if trail else [],
        additional_instruction="Confirm the pages actually describe the trail and include some or all of the key specs."
    )


async def verify_cruise(evaluator: Evaluator, parent_node, cruise: Optional[CruiseSpec]) -> None:
    # Alaska_Cruise (critical)
    cruise_node = evaluator.add_parallel(
        id="Alaska_Cruise",
        desc="Alaska cruise selection meeting all requirements",
        parent=parent_node,
        critical=True
    )

    # Cruise_Line_And_Ship (critical)
    cls_node = evaluator.add_parallel(
        id="Cruise_Line_And_Ship",
        desc="Cruise operated by Princess Cruises on Ruby Princess",
        parent=cruise_node,
        critical=True
    )

    exists_cls = cruise is not None and bool(cruise.operator) and bool(cruise.ship) and bool(cruise.urls)
    evaluator.add_custom_node(
        result=exists_cls,
        id="Cruise_Line_Ship_Provided",
        desc="Cruise line and ship information with URL references are provided",
        parent=cls_node,
        critical=True
    )

    # Cruise_Line
    n_line = evaluator.add_leaf(
        id="Cruise_Line",
        desc="Cruise is operated by Princess Cruises",
        parent=cls_node,
        critical=True
    )
    line_claim = "This Alaska cruise is operated by Princess Cruises."
    await evaluator.verify(
        claim=line_claim,
        node=n_line,
        sources=cruise.urls if cruise else [],
        additional_instruction="Confirm operator is explicitly Princess Cruises."
    )

    # Ship_Name
    n_ship = evaluator.add_leaf(
        id="Ship_Name",
        desc="Cruise is on the Ruby Princess",
        parent=cls_node,
        critical=True
    )
    ship_claim = "The ship for this cruise is Ruby Princess."
    await evaluator.verify(
        claim=ship_claim,
        node=n_ship,
        sources=cruise.urls if cruise else [],
        additional_instruction="Confirm the ship name is Ruby Princess."
    )

    # Cruise_Line_Reference
    n_cls_ref = evaluator.add_leaf(
        id="Cruise_Line_Reference",
        desc="Provide URL reference confirming cruise line and ship",
        parent=cls_node,
        critical=True
    )
    cls_ref_claim = "The provided URLs confirm that Ruby Princess is a Princess Cruises ship and the selected cruise is operated by Princess Cruises."
    await evaluator.verify(
        claim=cls_ref_claim,
        node=n_cls_ref,
        sources=cruise.urls if cruise else [],
        additional_instruction="The pages should clearly tie Ruby Princess to Princess Cruises."
    )

    # Departure_Port (critical)
    port_node = evaluator.add_parallel(
        id="Departure_Port",
        desc="Cruise departs from and returns to San Francisco",
        parent=cruise_node,
        critical=True
    )

    exists_port = cruise is not None and bool(cruise.departure_port_city) and bool(cruise.return_port_city)
    evaluator.add_custom_node(
        result=exists_port,
        id="Port_Info_Provided",
        desc="Departure and return port info provided",
        parent=port_node,
        critical=True
    )

    # Departure_City
    n_dep_city = evaluator.add_leaf(
        id="Departure_City",
        desc="Cruise departs from San Francisco",
        parent=port_node,
        critical=True
    )
    dep_city_claim = "This cruise departs from San Francisco."
    await evaluator.verify(
        claim=dep_city_claim,
        node=n_dep_city,
        sources=cruise.urls if cruise else [],
        additional_instruction="Confirm the departure port city is San Francisco."
    )

    # Return_City
    n_ret_city = evaluator.add_leaf(
        id="Return_City",
        desc="Cruise returns to San Francisco (round-trip)",
        parent=port_node,
        critical=True
    )
    ret_city_claim = "This cruise returns to San Francisco (round-trip)."
    await evaluator.verify(
        claim=ret_city_claim,
        node=n_ret_city,
        sources=cruise.urls if cruise else [],
        additional_instruction="Confirm round-trip San Francisco."
    )

    # Port_Reference
    n_port_ref = evaluator.add_leaf(
        id="Port_Reference",
        desc="Provide URL reference confirming departure and return port",
        parent=port_node,
        critical=True
    )
    port_ref_claim = "The provided URLs confirm the cruise is round-trip from San Francisco."
    await evaluator.verify(
        claim=port_ref_claim,
        node=n_port_ref,
        sources=cruise.urls if cruise else [],
        additional_instruction="Pages should indicate both departure and return to San Francisco."
    )

    # Cruise_Timing (critical)
    timing_node = evaluator.add_parallel(
        id="Cruise_Timing",
        desc="Cruise departs in May 2026 allowing time for pre-cruise hiking",
        parent=cruise_node,
        critical=True
    )

    exists_timing = cruise is not None and bool(cruise.departure_date) and bool(cruise.duration_days)
    evaluator.add_custom_node(
        result=exists_timing,
        id="Timing_Info_Provided",
        desc="Specific departure date and duration are provided",
        parent=timing_node,
        critical=True
    )

    # Departure_Month
    n_dep_month = evaluator.add_leaf(
        id="Departure_Month",
        desc="Cruise departs in May 2026",
        parent=timing_node,
        critical=True
    )
    dep_month_claim = "The cruise departure occurs in May 2026."
    await evaluator.verify(
        claim=dep_month_claim,
        node=n_dep_month,
        sources=cruise.urls if cruise else [],
        additional_instruction="Explicitly confirm month/year as May 2026; minor formatting differences are acceptable."
    )

    # Specific_Departure_Date
    n_dep_date = evaluator.add_leaf(
        id="Specific_Departure_Date",
        desc="Specific departure date is provided",
        parent=timing_node,
        critical=True
    )
    dep_date_text = cruise.departure_date if cruise and cruise.departure_date else ""
    dep_date_claim = f"The cruise departs on {dep_date_text}."
    await evaluator.verify(
        claim=dep_date_claim,
        node=n_dep_date,
        sources=cruise.urls if cruise else [],
        additional_instruction="Verify the exact date appears; allow differences in date formatting."
    )

    # Cruise_Duration
    n_duration = evaluator.add_leaf(
        id="Cruise_Duration",
        desc="Cruise duration is 10-11 days",
        parent=timing_node,
        critical=True
    )
    duration_claim = "The cruise duration is between 10 and 11 days inclusive."
    await evaluator.verify(
        claim=duration_claim,
        node=n_duration,
        sources=cruise.urls if cruise else [],
        additional_instruction="Confirm the listed duration is either 10 or 11 days."
    )

    # Timing_Reference
    n_timing_ref = evaluator.add_leaf(
        id="Timing_Reference",
        desc="Provide URL reference confirming departure date and duration",
        parent=timing_node,
        critical=True
    )
    timing_ref_claim = "The provided URLs confirm both the specific departure date and the cruise duration."
    await evaluator.verify(
        claim=timing_ref_claim,
        node=n_timing_ref,
        sources=cruise.urls if cruise else [],
        additional_instruction="Pages should include date and duration details."
    )

    # Itinerary_Ports (critical)
    itin_node = evaluator.add_parallel(
        id="Itinerary_Ports",
        desc="Cruise itinerary includes required Alaska ports",
        parent=cruise_node,
        critical=True
    )

    exists_itin = cruise is not None and bool(cruise.itinerary_ports) and bool(cruise.urls)
    evaluator.add_custom_node(
        result=exists_itin,
        id="Itinerary_Info_Provided",
        desc="Itinerary info and references provided",
        parent=itin_node,
        critical=True
    )

    # Juneau
    n_juneau = evaluator.add_leaf(
        id="Juneau_Port",
        desc="Itinerary includes Juneau, Alaska",
        parent=itin_node,
        critical=True
    )
    juneau_claim = "The cruise itinerary includes Juneau, Alaska."
    await evaluator.verify(
        claim=juneau_claim,
        node=n_juneau,
        sources=cruise.urls if cruise else [],
        additional_instruction="Look for 'Juneau' explicitly in the itinerary."
    )

    # Ketchikan
    n_ketchikan = evaluator.add_leaf(
        id="Ketchikan_Port",
        desc="Itinerary includes Ketchikan, Alaska",
        parent=itin_node,
        critical=True
    )
    ketchikan_claim = "The cruise itinerary includes Ketchikan, Alaska."
    await evaluator.verify(
        claim=ketchikan_claim,
        node=n_ketchikan,
        sources=cruise.urls if cruise else [],
        additional_instruction="Look for 'Ketchikan' explicitly in the itinerary."
    )

    # Glacier Bay
    n_glacier = evaluator.add_leaf(
        id="Glacier_Bay",
        desc="Itinerary includes Glacier Bay National Park",
        parent=itin_node,
        critical=True
    )
    glacier_claim = "The cruise itinerary includes Glacier Bay National Park (including scenic cruising is acceptable)."
    await evaluator.verify(
        claim=glacier_claim,
        node=n_glacier,
        sources=cruise.urls if cruise else [],
        additional_instruction="Allow 'Glacier Bay' or 'Glacier Bay National Park'; scenic cruising counts."
    )

    # Itinerary_Reference
    n_itin_ref = evaluator.add_leaf(
        id="Itinerary_Reference",
        desc="Provide URL reference confirming all itinerary ports",
        parent=itin_node,
        critical=True
    )
    itin_ref_claim = "The provided URLs confirm the itinerary includes Juneau, Ketchikan, and Glacier Bay National Park."
    await evaluator.verify(
        claim=itin_ref_claim,
        node=n_itin_ref,
        sources=cruise.urls if cruise else [],
        additional_instruction="All three locations should be evident across the provided pages."
    )


async def verify_bike_park(evaluator: Evaluator, parent_node, park: Optional[BikeParkSpec]) -> None:
    # Post_Cruise_Activities (critical)
    post_node = evaluator.add_parallel(
        id="Post_Cruise_Activities",
        desc="Mountain biking activities after cruise completion",
        parent=parent_node,
        critical=True
    )

    # Bike_Park_Selection (critical)
    park_sel_node = evaluator.add_parallel(
        id="Bike_Park_Selection",
        desc="Whistler Mountain Bike Park identified for post-cruise activities",
        parent=post_node,
        critical=True
    )

    exists_park_info = park is not None and bool(park.name) and bool(park.location_city_province) and bool(park.urls)
    evaluator.add_custom_node(
        result=exists_park_info,
        id="Park_Info_Provided",
        desc="Bike park identity, location, and URLs provided",
        parent=park_sel_node,
        critical=True
    )

    # Park_Name
    n_park_name = evaluator.add_leaf(
        id="Park_Name",
        desc="Park is Whistler Mountain Bike Park",
        parent=park_sel_node,
        critical=True
    )
    park_name_claim = "The selected park is Whistler Mountain Bike Park."
    await evaluator.verify(
        claim=park_name_claim,
        node=n_park_name,
        sources=park.urls if park else [],
        additional_instruction="Confirm explicit naming of 'Whistler Mountain Bike Park'."
    )

    # Park_Location
    n_park_loc = evaluator.add_leaf(
        id="Park_Location",
        desc="Park is located in Whistler, British Columbia",
        parent=park_sel_node,
        critical=True
    )
    park_loc_claim = "Whistler Mountain Bike Park is located in Whistler, British Columbia."
    await evaluator.verify(
        claim=park_loc_claim,
        node=n_park_loc,
        sources=park.urls if park else [],
        additional_instruction="Location must match 'Whistler, British Columbia'."
    )

    # Park_Reference
    n_park_ref = evaluator.add_leaf(
        id="Park_Reference",
        desc="Provide URL reference confirming park identity and location",
        parent=park_sel_node,
        critical=True
    )
    park_ref_claim = "The provided URLs confirm the park's identity and location."
    await evaluator.verify(
        claim=park_ref_claim,
        node=n_park_ref,
        sources=park.urls if park else [],
        additional_instruction="Pages should clearly show the park is in Whistler, BC."
    )

    # Operating_Status (critical)
    ops_node = evaluator.add_parallel(
        id="Operating_Status",
        desc="Bike park is open during the planned visit dates in late May 2026",
        parent=post_node,
        critical=True
    )

    exists_ops = park is not None and bool(park.opening_dates_info or park.operating_late_may_confirmation) and bool(park.urls)
    evaluator.add_custom_node(
        result=exists_ops,
        id="Operating_Info_Provided",
        desc="Operating season/date info and references provided",
        parent=ops_node,
        critical=True
    )

    # Season_Opening
    n_season_open = evaluator.add_leaf(
        id="Season_Opening",
        desc="Confirm bike park opens for summer season by mid-May 2026",
        parent=ops_node,
        critical=True
    )
    season_open_claim = "Whistler Mountain Bike Park opens for the summer riding season by mid-May 2026."
    await evaluator.verify(
        claim=season_open_claim,
        node=n_season_open,
        sources=park.urls if park else [],
        additional_instruction="Accept official statements or historical schedule pages indicating mid-May opening."
    )

    # Visit_Date_Feasibility
    n_visit_ok = evaluator.add_leaf(
        id="Visit_Date_Feasibility",
        desc="Confirm park will be operating during late May 2026 post-cruise period",
        parent=ops_node,
        critical=True
    )
    visit_ok_claim = "The park will be operating during late May 2026 (after the cruise)."
    await evaluator.verify(
        claim=visit_ok_claim,
        node=n_visit_ok,
        sources=park.urls if park else [],
        additional_instruction="Confirm that late May 2026 falls within operating dates; allow typical season calendars."
    )

    # Operating_Hours_Reference
    n_ops_ref = evaluator.add_leaf(
        id="Operating_Hours_Reference",
        desc="Provide URL reference confirming operating dates/season",
        parent=ops_node,
        critical=True
    )
    ops_ref_claim = "The provided URLs confirm the park's operating season/dates for 2026."
    await evaluator.verify(
        claim=ops_ref_claim,
        node=n_ops_ref,
        sources=park.urls if park else [],
        additional_instruction="Pages should indicate the 2026 season or typical opening timeframe."
    )

    # Trail_Variety (critical subset: blue trails + reference)
    trail_node = evaluator.add_parallel(
        id="Trail_Variety",
        desc="Bike park offers trails suitable for intermediate riders",
        parent=post_node,
        critical=True
    )

    exists_trails = park is not None and bool(park.intermediate_trails_info or park.blue_trail_examples) and bool(park.urls)
    evaluator.add_custom_node(
        result=exists_trails,
        id="Trail_Info_Provided",
        desc="Intermediate/blue trail info and URLs provided",
        parent=trail_node,
        critical=True
    )

    # Blue_Trail_Availability
    n_blue = evaluator.add_leaf(
        id="Blue_Trail_Availability",
        desc="Park has blue (intermediate) difficulty trails available",
        parent=trail_node,
        critical=True
    )
    blue_claim = "Whistler Mountain Bike Park offers blue (intermediate) mountain bike trails."
    await evaluator.verify(
        claim=blue_claim,
        node=n_blue,
        sources=park.urls if park else [],
        additional_instruction="Confirm 'blue' or 'intermediate' trails are offered; examples suffice."
    )

    # Trail_Information_Reference
    n_tr_ref = evaluator.add_leaf(
        id="Trail_Information_Reference",
        desc="Provide URL reference confirming trail difficulty levels and variety",
        parent=trail_node,
        critical=True
    )
    tr_ref_claim = "The provided URLs confirm trail difficulty levels and variety at the park, including intermediate/blue options."
    await evaluator.verify(
        claim=tr_ref_claim,
        node=n_tr_ref,
        sources=park.urls if park else [],
        additional_instruction="Pages should list trail difficulties or show blue-trail examples."
    )


async def verify_transport(evaluator: Evaluator, parent_node, sd_sf: Optional[TransportSDtoSF], sf_whistler: Optional[TransportSFtoWhistler]) -> None:
    # Transportation_Logistics (non-critical)
    trans_node = evaluator.add_parallel(
        id="Transportation_Logistics",
        desc="Transportation connections between all locations",
        parent=parent_node,
        critical=False
    )

    # San_Diego_To_San_Francisco (non-critical)
    sdsf_node = evaluator.add_parallel(
        id="San_Diego_To_San_Francisco",
        desc="Transportation method from San Diego to San Francisco for cruise departure",
        parent=trans_node,
        critical=False
    )

    # Flight_Availability
    n_flight_avail = evaluator.add_leaf(
        id="Flight_Availability",
        desc="Confirm flight options exist from San Diego to San Francisco",
        parent=sdsf_node,
        critical=False
    )
    flight_avail_claim = "Commercial flight options exist from San Diego (SAN) to San Francisco (SFO)."
    await evaluator.verify(
        claim=flight_avail_claim,
        node=n_flight_avail,
        sources=(sd_sf.flight_urls if sd_sf else []),
        additional_instruction="Any airline or aggregator page showing SAN→SFO flights suffices."
    )

    # Budget_Airline_Option
    n_budget = evaluator.add_leaf(
        id="Budget_Airline_Option",
        desc="Budget airline options available for this route",
        parent=sdsf_node,
        critical=False
    )
    budget_name = (sd_sf.budget_airlines[0] if (sd_sf and sd_sf.budget_airlines) else "a budget airline")
    budget_claim = f"Budget airline options (e.g., {budget_name}) operate or offer flights on the SAN to SFO route."
    await evaluator.verify(
        claim=budget_claim,
        node=n_budget,
        sources=(sd_sf.flight_urls if sd_sf else []),
        additional_instruction="Southwest is commonly considered budget; others may qualify. Confirm via cited pages."
    )

    # Flight_Reference
    n_flight_ref = evaluator.add_leaf(
        id="Flight_Reference",
        desc="Provide URL reference for flight availability information",
        parent=sdsf_node,
        critical=False
    )
    flight_ref_claim = "The provided URLs substantiate flight availability between San Diego and San Francisco."
    await evaluator.verify(
        claim=flight_ref_claim,
        node=n_flight_ref,
        sources=(sd_sf.flight_urls if sd_sf else []),
        additional_instruction="Any legit airline/OTA schedule or route listing counts."
    )

    # San_Francisco_To_Whistler (non-critical)
    sfw_node = evaluator.add_parallel(
        id="San_Francisco_To_Whistler",
        desc="Transportation method from San Francisco to Whistler after cruise",
        parent=trans_node,
        critical=False
    )

    # Connection_Route
    n_conn_route = evaluator.add_leaf(
        id="Connection_Route",
        desc="Identify flight connection route (e.g., via Vancouver or Seattle)",
        parent=sfw_node,
        critical=False
    )
    conn_route_claim = "Travel from San Francisco to Whistler is typically via flight to Vancouver (YVR) or Seattle (SEA), followed by ground transfer."
    await evaluator.verify(
        claim=conn_route_claim,
        node=n_conn_route,
        sources=(sf_whistler.connection_urls if sf_whistler else []),
        additional_instruction="Any page indicating SFO→YVR or SFO→SEA then ground transport to Whistler suffices."
    )

    # Connection_Feasibility
    n_conn_feas = evaluator.add_leaf(
        id="Connection_Feasibility",
        desc="Confirm connection is feasible within reasonable timeframe",
        parent=sfw_node,
        critical=False
    )
    conn_feas_claim = "The connection route from San Francisco to Whistler is feasible within a reasonable timeframe."
    await evaluator.verify(
        claim=conn_feas_claim,
        node=n_conn_feas,
        sources=(sf_whistler.connection_urls if sf_whistler else []),
        additional_instruction="Airline schedules plus typical ground transfers (e.g., bus/shuttle) demonstrate feasibility."
    )

    # Connection_Reference
    n_conn_ref = evaluator.add_leaf(
        id="Connection_Reference",
        desc="Provide URL reference for connection information",
        parent=sfw_node,
        critical=False
    )
    conn_ref_claim = "The provided URLs substantiate the connection route and feasibility."
    await evaluator.verify(
        claim=conn_ref_claim,
        node=n_conn_ref,
        sources=(sf_whistler.connection_urls if sf_whistler else []),
        additional_instruction="Airline or transit sites are acceptable references."
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
    Evaluate an answer for the multi-activity outdoor vacation plan (late spring 2026).
    """
    # Initialize evaluator (root non-critical to allow adding both critical and non-critical children)
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

    # Extract structured plan info
    plan: PlanExtraction = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="vacation_plan_extraction",
    )

    # Add ground truth constraints (for context in summary)
    evaluator.add_ground_truth({
        "requirements": {
            "trail": {
                "distance_roundtrip": "≈3 miles",
                "elevation_gain": "≈900 ft",
                "difficulty": "moderate",
                "location": "San Diego area"
            },
            "cruise": {
                "operator": "Princess Cruises",
                "ship": "Ruby Princess",
                "port_roundtrip": "San Francisco",
                "departure_month": "May 2026",
                "duration_days": "10–11 days",
                "itinerary": ["Juneau", "Ketchikan", "Glacier Bay National Park"]
            },
            "bike_park": {
                "name": "Whistler Mountain Bike Park",
                "location": "Whistler, British Columbia",
                "operating": "Open by mid-May 2026; operating in late May",
                "trails": "Intermediate (blue) available"
            }
        }
    })

    # Build an essential aggregator to enforce critical components
    essential_node = evaluator.add_parallel(
        id="Essential_Components",
        desc="All essential components of the vacation plan must meet the specified requirements",
        parent=root,
        critical=True
    )

    # Verify hiking component
    await verify_hiking(evaluator, essential_node, plan.trail)

    # Verify cruise component
    await verify_cruise(evaluator, essential_node, plan.cruise)

    # Verify bike park component
    await verify_bike_park(evaluator, essential_node, plan.bike_park)

    # Transportation logistics (non-critical)
    await verify_transport(evaluator, root, plan.transport_sd_to_sf, plan.transport_sf_to_whistler)

    # Return structured summary
    return evaluator.get_summary()