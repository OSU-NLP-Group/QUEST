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
TASK_ID = "vacation_itinerary_2025"
TASK_DESCRIPTION = """A family is planning a comprehensive vacation in late November 2025 and needs to identify the specific details that meet all of their requirements:

Cruise Requirements:
1. The cruise must depart from a Florida port on the maiden voyage date of Disney Cruise Line's newest ship entering service in 2025
2. The ship must be themed around Disney heroes and villains
3. The ship must have a passenger capacity of approximately 4,000 passengers
4. The cruise must offer 4-night and 5-night itinerary options to the Bahamas and Western Caribbean

Theme Park Requirements:
5. The family will visit a California theme park located in Valencia
6. The theme park must currently have exactly 19 roller coasters
7. The theme park must have at least one roller coaster with a minimum height requirement of 54 inches
8. The theme park must be a Six Flags property

International Destination Requirements:
9. The destination must be accessible via direct flights from Istanbul operated by Turkish Airlines
10. US citizens must be able to enter this destination visa-free for up to 14 days with a confirmed hotel booking, health insurance, and return ticket
11. The destination must have exactly 5 UNESCO World Heritage Sites
12. The destination must offer adventure activities including hiking, diving, canyoning, and kayaking
13. One of the diving locations must feature coral reefs and is made up of islands
14. US passport holders need their passport valid for at least 6 months to enter

Budget Airline Requirements:
15. A budget airline must serve Fort Lauderdale, Florida
16. The same airline must have added Nassau, Bahamas as an international destination in 2025
17. The airline's total network must include approximately 56 destinations as of early 2025

Specify:
- The name of the cruise ship
- The exact maiden voyage departure date
- The departure port name and city
- The name of the theme park
- The name and capital city of the international destination country
- The name of the diving location in the international destination
- The names of at least 3 UNESCO World Heritage Sites in the international destination
- The name of the budget airline
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CruiseData(BaseModel):
    ship_name: Optional[str] = None
    ship_theme: Optional[str] = None
    maiden_voyage_date: Optional[str] = None  # e.g., "November 20, 2025"
    departure_port: Optional[str] = None      # e.g., "Port Everglades"
    departure_city: Optional[str] = None      # e.g., "Fort Lauderdale"
    departure_state: Optional[str] = None     # e.g., "Florida"
    passenger_capacity: Optional[str] = None  # allow strings like "around 4,000"
    itinerary_options: List[str] = Field(default_factory=list)

    ship_urls: List[str] = Field(default_factory=list)       # supports name/theme/newest ship entering service 2025
    voyage_urls: List[str] = Field(default_factory=list)     # supports maiden voyage date & port
    capacity_urls: List[str] = Field(default_factory=list)   # supports capacity
    itinerary_urls: List[str] = Field(default_factory=list)  # supports itinerary durations/destinations


class ThemeParkData(BaseModel):
    park_name: Optional[str] = None
    park_location_city: Optional[str] = None  # "Valencia"
    park_location_state: Optional[str] = None  # "California"
    roller_coaster_count: Optional[str] = None

    park_urls: List[str] = Field(default_factory=list)             # general park identification/location
    coaster_count_urls: List[str] = Field(default_factory=list)    # supports coaster count
    height_urls: List[str] = Field(default_factory=list)           # supports 54" min height ride
    operator_urls: List[str] = Field(default_factory=list)         # supports Six Flags operator


class DestinationData(BaseModel):
    country_name: Optional[str] = None  # "Oman"
    capital_city: Optional[str] = None  # "Muscat"
    diving_name: Optional[str] = None   # "Damaniyat Islands"
    unesco_sites: List[str] = Field(default_factory=list)  # at least 3 site names if provided in answer

    country_urls: List[str] = Field(default_factory=list)     # confirms country & capital
    flight_urls: List[str] = Field(default_factory=list)      # confirms Turkish Airlines direct flights
    visa_urls: List[str] = Field(default_factory=list)        # confirms visa-free 14 days + docs + 6-month validity
    unesco_urls: List[str] = Field(default_factory=list)      # confirms UNESCO count & sites
    adventure_urls: List[str] = Field(default_factory=list)   # confirms hiking/diving/canyoning/kayaking
    diving_urls: List[str] = Field(default_factory=list)      # confirms Damaniyat Islands coral reefs/islands


class AirlineData(BaseModel):
    airline_name: Optional[str] = None  # "Avelo Airlines"

    airline_urls: List[str] = Field(default_factory=list)   # general airline details
    fll_urls: List[str] = Field(default_factory=list)       # confirms service to Fort Lauderdale (FLL)
    nassau_urls: List[str] = Field(default_factory=list)    # confirms Nassau added in 2025
    network_urls: List[str] = Field(default_factory=list)   # confirms ~56 destinations as of early 2025


class VacationExtraction(BaseModel):
    cruise: Optional[CruiseData] = None
    theme_park: Optional[ThemeParkData] = None
    destination: Optional[DestinationData] = None
    airline: Optional[AirlineData] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_vacation_details() -> str:
    return """
    Extract all specified vacation details strictly from the provided answer. Organize the result into the following JSON structure with four top-level sections: cruise, theme_park, destination, airline. For each field, extract exactly what the answer states. If any field is missing, set it to null (or an empty list for list fields). Also extract all URLs explicitly mentioned in the answer that support each requirement.

    cruise:
      - ship_name: the name of Disney Cruise Line's newest ship entering service in 2025
      - ship_theme: the theme described for the ship
      - maiden_voyage_date: the date of the maiden voyage (e.g., "November 20, 2025")
      - departure_port: the port name (e.g., "Port Everglades")
      - departure_city: the city name (e.g., "Fort Lauderdale")
      - departure_state: the state name (e.g., "Florida")
      - passenger_capacity: a string for capacity (e.g., "~4,000", "approximately 4,000")
      - itinerary_options: list of textual items mentioning 4-night and/or 5-night options and destinations (Bahamas, Western Caribbean)
      - ship_urls: list of URLs in the answer that support the ship name, theme, and entry into service in 2025
      - voyage_urls: list of URLs in the answer that support the maiden voyage date and departure port/city/state
      - capacity_urls: list of URLs in the answer that support the passenger capacity
      - itinerary_urls: list of URLs in the answer that support the 4-night and 5-night Bahamas/Western Caribbean itineraries

    theme_park:
      - park_name: the park name (e.g., "Six Flags Magic Mountain")
      - park_location_city: the city (e.g., "Valencia")
      - park_location_state: the state (e.g., "California")
      - roller_coaster_count: the current number of roller coasters as stated (string; e.g., "19")
      - park_urls: URLs supporting the park identity/location
      - coaster_count_urls: URLs supporting the roller coaster count
      - height_urls: URLs supporting a roller coaster with minimum height 54 inches
      - operator_urls: URLs supporting that it is a Six Flags property

    destination:
      - country_name: the destination country name (e.g., "Oman")
      - capital_city: the capital city (e.g., "Muscat")
      - diving_name: the diving location name (e.g., "Damaniyat Islands")
      - unesco_sites: list of UNESCO site names mentioned in the answer (include all names given)
      - country_urls: URLs supporting the country and capital information
      - flight_urls: URLs supporting direct flights from Istanbul by Turkish Airlines (e.g., IST→MCT)
      - visa_urls: URLs supporting visa-free up to 14 days for US citizens with hotel booking, health insurance, return ticket; and 6-month passport validity
      - unesco_urls: URLs supporting the UNESCO count and site names
      - adventure_urls: URLs supporting availability of hiking, diving, canyoning, kayaking
      - diving_urls: URLs supporting Damaniyat Islands coral reefs and that it is an island group

    airline:
      - airline_name: the budget airline name (e.g., "Avelo Airlines")
      - airline_urls: URLs supporting the airline identification
      - fll_urls: URLs supporting service to Fort Lauderdale (FLL)
      - nassau_urls: URLs supporting Nassau, Bahamas added in 2025
      - network_urls: URLs supporting approximately 56 destinations as of early 2025

    IMPORTANT:
    - Extract only URLs explicitly present in the answer (plain urls or markdown links). Do not invent URLs.
    - If a URL is missing protocol, prepend http://.
    - If a required detail is not present in the answer, return null (or an empty list for lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_cruise(evaluator: Evaluator, parent_node, cruise: Optional[CruiseData]) -> None:
    comp_node = evaluator.add_parallel(
        id="cruise_component",
        desc="Verification of cruise ship selection and departure details",
        parent=parent_node,
        critical=False
    )

    # Ship identification
    ship_id_node = evaluator.add_parallel(
        id="cruise_ship_identification",
        desc="Identify Disney Cruise Line's newest ship entering service in 2025 themed around heroes and villains",
        parent=comp_node,
        critical=True
    )

    # Ship name verification
    ship_name_leaf = evaluator.add_leaf(
        id="ship_name_verification",
        desc="The cruise ship name is Disney Destiny",
        parent=ship_id_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cruise ship name is Disney Destiny.",
        node=ship_name_leaf,
        sources=cruise.ship_urls if cruise else [],
        additional_instruction="Confirm that Disney Cruise Line's newest ship entering service in 2025 is named Disney Destiny."
    )

    # Ship theme verification
    ship_theme_leaf = evaluator.add_leaf(
        id="ship_theme_verification",
        desc="The ship is themed around Disney heroes and villains",
        parent=ship_id_node,
        critical=True
    )
    await evaluator.verify(
        claim="The ship is themed around Disney heroes and villains.",
        node=ship_theme_leaf,
        sources=cruise.ship_urls if cruise else [],
        additional_instruction="Confirm that Disney Destiny is themed around Disney heroes and villains (accept reasonable phrasing variants)."
    )

    # Reference URL existence for ship details
    evaluator.add_custom_node(
        result=_non_empty_urls(cruise.ship_urls if cruise else []),
        id="ship_reference_url",
        desc="URL reference confirming ship details",
        parent=ship_id_node,
        critical=True
    )

    # Maiden voyage details
    voyage_node = evaluator.add_parallel(
        id="maiden_voyage_details",
        desc="Verify the maiden voyage departure date and port",
        parent=comp_node,
        critical=True
    )

    date_leaf = evaluator.add_leaf(
        id="departure_date_verification",
        desc="The maiden voyage departs on November 20, 2025",
        parent=voyage_node,
        critical=True
    )
    await evaluator.verify(
        claim="The maiden voyage departs on November 20, 2025.",
        node=date_leaf,
        sources=cruise.voyage_urls if cruise else [],
        additional_instruction="Verify the maiden voyage date is explicitly November 20, 2025."
    )

    port_leaf = evaluator.add_leaf(
        id="departure_port_verification",
        desc="The departure port is Port Everglades in Fort Lauderdale, Florida",
        parent=voyage_node,
        critical=True
    )
    await evaluator.verify(
        claim="The departure port is Port Everglades in Fort Lauderdale, Florida.",
        node=port_leaf,
        sources=cruise.voyage_urls if cruise else [],
        additional_instruction="Confirm the maiden voyage departs from Port Everglades located in Fort Lauderdale, Florida."
    )

    evaluator.add_custom_node(
        result=_non_empty_urls(cruise.voyage_urls if cruise else []),
        id="voyage_reference_url",
        desc="URL reference confirming maiden voyage details",
        parent=voyage_node,
        critical=True
    )

    # Capacity
    capacity_node = evaluator.add_parallel(
        id="ship_capacity_verification",
        desc="Verify ship passenger capacity meets requirements",
        parent=comp_node,
        critical=True
    )
    capacity_leaf = evaluator.add_leaf(
        id="passenger_capacity_check",
        desc="Ship has passenger capacity of approximately 4,000",
        parent=capacity_node,
        critical=True
    )
    await evaluator.verify(
        claim="The ship has a passenger capacity of approximately 4,000.",
        node=capacity_leaf,
        sources=cruise.capacity_urls if cruise else [],
        additional_instruction="Interpret 'approximately 4,000' generously (e.g., 3,900–4,100 range acceptable). Confirm approximate capacity around 4,000."
    )

    evaluator.add_custom_node(
        result=_non_empty_urls(cruise.capacity_urls if cruise else []),
        id="capacity_reference_url",
        desc="URL reference confirming capacity details",
        parent=capacity_node,
        critical=True
    )

    # Itinerary options
    itinerary_node = evaluator.add_parallel(
        id="itinerary_options_verification",
        desc="Verify cruise offers required itinerary durations",
        parent=comp_node,
        critical=True
    )
    itinerary_leaf = evaluator.add_leaf(
        id="itinerary_duration_check",
        desc="Cruise offers both 4-night and 5-night itinerary options to Bahamas and Western Caribbean",
        parent=itinerary_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cruise offers both 4-night and 5-night itinerary options to the Bahamas and the Western Caribbean.",
        node=itinerary_leaf,
        sources=cruise.itinerary_urls if cruise else [],
        additional_instruction="Confirm BOTH 4-night and 5-night itineraries are offered and destinations include Bahamas and Western Caribbean."
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(cruise.itinerary_urls if cruise else []),
        id="itinerary_reference_url",
        desc="URL reference confirming itinerary options",
        parent=itinerary_node,
        critical=True
    )


async def verify_theme_park(evaluator: Evaluator, parent_node, park: Optional[ThemeParkData]) -> None:
    comp_node = evaluator.add_parallel(
        id="theme_park_component",
        desc="Verification of California theme park selection and characteristics",
        parent=parent_node,
        critical=False
    )

    # Identification
    id_node = evaluator.add_parallel(
        id="theme_park_identification",
        desc="Identify the California theme park in Valencia with 19 roller coasters",
        parent=comp_node,
        critical=True
    )

    park_name_leaf = evaluator.add_leaf(
        id="park_name_verification",
        desc="The theme park name is Six Flags Magic Mountain",
        parent=id_node,
        critical=True
    )
    await evaluator.verify(
        claim="The theme park name is Six Flags Magic Mountain.",
        node=park_name_leaf,
        sources=park.park_urls if park else [],
        additional_instruction="Confirm the park identified is Six Flags Magic Mountain."
    )

    park_location_leaf = evaluator.add_leaf(
        id="park_location_verification",
        desc="The park is located in Valencia, California",
        parent=id_node,
        critical=True
    )
    await evaluator.verify(
        claim="The park is located in Valencia, California.",
        node=park_location_leaf,
        sources=park.park_urls if park else [],
        additional_instruction="Confirm the location 'Valencia, California' for Six Flags Magic Mountain."
    )

    evaluator.add_custom_node(
        result=_non_empty_urls(park.park_urls if park else []),
        id="park_reference_url",
        desc="URL reference confirming park details",
        parent=id_node,
        critical=True
    )

    # Roller coaster count
    count_node = evaluator.add_parallel(
        id="roller_coaster_count",
        desc="Verify the theme park has exactly 19 roller coasters",
        parent=comp_node,
        critical=True
    )
    count_leaf = evaluator.add_leaf(
        id="coaster_count_check",
        desc="Park currently has 19 roller coasters",
        parent=count_node,
        critical=True
    )
    await evaluator.verify(
        claim="The park currently has exactly 19 roller coasters.",
        node=count_leaf,
        sources=park.coaster_count_urls if park else [],
        additional_instruction="Confirm the current roller coaster count is 19 (exact)."
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(park.coaster_count_urls if park else []),
        id="coaster_count_reference_url",
        desc="URL reference confirming roller coaster count",
        parent=count_node,
        critical=True
    )

    # Height requirement
    height_node = evaluator.add_parallel(
        id="height_requirement_verification",
        desc="Verify park has rides with 54-inch minimum height requirement",
        parent=comp_node,
        critical=True
    )
    height_leaf = evaluator.add_leaf(
        id="height_requirement_check",
        desc="Park has at least one roller coaster requiring minimum 54 inches height",
        parent=height_node,
        critical=True
    )
    await evaluator.verify(
        claim="The park has at least one roller coaster with a minimum height requirement of 54 inches.",
        node=height_leaf,
        sources=park.height_urls if park else [],
        additional_instruction="Confirm any ride with minimum height requirement 54 inches at Six Flags Magic Mountain."
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(park.height_urls if park else []),
        id="height_reference_url",
        desc="URL reference confirming height requirements",
        parent=height_node,
        critical=True
    )

    # Operator
    operator_node = evaluator.add_parallel(
        id="park_operator_verification",
        desc="Verify the park is a Six Flags property",
        parent=comp_node,
        critical=True
    )
    operator_leaf = evaluator.add_leaf(
        id="operator_check",
        desc="The park is operated by Six Flags",
        parent=operator_node,
        critical=True
    )
    await evaluator.verify(
        claim="The park is operated by Six Flags.",
        node=operator_leaf,
        sources=park.operator_urls if park else [],
        additional_instruction="Confirm the park is a Six Flags property/operator."
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(park.operator_urls if park else []),
        id="operator_reference_url",
        desc="URL reference confirming park operator",
        parent=operator_node,
        critical=True
    )


async def verify_destination(evaluator: Evaluator, parent_node, dest: Optional[DestinationData]) -> None:
    comp_node = evaluator.add_parallel(
        id="international_destination_component",
        desc="Verification of international destination selection and requirements",
        parent=parent_node,
        critical=False
    )

    # Country identification
    country_node = evaluator.add_parallel(
        id="destination_country_identification",
        desc="Identify the international destination country and capital city",
        parent=comp_node,
        critical=True
    )

    country_leaf = evaluator.add_leaf(
        id="country_name_verification",
        desc="The destination country is Oman (Sultanate of Oman)",
        parent=country_node,
        critical=True
    )
    await evaluator.verify(
        claim="The destination country is Oman (Sultanate of Oman).",
        node=country_leaf,
        sources=dest.country_urls if dest else [],
        additional_instruction="Confirm the country is Oman (Sultanate of Oman)."
    )

    capital_leaf = evaluator.add_leaf(
        id="capital_city_verification",
        desc="The capital city is Muscat",
        parent=country_node,
        critical=True
    )
    await evaluator.verify(
        claim="The capital city is Muscat.",
        node=capital_leaf,
        sources=dest.country_urls if dest else [],
        additional_instruction="Confirm that the capital city of Oman is Muscat."
    )

    evaluator.add_custom_node(
        result=_non_empty_urls(dest.country_urls if dest else []),
        id="country_reference_url",
        desc="URL reference confirming country details",
        parent=country_node,
        critical=True
    )

    # Flight connectivity
    flight_node = evaluator.add_parallel(
        id="flight_connectivity_verification",
        desc="Verify Turkish Airlines operates direct flights from Istanbul",
        parent=comp_node,
        critical=True
    )
    route_leaf = evaluator.add_leaf(
        id="turkish_airlines_route_check",
        desc="Turkish Airlines operates direct flights from Istanbul to the destination",
        parent=flight_node,
        critical=True
    )
    await evaluator.verify(
        claim="Turkish Airlines operates direct flights from Istanbul to Muscat, Oman.",
        node=route_leaf,
        sources=dest.flight_urls if dest else [],
        additional_instruction="Check for Turkish Airlines direct service IST→MCT (Muscat). Verify explicitly 'direct/nonstop'."
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(dest.flight_urls if dest else []),
        id="flight_reference_url",
        desc="URL reference confirming Turkish Airlines flights",
        parent=flight_node,
        critical=True
    )

    # Visa requirements
    visa_node = evaluator.add_parallel(
        id="visa_requirements_verification",
        desc="Verify US citizens can enter visa-free for 14 days with required documentation",
        parent=comp_node,
        critical=True
    )
    visa_free_leaf = evaluator.add_leaf(
        id="visa_free_duration_check",
        desc="US citizens can visit visa-free for up to 14 days",
        parent=visa_node,
        critical=True
    )
    await evaluator.verify(
        claim="US citizens can visit Oman visa-free for up to 14 days.",
        node=visa_free_leaf,
        sources=dest.visa_urls if dest else [],
        additional_instruction="Confirm visa-free entry (or visa-on-arrival waiver) for US citizens up to 14 days in Oman. Use official or authoritative sources."
    )

    entry_req_leaf = evaluator.add_leaf(
        id="entry_requirements_check",
        desc="Entry requires confirmed hotel booking, health insurance, and return ticket",
        parent=visa_node,
        critical=True
    )
    await evaluator.verify(
        claim="Entry to Oman requires a confirmed hotel booking, health insurance, and a return ticket.",
        node=entry_req_leaf,
        sources=dest.visa_urls if dest else [],
        additional_instruction="Confirm that these documents are required for visa-free short stay."
    )

    passport_leaf = evaluator.add_leaf(
        id="passport_validity_check",
        desc="US passport must be valid for at least 6 months",
        parent=visa_node,
        critical=True
    )
    await evaluator.verify(
        claim="A US passport must be valid for at least 6 months to enter Oman.",
        node=passport_leaf,
        sources=dest.visa_urls if dest else [],
        additional_instruction="Confirm 6-month passport validity requirement for entry."
    )

    evaluator.add_custom_node(
        result=_non_empty_urls(dest.visa_urls if dest else []),
        id="visa_reference_url",
        desc="URL reference confirming visa requirements",
        parent=visa_node,
        critical=True
    )

    # UNESCO sites
    unesco_node = evaluator.add_parallel(
        id="unesco_sites_verification",
        desc="Verify destination has exactly 5 UNESCO World Heritage Sites",
        parent=comp_node,
        critical=False  # set non-critical to allow partial credit for site names while count can be critical
    )
    unesco_count_leaf = evaluator.add_leaf(
        id="unesco_count_check",
        desc="The destination has exactly 5 UNESCO World Heritage Sites",
        parent=unesco_node,
        critical=True
    )
    await evaluator.verify(
        claim="Oman has exactly 5 UNESCO World Heritage Sites.",
        node=unesco_count_leaf,
        sources=dest.unesco_urls if dest else [],
        additional_instruction="Use UNESCO official listings or authoritative sources; verify exact count equals 5."
    )

    site_names_node = evaluator.add_parallel(
        id="unesco_site_names",
        desc="Provide names of at least 3 UNESCO sites",
        parent=unesco_node,
        critical=False
    )
    # Site 1: Bahla Fort
    site1_leaf = evaluator.add_leaf(
        id="unesco_site_1_verification",
        desc="First UNESCO site: Bahla Fort (inscribed 1987)",
        parent=site_names_node,
        critical=False
    )
    await evaluator.verify(
        claim="Bahla Fort is a UNESCO World Heritage Site in Oman.",
        node=site1_leaf,
        sources=dest.unesco_urls if dest else [],
        additional_instruction="Confirm Bahla Fort appears on UNESCO WH list for Oman."
    )
    # Site 2: Bat, Al-Khutm and Al-Ayn
    site2_leaf = evaluator.add_leaf(
        id="unesco_site_2_verification",
        desc="Second UNESCO site: Archaeological Sites of Bat, Al-Khutm and Al-Ayn (inscribed 1988)",
        parent=site_names_node,
        critical=False
    )
    await evaluator.verify(
        claim="The Archaeological Sites of Bat, Al-Khutm and Al-Ayn are a UNESCO World Heritage Site in Oman.",
        node=site2_leaf,
        sources=dest.unesco_urls if dest else [],
        additional_instruction="Confirm Bat, Al-Khutm and Al-Ayn ensemble is listed as UNESCO WH site in Oman."
    )
    # Site 3: Land of Frankincense or Aflaj Irrigation System or Ancient City of Qalhat
    site3_leaf = evaluator.add_leaf(
        id="unesco_site_3_verification",
        desc="Third UNESCO site: Land of Frankincense or Aflaj Irrigation System or Ancient City of Qalhat",
        parent=site_names_node,
        critical=False
    )
    await evaluator.verify(
        claim="Land of Frankincense is a UNESCO World Heritage Site in Oman.",
        node=site3_leaf,
        sources=dest.unesco_urls if dest else [],
        additional_instruction="Alternatively accept Aflaj Irrigation Systems of Oman or Ancient City of Qalhat as UNESCO sites; confirm at least one of these."
    )

    evaluator.add_custom_node(
        result=_non_empty_urls(dest.unesco_urls if dest else []),
        id="unesco_reference_url",
        desc="URL reference confirming UNESCO World Heritage Sites",
        parent=unesco_node,
        critical=True
    )

    # Adventure activities
    adventure_node = evaluator.add_parallel(
        id="adventure_activities_verification",
        desc="Verify destination offers required adventure activities",
        parent=comp_node,
        critical=True
    )
    activities_leaf = evaluator.add_leaf(
        id="activity_types_check",
        desc="Destination offers hiking, diving, canyoning, and kayaking",
        parent=adventure_node,
        critical=True
    )
    await evaluator.verify(
        claim="Oman offers adventure activities including hiking, diving, canyoning, and kayaking.",
        node=activities_leaf,
        sources=dest.adventure_urls if dest else [],
        additional_instruction="Confirm availability of all four activities: hiking, diving, canyoning, kayaking."
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(dest.adventure_urls if dest else []),
        id="activities_reference_url",
        desc="URL reference confirming adventure activities",
        parent=adventure_node,
        critical=True
    )

    # Diving location
    diving_node = evaluator.add_parallel(
        id="diving_location_identification",
        desc="Identify the diving location with coral reefs",
        parent=comp_node,
        critical=True
    )
    diving_leaf = evaluator.add_leaf(
        id="diving_location_name_verification",
        desc="The diving location is Damaniyat Islands featuring coral reefs",
        parent=diving_node,
        critical=True
    )
    await evaluator.verify(
        claim="The diving location is the Damaniyat Islands, which are an island group featuring coral reefs.",
        node=diving_leaf,
        sources=dest.diving_urls if dest else [],
        additional_instruction="Confirm Damaniyat Islands are made up of islands and feature coral reefs suitable for diving."
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(dest.diving_urls if dest else []),
        id="diving_reference_url",
        desc="URL reference confirming diving location",
        parent=diving_node,
        critical=True
    )


async def verify_airline(evaluator: Evaluator, parent_node, air: Optional[AirlineData]) -> None:
    comp_node = evaluator.add_parallel(
        id="budget_airline_component",
        desc="Verification of budget airline selection and route network",
        parent=parent_node,
        critical=False
    )

    # Airline identification
    id_node = evaluator.add_parallel(
        id="airline_identification",
        desc="Identify the budget airline meeting all requirements",
        parent=comp_node,
        critical=True
    )
    airline_leaf = evaluator.add_leaf(
        id="airline_name_verification",
        desc="The budget airline name is Avelo Airlines",
        parent=id_node,
        critical=True
    )
    await evaluator.verify(
        claim="The budget airline name is Avelo Airlines.",
        node=airline_leaf,
        sources=air.airline_urls if air else [],
        additional_instruction="Confirm the airline identified in the answer is Avelo Airlines."
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(air.airline_urls if air else []),
        id="airline_reference_url",
        desc="URL reference confirming airline details",
        parent=id_node,
        critical=True
    )

    # Fort Lauderdale service
    fll_node = evaluator.add_parallel(
        id="fort_lauderdale_service_verification",
        desc="Verify airline serves Fort Lauderdale, Florida",
        parent=comp_node,
        critical=True
    )
    fll_leaf = evaluator.add_leaf(
        id="fort_lauderdale_route_check",
        desc="Airline serves Fort Lauderdale (FLL)",
        parent=fll_node,
        critical=True
    )
    await evaluator.verify(
        claim="The airline serves Fort Lauderdale (FLL).",
        node=fll_leaf,
        sources=air.fll_urls if air else [],
        additional_instruction="Confirm service/operations at Fort Lauderdale-Hollywood International Airport (FLL)."
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(air.fll_urls if air else []),
        id="fll_reference_url",
        desc="URL reference confirming Fort Lauderdale service",
        parent=fll_node,
        critical=True
    )

    # Nassau service added in 2025
    nas_node = evaluator.add_parallel(
        id="nassau_service_verification",
        desc="Verify airline added Nassau, Bahamas in 2025",
        parent=comp_node,
        critical=True
    )
    nas_leaf = evaluator.add_leaf(
        id="nassau_route_check",
        desc="Airline added Nassau, Bahamas as international destination in 2025",
        parent=nas_node,
        critical=True
    )
    await evaluator.verify(
        claim="In 2025, the airline added Nassau, Bahamas as an international destination.",
        node=nas_leaf,
        sources=air.nassau_urls if air else [],
        additional_instruction="Confirm that Nassau (NAS) was added to the airline's network in the year 2025 (press releases or news acceptable)."
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(air.nassau_urls if air else []),
        id="nassau_reference_url",
        desc="URL reference confirming Nassau service added in 2025",
        parent=nas_node,
        critical=True
    )

    # Network size ~56
    net_node = evaluator.add_parallel(
        id="network_size_verification",
        desc="Verify airline network includes approximately 56 destinations",
        parent=comp_node,
        critical=True
    )
    net_leaf = evaluator.add_leaf(
        id="network_size_check",
        desc="Airline network includes approximately 56 destinations as of early 2025",
        parent=net_node,
        critical=True
    )
    await evaluator.verify(
        claim="As of early 2025, the airline's network includes approximately 56 destinations.",
        node=net_leaf,
        sources=air.network_urls if air else [],
        additional_instruction="Accept reasonable approximations around 56 (e.g., 54–58) if explicitly stated as 'approximately' in authoritative sources."
    )
    evaluator.add_custom_node(
        result=_non_empty_urls(air.network_urls if air else []),
        id="network_reference_url",
        desc="URL reference confirming network size",
        parent=net_node,
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
    Evaluate the multi-component vacation itinerary answer against the rubric.
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

    # Extract all details in one pass
    extraction = await evaluator.extract(
        prompt=prompt_extract_vacation_details(),
        template_class=VacationExtraction,
        extraction_name="vacation_details",
    )

    # Top-level verification node (non-critical to allow partial credit across components)
    top_node = evaluator.add_parallel(
        id="vacation_itinerary_verification",
        desc="Verify that all components of the planned multi-destination vacation meet the specified requirements",
        parent=root,
        critical=False
    )

    # Build and verify subcomponents
    await verify_cruise(evaluator, top_node, extraction.cruise)
    await verify_theme_park(evaluator, top_node, extraction.theme_park)
    await verify_destination(evaluator, top_node, extraction.destination)
    await verify_airline(evaluator, top_node, extraction.airline)

    # Return structured summary
    return evaluator.get_summary()