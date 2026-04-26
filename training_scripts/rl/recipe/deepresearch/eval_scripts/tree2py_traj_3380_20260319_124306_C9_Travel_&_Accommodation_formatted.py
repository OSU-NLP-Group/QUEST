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
TASK_ID = "hal_apac_2026_2027_extension"
TASK_DESCRIPTION = """
Identify a Holland America Line cruise itinerary scheduled for the 2026-2027 season that satisfies all of the following criteria:

Cruise Specifications:
1. The cruise must operate in the Asia-Pacific region
2. The ship must have a passenger capacity between 1,400 and 2,700 guests
3. The itinerary must be between 7 and 16 days in duration
4. At least one port of call must provide access to a UNESCO World Heritage Site through a shore excursion or within reasonable travel distance (under 3 hours)

Connectivity Requirements:
5. The cruise's departure port must be accessible via commercial airline service with no more than one connection from a major US hub airport
6. The departure port's international airport must have at least one of the following transit amenities: airline lounge facilities, airport rest facilities (such as sleeping pods or day rooms), or an airport transit hotel

Post-Cruise Extension Option:
7. The cruise's final port or a nearby airport (within 2 hours) must offer commercial flight connections to a destination that has a Four Seasons resort property
8. This Four Seasons resort must be located in a country that offers either visa-free entry or e-visa/visa-on-arrival for US passport holders
9. The Four Seasons property must have at least 60 rooms, villas, or accommodations

Provide the following information for your identified solution:
- Cruise name and ship name
- Cruise duration and operating region
- Departure and final ports
- Ship passenger capacity
- Name and location of at least one accessible UNESCO World Heritage Site with its corresponding port of call
- Specific transit amenity available at the departure port's airport
- Four Seasons resort name and location that can be reached post-cruise
- Visa entry category for US citizens for the Four Seasons resort's country
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class CruiseCore(BaseModel):
    cruise_name: Optional[str] = None
    ship_name: Optional[str] = None
    region: Optional[str] = None  # e.g., Asia-Pacific, Asia, Pacific
    duration_days: Optional[str] = None  # keep as string to be robust, e.g., "14", "14-day"
    departure_port: Optional[str] = None
    final_port: Optional[str] = None
    cruise_urls: List[str] = Field(default_factory=list)  # itinerary/official HAL page(s)


class ShipSpec(BaseModel):
    passenger_capacity: Optional[str] = None  # keep as string; e.g., "1,964"
    capacity_urls: List[str] = Field(default_factory=list)  # URL(s) stating capacity (ship page, credible source)
    ship_spec_urls: List[str] = Field(default_factory=list)  # additional ship details/specs URLs


class SeasonInfo(BaseModel):
    season_label: Optional[str] = None  # e.g., "2026–2027" or "2026-2027"
    season_urls: List[str] = Field(default_factory=list)  # URL(s) confirming 2026-2027 schedule


class Connectivity(BaseModel):
    departure_airport_name: Optional[str] = None
    departure_airport_code: Optional[str] = None  # e.g., HND, NRT, SIN
    us_hub: Optional[str] = None  # e.g., LAX, JFK, SFO, ORD, ATL
    flight_route_example: Optional[str] = None  # free text route example from the answer
    flight_urls: List[str] = Field(default_factory=list)  # airline/OTA route proof URLs


class AirportAmenity(BaseModel):
    amenity_airport_code: Optional[str] = None  # usually same as departure_airport_code
    amenity_type: Optional[str] = None  # one of: airline lounge, rest facility, transit hotel
    amenity_name: Optional[str] = None  # specific lounge/hotel/pod brand if given
    amenity_urls: List[str] = Field(default_factory=list)  # airport official page or credible source


class UNESCOAccess(BaseModel):
    site_name: Optional[str] = None
    site_location: Optional[str] = None  # city/area + country
    port_of_call: Optional[str] = None  # which cruise port provides access
    est_travel_time: Optional[str] = None  # e.g., "2h 30m", "90 minutes", must be <= 3 hours
    site_official_urls: List[str] = Field(default_factory=list)  # UNESCO official page(s)
    port_to_site_urls: List[str] = Field(default_factory=list)  # shore ex / guide pages showing access/time
    itinerary_urls: List[str] = Field(default_factory=list)  # itinerary URL showing call at the port


class PostCruiseFS(BaseModel):
    final_airport_code: Optional[str] = None  # airport near final port (<=2 hours)
    fs_property_name: Optional[str] = None
    fs_location_city: Optional[str] = None
    fs_location_country: Optional[str] = None
    room_count: Optional[str] = None  # keep as string; e.g., "200" or "197 keys"
    fs_official_urls: List[str] = Field(default_factory=list)  # Four Seasons official property page(s)
    room_count_urls: List[str] = Field(default_factory=list)  # can be same as official page; add others if any
    flight_connectivity_urls: List[str] = Field(default_factory=list)  # final airport -> FS destination flights
    visa_category: Optional[str] = None  # "visa-free", "e-visa", "visa-on-arrival"
    visa_urls: List[str] = Field(default_factory=list)  # gov/embassy/credible travel policy page(s)


class CruisePackageExtraction(BaseModel):
    core: Optional[CruiseCore] = None
    ship: Optional[ShipSpec] = None
    season: Optional[SeasonInfo] = None
    connectivity: Optional[Connectivity] = None
    amenity: Optional[AirportAmenity] = None
    unesco: Optional[UNESCOAccess] = None
    post_cruise: Optional[PostCruiseFS] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cruise_package() -> str:
    return """
Extract a single Holland America Line cruise itinerary from the answer that best satisfies ALL requirements. If multiple are mentioned, pick the one that most clearly meets the criteria and has the most complete sources. Return structured JSON with the following fields. Only extract information explicitly present in the answer; do not invent anything. For any missing field, return null or an empty list as appropriate.

core:
  cruise_name: the marketed cruise name, if provided
  ship_name: the precise ship name
  region: the operating region label (e.g., "Asia-Pacific", "Asia", "Pacific")
  duration_days: number of days as given in the answer (keep as string, e.g., "14" or "14-day")
  departure_port: city + country if given (example: "Yokohama, Japan")
  final_port: city + country if given
  cruise_urls: list of URL(s) from the answer that specifically show the HAL itinerary details (HAL official preferred)

ship:
  passenger_capacity: passenger capacity as stated in the answer (keep as string)
  capacity_urls: URL(s) in the answer that document the ship's capacity
  ship_spec_urls: URL(s) in the answer to the ship's specification page(s)

season:
  season_label: season string mentioned in the answer, e.g., "2026–2027"
  season_urls: URL(s) that confirm the schedule is in 2026 or 2027 (HAL official preferred)

connectivity:
  departure_airport_name: the name of the international airport serving the departure port, if stated
  departure_airport_code: IATA code if given, e.g. "HND"
  us_hub: a major US hub mentioned in the answer (e.g., "LAX", "JFK", "SFO", "ORD", "ATL")
  flight_route_example: a concise example of a valid routing from the hub to the departure airport with <=1 connection, if given in the answer
  flight_urls: URL(s) from the answer to airline/OTA pages that support flight availability with no more than one connection

amenity:
  amenity_airport_code: IATA code of the departure airport for the amenity
  amenity_type: one of ["airline lounge", "rest facility", "transit hotel"] as stated in the answer
  amenity_name: specific brand/name if provided (e.g., "Aerotel", "YYZ Plaza Premium Lounge")
  amenity_urls: URL(s) from the answer that show the amenity exists at the departure airport

unesco:
  site_name: the UNESCO World Heritage site's proper name
  site_location: city/area and country of the site
  port_of_call: which cruise port gives access to this site
  est_travel_time: the estimated travel time from port to the site (e.g., "2h", "2 hours 30 minutes")
  site_official_urls: URL(s) to UNESCO listing or official site page(s)
  port_to_site_urls: URL(s) that document access from the port and (ideally) the sub-3h travel time (shore excursion page/guide acceptable)
  itinerary_urls: URL(s) that show the cruise actually calls at the specified port (can reuse core.cruise_urls if applicable)

post_cruise:
  final_airport_code: the airport near the final port used to fly to the Four Seasons destination (within ~2h from final port if stated)
  fs_property_name: the Four Seasons property name
  fs_location_city: city/island of the Four Seasons resort
  fs_location_country: country of the Four Seasons resort
  room_count: total rooms/villas/keys count as stated in the answer (keep as string)
  fs_official_urls: Four Seasons official property page URL(s) from the answer
  room_count_urls: URL(s) that state the room/villa count (may be the same as fs_official_urls)
  flight_connectivity_urls: URL(s) from the answer that show commercial flight options from the final airport to the FS destination with <=1 connection
  visa_category: one of ["visa-free", "e-visa", "visa-on-arrival"] as stated for US citizens for the Four Seasons country
  visa_urls: URL(s) that document the visa policy for US citizens for that country

Important:
- Extract only URLs explicitly present in the answer; do not infer or create new URLs.
- When multiple URLs are present for a field, include them all.
- If a field is not explicitly provided in the answer, set it to null (or [] for URL lists).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


def _first_or_none(lst: Optional[List[str]]) -> Optional[str]:
    arr = _safe_list(lst)
    return arr[0] if arr else None


def _merge_sources(*args: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for lst in args:
        for url in _safe_list(lst):
            if url and url not in seen:
                seen.add(url)
                out.append(url)
    return out


# --------------------------------------------------------------------------- #
# Verification section builders                                               #
# --------------------------------------------------------------------------- #
async def build_cruise_product_section(evaluator: Evaluator, parent, data: CruisePackageExtraction):
    section = evaluator.add_parallel(
        id="cruise_product_identification_and_verification",
        desc="Complete identification and verification of the Holland America Line cruise product meeting all core specifications",
        parent=parent,
        critical=False
    )

    core = data.core or CruiseCore()
    ship = data.ship or ShipSpec()
    season = data.season or SeasonInfo()

    # 1) Cruise Line & Season Verification
    brand_season = evaluator.add_parallel(
        id="cruise_line_and_season_verification",
        desc="Verification of cruise operator and operating season",
        parent=section,
        critical=False
    )

    # 1.a) Cruise Line brand confirmation
    brand_group = evaluator.add_parallel(
        id="cruise_line_brand_confirmation_group",
        desc="The cruise must be operated by Holland America Line",
        parent=brand_season,
        critical=False
    )

    # Existence of HAL cruise reference URL(s)
    evaluator.add_custom_node(
        result=bool(_safe_list(core.cruise_urls)),
        id="cruise_line_reference_url",
        desc="Provide URL reference to Holland America Line cruise page confirming the operator",
        parent=brand_group,
        critical=True
    )

    brand_leaf = evaluator.add_leaf(
        id="cruise_line_brand_confirmation",
        desc="The cruise is operated by Holland America Line",
        parent=brand_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage confirms that the cruise/itinerary is operated by Holland America Line (HAL).",
        node=brand_leaf,
        sources=_safe_list(core.cruise_urls),
        additional_instruction="Look for HAL branding, official site domain, and operator wording on the page. Minor formatting differences are acceptable."
    )

    # 1.b) Operating season timeframe
    season_group = evaluator.add_parallel(
        id="operating_season_timeframe_group",
        desc="The cruise must be scheduled for the 2026-2027 season",
        parent=brand_season,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(season.season_urls)),
        id="season_reference_url",
        desc="Provide URL reference confirming the 2026-2027 operating schedule",
        parent=season_group,
        critical=True
    )

    season_leaf = evaluator.add_leaf(
        id="operating_season_timeframe",
        desc="The cruise is scheduled for the 2026-2027 season",
        parent=season_group,
        critical=True
    )
    await evaluator.verify(
        claim="This page indicates that the sailing(s) for the identified itinerary occur in 2026 or 2027 (i.e., the 2026–2027 season).",
        node=season_leaf,
        sources=_safe_list(season.season_urls) or _safe_list(core.cruise_urls),
        additional_instruction="Accept listings or date pickers showing departures in 2026 and/or 2027."
    )

    # 2) Ship Specifications Verification
    ship_specs = evaluator.add_parallel(
        id="ship_specifications_verification",
        desc="Verification of ship capacity and suitability",
        parent=section,
        critical=False
    )

    # 2.a) Passenger capacity within range
    capacity_group = evaluator.add_parallel(
        id="ship_passenger_capacity_range_group",
        desc="The ship's passenger capacity must be between 1,400 and 2,700 guests",
        parent=ship_specs,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(ship.capacity_urls)),
        id="capacity_reference_url",
        desc="Provide URL reference documenting the ship's passenger capacity",
        parent=capacity_group,
        critical=True
    )

    capacity_leaf = evaluator.add_leaf(
        id="ship_passenger_capacity_range",
        desc="Ship passenger capacity within [1,400, 2,700]",
        parent=capacity_group,
        critical=True
    )
    capacity_claim = f"The ship's stated passenger capacity is {ship.passenger_capacity or 'provided on the page'}, which lies between 1,400 and 2,700 guests inclusive."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=_merge_sources(ship.capacity_urls, ship.ship_spec_urls),
        additional_instruction="Verify the total passenger capacity or guest count from the ship page or a credible source; allow standard phrasing variations like 'guests' or 'passengers'."
    )

    # 2.b) Ship name identification (non-critical)
    ship_name_group = evaluator.add_parallel(
        id="ship_name_identification_group",
        desc="Provide the specific name of the cruise ship operating the itinerary",
        parent=ship_specs,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(ship.ship_spec_urls)),
        id="ship_details_reference_url",
        desc="Provide URL reference to the ship's specifications page",
        parent=ship_name_group,
        critical=False
    )

    if core.ship_name:
        ship_name_leaf = evaluator.add_leaf(
            id="ship_name_identification",
            desc="Ship name matches the identified vessel operating the itinerary",
            parent=ship_name_group,
            critical=False
        )
        await evaluator.verify(
            claim=f"The ship operating the itinerary is named '{core.ship_name}'.",
            node=ship_name_leaf,
            sources=_merge_sources(core.cruise_urls, ship.ship_spec_urls),
            additional_instruction="Allow minor formatting differences (e.g., dashes, punctuation) when matching ship name."
        )

    # 3) Itinerary Specifications Verification
    itin_specs = evaluator.add_parallel(
        id="itinerary_specifications_verification",
        desc="Verification of itinerary duration, region, and port details",
        parent=section,
        critical=False
    )

    # 3.a) Duration in range
    duration_group = evaluator.add_parallel(
        id="cruise_duration_range_compliance_group",
        desc="The cruise duration must be between 7 and 16 days",
        parent=itin_specs,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(core.cruise_urls)),
        id="duration_reference_url",
        desc="Provide URL reference confirming the cruise duration",
        parent=duration_group,
        critical=True
    )

    duration_leaf = evaluator.add_leaf(
        id="cruise_duration_range_compliance",
        desc="Cruise duration within [7, 16] days",
        parent=duration_group,
        critical=True
    )
    duration_claim = f"The cruise duration is {core.duration_days or 'as shown on the page'}, which is between 7 and 16 days inclusive."
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=_safe_list(core.cruise_urls),
        additional_instruction="If the page states something like '14-day', that counts as 14 days. Ensure the value lies within 7–16 inclusive."
    )

    # 3.b) Geographic region is Asia-Pacific
    region_group = evaluator.add_parallel(
        id="geographic_region_specification_group",
        desc="The cruise must operate in the Asia-Pacific region",
        parent=itin_specs,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(core.cruise_urls)),
        id="region_reference_url",
        desc="Provide URL reference confirming Asia-Pacific routing",
        parent=region_group,
        critical=True
    )

    region_leaf = evaluator.add_leaf(
        id="geographic_region_specification",
        desc="Itinerary operates in the Asia-Pacific region",
        parent=region_group,
        critical=True
    )
    await evaluator.verify(
        claim="This cruise itinerary operates within the Asia-Pacific region (Asia and/or Pacific ports).",
        node=region_leaf,
        sources=_safe_list(core.cruise_urls),
        additional_instruction="Accept itineraries clearly within Asia-Pacific (e.g., Japan, Singapore, Australia, New Zealand, Southeast Asia)."
    )

    # 3.c) Departure & Final port details (non-critical)
    ports_group = evaluator.add_parallel(
        id="departure_and_final_port_details_group",
        desc="Specify the departure port and final port of the cruise itinerary",
        parent=itin_specs,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(core.cruise_urls)),
        id="port_details_reference_url",
        desc="Provide URL reference to the detailed itinerary showing ports",
        parent=ports_group,
        critical=False
    )

    if core.departure_port or core.final_port:
        ports_leaf = evaluator.add_leaf(
            id="departure_and_final_port_details",
            desc="Departure and final ports match the itinerary",
            parent=ports_group,
            critical=False
        )
        ports_claim = f"The itinerary shows departure from '{core.departure_port or 'the specified departure port'}' and ends at '{core.final_port or 'the specified final port'}'."
        await evaluator.verify(
            claim=ports_claim,
            node=ports_leaf,
            sources=_safe_list(core.cruise_urls),
            additional_instruction="Allow reasonable formatting differences in port names (e.g., with/without country)."
        )


async def build_connectivity_section(evaluator: Evaluator, parent, data: CruisePackageExtraction):
    section = evaluator.add_parallel(
        id="pre_cruise_connectivity_and_logistics",
        desc="Verification of flight connectivity from US hubs and airport transit facilities at the departure port",
        parent=parent,
        critical=False
    )

    conn = data.connectivity or Connectivity()
    amen = data.amenity or AirportAmenity()

    # 1) Flight Connection Accessibility
    flight_section = evaluator.add_parallel(
        id="flight_connection_accessibility",
        desc="Verification that the departure port is accessible from US major hubs",
        parent=section,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(conn.flight_urls)),
        id="flight_route_reference_url",
        desc="Provide URL reference or documentation confirming commercial flight availability",
        parent=flight_section,
        critical=True
    )

    flight_leaf = evaluator.add_leaf(
        id="us_hub_connection_availability",
        desc="Departure port reachable from US hub with ≤1 connection",
        parent=flight_section,
        critical=True
    )
    flight_claim = f"There exists a commercial flight itinerary from the US hub '{conn.us_hub or 'a major US hub'}' to '{conn.departure_airport_code or 'the departure airport'}' with no more than one connection."
    await evaluator.verify(
        claim=flight_claim,
        node=flight_leaf,
        sources=_safe_list(conn.flight_urls),
        additional_instruction="Verify using the provided airline or OTA link(s) that routing requires at most one connection (nonstop or 1-stop)."
    )

    # 2) Airport Transit Amenities Verification
    amen_section = evaluator.add_parallel(
        id="airport_transit_amenities_verification",
        desc="Verification of transit passenger facilities at the departure port's airport",
        parent=section,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(amen.amenity_urls)),
        id="amenity_reference_url",
        desc="Provide URL reference documenting the specific transit amenity available",
        parent=amen_section,
        critical=True
    )

    amen_type_leaf = evaluator.add_leaf(
        id="transit_amenity_type_availability",
        desc="Airport has at least one of: airline lounge, rest facility, or transit hotel",
        parent=amen_section,
        critical=True
    )
    amen_type = amen.amenity_type or "one of: airline lounge, rest facility, or transit hotel"
    amen_airport = amen.amenity_airport_code or "the departure airport"
    await evaluator.verify(
        claim=f"The airport {amen_airport} offers {amen_type} for transit passengers.",
        node=amen_type_leaf,
        sources=_safe_list(amen.amenity_urls),
        additional_instruction="Accept any one of the listed categories if clearly available at the airport."
    )

    # Specific amenity identification (non-critical)
    evaluator.add_custom_node(
        result=bool(_safe_list(amen.amenity_urls)),
        id="amenity_details_reference_url",
        desc="Provide URL reference with details about the amenity",
        parent=amen_section,
        critical=False
    )

    if amen.amenity_name:
        amen_name_leaf = evaluator.add_leaf(
            id="specific_amenity_identification",
            desc="Specific amenity identification (name/brand) at the airport",
            parent=amen_section,
            critical=False
        )
        await evaluator.verify(
            claim=f"The specific amenity at {amen_airport} includes '{amen.amenity_name}'.",
            node=amen_name_leaf,
            sources=_safe_list(amen.amenity_urls),
            additional_instruction="Match the amenity name/brand; minor variations in formatting/spelling are acceptable."
        )


async def build_unesco_section(evaluator: Evaluator, parent, data: CruisePackageExtraction):
    section = evaluator.add_parallel(
        id="unesco_world_heritage_site_access",
        desc="Verification that at least one UNESCO World Heritage Site is accessible from a port of call within the itinerary",
        parent=parent,
        critical=False
    )

    u = data.unesco or UNESCOAccess()
    core = data.core or CruiseCore()

    # 1) UNESCO Site Port Accessibility
    access = evaluator.add_parallel(
        id="unesco_site_port_accessibility",
        desc="Verification of UNESCO site accessibility from a cruise port",
        parent=section,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(u.port_to_site_urls)),
        id="unesco_access_reference_url",
        desc="Provide URL reference confirming UNESCO site accessibility from the port",
        parent=access,
        critical=True
    )

    within_range_leaf = evaluator.add_leaf(
        id="unesco_site_within_range",
        desc="At least one port provides access to a UNESCO site within 3 hours",
        parent=access,
        critical=True
    )
    within_range_claim = f"From the port of '{u.port_of_call or 'the specified port'}', the UNESCO site '{u.site_name or 'the specified site'}' is reachable within 3 hours of travel."
    await evaluator.verify(
        claim=within_range_claim,
        node=within_range_leaf,
        sources=_merge_sources(u.port_to_site_urls, u.site_official_urls),
        additional_instruction="Prefer explicit travel-time statements ≤ 3 hours from the port to the site. Shore excursion pages or credible guides are acceptable."
    )

    # 2) UNESCO Site Identification Details (non-critical)
    id_detail = evaluator.add_parallel(
        id="unesco_site_identification_details",
        desc="Specific identification of the accessible UNESCO World Heritage Site",
        parent=section,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(u.site_official_urls)),
        id="unesco_official_reference_url",
        desc="Provide URL reference to UNESCO official listing or site information",
        parent=id_detail,
        critical=False
    )

    if u.site_name and u.site_location:
        name_loc_leaf = evaluator.add_leaf(
            id="unesco_site_name_and_location",
            desc="UNESCO site name and location are correct",
            parent=id_detail,
            critical=False
        )
        await evaluator.verify(
            claim=f"The UNESCO World Heritage Site '{u.site_name}' is located in '{u.site_location}'.",
            node=name_loc_leaf,
            sources=_safe_list(u.site_official_urls),
            additional_instruction="Verify via UNESCO official listing or authoritative site page."
        )

    # Port of call with UNESCO access (non-critical)
    evaluator.add_custom_node(
        result=bool(_merge_sources(u.itinerary_urls, core.cruise_urls)),
        id="port_unesco_connection_reference_url",
        desc="Provide URL reference documenting the connection between port and UNESCO site",
        parent=id_detail,
        critical=False
    )

    if u.port_of_call:
        port_call_leaf = evaluator.add_leaf(
            id="port_of_call_with_unesco_access",
            desc="Itinerary includes the specified port of call providing UNESCO access",
            parent=id_detail,
            critical=False
        )
        await evaluator.verify(
            claim=f"The cruise itinerary includes a call at '{u.port_of_call}'.",
            node=port_call_leaf,
            sources=_merge_sources(u.itinerary_urls, core.cruise_urls),
            additional_instruction="Match the port-of-call name; minor naming variants are acceptable."
        )


async def build_post_cruise_section(evaluator: Evaluator, parent, data: CruisePackageExtraction):
    section = evaluator.add_parallel(
        id="post_cruise_four_seasons_extension",
        desc="Verification of Four Seasons resort accessibility, visa requirements, and property specifications for post-cruise extension",
        parent=parent,
        critical=False
    )

    pc = data.post_cruise or PostCruiseFS()

    # 1) Four Seasons Resort Accessibility
    access = evaluator.add_parallel(
        id="four_seasons_resort_accessibility",
        desc="Verification of Four Seasons resort connectivity from cruise endpoint",
        parent=section,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(pc.flight_connectivity_urls)),
        id="flight_connection_reference_url",
        desc="Provide URL reference or documentation confirming flight connectivity",
        parent=access,
        critical=True
    )

    flight_leaf = evaluator.add_leaf(
        id="resort_flight_connection_availability",
        desc="Commercial flights available from final airport to FS destination with ≤1 connection",
        parent=access,
        critical=True
    )
    flight_conn_claim = f"From final/nearby airport '{pc.final_airport_code or 'the final airport'}', flights to the Four Seasons destination ({pc.fs_location_city or 'destination'}) are available with no more than one connection."
    await evaluator.verify(
        claim=flight_conn_claim,
        node=flight_leaf,
        sources=_safe_list(pc.flight_connectivity_urls),
        additional_instruction="Use airline/OTA links to confirm ≤1 connection routing to the FS city/airport."
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(pc.fs_official_urls)),
        id="four_seasons_official_reference_url",
        desc="Provide URL reference to Four Seasons official website confirming the property",
        parent=access,
        critical=True
    )

    brand_leaf = evaluator.add_leaf(
        id="four_seasons_brand_confirmation",
        desc="Resort is an official Four Seasons property",
        parent=access,
        critical=True
    )
    await evaluator.verify(
        claim=f"The property '{pc.fs_property_name or 'the identified resort'}' is an official Four Seasons hotel/resort.",
        node=brand_leaf,
        sources=_safe_list(pc.fs_official_urls),
        additional_instruction="Check that the page is on fourseasons.com (or an official FS domain) and names the property."
    )

    # 2) Resort Property Specifications
    specs = evaluator.add_parallel(
        id="resort_property_specifications",
        desc="Verification of Four Seasons resort size and accommodation capacity",
        parent=section,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(_merge_sources(pc.room_count_urls, pc.fs_official_urls)),
        id="resort_capacity_reference_url",
        desc="Provide URL reference documenting the resort's room/villa count",
        parent=specs,
        critical=True
    )

    capacity_leaf = evaluator.add_leaf(
        id="resort_accommodation_capacity_requirement",
        desc="Four Seasons property has at least 60 rooms/villas/accommodations",
        parent=specs,
        critical=True
    )
    capacity_claim = f"The property has at least 60 accommodations (rooms/villas); the stated count is {pc.room_count or 'provided on the page'}."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=_merge_sources(pc.room_count_urls, pc.fs_official_urls),
        additional_instruction="Verify the total number of accommodations is ≥ 60; accept 'keys', 'rooms', or 'villas' phrasing."
    )

    # Location identification (non-critical)
    evaluator.add_custom_node(
        result=bool(_safe_list(pc.fs_official_urls)),
        id="resort_location_reference_url",
        desc="Provide URL reference to the resort's location information",
        parent=specs,
        critical=False
    )

    if pc.fs_location_city or pc.fs_location_country:
        loc_leaf = evaluator.add_leaf(
            id="resort_location_identification",
            desc="Four Seasons resort location (city/island and country) is correct",
            parent=specs,
            critical=False
        )
        location_text = ", ".join([t for t in [pc.fs_location_city, pc.fs_location_country] if t])
        await evaluator.verify(
            claim=f"The Four Seasons resort is located in {location_text}.",
            node=loc_leaf,
            sources=_safe_list(pc.fs_official_urls),
            additional_instruction="Match city/island and country; minor formatting differences are acceptable."
        )

    # 3) Visa Entry Requirements Verification
    visa = evaluator.add_parallel(
        id="visa_entry_requirements_verification",
        desc="Verification of visa requirements for US passport holders at the Four Seasons resort's country",
        parent=section,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(_safe_list(pc.visa_urls)),
        id="visa_policy_reference_url",
        desc="Provide URL reference documenting the visa policy for US citizens",
        parent=visa,
        critical=True
    )

    visa_leaf = evaluator.add_leaf(
        id="visa_category_for_us_citizens",
        desc="Country offers visa-free, e-visa, or visa-on-arrival for US citizens",
        parent=visa,
        critical=True
    )
    visa_claim = f"US citizens can enter {pc.fs_location_country or 'the resort country'} with {pc.visa_category or 'visa-free/e-visa/visa-on-arrival'} entry."
    await evaluator.verify(
        claim=visa_claim,
        node=visa_leaf,
        sources=_safe_list(pc.visa_urls),
        additional_instruction="Accept authoritative sources (e.g., government, embassy, IATA Timatic, or current official policy pages)."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 'Holland America Line Asia-Pacific 2026–2027 cruise + post-cruise Four Seasons extension' task.
    """
    # Initialize evaluator (root kept non-critical to allow partial credit across major sections)
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
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_cruise_package(),
        template_class=CruisePackageExtraction,
        extraction_name="cruise_package_extraction"
    )

    # Build verification tree sections
    await build_cruise_product_section(evaluator, root, extracted)
    await build_connectivity_section(evaluator, root, extracted)
    await build_unesco_section(evaluator, root, extracted)
    await build_post_cruise_section(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()