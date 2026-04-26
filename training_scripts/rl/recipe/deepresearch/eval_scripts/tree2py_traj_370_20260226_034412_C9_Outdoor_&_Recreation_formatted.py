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
TASK_ID = "indian_ocean_eco_cruise_2026"
TASK_DESCRIPTION = """
You are planning an eco-conscious cruise vacation in the Indian Ocean region departing in March 2026. The cruise must meet the following requirements:

Cruise Ship Requirements:
1. The ship must be classified as a small ship with a passenger capacity between 600-800 passengers (based on double occupancy)
2. The cruise line or ship must have environmental certification (either Green Marine certification or ISO 14001 standard)
3. The cruise must depart from a port city in Asia and arrive at a port city in Africa
4. The total cruise duration must be between 18-25 days
5. The itinerary must include ports in all three of these destinations: Madagascar, Mauritius, and Seychelles

Madagascar Port Requirements:
6. Shore excursions must include lemur viewing at Lokobe Nature Reserve (also called Lokobe Natural Reserve)
7. The lemur viewing excursion must feature at least three different lemur species that can be observed
8. At least one of the observable species must be black lemurs

Mauritius Port Requirements:
9. Shore excursions must include a guided hiking excursion in Black River Gorges National Park
10. The hiking excursion must feature a specific named trail within the park (the trail name must be provided)
11. Shore excursions must also include snorkeling at Blue Bay Marine Park
12. Blue Bay Marine Park must be identified as a protected marine park with coral reefs

Seychelles Port Requirements:
13. Shore excursions must include either diving or snorkeling activities at St Anne Marine Park (also called Ste Anne Marine Park)

Identify a specific cruise that meets all these requirements. Your answer must include:
- The cruise line name
- The ship name
- The voyage/itinerary name
- The departure date
- The departure port city
- The arrival port city
- The cruise duration in days
- Reference URLs confirming the ship's capacity
- Reference URLs confirming the environmental certification
- Reference URLs confirming the itinerary details
- For each of the three ports (Madagascar, Mauritius, Seychelles), provide the specific excursion names or descriptions that meet the stated requirements, along with reference URLs
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CruiseSelection(BaseModel):
    cruise_line: Optional[str] = None
    ship_name: Optional[str] = None
    voyage_name: Optional[str] = None
    departure_date: Optional[str] = None
    departure_port_city: Optional[str] = None
    arrival_port_city: Optional[str] = None
    duration_days: Optional[str] = None
    capacity_reference_urls: List[str] = Field(default_factory=list)
    certification_reference_urls: List[str] = Field(default_factory=list)
    itinerary_reference_urls: List[str] = Field(default_factory=list)


class MadagascarInfo(BaseModel):
    excursion_name: Optional[str] = None
    location_name: Optional[str] = None  # Expect "Lokobe Nature Reserve" or "Lokobe Natural Reserve"
    species_observable: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class MauritiusHike(BaseModel):
    excursion_name: Optional[str] = None
    park_name: Optional[str] = None  # Expect "Black River Gorges National Park"
    trail_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class MauritiusSnorkel(BaseModel):
    excursion_name: Optional[str] = None
    park_name: Optional[str] = None  # Expect "Blue Bay Marine Park"
    reference_urls: List[str] = Field(default_factory=list)


class MauritiusInfo(BaseModel):
    hiking: Optional[MauritiusHike] = None
    snorkeling: Optional[MauritiusSnorkel] = None


class SeychellesInfo(BaseModel):
    activity_type: Optional[str] = None  # "snorkeling" or "diving"
    park_name: Optional[str] = None  # Expect "St Anne Marine Park" or "Ste Anne Marine Park"
    reference_urls: List[str] = Field(default_factory=list)


class CruisePlanExtraction(BaseModel):
    cruise: Optional[CruiseSelection] = None
    madagascar: Optional[MadagascarInfo] = None
    mauritius: Optional[MauritiusInfo] = None
    seychelles: Optional[SeychellesInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_cruise_plan() -> str:
    return """
Extract the structured information from the answer needed to evaluate the eco-conscious Indian Ocean cruise plan.

Return a JSON object with the following structure and fields:

{
  "cruise": {
    "cruise_line": string | null,
    "ship_name": string | null,
    "voyage_name": string | null,
    "departure_date": string | null,
    "departure_port_city": string | null,
    "arrival_port_city": string | null,
    "duration_days": string | null,
    "capacity_reference_urls": string[]  // URLs explicitly cited for ship capacity
    "certification_reference_urls": string[]  // URLs explicitly cited for environmental certification (Green Marine or ISO 14001)
    "itinerary_reference_urls": string[]  // URLs explicitly cited for itinerary details, route, departure timing, duration, ports
  },
  "madagascar": {
    "excursion_name": string | null,  // e.g., "Lokobe Nature Reserve lemur viewing"
    "location_name": string | null,   // e.g., "Lokobe Nature Reserve" or "Lokobe Natural Reserve"
    "species_observable": string[],   // list the lemur species names mentioned in the answer, if any
    "reference_urls": string[]        // URLs for this excursion/location (tour operator, park, or cruise line excursion pages)
  },
  "mauritius": {
    "hiking": {
      "excursion_name": string | null,
      "park_name": string | null,     // should be "Black River Gorges National Park" if present
      "trail_name": string | null,    // specific trail name within the park
      "reference_urls": string[]      // URLs for the hiking excursion/trail
    },
    "snorkeling": {
      "excursion_name": string | null,
      "park_name": string | null,     // should be "Blue Bay Marine Park" if present
      "reference_urls": string[]      // URLs for the snorkeling excursion/park
    }
  },
  "seychelles": {
    "activity_type": string | null,   // "snorkeling" or "diving"
    "park_name": string | null,       // "St Anne Marine Park" or "Ste Anne Marine Park"
    "reference_urls": string[]        // URLs for the activity/park
  }
}

Rules:
- Extract only what is explicitly present in the answer text.
- Collect all URLs mentioned in the answer that directly support the specified items; if a URL is missing, return an empty array for that field.
- Do not invent any URLs.
- Keep strings as-is; do not normalize names beyond exact extraction.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(value: Optional[str], default_text: str) -> str:
    return value.strip() if value else default_text


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_cruise_selection(evaluator: Evaluator, parent, cruise: CruiseSelection) -> None:
    # Cruise_Selection (critical, parallel)
    cruise_sel = evaluator.add_parallel(
        id="Cruise_Selection",
        desc="The selected cruise must meet all environmental, capacity, and itinerary requirements",
        parent=parent,
        critical=True
    )

    # Ship_Specifications (critical, parallel)
    ship_specs = evaluator.add_parallel(
        id="Ship_Specifications",
        desc="The cruise ship must meet small ship criteria",
        parent=cruise_sel,
        critical=True
    )

    # Ship_Capacity_Reference - existence of at least one capacity URL
    cap_ref_node = evaluator.add_custom_node(
        result=bool(cruise.capacity_reference_urls),
        id="Ship_Capacity_Reference",
        desc="Provide a reference URL confirming the ship's passenger capacity",
        parent=ship_specs,
        critical=True
    )

    # Passenger_Capacity - verify capacity range 600-800 with given URLs
    cap_leaf = evaluator.add_leaf(
        id="Passenger_Capacity",
        desc="The ship must have a passenger capacity between 600-800 passengers at double occupancy",
        parent=ship_specs,
        critical=True
    )
    ship_name_txt = _safe(cruise.ship_name, "the ship")
    capacity_claim = (
        f"The ship {ship_name_txt} has a passenger capacity between 600 and 800 passengers "
        f"(based on double occupancy)."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=cap_leaf,
        sources=cruise.capacity_reference_urls,
        additional_instruction="Verify the capacity number reported on the ship's official or reliable source page and confirm it lies within [600, 800] guests."
    )

    # Environmental_Certification (critical, parallel)
    env_cert = evaluator.add_parallel(
        id="Environmental_Certification",
        desc="The cruise line or ship must have recognized environmental certification",
        parent=cruise_sel,
        critical=True
    )

    cert_ref_node = evaluator.add_custom_node(
        result=bool(cruise.certification_reference_urls),
        id="Certification_Reference",
        desc="Provide a reference URL confirming the environmental certification",
        parent=env_cert,
        critical=True
    )

    cert_leaf = evaluator.add_leaf(
        id="Certification_Type",
        desc="The certification must be either Green Marine or ISO 14001 standard",
        parent=env_cert,
        critical=True
    )
    line_txt = _safe(cruise.cruise_line, "the cruise line")
    cert_claim = (
        f"{line_txt} or the ship {ship_name_txt} has an environmental certification that is either "
        f"Green Marine or ISO 14001."
    )
    await evaluator.verify(
        claim=cert_claim,
        node=cert_leaf,
        sources=cruise.certification_reference_urls,
        additional_instruction="Look for explicit mentions of 'Green Marine' certification or 'ISO 14001' environmental management certification on the provided pages."
    )

    # Itinerary_Requirements (critical, parallel)
    itin_reqs = evaluator.add_parallel(
        id="Itinerary_Requirements",
        desc="The cruise itinerary must meet specific route and timing requirements",
        parent=cruise_sel,
        critical=True
    )

    # Port_Inclusions (critical, parallel)
    port_inclusions = evaluator.add_parallel(
        id="Port_Inclusions",
        desc="The itinerary must include all three required port destinations",
        parent=itin_reqs,
        critical=True
    )

    port_ref_node = evaluator.add_custom_node(
        result=bool(cruise.itinerary_reference_urls),
        id="Port_Inclusions_Reference",
        desc="Provide a reference URL confirming all three ports are included in the itinerary",
        parent=port_inclusions,
        critical=True
    )

    # Madagascar_Port
    mg_port_leaf = evaluator.add_leaf(
        id="Madagascar_Port",
        desc="The itinerary must include a port in Madagascar",
        parent=port_inclusions,
        critical=True
    )
    await evaluator.verify(
        claim="The cruise itinerary includes at least one port call in Madagascar.",
        node=mg_port_leaf,
        sources=cruise.itinerary_reference_urls,
        additional_instruction="Confirm that the itinerary explicitly lists a stop in Madagascar (e.g., Nosy Be, Antsiranana, or another Madagascar port)."
    )

    # Mauritius_Port
    mu_port_leaf = evaluator.add_leaf(
        id="Mauritius_Port",
        desc="The itinerary must include a port in Mauritius",
        parent=port_inclusions,
        critical=True
    )
    await evaluator.verify(
        claim="The cruise itinerary includes at least one port call in Mauritius.",
        node=mu_port_leaf,
        sources=cruise.itinerary_reference_urls,
        additional_instruction="Confirm that the itinerary explicitly lists a stop in Mauritius (e.g., Port Louis or another Mauritius port)."
    )

    # Seychelles_Port
    sc_port_leaf = evaluator.add_leaf(
        id="Seychelles_Port",
        desc="The itinerary must include a port in Seychelles",
        parent=port_inclusions,
        critical=True
    )
    await evaluator.verify(
        claim="The cruise itinerary includes at least one port call in Seychelles.",
        node=sc_port_leaf,
        sources=cruise.itinerary_reference_urls,
        additional_instruction="Confirm that the itinerary explicitly lists a stop in Seychelles (e.g., Victoria/Mahé, Praslin, La Digue, etc.)."
    )

    # Departure_Timing (critical, parallel)
    dep_timing = evaluator.add_parallel(
        id="Departure_Timing",
        desc="The cruise must depart in March 2026",
        parent=itin_reqs,
        critical=True
    )

    dep_ref_node = evaluator.add_custom_node(
        result=bool(cruise.itinerary_reference_urls),
        id="Departure_Timing_Reference",
        desc="Provide a reference URL confirming the departure date",
        parent=dep_timing,
        critical=True
    )

    dep_month_leaf = evaluator.add_leaf(
        id="Departure_Month",
        desc="The departure month must be March 2026",
        parent=dep_timing,
        critical=True
    )
    await evaluator.verify(
        claim="The cruise departs in March 2026.",
        node=dep_month_leaf,
        sources=cruise.itinerary_reference_urls,
        additional_instruction="Check the itinerary or schedule page for the specific departure date and confirm the month is March and the year is 2026. Allow minor date formatting variations."
    )

    # Cruise_Duration (critical, parallel)
    duration_node = evaluator.add_parallel(
        id="Cruise_Duration",
        desc="The total cruise duration must be between 18-25 days",
        parent=itin_reqs,
        critical=True
    )

    duration_ref_node = evaluator.add_custom_node(
        result=bool(cruise.itinerary_reference_urls),
        id="Duration_Reference",
        desc="Provide a reference URL confirming the cruise duration",
        parent=duration_node,
        critical=True
    )

    duration_range_leaf = evaluator.add_leaf(
        id="Duration_Range",
        desc="The cruise must be at least 18 days and no more than 25 days in length",
        parent=duration_node,
        critical=True
    )
    await evaluator.verify(
        claim="The total cruise duration is between 18 and 25 days inclusive.",
        node=duration_range_leaf,
        sources=cruise.itinerary_reference_urls,
        additional_instruction="Verify the total number of days shown on the itinerary page is >= 18 and <= 25."
    )

    # Route_Specifications (critical, parallel)
    route_specs = evaluator.add_parallel(
        id="Route_Specifications",
        desc="The cruise route must connect Asia to Africa",
        parent=cruise_sel,
        critical=True
    )

    route_ref_node = evaluator.add_custom_node(
        result=bool(cruise.itinerary_reference_urls),
        id="Route_Reference",
        desc="Provide a reference URL confirming the departure and arrival cities",
        parent=route_specs,
        critical=True
    )

    dep_region_leaf = evaluator.add_leaf(
        id="Departure_Region",
        desc="The cruise must depart from a port city in Asia",
        parent=route_specs,
        critical=True
    )
    dep_city_txt = _safe(cruise.departure_port_city, "the departure city")
    await evaluator.verify(
        claim=f"The cruise departs from {dep_city_txt}, which is in Asia.",
        node=dep_region_leaf,
        sources=cruise.itinerary_reference_urls,
        additional_instruction="Confirm the departure city and its country on the itinerary page. It is acceptable to infer the continent (Asia) from the country shown."
    )

    arr_region_leaf = evaluator.add_leaf(
        id="Arrival_Region",
        desc="The cruise must end at a port city in Africa",
        parent=route_specs,
        critical=True
    )
    arr_city_txt = _safe(cruise.arrival_port_city, "the arrival city")
    await evaluator.verify(
        claim=f"The cruise arrives at {arr_city_txt}, which is in Africa.",
        node=arr_region_leaf,
        sources=cruise.itinerary_reference_urls,
        additional_instruction="Confirm the arrival city and its country on the itinerary page. It is acceptable to infer the continent (Africa) from the country shown."
    )


async def build_madagascar_port(evaluator: Evaluator, parent, mg: MadagascarInfo) -> None:
    # Madagascar_Port_Activities (critical, parallel)
    mg_root = evaluator.add_parallel(
        id="Madagascar_Port_Activities",
        desc="Shore excursions at the Madagascar port must include lemur viewing with specific requirements",
        parent=parent,
        critical=True
    )

    # Lemur_Viewing_Excursion (critical, parallel)
    lemur_view = evaluator.add_parallel(
        id="Lemur_Viewing_Excursion",
        desc="A shore excursion featuring lemur viewing at the appropriate nature reserve",
        parent=mg_root,
        critical=True
    )

    # Viewing_Location (critical, parallel)
    view_loc = evaluator.add_parallel(
        id="Viewing_Location",
        desc="Lemur viewing must take place at Lokobe Nature Reserve or Lokobe Natural Reserve",
        parent=lemur_view,
        critical=True
    )

    # Location_Reference - existence
    loc_ref_node = evaluator.add_custom_node(
        result=bool(mg.reference_urls),
        id="Location_Reference",
        desc="Provide a reference URL confirming lemur viewing at Lokobe",
        parent=view_loc,
        critical=True
    )

    # Reserve_Name - verify location is Lokobe
    reserve_leaf = evaluator.add_leaf(
        id="Reserve_Name",
        desc="The reserve must be identified as Lokobe Nature Reserve or Lokobe Natural Reserve",
        parent=view_loc,
        critical=True
    )
    loc_name_txt = _safe(mg.location_name, "Lokobe Nature Reserve")
    await evaluator.verify(
        claim=f"The lemur viewing shore excursion takes place at {loc_name_txt}, also known as Lokobe Nature Reserve (Lokobe Natural Reserve), in Madagascar.",
        node=reserve_leaf,
        sources=mg.reference_urls,
        additional_instruction="Confirm the excursion page explicitly mentions Lokobe Nature Reserve (or Lokobe Natural Reserve) as the location."
    )

    # Species_Diversity (critical, parallel)
    species_div = evaluator.add_parallel(
        id="Species_Diversity",
        desc="The excursion must feature at least three different lemur species",
        parent=lemur_view,
        critical=True
    )

    # Species_Count_Reference - existence
    sp_ref_node = evaluator.add_custom_node(
        result=bool(mg.reference_urls),
        id="Species_Count_Reference",
        desc="Provide a reference URL confirming the number of lemur species viewable",
        parent=species_div,
        critical=True
    )

    # Minimum_Species_Count - verify >=3 species
    min_species_leaf = evaluator.add_leaf(
        id="Minimum_Species_Count",
        desc="At least three different lemur species must be observable during the excursion",
        parent=species_div,
        critical=True
    )
    species_list_txt = ", ".join(mg.species_observable) if mg.species_observable else "multiple species"
    await evaluator.verify(
        claim=f"The Lokobe excursion features lemur viewing of at least three different species (for example: {species_list_txt}).",
        node=min_species_leaf,
        sources=mg.reference_urls,
        additional_instruction="Look for a list or mentions of multiple lemur species on the page; consider reasonable name variants; confirm that at least three distinct species are observable."
    )

    # Black_Lemur_Requirement (critical, parallel)
    black_req = evaluator.add_parallel(
        id="Black_Lemur_Requirement",
        desc="Black lemurs must be one of the observable species (explicit requirement #8)",
        parent=lemur_view,
        critical=True
    )

    # Black_Lemur_Reference - existence
    black_ref_node = evaluator.add_custom_node(
        result=bool(mg.reference_urls),
        id="Black_Lemur_Reference",
        desc="Provide a reference URL confirming black lemurs are viewable at Lokobe",
        parent=black_req,
        critical=True
    )

    # Black_Lemurs_Present - verify black lemurs observable
    black_present_leaf = evaluator.add_leaf(
        id="Black_Lemurs_Present",
        desc="Black lemurs must be identified as observable at the location",
        parent=black_req,
        critical=True
    )
    await evaluator.verify(
        claim="Black lemurs are among the observable species at Lokobe Nature Reserve.",
        node=black_present_leaf,
        sources=mg.reference_urls,
        additional_instruction="Check that the page explicitly mentions 'black lemur' (or a recognized equivalent common name) as an observable species at Lokobe."
    )


async def build_mauritius_port(evaluator: Evaluator, parent, mu: MauritiusInfo) -> None:
    # Mauritius_Port_Activities (critical, parallel)
    mu_root = evaluator.add_parallel(
        id="Mauritius_Port_Activities",
        desc="Shore excursions at the Mauritius port must include both hiking and snorkeling activities",
        parent=parent,
        critical=True
    )

    # Hiking_Excursion (critical, parallel)
    hike = mu.hiking or MauritiusHike()
    hiking_node = evaluator.add_parallel(
        id="Hiking_Excursion",
        desc="A hiking excursion in the appropriate national park with a named trail",
        parent=mu_root,
        critical=True
    )

    # National_Park (critical, parallel)
    park_node = evaluator.add_parallel(
        id="National_Park",
        desc="Hiking must take place in Black River Gorges National Park",
        parent=hiking_node,
        critical=True
    )

    park_ref_exists = evaluator.add_custom_node(
        result=bool(hike.reference_urls),
        id="Park_Reference",
        desc="Provide a reference URL confirming hiking in Black River Gorges National Park",
        parent=park_node,
        critical=True
    )

    park_name_leaf = evaluator.add_leaf(
        id="Park_Name",
        desc="The national park must be identified as Black River Gorges National Park",
        parent=park_node,
        critical=True
    )
    park_name_txt = _safe(hike.park_name, "Black River Gorges National Park")
    await evaluator.verify(
        claim=f"The guided hiking excursion takes place in {park_name_txt} in Mauritius.",
        node=park_name_leaf,
        sources=hike.reference_urls,
        additional_instruction="Confirm the page explicitly names 'Black River Gorges National Park' as the hiking location."
    )

    # Trail_Identification (critical, parallel)
    trail_node = evaluator.add_parallel(
        id="Trail_Identification",
        desc="The hiking excursion must feature a specific named trail within the park",
        parent=hiking_node,
        critical=True
    )

    trail_ref_exists = evaluator.add_custom_node(
        result=bool(hike.reference_urls),
        id="Trail_Reference",
        desc="Provide a reference URL confirming the named trail in Black River Gorges",
        parent=trail_node,
        critical=True
    )

    trail_name_leaf = evaluator.add_leaf(
        id="Trail_Name",
        desc="A specific trail name must be provided (e.g., Machabee Trail, Paille en Queue Trail, Mare aux Joncs Trail, Black River Peak Trail)",
        parent=trail_node,
        critical=True
    )
    trail_name_txt = _safe(hike.trail_name, "a named trail")
    await evaluator.verify(
        claim=f"The hiking excursion includes the named trail '{trail_name_txt}' within Black River Gorges National Park.",
        node=trail_name_leaf,
        sources=hike.reference_urls,
        additional_instruction="Verify that the trail name is explicitly stated on the provided page and that it is within Black River Gorges National Park."
    )

    # Snorkeling_Excursion (critical, parallel)
    snork = mu.snorkeling or MauritiusSnorkel()
    snorkel_node = evaluator.add_parallel(
        id="Snorkeling_Excursion",
        desc="A snorkeling excursion at the designated marine park",
        parent=mu_root,
        critical=True
    )

    # Marine_Park (critical, parallel)
    marine_park_node = evaluator.add_parallel(
        id="Marine_Park",
        desc="Snorkeling must take place at Blue Bay Marine Park",
        parent=snorkel_node,
        critical=True
    )

    marine_ref_exists = evaluator.add_custom_node(
        result=bool(snork.reference_urls),
        id="Marine_Park_Reference",
        desc="Provide a reference URL confirming snorkeling at Blue Bay Marine Park",
        parent=marine_park_node,
        critical=True
    )

    marine_park_leaf = evaluator.add_leaf(
        id="Park_Name_BlueBay",
        desc="The marine park must be identified as Blue Bay Marine Park",
        parent=marine_park_node,
        critical=True
    )
    snork_park_name_txt = _safe(snork.park_name, "Blue Bay Marine Park")
    await evaluator.verify(
        claim=f"The snorkeling excursion takes place at {snork_park_name_txt} in Mauritius.",
        node=marine_park_leaf,
        sources=snork.reference_urls,
        additional_instruction="Confirm the page explicitly mentions 'Blue Bay Marine Park' as the snorkeling location."
    )

    # Protected_Status (critical, parallel)
    protected_node = evaluator.add_parallel(
        id="Protected_Status",
        desc="Blue Bay Marine Park must be described as a protected marine park",
        parent=snorkel_node,
        critical=True
    )

    protection_ref_exists = evaluator.add_custom_node(
        result=bool(snork.reference_urls),
        id="Protection_Reference",
        desc="Provide a reference URL confirming the protected status",
        parent=protected_node,
        critical=True
    )

    protection_leaf = evaluator.add_leaf(
        id="Protection_Designation",
        desc="The area must be identified as a protected or designated marine park",
        parent=protected_node,
        critical=True
    )
    await evaluator.verify(
        claim="Blue Bay Marine Park is a protected marine park.",
        node=protection_leaf,
        sources=snork.reference_urls,
        additional_instruction="Look for designation language indicating legal protection or official marine park status."
    )

    # Coral_Reef_Requirement (critical, parallel)
    coral_node = evaluator.add_parallel(
        id="Coral_Reef_Requirement",
        desc="Blue Bay Marine Park must have coral reefs (explicit requirement #12)",
        parent=snorkel_node,
        critical=True
    )

    coral_ref_exists = evaluator.add_custom_node(
        result=bool(snork.reference_urls),
        id="Coral_Reef_Reference",
        desc="Provide a reference URL confirming coral reefs at Blue Bay Marine Park",
        parent=coral_node,
        critical=True
    )

    coral_leaf = evaluator.add_leaf(
        id="Coral_Reefs_Present",
        desc="The presence of coral reefs must be confirmed at Blue Bay Marine Park",
        parent=coral_node,
        critical=True
    )
    await evaluator.verify(
        claim="Blue Bay Marine Park has coral reefs.",
        node=coral_leaf,
        sources=snork.reference_urls,
        additional_instruction="Confirm explicit mentions of coral reefs at Blue Bay Marine Park on the provided page(s)."
    )


async def build_seychelles_port(evaluator: Evaluator, parent, sc: SeychellesInfo) -> None:
    # Seychelles_Port_Activities (critical, parallel)
    sc_root = evaluator.add_parallel(
        id="Seychelles_Port_Activities",
        desc="Shore excursions at the Seychelles port must include marine recreation activities",
        parent=parent,
        critical=True
    )

    # Marine_Recreation_Activity (critical, parallel)
    marine_act = evaluator.add_parallel(
        id="Marine_Recreation_Activity",
        desc="A diving or snorkeling activity at the designated marine park",
        parent=sc_root,
        critical=True
    )

    # Activity_Type (critical, parallel)
    act_type = evaluator.add_parallel(
        id="Activity_Type",
        desc="The activity must be either diving or snorkeling",
        parent=marine_act,
        critical=True
    )

    act_ref_exists = evaluator.add_custom_node(
        result=bool(sc.reference_urls),
        id="Activity_Reference",
        desc="Provide a reference URL confirming the diving or snorkeling activity",
        parent=act_type,
        critical=True
    )

    act_cat_leaf = evaluator.add_leaf(
        id="Activity_Category",
        desc="The activity must be identified as diving, snorkeling, or scuba diving",
        parent=act_type,
        critical=True
    )
    activity_txt = _safe(sc.activity_type, "snorkeling or diving")
    await evaluator.verify(
        claim=f"The Seychelles shore excursion involves {activity_txt} (diving or snorkeling).",
        node=act_cat_leaf,
        sources=sc.reference_urls,
        additional_instruction="Confirm that the activity is clearly categorized as snorkeling, diving, or scuba diving on the provided page(s)."
    )

    # Activity_Location (critical, parallel)
    act_loc = evaluator.add_parallel(
        id="Activity_Location",
        desc="The marine activity must take place at St Anne Marine Park",
        parent=marine_act,
        critical=True
    )

    loc_ref_exists = evaluator.add_custom_node(
        result=bool(sc.reference_urls),
        id="Activity_Location_Reference",
        desc="Provide a reference URL confirming activities at St Anne Marine Park",
        parent=act_loc,
        critical=True
    )

    loc_name_leaf = evaluator.add_leaf(
        id="Marine_Park_Name",
        desc="The location must be identified as St Anne Marine Park or Ste Anne Marine Park",
        parent=act_loc,
        critical=True
    )
    sc_park_txt = _safe(sc.park_name, "St Anne Marine Park")
    await evaluator.verify(
        claim=f"The {activity_txt} activity takes place at {sc_park_txt} (also spelled Ste Anne Marine Park) in Seychelles.",
        node=loc_name_leaf,
        sources=sc.reference_urls,
        additional_instruction="Confirm that the page explicitly mentions 'St Anne Marine Park' or 'Ste Anne Marine Park' as the activity location."
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
    Evaluate an answer for the eco-conscious Indian Ocean cruise task.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured plan data
    extracted: CruisePlanExtraction = await evaluator.extract(
        prompt=prompt_extract_cruise_plan(),
        template_class=CruisePlanExtraction,
        extraction_name="cruise_plan_extraction"
    )

    # Top-level critical umbrella to mirror rubric
    eco_root = evaluator.add_parallel(
        id="Indian_Ocean_Eco_Cruise_Vacation",
        desc="A complete eco-conscious cruise vacation plan in the Indian Ocean with specific outdoor recreation activities at multiple ports",
        parent=root,
        critical=True
    )

    # Build Cruise Selection subtree
    cruise = extracted.cruise or CruiseSelection()
    await build_cruise_selection(evaluator, eco_root, cruise)

    # Build Madagascar subtree
    madagascar = extracted.madagascar or MadagascarInfo()
    await build_madagascar_port(evaluator, eco_root, madagascar)

    # Build Mauritius subtree
    mauritius = extracted.mauritius or MauritiusInfo()
    await build_mauritius_port(evaluator, eco_root, mauritius)

    # Build Seychelles subtree
    seychelles = extracted.seychelles or SeychellesInfo()
    await build_seychelles_port(evaluator, eco_root, seychelles)

    # Add a bit of custom info for debugging
    evaluator.add_custom_info(
        {
            "cruise_line": cruise.cruise_line,
            "ship_name": cruise.ship_name,
            "voyage_name": cruise.voyage_name,
            "departure_date": cruise.departure_date,
            "departure_port_city": cruise.departure_port_city,
            "arrival_port_city": cruise.arrival_port_city,
            "duration_days": cruise.duration_days
        },
        info_type="overview",
        info_name="extracted_overview"
    )

    # Return the evaluation summary
    return evaluator.get_summary()