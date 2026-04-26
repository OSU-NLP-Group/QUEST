import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "southern_caribbean_cruise_hotels"
TASK_DESCRIPTION = (
    "You are planning a Southern Caribbean cruise vacation departing from Fort Lauderdale, Florida. "
    "The cruise will visit four ports: Grenada, Curacao, Aruba, and Barbados. You need to arrange hotel "
    "accommodations for one night at each of the four Caribbean islands visited during the cruise, plus one night "
    "in Fort Lauderdale before embarking on the cruise.\n\n"
    "For each of the five required hotels, identify a specific property that meets ALL of the following requirements:\n\n"
    "Grenada Hotel Requirements:\n"
    "- Must be affiliated with Marriott, Hilton, Hyatt, or Radisson hotel chains\n"
    "- Must have a minimum 4-star rating\n"
    "- Must be located on or provide direct beach access\n"
    "- Must offer ocean view room accommodations\n"
    "- Must have an outdoor swimming pool\n"
    "- Must have at least one on-site restaurant\n\n"
    "Curacao Hotel Requirements:\n"
    "- Must be affiliated with Marriott, Hilton, Hyatt, or Radisson hotel chains\n"
    "- Must be located in or near Willemstad\n"
    "- Must provide beach access (private beach or beach location)\n"
    "- Must offer rooms with balconies or terraces\n"
    "- Must have a fitness center\n"
    "- Must have swimming pool facilities\n\n"
    "Aruba Hotel Requirements:\n"
    "- Must be affiliated with Marriott, Hilton, Hyatt, or Radisson hotel chains\n"
    "- Must have a minimum 4-star rating\n"
    "- Must be located in either Palm Beach or Eagle Beach area\n"
    "- Must offer ocean view accommodations\n"
    "- Must have spa facilities or spa services\n"
    "- Must have multiple on-site dining options (at least 2 restaurants or bars)\n\n"
    "Barbados Hotel Requirements:\n"
    "- Must be affiliated with Marriott, Hilton, Hyatt, or Radisson hotel chains\n"
    "- Must be located in or near Bridgetown\n"
    "- Must be a beachfront property or provide beach access\n"
    "- Must offer ocean view room options\n"
    "- Must have outdoor pool facilities\n"
    "- Must have at least one on-site restaurant\n\n"
    "Fort Lauderdale Hotel Requirements:\n"
    "- Must be affiliated with a major hotel chain (Marriott, Hilton, Hyatt, IHG, or similar recognized international chain)\n"
    "- Must be located within 3 miles of Port Everglades cruise terminal\n"
    "- Must be located within 15 minutes drive of Fort Lauderdale-Hollywood International Airport (FLL)\n"
    "- Must offer shuttle service to Port Everglades cruise terminal (complimentary or paid)\n"
    "- Must have a swimming pool\n\n"
    "For each hotel, provide: (1) the hotel name, (2) a brief description confirming it meets the requirements, "
    "(3) reference URLs to official hotel websites, booking pages, or property information pages that verify the hotel meets the specified criteria."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelEntry(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    grenada: Optional[HotelEntry] = None
    curacao: Optional[HotelEntry] = None
    aruba: Optional[HotelEntry] = None
    barbados: Optional[HotelEntry] = None
    fort_lauderdale: Optional[HotelEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract exactly one specific hotel for each of the following locations from the provided answer text:
    - grenada
    - curacao
    - aruba
    - barbados
    - fort_lauderdale

    For each location, extract:
    1) name: the hotel's name as stated in the answer.
    2) description: a brief summary sentence or two provided in the answer describing why it meets the requirements.
    3) sources: an array of URLs explicitly cited in the answer that support the hotel's details (official site, brand site, booking site, or property info page). 
       Only include URLs that appear in the answer. If no URLs are provided for a location, return an empty array for that location.

    Return a JSON object with the following top-level fields:
    {
      "grenada": { "name": ..., "description": ..., "sources": [...] } | null,
      "curacao": { ... } | null,
      "aruba": { ... } | null,
      "barbados": { ... } | null,
      "fort_lauderdale": { ... } | null
    }

    Rules:
    - Do not invent URLs. Only include URLs explicitly present in the answer.
    - If multiple hotels are mentioned for a location, select the first clearly recommended hotel and its corresponding URLs.
    - If a location has no hotel mentioned, set that field to null.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(h: Optional[HotelEntry]) -> str:
    return h.name if (h and h.name) else "the selected hotel"

def _safe_sources(h: Optional[HotelEntry]) -> List[str]:
    return h.sources if (h and h.sources) else []

async def _add_and_verify(
    evaluator: Evaluator,
    node_id: str,
    desc: str,
    parent,
    claim: str,
    sources: List[str],
    critical: bool = True,
    additional_instruction: str = ""
):
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction
    )
    return leaf


# --------------------------------------------------------------------------- #
# Verification builders for each destination                                  #
# --------------------------------------------------------------------------- #
async def verify_grenada(evaluator: Evaluator, parent, hotel: Optional[HotelEntry]):
    hotel_node = evaluator.add_parallel(
        id="Grenada_Hotel",
        desc="Identify a suitable hotel in Grenada meeting all specified criteria",
        parent=parent,
        critical=False
    )
    sources = _safe_sources(hotel)
    name = _safe_name(hotel)

    # Group: Basic Requirements (critical)
    basic_node = evaluator.add_parallel(
        id="Grenada_Basic_Requirements",
        desc="Chain affiliation and rating requirements for Grenada hotel",
        parent=hotel_node,
        critical=True
    )
    # URL presence for basic (critical, existence check)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Grenada_Basic_URL",
        desc="Provide URL confirming chain affiliation and rating",
        parent=basic_node,
        critical=True
    )
    # Chain affiliation
    await _add_and_verify(
        evaluator,
        "Grenada_Chain",
        "The hotel must be affiliated with Marriott, Hilton, Hyatt, or Radisson chains",
        basic_node,
        claim=f"The hotel '{name}' is affiliated with Marriott, Hilton, Hyatt, or Radisson (including their sub-brands such as Autograph Collection, Renaissance, Courtyard, JW Marriott; DoubleTree, Curio, Hilton Garden Inn; Hyatt Regency, Grand Hyatt, Hyatt Centric; Radisson, Radisson Blu).",
        sources=sources,
        additional_instruction="Confirm the brand family explicitly on the provided page(s)."
    )
    # Rating >= 4-star
    await _add_and_verify(
        evaluator,
        "Grenada_Rating",
        "The hotel must have a minimum 4-star rating",
        basic_node,
        claim=f"The hotel '{name}' is rated at least 4 stars.",
        sources=sources,
        additional_instruction="Accept ratings from official sites or credible booking platforms (e.g., Booking.com, Expedia, Hotels.com, Google Hotels). '4-star', '4.5-star', or '5-star' all satisfy the minimum."
    )

    # Group: Location & Beach Access (critical)
    loc_node = evaluator.add_parallel(
        id="Grenada_Location_Requirements",
        desc="Location and beach access requirements for Grenada hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Grenada_Location_URL",
        desc="Provide URL confirming beach location",
        parent=loc_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Grenada_Beach_Access",
        "The hotel must be located on or provide direct access to a beach",
        loc_node,
        claim=f"The hotel '{name}' is beachfront or provides direct beach access.",
        sources=sources,
        additional_instruction="Look for terms like 'beachfront', 'on the beach', 'private beach', or equivalent. 'Across the street' generally does not count as 'direct' access unless the property has a private designated access pathway."
    )

    # Group: Accommodation Features (critical)
    acc_node = evaluator.add_parallel(
        id="Grenada_Accommodation_Features",
        desc="Room type requirements for Grenada hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Grenada_Room_URL",
        desc="Provide URL confirming ocean view room availability",
        parent=acc_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Grenada_Ocean_View",
        "The hotel must offer ocean view room accommodations",
        acc_node,
        claim=f"The hotel '{name}' offers ocean view (or sea view) room categories.",
        sources=sources,
        additional_instruction="Accept 'ocean view', 'sea view', or 'partial ocean view' room descriptions."
    )

    # Group: Amenities (critical)
    amen_node = evaluator.add_parallel(
        id="Grenada_Amenity_Requirements",
        desc="Facility and amenity requirements for Grenada hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Grenada_Amenity_URL",
        desc="Provide URL confirming pool and dining facilities",
        parent=amen_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Grenada_Pool",
        "The hotel must have an outdoor swimming pool",
        amen_node,
        claim=f"The hotel '{name}' has an outdoor swimming pool.",
        sources=sources,
        additional_instruction="Look for explicit mention of 'outdoor pool' or photos/text indicating an outdoor pool."
    )
    await _add_and_verify(
        evaluator,
        "Grenada_Dining",
        "The hotel must have at least one on-site restaurant",
        amen_node,
        claim=f"The hotel '{name}' has at least one on-site restaurant.",
        sources=sources,
        additional_instruction="On-site bars can be mentioned, but there must be at least one on-property restaurant."
    )


async def verify_curacao(evaluator: Evaluator, parent, hotel: Optional[HotelEntry]):
    hotel_node = evaluator.add_parallel(
        id="Curacao_Hotel",
        desc="Identify a suitable hotel in Curacao meeting all specified criteria",
        parent=parent,
        critical=False
    )
    sources = _safe_sources(hotel)
    name = _safe_name(hotel)

    # Basic (critical)
    basic_node = evaluator.add_parallel(
        id="Curacao_Basic_Requirements",
        desc="Chain affiliation requirements for Curacao hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Curacao_Basic_URL",
        desc="Provide URL confirming chain affiliation",
        parent=basic_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Curacao_Chain",
        "The hotel must be affiliated with Marriott, Hilton, Hyatt, or Radisson chains",
        basic_node,
        claim=f"The hotel '{name}' is affiliated with Marriott, Hilton, Hyatt, or Radisson (including sub-brands).",
        sources=sources,
        additional_instruction="Confirm the chain on brand or property pages."
    )

    # Location (critical)
    loc_node = evaluator.add_parallel(
        id="Curacao_Location_Requirements",
        desc="Location requirements for Curacao hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Curacao_Location_URL",
        desc="Provide URL confirming Willemstad location and beach access",
        parent=loc_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Curacao_Willemstad",
        "The hotel must be located in or near Willemstad",
        loc_node,
        claim=f"The hotel '{name}' is located in or near Willemstad, Curacao (including neighborhoods like Punda, Otrobanda, Pietermaai, or within a short distance of central Willemstad).",
        sources=sources,
        additional_instruction="Accept near-city neighborhoods commonly considered part of the Willemstad area."
    )
    await _add_and_verify(
        evaluator,
        "Curacao_Beach",
        "The hotel must provide beach access",
        loc_node,
        claim=f"The hotel '{name}' has beach access (on a beach, private beach, or direct access to a beach).",
        sources=sources,
        additional_instruction="Look for 'beachfront', 'private beach', or clear guest access to a beach."
    )

    # Accommodation features (critical)
    acc_node = evaluator.add_parallel(
        id="Curacao_Accommodation_Features",
        desc="Room features for Curacao hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Curacao_Room_URL",
        desc="Provide URL confirming balcony room availability",
        parent=acc_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Curacao_Balcony",
        "The hotel must offer rooms with balconies or terraces",
        acc_node,
        claim=f"The hotel '{name}' offers room categories featuring balconies or terraces.",
        sources=sources,
        additional_instruction="Accept wording such as 'balcony', 'terrace', 'veranda', or similar."
    )

    # Amenities (critical)
    amen_node = evaluator.add_parallel(
        id="Curacao_Amenity_Requirements",
        desc="Facility requirements for Curacao hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Curacao_Amenity_URL",
        desc="Provide URL confirming fitness and pool facilities",
        parent=amen_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Curacao_Fitness",
        "The hotel must have a fitness center",
        amen_node,
        claim=f"The hotel '{name}' has a fitness center or gym.",
        sources=sources,
        additional_instruction="Confirm 'fitness center', 'gym', or equivalent."
    )
    await _add_and_verify(
        evaluator,
        "Curacao_Pool",
        "The hotel must have swimming pool facilities",
        amen_node,
        claim=f"The hotel '{name}' has at least one swimming pool.",
        sources=sources,
        additional_instruction="Any on-site pool qualifies."
    )


async def verify_aruba(evaluator: Evaluator, parent, hotel: Optional[HotelEntry]):
    hotel_node = evaluator.add_parallel(
        id="Aruba_Hotel",
        desc="Identify a suitable hotel in Aruba meeting all specified criteria",
        parent=parent,
        critical=False
    )
    sources = _safe_sources(hotel)
    name = _safe_name(hotel)

    # Basic (critical)
    basic_node = evaluator.add_parallel(
        id="Aruba_Basic_Requirements",
        desc="Chain affiliation and rating requirements for Aruba hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Aruba_Basic_URL",
        desc="Provide URL confirming chain affiliation and rating",
        parent=basic_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Aruba_Chain",
        "The hotel must be affiliated with Marriott, Hilton, Hyatt, or Radisson chains",
        basic_node,
        claim=f"The hotel '{name}' is affiliated with Marriott, Hilton, Hyatt, or Radisson (including sub-brands).",
        sources=sources,
        additional_instruction="Confirm the chain on brand or property pages."
    )
    await _add_and_verify(
        evaluator,
        "Aruba_Rating",
        "The hotel must have a minimum 4-star rating",
        basic_node,
        claim=f"The hotel '{name}' is rated at least 4 stars.",
        sources=sources,
        additional_instruction="Accept 4-star or higher on credible platforms or official statements."
    )

    # Location (critical)
    loc_node = evaluator.add_parallel(
        id="Aruba_Location_Requirements",
        desc="Location requirements for Aruba hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Aruba_Location_URL",
        desc="Provide URL confirming Palm Beach or Eagle Beach location",
        parent=loc_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Aruba_Beach_Area",
        "The hotel must be located in Palm Beach or Eagle Beach area",
        loc_node,
        claim=f"The hotel '{name}' is located in Aruba's Palm Beach or Eagle Beach area (including Noord near Palm Beach).",
        sources=sources,
        additional_instruction="Accept addresses/neighborhoods clearly in the Palm Beach or Eagle Beach zones."
    )

    # Accommodation features (critical)
    acc_node = evaluator.add_parallel(
        id="Aruba_Accommodation_Features",
        desc="Room features for Aruba hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Aruba_Room_URL",
        desc="Provide URL confirming ocean view room availability",
        parent=acc_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Aruba_Ocean_View",
        "The hotel must offer ocean view accommodations",
        acc_node,
        claim=f"The hotel '{name}' offers ocean view (or sea view) room categories.",
        sources=sources,
        additional_instruction="Accept 'ocean view', 'sea view', or 'partial ocean view' room descriptions."
    )

    # Amenities (critical)
    amen_node = evaluator.add_parallel(
        id="Aruba_Amenity_Requirements",
        desc="Facility requirements for Aruba hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Aruba_Amenity_URL",
        desc="Provide URL confirming spa and multiple dining facilities",
        parent=amen_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Aruba_Spa",
        "The hotel must have spa facilities or spa services",
        amen_node,
        claim=f"The hotel '{name}' has spa facilities or on-site spa services.",
        sources=sources,
        additional_instruction="Look for 'spa', 'spa treatment rooms', or comparable terminology."
    )
    await _add_and_verify(
        evaluator,
        "Aruba_Dining",
        "The hotel must have multiple dining options (at least 2 restaurants or bars)",
        amen_node,
        claim=f"The hotel '{name}' offers multiple on-site dining venues, with at least two restaurants or bars.",
        sources=sources,
        additional_instruction="Confirm there are two or more distinct on-property dining/bar outlets."
    )


async def verify_barbados(evaluator: Evaluator, parent, hotel: Optional[HotelEntry]):
    hotel_node = evaluator.add_parallel(
        id="Barbados_Hotel",
        desc="Identify a suitable hotel in Barbados meeting all specified criteria",
        parent=parent,
        critical=False
    )
    sources = _safe_sources(hotel)
    name = _safe_name(hotel)

    # Basic (critical)
    basic_node = evaluator.add_parallel(
        id="Barbados_Basic_Requirements",
        desc="Chain affiliation requirements for Barbados hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Barbados_Basic_URL",
        desc="Provide URL confirming chain affiliation",
        parent=basic_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Barbados_Chain",
        "The hotel must be affiliated with Marriott, Hilton, Hyatt, or Radisson chains",
        basic_node,
        claim=f"The hotel '{name}' is affiliated with Marriott, Hilton, Hyatt, or Radisson (including sub-brands).",
        sources=sources,
        additional_instruction="Confirm brand family on official or reliable pages."
    )

    # Location (critical)
    loc_node = evaluator.add_parallel(
        id="Barbados_Location_Requirements",
        desc="Location requirements for Barbados hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Barbados_Location_URL",
        desc="Provide URL confirming Bridgetown location and beach access",
        parent=loc_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Barbados_Bridgetown",
        "The hotel must be located in or near Bridgetown",
        loc_node,
        claim=f"The hotel '{name}' is located in or near Bridgetown, Barbados (including nearby areas like Garrison, Hastings, Carlisle Bay, or St. Michael near Bridgetown).",
        sources=sources,
        additional_instruction="Accept clearly near-city locations within the Bridgetown area."
    )
    await _add_and_verify(
        evaluator,
        "Barbados_Beach",
        "The hotel must be beachfront or provide beach access",
        loc_node,
        claim=f"The hotel '{name}' is on a beach or provides guest access to a beach.",
        sources=sources,
        additional_instruction="Accept beachfront, private beach, or direct access statements."
    )

    # Accommodation features (critical)
    acc_node = evaluator.add_parallel(
        id="Barbados_Accommodation_Features",
        desc="Room features for Barbados hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Barbados_Room_URL",
        desc="Provide URL confirming ocean view room availability",
        parent=acc_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Barbados_Ocean_View",
        "The hotel must offer ocean view room options",
        acc_node,
        claim=f"The hotel '{name}' offers ocean view (or sea view) room options.",
        sources=sources,
        additional_instruction="Accept 'ocean view', 'sea view', or 'partial ocean view'."
    )

    # Amenities (critical)
    amen_node = evaluator.add_parallel(
        id="Barbados_Amenity_Requirements",
        desc="Facility requirements for Barbados hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Barbados_Amenity_URL",
        desc="Provide URL confirming pool and restaurant facilities",
        parent=amen_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "Barbados_Pool",
        "The hotel must have outdoor pool facilities",
        amen_node,
        claim=f"The hotel '{name}' has at least one outdoor swimming pool.",
        sources=sources,
        additional_instruction="Confirm explicitly outdoor pool; photos may help."
    )
    await _add_and_verify(
        evaluator,
        "Barbados_Restaurant",
        "The hotel must have at least one on-site restaurant",
        amen_node,
        claim=f"The hotel '{name}' has at least one on-site restaurant.",
        sources=sources,
        additional_instruction="Bars are acceptable as additional venues, but at least one restaurant is required."
    )


async def verify_fort_lauderdale(evaluator: Evaluator, parent, hotel: Optional[HotelEntry]):
    hotel_node = evaluator.add_parallel(
        id="Fort_Lauderdale_Hotel",
        desc="Identify a suitable hotel in Fort Lauderdale for pre-cruise stay",
        parent=parent,
        critical=False
    )
    sources = _safe_sources(hotel)
    name = _safe_name(hotel)

    # Basic (critical)
    basic_node = evaluator.add_parallel(
        id="FLL_Basic_Requirements",
        desc="Chain affiliation requirements for Fort Lauderdale hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="FLL_Basic_URL",
        desc="Provide URL confirming chain affiliation",
        parent=basic_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "FLL_Chain",
        "The hotel must be affiliated with a major hotel chain",
        basic_node,
        claim=f"The hotel '{name}' is affiliated with a recognized major international chain such as Marriott (and Bonvoy brands), Hilton (and sub-brands), Hyatt (and sub-brands), IHG (e.g., Holiday Inn, Crowne Plaza), or a comparable global chain.",
        sources=sources,
        additional_instruction="Look for the brand logo or explicit brand wording on official or property pages."
    )

    # Location & proximity (critical)
    loc_node = evaluator.add_parallel(
        id="FLL_Location_Requirements",
        desc="Location and proximity requirements for Fort Lauderdale hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="FLL_Location_URL",
        desc="Provide URL confirming cruise port and airport proximity",
        parent=loc_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "FLL_Cruise_Port",
        "The hotel must be within 3 miles of Port Everglades",
        loc_node,
        claim=f"The hotel '{name}' is located within approximately 3 miles of Port Everglades cruise terminal.",
        sources=sources,
        additional_instruction="Accept statements such as '2 miles from Port Everglades' or similar. If distance in miles is given less than or equal to 3, this satisfies the requirement."
    )
    await _add_and_verify(
        evaluator,
        "FLL_Airport",
        "The hotel must be within 15 minutes drive of FLL airport",
        loc_node,
        claim=f"The hotel '{name}' is within about a 15-minute drive of Fort Lauderdale-Hollywood International Airport (FLL).",
        sources=sources,
        additional_instruction="Accept explicit driving-time mention (<=15 minutes) or very short distance (e.g., 1–3 miles), which reasonably implies <=15 minutes under normal conditions."
    )

    # Services (critical)
    svc_node = evaluator.add_parallel(
        id="FLL_Service_Requirements",
        desc="Transportation service requirements for Fort Lauderdale hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="FLL_Service_URL",
        desc="Provide URL confirming shuttle service availability",
        parent=svc_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "FLL_Shuttle",
        "The hotel must offer shuttle service to Port Everglades",
        svc_node,
        claim=f"The hotel '{name}' offers shuttle service to Port Everglades (complimentary or paid).",
        sources=sources,
        additional_instruction="Look for 'cruise port shuttle', 'Port Everglades shuttle', or 'transportation to the cruise port'."
    )

    # Amenities (critical)
    amen_node = evaluator.add_parallel(
        id="FLL_Amenity_Requirements",
        desc="Facility requirements for Fort Lauderdale hotel",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="FLL_Amenity_URL",
        desc="Provide URL confirming pool facility",
        parent=amen_node,
        critical=True
    )
    await _add_and_verify(
        evaluator,
        "FLL_Pool",
        "The hotel must have a swimming pool",
        amen_node,
        claim=f"The hotel '{name}' has a swimming pool on site.",
        sources=sources,
        additional_instruction="Any on-site pool (indoor or outdoor) qualifies."
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
    # Initialize evaluator
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

    # Add a top-level plan node (non-critical to allow partial credit across destinations)
    plan_node = evaluator.add_parallel(
        id="Southern_Caribbean_Cruise_Vacation_Plan",
        desc="Complete planning of a Southern Caribbean cruise vacation with hotel accommodations at four cruise ports and Fort Lauderdale",
        parent=root,
        critical=False
    )

    # Extract the hotels from the answer
    hotels = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="selected_hotels"
    )

    # Build verification subtrees per destination
    await verify_grenada(evaluator, plan_node, hotels.grenada)
    await verify_curacao(evaluator, plan_node, hotels.curacao)
    await verify_aruba(evaluator, plan_node, hotels.aruba)
    await verify_barbados(evaluator, plan_node, hotels.barbados)
    await verify_fort_lauderdale(evaluator, plan_node, hotels.fort_lauderdale)

    # Return evaluation summary
    return evaluator.get_summary()