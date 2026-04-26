import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "accessible_beach_florida"
TASK_DESCRIPTION = """
Identify a beach in Florida that provides comprehensive accessibility features for wheelchair users, including: (1) Mobi-Mat pathways (non-slip modular mats forming a pathway across sand to the shoreline), (2) beach wheelchairs available for rent or loan (specify whether manual, motorized, or both), and (3) accessible restrooms. Provide the beach name, its specific location in Florida, a description of each accessibility feature, and a reference URL documenting these amenities. Additionally, if available, provide information about a nearby hotel that offers ADA-compliant accessible rooms.
"""


class BeachFeature(BaseModel):
    description: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class WheelchairFeature(BaseModel):
    type: Optional[str] = None
    description: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class BeachInfoExtraction(BaseModel):
    beach_name: Optional[str] = None
    beach_location: Optional[str] = None
    mobi_mat: BeachFeature = Field(default_factory=BeachFeature)
    beach_wheelchair: WheelchairFeature = Field(default_factory=WheelchairFeature)
    accessible_restrooms: BeachFeature = Field(default_factory=BeachFeature)
    official_reference_urls: List[str] = Field(default_factory=list)
    wooden_boardwalk: BeachFeature = Field(default_factory=BeachFeature)


class HotelInfo(BaseModel):
    hotel_name: Optional[str] = None
    hotel_location: Optional[str] = None
    ada_compliance_description: Optional[str] = None
    hotel_url: Optional[str] = None


class VacationPlanExtraction(BaseModel):
    beach: BeachInfoExtraction = Field(default_factory=BeachInfoExtraction)
    hotel: HotelInfo = Field(default_factory=HotelInfo)


def prompt_extract_vacation_plan() -> str:
    return """
    Extract the structured information about the proposed Florida beach accessibility plan from the answer.

    Return a JSON object with this schema:

    {
      "beach": {
        "beach_name": string | null,
        "beach_location": string | null,
        "mobi_mat": {
          "description": string | null,
          "source_urls": string[]  // URLs explicitly mentioned in the answer that document the Mobi-Mat feature
        },
        "beach_wheelchair": {
          "type": string | null,   // one of: "manual", "motorized", "both", or null if unspecified
          "description": string | null,
          "source_urls": string[]  // URLs explicitly mentioned in the answer that document beach wheelchair availability
        },
        "accessible_restrooms": {
          "description": string | null,
          "source_urls": string[]  // URLs explicitly mentioned that document accessible restrooms
        },
        "official_reference_urls": string[], // URLs from official tourism/state park/municipal sources documenting accessibility at the beach
        "wooden_boardwalk": {
          "description": string | null,
          "source_urls": string[]  // URLs documenting wooden boardwalk access if mentioned
        }
      },
      "hotel": {
        "hotel_name": string | null,
        "hotel_location": string | null,
        "ada_compliance_description": string | null, // e.g., "ADA-compliant accessible rooms"
        "hotel_url": string | null
      }
    }

    Rules:
    - Extract only what is explicitly present in the answer text.
    - For URL fields, extract full valid URLs that are present in the answer (plain or markdown links).
    - If multiple beaches/hotels are mentioned, select the first one or the one most prominently described and ignore the rest.
    - Normalize the wheelchair "type" to one of: "manual", "motorized", "both"; if unclear, set to null.
    - If any required item is not present, set it to null (or empty list for URLs).
    """


def _combine_sources(*lists: List[str]) -> List[str]:
    uniq = []
    seen = set()
    for lst in lists:
        for u in lst or []:
            if u and u not in seen:
                uniq.append(u)
                seen.add(u)
    return uniq


async def verify_beach_accessibility(
    evaluator: Evaluator,
    parent_node,
    plan: VacationPlanExtraction,
) -> None:
    beach = plan.beach

    beach_node = evaluator.add_parallel(
        id="Beach_Accessibility_Information",
        desc="Verify the identified beach has all required accessibility features with proper documentation",
        parent=parent_node,
        critical=True,
    )

    name_ok = evaluator.add_custom_node(
        result=bool(beach.beach_name and beach.beach_name.strip()),
        id="Beach_Name_Provided",
        desc="Provide the beach name",
        parent=beach_node,
        critical=True,
    )

    location_ok = evaluator.add_custom_node(
        result=bool(beach.beach_location and beach.beach_location.strip()),
        id="Beach_Location_Provided",
        desc="Specific beach location is provided",
        parent=beach_node,
        critical=True,
    )

    refs_ok = evaluator.add_custom_node(
        result=bool(beach.official_reference_urls and len(beach.official_reference_urls) > 0),
        id="Official_Reference_URLs_Provided",
        desc="Provide valid reference URL(s) documenting accessibility amenities",
        parent=beach_node,
        critical=True,
    )

    located_leaf = evaluator.add_leaf(
        id="Beach_Located_In_Florida",
        desc="Confirm the beach is located in Florida (specific location provided)",
        parent=beach_node,
        critical=True,
    )
    located_claim = f"The beach '{beach.beach_name or ''}' is located in Florida. The provided location is '{beach.beach_location or ''}'."
    await evaluator.verify(
        claim=located_claim,
        node=located_leaf,
        sources=beach.official_reference_urls,
        additional_instruction="Verify from the provided reference URLs that the beach is in Florida. Accept municipal/county/state park or official tourism pages indicating the Florida location.",
    )

    mobi_desc_ok = evaluator.add_custom_node(
        result=bool(beach.mobi_mat.description and beach.mobi_mat.description.strip()),
        id="Mobi_Mat_Description_Provided",
        desc="Description of Mobi-Mat feature is provided",
        parent=beach_node,
        critical=True,
    )

    mobi_leaf = evaluator.add_leaf(
        id="Mobi_Mat_Features",
        desc="Beach must have Mobi-Mat pathways with description",
        parent=beach_node,
        critical=True,
    )
    mobi_sources = _combine_sources(beach.mobi_mat.source_urls, beach.official_reference_urls)
    mobi_claim = "This beach provides Mobi-Mat pathways (non-slip modular mats forming wheelchair-accessible pathways across sand to the shoreline)."
    await evaluator.verify(
        claim=mobi_claim,
        node=mobi_leaf,
        sources=mobi_sources,
        additional_instruction="Check the page(s) for terms like Mobi-Mat, access mat, beach access mat, mobility mat, or similar. Verify the amenity is explicitly provided at the named beach.",
    )

    wc_type_ok = evaluator.add_custom_node(
        result=bool(beach.beach_wheelchair.type and beach.beach_wheelchair.type.strip()),
        id="Beach_Wheelchair_Type_Specified",
        desc="Beach wheelchair type is specified (manual, motorized, or both)",
        parent=beach_node,
        critical=True,
    )

    wc_desc_ok = evaluator.add_custom_node(
        result=bool(beach.beach_wheelchair.description and beach.beach_wheelchair.description.strip()),
        id="Beach_Wheelchair_Description_Provided",
        desc="Description of beach wheelchair availability is provided",
        parent=beach_node,
        critical=True,
    )

    wc_leaf = evaluator.add_leaf(
        id="Beach_Wheelchair_Features",
        desc="Beach must offer beach wheelchairs for rent or loan; type(s) specified",
        parent=beach_node,
        critical=True,
    )
    wc_sources = _combine_sources(beach.beach_wheelchair.source_urls, beach.official_reference_urls)
    wc_type_text = beach.beach_wheelchair.type or "unspecified"
    wc_claim = f"The beach offers beach wheelchairs for rent or loan. Available type(s): {wc_type_text}."
    await evaluator.verify(
        claim=wc_claim,
        node=wc_leaf,
        sources=wc_sources,
        additional_instruction="Verify the availability of beach wheelchairs at the beach. If possible, confirm whether they are manual, motorized, or both. Accept synonyms like 'beach wheelchair', 'mobility beach chair'.",
    )

    rest_desc_ok = evaluator.add_custom_node(
        result=bool(beach.accessible_restrooms.description and beach.accessible_restrooms.description.strip()),
        id="Accessible_Restrooms_Description_Provided",
        desc="Description of accessible restrooms is provided",
        parent=beach_node,
        critical=True,
    )

    rest_leaf = evaluator.add_leaf(
        id="Accessible_Restrooms",
        desc="Beach must have accessible restrooms meeting ADA standards",
        parent=beach_node,
        critical=True,
    )
    rest_sources = _combine_sources(beach.accessible_restrooms.source_urls, beach.official_reference_urls)
    rest_claim = "Accessible restrooms are available at or adjacent to the beach."
    await evaluator.verify(
        claim=rest_claim,
        node=rest_leaf,
        sources=rest_sources,
        additional_instruction="Look for clear indications such as 'accessible restrooms', 'ADA-compliant restrooms', or similar phrasing on official pages for the beach or park.",
    )

    official_leaf = evaluator.add_leaf(
        id="Official_Reference_URLs",
        desc="Provide valid reference URL(s) from official tourism, state park, or municipal sources",
        parent=beach_node,
        critical=True,
    )
    official_claim = "This page is an official tourism, state park, or municipal source (e.g., government or official city/county site) documenting accessibility at the beach."
    await evaluator.verify(
        claim=official_claim,
        node=official_leaf,
        sources=beach.official_reference_urls,
        additional_instruction="Use domain cues (.gov, .fl.us, city/county official websites) or explicit page statements to confirm the page is an official source. If none of the provided URLs are official, fail.",
    )


async def verify_optional_boardwalk(
    evaluator: Evaluator,
    parent_node,
    plan: VacationPlanExtraction,
) -> None:
    boardwalk_node = evaluator.add_parallel(
        id="Supplemental_Amenities",
        desc="Optional supplemental accessibility information",
        parent=parent_node,
        critical=False,
    )

    bw_info_ok = evaluator.add_custom_node(
        result=bool(plan.beach.wooden_boardwalk.description and plan.beach.wooden_boardwalk.description.strip()),
        id="Wooden_Boardwalk_Info_Provided",
        desc="Information about wooden boardwalk access is provided (optional)",
        parent=boardwalk_node,
        critical=False,
    )

    bw_leaf = evaluator.add_leaf(
        id="Wooden_Boardwalk_Information",
        desc="Information about wooden boardwalk access if present at the location",
        parent=boardwalk_node,
        critical=False,
    )
    bw_sources = _combine_sources(plan.beach.wooden_boardwalk.source_urls, plan.beach.official_reference_urls)
    bw_claim = "There is a wooden boardwalk providing access at or near the beach."
    await evaluator.verify(
        claim=bw_claim,
        node=bw_leaf,
        sources=bw_sources,
        additional_instruction="Accept synonyms like 'wooden boardwalk', 'boardwalk access', or 'wooden walkway'. If no sources are provided or the page is unrelated, fail this optional check.",
    )


async def verify_hotel_info(
    evaluator: Evaluator,
    parent_node,
    plan: VacationPlanExtraction,
) -> None:
    hotel = plan.hotel

    hotel_node = evaluator.add_parallel(
        id="Nearby_Accommodation",
        desc="Information about accessible lodging near the selected beach (if available)",
        parent=parent_node,
        critical=False,
    )

    url_provided = evaluator.add_custom_node(
        result=bool(hotel.hotel_url and hotel.hotel_url.strip()),
        id="Hotel_URL_Provided",
        desc="Hotel reference URL provided",
        parent=hotel_node,
        critical=True,
    )

    name_loc_leaf = evaluator.add_custom_node(
        result=bool(hotel.hotel_name and hotel.hotel_name.strip() and hotel.hotel_location and hotel.hotel_location.strip()),
        id="Hotel_Name_and_Location",
        desc="Name and location of a nearby hotel (reasonable proximity to the beach)",
        parent=hotel_node,
        critical=False,
    )

    page_match_leaf = evaluator.add_leaf(
        id="Hotel_Reference_URL",
        desc="Provide a valid reference URL for the hotel information",
        parent=hotel_node,
        critical=False,
    )
    page_match_claim = f"This webpage corresponds to the hotel '{hotel.hotel_name or ''}' located in '{hotel.hotel_location or ''}'."
    await evaluator.verify(
        claim=page_match_claim,
        node=page_match_leaf,
        sources=hotel.hotel_url,
        additional_instruction="Verify the page shows the same hotel name and general location. Minor variations or neighborhood names are acceptable.",
    )

    ada_leaf = evaluator.add_leaf(
        id="ADA_Compliance_Confirmation",
        desc="Confirm the hotel offers ADA-compliant accessible rooms",
        parent=hotel_node,
        critical=False,
    )
    ada_claim = "The hotel offers ADA-compliant accessible rooms (e.g., ADA or accessibility features listed for guest rooms)."
    await evaluator.verify(
        claim=ada_claim,
        node=ada_leaf,
        sources=hotel.hotel_url,
        additional_instruction="Look for terms like 'ADA', 'accessible rooms', or detailed accessibility features on the hotel's official or booking page.",
    )


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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluate whether the proposed Florida beach vacation plan meets all accessibility requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    plan = await evaluator.extract(
        prompt=prompt_extract_vacation_plan(),
        template_class=VacationPlanExtraction,
        extraction_name="vacation_plan",
    )

    await verify_beach_accessibility(evaluator, root, plan)
    await verify_optional_boardwalk(evaluator, root, plan)
    await verify_hotel_info(evaluator, root, plan)

    return evaluator.get_summary()