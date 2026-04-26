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
TASK_ID = "allegiant_np_camping"
TASK_DESCRIPTION = (
    "Identify a U.S. National Park destination that meets all of the following criteria for a summer outdoor recreation trip:\n\n"
    "1. The park must be accessible via an airport that Allegiant Airlines serves with nonstop flights from Las Vegas (LAS)\n"
    "2. The park must be explicitly listed on Allegiant Airlines' National Parks destinations page\n"
    "3. The park must have at least one campground that accepts reservations through Recreation.gov\n"
    "4. The campground must allow reservations to be made at least 4 months in advance of the arrival date\n"
    "5. The destination must offer outdoor recreational activities during the summer months (May through September)\n"
    "6. The selected campground must have at least 100 campsites available for reservation\n\n"
    "Provide the name of the national park, the nearest airport code served by Allegiant from Las Vegas, the campground name, and the number of campsites at that campground."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DestinationExtraction(BaseModel):
    park_name: Optional[str] = None
    airport_code: Optional[str] = None
    airport_name: Optional[str] = None

    allegiant_np_urls: List[str] = Field(default_factory=list)         # URLs to Allegiant National Parks destinations page(s) cited in the answer
    allegiant_route_urls: List[str] = Field(default_factory=list)      # Allegiant route/airport/destination URLs evidencing LAS -> airport nonstop service

    campground_name: Optional[str] = None
    campground_recreation_gov_url: Optional[str] = None
    campsite_count: Optional[str] = None

    summer_activity_desc: Optional[str] = None                         # Description of summer outdoor activities from the answer
    summer_activity_urls: List[str] = Field(default_factory=list)      # NPS/Recreation.gov pages evidencing summer activities

    other_sources: List[str] = Field(default_factory=list)             # Any other relevant support URLs the answer cites


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_destination() -> str:
    return (
        "Extract the key facts the answer provides for the proposed U.S. National Park destination.\n"
        "Return a JSON object with the following fields:\n"
        "- park_name: The name of the national park (string)\n"
        "- airport_code: The nearest airport IATA code that Allegiant serves from Las Vegas (string, e.g., 'BZN')\n"
        "- airport_name: The airport name if provided (string)\n"
        "- allegiant_np_urls: An array of URLs to Allegiant’s National Parks destinations page(s). Only include actual URLs explicitly present in the answer.\n"
        "- allegiant_route_urls: An array of Allegiant URLs (route/destination/airport pages) that support nonstop service from Las Vegas (LAS) to the airport_code. Extract only URLs explicitly shown.\n"
        "- campground_name: The selected campground name (string)\n"
        "- campground_recreation_gov_url: The Recreation.gov page URL for the selected campground (string URL)\n"
        "- campsite_count: The number of campsites at the selected campground as stated in the answer (string, keep formatting exactly; if a range is given, keep it as-is)\n"
        "- summer_activity_desc: Brief description of summer outdoor activities (string) if provided\n"
        "- summer_activity_urls: An array of URLs (e.g., NPS or Recreation.gov) that support summer outdoor activities occurring May–September\n"
        "- other_sources: Any other support URLs cited in the answer (exclude duplicates of the above)\n\n"
        "IMPORTANT:\n"
        "• Extract only what appears explicitly in the answer. Do not invent or infer.\n"
        "• For URLs, include fully qualified URLs exactly as shown. If a URL is missing protocol, prepend http://.\n"
        "• If any field is missing, set it to null (or empty array for URL lists)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())

def _has_urls(urls: List[str]) -> bool:
    return isinstance(urls, list) and len(urls) > 0

def _urls_or_none(urls: List[str]) -> Optional[List[str]]:
    return urls if _has_urls(urls) else None

def _contains_digit(s: Optional[str]) -> bool:
    if not _nonempty(s):
        return False
    return any(ch.isdigit() for ch in str(s))


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_destination(
    evaluator: Evaluator,
    root: Any,
    dest: DestinationExtraction,
) -> None:
    """
    Build the verification tree under a critical 'destination_evaluation' node
    and run the required checks.
    """
    # Create the top-level critical node (parallel aggregation)
    dest_node = evaluator.add_parallel(
        id="destination_evaluation",
        desc="Evaluate whether the proposed national park destination meets all specified criteria for outdoor recreation accessibility and camping, and whether all required information is provided",
        parent=root,
        critical=True
    )

    # ----- Required info existence checks (critical) -----
    park_name_provided_node = evaluator.add_custom_node(
        result=_nonempty(dest.park_name),
        id="park_name_provided",
        desc="The solution provides the name of the national park",
        parent=dest_node,
        critical=True
    )

    airport_code_provided_node = evaluator.add_custom_node(
        result=_nonempty(dest.airport_code) and len(str(dest.airport_code).strip()) >= 3,
        id="airport_code_provided",
        desc="The solution provides the airport code served by Allegiant from Las Vegas",
        parent=dest_node,
        critical=True
    )

    campground_name_provided_node = evaluator.add_custom_node(
        result=_nonempty(dest.campground_name),
        id="campground_name_provided",
        desc="The solution provides the name of the campground",
        parent=dest_node,
        critical=True
    )

    campsite_count_provided_node = evaluator.add_custom_node(
        result=_contains_digit(dest.campsite_count),
        id="campsite_count_provided",
        desc="The solution provides the number of campsites at the campground",
        parent=dest_node,
        critical=True
    )

    # ----- Allegiant accessibility (critical leaf with source-grounding) -----
    allegiant_access_leaf = evaluator.add_leaf(
        id="allegiant_accessibility",
        desc="The airport serving the destination is accessible via nonstop Allegiant Airlines flights from Las Vegas (LAS)",
        parent=dest_node,
        critical=True
    )
    allegiant_access_claim = (
        f"Allegiant Airlines operates or lists nonstop service from Las Vegas (LAS) to the airport '{dest.airport_code}'. "
        f"This must be explicitly supported by Allegiant-owned pages."
    )
    allegiant_access_add_ins = (
        "Use the cited Allegiant route/destination/airport pages to confirm nonstop service from LAS to the given airport code. "
        "Look for phrases like 'Nonstop', 'Direct', or explicit LAS→[airport] route listings. "
        "If no Allegiant URLs are provided or the pages do not support this, judge as not supported."
    )
    await evaluator.verify(
        claim=allegiant_access_claim,
        node=allegiant_access_leaf,
        sources=_urls_or_none(dest.allegiant_route_urls),
        additional_instruction=allegiant_access_add_ins,
        extra_prerequisites=[airport_code_provided_node]
    )

    # ----- National park listed on Allegiant NP page (critical leaf) -----
    allegiant_np_leaf = evaluator.add_leaf(
        id="national_park_listing",
        desc="The destination provides access to a U.S. National Park explicitly listed on Allegiant Airlines' National Parks destinations page",
        parent=dest_node,
        critical=True
    )
    np_claim = (
        f"The Allegiant National Parks destinations page explicitly lists the park '{dest.park_name}' (or a clearly equivalent named unit of the National Park System)."
    )
    np_add_ins = (
        "Open the cited Allegiant National Parks destinations page(s) and check whether the specific park name (or a clearly equivalent National Park unit name) "
        "appears as a listed destination. Allow reasonable naming variants (e.g., 'Grand Canyon National Park' vs. 'Grand Canyon'). "
        "If no Allegiant NP page URLs are provided, judge as not supported."
    )
    await evaluator.verify(
        claim=np_claim,
        node=allegiant_np_leaf,
        sources=_urls_or_none(dest.allegiant_np_urls),
        additional_instruction=np_add_ins,
        extra_prerequisites=[park_name_provided_node]
    )

    # ----- Recreation.gov booking (critical leaf) -----
    recreation_booking_leaf = evaluator.add_leaf(
        id="recreation_gov_booking",
        desc="The national park has at least one campground that accepts reservations through Recreation.gov",
        parent=dest_node,
        critical=True
    )
    booking_claim = (
        f"The campground '{dest.campground_name}' accepts reservations on Recreation.gov (i.e., it is reservable there)."
    )
    booking_add_ins = (
        "Use the Recreation.gov campground page to confirm that reservations are accepted (e.g., presence of a 'Reserve' or 'Book Now' button, or text indicating 'Reservations'). "
        "If the page indicates the campground is first-come, first-served only or lacks reservation capability, judge as not supported. "
        "If no Recreation.gov URL is provided, judge as not supported."
    )
    await evaluator.verify(
        claim=booking_claim,
        node=recreation_booking_leaf,
        sources=dest.campground_recreation_gov_url,
        additional_instruction=booking_add_ins,
        extra_prerequisites=[campground_name_provided_node]
    )

    # ----- Advanced reservation window (critical leaf) -----
    advance_window_leaf = evaluator.add_leaf(
        id="advanced_reservation_window",
        desc="The campground allows reservations to be made at least 4 months in advance of the arrival date",
        parent=dest_node,
        critical=True
    )
    window_claim = (
        f"Reservations for the campground '{dest.campground_name}' can be made at least 4 months prior to the arrival date."
    )
    window_add_ins = (
        "On the Recreation.gov campground page (and any policy subpages it links to), verify the booking window policy. "
        "Accept language like 'reservations open 6 months in advance' or 'rolling 6-month window'. "
        "If the booking window is less than 4 months, or the page does not state a booking window at least 4 months prior, judge as not supported. "
        "If no Recreation.gov URL is provided, judge as not supported."
    )
    await evaluator.verify(
        claim=window_claim,
        node=advance_window_leaf,
        sources=dest.campground_recreation_gov_url,
        additional_instruction=window_add_ins,
        extra_prerequisites=[campground_name_provided_node]
    )

    # ----- Summer season operation (critical leaf) -----
    summer_ops_leaf = evaluator.add_leaf(
        id="summer_season_operation",
        desc="The destination offers outdoor recreational activities during the summer months (May through September)",
        parent=dest_node,
        critical=True
    )
    summer_claim = (
        "The selected destination offers outdoor recreational activities during May through September (summer months)."
    )
    summer_add_ins = (
        "Use the cited NPS/Recreation.gov pages to confirm that outdoor recreational activities (e.g., hiking, camping, boating) are available during May–September. "
        "Seasonal closures for winter are fine; ensure summer activities are available. "
        "If no relevant URLs are provided and the page does not support summer activities, judge as not supported."
    )
    summer_sources = _urls_or_none(dest.summer_activity_urls) or dest.campground_recreation_gov_url
    await evaluator.verify(
        claim=summer_claim,
        node=summer_ops_leaf,
        sources=summer_sources,
        additional_instruction=summer_add_ins,
        extra_prerequisites=[park_name_provided_node]
    )

    # ----- Minimum campsite capacity (critical leaf) -----
    capacity_leaf = evaluator.add_leaf(
        id="minimum_campsite_capacity",
        desc="The selected campground has at least 100 campsites available for reservation",
        parent=dest_node,
        critical=True
    )
    capacity_claim = (
        f"The campground '{dest.campground_name}' has at least 100 reservable campsites."
    )
    capacity_add_ins = (
        "Use the Recreation.gov campground page to verify the total number of reservable campsites (e.g., 'Sites', 'Number of Campsites'). "
        "If the total reservable sites are fewer than 100, judge as not supported. "
        "If no Recreation.gov URL is provided or the page does not support the claim, judge as not supported."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=dest.campground_recreation_gov_url,
        additional_instruction=capacity_add_ins,
        extra_prerequisites=[campground_name_provided_node, campsite_count_provided_node]
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the Allegiant National Parks camping accessibility task.
    """
    # Initialize evaluator with a parallel root (we add a critical child node under root)
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

    # Extract destination details from the answer
    dest = await evaluator.extract(
        prompt=prompt_extract_destination(),
        template_class=DestinationExtraction,
        extraction_name="destination_extraction"
    )

    # Optionally record custom info for debugging
    evaluator.add_custom_info(
        info={
            "park_name": dest.park_name,
            "airport_code": dest.airport_code,
            "campground_name": dest.campground_name,
            "campsite_count": dest.campsite_count,
            "allegiant_np_urls": dest.allegiant_np_urls,
            "allegiant_route_urls": dest.allegiant_route_urls,
            "campground_recreation_gov_url": dest.campground_recreation_gov_url,
            "summer_activity_urls": dest.summer_activity_urls
        },
        info_type="extraction_summary"
    )

    # Build verification tree and run checks
    await build_and_verify_destination(evaluator, root, dest)

    # Return standardized summary
    return evaluator.get_summary()