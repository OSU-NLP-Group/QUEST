import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "thanksgiving_baltimore_dc_options"
TASK_DESCRIPTION = (
    "Find three Thanksgiving dining options in the Baltimore/Washington D.C. metro area that meet the following requirements:\n\n"
    "1. One casual dining chain restaurant that is open for dine-in on Thanksgiving Day. Provide:\n"
    "- Restaurant name and specific location address\n"
    "- Thanksgiving Day operating hours (opening and closing time)\n"
    "- Adult Thanksgiving meal pricing\n"
    "- Menu type (special Thanksgiving menu, regular menu, or both)\n"
    "- Reservation requirements (required, recommended, or not needed) and contact information\n"
    "- Reference URL confirming all information\n\n"
    "2. One upscale steakhouse restaurant that is open for dine-in on Thanksgiving Day. Provide:\n"
    "- Restaurant name and specific location address\n"
    "- Thanksgiving Day operating hours (opening and closing time)\n"
    "- Adult Thanksgiving meal pricing\n"
    "- Menu type (e.g., 2-course or 3-course special menu, or regular menu)\n"
    "- Reservation requirements and contact information (phone number or reservation link)\n"
    "- Reference URL confirming all information\n\n"
    "3. One restaurant or grocery store offering prepared Thanksgiving meal packages for takeout/pickup in the Baltimore/DC area. Provide:\n"
    "- Provider name and specific location or service area\n"
    "- Complete description of what is included in the meal package (main items with sizes/weights and all side dishes)\n"
    "- Number of people the package serves\n"
    "- Total price for the package\n"
    "- Advance ordering deadline and pickup date/time options\n"
    "- How to place an order (phone number, website, or online ordering link)\n"
    "- Reference URL confirming all information\n\n"
    "For all three options, information must be verified through official sources (restaurant websites, corporate announcements, or official store pages). "
    "All reference URLs must be provided to support the information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HoursInfo(BaseModel):
    hours_text: Optional[str] = None
    opening_time: Optional[str] = None
    closing_time: Optional[str] = None


class CasualChainOption(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    thanksgiving_hours: Optional[HoursInfo] = None
    adult_pricing: Optional[str] = None
    menu_type: Optional[str] = None  # e.g., "Thanksgiving special menu", "regular menu", "both"
    reservation_requirement: Optional[str] = None  # required / recommended / not needed
    contact_phone: Optional[str] = None
    reservation_link: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)


class UpscaleSteakhouseOption(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    thanksgiving_hours: Optional[HoursInfo] = None
    adult_pricing: Optional[str] = None
    menu_type: Optional[str] = None  # e.g., "3-course prix fixe", "regular menu", etc.
    reservation_requirement: Optional[str] = None
    contact_phone: Optional[str] = None
    reservation_link: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)


class PreparedMealOption(BaseModel):
    provider_name: Optional[str] = None
    location_or_service_area: Optional[str] = None
    package_contents: Optional[str] = None  # include main items with sizes/weights and all sides
    serves: Optional[str] = None
    total_price: Optional[str] = None
    ordering_deadline: Optional[str] = None
    pickup_options: Optional[str] = None
    how_to_order: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)


class ThanksgivingOptionsExtraction(BaseModel):
    casual_chain: Optional[CasualChainOption] = None
    upscale_steakhouse: Optional[UpscaleSteakhouseOption] = None
    prepared_meal: Optional[PreparedMealOption] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_thanksgiving_options() -> str:
    return """
You will extract exactly three entries from the answer text, one for each required category. If multiple candidates are provided for a category, choose the best one that clearly satisfies the requirements and appears first. If any field is missing in the answer, return null for that field. Extract strings exactly as written in the answer. For URLs, only extract URLs explicitly present in the answer text, and prefer official sources (restaurant corporate site, official location page, official announcement, or official grocery/store page). Do not invent any URLs.

Return a JSON object with this schema:

{
  "casual_chain": {
    "name": string|null,
    "address": string|null,
    "city": string|null,
    "state": string|null,
    "thanksgiving_hours": {
      "hours_text": string|null,
      "opening_time": string|null,
      "closing_time": string|null
    } | null,
    "adult_pricing": string|null,
    "menu_type": string|null,
    "reservation_requirement": string|null,  // one of: "required", "recommended", "not needed" (if stated; otherwise null)
    "contact_phone": string|null,
    "reservation_link": string|null,
    "official_urls": [string, ...] // only official brand/restaurant pages or official location/ordering pages; if none in the answer, return []
  },
  "upscale_steakhouse": {
    "name": string|null,
    "address": string|null,
    "city": string|null,
    "state": string|null,
    "thanksgiving_hours": {
      "hours_text": string|null,
      "opening_time": string|null,
      "closing_time": string|null
    } | null,
    "adult_pricing": string|null,
    "menu_type": string|null,              // e.g., "2-course prix fixe", "3-course prix fixe", "regular menu", or both
    "reservation_requirement": string|null,
    "contact_phone": string|null,
    "reservation_link": string|null,
    "official_urls": [string, ...]
  },
  "prepared_meal": {
    "provider_name": string|null,          // could be a restaurant or grocery chain
    "location_or_service_area": string|null,
    "package_contents": string|null,       // include main item(s) with sizes/weights and all sides if given
    "serves": string|null,                 // number of people it serves (string allowed, e.g., "4-6", "8-10")
    "total_price": string|null,
    "ordering_deadline": string|null,
    "pickup_options": string|null,         // pickup date/time windows
    "how_to_order": string|null,           // phone or website/ordering link description
    "official_urls": [string, ...]
  }
}

Rules:
- For hours, if the answer provides a single range (e.g., "11am–6pm"), put that in hours_text, and also try to split opening_time and closing_time if clearly stated in the answer; otherwise leave them null.
- All pricing fields must be strings as shown (e.g., "$39.99", "$40–$55", "starts at $45").
- For URLs, only include URLs explicitly present in the answer; prefer official pages. If no official URL is present in the answer, keep the list empty.
- Do not add or infer information not in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _join_hours(h: Optional[HoursInfo]) -> str:
    if not h:
        return ""
    if h.hours_text and h.hours_text.strip():
        return h.hours_text.strip()
    # Fallback: try to combine opening and closing
    open_t = (h.opening_time or "").strip()
    close_t = (h.closing_time or "").strip()
    if open_t and close_t:
        return f"{open_t} - {close_t}"
    return open_t or close_t


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _normalize_sources(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    # Filter out obviously empty strings
    cleaned = [u for u in urls if isinstance(u, str) and u.strip()]
    return cleaned if cleaned else None


def _contact_summary(res_req: Optional[str], phone: Optional[str], link: Optional[str]) -> str:
    parts = []
    if res_req:
        parts.append(f"reservations are {res_req}")
    if phone:
        parts.append(f"phone: {phone}")
    if link:
        parts.append(f"reservation/contact link: {link}")
    return "; ".join(parts) if parts else "no contact method provided"


def _metro_area_instruction() -> str:
    return (
        "Treat addresses in Washington, DC, Baltimore, MD, or nearby Maryland counties (e.g., Montgomery, Prince George's, Anne Arundel, Howard, Baltimore City/County) "
        "and Northern Virginia (e.g., Arlington, Alexandria, Fairfax, Loudoun) as within the Baltimore/Washington D.C. metro area. "
        "Use the official page to confirm the address/city shown; minor common-sense about metro boundaries is acceptable."
    )


def _official_source_instruction() -> str:
    return (
        "Use only the provided official URL(s) (restaurant corporate site, official location page, official brand announcement, or official grocery/store page) "
        "to verify each claim. Do not rely on third-party blogs or news articles. Allow minor formatting/wording differences. "
        "If the provided URL is clearly not official or does not support the claim, the claim should be considered not supported."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_option_casual_chain(
    evaluator: Evaluator,
    parent,
    data: Optional[CasualChainOption],
) -> None:
    option_node = evaluator.add_sequential(
        id="option_1_casual_dine_in_chain",
        desc="Option 1: Casual dining chain restaurant open for dine-in on Thanksgiving Day (Baltimore/DC metro).",
        parent=parent,
        critical=False,
    )

    # Eligibility (critical parallel)
    eligibility = evaluator.add_parallel(
        id="opt1_eligibility",
        desc="Meets category/location/Thanksgiving-open requirements for the casual dine-in option.",
        parent=option_node,
        critical=True,
    )

    urls = _normalize_sources(data.official_urls if data else None)
    name = (data.name if data and data.name else "the restaurant").strip() if data else "the restaurant"
    address = (data.address or "").strip() if data else ""
    city = (data.city or "").strip() if data else ""
    state = (data.state or "").strip() if data else ""
    full_loc = ", ".join([p for p in [address, city, state] if p])

    # Leaves for eligibility
    leaf_cat = evaluator.add_leaf(
        id="opt1_category_casual_chain",
        desc="Restaurant qualifies as a casual dining chain (per question/constraints).",
        parent=eligibility,
        critical=True,
    )
    leaf_dinein = evaluator.add_leaf(
        id="opt1_service_format_dine_in",
        desc="Service format is explicitly stated as dine-in (or dine-in available) for Thanksgiving Day.",
        parent=eligibility,
        critical=True,
    )
    leaf_within = evaluator.add_leaf(
        id="opt1_within_metro_area",
        desc="Specific location address is provided and is within the Baltimore/Washington D.C. metro area.",
        parent=eligibility,
        critical=True,
    )
    leaf_open = evaluator.add_leaf(
        id="opt1_open_on_thanksgiving",
        desc="Officially confirmed open for dine-in on Thanksgiving Day.",
        parent=eligibility,
        critical=True,
    )

    claims_elig = [
        (
            f"{name} is a casual dining chain restaurant brand.",
            urls,
            leaf_cat,
            _official_source_instruction(),
        ),
        (
            f"The {name} location at {full_loc if full_loc else '(location unspecified)'} offers dine-in service on Thanksgiving Day.",
            urls,
            leaf_dinein,
            _official_source_instruction(),
        ),
        (
            f"The location address for {name} is {full_loc if full_loc else '(address missing)'}, which is within the Baltimore/Washington D.C. metro area.",
            urls,
            leaf_within,
            _official_source_instruction() + " " + _metro_area_instruction(),
        ),
        (
            f"{name} is open for dine-in on Thanksgiving Day at {full_loc if full_loc else '(location)'}.",
            urls,
            leaf_open,
            _official_source_instruction(),
        ),
    ]
    await evaluator.batch_verify(claims_elig)

    # Required details (critical parallel)
    req = evaluator.add_parallel(
        id="opt1_required_details",
        desc="All required details for the casual dine-in option are provided.",
        parent=option_node,
        critical=True,
    )

    # name_provided (custom existence)
    evaluator.add_custom_node(
        result=bool(data and data.name and data.name.strip()),
        id="opt1_name_provided",
        desc="Restaurant name is provided.",
        parent=req,
        critical=True,
    )

    # official_source_url (custom existence) - add BEFORE detailed verifications to enable precondition gating
    evaluator.add_custom_node(
        result=_has_any_url(urls),
        id="opt1_official_source_url",
        desc="At least one official source URL is provided that supports the key claims (open status, hours, pricing, menu type, reservation/contact).",
        parent=req,
        critical=True,
    )

    # thanksgiving_hours
    leaf_hours = evaluator.add_leaf(
        id="opt1_thanksgiving_hours",
        desc="Thanksgiving Day operating hours include both opening and closing times.",
        parent=req,
        critical=True,
    )
    hours_str = _join_hours(data.thanksgiving_hours if data else None)
    claim_hours = (
        f"The Thanksgiving Day hours for {name} at {full_loc if full_loc else '(location)'} are '{hours_str}'. "
        "They should represent both opening and closing times."
    )

    # adult_pricing
    leaf_price = evaluator.add_leaf(
        id="opt1_adult_pricing",
        desc="Adult Thanksgiving meal pricing is provided (specific price or price range).",
        parent=req,
        critical=True,
    )
    price_str = (data.adult_pricing or "").strip() if data else ""
    claim_price = f"The adult Thanksgiving meal pricing for {name} is '{price_str}'."

    # menu_type
    leaf_menu = evaluator.add_leaf(
        id="opt1_menu_type",
        desc="Menu type is specified (special Thanksgiving menu, regular menu, or both).",
        parent=req,
        critical=True,
    )
    menu_type = (data.menu_type or "").strip() if data else ""
    claim_menu = f"The Thanksgiving menu type for {name} is '{menu_type}'."

    # reservation_and_contact
    leaf_res = evaluator.add_leaf(
        id="opt1_reservation_and_contact",
        desc="Reservation requirement status (required/recommended/not needed) and contact method (phone and/or website/reservation link) are provided.",
        parent=req,
        critical=True,
    )
    contact_text = _contact_summary(
        (data.reservation_requirement or "").strip() if data else "",
        (data.contact_phone or "").strip() if data else "",
        (data.reservation_link or "").strip() if data else "",
    )
    claim_res = (
        f"For {name}, {contact_text}. This should reflect the Thanksgiving Day dine-in reservation requirement and contact method."
    )

    # Verify required details (respecting official source gating via critical sibling preconditions)
    await evaluator.batch_verify([
        (claim_hours, urls, leaf_hours, _official_source_instruction()),
        (claim_price, urls, leaf_price, _official_source_instruction()),
        (claim_menu, urls, leaf_menu, _official_source_instruction()),
        (claim_res, urls, leaf_res, _official_source_instruction()),
    ])


async def build_option_upscale_steakhouse(
    evaluator: Evaluator,
    parent,
    data: Optional[UpscaleSteakhouseOption],
) -> None:
    option_node = evaluator.add_sequential(
        id="option_2_upscale_steakhouse_dine_in",
        desc="Option 2: Upscale steakhouse restaurant open for dine-in on Thanksgiving Day (Baltimore/DC metro).",
        parent=parent,
        critical=False,
    )

    # Eligibility (critical parallel)
    eligibility = evaluator.add_parallel(
        id="opt2_eligibility",
        desc="Meets category/location/Thanksgiving-open requirements for the upscale steakhouse option.",
        parent=option_node,
        critical=True,
    )

    urls = _normalize_sources(data.official_urls if data else None)
    name = (data.name if data and data.name else "the restaurant").strip() if data else "the restaurant"
    address = (data.address or "").strip() if data else ""
    city = (data.city or "").strip() if data else ""
    state = (data.state or "").strip() if data else ""
    full_loc = ", ".join([p for p in [address, city, state] if p])

    leaf_cat = evaluator.add_leaf(
        id="opt2_category_upscale_steakhouse",
        desc="Restaurant qualifies as an upscale steakhouse (and as a chain if required by constraints).",
        parent=eligibility,
        critical=True,
    )
    leaf_dinein = evaluator.add_leaf(
        id="opt2_service_format_dine_in",
        desc="Service format is explicitly stated as dine-in (or dine-in available) for Thanksgiving Day.",
        parent=eligibility,
        critical=True,
    )
    leaf_within = evaluator.add_leaf(
        id="opt2_within_metro_area",
        desc="Specific location address is provided and is within the Baltimore/Washington D.C. metro area.",
        parent=eligibility,
        critical=True,
    )
    leaf_open = evaluator.add_leaf(
        id="opt2_open_on_thanksgiving",
        desc="Officially confirmed open for dine-in on Thanksgiving Day.",
        parent=eligibility,
        critical=True,
    )

    claims_elig = [
        (
            f"{name} is an upscale steakhouse restaurant.",
            urls,
            leaf_cat,
            _official_source_instruction(),
        ),
        (
            f"The {name} location at {full_loc if full_loc else '(location unspecified)'} offers dine-in service on Thanksgiving Day.",
            urls,
            leaf_dinein,
            _official_source_instruction(),
        ),
        (
            f"The location address for {name} is {full_loc if full_loc else '(address missing)'}, which is within the Baltimore/Washington D.C. metro area.",
            urls,
            leaf_within,
            _official_source_instruction() + " " + _metro_area_instruction(),
        ),
        (
            f"{name} is open for dine-in on Thanksgiving Day at {full_loc if full_loc else '(location)'}.",
            urls,
            leaf_open,
            _official_source_instruction(),
        ),
    ]
    await evaluator.batch_verify(claims_elig)

    # Required details (critical parallel)
    req = evaluator.add_parallel(
        id="opt2_required_details",
        desc="All required details for the upscale steakhouse option are provided.",
        parent=option_node,
        critical=True,
    )

    # name_provided
    evaluator.add_custom_node(
        result=bool(data and data.name and data.name.strip()),
        id="opt2_name_provided",
        desc="Restaurant name is provided.",
        parent=req,
        critical=True,
    )

    # official_source_url (existence; add before other verifications for precondition gating)
    evaluator.add_custom_node(
        result=_has_any_url(urls),
        id="opt2_official_source_url",
        desc="At least one official source URL is provided that supports the key claims (open status, hours, pricing, menu type, reservation/contact).",
        parent=req,
        critical=True,
    )

    # thanksgiving_hours
    leaf_hours = evaluator.add_leaf(
        id="opt2_thanksgiving_hours",
        desc="Thanksgiving Day operating hours include both opening and closing times.",
        parent=req,
        critical=True,
    )
    hours_str = _join_hours(data.thanksgiving_hours if data else None)
    claim_hours = (
        f"The Thanksgiving Day hours for {name} at {full_loc if full_loc else '(location)'} are '{hours_str}', including opening and closing times."
    )

    # adult_pricing
    leaf_price = evaluator.add_leaf(
        id="opt2_adult_pricing",
        desc="Adult Thanksgiving meal pricing is provided (specific price or price range).",
        parent=req,
        critical=True,
    )
    price_str = (data.adult_pricing or "").strip() if data else ""
    claim_price = f"The adult Thanksgiving meal pricing for {name} is '{price_str}'."

    # menu_type
    leaf_menu = evaluator.add_leaf(
        id="opt2_menu_type",
        desc="Menu type is specified (e.g., special prix-fixe/coursed menu vs regular menu vs both).",
        parent=req,
        critical=True,
    )
    menu_type = (data.menu_type or "").strip() if data else ""
    claim_menu = f"The Thanksgiving menu type for {name} is '{menu_type}'."

    # reservation_and_contact
    leaf_res = evaluator.add_leaf(
        id="opt2_reservation_and_contact",
        desc="Reservation requirement status and contact method (phone and/or reservation link) are provided.",
        parent=req,
        critical=True,
    )
    contact_text = _contact_summary(
        (data.reservation_requirement or "").strip() if data else "",
        (data.contact_phone or "").strip() if data else "",
        (data.reservation_link or "").strip() if data else "",
    )
    claim_res = (
        f"For {name}, {contact_text}. This should reflect Thanksgiving Day dine-in reservations and contact method."
    )

    await evaluator.batch_verify([
        (claim_hours, urls, leaf_hours, _official_source_instruction()),
        (claim_price, urls, leaf_price, _official_source_instruction()),
        (claim_menu, urls, leaf_menu, _official_source_instruction()),
        (claim_res, urls, leaf_res, _official_source_instruction()),
    ])


async def build_option_prepared_meal(
    evaluator: Evaluator,
    parent,
    data: Optional[PreparedMealOption],
) -> None:
    option_node = evaluator.add_sequential(
        id="option_3_prepared_meal_package_takeout",
        desc="Option 3: Restaurant or grocery store offering a prepared Thanksgiving meal package for takeout/pickup (Baltimore/DC metro).",
        parent=parent,
        critical=False,
    )

    # Eligibility (critical parallel)
    eligibility = evaluator.add_parallel(
        id="opt3_eligibility",
        desc="Meets prepared-meal-package and location requirements.",
        parent=option_node,
        critical=True,
    )

    urls = _normalize_sources(data.official_urls if data else None)
    provider = (data.provider_name if data and data.provider_name else "the provider").strip() if data else "the provider"
    service_area = (data.location_or_service_area or "").strip() if data else ""

    leaf_takeout = evaluator.add_leaf(
        id="opt3_service_format_takeout_pickup",
        desc="Service format is explicitly stated as takeout/pickup (prepared meal package).",
        parent=eligibility,
        critical=True,
    )
    leaf_within = evaluator.add_leaf(
        id="opt3_within_metro_area",
        desc="Specific location address or clearly stated service area within the Baltimore/Washington D.C. metro area is provided.",
        parent=eligibility,
        critical=True,
    )
    leaf_pkg = evaluator.add_leaf(
        id="opt3_prepared_package_confirmed",
        desc="Officially confirmed to offer a complete prepared Thanksgiving meal package (not just individual items).",
        parent=eligibility,
        critical=True,
    )

    claims_elig = [
        (
            f"{provider} offers a prepared Thanksgiving meal package for takeout or pickup.",
            urls,
            leaf_takeout,
            _official_source_instruction(),
        ),
        (
            f"The stated location or service area for {provider} is '{service_area}', which is within the Baltimore/Washington D.C. metro area.",
            urls,
            leaf_within,
            _official_source_instruction() + " " + _metro_area_instruction(),
        ),
        (
            f"{provider} offers a complete prepared Thanksgiving meal package (includes a main and sides), not only individual items.",
            urls,
            leaf_pkg,
            _official_source_instruction(),
        ),
    ]
    await evaluator.batch_verify(claims_elig)

    # Required details (critical parallel)
    req = evaluator.add_parallel(
        id="opt3_required_details",
        desc="All required details for the prepared meal package are provided.",
        parent=option_node,
        critical=True,
    )

    # provider_name_provided
    evaluator.add_custom_node(
        result=bool(data and data.provider_name and data.provider_name.strip()),
        id="opt3_provider_name_provided",
        desc="Provider name is provided.",
        parent=req,
        critical=True,
    )

    # official_source_url (existence; add before field verifications)
    evaluator.add_custom_node(
        result=_has_any_url(urls),
        id="opt3_official_source_url",
        desc="At least one official source URL is provided that supports the key claims (package contents, serving size, price, deadline, pickup options, ordering method).",
        parent=req,
        critical=True,
    )

    # package_contents_complete
    leaf_contents = evaluator.add_leaf(
        id="opt3_package_contents_complete",
        desc="Complete description of package contents is provided, including main item(s) with sizes/weights and all included side dishes.",
        parent=req,
        critical=True,
    )
    contents = (data.package_contents or "").strip() if data else ""
    claim_contents = (
        f"The package contents for {provider} are: '{contents}'. This description should include the main item(s) with sizes/weights and all sides."
    )

    # serving_size
    leaf_serves = evaluator.add_leaf(
        id="opt3_serving_size",
        desc="Number of people the package serves is specified.",
        parent=req,
        critical=True,
    )
    serves = (data.serves or "").strip() if data else ""
    claim_serves = f"The package from {provider} serves '{serves}' people."

    # total_price
    leaf_price = evaluator.add_leaf(
        id="opt3_total_price",
        desc="Total price for the package is provided.",
        parent=req,
        critical=True,
    )
    total_price = (data.total_price or "").strip() if data else ""
    claim_price = f"The total price for {provider}'s Thanksgiving meal package is '{total_price}'."

    # ordering_deadline
    leaf_deadline = evaluator.add_leaf(
        id="opt3_ordering_deadline",
        desc="Advance ordering deadline is specified.",
        parent=req,
        critical=True,
    )
    deadline = (data.ordering_deadline or "").strip() if data else ""
    claim_deadline = f"The advance ordering deadline for {provider}'s package is '{deadline}'."

    # pickup_options
    leaf_pickup = evaluator.add_leaf(
        id="opt3_pickup_options",
        desc="Pickup date/time options are provided.",
        parent=req,
        critical=True,
    )
    pickup = (data.pickup_options or "").strip() if data else ""
    claim_pickup = f"The pickup date/time options are '{pickup}'."

    # how_to_order
    leaf_order = evaluator.add_leaf(
        id="opt3_how_to_order",
        desc="How to place an order is provided (phone number and/or website/online ordering link).",
        parent=req,
        critical=True,
    )
    order_how = (data.how_to_order or "").strip() if data else ""
    claim_order = f"How to order for {provider}: '{order_how}'."

    await evaluator.batch_verify([
        (claim_contents, urls, leaf_contents, _official_source_instruction()),
        (claim_serves, urls, leaf_serves, _official_source_instruction()),
        (claim_price, urls, leaf_price, _official_source_instruction()),
        (claim_deadline, urls, leaf_deadline, _official_source_instruction()),
        (claim_pickup, urls, leaf_pickup, _official_source_instruction()),
        (claim_order, urls, leaf_order, _official_source_instruction()),
    ])


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Baltimore/DC Thanksgiving options task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Evaluate three options independently; critical gating is inside each option
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

    # Extract structured info for the three options
    extracted = await evaluator.extract(
        prompt=prompt_extract_thanksgiving_options(),
        template_class=ThanksgivingOptionsExtraction,
        extraction_name="thanksgiving_options_extraction",
    )

    # Build and verify option 1 (casual chain dine-in)
    await build_option_casual_chain(evaluator, root, extracted.casual_chain if extracted else None)

    # Build and verify option 2 (upscale steakhouse dine-in)
    await build_option_upscale_steakhouse(evaluator, root, extracted.upscale_steakhouse if extracted else None)

    # Build and verify option 3 (prepared meal package takeout/pickup)
    await build_option_prepared_meal(evaluator, root, extracted.prepared_meal if extracted else None)

    return evaluator.get_summary()