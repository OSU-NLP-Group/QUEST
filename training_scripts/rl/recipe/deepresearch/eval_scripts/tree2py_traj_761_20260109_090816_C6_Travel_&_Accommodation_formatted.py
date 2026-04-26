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
TASK_ID = "two_in_park_lodges_summer_pet_access_accessible"
TASK_DESCRIPTION = (
    "I am planning a summer trip (June through August) to U.S. National Parks with my elderly mother who uses a "
    "wheelchair and my dog. I need to find two lodges that are located inside national park boundaries (not just nearby) "
    "where we can stay. For each lodge, please provide the following information: "
    "1. Lodge name and national park location. "
    "2. Confirmation that the lodge is: located within the park boundaries (not outside), operated by an authorized "
    "National Park Service concessioner, and open during summer months (June-August). "
    "3. Pet accommodation details: availability of pet-friendly rooms or cabins that allow dogs, and the nightly pet fee amount. "
    "4. Accessibility features: availability of ADA-compliant guest rooms, specific accessible bathroom features in these rooms "
    "(such as grab bars, roll-in showers, or accessible tubs), and confirmation that the lodge's dining room is wheelchair accessible. "
    "5. Dining services: availability of an on-site restaurant or dining room, and whether dinner reservations are required or recommended. "
    "6. Amenities and booking: confirmation of a gift shop on-site, availability of WiFi (at least in common areas/lobby), the check-in time "
    "(must be 4:00 PM or later), and the official method for making reservations (website or phone number). For each piece of information, "
    "please provide a direct URL link to an official source (such as the lodge's official website, the National Park Service website, or the "
    "authorized concessioner's website) that verifies the information."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class Lodge(BaseModel):
    # Identification
    name: Optional[str] = None
    national_park: Optional[str] = None

    # Textual/Value details from the answer (used in claims)
    pet_fee_amount: Optional[str] = None
    check_in_time: Optional[str] = None
    dinner_reservation_policy: Optional[str] = None
    bathroom_features: Optional[str] = None
    wifi_scope: Optional[str] = None
    reservation_method: Optional[str] = None
    reservation_phone: Optional[str] = None

    # Per-attribute official source URLs (as cited in the answer)
    url_location: List[str] = Field(default_factory=list)
    url_within_park: List[str] = Field(default_factory=list)
    url_concessioner: List[str] = Field(default_factory=list)
    url_open_summer: List[str] = Field(default_factory=list)
    url_pet_friendly: List[str] = Field(default_factory=list)
    url_pet_fee: List[str] = Field(default_factory=list)
    url_ada_rooms: List[str] = Field(default_factory=list)
    url_bathroom_features: List[str] = Field(default_factory=list)
    url_dining_wheelchair: List[str] = Field(default_factory=list)
    url_on_site_restaurant: List[str] = Field(default_factory=list)
    url_dinner_reservation_policy: List[str] = Field(default_factory=list)
    url_gift_shop: List[str] = Field(default_factory=list)
    url_wifi: List[str] = Field(default_factory=list)
    url_checkin_time: List[str] = Field(default_factory=list)
    url_reservation_method: List[str] = Field(default_factory=list)


class LodgesExtraction(BaseModel):
    lodges: List[Lodge] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_lodges() -> str:
    return """
Extract exactly the lodging options the answer proposes for staying INSIDE U.S. National Park boundaries.

Return a JSON object with a `lodges` array (in the same order as in the answer). Each item must include the following fields:

Required identification (strings):
- name
- national_park

Text/value details as written in the answer (strings, can be null if not given):
- pet_fee_amount               # e.g., "$25 per night" (must reflect 'nightly' if stated)
- check_in_time                # e.g., "4:00 PM"
- dinner_reservation_policy    # e.g., "reservations recommended"
- bathroom_features            # e.g., "roll-in shower; grab bars"
- wifi_scope                   # e.g., "WiFi available in lobby"
- reservation_method           # e.g., "Book online" or "Call reservations"
- reservation_phone            # e.g., "+1-307-555-1234"

For each required verification, extract the official source URLs actually cited in the answer. Include ALL provided official URLs; if none given for that field, use an empty list.

URL list fields (arrays of strings):
- url_location                  # supports the name+park identification
- url_within_park               # supports that the lodge is inside the park boundary (not outside)
- url_concessioner              # supports that it is operated by an authorized NPS concessioner
- url_open_summer               # supports being open June–August
- url_pet_friendly              # supports designated pet-friendly rooms/cabins allowing dogs
- url_pet_fee                   # supports the nightly pet fee amount
- url_ada_rooms                 # supports ADA/accessible rooms available
- url_bathroom_features         # supports specific accessible bathroom features (grab bars, roll-in shower, accessible tub, etc.)
- url_dining_wheelchair         # supports that dining room/restaurant is wheelchair accessible
- url_on_site_restaurant        # supports that an on-site restaurant/dining room exists
- url_dinner_reservation_policy # supports dinner reservations policy (required/recommended)
- url_gift_shop                 # supports that an on-site gift shop exists
- url_wifi                      # supports WiFi available at least in lobby/common areas
- url_checkin_time              # supports check-in time (must be 4:00 PM or later)
- url_reservation_method        # supports official reservation method (website or phone)

General rules:
- Only extract what is explicitly present in the answer. Do not invent.
- URLs must be full and valid. Extract all URLs the answer associates with each corresponding fact.
- If a field is missing in the answer, set it to null (for strings) or an empty array (for URL lists).

Extract all lodges the answer actually proposes as options (not “nearby” or examples). Do NOT add extra lodges.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _urls_present(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


def _safe_park(lodge: Lodge) -> str:
    return lodge.national_park or "the national park"


def _norm_name(s: Optional[str]) -> str:
    return (s or "").strip().lower()


# --------------------------------------------------------------------------- #
# Lodge verification                                                          #
# --------------------------------------------------------------------------- #
async def verify_lodge(evaluator: Evaluator, parent_node, lodge: Lodge, index: int) -> None:
    """
    Build verification sub-tree for one lodge (parallel block with per-attribute sub-blocks).
    Follow the rubric: each attribute requires support by an official URL and the stated info.
    """

    lodge_idx = index + 1
    lodge_node = evaluator.add_parallel(
        id=f"Lodge_{lodge_idx}",
        desc=f"{'First' if lodge_idx == 1 else 'Second'} lodge and its required attributes (with supporting official URLs).",
        parent=parent_node,
        critical=False
    )

    # 1) Name and National Park Location (with URL)
    name_loc_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_name_park_main",
        desc="Provides the lodge name and the associated National Park location, with an official URL supporting this identification/location.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_non_empty(lodge.name) and _non_empty(lodge.national_park),
        id=f"l{lodge_idx}_name_park_value_provided",
        desc="Name & park values are provided.",
        parent=name_loc_block,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_location),
        id=f"l{lodge_idx}_name_park_url_provided",
        desc="Name & park identification has at least one official URL provided.",
        parent=name_loc_block,
        critical=True
    )
    name_loc_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_name_park_supported",
        desc="Name & park identification is supported by the cited official URL(s).",
        parent=name_loc_block,
        critical=True
    )
    claim_name_loc = f"The official page confirms that '{lodge.name}' is a lodge located in {_safe_park(lodge)}."
    await evaluator.verify(
        claim=claim_name_loc,
        node=name_loc_supported,
        sources=lodge.url_location,
        additional_instruction="Treat official sources as the lodge's own website, NPS site, or the authorized concessioner's site. Minor name variations are acceptable."
    )

    # 2) Within Park Boundaries (with URL)
    within_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_within_park_main",
        desc="Confirms the lodge is physically within the National Park boundary (not outside/nearby), with an official URL supporting this.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_within_park),
        id=f"l{lodge_idx}_within_park_url_provided",
        desc="Within-park confirmation has at least one official URL provided.",
        parent=within_block,
        critical=True
    )
    within_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_within_park_supported",
        desc="Within-park confirmation is supported by the cited official URL(s).",
        parent=within_block,
        critical=True
    )
    claim_within = f"The lodge '{lodge.name}' is located inside the boundaries of {_safe_park(lodge)}, not outside the park."
    await evaluator.verify(
        claim=claim_within,
        node=within_supported,
        sources=lodge.url_within_park,
        additional_instruction="Accept explicit statements like 'in-park lodging', or pages clearly listing this lodge as within park boundaries."
    )

    # 3) Authorized NPS Concessioner (with URL)
    conc_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_concessioner_main",
        desc="Confirms the lodge is operated by an authorized NPS concessioner, with an official URL supporting this.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_concessioner),
        id=f"l{lodge_idx}_concessioner_url_provided",
        desc="Concessioner claim has at least one official URL provided.",
        parent=conc_block,
        critical=True
    )
    conc_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_concessioner_supported",
        desc="Authorized NPS concessioner claim is supported by the cited official URL(s).",
        parent=conc_block,
        critical=True
    )
    claim_conc = "This lodge is operated by an authorized National Park Service concessioner."
    await evaluator.verify(
        claim=claim_conc,
        node=conc_supported,
        sources=lodge.url_concessioner,
        additional_instruction="The supporting page should be from NPS or the official concessioner and clearly indicate authorized NPS concession status for the property."
    )

    # 4) Open June–August (with URL)
    open_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_open_summer_main",
        desc="Confirms the lodge is open during summer months (June–August), with an official URL supporting this.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_open_summer),
        id=f"l{lodge_idx}_open_summer_url_provided",
        desc="Open-in-summer claim has at least one official URL provided.",
        parent=open_block,
        critical=True
    )
    open_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_open_summer_supported",
        desc="Open June–August is supported by the cited official URL(s).",
        parent=open_block,
        critical=True
    )
    claim_open = "The lodge is open during the summer months of June, July, and August."
    await evaluator.verify(
        claim=claim_open,
        node=open_supported,
        sources=lodge.url_open_summer,
        additional_instruction="Seasonal calendar, operating dates, or booking availability for June–August count as support if clearly stated."
    )

    # 5) Pet-friendly allows dogs (with URL)
    pet_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_pet_friendly_main",
        desc="Confirms availability of designated pet-friendly rooms/cabins that allow dogs, with an official URL supporting this.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_pet_friendly),
        id=f"l{lodge_idx}_pet_friendly_url_provided",
        desc="Pet-friendly (dogs allowed) claim has at least one official URL provided.",
        parent=pet_block,
        critical=True
    )
    pet_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_pet_friendly_supported",
        desc="Pet-friendly rooms/cabins allowing dogs are supported by the cited official URL(s).",
        parent=pet_block,
        critical=True
    )
    claim_pet = "Designated pet-friendly rooms or cabins that allow dogs are available at this lodge."
    await evaluator.verify(
        claim=claim_pet,
        node=pet_supported,
        sources=lodge.url_pet_friendly,
        additional_instruction="Look for 'pet-friendly rooms' or pet policy specifically allowing dogs."
    )

    # 6) Nightly pet fee amount (with URL)
    petfee_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_pet_fee_main",
        desc="Provides a stated nightly pet fee amount, with an official URL supporting the amount and that it is nightly.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_non_empty(lodge.pet_fee_amount),
        id=f"l{lodge_idx}_pet_fee_value_provided",
        desc="Pet fee amount value is provided.",
        parent=petfee_block,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_pet_fee),
        id=f"l{lodge_idx}_pet_fee_url_provided",
        desc="Pet fee amount has at least one official URL provided.",
        parent=petfee_block,
        critical=True
    )
    petfee_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_pet_fee_supported",
        desc="Nightly pet fee amount is supported by the cited official URL(s).",
        parent=petfee_block,
        critical=True
    )
    claim_pet_fee = f"The nightly pet fee amount is {lodge.pet_fee_amount}."
    await evaluator.verify(
        claim=claim_pet_fee,
        node=petfee_supported,
        sources=lodge.url_pet_fee,
        additional_instruction="Verify that the fee is charged per night (not per stay) and that the amount matches."
    )

    # 7) ADA accessible rooms available (with URL)
    ada_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_ada_rooms_main",
        desc="Confirms ADA-compliant/accessible guest rooms are available, with an official URL supporting this.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_ada_rooms),
        id=f"l{lodge_idx}_ada_rooms_url_provided",
        desc="Accessible rooms availability has at least one official URL provided.",
        parent=ada_block,
        critical=True
    )
    ada_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_ada_rooms_supported",
        desc="ADA-compliant/accessible rooms availability is supported by the cited official URL(s).",
        parent=ada_block,
        critical=True
    )
    claim_ada = "ADA-compliant or accessible guest rooms are available at this lodge."
    await evaluator.verify(
        claim=claim_ada,
        node=ada_supported,
        sources=lodge.url_ada_rooms,
        additional_instruction="Look for accessibility statements or room type listings showing accessible/ADA rooms."
    )

    # 8) Accessible bathroom features specified (with URL)
    bath_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_bathroom_features_main",
        desc="Specifies accessible bathroom features in accessible rooms (e.g., grab bars, roll-in showers, accessible tubs), with an official URL supporting these features.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_non_empty(lodge.bathroom_features),
        id=f"l{lodge_idx}_bathroom_features_value_provided",
        desc="One or more accessible bathroom features are specified.",
        parent=bath_block,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_bathroom_features),
        id=f"l{lodge_idx}_bathroom_features_url_provided",
        desc="Bathroom features claim has at least one official URL provided.",
        parent=bath_block,
        critical=True
    )
    bath_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_bathroom_features_supported",
        desc="Accessible bathroom features are supported by the cited official URL(s).",
        parent=bath_block,
        critical=True
    )
    claim_bath = f"The accessible room bathrooms include the following features: {lodge.bathroom_features}."
    await evaluator.verify(
        claim=claim_bath,
        node=bath_supported,
        sources=lodge.url_bathroom_features,
        additional_instruction="Verify explicit feature mentions such as grab bars, roll-in shower, or accessible tub."
    )

    # 9) Dining facility wheelchair accessible (with URL)
    dine_access_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_dining_accessible_main",
        desc="Confirms the lodge’s dining room/restaurant is wheelchair accessible, with an official URL supporting this.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_dining_wheelchair),
        id=f"l{lodge_idx}_dining_accessible_url_provided",
        desc="Dining wheelchair accessibility has at least one official URL provided.",
        parent=dine_access_block,
        critical=True
    )
    dine_access_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_dining_accessible_supported",
        desc="Dining wheelchair accessibility is supported by the cited official URL(s).",
        parent=dine_access_block,
        critical=True
    )
    claim_dine_access = "The lodge's dining room or on-site restaurant is wheelchair accessible."
    await evaluator.verify(
        claim=claim_dine_access,
        node=dine_access_supported,
        sources=lodge.url_dining_wheelchair,
        additional_instruction="Accessibility page or statement that public areas (including restaurant/dining room) are accessible suffices."
    )

    # 10) On-site restaurant or dining room (with URL)
    restaurant_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_restaurant_main",
        desc="Confirms an on-site restaurant or dining room exists, with an official URL supporting this.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_on_site_restaurant),
        id=f"l{lodge_idx}_restaurant_url_provided",
        desc="On-site restaurant/dining has at least one official URL provided.",
        parent=restaurant_block,
        critical=True
    )
    restaurant_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_restaurant_supported",
        desc="On-site restaurant/dining room is supported by the cited official URL(s).",
        parent=restaurant_block,
        critical=True
    )
    claim_restaurant = "The lodge has an on-site restaurant or dining room."
    await evaluator.verify(
        claim=claim_restaurant,
        node=restaurant_supported,
        sources=lodge.url_on_site_restaurant,
        additional_instruction="Menus or dining pages count as support for on-site dining."
    )

    # 11) Dinner reservation policy (with URL)
    dinner_res_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_dinner_policy_main",
        desc="States whether dinner reservations are required or recommended, with an official URL supporting the policy.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_non_empty(lodge.dinner_reservation_policy),
        id=f"l{lodge_idx}_dinner_policy_value_provided",
        desc="Dinner reservation policy value is provided (e.g., required/recommended/not required).",
        parent=dinner_res_block,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_dinner_reservation_policy),
        id=f"l{lodge_idx}_dinner_policy_url_provided",
        desc="Dinner reservation policy has at least one official URL provided.",
        parent=dinner_res_block,
        critical=True
    )
    dinner_res_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_dinner_policy_supported",
        desc="Dinner reservation policy is supported by the cited official URL(s).",
        parent=dinner_res_block,
        critical=True
    )
    claim_dinner_policy = f"The dinner reservation policy is: {lodge.dinner_reservation_policy}."
    await evaluator.verify(
        claim=claim_dinner_policy,
        node=dinner_res_supported,
        sources=lodge.url_dinner_reservation_policy,
        additional_instruction="Look for phrasing like 'reservations required' or 'reservations recommended' specific to dinner service."
    )

    # 12) Gift shop on-site (with URL)
    gift_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_gift_shop_main",
        desc="Confirms an on-site gift shop, with an official URL supporting this.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_gift_shop),
        id=f"l{lodge_idx}_gift_shop_url_provided",
        desc="Gift shop has at least one official URL provided.",
        parent=gift_block,
        critical=True
    )
    gift_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_gift_shop_supported",
        desc="On-site gift shop is supported by the cited official URL(s).",
        parent=gift_block,
        critical=True
    )
    claim_gift = "The lodge has an on-site gift shop."
    await evaluator.verify(
        claim=claim_gift,
        node=gift_supported,
        sources=lodge.url_gift_shop,
        additional_instruction="Any official amenities page mentioning gift shop is acceptable."
    )

    # 13) WiFi available at least in common areas (with URL)
    wifi_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_wifi_main",
        desc="Confirms WiFi availability at least in common areas/lobby, with an official URL supporting this.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_wifi),
        id=f"l{lodge_idx}_wifi_url_provided",
        desc="WiFi claim has at least one official URL provided.",
        parent=wifi_block,
        critical=True
    )
    wifi_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_wifi_supported",
        desc="WiFi availability (at least lobby/common areas) is supported by the cited official URL(s).",
        parent=wifi_block,
        critical=True
    )
    wifi_scope_text = lodge.wifi_scope or "WiFi is available at least in common areas or lobby."
    await evaluator.verify(
        claim=wifi_scope_text,
        node=wifi_supported,
        sources=lodge.url_wifi,
        additional_instruction="It is sufficient if the page states WiFi is available in public spaces, the lobby, or anywhere on property (rooms also acceptable)."
    )

    # 14) Check-in time 4:00 PM or later (with URL)
    checkin_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_checkin_main",
        desc="Provides check-in time and confirms it is 4:00 PM or later, with an official URL supporting this.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_non_empty(lodge.check_in_time),
        id=f"l{lodge_idx}_checkin_value_provided",
        desc="Check-in time value is provided.",
        parent=checkin_block,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_checkin_time),
        id=f"l{lodge_idx}_checkin_url_provided",
        desc="Check-in time has at least one official URL provided.",
        parent=checkin_block,
        critical=True
    )
    checkin_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_checkin_supported",
        desc="Check-in time being 4:00 PM or later is supported by the cited official URL(s).",
        parent=checkin_block,
        critical=True
    )
    claim_checkin = "The check-in time is 4:00 PM or later (e.g., 4:00 PM, 5:00 PM, or later)."
    await evaluator.verify(
        claim=claim_checkin,
        node=checkin_supported,
        sources=lodge.url_checkin_time,
        additional_instruction="Fail if the page shows a time earlier than 4:00 PM; pass if 4:00 PM or any later time is clearly stated."
    )

    # 15) Official reservation method (with URL)
    reserve_block = evaluator.add_parallel(
        id=f"l{lodge_idx}_reservation_method_main",
        desc="Provides an official reservation method (website and/or phone number), with an official URL supporting this method.",
        parent=lodge_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_urls_present(lodge.url_reservation_method),
        id=f"l{lodge_idx}_reservation_method_url_provided",
        desc="Reservation method has at least one official URL provided.",
        parent=reserve_block,
        critical=True
    )
    reserve_supported = evaluator.add_leaf(
        id=f"l{lodge_idx}_reservation_method_supported",
        desc="Official reservation method is supported by the cited official URL(s).",
        parent=reserve_block,
        critical=True
    )
    reserve_text = lodge.reservation_method or "Reservations can be made via the official website or by phone."
    extra = "If a phone number is claimed, the page should show that phone number for reservations; otherwise confirm 'Book online' or 'Reserve' is offered on the official site."
    await evaluator.verify(
        claim=reserve_text,
        node=reserve_supported,
        sources=lodge.url_reservation_method,
        additional_instruction=extra
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
    Evaluate an answer for the two in-park lodges task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Root aggregates in parallel; critical children can gate
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

    # Extract structured lodge info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_lodges(),
        template_class=LodgesExtraction,
        extraction_name="lodges_extraction"
    )

    # Record custom info about extraction
    provided_named = [l for l in extraction.lodges if _non_empty(l.name)]
    evaluator.add_custom_info(
        info={
            "total_lodges_extracted": len(extraction.lodges),
            "lodges_with_names": len(provided_named),
            "lodges_names": [l.name for l in provided_named]
        },
        info_type="extraction_stats"
    )

    # Top-level critical check: exactly two distinct lodge options provided
    names_norm = [_norm_name(l.name) for l in extraction.lodges if _non_empty(l.name)]
    unique_names_norm = set(names_norm)
    exactly_two = (len(names_norm) == 2) and (len(unique_names_norm) == 2)

    evaluator.add_custom_node(
        result=exactly_two,
        id="Provide_Exactly_Two_Lodges",
        desc="Response provides exactly two distinct lodge options (not one, not more than two).",
        parent=root,
        critical=True
    )

    # Verify only the first two lodges with names (if provided)
    lodges_to_check: List[Lodge] = provided_named[:2]
    # If fewer than 2, pad with empty placeholders to keep the tree shape stable
    while len(lodges_to_check) < 2:
        lodges_to_check.append(Lodge())

    # Lodge 1
    await verify_lodge(evaluator, root, lodges_to_check[0], 0)
    # Lodge 2
    await verify_lodge(evaluator, root, lodges_to_check[1], 1)

    return evaluator.get_summary()