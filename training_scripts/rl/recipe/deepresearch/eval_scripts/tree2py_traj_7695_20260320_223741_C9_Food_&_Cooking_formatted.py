import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nc_holiday_2025"
TASK_DESCRIPTION = """
In North Carolina, identify 4 different establishments that meet the following specific requirements for the 2025 holiday season:

1. A restaurant chain location that:
   - Operates 24 hours a day, 365 days a year
   - Is confirmed to be open on both Thanksgiving Day (November 27, 2025) and Christmas Day (December 25, 2025)
   - Serves breakfast items
   - Has at least one location in North Carolina
   - Provide the chain name and a URL reference confirming its 24/7/365 operations or holiday hours policy

2. A convenience store chain location that:
   - Operates 24 hours a day, 7 days a week
   - Is confirmed to be open on both Thanksgiving Day (November 27, 2025) and Christmas Day (December 25, 2025)
   - Provides gasoline/fuel services
   - Sells food items and convenience store products
   - Has multiple locations in North Carolina
   - Provide the chain name and a URL reference confirming its 24/7 operations and/or holiday hours

3. A specific 24-hour pharmacy location that:
   - Operates 24 hours per day
   - Remains open on Thanksgiving Day (November 27, 2025) as a 24-hour location
   - Is from either the CVS or Walgreens pharmacy chain
   - Provide the complete street address, city, and state (North Carolina)
   - Provide a URL reference confirming the 24-hour pharmacy location details or store hours

4. A grocery store chain location that:
   - Is confirmed to be open on Thanksgiving Day (November 27, 2025)
   - Provide the specific operating hours for Thanksgiving Day 2025
   - Is NOT one of the following chains: Walmart, Target, Costco, Aldi, or Trader Joe's
   - Is classified as a grocery store or supermarket
   - Has locations in North Carolina
   - Provide the chain name and a URL reference confirming its Thanksgiving 2025 hours

For each establishment, provide all required information including names, addresses (where applicable), and supporting URL references.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RestaurantChain(BaseModel):
    chain_name: Optional[str] = None
    support_urls: List[str] = Field(default_factory=list)


class ConvenienceStoreChain(BaseModel):
    chain_name: Optional[str] = None
    support_urls: List[str] = Field(default_factory=list)


class PharmacyLocation(BaseModel):
    chain_name: Optional[str] = None  # Expect CVS or Walgreens
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    support_urls: List[str] = Field(default_factory=list)


class GroceryChain(BaseModel):
    chain_name: Optional[str] = None
    # If the answer stated explicit Thanksgiving hours, capture the text (e.g., "Open 7am–3pm")
    thanksgiving_2025_hours_text: Optional[str] = None
    support_urls: List[str] = Field(default_factory=list)


class HolidayEstablishments(BaseModel):
    restaurant: Optional[RestaurantChain] = None
    convenience_store: Optional[ConvenienceStoreChain] = None
    pharmacy: Optional[PharmacyLocation] = None
    grocery: Optional[GroceryChain] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_establishments() -> str:
    return """
Extract the four requested North Carolina establishments as structured JSON.

For each category below, extract exactly the fields listed (do not invent information). If a value is not present in the answer, return null (or [] for lists).

1) restaurant:
- chain_name: The restaurant chain name mentioned in the answer.
- support_urls: All URLs in the answer that are intended to support 24/7/365 operations, holiday-hours policy (Thanksgiving/Christmas), menu/breakfast availability, or NC presence (store locator or location pages). Include every relevant URL cited.

2) convenience_store:
- chain_name: The convenience store chain name mentioned in the answer.
- support_urls: All URLs in the answer that are intended to support 24/7 operations and/or holiday hours, gasoline/fuel services, food/convenience products, and NC presence (preferably indicating multiple NC locations). Include every relevant URL cited.

3) pharmacy:
- chain_name: The pharmacy chain name mentioned in the answer (should be CVS or Walgreens if the answer is valid).
- street_address: The full street address of the specific 24-hour pharmacy location provided in the answer (e.g., "123 Main St").
- city: The city for that location.
- state: The state for that location (should be "NC" or "North Carolina" if valid).
- support_urls: All URLs in the answer that are intended to support the store details/hours page and 24-hour status and/or Thanksgiving 2025 status for this exact location. Include every relevant URL cited.

4) grocery:
- chain_name: The grocery store chain name mentioned in the answer (must not be Walmart, Target, Costco, Aldi, or Trader Joe's).
- thanksgiving_2025_hours_text: The specific Thanksgiving Day 2025 operating hours stated in the answer text for this chain (e.g., "Open 6am–2pm"). If not explicitly provided in the answer text, set this to null.
- support_urls: All URLs in the answer that are intended to support Thanksgiving Day 2025 hours for this chain, grocery/supermarket classification, and NC presence. Include every relevant URL cited.

Return the final JSON object with the following top-level fields:
{
  "restaurant": { ... },
  "convenience_store": { ... },
  "pharmacy": { ... },
  "grocery": { ... }
}
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def _is_nc_state_str(s: Optional[str]) -> bool:
    if not s:
        return False
    s_norm = s.strip().lower()
    return s_norm in {"nc", "north carolina"}


# --------------------------------------------------------------------------- #
# Verification logic per establishment                                        #
# --------------------------------------------------------------------------- #
async def verify_restaurant(evaluator: Evaluator, parent_node, data: Optional[RestaurantChain]) -> None:
    node = evaluator.add_parallel(
        id="establishment_1",
        desc="A restaurant chain location in North Carolina operating 24/7/365 with breakfast service",
        parent=parent_node,
        critical=False,
    )

    chain_name = (data.chain_name if data else None) or ""
    urls = (data.support_urls if data else []) or []

    # Critical existence: chain name provided
    evaluator.add_custom_node(
        result=bool(chain_name.strip()),
        id="est1_chain_name",
        desc="Restaurant chain name is provided",
        parent=node,
        critical=True
    )

    # 1) URL reference confirms 24/7/365 operations or holiday-hours policy (verify first to gate others)
    url_ref_leaf = evaluator.add_leaf(
        id="est1_url_reference",
        desc="URL reference confirms 24/7/365 operations or holiday hours policy",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these pages clearly states that the restaurant chain either operates 24 hours a day, 365 days a year OR provides an official holiday-hours policy (including Thanksgiving and/or Christmas).",
        node=url_ref_leaf,
        sources=urls,
        additional_instruction="Accept official brand pages, store-locator pages with policy statements, or credible news/press pages. Generic third-party blogs without concrete chain policy should not be considered sufficient."
    )

    # 2) 24/7/365 operation
    op_leaf = evaluator.add_leaf(
        id="est1_247365_operation",
        desc="Restaurant operates 24 hours a day, 365 days a year",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The restaurant chain '{chain_name}' operates 24 hours a day, 365 days a year (i.e., open 24/7/365).",
        node=op_leaf,
        sources=urls,
        additional_instruction="Look for explicit phrases like 'open 24 hours', '24/7', and '365 days a year'. It can be a chain-level policy or a widely consistent practice stated by the brand."
    )

    # 3) Open on Thanksgiving Day 2025
    tg_leaf = evaluator.add_leaf(
        id="est1_thanksgiving_open",
        desc="Confirmed to be open on Thanksgiving Day (November 27, 2025)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The restaurant chain '{chain_name}' is open on Thanksgiving Day 2025 (November 27, 2025).",
        node=tg_leaf,
        sources=urls,
        additional_instruction="Evidence should indicate Thanksgiving Day opening. Accept chain policy pages that state they remain open on holidays including Thanksgiving. The year must be 2025 or a timeless policy explicitly including Thanksgiving."
    )

    # 4) Open on Christmas Day 2025
    xmas_leaf = evaluator.add_leaf(
        id="est1_christmas_open",
        desc="Confirmed to be open on Christmas Day (December 25, 2025)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The restaurant chain '{chain_name}' is open on Christmas Day 2025 (December 25, 2025).",
        node=xmas_leaf,
        sources=urls,
        additional_instruction="Evidence should indicate Christmas Day opening. Accept chain policy pages that state they remain open on holidays including Christmas. The year must be 2025 or a timeless policy explicitly including Christmas."
    )

    # 5) Serves breakfast
    bf_leaf = evaluator.add_leaf(
        id="est1_serves_breakfast",
        desc="Restaurant serves breakfast items",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The restaurant chain '{chain_name}' serves breakfast or has a breakfast menu.",
        node=bf_leaf,
        sources=urls,
        additional_instruction="Accept official menu pages or credible references indicating breakfast availability."
    )

    # 6) Has at least one NC location
    nc_leaf = evaluator.add_leaf(
        id="est1_nc_location",
        desc="Has at least one location in North Carolina",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The restaurant chain '{chain_name}' has at least one location in North Carolina.",
        node=nc_leaf,
        sources=urls,
        additional_instruction="Accept store locator results/pages filtered for North Carolina, or an official list showing NC locations."
    )


async def verify_convenience_store(evaluator: Evaluator, parent_node, data: Optional[ConvenienceStoreChain]) -> None:
    node = evaluator.add_parallel(
        id="establishment_2",
        desc="A convenience store chain with gas in North Carolina operating 24/7 on holidays",
        parent=parent_node,
        critical=False,
    )

    chain_name = (data.chain_name if data else None) or ""
    urls = (data.support_urls if data else []) or []

    # Critical existence: chain name provided
    evaluator.add_custom_node(
        result=bool(chain_name.strip()),
        id="est2_chain_name",
        desc="Convenience store chain name is provided",
        parent=node,
        critical=True
    )

    # 1) URL reference confirms 24/7 ops and/or holiday hours (verify first to gate others)
    url_ref_leaf = evaluator.add_leaf(
        id="est2_url_reference",
        desc="URL reference confirms 24/7 operations and/or holiday hours",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these pages clearly states that the convenience store chain operates 24 hours a day and/or articulates official holiday-hours (including Thanksgiving or Christmas).",
        node=url_ref_leaf,
        sources=urls,
        additional_instruction="Prefer official chain pages or credible retailer information. The content should clearly describe 24/7 operations or holiday hours."
    )

    # 2) 24/7 operation
    op_leaf = evaluator.add_leaf(
        id="est2_247_operation",
        desc="Store operates 24 hours a day, 7 days a week",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The convenience store chain '{chain_name}' operates 24/7.",
        node=op_leaf,
        sources=urls,
        additional_instruction="Look for explicit chain-level 24/7 operation statements."
    )

    # 3) Thanksgiving Day open
    tg_leaf = evaluator.add_leaf(
        id="est2_thanksgiving_open",
        desc="Confirmed to be open on Thanksgiving Day (November 27, 2025)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The convenience store chain '{chain_name}' is open on Thanksgiving Day 2025 (November 27, 2025).",
        node=tg_leaf,
        sources=urls,
        additional_instruction="Holiday policies that explicitly include Thanksgiving are acceptable, or specific 2025 hours pages."
    )

    # 4) Christmas Day open
    xmas_leaf = evaluator.add_leaf(
        id="est2_christmas_open",
        desc="Confirmed to be open on Christmas Day (December 25, 2025)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The convenience store chain '{chain_name}' is open on Christmas Day 2025 (December 25, 2025).",
        node=xmas_leaf,
        sources=urls,
        additional_instruction="Holiday policies that explicitly include Christmas are acceptable, or specific 2025 hours pages."
    )

    # 5) Gasoline/fuel services
    gas_leaf = evaluator.add_leaf(
        id="est2_gas_services",
        desc="Provides gasoline/fuel services",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The convenience store chain '{chain_name}' provides gasoline/fuel services at its locations.",
        node=gas_leaf,
        sources=urls,
        additional_instruction="Accept official brand/service pages indicating fuel/pumps at stores."
    )

    # 6) Sells food items and convenience products
    food_leaf = evaluator.add_leaf(
        id="est2_food_items",
        desc="Sells food items and convenience store products",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The convenience store chain '{chain_name}' sells food items and convenience store products.",
        node=food_leaf,
        sources=urls,
        additional_instruction="Accept official product category pages, menus, or store overview pages."
    )

    # 7) Multiple NC locations
    multi_nc_leaf = evaluator.add_leaf(
        id="est2_multiple_nc_locations",
        desc="Has multiple locations in North Carolina",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The convenience store chain '{chain_name}' has multiple (two or more) locations in North Carolina.",
        node=multi_nc_leaf,
        sources=urls,
        additional_instruction="Accept store locator pages indicating more than one NC location (e.g., 'locations in North Carolina' with counts or multiple entries)."
    )


async def verify_pharmacy(evaluator: Evaluator, parent_node, data: Optional[PharmacyLocation]) -> None:
    node = evaluator.add_parallel(
        id="establishment_3",
        desc="A specific 24-hour pharmacy location in North Carolina open on Thanksgiving",
        parent=parent_node,
        critical=False,
    )

    chain_name = (data.chain_name if data else None) or ""
    address = (data.street_address if data else None) or ""
    city = (data.city if data else None) or ""
    state = (data.state if data else None) or ""
    urls = (data.support_urls if data else []) or []

    # Critical existence checks for address components
    evaluator.add_custom_node(
        result=bool(address.strip()),
        id="est3_street_address",
        desc="Complete street address is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(city.strip()),
        id="est3_city",
        desc="City in North Carolina is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nc_state_str(state),
        id="est3_state",
        desc="State (North Carolina) is specified",
        parent=node,
        critical=True
    )

    # 1) Pharmacy chain is CVS or Walgreens (simple verification)
    chain_leaf = evaluator.add_leaf(
        id="est3_pharmacy_chain",
        desc="Pharmacy chain is either CVS or Walgreens",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The pharmacy chain '{chain_name}' is either CVS or Walgreens.",
        node=chain_leaf,
        additional_instruction="Allow minor naming variants such as 'CVS Pharmacy' or 'Walgreens Pharmacy'."
    )

    # 2) URL reference confirms store details/hours for this specific location (verify early to gate others)
    url_ref_leaf = evaluator.add_leaf(
        id="est3_url_reference",
        desc="URL reference confirms 24-hour pharmacy location details or store hours",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one of these pages is an official store details or hours page (or an authoritative listing) for this specific location at '{address}, {city}, {state}', and it includes store hours and/or 24-hour designation.",
        node=url_ref_leaf,
        sources=urls,
        additional_instruction="Prefer official CVS/Walgreens store pages. Third-party listings are acceptable only if authoritative and clearly match the exact location."
    )

    # 3) 24-hour operation (specific location)
    op_leaf = evaluator.add_leaf(
        id="est3_24hour_operation",
        desc="Pharmacy location operates 24 hours per day",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The specific pharmacy location at '{address}, {city}, {state}' operates 24 hours per day.",
        node=op_leaf,
        sources=urls,
        additional_instruction="The page should clearly state 'Open 24 hours' (or equivalent) for this exact location."
    )

    # 4) Thanksgiving Day open (as a 24-hour location)
    tg_leaf = evaluator.add_leaf(
        id="est3_thanksgiving_open",
        desc="24-hour location remains open on Thanksgiving Day (November 27, 2025)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The specific 24-hour pharmacy location at '{address}, {city}, {state}' remains open on Thanksgiving Day 2025 (November 27, 2025).",
        node=tg_leaf,
        sources=urls,
        additional_instruction="Accept explicit Thanksgiving 2025 hours OR a chain policy stating 24-hour stores remain open on holidays including Thanksgiving. The page should imply no closure on that day."
    )


async def verify_grocery(evaluator: Evaluator, parent_node, data: Optional[GroceryChain]) -> None:
    node = evaluator.add_parallel(
        id="establishment_4",
        desc="A grocery store chain in North Carolina open on Thanksgiving 2025 with hours",
        parent=parent_node,
        critical=False,
    )

    chain_name = (data.chain_name if data else None) or ""
    tg_hours_text = (data.thanksgiving_2025_hours_text if data else None) or ""
    urls = (data.support_urls if data else []) or []

    # Critical existence: chain name provided
    evaluator.add_custom_node(
        result=bool(chain_name.strip()),
        id="est4_chain_name",
        desc="Grocery chain name is provided",
        parent=node,
        critical=True
    )

    # 1) Chain is NOT one of the excluded list (simple verify)
    not_excluded_leaf = evaluator.add_leaf(
        id="est4_not_excluded_chain",
        desc="Store is NOT Walmart, Target, Costco, Aldi, or Trader Joe's",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The grocery chain '{chain_name}' is not Walmart, Target, Costco, Aldi, or Trader Joe's.",
        node=not_excluded_leaf,
        additional_instruction="This is a simple name check; allow minor formatting differences."
    )

    # 2) URL reference confirms Thanksgiving 2025 hours (verify early to gate others)
    url_ref_leaf = evaluator.add_leaf(
        id="est4_url_reference",
        desc="URL reference confirms Thanksgiving 2025 hours for this chain",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one of these pages clearly states Thanksgiving Day 2025 (November 27, 2025) operating hours for the grocery chain '{chain_name}'.",
        node=url_ref_leaf,
        sources=urls,
        additional_instruction="The page should be an official chain communication or a highly credible source listing Thanksgiving 2025 hours specifically."
    )

    # 3) Confirm open on Thanksgiving Day 2025
    tg_open_leaf = evaluator.add_leaf(
        id="est4_thanksgiving_open",
        desc="Confirmed to be open on Thanksgiving Day (November 27, 2025)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The grocery chain '{chain_name}' is open on Thanksgiving Day 2025 (November 27, 2025).",
        node=tg_open_leaf,
        sources=urls,
        additional_instruction="Evidence should indicate that stores are open (not closed) on Thanksgiving Day 2025."
    )

    # 4) Specific operating hours for Thanksgiving Day 2025 are provided (page-level verification)
    tg_hours_leaf = evaluator.add_leaf(
        id="est4_thanksgiving_hours",
        desc="Specific operating hours for Thanksgiving Day 2025 are provided",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page(s) give specific opening/closing times for '{chain_name}' on Thanksgiving Day 2025 (e.g., 'Open 7am–3pm').",
        node=tg_hours_leaf,
        sources=urls,
        additional_instruction="Look for explicit time ranges for Thanksgiving Day 2025, not just a generic 'open' statement."
    )

    # 5) Grocery classification
    grocery_cls_leaf = evaluator.add_leaf(
        id="est4_grocery_classification",
        desc="Establishment is classified as a grocery store or supermarket",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{chain_name}' is a grocery store or supermarket brand (i.e., its primary business is food/grocery retail).",
        node=grocery_cls_leaf,
        sources=urls,
        additional_instruction="Prefer official 'About' pages or credible sources describing the retailer as a grocery store or supermarket."
    )

    # 6) Store has locations in North Carolina
    nc_leaf = evaluator.add_leaf(
        id="est4_nc_location",
        desc="Store has locations in North Carolina",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The grocery chain '{chain_name}' has locations in North Carolina.",
        node=nc_leaf,
        sources=urls,
        additional_instruction="Accept store locator pages filtered for North Carolina or credible listings showing NC stores."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator (root should be non-critical parallel to allow partial credit across establishments)
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

    # Extract structured info for all four establishments
    extracted = await evaluator.extract(
        prompt=prompt_extract_establishments(),
        template_class=HolidayEstablishments,
        extraction_name="holiday_establishments_extraction"
    )

    # Optionally record exclusion ground-truth for transparency
    evaluator.add_ground_truth({
        "excluded_grocery_chains": ["Walmart", "Target", "Costco", "Aldi", "Trader Joe's"],
        "holiday_dates_2025": {
            "Thanksgiving": "November 27, 2025",
            "Christmas": "December 25, 2025"
        }
    })

    # Build and verify sub-trees for each establishment
    await verify_restaurant(evaluator, root, extracted.restaurant)
    await verify_convenience_store(evaluator, root, extracted.convenience_store)
    await verify_pharmacy(evaluator, root, extracted.pharmacy)
    await verify_grocery(evaluator, root, extracted.grocery)

    return evaluator.get_summary()