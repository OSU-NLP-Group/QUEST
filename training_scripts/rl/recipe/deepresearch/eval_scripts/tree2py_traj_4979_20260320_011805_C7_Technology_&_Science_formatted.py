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
TASK_ID = "columbus_best_buy_store_eval"
TASK_DESCRIPTION = (
    "Identify a Best Buy retail store location in Columbus, Ohio that offers on-site Geek Squad services and is open on "
    "weekday evenings past 5:00 PM. For this store location, provide the following comprehensive information: "
    "(1) The complete street address, (2) The store's contact phone number, (3) The store's weekday operating hours "
    "(both opening and closing times), (4) Confirmation that Geek Squad services are available at this location, "
    "(5) Confirmation that same-day screen repair services are offered, (6) Confirmation that Apple TV 4K streaming "
    "devices are available for purchase, (7) The retail price of the Apple TV 4K Wi-Fi model (64GB), "
    "(8) The retail price of the Apple TV 4K Wi-Fi + Ethernet model (128GB), "
    "(9) The Geek Squad service fee for mobile device screen repairs, "
    "(10) The Geek Squad service fee for other mobile device damage repairs, "
    "(11) The starting price for Geek Squad smart home installation services, "
    "(12) Verification that the store location has T-Mobile 5G network coverage, "
    "(13) Confirmation that Apple-trained Geek Squad agents are available at this location. "
    "Additionally, provide reference URLs from official sources (such as Best Buy's website, Google Maps, or carrier "
    "coverage maps) that verify the store location details and service pricing information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreExtraction(BaseModel):
    # Core store details
    store_name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    weekday_open: Optional[str] = None  # e.g., "10:00 AM"
    weekday_close: Optional[str] = None  # e.g., "8:00 PM"

    # Availability confirmations (as booleans if present in answer)
    geek_squad_available: Optional[bool] = None
    same_day_screen_repair: Optional[bool] = None
    apple_tv_available: Optional[bool] = None
    tmobile_5g_coverage: Optional[bool] = None
    apple_trained_geek_squad: Optional[bool] = None

    # Prices (keep as strings to allow variants like "$129", "129", "129.00 USD")
    apple_tv_wifi_64gb_price: Optional[str] = None
    apple_tv_ethernet_128gb_price: Optional[str] = None
    screen_repair_fee: Optional[str] = None
    other_damage_fee: Optional[str] = None
    install_base_price: Optional[str] = None

    # URLs for verification
    store_urls: List[str] = Field(default_factory=list)                 # BestBuy store page or Google Maps listing
    service_pricing_urls: List[str] = Field(default_factory=list)       # Geek Squad pricing/availability pages
    apple_tv_urls: List[str] = Field(default_factory=list)              # Apple TV 4K product page(s) on BestBuy.com
    tmobile_coverage_urls: List[str] = Field(default_factory=list)      # T-Mobile coverage map URL(s)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_store_info() -> str:
    return """
    Extract from the answer a single Best Buy retail store location in Columbus, Ohio that meets the task requirements.
    If multiple stores are mentioned, extract only the first one that is in Columbus, OH.

    Required fields to extract (use null if missing):
    - store_name: The store's name as provided (e.g., "Best Buy Easton").
    - address: Full street address including city and state (should be Columbus, OH).
    - phone: Store contact phone number.
    - weekday_open: Store opening time on weekdays (Mon–Fri), e.g., "10:00 AM".
    - weekday_close: Store closing time on weekdays (Mon–Fri), e.g., "8:00 PM".

    Availability confirmations (booleans; true/false if explicitly stated, otherwise null):
    - geek_squad_available
    - same_day_screen_repair
    - apple_tv_available
    - tmobile_5g_coverage
    - apple_trained_geek_squad

    Prices (as strings, exactly as written in the answer if present, otherwise null):
    - apple_tv_wifi_64gb_price
    - apple_tv_ethernet_128gb_price
    - screen_repair_fee
    - other_damage_fee
    - install_base_price

    Verification URLs (extract actual URLs only; if not present, return empty array):
    - store_urls: Official Best Buy store locator page and/or Google Maps URL for the store (include all provided).
    - service_pricing_urls: Official Best Buy/Geek Squad pages that mention service pricing and availability (e.g., repair fees, installation services).
    - apple_tv_urls: Best Buy product page(s) for Apple TV 4K that show availability and pricing.
    - tmobile_coverage_urls: Official T-Mobile coverage map URL(s) relevant to the store location.

    IMPORTANT:
    - Return exactly the information as it appears in the answer; do not invent or infer.
    - Only include URLs explicitly present in the answer text (including Markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _flatten_unique(*lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        for u in lst or []:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    result.append(uu)
    return result


def _nz(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Build verification tree and run checks                                      #
# --------------------------------------------------------------------------- #
async def _build_and_verify(evaluator: Evaluator, extracted: StoreExtraction):
    # Create a critical aggregator node to mirror rubric root
    col_node = evaluator.add_parallel(
        id="Columbus_Best_Buy_Store_Information",
        desc="Verify that the provided Best Buy store location in Columbus, Ohio meets all specified criteria and that all required information is accurately provided",
        parent=evaluator.root,
        critical=True
    )

    # Consolidate sources for convenience
    store_sources = extracted.store_urls
    service_sources = extracted.service_pricing_urls
    apple_tv_sources = extracted.apple_tv_urls
    tmo_sources = extracted.tmobile_coverage_urls

    # 1) Store Address
    leaf_addr = evaluator.add_leaf(
        id="Store_Address",
        desc="The complete street address of the Best Buy store location in Columbus, Ohio is provided",
        parent=col_node,
        critical=True
    )
    claim_addr = (
        f"The official store page or Google Maps listing confirms this Best Buy store's address as '{_nz(extracted.address)}' "
        f"in Columbus, OH (or Columbus, Ohio)."
    )
    await evaluator.verify(
        claim=claim_addr,
        node=leaf_addr,
        sources=store_sources,
        additional_instruction="Verify the address text matches (allow minor formatting differences) and that the city is Columbus, OH."
    )

    # 2) Store Phone Number
    leaf_phone = evaluator.add_leaf(
        id="Store_Phone_Number",
        desc="A valid contact phone number for the Best Buy store is provided",
        parent=col_node,
        critical=True
    )
    claim_phone = f"The store's contact phone number is '{_nz(extracted.phone)}'."
    await evaluator.verify(
        claim=claim_phone,
        node=leaf_phone,
        sources=store_sources,
        additional_instruction="Confirm the page lists the same phone number (ignore punctuation or spacing differences)."
    )

    # 3) Weekday Opening Time
    leaf_open = evaluator.add_leaf(
        id="Weekday_Hours_Start",
        desc="The store's opening time on weekdays (Monday-Friday) is provided",
        parent=col_node,
        critical=True
    )
    claim_open = f"On weekdays (Mon–Fri), the store opens at '{_nz(extracted.weekday_open)}'."
    await evaluator.verify(
        claim=claim_open,
        node=leaf_open,
        sources=store_sources,
        additional_instruction="If hours vary by day, use the general weekday opening time shown on the store page."
    )

    # 4) Weekday Closing Time and 'past 5 PM' requirement
    leaf_close = evaluator.add_leaf(
        id="Weekday_Hours_End",
        desc="The store's closing time on weekdays is provided and shows the store is open past 5:00 PM",
        parent=col_node,
        critical=True
    )
    claim_close = (
        f"On weekdays (Mon–Fri), the store closes at '{_nz(extracted.weekday_close)}', which is later than 5:00 PM local time."
    )
    await evaluator.verify(
        claim=claim_close,
        node=leaf_close,
        sources=store_sources,
        additional_instruction="Confirm the weekday closing time is after 5:00 PM (17:00). Treat exactly 5:00 PM as not 'past 5:00 PM'."
    )

    # 5) Geek Squad availability at this location
    leaf_geek = evaluator.add_leaf(
        id="Geek_Squad_Availability",
        desc="Confirmation that Geek Squad services are available at this store location",
        parent=col_node,
        critical=True
    )
    claim_geek = "This Best Buy store location provides on-site Geek Squad services."
    await evaluator.verify(
        claim=claim_geek,
        node=leaf_geek,
        sources=store_sources,
        additional_instruction="Look for 'Geek Squad' or 'in-store services' indicators for this specific store."
    )

    # 6) Same-day screen repair service availability at this location
    leaf_same_day = evaluator.add_leaf(
        id="Same_Day_Repair_Service",
        desc="Confirmation that same-day screen repair services are offered at this location",
        parent=col_node,
        critical=True
    )
    claim_same_day = "Same-day screen repair services are offered at this Best Buy store location."
    await evaluator.verify(
        claim=claim_same_day,
        node=leaf_same_day,
        sources=_flatten_unique(store_sources, service_sources),
        additional_instruction="Evidence can be on the store page or an official Geek Squad service page that explicitly states same-day screen repairs and indicates availability at this store."
    )

    # 7) Apple TV 4K devices available for purchase
    leaf_appletv_stock = evaluator.add_leaf(
        id="Apple_TV_4K_Stock",
        desc="Confirmation that Apple TV 4K devices are available for purchase at this store",
        parent=col_node,
        critical=True
    )
    claim_appletv_stock = "Apple TV 4K streaming devices are available for purchase from Best Buy."
    await evaluator.verify(
        claim=claim_appletv_stock,
        node=leaf_appletv_stock,
        sources=apple_tv_sources,
        additional_instruction="Use Best Buy product pages to verify sellable Apple TV 4K items (not open-box only)."
    )

    # 8) Apple TV 4K Wi‑Fi 64GB price = $129
    leaf_price_wifi = evaluator.add_leaf(
        id="Apple_TV_WiFi_Price",
        desc="The price of the Apple TV 4K Wi-Fi model (64GB) is stated as $129",
        parent=col_node,
        critical=True
    )
    claim_price_wifi = "The Apple TV 4K (Wi‑Fi, 64GB) is priced at $129."
    await evaluator.verify(
        claim=claim_price_wifi,
        node=leaf_price_wifi,
        sources=apple_tv_sources,
        additional_instruction="Check the current sell price on the official Best Buy product page (ignore open-box or trade-in)."
    )

    # 9) Apple TV 4K Wi‑Fi + Ethernet 128GB price = $149
    leaf_price_eth = evaluator.add_leaf(
        id="Apple_TV_Ethernet_Price",
        desc="The price of the Apple TV 4K Wi-Fi + Ethernet model (128GB) is stated as $149",
        parent=col_node,
        critical=True
    )
    claim_price_eth = "The Apple TV 4K (Wi‑Fi + Ethernet, 128GB) is priced at $149."
    await evaluator.verify(
        claim=claim_price_eth,
        node=leaf_price_eth,
        sources=apple_tv_sources,
        additional_instruction="Verify price on the Best Buy product page; disregard memberships, financing, or open-box."
    )

    # 10) Geek Squad mobile device screen repair service fee = $29
    leaf_screen_fee = evaluator.add_leaf(
        id="Screen_Repair_Fee",
        desc="The Geek Squad mobile device screen repair service fee of $29 is provided",
        parent=col_node,
        critical=True
    )
    claim_screen_fee = "Geek Squad's mobile device screen repair service fee is $29."
    await evaluator.verify(
        claim=claim_screen_fee,
        node=leaf_screen_fee,
        sources=service_sources,
        additional_instruction="Typically tied to AppleCare+ service fee; confirm the $29 screen repair fee on an official Best Buy/Geek Squad or AppleCare+ pricing page referenced in the answer."
    )

    # 11) Geek Squad mobile device other damage repair fee = $99
    leaf_other_fee = evaluator.add_leaf(
        id="Other_Damage_Fee",
        desc="The Geek Squad mobile device other damage repair fee of $99 is provided",
        parent=col_node,
        critical=True
    )
    claim_other_fee = "Geek Squad's mobile device other damage repair service fee is $99."
    await evaluator.verify(
        claim=claim_other_fee,
        node=leaf_other_fee,
        sources=service_sources,
        additional_instruction="Typically tied to AppleCare+ other damage service fee; verify $99 on referenced official page(s)."
    )

    # 12) Geek Squad smart home installation services starting at $99.99
    leaf_install_base = evaluator.add_leaf(
        id="Installation_Service_Base_Price",
        desc="The base price for Geek Squad smart home installation services starting at $99.99 is provided",
        parent=col_node,
        critical=True
    )
    claim_install_base = "Geek Squad smart home installation services start at $99.99."
    await evaluator.verify(
        claim=claim_install_base,
        node=leaf_install_base,
        sources=service_sources,
        additional_instruction="Use an official Geek Squad page that lists starting prices for smart home installation services."
    )

    # 13) T-Mobile 5G coverage at the store location
    leaf_tmo = evaluator.add_leaf(
        id="T_Mobile_5G_Coverage",
        desc="Verification that the store location has T-Mobile 5G network coverage in Columbus, Ohio",
        parent=col_node,
        critical=True
    )
    claim_tmo = (
        f"The area around the store address '{_nz(extracted.address)}' in Columbus, OH has T-Mobile 5G coverage."
    )
    await evaluator.verify(
        claim=claim_tmo,
        node=leaf_tmo,
        sources=tmo_sources,
        additional_instruction="Use the official T-Mobile coverage map to verify 5G coverage (5G or 5G UC) at or around the given address."
    )

    # 14) Apple-trained Geek Squad agents available at this location
    leaf_apple_trained = evaluator.add_leaf(
        id="Apple_Trained_Staff",
        desc="Confirmation that Apple-trained Geek Squad agents are available at this location",
        parent=col_node,
        critical=True
    )
    claim_apple_trained = "Apple-trained (Apple Authorized Service Provider) Geek Squad agents are available at this Best Buy store location."
    await evaluator.verify(
        claim=claim_apple_trained,
        node=leaf_apple_trained,
        sources=_flatten_unique(store_sources, service_sources),
        additional_instruction="Evidence can include 'Apple Authorized Service Provider' designation tied to Best Buy/Geek Squad, ideally referencing this store or general AASP status applicable in-store."
    )

    # 15) Store reference URL validity (official store page or Google Maps)
    leaf_store_ref = evaluator.add_leaf(
        id="Store_Reference_URL",
        desc="A valid reference URL from Best Buy's official website or Google Maps confirming the store location and details",
        parent=col_node,
        critical=True
    )
    claim_store_ref = "This URL is an official Best Buy store locator page or a Google Maps listing for the specified Best Buy store in Columbus, OH."
    await evaluator.verify(
        claim=claim_store_ref,
        node=leaf_store_ref,
        sources=store_sources,
        additional_instruction="The page should clearly indicate the Best Buy store's name, address, and contact details."
    )

    # 16) Service pricing reference URL validity
    leaf_service_ref = evaluator.add_leaf(
        id="Service_Pricing_Reference_URL",
        desc="A valid reference URL confirming Geek Squad service pricing and availability",
        parent=col_node,
        critical=True
    )
    claim_service_ref = "This URL is an official source that confirms Geek Squad service pricing and availability (e.g., $29 screen repair fee and $99 other damage fee) and/or service offerings."
    await evaluator.verify(
        claim=claim_service_ref,
        node=leaf_service_ref,
        sources=service_sources,
        additional_instruction="Prefer Best Buy/Geek Squad official pages. The page should explicitly state pricing/fees or clear service availability."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Columbus Best Buy store information task.
    """
    # Initialize evaluator root
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel; we add a critical child aggregator to gate all
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

    # Extract structured information from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_store_info(),
        template_class=StoreExtraction,
        extraction_name="store_extraction",
    )

    # Record a compact snapshot of extracted info for debugging
    evaluator.add_custom_info(
        info={
            "store_name": extracted.store_name,
            "address": extracted.address,
            "phone": extracted.phone,
            "weekday_open": extracted.weekday_open,
            "weekday_close": extracted.weekday_close,
            "urls": {
                "store_urls": extracted.store_urls,
                "service_pricing_urls": extracted.service_pricing_urls,
                "apple_tv_urls": extracted.apple_tv_urls,
                "tmobile_coverage_urls": extracted.tmobile_coverage_urls,
            },
            "prices": {
                "apple_tv_wifi_64gb_price": extracted.apple_tv_wifi_64gb_price,
                "apple_tv_ethernet_128gb_price": extracted.apple_tv_ethernet_128gb_price,
                "screen_repair_fee": extracted.screen_repair_fee,
                "other_damage_fee": extracted.other_damage_fee,
                "install_base_price": extracted.install_base_price,
            },
            "flags": {
                "geek_squad_available": extracted.geek_squad_available,
                "same_day_screen_repair": extracted.same_day_screen_repair,
                "apple_tv_available": extracted.apple_tv_available,
                "tmobile_5g_coverage": extracted.tmobile_5g_coverage,
                "apple_trained_geek_squad": extracted.apple_trained_geek_squad,
            },
        },
        info_type="extraction_snapshot",
        info_name="extracted_store_info",
    )

    # Build verification tree and run verifications
    await _build_and_verify(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()