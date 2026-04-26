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
TASK_ID = "wireless_earbuds_chicago"
TASK_DESCRIPTION = """I'm a professional working in Chicago, Illinois, and I need to purchase wireless earbuds that will serve multiple purposes throughout my day. I commute daily on public transportation (requiring strong noise cancellation), participate in video conferences at work (requiring clear audio and the ability to connect to multiple devices), and occasionally work out at the gym after work (requiring sweat resistance).

Find ONE specific wireless earbud model that meets ALL of the following requirements:

Technical Requirements:
- Active Noise Cancellation (ANC)
- Transparency mode
- Water resistance rating of at least IPX4
- Battery life of at least 8 hours on a single charge with ANC enabled
- Multipoint Bluetooth connection support (can connect to at least 2 devices simultaneously)
- AAC Bluetooth codec support (minimum)
- Touch controls
- Voice assistant integration (Siri, Google Assistant, or Alexa)
- Wireless charging case
- Weight specification per earbud must be provided
- Standard manufacturer warranty of at least 1 year

Availability Requirements:
- Must be available for purchase at physical retail stores in Chicago, Illinois
- Must be available from at least 3 different retailers in Chicago
- For each retailer, provide: store name, specific store address/location, and confirmation of availability

Price Requirement:
- Must be priced between $150 and $250

Verification Requirement:
- Provide the manufacturer's official product page URL where these specifications can be verified

For your answer, provide:
1. The specific model name and manufacturer
2. The manufacturer's official product page URL
3. Complete information for at least 3 Chicago retailers (name, address, availability)
4. Verification of each technical specification listed above
5. Current price range
"""

PRICE_MIN = 150
PRICE_MAX = 250

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProductBasic(BaseModel):
    manufacturer: Optional[str] = None
    model_name: Optional[str] = None
    manufacturer_url: Optional[str] = None


class RetailerInfo(BaseModel):
    name: Optional[str] = None
    store_address: Optional[str] = None
    availability_confirmation: Optional[str] = None
    retailer_url: Optional[str] = None
    price_text: Optional[str] = None


class RetailersExtraction(BaseModel):
    retailers: List[RetailerInfo] = Field(default_factory=list)


class PriceRangeExtraction(BaseModel):
    price_min_usd: Optional[str] = None
    price_max_usd: Optional[str] = None


# (Optional) Capture the answer's stated spec claims for record keeping
class TechSpecClaims(BaseModel):
    anc: Optional[str] = None
    transparency_mode: Optional[str] = None
    water_resistance_rating: Optional[str] = None
    battery_life_anc: Optional[str] = None
    multipoint: Optional[str] = None
    codec_aac: Optional[str] = None
    touch_controls: Optional[str] = None
    voice_assistant: Optional[str] = None
    wireless_charging_case: Optional[str] = None
    earbud_weight: Optional[str] = None
    warranty: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_product_basic() -> str:
    return """
    Extract the single specific wireless earbud model recommended in the answer.
    Return a JSON object with:
    - manufacturer: The brand/manufacturer name (e.g., Sony, Apple, Bose)
    - model_name: The exact earbud model name (e.g., WF-1000XM5)
    - manufacturer_url: The official manufacturer product page URL provided in the answer (not a review or retailer site)
    If multiple models are mentioned, extract the one that the answer ultimately recommends as the chosen model.
    If the manufacturer URL is missing, return null for it.
    """


def prompt_extract_retailers() -> str:
    return """
    Extract up to 5 Chicago retailers for the recommended earbuds from the answer.
    For each retailer, return:
    - name: Retailer name (e.g., Best Buy, Target)
    - store_address: The specific Chicago store address or location text (should indicate it's a Chicago, IL location)
    - availability_confirmation: The text or summary indicating availability at that Chicago store (e.g., "In stock", "Pickup today")
    - retailer_url: A URL provided in the answer that supports the Chicago store's availability of the product (ideally the store or product page specific to that location)
    - price_text: The price shown for the product on that retailer (if present in the answer)
    Only include retailers where the answer provides Chicago store/location information.
    If fewer than 3 are provided, return what is available.
    """


def prompt_extract_price_range() -> str:
    return f"""
    Extract the current price range for the recommended earbuds as stated in the answer.
    Return:
    - price_min_usd: The lowest price mentioned in USD (numbers only if possible, e.g., "169.99"); if a single price is mentioned, set both min and max to it.
    - price_max_usd: The highest price mentioned in USD (numbers only if possible)
    If no price is provided, return null for both fields.
    """


def prompt_extract_spec_claims() -> str:
    return """
    From the answer text, extract whether each technical requirement is claimed to be met. Use brief phrases or 'yes'/'no' if explicit.
    Return:
    - anc
    - transparency_mode
    - water_resistance_rating
    - battery_life_anc
    - multipoint
    - codec_aac
    - touch_controls
    - voice_assistant
    - wireless_charging_case
    - earbud_weight
    - warranty
    If a field is not explicitly mentioned, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def full_model_str(product: ProductBasic) -> str:
    manu = (product.manufacturer or "").strip()
    model = (product.model_name or "").strip()
    if manu and model:
        return f"{manu} {model}"
    return manu or model or "the earbuds"


def retailer_is_chicago(ret: RetailerInfo) -> bool:
    if ret.store_address and isinstance(ret.store_address, str):
        return "chicago" in ret.store_address.lower()
    return False


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_product_identification(evaluator: Evaluator, parent_node, product: ProductBasic):
    pid = evaluator.add_sequential(
        id="Product_Identification",
        desc="Verify that a specific wireless earbud model has been identified with manufacturer verification",
        parent=parent_node,
        critical=True
    )

    # 1) Model name + manufacturer provided
    model_ok = bool(product.manufacturer and product.model_name)
    evaluator.add_custom_node(
        result=model_ok,
        id="Model_Name_Provided",
        desc="A specific earbud model name and manufacturer are clearly stated",
        parent=pid,
        critical=True
    )

    # 2) Manufacturer URL provided
    url_ok = is_valid_url(product.manufacturer_url)
    evaluator.add_custom_node(
        result=url_ok,
        id="Manufacturer_URL",
        desc="Official manufacturer product page URL is provided for specification verification",
        parent=pid,
        critical=True
    )


async def verify_price_verification(evaluator: Evaluator, parent_node, retailers: RetailersExtraction, product: ProductBasic):
    """
    Verify the price range across individual retailers (critical).
    We create one critical leaf per retailer to avoid OR semantics of multi-URL verification.
    """
    price_node = evaluator.add_parallel(
        id="Price_Verification",
        desc=f"Verify the model falls within the ${PRICE_MIN}-${PRICE_MAX} price range",
        parent=parent_node,
        critical=True
    )

    full_model = full_model_str(product)
    # Verify up to first 3 retailers individually
    for idx in range(3):
        node_id = f"Price_Within_Range_Retailer_{idx+1}"
        desc = f"Retailer #{idx+1} price is between ${PRICE_MIN} and ${PRICE_MAX}"
        if idx < len(retailers.retailers):
            r = retailers.retailers[idx]
            if is_valid_url(r.retailer_url):
                leaf = evaluator.add_leaf(
                    id=node_id,
                    desc=desc,
                    parent=price_node,
                    critical=True
                )
                claim = f"The price of {full_model} on this retailer page is between ${PRICE_MIN} and ${PRICE_MAX} USD."
                await evaluator.verify(
                    claim=claim,
                    node=leaf,
                    sources=r.retailer_url,
                    additional_instruction=f"Check the visible product price on the page. "
                                           f"Accept sale price or regular price as long as it is within ${PRICE_MIN}-{PRICE_MAX}. "
                                           f"If the price is hidden ('see price in cart') or membership-only with no visible number, consider unsupported."
                )
            else:
                evaluator.add_custom_node(
                    result=False,
                    id=node_id,
                    desc=desc + " (missing/invalid retailer URL)",
                    parent=price_node,
                    critical=True
                )
        else:
            evaluator.add_custom_node(
                result=False,
                id=node_id,
                desc=desc + " (no retailer provided)",
                parent=price_node,
                critical=True
            )


async def verify_availability_minimum(evaluator: Evaluator, parent_node, retailers: RetailersExtraction):
    """
    Critical gate: At least 3 different Chicago retailers with complete info.
    """
    # Define "complete" minimally: name, Chicago address, availability text, and a URL
    complete = []
    seen_names = set()
    for r in retailers.retailers:
        if (r.name and retailer_is_chicago(r) and r.availability_confirmation and is_valid_url(r.retailer_url)):
            key = r.name.strip().lower()
            if key not in seen_names:
                complete.append(r)
                seen_names.add(key)
    evaluator.add_custom_node(
        result=(len(complete) >= 3),
        id="Minimum_Three_Retailers",
        desc="At least three different retailers are provided with complete information",
        parent=parent_node,
        critical=True
    )
    return complete[:3]


async def verify_availability_details(evaluator: Evaluator, parent_node, retailers: List[RetailerInfo], product: ProductBasic):
    """
    Non-critical: verify each provided Chicago retailer's page indicates in-store availability/pickup.
    """
    avail = evaluator.add_parallel(
        id="Availability_Verification",
        desc="Verify the model is available from at least 3 Chicago retailers with specific store information",
        parent=parent_node,
        critical=False
    )

    full_model = full_model_str(product)

    # Up to three detail checks to mirror rubric's three retailer items
    for idx in range(3):
        node_id = f"Retailer_{idx+1}_Information"
        desc = [
            "First", "Second", "Third"
        ][idx] + " Chicago retailer name, specific store address, and availability confirmation provided"
        if idx < len(retailers):
            r = retailers[idx]
            if is_valid_url(r.retailer_url) and retailer_is_chicago(r) and r.name and r.store_address and r.availability_confirmation:
                leaf = evaluator.add_leaf(
                    id=node_id,
                    desc=desc,
                    parent=avail,
                    critical=False
                )
                claim = (
                    f"The retailer '{r.name}' at '{r.store_address}' in Chicago shows that {full_model} "
                    f"is currently available for in-store purchase or same-day pickup."
                )
                await evaluator.verify(
                    claim=claim,
                    node=leaf,
                    sources=r.retailer_url,
                    additional_instruction=(
                        "Check that the page corresponds to a Chicago, IL store/location and indicates current availability "
                        "(e.g., 'In stock', 'Pickup today', 'Available at this store'). If the page shows 'Out of stock' or only online-only availability, the claim is not supported."
                    )
                )
            else:
                evaluator.add_custom_node(
                    result=False,
                    id=node_id,
                    desc=desc + " (insufficient info or invalid URL)",
                    parent=avail,
                    critical=False
                )
        else:
            evaluator.add_custom_node(
                result=False,
                id=node_id,
                desc=desc + " (not provided)",
                parent=avail,
                critical=False
            )


async def verify_technical_specs(evaluator: Evaluator, parent_node, product: ProductBasic):
    """
    Critical: Verify all required technical specs against the official manufacturer page.
    """
    tech = evaluator.add_parallel(
        id="Technical_Specifications",
        desc="Verify all required technical specifications are met and documented",
        parent=parent_node,
        critical=True
    )

    manu_url = product.manufacturer_url if is_valid_url(product.manufacturer_url) else None
    full_model = full_model_str(product)

    # Define spec claims
    checks = [
        (
            "Active_Noise_Cancellation",
            f"The {full_model} includes Active Noise Cancellation (ANC).",
            "Confirm that the official page explicitly mentions Active Noise Cancellation or noise cancelling for the earbuds themselves."
        ),
        (
            "Transparency_Mode",
            f"The {full_model} includes a Transparency mode (also referred to as Ambient/Awareness/Hear-Through).",
            "Accept synonymous terms such as Transparency, Ambient sound, Aware mode, Hear-through, or similar."
        ),
        (
            "Water_Resistance_Rating",
            f"The {full_model} has a water resistance rating of at least IPX4.",
            "Accept IPX4 or any higher/equivalent rating (e.g., IPX5, IP54, IP55, IP57). Sweat resistance tied to an IPX rating qualifies."
        ),
        (
            "Battery_Life_Single_Charge",
            f"The {full_model} provides at least 8 hours of playtime on a single charge with ANC enabled.",
            "Specifically check for the duration with ANC ON. If only battery life with ANC OFF is listed and it's unclear about ANC ON, treat as not supported."
        ),
        (
            "Multipoint_Bluetooth",
            f"The {full_model} supports multipoint Bluetooth connection to at least two devices simultaneously.",
            "Look for 'multipoint', 'connect to 2 devices', or similar language on the official page."
        ),
        (
            "Bluetooth_Codec_Support",
            f"The {full_model} supports the AAC Bluetooth audio codec.",
            "AAC must be supported. Additional codecs (SBC, aptX, LDAC) are acceptable but AAC support is required."
        ),
        (
            "Touch_Controls",
            f"The {full_model} provides touch controls for playback or functions.",
            "Look for mentions of 'touch controls', 'tap', or 'touch gestures' on the earbuds."
        ),
        (
            "Voice_Assistant_Integration",
            f"The {full_model} integrates with at least one major voice assistant (Siri, Google Assistant, or Alexa).",
            "Any one of Siri, Google Assistant, or Alexa is sufficient."
        ),
        (
            "Wireless_Charging_Case",
            f"The charging case for {full_model} supports wireless charging (e.g., Qi).",
            "Look for 'wireless charging' or 'Qi-compatible' for the case."
        ),
        (
            "Weight_Specification",
            f"The official specifications for {full_model} include the weight per single earbud.",
            "Must specify per-earbud (e.g., 'each earbud weighs X g'); not just the case weight or total combined."
        ),
        (
            "Warranty_Information",
            f"The {full_model} comes with a standard manufacturer warranty of at least 1 year.",
            "Look for '1-year limited warranty' or greater. If region-specific, accept US-standard 1-year language if present."
        ),
    ]

    # Create leaf nodes and batch verify
    claims_and_sources = []
    for node_id, claim, add_ins in checks:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=claim.replace(full_model, "{model}").format(model=full_model),  # desc placeholder-friendly
            parent=tech,
            critical=True
        )
        claims_and_sources.append((
            claim,
            manu_url,  # verify against manufacturer page as required
            leaf,
            add_ins
        ))

    # Run in parallel
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the Wireless Earbuds Chicago task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Follow rubric's sequential high-level evaluation
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

    # Extract structured information
    product_task = evaluator.extract(
        prompt=prompt_extract_product_basic(),
        template_class=ProductBasic,
        extraction_name="product_basic"
    )
    retailers_task = evaluator.extract(
        prompt=prompt_extract_retailers(),
        template_class=RetailersExtraction,
        extraction_name="retailers"
    )
    price_task = evaluator.extract(
        prompt=prompt_extract_price_range(),
        template_class=PriceRangeExtraction,
        extraction_name="price_range"
    )
    spec_claims_task = evaluator.extract(
        prompt=prompt_extract_spec_claims(),
        template_class=TechSpecClaims,
        extraction_name="answer_spec_claims"
    )

    product, retailers, _price_rng, _spec_claims = await asyncio.gather(
        product_task, retailers_task, price_task, spec_claims_task
    )

    # 1) Product Identification (critical)
    await verify_product_identification(evaluator, root, product)

    # 2) Price Verification (critical) — check price per retailer
    await verify_price_verification(evaluator, root, retailers, product)

    # 3) Availability minimum (critical)
    top3_complete = await verify_availability_minimum(evaluator, root, retailers)

    # 4) Availability details (non-critical, parallel) — verify each of the first 3 retailers
    await verify_availability_details(evaluator, root, top3_complete, product)

    # 5) Technical Specifications (critical, parallel) — verify against manufacturer product page
    await verify_technical_specs(evaluator, root, product)

    # Return structured evaluation summary
    return evaluator.get_summary()