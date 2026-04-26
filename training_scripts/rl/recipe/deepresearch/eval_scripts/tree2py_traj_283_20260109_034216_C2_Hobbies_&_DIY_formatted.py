import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "beginner_candle_kit_under_60"
TASK_DESCRIPTION = (
    "I am looking to purchase a complete candle making kit for beginners that I can order online from a major US craft "
    "retailer (such as Michaels, CandleScience, Bramble Berry, or similar well-known suppliers). The kit must meet all "
    "of the following requirements: (1) Be explicitly designed and marketed for beginners or as a \"starter kit\", (2) "
    "Be priced under $60 (before any shipping costs), (3) Include all of the following essential supplies in the kit: "
    "candle wax (any type suitable for candle making), pre-tabbed wicks, containers or molds for forming the candles, "
    "at least one fragrance oil or essential oil, and a thermometer for monitoring wax temperature, (4) Include written "
    "instructions or provide clear access to instructional materials, (5) Be currently available for purchase (not out "
    "of stock). Please identify one candle making kit that satisfies all these requirements. For the kit you identify, "
    "provide: the exact name of the kit, the retailer selling it, the current price, a direct link to the product page "
    "on the retailer's website, and confirmation that all five required components (wax, wicks, containers/molds, "
    "fragrance, thermometer) are included in the kit."
)


# --------------------------------------------------------------------------- #
# Major US craft retailers / well-known candle suppliers                      #
# (Used for deterministic domain/name checking of retailer credibility)       #
# --------------------------------------------------------------------------- #
MAJOR_RETAILER_DOMAIN_SUFFIXES = [
    "michaels.com",
    "joann.com",
    "hobbylobby.com",
    "candlescience.com",
    "brambleberry.com",
    "theflamingcandle.com",
    "lonestarcandlesupply.com",
    "naturesgardencandles.com",
    "bulkapothecary.com",
    "makesy.com",
    "woodenwick.com",
    "candlewic.com",
    "candlesupply.com",           # Bitter Creek Candle Supply
    "candlemakingsupplies.net",   # Aztec Candle & Soap Making Supplies
]

MAJOR_RETAILER_NAMES = [
    "michaels",
    "joann",
    "jo-ann",
    "hobby lobby",
    "candlescience",
    "bramble berry",
    "the flaming candle",
    "lone star candle supply",
    "nature's garden",
    "natures garden",
    "bulk apothecary",
    "makesy",
    "wooden wick",
    "candlewic",
    "bitter creek",
    "aztec candle",
]


def _extract_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return None


def _is_major_retailer_domain(url: Optional[str]) -> bool:
    dom = _extract_domain(url)
    if not dom:
        return False
    return any(dom == sfx or dom.endswith("." + sfx) for sfx in MAJOR_RETAILER_DOMAIN_SUFFIXES)


def _is_major_retailer_name(name: Optional[str]) -> bool:
    if not name:
        return False
    n = name.strip().lower()
    return any(key in n for key in MAJOR_RETAILER_NAMES)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CandleKitExtraction(BaseModel):
    kit_name: Optional[str] = None
    retailer_name: Optional[str] = None
    price_text: Optional[str] = None
    product_url: Optional[str] = None
    instructions_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_candle_kit() -> str:
    return """
    You must extract the single primary candle making kit identified in the answer for purchase.

    Extract exactly these fields:
    - kit_name: the exact product/kit name as presented in the answer
    - retailer_name: the retailer or supplier selling this kit (e.g., Michaels, CandleScience, Bramble Berry, etc.)
    - price_text: the current listed price as stated in the answer (keep it as a string; do not normalize)
    - product_url: a direct link (URL) to the product page on the retailer's website
    - instructions_urls: an array of any URLs mentioned in the answer that provide instructions or tutorials (if none are provided, return an empty array)

    Rules:
    - If multiple kits are mentioned, choose the one the answer actually recommends or the first listed kit.
    - Only return URLs that appear explicitly in the answer text.
    - If a field is missing in the answer, return null (or an empty array for instructions_urls).
    """


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_kit_tree(evaluator: Evaluator, parent_node, kit: CandleKitExtraction) -> None:
    """
    Build the critical verification tree for the candle kit and run all checks.
    All nodes under this critical subtree must also be critical to satisfy the framework's constraint.
    """
    # Critical task-level node
    task_node = evaluator.add_parallel(
        id="Complete_Beginner_Candle_Making_Kit",
        desc="Identifies one candle making kit and provides required purchase info, such that the kit meets all specified constraints",
        parent=parent_node,
        critical=True,
    )

    # 1) Required response fields (critical)
    required_fields_node = evaluator.add_parallel(
        id="Required_Response_Fields",
        desc="Answer provides all requested identifying and purchasing information for the kit",
        parent=task_node,
        critical=True,
    )

    # Provide exact kit name
    evaluator.add_custom_node(
        result=bool(kit.kit_name and kit.kit_name.strip()),
        id="Provide_Exact_Kit_Name",
        desc="Provides the exact name of the identified kit",
        parent=required_fields_node,
        critical=True,
    )

    # Provide retailer name
    evaluator.add_custom_node(
        result=bool(kit.retailer_name and kit.retailer_name.strip()),
        id="Provide_Retailer_Name",
        desc="Provides the retailer selling the kit",
        parent=required_fields_node,
        critical=True,
    )

    # Provide current price (ensure a number present)
    evaluator.add_custom_node(
        result=bool(kit.price_text and re.search(r"\d", kit.price_text or "")),
        id="Provide_Current_Price",
        desc="Provides the current listed price of the kit",
        parent=required_fields_node,
        critical=True,
    )

    # Provide direct product link
    evaluator.add_custom_node(
        result=bool(kit.product_url and kit.product_url.strip().startswith(("http://", "https://"))),
        id="Provide_Direct_Product_Link",
        desc="Provides a direct link to the product page on the retailer's website",
        parent=required_fields_node,
        critical=True,
    )

    # 2) Marketing & design (critical)
    marketing_node = evaluator.add_parallel(
        id="Marketing_And_Design",
        desc="Verifies the kit is marketed for the intended audience",
        parent=task_node,
        critical=True,
    )
    beginner_node = evaluator.add_leaf(
        id="Beginner_Targeted",
        desc="The kit is explicitly designed/marketed for beginners or described as a starter kit",
        parent=marketing_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This product page describes the kit as designed for beginners or markets it clearly as a starter kit.",
        node=beginner_node,
        sources=kit.product_url,
        additional_instruction=(
            "Look for explicit phrases like 'beginner', 'beginner-friendly', 'for beginners', 'starter kit', "
            "'starter set', 'learn candle making', or similar. Synonyms and close phrases count."
        ),
    )

    # 3) Components verification (critical)
    components_node = evaluator.add_parallel(
        id="Kit_Components_Verification",
        desc="Verifies that the kit includes all essential candle making supplies listed in the question",
        parent=task_node,
        critical=True,
    )

    # Build component leaves
    wax_node = evaluator.add_leaf(
        id="Wax_Included",
        desc="The kit includes candle wax suitable for candle making",
        parent=components_node,
        critical=True,
    )
    wicks_node = evaluator.add_leaf(
        id="Pre_Tabbed_Wicks_Included",
        desc="The kit includes pre-tabbed wicks",
        parent=components_node,
        critical=True,
    )
    containers_node = evaluator.add_leaf(
        id="Containers_Or_Molds_Included",
        desc="The kit includes containers or molds for forming candles",
        parent=components_node,
        critical=True,
    )
    fragrance_node = evaluator.add_leaf(
        id="Fragrance_Included",
        desc="The kit includes at least one fragrance oil or essential oil",
        parent=components_node,
        critical=True,
    )
    thermo_node = evaluator.add_leaf(
        id="Thermometer_Included",
        desc="The kit includes a thermometer for monitoring wax temperature",
        parent=components_node,
        critical=True,
    )

    # Parallel verify for components
    await evaluator.batch_verify([
        (
            "This kit includes candle wax suitable for candle making (e.g., soy, paraffin, beeswax, etc.).",
            kit.product_url,
            wax_node,
            "Check the 'What's included' or product description sections for wax. Any candle-making wax satisfies this."
        ),
        (
            "This kit includes pre-tabbed candle wicks.",
            kit.product_url,
            wicks_node,
            "Look for 'pre-tabbed wicks', 'wicks with tabs', 'pre-waxed wicks with tabs', or similar wording."
        ),
        (
            "This kit includes containers or molds for forming the candles.",
            kit.product_url,
            containers_node,
            "Containers may be jars, tins, vessels, etc. Molds also count. Look at 'Kit contents' or description."
        ),
        (
            "This kit includes at least one fragrance oil or essential oil for scenting candles.",
            kit.product_url,
            fragrance_node,
            "Look for 'fragrance oil', 'essential oil', 'scent', 'scented', or 'includes fragrance' in contents."
        ),
        (
            "This kit includes a thermometer for monitoring wax temperature.",
            kit.product_url,
            thermo_node,
            "Wording may be 'thermometer', 'digital thermometer', 'candy thermometer', 'temperature gauge', etc."
        ),
    ])

    # 4) Purchase criteria (critical)
    purchase_node = evaluator.add_parallel(
        id="Purchase_Criteria",
        desc="Verifies price, instructions, retailer, and availability constraints",
        parent=task_node,
        critical=True,
    )

    # Price under $60 (before shipping)
    price_under_60_node = evaluator.add_leaf(
        id="Price_Under_60_Before_Shipping",
        desc="The kit is priced under $60 before any shipping costs",
        parent=purchase_node,
        critical=True,
    )

    # Currently available
    available_node = evaluator.add_leaf(
        id="Currently_Available_Not_Out_Of_Stock",
        desc="The kit is currently available for online purchase (not out of stock)",
        parent=purchase_node,
        critical=True,
    )

    # Instructions accessible
    instructions_node = evaluator.add_leaf(
        id="Instructions_Accessible",
        desc="The kit includes written instructions or provides clear access to instructional materials",
        parent=purchase_node,
        critical=True,
    )

    # Major US craft retailer check (deterministic via domain/name)
    major_retailer_bool = _is_major_retailer_domain(kit.product_url) or _is_major_retailer_name(kit.retailer_name)
    evaluator.add_custom_node(
        result=major_retailer_bool,
        id="Major_US_Craft_Retailer",
        desc="The kit is sold by a major US craft retailer (e.g., Michaels, CandleScience, Bramble Berry, or similar well-known supplier)",
        parent=purchase_node,
        critical=True,
    )

    # Record retailer/domain diagnostic info
    evaluator.add_custom_info(
        {
            "product_url": kit.product_url,
            "url_domain": _extract_domain(kit.product_url),
            "retailer_name": kit.retailer_name,
            "major_retailer_detected": major_retailer_bool,
        },
        info_type="diagnostics",
        info_name="retailer_check"
    )

    # Run purchase criteria verifications (parallel)
    # For instructions, use both product_url and any explicit instruction URLs if provided
    instr_sources: List[str] = []
    if kit.product_url:
        instr_sources.append(kit.product_url)
    if kit.instructions_urls:
        # de-duplicate while preserving order
        seen = set()
        for u in kit.instructions_urls:
            if isinstance(u, str) and u and (u not in seen):
                instr_sources.append(u)
                seen.add(u)

    await evaluator.batch_verify([
        (
            "The current listed price shown on this product page is under $60 USD (before any shipping costs).",
            kit.product_url,
            price_under_60_node,
            "If a price range or multiple variants are shown, use the default/primary configuration's price. "
            "Ignore shipping, taxes, or optional add-ons. Discounted sale price is acceptable."
        ),
        (
            "This product is currently available for purchase online (i.e., not out of stock).",
            kit.product_url,
            available_node,
            "Look for 'In Stock', 'Add to Cart', active purchase buttons, and absence of 'Out of Stock' indicators."
        ),
        (
            "The kit includes written instructions or provides clear access to instructional materials (printed or online).",
            instr_sources if instr_sources else kit.product_url,
            instructions_node,
            "Check 'What's included', description, or links to PDFs/tutorials. Phrases like 'instructions included', "
            "'instruction guide', 'how-to', 'tutorial', 'learn guide', or similar count."
        ),
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
    Evaluate an answer for the 'beginner candle kit under $60' task and return a structured result dictionary.
    """
    # Initialize evaluator (root is non-critical, we will add a critical child node for the task)
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

    # Extract the identified kit information
    kit_info = await evaluator.extract(
        prompt=prompt_extract_candle_kit(),
        template_class=CandleKitExtraction,
        extraction_name="candle_kit_extraction",
    )

    # Build and verify the critical subtree for the kit
    await build_and_verify_kit_tree(evaluator, root, kit_info)

    # Return summary
    return evaluator.get_summary()