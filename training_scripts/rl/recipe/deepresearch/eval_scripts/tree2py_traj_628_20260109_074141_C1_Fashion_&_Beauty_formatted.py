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
TASK_ID = "sunscreen_mineral_lb_spf30"
TASK_DESCRIPTION = (
    "I am looking for a facial or body sunscreen that meets ethical and safety standards. "
    "Please identify one sunscreen product that is Leaping Bunny certified (cruelty-free certification by CCIC), "
    "uses only mineral active ingredients (zinc oxide and/or titanium dioxide with no chemical UV filters), "
    "and provides SPF 30 or higher broad-spectrum protection. Provide the product name, brand, and a link to the "
    "product page or the brand's official website where this product is listed."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SunscreenProductExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer for one sunscreen product.
    """
    product_name: Optional[str] = None
    brand_name: Optional[str] = None
    product_url: Optional[str] = None

    # Sources that may support Leaping Bunny certification (brand page, certification database, product page)
    certification_urls: List[str] = Field(default_factory=list)

    # Active sunscreen ingredients mentioned in the answer (e.g., ["Zinc Oxide", "Titanium Dioxide"])
    active_ingredients: List[str] = Field(default_factory=list)

    # URLs that may list ingredients or product technical info
    ingredient_urls: List[str] = Field(default_factory=list)

    # SPF and labeling fields mentioned in the answer
    spf_value: Optional[str] = None            # e.g., "SPF 30", "30", "SPF 50+"
    broad_spectrum_label: Optional[str] = None # e.g., "Broad Spectrum", "UVA/UVB"

    # URLs that may mention SPF or broad-spectrum labeling (often the product page)
    spf_urls: List[str] = Field(default_factory=list)

    # URLs that may indicate availability (product page, authorized retailer)
    availability_urls: List[str] = Field(default_factory=list)

    # Any other relevant URLs referenced by the answer
    other_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_sunscreen_product() -> str:
    return (
        "Extract the details for one sunscreen product mentioned in the answer that satisfies the constraints. "
        "Return a JSON object with the following fields:\n"
        "1. product_name: The product name exactly as stated in the answer.\n"
        "2. brand_name: The brand name exactly as stated in the answer.\n"
        "3. product_url: A URL to the product page on the brand's official website or an authorized retailer where the product is listed.\n"
        "4. certification_urls: Array of URLs that support Leaping Bunny certification (e.g., official Leaping Bunny/CCIC directory page, brand page showing the Leaping Bunny logo/certification).\n"
        "5. active_ingredients: Array of listed active sunscreen ingredients for the product (extract the names exactly as mentioned in the answer).\n"
        "6. ingredient_urls: Array of URLs that list the product ingredients (often the product page or brand site technical pages).\n"
        "7. spf_value: The SPF value as stated in the answer (e.g., 'SPF 30', '30', 'SPF 50+').\n"
        "8. broad_spectrum_label: The broad-spectrum labeling text if mentioned (e.g., 'Broad Spectrum', 'UVA/UVB'); otherwise null.\n"
        "9. spf_urls: Array of URLs that mention or show the SPF and broad-spectrum claim (often the product page).\n"
        "10. availability_urls: Array of URLs that indicate the product is available for purchase (product page, authorized retailer).\n"
        "11. other_urls: Any other relevant URLs cited in the answer.\n\n"
        "GENERAL RULES:\n"
        "- Extract exactly as presented in the answer; do not invent anything.\n"
        "- If a field is missing in the answer, set it to null (for single fields) or an empty array (for list fields).\n"
        "- For URLs, extract only valid web URLs that appear in the answer (plain or markdown). If protocol is missing, prepend http://.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty_str(val: Optional[str]) -> bool:
    return bool(val and isinstance(val, str) and val.strip())


def _is_valid_url(url: Optional[str]) -> bool:
    if not _is_nonempty_str(url):
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def _combine_sources(*args: List[str | None] | List[List[str]]) -> List[str]:
    """
    Combine multiple source lists or single URLs into a unique list preserving order.
    Accepts single URL strings or lists of URL strings.
    """
    combined: List[str] = []
    def _add(url: str):
        if _is_valid_url(url) and url not in combined:
            combined.append(url)

    for arg in args:
        if arg is None:
            continue
        if isinstance(arg, list):
            for u in arg:
                if isinstance(u, str):
                    _add(u)
        elif isinstance(arg, str):
            _add(arg)
    return combined


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_sunscreen_product(
    evaluator: Evaluator,
    parent_node,
    info: SunscreenProductExtraction,
) -> None:
    """
    Build the verification tree for the sunscreen product and run all checks.
    """
    # Create main critical node for the sunscreen product
    product_node = evaluator.add_parallel(
        id="sunscreen_product",
        desc="Identify one facial or body sunscreen product that satisfies all stated ethical, formulation, and protection constraints, and provide the required identifying information and link.",
        parent=parent_node,
        critical=True,
    )

    # --------------------------------------------------------------------- #
    # Provided product details (critical group)
    # --------------------------------------------------------------------- #
    details_node = evaluator.add_parallel(
        id="provided_product_details",
        desc="Response includes the required product identification details and link.",
        parent=product_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_nonempty_str(info.product_name),
        id="product_name_provided",
        desc="Product name is provided.",
        parent=details_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_nonempty_str(info.brand_name),
        id="brand_name_provided",
        desc="Brand name is provided.",
        parent=details_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_valid_url(info.product_url),
        id="product_page_url_provided",
        desc="A verifiable URL is provided to the product page on the brand's official website or an authorized retailer.",
        parent=details_node,
        critical=True,
    )

    # --------------------------------------------------------------------- #
    # Leaping Bunny certification (critical leaf)
    # --------------------------------------------------------------------- #
    lb_leaf = evaluator.add_leaf(
        id="leaping_bunny_certified",
        desc="Product/brand is Leaping Bunny certified by CCIC, verifiable via an official certification database entry or official brand/product materials showing the Leaping Bunny certification.",
        parent=product_node,
        critical=True,
    )

    lb_claim = (
        f"The brand '{info.brand_name or ''}' or the product '{info.product_name or ''}' "
        f"is Leaping Bunny certified by CCIC."
    )
    lb_sources = _combine_sources(
        info.certification_urls,
        info.product_url,
        info.other_urls,
    )
    await evaluator.verify(
        claim=lb_claim,
        node=lb_leaf,
        sources=lb_sources,
        additional_instruction=(
            "Verify that the provided page(s) explicitly indicate Leaping Bunny certification by CCIC. "
            "Accept either an official Leaping Bunny/CCIC directory entry listing the brand or an official brand/product page "
            "showing the Leaping Bunny logo or statement. Ignore non-official third-party blog posts unless the brand or CCIC site is provided."
        ),
    )

    # --------------------------------------------------------------------- #
    # Mineral-only active ingredients (critical leaf)
    # --------------------------------------------------------------------- #
    mineral_leaf = evaluator.add_leaf(
        id="mineral_only_active_ingredients",
        desc="Active ingredients are only zinc oxide and/or titanium dioxide, with no chemical UV filters (e.g., oxybenzone, octinoxate, avobenzone, octocrylene).",
        parent=product_node,
        critical=True,
    )

    mineral_claim = (
        "The product's listed active sunscreen ingredients are exclusively zinc oxide and/or titanium dioxide, "
        "with no chemical UV filters such as oxybenzone, octinoxate, avobenzone, homosalate, octisalate, or octocrylene."
    )
    mineral_sources = _combine_sources(
        info.product_url,
        info.ingredient_urls,
        info.other_urls,
    )
    await evaluator.verify(
        claim=mineral_claim,
        node=mineral_leaf,
        sources=mineral_sources,
        additional_instruction=(
            "Check the product page or official brand materials for the 'Active Ingredients' section. "
            "Allow reasonable naming variations (e.g., 'Zinc Oxide', 'Titanium Dioxide'). "
            "If any chemical UV filter is present, the claim is not supported."
        ),
    )

    # --------------------------------------------------------------------- #
    # SPF 30+ and Broad Spectrum (critical leaf)
    # --------------------------------------------------------------------- #
    spf_leaf = evaluator.add_leaf(
        id="spf_30_plus_broad_spectrum",
        desc="Product is labeled/provided as SPF 30 or higher and broad-spectrum (UVA/UVB) protection.",
        parent=product_node,
        critical=True,
    )

    spf_claim = (
        "The product is labeled as SPF 30 or higher and explicitly described as broad-spectrum (UVA/UVB) protection."
    )
    spf_sources = _combine_sources(
        info.product_url,
        info.spf_urls,
        info.other_urls,
    )
    await evaluator.verify(
        claim=spf_claim,
        node=spf_leaf,
        sources=spf_sources,
        additional_instruction=(
            "Look for labeling such as 'Broad Spectrum SPF 30', 'SPF 30 UVA/UVB', or any equivalent wording indicating "
            "broad-spectrum protection with SPF of at least 30. Equivalent phrasing is acceptable."
        ),
    )

    # --------------------------------------------------------------------- #
    # Currently available for purchase (critical leaf)
    # --------------------------------------------------------------------- #
    avail_leaf = evaluator.add_leaf(
        id="currently_available_for_purchase",
        desc="Product is currently available for purchase, verifiable via a product page indicating it can be purchased (not discontinued/unavailable).",
        parent=product_node,
        critical=True,
    )

    avail_claim = (
        "The product page indicates the item is currently available for purchase (e.g., shows price, in-stock status, or an Add to Cart/Buy button) and is not discontinued or permanently unavailable."
    )
    avail_sources = _combine_sources(
        info.product_url,
        info.availability_urls,
        info.other_urls,
    )
    await evaluator.verify(
        claim=avail_claim,
        node=avail_leaf,
        sources=avail_sources,
        additional_instruction=(
            "Accept brand official product pages or authorized retailer pages that clearly indicate purchase availability "
            "such as 'Add to Cart', price shown with in-stock status, or similar. If the page clearly states 'discontinued', "
            "'unavailable', or permanently out of stock, the claim is not supported."
        ),
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
    Evaluate the agent's answer for the sunscreen product task and return the structured summary.
    """
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
        default_model=model,
    )

    # Extract product information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_sunscreen_product(),
        template_class=SunscreenProductExtraction,
        extraction_name="sunscreen_product_info",
    )

    # Build verification tree and run checks
    await verify_sunscreen_product(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()