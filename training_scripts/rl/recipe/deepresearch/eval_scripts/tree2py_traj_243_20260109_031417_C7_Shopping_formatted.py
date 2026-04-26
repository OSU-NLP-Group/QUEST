import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "portable_ac_selection"
TASK_DESCRIPTION = (
    "I'm looking for a portable air conditioner for my apartment and have specific requirements. "
    "I need a model with a DOE SACC rating between 10,000 and 12,000 BTU that can cool rooms of at least 450 square feet. "
    "The unit must operate at 55 decibels (dB) or lower at its highest setting, as I'm noise-sensitive. "
    "Energy efficiency is important to me, so it must have a CEER rating of at least 8.5. "
    "I specifically want a dual-hose configuration for better efficiency. "
    "The air conditioner must have Wi-Fi connectivity and app control, and it needs to include cooling, fan, and dehumidify modes. "
    "To make it easier to move around, it should weigh 80 pounds or less. "
    "My budget is between $400 and $700. "
    "The unit must be available for purchase online from a major retailer (such as Amazon, Home Depot, Lowe's, or the manufacturer's website) "
    "and come with at least a 1-year manufacturer warranty. "
    "Please identify a specific portable air conditioner model that meets all these requirements and provide the model name/number along with a direct URL to the product page showing these specifications."
)

MAJOR_RETAILERS = ["amazon.com", "homedepot.com", "lowes.com", "bestbuy.com"]  # plus manufacturer website


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PortableACInfo(BaseModel):
    """Structured extraction of the selected portable AC and its cited specs/links from the answer text."""
    model_name: Optional[str] = None
    model_number: Optional[str] = None

    # Direct product/specifications page URL (prefer manufacturer or retailer product page with full specs)
    product_page_url: Optional[str] = None

    # Optional additional retailer URLs mentioned in the answer
    retailer_urls: List[str] = Field(default_factory=list)

    # Specs mentioned in the answer (kept as strings for flexibility)
    doe_sacc_btu: Optional[str] = None
    coverage_sqft: Optional[str] = None
    max_noise_db: Optional[str] = None
    ceer_rating: Optional[str] = None
    hose_configuration: Optional[str] = None
    wifi_connectivity: Optional[str] = None
    app_control: Optional[str] = None
    includes_cooling_mode: Optional[str] = None
    includes_fan_mode: Optional[str] = None
    includes_dehumidify_mode: Optional[str] = None
    weight_lbs: Optional[str] = None
    price_usd: Optional[str] = None
    warranty_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_portable_ac() -> str:
    return """
    Extract the single portable air conditioner model recommended in the answer and all relevant information explicitly stated in the answer text.
    Return a JSON object with the following fields:
    1. model_name: The model name or marketing name (string).
    2. model_number: The model number/identifier (string). If not explicitly mentioned, return null.
    3. product_page_url: A direct URL to the product page that shows specifications (prefer manufacturer official product/specs page; a major retailer product page with full specs is also acceptable). If not provided, return null.
    4. retailer_urls: An array of additional retailer URLs (e.g., Amazon, Home Depot, Lowe's, Best Buy) explicitly mentioned in the answer for the same product. If none are mentioned, return an empty array.
    5. doe_sacc_btu: The DOE SACC rating mentioned in BTU, as written (string). If not explicitly mentioned, return null.
    6. coverage_sqft: The stated room coverage in square feet (string). If not explicitly mentioned, return null.
    7. max_noise_db: The maximum noise level in decibels at highest setting (string). If not explicitly mentioned, return null.
    8. ceer_rating: The CEER rating (string). If CEER is not mentioned, return null (do NOT substitute EER).
    9. hose_configuration: The stated hose configuration (string; e.g., 'dual-hose', 'single-hose'). If not mentioned, return null.
    10. wifi_connectivity: Whether Wi‑Fi connectivity is mentioned ('yes'/'no' or text; return null if not mentioned).
    11. app_control: Whether app/mobile control is mentioned ('yes'/'no' or text; return null if not mentioned).
    12. includes_cooling_mode: Whether a cooling mode is mentioned ('yes'/'no' or text; return null if not mentioned).
    13. includes_fan_mode: Whether a fan mode is mentioned ('yes'/'no' or text; return null if not mentioned).
    14. includes_dehumidify_mode: Whether a dehumidify/dry mode is mentioned ('yes'/'no' or text; return null if not mentioned).
    15. weight_lbs: The product weight in pounds as written (string). If not explicitly mentioned, return null.
    16. price_usd: The stated price in USD as written (string). If not explicitly mentioned, return null.
    17. warranty_text: The warranty mentioned (string; e.g., '1-year limited manufacturer warranty'). If not mentioned, return null.

    Rules:
    - Extract only what is explicitly present in the answer text. Do not infer or invent values.
    - For URLs, extract the actual URLs (plain text or markdown links). If no URL is given, return null for product_page_url and an empty list for retailer_urls.
    - Keep numbers as strings exactly as written; do not normalize units.
    - If multiple models are mentioned, choose the primary recommended one (first clearly recommended).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        u_norm = u.strip()
        if u_norm and u_norm not in seen:
            seen.add(u_norm)
            out.append(u_norm)
    return out


def get_verification_urls(info: PortableACInfo) -> List[str]:
    """Gather all URLs to use for verification: the main product page and any retailer URLs."""
    return _dedupe_urls([info.product_page_url] + list(info.retailer_urls))


def safe_model_label(info: PortableACInfo) -> str:
    """Return a human-friendly label for the model."""
    if info.model_name and info.model_number:
        return f"{info.model_name} ({info.model_number})"
    if info.model_name:
        return info.model_name
    if info.model_number:
        return info.model_number
    return "the product"


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, info: PortableACInfo, parent_node) -> None:
    """
    Build verification leaves according to the rubric and run verifications against cited webpages.
    All checks here are critical and aggregated in parallel under the parent.
    """

    # Pre-check existence of core identification: model and direct product page URL
    has_model_and_url = evaluator.add_custom_node(
        result=bool((info.model_name or info.model_number) and info.product_page_url),
        id="has_model_and_specs_url",
        desc="The answer provides a specific model name/number and a direct product/specs page URL",
        parent=parent_node,
        critical=True,
    )

    # Leaf: model_and_specs_url
    model_specs_node = evaluator.add_leaf(
        id="model_and_specs_url",
        desc="Provide the specific model name/number and a direct URL to the product page showing the specifications",
        parent=parent_node,
        critical=True,
    )
    model_label = safe_model_label(info)
    claim_model_specs = (
        f"The provided product page URL is a direct product/specifications page for {model_label}, "
        f"and it lists technical specifications for this model."
    )
    await evaluator.verify(
        claim=claim_model_specs,
        node=model_specs_node,
        sources=info.product_page_url,
        additional_instruction=(
            "Confirm the page shows the exact model name/number and a 'Specifications' or detailed specs section. "
            "Marketing overview pages without specs do not count. Manufacturer official product page OR a retailer product page "
            "that lists specs is acceptable."
        ),
    )

    # Common sources for spec verification: use any of the URLs that can show specs
    all_sources = get_verification_urls(info)

    # Leaf: btu_capacity (DOE SACC between 10,000 and 12,000 BTU)
    btu_node = evaluator.add_leaf(
        id="btu_capacity",
        desc="The portable air conditioner must have a DOE SACC (Seasonally Adjusted Cooling Capacity) rating between 10,000 and 12,000 BTU",
        parent=parent_node,
        critical=True,
    )
    claim_btu = (
        f"For {model_label}, the DOE SACC (Seasonally Adjusted Cooling Capacity) rating is between 10,000 and 12,000 BTU (inclusive)."
    )
    await evaluator.verify(
        claim=claim_btu,
        node=btu_node,
        sources=all_sources,
        additional_instruction=(
            "Look specifically for 'DOE SACC', 'SACC', or 'Seasonally Adjusted Cooling Capacity'. "
            "Do NOT count ASHRAE or plain 'cooling capacity' values. If only ASHRAE is present and DOE SACC is not provided, this should fail."
        ),
    )

    # Leaf: room_coverage (≥ 450 sq ft)
    coverage_node = evaluator.add_leaf(
        id="room_coverage",
        desc="The portable air conditioner must be rated to cool rooms of at least 450 square feet",
        parent=parent_node,
        critical=True,
    )
    claim_coverage = f"{model_label} is rated to cool rooms of at least 450 square feet."
    await evaluator.verify(
        claim=claim_coverage,
        node=coverage_node,
        sources=all_sources,
        additional_instruction=(
            "Check the product specs or description for room coverage (e.g., 'up to 450 sq ft' or higher). "
            "If metric area is shown (e.g., m²), convert to sq ft (1 m² ≈ 10.764 sq ft). "
            "Requirement is at least 450 sq ft."
        ),
    )

    # Leaf: noise_level (≤ 55 dB at highest setting)
    noise_node = evaluator.add_leaf(
        id="noise_level",
        desc="The portable air conditioner must have a maximum noise level of 55 decibels (dB) or lower at its highest setting",
        parent=parent_node,
        critical=True,
    )
    claim_noise = f"At its highest setting, {model_label} has a maximum noise level of 55 dB (dBA) or lower."
    await evaluator.verify(
        claim=claim_noise,
        node=noise_node,
        sources=all_sources,
        additional_instruction=(
            "Use the maximum noise value (highest fan/compressor setting). If a range is given (e.g., 52–56 dB), "
            "the maximum is 56 and this fails. Accept '≤55 dB(A)' or similar."
        ),
    )

    # Leaf: energy_efficiency (CEER ≥ 8.5)
    ceer_node = evaluator.add_leaf(
        id="energy_efficiency",
        desc="The portable air conditioner must have a CEER (Combined Energy Efficiency Ratio) rating of at least 8.5",
        parent=parent_node,
        critical=True,
    )
    claim_ceer = f"{model_label} has a CEER (Combined Energy Efficiency Ratio) rating of at least 8.5."
    await evaluator.verify(
        claim=claim_ceer,
        node=ceer_node,
        sources=all_sources,
        additional_instruction=(
            "Look specifically for CEER. Do NOT substitute EER; if only EER is provided and CEER is absent, treat this as not meeting the requirement."
        ),
    )

    # Leaf: hose_configuration (dual-hose)
    hose_node = evaluator.add_leaf(
        id="hose_configuration",
        desc="The portable air conditioner must have a dual-hose configuration",
        parent=parent_node,
        critical=True,
    )
    claim_hose = f"{model_label} uses a dual-hose (two-hose) configuration."
    await evaluator.verify(
        claim=claim_hose,
        node=hose_node,
        sources=all_sources,
        additional_instruction="Confirm 'dual hose', 'two hoses', or an explicit statement indicating a dual-hose design.",
    )

    # Leaf: smart_features (Wi‑Fi and app control)
    smart_node = evaluator.add_leaf(
        id="smart_features",
        desc="The portable air conditioner must include Wi-Fi connectivity and app control capability",
        parent=parent_node,
        critical=True,
    )
    claim_smart = f"{model_label} supports Wi‑Fi connectivity and mobile app control."
    await evaluator.verify(
        claim=claim_smart,
        node=smart_node,
        sources=all_sources,
        additional_instruction=(
            "Both features are required: Wi‑Fi connectivity AND app/mobile control. "
            "Terms like 'smart app', 'remote app control', 'Wi‑Fi enabled', 'compatible with smartphone app' count. "
            "If only one of the two is present, this fails."
        ),
    )

    # Leaf: cooling_mode
    cooling_node = evaluator.add_leaf(
        id="cooling_mode",
        desc="The portable air conditioner must include a cooling mode",
        parent=parent_node,
        critical=True,
    )
    claim_cooling = f"{model_label} includes a cooling mode."
    await evaluator.verify(
        claim=claim_cooling,
        node=cooling_node,
        sources=all_sources,
        additional_instruction="Verify that 'cooling' mode is listed among the operating modes (e.g., Cooling/AC).",
    )

    # Leaf: fan_mode
    fan_node = evaluator.add_leaf(
        id="fan_mode",
        desc="The portable air conditioner must include a fan mode",
        parent=parent_node,
        critical=True,
    )
    claim_fan = f"{model_label} includes a fan mode."
    await evaluator.verify(
        claim=claim_fan,
        node=fan_node,
        sources=all_sources,
        additional_instruction="Verify that a 'fan' or 'fan-only' mode is listed among the operating modes.",
    )

    # Leaf: dehumidify_mode
    dehum_node = evaluator.add_leaf(
        id="dehumidify_mode",
        desc="The portable air conditioner must include a dehumidify or dry mode",
        parent=parent_node,
        critical=True,
    )
    claim_dehum = f"{model_label} includes a dehumidify (dry) mode."
    await evaluator.verify(
        claim=claim_dehum,
        node=dehum_node,
        sources=all_sources,
        additional_instruction="Verify that 'dehumidify', 'dry', or equivalent mode is listed among the operating modes.",
    )

    # Leaf: weight_limit (≤ 80 lbs)
    weight_node = evaluator.add_leaf(
        id="weight_limit",
        desc="The portable air conditioner must weigh 80 pounds or less",
        parent=parent_node,
        critical=True,
    )
    claim_weight = f"The product weight of {model_label} is 80 pounds or less."
    await evaluator.verify(
        claim=claim_weight,
        node=weight_node,
        sources=all_sources,
        additional_instruction=(
            "Use product/Net weight (not shipping weight) if available. If only shipping weight is shown and it exceeds 80 lbs, treat as failing. "
            "80.0 lbs exactly is acceptable."
        ),
    )

    # Leaf: price_range ($400–$700)
    price_node = evaluator.add_leaf(
        id="price_range",
        desc="The portable air conditioner must be priced between $400 and $700",
        parent=parent_node,
        critical=True,
    )
    claim_price = f"The current selling price for {model_label} is between $400 and $700 USD."
    await evaluator.verify(
        claim=claim_price,
        node=price_node,
        sources=all_sources,
        additional_instruction=(
            "Check the listed price on any of the provided product/retailer pages. Use current or sale price shown. "
            "If multiple variants exist, match the variant implied by the model name/number. "
            "Accept a price within the inclusive range $400–$700."
        ),
    )

    # Leaf: online_availability (major retailer or manufacturer site)
    availability_node = evaluator.add_leaf(
        id="online_availability",
        desc="The portable air conditioner must be available for purchase online from at least one major retailer (e.g., Amazon, Home Depot, Lowe's, manufacturer's website)",
        parent=parent_node,
        critical=True,
    )
    claim_availability = (
        f"{model_label} is available for purchase online from a major retailer (Amazon, Home Depot, Lowe's, Best Buy) "
        f"or the manufacturer's official website."
    )
    await evaluator.verify(
        claim=claim_availability,
        node=availability_node,
        sources=all_sources,
        additional_instruction=(
            "Confirm the page is a product listing with purchase options (e.g., Add to Cart/Buy/Price) on a recognized major retailer "
            "domain (amazon.com, homedepot.com, lowes.com, bestbuy.com) or the manufacturer's official site with direct purchase. "
            "If pages are informational only and do not offer purchase, this fails."
        ),
    )

    # Leaf: warranty (≥ 1-year manufacturer warranty)
    warranty_node = evaluator.add_leaf(
        id="warranty",
        desc="The portable air conditioner must include at least a 1-year manufacturer warranty",
        parent=parent_node,
        critical=True,
    )
    claim_warranty = f"{model_label} includes a manufacturer warranty of at least 1 year."
    await evaluator.verify(
        claim=claim_warranty,
        node=warranty_node,
        sources=all_sources,
        additional_instruction=(
            "Look for '1-year' or longer manufacturer warranty (limited warranty acceptable). "
            "If only satisfaction guarantee or return policy is present without manufacturer warranty, this fails."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the portable AC selection task using Mind2Web2 LLM-as-a-Judge framework.
    Builds a parallel, all-critical verification tree aligned with the rubric.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregating all checks in parallel
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

    # Create a top-level parallel node to mirror rubric root description
    rubric_root = evaluator.add_parallel(
        id="portable_ac_requirements",
        desc="Find a portable air conditioner that meets all specified requirements",
        parent=root,
        critical=False,  # Parent may be non-critical; all children below are critical leaves
    )

    # Extract structured info from answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_portable_ac(),
        template_class=PortableACInfo,
        extraction_name="portable_ac_extraction",
    )

    # Record custom info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "model_label": safe_model_label(extracted_info),
            "product_page_url": extracted_info.product_page_url,
            "retailer_urls": extracted_info.retailer_urls,
            "doe_sacc_btu": extracted_info.doe_sacc_btu,
            "coverage_sqft": extracted_info.coverage_sqft,
            "max_noise_db": extracted_info.max_noise_db,
            "ceer_rating": extracted_info.ceer_rating,
            "hose_configuration": extracted_info.hose_configuration,
            "wifi_connectivity": extracted_info.wifi_connectivity,
            "app_control": extracted_info.app_control,
            "includes_cooling_mode": extracted_info.includes_cooling_mode,
            "includes_fan_mode": extracted_info.includes_fan_mode,
            "includes_dehumidify_mode": extracted_info.includes_dehumidify_mode,
            "weight_lbs": extracted_info.weight_lbs,
            "price_usd": extracted_info.price_usd,
            "warranty_text": extracted_info.warranty_text,
        },
        info_type="extracted_debug",
        info_name="extracted_portable_ac_fields",
    )

    # Build and run verification leaves
    await build_and_verify_tree(evaluator, extracted_info, rubric_root)

    # Return the evaluation summary (score + verification tree + recorded info)
    return evaluator.get_summary()