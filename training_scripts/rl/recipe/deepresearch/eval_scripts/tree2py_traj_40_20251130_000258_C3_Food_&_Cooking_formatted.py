import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "thanksgiving_eve_bogo_2025"
TASK_DESCRIPTION = (
    "You are planning to take advantage of restaurant promotions during the Thanksgiving 2025 holiday period. "
    "Research and identify a major restaurant chain that is offering a buy-one-get-one-free (BOGO) entrée promotion "
    "specifically on Thanksgiving Eve (Wednesday, November 26, 2025) starting at 4:00 PM or later. For this restaurant chain, provide:\n\n"
    "1. The name of the restaurant chain\n"
    "2. An official URL from the restaurant's website or newsroom that documents this promotion\n"
    "3. The specific start time when the promotion becomes available on November 26, 2025\n"
    "4. The order channels where this promotion is valid (specify whether it is available for in-restaurant orders, online orders, mobile app orders, and/or delivery orders)\n"
    "5. The maximum number of free items allowed per single transaction or check\n"
    "6. Whether this restaurant chain is open or closed on Thanksgiving Day (Thursday, November 27, 2025)"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PromotionExtraction(BaseModel):
    # Identity and sourcing
    chain_name: Optional[str] = None
    official_url: Optional[str] = None
    additional_official_urls: List[str] = Field(default_factory=list)
    locations_url: Optional[str] = None

    # Promotion constraints & details
    bogo_offer_text: Optional[str] = None                    # e.g., "BOGO entrée", "Buy one get one free entrées"
    promo_date_text: Optional[str] = None                    # e.g., "Thanksgiving Eve, Wednesday, Nov 26, 2025"
    start_time_text: Optional[str] = None                    # e.g., "4 PM", "5:00 p.m.", "after 4 PM"
    order_channels_text: Optional[str] = None                # free-form description of channels
    in_restaurant: Optional[bool] = None
    online_ordering: Optional[bool] = None
    mobile_app_ordering: Optional[bool] = None
    delivery_ordering: Optional[bool] = None
    exclusions_text: Optional[str] = None                    # if any exclusions noted in answer
    max_free_items_text: Optional[str] = None                # e.g., "1 free entrée per check"
    thanksgiving_day_status_text: Optional[str] = None       # e.g., "Closed on Thanksgiving Day", "Open with special hours"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_promotion_details() -> str:
    return """
    Extract the restaurant and promotion information as presented in the answer. Return a JSON object with the following fields:

    Identity & Official Sourcing:
    - chain_name: The name of the restaurant chain.
    - official_url: A single official URL from the restaurant's website or official newsroom explicitly documenting the promotion.
    - additional_official_urls: Any other official restaurant URLs mentioned in the answer (e.g., newsroom posts, terms pages, FAQ, homepage).
    - locations_url: If the answer mentions an official locations or store locator URL, extract it here; otherwise, return null.

    Promotion Constraints & Required Details:
    - bogo_offer_text: The answer's wording indicating a buy-one-get-one-free entrée offer; include the exact phrasing from the answer.
    - promo_date_text: The answer's stated date for the promotion (e.g., "Wednesday, November 26, 2025", "Thanksgiving Eve").
    - start_time_text: The specific start time stated for Nov 26, 2025 (e.g., "4 PM", "5:00 PM"). If the answer says "after 4 PM" or "starting at 4 PM", include that exact phrase.
    - order_channels_text: The answer's description of eligible order channels and any exclusions (e.g., "valid for dine-in and online orders; delivery excluded").
    - in_restaurant: true/false if the answer explicitly states the promotion is valid for in-restaurant (dine-in) orders; null if unspecified.
    - online_ordering: true/false if valid for online web orders; null if unspecified.
    - mobile_app_ordering: true/false if valid via mobile app; null if unspecified.
    - delivery_ordering: true/false if valid for delivery; null if unspecified.
    - exclusions_text: Any exclusions or restrictions explicitly mentioned (e.g., "excludes delivery", "limit one free entrée per check").
    - max_free_items_text: The answer's stated maximum number of free items allowed per single transaction/check.
    - thanksgiving_day_status_text: The answer's statement about whether the chain is open or closed on Thanksgiving Day (Thursday, Nov 27, 2025). Return the exact wording from the answer.

    IMPORTANT:
    - Only extract information explicitly present in the answer. Do not infer or invent.
    - For URLs, include only official restaurant domains (e.g., brand.com, newsroom.brand.com, store-locator.brand.com). If no official URL is present, return null or an empty list as applicable.
    - If a field is not mentioned, return null (or empty list for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    uniq = []
    seen = set()
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def get_official_urls(extracted: PromotionExtraction) -> List[str]:
    urls = []
    if extracted.official_url:
        urls.append(extracted.official_url)
    if extracted.locations_url:
        urls.append(extracted.locations_url)
    urls.extend(extracted.additional_official_urls or [])
    return _dedup_urls(urls)


def format_channels(extracted: PromotionExtraction) -> str:
    parts = []
    if extracted.in_restaurant is True:
        parts.append("in-restaurant (dine-in)")
    elif extracted.in_restaurant is False:
        parts.append("in-restaurant: not eligible")

    if extracted.online_ordering is True:
        parts.append("online ordering")
    elif extracted.online_ordering is False:
        parts.append("online ordering: not eligible")

    if extracted.mobile_app_ordering is True:
        parts.append("mobile app ordering")
    elif extracted.mobile_app_ordering is False:
        parts.append("mobile app ordering: not eligible")

    if extracted.delivery_ordering is True:
        parts.append("delivery orders")
    elif extracted.delivery_ordering is False:
        parts.append("delivery orders: not eligible")

    # If no explicit booleans, fall back to free-form text
    if not parts and extracted.order_channels_text:
        return extracted.order_channels_text.strip()

    if parts:
        text = ", ".join(parts)
        if extracted.exclusions_text:
            text += f"; exclusions: {extracted.exclusions_text.strip()}"
        return text

    return extracted.order_channels_text.strip() if extracted.order_channels_text else "unspecified"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_official_sourcing_group(
    evaluator: Evaluator,
    parent_node,
    extracted: PromotionExtraction,
):
    """
    Build the 'official_sourcing' group and return the nodes for later use.
    """
    official_node = evaluator.add_parallel(
        id="official_sourcing",
        desc="All claims are supported by official restaurant website or official newsroom sources",
        parent=parent_node,
        critical=True,
    )

    # Leaf 1: Official URL provided (existence check)
    has_official = bool(extracted.official_url) or bool(extracted.additional_official_urls)
    official_url_node = evaluator.add_custom_node(
        result=has_official,
        id="official_url_provided",
        desc="Provides at least one official restaurant website or official newsroom URL",
        parent=official_node,
        critical=True
    )

    # Leaf 2: Official sources support all reported details (verification later)
    support_all_node = evaluator.add_leaf(
        id="official_sources_support_all_reported_details",
        desc="Official source URL(s) explicitly support the promotion and each reported required attribute",
        parent=official_node,
        critical=True
    )

    return official_node, official_url_node, support_all_node


async def build_chain_identity_group(
    evaluator: Evaluator,
    parent_node,
    extracted: PromotionExtraction,
    official_url_prereq_node,
):
    """
    Build and verify the 'chain_identity_and_scale' group.
    """
    chain_node = evaluator.add_parallel(
        id="chain_identity_and_scale",
        desc="Restaurant chain is identified and meets the 'major chain with multiple U.S. locations' constraint",
        parent=parent_node,
        critical=True
    )

    # Leaf: Restaurant name provided (existence)
    name_provided = bool(extracted.chain_name and extracted.chain_name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id="restaurant_name_provided",
        desc="Provides the name of the restaurant chain",
        parent=chain_node,
        critical=True
    )

    # Leaf: Multiple U.S. locations evidence (verification by official sources)
    multi_loc_node = evaluator.add_leaf(
        id="multiple_us_locations_evidence",
        desc="Provides verifiable evidence that the chain operates multiple U.S. locations",
        parent=chain_node,
        critical=True
    )

    claim = (
        f"{extracted.chain_name or 'The chain'} operates multiple locations across the United States "
        f"(i.e., is a major chain, not a single-location restaurant). Look for an official store locator, "
        f"locations page, or official wording indicating multiple U.S. locations or nationwide presence."
    )
    sources = get_official_urls(extracted)

    await evaluator.verify(
        claim=claim,
        node=multi_loc_node,
        sources=sources,
        additional_instruction=(
            "Verify using official restaurant pages only (store locator, locations page, newsroom statements, "
            "or official site text like 'nationwide', 'multiple locations', 'find a location'). "
            "If no official source evidences multiple U.S. locations, the claim is not supported."
        ),
        extra_prerequisites=[official_url_prereq_node]
    )


async def build_promotion_constraints_group(
    evaluator: Evaluator,
    parent_node,
    extracted: PromotionExtraction,
    official_url_prereq_node,
):
    """
    Build and verify the 'promotion_constraints_and_required_details' group.
    """
    promo_node = evaluator.add_parallel(
        id="promotion_constraints_and_required_details",
        desc="Promotion meets all constraints and all requested attributes are provided",
        parent=parent_node,
        critical=True
    )

    sources = get_official_urls(extracted)

    # Leaf: BOGO entrée offer
    bogo_node = evaluator.add_leaf(
        id="bogo_entree_offer",
        desc="Promotion is explicitly a buy-one-get-one-free (BOGO) entrée offer",
        parent=promo_node,
        critical=True
    )
    bogo_claim = (
        "The promotion is a buy-one-get-one-free (BOGO) offer specifically for entrée items (not sides or beverages). "
        f"Answer text indicates: {extracted.bogo_offer_text or 'unspecified'}."
    )
    await evaluator.verify(
        claim=bogo_claim,
        node=bogo_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the official page states a BOGO offer for entrées. Synonyms like 'Buy One, Get One Free' "
            "are acceptable, but the item category must be entrées."
        ),
        extra_prerequisites=[official_url_prereq_node]
    )

    # Leaf: Thanksgiving Eve date (Wednesday, November 26, 2025)
    date_node = evaluator.add_leaf(
        id="thanksgiving_eve_date",
        desc="Promotion is available specifically on Thanksgiving Eve (Wednesday, November 26, 2025)",
        parent=promo_node,
        critical=True
    )
    date_claim = (
        "The promotion is available specifically on Thanksgiving Eve, Wednesday, November 26, 2025."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=sources,
        additional_instruction=(
            "Confirm the official page explicitly mentions the promotion date as Wednesday, Nov 26, 2025, "
            "or 'Thanksgiving Eve' with the 2025 date."
        ),
        extra_prerequisites=[official_url_prereq_node]
    )

    # Leaf: Start time specified and at/after 4:00 PM
    start_node = evaluator.add_leaf(
        id="start_time_specified_and_ge_4pm",
        desc="Provides a specific start time for Nov 26, 2025, and that start time is 4:00 PM local time or later",
        parent=promo_node,
        critical=True
    )
    start_claim = (
        f"The promotion starts at {extracted.start_time_text or 'an explicitly stated time'} on November 26, 2025, "
        f"and that start time is 4:00 PM local time or later."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_node,
        sources=sources,
        additional_instruction=(
            "Check that the official page specifies a start time on Nov 26, 2025, and confirm that the start time is "
            ">= 4:00 PM local time. Phrases like 'after 4 PM' or 'starting at 5 PM' should be considered compliant."
        ),
        extra_prerequisites=[official_url_prereq_node]
    )

    # Leaf: Order channels specified
    channels_node = evaluator.add_leaf(
        id="order_channels_specified",
        desc="Specifies which order channels are eligible (in-restaurant, online, mobile app, and/or delivery), including exclusions if any",
        parent=promo_node,
        critical=True
    )
    channels_text = format_channels(extracted)
    channels_claim = (
        f"The official page specifies eligible order channels for the promotion on Nov 26, 2025: {channels_text}."
    )
    await evaluator.verify(
        claim=channels_claim,
        node=channels_node,
        sources=sources,
        additional_instruction=(
            "Verify that the official source clearly states which order channels are eligible (e.g., dine-in, online, "
            "mobile app, delivery) and any exclusions (e.g., 'delivery excluded'). If channels are not specified on the "
            "official page, the claim is not supported."
        ),
        extra_prerequisites=[official_url_prereq_node]
    )

    # Leaf: Max free items per transaction/check specified
    limit_node = evaluator.add_leaf(
        id="max_free_items_limit_specified",
        desc="Specifies the maximum number of free items allowed per single transaction/check",
        parent=promo_node,
        critical=True
    )
    limit_claim = (
        f"The official page states the maximum number of free items allowed per single transaction/check: "
        f"{extracted.max_free_items_text or 'unspecified'}."
    )
    await evaluator.verify(
        claim=limit_claim,
        node=limit_node,
        sources=sources,
        additional_instruction=(
            "Verify that the official page explicitly mentions a limit (e.g., 'limit one free entrée per check'). "
            "If no such limit is stated, the claim is not supported."
        ),
        extra_prerequisites=[official_url_prereq_node]
    )

    # Leaf: Thanksgiving Day open/closed status specified
    tgday_node = evaluator.add_leaf(
        id="thanksgiving_day_open_closed_specified",
        desc="Clearly states whether the chain is open or closed on Thanksgiving Day (Thursday, November 27, 2025)",
        parent=promo_node,
        critical=True
    )
    tg_status_text = extracted.thanksgiving_day_status_text or "unspecified"
    tgday_claim = (
        f"The official page clearly states whether the chain is open or closed on Thanksgiving Day "
        f"(Thursday, November 27, 2025): {tg_status_text}."
    )
    await evaluator.verify(
        claim=tgday_claim,
        node=tgday_node,
        sources=sources,
        additional_instruction=(
            "Confirm on official sources whether the chain is open or closed on Thanksgiving Day 2025. "
            "If not stated on official sources, the claim is not supported."
        ),
        extra_prerequisites=[official_url_prereq_node]
    )


async def verify_official_support_all_details(
    evaluator: Evaluator,
    support_all_node,
    extracted: PromotionExtraction,
):
    """
    Verify that official sources support all reported details in a single combined check.
    """
    sources = get_official_urls(extracted)
    combined_claim = (
        "Official restaurant sources explicitly confirm ALL of the following for the promotion:\n"
        f"- BOGO entrée offer: {extracted.bogo_offer_text or 'unspecified'}\n"
        "- Date: Wednesday, November 26, 2025 (Thanksgiving Eve)\n"
        f"- Start time on Nov 26, 2025: {extracted.start_time_text or 'unspecified'}, and it is at or after 4:00 PM local time\n"
        f"- Eligible order channels: {format_channels(extracted)}\n"
        f"- Maximum free items per single transaction/check: {extracted.max_free_items_text or 'unspecified'}\n"
        f"- Thanksgiving Day (Nov 27, 2025) open/closed status: {extracted.thanksgiving_day_status_text or 'unspecified'}\n"
        "If ANY of the above items is not supported by the official sources, return NOT SUPPORTED."
    )

    await evaluator.verify(
        claim=combined_claim,
        node=support_all_node,
        sources=sources,
        additional_instruction=(
            "Cross-check each bullet against the official webpage(s)/newsroom. If even one bullet is missing or contradicted, "
            "the verification should fail."
        )
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Thanksgiving Eve 2025 BOGO entrée promotion task.
    """

    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
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

    # Create a critical task node under the root to satisfy the rubric root being critical
    task_node = evaluator.add_parallel(
        id="task_root",
        desc="Answer identifies one qualifying major U.S. restaurant chain with a Thanksgiving Eve BOGO entrée promo and provides all required promo details with official sourcing",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_promotion_details(),
        template_class=PromotionExtraction,
        extraction_name="promotion_extraction"
    )

    # Record expected constraints for transparency
    evaluator.add_ground_truth({
        "required_date": "Wednesday, November 26, 2025 (Thanksgiving Eve)",
        "required_start_time_threshold": ">= 4:00 PM local time",
        "required_offer": "BOGO entrée",
        "required_details": [
            "Chain name",
            "Official URL",
            "Start time on Nov 26, 2025",
            "Order channels and any exclusions",
            "Max free items per transaction/check",
            "Thanksgiving Day (Nov 27, 2025) open/closed status"
        ]
    }, gt_type="constraints")

    # Build official sourcing group first (to use its 'official_url_provided' as prerequisite)
    official_node, official_url_prereq_node, support_all_node = await build_official_sourcing_group(
        evaluator, task_node, extracted
    )

    # Build chain identity & scale group
    await build_chain_identity_group(
        evaluator, task_node, extracted, official_url_prereq_node
    )

    # Build promotion constraints & required details group
    await build_promotion_constraints_group(
        evaluator, task_node, extracted, official_url_prereq_node
    )

    # Verify the combined official support for all reported details
    await verify_official_support_all_details(
        evaluator, support_all_node, extracted
    )

    # Return summary
    return evaluator.get_summary()