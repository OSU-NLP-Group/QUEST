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
TASK_ID = "holiday_grocery_2025"
TASK_DESCRIPTION = (
    "Identify 4 different major grocery store chains operating in the United States that each match one of the following "
    "distinct holiday operating patterns for Thanksgiving 2025 (November 27) and/or Christmas 2025 (December 25):\n\n"
    "Pattern A: A chain that closed ALL store locations on Christmas Day 2025\n"
    "Pattern B: A chain that kept select locations open on Christmas Day 2025 with reduced operating hours (must specify the hours)\n"
    "Pattern C: A chain whose stores operated on Thanksgiving Day 2025 with closing time at or before 4:00 PM (must specify the hours)\n"
    "Pattern D: A chain whose stores operated on Thanksgiving Day 2025 with closing time at or before 2:00 PM (must specify the hours)\n\n"
    "Additional Requirements (all 4 chains must meet these):\n"
    "- Must offer grocery delivery services through their own platform or a third-party service (e.g., Instacart)\n"
    "- Must have a stated minimum order requirement for delivery of $35 or less\n"
    "- Must have publicly available, verifiable information about their holiday operating hours from official company sources or reliable news outlets\n\n"
    "For each of the 4 grocery chains you identify, provide:\n"
    "1. The chain name\n"
    "2. Which pattern (A, B, C, or D) it matches\n"
    "3. The specific holiday operating hours or closure policy that qualifies it for that pattern\n"
    "4. The grocery delivery service they offer and the minimum order requirement\n"
    "5. Reference URL(s) from official sources or reliable news outlets confirming the information\n\n"
    "Note: Each of the 4 chains you identify must match a DIFFERENT pattern (one chain for Pattern A, one for Pattern B, one for Pattern C, and one for Pattern D)."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ChainHolidayHours(BaseModel):
    holiday: Optional[str] = None  # "Christmas 2025" or "Thanksgiving 2025"
    policy_summary: Optional[str] = None  # e.g., "All locations closed", "Select locations open 8am-3pm"
    open_time: Optional[str] = None  # e.g., "8:00 AM"
    close_time: Optional[str] = None  # e.g., "3:00 PM"
    select_locations_only: Optional[bool] = None  # For Pattern B
    closure_all_day: Optional[bool] = None  # For Pattern A
    urls: List[str] = Field(default_factory=list)  # Holiday hours/reference URLs


class DeliveryInfo(BaseModel):
    service_name: Optional[str] = None  # e.g., "Instacart", "Shipt", "DoorDash", or "Own delivery"
    min_order: Optional[str] = None  # e.g., "$35", "35 dollars", "Minimum order $30"
    urls: List[str] = Field(default_factory=list)  # Delivery policy/FAQ/terms URLs


class ChainItem(BaseModel):
    name: Optional[str] = None
    pattern: Optional[str] = None  # 'A', 'B', 'C', or 'D'
    christmas: Optional[ChainHolidayHours] = None
    thanksgiving: Optional[ChainHolidayHours] = None
    delivery: Optional[DeliveryInfo] = None


class ChainsExtraction(BaseModel):
    chains: List[ChainItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chains() -> str:
    return (
        "From the provided answer, extract details for up to 4 major U.S. grocery store chains, each matching a unique "
        "holiday operating pattern among {A, B, C, D}. Extract only one chain per pattern; if multiple are mentioned for the "
        "same pattern, pick the first one.\n\n"
        "For each chain, return the following structured fields:\n"
        "- name: The chain name.\n"
        "- pattern: One of 'A', 'B', 'C', or 'D', corresponding to the described patterns.\n"
        "- christmas: Holiday hours object ONLY IF the chain's qualifying information relates to Christmas Day 2025. Otherwise, set to null.\n"
        "  - holiday: Exactly 'Christmas 2025' if applicable.\n"
        "  - policy_summary: A concise summary of the policy (e.g., 'All locations closed', 'Select locations open 9am-3pm').\n"
        "  - open_time: Opening time if hours are specified (e.g., '9:00 AM'); null if closed all day or not applicable.\n"
        "  - close_time: Closing time if hours are specified (e.g., '3:00 PM'); null if closed all day or not applicable.\n"
        "  - select_locations_only: true if only select locations are open; false otherwise; null if not applicable.\n"
        "  - closure_all_day: true if all locations are closed all day; false otherwise; null if not applicable.\n"
        "  - urls: An array of URL strings from official company sources or reliable news outlets that confirm the policy.\n"
        "- thanksgiving: Holiday hours object ONLY IF the chain's qualifying information relates to Thanksgiving Day 2025. Otherwise, set to null.\n"
        "  - holiday: Exactly 'Thanksgiving 2025' if applicable.\n"
        "  - policy_summary: A concise summary of the policy.\n"
        "  - open_time: Opening time where hours are specified.\n"
        "  - close_time: Closing time where hours are specified.\n"
        "  - select_locations_only: null for Thanksgiving unless explicitly stated.\n"
        "  - closure_all_day: null for Thanksgiving unless explicitly stated.\n"
        "  - urls: An array of official/reliable URLs confirming the Thanksgiving policy.\n"
        "- delivery: Delivery information object.\n"
        "  - service_name: The delivery service used (e.g., 'Instacart', 'Shipt', 'DoorDash', or 'Own platform').\n"
        "  - min_order: The minimum order requirement value as mentioned (string; e.g., '$35', '30 dollars'). If not stated, set to null.\n"
        "  - urls: URL(s) to official company pages or reliable sources confirming delivery availability and/or minimum order.\n\n"
        "Rules and clarifications:\n"
        "1) Extract URLs only if they are explicitly present in the answer text. Use full URLs; convert markdown links to plain URLs.\n"
        "2) Prefer official company pages or reliable news outlets; avoid casual blogs. If multiple holiday URLs are present, include all.\n"
        "3) For Pattern A (Christmas): closure_all_day must be true and select_locations_only should be false or null.\n"
        "4) For Pattern B (Christmas): select_locations_only must be true, and both open_time and close_time must be provided for reduced hours.\n"
        "5) For Pattern C (Thanksgiving): include both open_time and close_time; the close_time must be at or before 4:00 PM.\n"
        "6) For Pattern D (Thanksgiving): include both open_time and close_time; the close_time must be at or before 2:00 PM.\n"
        "7) Delivery min_order should be the value stated; do not infer. If the answer only states '≤ $35' without a specific value, set min_order to that text.\n"
        "8) If any required information is missing, set the field to null (or empty list for URLs).\n"
        "9) Return a JSON object with a 'chains' array containing up to 4 ChainItem objects."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def pick_first_chain_for_pattern(extraction: ChainsExtraction, pattern_code: str) -> ChainItem:
    for item in extraction.chains:
        if item.pattern and item.pattern.strip().upper() == pattern_code:
            return item
    return ChainItem()  # Empty placeholder if not found


def _safe_holiday_urls(h: Optional[ChainHolidayHours]) -> List[str]:
    return (h.urls if h and h.urls else [])


def _safe_delivery_urls(d: Optional[DeliveryInfo]) -> List[str]:
    return (d.urls if d and d.urls else [])


# --------------------------------------------------------------------------- #
# Verification routines for each pattern                                      #
# --------------------------------------------------------------------------- #
async def verify_pattern_a(evaluator: Evaluator, parent_node, chain: ChainItem) -> None:
    """
    Pattern A: A chain that closed ALL store locations on Christmas Day 2025.
    """
    node = evaluator.add_sequential(
        id="Pattern_A_Chain",
        desc="Identify and verify one chain that closed ALL locations on Christmas Day 2025",
        parent=parent_node,
        critical=False
    )

    # 1) Chain name provided
    evaluator.add_custom_node(
        result=bool(chain.name and chain.name.strip()),
        id="Pattern_A_Chain_Name",
        desc="Provide the name of a major grocery chain that was closed on Christmas Day 2025",
        parent=node,
        critical=True
    )

    # 2) Verify ALL locations closed on Christmas 2025
    closure_node = evaluator.add_leaf(
        id="Pattern_A_Christmas_Closure_Verification",
        desc="Verify that ALL store locations were closed on Christmas Day 2025 (not just select locations)",
        parent=node,
        critical=True
    )
    christmas_urls = _safe_holiday_urls(chain.christmas)
    closure_claim = (
        f"All {chain.name or 'the chain'} store locations were closed on Christmas Day 2025 (December 25, 2025)."
    )
    await evaluator.verify(
        claim=closure_claim,
        node=closure_node,
        sources=christmas_urls,
        additional_instruction=(
            "Confirm the page explicitly states that all stores were closed on Christmas Day 2025. "
            "Statements like 'all locations closed' or the chain's blanket closure policy qualify."
        )
    )

    # 3) Holiday reference URL credibility and relevance
    holiday_ref_node = evaluator.add_leaf(
        id="Pattern_A_Holiday_Reference_URL",
        desc="Provide reference URL from official source or reliable news outlet confirming Christmas Day closure policy",
        parent=node,
        critical=True
    )
    holiday_ref_claim = (
        f"The provided page(s) are official company sources or reliable news outlets and they clearly confirm "
        f"the Christmas Day 2025 closure policy for {chain.name or 'the chain'}."
    )
    await evaluator.verify(
        claim=holiday_ref_claim,
        node=holiday_ref_node,
        sources=christmas_urls,
        additional_instruction=(
            "Evaluate whether the URL is from an official company domain or a well-known reliable news outlet, "
            "and whether it explicitly confirms the specific holiday policy for 2025."
        )
    )

    # 4) Delivery service availability
    delivery_service_node = evaluator.add_leaf(
        id="Pattern_A_Delivery_Service_Available",
        desc="Verify the chain offers grocery delivery services (own platform or third-party like Instacart)",
        parent=node,
        critical=True
    )
    delivery_urls = _safe_delivery_urls(chain.delivery)
    delivery_claim = (
        f"{chain.name or 'The chain'} offers grocery delivery services via "
        f"{chain.delivery.service_name if chain.delivery and chain.delivery.service_name else 'a delivery platform'}."
    )
    await evaluator.verify(
        claim=delivery_claim,
        node=delivery_service_node,
        sources=delivery_urls,
        additional_instruction=(
            "Accept official company delivery pages, FAQs, or trusted partner listings (e.g., Instacart, Shipt, DoorDash)."
        )
    )

    # 5) Delivery minimum order requirement is $35 or less
    delivery_min_node = evaluator.add_leaf(
        id="Pattern_A_Delivery_Minimum_Order",
        desc="Verify and specify the delivery minimum order requirement is $35 or less",
        parent=node,
        critical=True
    )
    min_order_text = chain.delivery.min_order if chain.delivery and chain.delivery.min_order else ""
    delivery_min_claim = (
        f"The delivery minimum order requirement for {chain.name or 'the chain'} is {min_order_text}, which is $35 or less."
    )
    await evaluator.verify(
        claim=delivery_min_claim,
        node=delivery_min_node,
        sources=delivery_urls,
        additional_instruction=(
            "Check the page for the minimum order value for grocery delivery. Confirm that it is at most $35 "
            "(e.g., $35, $30, $25 are valid; $40 is not). If the page states a threshold '≥ $35' for free delivery but "
            "no minimum order, do not consider it valid."
        )
    )

    # 6) Delivery reference URL presence and relevance (existence check)
    evaluator.add_custom_node(
        result=bool(delivery_urls),
        id="Pattern_A_Delivery_Reference_URL",
        desc="Provide reference URL confirming delivery service availability and minimum order requirement",
        parent=node,
        critical=True
    )


async def verify_pattern_b(evaluator: Evaluator, parent_node, chain: ChainItem) -> None:
    """
    Pattern B: Select locations open on Christmas Day 2025 with reduced hours (must specify hours).
    """
    node = evaluator.add_sequential(
        id="Pattern_B_Chain",
        desc="Identify and verify one chain with select locations open on Christmas Day 2025 with reduced hours",
        parent=parent_node,
        critical=False
    )

    # 1) Chain name
    evaluator.add_custom_node(
        result=bool(chain.name and chain.name.strip()),
        id="Pattern_B_Chain_Name",
        desc="Provide the name of a major grocery chain with select locations open on Christmas Day 2025",
        parent=node,
        critical=True
    )

    # 2) Verify only select locations open
    select_node = evaluator.add_leaf(
        id="Pattern_B_Select_Locations_Only",
        desc="Verify that only SELECT locations (not all) were open on Christmas Day 2025",
        parent=node,
        critical=True
    )
    christmas_urls = _safe_holiday_urls(chain.christmas)
    select_claim = (
        f"On Christmas Day 2025, only select locations of {chain.name or 'the chain'} were open (not all stores)."
    )
    await evaluator.verify(
        claim=select_claim,
        node=select_node,
        sources=christmas_urls,
        additional_instruction=(
            "Look for language such as 'select stores', 'limited locations', or 'some locations'. "
            "Statements indicating all stores open should fail."
        )
    )

    # 3) Specify reduced hours (must include opening and closing times)
    hours_spec_node = evaluator.add_leaf(
        id="Pattern_B_Christmas_Hours_Specified",
        desc="Specify the reduced operating hours for Christmas Day 2025 (must include both opening and closing times)",
        parent=node,
        critical=True
    )
    open_t = chain.christmas.open_time if chain.christmas else None
    close_t = chain.christmas.close_time if chain.christmas else None
    hours_claim = (
        f"On Christmas Day 2025, select locations of {chain.name or 'the chain'} operated reduced hours "
        f"from {open_t or 'UNKNOWN'} to {close_t or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=hours_claim,
        node=hours_spec_node,
        sources=christmas_urls,
        additional_instruction=(
            "Confirm that both an opening time and a closing time for Christmas Day are explicitly stated. "
            "Minor format variations (e.g., '8am-3pm') are acceptable."
        )
    )

    # 4) Holiday reference URL credibility
    holiday_ref_node = evaluator.add_leaf(
        id="Pattern_B_Holiday_Reference_URL",
        desc="Provide reference URL from official source or reliable news outlet confirming Christmas Day operating hours",
        parent=node,
        critical=True
    )
    holiday_ref_claim = (
        f"The provided page(s) are official company sources or reliable news outlets and they clearly confirm "
        f"Christmas Day 2025 operating hours for {chain.name or 'the chain'}."
    )
    await evaluator.verify(
        claim=holiday_ref_claim,
        node=holiday_ref_node,
        sources=christmas_urls,
        additional_instruction=(
            "Assess whether the domain is official or a well-known news outlet, and whether the page explicitly "
            "states the Christmas Day hours."
        )
    )

    # 5) Delivery service available
    delivery_urls = _safe_delivery_urls(chain.delivery)
    delivery_service_node = evaluator.add_leaf(
        id="Pattern_B_Delivery_Service_Available",
        desc="Verify the chain offers grocery delivery services (own platform or third-party like Instacart)",
        parent=node,
        critical=True
    )
    delivery_claim = (
        f"{chain.name or 'The chain'} offers grocery delivery services via "
        f"{chain.delivery.service_name if chain.delivery and chain.delivery.service_name else 'a delivery platform'}."
    )
    await evaluator.verify(
        claim=delivery_claim,
        node=delivery_service_node,
        sources=delivery_urls,
        additional_instruction="Official delivery pages, FAQs, or trusted partner listings are acceptable evidence."
    )

    # 6) Delivery minimum order requirement ≤ $35
    delivery_min_node = evaluator.add_leaf(
        id="Pattern_B_Delivery_Minimum_Order",
        desc="Verify and specify the delivery minimum order requirement is $35 or less",
        parent=node,
        critical=True
    )
    min_order_text = chain.delivery.min_order if chain.delivery and chain.delivery.min_order else ""
    delivery_min_claim = (
        f"The delivery minimum order requirement for {chain.name or 'the chain'} is {min_order_text}, which is $35 or less."
    )
    await evaluator.verify(
        claim=delivery_min_claim,
        node=delivery_min_node,
        sources=delivery_urls,
        additional_instruction=(
            "Confirm the minimum order value on the page and judge whether it is at most $35."
        )
    )

    # 7) Delivery reference URL existence (must provide URL)
    evaluator.add_custom_node(
        result=bool(delivery_urls),
        id="Pattern_B_Delivery_Reference_URL",
        desc="Provide reference URL confirming delivery service availability and minimum order requirement",
        parent=node,
        critical=True
    )


async def verify_pattern_c(evaluator: Evaluator, parent_node, chain: ChainItem) -> None:
    """
    Pattern C: Thanksgiving Day 2025 hours with closing time at or before 4:00 PM (must specify hours).
    """
    node = evaluator.add_sequential(
        id="Pattern_C_Chain",
        desc="Identify and verify one chain open on Thanksgiving 2025 with closing time at or before 4:00 PM",
        parent=parent_node,
        critical=False
    )

    # 1) Chain name
    evaluator.add_custom_node(
        result=bool(chain.name and chain.name.strip()),
        id="Pattern_C_Chain_Name",
        desc="Provide the name of a major grocery chain open on Thanksgiving Day 2025 with closing time at or before 4:00 PM",
        parent=node,
        critical=True
    )

    # 2) Thanksgiving hours verified (closing time ≤ 4:00 PM)
    tg_urls = _safe_holiday_urls(chain.thanksgiving)
    hours_node = evaluator.add_leaf(
        id="Pattern_C_Thanksgiving_Hours_Verified",
        desc="Specify the Thanksgiving Day 2025 operating hours and confirm closing time is at or before 4:00 PM",
        parent=node,
        critical=True
    )
    open_t = chain.thanksgiving.open_time if chain.thanksgiving else None
    close_t = chain.thanksgiving.close_time if chain.thanksgiving else None
    hours_claim = (
        f"On Thanksgiving Day 2025 (November 27, 2025), {chain.name or 'the chain'} stores operated from "
        f"{open_t or 'UNKNOWN'} to {close_t or 'UNKNOWN'}, and the closing time is at or before 4:00 PM."
    )
    await evaluator.verify(
        claim=hours_claim,
        node=hours_node,
        sources=tg_urls,
        additional_instruction=(
            "Confirm both the opening and closing times for Thanksgiving Day 2025 and ensure the stated closing time is ≤ 4:00 PM."
        )
    )

    # 3) Holiday reference URL credibility
    holiday_ref_node = evaluator.add_leaf(
        id="Pattern_C_Holiday_Reference_URL",
        desc="Provide reference URL from official source or reliable news outlet confirming Thanksgiving Day operating hours",
        parent=node,
        critical=True
    )
    holiday_ref_claim = (
        f"The provided page(s) are official company sources or reliable news outlets and they clearly confirm "
        f"Thanksgiving Day 2025 operating hours for {chain.name or 'the chain'}."
    )
    await evaluator.verify(
        claim=holiday_ref_claim,
        node=holiday_ref_node,
        sources=tg_urls,
        additional_instruction=(
            "Assess whether the domain is official or a well-known reliable news outlet, and whether the page explicitly "
            "states the Thanksgiving Day hours for 2025."
        )
    )

    # 4) Delivery service available
    delivery_urls = _safe_delivery_urls(chain.delivery)
    delivery_service_node = evaluator.add_leaf(
        id="Pattern_C_Delivery_Service_Available",
        desc="Verify the chain offers grocery delivery services (own platform or third-party like Instacart)",
        parent=node,
        critical=True
    )
    delivery_claim = (
        f"{chain.name or 'The chain'} offers grocery delivery services via "
        f"{chain.delivery.service_name if chain.delivery and chain.delivery.service_name else 'a delivery platform'}."
    )
    await evaluator.verify(
        claim=delivery_claim,
        node=delivery_service_node,
        sources=delivery_urls,
        additional_instruction="Official delivery pages, FAQs, or trusted partner listings are acceptable evidence."
    )

    # 5) Delivery minimum order requirement ≤ $35
    delivery_min_node = evaluator.add_leaf(
        id="Pattern_C_Delivery_Minimum_Order",
        desc="Verify and specify the delivery minimum order requirement is $35 or less",
        parent=node,
        critical=True
    )
    min_order_text = chain.delivery.min_order if chain.delivery and chain.delivery.min_order else ""
    delivery_min_claim = (
        f"The delivery minimum order requirement for {chain.name or 'the chain'} is {min_order_text}, which is $35 or less."
    )
    await evaluator.verify(
        claim=delivery_min_claim,
        node=delivery_min_node,
        sources=delivery_urls,
        additional_instruction="Confirm the minimum order value on the page and judge that it is ≤ $35."
    )

    # 6) Delivery reference URL existence (must provide URL)
    evaluator.add_custom_node(
        result=bool(delivery_urls),
        id="Pattern_C_Delivery_Reference_URL",
        desc="Provide reference URL confirming delivery service availability and minimum order requirement",
        parent=node,
        critical=True
    )


async def verify_pattern_d(evaluator: Evaluator, parent_node, chain: ChainItem) -> None:
    """
    Pattern D: Thanksgiving Day 2025 hours with closing time at or before 2:00 PM (must specify hours).
    """
    node = evaluator.add_sequential(
        id="Pattern_D_Chain",
        desc="Identify and verify one chain open on Thanksgiving 2025 with closing time at or before 2:00 PM",
        parent=parent_node,
        critical=False
    )

    # 1) Chain name
    evaluator.add_custom_node(
        result=bool(chain.name and chain.name.strip()),
        id="Pattern_D_Chain_Name",
        desc="Provide the name of a major grocery chain open on Thanksgiving Day 2025 with closing time at or before 2:00 PM",
        parent=node,
        critical=True
    )

    # 2) Thanksgiving hours verified (closing time ≤ 2:00 PM)
    tg_urls = _safe_holiday_urls(chain.thanksgiving)
    hours_node = evaluator.add_leaf(
        id="Pattern_D_Thanksgiving_Hours_Verified",
        desc="Specify the Thanksgiving Day 2025 operating hours and confirm closing time is at or before 2:00 PM",
        parent=node,
        critical=True
    )
    open_t = chain.thanksgiving.open_time if chain.thanksgiving else None
    close_t = chain.thanksgiving.close_time if chain.thanksgiving else None
    hours_claim = (
        f"On Thanksgiving Day 2025 (November 27, 2025), {chain.name or 'the chain'} stores operated from "
        f"{open_t or 'UNKNOWN'} to {close_t or 'UNKNOWN'}, and the closing time is at or before 2:00 PM."
    )
    await evaluator.verify(
        claim=hours_claim,
        node=hours_node,
        sources=tg_urls,
        additional_instruction=(
            "Confirm both opening and closing times for Thanksgiving Day 2025 and ensure the stated closing time is ≤ 2:00 PM."
        )
    )

    # 3) Holiday reference URL credibility
    holiday_ref_node = evaluator.add_leaf(
        id="Pattern_D_Holiday_Reference_URL",
        desc="Provide reference URL from official source or reliable news outlet confirming Thanksgiving Day operating hours",
        parent=node,
        critical=True
    )
    holiday_ref_claim = (
        f"The provided page(s) are official company sources or reliable news outlets and they clearly confirm "
        f"Thanksgiving Day 2025 operating hours for {chain.name or 'the chain'}."
    )
    await evaluator.verify(
        claim=holiday_ref_claim,
        node=holiday_ref_node,
        sources=tg_urls,
        additional_instruction=(
            "Assess whether the domain is official or a well-known reliable news outlet, and whether the page explicitly "
            "states the Thanksgiving Day hours for 2025."
        )
    )

    # 4) Delivery service available
    delivery_urls = _safe_delivery_urls(chain.delivery)
    delivery_service_node = evaluator.add_leaf(
        id="Pattern_D_Delivery_Service_Available",
        desc="Verify the chain offers grocery delivery services (own platform or third-party like Instacart)",
        parent=node,
        critical=True
    )
    delivery_claim = (
        f"{chain.name or 'The chain'} offers grocery delivery services via "
        f"{chain.delivery.service_name if chain.delivery and chain.delivery.service_name else 'a delivery platform'}."
    )
    await evaluator.verify(
        claim=delivery_claim,
        node=delivery_service_node,
        sources=delivery_urls,
        additional_instruction="Official delivery pages, FAQs, or trusted partner listings are acceptable evidence."
    )

    # 5) Delivery minimum order requirement ≤ $35
    delivery_min_node = evaluator.add_leaf(
        id="Pattern_D_Delivery_Minimum_Order",
        desc="Verify and specify the delivery minimum order requirement is $35 or less",
        parent=node,
        critical=True
    )
    min_order_text = chain.delivery.min_order if chain.delivery and chain.delivery.min_order else ""
    delivery_min_claim = (
        f"The delivery minimum order requirement for {chain.name or 'the chain'} is {min_order_text}, which is $35 or less."
    )
    await evaluator.verify(
        claim=delivery_min_claim,
        node=delivery_min_node,
        sources=delivery_urls,
        additional_instruction="Confirm the minimum order value on the page and judge that it is ≤ $35."
    )

    # 6) Delivery reference URL existence (must provide URL)
    evaluator.add_custom_node(
        result=bool(delivery_urls),
        id="Pattern_D_Delivery_Reference_URL",
        desc="Provide reference URL confirming delivery service availability and minimum order requirement",
        parent=node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate the agent's answer for the Holiday Grocery Chain Identification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Evaluate each pattern independently
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

    # Extract structured chain information
    extraction = await evaluator.extract(
        prompt=prompt_extract_chains(),
        template_class=ChainsExtraction,
        extraction_name="chains_extraction"
    )

    # Record custom info about extracted patterns (useful for debugging)
    extracted_patterns = [c.pattern for c in extraction.chains]
    evaluator.add_custom_info(
        info={"extracted_patterns": extracted_patterns, "total_chains_extracted": len(extraction.chains)},
        info_type="extraction_stats",
        info_name="extraction_overview"
    )

    # Create the main node (non-critical root aggregator)
    main_node = evaluator.add_parallel(
        id="Holiday_Grocery_Chain_Identification",
        desc="Identify 4 major grocery store chains, each matching a distinct holiday operating pattern (A, B, C, or D) with verified delivery services",
        parent=root,
        critical=False
    )

    # Select chains per pattern (pick the first occurrence for each pattern; pad with empty placeholders if missing)
    chain_a = pick_first_chain_for_pattern(extraction, "A")
    chain_b = pick_first_chain_for_pattern(extraction, "B")
    chain_c = pick_first_chain_for_pattern(extraction, "C")
    chain_d = pick_first_chain_for_pattern(extraction, "D")

    # Verify each pattern
    await verify_pattern_a(evaluator, main_node, chain_a)
    await verify_pattern_b(evaluator, main_node, chain_b)
    await verify_pattern_c(evaluator, main_node, chain_c)
    await verify_pattern_d(evaluator, main_node, chain_d)

    # Return evaluation summary
    return evaluator.get_summary()