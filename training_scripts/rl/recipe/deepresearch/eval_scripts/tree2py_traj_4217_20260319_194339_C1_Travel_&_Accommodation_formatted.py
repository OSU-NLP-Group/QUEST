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
TASK_ID = "jfk_airtrain_fare_payment_2026"
TASK_DESCRIPTION = (
    "As of March 2026, what is the current one-way fare for the AirTrain at JFK Airport, and what is the name of the "
    "contactless payment system that became the only payment option after MetroCard was discontinued in January 2026?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FareInfo(BaseModel):
    fare_amount: Optional[str] = None
    fare_sources: List[str] = Field(default_factory=list)

    mentions_march_2026_increase: Optional[bool] = None
    mentions_from_850_to_875: Optional[bool] = None
    increase_sources: List[str] = Field(default_factory=list)


class PaymentSystemInfo(BaseModel):
    system_name: Optional[str] = None
    system_sources: List[str] = Field(default_factory=list)

    metrocard_discontinued_jan_2026: Optional[bool] = None
    discontinued_sources: List[str] = Field(default_factory=list)

    only_payment_option_after_discontinuation: Optional[bool] = None
    only_payment_sources: List[str] = Field(default_factory=list)

    accepted_media: List[str] = Field(default_factory=list)
    accepted_media_sources: List[str] = Field(default_factory=list)


class JFKAirTrainExtraction(BaseModel):
    fare: FareInfo = Field(default_factory=FareInfo)
    payment: PaymentSystemInfo = Field(default_factory=PaymentSystemInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_jfk_airtrain_info() -> str:
    return """
    Extract the specific pieces of information about JFK AirTrain fare and payment mentioned in the answer.

    Return a JSON object with this structure:
    {
      "fare": {
        "fare_amount": string | null,               // The one-way AirTrain JFK fare explicitly stated as of March 2026 (e.g., "$8.75"). If not present, null.
        "fare_sources": string[]                    // All URLs cited that directly support the fare amount. If none, return [].
        "mentions_march_2026_increase": boolean | null,  // True if the answer states there was a fare increase in March 2026 (even without exact amounts). False if it explicitly states no increase; null if not mentioned.
        "mentions_from_850_to_875": boolean | null, // True if the answer states or clearly implies an increase from $8.50 to $8.75 in March 2026. False if states otherwise; null if not mentioned.
        "increase_sources": string[]                // All URLs cited that support the increase (month and/or amounts). If none, return [].
      },
      "payment": {
        "system_name": string | null,               // The name of the contactless payment system that replaced MetroCard (e.g., "OMNY"). If not present, null.
        "system_sources": string[],                 // All URLs cited that support the identification of this system. If none, return [].
        "metrocard_discontinued_jan_2026": boolean | null, // True if the answer states MetroCard was discontinued in January 2026; False if claims differently; null if not mentioned.
        "discontinued_sources": string[],           // All URLs cited that support the MetroCard discontinuation timing. If none, return [].
        "only_payment_option_after_discontinuation": boolean | null, // True if the answer claims the named contactless system became the ONLY payment option after discontinuation; False if claims otherwise; null if not mentioned.
        "only_payment_sources": string[],           // All URLs cited that support the "only payment option" claim. If none, return [].
        "accepted_media": string[],                 // A list of accepted media explicitly mentioned in the answer. Normalize each to one of:
                                                    //   "contactless_card" (bank credit/debit card), "mobile_device" (phone/watch wallets),
                                                    //   "dedicated_card" (e.g., an OMNY card or similar), or use the closest of these if clear.
                                                    // Include each category at most once.
        "accepted_media_sources": string[]          // All URLs cited that support accepted media. If none, return [].
      }
    }

    Rules:
    - Extract only what is explicitly present in the answer. Do not infer or invent.
    - For URL fields, extract only valid URLs that appear in the answer (including in markdown links).
    - If a requested value is missing from the answer, return null (for scalars) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_fare(evaluator: Evaluator, parent_node, fare: FareInfo) -> None:
    """
    Build the Fare Information subtree and run verifications.
    This corresponds to the rubric group: Fare_Information (critical, parallel).
    """
    fare_node = evaluator.add_parallel(
        id="Fare_Information",
        desc="Provide the required AirTrain fare information as constrained.",
        parent=parent_node,
        critical=True
    )

    # Existence: the agent must provide the one-way fare as of March 2026
    evaluator.add_custom_node(
        result=bool(fare.fare_amount and fare.fare_amount.strip()),
        id="Fare_Value_Provided",
        desc="Answer provides the one-way AirTrain fare as of March 2026.",
        parent=fare_node,
        critical=True
    )

    # Presence: the agent should mention there was a March 2026 increase
    evaluator.add_custom_node(
        result=bool(fare.mentions_march_2026_increase),
        id="Fare_Increase_March_2026_Mentioned",
        desc="Answer states there was a fare increase in March 2026 (even if not giving exact amounts).",
        parent=fare_node,
        critical=True
    )

    # Leaf: Verify the fare amount as of March 2026 (use provided sources if any)
    fare_amount_leaf = evaluator.add_leaf(
        id="Fare_AsOf_March_2026_Amount_Correct",
        desc="As of March 2026, the one-way JFK AirTrain fare amount stated in the answer is correct.",
        parent=fare_node,
        critical=True
    )
    fare_amount_claim = f"As of March 2026, the one-way fare for the AirTrain at JFK Airport is {fare.fare_amount}."
    await evaluator.verify(
        claim=fare_amount_claim,
        node=fare_amount_leaf,
        sources=_merge_sources(fare.fare_sources, fare.increase_sources),
        additional_instruction=(
            "Confirm the current (post-March-2026) JFK AirTrain one-way fare. "
            "Focus specifically on AirTrain JFK fare (Port Authority), not NYC Subway/LIRR base fares. "
            "Minor formatting differences (like including a dollar sign) are acceptable as long as the numeric amount matches."
        ),
    )

    # Leaf: Verify that there was an increase in March 2026 from $8.50 to $8.75 (or consistent with that change)
    fare_increase_leaf = evaluator.add_leaf(
        id="Fare_Increase_March_2026_850_to_875_Correct",
        desc="States the March 2026 increase from $8.50 to $8.75 (or information fully consistent with that increase) is correct.",
        parent=fare_node,
        critical=True
    )
    increase_claim = (
        "In March 2026, the JFK AirTrain one-way fare increased from $8.50 to $8.75 "
        "(wording reasonably similar is acceptable as long as the amounts and month match)."
    )
    await evaluator.verify(
        claim=increase_claim,
        node=fare_increase_leaf,
        sources=_merge_sources(fare.increase_sources, fare.fare_sources),
        additional_instruction=(
            "Verify that a fare change occurred in March 2026 and that it raised the AirTrain JFK one-way fare "
            "from approximately $8.50 to approximately $8.75. Allow small phrasing variations."
        ),
    )


async def build_and_verify_contactless(evaluator: Evaluator, parent_node, payment: PaymentSystemInfo) -> None:
    """
    Build the Contactless Payment System subtree and run verifications.
    This corresponds to the rubric group: Contactless_Payment_System (critical, parallel).
    """
    pay_node = evaluator.add_parallel(
        id="Contactless_Payment_System",
        desc="Identify the contactless payment system that replaced MetroCard at JFK and verify required properties.",
        parent=parent_node,
        critical=True
    )

    # Presence: system name provided
    evaluator.add_custom_node(
        result=bool(payment.system_name and payment.system_name.strip()),
        id="System_Name_Provided",
        desc="Provides the name of the contactless payment system that replaced MetroCard at JFK Airport.",
        parent=pay_node,
        critical=True
    )

    # Leaf: MetroCard discontinued in January 2026 (factual verification)
    metrocard_disc_leaf = evaluator.add_leaf(
        id="MetroCard_Discontinued_Jan_2026",
        desc="States that MetroCard was discontinued at JFK Airport in January 2026.",
        parent=pay_node,
        critical=True
    )
    metrocard_claim = "MetroCard was discontinued in January 2026."
    await evaluator.verify(
        claim=metrocard_claim,
        node=metrocard_disc_leaf,
        sources=_merge_sources(payment.discontinued_sources, payment.system_sources),
        additional_instruction=(
            "Confirm that MetroCard was discontinued (retired/ended) in January 2026. "
            "Accept language like 'retired', 'phased out', or 'ended' if it clearly indicates discontinuation in Jan 2026."
        ),
    )

    # Presence: the answer explicitly says the named system became the only payment option
    evaluator.add_custom_node(
        result=bool(payment.only_payment_option_after_discontinuation),
        id="Only_Payment_Option_Stated",
        desc="Answer states the named contactless system became the only payment option after MetroCard was discontinued.",
        parent=pay_node,
        critical=True
    )

    # Leaf: Verify the 'only payment option' claim (factual verification)
    only_option_leaf = evaluator.add_leaf(
        id="System_Became_Only_Payment_Option_After_Discontinuation",
        desc="After MetroCard was discontinued, the named contactless system became the only payment option.",
        parent=pay_node,
        critical=True
    )
    system_name_text = payment.system_name if payment.system_name else "the contactless payment system"
    only_option_claim = (
        f"After MetroCard was discontinued in January 2026, {system_name_text} became the only payment option "
        f"for paying the JFK AirTrain fare (i.e., MetroCard and cash are not accepted)."
    )
    await evaluator.verify(
        claim=only_option_claim,
        node=only_option_leaf,
        sources=_merge_sources(payment.only_payment_sources, payment.system_sources),
        additional_instruction=(
            "Confirm exclusivity: that after MetroCard's discontinuation, the named contactless system is the sole accepted method "
            "to pay the JFK AirTrain fare at fare gates (no MetroCard or cash accepted)."
        ),
    )

    # Presence: the answer indicates accepted contactless media
    has_any_media = any(m in {"contactless_card", "mobile_device", "dedicated_card"} for m in payment.accepted_media)
    evaluator.add_custom_node(
        result=has_any_media,
        id="Accepted_Contactless_Media_Stated",
        desc="Answer indicates accepted media (contactless bank cards, mobile devices, or a dedicated card).",
        parent=pay_node,
        critical=True
    )

    # Leaf: Verify accepted media with sources
    accepted_media_leaf = evaluator.add_leaf(
        id="Accepted_Contactless_Media_Supported",
        desc="Indicates the system accepts contactless credit/debit cards, mobile devices, or dedicated payment cards (or clear equivalents).",
        parent=pay_node,
        critical=True
    )
    accepted_media_claim = (
        f"{system_name_text} accepts contactless credit/debit bank cards (tap-to-pay), mobile wallets on phones/watches "
        f"(e.g., Apple Pay or Google Pay), and/or a dedicated payment card (e.g., an OMNY card)."
    )
    await evaluator.verify(
        claim=accepted_media_claim,
        node=accepted_media_leaf,
        sources=_merge_sources(payment.accepted_media_sources, payment.system_sources),
        additional_instruction=(
            "You should verify that the system supports: (1) open-loop contactless bank cards; "
            "(2) mobile device wallets (phones/watches); and/or (3) a dedicated proprietary card. "
            "Accept equivalent phrasing (e.g., 'OMNY card' counts as a dedicated card)."
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
    Entry point for evaluating the JFK AirTrain fare and payment system question.
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

    # Top-level critical group node reflecting the rubric root
    jfk_info_node = evaluator.add_parallel(
        id="JFK_AirTrain_Information",
        desc="Provide (1) the current JFK AirTrain one-way fare as of March 2026 and "
             "(2) the name and required properties of the contactless payment system that became the only payment option after MetroCard was discontinued in January 2026.",
        parent=root,
        critical=True
    )

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_jfk_airtrain_info(),
        template_class=JFKAirTrainExtraction,
        extraction_name="jfk_airtrain_extraction"
    )

    # Build and verify subtrees
    await build_and_verify_fare(evaluator, jfk_info_node, extraction.fare)
    await build_and_verify_contactless(evaluator, jfk_info_node, extraction.payment)

    # Return structured summary
    return evaluator.get_summary()