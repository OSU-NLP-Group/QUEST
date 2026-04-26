import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "premium_card_airport_lounges_2026"
TASK_DESCRIPTION = (
    "A business traveler frequently flies through Boston Logan International Airport (BOS), Philadelphia International Airport (PHL), "
    "and New York John F. Kennedy International Airport (JFK). They are looking for a premium credit card that provides access to a proprietary lounge network "
    "(lounges operated by the card issuer or partner bank, not just third-party networks like Priority Pass) at all three of these airports. "
    "The card must allow the primary cardholder to bring at least 2 complimentary guests into these proprietary lounges, and lounge access for the cardholder must be complimentary "
    "without requiring additional spending thresholds beyond the annual fee. Identify a specific credit card product currently available as of January 2026 that meets all of these requirements. "
    "Provide the card name and verify that it satisfies each criterion with supporting evidence including reference URLs."
)

CURRENT_AVAILABILITY_MONTH_YEAR = "January 2026"


class CreditCardExtraction(BaseModel):
    card_name: Optional[str] = None
    card_availability_urls: List[str] = Field(default_factory=list)
    bos_lounge_urls: List[str] = Field(default_factory=list)
    phl_lounge_urls: List[str] = Field(default_factory=list)
    jfk_lounge_urls: List[str] = Field(default_factory=list)
    guest_policy_urls: List[str] = Field(default_factory=list)
    complimentary_access_urls: List[str] = Field(default_factory=list)


def prompt_extract_card_sources() -> str:
    return (
        "Extract the specific credit card product recommended in the answer and all cited reference URLs that support each verification criterion.\n"
        "Return a JSON object with the following fields:\n"
        "- card_name: The exact product name of the recommended credit card.\n"
        "- card_availability_urls: Array of URLs the answer cites to show the card is currently available (open to apply) as of January 2026.\n"
        "- bos_lounge_urls: Array of URLs the answer cites to show this card grants access to an issuer- or partner-bank-operated proprietary lounge at BOS (not just Priority Pass or other third-party networks).\n"
        "- phl_lounge_urls: Array of URLs the answer cites to show proprietary lounge access at PHL.\n"
        "- jfk_lounge_urls: Array of URLs the answer cites to show proprietary lounge access at JFK.\n"
        "- guest_policy_urls: Array of URLs the answer cites to show the primary cardholder may bring at least 2 complimentary guests into these proprietary lounges.\n"
        "- complimentary_access_urls: Array of URLs the answer cites to show the primary cardholder's lounge access is complimentary without additional spending thresholds beyond the annual fee.\n\n"
        "IMPORTANT:\n"
        "• Only include URLs that are explicitly present in the answer text (including markdown links). Do not invent or infer URLs.\n"
        "• If the answer does not provide any URL for a field, return an empty array for that field.\n"
        "• If multiple cards are mentioned, choose the main recommended card (or the first clearly recommended card) and extract URLs relevant to that card only.\n"
    )


async def build_verification_tree(evaluator: Evaluator, root, data: CreditCardExtraction) -> None:
    # Card Identification & Current Availability (aggregate node)
    card_ident_node = evaluator.add_parallel(
        id="Card_Identification_and_Current_Availability",
        desc="Provides the specific credit card product name and includes at least one reference URL showing the product is currently available as of January 2026.",
        parent=root,
        critical=True,
    )
    # Existence of card name
    evaluator.add_custom_node(
        result=bool(data.card_name and data.card_name.strip()),
        id="card_name_present",
        desc="Specific credit card product name is provided in the answer.",
        parent=card_ident_node,
        critical=True,
    )
    # Presence of availability URLs
    evaluator.add_custom_node(
        result=len(data.card_availability_urls) > 0,
        id="availability_urls_present",
        desc="At least one availability/reference URL is provided.",
        parent=card_ident_node,
        critical=True,
    )
    # Verification of current availability via URLs
    card_avail_leaf = evaluator.add_leaf(
        id="availability_verified",
        desc="Product is currently available (open for applications) as of January 2026, supported by provided URLs.",
        parent=card_ident_node,
        critical=True,
    )
    card_name_for_claim = data.card_name or ""
    availability_claim = (
        f"The credit card product '{card_name_for_claim}' is currently available (open to apply) as of {CURRENT_AVAILABILITY_MONTH_YEAR}."
    )
    await evaluator.verify(
        claim=availability_claim,
        node=card_avail_leaf,
        sources=data.card_availability_urls,
        additional_instruction=(
            "Use the provided URLs to determine if the product page indicates availability for new applications (e.g., 'Apply' button, current product page). "
            "Issuer or bank official pages and credible sources are acceptable. "
            "If the URLs do not show current availability or no URLs were extracted from the answer, mark this as Incorrect."
        ),
    )

    # Proprietary Lounge Access BOS
    bos_node = evaluator.add_parallel(
        id="Proprietary_Lounge_Access_BOS_with_Evidence",
        desc="Includes at least one reference URL showing the card provides access to an issuer/partner-operated (proprietary) lounge at Boston Logan International Airport (BOS), not merely a third-party network lounge.",
        parent=root,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(data.bos_lounge_urls) > 0,
        id="bos_urls_present",
        desc="At least one BOS lounge URL is provided in the answer.",
        parent=bos_node,
        critical=True,
    )
    bos_verify_leaf = evaluator.add_leaf(
        id="bos_proprietary_verified",
        desc="Card provides access to an issuer/partner-operated proprietary lounge at BOS.",
        parent=bos_node,
        critical=True,
    )
    bos_claim = (
        f"The card '{card_name_for_claim}' provides access to a proprietary lounge (issuer- or partner-bank-operated, e.g., Centurion Lounge, Chase Sapphire Lounge, Capital One Lounge) at BOS."
    )
    await evaluator.verify(
        claim=bos_claim,
        node=bos_verify_leaf,
        sources=data.bos_lounge_urls,
        additional_instruction=(
            "Confirm that the cited BOS lounge is part of an issuer/bank proprietary network (e.g., American Express Centurion Lounge, Chase Sapphire Lounge by The Club, Capital One Lounge), "
            "not merely a third-party network like Priority Pass. The page should clearly connect card eligibility to this lounge network. "
            "If URLs fail to show issuer/partner-operated lounge access, mark Incorrect."
        ),
    )

    # Proprietary Lounge Access PHL
    phl_node = evaluator.add_parallel(
        id="Proprietary_Lounge_Access_PHL_with_Evidence",
        desc="Includes at least one reference URL showing the card provides access to an issuer/partner-operated (proprietary) lounge at Philadelphia International Airport (PHL), not merely a third-party network lounge.",
        parent=root,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(data.phl_lounge_urls) > 0,
        id="phl_urls_present",
        desc="At least one PHL lounge URL is provided in the answer.",
        parent=phl_node,
        critical=True,
    )
    phl_verify_leaf = evaluator.add_leaf(
        id="phl_proprietary_verified",
        desc="Card provides access to an issuer/partner-operated proprietary lounge at PHL.",
        parent=phl_node,
        critical=True,
    )
    phl_claim = (
        f"The card '{card_name_for_claim}' provides access to a proprietary lounge (issuer- or partner-bank-operated) at PHL."
    )
    await evaluator.verify(
        claim=phl_claim,
        node=phl_verify_leaf,
        sources=data.phl_lounge_urls,
        additional_instruction=(
            "Verify the PHL lounge is part of an issuer/bank proprietary network and the card grants access to it. "
            "Citations to third-party-only networks (e.g., Priority Pass listings without issuer lounge branding) do not qualify."
        ),
    )

    # Proprietary Lounge Access JFK
    jfk_node = evaluator.add_parallel(
        id="Proprietary_Lounge_Access_JFK_with_Evidence",
        desc="Includes at least one reference URL showing the card provides access to an issuer/partner-operated (proprietary) lounge at New York John F. Kennedy International Airport (JFK), not merely a third-party network lounge.",
        parent=root,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(data.jfk_lounge_urls) > 0,
        id="jfk_urls_present",
        desc="At least one JFK lounge URL is provided in the answer.",
        parent=jfk_node,
        critical=True,
    )
    jfk_verify_leaf = evaluator.add_leaf(
        id="jfk_proprietary_verified",
        desc="Card provides access to an issuer/partner-operated proprietary lounge at JFK.",
        parent=jfk_node,
        critical=True,
    )
    jfk_claim = (
        f"The card '{card_name_for_claim}' provides access to a proprietary lounge (issuer- or partner-bank-operated) at JFK."
    )
    await evaluator.verify(
        claim=jfk_claim,
        node=jfk_verify_leaf,
        sources=data.jfk_lounge_urls,
        additional_instruction=(
            "Verify the JFK lounge belongs to the issuer/bank proprietary network and the card grants access. "
            "Do not accept third-party-only networks as sufficient evidence."
        ),
    )

    # Guest Policy: At least two complimentary guests
    guest_node = evaluator.add_parallel(
        id="Guest_Policy_At_Least_Two_with_Evidence",
        desc="Includes at least one reference URL showing the primary cardholder may bring at least 2 complimentary guests into the proprietary lounges.",
        parent=root,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(data.guest_policy_urls) > 0,
        id="guest_policy_urls_present",
        desc="At least one guest policy URL is provided in the answer.",
        parent=guest_node,
        critical=True,
    )
    guest_verify_leaf = evaluator.add_leaf(
        id="guest_policy_verified",
        desc="Primary cardholder may bring at least two complimentary guests into proprietary lounges.",
        parent=guest_node,
        critical=True,
    )
    guest_claim = (
        f"The primary cardholder of '{card_name_for_claim}' may bring at least two guests into the proprietary lounge(s) at no charge."
    )
    await evaluator.verify(
        claim=guest_claim,
        node=guest_verify_leaf,
        sources=data.guest_policy_urls,
        additional_instruction=(
            "Confirm the guest policy explicitly allows at least two guests complimentary for the primary cardholder. "
            "If complimentary guest access requires meeting an additional spend threshold (e.g., annual spend requirement), this should be marked Incorrect."
        ),
    )

    # Complimentary access for primary cardholder without spend threshold
    comp_node = evaluator.add_parallel(
        id="Complimentary_Access_No_Spend_Threshold_with_Evidence",
        desc="Includes at least one reference URL showing lounge access for the primary cardholder is complimentary without additional spending thresholds beyond holding the card/annual fee.",
        parent=root,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(data.complimentary_access_urls) > 0,
        id="complimentary_urls_present",
        desc="At least one complimentary-access policy URL is provided in the answer.",
        parent=comp_node,
        critical=True,
    )
    comp_verify_leaf = evaluator.add_leaf(
        id="complimentary_access_verified",
        desc="Primary cardholder lounge access is complimentary without an extra spending threshold beyond annual fee.",
        parent=comp_node,
        critical=True,
    )
    comp_claim = (
        f"Lounge access for the primary cardholder of '{card_name_for_claim}' is complimentary and does not require meeting an additional spending threshold beyond the annual fee."
    )
    await evaluator.verify(
        claim=comp_claim,
        node=comp_verify_leaf,
        sources=data.complimentary_access_urls,
        additional_instruction=(
            "Confirm the cardholder’s own entry is complimentary based on holding the card (or included membership) without any extra spend requirement. "
            "Reasonable conditions like same-day boarding pass do not count as spend thresholds. "
            "If a spend threshold is required for complimentary entry, mark Incorrect."
        ),
    )


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_card_sources(),
        template_class=CreditCardExtraction,
        extraction_name="credit_card_sources",
    )

    # Build and evaluate the verification tree
    solution_root = evaluator.add_parallel(
        id="Credit_Card_Solution",
        desc="Identify a specific credit card and verify it meets all lounge/guest/complimentary-access requirements with supporting reference URLs.",
        parent=root,
        critical=True,
    )

    await build_verification_tree(evaluator, solution_root, extracted)

    return evaluator.get_summary()