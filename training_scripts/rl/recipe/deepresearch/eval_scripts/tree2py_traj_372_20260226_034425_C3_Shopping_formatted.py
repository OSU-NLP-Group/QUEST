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
TASK_ID = "ct_grocery_pharmacy_xmas_2025"
TASK_DESCRIPTION = (
    "Which grocery store chain operating in Connecticut has in-store pharmacies that close for a lunch break "
    "from 1:30 PM to 2:00 PM daily and will be open on Christmas Eve 2025 (December 24, 2025) until at least 6:00 PM?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StoreChainExtraction(BaseModel):
    """Structured information extracted from the agent's answer."""
    chain_name: Optional[str] = None
    ct_presence_urls: List[str] = Field(default_factory=list)
    pharmacy_urls: List[str] = Field(default_factory=list)
    christmas_eve_hours_urls: List[str] = Field(default_factory=list)
    pharmacy_christmas_eve_urls: List[str] = Field(default_factory=list)
    lunch_break_policy_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_chain_info() -> str:
    return """
    Extract the grocery store chain name and the exact URLs the answer cites to support each of the following aspects:
    1) chain_name: The name of the grocery store chain identified in the answer.
    2) ct_presence_urls: URLs that show the chain operates in Connecticut (e.g., store locator pages, lists of locations including CT).
    3) pharmacy_urls: URLs that show the chain offers in-store pharmacy services (preferably relevant to Connecticut or general chain pharmacy page).
    4) christmas_eve_hours_urls: URLs that show store hours for Christmas Eve 2025 (December 24, 2025), including closing time information.
    5) pharmacy_christmas_eve_urls: URLs that show the in-store pharmacy is operating on Christmas Eve 2025 (December 24, 2025).
    6) lunch_break_policy_urls: URLs that show the pharmacy closes daily for lunch from 1:30 PM to 2:00 PM.

    IMPORTANT:
    - Only include URLs explicitly present in the answer text. Do not infer or add new URLs.
    - Extract valid, complete URLs (include http:// or https://). If protocol is missing, prepend http://.
    - If the answer does not provide URLs for a category, return an empty list for that category.
    - If the chain name is not provided, set chain_name to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _display_chain_name(data: StoreChainExtraction) -> str:
    return data.chain_name if data.chain_name and data.chain_name.strip() else "the identified grocery store chain"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_store_chain_identification(evaluator: Evaluator, parent_node, data: StoreChainExtraction) -> None:
    """
    Build the StoreChainIdentification subtree:
    - Geographic and service verification (parallel)
      - Operates in Connecticut (sequential)
         - ConnecticutPresenceCheck (verify claim by URLs)
         - ConnecticutPresenceURL (existence of URLs)
      - Has in-store pharmacy (sequential)
         - PharmacyServiceCheck (verify claim by URLs)
         - PharmacyServiceURL (existence of URLs)
    """
    chain_name_disp = _display_chain_name(data)

    store_chain_node = evaluator.add_sequential(
        id="StoreChainIdentification",
        desc="Identify a grocery store chain that operates in Connecticut with in-store pharmacy services",
        parent=parent_node,
        critical=False
    )

    geo_svc_node = evaluator.add_parallel(
        id="GeographicAndServiceVerification",
        desc="Verify the store chain's geographic presence and pharmacy services",
        parent=store_chain_node,
        critical=True
    )

    # Operates in Connecticut
    operates_ct_node = evaluator.add_sequential(
        id="OperatesInConnecticut",
        desc="The identified grocery store chain must have store locations operating in Connecticut",
        parent=geo_svc_node,
        critical=True
    )

    ct_presence_check = evaluator.add_leaf(
        id="ConnecticutPresenceCheck",
        desc="Confirm the store chain has Connecticut locations",
        parent=operates_ct_node,
        critical=True
    )
    ct_claim = f"{chain_name_disp} has store locations operating in the state of Connecticut."
    await evaluator.verify(
        claim=ct_claim,
        node=ct_presence_check,
        sources=data.ct_presence_urls,
        additional_instruction=(
            "Use the provided URLs (e.g., store locator pages or lists of locations) to confirm presence in Connecticut. "
            "Look for explicit mentions of 'Connecticut' or 'CT', or store entries located in CT."
        )
    )

    ct_presence_url_node = evaluator.add_custom_node(
        result=bool(data.ct_presence_urls),
        id="ConnecticutPresenceURL",
        desc="Provide reference URL confirming Connecticut presence",
        parent=operates_ct_node,
        critical=True
    )

    # Has in-store pharmacy
    has_pharmacy_node = evaluator.add_sequential(
        id="HasInStorePharmacy",
        desc="The identified grocery store chain must offer in-store pharmacy services at their Connecticut locations",
        parent=geo_svc_node,
        critical=True
    )

    pharmacy_service_check = evaluator.add_leaf(
        id="PharmacyServiceCheck",
        desc="Confirm in-store pharmacy services are available",
        parent=has_pharmacy_node,
        critical=True
    )
    pharmacy_claim = (
        f"{chain_name_disp} offers in-store pharmacy services (preferably at Connecticut locations or as a chain-wide service)."
    )
    await evaluator.verify(
        claim=pharmacy_claim,
        node=pharmacy_service_check,
        sources=data.pharmacy_urls,
        additional_instruction=(
            "Use the provided URLs to verify that the chain operates in-store pharmacies. "
            "Accept official pharmacy pages, pharmacy services sections, or store pages explicitly mentioning 'Pharmacy'. "
            "It's acceptable if the page is chain-wide as long as the chain operates in CT."
        )
    )

    pharmacy_service_url_node = evaluator.add_custom_node(
        result=bool(data.pharmacy_urls),
        id="PharmacyServiceURL",
        desc="Provide reference URL confirming pharmacy services",
        parent=has_pharmacy_node,
        critical=True
    )


async def build_christmas_eve_hours(evaluator: Evaluator, parent_node, data: StoreChainExtraction) -> None:
    """
    Build the ChristmasEveHours subtree:
    - HoursVerification (parallel)
      - OpenOnChristmasEve (verify claim by URLs)
      - ClosingTimeRequirement (verify claim by URLs)
    - ChristmasEveReferenceURL (existence of URLs)
    """
    chain_name_disp = _display_chain_name(data)

    christmas_node = evaluator.add_sequential(
        id="ChristmasEveHours",
        desc="Verify the store's Christmas Eve 2025 operating hours meet the minimum requirement",
        parent=parent_node,
        critical=False
    )

    hours_ver_node = evaluator.add_parallel(
        id="HoursVerification",
        desc="Verify Christmas Eve 2025 store hours",
        parent=christmas_node,
        critical=True
    )

    open_eve_leaf = evaluator.add_leaf(
        id="OpenOnChristmasEve",
        desc="The store must be open on Christmas Eve 2025 (December 24, 2025)",
        parent=hours_ver_node,
        critical=True
    )
    open_claim = f"{chain_name_disp} stores are open on Christmas Eve 2025 (December 24, 2025)."
    await evaluator.verify(
        claim=open_claim,
        node=open_eve_leaf,
        sources=data.christmas_eve_hours_urls,
        additional_instruction=(
            "Use holiday hours pages or official announcements to confirm that stores are open on December 24, 2025. "
            "Explicit mention of 'Christmas Eve 2025' or 'Dec 24, 2025' is preferred. If the page clearly indicates "
            "Christmas Eve hours for 2025, consider it sufficient."
        )
    )

    closing_req_leaf = evaluator.add_leaf(
        id="ClosingTimeRequirement",
        desc="The store must remain open until at least 6:00 PM on Christmas Eve 2025",
        parent=hours_ver_node,
        critical=True
    )
    closing_claim = (
        f"On December 24, 2025 (Christmas Eve), {chain_name_disp} stores close at or after 6:00 PM local time."
    )
    await evaluator.verify(
        claim=closing_claim,
        node=closing_req_leaf,
        sources=data.christmas_eve_hours_urls,
        additional_instruction=(
            "Check the closing time listed for Christmas Eve 2025. Pass if the closing time is 6:00 PM or later "
            "(e.g., 6 PM, 7 PM, etc.). If multiple locations have varying hours, evidence that at least one Connecticut "
            "store is open until 6 PM or later is acceptable."
        )
    )

    christmas_ref_url_leaf = evaluator.add_custom_node(
        result=bool(data.christmas_eve_hours_urls),
        id="ChristmasEveReferenceURL",
        desc="Provide reference URL confirming the Christmas Eve 2025 hours",
        parent=christmas_node,
        critical=True
    )


async def build_pharmacy_availability(evaluator: Evaluator, parent_node, data: StoreChainExtraction) -> None:
    """
    Build the PharmacyAvailability subtree:
    - PharmacyOperatesChristmasEve (verify claim by URLs)
    - PharmacyChristmasEveReferenceURL (existence of URLs)
    """
    chain_name_disp = _display_chain_name(data)

    pharm_avail_node = evaluator.add_sequential(
        id="PharmacyAvailability",
        desc="Verify the in-store pharmacy operates on Christmas Eve 2025",
        parent=parent_node,
        critical=False
    )

    pharm_oper_leaf = evaluator.add_leaf(
        id="PharmacyOperatesChristmasEve",
        desc="The in-store pharmacy must be operational on Christmas Eve 2025",
        parent=pharm_avail_node,
        critical=True
    )
    pharm_oper_claim = (
        f"The in-store pharmacy of {chain_name_disp} is open/operational on Christmas Eve 2025 (December 24, 2025)."
    )
    await evaluator.verify(
        claim=pharm_oper_claim,
        node=pharm_oper_leaf,
        sources=data.pharmacy_christmas_eve_urls,
        additional_instruction=(
            "Use pharmacy hours pages or official holiday notices to confirm pharmacy operations on December 24, 2025. "
            "Explicit mention of 'Christmas Eve 2025' or 'Dec 24, 2025' is preferred. "
            "If the page lists specific holiday hours that include Christmas Eve (2025), consider it sufficient."
        )
    )

    pharm_eve_ref_leaf = evaluator.add_custom_node(
        result=bool(data.pharmacy_christmas_eve_urls),
        id="PharmacyChristmasEveReferenceURL",
        desc="Provide reference URL confirming pharmacy operations on Christmas Eve 2025",
        parent=pharm_avail_node,
        critical=True
    )


async def build_lunch_break_policy(evaluator: Evaluator, parent_node, data: StoreChainExtraction) -> None:
    """
    Build the LunchBreakPolicy subtree:
    - LunchBreakTimeWindow (verify claim by URLs)
    - LunchBreakReferenceURL (existence of URLs)
    """
    chain_name_disp = _display_chain_name(data)

    lunch_node = evaluator.add_sequential(
        id="LunchBreakPolicy",
        desc="Verify the pharmacy's daily lunch break closure policy",
        parent=parent_node,
        critical=False
    )

    lunch_window_leaf = evaluator.add_leaf(
        id="LunchBreakTimeWindow",
        desc="The pharmacy must close for lunch break from 1:30 PM to 2:00 PM daily",
        parent=lunch_node,
        critical=True
    )
    lunch_claim = (
        f"The in-store pharmacy of {chain_name_disp} closes daily for a lunch break from 1:30 PM to 2:00 PM."
    )
    await evaluator.verify(
        claim=lunch_claim,
        node=lunch_window_leaf,
        sources=data.lunch_break_policy_urls,
        additional_instruction=(
            "Confirm that the provided URLs explicitly state the pharmacy is closed for lunch from 1:30 PM to 2:00 PM. "
            "Accept phrasing such as 'Pharmacy lunch break 1:30–2:00' or 'Closed for lunch 1:30 PM - 2:00 PM'. "
            "If the page lists the same lunch break across all days or explicitly says 'daily', consider it compliant."
        )
    )

    lunch_ref_leaf = evaluator.add_custom_node(
        result=bool(data.lunch_break_policy_urls),
        id="LunchBreakReferenceURL",
        desc="Provide reference URL confirming the lunch break policy",
        parent=lunch_node,
        critical=True
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
) -> Dict[str, Any]:
    """
    Entry point for evaluating the agent's answer for the Connecticut grocery store chain with specific pharmacy policies and Christmas Eve 2025 hours.
    """
    # Initialize evaluator; Keep root non-critical to allow mixed criticality in children.
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_chain_info(),
        template_class=StoreChainExtraction,
        extraction_name="store_chain_extraction"
    )

    # Record the extracted chain name for convenience
    evaluator.add_custom_info(
        info={"chain_name": extracted.chain_name or None},
        info_type="extraction_summary",
        info_name="identified_chain"
    )

    # Build verification tree according to rubric
    # Root (JSON Root) was critical in the provided rubric, but to satisfy framework constraints
    # and allow non-critical children, we keep the actual root non-critical and mirror its description.
    root.desc = "Identify the grocery store chain in Connecticut that meets all specified criteria regarding Christmas Eve hours, pharmacy services, and lunch break policies"

    # Subtrees according to rubric
    await build_store_chain_identification(evaluator, root, extracted)
    await build_christmas_eve_hours(evaluator, root, extracted)
    await build_pharmacy_availability(evaluator, root, extracted)
    await build_lunch_break_policy(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()