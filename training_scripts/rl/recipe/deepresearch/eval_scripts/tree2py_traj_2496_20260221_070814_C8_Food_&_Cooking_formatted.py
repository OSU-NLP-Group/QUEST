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
TASK_ID = "holiday_resource_guide"
TASK_DESCRIPTION = (
    "You are helping create a winter holiday resource guide for a community to ensure residents can access "
    "essential services, food, and pharmacy needs during major holidays.\n\n"
    "Identify four specific national chain establishments, one from each of the following categories, that meet "
    "these holiday operating requirements:\n\n"
    "1. A standalone pharmacy chain (not an in-store pharmacy within a retail store) that provides prescription "
    "services on Christmas Day (December 25)\n\n"
    "2. A convenience store chain that operates continuously (24 hours a day, 7 days a week) through major holidays "
    "including Christmas Day\n\n"
    "3. A breakfast or brunch restaurant chain that serves customers on New Year's Day (January 1)\n\n"
    "4. A fast food chain that maintains operations on Christmas Day\n\n"
    "For each establishment you identify, provide:\n"
    "- The name of the national chain\n"
    "- Verification of its holiday operating policy relevant to the specified requirement\n"
    "- A reference URL from a reliable source (news article, corporate website, or business directory) that confirms "
    "the chain's holiday operations\n\n"
    "Note: Hours may vary by individual location, but the chain's overall policy should support the stated requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Establishment(BaseModel):
    name: Optional[str] = None
    policy_excerpt: Optional[str] = None
    support_urls: List[str] = Field(default_factory=list)


class HolidayResourcesExtraction(BaseModel):
    pharmacy: Optional[Establishment] = None
    convenience_store: Optional[Establishment] = None
    breakfast_restaurant: Optional[Establishment] = None
    fast_food: Optional[Establishment] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_holiday_resources() -> str:
    return """
    Extract the four national chains (one per category) that the answer proposes for the holiday resource guide.

    For each category, extract:
    - name: the chain's name exactly as written in the answer
    - policy_excerpt: a short excerpt or sentence quoted or closely paraphrased from the answer that describes the chain’s holiday operations relevant to the requirement (e.g., “open on Christmas Day,” “24/7 including holidays,” “open New Year’s Day serving breakfast”)
    - support_urls: a list of up to 5 URLs that the answer cites for that category’s chain to support the holiday-hours claim (news, corporate site, or business directory). Extract only actual URLs present in the answer text (including those inside markdown links).

    Categories to extract (choose the first clearly matching chain for each category if multiple are provided):
    - pharmacy: a standalone pharmacy chain (not just an in-store pharmacy within a retail store) that provides prescription services on Christmas Day
    - convenience_store: a convenience store chain that operates 24/7 and remains open on Christmas Day
    - breakfast_restaurant: a breakfast/brunch restaurant chain that is open on New Year’s Day
    - fast_food: a fast food chain that has locations open on Christmas Day

    If a category is missing from the answer, set that field to null.
    If no URLs are provided for a category, return an empty list for support_urls.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _urls_or_empty(est: Optional[Establishment]) -> List[str]:
    return est.support_urls if (est and est.support_urls) else []


def _name_or_blank(est: Optional[Establishment]) -> str:
    return est.name or ""


# --------------------------------------------------------------------------- #
# Verification functions for each category                                    #
# --------------------------------------------------------------------------- #
async def verify_pharmacy_chain(evaluator: Evaluator, parent_node, est: Optional[Establishment]) -> None:
    """
    Standalone pharmacy chain verifications.
    """
    cat_node = evaluator.add_parallel(
        id="Standalone_Pharmacy_Chain",
        desc="A national standalone pharmacy chain that provides prescription services on Christmas Day",
        parent=parent_node,
        critical=False
    )

    # Reference URL presence (critical)
    evaluator.add_custom_node(
        result=bool(est and est.support_urls and len(est.support_urls) > 0),
        id="Reference_URL_Pharmacy",
        desc="Provide valid URL reference supporting the pharmacy's Christmas Day operations from search results",
        parent=cat_node,
        critical=True
    )

    # Chain identification (critical)
    chain_ident_node = evaluator.add_leaf(
        id="Chain_Identification",
        desc="Identify a valid national standalone pharmacy chain (not an in-store pharmacy)",
        parent=cat_node,
        critical=True
    )
    chain_name = _name_or_blank(est)
    await evaluator.verify(
        claim=f"'{chain_name}' is a national standalone pharmacy chain (not merely in-store pharmacies within supermarkets or big-box stores).",
        node=chain_ident_node,
        sources=_urls_or_empty(est),
        additional_instruction=(
            "Accept Walgreens, CVS Pharmacy, Rite Aid, etc., as standalone chains with their own stores. "
            "It is acceptable if some locations also operate inside other stores, as long as the chain primarily operates standalone pharmacies. "
            "Reject purely in-store brands of grocery chains."
        )
    )

    # Christmas Day operations (critical)
    xmas_ops_node = evaluator.add_leaf(
        id="Christmas_Day_Operations",
        desc="Verify that the pharmacy chain operates and provides prescription services on Christmas Day",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{chain_name}' has locations operating on Christmas Day (Dec 25) providing pharmacy prescription services. Hours may vary by location.",
        node=xmas_ops_node,
        sources=_urls_or_empty(est),
        additional_instruction=(
            "Look for explicit confirmation they are open on Christmas Day and that pharmacy/prescription services are available. "
            "Accept formulations like 'some locations open' or 'reduced holiday hours'; hours can vary by location."
        )
    )

    # Operating hours/policy confirmation (critical)
    hours_policy_node = evaluator.add_leaf(
        id="Operating_Hours_Verification",
        desc="Confirm the chain's Christmas Day operating hours or policy (hours may vary by location)",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided source(s) give explicit information about Christmas Day hours or holiday-hours policy for '{chain_name}' (e.g., lists special hours or notes hours vary by location).",
        node=hours_policy_node,
        sources=_urls_or_empty(est),
        additional_instruction=(
            "Accept explicit policy statements or store-hour listings for Christmas Day. "
            "A general holiday-hours notice that mentions Christmas is sufficient."
        )
    )


async def verify_convenience_store_chain(evaluator: Evaluator, parent_node, est: Optional[Establishment]) -> None:
    """
    Convenience store chain verifications.
    """
    cat_node = evaluator.add_parallel(
        id="Convenience_Store_Chain",
        desc="A national convenience store chain that operates continuously (24/7) through major holidays including Christmas",
        parent=parent_node,
        critical=False
    )

    # Reference URL presence (critical)
    evaluator.add_custom_node(
        result=bool(est and est.support_urls and len(est.support_urls) > 0),
        id="Reference_URL_Convenience",
        desc="Provide valid URL reference supporting the convenience store's holiday operations from search results",
        parent=cat_node,
        critical=True
    )

    # Chain identification (critical)
    chain_ident_node = evaluator.add_leaf(
        id="Chain_Identification_Convenience",
        desc="Identify a valid national convenience store chain",
        parent=cat_node,
        critical=True
    )
    chain_name = _name_or_blank(est)
    await evaluator.verify(
        claim=f"'{chain_name}' is a national convenience store chain.",
        node=chain_ident_node,
        sources=_urls_or_empty(est),
        additional_instruction=(
            "Accept well-known convenience store chains (e.g., 7-Eleven, Wawa, Sheetz, Circle K, QuikTrip). "
            "Must be multi-state or broadly national in scope."
        )
    )

    # 24/7 operations (critical)
    ops_24_node = evaluator.add_leaf(
        id="24_Hour_Operations",
        desc="Verify that the chain operates 24/7 or has 24-hour locations",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{chain_name}' operates 24 hours a day, 7 days a week (at least for many or typical locations).",
        node=ops_24_node,
        sources=_urls_or_empty(est),
        additional_instruction=(
            "Accept claims that many or most locations are 24/7, or that the chain is known for 24/7 service."
        )
    )

    # Christmas availability (critical)
    xmas_open_node = evaluator.add_leaf(
        id="Christmas_Availability",
        desc="Confirm the chain remains open on Christmas Day (all day or with most locations open)",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{chain_name}' remains open on Christmas Day (Dec 25), consistent with 24/7 operations; exceptions may exist by location.",
        node=xmas_open_node,
        sources=_urls_or_empty(est),
        additional_instruction=(
            "Accept formulations such as 'most locations open' or 'hours vary by location'. "
            "24/7 policies generally imply being open on Christmas."
        )
    )

    # Food and beverage services (non-critical)
    fnb_node = evaluator.add_leaf(
        id="Food_Beverage_Services",
        desc="Verify that the chain provides food and beverage options",
        parent=cat_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"'{chain_name}' sells food and beverages such as snacks, prepared foods, and drinks.",
        node=fnb_node,
        sources=_urls_or_empty(est),
        additional_instruction="Confirm typical convenience-store food/beverage offerings are available."
    )


async def verify_breakfast_restaurant_chain(evaluator: Evaluator, parent_node, est: Optional[Establishment]) -> None:
    """
    Breakfast/brunch restaurant chain verifications.
    """
    cat_node = evaluator.add_parallel(
        id="Breakfast_Restaurant_Chain",
        desc="A national breakfast/brunch restaurant chain that serves customers on New Year's Day",
        parent=parent_node,
        critical=False
    )

    # Reference URL presence (critical)
    evaluator.add_custom_node(
        result=bool(est and est.support_urls and len(est.support_urls) > 0),
        id="Reference_URL_Restaurant",
        desc="Provide valid URL reference supporting the restaurant's New Year's Day operations from search results",
        parent=cat_node,
        critical=True
    )

    # Chain identification (critical)
    chain_ident_node = evaluator.add_leaf(
        id="Chain_Identification_Breakfast",
        desc="Identify a valid national restaurant chain that serves breakfast or brunch",
        parent=cat_node,
        critical=True
    )
    chain_name = _name_or_blank(est)
    await evaluator.verify(
        claim=f"'{chain_name}' is a national restaurant chain known for serving breakfast or brunch.",
        node=chain_ident_node,
        sources=_urls_or_empty(est),
        additional_instruction=(
            "Accept chains like IHOP, Denny's, Perkins, etc., that prominently serve breakfast/brunch. "
            "All-day breakfast also qualifies."
        )
    )

    # New Year's Day operations (critical)
    nyd_open_node = evaluator.add_leaf(
        id="New_Years_Day_Operations",
        desc="Verify that the restaurant chain is open on New Year's Day",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{chain_name}' is open on New Year's Day (Jan 1).",
        node=nyd_open_node,
        sources=_urls_or_empty(est),
        additional_instruction=(
            "Accept explicit statements that locations are open on New Year's Day; hours may vary by location."
        )
    )

    # Breakfast service confirmation (critical)
    breakfast_confirm_node = evaluator.add_leaf(
        id="Breakfast_Service_Confirmation",
        desc="Confirm the chain serves breakfast or brunch during New Year's Day hours",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"On New Year's Day, '{chain_name}' serves breakfast or brunch (e.g., breakfast menu or all-day breakfast applies).",
        node=breakfast_confirm_node,
        sources=_urls_or_empty(est),
        additional_instruction=(
            "Accept if the chain normally serves breakfast (including all-day breakfast), implying availability on New Year's Day when open."
        )
    )

    # Operating hours information (non-critical)
    hours_info_node = evaluator.add_leaf(
        id="Operating_Hours_Info",
        desc="Provide information about New Year's Day operating hours or policy",
        parent=cat_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The provided source(s) include information about New Year's Day hours or a holiday-hours policy for '{chain_name}'.",
        node=hours_info_node,
        sources=_urls_or_empty(est),
        additional_instruction="General New Year's Day hours or policy statements count, even if hours vary by location."
    )


async def verify_fast_food_chain(evaluator: Evaluator, parent_node, est: Optional[Establishment]) -> None:
    """
    Fast food chain verifications.
    """
    cat_node = evaluator.add_parallel(
        id="Fast_Food_Chain",
        desc="A national fast food chain that maintains operations on Christmas Day",
        parent=parent_node,
        critical=False
    )

    # Reference URL presence (critical)
    evaluator.add_custom_node(
        result=bool(est and est.support_urls and len(est.support_urls) > 0),
        id="Reference_URL_FastFood",
        desc="Provide valid URL reference supporting the fast food chain's Christmas Day operations from search results",
        parent=cat_node,
        critical=True
    )

    # Chain identification (critical)
    chain_ident_node = evaluator.add_leaf(
        id="Chain_Identification_FastFood",
        desc="Identify a valid national fast food chain",
        parent=cat_node,
        critical=True
    )
    chain_name = _name_or_blank(est)
    await evaluator.verify(
        claim=f"'{chain_name}' is a national fast food (quick-service) restaurant chain.",
        node=chain_ident_node,
        sources=_urls_or_empty(est),
        additional_instruction="Accept typical QSR chains operating nationally across multiple states."
    )

    # Christmas Day availability (critical)
    xmas_avail_node = evaluator.add_leaf(
        id="Christmas_Day_Availability",
        desc="Verify that the chain has locations open on Christmas Day",
        parent=cat_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{chain_name}' has locations open on Christmas Day (Dec 25).",
        node=xmas_avail_node,
        sources=_urls_or_empty(est),
        additional_instruction=(
            "Accept 'some locations open' or 'hours vary by location'; explicit confirmation for Christmas Day is required."
        )
    )

    # Multi-holiday operations stance (non-critical)
    multi_holiday_node = evaluator.add_leaf(
        id="Multi_Holiday_Operations",
        desc="Confirm the chain's general approach to holiday operations (e.g., typically open on major holidays)",
        parent=cat_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The provided source(s) indicate '{chain_name}' general holiday operations policy (e.g., typically open on major holidays or with modified hours).",
        node=multi_holiday_node,
        sources=_urls_or_empty(est),
        additional_instruction="Any credible indication of general holiday-hours policy qualifies (corporate or reliable news/directory)."
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
    Evaluate an answer for the Holiday Resource Guide task.
    """
    # Initialize evaluator and root node
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
        default_model=model
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_holiday_resources(),
        template_class=HolidayResourcesExtraction,
        extraction_name="holiday_resources"
    )

    # Add a small debug summary of extracted names and URLs
    evaluator.add_custom_info(
        {
            "pharmacy": {
                "name": extracted.pharmacy.name if extracted.pharmacy else None,
                "urls": extracted.pharmacy.support_urls if (extracted.pharmacy and extracted.pharmacy.support_urls) else []
            },
            "convenience_store": {
                "name": extracted.convenience_store.name if extracted.convenience_store else None,
                "urls": extracted.convenience_store.support_urls if (extracted.convenience_store and extracted.convenience_store.support_urls) else []
            },
            "breakfast_restaurant": {
                "name": extracted.breakfast_restaurant.name if extracted.breakfast_restaurant else None,
                "urls": extracted.breakfast_restaurant.support_urls if (extracted.breakfast_restaurant and extracted.breakfast_restaurant.support_urls) else []
            },
            "fast_food": {
                "name": extracted.fast_food.name if extracted.fast_food else None,
                "urls": extracted.fast_food.support_urls if (extracted.fast_food and extracted.fast_food.support_urls) else []
            }
        },
        info_type="extracted_overview"
    )

    # Build subtrees for each category
    await verify_pharmacy_chain(evaluator, root, extracted.pharmacy)
    await verify_convenience_store_chain(evaluator, root, extracted.convenience_store)
    await verify_breakfast_restaurant_chain(evaluator, root, extracted.breakfast_restaurant)
    await verify_fast_food_chain(evaluator, root, extracted.fast_food)

    # Return final structured summary
    return evaluator.get_summary()