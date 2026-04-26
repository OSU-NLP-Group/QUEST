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
TASK_ID = "bf_craft_store_2025"
TASK_DESCRIPTION = (
    "You live in California and are planning to shop for craft supplies early on Black Friday morning 2025 "
    "(November 28) to complete three DIY Thanksgiving projects before guests arrive: (1) a decorative wreath, "
    "(2) painted wooden decorations, and (3) a yarn-based table centerpiece. You want to visit a major craft store chain "
    "that opens earliest to avoid crowds and take advantage of Black Friday deals.\n\n"
    "Identify which major national craft store chain (between Michaels and Hobby Lobby) opens earliest on Black Friday 2025, "
    "and provide the following information:\n"
    "- The store's Black Friday opening time\n"
    "- Confirmation that this store is closed on Thanksgiving Day (November 27, 2025)\n"
    "- Confirmation that this store has locations in California\n"
    "- Three specific product categories available at this store that match your supply needs (one for wreath materials, "
    "one for paint supplies, and one for yarn/fabric), along with the Black Friday 2025 discount percentage for each category"
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CategoryDeal(BaseModel):
    """A product category and its Black Friday 2025 discount with sources."""
    category: Optional[str] = None
    discount_percent: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ShoppingPlanExtraction(BaseModel):
    """Structured extraction for store choice, opening times, confirmations, and deals."""
    chosen_chain: Optional[str] = None  # "Michaels" or "Hobby Lobby"
    black_friday_opening_time: Optional[str] = None
    black_friday_open_time_sources: List[str] = Field(default_factory=list)

    thanksgiving_closed_confirmation: Optional[str] = None  # e.g., "Closed on Thanksgiving"
    thanksgiving_sources: List[str] = Field(default_factory=list)

    california_locations_confirmation: Optional[str] = None  # e.g., "Has CA stores"
    california_sources: List[str] = Field(default_factory=list)

    # Opening time details for both chains, if mentioned
    michaels_black_friday_opening_time: Optional[str] = None
    michaels_opening_sources: List[str] = Field(default_factory=list)
    hobby_lobby_black_friday_opening_time: Optional[str] = None
    hobby_lobby_opening_sources: List[str] = Field(default_factory=list)

    # Category deals
    wreath_deal: Optional[CategoryDeal] = None
    paint_deal: Optional[CategoryDeal] = None
    yarn_or_fabric_deal: Optional[CategoryDeal] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shopping_plan() -> str:
    return (
        "Extract the following structured information from the answer:\n"
        "1) chosen_chain: Which chain is identified as opening earliest on Black Friday 2025; must be exactly 'Michaels' or 'Hobby Lobby'.\n"
        "2) black_friday_opening_time: The opening time for the chosen chain on Black Friday 2025 (Nov 28), as stated.\n"
        "3) black_friday_open_time_sources: URLs that support the stated Black Friday opening time for the chosen chain. Extract only explicit URLs.\n"
        "4) thanksgiving_closed_confirmation: A statement confirming the chosen chain is closed on Thanksgiving Day (Nov 27, 2025), if present; otherwise null.\n"
        "5) thanksgiving_sources: URLs that support the Thanksgiving closure statement for the chosen chain. Extract only explicit URLs.\n"
        "6) california_locations_confirmation: A statement confirming the chosen chain has store locations in California, if present; otherwise null.\n"
        "7) california_sources: URLs that support the California locations confirmation (e.g., store locator pages). Extract only explicit URLs.\n"
        "8) michaels_black_friday_opening_time: If the answer mentions Michaels' Black Friday 2025 opening time, extract it; otherwise null.\n"
        "9) michaels_opening_sources: URLs supporting Michaels' Black Friday opening time.\n"
        "10) hobby_lobby_black_friday_opening_time: If the answer mentions Hobby Lobby's Black Friday 2025 opening time, extract it; otherwise null.\n"
        "11) hobby_lobby_opening_sources: URLs supporting Hobby Lobby's Black Friday opening time.\n"
        "12) wreath_deal: One category suitable for wreath materials (e.g., 'Floral & wreath supplies', 'Floral picks', 'Wreath forms'), the Black Friday 2025 discount percent, and URLs supporting it.\n"
        "13) paint_deal: One category suitable for paint supplies (e.g., 'Acrylic paint', 'Craft paint', 'Paint pens'), the Black Friday 2025 discount percent, and URLs supporting it.\n"
        "14) yarn_or_fabric_deal: One category suitable for yarn/fabric supplies (e.g., 'Yarn', 'Fleece fabric'), the Black Friday 2025 discount percent, and URLs supporting it.\n\n"
        "Return a JSON object with keys exactly matching the template fields. If any item is missing in the answer, set it to null or an empty list as appropriate. "
        "For URLs, extract only valid URLs explicitly present in the answer (including markdown link targets)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_chain_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower()
    if "michaels" in n:
        return "Michaels"
    if "hobby" in n and "lobby" in n:
        return "Hobby Lobby"
    return name.strip()


def other_chain(chain: Optional[str]) -> Optional[str]:
    c = normalize_chain_name(chain)
    if c == "Michaels":
        return "Hobby Lobby"
    if c == "Hobby Lobby":
        return "Michaels"
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_store_facts(
    evaluator: Evaluator,
    parent_node,
    info: ShoppingPlanExtraction,
) -> None:
    """
    Build and verify the 'earliest_store_choice_and_store_facts' subtree.
    """
    node = evaluator.add_parallel(
        id="earliest_store_choice_and_store_facts",
        desc="Correctly select the earliest-opening chain (between Michaels and Hobby Lobby) for Black Friday 2025 and provide required confirmations.",
        parent=parent_node,
        critical=True,
    )

    chosen = normalize_chain_name(info.chosen_chain) or ""
    m_time = info.michaels_black_friday_opening_time or ""
    h_time = info.hobby_lobby_black_friday_opening_time or ""
    m_srcs = info.michaels_opening_sources or []
    h_srcs = info.hobby_lobby_opening_sources or []

    # 1) Earliest store identified correctly
    earliest_leaf = evaluator.add_leaf(
        id="earliest_store_identified_correctly",
        desc="Identifies the correct earliest-opening chain for Black Friday 2025 (as determined by the provided Black Friday opening-time constraints).",
        parent=node,
        critical=True,
    )
    if m_time.strip() and h_time.strip():
        earliest_claim = (
            f"On Black Friday 2025 (Nov 28), Michaels opens at '{m_time}', and Hobby Lobby opens at '{h_time}'. "
            f"Therefore, the earliest-opening chain between Michaels and Hobby Lobby is '{chosen}'."
        )
        earliest_sources: List[str] = list(m_srcs) + list(h_srcs)
    else:
        # Fall back to whatever sources are available (including chosen chain's opening time sources)
        earliest_claim = (
            f"The earliest-opening chain between Michaels and Hobby Lobby on Black Friday 2025 is '{chosen}'. "
            f"This judgment should be supported by the provided opening-time sources."
        )
        earliest_sources = list(info.black_friday_open_time_sources or []) + list(m_srcs) + list(h_srcs)

    await evaluator.verify(
        claim=earliest_claim,
        node=earliest_leaf,
        sources=earliest_sources,
        additional_instruction=(
            "Use the provided URLs to compare Michaels vs Hobby Lobby opening times for Black Friday 2025. "
            "Accept common corporate or ad-published opening times (e.g., 'doors open at X AM'). "
            "If both times are available, pick the chain with the earlier time as earliest."
        ),
    )

    # 2) Black Friday opening time provided (verify the chosen chain's time against sources)
    opening_leaf = evaluator.add_leaf(
        id="black_friday_opening_time_provided",
        desc="Provides the identified store's Black Friday 2025 opening time.",
        parent=node,
        critical=True,
    )
    # Combine chosen chain time sources with chain-specific ones
    chosen_time = info.black_friday_opening_time or ""
    chosen_sources = list(info.black_friday_open_time_sources or [])
    if chosen == "Michaels":
        chosen_sources = chosen_sources + list(m_srcs)
    elif chosen == "Hobby Lobby":
        chosen_sources = chosen_sources + list(h_srcs)

    opening_claim = f"On Black Friday 2025 (November 28), '{chosen}' opens at '{chosen_time}'."
    await evaluator.verify(
        claim=opening_claim,
        node=opening_leaf,
        sources=chosen_sources,
        additional_instruction=(
            "Verify the stated opening time for the chosen chain using the provided URLs. "
            "Minor phrasing variations are acceptable (e.g., 'doors open' vs 'store opens')."
        ),
    )

    # 3) Thanksgiving closed confirmation
    thanksgiving_leaf = evaluator.add_leaf(
        id="thanksgiving_closed_confirmation",
        desc="Confirms the identified store is closed on Thanksgiving Day (Nov 27, 2025).",
        parent=node,
        critical=True,
    )
    thanksgiving_claim = f"'{chosen}' is closed on Thanksgiving Day (November 27, 2025)."
    await evaluator.verify(
        claim=thanksgiving_claim,
        node=thanksgiving_leaf,
        sources=info.thanksgiving_sources,
        additional_instruction=(
            "Confirm that the chosen chain states it is closed on Thanksgiving Day. "
            "Corporate policy pages, holiday hours pages, or official announcements are acceptable."
        ),
    )

    # 4) California locations confirmation
    ca_leaf = evaluator.add_leaf(
        id="california_locations_confirmation",
        desc="Confirms the identified store has locations in California.",
        parent=node,
        critical=True,
    )
    ca_claim = f"'{chosen}' has store locations in California."
    await evaluator.verify(
        claim=ca_claim,
        node=ca_leaf,
        sources=info.california_sources,
        additional_instruction=(
            "Use store locator or official pages to confirm California presence. "
            "Listings or maps showing California stores are acceptable evidence."
        ),
    )


async def verify_product_categories_and_discounts(
    evaluator: Evaluator,
    parent_node,
    info: ShoppingPlanExtraction,
) -> None:
    """
    Build and verify the 'product_categories_and_discounts' subtree for wreath, paint, and yarn/fabric needs.
    """
    node = evaluator.add_parallel(
        id="product_categories_and_discounts",
        desc="Provides three product categories at the identified store matching the three DIY supply needs, each with a Black Friday 2025 discount percentage.",
        parent=parent_node,
        critical=True,
    )

    chosen = normalize_chain_name(info.chosen_chain) or ""

    # Helper to add a single category verification
    async def add_category_leaf(
        leaf_id: str,
        leaf_desc: str,
        deal: Optional[CategoryDeal],
    ):
        leaf = evaluator.add_leaf(
            id=leaf_id,
            desc=leaf_desc,
            parent=node,
            critical=True,
        )
        category_name = (deal.category if deal else "") or ""
        discount = (deal.discount_percent if deal else "") or ""
        sources = list(deal.sources if deal and deal.sources else [])

        claim = (
            f"At '{chosen}', the product category '{category_name}' is available and has a Black Friday 2025 discount of '{discount}'."
        )
        add_ins = (
            "Verify that the category exists at the chosen chain, and that the stated Black Friday 2025 discount is supported. "
            "Accept phrasing like 'up to X% off' or 'X% off regular price' as matching if consistent. "
            "Use official ads, sale pages, or weekly ad content from the provided URLs."
        )
        await evaluator.verify(claim=claim, node=leaf, sources=sources, additional_instruction=add_ins)

    # Wreath materials category
    await add_category_leaf(
        leaf_id="wreath_category_and_discount",
        leaf_desc="Gives one store-available product category suitable for wreath materials and states its Black Friday 2025 discount percentage (consistent with the constraints for that store/category).",
        deal=info.wreath_deal,
    )
    # Paint supplies category
    await add_category_leaf(
        leaf_id="paint_category_and_discount",
        leaf_desc="Gives one store-available product category suitable for paint supplies and states its Black Friday 2025 discount percentage (consistent with the constraints for that store/category).",
        deal=info.paint_deal,
    )
    # Yarn/fabric category
    await add_category_leaf(
        leaf_id="yarn_or_fabric_category_and_discount",
        leaf_desc="Gives one store-available product category suitable for yarn/fabric supplies and states its Black Friday 2025 discount percentage (consistent with the constraints for that store/category).",
        deal=info.yarn_or_fabric_deal,
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
    Evaluate an answer for the Black Friday 2025 craft store planning task.
    """
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

    # Extract structured information from the answer
    plan_info = await evaluator.extract(
        prompt=prompt_extract_shopping_plan(),
        template_class=ShoppingPlanExtraction,
        extraction_name="shopping_plan_extraction",
    )

    # Build top-level critical node as per rubric
    shopping_plan_node = evaluator.add_parallel(
        id="shopping_plan",
        desc="Answer identifies the earliest-opening chain between Michaels and Hobby Lobby for Black Friday 2025 and provides required store facts and three relevant product categories with discounts.",
        parent=root,
        critical=True,
    )

    # Subtree: earliest store choice and store facts
    await verify_store_facts(evaluator, shopping_plan_node, plan_info)

    # Subtree: product categories and discounts
    await verify_product_categories_and_discounts(evaluator, shopping_plan_node, plan_info)

    # Return structured evaluation summary
    return evaluator.get_summary()