import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "woodworking_store_msp"
TASK_DESCRIPTION = (
    "Identify a craft or hardware store located in the Minneapolis-Saint Paul metropolitan area, Minnesota, "
    "that sells beginner woodworking tools (such as drills, saws, measuring tools, sanders, or clamps) and safety "
    "equipment (including safety glasses and dust masks). Provide the store's complete address, weekend operating "
    "hours (Saturday and Sunday), and contact information (phone number or website)."
)

ESSENTIAL_TOOL_CATEGORIES = [
    "cordless drill",
    "circular saw or miter saw",
    "measuring tools",
    "sander",
    "clamps",
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WeekendHours(BaseModel):
    saturday: Optional[str] = None
    sunday: Optional[str] = None


class StoreExtraction(BaseModel):
    store_name: Optional[str] = None
    address_full: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None

    weekend_hours: WeekendHours = WeekendHours()

    phone: Optional[str] = None
    website: Optional[str] = None

    # Tools: restrict to the essential set; use exactly the category names listed in ESSENTIAL_TOOL_CATEGORIES
    tool_categories: List[str] = Field(default_factory=list)

    # Safety gear: list items mentioned in the answer; include canonical entries if present:
    # "safety glasses" and "dust masks or respirators"
    safety_items: List[str] = Field(default_factory=list)

    # Source URLs cited in the answer; each list should contain actual URLs
    address_sources: List[str] = Field(default_factory=list)
    hours_sources: List[str] = Field(default_factory=list)
    contact_sources: List[str] = Field(default_factory=list)
    product_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_store_info() -> str:
    return """
    Extract information about a single craft or hardware store suitable for purchasing beginner woodworking supplies
    from the provided answer. If multiple stores are mentioned, pick the first and ignore the rest.

    Return a JSON object following this schema:

    {
      "store_name": string | null,
      "address_full": string | null,  // Full street address as written in the answer
      "city": string | null,
      "state": string | null,         // Prefer "MN" or "Minnesota"
      "zip_code": string | null,
      "weekend_hours": {
        "saturday": string | null,    // e.g., "9am–6pm" or "Closed"
        "sunday": string | null
      },
      "phone": string | null,         // digits, possibly with formatting; null if not present
      "website": string | null,       // full URL if provided, null otherwise

      "tool_categories": [string],    // Categories (as explicitly claimed in the answer) chosen from:
                                      // ["cordless drill", "circular saw or miter saw", "measuring tools", "sander", "clamps"]
                                      // Include only those categories the answer explicitly claims the store sells.
      "safety_items": [string],       // Include items explicitly claimed. Normalize to include canonical entries if present:
                                      // "safety glasses" and "dust masks or respirators" (accept synonyms like "dust masks", "respirator", "face mask")

      "address_sources": [url],       // URLs cited in the answer that support the address/location
      "hours_sources": [url],         // URLs cited in the answer that support the weekend hours
      "contact_sources": [url],       // URLs cited in the answer that support phone or website
      "product_sources": [url]        // URLs cited in the answer that support product availability (tools/safety gear)
    }

    Rules:
    - Extract only what is explicitly present in the answer. Do not invent missing information.
    - Each URL list should contain actual URLs mentioned in the answer. If no URLs are provided, return an empty list.
    - For "tool_categories", include only values from the provided canonical set exactly as written above.
    - For safety items, include canonical entries when possible. Accept synonyms but normalize them to canonical entries if clearly equivalent.
    - If any required field is missing, set it to null. If a sources list is missing, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def _merge_sources(primary: List[str], secondary: List[str], website: Optional[str]) -> List[str]:
    """
    Merge source lists with a website fallback. Ensures at least one URL if possible.
    """
    merged = []
    for src in primary:
        if isinstance(src, str) and src.strip():
            merged.append(src.strip())
    for src in secondary:
        if isinstance(src, str) and src.strip():
            merged.append(src.strip())
    if not merged and website and isinstance(website, str) and website.strip():
        merged.append(website.strip())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in merged:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def _normalize_bool(val: Optional[str]) -> bool:
    return bool(val and str(val).strip())


def _select_first_k(items: List[str], k: int) -> List[str]:
    return items[:k] if len(items) >= k else items


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_store_location(evaluator: Evaluator, parent_node, info: StoreExtraction) -> None:
    """
    Build and verify the 'store_location_verification' subtree.
    """
    node = evaluator.add_parallel(
        id="store_location_verification",
        desc="The store must be physically located within the Minneapolis-Saint Paul metropolitan area, Minnesota",
        parent=parent_node,
        critical=True,
    )

    # Existence prerequisite
    exists = _normalize_bool(info.address_full) and _normalize_bool(info.city) and _normalize_bool(info.state)
    existence_node = evaluator.add_custom_node(
        result=exists,
        id="store_location_info_provided",
        desc="Location information (address/city/state) is provided",
        parent=node,
        critical=True,
    )

    # Sources
    location_sources = _merge_sources(info.address_sources, info.contact_sources, info.website)

    # Address supported
    addr_leaf = evaluator.add_leaf(
        id="address_supported_location",
        desc="Store address is supported by cited sources",
        parent=node,
        critical=True,
    )
    address_claim = f"The store's address is '{info.address_full}' in {info.city}, {info.state} {info.zip_code or ''}."
    await evaluator.verify(
        claim=address_claim,
        node=addr_leaf,
        sources=location_sources,
        additional_instruction="Verify that the cited page displays the same address (minor formatting differences acceptable).",
        extra_prerequisites=[existence_node],
    )

    # State supported (Minnesota)
    state_leaf = evaluator.add_leaf(
        id="in_minnesota_supported",
        desc="Store is located in Minnesota",
        parent=node,
        critical=True,
    )
    mn_claim = f"The store is located in the state of Minnesota."
    await evaluator.verify(
        claim=mn_claim,
        node=state_leaf,
        sources=location_sources,
        additional_instruction="Confirm the page shows the city/state or address indicating Minnesota (MN).",
        extra_prerequisites=[existence_node],
    )

    # Metro area supported
    metro_leaf = evaluator.add_leaf(
        id="in_msp_metro_supported",
        desc="Store is within the Minneapolis–Saint Paul metropolitan area",
        parent=node,
        critical=True,
    )
    metro_claim = (
        f"This store at '{info.address_full}' in {info.city}, {info.state} is in the Minneapolis–Saint Paul metropolitan area."
    )
    await evaluator.verify(
        claim=metro_claim,
        node=metro_leaf,
        sources=location_sources,
        additional_instruction=(
            "You are explicitly allowed to use general US metro-area geography knowledge to determine whether the city "
            "belongs to the Minneapolis–Saint Paul metropolitan area. If the page clearly shows the city (e.g., Minneapolis, "
            "Saint Paul, Bloomington, Minnetonka, Maplewood, Roseville, etc.), consider it within MSP metro."
        ),
        extra_prerequisites=[existence_node],
    )


async def verify_essential_tools(evaluator: Evaluator, parent_node, info: StoreExtraction) -> None:
    """
    Build and verify the 'essential_tools_availability' subtree.
    Must sell at least three categories among ESSENTIAL_TOOL_CATEGORIES.
    We verify at most the first 3 categories claimed in the answer to avoid over-penalization.
    """
    node = evaluator.add_parallel(
        id="essential_tools_availability",
        desc="The store must sell at least three of the five essential beginner woodworking tools",
        parent=parent_node,
        critical=True,
    )

    # Normalize categories: keep only those in the canonical list and pick first 3
    claimed = [c for c in info.tool_categories if c in ESSENTIAL_TOOL_CATEGORIES]
    first_three = _select_first_k(claimed, 3)

    # Enforce at least three claimed categories
    min_three_node = evaluator.add_custom_node(
        result=len(first_three) >= 3,
        id="tools_minimum_three_claimed",
        desc="At least three essential tool categories are claimed for this store",
        parent=node,
        critical=True,
    )

    # Sources for products
    product_sources = _merge_sources(info.product_sources, [], info.website)

    # For each selected category, verify sale support
    for idx, cat in enumerate(first_three):
        leaf = evaluator.add_leaf(
            id=f"tool_category_{idx}_{cat.replace(' ', '_').replace('/', '_')}_sold",
            desc=f"Store sells '{cat}'",
            parent=node,
            critical=True,
        )

        if cat == "circular saw or miter saw":
            claim = (
                "This store sells either a circular saw or a miter saw (any variant such as 'compound miter saw' counts)."
            )
            add_ins = (
                "Look for product pages or category listings for circular saws or miter saws. Minor naming differences are acceptable."
            )
        elif cat == "measuring tools":
            claim = (
                "This store sells measuring tools (e.g., tape measure, square, calipers, ruler, marking gauge)."
            )
            add_ins = (
                "Any typical measuring device in woodworking counts. Confirm that the page shows such items available for purchase."
            )
        elif cat == "cordless drill":
            claim = "This store sells a cordless drill (including driver kits or drill/driver combos)."
            add_ins = "Any cordless drill/driver kit or product listing suffices."
        elif cat == "sander":
            claim = "This store sells a sander (e.g., random orbital sander, belt sander, palm sander)."
            add_ins = "Any sander type counts."
        elif cat == "clamps":
            claim = "This store sells clamps (e.g., bar clamps, F-clamps, C-clamps, spring clamps)."
            add_ins = "Any common clamp type counts."
        else:
            claim = f"This store sells items from the category '{cat}'."
            add_ins = "Confirm sale using product/category pages or listings."

        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=product_sources,
            additional_instruction=add_ins,
            extra_prerequisites=[min_three_node],
        )


async def verify_safety_equipment(evaluator: Evaluator, parent_node, info: StoreExtraction) -> None:
    """
    Build and verify the 'safety_equipment_availability' subtree.
    Must sell both safety glasses and dust masks or respirators.
    """
    node = evaluator.add_parallel(
        id="safety_equipment_availability",
        desc="The store must sell woodworking safety equipment including safety glasses and dust masks or respirators",
        parent=parent_node,
        critical=True,
    )

    # Existence prerequisite: both safety glasses and dust masks/respirators claimed
    has_glasses = any(si.lower() in ["safety glasses", "glasses"] for si in [*(info.safety_items or [])])
    has_masks_or_resps = any(
        ("dust mask" in si.lower()) or ("respirator" in si.lower()) for si in [*(info.safety_items or [])]
    )
    existence_node = evaluator.add_custom_node(
        result=has_glasses and has_masks_or_resps,
        id="safety_items_both_claimed",
        desc="Both 'safety glasses' and 'dust masks or respirators' are claimed in the answer",
        parent=node,
        critical=True,
    )

    product_sources = _merge_sources(info.product_sources, [], info.website)

    # Safety glasses
    glasses_leaf = evaluator.add_leaf(
        id="safety_glasses_sold",
        desc="Store sells safety glasses",
        parent=node,
        critical=True,
    )
    glasses_claim = "This store sells safety glasses (any protective eyewear for woodworking)."
    await evaluator.verify(
        claim=glasses_claim,
        node=glasses_leaf,
        sources=product_sources,
        additional_instruction="Confirm via product listings or category pages that safety glasses are available.",
        extra_prerequisites=[existence_node],
    )

    # Dust masks or respirators
    masks_leaf = evaluator.add_leaf(
        id="dust_masks_or_respirators_sold",
        desc="Store sells dust masks or respirators",
        parent=node,
        critical=True,
    )
    masks_claim = "This store sells dust masks or respirators suitable for woodworking."
    await evaluator.verify(
        claim=masks_claim,
        node=masks_leaf,
        sources=product_sources,
        additional_instruction="Listings for N95 masks, reusable respirators, or disposable dust masks all count.",
        extra_prerequisites=[existence_node],
    )


async def verify_weekend_hours(evaluator: Evaluator, parent_node, info: StoreExtraction) -> None:
    """
    Build and verify the 'weekend_hours_provided' subtree.
    Must provide Saturday and Sunday hours, supported by sources.
    """
    node = evaluator.add_parallel(
        id="weekend_hours_provided",
        desc="Weekend operating hours (Saturday and Sunday) must be provided",
        parent=parent_node,
        critical=True,
    )

    exists = _normalize_bool(info.weekend_hours.saturday) and _normalize_bool(info.weekend_hours.sunday)
    existence_node = evaluator.add_custom_node(
        result=exists,
        id="weekend_hours_existence",
        desc="Both Saturday and Sunday hours are provided in the answer",
        parent=node,
        critical=True,
    )

    hours_sources = _merge_sources(info.hours_sources, info.address_sources, info.website)

    # Saturday hours
    sat_leaf = evaluator.add_leaf(
        id="saturday_hours_supported",
        desc="Saturday hours are supported by cited sources",
        parent=node,
        critical=True,
    )
    sat_claim = f"Saturday hours are '{info.weekend_hours.saturday}'."
    await evaluator.verify(
        claim=sat_claim,
        node=sat_leaf,
        sources=hours_sources,
        additional_instruction=(
            "Verify the store's Saturday hours on the cited page. Minor formatting differences are acceptable."
        ),
        extra_prerequisites=[existence_node],
    )

    # Sunday hours
    sun_leaf = evaluator.add_leaf(
        id="sunday_hours_supported",
        desc="Sunday hours are supported by cited sources",
        parent=node,
        critical=True,
    )
    sun_claim = f"Sunday hours are '{info.weekend_hours.sunday}'."
    await evaluator.verify(
        claim=sun_claim,
        node=sun_leaf,
        sources=hours_sources,
        additional_instruction=(
            "Verify the store's Sunday hours on the cited page. Minor formatting differences are acceptable."
        ),
        extra_prerequisites=[existence_node],
    )


async def verify_contact_information(evaluator: Evaluator, parent_node, info: StoreExtraction) -> None:
    """
    Build and verify the 'store_contact_information' subtree.
    Must provide address and at least one of phone or website, supported by sources.
    """
    node = evaluator.add_parallel(
        id="store_contact_information",
        desc="Store address and phone number or website must be provided",
        parent=parent_node,
        critical=True,
    )

    has_address = _normalize_bool(info.address_full)
    has_phone_or_site = _normalize_bool(info.phone) or _normalize_bool(info.website)
    existence_node = evaluator.add_custom_node(
        result=has_address and has_phone_or_site,
        id="contact_info_existence",
        desc="Address plus at least one contact method (phone or website) are provided",
        parent=node,
        critical=True,
    )

    contact_sources = _merge_sources(info.contact_sources, info.address_sources, info.website)

    # Address supported (contact block)
    addr_leaf = evaluator.add_leaf(
        id="address_supported_contact",
        desc="Store address (contact section) is supported by cited sources",
        parent=node,
        critical=True,
    )
    addr_claim = f"The store's address is '{info.address_full}' in {info.city}, {info.state} {info.zip_code or ''}."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_leaf,
        sources=contact_sources,
        additional_instruction="Verify the address matches the cited page. Minor formatting differences are acceptable.",
        extra_prerequisites=[existence_node],
    )

    # Phone OR website supported
    contact_leaf = evaluator.add_leaf(
        id="phone_or_website_supported",
        desc="At least one contact method (phone or website) is supported by cited sources",
        parent=node,
        critical=True,
    )
    contact_claim = (
        f"At least one contact method is provided and supported: phone='{info.phone or ''}' or website='{info.website or ''}'."
    )
    await evaluator.verify(
        claim=contact_claim,
        node=contact_leaf,
        sources=contact_sources,
        additional_instruction=(
            "Pass if either the provided phone number appears on the cited page or the provided website URL is the store's site."
        ),
        extra_prerequisites=[existence_node],
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
    Evaluate an answer for the Minneapolis–Saint Paul woodworking store identification task.
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

    # Extract structured store information from the answer
    store_info = await evaluator.extract(
        prompt=prompt_extract_store_info(),
        template_class=StoreExtraction,
        extraction_name="store_info",
    )

    # Build top-level critical node per rubric
    main_node = evaluator.add_parallel(
        id="beginner_woodworking_store_identification",
        desc="Identify a craft or hardware store suitable for purchasing beginner woodworking supplies",
        parent=root,
        critical=True,
    )

    # Sub-verifications (all critical under the main critical node)
    await verify_store_location(evaluator, main_node, store_info)
    await verify_essential_tools(evaluator, main_node, store_info)
    await verify_safety_equipment(evaluator, main_node, store_info)
    await verify_weekend_hours(evaluator, main_node, store_info)
    await verify_contact_information(evaluator, main_node, store_info)

    # Return summary
    return evaluator.get_summary()