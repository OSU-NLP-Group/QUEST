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
TASK_ID = "la_xmas_open_2025_4types"
TASK_DESCRIPTION = """
Find four different types of retail establishments in Los Angeles, California that are confirmed to be open on Christmas Day 2025. You must identify one establishment from each of the following categories: (1) a pharmacy, (2) a convenience store, (3) a grocery store, and (4) a restaurant.

For each establishment, provide the following information:
- The specific store/restaurant name and brand
- The complete street address in Los Angeles, California
- The operating hours on December 25, 2025 (or confirmation of 24-hour operation)
- A reference URL that confirms the establishment is open on Christmas Day 2025
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Establishment(BaseModel):
    """Information for a single establishment."""
    name: Optional[str] = None
    address: Optional[str] = None
    hours_dec25_2025: Optional[str] = None  # e.g., "Open 24 hours", "8am–5pm", "Normal Sunday hours"
    reference_urls: List[str] = Field(default_factory=list)  # Pages confirming open on Dec 25, 2025


class XmasOpenExtraction(BaseModel):
    """Extraction container for the four required categories."""
    pharmacy: Optional[Establishment] = None
    convenience_store: Optional[Establishment] = None
    grocery_store: Optional[Establishment] = None
    restaurant: Optional[Establishment] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_establishments() -> str:
    return """
    Extract structured information for FOUR different Los Angeles (California) establishments that are open on Christmas Day 2025, one per each category:
    1) pharmacy, 2) convenience_store, 3) grocery_store, 4) restaurant.

    For each category, extract these fields from the answer exactly as written:
    - name: The store/restaurant name and brand (e.g., "CVS Pharmacy", "Walgreens", "7-Eleven", "Ralphs", "Denny's").
    - address: The complete street address for the specific Los Angeles location, including city and state (must be in Los Angeles, CA).
    - hours_dec25_2025: The stated operating hours for December 25, 2025 (e.g., "Open 24 hours", "8:00 AM–5:00 PM", "Open normal hours", or any explicit Dec 25 hours).
    - reference_urls: An array of 1–3 URLs that confirm the location is open on December 25, 2025. Prefer location-specific pages (store locator page, location detail page, hours page, or a Christmas/holiday-hours announcement that applies to that specific location). If multiple are provided in the answer, include all of them (up to 3). If the answer only gives a corporate policy page (e.g., "open 24/7"), include that URL.

    Important rules:
    - Do not invent any information. Extract only what appears in the answer.
    - If any of the four categories is missing in the answer, return null for that category.
    - If a field is not provided for a category, return null (or an empty array for reference_urls).
    - Keep addresses and hours as strings exactly as presented (do not normalize).
    - Only include URLs that appear in the answer. If no URLs are provided for a category, return an empty array.

    Output a JSON object mapping fields to these exact keys: pharmacy, convenience_store, grocery_store, restaurant.
    Each of those keys should map to an object with: name, address, hours_dec25_2025, reference_urls.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


async def verify_establishment(
    evaluator: Evaluator,
    parent_node,
    category_id: str,            # e.g., "Pharmacy_location"
    category_desc: str,          # description for the category node
    prefix: str,                 # e.g., "Pharmacy_" used for leaf IDs
    est: Optional[Establishment],
    kind_readable: str           # e.g., "pharmacy", "convenience store", "grocery store", "restaurant"
) -> None:
    """
    Build verification subtree for a single establishment category.
    Conforms to the rubric: a parallel node with 4 critical verification leaves.
    Adds a few critical "provided" gating checks as custom nodes to ensure robust dependency handling.
    """
    # Create the category node (parallel; non-critical to allow partial credit across categories)
    cat_node = evaluator.add_parallel(
        id=category_id,
        desc=category_desc,
        parent=parent_node,
        critical=False
    )

    # Basic provided-field gates (critical siblings). These gate the verification leaves.
    name_provided = _nonempty_str(est.name) if est else False
    addr_provided = _nonempty_str(est.address) if est else False
    hours_provided = _nonempty_str(est.hours_dec25_2025) if est else False
    urls_provided = bool(est and est.reference_urls and len(est.reference_urls) > 0)

    evaluator.add_custom_node(
        result=name_provided,
        id=f"{prefix}name_provided",
        desc=f"{kind_readable.capitalize()} name was provided in the answer",
        parent=cat_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=addr_provided,
        id=f"{prefix}address_provided",
        desc=f"{kind_readable.capitalize()} address was provided in the answer",
        parent=cat_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=hours_provided,
        id=f"{prefix}hours_provided",
        desc=f"{kind_readable.capitalize()} hours for Dec 25, 2025 were provided in the answer",
        parent=cat_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=urls_provided,
        id=f"{prefix}url_provided",
        desc=f"{kind_readable.capitalize()} has at least one reference URL",
        parent=cat_node,
        critical=True
    )

    # Prepare source URLs for evidence-grounded verification
    urls: List[str] = est.reference_urls[:5] if (est and est.reference_urls) else []
    srcs: List[str] | None = urls if urls else None

    # 1) Store/restaurant name verification
    name_leaf = evaluator.add_leaf(
        id=f"{prefix}store_name",   # e.g., "Pharmacy_store_name"
        desc=f"The specific name and brand of the {kind_readable}",
        parent=cat_node,
        critical=True
    )
    est_name = est.name if est else ""
    claim_name = (
        f"The provided webpage confirms that the establishment's name/brand for this Los Angeles location is "
        f"\"{est_name}\" (minor formatting or punctuation differences are acceptable), and the page is about a "
        f"location in Los Angeles, California."
    )
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        sources=srcs,
        additional_instruction=(
            "Determine if the page is clearly about the same establishment and location in Los Angeles, CA. "
            "Treat minor variations (case, punctuation, apostrophes, abbreviations) as a match. The page should "
            "be reasonably tied to the specific location (e.g., store locator page, location details, or a "
            "corporate page that unambiguously applies to all locations including Los Angeles)."
        )
    )

    # 2) Full address verification
    addr_leaf = evaluator.add_leaf(
        id=f"{prefix}full_address",  # e.g., "Pharmacy_full_address"
        desc=f"The complete street address of the {kind_readable} location in Los Angeles, California",
        parent=cat_node,
        critical=True
    )
    est_addr = est.address if est else ""
    claim_addr = (
        f"The webpage lists the full street address for the same location as \"{est_addr}\", and it is in Los Angeles, CA."
    )
    await evaluator.verify(
        claim=claim_addr,
        node=addr_leaf,
        sources=srcs,
        additional_instruction=(
            "Allow minor formatting differences (e.g., punctuation, abbreviations like 'Ave' vs 'Avenue', "
            "ZIP code presence/absence). The address content must clearly match and be in Los Angeles, California."
        )
    )

    # 3) Christmas Day 2025 hours / open verification
    hours_leaf = evaluator.add_leaf(
        id=f"{prefix}Christmas_hours",  # e.g., "Pharmacy_Christmas_hours"
        desc=f"The specific operating hours for this {kind_readable} on December 25, 2025 (or 24-hour confirmation)",
        parent=cat_node,
        critical=True
    )
    hours_text = est.hours_dec25_2025 if est else ""
    claim_hours = (
        "On December 25, 2025 (Christmas Day), this specific Los Angeles location is OPEN. "
        f"The page either explicitly lists Dec 25, 2025 hours matching or equivalent to \"{hours_text}\", "
        "mentions it is open on Christmas Day, or states 'Open 24 hours/24-7' in a way that reasonably applies to that date."
    )
    await evaluator.verify(
        claim=claim_hours,
        node=hours_leaf,
        sources=srcs,
        additional_instruction=(
            "Accept explicit 2025 holiday hour listings for Christmas Day, statements like 'Open on Christmas', "
            "or reliable indicators such as 'Open 24 hours / Open 24/7' that apply to that date. "
            "Small time-format differences (12h vs 24h, punctuation) should be tolerated when comparing the given "
            "hours text. If the page clearly indicates closure on Christmas Day, mark as not supported."
        )
    )

    # 4) Reference URL actually confirms open on Dec 25, 2025
    ref_leaf = evaluator.add_leaf(
        id=f"{prefix}reference_URL",  # e.g., "Pharmacy_reference_URL"
        desc=f"A URL that confirms this specific {kind_readable} location is open on Christmas Day 2025",
        parent=cat_node,
        critical=True
    )
    claim_ref = (
        f"The referenced webpage(s) confirm that the specific location at \"{est_addr}\" ({est_name}) in Los Angeles, "
        "CA is open on December 25, 2025—either via explicit holiday hours for that date, an 'open on Christmas' "
        "statement, or a clear 'open 24 hours' policy that applies to that date."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=srcs,
        additional_instruction=(
            "Focus on whether the provided page(s) support being open on Dec 25, 2025 for the specific location. "
            "Location-specific pages (store detail/locator) are ideal. Corporate policy pages that credibly apply "
            "to all locations (e.g., 'open 24/7 every day') are acceptable when reasonable."
        )
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
    Evaluate an answer for the 'Four retail establishments open on Christmas Day 2025 in Los Angeles' task.
    """
    # 1) Initialize evaluator and root node
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

    # 2) Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_establishments(),
        template_class=XmasOpenExtraction,
        extraction_name="extracted_establishments"
    )

    # 3) Build rubric root node (parallel over the four categories)
    rubric_root = evaluator.add_parallel(
        id="Four_retail_establishments_open_Christmas_Day_2025_Los_Angeles",
        desc="Four different types of retail establishments in Los Angeles, California that are confirmed to be open on Christmas Day 2025, including one pharmacy, one convenience store, one grocery store, and one restaurant.",
        parent=root,
        critical=False
    )

    # 4) Verify each category
    # Pharmacy
    await verify_establishment(
        evaluator=evaluator,
        parent_node=rubric_root,
        category_id="Pharmacy_location",
        category_desc="A pharmacy location in Los Angeles, California that is confirmed to be open on Christmas Day 2025",
        prefix="Pharmacy_",
        est=extracted.pharmacy,
        kind_readable="pharmacy"
    )

    # Convenience store
    await verify_establishment(
        evaluator=evaluator,
        parent_node=rubric_root,
        category_id="Convenience_store_location",
        category_desc="A convenience store location in Los Angeles, California that is confirmed to be open on Christmas Day 2025",
        prefix="Convenience_store_",
        est=extracted.convenience_store,
        kind_readable="convenience store"
    )

    # Grocery store
    await verify_establishment(
        evaluator=evaluator,
        parent_node=rubric_root,
        category_id="Grocery_store_location",
        category_desc="A grocery store location in Los Angeles, California that is confirmed to be open on Christmas Day 2025",
        prefix="Grocery_store_",
        est=extracted.grocery_store,
        kind_readable="grocery store"
    )

    # Restaurant
    await verify_establishment(
        evaluator=evaluator,
        parent_node=rubric_root,
        category_id="Restaurant_location",
        category_desc="A restaurant location in Los Angeles, California that is confirmed to be open on Christmas Day 2025",
        prefix="Restaurant_",
        est=extracted.restaurant,
        kind_readable="restaurant"
    )

    # 5) Return structured summary
    return evaluator.get_summary()