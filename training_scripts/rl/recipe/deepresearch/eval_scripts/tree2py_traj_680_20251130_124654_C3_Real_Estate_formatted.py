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
TASK_ID = "charleston_record_sale_2025"
TASK_DESCRIPTION = (
    "In 2025, a hedge fund manager who became U.S. Treasury Secretary sold a historic property in Charleston, "
    "South Carolina, setting a new record for the Charleston peninsula. This individual is known for restoring "
    "historic properties and purchased this particular property in the 2010s. The property is located in the South "
    "of Broad historic district on the High Battery. Identify the following information about this transaction: "
    "(1) The full name of the person, (2) The complete street address of the property, (3) The year the property was "
    "originally built (circa), (4) The architectural style of the property, (5) The year this person purchased the "
    "property, (6) The purchase price in millions of dollars, (7) The date the sale closed in 2025, (8) The sale price "
    "of the real estate portion (excluding contents) in millions of dollars."
)

# Ground-truth constraints known for the correct property (used for context in summary and guidance for verification).
GROUND_TRUTH_CONSTRAINTS = {
    "built_year_circa": "1848",
    "architectural_style": "Italianate",
    "purchase_year": "2016",
    "purchase_price_millions": "6.5",
    "sale_close_date_2025": "February 28, 2025",
    "sale_price_real_estate_only_millions": "18.25",
    # Person and full address are verified against sources rather than fixed strings
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    """Represents a single field value and its cited URLs from the answer."""
    text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TransactionExtraction(BaseModel):
    """Complete extraction schema for the eight requested fields."""
    person_full_name: Optional[FieldWithSources] = None
    property_street_address: Optional[FieldWithSources] = None
    built_year_circa: Optional[FieldWithSources] = None
    architectural_style: Optional[FieldWithSources] = None
    purchase_year: Optional[FieldWithSources] = None
    purchase_price_millions: Optional[FieldWithSources] = None
    sale_close_date_2025: Optional[FieldWithSources] = None
    sale_price_real_estate_only_millions: Optional[FieldWithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_transaction() -> str:
    return """
    Extract the eight requested fields for the described 2025 Charleston historic property transaction from the provided answer text.
    For each field, return an object with:
    - text: the value exactly as stated in the answer (string, or null if missing)
    - sources: an array of URLs explicitly cited in the answer that support this specific field (can be empty if none given)

    The eight fields and their JSON keys are:
    1. person_full_name
    2. property_street_address
    3. built_year_circa
    4. architectural_style
    5. purchase_year
    6. purchase_price_millions
    7. sale_close_date_2025
    8. sale_price_real_estate_only_millions

    Special instructions:
    - Only include URLs explicitly present in the answer text. Do not infer or add new URLs.
    - For money fields, if the answer uses full amounts like "6,500,000", still extract the text exactly as shown (do not transform).
    - For dates, return the exact formatting as stated (e.g., "February 28, 2025", "Feb. 28, 2025", "2025-02-28").
    - If the answer provides one general source list, attribute each relevant URL to the corresponding fields where appropriate; otherwise leave sources empty for fields without explicit citation.
    - If a field is not mentioned in the answer, set its "text" to null and "sources" to an empty list.

    Return a single JSON object with the eight keys above, each mapping to { "text": ..., "sources": [...] }.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _get_text(field: Optional[FieldWithSources]) -> str:
    return (field.text or "").strip() if field else ""


def _get_sources(field: Optional[FieldWithSources]) -> List[str]:
    return field.sources if (field and field.sources) else []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _build_sequential_field_check(
    evaluator: Evaluator,
    parent,
    group_id: str,
    group_desc: str,
    existence_leaf_id: str,
    existence_desc: str,
    value_text: str,
    verify_leaf_id: str,
    verify_desc: str,
    claim: str,
    sources: List[str],
    additional_instruction: str,
) -> None:
    """
    Create a critical sequential group with:
    - A critical existence check leaf
    - A critical verification leaf that checks the claim against sources (auto-skipped if existence fails)
    """
    group_node = evaluator.add_sequential(
        id=group_id,
        desc=group_desc,
        parent=parent,
        critical=True
    )

    exists_node = evaluator.add_custom_node(
        result=bool(value_text),
        id=existence_leaf_id,
        desc=existence_desc,
        parent=group_node,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id=verify_leaf_id,
        desc=verify_desc,
        parent=group_node,
        critical=True
    )

    # Perform URL-backed verification; if sources empty, framework auto-routes to simple verification.
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=sources,
        additional_instruction=additional_instruction
    )


async def verify_transaction_fields(
    evaluator: Evaluator,
    critical_root,
    tx: TransactionExtraction
) -> None:
    """
    Build the verification tree for all eight fields, enforcing existence first and then URL-backed verification.
    All nodes are critical and organized as sequential groups under a critical parallel root to reflect the rubric.
    """

    # 1) Person full name
    person_value = _get_text(tx.person_full_name)
    person_sources = _get_sources(tx.person_full_name)
    await _build_sequential_field_check(
        evaluator=evaluator,
        parent=critical_root,
        group_id="person_full_name_group",
        group_desc="Person identity field checks",
        existence_leaf_id="person_full_name_exists",
        existence_desc="Person full name is provided in the answer",
        value_text=person_value,
        verify_leaf_id="person_full_name",
        verify_desc="Identifies the correct person by full name: the hedge fund manager who became U.S. Treasury Secretary in 2025 and is known for restoring historic properties (per constraints)",
        claim=f"The seller is {person_value}.",
        sources=person_sources,
        additional_instruction=(
            "Verify that the cited page(s) indicate this person is the seller of the property in question. "
            "Context: The person is a hedge fund manager who became U.S. Treasury Secretary in 2025 and is known for restoring historic properties. "
            "Allow minor name variants (e.g., middle initials) but ensure the identity matches."
        ),
    )

    # 2) Property complete street address
    addr_value = _get_text(tx.property_street_address)
    addr_sources = _get_sources(tx.property_street_address)
    await _build_sequential_field_check(
        evaluator=evaluator,
        parent=critical_root,
        group_id="property_street_address_group",
        group_desc="Property address field checks",
        existence_leaf_id="property_street_address_exists",
        existence_desc="Property street address is provided in the answer",
        value_text=addr_value,
        verify_leaf_id="property_street_address",
        verify_desc="Provides the complete street address of the correct property in Charleston, South Carolina, located in the South of Broad historic district on the High Battery (per constraints)",
        claim=f"The complete street address of the property is {addr_value}, Charleston, South Carolina.",
        sources=addr_sources,
        additional_instruction=(
            "Verify that the page(s) state this exact address, and that it is located in the South of Broad historic district on the High Battery. "
            "Allow minor formatting variations (commas, 'St.' vs 'Street')."
        ),
    )

    # 3) Built year (circa)
    built_value = _get_text(tx.built_year_circa)
    built_sources = _get_sources(tx.built_year_circa)
    await _build_sequential_field_check(
        evaluator=evaluator,
        parent=critical_root,
        group_id="built_year_circa_group",
        group_desc="Original build year (circa) field checks",
        existence_leaf_id="built_year_circa_exists",
        existence_desc="Original build year (circa) is provided in the answer",
        value_text=built_value,
        verify_leaf_id="built_year_circa",
        verify_desc="States the property's original build year as circa 1848 (per constraints)",
        claim=f"The property was originally built circa {built_value}.",
        sources=built_sources,
        additional_instruction=(
            "Verify that the page(s) indicate the original construction around 1848. "
            "Treat 'circa' as approximate; acceptable if the page shows c.1848 or an immediately adjacent year, but the claim must match what the page states."
        ),
    )

    # 4) Architectural style
    style_value = _get_text(tx.architectural_style)
    style_sources = _get_sources(tx.architectural_style)
    await _build_sequential_field_check(
        evaluator=evaluator,
        parent=critical_root,
        group_id="architectural_style_group",
        group_desc="Architectural style field checks",
        existence_leaf_id="architectural_style_exists",
        existence_desc="Architectural style is provided in the answer",
        value_text=style_value,
        verify_leaf_id="architectural_style",
        verify_desc="States the property's architectural style as Italianate (per constraints)",
        claim=f"The property's architectural style is {style_value}.",
        sources=style_sources,
        additional_instruction=(
            "Verify that the page(s) state the property is Italianate. "
            "Allow reasonable synonyms (e.g., 'Italianate-style') but the claim must match what the source says."
        ),
    )

    # 5) Purchase year
    purchase_year_value = _get_text(tx.purchase_year)
    purchase_year_sources = _get_sources(tx.purchase_year)
    await _build_sequential_field_check(
        evaluator=evaluator,
        parent=critical_root,
        group_id="purchase_year_group",
        group_desc="Purchase year field checks",
        existence_leaf_id="purchase_year_exists",
        existence_desc="Purchase year is provided in the answer",
        value_text=purchase_year_value,
        verify_leaf_id="purchase_year",
        verify_desc="States that the person purchased the property in 2016 (2010s) (per constraints)",
        claim=f"The property was purchased in {purchase_year_value} by {person_value}.",
        sources=purchase_year_sources,
        additional_instruction=(
            "Verify that the page(s) indicate the purchase year as 2016 (within the 2010s), and that the buyer is the same person identified. "
            "Minor date formatting variations are acceptable."
        ),
    )

    # 6) Purchase price in millions
    purchase_price_value = _get_text(tx.purchase_price_millions)
    purchase_price_sources = _get_sources(tx.purchase_price_millions)
    await _build_sequential_field_check(
        evaluator=evaluator,
        parent=critical_root,
        group_id="purchase_price_millions_group",
        group_desc="Purchase price (millions) field checks",
        existence_leaf_id="purchase_price_millions_exists",
        existence_desc="Purchase price in millions is provided in the answer",
        value_text=purchase_price_value,
        verify_leaf_id="purchase_price_millions",
        verify_desc="States the purchase price as $6.5 million (per constraints)",
        claim=f"The purchase price was {purchase_price_value} million dollars.",
        sources=purchase_price_sources,
        additional_instruction=(
            "Verify that the page(s) indicate the purchase price equals $6.5 million. "
            "Consider numeric equivalence: 6,500,000 equals 6.5 million; $6.5M equals $6.5 million. "
            "If the answer includes currency formatting or symbols, focus on the underlying amount."
        ),
    )

    # 7) Sale closing date in 2025
    sale_close_value = _get_text(tx.sale_close_date_2025)
    sale_close_sources = _get_sources(tx.sale_close_date_2025)
    await _build_sequential_field_check(
        evaluator=evaluator,
        parent=critical_root,
        group_id="sale_close_date_2025_group",
        group_desc="Sale closing date (2025) field checks",
        existence_leaf_id="sale_close_date_2025_exists",
        existence_desc="Sale closing date in 2025 is provided in the answer",
        value_text=sale_close_value,
        verify_leaf_id="sale_close_date_2025",
        verify_desc="States the sale closing date as February 28, 2025 (per constraints)",
        claim=f"The sale closed on {sale_close_value}.",
        sources=sale_close_sources,
        additional_instruction=(
            "Verify that the page(s) indicate the sale closing date was February 28, 2025 (format variants such as 'Feb. 28, 2025' or '2025-02-28' are acceptable). "
            "The claim should match the source date."
        ),
    )

    # 8) Sale price (real estate portion only) in millions
    sale_price_value = _get_text(tx.sale_price_real_estate_only_millions)
    sale_price_sources = _get_sources(tx.sale_price_real_estate_only_millions)
    await _build_sequential_field_check(
        evaluator=evaluator,
        parent=critical_root,
        group_id="sale_price_real_estate_only_millions_group",
        group_desc="Sale price (real estate-only, millions) field checks",
        existence_leaf_id="sale_price_real_estate_only_millions_exists",
        existence_desc="Sale price for the real estate portion (excluding contents) in millions is provided in the answer",
        value_text=sale_price_value,
        verify_leaf_id="sale_price_real_estate_only_millions",
        verify_desc="States the sale price for the real estate portion (excluding contents) as $18.25 million (per constraints)",
        claim=f"The sale price for the real estate portion (excluding contents) was {sale_price_value} million dollars.",
        sources=sale_price_sources,
        additional_instruction=(
            "Verify that the page(s) indicate the real estate-only sale price equals $18.25 million. "
            "If sources discuss a separate price for contents/furnishings, ignore those and focus on the real estate portion."
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
    Evaluate an answer for the Charleston 2025 record sale task.
    Builds a critical verification tree ensuring each of the eight fields is present and supported by sources.
    """
    # Initialize evaluator (root is non-critical by framework; we add a critical child grouping node below)
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

    # Extract all fields
    tx = await evaluator.extract(
        prompt=prompt_extract_transaction(),
        template_class=TransactionExtraction,
        extraction_name="transaction_extraction",
    )

    # Record ground-truth constraints/context
    evaluator.add_ground_truth(
        gt_info=GROUND_TRUTH_CONSTRAINTS,
        gt_type="constraints_context"
    )

    # Add a critical parallel node to emulate a critical root (framework's root is always non-critical).
    critical_root = evaluator.add_parallel(
        id="critical_root",
        desc="Answer correctly provides all 8 requested fields for the described 2025 Charleston historic property transaction, satisfying the given constraints",
        parent=root,
        critical=True
    )

    # Build verification nodes for all requested fields
    await verify_transaction_fields(evaluator, critical_root, tx)

    # Optional: Add a compact snapshot of extracted values to the summary
    evaluator.add_custom_info(
        info={
            "person_full_name": _get_text(tx.person_full_name),
            "property_street_address": _get_text(tx.property_street_address),
            "built_year_circa": _get_text(tx.built_year_circa),
            "architectural_style": _get_text(tx.architectural_style),
            "purchase_year": _get_text(tx.purchase_year),
            "purchase_price_millions": _get_text(tx.purchase_price_millions),
            "sale_close_date_2025": _get_text(tx.sale_close_date_2025),
            "sale_price_real_estate_only_millions": _get_text(tx.sale_price_real_estate_only_millions),
        },
        info_type="extracted_values_snapshot",
        info_name="extracted_values"
    )

    # Return full evaluation summary
    return evaluator.get_summary()