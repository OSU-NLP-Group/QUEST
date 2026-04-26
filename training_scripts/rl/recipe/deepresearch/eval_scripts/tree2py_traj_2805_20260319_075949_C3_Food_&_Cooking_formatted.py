import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "frozen_blueberry_recall_2026"
TASK_DESCRIPTION = (
    "A food service procurement manager in Wisconsin needs to investigate the frozen blueberry recall announced in February 2026. "
    "Provide a complete report including: (1) the company name that issued the recall, (2) the brand name on recalled products, "
    "(3) confirmation that Wisconsin is an affected state, (4) the FDA recall classification level, and (5) all specific lot codes "
    "with expiration dates for both 30-pound cases and totes."
)

# Expected ground truth (as per rubric)
EXPECTED_COMPANY = "Oregon Potato Company LLC"
EXPECTED_BRAND = "Willamette Valley Fruit Company"
EXPECTED_AFFECTED_STATES = ["Michigan", "Oregon", "Washington", "Wisconsin"]  # exact set, order-agnostic
EXPECTED_FDA_CLASSIFICATION = "Class I"
EXPECTED_RECALL_INITIATION_DATE = "February 12, 2026"
EXPECTED_CLASSIFICATION_DATE = "February 24, 2026"
EXPECTED_QUANTITY_TEXT = "55,689 pounds of frozen blueberries"  # for claims; extractor may store as "55,689 pounds"
EXPECTED_PACKAGING_PHRASE = "30-pound corrugated cases with polyethylene liners and 1,400-pound totes"
EXPECTED_RETAIL_STATEMENT = "Products were not sold directly to consumers in retail stores"
EXPECTED_CAUSE = "Listeria monocytogenes"
EXPECTED_30LB_CASE_LOTS = [
    ("2055 B2", "7/23/2027"),
    ("2065 B1", "7/24/2027"),
    ("2065 B3", "7/24/2027"),
]
EXPECTED_TOTE_LOTS = [
    ("3305 A1", "11/25/2027"),
    ("3305 B1", "11/25/2027"),
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LotItem(BaseModel):
    lot_code: Optional[str] = None
    expiration_date: Optional[str] = None


class RecallReportExtraction(BaseModel):
    company_name: Optional[str] = None
    brand_name: Optional[str] = None
    affected_states: List[str] = Field(default_factory=list)
    fda_classification: Optional[str] = None
    recall_initiated_date: Optional[str] = None
    fda_classification_date: Optional[str] = None

    recalled_quantity: Optional[str] = None
    packaging_description: Optional[str] = None
    retail_channel_statement: Optional[str] = None
    recall_cause: Optional[str] = None

    case_30lb_lots: List[LotItem] = Field(default_factory=list)
    tote_1400lb_lots: List[LotItem] = Field(default_factory=list)

    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_report_fields() -> str:
    return """
    Extract the following fields from the answer exactly as stated (do not invent anything):
    1) company_name: the company that issued the recall
    2) brand_name: the brand appearing on the recalled products
    3) affected_states: an array of U.S. state names explicitly listed as affected in the answer (title case each if needed)
    4) fda_classification: FDA recall classification (e.g., "Class I", "Class II", etc.)
    5) recall_initiated_date: the recall initiation date as written in the answer (e.g., "February 12, 2026" or "Feb 12, 2026")
    6) fda_classification_date: the FDA classification date as written (e.g., "February 24, 2026")
    7) recalled_quantity: the amount recalled as written (e.g., "55,689 pounds")
    8) packaging_description: the packaging formats phrase or sentence as written (e.g., "30-pound corrugated cases with polyethylene liners and 1,400-pound totes")
    9) retail_channel_statement: the statement about retail sales (e.g., "not sold directly to consumers in retail stores")
    10) recall_cause: the cause/reason (e.g., "Listeria monocytogenes" contamination)
    11) case_30lb_lots: array of objects for 30-pound cases; each object must have:
        - lot_code (e.g., "2055 B2")
        - expiration_date (e.g., "7/23/2027")
    12) tote_1400lb_lots: array of objects for 1,400-pound totes; each object must have:
        - lot_code (e.g., "3305 A1")
        - expiration_date (e.g., "11/25/2027")
    13) source_urls: array of all URLs explicitly cited in the answer that support the above information (only actual URLs; include full http/https).

    Rules:
    - If a field is missing in the answer, return null for strings and empty arrays for lists.
    - Do not infer states or codes not explicitly listed in the answer text.
    - Keep strings exactly as presented (including punctuation and casing), except normalize state names to Title Case in affected_states.
    """


# --------------------------------------------------------------------------- #
# Helper formatting functions                                                 #
# --------------------------------------------------------------------------- #
def join_states(states: List[str]) -> str:
    return ", ".join(states)


def lot_items_to_text(items: List[LotItem]) -> str:
    parts = []
    for it in items:
        if it and it.lot_code and it.expiration_date:
            parts.append(f"{it.lot_code} (expires {it.expiration_date})")
    return "; ".join(parts)


def expected_lot_items_text(expected_lots: List[tuple]) -> str:
    return "; ".join([f"{code} (expires {date})" for code, date in expected_lots])


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_company_name(evaluator: Evaluator, parent, data: RecallReportExtraction):
    node = evaluator.add_sequential(
        id="company_name_main",
        desc="Company name that issued the recall is Oregon Potato Company LLC",
        parent=parent,
        critical=True,
    )

    # Existence + sources gate
    evaluator.add_custom_node(
        result=bool(data.company_name) and bool(data.source_urls),
        id="company_name_provided",
        desc="Company name provided in answer and at least one source URL is cited",
        parent=node,
        critical=True,
    )

    # Match (answer -> expected)
    match_leaf = evaluator.add_leaf(
        id="company_name_match",
        desc="Answer names the recall issuer equivalent to 'Oregon Potato Company LLC'",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The company named in the answer ('{data.company_name}') and 'Oregon Potato Company LLC' refer to the same entity.",
        node=match_leaf,
        additional_instruction="Allow minor formatting or suffix variations (e.g., with/without LLC punctuation); judge equivalence, not strict character match."
    )

    # Source support
    src_leaf = evaluator.add_leaf(
        id="company_name_source_support",
        desc="Sources support that the recall was issued by Oregon Potato Company LLC",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The recall was issued by Oregon Potato Company LLC.",
        node=src_leaf,
        sources=data.source_urls,
        additional_instruction="Confirm the issuing company on the provided webpage(s). Minor naming variants are acceptable if clearly the same company."
    )


async def verify_brand_name(evaluator: Evaluator, parent, data: RecallReportExtraction):
    node = evaluator.add_sequential(
        id="brand_name_main",
        desc="Brand name on recalled products is Willamette Valley Fruit Company",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.brand_name) and bool(data.source_urls),
        id="brand_name_provided",
        desc="Brand name provided in answer and at least one source URL is cited",
        parent=node,
        critical=True,
    )

    match_leaf = evaluator.add_leaf(
        id="brand_name_match",
        desc="Answer names the brand equivalent to 'Willamette Valley Fruit Company'",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The brand named in the answer ('{data.brand_name}') and 'Willamette Valley Fruit Company' refer to the same brand.",
        node=match_leaf,
        additional_instruction="Allow minor formatting or abbreviation variants (e.g., 'Co.'); judge equivalence."
    )

    src_leaf = evaluator.add_leaf(
        id="brand_name_source_support",
        desc="Sources support that the recalled products are branded as Willamette Valley Fruit Company",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The recalled products are branded as Willamette Valley Fruit Company.",
        node=src_leaf,
        sources=data.source_urls,
        additional_instruction="Verify that the brand name on the recalled product(s) is Willamette Valley Fruit Company."
    )


async def verify_affected_states(evaluator: Evaluator, parent, data: RecallReportExtraction):
    node = evaluator.add_sequential(
        id="affected_states_main",
        desc="Wisconsin is an affected state and affected states are exactly: Michigan, Oregon, Washington, Wisconsin",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.affected_states) and bool(data.source_urls),
        id="states_list_provided",
        desc="Affected states list provided in answer and at least one source URL is cited",
        parent=node,
        critical=True,
    )

    # Answer exactness check (as a set; order-agnostic; ensure Wisconsin present)
    answer_match_leaf = evaluator.add_leaf(
        id="states_exact_match_answer",
        desc="Answer lists exactly the affected states (MI, OR, WA, WI), including Wisconsin",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"Based on the answer, the affected states are: {join_states(data.affected_states)}. "
            f"This set is exactly equal to: {join_states(EXPECTED_AFFECTED_STATES)} (order does not matter), and Wisconsin is included."
        ),
        node=answer_match_leaf,
        additional_instruction="Treat equality as set-equality ignoring order/case; confirm that no extra states are included and none missing; ensure Wisconsin is included."
    )

    # Source support
    src_leaf = evaluator.add_leaf(
        id="states_source_support",
        desc="Sources support that the affected states are exactly MI, OR, WA, WI and include Wisconsin",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The recall affected the U.S. states: Michigan, Oregon, Washington, and Wisconsin (and Wisconsin is affected). No other states are listed.",
        node=src_leaf,
        sources=data.source_urls,
        additional_instruction="Confirm the exact set of states on the webpage(s); accept order differences; reject if extra/missing states."
    )


async def verify_fda_class_and_timeline(evaluator: Evaluator, parent, data: RecallReportExtraction):
    node = evaluator.add_sequential(
        id="fda_class_timeline_main",
        desc="Recall timeline and FDA classification are correct: initiated Feb 12, 2026; Class I; classification date Feb 24, 2026",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.fda_classification) and bool(data.recall_initiated_date) and bool(data.fda_classification_date) and bool(data.source_urls),
        id="fda_timeline_fields_provided",
        desc="FDA class, initiation date, classification date provided and at least one source URL is cited",
        parent=node,
        critical=True,
    )

    match_leaf = evaluator.add_leaf(
        id="fda_timeline_answer_match",
        desc="Answer reports Class I and dates (Feb 12, 2026 initiation; Feb 24, 2026 classification)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The answer reports FDA classification '{data.fda_classification}', initiation date '{data.recall_initiated_date}', "
            f"and classification date '{data.fda_classification_date}', which together equal: "
            f"classification '{EXPECTED_FDA_CLASSIFICATION}', initiation '{EXPECTED_RECALL_INITIATION_DATE}', classification date '{EXPECTED_CLASSIFICATION_DATE}'."
        ),
        node=match_leaf,
        additional_instruction="Judge semantic equivalence; allow common date formatting variants (e.g., Feb vs February; zero-padded vs non-padded days); 'Class I' equivalence acceptable."
    )

    src_leaf = evaluator.add_leaf(
        id="fda_timeline_source_support",
        desc="Sources support Class I and the specific initiation and classification dates",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The recall was initiated on {EXPECTED_RECALL_INITIATION_DATE}, and FDA classified it as {EXPECTED_FDA_CLASSIFICATION} on {EXPECTED_CLASSIFICATION_DATE}."
        ),
        node=src_leaf,
        sources=data.source_urls,
        additional_instruction="Verify these three details (initiation date, classification level, classification date) on the provided webpage(s)."
    )


async def verify_recalled_quantity(evaluator: Evaluator, parent, data: RecallReportExtraction):
    node = evaluator.add_sequential(
        id="recalled_quantity_main",
        desc="Report states the recall involves 55,689 pounds of frozen blueberries",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.recalled_quantity) and bool(data.source_urls),
        id="recalled_quantity_provided",
        desc="Recalled quantity provided in answer and at least one source URL is cited",
        parent=node,
        critical=True,
    )

    match_leaf = evaluator.add_leaf(
        id="recalled_quantity_answer_match",
        desc="Answer quantity matches '55,689 pounds' (of frozen blueberries)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The answer's recalled quantity '{data.recalled_quantity}' is equivalent to '55,689 pounds'.",
        node=match_leaf,
        additional_instruction="Allow minor formatting (commas/spaces) and presence/absence of 'of frozen blueberries' suffix."
    )

    src_leaf = evaluator.add_leaf(
        id="recalled_quantity_source_support",
        desc="Sources support that the recall involves 55,689 pounds of frozen blueberries",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The recall involves {EXPECTED_QUANTITY_TEXT}.",
        node=src_leaf,
        sources=data.source_urls,
        additional_instruction="Confirm the total recalled quantity on the provided webpage(s)."
    )


async def verify_packaging_formats(evaluator: Evaluator, parent, data: RecallReportExtraction):
    node = evaluator.add_sequential(
        id="packaging_formats_main",
        desc="Products were packaged in 30-pound corrugated cases with polyethylene liners and 1,400-pound totes",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.packaging_description) and bool(data.source_urls),
        id="packaging_desc_provided",
        desc="Packaging description provided in answer and at least one source URL is cited",
        parent=node,
        critical=True,
    )

    match_leaf = evaluator.add_leaf(
        id="packaging_answer_match",
        desc="Answer packaging includes both '30-pound corrugated cases with polyethylene liners' and '1,400-pound totes'",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The answer's packaging description ('{data.packaging_description}') includes BOTH of the following formats: "
            f"'30-pound corrugated cases with polyethylene liners' AND '1,400-pound totes'."
        ),
        node=match_leaf,
        additional_instruction="Judge semantic inclusion; allow mild wording/formatting variants (e.g., '1400 lb' vs '1,400-pound')."
    )

    src_leaf = evaluator.add_leaf(
        id="packaging_source_support",
        desc="Sources support the packaging formats (30-lb corrugated cases with polyethylene liners; 1,400-lb totes)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The recalled products were packaged in 30-pound corrugated cases with polyethylene liners and 1,400-pound totes.",
        node=src_leaf,
        sources=data.source_urls,
        additional_instruction="Verify both packaging formats appear on the provided webpage(s)."
    )


async def verify_retail_channel(evaluator: Evaluator, parent, data: RecallReportExtraction):
    node = evaluator.add_sequential(
        id="retail_channel_main",
        desc="Products were not sold directly to consumers in retail stores",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.retail_channel_statement) and bool(data.source_urls),
        id="retail_statement_provided",
        desc="Retail channel statement provided in answer and at least one source URL is cited",
        parent=node,
        critical=True,
    )

    match_leaf = evaluator.add_leaf(
        id="retail_answer_match",
        desc="Answer confirms products were not sold directly to consumers in retail stores",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The answer's retail statement ('{data.retail_channel_statement}') asserts that products were not sold directly to consumers in retail stores.",
        node=match_leaf,
        additional_instruction="Judge semantic equivalence to 'not sold at retail / not sold to consumers in retail stores'."
    )

    src_leaf = evaluator.add_leaf(
        id="retail_source_support",
        desc="Sources support that products were not sold directly to consumers in retail stores",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The recalled products were not sold directly to consumers in retail stores.",
        node=src_leaf,
        sources=data.source_urls,
        additional_instruction="Confirm the non-retail distribution statement on the provided webpage(s)."
    )


async def verify_recall_cause(evaluator: Evaluator, parent, data: RecallReportExtraction):
    node = evaluator.add_sequential(
        id="recall_cause_main",
        desc="Reason: potential Listeria monocytogenes contamination",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.recall_cause) and bool(data.source_urls),
        id="recall_cause_provided",
        desc="Recall cause provided in answer and at least one source URL is cited",
        parent=node,
        critical=True,
    )

    match_leaf = evaluator.add_leaf(
        id="recall_cause_answer_match",
        desc="Answer identifies potential 'Listeria monocytogenes' contamination as the reason",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The answer's cause ('{data.recall_cause}') is equivalent to potential 'Listeria monocytogenes' contamination.",
        node=match_leaf,
        additional_instruction="Judge semantic equivalence (e.g., 'due to Listeria monocytogenes')."
    )

    src_leaf = evaluator.add_leaf(
        id="recall_cause_source_support",
        desc="Sources support Listeria monocytogenes as the reason for the recall",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The reason for the recall is potential contamination with Listeria monocytogenes.",
        node=src_leaf,
        sources=data.source_urls,
        additional_instruction="Confirm on the provided webpage(s)."
    )


async def verify_30lb_case_lots(evaluator: Evaluator, parent, data: RecallReportExtraction):
    node = evaluator.add_sequential(
        id="lot_30lb_main",
        desc="All 30-pound case lot codes and expiration dates are provided",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(len(data.case_30lb_lots) >= 3) and bool(data.source_urls),
        id="lot_30lb_present",
        desc="Answer lists lot codes for 30-lb cases (at least 3 entries) and at least one source URL is cited",
        parent=node,
        critical=True,
    )

    # Answer exactness check
    answer_match = evaluator.add_leaf(
        id="lot_30lb_answer_exact",
        desc="Answer lists exactly the 30-lb case lots: 2055 B2 (7/23/2027), 2065 B1 (7/24/2027), 2065 B3 (7/24/2027)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"For 30-lb cases, the answer lists: {lot_items_to_text(data.case_30lb_lots)}. "
            f"This exactly matches the required set: {expected_lot_items_text(EXPECTED_30LB_CASE_LOTS)} (no extras or omissions)."
        ),
        node=answer_match,
        additional_instruction="Judge set-equality; allow minor date formatting differences (e.g., 7/23/2027 vs 07/23/2027) and spacing in lot codes."
    )

    # Source support
    src_leaf = evaluator.add_leaf(
        id="lot_30lb_source_support",
        desc="Sources support the 30-lb case lots and expiration dates",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The 30-lb case lots and expirations are: {expected_lot_items_text(EXPECTED_30LB_CASE_LOTS)}.",
        node=src_leaf,
        sources=data.source_urls,
        additional_instruction="Confirm lot codes and dates for 30-lb cases on the provided webpage(s)."
    )


async def verify_tote_lots(evaluator: Evaluator, parent, data: RecallReportExtraction):
    node = evaluator.add_sequential(
        id="lot_tote_main",
        desc="All 1,400-pound tote lot codes and expiration dates are provided",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(len(data.tote_1400lb_lots) >= 2) and bool(data.source_urls),
        id="lot_tote_present",
        desc="Answer lists lot codes for 1,400-lb totes (at least 2 entries) and at least one source URL is cited",
        parent=node,
        critical=True,
    )

    # Answer exactness check
    answer_match = evaluator.add_leaf(
        id="lot_tote_answer_exact",
        desc="Answer lists exactly the tote lots: 3305 A1 (11/25/2027), 3305 B1 (11/25/2027)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"For 1,400-lb totes, the answer lists: {lot_items_to_text(data.tote_1400lb_lots)}. "
            f"This exactly matches the required set: {expected_lot_items_text(EXPECTED_TOTE_LOTS)} (no extras or omissions)."
        ),
        node=answer_match,
        additional_instruction="Judge set-equality; allow minor date formatting differences and spacing in lot codes."
    )

    # Source support
    src_leaf = evaluator.add_leaf(
        id="lot_tote_source_support",
        desc="Sources support the 1,400-lb tote lots and expiration dates",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The 1,400-lb tote lots and expirations are: {expected_lot_items_text(EXPECTED_TOTE_LOTS)}.",
        node=src_leaf,
        sources=data.source_urls,
        additional_instruction="Confirm lot codes and dates for 1,400-lb totes on the provided webpage(s)."
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
    # Initialize evaluator (root is non-critical by design)
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_report_fields(),
        template_class=RecallReportExtraction,
        extraction_name="recall_report_extraction",
    )

    # Record ground truth for transparency
    evaluator.add_ground_truth({
        "expected_company": EXPECTED_COMPANY,
        "expected_brand": EXPECTED_BRAND,
        "expected_affected_states": EXPECTED_AFFECTED_STATES,
        "expected_fda_classification": EXPECTED_FDA_CLASSIFICATION,
        "expected_recall_initiation_date": EXPECTED_RECALL_INITIATION_DATE,
        "expected_classification_date": EXPECTED_CLASSIFICATION_DATE,
        "expected_recalled_quantity": EXPECTED_QUANTITY_TEXT,
        "expected_packaging_phrase": EXPECTED_PACKAGING_PHRASE,
        "expected_retail_statement": EXPECTED_RETAIL_STATEMENT,
        "expected_cause": EXPECTED_CAUSE,
        "expected_30lb_case_lots": expected_lot_items_text(EXPECTED_30LB_CASE_LOTS),
        "expected_tote_lots": expected_lot_items_text(EXPECTED_TOTE_LOTS),
    })

    # Build the critical main node mirroring the rubric root
    main = evaluator.add_parallel(
        id="frozen_blueberry_recall_investigation",
        desc="Complete investigation of the February 2026 frozen blueberry recall affecting Wisconsin, satisfying all stated constraints",
        parent=root,
        critical=True,
    )

    # Company name verification (critical)
    await verify_company_name(evaluator, main, extracted)

    # Brand name verification (critical)
    await verify_brand_name(evaluator, main, extracted)

    # Affected states verification (critical)
    await verify_affected_states(evaluator, main, extracted)

    # FDA classification and timeline verification (critical)
    await verify_fda_class_and_timeline(evaluator, main, extracted)

    # Recall scope and product details (critical, parallel group)
    scope = evaluator.add_parallel(
        id="recall_scope_and_product_details",
        desc="Report includes required recall scope/product details from constraints",
        parent=main,
        critical=True,
    )

    await verify_recalled_quantity(evaluator, scope, extracted)
    await verify_packaging_formats(evaluator, scope, extracted)
    await verify_retail_channel(evaluator, scope, extracted)
    await verify_recall_cause(evaluator, scope, extracted)

    # Lot codes and expiration dates (critical, parallel group)
    lot_group = evaluator.add_parallel(
        id="lot_codes_and_expiration_dates",
        desc="All specific lot codes with expiration dates are provided for both package formats",
        parent=main,
        critical=True,
    )

    await verify_30lb_case_lots(evaluator, lot_group, extracted)
    await verify_tote_lots(evaluator, lot_group, extracted)

    # Return standard summary
    return evaluator.get_summary()