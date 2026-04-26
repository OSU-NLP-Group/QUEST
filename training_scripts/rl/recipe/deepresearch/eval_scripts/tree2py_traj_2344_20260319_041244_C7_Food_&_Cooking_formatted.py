import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "food_recalls_q1_2026"
TASK_DESCRIPTION = (
    "Identify three distinct food product recalls that were announced or expanded between January 1, 2026 and March 19, 2026, "
    "where each recall meets all of the specified contamination, severity, geographic, product, retail, official documentation, "
    "and product identification criteria. For each, provide: the product name or category, the manufacturer or brand, the type of "
    "contamination, the key severity metric (death count, hospitalization count, or pound volume), and at least one product identifier."
)

DATE_WINDOW_START = "2026-01-01"
DATE_WINDOW_END = "2026-03-19"


# -----------------------------------------------------------------------------
# Data Models (extraction)
# -----------------------------------------------------------------------------
class IdentifierInfo(BaseModel):
    upc_codes: List[str] = Field(default_factory=list)
    establishment_numbers: List[str] = Field(default_factory=list)
    lot_codes: List[str] = Field(default_factory=list)
    date_codes: List[str] = Field(default_factory=list)  # best-by / use-by / sell-by / exp ranges or dates
    other_identifiers: List[str] = Field(default_factory=list)


class GeoInfo(BaseModel):
    us_states: List[str] = Field(default_factory=list)
    canadian_provinces: List[str] = Field(default_factory=list)
    distribution_text: Optional[str] = None  # optional free-text description extracted from the answer


class RetailInfo(BaseModel):
    retailer_names: List[str] = Field(default_factory=list)
    retailer_urls: List[str] = Field(default_factory=list)


class SeverityInfo(BaseModel):
    severity_metric_type: Optional[str] = None  # one of: "deaths", "hospitalizations", "pounds", or free text if unknown
    severity_value: Optional[str] = None        # numeric as string or phrase exactly as in the answer


class RecallItem(BaseModel):
    product_name_or_category: Optional[str] = None
    brand_or_manufacturer: Optional[str] = None

    contamination_category: Optional[str] = None  # "pathogen" or "foreign_material" if provided
    specific_contaminant: Optional[str] = None    # e.g., "Listeria monocytogenes", "Salmonella", "glass fragments"

    announcement_or_expansion_date: Optional[str] = None  # string as given in the answer
    product_type: Optional[str] = None  # e.g., "ready-to-eat", "frozen prepared", etc.

    identifiers: IdentifierInfo = Field(default_factory=IdentifierInfo)
    distribution: GeoInfo = Field(default_factory=GeoInfo)
    retail: RetailInfo = Field(default_factory=RetailInfo)
    severity: SeverityInfo = Field(default_factory=SeverityInfo)

    official_urls: List[str] = Field(default_factory=list)   # government recall/outbreak pages (FDA/FSIS/CFIA/PHAC)
    supporting_urls: List[str] = Field(default_factory=list) # retailer notices, CDC/state health dept, press releases, etc.


class RecallsExtraction(BaseModel):
    recalls: List[RecallItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_recalls() -> str:
    return """
Extract up to three distinct food recalls described in the answer that the author claims meet the specified criteria. For each selected recall, return a JSON object with the following fields (use the exact field names and structure):

- product_name_or_category: string or null
- brand_or_manufacturer: string or null

- contamination_category: prefer "pathogen" or "foreign_material" if explicitly clear in the answer; otherwise copy the answer's phrasing or set null
- specific_contaminant: for pathogen, e.g., "Listeria monocytogenes" or "Salmonella"; for foreign material, e.g., "glass fragments"; or null if unspecified

- announcement_or_expansion_date: the date from the answer explicitly linked to the recall announcement or its official expansion/update (prefer ISO-like 'YYYY-MM-DD' if present; otherwise copy as-is), or null
- product_type: concise phrase describing the product type (e.g., "ready-to-eat salad", "frozen prepared meal"); or null

- identifiers: object with arrays (empty if not provided)
  - upc_codes: []
  - establishment_numbers: []
  - lot_codes: []
  - date_codes: []       # best-by / use-by / exp ranges/dates
  - other_identifiers: []

- distribution: object
  - us_states: []            # list of U.S. states mentioned, if any
  - canadian_provinces: []   # list of Canadian provinces/territories mentioned, if any
  - distribution_text: string or null  # any free-text distribution info mentioned (e.g., "nationwide", "multiple states")

- retail: object
  - retailer_names: []   # retailer brand names (e.g., Walmart, Kroger, Trader Joe's, Costco, Target, Sam's Club, Safeway, Publix, Whole Foods, Aldi)
  - retailer_urls: []    # any retailer announcement pages mentioned

- severity: object
  - severity_metric_type: "deaths" | "hospitalizations" | "pounds" | null (choose what the answer emphasizes for this recall's severity criterion)
  - severity_value: string or null (copy exactly from the answer; examples: "1 death", "23 hospitalizations", "35,000,000 pounds")

- official_urls: []  # ONLY include direct links explicitly mentioned in the answer to official government pages for this recall:
                     #   FDA:           https://www.fda.gov/...
                     #   USDA FSIS:     https://www.fsis.usda.gov/...
                     #   CFIA:          https://inspection.canada.ca/...   or https://www.inspection.gc.ca/...
                     #   PHAC:          https://www.canada.ca/en/public-health/...
                     # Do NOT include non-government domains here.

- supporting_urls: []  # other URLs explicitly mentioned in the answer, such as CDC/state health authorities, retailer notices, or press releases

GENERAL RULES:
1) Extract only what is explicitly present in the answer text. If a field is not present, set it to null or an empty array as specified.
2) Do not invent URLs. For official_urls, ensure each URL is actually present in the answer and is a government page per the above list.
3) Return exactly one top-level JSON object with a single key "recalls" that is an array of up to three recall objects.
4) If the answer lists more than three recalls, include the first three that appear to meet the requested criteria based on the answer's own text.
"""


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _merged_sources(item: RecallItem) -> List[str]:
    return _unique_urls((item.official_urls or []) + (item.supporting_urls or []) + (item.retail.retailer_urls or []))


def _any_identifier_present(item: RecallItem) -> bool:
    ids = item.identifiers
    return any([
        bool(ids.upc_codes),
        bool(ids.establishment_numbers),
        bool(ids.lot_codes),
        bool(ids.date_codes),
        bool(ids.other_identifiers),
    ])


def _recall_context(item: RecallItem, idx: int) -> str:
    pn = item.product_name_or_category or "the product"
    br = item.brand_or_manufacturer or "the brand/manufacturer"
    return f"Recall #{idx}: product={pn}; brand/manufacturer={br}"


# -----------------------------------------------------------------------------
# Verification for a single recall
# -----------------------------------------------------------------------------
async def verify_single_recall(evaluator: Evaluator, parent_node, item: RecallItem, idx: int) -> None:
    rn = idx  # 1-based display in descriptions, but using 1..3 mapping for IDs
    recall_node = evaluator.add_parallel(
        id=f"recall_{rn}",
        desc=f"{['First','Second','Third'][rn-1]} qualifying food recall meeting all specified criteria",
        parent=parent_node,
        critical=False,
    )

    # Precondition existence checks (so missing essentials deterministically fail)
    # 1) At least one official government URL must be provided
    evaluator.add_custom_node(
        result=bool(item.official_urls),
        id=f"recall_{rn}_official_urls_present",
        desc=f"Recall #{rn}: At least one official government recall page URL is provided in the answer",
        parent=recall_node,
        critical=True,
    )

    # 2) At least one product identifier present
    evaluator.add_custom_node(
        result=_any_identifier_present(item),
        id=f"recall_{rn}_identifiers_exist",
        desc=f"Recall #{rn}: The answer lists at least one product identifier (UPC, est. number, lot, or date range)",
        parent=recall_node,
        critical=True,
    )

    # Build leaves according to rubric JSON (all critical per recall)
    # 1) Timeframe
    timeframe_node = evaluator.add_leaf(
        id=f"recall_{rn}_timeframe",
        desc=f"The recall was announced or expanded between January 1, 2026 and March 19, 2026",
        parent=recall_node,
        critical=True,
    )
    tf_claim = (
        f"{_recall_context(item, rn)}. The official announcement or an official update for this recall occurred "
        f"between {DATE_WINDOW_START} and {DATE_WINDOW_END} (inclusive)."
    )
    await evaluator.verify(
        claim=tf_claim,
        node=timeframe_node,
        sources=item.official_urls,
        additional_instruction=(
            "Check the date shown on the provided official government recall page(s). "
            "If the recall page indicates an announcement or an expansion/update date within 2026-01-01 to 2026-03-19 inclusive, pass. "
            "If the page clearly shows only dates outside this range, fail."
        ),
    )

    # 2) Contamination type: Listeria monocytogenes or Salmonella OR foreign material = glass fragments
    cont_node = evaluator.add_leaf(
        id=f"recall_{rn}_contamination_type",
        desc="The recall involves either a foodborne pathogen (Listeria monocytogenes or Salmonella) OR foreign material contamination (glass fragments)",
        parent=recall_node,
        critical=True,
    )
    cont_claim = (
        f"{_recall_context(item, rn)}. This recall involves either Listeria monocytogenes, Salmonella, or glass fragments."
    )
    await evaluator.verify(
        claim=cont_claim,
        node=cont_node,
        sources=item.official_urls,
        additional_instruction=(
            "Examine the government recall page(s). Pass if it explicitly mentions either Listeria monocytogenes, "
            "Salmonella (any serotype), or glass fragments as the contaminant. Otherwise, fail."
        ),
    )

    # 3) Severity threshold
    sev_node = evaluator.add_leaf(
        id=f"recall_{rn}_severity",
        desc="For pathogen recalls: resulted in at least one confirmed death OR at least 20 hospitalizations; For foreign material recalls: scope exceeds 30 million pounds",
        parent=recall_node,
        critical=True,
    )
    sev_claim = (
        f"{_recall_context(item, rn)}. The severity threshold is met: "
        f"either (pathogen-related) at least one confirmed death OR at least 20 hospitalizations, "
        f"or (foreign material) the recall scope exceeds 30 million pounds."
    )
    await evaluator.verify(
        claim=sev_claim,
        node=sev_node,
        sources=_merged_sources(item),
        additional_instruction=(
            "Use the provided sources (government pages preferred; supporting sources such as CDC, state health departments, "
            "or retailer notices may be used if they directly state the metric). "
            "For pathogen recalls: pass if the evidence clearly shows ≥1 death OR ≥20 hospitalizations linked to this recall/outbreak. "
            "For foreign material recalls: pass if the evidence indicates the affected product volume exceeds 30,000,000 pounds."
        ),
    )

    # 4) Geographic distribution scope
    geo_node = evaluator.add_leaf(
        id=f"recall_{rn}_geographic_scope",
        desc="The product was distributed to multiple U.S. states (at least 6) OR multiple Canadian provinces (at least 3)",
        parent=recall_node,
        critical=True,
    )
    geo_claim = (
        f"{_recall_context(item, rn)}. The distribution covers at least 6 U.S. states or at least 3 Canadian provinces/territories."
    )
    await evaluator.verify(
        claim=geo_claim,
        node=geo_node,
        sources=_merged_sources(item),
        additional_instruction=(
            "Look for an explicit list of states/provinces or a statement like 'nationwide' in the U.S. "
            "If 'nationwide' is clearly indicated for the U.S., treat it as ≥6 states. "
            "Otherwise, count the listed states/provinces; pass if U.S. states count ≥6 or Canadian provinces/territories count ≥3."
        ),
    )

    # 5) Product type: ready-to-eat or frozen prepared foods (not raw agricultural commodities alone)
    ptype_node = evaluator.add_leaf(
        id=f"recall_{rn}_product_type",
        desc="Involves ready-to-eat or frozen prepared food products (not raw agricultural commodities alone)",
        parent=recall_node,
        critical=True,
    )
    ptype_claim = (
        f"{_recall_context(item, rn)}. The recalled product(s) are ready-to-eat or frozen prepared foods (not raw agricultural commodities alone)."
    )
    await evaluator.verify(
        claim=ptype_claim,
        node=ptype_node,
        sources=item.official_urls,
        additional_instruction=(
            "Check product descriptions, category, and preparation state (e.g., 'ready-to-eat', 'heat-and-eat', 'frozen meal'). "
            "If clearly prepared/ready foods, pass; if only raw agricultural commodities, fail."
        ),
    )

    # 6) Retail distribution at major national chains
    retail_node = evaluator.add_leaf(
        id=f"recall_{rn}_retail_distribution",
        desc="The product was sold at major national retail chains such as Walmart, Kroger, Trader Joe's, or equivalent major retailers",
        parent=recall_node,
        critical=True,
    )
    retail_claim = (
        f"{_recall_context(item, rn)}. The recalled products were sold at major national retail chains (e.g., Walmart, Kroger, Trader Joe's, "
        f"Costco, Target, Sam's Club, Safeway/Albertsons, Publix, Whole Foods, Aldi, or similar)."
    )
    await evaluator.verify(
        claim=retail_claim,
        node=retail_node,
        sources=_merged_sources(item),
        additional_instruction=(
            "Look for explicit retailer mentions on government recall pages or retailer notice pages. "
            "Pass if at least one clearly major U.S./Canada national retailer is listed."
        ),
    )

    # 7) Official documentation (government page presence)
    official_node = evaluator.add_leaf(
        id=f"recall_{rn}_official_documentation",
        desc="The recall is documented on FDA, USDA FSIS, CFIA, or Public Health Agency of Canada official pages",
        parent=recall_node,
        critical=True,
    )
    official_claim = (
        f"{_recall_context(item, rn)}. There is at least one official government recall/outbreak page for this recall "
        f"(FDA, USDA FSIS, CFIA, or PHAC)."
    )
    await evaluator.verify(
        claim=official_claim,
        node=official_node,
        sources=item.official_urls,
        additional_instruction=(
            "Verify that at least one provided URL is an official page: "
            "fda.gov, fsis.usda.gov, inspection.canada.ca (CFIA), or canada.ca/en/public-health (PHAC). "
            "Also ensure the page actually corresponds to this specific recall (matching product/brand context)."
        ),
    )

    # 8) Product identifiers present on announcement
    id_node = evaluator.add_leaf(
        id=f"recall_{rn}_product_identifiers",
        desc="The recall includes specific product identifiers such as UPC codes, establishment numbers, lot numbers, or best-by/expiration date ranges",
        parent=recall_node,
        critical=True,
    )
    id_claim = (
        f"{_recall_context(item, rn)}. The government recall announcement includes at least one specific product identifier "
        f"(UPC code, USDA establishment number, lot/batch code, or best-by/expiration date range)."
    )
    await evaluator.verify(
        claim=id_claim,
        node=id_node,
        sources=item.official_urls,
        additional_instruction=(
            "Examine the government recall page(s) for explicit identifiers such as UPC(s), establishment numbers (EST/EST#), "
            "lot/batch codes, or best-by/use-by/expiration date lists or ranges. "
            "Pass if at least one such identifier is present."
        ),
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    # Initialize evaluator (root kept non-critical to allow partial credit across recalls)
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

    # Extract recalls
    extracted: RecallsExtraction = await evaluator.extract(
        prompt=prompt_extract_recalls(),
        template_class=RecallsExtraction,
        extraction_name="recalls_extraction",
    )

    # Keep only first 3; pad with placeholders if fewer
    recalls: List[RecallItem] = list(extracted.recalls[:3])
    while len(recalls) < 3:
        recalls.append(RecallItem())  # placeholder (will likely fail critical checks)

    # Optional: record task window and rubric highlights as ground truth context
    evaluator.add_ground_truth({
        "date_window": {"start": DATE_WINDOW_START, "end": DATE_WINDOW_END},
        "allowed_contaminants": ["Listeria monocytogenes", "Salmonella (any serotype)", "glass fragments"],
        "severity_thresholds": {
            "pathogen": ">=1 death OR >=20 hospitalizations",
            "foreign_material": ">30,000,000 pounds",
        },
        "geographic_thresholds": {
            "us_states": ">=6",
            "canadian_provinces": ">=3",
        },
        "product_type_required": "ready-to-eat or frozen prepared foods",
        "official_docs_required": ["FDA", "USDA FSIS", "CFIA", "PHAC"],
        "retail_major_chains_examples": [
            "Walmart", "Kroger", "Trader Joe's", "Costco", "Target", "Sam's Club",
            "Safeway/Albertsons", "Publix", "Whole Foods", "Aldi"
        ],
        "product_identifiers_required": ["UPC", "USDA EST #", "Lot/Batch", "Best-by/Expiration range"],
    })

    # Build and verify each recall subtree
    for i in range(1, 4):
        await verify_single_recall(evaluator, root, recalls[i - 1], i)

    return evaluator.get_summary()