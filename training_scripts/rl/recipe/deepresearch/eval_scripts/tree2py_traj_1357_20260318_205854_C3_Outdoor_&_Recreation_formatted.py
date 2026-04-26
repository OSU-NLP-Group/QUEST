import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_parks_pass_2026_costs"
TASK_DESCRIPTION = """
A US resident aged 67 is planning to visit Yellowstone National Park, Grand Teton National Park, Glacier National Park, Yosemite National Park, and Rocky Mountain National Park during summer 2026, arriving at each park by private vehicle. They will spend one week traveling and visiting all five parks.

Provide:
1. The name and price of the least expensive 2026 America the Beautiful pass for which this person is eligible
2. The total that would be spent on vehicle entrance fees if visiting all 5 parks without any pass (assume standard 7-day vehicle pass at each park)
3. Whether purchasing the eligible pass would result in cost savings compared to paying individual entrance fees
4. One category of fees that the pass would NOT cover during park visits
5. The official website URL where this pass can be purchased digitally for immediate use
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkFee(BaseModel):
    park: Optional[str] = None  # The park name as written in the answer ("Yellowstone National Park", etc.)
    vehicle_7day_fee: Optional[str] = None  # e.g., "$35", "35 USD", etc.
    urls: List[str] = Field(default_factory=list)  # URLs explicitly cited for this park's fee


class AnswerExtraction(BaseModel):
    # 1) Eligible pass & price
    pass_name: Optional[str] = None
    pass_price: Optional[str] = None
    pass_urls: List[str] = Field(default_factory=list)

    # 2) Individual fees and total
    parks: List[ParkFee] = Field(default_factory=list)  # Should include 5 specified parks (if present in answer)
    total_without_pass: Optional[str] = None  # e.g., "$175"
    total_urls: List[str] = Field(default_factory=list)  # Any URL(s) the answer cites for the total

    # 3) Cost-benefit determination (answer's stated conclusion)
    savings_determination: Optional[str] = None  # "yes"/"no" or equivalent wording

    # 4) Coverage limitation
    coverage_limitation: Optional[str] = None  # e.g., "camping", "special tours", "special permits", "parking", "ferries"
    coverage_urls: List[str] = Field(default_factory=list)

    # 5) Purchase URL
    purchase_url: Optional[str] = None  # URL to official purchase page (Recreation.gov or USGS Store)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_answer() -> str:
    return """
    Extract the following fields from the answer exactly as stated. Do not invent or infer anything not present in the answer text.

    1) pass_name: The name of the least expensive America the Beautiful pass the 67-year-old is eligible for (should be a Senior pass if present).
    2) pass_price: The stated price for that identified pass (e.g., "$20").
    3) pass_urls: Array of every URL the answer cites specifically for this pass’s details/price/eligibility (official sources like USGS Store or Recreation.gov if provided).

    4) parks: An array of objects, one per park mentioned. Only include the five specific parks if the answer mentions their fees:
       - park: The park name as written in the answer (e.g., "Yellowstone National Park").
       - vehicle_7day_fee: The stated standard 7-day private vehicle entrance fee for that park, as written (e.g., "$35").
       - urls: Array of all URLs cited for that park’s entrance fee in the answer.
       The five parks of interest are:
         • Yellowstone National Park
         • Grand Teton National Park
         • Glacier National Park
         • Yosemite National Park
         • Rocky Mountain National Park

    5) total_without_pass: The total vehicle entrance fees to visit all five parks without any pass, exactly as stated (e.g., "$175").
    6) total_urls: All URLs the answer cites for the total calculation (if any).

    7) savings_determination: The answer’s explicit determination whether buying the identified pass would save money compared to paying individual entrance fees (return "yes" or "no" if clearly stated, else return the phrase used or null).

    8) coverage_limitation: One category of fees the pass does NOT cover, as stated (e.g., "camping", "special tours", "special permits", "parking", or "ferries").
    9) coverage_urls: All URLs the answer cites to support the coverage limitation.

    10) purchase_url: A single official URL (if provided) where this specific pass can be purchased online (preferably digital for immediate use) — usually Recreation.gov or the USGS Store.

    IMPORTANT URL RULES:
    - Only include URLs explicitly present in the answer.
    - If a referenced source is mentioned without an actual URL, do not create one; leave the field empty or null as appropriate.

    Return a single JSON object with the schema provided.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
PARK_CANONICALS: List[Tuple[str, str]] = [
    ("yellowstone", "Yellowstone National Park"),
    ("grand_teton", "Grand Teton National Park"),
    ("glacier", "Glacier National Park"),
    ("yosemite", "Yosemite National Park"),
    ("rocky_mountain", "Rocky Mountain National Park"),
]


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _extract_amount(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    # Capture the first valid number with optional $ and commas/decimals
    m = re.search(r"(?<!\d)(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{1,2})?", s.replace("$", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except Exception:
        return None


def _is_official_purchase_domain(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    return (
        netloc.endswith("recreation.gov")
        or netloc.endswith("store.usgs.gov")
        or netloc == "usgs.gov"
        or netloc.endswith(".usgs.gov")
    )


def _map_yesno(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    val = _norm(s)
    if any(k in val for k in ["yes", "true", "saves", "save", "cheaper", "cost-saving", "cost saving"]):
        return True
    if any(k in val for k in ["no", "false", "not save", "no savings", "more expensive", "cost increase"]):
        return False
    return None


def _find_park_item(parks: List[ParkFee], target_key: str) -> Optional[ParkFee]:
    target_full = dict(PARK_CANONICALS).get(target_key, "")
    for p in parks:
        name = _norm(p.park)
        if not name:
            continue
        if target_key.replace("_", " ") in name:
            return p
        if _norm(target_full) in name:
            return p
        # Loose contains for cases like "Yosemite NP"
        key_frag = target_full.split(" ")[0].lower()
        if key_frag and key_frag in name:
            # Make sure it's reasonably matching the correct park
            if any(k in name for k, _ in PARK_CANONICALS if k != target_key):
                continue
            return p
    return None


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_pass_info(evaluator: Evaluator, parent_node, ext: AnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="pass_identification_and_price",
        desc="Correctly identifies eligible least expensive Senior pass and states the correct price",
        parent=parent_node,
        critical=True,
    )

    # Existence checks
    evaluator.add_custom_node(
        result=bool(ext.pass_name and ext.pass_price),
        id="pass_name_price_present",
        desc="Pass name and price are both provided",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(ext.pass_urls),
        id="pass_sources_present",
        desc="Sources for pass details are provided",
        parent=node,
        critical=True,
    )

    # Verify the pass is a Senior pass (eligible for 62+)
    pass_is_senior = evaluator.add_leaf(
        id="pass_is_senior",
        desc="Identified pass is a Senior pass (eligible for age 62+)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The identified pass '{ext.pass_name}' is a Senior Pass (either the Senior Annual or Senior Lifetime) intended for U.S. citizens or permanent residents aged 62+.",
        node=pass_is_senior,
        sources=ext.pass_urls,
        additional_instruction="Verify that the named pass is a Senior pass and that the eligibility mentions age 62+."
    )

    # Verify the stated price for the identified pass
    pass_price_correct = evaluator.add_leaf(
        id="pass_price_correct",
        desc="Stated pass price is correct per official sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The price of the {ext.pass_name} is {ext.pass_price}.",
        node=pass_price_correct,
        sources=ext.pass_urls,
        additional_instruction="Confirm the exact price on the official source page(s)."
    )

    # Verify it is the least expensive eligible option for a 67-year-old
    least_expensive_ok = evaluator.add_leaf(
        id="least_expensive_eligible",
        desc="Identified pass is the least expensive eligible option for a 67-year-old",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"For a 67-year-old U.S. resident, among eligible America the Beautiful options (Senior Annual vs. Senior Lifetime), the least expensive choice is the Senior Annual Pass at around $20. The identified pass '{ext.pass_name}' priced at '{ext.pass_price}' is the least expensive eligible option.",
        node=least_expensive_ok,
        sources=ext.pass_urls,
        additional_instruction="Use the official Senior Pass info to compare Annual vs Lifetime pricing; confirm the chosen pass is the cheaper eligible option."
    )


async def verify_individual_fees_total(evaluator: Evaluator, parent_node, ext: AnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="individual_fees_total",
        desc="Correct total of 7-day vehicle entrance fees for all 5 specified parks without any pass",
        parent=parent_node,
        critical=True,
    )

    # Require a stated total
    evaluator.add_custom_node(
        result=bool(ext.total_without_pass),
        id="total_stated_present",
        desc="Total entrance fees without pass is stated",
        parent=node,
        critical=True,
    )

    # Per-park fee verifications
    park_urls_aggregate: List[str] = []
    numeric_fees: Dict[str, float] = {}

    for short_key, full_name in PARK_CANONICALS:
        sub = evaluator.add_parallel(
            id=f"{short_key}_fee_block",
            desc=f"{full_name}: verify 7-day private vehicle entrance fee",
            parent=node,
            critical=True,
        )
        park_item = _find_park_item(ext.parks or [], short_key)

        has_fee_and_source = evaluator.add_custom_node(
            result=bool(park_item and park_item.vehicle_7day_fee and park_item.urls),
            id=f"{short_key}_fee_and_source_present",
            desc=f"{full_name}: fee and at least one source URL are provided",
            parent=sub,
            critical=True,
        )

        fee_leaf = evaluator.add_leaf(
            id=f"{short_key}_fee_supported",
            desc=f"{full_name}: stated standard 7-day private vehicle fee is correct",
            parent=sub,
            critical=True,
        )

        fee_value_str = park_item.vehicle_7day_fee if park_item else ""
        urls = park_item.urls if (park_item and park_item.urls) else []
        if urls:
            park_urls_aggregate.extend(urls)

        # Store numeric fee if parsable
        amt = _extract_amount(fee_value_str)
        if amt is not None:
            numeric_fees[short_key] = amt

        await evaluator.verify(
            claim=f"The standard 7-day private vehicle entrance fee for {full_name} is {fee_value_str}.",
            node=fee_leaf,
            sources=urls,
            additional_instruction="Ignore timed-entry, shuttles, or special reservations; verify the standard 7-day non-commercial private vehicle entrance fee only."
        )

    # Verify the total using the combined per-park sources (LLM sums across pages)
    total_supported = evaluator.add_leaf(
        id="total_supported_by_sources",
        desc="Total vehicle entrance fees across all 5 parks is correct per official sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=("The combined total for standard 7-day private vehicle entrance fees for Yellowstone National Park, "
               "Grand Teton National Park, Glacier National Park, Yosemite National Park, and Rocky Mountain National Park "
               f"is {ext.total_without_pass}."),
        node=total_supported,
        sources=park_urls_aggregate,
        additional_instruction="Verify each park's posted vehicle fee on its official page(s) and sum them to check the stated total."
    )

    # Math check: sum of extracted per-park fees equals the stated total
    stated_total_amt = _extract_amount(ext.total_without_pass)
    computed_sum = sum(numeric_fees.get(k, 0.0) for k, _ in PARK_CANONICALS)
    can_check_math = (stated_total_amt is not None) and all(k in numeric_fees for k, _ in PARK_CANONICALS)

    evaluator.add_custom_node(
        result=(can_check_math and abs(computed_sum - stated_total_amt) < 0.01),
        id="total_math_matches_sum",
        desc=f"Sum of per-park fees (${computed_sum:.2f} if computed) matches the stated total ({ext.total_without_pass})",
        parent=node,
        critical=True,
    )


async def verify_cost_benefit(evaluator: Evaluator, parent_node, ext: AnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="cost_benefit_determination",
        desc="Correctly determines cost savings from purchasing the eligible pass vs paying individual fees",
        parent=parent_node,
        critical=True,
    )

    # Require both prices to be present
    evaluator.add_custom_node(
        result=bool(ext.pass_price and ext.total_without_pass),
        id="cost_inputs_present",
        desc="Pass price and total individual fees are both provided",
        parent=node,
        critical=True,
    )

    pass_amt = _extract_amount(ext.pass_price)
    total_amt = _extract_amount(ext.total_without_pass)
    computed_saves = (pass_amt is not None and total_amt is not None and pass_amt < total_amt)

    evaluator.add_custom_node(
        result=computed_saves,
        id="computed_savings_true",
        desc=f"Computed comparison shows savings: pass ({ext.pass_price}) < total ({ext.total_without_pass})",
        parent=node,
        critical=True,
    )

    # Check the answer's stated determination matches computed truth
    stated_bool = _map_yesno(ext.savings_determination)
    evaluator.add_custom_node(
        result=(stated_bool is not None and stated_bool == computed_saves),
        id="stated_savings_matches",
        desc="Stated cost-benefit determination matches the computed comparison",
        parent=node,
        critical=True,
    )

    # Additional LLM check on simple logic (no web sources needed)
    logic_leaf = evaluator.add_leaf(
        id="llm_logic_confirms_savings",
        desc="LLM confirms that the pass yields savings given the two amounts",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(f"Given a pass price of {ext.pass_price} and a total of individual entrance fees of "
               f"{ext.total_without_pass}, buying the pass would result in overall cost savings."),
        node=logic_leaf,
        additional_instruction="Purely logical check: compare the two amounts numerically to determine if buying the pass saves money."
    )


async def verify_coverage_limitation(evaluator: Evaluator, parent_node, ext: AnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="coverage_limitation_identified",
        desc="Identifies a valid category of fees not covered by the pass",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(ext.coverage_limitation),
        id="coverage_text_present",
        desc="Coverage limitation text is provided",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(ext.coverage_urls),
        id="coverage_sources_present",
        desc="Sources for coverage limitation are provided",
        parent=node,
        critical=True,
    )

    coverage_leaf = evaluator.add_leaf(
        id="coverage_not_covered_supported",
        desc="Stated category is indeed not covered by the pass (per official policy)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(f"The America the Beautiful passes do NOT cover '{ext.coverage_limitation}' fees. "
               "Valid examples of non-covered categories include camping, special tours, special permits, parking, or ferries."),
        node=coverage_leaf,
        sources=ext.coverage_urls,
        additional_instruction="Check official pass policy pages to confirm the named category is explicitly not covered."
    )


async def verify_purchase_url(evaluator: Evaluator, parent_node, ext: AnswerExtraction) -> None:
    node = evaluator.add_parallel(
        id="purchase_url_provided",
        desc="Provides a valid, official purchase URL for the identified pass",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(ext.purchase_url),
        id="purchase_url_present",
        desc="A purchase URL is provided",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_official_purchase_domain(ext.purchase_url),
        id="purchase_url_official_domain",
        desc="Purchase URL is on an official domain (Recreation.gov or USGS Store)",
        parent=node,
        critical=True,
    )

    sells_leaf = evaluator.add_leaf(
        id="purchase_page_sells_pass",
        desc="The provided page allows purchasing the identified pass",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This page allows purchasing the {ext.pass_name}.",
        node=sells_leaf,
        sources=ext.purchase_url,
        additional_instruction="Verify that the page is a purchase page (e.g., Add to Cart/Buy Now) for the specified pass."
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
    Evaluate an answer for the national parks pass and entrance fee comparison task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level checks are independent
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

    # Extract structured data from the answer
    ext = await evaluator.extract(
        prompt=prompt_extract_answer(),
        template_class=AnswerExtraction,
        extraction_name="answer_extraction",
    )

    # Build "Complete Task" aggregator (children are critical to enforce completeness)
    complete = evaluator.add_parallel(
        id="complete_task",
        desc="Provides all five required pieces of information",
        parent=root,
        critical=False,
    )

    # Run verifications (each major branch is critical; failure of any causes overall failure of 'complete_task')
    await verify_pass_info(evaluator, complete, ext)
    await verify_individual_fees_total(evaluator, complete, ext)
    await verify_cost_benefit(evaluator, complete, ext)
    await verify_coverage_limitation(evaluator, complete, ext)
    await verify_purchase_url(evaluator, complete, ext)

    return evaluator.get_summary()