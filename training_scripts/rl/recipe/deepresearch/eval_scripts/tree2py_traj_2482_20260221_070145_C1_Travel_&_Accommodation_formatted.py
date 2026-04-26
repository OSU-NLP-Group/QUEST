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
TASK_ID = "clt_parking_5d_budget_100_mar2026"
TASK_DESCRIPTION = (
    "I'm planning to park my car at Charlotte Douglas International Airport for 5 days while I travel. "
    "My budget for parking is a maximum of $100 total. What is the most cost-effective parking option at Charlotte airport "
    "that stays within my budget? Please provide the name of the parking lot, the total cost for 5 days, and any special "
    "booking requirements for this option. Use current rates as of March 2026."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ParkingSelectionExtraction(BaseModel):
    """
    The single parking option selected/recommended in the answer for a 5‑day stay at CLT,
    including basic pricing and booking requirement text, plus all cited URLs.
    """
    lot_name: Optional[str] = None
    daily_rate: Optional[str] = None            # As written in the answer (e.g., "$18/day")
    total_cost_5d: Optional[str] = None         # As written in the answer (e.g., "$90")
    booking_requirements: Optional[str] = None  # The exact phrase/sentence describing booking method
    sources: List[str] = Field(default_factory=list)  # All URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parking_option() -> str:
    return """
Extract the single parking option that the answer recommends as the most cost‑effective for a 5‑day stay at Charlotte Douglas International Airport (CLT) within a $100 total budget (as of March 2026). Return:

- lot_name: The exact name of the selected parking product (e.g., “Express Deck 1”, “Daily Decks”, “Long Term Lot”, “Economy Lot”, etc.). If multiple lots are listed, pick the one the answer ultimately recommends as the cheapest within budget. If no clear recommendation, return null.
- daily_rate: The per‑day rate mentioned in the answer for the selected lot, exactly as written (include currency symbol if present). If no daily rate is stated in the answer, return null.
- total_cost_5d: The total cost for 5 days mentioned in the answer for the selected lot, exactly as written. Do NOT compute or invent; extract only if explicitly stated. If not given, return null.
- booking_requirements: The sentence or phrase from the answer that describes booking method or restrictions (e.g., “must book online via ParkCLT/app; no walk‑up”). If not mentioned, return null.
- sources: A list of all URLs cited in the answer (e.g., CLT/ParkCLT pages). Extract only actual URLs that appear in the answer.

Only extract what is explicitly present in the answer. Do not infer or compute new values.
"""


# --------------------------------------------------------------------------- #
# Helper for booking requirement mention check                                #
# --------------------------------------------------------------------------- #
def _mentions_online_and_no_walkup(text: Optional[str]) -> bool:
    """
    Heuristic check whether the booking requirement text explicitly mentions
    online/app (ParkCLT) booking and indicates no walk‑up/drive‑up.
    """
    if not text:
        return False
    t = text.lower()
    mentions_online = any(kw in t for kw in [
        "online", "parkclt", "pre-book", "prebook", "pre book", "advance booking", "app"
    ])
    mentions_no_walkup = any(kw in t for kw in [
        "no walk", "not available for walk-up", "walk-up not available",
        "no drive", "drive-up not", "no drive-up", "no pay-on-entry", "no pay on entry"
    ])
    return mentions_online and mentions_no_walkup


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root,
    extracted: ParkingSelectionExtraction
) -> None:
    """
    Build the verification tree per rubric and run verifications.
    Root rubric node is critical; all child criteria are critical as well.
    """

    # Root rubric node (critical)
    budget_node = evaluator.add_parallel(
        id="Budget_Compliant_Parking_Selection",
        desc=("Evaluates whether the traveler has correctly identified the most cost-effective parking option at "
              "Charlotte airport for a 5-day stay that remains within a $100 budget, including the lot name, total cost, "
              "and booking requirements."),
        parent=root,
        critical=True
    )

    # ----------------------------- Criterion 1 ----------------------------- #
    # Cheapest qualifying lot identified (critical)
    cheapest_node = evaluator.add_parallel(
        id="Cheapest_Qualifying_Lot_Identified",
        desc=("The parking lot identified is the option with the lowest daily rate among all Charlotte airport parking "
              "facilities that result in a total cost under $100 for 5 days."),
        parent=budget_node,
        critical=True
    )

    # Existence check: lot name and sources provided
    evaluator.add_custom_node(
        result=bool(extracted.lot_name) and bool(extracted.sources),
        id="cheapest_inputs_present",
        desc="Lot name is provided and at least one source URL is cited for rate comparison.",
        parent=cheapest_node,
        critical=True
    )

    # Evidence-backed verification for cheapest qualifying selection
    cheapest_supported_leaf = evaluator.add_leaf(
        id="cheapest_supported_by_sources",
        desc=("The selected lot is indeed the cheapest CLT on‑airport option among those whose 5‑day total is ≤ $100 "
              "as of March 2026."),
        parent=cheapest_node,
        critical=True
    )

    cheapest_claim = (
        f"As of March 2026 at Charlotte Douglas International Airport (CLT), among on‑airport ParkCLT parking products, "
        f"the product with the lowest daily price that yields a 5‑day total at or below $100 is '{extracted.lot_name}'."
    )
    await evaluator.verify(
        claim=cheapest_claim,
        node=cheapest_supported_leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Use only official CLT/ParkCLT pages that list current product rates. Compute 5‑day totals as 5 × daily rate; "
            "ignore taxes/fees unless the page explicitly includes them in the daily price. "
            "If multiple products tie at the lowest qualifying rate, selecting any one of the tied products counts as correct. "
            "Exclude off‑airport third‑party lots from consideration."
        )
    )

    # ----------------------------- Criterion 2 ----------------------------- #
    # Correct total cost calculated (critical)
    cost_node = evaluator.add_parallel(
        id="Correct_Total_Cost_Calculated",
        desc=("The total parking cost for 5 days is correctly calculated by multiplying the identified lot's daily rate by 5 days."),
        parent=budget_node,
        critical=True
    )

    # Existence check: daily rate and total provided in the answer
    evaluator.add_custom_node(
        result=bool(extracted.daily_rate) and bool(extracted.total_cost_5d),
        id="cost_inputs_present",
        desc="Daily rate and 5‑day total are both explicitly provided in the answer.",
        parent=cost_node,
        critical=True
    )

    # Arithmetic check via simple verification (non‑web factual)
    calc_leaf = evaluator.add_leaf(
        id="arithmetic_total_equals_5x_rate",
        desc="The 5‑day total equals 5 × the stated daily rate (allowing minor rounding).",
        parent=cost_node,
        critical=True
    )

    calc_claim = (
        f"For the selected lot '{extracted.lot_name}', the daily rate is '{extracted.daily_rate}' per day and the "
        f"5‑day total is stated as '{extracted.total_cost_5d}'. The 5‑day total equals five times the daily rate "
        f"within normal rounding."
    )
    await evaluator.verify(
        claim=calc_claim,
        node=calc_leaf,
        additional_instruction=(
            "Treat this as a simple arithmetic check (no web lookup needed). "
            "Extract numeric values from the strings, ignore currency symbols and text, and allow cents‑level rounding differences."
        )
    )

    # ----------------------------- Criterion 3 ----------------------------- #
    # Booking method requirement stated (critical)
    booking_node = evaluator.add_parallel(
        id="Booking_Method_Requirement_Stated",
        desc=("The answer mentions that the identified parking lot requires online booking (via parkclt.com or the CLT Airport app) "
              "and is not available for walk‑up."),
        parent=budget_node,
        critical=True
    )

    # Existence check: the answer explicitly mentions online/app only and that walk‑up/drive‑up is not accepted
    evaluator.add_custom_node(
        result=_mentions_online_and_no_walkup(extracted.booking_requirements),
        id="booking_requirement_mention_present",
        desc="Answer text explicitly states online/app booking only and no walk‑up/drive‑up.",
        parent=booking_node,
        critical=True
    )

    # Evidence-backed verification that this policy is correct
    booking_supported_leaf = evaluator.add_leaf(
        id="booking_requirement_supported_by_sources",
        desc="Online‑only (ParkCLT/app) and no walk‑up requirement is supported by cited sources.",
        parent=booking_node,
        critical=True
    )

    booking_claim = (
        f"The '{extracted.lot_name}' parking option at CLT requires online booking via ParkCLT.com or the official CLT Airport app, "
        f"and does not accept walk‑up/drive‑up purchase."
    )
    await evaluator.verify(
        claim=booking_claim,
        node=booking_supported_leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Look for language like 'pre‑book only', 'online only', 'no drive‑up', or 'no walk‑up' on official CLT/ParkCLT pages. "
            "Equivalent phrasing counts as support."
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
    Evaluate an answer for the CLT parking (5 days, ≤ $100) task and return an evaluation summary.
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
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_parking_option(),
        template_class=ParkingSelectionExtraction,
        extraction_name="selected_parking_option"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()