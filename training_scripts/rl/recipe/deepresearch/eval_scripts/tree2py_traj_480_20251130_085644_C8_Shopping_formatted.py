import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bf2024_retailers"
TASK_DESCRIPTION = (
    "You are planning a comprehensive Black Friday 2024 shopping strategy and need to identify 5 major national "
    "retailers that meet specific criteria to maximize your shopping efficiency. For each of the 5 retailers, you "
    "must provide: (1) The retailer's name, (2) Their Black Friday 2024 (November 29, 2024) store opening time, "
    "(3) Their Black Friday 2024 store closing time, (4) Confirmation that they are closed on Thanksgiving Day "
    "(November 28, 2024), (5) Details of their extended holiday return policy, including the specific return deadline "
    "date, (6) Whether they offer in-store pickup or curbside pickup for online orders on Black Friday, and (7) A "
    "reference URL from an official source or credible news outlet supporting this information. All 5 retailers must "
    "be major national chains operating stores across the United States. Each retailer must be different from the others."
)

BF_DATE = "November 29, 2024"
THANKSGIVING_DATE = "November 28, 2024"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RetailerItem(BaseModel):
    name: Optional[str] = None
    opening_time: Optional[str] = None
    closing_time: Optional[str] = None
    thanksgiving_closed: Optional[str] = None  # e.g., "closed", "open", "confirmed closed", etc.
    return_policy_desc: Optional[str] = None
    return_deadline_date: Optional[str] = None  # e.g., "January 15, 2025"
    pickup_statement: Optional[str] = None  # e.g., "in-store pickup available", "curbside pickup available", etc.
    reference_urls: List[str] = Field(default_factory=list)


class RetailersExtraction(BaseModel):
    retailers: List[RetailerItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_retailers() -> str:
    return (
        "Extract the Black Friday 2024 information for major national U.S. retailers as presented in the ANSWER. "
        "Return exactly the retailers mentioned by the answer, preserving their order; do not invent any new ones.\n\n"
        "For each retailer, extract the following fields:\n"
        "1. name: The retailer's name as written in the answer.\n"
        "2. opening_time: The store opening time for Black Friday 2024 (Nov 29, 2024) as stated in the answer.\n"
        "3. closing_time: The store closing time for Black Friday 2024 (Nov 29, 2024) as stated in the answer.\n"
        "4. thanksgiving_closed: Whether the retailer is closed on Thanksgiving Day (Nov 28, 2024) according to the answer; "
        "   extract the phrasing used (e.g., 'closed', 'confirmed closed', 'not open'). If unknown, return null.\n"
        "5. return_policy_desc: The extended holiday return policy description as stated in the answer.\n"
        "6. return_deadline_date: The exact stated return deadline date (e.g., 'January 15, 2025'). If not provided, return null.\n"
        "7. pickup_statement: Whether in-store pickup and/or curbside pickup is available for online orders on Black Friday; "
        "   extract the answer's statement verbatim (e.g., 'in-store pickup available', 'curbside pickup available', 'both', 'not available'). If not provided, return null.\n"
        "8. reference_urls: A list of URLs (official retailer site or credible news outlets) cited in the answer for this retailer. "
        "   Extract only actual URLs visible in the answer (plain or markdown link). If none, return an empty list.\n\n"
        "Notes:\n"
        "- Do not infer or add anything not explicitly stated in the answer.\n"
        "- If the answer provides more than 5 retailers, still extract all; evaluation will use only the first 5.\n"
        "- If the answer provides fewer than 5 retailers, extract as many as present; missing ones will be handled by evaluation.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    cleaned = "".join(ch for ch in name.lower() if ch.isalnum() or ch.isspace()).strip()
    return cleaned or None


def first_five(items: List[RetailerItem]) -> List[RetailerItem]:
    return items[:5]


def pad_to_five(items: List[RetailerItem]) -> List[RetailerItem]:
    if len(items) >= 5:
        return items[:5]
    padded = list(items)
    while len(padded) < 5:
        padded.append(RetailerItem())
    return padded


# --------------------------------------------------------------------------- #
# Verification for one retailer                                               #
# --------------------------------------------------------------------------- #
async def verify_retailer(
    evaluator: Evaluator,
    parent_node,
    retailer: RetailerItem,
    idx: int,
) -> None:
    """
    Build the verification sub-tree for a single retailer (parallel aggregation).
    Each requirement is a critical leaf node under this retailer node.
    """
    retailer_idx = idx + 1
    retailer_node = evaluator.add_parallel(
        id=f"retailer_{retailer_idx}",
        desc=f"Retailer {retailer_idx} requirements",
        parent=parent_node,
        critical=False,  # Non-critical at retailer level; children are critical per rubric
    )

    # Reference URL existence (critical)
    ref_exists_result = bool(retailer.reference_urls)
    ref_node = evaluator.add_custom_node(
        result=ref_exists_result,
        id=f"retailer_{retailer_idx}_reference_url",
        desc=(
            f"Retailer {retailer_idx} provides at least one reference URL from an official source or credible news outlet "
            f"supporting the provided information"
        ),
        parent=retailer_node,
        critical=True,
    )

    # Major national chain verification (critical)
    major_chain_leaf = evaluator.add_leaf(
        id=f"retailer_{retailer_idx}_is_major_national_chain",
        desc=(
            f"Retailer {retailer_idx} is identified by name and is a major national chain operating across the United States"
        ),
        parent=retailer_node,
        critical=True,
    )
    claim_major = (
        f"The retailer named '{retailer.name}' is a major national chain operating stores across the United States."
    )
    await evaluator.verify(
        claim=claim_major,
        node=major_chain_leaf,
        sources=retailer.reference_urls,
        additional_instruction=(
            "Confirm that the page(s) clearly indicate national, nationwide, or multi-state U.S. presence for the retailer. "
            "Look for wording like 'nationwide', 'across the U.S.', 'stores in many states', or explicit scale (hundreds/thousands "
            "of U.S. stores). Official 'About' pages or credible news outlets are acceptable evidence."
        ),
        extra_prerequisites=[ref_node],
    )

    # Opening time must be either 6 a.m. or 7 a.m. (critical)
    opening_leaf = evaluator.add_leaf(
        id=f"retailer_{retailer_idx}_opening_time",
        desc=(
            f"Retailer {retailer_idx} Black Friday 2024 (Nov 29, 2024) opening time is stated and is either 6 a.m. or 7 a.m."
        ),
        parent=retailer_node,
        critical=True,
    )
    claim_opening = (
        f"On Black Friday 2024 ({BF_DATE}), '{retailer.name}' store opening time is '{retailer.opening_time}', "
        f"and that time is either 6 a.m. or 7 a.m."
    )
    await evaluator.verify(
        claim=claim_opening,
        node=opening_leaf,
        sources=retailer.reference_urls,
        additional_instruction=(
            "Verify the Black Friday store opening time for Nov 29, 2024. The claim must be explicitly supported by the page(s) "
            "and the time must be 6 a.m. or 7 a.m. Accept reasonable formatting variants (e.g., '6 AM', '6:00 a.m.', '7am')."
        ),
        extra_prerequisites=[ref_node],
    )

    # Closing time is stated (critical)
    closing_leaf = evaluator.add_leaf(
        id=f"retailer_{retailer_idx}_closing_time",
        desc=f"Retailer {retailer_idx} Black Friday 2024 (Nov 29, 2024) closing time is stated",
        parent=retailer_node,
        critical=True,
    )
    claim_closing = (
        f"On Black Friday 2024 ({BF_DATE}), '{retailer.name}' store closing time is stated as '{retailer.closing_time}'."
    )
    await evaluator.verify(
        claim=claim_closing,
        node=closing_leaf,
        sources=retailer.reference_urls,
        additional_instruction=(
            "Check that the page(s) clearly state a closing time for Black Friday 2024 (Nov 29, 2024) and that it matches "
            "the provided closing time value."
        ),
        extra_prerequisites=[ref_node],
    )

    # Thanksgiving closed confirmation (critical)
    tg_leaf = evaluator.add_leaf(
        id=f"retailer_{retailer_idx}_thanksgiving_closed",
        desc=f"Retailer {retailer_idx} is confirmed closed on Thanksgiving Day (Nov 28, 2024)",
        parent=retailer_node,
        critical=True,
    )
    claim_tg = (
        f"'{retailer.name}' is closed on Thanksgiving Day ({THANKSGIVING_DATE})."
    )
    await evaluator.verify(
        claim=claim_tg,
        node=tg_leaf,
        sources=retailer.reference_urls,
        additional_instruction=(
            "Verify that the page(s) explicitly state the stores are closed on Thanksgiving Day (Thursday, Nov 28, 2024). "
            "Accept phrases like 'closed Thanksgiving' or 'not open on Thanksgiving'."
        ),
        extra_prerequisites=[ref_node],
    )

    # Return policy deadline with specific date (critical)
    return_leaf = evaluator.add_leaf(
        id=f"retailer_{retailer_idx}_return_policy_deadline",
        desc=(
            f"Retailer {retailer_idx} extended holiday return policy is described and includes a specific return deadline date"
        ),
        parent=retailer_node,
        critical=True,
    )
    claim_return = (
        f"'{retailer.name}' has an extended holiday return policy with a specific return deadline date of "
        f"'{retailer.return_deadline_date}'."
    )
    await evaluator.verify(
        claim=claim_return,
        node=return_leaf,
        sources=retailer.reference_urls,
        additional_instruction=(
            "Verify that the page(s) state an extended holiday return window with a clear deadline date (e.g., 'returns accepted "
            "until January 15, 2025'). Minor date format variations are acceptable, but a specific calendar date must be present."
        ),
        extra_prerequisites=[ref_node],
    )

    # Pickup option availability on Black Friday (critical)
    pickup_leaf = evaluator.add_leaf(
        id=f"retailer_{retailer_idx}_pickup_option",
        desc=(
            f"Retailer {retailer_idx} states whether in-store pickup and/or curbside pickup is available for online orders "
            f"on Black Friday"
        ),
        parent=retailer_node,
        critical=True,
    )
    claim_pickup = (
        f"On Black Friday 2024 ({BF_DATE}), '{retailer.name}' offers the following for online orders: '{retailer.pickup_statement}'."
    )
    await evaluator.verify(
        claim=claim_pickup,
        node=pickup_leaf,
        sources=retailer.reference_urls,
        additional_instruction=(
            "Verify that the page(s) state whether in-store pickup and/or curbside pickup is available for online orders. "
            "If the page indicates normal availability and does not mention any Black Friday suspension, consider availability "
            "as applying on Black Friday as well."
        ),
        extra_prerequisites=[ref_node],
    )


# --------------------------------------------------------------------------- #
# Cross-retailer constraints                                                  #
# --------------------------------------------------------------------------- #
def build_cross_constraints(
    evaluator: Evaluator,
    parent_node,
    extracted_retailers: List[RetailerItem],
    used_retailers: List[RetailerItem],
) -> None:
    """
    Build the cross-retailer constraints node (critical).
    - exactly_five_retailers_provided: The original answer provides exactly 5 retailers.
    - all_retailers_distinct: All 5 retailers used for evaluation are distinct.
    """
    constraints_node = evaluator.add_parallel(
        id="set_level_constraints",
        desc="Cross-retailer constraints",
        parent=parent_node,
        critical=True,
    )

    # Exactly five provided in the original answer
    exactly_five = len(extracted_retailers) == 5
    evaluator.add_custom_node(
        result=exactly_five,
        id="exactly_five_retailers_provided",
        desc="Response provides 5 retailers (not fewer or more)",
        parent=constraints_node,
        critical=True,
    )

    # All five distinct (based on normalized names in the first five used)
    used_names = [normalize_name(r.name) for r in used_retailers]
    all_present = all(n is not None and n != "" for n in used_names)
    all_distinct = len(set(used_names)) == 5 if all_present else False

    evaluator.add_custom_node(
        result=all_distinct,
        id="all_retailers_distinct",
        desc="All 5 retailers are different from each other (no duplicates)",
        parent=constraints_node,
        critical=True,
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
    Evaluate an answer for the Black Friday 2024 retailers task.
    """
    # Initialize evaluator (root is non-critical by framework design; we add critical sub-nodes where needed)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: each retailer evaluated independently + cross constraints
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

    # Extract structured retailer information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_retailers(),
        template_class=RetailersExtraction,
        extraction_name="retailers_extraction",
    )

    original_count = len(extraction.retailers)
    used_retailers = first_five(extraction.retailers)
    padded_retailers = pad_to_five(used_retailers)

    # Record some auxiliary info for transparency
    evaluator.add_custom_info(
        info={
            "original_retailer_count": original_count,
            "used_retailer_count": len(used_retailers),
            "padded_to_five": len(padded_retailers),
            "black_friday_date": BF_DATE,
            "thanksgiving_date": THANKSGIVING_DATE,
        },
        info_type="meta",
        info_name="evaluation_setup",
    )

    # Build cross-retailer constraints (critical)
    build_cross_constraints(
        evaluator=evaluator,
        parent_node=root,
        extracted_retailers=extraction.retailers,
        used_retailers=padded_retailers,
    )

    # Build retailer-specific verification nodes (non-critical at top level)
    for idx, retailer in enumerate(padded_retailers):
        await verify_retailer(
            evaluator=evaluator,
            parent_node=root,
            retailer=retailer,
            idx=idx,
        )

    # Return summary
    return evaluator.get_summary()