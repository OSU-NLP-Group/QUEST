import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "holiday_craft_hours_2025"
TASK_DESCRIPTION = (
    "I'm planning a holiday craft shopping trip in late 2025 and want to compare the operating hours of major craft "
    "store chains. For both Hobby Lobby and Michaels stores in the United States, please provide the following "
    "information for the 2025 holiday season:\n\n"
    "1. Black Friday (November 28, 2025): What are the store hours?\n"
    "2. Christmas Eve (December 24, 2025): What are the store hours?\n\n"
    "For each store, include a reference URL from the store's official website, newsroom, or press release that "
    "confirms these holiday hours."
)

# Optional ground-truth expectation reference (not used for scoring directly)
GROUND_TRUTH_EXPECTED = {
    "Hobby Lobby": {
        "black_friday": "8:00 a.m. to 9:00 p.m.",
        "christmas_eve": "9:00 a.m. to 5:30 p.m.",
    },
    "Michaels": {
        "black_friday": "7:00 a.m. to 10:00 p.m.",
        "christmas_eve_expected_options": "Open at 7:00 a.m. or 8:00 a.m.; close at 6:00 p.m.",
    },
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreHolidayHours(BaseModel):
    black_friday: Optional[str] = None
    christmas_eve: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class HolidayHoursExtraction(BaseModel):
    hobby_lobby: Optional[StoreHolidayHours] = None
    michaels: Optional[StoreHolidayHours] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_holiday_hours() -> str:
    return """
    Extract the 2025 holiday store hours for each of the following U.S. chains as explicitly stated in the provided answer text:
    - Hobby Lobby
    - Michaels

    For each store, extract:
    1) black_friday: The hours for Black Friday 2025 (Nov 28, 2025), exactly as written in the answer (keep formatting such as "7am-10pm", "7:00 a.m. to 10:00 p.m.", etc.).
    2) christmas_eve: The hours for Christmas Eve 2025 (Dec 24, 2025), exactly as written in the answer.
    3) reference_urls: A list of all URLs that the answer cites as references supporting those hours for that store. Extract only URLs explicitly present in the answer text (plain links or markdown links). Do not invent any URLs.

    Notes:
    - If a store or a specific date's hours are not mentioned, set that field to null.
    - If no reference URLs are provided for a store, return an empty list for reference_urls.
    - Prefer the general, chain-wide hours if multiple hours are mentioned (e.g., not store-specific exceptions).

    Return a JSON object with two top-level keys: "hobby_lobby" and "michaels".
    Each should be an object with fields: black_friday, christmas_eve, reference_urls.
    """


# --------------------------------------------------------------------------- #
# Helper: verification per store                                              #
# --------------------------------------------------------------------------- #
async def verify_store_hours(
    evaluator: Evaluator,
    parent_node,
    store_key: str,
    store_name: str,
    store_data: Optional[StoreHolidayHours],
) -> None:
    """
    Build verification subtree for a single store.
    - Ensures hours and at least one reference URL are provided.
    - Verifies Black Friday hours against cited official sources.
    - Verifies Christmas Eve hours against cited official sources.
    - Verifies that at least one reference URL is an official company page (website/newsroom/press).
    """
    group_node = evaluator.add_parallel(
        id=f"{store_key}_holiday_hours",
        desc=f"Provides accurate 2025 holiday hours for {store_name} stores",
        parent=parent_node,
        critical=True,  # Critical: this store's correctness is required
    )

    # Gather values safely
    bf_hours = store_data.black_friday.strip() if (store_data and store_data.black_friday) else ""
    ce_hours = store_data.christmas_eve.strip() if (store_data and store_data.christmas_eve) else ""
    urls: List[str] = (store_data.reference_urls if store_data else []) or []

    # Existence gate (critical sibling) – if this fails, other leaves will auto-skip
    existence_ok = (bf_hours != "") and (ce_hours != "") and (len(urls) > 0)
    evaluator.add_custom_node(
        result=existence_ok,
        id=f"{store_key}_existence",
        desc=f"{store_name} holiday hours info (both dates) and at least one reference URL are provided in the answer",
        parent=group_node,
        critical=True,
    )

    # Black Friday verification (leaf)
    bf_leaf = evaluator.add_leaf(
        id=f"{'Hobby_Lobby' if store_key=='hobby_lobby' else 'Michaels'}_Black_Friday",
        desc=(
            "Specifies that Hobby Lobby is open on Black Friday 2025 from 8:00 a.m. to 9:00 p.m."
            if store_key == "hobby_lobby"
            else "Specifies that Michaels is open on Black Friday 2025 from 7:00 a.m. to 10:00 p.m."
        ),
        parent=group_node,
        critical=True,
    )
    bf_claim = (
        f"For U.S. stores on Black Friday 2025 (Nov 28, 2025), {store_name} announced store hours as: {bf_hours}. "
        f"The cited page should explicitly confirm the 2025 Black Friday hours (equivalent time formats acceptable; "
        f'allow variants like "7am" vs "7:00 a.m.", en-dash vs "to").'
    )
    bf_additional = (
        "Verify that the page clearly refers to 2025 Black Friday hours for U.S. stores. "
        "Minor formatting differences (am/pm vs a.m./p.m.; hyphen vs en‑dash) should be considered equivalent. "
        "If the URLs are irrelevant, invalid, or not official, the claim is not supported."
    )

    # Christmas Eve verification (leaf)
    ce_leaf = evaluator.add_leaf(
        id=f"{'Hobby_Lobby' if store_key=='hobby_lobby' else 'Michaels'}_Christmas_Eve",
        desc=(
            "Specifies that Hobby Lobby is open on Christmas Eve 2025 from 9:00 a.m. to 5:30 p.m."
            if store_key == "hobby_lobby"
            else "Specifies that Michaels is open on Christmas Eve 2025 with early closing (acceptable opening times: 7:00 a.m. or 8:00 a.m.; closing time: 6:00 p.m.)"
        ),
        parent=group_node,
        critical=True,
    )
    ce_claim = (
        f"For U.S. stores on Christmas Eve 2025 (Dec 24, 2025), {store_name} announced store hours as: {ce_hours}. "
        f"The cited page should explicitly confirm the 2025 Christmas Eve hours (equivalent formatting acceptable)."
    )
    ce_additional = (
        "Confirm the date is 2025-12-24 and that the page states the chain-wide hours. "
        "For Michaels, early opening at 7:00 a.m. or 8:00 a.m. with a 6:00 p.m. close is typical; however, only pass "
        "if the hours on the page match the claim text. Minor format differences are acceptable. "
        "If the URLs are irrelevant, invalid, or not official, the claim is not supported."
    )

    # Reference URL validity (leaf) – checks official nature of at least one URL
    ref_leaf = evaluator.add_leaf(
        id=f"{'Hobby_Lobby' if store_key=='hobby_lobby' else 'Michaels'}_Reference_URL",
        desc=(
            "Provides a valid reference URL from Hobby Lobby's official source that confirms the holiday hours"
            if store_key == "hobby_lobby"
            else "Provides a valid reference URL from Michaels' official source that confirms the holiday hours"
        ),
        parent=group_node,
        critical=True,
    )
    if store_key == "hobby_lobby":
        ref_claim = (
            "This page is an official page from Hobby Lobby (e.g., hobbylobby.com or the company's newsroom/press site) "
            "and it discusses 2025 holiday store hours (Black Friday or Christmas Eve)."
        )
    else:
        ref_claim = (
            "This page is an official page from Michaels (e.g., michaels.com or the company's newsroom/press site) "
            "and it discusses 2025 holiday store hours (Black Friday or Christmas Eve)."
        )
    ref_additional = (
        "Look for brand indicators such as official domain, site header/footer, copyright, or pressroom. "
        "Third-party blogs, news aggregators, or coupon sites are not official. "
        "The page should pertain to holiday hours and reference the 2025 season."
    )

    # Run verifications in parallel under this group
    await evaluator.batch_verify(
        [
            (bf_claim, urls, bf_leaf, bf_additional),
            (ce_claim, urls, ce_leaf, ce_additional),
            (ref_claim, urls, ref_leaf, ref_additional),
        ]
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
    Evaluate an answer for the 2025 holiday shopping hours (Hobby Lobby & Michaels).
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Two stores evaluated independently
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

    # Add high-level grouping node (non-critical root; children will be critical to enforce all-or-nothing)
    top_node = evaluator.add_parallel(
        id="Holiday_Shopping_Hours_Comparison",
        desc="Provides accurate 2025 holiday shopping hours for both Hobby Lobby and Michaels stores for the specified holidays",
        parent=root,
        critical=False,
    )

    # Record ground-truth reference info (not used for scoring)
    evaluator.add_ground_truth(
        {
            "expected_notes": GROUND_TRUTH_EXPECTED,
            "holidays": {
                "black_friday": "2025-11-28",
                "christmas_eve": "2025-12-24",
            },
        },
        gt_type="rubric_reference",
    )

    # Extract structured info from answer
    extracted: HolidayHoursExtraction = await evaluator.extract(
        prompt=prompt_extract_holiday_hours(),
        template_class=HolidayHoursExtraction,
        extraction_name="holiday_hours_2025",
    )

    # Build critical per-store nodes under top node
    # Hobby Lobby subtree
    await verify_store_hours(
        evaluator=evaluator,
        parent_node=top_node,
        store_key="hobby_lobby",
        store_name="Hobby Lobby",
        store_data=extracted.hobby_lobby,
    )

    # Michaels subtree
    await verify_store_hours(
        evaluator=evaluator,
        parent_node=top_node,
        store_key="michaels",
        store_name="Michaels",
        store_data=extracted.michaels,
    )

    return evaluator.get_summary()