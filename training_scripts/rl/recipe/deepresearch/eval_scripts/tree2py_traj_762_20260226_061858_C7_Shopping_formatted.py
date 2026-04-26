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
TASK_ID = "christmas_eve_louisville_2025"
TASK_DESCRIPTION = (
    "I need to create a comprehensive Christmas Eve 2025 shopping reference guide for my family in Louisville, Kentucky. "
    "Please provide the following specific information for Christmas Eve (December 24, 2025) in Louisville, KY:\n"
    "1. What time do Walmart stores close?\n"
    "2. What time do Target stores close?\n"
    "3. What time do Kroger stores close?\n"
    "4. What time do ALDI stores close?\n"
    "5. What time do Publix stores close?\n"
    "6. What are Walmart pharmacy hours on Christmas Eve?\n"
    "7. What time do Kroger pharmacies close on Christmas Eve?\n"
    "8. Does Walmart offer curbside pickup service on Christmas Eve?\n"
    "9. Does Target offer same-day delivery service on Christmas Eve?\n"
    "10. Does Kroger offer curbside pickup service on Christmas Eve?\n"
    "11. Does ALDI offer curbside pickup service on Christmas Eve?\n"
    "12. Which of these major stores (Walmart, Target, Kroger, ALDI, Publix) are closed on Christmas Day (December 25, 2025)?\n"
    "13. Please provide reference URLs to verify this information."
)

# Expected values per rubric (used to formulate verification claims)
EXPECTED = {
    "walmart_close": "6 p.m.",
    "target_close": "8 p.m.",
    "kroger_close": "6 p.m.",
    "aldi_close": "4 p.m.",
    "publix_close": "7 p.m.",
    "walmart_rx_hours": "9 a.m. to 5 p.m.",
    "kroger_rx_close_by": "5 p.m. or earlier",
    "service_yes": "yes",
    "closed_yes": "closed"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreInfo(BaseModel):
    # Christmas Eve (Dec 24, 2025) — Louisville, KY
    closing_time: Optional[str] = None
    closing_sources: List[str] = Field(default_factory=list)

    # Pharmacies (store-specific: Walmart has hours range; Kroger has closing time)
    pharmacy_hours: Optional[str] = None  # e.g., "9 a.m. to 5 p.m."
    pharmacy_sources: List[str] = Field(default_factory=list)

    pharmacy_closing_time: Optional[str] = None  # e.g., "5 p.m."
    pharmacy_closing_sources: List[str] = Field(default_factory=list)

    # Services (Christmas Eve; yes/no)
    curbside_pickup: Optional[str] = None
    curbside_sources: List[str] = Field(default_factory=list)

    same_day_delivery: Optional[str] = None  # Target-specific
    same_day_sources: List[str] = Field(default_factory=list)

    # Christmas Day (Dec 25, 2025) — closure status for Louisville, KY
    christmas_closed: Optional[str] = None  # expected "closed"
    christmas_closed_sources: List[str] = Field(default_factory=list)


class ShoppingExtraction(BaseModel):
    walmart: Optional[StoreInfo] = None
    target: Optional[StoreInfo] = None
    kroger: Optional[StoreInfo] = None
    aldi: Optional[StoreInfo] = None
    publix: Optional[StoreInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shopping_info() -> str:
    return """
    Extract structured information exactly as it appears in the provided answer text for Louisville, Kentucky, focusing on Christmas Eve (December 24, 2025) and Christmas Day (December 25, 2025). Do NOT invent any data. Return null for any missing field. For each field that requires URLs, extract only explicit URLs from the answer (including Markdown links).

    Return a JSON object with the following schema:

    {
      "walmart": {
        "closing_time": string | null,                 // e.g., "6 p.m.", "6 PM", "6pm"
        "closing_sources": string[],

        "pharmacy_hours": string | null,               // e.g., "9 a.m. to 5 p.m."
        "pharmacy_sources": string[],

        "pharmacy_closing_time": null,                 // leave null for Walmart
        "pharmacy_closing_sources": [],

        "curbside_pickup": string | null,              // "yes" / "no" or textual indication
        "curbside_sources": string[],

        "same_day_delivery": null,                     // leave null for Walmart
        "same_day_sources": [],

        "christmas_closed": string | null,             // e.g., "closed", "open", "yes", "no"
        "christmas_closed_sources": string[]
      },

      "target": {
        "closing_time": string | null,
        "closing_sources": string[],

        "pharmacy_hours": null,
        "pharmacy_sources": [],

        "pharmacy_closing_time": null,
        "pharmacy_closing_sources": [],

        "curbside_pickup": null,
        "curbside_sources": [],

        "same_day_delivery": string | null,            // "yes" / "no" or textual indication
        "same_day_sources": string[],

        "christmas_closed": string | null,
        "christmas_closed_sources": string[]
      },

      "kroger": {
        "closing_time": string | null,
        "closing_sources": string[],

        "pharmacy_hours": null,
        "pharmacy_sources": [],

        "pharmacy_closing_time": string | null,        // e.g., "5 p.m."
        "pharmacy_closing_sources": string[],

        "curbside_pickup": string | null,              // "yes" / "no" or textual indication
        "curbside_sources": string[],

        "same_day_delivery": null,
        "same_day_sources": [],

        "christmas_closed": string | null,
        "christmas_closed_sources": string[]
      },

      "aldi": {
        "closing_time": string | null,
        "closing_sources": string[],

        "pharmacy_hours": null,
        "pharmacy_sources": [],

        "pharmacy_closing_time": null,
        "pharmacy_closing_sources": [],

        "curbside_pickup": string | null,              // "yes" / "no" or textual indication
        "curbside_sources": string[],

        "same_day_delivery": null,
        "same_day_sources": [],

        "christmas_closed": string | null,
        "christmas_closed_sources": string[]
      },

      "publix": {
        "closing_time": string | null,
        "closing_sources": string[],

        "pharmacy_hours": null,
        "pharmacy_sources": [],

        "pharmacy_closing_time": null,
        "pharmacy_closing_sources": [],

        "curbside_pickup": null,
        "curbside_sources": [],

        "same_day_delivery": null,
        "same_day_sources": [],

        "christmas_closed": string | null,
        "christmas_closed_sources": string[]
      }
    }

    Rules:
    - All times should be returned exactly as they appear in the answer (keep formatting like "6 p.m.", "6 PM", etc.).
    - Only include URLs actually present in the answer; do not infer or construct any.
    - For yes/no type fields, prefer "yes" or "no" if the answer uses those words; otherwise return the exact phrase used (e.g., "offers curbside", "closed", etc.).
    - If the answer mentions different hours by location, choose the Louisville, KY specific information; if unavailable, keep what the answer states and the corresponding URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(x: Optional[str]) -> bool:
    return bool(x and str(x).strip())


def _nonempty(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def _urls_or_empty(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# --------------------------------------------------------------------------- #
# Verification logic by store                                                 #
# --------------------------------------------------------------------------- #
async def verify_walmart(evaluator: Evaluator, parent, data: ShoppingExtraction) -> None:
    node = evaluator.add_parallel(
        id="Walmart",
        desc="Walmart required information for Louisville, KY on Dec 24, 2025.",
        parent=parent,
        critical=True
    )
    w = data.walmart or StoreInfo()

    # Store closing time (expected 6 p.m.) — add existence checks
    evaluator.add_custom_node(
        result=_has_text(w.closing_time),
        id="walmart_closing_value_present",
        desc="Walmart closing time value is present in answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(w.closing_sources),
        id="walmart_closing_sources_present",
        desc="Walmart closing time sources are provided",
        parent=node,
        critical=True
    )

    leaf_close = evaluator.add_leaf(
        id="Walmart_Store_Closing_Time",
        desc="States Walmart store closing time on Christmas Eve (Dec 24, 2025) in Louisville is 6 p.m. local time.",
        parent=node,
        critical=True
    )
    claim_close = (
        "On Christmas Eve (December 24, 2025) in Louisville, Kentucky, Walmart stores close at 6 p.m. local time."
    )
    await evaluator.verify(
        claim=claim_close,
        node=leaf_close,
        sources=_urls_or_empty(w.closing_sources),
        additional_instruction=(
            "Verify that the cited page(s) indicate Christmas Eve hours for Walmart applicable to Louisville, KY "
            "or a clearly relevant local/official context for the Louisville area on December 24, 2025. "
            "Allow minor formatting variants like '6 PM'/'6 p.m.' but the time should be 6 p.m."
        ),
    )

    # Pharmacy hours (expected 9 a.m. to 5 p.m.)
    evaluator.add_custom_node(
        result=_has_text(w.pharmacy_hours),
        id="walmart_rx_value_present",
        desc="Walmart pharmacy hours value is present in answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(w.pharmacy_sources),
        id="walmart_rx_sources_present",
        desc="Walmart pharmacy hours sources are provided",
        parent=node,
        critical=True
    )

    leaf_rx = evaluator.add_leaf(
        id="Walmart_Pharmacy_Hours",
        desc="States Walmart pharmacy hours on Christmas Eve (Dec 24, 2025) in Louisville are 9 a.m. to 5 p.m.",
        parent=node,
        critical=True
    )
    claim_rx = (
        "On Christmas Eve (December 24, 2025) in Louisville, Kentucky, Walmart pharmacy hours are 9 a.m. to 5 p.m."
    )
    await evaluator.verify(
        claim=claim_rx,
        node=leaf_rx,
        sources=_urls_or_empty(w.pharmacy_sources),
        additional_instruction=(
            "Confirm that the page(s) provide Walmart pharmacy hours for Christmas Eve specific to Louisville, KY "
            "or a directly-relevant official context. Accept '9 AM to 5 PM' variations."
        ),
    )

    # Curbside pickup availability (expected: yes)
    evaluator.add_custom_node(
        result=_has_text(w.curbside_pickup),
        id="walmart_curbside_value_present",
        desc="Walmart curbside pickup availability value is present in answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(w.curbside_sources),
        id="walmart_curbside_sources_present",
        desc="Walmart curbside pickup availability sources are provided",
        parent=node,
        critical=True
    )

    leaf_curb = evaluator.add_leaf(
        id="Walmart_Curbside_Pickup_Availability",
        desc="Answers whether Walmart offers curbside pickup on Christmas Eve in Louisville (expected: yes).",
        parent=node,
        critical=True
    )
    claim_curb = (
        "On Christmas Eve (December 24, 2025) in Louisville, Kentucky, Walmart offers curbside pickup service."
    )
    await evaluator.verify(
        claim=claim_curb,
        node=leaf_curb,
        sources=_urls_or_empty(w.curbside_sources),
        additional_instruction=(
            "Evidence may be general service availability (e.g., Walmart Pickup) or holiday-specific guidance. "
            "Unless explicitly suspended for Christmas Eve, service availability generally implies 'yes'. "
            "Prefer Louisville- or Kentucky-relevant official pages."
        ),
    )


async def verify_target(evaluator: Evaluator, parent, data: ShoppingExtraction) -> None:
    node = evaluator.add_parallel(
        id="Target",
        desc="Target required information for Louisville, KY on Dec 24, 2025.",
        parent=parent,
        critical=True
    )
    t = data.target or StoreInfo()

    # Store closing time (expected 8 p.m.)
    evaluator.add_custom_node(
        result=_has_text(t.closing_time),
        id="target_closing_value_present",
        desc="Target closing time value is present in answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(t.closing_sources),
        id="target_closing_sources_present",
        desc="Target closing time sources are provided",
        parent=node,
        critical=True
    )

    leaf_close = evaluator.add_leaf(
        id="Target_Store_Closing_Time",
        desc="States Target store closing time on Christmas Eve (Dec 24, 2025) in Louisville is 8 p.m.",
        parent=node,
        critical=True
    )
    claim_close = (
        "On Christmas Eve (December 24, 2025) in Louisville, Kentucky, Target stores close at 8 p.m. local time."
    )
    await evaluator.verify(
        claim=claim_close,
        node=leaf_close,
        sources=_urls_or_empty(t.closing_sources),
        additional_instruction=(
            "Verify Christmas Eve hours for Target with Louisville/Kentucky-applicable information. "
            "Allow '8 PM'/'8 p.m.' variants."
        ),
    )

    # Same-day delivery availability (expected: yes)
    evaluator.add_custom_node(
        result=_has_text(t.same_day_delivery),
        id="target_sameday_value_present",
        desc="Target same-day delivery availability value is present in answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(t.same_day_sources),
        id="target_sameday_sources_present",
        desc="Target same-day delivery availability sources are provided",
        parent=node,
        critical=True
    )

    leaf_sameday = evaluator.add_leaf(
        id="Target_Same_Day_Delivery_Availability",
        desc="Answers whether Target offers same-day delivery on Christmas Eve in Louisville (expected: yes).",
        parent=node,
        critical=True
    )
    claim_sameday = (
        "On Christmas Eve (December 24, 2025) in Louisville, Kentucky, Target offers same-day delivery (e.g., via Shipt) and/or Drive Up."
    )
    await evaluator.verify(
        claim=claim_sameday,
        node=leaf_sameday,
        sources=_urls_or_empty(t.same_day_sources),
        additional_instruction=(
            "Evidence may be general service pages (Shipt/Drive Up). Unless explicitly suspended on Christmas Eve, "
            "consider service available. Prefer official pages or Louisville-relevant guidance."
        ),
    )


async def verify_kroger(evaluator: Evaluator, parent, data: ShoppingExtraction) -> None:
    node = evaluator.add_parallel(
        id="Kroger",
        desc="Kroger required information for Louisville, KY on Dec 24, 2025.",
        parent=parent,
        critical=True
    )
    k = data.kroger or StoreInfo()

    # Store closing time (expected 6 p.m.)
    evaluator.add_custom_node(
        result=_has_text(k.closing_time),
        id="kroger_closing_value_present",
        desc="Kroger closing time value is present in answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(k.closing_sources),
        id="kroger_closing_sources_present",
        desc="Kroger closing time sources are provided",
        parent=node,
        critical=True
    )

    leaf_close = evaluator.add_leaf(
        id="Kroger_Store_Closing_Time",
        desc="States Kroger store closing time on Christmas Eve (Dec 24, 2025) in Louisville is 6 p.m.",
        parent=node,
        critical=True
    )
    claim_close = (
        "On Christmas Eve (December 24, 2025) in Louisville, Kentucky, Kroger stores close at 6 p.m. local time."
    )
    await evaluator.verify(
        claim=claim_close,
        node=leaf_close,
        sources=_urls_or_empty(k.closing_sources),
        additional_instruction=(
            "Verify that Christmas Eve hours for Kroger (Louisville/Kentucky context) specify a 6 p.m. closing time. "
            "Allow '6 PM'/'6 p.m.' variants."
        ),
    )

    # Pharmacy closing time (expected: 5 p.m. or earlier)
    evaluator.add_custom_node(
        result=_has_text(k.pharmacy_closing_time),
        id="kroger_rx_value_present",
        desc="Kroger pharmacy closing time value is present in answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(k.pharmacy_closing_sources),
        id="kroger_rx_sources_present",
        desc="Kroger pharmacy closing time sources are provided",
        parent=node,
        critical=True
    )

    leaf_rx = evaluator.add_leaf(
        id="Kroger_Pharmacy_Closing_Time",
        desc="States Kroger pharmacy closing time on Christmas Eve (Dec 24, 2025) in Louisville is 5 p.m. or earlier.",
        parent=node,
        critical=True
    )
    claim_rx = (
        "On Christmas Eve (December 24, 2025) in Louisville, Kentucky, Kroger pharmacies close by 5 p.m. or earlier."
    )
    await evaluator.verify(
        claim=claim_rx,
        node=leaf_rx,
        sources=_urls_or_empty(k.pharmacy_closing_sources),
        additional_instruction=(
            "Confirm pharmacy hours indicate closing at 5 p.m. or earlier in Louisville/Kentucky for Christmas Eve."
        ),
    )

    # Curbside pickup availability (expected: yes)
    evaluator.add_custom_node(
        result=_has_text(k.curbside_pickup),
        id="kroger_curbside_value_present",
        desc="Kroger curbside pickup availability value is present in answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(k.curbside_sources),
        id="kroger_curbside_sources_present",
        desc="Kroger curbside pickup availability sources are provided",
        parent=node,
        critical=True
    )

    leaf_curb = evaluator.add_leaf(
        id="Kroger_Curbside_Pickup_Availability",
        desc="Answers whether Kroger offers curbside pickup on Christmas Eve in Louisville (expected: yes).",
        parent=node,
        critical=True
    )
    claim_curb = (
        "On Christmas Eve (December 24, 2025) in Louisville, Kentucky, Kroger offers curbside pickup service."
    )
    await evaluator.verify(
        claim=claim_curb,
        node=leaf_curb,
        sources=_urls_or_empty(k.curbside_sources),
        additional_instruction=(
            "Evidence can be general service pages (Kroger Pickup). Unless specifically suspended on Christmas Eve, "
            "consider service available. Prefer official/local pages."
        ),
    )


async def verify_aldi(evaluator: Evaluator, parent, data: ShoppingExtraction) -> None:
    node = evaluator.add_parallel(
        id="ALDI",
        desc="ALDI required information for Louisville, KY on Dec 24, 2025.",
        parent=parent,
        critical=True
    )
    a = data.aldi or StoreInfo()

    # Store closing time (expected 4 p.m.)
    evaluator.add_custom_node(
        result=_has_text(a.closing_time),
        id="aldi_closing_value_present",
        desc="ALDI closing time value is present in answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(a.closing_sources),
        id="aldi_closing_sources_present",
        desc="ALDI closing time sources are provided",
        parent=node,
        critical=True
    )

    leaf_close = evaluator.add_leaf(
        id="ALDI_Store_Closing_Time",
        desc="States ALDI store closing time on Christmas Eve (Dec 24, 2025) in Louisville is 4 p.m.",
        parent=node,
        critical=True
    )
    claim_close = (
        "On Christmas Eve (December 24, 2025) in Louisville, Kentucky, ALDI stores close at 4 p.m. local time."
    )
    await evaluator.verify(
        claim=claim_close,
        node=leaf_close,
        sources=_urls_or_empty(a.closing_sources),
        additional_instruction=(
            "Verify ALDI Christmas Eve hours for Louisville/Kentucky. Allow variants '4 PM'/'4 p.m.'."
        ),
    )

    # Curbside pickup availability (expected: yes)
    evaluator.add_custom_node(
        result=_has_text(a.curbside_pickup),
        id="aldi_curbside_value_present",
        desc="ALDI curbside pickup availability value is present in answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(a.curbside_sources),
        id="aldi_curbside_sources_present",
        desc="ALDI curbside pickup availability sources are provided",
        parent=node,
        critical=True
    )

    leaf_curb = evaluator.add_leaf(
        id="ALDI_Curbside_Pickup_Availability",
        desc="Answers whether ALDI offers curbside pickup on Christmas Eve in Louisville (expected: yes).",
        parent=node,
        critical=True
    )
    claim_curb = (
        "On Christmas Eve (December 24, 2025) in Louisville, Kentucky, ALDI offers curbside pickup service (e.g., via Instacart)."
    )
    await evaluator.verify(
        claim=claim_curb,
        node=leaf_curb,
        sources=_urls_or_empty(a.curbside_sources),
        additional_instruction=(
            "General curbside service pages (e.g., Instacart for ALDI) are acceptable unless they explicitly suspend service on Christmas Eve. "
            "Prefer official/local guidance."
        ),
    )


async def verify_publix(evaluator: Evaluator, parent, data: ShoppingExtraction) -> None:
    node = evaluator.add_parallel(
        id="Publix",
        desc="Publix required information for Louisville, KY on Dec 24, 2025.",
        parent=parent,
        critical=True
    )
    p = data.publix or StoreInfo()

    # Store closing time (expected 7 p.m.)
    evaluator.add_custom_node(
        result=_has_text(p.closing_time),
        id="publix_closing_value_present",
        desc="Publix closing time value is present in answer",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(p.closing_sources),
        id="publix_closing_sources_present",
        desc="Publix closing time sources are provided",
        parent=node,
        critical=True
    )

    leaf_close = evaluator.add_leaf(
        id="Publix_Store_Closing_Time",
        desc="States Publix store closing time on Christmas Eve (Dec 24, 2025) in Louisville is 7 p.m.",
        parent=node,
        critical=True
    )
    claim_close = (
        "On Christmas Eve (December 24, 2025) in Louisville, Kentucky, Publix stores close at 7 p.m. local time."
    )
    await evaluator.verify(
        claim=claim_close,
        node=leaf_close,
        sources=_urls_or_empty(p.closing_sources),
        additional_instruction=(
            "Verify Publix Christmas Eve hours applicable to Louisville/Kentucky market. Allow '7 PM'/'7 p.m.' variants."
        ),
    )


# --------------------------------------------------------------------------- #
# Christmas Day closures verification                                         #
# --------------------------------------------------------------------------- #
async def verify_christmas_day_closures(evaluator: Evaluator, parent, data: ShoppingExtraction) -> None:
    node = evaluator.add_parallel(
        id="Christmas_Day_Closures",
        desc="States which of Walmart, Target, Kroger, ALDI, and Publix are closed on Christmas Day (Dec 25, 2025) in Louisville, consistent with all five closed.",
        parent=parent,
        critical=True
    )

    # Helper to add per-store closure verification
    async def _add_store_closure(store_id: str, store_label: str, store: StoreInfo):
        container = evaluator.add_parallel(
            id=f"{store_id}_closure_group",
            desc=f"{store_label} closure status on Christmas Day (Dec 25, 2025) in Louisville",
            parent=node,
            critical=True
        )
        evaluator.add_custom_node(
            result=_has_text(store.christmas_closed),
            id=f"{store_id}_closure_value_present",
            desc=f"{store_label} Christmas Day closure value is present in answer",
            parent=container,
            critical=True
        )
        evaluator.add_custom_node(
            result=_nonempty(store.christmas_closed_sources),
            id=f"{store_id}_closure_sources_present",
            desc=f"{store_label} Christmas Day closure sources are provided",
            parent=container,
            critical=True
        )
        leaf = evaluator.add_leaf(
            id=f"{store_id}_closed_on_christmas",
            desc=f"{store_label} is closed on Christmas Day (Dec 25, 2025) in Louisville.",
            parent=container,
            critical=True
        )
        claim = f"On Christmas Day (December 25, 2025), {store_label} stores in Louisville, Kentucky are closed."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=_urls_or_empty(store.christmas_closed_sources),
            additional_instruction=(
                "Verify the page(s) state Christmas Day closure for this brand. "
                "National policy pages are acceptable if they apply to Louisville/KY. "
                "Allow minor wording variations."
            ),
        )

    await _add_store_closure("walmart", "Walmart", data.walmart or StoreInfo())
    await _add_store_closure("target", "Target", data.target or StoreInfo())
    await _add_store_closure("kroger", "Kroger", data.kroger or StoreInfo())
    await _add_store_closure("aldi", "ALDI", data.aldi or StoreInfo())
    await _add_store_closure("publix", "Publix", data.publix or StoreInfo())


# --------------------------------------------------------------------------- #
# Reference URLs sufficiency (coverage)                                       #
# --------------------------------------------------------------------------- #
def compute_reference_coverage(ex: ShoppingExtraction) -> Dict[str, Any]:
    w = ex.walmart or StoreInfo()
    t = ex.target or StoreInfo()
    k = ex.kroger or StoreInfo()
    a = ex.aldi or StoreInfo()
    p = ex.publix or StoreInfo()

    checks = {
        # Walmart
        "walmart_closing_sources": _nonempty(w.closing_sources),
        "walmart_rx_sources": _nonempty(w.pharmacy_sources),
        "walmart_curbside_sources": _nonempty(w.curbside_sources),

        # Target
        "target_closing_sources": _nonempty(t.closing_sources),
        "target_sameday_sources": _nonempty(t.same_day_sources),

        # Kroger
        "kroger_closing_sources": _nonempty(k.closing_sources),
        "kroger_rx_sources": _nonempty(k.pharmacy_closing_sources),
        "kroger_curbside_sources": _nonempty(k.curbside_sources),

        # ALDI
        "aldi_closing_sources": _nonempty(a.closing_sources),
        "aldi_curbside_sources": _nonempty(a.curbside_sources),

        # Publix
        "publix_closing_sources": _nonempty(p.closing_sources),

        # Christmas Day closures for all five
        "walmart_christmas_closed_sources": _nonempty(w.christmas_closed_sources),
        "target_christmas_closed_sources": _nonempty(t.christmas_closed_sources),
        "kroger_christmas_closed_sources": _nonempty(k.christmas_closed_sources),
        "aldi_christmas_closed_sources": _nonempty(a.christmas_closed_sources),
        "publix_christmas_closed_sources": _nonempty(p.christmas_closed_sources),
    }
    return {
        "all_present": all(checks.values()),
        "detail": checks
    }


async def add_reference_urls_node(evaluator: Evaluator, parent, ex: ShoppingExtraction) -> None:
    coverage = compute_reference_coverage(ex)
    evaluator.add_custom_info(
        info=coverage,
        info_type="coverage_detail",
        info_name="reference_urls_coverage"
    )
    evaluator.add_custom_node(
        result=coverage["all_present"],
        id="Reference_URLs",
        desc="Provides sufficient reference URLs to verify all required claims and specific to Louisville, KY / specified dates.",
        parent=parent,
        critical=True
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

    # Extract structured info from the answer
    extraction: ShoppingExtraction = await evaluator.extract(
        prompt=prompt_extract_shopping_info(),
        template_class=ShoppingExtraction,
        extraction_name="shopping_info_extraction",
    )

    # Add ground truth expectations for transparency (used for human-readable context, not enforced directly)
    evaluator.add_ground_truth({
        "expected": {
            "walmart_close": EXPECTED["walmart_close"],
            "target_close": EXPECTED["target_close"],
            "kroger_close": EXPECTED["kroger_close"],
            "aldi_close": EXPECTED["aldi_close"],
            "publix_close": EXPECTED["publix_close"],
            "walmart_pharmacy_hours": EXPECTED["walmart_rx_hours"],
            "kroger_pharmacy_close_by": EXPECTED["kroger_rx_close_by"],
            "services_expected_yes": ["Walmart curbside", "Target same-day delivery", "Kroger curbside", "ALDI curbside"],
            "christmas_day_closed": ["Walmart", "Target", "Kroger", "ALDI", "Publix"]
        }
    }, gt_type="expected_constraints")

    # Build evaluation tree according to rubric
    complete_info = evaluator.add_parallel(
        id="Complete_Shopping_Information",
        desc="Provide the requested Christmas Eve (Dec 24, 2025) shopping guide information for Louisville, KY for the five named stores, plus Christmas Day (Dec 25, 2025) closures and reference URLs.",
        parent=root,
        critical=True
    )

    # Christmas Eve info by store
    eve_by_store = evaluator.add_parallel(
        id="Christmas_Eve_Info_By_Store",
        desc="Provide required Christmas Eve information for each of the five named stores in Louisville, KY.",
        parent=complete_info,
        critical=True
    )

    # Per-store verifications
    await verify_walmart(evaluator, eve_by_store, extraction)
    await verify_target(evaluator, eve_by_store, extraction)
    await verify_kroger(evaluator, eve_by_store, extraction)
    await verify_aldi(evaluator, eve_by_store, extraction)
    await verify_publix(evaluator, eve_by_store, extraction)

    # Christmas Day closures across all five stores
    await verify_christmas_day_closures(evaluator, complete_info, extraction)

    # Reference URLs sufficiency/coverage
    await add_reference_urls_node(evaluator, complete_info, extraction)

    # Return final summary with verification tree
    return evaluator.get_summary()