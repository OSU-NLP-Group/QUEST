import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "two_largest_2025_bank_mergers"
TASK_DESCRIPTION = (
    "Identify the two largest U.S. bank merger transactions from 2025, measured by transaction value in dollars, "
    "that have completed (closed) by March 18, 2026. For each of these two mergers, provide the following information: "
    "(1) Acquiring Institution Name; (2) Target Institution Name; (3) Transaction Value (billions of dollars); "
    "(4) Announcement Date (month and year); (5) Completion Date (month and year); "
    "(6) Combined Total Assets (billions of dollars); "
    "(7) Transaction Structure (confirm all-stock merger); "
    "(8) Reference URLs supporting transaction value, completion date, and combined assets. "
    "Present the largest merger first, followed by the second-largest merger."
)

CUTOFF_DATE_STR = "March 18, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MergerItem(BaseModel):
    acquiring_institution: Optional[str] = None
    target_institution: Optional[str] = None
    transaction_value_billion: Optional[str] = None
    announcement_date: Optional[str] = None  # Expect "Month YYYY" format per answer
    completion_date: Optional[str] = None    # Expect "Month YYYY" (or full date), but month+year acceptable
    combined_total_assets_billion: Optional[str] = None
    transaction_structure: Optional[str] = None  # Expect text like "all-stock", "stock-for-stock", etc.
    reference_urls: List[str] = Field(default_factory=list)


class MergersExtraction(BaseModel):
    mergers: List[MergerItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_mergers() -> str:
    return """
Extract information about U.S. bank mergers listed in the answer. The task expects exactly two mergers: the largest and the second-largest (by transaction value) among those announced in 2025 and completed by March 18, 2026, presented in descending order (largest first). Extract all merger entries mentioned in the answer (preserve the original order). For each merger, return:
- acquiring_institution: the acquiring bank's name (string)
- target_institution: the acquired bank's name (string)
- transaction_value_billion: the transaction value expressed in billions of U.S. dollars, exactly as written in the answer (string; do not compute or convert)
- announcement_date: the month and year of the public announcement exactly as stated (e.g., "January 2025"; string)
- completion_date: the month and year (or full date) when the merger completed/closed exactly as stated (e.g., "February 2026"; string)
- combined_total_assets_billion: combined total assets of the merged entity, in billions of U.S. dollars, exactly as stated (string)
- transaction_structure: the text describing the structure (e.g., "all-stock"; string; return null if missing)
- reference_urls: an array of all URLs cited in the answer that pertain to this merger; include official press releases, SEC/FDIC/OCC/FRB/regulators, or credible financial media; if none are given, return an empty array

Return a JSON object with a field:
- mergers: an array of merger objects in the same order they appear in the answer.

Rules:
- Extract strictly from the provided answer text; do not invent or infer fields.
- If a field is missing in the answer, set it to null (or [] for reference_urls).
- Do not normalize or convert numeric values; keep the original strings as written.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ensure_two_mergers(extraction: MergersExtraction) -> List[MergerItem]:
    items = list(extraction.mergers) if extraction and extraction.mergers else []
    if len(items) >= 2:
        return items[:2]
    # Pad with empty items to always have length 2
    while len(items) < 2:
        items.append(MergerItem())
    return items[:2]


def has_any_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_largest_merger_checks(evaluator: Evaluator, parent, item: MergerItem) -> None:
    # 1) Largest_Merger_Identification
    grp_ident = evaluator.add_parallel(
        id="Largest_Merger_Identification",
        desc="The largest merger is correctly identified with both acquiring and target names, and is verifiably the #1 ranked 2025 U.S. bank merger by transaction value completed by March 18, 2026",
        parent=parent,
        critical=True
    )
    # Existence: acquiring and target provided
    evaluator.add_custom_node(
        result=bool(item.acquiring_institution and item.target_institution),
        id="largest_id_names_provided",
        desc="Largest merger: acquiring and target institution names are provided",
        parent=grp_ident,
        critical=True
    )
    # Ranking verification (source-backed)
    node_rank = evaluator.add_leaf(
        id="largest_id_is_rank1_2025",
        desc="Largest merger is verified as the #1 2025 U.S. bank merger by transaction value that completed by March 18, 2026",
        parent=grp_ident,
        critical=True
    )
    claim_rank = (
        f"The merger between '{_safe(item.acquiring_institution)}' and '{_safe(item.target_institution)}' "
        f"was the largest U.S. bank merger announced in 2025 that had completed by {CUTOFF_DATE_STR}."
    )
    await evaluator.verify(
        claim=claim_rank,
        node=node_rank,
        sources=item.reference_urls,
        additional_instruction="Confirm ranking strictly by transaction value among U.S. bank mergers announced in 2025 that had closed by the cutoff date. Accept credible sources (official releases, regulators, or major financial media)."
    )

    # 2) Largest_Merger_Transaction_Value
    grp_value = evaluator.add_parallel(
        id="Largest_Merger_Transaction_Value",
        desc="The transaction value for the largest merger is accurately reported in billions of dollars and matches authoritative sources",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item.transaction_value_billion),
        id="largest_value_provided",
        desc="Largest merger: transaction value provided",
        parent=grp_value,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_urls(item.reference_urls),
        id="largest_value_sources_present",
        desc="Largest merger: at least one reference URL is provided for transaction value verification",
        parent=grp_value,
        critical=True
    )
    node_value = evaluator.add_leaf(
        id="largest_value_supported",
        desc="Largest merger: transaction value matches authoritative sources",
        parent=grp_value,
        critical=True
    )
    claim_value = (
        f"The transaction value for the acquisition of '{_safe(item.target_institution)}' by "
        f"'{_safe(item.acquiring_institution)}' is {_safe(item.transaction_value_billion)} (billion U.S. dollars)."
    )
    await evaluator.verify(
        claim=claim_value,
        node=node_value,
        sources=item.reference_urls,
        additional_instruction="Minor rounding differences are acceptable (e.g., 12.7 vs 12.65). Ensure the unit is billions of U.S. dollars."
    )

    # 3) Largest_Merger_Timeline
    grp_time = evaluator.add_parallel(
        id="Largest_Merger_Timeline",
        desc=f"Announcement date (month/year) and completion date (month/year on or before {CUTOFF_DATE_STR}) are provided and documented for the largest merger",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item.announcement_date),
        id="largest_timeline_announce_provided",
        desc="Largest merger: announcement date provided",
        parent=grp_time,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item.completion_date),
        id="largest_timeline_complete_provided",
        desc="Largest merger: completion date provided",
        parent=grp_time,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_urls(item.reference_urls),
        id="largest_timeline_sources_present",
        desc="Largest merger: at least one reference URL is provided for timeline verification",
        parent=grp_time,
        critical=True
    )
    node_announce = evaluator.add_leaf(
        id="largest_timeline_announce_supported",
        desc="Largest merger: announcement month/year supported by sources",
        parent=grp_time,
        critical=True
    )
    claim_announce = (
        f"The merger between '{_safe(item.acquiring_institution)}' and '{_safe(item.target_institution)}' "
        f"was publicly announced in {_safe(item.announcement_date)}."
    )
    await evaluator.verify(
        claim=claim_announce,
        node=node_announce,
        sources=item.reference_urls,
        additional_instruction="Confirm announcement month and year from the provided credible sources."
    )
    node_complete = evaluator.add_leaf(
        id="largest_timeline_complete_supported_by_cutoff",
        desc=f"Largest merger: completion date supported and on or before {CUTOFF_DATE_STR}",
        parent=grp_time,
        critical=True
    )
    claim_complete = (
        f"The merger between '{_safe(item.acquiring_institution)}' and '{_safe(item.target_institution)}' "
        f"closed (completed) in {_safe(item.completion_date)}, which is on or before {CUTOFF_DATE_STR}."
    )
    await evaluator.verify(
        claim=claim_complete,
        node=node_complete,
        sources=item.reference_urls,
        additional_instruction="Ensure the source explicitly states that the merger closed/completed, and that the completion date is not later than the cutoff date."
    )

    # 4) Largest_Merger_Combined_Assets
    grp_assets = evaluator.add_parallel(
        id="Largest_Merger_Combined_Assets",
        desc="Combined total assets for the largest merger are reported (in billions) and documented as exceeding $100 billion",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item.combined_total_assets_billion),
        id="largest_assets_provided",
        desc="Largest merger: combined total assets value provided",
        parent=grp_assets,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_urls(item.reference_urls),
        id="largest_assets_sources_present",
        desc="Largest merger: at least one reference URL is provided for assets verification",
        parent=grp_assets,
        critical=True
    )
    node_assets_val = evaluator.add_leaf(
        id="largest_assets_value_supported",
        desc="Largest merger: combined total assets value supported by sources",
        parent=grp_assets,
        critical=True
    )
    claim_assets_val = (
        f"The combined total assets of the merged entity for '{_safe(item.acquiring_institution)}' and "
        f"'{_safe(item.target_institution)}' were approximately {_safe(item.combined_total_assets_billion)} (billion U.S. dollars)."
    )
    await evaluator.verify(
        claim=claim_assets_val,
        node=node_assets_val,
        sources=item.reference_urls,
        additional_instruction="Minor rounding/language like 'about' or 'approximately' is acceptable."
    )
    node_assets_100 = evaluator.add_leaf(
        id="largest_assets_exceed_100b",
        desc="Largest merger: combined total assets exceed $100 billion",
        parent=grp_assets,
        critical=True
    )
    claim_assets_100 = (
        f"The combined total assets of the merged entity for '{_safe(item.acquiring_institution)}' and "
        f"'{_safe(item.target_institution)}' were at least 100 billion U.S. dollars."
    )
    await evaluator.verify(
        claim=claim_assets_100,
        node=node_assets_100,
        sources=item.reference_urls,
        additional_instruction="Confirm the combined assets meet or exceed $100B per the provided sources."
    )

    # 5) Largest_Merger_Transaction_Structure
    grp_struct = evaluator.add_parallel(
        id="Largest_Merger_Transaction_Structure",
        desc="Largest merger is documented and confirmed as an all-stock transaction",
        parent=parent,
        critical=True
    )
    # Local existence check
    evaluator.add_custom_node(
        result=bool(item.transaction_structure and ("stock" in item.transaction_structure.lower())),
        id="largest_structure_provided",
        desc="Largest merger: transaction structure provided indicating stock-based structure",
        parent=grp_struct,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_urls(item.reference_urls),
        id="largest_structure_sources_present",
        desc="Largest merger: at least one reference URL is provided for structure verification",
        parent=grp_struct,
        critical=True
    )
    node_struct = evaluator.add_leaf(
        id="largest_structure_all_stock_confirmed",
        desc="Largest merger: all-stock transaction confirmed by sources",
        parent=grp_struct,
        critical=True
    )
    claim_struct = (
        f"The merger between '{_safe(item.acquiring_institution)}' and '{_safe(item.target_institution)}' "
        f"was structured as an all-stock transaction."
    )
    await evaluator.verify(
        claim=claim_struct,
        node=node_struct,
        sources=item.reference_urls,
        additional_instruction="Look for phrases such as 'all-stock', 'stock-for-stock', or 'all shares' indicating no cash consideration."
    )


async def build_second_merger_checks(evaluator: Evaluator, parent, item: MergerItem) -> None:
    # 6) Second_Largest_Merger_Identification
    grp_ident = evaluator.add_parallel(
        id="Second_Largest_Merger_Identification",
        desc="The second-largest merger is correctly identified with both acquiring and target names, and is verifiably the #2 ranked 2025 U.S. bank merger by transaction value that completed by March 18, 2026",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item.acquiring_institution and item.target_institution),
        id="second_id_names_provided",
        desc="Second-largest merger: acquiring and target institution names are provided",
        parent=grp_ident,
        critical=True
    )
    node_rank = evaluator.add_leaf(
        id="second_id_is_rank2_2025",
        desc="Second-largest merger is verified as the #2 2025 U.S. bank merger by transaction value that completed by March 18, 2026",
        parent=grp_ident,
        critical=True
    )
    claim_rank = (
        f"The merger between '{_safe(item.acquiring_institution)}' and '{_safe(item.target_institution)}' "
        f"was the second-largest U.S. bank merger announced in 2025 that had completed by {CUTOFF_DATE_STR}."
    )
    await evaluator.verify(
        claim=claim_rank,
        node=node_rank,
        sources=item.reference_urls,
        additional_instruction="Confirm ranking strictly by transaction value among U.S. bank mergers announced in 2025 that had closed by the cutoff date. Accept credible sources (official releases, regulators, or major financial media)."
    )

    # 7) Second_Largest_Merger_Transaction_Value
    grp_value = evaluator.add_parallel(
        id="Second_Largest_Merger_Transaction_Value",
        desc="The transaction value for the second-largest merger is accurately reported in billions of dollars and matches authoritative sources",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item.transaction_value_billion),
        id="second_value_provided",
        desc="Second-largest merger: transaction value provided",
        parent=grp_value,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_urls(item.reference_urls),
        id="second_value_sources_present",
        desc="Second-largest merger: at least one reference URL is provided for transaction value verification",
        parent=grp_value,
        critical=True
    )
    node_value = evaluator.add_leaf(
        id="second_value_supported",
        desc="Second-largest merger: transaction value matches authoritative sources",
        parent=grp_value,
        critical=True
    )
    claim_value = (
        f"The transaction value for the acquisition of '{_safe(item.target_institution)}' by "
        f"'{_safe(item.acquiring_institution)}' is {_safe(item.transaction_value_billion)} (billion U.S. dollars)."
    )
    await evaluator.verify(
        claim=claim_value,
        node=node_value,
        sources=item.reference_urls,
        additional_instruction="Minor rounding differences are acceptable. Ensure the unit is billions of U.S. dollars."
    )

    # 8) Second_Largest_Merger_Timeline
    grp_time = evaluator.add_parallel(
        id="Second_Largest_Merger_Timeline",
        desc=f"Announcement date (month/year) and completion date (month/year on or before {CUTOFF_DATE_STR}) are provided and documented for the second-largest merger",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item.announcement_date),
        id="second_timeline_announce_provided",
        desc="Second-largest merger: announcement date provided",
        parent=grp_time,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item.completion_date),
        id="second_timeline_complete_provided",
        desc="Second-largest merger: completion date provided",
        parent=grp_time,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_urls(item.reference_urls),
        id="second_timeline_sources_present",
        desc="Second-largest merger: at least one reference URL is provided for timeline verification",
        parent=grp_time,
        critical=True
    )
    node_announce = evaluator.add_leaf(
        id="second_timeline_announce_supported",
        desc="Second-largest merger: announcement month/year supported by sources",
        parent=grp_time,
        critical=True
    )
    claim_announce = (
        f"The merger between '{_safe(item.acquiring_institution)}' and '{_safe(item.target_institution)}' "
        f"was publicly announced in {_safe(item.announcement_date)}."
    )
    await evaluator.verify(
        claim=claim_announce,
        node=node_announce,
        sources=item.reference_urls,
        additional_instruction="Confirm announcement month and year from the provided credible sources."
    )
    node_complete = evaluator.add_leaf(
        id="second_timeline_complete_supported_by_cutoff",
        desc=f"Second-largest merger: completion date supported and on or before {CUTOFF_DATE_STR}",
        parent=grp_time,
        critical=True
    )
    claim_complete = (
        f"The merger between '{_safe(item.acquiring_institution)}' and '{_safe(item.target_institution)}' "
        f"closed (completed) in {_safe(item.completion_date)}, which is on or before {CUTOFF_DATE_STR}."
    )
    await evaluator.verify(
        claim=claim_complete,
        node=node_complete,
        sources=item.reference_urls,
        additional_instruction="Ensure the source explicitly states that the merger closed/completed, and that the completion date is not later than the cutoff date."
    )

    # 9) Second_Largest_Merger_Combined_Assets
    grp_assets = evaluator.add_parallel(
        id="Second_Largest_Merger_Combined_Assets",
        desc="Combined total assets for the second-largest merger are reported (in billions) and documented as exceeding $100 billion",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item.combined_total_assets_billion),
        id="second_assets_provided",
        desc="Second-largest merger: combined total assets value provided",
        parent=grp_assets,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_urls(item.reference_urls),
        id="second_assets_sources_present",
        desc="Second-largest merger: at least one reference URL is provided for assets verification",
        parent=grp_assets,
        critical=True
    )
    node_assets_val = evaluator.add_leaf(
        id="second_assets_value_supported",
        desc="Second-largest merger: combined total assets value supported by sources",
        parent=grp_assets,
        critical=True
    )
    claim_assets_val = (
        f"The combined total assets of the merged entity for '{_safe(item.acquiring_institution)}' and "
        f"'{_safe(item.target_institution)}' were approximately {_safe(item.combined_total_assets_billion)} (billion U.S. dollars)."
    )
    await evaluator.verify(
        claim=claim_assets_val,
        node=node_assets_val,
        sources=item.reference_urls,
        additional_instruction="Minor rounding/language like 'about' or 'approximately' is acceptable."
    )
    node_assets_100 = evaluator.add_leaf(
        id="second_assets_exceed_100b",
        desc="Second-largest merger: combined total assets exceed $100 billion",
        parent=grp_assets,
        critical=True
    )
    claim_assets_100 = (
        f"The combined total assets of the merged entity for '{_safe(item.acquiring_institution)}' and "
        f"'{_safe(item.target_institution)}' were at least 100 billion U.S. dollars."
    )
    await evaluator.verify(
        claim=claim_assets_100,
        node=node_assets_100,
        sources=item.reference_urls,
        additional_instruction="Confirm the combined assets meet or exceed $100B per the provided sources."
    )

    # 10) Second_Largest_Merger_Transaction_Structure
    grp_struct = evaluator.add_parallel(
        id="Second_Largest_Merger_Transaction_Structure",
        desc="Second-largest merger is documented and confirmed as an all-stock transaction",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(item.transaction_structure and ("stock" in item.transaction_structure.lower())),
        id="second_structure_provided",
        desc="Second-largest merger: transaction structure provided indicating stock-based structure",
        parent=grp_struct,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_urls(item.reference_urls),
        id="second_structure_sources_present",
        desc="Second-largest merger: at least one reference URL is provided for structure verification",
        parent=grp_struct,
        critical=True
    )
    node_struct = evaluator.add_leaf(
        id="second_structure_all_stock_confirmed",
        desc="Second-largest merger: all-stock transaction confirmed by sources",
        parent=grp_struct,
        critical=True
    )
    claim_struct = (
        f"The merger between '{_safe(item.acquiring_institution)}' and '{_safe(item.target_institution)}' "
        f"was structured as an all-stock transaction."
    )
    await evaluator.verify(
        claim=claim_struct,
        node=node_struct,
        sources=item.reference_urls,
        additional_instruction="Look for phrases such as 'all-stock', 'stock-for-stock', or 'all shares' indicating no cash consideration."
    )


async def build_reference_documentation_checks(evaluator: Evaluator, parent, largest: MergerItem, second: MergerItem) -> None:
    grp_ref = evaluator.add_parallel(
        id="Reference_Documentation",
        desc="For each merger, at least one credible reference URL supports the transaction value, completion date, and combined assets",
        parent=parent,
        critical=True
    )

    # Presence checks
    evaluator.add_custom_node(
        result=has_any_urls(largest.reference_urls),
        id="refs_largest_has_url",
        desc="Largest merger: at least one reference URL provided",
        parent=grp_ref,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_any_urls(second.reference_urls),
        id="refs_second_has_url",
        desc="Second-largest merger: at least one reference URL provided",
        parent=grp_ref,
        critical=True
    )

    # Largest: value, completion, assets supported by refs
    node_l_val = evaluator.add_leaf(
        id="refs_largest_value_supported",
        desc="Largest merger: transaction value is supported by provided reference URLs",
        parent=grp_ref,
        critical=True
    )
    claim_l_val = (
        f"The transaction value for the acquisition of '{_safe(largest.target_institution)}' by "
        f"'{_safe(largest.acquiring_institution)}' is {_safe(largest.transaction_value_billion)} (billion U.S. dollars)."
    )
    await evaluator.verify(
        claim=claim_l_val,
        node=node_l_val,
        sources=largest.reference_urls,
        additional_instruction="Use the provided references to confirm the stated transaction value. Minor rounding differences are acceptable."
    )

    node_l_comp = evaluator.add_leaf(
        id="refs_largest_completion_supported",
        desc=f"Largest merger: completion date is supported by provided reference URLs (on or before {CUTOFF_DATE_STR})",
        parent=grp_ref,
        critical=True
    )
    claim_l_comp = (
        f"The merger between '{_safe(largest.acquiring_institution)}' and '{_safe(largest.target_institution)}' "
        f"closed (completed) in {_safe(largest.completion_date)}, which is on or before {CUTOFF_DATE_STR}."
    )
    await evaluator.verify(
        claim=claim_l_comp,
        node=node_l_comp,
        sources=largest.reference_urls,
        additional_instruction="Confirm the close/completion date and that it is not later than the cutoff date."
    )

    node_l_assets = evaluator.add_leaf(
        id="refs_largest_assets_supported",
        desc="Largest merger: combined total assets are supported by provided reference URLs",
        parent=grp_ref,
        critical=True
    )
    claim_l_assets = (
        f"The combined total assets of the merged entity for '{_safe(largest.acquiring_institution)}' and "
        f"'{_safe(largest.target_institution)}' were approximately {_safe(largest.combined_total_assets_billion)} (billion U.S. dollars)."
    )
    await evaluator.verify(
        claim=claim_l_assets,
        node=node_l_assets,
        sources=largest.reference_urls,
        additional_instruction="Minor rounding/language like 'about' or 'approximately' is acceptable."
    )

    # Second: value, completion, assets supported by refs
    node_s_val = evaluator.add_leaf(
        id="refs_second_value_supported",
        desc="Second-largest merger: transaction value is supported by provided reference URLs",
        parent=grp_ref,
        critical=True
    )
    claim_s_val = (
        f"The transaction value for the acquisition of '{_safe(second.target_institution)}' by "
        f"'{_safe(second.acquiring_institution)}' is {_safe(second.transaction_value_billion)} (billion U.S. dollars)."
    )
    await evaluator.verify(
        claim=claim_s_val,
        node=node_s_val,
        sources=second.reference_urls,
        additional_instruction="Use the provided references to confirm the stated transaction value. Minor rounding differences are acceptable."
    )

    node_s_comp = evaluator.add_leaf(
        id="refs_second_completion_supported",
        desc=f"Second-largest merger: completion date is supported by provided reference URLs (on or before {CUTOFF_DATE_STR})",
        parent=grp_ref,
        critical=True
    )
    claim_s_comp = (
        f"The merger between '{_safe(second.acquiring_institution)}' and '{_safe(second.target_institution)}' "
        f"closed (completed) in {_safe(second.completion_date)}, which is on or before {CUTOFF_DATE_STR}."
    )
    await evaluator.verify(
        claim=claim_s_comp,
        node=node_s_comp,
        sources=second.reference_urls,
        additional_instruction="Confirm the close/completion date and that it is not later than the cutoff date."
    )

    node_s_assets = evaluator.add_leaf(
        id="refs_second_assets_supported",
        desc="Second-largest merger: combined total assets are supported by provided reference URLs",
        parent=grp_ref,
        critical=True
    )
    claim_s_assets = (
        f"The combined total assets of the merged entity for '{_safe(second.acquiring_institution)}' and "
        f"'{_safe(second.target_institution)}' were approximately {_safe(second.combined_total_assets_billion)} (billion U.S. dollars)."
    )
    await evaluator.verify(
        claim=claim_s_assets,
        node=node_s_assets,
        sources=second.reference_urls,
        additional_instruction="Minor rounding/language like 'about' or 'approximately' is acceptable."
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
    Evaluate an answer for the 'two_largest_2025_bank_mergers' task using the Mind2Web2 framework.
    """
    # Initialize evaluator and root
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

    # Add an explicit critical top-level node to mirror the rubric root
    rubric_root = evaluator.add_parallel(
        id="Two_Largest_2025_Bank_Mergers_Task",
        desc="Verify that the two identified bank mergers are the two largest U.S. bank merger transactions from 2025 (by value) that completed by March 18, 2026, with all required information provided",
        parent=root,
        critical=True
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_mergers(),
        template_class=MergersExtraction,
        extraction_name="mergers_extraction"
    )
    m0, m1 = ensure_two_mergers(extracted)

    # Build verification nodes following the rubric
    await build_largest_merger_checks(evaluator, rubric_root, m0)
    await build_second_merger_checks(evaluator, rubric_root, m1)
    await build_reference_documentation_checks(evaluator, rubric_root, m0, m1)

    # Return the evaluation summary
    return evaluator.get_summary()