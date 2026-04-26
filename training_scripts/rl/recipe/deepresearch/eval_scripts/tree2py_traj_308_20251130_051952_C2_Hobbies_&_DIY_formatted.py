import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "thanksgiving_black_friday_craft_columbus"
TASK_DESCRIPTION = (
    "You're planning a DIY Thanksgiving craft project in Columbus, Ohio, but you realize on Thanksgiving evening that "
    "you forgot to buy supplies. Since all major craft stores are closed on Thanksgiving Day, you plan to go shopping "
    "first thing on Black Friday morning (November 29, 2025). Between the two major craft store chains that have "
    "locations in Columbus, OH—Michaels and Hobby Lobby—which one opens earliest on Black Friday 2025, what time does "
    "it open, and what is the maximum discount percentage being offered during their Black Friday sale?"
)


class StoreBFInfo(BaseModel):
    name: Optional[str] = None
    opening_time: Optional[str] = None
    max_discount_percentage: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BFChainsExtraction(BaseModel):
    earliest_chain: Optional[str] = None
    earliest_opening_time: Optional[str] = None
    earliest_max_discount_percentage: Optional[str] = None
    earliest_chain_sources: List[str] = Field(default_factory=list)

    michaels: Optional[StoreBFInfo] = None
    hobby_lobby: Optional[StoreBFInfo] = None


def prompt_extract_bf_chains() -> str:
    return (
        "Extract structured information from the answer about Black Friday 2025 (Nov 29, 2025) store hours and sales "
        "for Michaels and Hobby Lobby in Columbus, Ohio.\n"
        "Return a JSON matching this schema:\n"
        "1) earliest_chain: Which chain the answer claims opens earliest (exactly 'Michaels' or 'Hobby Lobby'). If not explicitly stated, return null.\n"
        "2) earliest_opening_time: The opening time string for the earliest chain (e.g., '6 AM', '8:00 a.m.'). If not stated, return null.\n"
        "3) earliest_max_discount_percentage: The maximum discount percentage the earliest-opening chain advertises for Black Friday (e.g., '70% off', 'up to 60%'). If not stated, return null.\n"
        "4) earliest_chain_sources: An array of URLs the answer cites that support the earliest chain’s opening time and/or sale. Only include explicit URLs; if none, return an empty list.\n"
        "5) michaels: An object with fields:\n"
        "   - name: 'Michaels' if present, else null\n"
        "   - opening_time: The Michaels Black Friday 2025 opening time for Columbus, OH as written in the answer\n"
        "   - max_discount_percentage: The largest percent-off Michaels advertises for Black Friday per the answer\n"
        "   - sources: Array of URLs explicitly cited for Michaels (store hours page, ad, flyer, store locator, etc.). If none, return []\n"
        "6) hobby_lobby: Same object format as 'michaels' but for Hobby Lobby.\n\n"
        "Rules:\n"
        "- Extract only what is explicitly present in the answer. Do not infer or invent.\n"
        "- For URLs, include only actual URLs (plain or in markdown). If missing protocol, prepend http://.\n"
        "- Keep times and percentages as strings (e.g., '6 AM', 'up to 70%').\n"
        "- If the answer gives ranges or notes like 'varies by location', extract the exact phrasing given (e.g., 'opens at 6–8 AM', 'varies by store').\n"
        "- If a field is not present, return null (or [] for lists)."
    )


def _normalize_chain_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower()
    if "michaels" in n:
        return "Michaels"
    if "hobby lobby" in n:
        return "Hobby Lobby"
    return name.strip()


def _select_earliest_chain_sources(extracted: BFChainsExtraction) -> List[str]:
    # Prefer earliest_chain_sources; if empty, fall back to the corresponding chain's sources
    chain_norm = _normalize_chain_name(extracted.earliest_chain)
    if extracted.earliest_chain_sources:
        return extracted.earliest_chain_sources
    if chain_norm == "Michaels" and extracted.michaels:
        return extracted.michaels.sources
    if chain_norm == "Hobby Lobby" and extracted.hobby_lobby:
        return extracted.hobby_lobby.sources
    return []


def _get_chain_opening_time_for_claim(extracted: BFChainsExtraction) -> str:
    # Prefer earliest_opening_time; if missing, fall back to chain-specific
    chain_norm = _normalize_chain_name(extracted.earliest_chain)
    if extracted.earliest_opening_time:
        return extracted.earliest_opening_time
    if chain_norm == "Michaels" and extracted.michaels and extracted.michaels.opening_time:
        return extracted.michaels.opening_time
    if chain_norm == "Hobby Lobby" and extracted.hobby_lobby and extracted.hobby_lobby.opening_time:
        return extracted.hobby_lobby.opening_time
    return ""


def _get_chain_discount_for_claim(extracted: BFChainsExtraction) -> str:
    # Prefer earliest_max_discount_percentage; if missing, fall back to chain-specific
    chain_norm = _normalize_chain_name(extracted.earliest_chain)
    if extracted.earliest_max_discount_percentage:
        return extracted.earliest_max_discount_percentage
    if chain_norm == "Michaels" and extracted.michaels and extracted.michaels.max_discount_percentage:
        return extracted.michaels.max_discount_percentage
    if chain_norm == "Hobby Lobby" and extracted.hobby_lobby and extracted.hobby_lobby.max_discount_percentage:
        return extracted.hobby_lobby.max_discount_percentage
    return ""


async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    # Initialize evaluator with sequential aggregation as per rubric
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Identify which of the two chains (Michaels vs Hobby Lobby) opens earliest on Black Friday 2025 in "
            "Columbus, OH; provide that opening time; and provide the maximum discount percentage offered by that "
            "earliest-opening chain."
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_bf_chains(),
        template_class=BFChainsExtraction,
        extraction_name="bf_chains_extraction",
    )

    # Record custom info for transparency
    evaluator.add_custom_info(
        {
            "earliest_chain": extracted.earliest_chain,
            "earliest_opening_time": extracted.earliest_opening_time,
            "earliest_max_discount_percentage": extracted.earliest_max_discount_percentage,
            "earliest_chain_sources": extracted.earliest_chain_sources,
            "michaels": (extracted.michaels.dict() if extracted.michaels else None),
            "hobby_lobby": (extracted.hobby_lobby.dict() if extracted.hobby_lobby else None),
        },
        info_type="extracted_overview",
    )

    # Build claims and sources for verification
    m_time = extracted.michaels.opening_time if extracted.michaels else None
    h_time = extracted.hobby_lobby.opening_time if extracted.hobby_lobby else None

    # 1) Earliest Chain Identification (critical leaf)
    earliest_chain_leaf = evaluator.add_leaf(
        id="Earliest_Chain_Identification",
        desc="Correctly identify which chain (between Michaels and Hobby Lobby) opens earliest on Black Friday 2025.",
        parent=root,
        critical=True,
    )
    earliest_chain_norm = _normalize_chain_name(extracted.earliest_chain) or ""

    earliest_chain_claim = (
        f"Between Michaels and Hobby Lobby in Columbus, OH on Black Friday (Nov 29, 2025), the earlier opening chain "
        f"is '{earliest_chain_norm}'. According to the answer text, Michaels opens at '{m_time or ''}' and "
        f"Hobby Lobby opens at '{h_time or ''}'. The identified earliest chain should be consistent with these times."
    )
    await evaluator.verify(
        claim=earliest_chain_claim,
        node=earliest_chain_leaf,
        additional_instruction=(
            "Judge the claim logically using the times stated in the provided answer text. Allow minor formatting "
            "differences (e.g., '6 AM' vs '6:00 a.m.'). If a time is missing for one chain, rely on what is stated; "
            "do not invent times."
        ),
    )

    # 2) Earliest Opening Time (critical leaf)
    opening_time_leaf = evaluator.add_leaf(
        id="Earliest_Opening_Time",
        desc="State the correct opening time for the earliest-opening chain on Black Friday 2025.",
        parent=root,
        critical=True,
    )
    earliest_opening_time_str = _get_chain_opening_time_for_claim(extracted)
    opening_sources = _select_earliest_chain_sources(extracted)

    opening_time_claim = (
        f"{earliest_chain_norm} opens at '{earliest_opening_time_str}' on Black Friday 2025 (Nov 29, 2025) in Columbus, Ohio."
    )
    await evaluator.verify(
        claim=opening_time_claim,
        node=opening_time_leaf,
        sources=opening_sources if opening_sources else None,
        additional_instruction=(
            "Verify the opening time using the cited webpage(s). Prefer official store hours, store locator entries "
            "for Columbus, OH, or Black Friday hours announcements. Consider equivalence such as '6 AM' vs '6:00 a.m.'. "
            "If the webpage is irrelevant or does not support the claimed time, mark as not supported."
        ),
    )

    # 3) Max Discount Percentage (critical leaf)
    max_discount_leaf = evaluator.add_leaf(
        id="Max_Discount_Percentage",
        desc="State the maximum discount percentage offered during the Black Friday sale by the earliest-opening chain.",
        parent=root,
        critical=True,
    )
    discount_str = _get_chain_discount_for_claim(extracted)
    discount_sources = opening_sources  # Use same sources bundle; typically ads and promo pages

    discount_claim = (
        f"During its Black Friday 2025 sale, {earliest_chain_norm} offers a maximum discount of '{discount_str}'."
    )
    await evaluator.verify(
        claim=discount_claim,
        node=max_discount_leaf,
        sources=discount_sources if discount_sources else None,
        additional_instruction=(
            "Verify the largest advertised percent-off for Black Friday 2025 from the cited webpage(s). Treat 'up to X%' "
            "as the maximum discount. If multiple percentages are listed, select the highest. If the page is irrelevant "
            "or does not substantiate the claimed maximum percent, mark as not supported."
        ),
    )

    return evaluator.get_summary()