import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sp500_lowest_cost_2025"
TASK_DESCRIPTION = """
Among the major S&P 500 index funds offered by Vanguard, Fidelity, Schwab, State Street, and BlackRock, which fund has the lowest expense ratio as of 2025? Provide the fund name, its expense ratio, and a reference URL from the official fund provider page.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FundSelectionExtraction(BaseModel):
    fund_name: Optional[str] = None
    ticker: Optional[str] = None
    provider: Optional[str] = None
    expense_ratio: Optional[str] = None
    reference_url: Optional[str] = None
    other_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_fund_selection() -> str:
    return """
    Extract the fund the answer claims has the lowest expense ratio among the S&P 500 index funds from Vanguard, Fidelity, Schwab, State Street (SPDR), and BlackRock (iShares), as of 2025.

    Required fields:
    - fund_name: The specific fund name as written (e.g., "Fidelity 500 Index Fund", "iShares Core S&P 500 ETF").
    - ticker: If a ticker is given (e.g., FXAIX, IVV, VOO, SWPPX, SPY). If none is given, return null.
    - provider: The fund provider brand as stated in the answer. If present, use a clear brand like "Vanguard", "Fidelity", "Schwab", "State Street (SPDR)", or "BlackRock (iShares)". If not explicitly stated, return null.
    - expense_ratio: The explicit expense ratio value stated for the identified fund (keep the original formatting such as "0.015%" or "0.02%").
    - reference_url: A single official fund provider URL that supports the expense ratio for the identified fund. Only extract a URL that is actually present in the answer text. If the answer does not provide any official provider URL, return null.
    - other_reference_urls: Extract any additional official provider URLs (if present in the answer) that are relevant for cross-checking expense ratios of other S&P 500 funds from the five specified providers. Only include URLs explicitly present in the answer text. If none, return an empty list.

    URL extraction rules:
    - Extract only URLs that are explicitly present in the answer (including markdown links).
    - Do not invent or infer URLs.
    - Include the full URL with protocol (prepend http:// if missing).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
_ALLOWED_PROVIDERS = {
    "vanguard": "Vanguard",
    "fidelity": "Fidelity",
    "schwab": "Schwab",
    "state street (spdr)": "State Street (SPDR)",
    "spdr": "State Street (SPDR)",
    "ssga": "State Street (SPDR)",
    "blackrock (ishares)": "BlackRock (iShares)",
    "ishares": "BlackRock (iShares)",
    "blackrock": "BlackRock (iShares)",
}


def normalize_provider_name(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower()
    # Simple normalization rules
    if "vanguard" in s:
        return _ALLOWED_PROVIDERS["vanguard"]
    if "fidelity" in s:
        return _ALLOWED_PROVIDERS["fidelity"]
    if "schwab" in s or "charles schwab" in s:
        return _ALLOWED_PROVIDERS["schwab"]
    if "spdr" in s or "state street" in s or "ssga" in s:
        return _ALLOWED_PROVIDERS["spdr"]
    if "ishares" in s or "blackrock" in s:
        return _ALLOWED_PROVIDERS["ishares"]
    # Exact match fallback
    return _ALLOWED_PROVIDERS.get(s, None)


def infer_provider_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return None
    if "vanguard.com" in netloc:
        return _ALLOWED_PROVIDERS["vanguard"]
    if "fidelity.com" in netloc:
        return _ALLOWED_PROVIDERS["fidelity"]
    if "schwab.com" in netloc:
        return _ALLOWED_PROVIDERS["schwab"]
    if "spdrs.com" in netloc or "ssga.com" in netloc:
        return _ALLOWED_PROVIDERS["spdr"]
    if "ishares.com" in netloc or "blackrock.com" in netloc:
        return _ALLOWED_PROVIDERS["ishares"]
    return None


def build_fund_identifier(fund_name: Optional[str], ticker: Optional[str]) -> str:
    if fund_name and ticker:
        return f"{fund_name} ({ticker})"
    if fund_name:
        return fund_name
    if ticker:
        return ticker
    return "the identified fund"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root,
    extracted: FundSelectionExtraction,
) -> None:
    # Normalize provider or infer from URL
    norm_provider = normalize_provider_name(extracted.provider) or infer_provider_from_url(extracted.reference_url)
    fund_identifier = build_fund_identifier(extracted.fund_name, extracted.ticker)
    expense_ratio_str = extracted.expense_ratio or ""

    # 1) Candidate_Fund_Validity (Parallel, Critical)
    candidate_node = evaluator.add_parallel(
        id="Candidate_Fund_Validity",
        desc="The proposed fund is an eligible candidate under the question constraints.",
        parent=root,
        critical=True
    )

    # 1.a) Fund_Name_Provided (Leaf via custom existence)
    fund_name_exists = bool((extracted.fund_name and extracted.fund_name.strip()) or (extracted.ticker and extracted.ticker.strip()))
    evaluator.add_custom_node(
        result=fund_name_exists,
        id="Fund_Name_Provided",
        desc="Provides a specific fund name/ticker to identify the fund unambiguously.",
        parent=candidate_node,
        critical=True
    )

    # 1.b) Tracks_S&P_500_Index (Leaf – verify using official page if available)
    tracks_node = evaluator.add_leaf(
        id="Tracks_S&P_500_Index",
        desc="The fund is an S&P 500 index fund that tracks the S&P 500 Index.",
        parent=candidate_node,
        critical=True
    )
    tracks_claim = f"This official fund page shows that {fund_identifier} tracks the S&P 500 Index (allow variants such as 'S&P 500', 'S&P 500 Index', or 'Standard & Poor's 500')."
    await evaluator.verify(
        claim=tracks_claim,
        node=tracks_node,
        sources=extracted.reference_url,
        additional_instruction="Confirm on the page that the fund tracks the S&P 500. Small naming variants are acceptable (e.g., 'S&P 500', 'S&P 500 Index', 'Standard & Poor's 500')."
    )

    # 1.c) Provider_Is_One_Of_Five (Leaf – simple verify)
    provider_node = evaluator.add_leaf(
        id="Provider_Is_One_Of_Five",
        desc="The fund is offered by one of: Vanguard, Fidelity, Schwab, State Street (SPDR), or BlackRock (iShares).",
        parent=candidate_node,
        critical=True
    )
    provider_to_check = norm_provider or (extracted.provider or "").strip()
    provider_claim = f"The provider '{provider_to_check}' is one of: Vanguard, Fidelity, Schwab, State Street (SPDR), or BlackRock (iShares)."
    await evaluator.verify(
        claim=provider_claim,
        node=provider_node,
        additional_instruction="Treat simple brand synonyms as acceptable (e.g., 'Charles Schwab' ~ 'Schwab', 'SPDR'/'SSGA' ~ 'State Street (SPDR)', 'iShares'/'BlackRock' ~ 'BlackRock (iShares)')."
    )

    # 2) Expense_Ratio_Evidence_and_Value (Parallel, Critical)
    expense_node = evaluator.add_parallel(
        id="Expense_Ratio_Evidence_and_Value",
        desc="Provides the expense ratio and demonstrates it is correct and applicable as of 2025.",
        parent=root,
        critical=True
    )

    # 2.a) Expense_Ratio_Value_Provided (Leaf via custom existence)
    ratio_exists = bool(extracted.expense_ratio and extracted.expense_ratio.strip())
    evaluator.add_custom_node(
        result=ratio_exists,
        id="Expense_Ratio_Value_Provided",
        desc="States an explicit expense ratio value for the identified fund.",
        parent=expense_node,
        critical=True
    )

    # 2.b) Expense_Ratio_Matches_Official_AsOf_2025 (Leaf – verify by official URL)
    ratio_match_node = evaluator.add_leaf(
        id="Expense_Ratio_Matches_Official_AsOf_2025",
        desc="The stated expense ratio matches what is shown on the official fund provider page and is current as of 2025.",
        parent=expense_node,
        critical=True
    )
    ratio_claim = f"The official provider page shows that the expense ratio for {fund_identifier} is {expense_ratio_str}, applicable in or across 2025."
    await evaluator.verify(
        claim=ratio_claim,
        node=ratio_match_node,
        sources=extracted.reference_url,
        additional_instruction=(
            "Confirm that the page explicitly shows the same expense ratio. "
            "Minor formatting/rounding differences are acceptable (e.g., 0.015% vs 0.0150%). "
            "If both 'net' and 'gross' expense ratios are presented, prefer 'net' unless the answer clearly states otherwise. "
            "For 'as of' dates: if the page indicates the same ratio in 2025 or later with no changes, treat it as consistent with 'as of 2025'."
        )
    )

    # 3) Lowest_Among_Eligible_Funds (Leaf – verify across provided URLs if available)
    lowest_node = evaluator.add_leaf(
        id="Lowest_Among_Eligible_Funds",
        desc="The identified fund’s expense ratio is the lowest among all eligible S&P 500 index funds from the five specified providers (as of 2025).",
        parent=root,
        critical=True
    )
    # Combine the main official page + any other provider URLs from the answer (if any)
    compare_urls: List[str] = []
    if extracted.reference_url:
        compare_urls.append(extracted.reference_url)
    if extracted.other_reference_urls:
        compare_urls.extend([u for u in extracted.other_reference_urls if isinstance(u, str) and u.strip()])

    lowest_claim = (
        f"As of 2025, among S&P 500 index funds offered by Vanguard, Fidelity, Schwab, State Street (SPDR), and BlackRock (iShares), "
        f"the lowest expense ratio is {expense_ratio_str}, and it belongs to {fund_identifier}. "
        f"If there is a tie at the same minimum expense ratio across multiple providers, treat this claim as valid if {fund_identifier}'s ratio equals that tied minimum."
    )
    await evaluator.verify(
        claim=lowest_claim,
        node=lowest_node,
        sources=compare_urls if compare_urls else None,
        additional_instruction=(
            "Use the provided official fund pages (if multiple URLs are given) to compare expense ratios across the five providers' S&P 500 index funds. "
            "If insufficient evidence is provided to verify cross-provider comparison, mark as not supported."
        )
    )

    # 4) Supporting_Reference_URL (Leaf – check official provider page presence and relevance)
    support_url_node = evaluator.add_leaf(
        id="Supporting_Reference_URL",
        desc="Provides a reference URL from the official fund provider website that supports the expense ratio claim (as of 2025).",
        parent=root,
        critical=True
    )
    support_claim = (
        f"The provided URL is an official fund provider page for {fund_identifier}, and it supports the expense ratio information "
        f"(matching or consistent with {expense_ratio_str}) as current or applicable as of 2025."
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_url_node,
        sources=extracted.reference_url,
        additional_instruction=(
            "Confirm that the URL domain belongs to the fund's official provider (e.g., vanguard.com, fidelity.com, schwab.com, spdrs.com/ssga.com, ishares.com/blackrock.com), "
            "and that the page is specifically about the identified fund and includes its expense ratio details."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the S&P 500 lowest-cost fund identification task.
    """
    # Initialize evaluator with a sequential root, mirroring rubric sequential flow
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract selected fund and references from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_fund_selection(),
        template_class=FundSelectionExtraction,
        extraction_name="selected_fund_extraction",
    )

    # Add helper info for debugging (normalized provider and inferred provider)
    normalized_provider = normalize_provider_name(extracted.provider)
    inferred_provider = infer_provider_from_url(extracted.reference_url)
    evaluator.add_custom_info(
        info={
            "raw_provider": extracted.provider,
            "normalized_provider": normalized_provider,
            "inferred_provider_from_url": inferred_provider,
            "fund_identifier": build_fund_identifier(extracted.fund_name, extracted.ticker),
            "expense_ratio_raw": extracted.expense_ratio,
            "reference_url": extracted.reference_url,
            "other_reference_urls": extracted.other_reference_urls,
        },
        info_type="extraction_postprocess",
        info_name="post_extraction_summary"
    )

    # Build verification tree in the order defined by the rubric
    await build_verification_tree(evaluator, root, extracted)

    # Return standardized summary
    return evaluator.get_summary()