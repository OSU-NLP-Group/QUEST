import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "xrp_spot_etf_lowest_expense"
TASK_DESCRIPTION = (
    "As an investor interested in gaining exposure to XRP through a regulated investment vehicle in the United States, "
    "identify which spot XRP exchange-traded fund (ETF) from a major asset management firm with at least $1 billion in global assets under management that is currently approved and trading in the US market has the lowest expense ratio. "
    "Provide the ETF's ticker symbol, its expense ratio as a percentage, and include at least one reference URL from a reputable source to verify this information."
)


class EtfInfo(BaseModel):
    """Information about the identified ETF extracted from the answer."""
    etf_name: Optional[str] = None
    ticker: Optional[str] = None
    issuer: Optional[str] = None
    expense_ratio: Optional[str] = None  # keep as string e.g., "0.25%" or "0.25"
    reference_urls: List[str] = Field(default_factory=list)

    # Optional supporting evidence fields if provided by the answer
    competitor_urls: List[str] = Field(default_factory=list)  # URLs discussing other XRP ETFs or comparisons
    issuer_aum_amount: Optional[str] = None  # e.g., "$9.7T"
    issuer_aum_urls: List[str] = Field(default_factory=list)  # URLs evidencing issuer AUM
    market: Optional[str] = None  # e.g., "US", "United States"
    trading_status: Optional[str] = None  # e.g., "approved and trading"
    category: Optional[str] = None  # e.g., "spot XRP ETF", "futures XRP ETF"
    asset: Optional[str] = None  # e.g., "XRP"


def prompt_extract_etf_info() -> str:
    return (
        "From the provided answer, extract details for the single ETF the answer selects as having the lowest expense ratio "
        "among qualifying U.S.-listed spot XRP ETFs from major asset managers (≥$1B AUM). "
        "If multiple ETFs are mentioned, choose the one explicitly presented as the lowest-fee choice.\n\n"
        "Return a JSON object with the following fields:\n"
        "- etf_name: The ETF's full name, if provided\n"
        "- ticker: The ETF's ticker symbol\n"
        "- issuer: The ETF issuer/asset manager name\n"
        "- expense_ratio: The ETF's expense ratio as stated (keep exactly as written, e.g., '0.25%' or '0.25')\n"
        "- reference_urls: An array of all URLs in the answer that directly support the ETF details (official fund page, prospectus, issuer site, finance portals, or regulatory filings). Include only valid URLs explicitly present in the answer.\n"
        "- competitor_urls: An array of any URLs in the answer that mention other spot XRP ETFs, compare fees, or list multiple XRP ETFs. Include only valid URLs explicitly present in the answer.\n"
        "- issuer_aum_amount: The issuer's AUM amount if the answer mentions it (e.g., '$2 trillion', '$1.2B'). Keep exactly as written.\n"
        "- issuer_aum_urls: An array of URLs that the answer cites to evidence the issuer's AUM. Include only valid URLs explicitly present in the answer.\n"
        "- market: If the answer mentions where the ETF is trading or listed, extract the market/country (e.g., 'US', 'United States').\n"
        "- trading_status: If the answer mentions status, extract a short phrase (e.g., 'approved and trading').\n"
        "- category: If the answer mentions product type, extract a short phrase (e.g., 'spot XRP ETF', 'futures-based').\n"
        "- asset: If the answer mentions the underlying asset, extract it (e.g., 'XRP').\n"
        "If any field is missing in the answer, set it to null or an empty array accordingly. "
        "Do not invent URLs or details not present in the answer."
    )


def _merge_urls(*lists: List[str]) -> List[str]:
    """Merge URL lists, remove duplicates and obvious empties."""
    seen = set()
    result: List[str] = []
    for lst in lists:
        for u in lst or []:
            if not isinstance(u, str):
                continue
            url = u.strip()
            if not url:
                continue
            if url.lower().startswith("http") and url not in seen:
                seen.add(url)
                result.append(url)
    return result


async def build_and_verify_tree(evaluator: Evaluator, extracted: EtfInfo, parent_root) -> None:
    """
    Build the verification tree according to the rubric and run the checks.
    """
    # Main critical aggregate node per rubric
    overall_node = evaluator.add_parallel(
        id="Correct_XRP_ETF_Identification",
        desc=(
            "The answer correctly identifies the spot XRP ETF from a major asset management firm (≥$1 billion AUM) "
            "that is currently trading in the US and has the lowest expense ratio among all qualifying ETFs"
        ),
        parent=parent_root,
        critical=True,
    )

    # 1) Required Information (critical) - existence check
    has_ticker = bool(extracted.ticker and extracted.ticker.strip())
    has_expense_ratio = bool(extracted.expense_ratio and extracted.expense_ratio.strip())
    has_at_least_one_url = bool(extracted.reference_urls and len(extracted.reference_urls) > 0)

    evaluator.add_custom_node(
        result=(has_ticker and has_expense_ratio and has_at_least_one_url),
        id="Required_Information",
        desc=(
            "The answer provides all required information: the ETF's ticker symbol, its expense ratio as a percentage, "
            "and at least one valid reference URL from a reputable source"
        ),
        parent=overall_node,
        critical=True,
    )

    # 2) ETF Qualification (critical) - break into specific leaf checks
    qual_node = evaluator.add_parallel(
        id="ETF_Qualification",
        desc=(
            "The identified ETF meets all qualifying criteria: (1) it is a spot XRP ETF holding actual XRP, "
            "(2) the issuer has at least $1 billion in global AUM, and (3) the ETF is currently approved and trading in the US market"
        ),
        parent=overall_node,
        critical=True,
    )

    # 2.1 Spot XRP holding actual XRP
    spot_leaf = evaluator.add_leaf(
        id="spot_xrp_check",
        desc="The ETF is a spot XRP ETF holding actual XRP (not futures-based or leveraged)",
        parent=qual_node,
        critical=True,
    )
    spot_claim = (
        f"The ETF {extracted.etf_name or ''} ({extracted.ticker or ''}) is a spot XRP ETF that holds actual XRP, "
        "not a futures-based, synthetic, or leveraged product."
    )
    await evaluator.verify(
        claim=spot_claim,
        node=spot_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Rely on official fund pages, prospectuses, issuer sites, or reputable finance portals. "
            "Accept reasonable synonyms indicating spot exposure or a trust holding XRP. "
            "If the product is clearly described as futures-based or synthetic, this should be incorrect."
        ),
    )

    # 2.2 Issuer AUM ≥ $1B
    aum_leaf = evaluator.add_leaf(
        id="issuer_aum_check",
        desc="The issuer has at least $1 billion in global assets under management (AUM)",
        parent=qual_node,
        critical=True,
    )
    aum_sources = _merge_urls(extracted.issuer_aum_urls, extracted.reference_urls)
    aum_claim = (
        f"The ETF issuer {extracted.issuer or ''} has global assets under management of at least $1 billion."
    )
    await evaluator.verify(
        claim=aum_claim,
        node=aum_leaf,
        sources=aum_sources if len(aum_sources) > 0 else None,
        additional_instruction=(
            "Use issuer profile pages, annual reports, trusted financial publications, or directories. "
            "If a specific AUM figure is shown and it is clearly ≥ $1B, treat as correct (allow approximate wording)."
        ),
    )

    # 2.3 Approved and trading in the US
    trading_leaf = evaluator.add_leaf(
        id="us_trading_check",
        desc="The ETF is currently approved and trading in the United States market",
        parent=qual_node,
        critical=True,
    )
    trading_claim = (
        f"The ETF {extracted.etf_name or ''} ({extracted.ticker or ''}) is approved and currently trading in the US market."
    )
    await evaluator.verify(
        claim=trading_claim,
        node=trading_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Confirm listing or trading on a US exchange or via reputable fund information sources. "
            "Regulatory filings or issuer pages stating US listing should count as support."
        ),
    )

    # 3) Lowest Expense Ratio (critical) - break into accuracy and 'lowest' claims
    lowest_node = evaluator.add_parallel(
        id="Lowest_Expense_Ratio",
        desc=(
            "Among all spot XRP ETFs from major asset managers (≥$1B AUM) that are currently trading in the US, "
            "the identified ETF has the lowest expense ratio (annual fee percentage)"
        ),
        parent=overall_node,
        critical=True,
    )

    # 3.1 Expense ratio accuracy
    ratio_leaf = evaluator.add_leaf(
        id="expense_ratio_accuracy",
        desc="The ETF's expense ratio is accurately cited",
        parent=lowest_node,
        critical=True,
    )
    ratio_claim = (
        f"The ETF {extracted.etf_name or ''} ({extracted.ticker or ''}) has an expense ratio of {extracted.expense_ratio or ''}."
    )
    await evaluator.verify(
        claim=ratio_claim,
        node=ratio_leaf,
        sources=extracted.reference_urls,
        additional_instruction=(
            "Match the stated expense ratio against official or reputable sources. "
            "Allow minor formatting differences (e.g., with or without the '%' sign) and reasonable rounding."
        ),
    )

    # 3.2 Lowest among qualifying ETFs
    lowest_leaf = evaluator.add_leaf(
        id="lowest_expense_ratio_claim",
        desc=(
            "The identified ETF has the lowest expense ratio among U.S.-listed spot XRP ETFs from ≥$1B AUM issuers that are currently trading"
        ),
        parent=lowest_node,
        critical=True,
    )
    lowest_sources = _merge_urls(extracted.competitor_urls, extracted.reference_urls)
    lowest_claim = (
        f"Among U.S.-listed spot XRP ETFs from major asset managers (≥$1B AUM) that are currently trading, "
        f"{extracted.etf_name or ''} ({extracted.ticker or ''}) has the lowest expense ratio."
    )
    await evaluator.verify(
        claim=lowest_claim,
        node=lowest_leaf,
        sources=lowest_sources if len(lowest_sources) > 0 else None,
        additional_instruction=(
            "Prefer comparison or aggregator pages that list multiple XRP ETFs and their fees, or multiple official sources. "
            "If the sources credibly indicate that only one qualifying U.S. spot XRP ETF exists, treat the 'lowest' as trivially true."
        ),
    )


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
    Entry point for evaluating the answer to the XRP spot ETF lowest expense ratio task.
    """
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

    # Extract ETF info from the answer
    extracted_etf = await evaluator.extract(
        prompt=prompt_extract_etf_info(),
        template_class=EtfInfo,
        extraction_name="identified_xrp_etf",
    )

    # Build tree and run verifications
    await build_and_verify_tree(evaluator, extracted_etf, root)

    # Return summary
    return evaluator.get_summary()