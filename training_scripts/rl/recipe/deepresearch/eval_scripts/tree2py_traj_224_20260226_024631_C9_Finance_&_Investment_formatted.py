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
TASK_ID = "xrp_etf_portfolio"
TASK_DESCRIPTION = (
    "An institutional investor wants to construct a diversified XRP ETF portfolio with exactly four US-listed spot XRP ETFs, "
    "each serving a different strategic purpose. For each of the four investment objectives, identify the single XRP ETF that "
    "best meets the stated criterion and provide attributes and supporting references."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ETFItem(BaseModel):
    """Represents a single ETF selection with required and optional attributes."""
    name: Optional[str] = None
    ticker: Optional[str] = None
    issuer: Optional[str] = None
    expense_ratio: Optional[str] = None  # keep as string to allow formats like "0.25%" or "25 bps"
    launch_date: Optional[str] = None    # month and year at minimum
    exchange: Optional[str] = None       # primary trading exchange
    references: List[str] = Field(default_factory=list)

    # Optional extras by role
    first_day_inflow: Optional[str] = None      # for First Mover
    parent_aum: Optional[str] = None            # for Institutional Credibility (total firm AUM)
    volume_metric: Optional[str] = None         # for High Liquidity (e.g., "avg daily volume $X over Y")


class PortfolioExtraction(BaseModel):
    """Structure holding the four role-based ETF selections."""
    lowest_cost: Optional[ETFItem] = None
    first_mover: Optional[ETFItem] = None
    institutional_credibility: Optional[ETFItem] = None
    high_liquidity: Optional[ETFItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_portfolio() -> str:
    return """
    Extract exactly four US-listed spot XRP ETFs selected in the answer, each mapped to one distinct role:

    Roles:
    - lowest_cost: The ETF selected for cost efficiency (lowest expense ratio)
    - first_mover: The ETF selected as the first to begin trading in the U.S.
    - institutional_credibility: The ETF selected for maximum institutional credibility (issuer with largest total firm AUM among XRP ETF issuers)
    - high_liquidity: The ETF selected for optimal liquidity (highest trading volume)

    For each role, extract the following fields from the answer text exactly as provided:
    - name: The fund or product name (if mentioned)
    - ticker: The ticker symbol
    - issuer: The issuer or sponsor name
    - expense_ratio: The expense ratio (percentage string; e.g., "0.25%" or "25 bps")
    - launch_date: The launch date (month and year at minimum; include day if provided)
    - exchange: The primary trading exchange (e.g., "NYSE Arca", "Nasdaq")
    - references: An array of URLs explicitly mentioned in the answer that support any of the above claims for the ETF. 
      Extract only actual URLs (including those inside markdown links), do not infer or create URLs.

    Optional role-specific fields (extract only if explicitly present in the answer; otherwise return null):
    - For first_mover: first_day_inflow (e.g., "First-day inflows of $150 million")
    - For institutional_credibility: parent_aum (total firm AUM; e.g., "$9.2T")
    - For high_liquidity: volume_metric (evidence for liquidity; e.g., "avg daily volume $50M")

    STRICT RULES:
    1. Only extract information explicitly stated in the answer. If a field is missing, set it to null (or empty array for references).
    2. For URL fields, return actual URLs as strings. Include protocol (http/https). Ignore invalid URLs.
    3. Do not add, guess, or transform values beyond reasonable normalization (e.g., preserve % symbols and currency signs).
    4. Return a JSON object with four top-level fields: lowest_cost, first_mover, institutional_credibility, high_liquidity.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _clean_sources(urls: Optional[List[str]]) -> List[str]:
    """Normalize and filter source URLs."""
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        # Basic validity guard; accept http(s) only
        if s.startswith("http://") or s.startswith("https://"):
            cleaned.append(s)
        else:
            # if missing protocol but seems like a domain, prepend http:// as per toolkit guidance
            if "." in s and "/" in s:
                cleaned.append(f"http://{s}")
    return cleaned


def _safe_str(x: Optional[str]) -> str:
    return (x or "").strip()


async def _verify_attribute_with_sources(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    sources: List[str],
    critical: bool = True,
    additional_instruction: Optional[str] = None,
) -> None:
    """Create a leaf node and verify an attribute claim against provided sources."""
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources if sources else None,
        additional_instruction=additional_instruction or "None",
    )


async def _verify_attributes_block(
    evaluator: Evaluator,
    role_prefix: str,
    attrs_node,
    etf: ETFItem,
) -> None:
    """
    Verify the core attributes for an ETF using its references.
    Enforces source-grounding via a critical 'references provided' gate before other attribute checks.
    """
    sources = _clean_sources(etf.references)

    # Gate: At least one supporting reference
    evaluator.add_custom_node(
        result=(len(sources) >= 1),
        id=f"{role_prefix}_References",
        desc=f"At least one supporting URL reference is provided for the {role_prefix.replace('_', ' ').lower()} ETF",
        parent=attrs_node,
        critical=True
    )

    # Verify attributes using sources; references gate runs first (created before other leaves).
    add_ins_common = (
        "Verify this specific attribute using the provided URLs. Prefer official issuer pages, regulatory filings, "
        "exchange listings, or reputable financial sources. Allow minor formatting variations (e.g., case, punctuation)."
    )

    # Ticker
    ticker_val = _safe_str(etf.ticker)
    await _verify_attribute_with_sources(
        evaluator,
        attrs_node,
        f"{role_prefix}_Ticker",
        f"The ticker symbol of the {role_prefix.replace('_', ' ').lower()} XRP ETF is provided",
        f"The ETF's ticker symbol is '{ticker_val}'.",
        sources,
        critical=True,
        additional_instruction=add_ins_common
    )

    # Issuer
    issuer_val = _safe_str(etf.issuer)
    await _verify_attribute_with_sources(
        evaluator,
        attrs_node,
        f"{role_prefix}_Issuer",
        f"The issuer name of the {role_prefix.replace('_', ' ').lower()} XRP ETF is provided",
        f"The ETF's issuer (sponsor) is '{issuer_val}'.",
        sources,
        critical=True,
        additional_instruction=add_ins_common
    )

    # Expense Ratio
    fee_val = _safe_str(etf.expense_ratio)
    await _verify_attribute_with_sources(
        evaluator,
        attrs_node,
        f"{role_prefix}_Fee",
        f"The expense ratio of the {role_prefix.replace('_', ' ').lower()} XRP ETF is provided",
        f"The ETF's expense ratio is '{fee_val}'.",
        sources,
        critical=True,
        additional_instruction=(
            add_ins_common + " Expense ratio may be stated as a percentage or basis points; treat 0.25% ≈ 25 bps."
        )
    )

    # Launch Date
    launch_val = _safe_str(etf.launch_date)
    await _verify_attribute_with_sources(
        evaluator,
        attrs_node,
        f"{role_prefix}_Launch_Date",
        f"The launch date of the {role_prefix.replace('_', ' ').lower()} XRP ETF is provided",
        f"The ETF launched (first trading date) on '{launch_val}'.",
        sources,
        critical=True,
        additional_instruction=add_ins_common + " If only month/year are present, ensure the page supports that."
    )

    # Exchange
    exch_val = _safe_str(etf.exchange)
    await _verify_attribute_with_sources(
        evaluator,
        attrs_node,
        f"{role_prefix}_Exchange",
        f"The trading exchange of the {role_prefix.replace('_', ' ').lower()} XRP ETF is provided",
        f"The ETF's primary trading exchange is '{exch_val}'.",
        sources,
        critical=True,
        additional_instruction=add_ins_common
    )


# --------------------------------------------------------------------------- #
# Role-specific verification                                                  #
# --------------------------------------------------------------------------- #
async def verify_lowest_cost(
    evaluator: Evaluator,
    portfolio_node,
    etf: Optional[ETFItem],
) -> None:
    role_node = evaluator.add_sequential(
        id="Lowest_Cost_ETF",
        desc="Identify the XRP ETF with the lowest expense ratio for long-term cost-efficient holding",
        parent=portfolio_node,
        critical=False,
    )

    etf = etf or ETFItem()
    sources = _clean_sources(etf.references)
    name_ticker = (_safe_str(etf.name) or _safe_str(etf.ticker) or "the selected ETF")

    # Selection criterion: lowest expense ratio among US-listed spot XRP ETFs
    sel_leaf = evaluator.add_leaf(
        id="Lowest_Cost_Selection_Criterion",
        desc="The selected ETF must be the spot XRP ETF with the lowest expense ratio among all US-listed options",
        parent=role_node,
        critical=True,
    )
    claim = (
        f"Among US-listed spot XRP ETFs, {name_ticker} has the lowest expense ratio."
    )
    await evaluator.verify(
        claim=claim,
        node=sel_leaf,
        sources=sources if sources else None,
        additional_instruction=(
            "Confirm that the selected ETF's expense ratio is the lowest among US-listed spot XRP ETFs. "
            "Look for explicit comparisons, league tables, or authoritative statements."
        )
    )

    # Attributes block (non-critical parent to allow optional items in other roles)
    attrs_node = evaluator.add_parallel(
        id="Lowest_Cost_Attributes",
        desc="Provide complete identifying attributes for the lowest-cost ETF",
        parent=role_node,
        critical=False
    )
    await _verify_attributes_block(evaluator, "Lowest_Cost", attrs_node, etf)


async def verify_first_mover(
    evaluator: Evaluator,
    portfolio_node,
    etf: Optional[ETFItem],
) -> None:
    role_node = evaluator.add_sequential(
        id="First_Mover_ETF",
        desc="Identify the first US spot XRP ETF to launch for historical significance",
        parent=portfolio_node,
        critical=False,
    )

    etf = etf or ETFItem()
    sources = _clean_sources(etf.references)
    name_ticker = (_safe_str(etf.name) or _safe_str(etf.ticker) or "the selected ETF")

    # Selection criterion: first to begin trading
    sel_leaf = evaluator.add_leaf(
        id="First_Mover_Selection_Criterion",
        desc="The selected ETF must be the first US spot XRP ETF to begin trading",
        parent=role_node,
        critical=True,
    )
    claim = f"{name_ticker} was the first US-listed spot XRP ETF to begin trading."
    await evaluator.verify(
        claim=claim,
        node=sel_leaf,
        sources=sources if sources else None,
        additional_instruction=(
            "Verify that this ETF began trading earlier than any other US-listed spot XRP ETF. "
            "Check launch/trading commencement dates and explicit 'first' statements from credible sources."
        )
    )

    # Attributes block
    attrs_node = evaluator.add_parallel(
        id="First_Mover_Attributes",
        desc="Provide complete identifying attributes for the first-mover ETF",
        parent=role_node,
        critical=False
    )
    await _verify_attributes_block(evaluator, "First_Mover", attrs_node, etf)

    # Optional: first-day inflows (do not penalize if missing)
    if _safe_str(etf.first_day_inflow):
        await _verify_attribute_with_sources(
            evaluator,
            attrs_node,
            "First_Mover_First_Day_Inflows",
            "The first-day inflow amount for the first-mover XRP ETF is provided if available",
            f"On its first trading day, the ETF recorded inflows of '{_safe_str(etf.first_day_inflow)}'.",
            sources,
            critical=False,
            additional_instruction=(
                "Confirm the stated first-day inflows figure from reputable sources (issuer releases, news, exchange data)."
            )
        )


async def verify_institutional_credibility(
    evaluator: Evaluator,
    portfolio_node,
    etf: Optional[ETFItem],
) -> None:
    role_node = evaluator.add_sequential(
        id="Institutional_Credibility_ETF",
        desc="Identify the XRP ETF from the largest traditional asset manager for institutional credibility",
        parent=portfolio_node,
        critical=False,
    )

    etf = etf or ETFItem()
    sources = _clean_sources(etf.references)
    issuer_name = _safe_str(etf.issuer) or "the issuer"

    # Selection criterion: largest total firm AUM among XRP ETF issuers
    sel_leaf = evaluator.add_leaf(
        id="Institutional_Selection_Criterion",
        desc="The selected ETF must be issued by the traditional asset manager with the largest total AUM (not crypto-specific AUM) among all XRP ETF issuers",
        parent=role_node,
        critical=True,
    )
    claim = (
        f"Among issuers of US-listed spot XRP ETFs, {issuer_name} has the largest total firm AUM."
    )
    await evaluator.verify(
        claim=claim,
        node=sel_leaf,
        sources=sources if sources else None,
        additional_instruction=(
            "Verify that the issuer's total firm AUM (parent company) is the largest among the set of XRP ETF issuers. "
            "Use reputable AUM figures and credible comparisons; do not use crypto-only AUM."
        )
    )

    # Attributes block
    attrs_node = evaluator.add_parallel(
        id="Institutional_Attributes",
        desc="Provide complete identifying attributes for the institutional credibility ETF",
        parent=role_node,
        critical=False
    )
    await _verify_attributes_block(evaluator, "Institutional", attrs_node, etf)

    # Optional: parent total AUM (non-critical)
    if _safe_str(etf.parent_aum):
        await _verify_attribute_with_sources(
            evaluator,
            attrs_node,
            "Institutional_Parent_AUM",
            "The total AUM of the parent asset manager is provided if available",
            f"The parent company's total firm AUM is '{_safe_str(etf.parent_aum)}'.",
            sources,
            critical=False,
            additional_instruction=(
                "Confirm the issuer's parent company total firm AUM from credible sources (e.g., annual reports, official fact sheets)."
            )
        )


async def verify_high_liquidity(
    evaluator: Evaluator,
    portfolio_node,
    etf: Optional[ETFItem],
) -> None:
    role_node = evaluator.add_sequential(
        id="High_Liquidity_ETF",
        desc="Identify the XRP ETF with the highest trading volume for active trading",
        parent=portfolio_node,
        critical=False,
    )

    etf = etf or ETFItem()
    sources = _clean_sources(etf.references)
    name_ticker = (_safe_str(etf.name) or _safe_str(etf.ticker) or "the selected ETF")

    # Selection criterion: highest trading volume
    sel_leaf = evaluator.add_leaf(
        id="High_Liquidity_Selection_Criterion",
        desc="The selected ETF must be the spot XRP ETF with the highest trading volume among all US-listed options",
        parent=role_node,
        critical=True,
    )
    claim = f"Among US-listed spot XRP ETFs, {name_ticker} has the highest trading volume."
    await evaluator.verify(
        claim=claim,
        node=sel_leaf,
        sources=sources if sources else None,
        additional_instruction=(
            "Verify that this ETF's trading volume is higher than other US-listed spot XRP ETFs over a reasonable period "
            "(e.g., recent days/weeks). Accept reputable comparative sources; if timeframe is stated in the source, respect it."
        )
    )

    # Attributes block
    attrs_node = evaluator.add_parallel(
        id="High_Liquidity_Attributes",
        desc="Provide complete identifying attributes for the high-liquidity ETF",
        parent=role_node,
        critical=False
    )
    await _verify_attributes_block(evaluator, "High_Liquidity", attrs_node, etf)

    # Optional: volume metric evidence (non-critical)
    if _safe_str(etf.volume_metric):
        await _verify_attribute_with_sources(
            evaluator,
            attrs_node,
            "High_Liquidity_Volume_Metric",
            "Evidence of the highest trading volume status is provided if available",
            f"The ETF's trading volume metric is '{_safe_str(etf.volume_metric)}'.",
            sources,
            critical=False,
            additional_instruction=(
                "Confirm the stated trading volume metric using the provided sources (exchange data, issuer pages, reputable analytics)."
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
    Evaluate the XRP ETF portfolio construction task according to the rubric tree.
    """
    # Initialize evaluator with a non-critical root (to allow partial credit across roles)
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

    # Extract the portfolio selections from the answer
    extracted_portfolio = await evaluator.extract(
        prompt=prompt_extract_portfolio(),
        template_class=PortfolioExtraction,
        extraction_name="xrp_etf_portfolio_extraction",
    )

    # Add an explicit portfolio-level node to host the four role subtrees (non-critical to enable partial scoring)
    portfolio_node = evaluator.add_parallel(
        id="Portfolio_Construction",
        desc="Construct a diversified XRP ETF portfolio with 4 different funds, each serving a specific investment purpose",
        parent=root,
        critical=False
    )

    # Record extraction summary info
    evaluator.add_custom_info(
        {
            "lowest_cost": (extracted_portfolio.lowest_cost.dict() if extracted_portfolio.lowest_cost else None),
            "first_mover": (extracted_portfolio.first_mover.dict() if extracted_portfolio.first_mover else None),
            "institutional_credibility": (extracted_portfolio.institutional_credibility.dict() if extracted_portfolio.institutional_credibility else None),
            "high_liquidity": (extracted_portfolio.high_liquidity.dict() if extracted_portfolio.high_liquidity else None)
        },
        info_type="extraction_overview",
        info_name="extracted_portfolio_items"
    )

    # Build and verify each role subtree (sequential nodes per role)
    await verify_lowest_cost(evaluator, portfolio_node, extracted_portfolio.lowest_cost)
    await verify_first_mover(evaluator, portfolio_node, extracted_portfolio.first_mover)
    await verify_institutional_credibility(evaluator, portfolio_node, extracted_portfolio.institutional_credibility)
    await verify_high_liquidity(evaluator, portfolio_node, extracted_portfolio.high_liquidity)

    # Return structured evaluation summary
    return evaluator.get_summary()