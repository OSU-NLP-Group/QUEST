import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "us_healthcare_top10_criteria_march2026"
TASK_DESCRIPTION = (
    "Identify the U.S.-based publicly traded healthcare company that meets all of the following criteria as of March 2026: "
    "(1) The company must be classified as a large-cap stock with market capitalization of at least $10 billion; "
    "(2) The company's market capitalization must be at least $250 billion; "
    "(3) The company must operate in the healthcare sector; "
    "(4) The company must rank within the top 10 largest healthcare companies by market capitalization; "
    "(5) The company must pay regular dividends to shareholders; "
    "(6) The company must be domiciled and headquartered in the United States; "
    "(7) The company must be listed on a major U.S. stock exchange (NYSE or NASDAQ). "
    "Provide the company's name, stock ticker symbol, current market capitalization, and supporting reference URLs that verify each criterion."
)

AS_OF_TIMEFRAME = "March 2026"


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class CompanyExtraction(BaseModel):
    # Required output fields
    company_name: Optional[str] = None
    stock_ticker: Optional[str] = None
    market_cap_value: Optional[str] = None  # Keep as string to be permissive with formats

    # Optional helpers (not strictly required but may be provided by answers)
    exchange: Optional[str] = None  # e.g., NYSE or NASDAQ (or variants)
    sector: Optional[str] = None
    domicile: Optional[str] = None
    headquarters_country: Optional[str] = None

    # References per criterion (URLs explicitly present in the answer)
    ref_market_cap_urls: List[str] = Field(default_factory=list)
    ref_healthcare_sector_urls: List[str] = Field(default_factory=list)
    ref_top10_marketcap_urls: List[str] = Field(default_factory=list)
    ref_dividends_urls: List[str] = Field(default_factory=list)
    ref_us_domicile_hq_urls: List[str] = Field(default_factory=list)
    ref_exchange_listing_urls: List[str] = Field(default_factory=list)
    ref_stock_price_minimum_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_company() -> str:
    return f"""
You are given an answer that should identify ONE U.S.-based publicly traded healthcare company meeting specific criteria as of {AS_OF_TIMEFRAME}. Extract the following fields strictly from the answer text. Do NOT invent any information. If a field is not provided, return null (for strings) or an empty list (for URLs).

Return a single JSON object with the fields below:

- company_name: The company's name (string).
- stock_ticker: The company's stock ticker symbol (string), e.g., "UNH", "LLY".
- market_cap_value: The company's market capitalization value as provided in the answer (string; allow any formatting e.g., "$510B", "USD 500 billion", etc.).

- exchange: If the exchange is explicitly mentioned, extract it (e.g., "NYSE", "NASDAQ", "NasdaqGS"). Otherwise null.
- sector: If mentioned, extract the sector classification (e.g., "Healthcare" / "Health Care"). Otherwise null.
- domicile: If mentioned, extract the company's domicile / place of incorporation (e.g., "United States", "Delaware, United States"). Otherwise null.
- headquarters_country: If mentioned, extract the country of headquarters. Otherwise null.

For each criterion below, extract the reference URL(s) explicitly cited in the answer text. Return empty list if none are provided.
- ref_market_cap_urls: URL(s) supporting the market capitalization value and that it is ≥ $250B as of {AS_OF_TIMEFRAME}.
- ref_healthcare_sector_urls: URL(s) supporting healthcare sector classification/operation.
- ref_top10_marketcap_urls: URL(s) supporting that the company is within top 10 largest healthcare companies by market cap as of {AS_OF_TIMEFRAME}.
- ref_dividends_urls: URL(s) supporting that the company pays regular dividends.
- ref_us_domicile_hq_urls: URL(s) supporting U.S. domicile AND U.S. headquarters.
- ref_exchange_listing_urls: URL(s) supporting that the company is listed on NYSE or NASDAQ.
- ref_stock_price_minimum_urls: URL(s) supporting stock price ≥ $4 per share (as of the relevant timeframe around {AS_OF_TIMEFRAME}).

Important URL extraction rules:
- Extract only URLs explicitly present in the answer text (plain or Markdown links).
- Do not infer or fabricate URLs.
- If a URL lacks protocol, prepend http://

If more than one company is mentioned, extract the one that the answer ultimately recommends or presents as the final choice. If ambiguous, pick the first one mentioned with the most complete information.
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _urls_exist(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str) and u.strip() and ("http://" in u or "https://" in u):
            return True
    return False


def _company_label(info: CompanyExtraction) -> str:
    if info.company_name and info.stock_ticker:
        return f"{info.company_name} (ticker: {info.stock_ticker})"
    if info.company_name:
        return info.company_name
    if info.stock_ticker:
        return f"the company with ticker {info.stock_ticker}"
    return "the company"


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_references_nodes(
    evaluator: Evaluator,
    parent_node,
    info: CompanyExtraction,
):
    """
    Create the 'References_For_Each_Criterion' critical group and its leaf existence checks.
    Return a dict mapping reference keys to their nodes for use as prerequisites.
    """
    refs_group = evaluator.add_parallel(
        id="references_for_each_criterion",
        desc="Provides supporting reference URL(s) that verify each required criterion (URLs may overlap across criteria).",
        parent=parent_node,
        critical=True
    )

    nodes = {}

    nodes["market_cap"] = evaluator.add_custom_node(
        result=_urls_exist(info.ref_market_cap_urls),
        id="reference_for_market_cap",
        desc=f"Provides reference URL(s) supporting the market capitalization (and ≥ $250B) as of {AS_OF_TIMEFRAME}.",
        parent=refs_group,
        critical=True
    )

    nodes["healthcare_sector"] = evaluator.add_custom_node(
        result=_urls_exist(info.ref_healthcare_sector_urls),
        id="reference_for_healthcare_sector",
        desc="Provides reference URL(s) supporting healthcare sector classification/operation.",
        parent=refs_group,
        critical=True
    )

    nodes["top10"] = evaluator.add_custom_node(
        result=_urls_exist(info.ref_top10_marketcap_urls),
        id="reference_for_top10_ranking",
        desc=f"Provides reference URL(s) supporting top-10 healthcare-by-market-cap ranking as of {AS_OF_TIMEFRAME}.",
        parent=refs_group,
        critical=True
    )

    nodes["dividends"] = evaluator.add_custom_node(
        result=_urls_exist(info.ref_dividends_urls),
        id="reference_for_dividends",
        desc="Provides reference URL(s) supporting regular dividend payments.",
        parent=refs_group,
        critical=True
    )

    nodes["us_domicile_hq"] = evaluator.add_custom_node(
        result=_urls_exist(info.ref_us_domicile_hq_urls),
        id="reference_for_us_domicile_and_hq",
        desc="Provides reference URL(s) supporting U.S. domicile and U.S. headquarters.",
        parent=refs_group,
        critical=True
    )

    nodes["listing"] = evaluator.add_custom_node(
        result=_urls_exist(info.ref_exchange_listing_urls),
        id="reference_for_exchange_listing",
        desc="Provides reference URL(s) supporting NYSE/NASDAQ listing.",
        parent=refs_group,
        critical=True
    )

    nodes["stock_price"] = evaluator.add_custom_node(
        result=_urls_exist(info.ref_stock_price_minimum_urls),
        id="reference_for_stock_price_minimum",
        desc="Provides reference URL(s) supporting the ≥ $4 stock price condition.",
        parent=refs_group,
        critical=True
    )

    return nodes


async def build_required_output_nodes(
    evaluator: Evaluator,
    parent_node,
    info: CompanyExtraction,
):
    """
    Create the 'Required_Output_Elements' critical group and its leaf existence checks.
    """
    req_group = evaluator.add_parallel(
        id="required_output_elements",
        desc="Answer includes the explicitly requested output fields (excluding references).",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.company_name and info.company_name.strip()),
        id="company_name_provided",
        desc="Provides the company's name.",
        parent=req_group,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.stock_ticker and info.stock_ticker.strip()),
        id="stock_ticker_provided",
        desc="Provides the company's stock ticker symbol.",
        parent=req_group,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.market_cap_value and info.market_cap_value.strip()),
        id="market_cap_value_provided",
        desc=f"Provides the company's market capitalization value as of {AS_OF_TIMEFRAME}.",
        parent=req_group,
        critical=True
    )


async def build_eligibility_nodes(
    evaluator: Evaluator,
    parent_node,
    info: CompanyExtraction,
    ref_nodes: Dict[str, Any],
):
    """
    Create the 'Eligibility_Criteria' critical group and add verification leaves for each constraint.
    Each verification is grounded in the corresponding reference URLs and depends on them.
    """
    elig_group = evaluator.add_parallel(
        id="eligibility_criteria",
        desc=f"Company satisfies all stated eligibility constraints as of {AS_OF_TIMEFRAME}.",
        parent=parent_node,
        critical=True
    )

    company_display = _company_label(info)

    # 1) Market cap ≥ $250B (satisfies ≥ $10B large-cap requirement)
    leaf_cap = evaluator.add_leaf(
        id="market_cap_at_least_250b",
        desc=f"Market capitalization is at least $250 billion as of {AS_OF_TIMEFRAME} (also satisfies ≥ $10B large-cap).",
        parent=elig_group,
        critical=True
    )
    cap_claim = (
        f"As of {AS_OF_TIMEFRAME}, {company_display} has a market capitalization of at least $250 billion (≥ 250B USD). "
        "If a specific numeric value is shown, it should meet or exceed this threshold."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=leaf_cap,
        sources=info.ref_market_cap_urls,
        additional_instruction=(
            "Accept clearly equivalent phrasings/numbers (e.g., 0.25T, 250B, $250,000,000,000). "
            "The page should either explicitly indicate the timeframe as March 2026 or be a contemporaneous resource "
            "from around that time. If multiple values are presented, use the primary/current figure."
        ),
        extra_prerequisites=[ref_nodes["market_cap"]]
    )

    # 2) Healthcare sector classification/operation
    leaf_sector = evaluator.add_leaf(
        id="healthcare_sector",
        desc="Company operates in / is classified in the healthcare sector.",
        parent=elig_group,
        critical=True
    )
    sector_claim = (
        f"{company_display} operates in or is classified within the healthcare/health care sector/industry "
        "(per widely used taxonomies such as GICS)."
    )
    await evaluator.verify(
        claim=sector_claim,
        node=leaf_sector,
        sources=info.ref_healthcare_sector_urls,
        additional_instruction=(
            "Allow reasonable synonyms: 'Healthcare', 'Health Care', 'Biopharma', 'Pharmaceuticals' (as a sub-industry), "
            "or equivalent sector classification indicating the company belongs to the healthcare sector."
        ),
        extra_prerequisites=[ref_nodes["healthcare_sector"]]
    )

    # 3) Top 10 healthcare by market cap
    leaf_top10 = evaluator.add_leaf(
        id="top_10_healthcare_by_market_cap",
        desc=f"Company ranks within the top 10 largest healthcare companies by market capitalization as of {AS_OF_TIMEFRAME}.",
        parent=elig_group,
        critical=True
    )
    top10_claim = (
        f"As of {AS_OF_TIMEFRAME}, {company_display} ranks within the top 10 largest healthcare companies by market capitalization."
    )
    await evaluator.verify(
        claim=top10_claim,
        node=leaf_top10,
        sources=info.ref_top10_marketcap_urls,
        additional_instruction=(
            "The supporting page should clearly indicate a ranking or list placing the company within the top 10 healthcare firms by market cap. "
            "Global or U.S.-focused rankings are acceptable as long as it's healthcare-specific."
        ),
        extra_prerequisites=[ref_nodes["top10"]]
    )

    # 4) Regular dividends
    leaf_div = evaluator.add_leaf(
        id="regular_dividends",
        desc="Company pays regular dividends to shareholders.",
        parent=elig_group,
        critical=True
    )
    div_claim = (
        f"{company_display} pays regular dividends to shareholders (e.g., quarterly)."
    )
    await evaluator.verify(
        claim=div_claim,
        node=leaf_div,
        sources=info.ref_dividends_urls,
        additional_instruction=(
            "Evidence might be an official investor relations dividend page, a press release showing recurring dividends, "
            "or a reputable financial data source listing regular dividend history."
        ),
        extra_prerequisites=[ref_nodes["dividends"]]
    )

    # 5) U.S. domicile and U.S. headquarters
    leaf_us = evaluator.add_leaf(
        id="us_domicile_and_headquarters",
        desc="Company is domiciled in the United States AND headquartered in the United States.",
        parent=elig_group,
        critical=True
    )
    us_claim = (
        f"{company_display} is domiciled (incorporated) in the United States and its headquarters are located in the United States."
    )
    await evaluator.verify(
        claim=us_claim,
        node=leaf_us,
        sources=info.ref_us_domicile_hq_urls,
        additional_instruction=(
            "Accept evidence such as 'Incorporated in Delaware (U.S.)' and headquarters address being in the U.S. "
            "Both conditions (domicile and HQ) must be satisfied."
        ),
        extra_prerequisites=[ref_nodes["us_domicile_hq"]]
    )

    # 6) Listed on NYSE or NASDAQ
    leaf_list = evaluator.add_leaf(
        id="listed_on_nyse_or_nasdaq",
        desc="Company is listed on NYSE or NASDAQ.",
        parent=elig_group,
        critical=True
    )
    list_claim = (
        f"{company_display} is listed on a major U.S. stock exchange: NYSE or NASDAQ."
    )
    await evaluator.verify(
        claim=list_claim,
        node=leaf_list,
        sources=info.ref_exchange_listing_urls,
        additional_instruction=(
            "Allow minor variants like 'NasdaqGS', 'Nasdaq Global Select Market', 'NYSE New York Stock Exchange'. "
            "Evidence can include the official exchange listing page, company IR page, or credible financial data sites."
        ),
        extra_prerequisites=[ref_nodes["listing"]]
    )

    # 7) Stock price ≥ $4 per share
    leaf_price = evaluator.add_leaf(
        id="stock_price_at_least_4",
        desc="Stock price is at least $4 per share.",
        parent=elig_group,
        critical=True
    )
    price_claim = (
        f"As of around {AS_OF_TIMEFRAME}, the stock price of {company_display} is at least $4 per share."
    )
    await evaluator.verify(
        claim=price_claim,
        node=leaf_price,
        sources=info.ref_stock_price_minimum_urls,
        additional_instruction=(
            "Accept reasonable contemporaneous pricing references (e.g., quote pages, historical charts) from around March 2026. "
            "If multiple share classes exist, use the class matching the given ticker. Accept 'last price', 'close', or 'current price' "
            "as long as it's clearly ≥ $4."
        ),
        extra_prerequisites=[ref_nodes["stock_price"]]
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Entry point for evaluating the company's compliance with all criteria.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates top-level groups in parallel
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

    # Extract structured information from the answer
    info: CompanyExtraction = await evaluator.extract(
        prompt=prompt_extract_company(),
        template_class=CompanyExtraction,
        extraction_name="company_extraction"
    )

    # Build top-level critical node for the whole evaluation
    company_node = evaluator.add_parallel(
        id="company_identification",
        desc="Identify one U.S.-based publicly traded healthcare company that satisfies all stated criteria as of March 2026, and provide the requested fields and references.",
        parent=root,
        critical=True
    )

    # 1) Required output fields present
    await build_required_output_nodes(evaluator, company_node, info)

    # 2) References existence checks (independent leaves; also used as prerequisites)
    ref_nodes = await build_references_nodes(evaluator, company_node, info)

    # 3) Eligibility criteria verification (web-grounded)
    await build_eligibility_nodes(evaluator, company_node, info, ref_nodes)

    # Add a small custom info block for context
    evaluator.add_custom_info(
        info={
            "as_of_timeframe": AS_OF_TIMEFRAME,
            "noted_company_name": info.company_name,
            "noted_ticker": info.stock_ticker,
            "noted_market_cap_value": info.market_cap_value,
        },
        info_type="context",
        info_name="evaluation_context"
    )

    return evaluator.get_summary()