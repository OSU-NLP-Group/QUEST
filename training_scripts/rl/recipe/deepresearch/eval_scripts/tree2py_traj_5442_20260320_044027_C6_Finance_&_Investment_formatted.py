import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "intl_equity_etf_selection"
TASK_DESCRIPTION = """
Identify an international equity ETF that satisfies all of the following criteria, and provide the requested specific information:

ETF Criteria:
1. The ETF must be issued by one of the following major providers: Vanguard, iShares (BlackRock), or State Street (SPDR)
2. The ETF's expense ratio must be 0.10% or lower
3. The ETF must invest primarily in international stocks (non-U.S. markets)
4. The ETF's geographic allocation to Europe must be between 35% and 40%, inclusive
5. The ETF must hold at least 5,000 individual stock holdings
6. The ETF must hold ASML Holding NV (the Netherlands-based semiconductor equipment manufacturer) with a portfolio weight of at least 1.0%
7. The ETF must hold at least one company that is both: (a) listed on the Oslo Stock Exchange (Oslo Børs) in Norway, and (b) operates in the energy sector (oil & gas)
8. The ETF must have a Morningstar Star Rating of either 4 stars or 5 stars

Required Information to Provide:
- The ETF's ticker symbol
- The ETF's full official name
- The ETF's exact expense ratio (as a percentage)
- The ETF's exact allocation to Europe (as a percentage)
- The exact portfolio weight of ASML Holding NV (as a percentage)
- The total number of individual stock holdings in the ETF
- The name of the Norwegian energy sector company held by the ETF
- The Morningstar Star Rating of the ETF

For each piece of numerical or factual data, provide a reference URL from an official or reputable source (such as the fund provider's website, Morningstar, or other established financial data providers) that confirms the information.
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ETFExtraction(BaseModel):
    # Basic identification
    ticker: Optional[str] = None
    ticker_source_urls: List[str] = Field(default_factory=list)

    official_name: Optional[str] = None
    official_name_source_urls: List[str] = Field(default_factory=list)

    provider: Optional[str] = None
    provider_source_urls: List[str] = Field(default_factory=list)

    # International focus
    international_focus_statement: Optional[str] = None
    international_focus_source_urls: List[str] = Field(default_factory=list)

    # Expense ratio
    expense_ratio_percent: Optional[str] = None
    expense_ratio_asof_date: Optional[str] = None
    expense_ratio_source_urls: List[str] = Field(default_factory=list)

    # Europe allocation
    europe_allocation_percent: Optional[str] = None
    europe_allocation_source_urls: List[str] = Field(default_factory=list)

    # Holdings count
    holdings_count: Optional[str] = None
    holdings_count_source_urls: List[str] = Field(default_factory=list)

    # ASML holding + weight
    asml_weight_percent: Optional[str] = None
    asml_holding_source_urls: List[str] = Field(default_factory=list)
    asml_weight_source_urls: List[str] = Field(default_factory=list)

    # Norwegian Oslo-listed energy company
    norwegian_energy_company_name: Optional[str] = None
    norwegian_company_holdings_source_urls: List[str] = Field(default_factory=list)
    norwegian_company_exchange_source_urls: List[str] = Field(default_factory=list)
    norwegian_company_sector_source_urls: List[str] = Field(default_factory=list)

    # Morningstar rating
    morningstar_star_rating: Optional[str] = None
    morningstar_rating_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etf_info() -> str:
    return """
Extract the following information exactly as presented in the answer and list the specific source URL(s) cited for each item (only URLs explicitly present in the answer; do not invent any):
1) ticker: ETF ticker symbol (string)
   ticker_source_urls: array of URL(s) that confirm the ticker

2) official_name: ETF's full official name (string)
   official_name_source_urls: array of URL(s) that confirm the official name

3) provider: ETF issuer/provider name (e.g., Vanguard, iShares, BlackRock, State Street, SPDR)
   provider_source_urls: array of URL(s) that confirm the issuer/provider

4) international_focus_statement: the phrase/sentence in the answer indicating the ETF primarily invests in non-U.S. (international) equities (string; can paraphrase from the answer)
   international_focus_source_urls: array of URL(s) that confirm the non-U.S. focus

5) expense_ratio_percent: exact expense ratio string as written in the answer (include symbol if present, e.g., "0.07%")
   expense_ratio_asof_date: the 'as of' or 'last updated' date for the expense ratio if provided (string)
   expense_ratio_source_urls: array of URL(s) that confirm the expense ratio

6) europe_allocation_percent: exact percentage of geographic allocation to Europe as written (e.g., "37.5%")
   europe_allocation_source_urls: array of URL(s) that confirm the Europe allocation

7) holdings_count: total number of individual stock holdings (string form as in the answer, e.g., "8,200")
   holdings_count_source_urls: array of URL(s) that confirm the total count

8) asml_weight_percent: exact portfolio weight of ASML Holding NV (e.g., "1.2%")
   asml_holding_source_urls: array of URL(s) that show ASML is a holding of the ETF (e.g., fund holdings page)
   asml_weight_source_urls: array of URL(s) that confirm the exact ASML weight

9) norwegian_energy_company_name: the name of at least one Norwegian energy (oil & gas) company the ETF holds (e.g., "Equinor ASA")
   norwegian_company_holdings_source_urls: array of URL(s) showing the ETF holds this company
   norwegian_company_exchange_source_urls: array of URL(s) confirming the company is listed on the Oslo Stock Exchange (Oslo Børs)
   norwegian_company_sector_source_urls: array of URL(s) confirming the company operates in the energy (oil & gas) sector

10) morningstar_star_rating: the Morningstar Star Rating (string; e.g., "4 stars", "****", or "4")
    morningstar_rating_source_urls: array of URL(s) that confirm the Morningstar Star Rating

Rules:
- Extract only what appears in the answer text. If any item is missing, set it to null (or empty array for URLs).
- Return the URLs exactly as shown in the answer (plain or markdown links are acceptable; extract the actual URL).
- Do not infer or fabricate values or URLs.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _parse_first_float(text: str) -> Optional[float]:
    m = re.search(r"([-+]?\d*\.?\d+)", text.replace(",", ""))
    return float(m.group(1)) if m else None


def parse_percentage(value: Optional[str]) -> Optional[float]:
    """
    Parse a percentage string to a float in percentage units.
    Examples:
    - "0.07%" -> 0.07
    - "37.5%" -> 37.5
    - "7 bps" -> 0.07
    - "37.5"  -> 37.5 (assume already in percent units)
    """
    if not value:
        return None
    s = value.strip().lower()
    if "bps" in s:
        num = _parse_first_float(s)
        return num / 100.0 if num is not None else None  # 100 bps = 1.0%
    num = _parse_first_float(s)
    if num is None:
        return None
    return num if "%" not in s else num  # already in percent units


def parse_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    m = re.search(r"(\d[\d,\.]*)", value)
    if not m:
        return None
    cleaned = m.group(1).replace(",", "")
    try:
        # Prefer integer; if decimal present, floor by int conversion
        return int(float(cleaned))
    except Exception:
        return None


def parse_rating(value: Optional[str]) -> Optional[int]:
    """
    Parse Morningstar Star Rating to 1-5 integer if possible.
    Supports formats like "4", "4 stars", "****", "★★★★", etc.
    """
    if not value:
        return None
    s = value.strip().lower()
    # Try explicit digit
    m = re.search(r"\b([1-5])\b", s)
    if m:
        return int(m.group(1))
    # Count star characters
    count_black = s.count("★")
    if count_black in {1, 2, 3, 4, 5}:
        return count_black
    count_ascii = s.count("*")
    if count_ascii in {1, 2, 3, 4, 5}:
        return count_ascii
    return None


def non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()]


def provider_is_allowed(name: Optional[str]) -> bool:
    if not name:
        return False
    n = name.strip().lower()
    allowed_keywords = ["vanguard", "ishares", "blackrock", "state street", "spdr"]
    return any(k in n for k in allowed_keywords)


async def verify_with_required_sources(
    evaluator: Evaluator,
    node,
    claim: str,
    sources: Optional[List[str]],
    additional_instruction: str = "None"
) -> bool:
    """
    Helper: If sources are empty, mark node failed directly; otherwise call evaluator.verify with URLs.
    """
    srcs = non_empty_urls(sources)
    if len(srcs) == 0:
        node.score = 0.0
        node.status = "failed"
        return False
    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=srcs,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_basic_identification(evaluator: Evaluator, parent, data: ETFExtraction):
    node = evaluator.add_parallel(
        id="ETF_Basic_Identification",
        desc="Provide ETF identifiers and verify issuer is allowed",
        parent=parent,
        critical=True,
    )

    # 1) Ticker symbol provided
    evaluator.add_custom_node(
        result=bool(data.ticker and data.ticker.strip()),
        id="Ticker_Symbol_Provided",
        desc="Provide the ETF ticker symbol",
        parent=node,
        critical=True
    )

    # 2) Ticker symbol source URL provided (existence of at least one URL)
    evaluator.add_custom_node(
        result=len(non_empty_urls(data.ticker_source_urls)) > 0,
        id="Ticker_Symbol_Source_URL_Provided",
        desc="Provide a reference URL confirming the ETF ticker symbol",
        parent=node,
        critical=True
    )

    # 3) Official name provided
    evaluator.add_custom_node(
        result=bool(data.official_name and data.official_name.strip()),
        id="Official_Name_Provided",
        desc="Provide the ETF full official name",
        parent=node,
        critical=True
    )

    # 4) Official name source URL provided
    evaluator.add_custom_node(
        result=len(non_empty_urls(data.official_name_source_urls)) > 0,
        id="Official_Name_Source_URL_Provided",
        desc="Provide a reference URL confirming the ETF official name",
        parent=node,
        critical=True
    )

    # 5) Provider is allowed
    evaluator.add_custom_node(
        result=provider_is_allowed(data.provider),
        id="Provider_Is_Allowed",
        desc="ETF issuer/provider is one of Vanguard, iShares (BlackRock), or State Street (SPDR)",
        parent=node,
        critical=True
    )

    # 6) Provider source URL provided
    evaluator.add_custom_node(
        result=len(non_empty_urls(data.provider_source_urls)) > 0,
        id="Provider_Source_URL_Provided",
        desc="Provide a reference URL confirming the ETF issuer/provider",
        parent=node,
        critical=True
    )


async def build_international_exposure(evaluator: Evaluator, parent, data: ETFExtraction):
    node = evaluator.add_parallel(
        id="International_Equity_Exposure",
        desc="Verify the ETF primarily invests in non-U.S. (international) equities",
        parent=parent,
        critical=True
    )

    # URL provided (existence) first for gating
    evaluator.add_custom_node(
        result=len(non_empty_urls(data.international_focus_source_urls)) > 0,
        id="International_Exposure_Source_URL_Provided",
        desc="Provide a reference URL confirming the ETF's international/non-U.S. equity focus",
        parent=node,
        critical=True
    )

    # Main verification using the URL(s)
    intl_leaf = evaluator.add_leaf(
        id="International_Stocks_Primary",
        desc="ETF invests primarily in international stocks (non-U.S. markets)",
        parent=node,
        critical=True
    )
    claim = "This ETF primarily invests in international (non-U.S.) equities."
    await evaluator.verify(
        claim=claim,
        node=intl_leaf,
        sources=non_empty_urls(data.international_focus_source_urls),
        additional_instruction="Confirm that the fund strategy or description explicitly states non-U.S. or international equity focus. Allow synonyms like 'developed ex-U.S.', 'non-U.S. markets', or 'global ex-U.S.'."
    )


async def build_expense_ratio(evaluator: Evaluator, parent, data: ETFExtraction):
    node = evaluator.add_sequential(
        id="Expense_Ratio",
        desc="Provide and verify the expense ratio requirement",
        parent=parent,
        critical=True
    )

    # Provide exact expense ratio (existence)
    evaluator.add_custom_node(
        result=bool(data.expense_ratio_percent and data.expense_ratio_percent.strip()),
        id="Provide_Exact_Expense_Ratio",
        desc="Provide the ETF's exact expense ratio (percentage)",
        parent=node,
        critical=True
    )

    # Verify expense ratio value via source URL(s)
    exp_src_leaf = evaluator.add_leaf(
        id="Expense_Ratio_Source_URL_Provided",
        desc="Provide a reference URL confirming the expense ratio",
        parent=node,
        critical=True
    )
    await verify_with_required_sources(
        evaluator,
        exp_src_leaf,
        claim=f"The fund's expense ratio is {data.expense_ratio_percent}.",
        sources=data.expense_ratio_source_urls,
        additional_instruction="Verify the current net expense ratio if multiple are shown. Allow minor formatting variations (e.g., '0.07%' vs '0.07 %')."
    )

    # As-of date provided (existence)
    evaluator.add_custom_node(
        result=bool(data.expense_ratio_asof_date and data.expense_ratio_asof_date.strip()),
        id="Expense_Ratio_Source_AsOf_Date_Provided",
        desc="Provide the as-of date / publication date / last-updated date for the cited expense ratio data (to support that it is based on the most recent available data)",
        parent=node,
        critical=True
    )

    # Threshold check: <= 0.10%
    exp_val = parse_percentage(data.expense_ratio_percent)
    evaluator.add_custom_node(
        result=(exp_val is not None and exp_val <= 0.10),
        id="Expense_Ratio_At_Most_010",
        desc="Expense ratio is 0.10% or lower",
        parent=node,
        critical=True
    )


async def build_europe_allocation(evaluator: Evaluator, parent, data: ETFExtraction):
    node = evaluator.add_sequential(
        id="Europe_Allocation",
        desc="Provide and verify Europe allocation requirement",
        parent=parent,
        critical=True
    )

    # Provide exact Europe allocation (existence)
    evaluator.add_custom_node(
        result=bool(data.europe_allocation_percent and data.europe_allocation_percent.strip()),
        id="Provide_Exact_Europe_Allocation",
        desc="Provide the ETF's exact allocation to Europe (percentage)",
        parent=node,
        critical=True
    )

    # Verify Europe allocation value via source URL(s)
    eu_src_leaf = evaluator.add_leaf(
        id="Europe_Allocation_Source_URL_Provided",
        desc="Provide a reference URL confirming the Europe allocation",
        parent=node,
        critical=True
    )
    await verify_with_required_sources(
        evaluator,
        eu_src_leaf,
        claim=f"The fund's geographic allocation to Europe is {data.europe_allocation_percent}.",
        sources=data.europe_allocation_source_urls,
        additional_instruction="Confirm the region allocation specifically for 'Europe'. If multiple timeframes or share classes are shown, use the one consistent with the answer."
    )

    # Threshold check: between 35% and 40% inclusive
    eu_val = parse_percentage(data.europe_allocation_percent)
    evaluator.add_custom_node(
        result=(eu_val is not None and 35.0 <= eu_val <= 40.0),
        id="Europe_Allocation_In_Range",
        desc="Europe allocation is between 35% and 40% inclusive",
        parent=node,
        critical=True
    )


async def build_holdings_count(evaluator: Evaluator, parent, data: ETFExtraction):
    node = evaluator.add_sequential(
        id="Holdings_Count",
        desc="Provide and verify total holdings count requirement",
        parent=parent,
        critical=True
    )

    # Provide total holdings (existence)
    evaluator.add_custom_node(
        result=bool(data.holdings_count and data.holdings_count.strip()),
        id="Provide_Total_Holdings_Count",
        desc="Provide the total number of individual stock holdings in the ETF",
        parent=node,
        critical=True
    )

    # Verify holdings count via source URL(s)
    hold_src_leaf = evaluator.add_leaf(
        id="Holdings_Count_Source_URL_Provided",
        desc="Provide a reference URL confirming the total holdings count",
        parent=node,
        critical=True
    )
    await verify_with_required_sources(
        evaluator,
        hold_src_leaf,
        claim=f"The ETF holds {data.holdings_count} individual stock holdings.",
        sources=data.holdings_count_source_urls,
        additional_instruction="Match the total number of equity holdings; allow minor rounding differences if source explicitly states 'approximately'."
    )

    # Threshold check: at least 5,000
    count_val = parse_int(data.holdings_count)
    evaluator.add_custom_node(
        result=(count_val is not None and count_val >= 5000),
        id="Holdings_Count_At_Least_5000",
        desc="ETF holds at least 5,000 individual stock holdings",
        parent=node,
        critical=True
    )


async def build_asml(evaluator: Evaluator, parent, data: ETFExtraction):
    node = evaluator.add_sequential(
        id="ASML_Holding_And_Weight",
        desc="Verify ASML is held and meets minimum weight; provide exact weight with source",
        parent=parent,
        critical=True
    )

    # We'll add both nodes first, then verify ASML_Is_Held with the existence node as an extra prerequisite
    asml_held_leaf = evaluator.add_leaf(
        id="ASML_Is_Held",
        desc="ETF holds ASML Holding NV as a portfolio holding",
        parent=node,
        critical=True
    )

    asml_holding_src_exist = evaluator.add_custom_node(
        result=len(non_empty_urls(data.asml_holding_source_urls) + non_empty_urls(data.asml_weight_source_urls)) > 0,
        id="ASML_Holding_Source_URL_Provided",
        desc="Provide a reference URL confirming ASML Holding NV is a holding of the ETF",
        parent=node,
        critical=True
    )

    # Now verify ASML held using combined URLs, gated by existence node
    await evaluator.verify(
        claim="The ETF holds ASML Holding NV (also referred to as ASML Holding N.V. or ASML) as a portfolio holding.",
        node=asml_held_leaf,
        sources=non_empty_urls(data.asml_holding_source_urls) + non_empty_urls(data.asml_weight_source_urls),
        additional_instruction="Accept reasonable name variants: 'ASML Holding NV', 'ASML Holding N.V.', or simply 'ASML'.",
        extra_prerequisites=[asml_holding_src_exist]
    )

    # Provide exact ASML weight (existence)
    evaluator.add_custom_node(
        result=bool(data.asml_weight_percent and data.asml_weight_percent.strip()),
        id="Provide_Exact_ASML_Weight",
        desc="Provide the exact portfolio weight of ASML Holding NV (percentage)",
        parent=node,
        critical=True
    )

    # Verify ASML weight via dedicated URL(s)
    asml_weight_leaf = evaluator.add_leaf(
        id="ASML_Weight_Source_URL_Provided",
        desc="Provide a reference URL confirming ASML Holding NV's portfolio weight in the ETF",
        parent=node,
        critical=True
    )
    await verify_with_required_sources(
        evaluator,
        asml_weight_leaf,
        claim=f"The portfolio weight of ASML Holding NV in the ETF is {data.asml_weight_percent}.",
        sources=data.asml_weight_source_urls,
        additional_instruction="Confirm the security weight (% of net assets). Allow minor rounding differences (e.g., 1.02% vs 1.0%)."
    )

    # Threshold check: ASML weight >= 1.0%
    asml_w = parse_percentage(data.asml_weight_percent)
    evaluator.add_custom_node(
        result=(asml_w is not None and asml_w >= 1.0),
        id="ASML_Weight_At_Least_1pct",
        desc="ASML Holding NV weight is at least 1.0%",
        parent=node,
        critical=True
    )


async def build_norwegian_energy(evaluator: Evaluator, parent, data: ETFExtraction):
    node = evaluator.add_sequential(
        id="Norwegian_Oslo_Listed_Energy_Company",
        desc="Verify the ETF holds at least one Oslo Stock Exchange-listed Norwegian oil & gas company and provide its name with sources",
        parent=parent,
        critical=True
    )

    # Provide company name (existence)
    evaluator.add_custom_node(
        result=bool(data.norwegian_energy_company_name and data.norwegian_energy_company_name.strip()),
        id="Provide_Norwegian_Energy_Company_Name",
        desc="Provide the name of the Norwegian energy (oil & gas) company held by the ETF",
        parent=node,
        critical=True
    )

    # Verify ETF holds the named company
    holds_leaf = evaluator.add_leaf(
        id="ETF_Holds_Named_Company_Source_URL_Provided",
        desc="Provide a reference URL confirming the ETF holds the named Norwegian company",
        parent=node,
        critical=True
    )
    await verify_with_required_sources(
        evaluator,
        holds_leaf,
        claim=f"The ETF holds {data.norwegian_energy_company_name}.",
        sources=data.norwegian_company_holdings_source_urls,
        additional_instruction="Confirm on the fund's holdings page or reputable data provider that the specified Norwegian company is included."
    )

    # Verify the company is listed on Oslo Stock Exchange
    oslo_leaf = evaluator.add_leaf(
        id="Company_Is_Oslo_Listed_Source_URL_Provided",
        desc="Provide a reference URL confirming the named company is listed on the Oslo Stock Exchange (Oslo Børs)",
        parent=node,
        critical=True
    )
    await verify_with_required_sources(
        evaluator,
        oslo_leaf,
        claim=f"The company {data.norwegian_energy_company_name} is listed on the Oslo Stock Exchange (Oslo Børs).",
        sources=data.norwegian_company_exchange_source_urls,
        additional_instruction="Accept mentions such as 'Oslo Børs', 'Oslo Stock Exchange', 'OSE', or Norwegian ticker suffixes like '.OL' that indicate Oslo listing."
    )

    # Verify the company operates in energy (oil & gas)
    sector_leaf = evaluator.add_leaf(
        id="Company_Is_Energy_Oil_And_Gas_Source_URL_Provided",
        desc="Provide a reference URL confirming the named company operates in the energy sector (oil & gas)",
        parent=node,
        critical=True
    )
    await verify_with_required_sources(
        evaluator,
        sector_leaf,
        claim=f"The company {data.norwegian_energy_company_name} operates in the energy sector (oil & gas).",
        sources=data.norwegian_company_sector_source_urls,
        additional_instruction="Accept classifications like 'Energy', 'Oil & Gas', 'Energy—Integrated Oil & Gas', etc."
    )


async def build_morningstar(evaluator: Evaluator, parent, data: ETFExtraction):
    node = evaluator.add_sequential(
        id="Morningstar_Star_Rating",
        desc="Provide and verify Morningstar rating requirement",
        parent=parent,
        critical=True
    )

    # Provide rating (existence)
    evaluator.add_custom_node(
        result=bool(data.morningstar_star_rating and data.morningstar_star_rating.strip()),
        id="Provide_Morningstar_Star_Rating",
        desc="Provide the ETF's Morningstar Star Rating",
        parent=node,
        critical=True
    )

    # Verify rating via source URL(s)
    rating_int = parse_rating(data.morningstar_star_rating)
    rating_leaf = evaluator.add_leaf(
        id="Morningstar_Rating_Source_URL_Provided",
        desc="Provide a reference URL confirming the Morningstar Star Rating",
        parent=node,
        critical=True
    )
    if rating_int is not None:
        claim = f"The ETF's Morningstar Star Rating is {rating_int} stars."
    else:
        claim = f"The ETF's Morningstar Star Rating is '{data.morningstar_star_rating}'."
    await verify_with_required_sources(
        evaluator,
        rating_leaf,
        claim=claim,
        sources=data.morningstar_rating_source_urls,
        additional_instruction="Confirm the Morningstar Star Rating displayed (overall rating). If multiple share classes/ratings exist, use the one consistent with the answer."
    )

    # Threshold: 4 or 5 stars
    evaluator.add_custom_node(
        result=(rating_int in {4, 5}),
        id="Rating_Is_4_Or_5",
        desc="Morningstar Star Rating is either 4 stars or 5 stars",
        parent=node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'international equity ETF' criteria task using obj_task_eval.
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
        default_model=model
    )

    # 1) Extraction
    etf_data: ETFExtraction = await evaluator.extract(
        prompt=prompt_extract_etf_info(),
        template_class=ETFExtraction,
        extraction_name="etf_extraction"
    )

    # 2) Add a critical top-level node aggregating all checks
    top = evaluator.add_parallel(
        id="ETF_Identification",
        desc="Identify an international equity ETF meeting all specified criteria and provide all requested information with references",
        parent=root,
        critical=True
    )

    # 3) Build subtrees
    await build_basic_identification(evaluator, top, etf_data)
    await build_international_exposure(evaluator, top, etf_data)
    await build_expense_ratio(evaluator, top, etf_data)
    await build_europe_allocation(evaluator, top, etf_data)
    await build_holdings_count(evaluator, top, etf_data)
    await build_asml(evaluator, top, etf_data)
    await build_norwegian_energy(evaluator, top, etf_data)
    await build_morningstar(evaluator, top, etf_data)

    # 4) Record parsed values (for transparency/debug)
    evaluator.add_custom_info(
        info={
            "parsed_values": {
                "expense_ratio_percent_num": parse_percentage(etf_data.expense_ratio_percent),
                "europe_allocation_percent_num": parse_percentage(etf_data.europe_allocation_percent),
                "asml_weight_percent_num": parse_percentage(etf_data.asml_weight_percent),
                "holdings_count_int": parse_int(etf_data.holdings_count),
                "morningstar_star_rating_int": parse_rating(etf_data.morningstar_star_rating),
                "provider_allowed": provider_is_allowed(etf_data.provider),
            }
        },
        info_type="parsed_debug_info"
    )

    # 5) Return evaluation summary
    return evaluator.get_summary()