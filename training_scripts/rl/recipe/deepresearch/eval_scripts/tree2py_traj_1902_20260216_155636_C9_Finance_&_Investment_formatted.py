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
TASK_ID = "financial_planning_practice_setup"
TASK_DESCRIPTION = """A newly established financial planning practice is developing comprehensive investment and retirement strategies for four distinct client profiles. For each client scenario described below, identify the specific financial products, contribution limits, and professional requirements that apply:

Client 1 - Young Professional:
Age 28, annual income $85,000, seeking aggressive growth with low-cost index investing, no existing retirement savings.
- Identify a specific S&P 500 ETF with an expense ratio at or below 0.03%
- Calculate the maximum 2026 employee contribution limit for a 401(k) plan
- Calculate the maximum 2026 IRA contribution limit

Client 2 - Mid-Career Professional:
Age 52, annual income $145,000, moderate risk tolerance, wants broader market exposure beyond the S&P 500, has existing retirement savings.
- Identify a specific total U.S. stock market ETF that holds at least 3,000 individual stocks and includes small, mid, and large-cap companies
- Calculate the maximum total 2026 employee contribution limit for a 401(k) plan including catch-up contributions
- Calculate the maximum total 2026 IRA contribution limit including catch-up contributions

Client 3 - Pre-Retiree Seeking Income:
Age 58, seeking income-focused investments from companies with exceptional dividend track records.
- Identify one specific company that is both an S&P 500 Dividend Aristocrat AND has increased its dividend for at least 69 consecutive years
- State the minimum number of consecutive years of dividend increases required to qualify as a Dividend Aristocrat
- Recommend an appropriate conservative asset allocation percentage range for stocks and bonds as this client approaches retirement

Client 4 - Recent Graduate:
Age 24, annual income $55,000, just starting career, interested in target date funds and wants to understand advisor qualifications.
- Identify an appropriate target date fund for someone planning to retire around 2060 and specify its approximate current equity allocation percentage
- State the number of questions, duration in hours, and minimum passing score percentage for the Series 65 examination
- State the total continuing education hours required for CFP® professionals every 2 years and how many of those hours must be devoted to Ethics

For each client scenario, provide the specific information requested along with reference URLs that verify each piece of information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFInfo(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    provider: Optional[str] = None
    index_tracked: Optional[str] = None
    expense_ratio: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ETFInfoExt(ETFInfo):
    holdings_count: Optional[str] = None
    market_coverage_desc: Optional[str] = None


class ContributionInfo(BaseModel):
    amount: Optional[str] = None  # e.g., "$24,500"
    breakdown_note: Optional[str] = None  # e.g., "base $24,500; catch-up $8,000"
    sources: List[str] = Field(default_factory=list)


class DividendCompanyInfo(BaseModel):
    name: Optional[str] = None
    consecutive_years_increases: Optional[str] = None  # e.g., "69"
    sp500_membership_note: Optional[str] = None  # e.g., "S&P 500 member"
    sources: List[str] = Field(default_factory=list)


class AristocratDefinitionInfo(BaseModel):
    min_years: Optional[str] = None  # e.g., "25"
    index_requirement: Optional[str] = None  # e.g., "Must be S&P 500 member"
    sources: List[str] = Field(default_factory=list)


class AssetAllocationInfo(BaseModel):
    stock_range: Optional[str] = None  # e.g., "40-60%"
    bond_range: Optional[str] = None   # e.g., "40-60%"
    sources: List[str] = Field(default_factory=list)


class TargetDateFundInfo(BaseModel):
    name: Optional[str] = None
    target_year: Optional[str] = None  # e.g., "2060"
    current_equity_pct: Optional[str] = None  # e.g., "95%"
    sources: List[str] = Field(default_factory=list)


class Series65Info(BaseModel):
    questions: Optional[str] = None  # e.g., "130"
    duration_hours: Optional[str] = None  # e.g., "3"
    passing_score_percent: Optional[str] = None  # e.g., "72%"
    sources: List[str] = Field(default_factory=list)


class CFPCEInfo(BaseModel):
    total_hours_2_years: Optional[str] = None  # e.g., "30"
    ethics_hours: Optional[str] = None  # e.g., "2"
    sources: List[str] = Field(default_factory=list)


class FinancialPlanExtraction(BaseModel):
    client1_sp500_etf: Optional[ETFInfo] = None
    client1_401k_limit_2026: Optional[ContributionInfo] = None
    client1_ira_limit_2026: Optional[ContributionInfo] = None

    client2_total_market_etf: Optional[ETFInfoExt] = None
    client2_401k_total_limit_2026: Optional[ContributionInfo] = None
    client2_ira_total_limit_2026: Optional[ContributionInfo] = None

    client3_company: Optional[DividendCompanyInfo] = None
    client3_aristocrat_definition: Optional[AristocratDefinitionInfo] = None
    client3_allocation: Optional[AssetAllocationInfo] = None

    client4_tdf: Optional[TargetDateFundInfo] = None
    client4_series65: Optional[Series65Info] = None
    client4_cfp_ce: Optional[CFPCEInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_financial_plan() -> str:
    return """
    Extract the structured information for each of the four client scenarios from the answer. Only extract details that are explicitly stated in the answer. For every numeric or factual item, include the source URLs cited in the answer that support the item.

    Field-by-field extraction schema:
    - client1_sp500_etf:
        • name: ETF name
        • ticker: ETF ticker
        • provider: fund family (e.g., Vanguard, iShares)
        • index_tracked: name of the index (must be S&P 500)
        • expense_ratio: stated expense ratio string (e.g., "0.03%")
        • sources: URLs in the answer that confirm the ETF's characteristics

    - client1_401k_limit_2026:
        • amount: employee 401(k) contribution limit for 2026 for under age 50 (string, e.g., "$24,500")
        • breakdown_note: any text describing base and catch-up if mentioned
        • sources: URLs confirming the 2026 401(k) base limit

    - client1_ira_limit_2026:
        • amount: IRA contribution limit for 2026 for under age 50 (string, e.g., "$7,500")
        • breakdown_note: any note if present
        • sources: URLs confirming the 2026 IRA base limit

    - client2_total_market_etf:
        • name, ticker, provider, index_tracked: ETF identifiers
        • holdings_count: number of holdings (string, e.g., "3,800")
        • market_coverage_desc: text indicating coverage of small, mid, and large caps
        • expense_ratio: if given
        • sources: URLs confirming total market coverage and holdings count

    - client2_401k_total_limit_2026:
        • amount: total employee 401(k) contribution limit for 2026 including catch-up for age 50+ (string, e.g., "$32,500")
        • breakdown_note: include base and catch-up amounts if stated (e.g., "$24,500 base + $8,000 catch-up")
        • sources: URLs confirming both base and catch-up limits

    - client2_ira_total_limit_2026:
        • amount: total IRA contribution limit for 2026 including catch-up for age 50+ (string, e.g., "$8,600")
        • breakdown_note: include base and catch-up amounts if stated (e.g., "$7,500 base + $1,100 catch-up")
        • sources: URLs confirming both base and catch-up IRA limits

    - client3_company:
        • name: company name
        • consecutive_years_increases: number of consecutive years of dividend increases (string)
        • sp500_membership_note: text indicating S&P 500 membership
        • sources: URLs confirming dividend streak and S&P 500 membership

    - client3_aristocrat_definition:
        • min_years: minimum consecutive years required (string, e.g., "25")
        • index_requirement: statement that S&P 500 membership is required
        • sources: URLs confirming Dividend Aristocrat requirements

    - client3_allocation:
        • stock_range: stock allocation range recommended for conservative pre-retiree (string, e.g., "40-60%")
        • bond_range: bond allocation range recommended (string, e.g., "40-60%")
        • sources: URLs supporting the recommended allocation

    - client4_tdf:
        • name: target date fund name
        • target_year: target year (string, e.g., "2060")
        • current_equity_pct: current equity allocation percentage (string, e.g., "95%")
        • sources: URLs confirming fund year and allocation

    - client4_series65:
        • questions: number of questions (string, e.g., "130")
        • duration_hours: duration in hours (string, e.g., "3")
        • passing_score_percent: passing score percentage (string, e.g., "72%")
        • sources: URLs confirming Series 65 exam details

    - client4_cfp_ce:
        • total_hours_2_years: total CE hours every 2 years (string, e.g., "30")
        • ethics_hours: required Ethics hours out of total (string, e.g., "2")
        • sources: URLs confirming CFP® CE requirements

    Rules:
    - Only include URLs explicitly present in the answer text. If a field is not provided, set it to null and use an empty sources list.
    - Keep values as strings. Do not infer or calculate new numbers; extract what's stated.
    - For sources, include all URLs linked for the specific item. If none are provided for that item, return an empty list for sources.
    """


# --------------------------------------------------------------------------- #
# Verification helpers per client                                             #
# --------------------------------------------------------------------------- #
async def verify_client_1(evaluator: Evaluator, root_node, data: FinancialPlanExtraction) -> None:
    client_node = evaluator.add_parallel(
        id="Client_1_Young_Professional",
        desc="Correctly address all requirements for Client 1: Age 28, annual income $85,000, seeking aggressive growth, no retirement savings yet",
        parent=root_node,
        critical=False
    )

    # ETF selection (S&P 500, <=0.03% ER)
    etf_sel_node = evaluator.add_parallel(
        id="Client_1_ETF_Selection",
        desc="Identify an appropriate low-cost S&P 500 ETF for this client's aggressive growth strategy",
        parent=client_node,
        critical=False  # allow non-critical child provider; critical leaves gate result
    )
    sp500 = data.client1_sp500_etf or ETFInfo()

    # Expense ratio <= 0.03%
    er_leaf = evaluator.add_leaf(
        id="Client_1_ETF_Expense_Ratio",
        desc="The selected ETF must have an expense ratio at or below 0.03%",
        parent=etf_sel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF {sp500.name or 'unknown'} ({sp500.ticker or 'unknown'}) has an expense ratio at or below 0.03%.",
        node=er_leaf,
        sources=sp500.sources,
        additional_instruction="Confirm the ETF's stated expense ratio on the official/provider fund page or fact sheet. Accept if 0.03% or lower."
    )

    # Tracks S&P 500 Index
    idx_leaf = evaluator.add_leaf(
        id="Client_1_ETF_Index_Tracking",
        desc="The ETF must track the S&P 500 Index specifically",
        parent=etf_sel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF {sp500.name or 'unknown'} ({sp500.ticker or 'unknown'}) tracks the S&P 500 Index.",
        node=idx_leaf,
        sources=sp500.sources,
        additional_instruction="Verify the fund objective/benchmark indicates S&P 500."
    )

    # Provider (non-critical)
    prov_leaf = evaluator.add_leaf(
        id="Client_1_ETF_Provider",
        desc="Identify the ETF provider (fund family)",
        parent=etf_sel_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The ETF {sp500.name or 'unknown'} ({sp500.ticker or 'unknown'}) is provided by {sp500.provider or 'unknown provider'}.",
        node=prov_leaf,
        sources=sp500.sources,
        additional_instruction="Check the fund page header to confirm the provider (e.g., Vanguard, iShares, Schwab, Fidelity)."
    )

    # Reference validity
    etf_ref_leaf = evaluator.add_leaf(
        id="Client_1_ETF_Reference",
        desc="Provide a valid URL confirming the ETF's characteristics",
        parent=etf_sel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided source(s) are the official/provider page(s) for {sp500.name or 'unknown'} ({sp500.ticker or 'unknown'}) and confirm both the S&P 500 tracking and the expense ratio.",
        node=etf_ref_leaf,
        sources=sp500.sources,
        additional_instruction="Accept provider official pages or fund fact sheets that show benchmark and expense ratio."
    )

    # 401(k) 2026 base limit
    k401_node = evaluator.add_parallel(
        id="Client_1_401k_Contribution",
        desc="Calculate the maximum 2026 employee contribution limit for 401(k) for someone under age 50",
        parent=client_node,
        critical=False
    )
    k401 = data.client1_401k_limit_2026 or ContributionInfo()
    k401_base_leaf = evaluator.add_leaf(
        id="Client_1_401k_Base_Limit",
        desc="The base 401(k) contribution limit for 2026 must be correctly identified as $24,500",
        parent=k401_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2026 401(k) employee contribution limit (under age 50) is $24,500.",
        node=k401_base_leaf,
        sources=k401.sources,
        additional_instruction="Confirm the IRS/official source shows the $24,500 employee deferral limit for 2026."
    )

    k401_ref_leaf = evaluator.add_leaf(
        id="Client_1_401k_Reference",
        desc="Provide a valid URL confirming the 2026 401(k) contribution limit",
        parent=k401_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided source(s) confirm the 2026 401(k) employee contribution limit is $24,500.",
        node=k401_ref_leaf,
        sources=k401.sources,
        additional_instruction="Prefer IRS pages or reputable financial institutions' summaries citing the limit."
    )

    # IRA 2026 base limit
    ira_node = evaluator.add_parallel(
        id="Client_1_IRA_Contribution",
        desc="Calculate the maximum 2026 IRA contribution limit for someone under age 50",
        parent=client_node,
        critical=False
    )
    ira = data.client1_ira_limit_2026 or ContributionInfo()
    ira_base_leaf = evaluator.add_leaf(
        id="Client_1_IRA_Base_Limit",
        desc="The base IRA contribution limit for 2026 must be correctly identified as $7,500",
        parent=ira_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2026 IRA contribution limit (under age 50) is $7,500.",
        node=ira_base_leaf,
        sources=ira.sources,
        additional_instruction="Confirm the IRS/official source shows the $7,500 IRA contribution limit for 2026."
    )

    ira_ref_leaf = evaluator.add_leaf(
        id="Client_1_IRA_Reference",
        desc="Provide a valid URL confirming the 2026 IRA contribution limit",
        parent=ira_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided source(s) confirm the 2026 IRA contribution limit is $7,500.",
        node=ira_ref_leaf,
        sources=ira.sources,
        additional_instruction="Prefer IRS pages or reputable financial institutions' summaries."
    )


async def verify_client_2(evaluator: Evaluator, root_node, data: FinancialPlanExtraction) -> None:
    client_node = evaluator.add_parallel(
        id="Client_2_Mid_Career",
        desc="Correctly address all requirements for Client 2: Age 52, annual income $145,000, moderate risk tolerance, existing retirement savings",
        parent=root_node,
        critical=False
    )

    # Total market ETF selection
    etf_sel_node = evaluator.add_parallel(
        id="Client_2_ETF_Selection",
        desc="Identify an appropriate total U.S. stock market ETF for diversification beyond the S&P 500",
        parent=client_node,
        critical=False
    )
    tmarket = data.client2_total_market_etf or ETFInfoExt()

    coverage_leaf = evaluator.add_leaf(
        id="Client_2_ETF_Market_Coverage",
        desc="The ETF must track the total U.S. stock market, including small, mid, and large-cap stocks",
        parent=etf_sel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF {tmarket.name or 'unknown'} ({tmarket.ticker or 'unknown'}) tracks the total U.S. stock market and includes small-, mid-, and large-cap stocks.",
        node=coverage_leaf,
        sources=tmarket.sources,
        additional_instruction="Verify benchmark and description indicate full market coverage across capitalization tiers."
    )

    holdings_leaf = evaluator.add_leaf(
        id="Client_2_ETF_Holdings_Count",
        desc="The ETF must hold at least 3,000 individual stocks",
        parent=etf_sel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF {tmarket.name or 'unknown'} ({tmarket.ticker or 'unknown'}) holds at least 3,000 individual stocks.",
        node=holdings_leaf,
        sources=tmarket.sources,
        additional_instruction="Confirm holdings count or approximate holdings shown exceeds 3,000."
    )

    prov_leaf = evaluator.add_leaf(
        id="Client_2_ETF_Provider",
        desc="Identify the ETF provider (fund family)",
        parent=etf_sel_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The ETF {tmarket.name or 'unknown'} ({tmarket.ticker or 'unknown'}) is provided by {tmarket.provider or 'unknown provider'}.",
        node=prov_leaf,
        sources=tmarket.sources,
        additional_instruction="Confirm via the official fund page."
    )

    etf_ref_leaf = evaluator.add_leaf(
        id="Client_2_ETF_Reference",
        desc="Provide a valid URL confirming the ETF's characteristics",
        parent=etf_sel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided source(s) for {tmarket.name or 'unknown'} ({tmarket.ticker or 'unknown'}) confirm total market coverage and the holdings count (≥3,000).",
        node=etf_ref_leaf,
        sources=tmarket.sources,
        additional_instruction="Accept provider official pages or prospectus/factsheet indicating coverage and holdings."
    )

    # 401(k) total limit including catch-up
    k401_node = evaluator.add_parallel(
        id="Client_2_401k_Contribution",
        desc="Calculate the maximum 2026 total employee contribution limit for 401(k) including catch-up for someone age 50+",
        parent=client_node,
        critical=False
    )
    k401 = data.client2_401k_total_limit_2026 or ContributionInfo()
    k401_total_leaf = evaluator.add_leaf(
        id="Client_2_401k_Total_Limit",
        desc="The total 401(k) contribution limit for 2026 including catch-up must be correctly calculated as $32,500 ($24,500 base + $8,000 catch-up)",
        parent=k401_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2026 total 401(k) employee contribution limit for age 50+ is $32,500 ($24,500 base + $8,000 catch-up).",
        node=k401_total_leaf,
        sources=k401.sources,
        additional_instruction="Confirm both the base and catch-up amounts for 2026 and their sum."
    )

    k401_ref_leaf = evaluator.add_leaf(
        id="Client_2_401k_Reference",
        desc="Provide a valid URL confirming both the base and catch-up contribution limits for 2026",
        parent=k401_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources confirm the 2026 401(k) base limit is $24,500 and the catch-up limit is $8,000 for age 50+.",
        node=k401_ref_leaf,
        sources=k401.sources,
        additional_instruction="Prefer IRS or official industry references explicitly stating both amounts."
    )

    # IRA total limit including catch-up
    ira_node = evaluator.add_parallel(
        id="Client_2_IRA_Contribution",
        desc="Calculate the maximum 2026 total IRA contribution limit including catch-up for someone age 50+",
        parent=client_node,
        critical=False
    )
    ira = data.client2_ira_total_limit_2026 or ContributionInfo()
    ira_total_leaf = evaluator.add_leaf(
        id="Client_2_IRA_Total_Limit",
        desc="The total IRA contribution limit for 2026 including catch-up must be correctly calculated as $8,600 ($7,500 base + $1,100 catch-up)",
        parent=ira_node,
        critical=True
    )
    await evaluator.verify(
        claim="The 2026 total IRA contribution limit for age 50+ is $8,600 ($7,500 base + $1,100 catch-up).",
        node=ira_total_leaf,
        sources=ira.sources,
        additional_instruction="Confirm both the base and catch-up amounts for IRA in 2026 and their sum."
    )

    ira_ref_leaf = evaluator.add_leaf(
        id="Client_2_IRA_Reference",
        desc="Provide a valid URL confirming both the base and catch-up IRA contribution limits for 2026",
        parent=ira_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources confirm the 2026 IRA base limit is $7,500 and the catch-up is $1,100 for age 50+.",
        node=ira_ref_leaf,
        sources=ira.sources,
        additional_instruction="Prefer IRS or official industry references."
    )


async def verify_client_3(evaluator: Evaluator, root_node, data: FinancialPlanExtraction) -> None:
    client_node = evaluator.add_parallel(
        id="Client_3_Dividend_Income_Seeker",
        desc="Correctly address all requirements for Client 3: Age 58, seeking income-focused investments with companies having long dividend histories",
        parent=root_node,
        critical=False
    )

    # Identify company with ≥69 dividend streak and S&P 500 membership
    comp_sel_node = evaluator.add_parallel(
        id="Client_3_Dividend_Aristocrat_Identification",
        desc="Identify a specific Dividend Aristocrat company with one of the longest dividend growth streaks",
        parent=client_node,
        critical=False
    )
    comp = data.client3_company or DividendCompanyInfo()

    streak_leaf = evaluator.add_leaf(
        id="Client_3_Company_Dividend_Streak",
        desc="The company must have increased its dividend for at least 69 consecutive years",
        parent=comp_sel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The company {comp.name or 'unknown'} has increased its dividend for at least 69 consecutive years.",
        node=streak_leaf,
        sources=comp.sources,
        additional_instruction="Verify via company investor relations or credible dividend track record sources."
    )

    sp500_leaf = evaluator.add_leaf(
        id="Client_3_Company_SP500_Membership",
        desc="The company must be a member of the S&P 500 Index",
        parent=comp_sel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The company {comp.name or 'unknown'} is a member of the S&P 500 Index.",
        node=sp500_leaf,
        sources=comp.sources,
        additional_instruction="Confirm using S&P index membership lists or reliable references."
    )

    comp_ref_leaf = evaluator.add_leaf(
        id="Client_3_Company_Reference",
        desc="Provide a valid URL confirming the company's dividend history and consecutive years of increases",
        parent=comp_sel_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided sources confirm {comp.name or 'unknown'}'s dividend increase streak (≥69 years) with explicit evidence.",
        node=comp_ref_leaf,
        sources=comp.sources,
        additional_instruction="Accept investor relations pages, dividend aristocrat lists, or authoritative datasets."
    )

    # Dividend Aristocrat definition
    def_node = evaluator.add_parallel(
        id="Client_3_Dividend_Aristocrat_Definition",
        desc="Correctly explain the definition and requirements for Dividend Aristocrat status",
        parent=client_node,
        critical=False
    )
    definition = data.client3_aristocrat_definition or AristocratDefinitionInfo()

    min_leaf = evaluator.add_leaf(
        id="Client_3_Minimum_Years",
        desc="Dividend Aristocrats must have increased dividends for at least 25 consecutive years",
        parent=def_node,
        critical=True
    )
    await evaluator.verify(
        claim="Dividend Aristocrats must have increased dividends for at least 25 consecutive years.",
        node=min_leaf,
        sources=definition.sources,
        additional_instruction="Confirm via S&P Dow Jones Indices documentation or authoritative sources."
    )

    index_req_leaf = evaluator.add_leaf(
        id="Client_3_Index_Requirement",
        desc="Companies must be members of the S&P 500 Index to qualify",
        parent=def_node,
        critical=True
    )
    await evaluator.verify(
        claim="To qualify as a Dividend Aristocrat, a company must be a member of the S&P 500 Index.",
        node=index_req_leaf,
        sources=definition.sources,
        additional_instruction="Confirm via S&P Dow Jones Indices documentation or authoritative sources."
    )

    def_ref_leaf = evaluator.add_leaf(
        id="Client_3_Definition_Reference",
        desc="Provide a valid URL confirming the Dividend Aristocrat requirements",
        parent=def_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources confirm the Dividend Aristocrat requirements (≥25 years of increases and S&P 500 membership).",
        node=def_ref_leaf,
        sources=definition.sources,
        additional_instruction="Prefer official S&P fact sheets."
    )

    # Asset allocation recommendation
    alloc_node = evaluator.add_parallel(
        id="Client_3_Asset_Allocation",
        desc="Recommend an appropriate conservative asset allocation as client approaches retirement",
        parent=client_node,
        critical=False
    )
    alloc = data.client3_allocation or AssetAllocationInfo()

    stock_leaf = evaluator.add_leaf(
        id="Client_3_Stock_Percentage",
        desc="For a conservative pre-retiree, stock allocation should be in the 40-60% range",
        parent=alloc_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The recommended stock allocation range '{alloc.stock_range or 'unknown'}' is within 40% to 60%.",
        node=stock_leaf,
        sources=None,
        additional_instruction="Interpret percentages; accept if the range falls between 40% and 60%."
    )

    bond_leaf = evaluator.add_leaf(
        id="Client_3_Bond_Percentage",
        desc="For a conservative pre-retiree, bond allocation should be in the 40-60% range",
        parent=alloc_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The recommended bond allocation range '{alloc.bond_range or 'unknown'}' is within 40% to 60%.",
        node=bond_leaf,
        sources=None,
        additional_instruction="Interpret percentages; accept if the range falls between 40% and 60%."
    )

    alloc_ref_leaf = evaluator.add_leaf(
        id="Client_3_Allocation_Reference",
        desc="Provide a valid URL supporting the recommended asset allocation for pre-retirees",
        parent=alloc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources support a conservative pre-retiree allocation around 40–60% stocks and 40–60% bonds.",
        node=alloc_ref_leaf,
        sources=alloc.sources,
        additional_instruction="Accept reputable sources (e.g., Vanguard glide paths, Fidelity, academic references)."
    )


async def verify_client_4(evaluator: Evaluator, root_node, data: FinancialPlanExtraction) -> None:
    client_node = evaluator.add_parallel(
        id="Client_4_New_Graduate",
        desc="Correctly address all requirements for Client 4: Age 24, annual income $55,000, just starting career, seeking target date fund",
        parent=root_node,
        critical=False
    )

    # Target Date Fund 2060 and equity allocation
    tdf_node = evaluator.add_parallel(
        id="Client_4_Target_Date_Fund",
        desc="Identify an appropriate target date fund for someone planning to retire around 2060",
        parent=client_node,
        critical=False
    )
    tdf = data.client4_tdf or TargetDateFundInfo()

    tdf_year_leaf = evaluator.add_leaf(
        id="Client_4_TDF_Year",
        desc="The target date fund should be for the year 2060 (±2 years)",
        parent=tdf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The target date fund {tdf.name or 'unknown'} is designed for the year 2060 or within ±2 years (2058–2062).",
        node=tdf_year_leaf,
        sources=tdf.sources,
        additional_instruction="Verify fund name/target year on the provider page."
    )

    tdf_alloc_leaf = evaluator.add_leaf(
        id="Client_4_TDF_Initial_Allocation",
        desc="The fund's current equity allocation should be approximately 90-99% given the long time horizon",
        parent=tdf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The fund's current equity allocation is approximately between 90% and 99%.",
        node=tdf_alloc_leaf,
        sources=tdf.sources,
        additional_instruction="Confirm using the fund's current glide path/allocation on provider page."
    )

    tdf_ref_leaf = evaluator.add_leaf(
        id="Client_4_TDF_Reference",
        desc="Provide a valid URL confirming the target date fund's allocation strategy",
        parent=tdf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources confirm the 2060 target date and the fund's current equity-heavy allocation consistent with a long horizon.",
        node=tdf_ref_leaf,
        sources=tdf.sources,
        additional_instruction="Prefer official provider pages or prospectus/fact sheets."
    )

    # Series 65 requirements
    s65_node = evaluator.add_parallel(
        id="Client_4_Series_65_Requirement",
        desc="Identify the examination requirements for an advisor to work with this client",
        parent=client_node,
        critical=False
    )
    s65 = data.client4_series65 or Series65Info()

    s65_q_leaf = evaluator.add_leaf(
        id="Client_4_Exam_Questions",
        desc="The Series 65 exam consists of 130 questions",
        parent=s65_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Series 65 exam consists of 130 questions.",
        node=s65_q_leaf,
        sources=s65.sources,
        additional_instruction="Confirm via NASAA/FINRA/Prometric official pages."
    )

    s65_dur_leaf = evaluator.add_leaf(
        id="Client_4_Exam_Duration",
        desc="The Series 65 exam duration is 3 hours",
        parent=s65_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Series 65 exam duration is 3 hours.",
        node=s65_dur_leaf,
        sources=s65.sources,
        additional_instruction="Confirm via official exam details."
    )

    s65_pass_leaf = evaluator.add_leaf(
        id="Client_4_Passing_Score",
        desc="The Series 65 exam requires a 72% passing score",
        parent=s65_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Series 65 exam requires a 72% passing score.",
        node=s65_pass_leaf,
        sources=s65.sources,
        additional_instruction="Confirm via NASAA/FINRA official pages."
    )

    s65_ref_leaf = evaluator.add_leaf(
        id="Client_4_Series_65_Reference",
        desc="Provide a valid URL confirming the Series 65 exam requirements",
        parent=s65_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources confirm the Series 65 exam details: 130 questions, 3 hours, and a 72% passing score.",
        node=s65_ref_leaf,
        sources=s65.sources,
        additional_instruction="Prefer NASAA/FINRA/Prometric official exam resources."
    )

    # CFP CE requirements
    cfp_node = evaluator.add_parallel(
        id="Client_4_CFP_Continuing_Education",
        desc="Identify the continuing education requirements for a CFP® professional serving this client",
        parent=client_node,
        critical=False
    )
    cfp = data.client4_cfp_ce or CFPCEInfo()

    cfp_total_leaf = evaluator.add_leaf(
        id="Client_4_CFP_Total_Hours",
        desc="CFP® professionals must complete 30 hours of continuing education every 2 years",
        parent=cfp_node,
        critical=True
    )
    await evaluator.verify(
        claim="CFP® professionals must complete 30 hours of continuing education every 2 years.",
        node=cfp_total_leaf,
        sources=cfp.sources,
        additional_instruction="Confirm via CFP Board official resources."
    )

    cfp_ethics_leaf = evaluator.add_leaf(
        id="Client_4_CFP_Ethics_Hours",
        desc="Of the 30 hours, 2 hours must be CFP Board-approved Ethics CE",
        parent=cfp_node,
        critical=True
    )
    await evaluator.verify(
        claim="Of the 30 hours, 2 hours must be CFP Board-approved Ethics CE.",
        node=cfp_ethics_leaf,
        sources=cfp.sources,
        additional_instruction="Confirm via CFP Board official resources."
    )

    cfp_ref_leaf = evaluator.add_leaf(
        id="Client_4_CFP_CE_Reference",
        desc="Provide a valid URL confirming the CFP® continuing education requirements",
        parent=cfp_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources confirm the CFP® continuing education requirements (30 hours every 2 years, including 2 hours Ethics).",
        node=cfp_ref_leaf,
        sources=cfp.sources,
        additional_instruction="Prefer CFP Board official pages."
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

    # Extract structured info from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_financial_plan(),
        template_class=FinancialPlanExtraction,
        extraction_name="financial_plan_extraction"
    )

    # Build the verification tree under a root thematic node
    setup_node = evaluator.add_parallel(
        id="Financial_Planning_Practice_Setup",
        desc="Evaluate a comprehensive financial planning practice setup that correctly identifies appropriate investment vehicles, contribution strategies, and advisor qualifications for four distinct client scenarios",
        parent=root,
        critical=False
    )

    # Run verifications for each client scenario
    await verify_client_1(evaluator, setup_node, extracted)
    await verify_client_2(evaluator, setup_node, extracted)
    await verify_client_3(evaluator, setup_node, extracted)
    await verify_client_4(evaluator, setup_node, extracted)

    return evaluator.get_summary()