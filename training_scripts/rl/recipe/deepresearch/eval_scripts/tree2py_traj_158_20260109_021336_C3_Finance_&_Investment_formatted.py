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
TASK_ID = "btc_spot_etf_lowest_fee_2025"
TASK_DESCRIPTION = (
    "Among the spot Bitcoin ETFs approved by the U.S. Securities and Exchange Commission on January 10, 2024, "
    "identify the ETF that currently has the lowest expense ratio as of late 2025. Provide the following information "
    "about this ETF: (1) The full name and ticker symbol of the ETF, (2) Its expense ratio, (3) The date when this ETF "
    "began trading, (4) The asset management company that issued this ETF, (5) The full name of the person who became "
    "Chief Executive Officer of this company in 2024, (6) The name of the university where this CEO obtained their "
    "bachelor's degree in engineering (which should be a federal university in Brazil), and (7) The name of the "
    "university where this CEO obtained their MBA degree. Please provide reference URLs to verify each piece of information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ETFCore(BaseModel):
    full_name: Optional[str] = None
    ticker: Optional[str] = None
    name_ticker_urls: List[str] = Field(default_factory=list)


class ExpenseInfo(BaseModel):
    expense_ratio: Optional[str] = None
    expense_ratio_urls: List[str] = Field(default_factory=list)
    lowest_expense_ratio_urls: List[str] = Field(default_factory=list)  # Comparative/coverage sources


class TradingInfo(BaseModel):
    trading_start_date: Optional[str] = None
    trading_start_date_urls: List[str] = Field(default_factory=list)


class IssuerInfo(BaseModel):
    issuer_name: Optional[str] = None
    issuer_urls: List[str] = Field(default_factory=list)


class CEOInfo(BaseModel):
    ceo_full_name: Optional[str] = None
    ceo_appointment_year: Optional[str] = None
    ceo_urls: List[str] = Field(default_factory=list)


class EducationInfo(BaseModel):
    bachelors_university: Optional[str] = None
    bachelors_degree_field: Optional[str] = None  # e.g., "engineering"
    bachelors_urls: List[str] = Field(default_factory=list)
    bachelors_federal_validation_urls: List[str] = Field(default_factory=list)  # proof it is a federal university in Brazil
    mba_university: Optional[str] = None
    mba_urls: List[str] = Field(default_factory=list)
    mba_top_tier_validation_urls: List[str] = Field(default_factory=list)  # rankings/M7 evidence


class EligibilitySources(BaseModel):
    approved_set_membership_urls: List[str] = Field(default_factory=list)  # inclusion in the Jan 10, 2024 SEC approval set


class ETFSelectionExtraction(BaseModel):
    etf: Optional[ETFCore] = None
    expense: Optional[ExpenseInfo] = None
    trading: Optional[TradingInfo] = None
    issuer: Optional[IssuerInfo] = None
    ceo: Optional[CEOInfo] = None
    education: Optional[EducationInfo] = None
    eligibility: Optional[EligibilitySources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etf_selection() -> str:
    return """
    Extract the single ETF that the answer identifies as having the lowest expense ratio (as of late 2025) among the
    spot Bitcoin ETFs approved by the U.S. SEC on January 10, 2024. If the answer discusses multiple ETFs, extract
    the one the answer claims is the lowest-expense pick.

    Return a JSON object with the following nested structure (use null for missing text fields and empty arrays for missing URL lists):

    {
      "etf": {
        "full_name": string|null,
        "ticker": string|null,
        "name_ticker_urls": string[]   // URLs in the answer that show the ETF name and ticker
      },
      "expense": {
        "expense_ratio": string|null,  // As written in the answer (e.g., "0.19%" or "0.19% (net)")
        "expense_ratio_urls": string[],  // URLs that directly show the ETF's own expense ratio
        "lowest_expense_ratio_urls": string[] // URLs that support the claim it's the lowest among the 11 as of late 2025
      },
      "trading": {
        "trading_start_date": string|null,  // The date it began trading (free-form, e.g., "January 11, 2024")
        "trading_start_date_urls": string[] // URLs that show the trading start date
      },
      "issuer": {
        "issuer_name": string|null,    // Asset management company/issuer of this ETF
        "issuer_urls": string[]        // URLs confirming the issuer
      },
      "ceo": {
        "ceo_full_name": string|null,        // The person who became CEO in 2024
        "ceo_appointment_year": string|null, // The year of appointment (should be "2024" if provided)
        "ceo_urls": string[]                 // URLs supporting CEO identity and 2024 appointment as CEO
      },
      "education": {
        "bachelors_university": string|null,           // University for bachelor's degree in engineering
        "bachelors_degree_field": string|null,         // Field of bachelor's (should indicate engineering)
        "bachelors_urls": string[],                    // URLs confirming bachelor's degree and institution
        "bachelors_federal_validation_urls": string[], // URLs confirming that this is a federal university in Brazil (can be Wikipedia/university site)
        "mba_university": string|null,                 // MBA university/business school name
        "mba_urls": string[],                          // URLs confirming MBA school
        "mba_top_tier_validation_urls": string[]       // URLs (e.g., rankings, M7) that show the MBA school is top-tier in the U.S.
      },
      "eligibility": {
        "approved_set_membership_urls": string[] // URLs confirming the ETF is one of the 11 approved by SEC on Jan 10, 2024
      }
    }

    STRICT RULES:
    - Extract only the URLs that appear in the answer; do not invent any.
    - If a data item (like expense ratio) is mentioned but no URL is provided, set the corresponding *_urls list to [].
    - Preserve strings exactly as written in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_eligibility_and_min_fee_checks(
    evaluator: Evaluator,
    parent_node,
    data: ETFSelectionExtraction,
) -> None:
    """
    Build 'Eligibility_and_Minimum_Expense_Ratio' subtree:
      - Approved_Set_Membership (with sources provided + by-URL verification)
      - Lowest_Expense_Ratio_As_Of_Late_2025 (with sources provided + by-URL verification)
    All nodes here are critical.
    """
    etf_name = _safe(data.etf.full_name if data.etf else None)
    etf_ticker = _safe(data.etf.ticker if data.etf else None)

    elig_node = evaluator.add_parallel(
        id="Eligibility_and_Minimum_Expense_Ratio",
        desc="Correctly select the ETF that meets the SEC-approval set constraint and has the lowest expense ratio among that set as of late 2025, supported by sources.",
        parent=parent_node,
        critical=True
    )

    # Approved set membership
    approved_node = evaluator.add_parallel(
        id="Approved_Set_Membership",
        desc="The chosen ETF is one of the 11 spot Bitcoin ETFs approved by the SEC on January 10, 2024, with a reference URL supporting inclusion in that approval set.",
        parent=elig_node,
        critical=True
    )

    approved_urls = _non_empty_urls(data.eligibility.approved_set_membership_urls if data.eligibility else None)
    evaluator.add_custom_node(
        result=len(approved_urls) > 0,
        id="Approved_Set_Membership_sources_provided",
        desc="Source URL(s) provided for SEC-approved set membership (Jan 10, 2024).",
        parent=approved_node,
        critical=True
    )

    approved_leaf = evaluator.add_leaf(
        id="Approved_Set_Membership_supported_by_urls",
        desc="The ETF is in the SEC's Jan 10, 2024 spot Bitcoin ETF approval set (supported by sources).",
        parent=approved_node,
        critical=True
    )
    approved_claim = (
        f"The ETF '{etf_name}' ({etf_ticker}) is one of the 11 spot Bitcoin ETFs approved by the U.S. SEC on January 10, 2024."
    )
    await evaluator.verify(
        claim=approved_claim,
        node=approved_leaf,
        sources=approved_urls,
        additional_instruction="Verify that the page(s) explicitly list this ETF as part of the SEC's spot Bitcoin ETF approvals on Jan 10, 2024."
    )

    # Lowest expense ratio as of late 2025
    lowest_node = evaluator.add_parallel(
        id="Lowest_Expense_Ratio_As_Of_Late_2025",
        desc="Demonstrate (with citations) that the chosen ETF’s expense ratio is the lowest among the 11 SEC-approved spot Bitcoin ETFs as of late 2025.",
        parent=elig_node,
        critical=True
    )
    lowest_urls = _non_empty_urls(data.expense.lowest_expense_ratio_urls if data.expense else None)
    evaluator.add_custom_node(
        result=len(lowest_urls) > 0,
        id="Lowest_Expense_Ratio_As_Of_Late_2025_sources_provided",
        desc="Source URL(s) provided supporting lowest expense ratio among the 11 as of late 2025.",
        parent=lowest_node,
        critical=True
    )
    lowest_leaf = evaluator.add_leaf(
        id="Lowest_Expense_Ratio_As_Of_Late_2025_supported_by_urls",
        desc="The ETF's expense ratio is the lowest among the 11 as of late 2025 (supported by sources).",
        parent=lowest_node,
        critical=True
    )
    lowest_claim = (
        f"As of late 2025, the ETF '{etf_name}' ({etf_ticker}) has the lowest expense ratio among the 11 spot Bitcoin ETFs approved by the SEC on January 10, 2024."
    )
    await evaluator.verify(
        claim=lowest_claim,
        node=lowest_leaf,
        sources=lowest_urls,
        additional_instruction=(
            "Prefer sources that compare fees across the 11 SEC-approved spot Bitcoin ETFs or otherwise substantiate "
            "that this ETF has the lowest fee as of late 2025. Accept ties for lowest as sufficient."
        )
    )


async def build_required_output_fields(
    evaluator: Evaluator,
    parent_node,
    data: ETFSelectionExtraction,
) -> None:
    """
    Build 'Required_Output_Fields_With_Citations' subtree.
    Each item is modeled as a critical parallel node with:
      - 'provided' existence check
      - one or more verification leaves
    """
    etf_name = _safe(data.etf.full_name if data.etf else None)
    etf_ticker = _safe(data.etf.ticker if data.etf else None)

    fields_node = evaluator.add_parallel(
        id="Required_Output_Fields_With_Citations",
        desc="Provide each requested field for the identified ETF and related CEO details, each with reference URL(s) sufficient to verify the claim.",
        parent=parent_node,
        critical=True
    )

    # 1) ETF Full Name and Ticker with URL
    name_ticker_node = evaluator.add_parallel(
        id="ETF_Full_Name_And_Ticker_With_URL",
        desc="Provide the ETF’s full name and ticker symbol, with a reference URL.",
        parent=fields_node,
        critical=True
    )
    name_ticker_urls = _non_empty_urls(data.etf.name_ticker_urls if data.etf else None)
    evaluator.add_custom_node(
        result=(etf_name.strip() != "" and etf_ticker.strip() != "" and len(name_ticker_urls) > 0),
        id="ETF_Full_Name_And_Ticker_provided",
        desc="ETF full name and ticker and source URL(s) are provided.",
        parent=name_ticker_node,
        critical=True
    )
    name_ticker_leaf = evaluator.add_leaf(
        id="ETF_Full_Name_And_Ticker_supported_by_url",
        desc="ETF full name and ticker match the cited page(s).",
        parent=name_ticker_node,
        critical=True
    )
    name_ticker_claim = f"This page shows the ETF full name '{etf_name}' and the ticker symbol '{etf_ticker}'."
    await evaluator.verify(
        claim=name_ticker_claim,
        node=name_ticker_leaf,
        sources=name_ticker_urls,
        additional_instruction="Allow minor formatting/case variations for names and tickers."
    )

    # 2) ETF Expense Ratio with URL
    expense_node = evaluator.add_parallel(
        id="ETF_Expense_Ratio_With_URL",
        desc="Provide the ETF’s expense ratio, with a reference URL (current/relevant to late 2025 when possible).",
        parent=fields_node,
        critical=True
    )
    expense_ratio = _safe(data.expense.expense_ratio if data.expense else None)
    expense_ratio_urls = _non_empty_urls(data.expense.expense_ratio_urls if data.expense else None)
    evaluator.add_custom_node(
        result=(expense_ratio.strip() != "" and len(expense_ratio_urls) > 0),
        id="ETF_Expense_Ratio_provided",
        desc="Expense ratio and source URL(s) are provided.",
        parent=expense_node,
        critical=True
    )
    expense_leaf = evaluator.add_leaf(
        id="ETF_Expense_Ratio_supported_by_url",
        desc="The stated expense ratio is supported by the cited page(s).",
        parent=expense_node,
        critical=True
    )
    expense_claim = f"The ETF '{etf_name}' ({etf_ticker}) has an expense ratio of '{expense_ratio}'."
    await evaluator.verify(
        claim=expense_claim,
        node=expense_leaf,
        sources=expense_ratio_urls,
        additional_instruction="Accept reasonable fee notation variations like 'net' or 'after waiver' if aligned with the claimed figure."
    )

    # 3) Trading Start Date in 2024 with URL
    start_node = evaluator.add_parallel(
        id="Trading_Start_Date_2024_With_URL",
        desc="Provide the date the ETF began trading, and it must be in 2024, with a reference URL.",
        parent=fields_node,
        critical=True
    )
    start_date = _safe(data.trading.trading_start_date if data.trading else None)
    start_urls = _non_empty_urls(data.trading.trading_start_date_urls if data.trading else None)
    evaluator.add_custom_node(
        result=(start_date.strip() != "" and len(start_urls) > 0),
        id="Trading_Start_Date_provided",
        desc="Trading start date and source URL(s) are provided.",
        parent=start_node,
        critical=True
    )
    start_date_leaf = evaluator.add_leaf(
        id="Trading_Start_Date_supported_by_url",
        desc="The trading start date is supported by the cited page(s).",
        parent=start_node,
        critical=True
    )
    start_claim = f"The ETF '{etf_name}' ({etf_ticker}) began trading on '{start_date}'."
    await evaluator.verify(
        claim=start_claim,
        node=start_date_leaf,
        sources=start_urls,
        additional_instruction="The page should clearly state the initial trading date for the ETF."
    )

    start_year_leaf = evaluator.add_leaf(
        id="Trading_Start_Date_is_2024",
        desc="The trading start date year is 2024.",
        parent=start_node,
        critical=True
    )
    year_check_claim = f"The year of the trading start date '{start_date}' is 2024."
    await evaluator.verify(
        claim=year_check_claim,
        node=start_year_leaf,
        additional_instruction="Verify that the provided date string corresponds to a calendar date in the year 2024."
    )

    # 4) Issuing Asset Manager with URL
    issuer_node = evaluator.add_parallel(
        id="Issuing_Asset_Manager_With_URL",
        desc="Identify the asset management company that issued the ETF, with a reference URL.",
        parent=fields_node,
        critical=True
    )
    issuer_name = _safe(data.issuer.issuer_name if data.issuer else None)
    issuer_urls = _non_empty_urls(data.issuer.issuer_urls if data.issuer else None)
    evaluator.add_custom_node(
        result=(issuer_name.strip() != "" and len(issuer_urls) > 0),
        id="Issuing_Asset_Manager_provided",
        desc="Issuer name and source URL(s) are provided.",
        parent=issuer_node,
        critical=True
    )
    issuer_leaf = evaluator.add_leaf(
        id="Issuing_Asset_Manager_supported_by_url",
        desc="The stated issuer is supported by the cited page(s).",
        parent=issuer_node,
        critical=True
    )
    issuer_claim = f"The ETF '{etf_name}' ({etf_ticker}) is issued by '{issuer_name}'."
    await evaluator.verify(
        claim=issuer_claim,
        node=issuer_leaf,
        sources=issuer_urls,
        additional_instruction="The issuer should match the asset management company behind the ETF."
    )

    # 5) CEO appointed in 2024 Name with URL
    ceo_node = evaluator.add_parallel(
        id="CEO_Appointed_In_2024_Name_With_URL",
        desc="Provide the full name of the person who became CEO of the issuing company in 2024, with a reference URL.",
        parent=fields_node,
        critical=True
    )
    ceo_name = _safe(data.ceo.ceo_full_name if data.ceo else None)
    ceo_year = _safe(data.ceo.ceo_appointment_year if data.ceo else None)
    ceo_urls = _non_empty_urls(data.ceo.ceo_urls if data.ceo else None)
    evaluator.add_custom_node(
        result=(ceo_name.strip() != "" and ceo_year.strip() != "" and len(ceo_urls) > 0),
        id="CEO_Provided",
        desc="CEO name, 2024 appointment year, and source URL(s) are provided.",
        parent=ceo_node,
        critical=True
    )
    ceo_is_leaf = evaluator.add_leaf(
        id="CEO_Name_supported_by_url",
        desc="The cited page(s) show this person is the CEO of the issuer.",
        parent=ceo_node,
        critical=True
    )
    ceo_is_claim = f"The person '{ceo_name}' is (or became) the CEO of '{issuer_name}'."
    await evaluator.verify(
        claim=ceo_is_claim,
        node=ceo_is_leaf,
        sources=ceo_urls,
        additional_instruction="The page should indicate the person holds the CEO title at the named issuer."
    )
    ceo_year_leaf = evaluator.add_leaf(
        id="CEO_Appointment_Year_2024_supported_by_url",
        desc="The cited page(s) indicate the CEO appointment occurred in 2024.",
        parent=ceo_node,
        critical=True
    )
    ceo_year_claim = f"'{ceo_name}' became CEO of '{issuer_name}' in 2024."
    await evaluator.verify(
        claim=ceo_year_claim,
        node=ceo_year_leaf,
        sources=ceo_urls,
        additional_instruction="The page should state (or strongly imply) that the appointment as CEO occurred in 2024."
    )

    # 6) CEO Bachelor's Engineering at Federal Brazilian University with URL
    bachelors_node = evaluator.add_parallel(
        id="CEO_Bachelors_Engineering_Federal_Brazil_University_With_URL",
        desc="Name the university where the CEO obtained their bachelor's degree in engineering and verify that it is a federal university in Brazil, with a reference URL.",
        parent=fields_node,
        critical=True
    )
    bachelors_uni = _safe(data.education.bachelors_university if data.education else None)
    bachelors_field = _safe(data.education.bachelors_degree_field if data.education else None)
    bachelors_urls = _non_empty_urls(data.education.bachelors_urls if data.education else None)
    bachelors_federal_urls = _non_empty_urls(data.education.bachelors_federal_validation_urls if data.education else None)
    evaluator.add_custom_node(
        result=(bachelors_uni.strip() != "" and len(bachelors_urls) > 0),
        id="Bachelors_info_provided",
        desc="Bachelor's university and source URL(s) are provided.",
        parent=bachelors_node,
        critical=True
    )
    bachelors_eng_leaf = evaluator.add_leaf(
        id="Bachelors_Engineering_supported_by_url",
        desc="The cited page(s) indicate the CEO earned a bachelor's degree in engineering at the stated university.",
        parent=bachelors_node,
        critical=True
    )
    bachelors_eng_claim = (
        f"'{ceo_name}' earned a bachelor's degree in engineering from '{bachelors_uni}'. "
        f"(Degree field extracted as '{bachelors_field}')"
    )
    await evaluator.verify(
        claim=bachelors_eng_claim,
        node=bachelors_eng_leaf,
        sources=bachelors_urls,
        additional_instruction="Confirm the bachelor's is in an engineering discipline; allow variants like 'Electrical Engineering', 'Industrial Engineering', etc."
    )

    federal_check_leaf = evaluator.add_leaf(
        id="Bachelors_Federal_Brazil_University_supported_by_url",
        desc="The cited page(s) indicate that the bachelor's university is a federal university in Brazil.",
        parent=bachelors_node,
        critical=True
    )
    federal_sources = bachelors_federal_urls if len(bachelors_federal_urls) > 0 else bachelors_urls
    federal_claim = f"'{bachelors_uni}' is a federal university in Brazil."
    await evaluator.verify(
        claim=federal_claim,
        node=federal_check_leaf,
        sources=federal_sources,
        additional_instruction="Accept evidence from reputable sources (e.g., official university site or Wikipedia) that the institution is part of Brazil's federal university system."
    )

    # 7) CEO MBA School with URL
    mba_node = evaluator.add_parallel(
        id="CEO_MBA_School_With_URL",
        desc="Name the university/business school where the CEO obtained their MBA, with a reference URL.",
        parent=fields_node,
        critical=True
    )
    mba_uni = _safe(data.education.mba_university if data.education else None)
    mba_urls = _non_empty_urls(data.education.mba_urls if data.education else None)
    evaluator.add_custom_node(
        result=(mba_uni.strip() != "" and len(mba_urls) > 0),
        id="MBA_info_provided",
        desc="MBA university and source URL(s) are provided.",
        parent=mba_node,
        critical=True
    )
    mba_leaf = evaluator.add_leaf(
        id="MBA_School_supported_by_url",
        desc="The cited page(s) indicate the CEO earned an MBA from the stated institution.",
        parent=mba_node,
        critical=True
    )
    mba_claim = f"'{ceo_name}' earned an MBA from '{mba_uni}'."
    await evaluator.verify(
        claim=mba_claim,
        node=mba_leaf,
        sources=mba_urls,
        additional_instruction="The page should clearly state that the person earned an MBA from the named institution."
    )

    # 8) MBA Top-tier US business school validation with URL
    top_tier_node = evaluator.add_parallel(
        id="MBA_Top_Tier_US_Business_School_Validation_With_URL",
        desc="Provide evidence (with a reference URL) that the MBA institution qualifies as a top-tier U.S. business school.",
        parent=fields_node,
        critical=True
    )
    top_tier_urls = _non_empty_urls(data.education.mba_top_tier_validation_urls if data.education else None)
    evaluator.add_custom_node(
        result=(len(top_tier_urls) > 0),
        id="MBA_Top_Tier_validation_urls_provided",
        desc="Top-tier validation source URL(s) are provided.",
        parent=top_tier_node,
        critical=True
    )
    top_tier_leaf = evaluator.add_leaf(
        id="MBA_Top_Tier_supported_by_url",
        desc="The cited page(s) demonstrate that the MBA school is top-tier among U.S. business schools.",
        parent=top_tier_node,
        critical=True
    )
    top_tier_claim = f"'{mba_uni}' is a top-tier U.S. business school."
    await evaluator.verify(
        claim=top_tier_claim,
        node=top_tier_leaf,
        sources=top_tier_urls,
        additional_instruction=(
            "Accept reputable rankings (e.g., U.S. News, Financial Times, Bloomberg), or recognized labels (e.g., M7). "
            "Evidence should indicate top-tier standing in the U.S."
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
    Evaluate an answer for the 'lowest-expense spot Bitcoin ETF as of late 2025' task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root sequential to reflect task flow
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

    # Create a critical task node (sequential) under root to enforce mandatory success of stages
    task_node = evaluator.add_sequential(
        id="ETF_With_Lowest_Expense_Ratio_Task",
        desc="Identify the spot Bitcoin ETF (from the 11 SEC-approved on Jan 10, 2024) with the lowest expense ratio as of late 2025, and provide all requested linked details with verifiable reference URLs.",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_etf_selection(),
        template_class=ETFSelectionExtraction,
        extraction_name="etf_lowest_fee_selection"
    )

    # Build Stage 1: Eligibility and Minimum Expense Ratio
    await build_eligibility_and_min_fee_checks(evaluator, task_node, extraction)

    # Build Stage 2: Required Output Fields with Citations
    await build_required_output_fields(evaluator, task_node, extraction)

    # Return summary
    return evaluator.get_summary()