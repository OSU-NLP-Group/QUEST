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
TASK_ID = "xrp_etf_and_state_rebates_2025_2026"
TASK_DESCRIPTION = """
You are preparing an investment research brief on US spot XRP ETFs and state tax rebate programs for Q4 2025/Q1 2026 tax planning. Your analysis must identify three specific XRP ETFs and two state tax rebate programs based on the criteria below.

Part A: XRP ETF Analysis

Identify the following three US spot XRP exchange-traded funds:

1. First Mover ETF: The first US spot XRP ETF to launch (earliest launch date among all US spot XRP ETFs)

2. Cost Leader ETF: The XRP ETF with the lowest stated management fee percentage (exclude temporary fee waivers when comparing; use the permanent stated fee)

3. Custodial Diversification ETF: An XRP ETF that uses three or more distinct custodians for holding XRP assets

For each of the three ETFs identified above, provide:
- Ticker symbol
- Stated management fee (percentage)
- Launch date
- Primary listing exchange
- Custodian(s) information

Part B: State Tax Rebate Programs

Identify and provide details for the following two state tax rebate/dividend programs:

1. Oregon's Program: Oregon's surplus revenue rebate program (commonly called the "kicker") that taxpayers can claim on their 2025 tax returns
   - Provide the 2025 kicker percentage
   - List the three main eligibility requirements
   - Explain the calculation method

2. Alaska's Program: Alaska's annual dividend program for 2025
   - Provide the 2025 dividend amount
   - List the key eligibility requirements including residency and intent requirements

All information must be supported by reference URLs from official sources, ETF issuer websites, or authoritative financial/government sources.
"""

# Optional ground-truth hints (used only for reference/logging)
OREGON_2025_KICKER_PCT = "9.863%"
ALASKA_2025_PFD_AMOUNT = "$1,000"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ETFInfo(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    stated_management_fee: Optional[str] = None            # e.g., "0.19%" (can include text)
    permanent_management_fee: Optional[str] = None         # non-waiver fee if explicitly stated
    temporary_fee_waiver_mentioned: Optional[bool] = None  # True if fee waiver mentioned
    launch_date: Optional[str] = None                      # e.g., "2025-04-15" or text date
    primary_exchange: Optional[str] = None                 # e.g., "NYSE Arca", "Nasdaq"
    custodians: List[str] = Field(default_factory=list)    # list of custodian names
    id_sources: List[str] = Field(default_factory=list)    # URLs verifying identity/selection claim
    selection_sources: List[str] = Field(default_factory=list)  # URLs for selection criterion
    trading_sources: List[str] = Field(default_factory=list)    # URLs verifying fee/launch/exchange
    custodian_sources: List[str] = Field(default_factory=list)  # URLs verifying custodian details


class ETFExtraction(BaseModel):
    first_mover: Optional[ETFInfo] = None
    cost_leader: Optional[ETFInfo] = None
    multi_custodian: Optional[ETFInfo] = None


class OregonKickerInfo(BaseModel):
    state_name: Optional[str] = None
    program_name: Optional[str] = None
    kicker_percentage_2025: Optional[str] = None
    eligibility_requirements: List[str] = Field(default_factory=list)
    calculation_method: Optional[str] = None
    id_sources: List[str] = Field(default_factory=list)       # for identification
    details_sources: List[str] = Field(default_factory=list)  # for percentage, eligibility, calculation


class AlaskaPFDInfo(BaseModel):
    state_name: Optional[str] = None
    program_name: Optional[str] = None
    dividend_amount_2025: Optional[str] = None
    eligibility_requirements: List[str] = Field(default_factory=list)
    id_sources: List[str] = Field(default_factory=list)
    details_sources: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    oregon: Optional[OregonKickerInfo] = None
    alaska: Optional[AlaskaPFDInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_etfs() -> str:
    return """
Extract details for three specific US spot XRP ETFs from the answer. The three categories are:
- first_mover: The first US spot XRP ETF to launch (earliest launch date)
- cost_leader: The XRP ETF with the lowest permanent stated management fee (exclude temporary waivers)
- multi_custodian: An XRP ETF that uses three or more distinct custodians

For each category, extract an object with:
- name: ETF name
- ticker: ticker symbol
- stated_management_fee: the stated (possibly promotional) fee percentage exactly as written (e.g., "0.19%")
- permanent_management_fee: the ongoing fee excluding temporary fee waivers if explicitly stated; else null
- temporary_fee_waiver_mentioned: true/false if a temporary/waived/introductory fee is mentioned
- launch_date: launch/listing/inception date as presented
- primary_exchange: primary listing exchange
- custodians: array of custodian names listed in the answer
- id_sources: array of URL(s) used to identify and/or support the selection criterion for this ETF
- selection_sources: array of URL(s) specifically supporting the selection claim (first-to-launch, lowest fee, or ≥3 custodians)
- trading_sources: array of URL(s) supporting fee/launch date/exchange details
- custodian_sources: array of URL(s) supporting custodian information

Return a JSON with keys: first_mover, cost_leader, multi_custodian, each mapped to the object above. If any category is missing in the answer, set it to null.
Follow URL extraction rules: only include actual URLs mentioned in the answer.
"""


def prompt_extract_programs() -> str:
    return """
Extract details for two state programs mentioned in the answer:

1) Oregon's 2025 "kicker" surplus revenue rebate (claimed on 2025 returns):
- state_name
- program_name (e.g., "kicker", "surplus rebate")
- kicker_percentage_2025 (e.g., "9.863%" if provided)
- eligibility_requirements: array of the main eligibility requirements listed (e.g., filed 2024 return, had 2024 tax liability, must file 2025 return)
- calculation_method: e.g., "2024 tax liability × 9.863%"
- id_sources: array of URL(s) identifying the program from Oregon government/tax authority
- details_sources: array of URL(s) supporting percentage, eligibility, and calculation

2) Alaska's 2025 Permanent Fund Dividend (PFD):
- state_name
- program_name (e.g., "Permanent Fund Dividend", "PFD")
- dividend_amount_2025 (e.g., "$1,000" if provided)
- eligibility_requirements: array including residency and intent requirements
- id_sources: array of URL(s) identifying the program from official Alaska sources
- details_sources: array of URL(s) supporting amount and eligibility

Return a JSON with keys: oregon, alaska. If any is missing, set it to null.
Follow URL extraction rules: only include actual URLs mentioned in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*url_lists: List[str]) -> List[str]:
    """Merge and de-duplicate URL lists preserving order."""
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if u and (u not in seen):
                seen.add(u)
                merged.append(u)
    return merged


def _effective_fee_str(etf: Optional[ETFInfo]) -> str:
    if not etf:
        return ""
    return (etf.permanent_management_fee or etf.stated_management_fee or "").strip()


# --------------------------------------------------------------------------- #
# Verification builders for ETF sections                                      #
# --------------------------------------------------------------------------- #
async def verify_first_mover_etf(evaluator: Evaluator, parent, etf: Optional[ETFInfo]) -> None:
    """
    Build and verify the 'First_Spot_XRP_ETF' subtree.
    """
    node = evaluator.add_parallel(
        id="First_Spot_XRP_ETF",
        desc="Identify and provide details about the first US spot XRP ETF to launch (earliest launch date)",
        parent=parent,
        critical=False
    )

    # Identification (critical)
    ident = evaluator.add_parallel(
        id="First_Spot_XRP_ETF_ETF_Identification",
        desc="Basic identification of the ETF meeting the selection criterion",
        parent=node,
        critical=True
    )

    # Selection criterion met
    sel_leaf = evaluator.add_leaf(
        id="First_Spot_XRP_ETF_Selection_Criterion_Met",
        desc="The identified ETF has the earliest launch date among all US spot XRP ETFs",
        parent=ident,
        critical=True
    )
    sel_sources = _merge_urls(getattr(etf, "selection_sources", []), getattr(etf, "id_sources", []), getattr(etf, "trading_sources", []))
    await evaluator.verify(
        claim=f"This ETF ({(etf.name or 'the ETF')}) was the first US spot XRP ETF to launch (earliest launch date).",
        node=sel_leaf,
        sources=sel_sources,
        additional_instruction="Look for phrases like 'first US spot XRP ETF', 'first to launch', or similar; accept equivalent phrasing on issuer, exchange, SEC, or reputable financial news sources."
    )

    # Ticker symbol
    tick_leaf = evaluator.add_leaf(
        id="First_Spot_XRP_ETF_Ticker_Symbol",
        desc="Correct ticker symbol is provided",
        parent=ident,
        critical=True
    )
    t_sources = _merge_urls(getattr(etf, "trading_sources", []), getattr(etf, "id_sources", []))
    await evaluator.verify(
        claim=f"The ETF's ticker symbol is '{(etf.ticker or '').strip()}'.",
        node=tick_leaf,
        sources=t_sources,
        additional_instruction="Verify that the page explicitly lists the ticker symbol for this ETF."
    )

    # Identification reference - Source_URL presence check
    id_ref = evaluator.add_parallel(
        id="First_Spot_XRP_ETF_Identification_Reference",
        desc="Reference documentation for ETF identification",
        parent=ident,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sel_sources) > 0,
        id="First_Spot_XRP_ETF_Identification_Source_URL",
        desc="Valid URL source verifying ETF identity and launch date",
        parent=id_ref,
        critical=True
    )

    # Trading details (critical)
    trading = evaluator.add_parallel(
        id="First_Spot_XRP_ETF_Trading_Details",
        desc="Detailed trading and operational information about the ETF",
        parent=node,
        critical=True
    )

    # Management Fee
    mgmt_leaf = evaluator.add_leaf(
        id="First_Spot_XRP_ETF_Management_Fee",
        desc="Stated management fee percentage is provided",
        parent=trading,
        critical=True
    )
    mgmt_fee_str = _effective_fee_str(etf)
    mgmt_sources = _merge_urls(getattr(etf, "trading_sources", []), getattr(etf, "id_sources", []))
    await evaluator.verify(
        claim=f"The ETF's permanent (non-waived) stated management fee (expense ratio) is '{mgmt_fee_str}'.",
        node=mgmt_leaf,
        sources=mgmt_sources,
        additional_instruction="Verify the ongoing management fee or expense ratio; ignore temporary fee waivers or promotional waiver periods if both are listed."
    )

    # Launch Date
    launch_leaf = evaluator.add_leaf(
        id="First_Spot_XRP_ETF_Launch_Date",
        desc="Launch date is provided",
        parent=trading,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF launched/listed on '{(etf.launch_date or '').strip()}'.",
        node=launch_leaf,
        sources=mgmt_sources,
        additional_instruction="Accept 'inception', 'listing', or 'launch' date if clearly referring to the ETF's first trading date."
    )

    # Primary Exchange
    exch_leaf = evaluator.add_leaf(
        id="First_Spot_XRP_ETF_Primary_Exchange",
        desc="Primary listing exchange is provided",
        parent=trading,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF is primarily listed on '{(etf.primary_exchange or '').strip()}'.",
        node=exch_leaf,
        sources=mgmt_sources,
        additional_instruction="Verify the primary listing venue; issuer pages, exchange listings, or fact sheets are acceptable."
    )

    # Custodian Information
    cust_leaf = evaluator.add_leaf(
        id="First_Spot_XRP_ETF_Custodian_Information",
        desc="Custodian(s) information is provided",
        parent=trading,
        critical=True
    )
    cust_sources = _merge_urls(getattr(etf, "custodian_sources", []), getattr(etf, "trading_sources", []))
    cust_list = ", ".join(getattr(etf, "custodians", []) or [])
    await evaluator.verify(
        claim=f"The ETF lists the following custodian(s): {cust_list}.",
        node=cust_leaf,
        sources=cust_sources,
        additional_instruction="Verify that the page lists the custodian(s) that hold the ETF's XRP assets (allow 'custodian'/'sub-custodian' terminology)."
    )

    # Trading Reference - Source URL presence
    tr_ref = evaluator.add_parallel(
        id="First_Spot_XRP_ETF_Trading_Reference",
        desc="Reference documentation for trading details",
        parent=trading,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(mgmt_sources) > 0,
        id="First_Spot_XRP_ETF_Trading_Source_URL",
        desc="Valid URL source verifying trading details",
        parent=tr_ref,
        critical=True
    )


async def verify_cost_leader_etf(evaluator: Evaluator, parent, etf: Optional[ETFInfo]) -> None:
    """
    Build and verify the 'Lowest_Fee_XRP_ETF' subtree.
    """
    node = evaluator.add_parallel(
        id="Lowest_Fee_XRP_ETF",
        desc="Identify and provide details about the XRP ETF with the lowest stated management fee",
        parent=parent,
        critical=False
    )

    # Identification (critical)
    ident = evaluator.add_parallel(
        id="Lowest_Fee_XRP_ETF_ETF_Identification",
        desc="Basic identification of the ETF meeting the selection criterion",
        parent=node,
        critical=True
    )

    sel_leaf = evaluator.add_leaf(
        id="Lowest_Fee_XRP_ETF_Selection_Criterion_Met",
        desc="The identified ETF has the lowest stated management fee among US spot XRP ETFs (excluding temporary waivers)",
        parent=ident,
        critical=True
    )
    sel_sources = _merge_urls(getattr(etf, "selection_sources", []), getattr(etf, "id_sources", []), getattr(etf, "trading_sources", []))
    await evaluator.verify(
        claim=f"This ETF ({(etf.name or 'the ETF')}) has the lowest permanent stated management fee among US spot XRP ETFs (excluding temporary waivers).",
        node=sel_leaf,
        sources=sel_sources,
        additional_instruction="Prefer explicit statements (e.g., 'lowest fee', 'lowest expense ratio') on issuer or reputable financial sources; where not explicit, fee comparison tables on authoritative sites are acceptable."
    )

    tick_leaf = evaluator.add_leaf(
        id="Lowest_Fee_XRP_ETF_Ticker_Symbol",
        desc="Correct ticker symbol is provided",
        parent=ident,
        critical=True
    )
    t_sources = _merge_urls(getattr(etf, "trading_sources", []), getattr(etf, "id_sources", []))
    await evaluator.verify(
        claim=f"The ETF's ticker symbol is '{(etf.ticker or '').strip()}'.",
        node=tick_leaf,
        sources=t_sources,
        additional_instruction="Verify the ticker on issuer/exchange/fact sheet pages."
    )

    id_ref = evaluator.add_parallel(
        id="Lowest_Fee_XRP_ETF_Identification_Reference",
        desc="Reference documentation for ETF identification",
        parent=ident,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sel_sources) > 0,
        id="Lowest_Fee_XRP_ETF_Identification_Source_URL",
        desc="Valid URL source verifying ETF identity and fee structure",
        parent=id_ref,
        critical=True
    )

    # Trading details (critical)
    trading = evaluator.add_parallel(
        id="Lowest_Fee_XRP_ETF_Trading_Details",
        desc="Detailed trading and operational information about the ETF",
        parent=node,
        critical=True
    )

    mgmt_leaf = evaluator.add_leaf(
        id="Lowest_Fee_XRP_ETF_Management_Fee",
        desc="Stated management fee percentage is provided",
        parent=trading,
        critical=True
    )
    mgmt_fee_str = _effective_fee_str(etf)
    mgmt_sources = _merge_urls(getattr(etf, "trading_sources", []), getattr(etf, "id_sources", []))
    await evaluator.verify(
        claim=f"The ETF's permanent (non-waived) stated management fee (expense ratio) is '{mgmt_fee_str}'.",
        node=mgmt_leaf,
        sources=mgmt_sources,
        additional_instruction="Verify the ongoing fee; ignore temporary/waived promotional fees."
    )

    launch_leaf = evaluator.add_leaf(
        id="Lowest_Fee_XRP_ETF_Launch_Date",
        desc="Launch date is provided",
        parent=trading,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF launched/listed on '{(etf.launch_date or '').strip()}'.",
        node=launch_leaf,
        sources=mgmt_sources,
        additional_instruction="Accept 'inception'/'listing' date if clearly referring to the first trading date."
    )

    exch_leaf = evaluator.add_leaf(
        id="Lowest_Fee_XRP_ETF_Primary_Exchange",
        desc="Primary listing exchange is provided",
        parent=trading,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF is primarily listed on '{(etf.primary_exchange or '').strip()}'.",
        node=exch_leaf,
        sources=mgmt_sources,
        additional_instruction="Issuer/exchange/fact sheet pages are acceptable."
    )

    cust_leaf = evaluator.add_leaf(
        id="Lowest_Fee_XRP_ETF_Custodian_Information",
        desc="Custodian(s) information is provided",
        parent=trading,
        critical=True
    )
    cust_sources = _merge_urls(getattr(etf, "custodian_sources", []), getattr(etf, "trading_sources", []))
    cust_list = ", ".join(getattr(etf, "custodians", []) or [])
    await evaluator.verify(
        claim=f"The ETF lists the following custodian(s): {cust_list}.",
        node=cust_leaf,
        sources=cust_sources,
        additional_instruction="Verify custodian(s) listed for the ETF."
    )

    tr_ref = evaluator.add_parallel(
        id="Lowest_Fee_XRP_ETF_Trading_Reference",
        desc="Reference documentation for trading details",
        parent=trading,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(mgmt_sources) > 0,
        id="Lowest_Fee_XRP_ETF_Trading_Source_URL",
        desc="Valid URL source verifying trading details",
        parent=tr_ref,
        critical=True
    )


async def verify_multi_custodian_etf(evaluator: Evaluator, parent, etf: Optional[ETFInfo]) -> None:
    """
    Build and verify the 'Multi_Custodian_XRP_ETF' subtree.
    """
    node = evaluator.add_parallel(
        id="Multi_Custodian_XRP_ETF",
        desc="Identify and provide details about an XRP ETF that uses three or more custodians",
        parent=parent,
        critical=False
    )

    # Identification (critical)
    ident = evaluator.add_parallel(
        id="Multi_Custodian_XRP_ETF_ETF_Identification",
        desc="Basic identification of the ETF meeting the selection criterion",
        parent=node,
        critical=True
    )

    sel_leaf = evaluator.add_leaf(
        id="Multi_Custodian_XRP_ETF_Selection_Criterion_Met",
        desc="The identified ETF uses three or more distinct custodians",
        parent=ident,
        critical=True
    )
    sel_sources = _merge_urls(getattr(etf, "selection_sources", []), getattr(etf, "custodian_sources", []), getattr(etf, "id_sources", []))
    await evaluator.verify(
        claim=f"This ETF ({(etf.name or 'the ETF')}) uses three or more distinct custodians.",
        node=sel_leaf,
        sources=sel_sources,
        additional_instruction="Verify that at least three distinct custodian entities are identified for the ETF; terms may include 'custodian' or 'sub-custodian'."
    )

    tick_leaf = evaluator.add_leaf(
        id="Multi_Custodian_XRP_ETF_Ticker_Symbol",
        desc="Correct ticker symbol is provided",
        parent=ident,
        critical=True
    )
    t_sources = _merge_urls(getattr(etf, "trading_sources", []), getattr(etf, "id_sources", []))
    await evaluator.verify(
        claim=f"The ETF's ticker symbol is '{(etf.ticker or '').strip()}'.",
        node=tick_leaf,
        sources=t_sources,
        additional_instruction="Verify the ticker on issuer or exchange pages."
    )

    id_ref = evaluator.add_parallel(
        id="Multi_Custodian_XRP_ETF_Identification_Reference",
        desc="Reference documentation for ETF identification",
        parent=ident,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(sel_sources) > 0,
        id="Multi_Custodian_XRP_ETF_Identification_Source_URL",
        desc="Valid URL source verifying ETF identity",
        parent=id_ref,
        critical=True
    )

    # Trading details (critical)
    trading = evaluator.add_parallel(
        id="Multi_Custodian_XRP_ETF_Trading_Details",
        desc="Detailed trading and operational information about the ETF",
        parent=node,
        critical=True
    )

    mgmt_leaf = evaluator.add_leaf(
        id="Multi_Custodian_XRP_ETF_Management_Fee",
        desc="Stated management fee percentage is provided",
        parent=trading,
        critical=True
    )
    mgmt_fee_str = _effective_fee_str(etf)
    mgmt_sources = _merge_urls(getattr(etf, "trading_sources", []), getattr(etf, "id_sources", []))
    await evaluator.verify(
        claim=f"The ETF's permanent (non-waived) stated management fee (expense ratio) is '{mgmt_fee_str}'.",
        node=mgmt_leaf,
        sources=mgmt_sources,
        additional_instruction="Verify the ongoing management fee; ignore temporary waivers."
    )

    launch_leaf = evaluator.add_leaf(
        id="Multi_Custodian_XRP_ETF_Launch_Date",
        desc="Launch date is provided",
        parent=trading,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF launched/listed on '{(etf.launch_date or '').strip()}'.",
        node=launch_leaf,
        sources=mgmt_sources,
        additional_instruction="Accept 'inception'/'listing' date as launch if clearly stated."
    )

    exch_leaf = evaluator.add_leaf(
        id="Multi_Custodian_XRP_ETF_Primary_Exchange",
        desc="Primary listing exchange is provided",
        parent=trading,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF is primarily listed on '{(etf.primary_exchange or '').strip()}'.",
        node=exch_leaf,
        sources=mgmt_sources,
        additional_instruction="Issuer/exchange/fact sheet pages are acceptable."
    )

    # Custodian details (critical, with three leaves)
    cust_details = evaluator.add_parallel(
        id="Multi_Custodian_XRP_ETF_Custodian_Details",
        desc="Detailed information about all custodians (minimum 3 required)",
        parent=trading,
        critical=True
    )

    cust_sources = _merge_urls(getattr(etf, "custodian_sources", []), getattr(etf, "trading_sources", []))
    cust_names = getattr(etf, "custodians", []) or []

    # First custodian
    first_cust_leaf = evaluator.add_leaf(
        id="Multi_Custodian_XRP_ETF_First_Custodian",
        desc="Name of first custodian is provided",
        parent=cust_details,
        critical=True
    )
    await evaluator.verify(
        claim=f"One of the ETF's custodians is '{cust_names[0] if len(cust_names) > 0 else ''}'.",
        node=first_cust_leaf,
        sources=cust_sources,
        additional_instruction="Verify that the custodian name appears on the page."
    )

    # Second custodian
    second_cust_leaf = evaluator.add_leaf(
        id="Multi_Custodian_XRP_ETF_Second_Custodian",
        desc="Name of second custodian is provided",
        parent=cust_details,
        critical=True
    )
    await evaluator.verify(
        claim=f"Another ETF custodian is '{cust_names[1] if len(cust_names) > 1 else ''}'.",
        node=second_cust_leaf,
        sources=cust_sources,
        additional_instruction="Verify the second custodian name appears on the page."
    )

    # Third custodian
    third_cust_leaf = evaluator.add_leaf(
        id="Multi_Custodian_XRP_ETF_Third_Custodian",
        desc="Name of third custodian is provided",
        parent=cust_details,
        critical=True
    )
    await evaluator.verify(
        claim=f"A third ETF custodian is '{cust_names[2] if len(cust_names) > 2 else ''}'.",
        node=third_cust_leaf,
        sources=cust_sources,
        additional_instruction="Verify the third custodian name appears on the page."
    )

    # Custodian reference - presence of URL
    cust_ref = evaluator.add_parallel(
        id="Multi_Custodian_XRP_ETF_Custodian_Reference",
        desc="Reference documentation for custodian information",
        parent=cust_details,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(cust_sources) > 0,
        id="Multi_Custodian_XRP_ETF_Custodian_Source_URL",
        desc="Valid URL source verifying custodian details",
        parent=cust_ref,
        critical=True
    )

    # Trading reference - presence of URL
    tr_ref = evaluator.add_parallel(
        id="Multi_Custodian_XRP_ETF_Trading_Reference",
        desc="Reference documentation for trading details",
        parent=trading,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(mgmt_sources) > 0,
        id="Multi_Custodian_XRP_ETF_Trading_Source_URL",
        desc="Valid URL source verifying trading details",
        parent=tr_ref,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Verification builders for Program sections                                  #
# --------------------------------------------------------------------------- #
async def verify_oregon_kicker(evaluator: Evaluator, parent, info: Optional[OregonKickerInfo]) -> None:
    """
    Build and verify the 'Oregon_Kicker_Program' subtree.
    """
    node = evaluator.add_parallel(
        id="Oregon_Kicker_Program",
        desc="Provide details about Oregon's surplus revenue (kicker) rebate program for 2025 tax returns",
        parent=parent,
        critical=False
    )

    # Program Identification (critical)
    ident = evaluator.add_parallel(
        id="Oregon_Kicker_Program_Identification",
        desc="Basic identification of the Oregon tax rebate program",
        parent=node,
        critical=True
    )

    state_leaf = evaluator.add_leaf(
        id="Oregon_Kicker_Program_State_Name",
        desc="State is identified as Oregon",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim="The program is in the state of Oregon.",
        node=state_leaf,
        sources=getattr(info, "id_sources", []),
        additional_instruction="The page should clearly indicate Oregon (preferably Department of Revenue or an official Oregon government domain)."
    )

    name_leaf = evaluator.add_leaf(
        id="Oregon_Kicker_Program_Program_Name",
        desc="Program name or type is provided (e.g., kicker, surplus rebate)",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim="The program is commonly called the 'kicker' (surplus revenue rebate).",
        node=name_leaf,
        sources=getattr(info, "id_sources", []),
        additional_instruction="Accept 'kicker', 'surplus revenue rebate', or equivalent terminology on official Oregon sources."
    )

    id_ref = evaluator.add_parallel(
        id="Oregon_Kicker_Program_Program_Reference",
        desc="Reference documentation for program identification",
        parent=ident,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(getattr(info, "id_sources", [])) > 0,
        id="Oregon_Kicker_Program_Identification_Source_URL",
        desc="Valid URL source from official Oregon government or tax authority",
        parent=id_ref,
        critical=True
    )

    # Program Details (critical)
    details = evaluator.add_parallel(
        id="Oregon_Kicker_Program_Details",
        desc="Detailed eligibility and calculation information",
        parent=node,
        critical=True
    )

    pct_leaf = evaluator.add_leaf(
        id="Oregon_Kicker_Program_Kicker_Percentage_2025",
        desc="The 2025 kicker percentage (9.863%) is provided",
        parent=details,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2025 kicker percentage is '{(getattr(info, 'kicker_percentage_2025', '') or '').strip()}'.",
        node=pct_leaf,
        sources=getattr(info, "details_sources", []),
        additional_instruction=f"Verify the official percentage (expected {OREGON_2025_KICKER_PCT}) shown on Oregon's official sources."
    )

    # Eligibility requirements (critical)
    elig = evaluator.add_parallel(
        id="Oregon_Kicker_Program_Eligibility_Requirements",
        desc="Complete eligibility requirements for claiming the 2025 kicker",
        parent=details,
        critical=True
    )

    filed_leaf = evaluator.add_leaf(
        id="Oregon_Kicker_Program_Filed_2024_Return",
        desc="Requirement to have filed 2024 Oregon return is stated",
        parent=elig,
        critical=True
    )
    await evaluator.verify(
        claim="To claim the 2025 kicker, taxpayers must have filed a 2024 Oregon personal income tax return.",
        node=filed_leaf,
        sources=getattr(info, "details_sources", []),
        additional_instruction="Verify this requirement on official Oregon sources."
    )

    liability_leaf = evaluator.add_leaf(
        id="Oregon_Kicker_Program_Had_2024_Tax_Liability",
        desc="Requirement to have had 2024 Oregon tax liability is stated",
        parent=elig,
        critical=True
    )
    await evaluator.verify(
        claim="To claim the 2025 kicker, taxpayers must have had Oregon tax liability for tax year 2024.",
        node=liability_leaf,
        sources=getattr(info, "details_sources", []),
        additional_instruction="Verify that the requirement mentions 2024 Oregon tax liability."
    )

    mustfile_leaf = evaluator.add_leaf(
        id="Oregon_Kicker_Program_Must_File_2025_Return",
        desc="Requirement to file 2025 Oregon return is stated",
        parent=elig,
        critical=True
    )
    await evaluator.verify(
        claim="To receive the 2025 kicker, taxpayers must file their 2025 Oregon tax return.",
        node=mustfile_leaf,
        sources=getattr(info, "details_sources", []),
        additional_instruction="Verify that filing the 2025 Oregon return is required to claim the kicker."
    )

    elig_ref = evaluator.add_parallel(
        id="Oregon_Kicker_Program_Eligibility_Reference",
        desc="Reference documentation for eligibility requirements",
        parent=elig,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(getattr(info, "details_sources", [])) > 0,
        id="Oregon_Kicker_Program_Eligibility_Source_URL",
        desc="Valid URL source verifying eligibility requirements",
        parent=elig_ref,
        critical=True
    )

    # Calculation method (critical)
    calc_leaf = evaluator.add_leaf(
        id="Oregon_Kicker_Program_Calculation_Method",
        desc="Calculation method is provided (2024 tax liability × 9.863%)",
        parent=details,
        critical=True
    )
    await evaluator.verify(
        claim=f"The calculation method is based on 2024 Oregon tax liability multiplied by {OREGON_2025_KICKER_PCT}.",
        node=calc_leaf,
        sources=getattr(info, "details_sources", []),
        additional_instruction="Verify phrasing such as 'multiply 2024 tax liability by the kicker percentage' or equivalent on official sources."
    )

    details_ref = evaluator.add_parallel(
        id="Oregon_Kicker_Program_Details_Reference",
        desc="Reference documentation for program details",
        parent=details,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(getattr(info, "details_sources", [])) > 0,
        id="Oregon_Kicker_Program_Details_Source_URL",
        desc="Valid URL source verifying calculation method and program details",
        parent=details_ref,
        critical=True
    )


async def verify_alaska_pfd(evaluator: Evaluator, parent, info: Optional[AlaskaPFDInfo]) -> None:
    """
    Build and verify the 'Alaska_PFD_Program' subtree.
    """
    node = evaluator.add_parallel(
        id="Alaska_PFD_Program",
        desc="Provide details about Alaska's Permanent Fund Dividend program for 2025",
        parent=parent,
        critical=False
    )

    # Program Identification (critical)
    ident = evaluator.add_parallel(
        id="Alaska_PFD_Program_Identification",
        desc="Basic identification of the Alaska dividend program",
        parent=node,
        critical=True
    )

    state_leaf = evaluator.add_leaf(
        id="Alaska_PFD_Program_State_Name",
        desc="State is identified as Alaska",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim="The program is in the state of Alaska.",
        node=state_leaf,
        sources=getattr(info, "id_sources", []),
        additional_instruction="The page should clearly indicate Alaska (official Alaska government domains preferred)."
    )

    name_leaf = evaluator.add_leaf(
        id="Alaska_PFD_Program_Program_Name",
        desc="Program name is provided (Permanent Fund Dividend or PFD)",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim="The program is called the Permanent Fund Dividend (PFD).",
        node=name_leaf,
        sources=getattr(info, "id_sources", []),
        additional_instruction="Accept 'Permanent Fund Dividend' or 'PFD' on official Alaska sources."
    )

    id_ref = evaluator.add_parallel(
        id="Alaska_PFD_Program_Program_Reference",
        desc="Reference documentation for program identification",
        parent=ident,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(getattr(info, "id_sources", [])) > 0,
        id="Alaska_PFD_Program_Identification_Source_URL",
        desc="Valid URL source from official Alaska government source",
        parent=id_ref,
        critical=True
    )

    # Program Details (critical)
    details = evaluator.add_parallel(
        id="Alaska_PFD_Program_Details",
        desc="Detailed amount and eligibility information",
        parent=node,
        critical=True
    )

    amt_leaf = evaluator.add_leaf(
        id="Alaska_PFD_Program_Dividend_Amount_2025",
        desc="The 2025 PFD amount ($1,000) is provided",
        parent=details,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2025 PFD amount is '{(getattr(info, 'dividend_amount_2025', '') or '').strip()}'.",
        node=amt_leaf,
        sources=getattr(info, "details_sources", []),
        additional_instruction=f"Verify the official 2025 PFD amount (expected {ALASKA_2025_PFD_AMOUNT}) on an official Alaska source."
    )

    elig = evaluator.add_parallel(
        id="Alaska_PFD_Program_Eligibility_Requirements",
        desc="Complete eligibility requirements for receiving the 2025 PFD",
        parent=details,
        critical=True
    )

    residency_leaf = evaluator.add_leaf(
        id="Alaska_PFD_Program_Full_Year_2024_Residency",
        desc="Requirement for full 2024 calendar year Alaska residency is stated",
        parent=elig,
        critical=True
    )
    await evaluator.verify(
        claim="Eligibility requires being an Alaska resident for the full 2024 calendar year.",
        node=residency_leaf,
        sources=getattr(info, "details_sources", []),
        additional_instruction="Verify residency requirements as stated by the Alaska PFD program (official)."
    )

    intent_leaf = evaluator.add_leaf(
        id="Alaska_PFD_Program_Intent_to_Remain",
        desc="Requirement to intend to remain permanently in Alaska is stated",
        parent=elig,
        critical=True
    )
    await evaluator.verify(
        claim="Eligibility requires the intent to remain in Alaska permanently.",
        node=intent_leaf,
        sources=getattr(info, "details_sources", []),
        additional_instruction="Verify 'intent to remain' or equivalent language on the official eligibility page."
    )

    elig_ref = evaluator.add_parallel(
        id="Alaska_PFD_Program_Eligibility_Reference",
        desc="Reference documentation for eligibility requirements",
        parent=elig,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(getattr(info, "details_sources", [])) > 0,
        id="Alaska_PFD_Program_Eligibility_Source_URL",
        desc="Valid URL source verifying eligibility requirements",
        parent=elig_ref,
        critical=True
    )

    details_ref = evaluator.add_parallel(
        id="Alaska_PFD_Program_Details_Reference",
        desc="Reference documentation for program details",
        parent=details,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(getattr(info, "details_sources", [])) > 0,
        id="Alaska_PFD_Program_Details_Source_URL",
        desc="Valid URL source verifying dividend amount",
        parent=details_ref,
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
    """
    Evaluate an answer for the XRP ETFs and State Rebate Programs task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: parallel across subsections
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

    # Extract ETF and program info in parallel
    etfs_task = evaluator.extract(
        prompt=prompt_extract_etfs(),
        template_class=ETFExtraction,
        extraction_name="xrp_etf_extraction"
    )
    progs_task = evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )
    etfs, programs = await asyncio.gather(etfs_task, progs_task)

    # Add ground-truth hints for context (non-scoring)
    evaluator.add_ground_truth({
        "oregon_expected_kicker_percentage_2025": OREGON_2025_KICKER_PCT,
        "alaska_expected_pfd_amount_2025": ALASKA_2025_PFD_AMOUNT
    }, gt_type="expected_values")

    # Build ETF sections
    await verify_first_mover_etf(evaluator, root, getattr(etfs, "first_mover", None))
    await verify_cost_leader_etf(evaluator, root, getattr(etfs, "cost_leader", None))
    await verify_multi_custodian_etf(evaluator, root, getattr(etfs, "multi_custodian", None))

    # Build Program sections
    await verify_oregon_kicker(evaluator, root, getattr(programs, "oregon", None))
    await verify_alaska_pfd(evaluator, root, getattr(programs, "alaska", None))

    return evaluator.get_summary()