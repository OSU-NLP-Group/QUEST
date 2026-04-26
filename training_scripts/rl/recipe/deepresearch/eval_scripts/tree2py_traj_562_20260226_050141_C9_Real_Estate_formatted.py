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
TASK_ID = "reit_sector_selection_feb2026"
TASK_DESCRIPTION = """
Identify three publicly traded U.S. real estate investment trusts (REITs), with one from each of the following three sectors: Multifamily/Residential, Data Centers, and Industrial. Each REIT must satisfy ALL of the criteria specified for its respective sector.

Multifamily/Residential REIT Requirements:
- Must be classified as a multifamily or residential apartment REIT
- Must own at least 80,000 apartment units across its portfolio
- Must have a geographic focus that includes properties in at least 3 of the following major coastal metropolitan markets: San Francisco, New York, Boston, Washington D.C., Los Angeles, or Seattle
- Must be a member of the S&P 500 index
- Must have a market capitalization between $20 billion and $30 billion as of February 2026

Data Center REIT Requirements:
- Must be classified as a data center REIT
- Must operate at least 250 data center facilities globally
- Must have a market capitalization exceeding $90 billion as of February 2026
- Must be a member of the S&P 500 index

Industrial REIT Requirements:
- Must be classified as an industrial REIT (focused on warehouse and logistics properties)
- Must be the largest industrial REIT by market capitalization as of February 2026
- Must own at least 1 billion square feet of industrial/warehouse space
- Must be a member of the S&P 500 index

For each identified REIT, provide: (1) the company name and ticker symbol, (2) verification that it meets each requirement, and (3) supporting reference URLs for each key fact.
"""

ALLOWED_COASTAL_MARKETS = {"San Francisco", "New York", "Boston", "Washington D.C.", "Los Angeles", "Seattle"}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class REITCommon(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    sector_label: Optional[str] = None
    sector_urls: List[str] = Field(default_factory=list)

    sp500_urls: List[str] = Field(default_factory=list)

    market_cap_feb_2026: Optional[str] = None
    market_cap_urls: List[str] = Field(default_factory=list)


class MultifamilyREIT(REITCommon):
    apartment_units: Optional[str] = None
    units_urls: List[str] = Field(default_factory=list)
    coastal_markets: List[str] = Field(default_factory=list)
    markets_urls: List[str] = Field(default_factory=list)


class DataCenterREIT(REITCommon):
    facility_count: Optional[str] = None
    facility_urls: List[str] = Field(default_factory=list)


class IndustrialREIT(REITCommon):
    total_sqft: Optional[str] = None
    sqft_urls: List[str] = Field(default_factory=list)
    largest_by_market_cap_urls: List[str] = Field(default_factory=list)


class REITsExtraction(BaseModel):
    multifamily: Optional[MultifamilyREIT] = None
    data_center: Optional[DataCenterREIT] = None
    industrial: Optional[IndustrialREIT] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_reits() -> str:
    return """
    Extract one REIT for each sector mentioned in the answer: Multifamily/Residential, Data Centers, and Industrial.
    For each sector, extract the following fields EXACTLY from the answer text. Do not invent anything.

    Common fields for each REIT:
    - name: company name of the REIT
    - ticker: stock ticker symbol
    - sector_label: sector classification label or description as written (e.g., "multifamily/residential", "data center", "industrial")
    - sector_urls: all URLs cited that directly support the sector classification (array)
    - sp500_urls: all URLs cited that directly support S&P 500 membership (array)
    - market_cap_feb_2026: market capitalization figure as of February 2026 (string, keep the original format like "$25B", "25,000,000,000", etc.)
    - market_cap_urls: all URLs cited for the February 2026 market cap figure (array)

    Multifamily-specific fields:
    - apartment_units: total apartment unit count (string)
    - units_urls: all URLs cited that provide the apartment unit count (array)
    - coastal_markets: list of coastal metropolitan markets (array) explicitly named in the answer; only include any of: San Francisco, New York, Boston, Washington D.C., Los Angeles, Seattle (use exact names)
    - markets_urls: all URLs cited that document the presence in these coastal markets (array)

    Data center-specific fields:
    - facility_count: total data center facilities operated globally (string)
    - facility_urls: all URLs cited that provide the facility count (array)

    Industrial-specific fields:
    - total_sqft: total owned industrial/warehouse square footage (string)
    - sqft_urls: all URLs cited that provide the square footage figure (array)
    - largest_by_market_cap_urls: all URLs cited that explicitly confirm it is the largest industrial REIT by market cap as of February 2026 (array)

    Return a JSON object with top-level keys: 'multifamily', 'data_center', 'industrial'.
    If the answer mentions multiple candidates per sector, choose the first one that is presented as satisfying the criteria.
    If a sector is missing or some fields are not provided, set that sector to null or set missing fields to null (for strings) or [] (for arrays).
    Ensure all URLs are full and valid; extract URLs exactly as shown (plain or markdown).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def first_n_markets(markets: List[str], n: int = 3) -> List[str]:
    # Select the first n that are in the allowed set; if fewer, just return whatever is present
    filtered = [m for m in markets if m in ALLOWED_COASTAL_MARKETS]
    if len(filtered) >= n:
        return filtered[:n]
    return filtered


def has_required_common_info(item: REITCommon) -> bool:
    return bool(item and item.name and item.ticker and item.sector_label and len(item.sector_urls) > 0)


def strict_source_instruction(reason: str) -> str:
    # Shared instruction used for source-grounded verification
    return (
        f"{reason}\n"
        "Strict source-grounding policy: Judge the claim only based on the provided URLs. "
        "If the provided list of URLs is empty, irrelevant, or does not explicitly support the claim, return Incorrect. "
        "Allow reasonable synonyms (e.g., 'apartment communities' for multifamily, 'warehouses/logistics' for industrial). "
        "If numbers are approximate (e.g., 'over 250', '~1 billion'), treat them as acceptable if they meet the threshold."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_multifamily_subtree(evaluator: Evaluator, root: Any, mf: Optional[MultifamilyREIT]) -> None:
    # Parent node (non-critical as per rubric)
    mf_node = evaluator.add_parallel(
        id="multifamily_reit",
        desc="Identify one multifamily/residential REIT meeting all specified criteria",
        parent=root,
        critical=False
    )

    # Required info gating (critical sibling to gate others)
    req_info_ok = bool(mf and has_required_common_info(mf))
    evaluator.add_custom_node(
        result=req_info_ok,
        id="multifamily_required_info",
        desc="Required info present for multifamily REIT (name, ticker, sector classification URL)",
        parent=mf_node,
        critical=True
    )

    if not mf:
        # Still construct subtree nodes with verification attempts; auto-preconditions will skip due to failed critical sibling
        mf = MultifamilyREIT()

    # Sector classification (critical)
    sector_node = evaluator.add_parallel(
        id="multifamily_sector_classification",
        desc="The REIT is classified as a multifamily or residential apartment REIT",
        parent=mf_node,
        critical=True
    )

    # sector_verification
    sector_ver_leaf = evaluator.add_leaf(
        id="sector_verification",
        desc="Verification that the REIT's primary business is owning and operating multifamily residential apartment properties",
        parent=sector_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{mf.name or 'The company'} ({mf.ticker or 'ticker unknown'}) is a multifamily/residential apartment REIT, primarily owning and operating apartment communities.",
        node=sector_ver_leaf,
        sources=mf.sector_urls,
        additional_instruction=strict_source_instruction(
            "Confirm the sector classification from investor relations, Nareit, or other authoritative sources."
        )
    )

    # sector_reference
    sector_ref_leaf = evaluator.add_leaf(
        id="sector_reference",
        desc="URL reference confirming the REIT's multifamily/residential sector classification",
        parent=sector_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one cited source explicitly classifies {mf.name or 'the company'} as a multifamily or residential apartment REIT.",
        node=sector_ref_leaf,
        sources=mf.sector_urls,
        additional_instruction=strict_source_instruction(
            "The source must clearly state multifamily/residential classification."
        )
    )

    # Portfolio size: >= 80,000 units (critical)
    portfolio_node = evaluator.add_parallel(
        id="multifamily_portfolio_size",
        desc="The REIT owns at least 80,000 apartment units in its portfolio",
        parent=mf_node,
        critical=True
    )

    units_threshold_leaf = evaluator.add_leaf(
        id="unit_count_verification",
        desc="Verification that the total apartment unit count is 80,000 or more",
        parent=portfolio_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{mf.name or 'The company'} owns at least 80,000 apartment units across its portfolio.",
        node=units_threshold_leaf,
        sources=mf.units_urls,
        additional_instruction=strict_source_instruction(
            "Check the total apartment unit count; approximations like 'over 80,000' satisfy the threshold."
        )
    )

    units_reference_leaf = evaluator.add_leaf(
        id="portfolio_reference",
        desc="URL reference providing the specific apartment unit count",
        parent=portfolio_node,
        critical=True
    )
    units_str = mf.apartment_units or "unknown"
    await evaluator.verify(
        claim=f"The apartment unit count for {mf.name or 'the company'} is {units_str}.",
        node=units_reference_leaf,
        sources=mf.units_urls,
        additional_instruction=strict_source_instruction(
            "Verify the specific unit count stated in the answer."
        )
    )

    # Geographic focus: at least 3 specified coastal markets (critical)
    geo_node = evaluator.add_parallel(
        id="multifamily_geographic_focus",
        desc="The REIT has a geographic focus including at least 3 major coastal metropolitan markets from the list: San Francisco, New York, Boston, Washington D.C., Los Angeles, Seattle",
        parent=mf_node,
        critical=True
    )

    coastal_ver_node = evaluator.add_parallel(
        id="coastal_markets_verification",
        desc="Verification that the REIT operates in at least 3 of the specified coastal markets",
        parent=geo_node,
        critical=True
    )

    markets_selected = first_n_markets(mf.coastal_markets or [], 3)
    # Ensure we create 3 market presence checks
    for i in range(3):
        market_name = markets_selected[i] if i < len(markets_selected) else "UNKNOWN"
        presence_leaf = evaluator.add_leaf(
            id=f"market_{i+1}_presence",
            desc=f"Presence confirmed in {'first' if i==0 else ('second' if i==1 else 'third')} qualifying coastal market",
            parent=coastal_ver_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"{mf.name or 'The company'} operates in the {market_name} metropolitan market.",
            node=presence_leaf,
            sources=mf.markets_urls,
            additional_instruction=strict_source_instruction(
                "Confirm presence/operations or owned properties in the specified coastal market. "
                "Allow synonyms and regional naming (e.g., 'Bay Area' for San Francisco; 'Greater Washington' for Washington D.C.)."
            )
        )

    geo_ref_leaf = evaluator.add_leaf(
        id="geographic_reference",
        desc="URL reference documenting the REIT's presence in the specified coastal markets",
        parent=geo_node,
        critical=True
    )
    stated_markets = ", ".join(markets_selected) if markets_selected else "none"
    await evaluator.verify(
        claim=f"{mf.name or 'The company'} operates in at least three of the specified coastal markets (e.g., {stated_markets}).",
        node=geo_ref_leaf,
        sources=mf.markets_urls,
        additional_instruction=strict_source_instruction(
            "The sources should document presence in at least three of the specified markets."
        )
    )

    # S&P 500 membership (critical)
    sp_node = evaluator.add_parallel(
        id="multifamily_sp500_membership",
        desc="The REIT is a member of the S&P 500 index",
        parent=mf_node,
        critical=True
    )

    sp_verify_leaf = evaluator.add_leaf(
        id="sp500_verification",
        desc="Verification of S&P 500 membership status",
        parent=sp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{mf.name or 'The company'} ({mf.ticker or 'ticker'}) is a current member of the S&P 500 index.",
        node=sp_verify_leaf,
        sources=mf.sp500_urls,
        additional_instruction=strict_source_instruction(
            "Prefer authoritative sources (S&P Global indices pages, official announcements, or reputable financial platforms) confirming membership."
        )
    )

    sp_ref_leaf = evaluator.add_leaf(
        id="sp500_reference",
        desc="URL reference confirming S&P 500 membership",
        parent=sp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one cited source explicitly confirms that {mf.name or 'the company'} is in the S&P 500 index.",
        node=sp_ref_leaf,
        sources=mf.sp500_urls,
        additional_instruction=strict_source_instruction(
            "The cited source must explicitly mention S&P 500 index membership."
        )
    )

    # Market cap range $20B–$30B (critical)
    mcap_node = evaluator.add_parallel(
        id="multifamily_market_cap",
        desc="The REIT has a market capitalization between $20 billion and $30 billion as of February 2026",
        parent=mf_node,
        critical=True
    )

    mcap_range_node = evaluator.add_parallel(
        id="market_cap_range_verification",
        desc="Verification that market cap is within the $20B-$30B range",
        parent=mcap_node,
        critical=True
    )

    mcap_lower_leaf = evaluator.add_leaf(
        id="market_cap_lower_bound",
        desc="Market cap is at least $20 billion",
        parent=mcap_range_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of February 2026, {mf.name or 'the company'} has a market capitalization of at least $20 billion.",
        node=mcap_lower_leaf,
        sources=mf.market_cap_urls,
        additional_instruction=strict_source_instruction(
            "Use February 2026 market cap figure(s); rounding and close approximations are acceptable if clearly ≥ $20B."
        )
    )

    mcap_upper_leaf = evaluator.add_leaf(
        id="market_cap_upper_bound",
        desc="Market cap does not exceed $30 billion",
        parent=mcap_range_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of February 2026, {mf.name or 'the company'} has a market capitalization no greater than $30 billion.",
        node=mcap_upper_leaf,
        sources=mf.market_cap_urls,
        additional_instruction=strict_source_instruction(
            "Confirm the market cap does not exceed $30B in February 2026."
        )
    )

    mcap_ref_leaf = evaluator.add_leaf(
        id="market_cap_reference",
        desc="URL reference providing the February 2026 market capitalization figure",
        parent=mcap_node,
        critical=True
    )
    mcap_str = mf.market_cap_feb_2026 or "unknown"
    await evaluator.verify(
        claim=f"The market capitalization of {mf.name or 'the company'} as of February 2026 is {mcap_str}.",
        node=mcap_ref_leaf,
        sources=mf.market_cap_urls,
        additional_instruction=strict_source_instruction(
            "Verify the specific market cap figure provided for February 2026."
        )
    )


async def build_datacenter_subtree(evaluator: Evaluator, root: Any, dc: Optional[DataCenterREIT]) -> None:
    dc_node = evaluator.add_parallel(
        id="data_center_reit",
        desc="Identify one data center REIT meeting all specified criteria",
        parent=root,
        critical=False
    )

    req_info_ok = bool(dc and has_required_common_info(dc))
    evaluator.add_custom_node(
        result=req_info_ok,
        id="datacenter_required_info",
        desc="Required info present for data center REIT (name, ticker, sector classification URL)",
        parent=dc_node,
        critical=True
    )

    if not dc:
        dc = DataCenterREIT()

    # Sector classification
    sector_node = evaluator.add_parallel(
        id="datacenter_sector_classification",
        desc="The REIT is classified as a data center REIT",
        parent=dc_node,
        critical=True
    )

    sector_ver_leaf = evaluator.add_leaf(
        id="datacenter_sector_verification",
        desc="Verification that the REIT's primary business is owning and operating data center facilities",
        parent=sector_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{dc.name or 'The company'} ({dc.ticker or 'ticker unknown'}) is a data center REIT that owns and operates data center facilities.",
        node=sector_ver_leaf,
        sources=dc.sector_urls,
        additional_instruction=strict_source_instruction(
            "Confirm data center sector classification from authoritative sources."
        )
    )

    sector_ref_leaf = evaluator.add_leaf(
        id="datacenter_sector_reference",
        desc="URL reference confirming the REIT's data center sector classification",
        parent=sector_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one cited source explicitly classifies {dc.name or 'the company'} as a data center REIT.",
        node=sector_ref_leaf,
        sources=dc.sector_urls,
        additional_instruction=strict_source_instruction(
            "The source must explicitly state 'data center REIT' or equivalent."
        )
    )

    # Facility count ≥ 250
    fac_node = evaluator.add_parallel(
        id="datacenter_facility_count",
        desc="The REIT operates at least 250 data center facilities globally",
        parent=dc_node,
        critical=True
    )

    fac_threshold_leaf = evaluator.add_leaf(
        id="facility_count_verification",
        desc="Verification that the REIT operates 250 or more data center facilities",
        parent=fac_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{dc.name or 'The company'} operates at least 250 data center facilities globally.",
        node=fac_threshold_leaf,
        sources=dc.facility_urls,
        additional_instruction=strict_source_instruction(
            "Approximate phrasing like 'over 250' is acceptable."
        )
    )

    fac_ref_leaf = evaluator.add_leaf(
        id="facility_count_reference",
        desc="URL reference providing the specific data center facility count",
        parent=fac_node,
        critical=True
    )
    fac_str = dc.facility_count or "unknown"
    await evaluator.verify(
        claim=f"The specific facility count for {dc.name or 'the company'} is {fac_str}.",
        node=fac_ref_leaf,
        sources=dc.facility_urls,
        additional_instruction=strict_source_instruction(
            "Verify the exact or stated facility count."
        )
    )

    # Market cap > $90B
    mcap_node = evaluator.add_parallel(
        id="datacenter_market_cap",
        desc="The REIT has a market capitalization exceeding $90 billion as of February 2026",
        parent=dc_node,
        critical=True
    )

    mcap_verify_leaf = evaluator.add_leaf(
        id="datacenter_market_cap_verification",
        desc="Verification that market cap exceeds $90 billion",
        parent=mcap_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of February 2026, {dc.name or 'the company'} has a market capitalization exceeding $90 billion.",
        node=mcap_verify_leaf,
        sources=dc.market_cap_urls,
        additional_instruction=strict_source_instruction(
            "Confirm that market cap is > $90B."
        )
    )

    mcap_ref_leaf = evaluator.add_leaf(
        id="datacenter_market_cap_reference",
        desc="URL reference providing the February 2026 market capitalization figure",
        parent=mcap_node,
        critical=True
    )
    dc_mcap_str = dc.market_cap_feb_2026 or "unknown"
    await evaluator.verify(
        claim=f"The market capitalization of {dc.name or 'the company'} as of February 2026 is {dc_mcap_str}.",
        node=mcap_ref_leaf,
        sources=dc.market_cap_urls,
        additional_instruction=strict_source_instruction(
            "Verify the specific market cap figure provided for February 2026."
        )
    )

    # S&P 500 membership
    sp_node = evaluator.add_parallel(
        id="datacenter_sp500_membership",
        desc="The REIT is a member of the S&P 500 index",
        parent=dc_node,
        critical=True
    )

    sp_verify_leaf = evaluator.add_leaf(
        id="datacenter_sp500_verification",
        desc="Verification of S&P 500 membership status",
        parent=sp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{dc.name or 'The company'} ({dc.ticker or 'ticker'}) is a current member of the S&P 500 index.",
        node=sp_verify_leaf,
        sources=dc.sp500_urls,
        additional_instruction=strict_source_instruction(
            "Prefer authoritative sources confirming S&P 500 membership."
        )
    )

    sp_ref_leaf = evaluator.add_leaf(
        id="datacenter_sp500_reference",
        desc="URL reference confirming S&P 500 membership",
        parent=sp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one cited source explicitly confirms that {dc.name or 'the company'} is in the S&P 500 index.",
        node=sp_ref_leaf,
        sources=dc.sp500_urls,
        additional_instruction=strict_source_instruction(
            "The cited source must explicitly mention S&P 500 membership."
        )
    )


async def build_industrial_subtree(evaluator: Evaluator, root: Any, ind: Optional[IndustrialREIT]) -> None:
    ind_node = evaluator.add_parallel(
        id="industrial_reit",
        desc="Identify one industrial REIT meeting all specified criteria",
        parent=root,
        critical=False
    )

    req_info_ok = bool(ind and has_required_common_info(ind))
    evaluator.add_custom_node(
        result=req_info_ok,
        id="industrial_required_info",
        desc="Required info present for industrial REIT (name, ticker, sector classification URL)",
        parent=ind_node,
        critical=True
    )

    if not ind:
        ind = IndustrialREIT()

    # Sector classification
    sector_node = evaluator.add_parallel(
        id="industrial_sector_classification",
        desc="The REIT is classified as an industrial REIT",
        parent=ind_node,
        critical=True
    )

    sector_ver_leaf = evaluator.add_leaf(
        id="industrial_sector_verification",
        desc="Verification that the REIT's primary business is owning and operating industrial warehouse and logistics properties",
        parent=sector_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ind.name or 'The company'} ({ind.ticker or 'ticker unknown'}) is an industrial REIT focused on warehouses and logistics properties.",
        node=sector_ver_leaf,
        sources=ind.sector_urls,
        additional_instruction=strict_source_instruction(
            "Confirm industrial sector classification from authoritative sources."
        )
    )

    sector_ref_leaf = evaluator.add_leaf(
        id="industrial_sector_reference",
        desc="URL reference confirming the REIT's industrial sector classification",
        parent=sector_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one cited source explicitly classifies {ind.name or 'the company'} as an industrial REIT.",
        node=sector_ref_leaf,
        sources=ind.sector_urls,
        additional_instruction=strict_source_instruction(
            "The source must explicitly state 'industrial REIT' or equivalent."
        )
    )

    # Largest industrial REIT by market cap (critical)
    leadership_node = evaluator.add_parallel(
        id="industrial_market_leadership",
        desc="The REIT is the largest industrial REIT by market capitalization as of February 2026",
        parent=ind_node,
        critical=True
    )

    leadership_verify_leaf = evaluator.add_leaf(
        id="market_cap_leadership_verification",
        desc="Verification that this REIT has the highest market cap among all industrial REITs",
        parent=leadership_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of February 2026, {ind.name or 'the company'} is the largest industrial REIT by market capitalization.",
        node=leadership_verify_leaf,
        sources=ind.largest_by_market_cap_urls,
        additional_instruction=strict_source_instruction(
            "The sources should compare or explicitly assert that this is the largest industrial REIT by market cap."
        )
    )

    leadership_ref_leaf = evaluator.add_leaf(
        id="market_leadership_reference",
        desc="URL reference confirming the REIT's status as the largest industrial REIT by market cap",
        parent=leadership_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one cited source confirms that {ind.name or 'the company'} is the largest industrial REIT by market capitalization (February 2026).",
        node=leadership_ref_leaf,
        sources=ind.largest_by_market_cap_urls,
        additional_instruction=strict_source_instruction(
            "Confirm explicit 'largest by market cap' status among industrial REITs."
        )
    )

    # Portfolio size ≥ 1 billion sq ft (critical)
    sqft_node = evaluator.add_parallel(
        id="industrial_portfolio_size",
        desc="The REIT owns at least 1 billion square feet of industrial/warehouse space",
        parent=ind_node,
        critical=True
    )

    sqft_threshold_leaf = evaluator.add_leaf(
        id="square_footage_verification",
        desc="Verification that the REIT's total industrial space is 1 billion square feet or more",
        parent=sqft_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ind.name or 'The company'} owns at least 1 billion square feet of industrial/warehouse space.",
        node=sqft_threshold_leaf,
        sources=ind.sqft_urls,
        additional_instruction=strict_source_instruction(
            "Accept '1 billion+', '~1 billion', or explicit figures ≥ 1,000,000,000 sq ft."
        )
    )

    sqft_ref_leaf = evaluator.add_leaf(
        id="portfolio_size_reference",
        desc="URL reference providing the specific square footage of industrial space owned",
        parent=sqft_node,
        critical=True
    )
    sqft_str = ind.total_sqft or "unknown"
    await evaluator.verify(
        claim=f"The total industrial/warehouse square footage owned by {ind.name or 'the company'} is {sqft_str}.",
        node=sqft_ref_leaf,
        sources=ind.sqft_urls,
        additional_instruction=strict_source_instruction(
            "Verify the specific square footage figure stated in the answer."
        )
    )

    # S&P 500 membership
    sp_node = evaluator.add_parallel(
        id="industrial_sp500_membership",
        desc="The REIT is a member of the S&P 500 index",
        parent=ind_node,
        critical=True
    )

    sp_verify_leaf = evaluator.add_leaf(
        id="industrial_sp500_verification",
        desc="Verification of S&P 500 membership status",
        parent=sp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ind.name or 'The company'} ({ind.ticker or 'ticker'}) is a current member of the S&P 500 index.",
        node=sp_verify_leaf,
        sources=ind.sp500_urls,
        additional_instruction=strict_source_instruction(
            "Prefer authoritative sources confirming S&P 500 membership."
        )
    )

    sp_ref_leaf = evaluator.add_leaf(
        id="industrial_sp500_reference",
        desc="URL reference confirming S&P 500 membership",
        parent=sp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one cited source explicitly confirms that {ind.name or 'the company'} is in the S&P 500 index.",
        node=sp_ref_leaf,
        sources=ind.sp500_urls,
        additional_instruction=strict_source_instruction(
            "The cited source must explicitly mention S&P 500 membership."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the REIT sector selection and verification task (February 2026).
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates three sector subtrees independently
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

    # 1) Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_reits(),
        template_class=REITsExtraction,
        extraction_name="reits_extraction"
    )

    # 2) Build verification subtrees per sector
    await build_multifamily_subtree(evaluator, root, extraction.multifamily)
    await build_datacenter_subtree(evaluator, root, extraction.data_center)
    await build_industrial_subtree(evaluator, root, extraction.industrial)

    # 3) Return evaluation summary
    return evaluator.get_summary()