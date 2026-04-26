import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "dividend_aristocrats_largest_etf_2026"
TASK_DESCRIPTION = (
    "As of early 2026, identify the largest exchange-traded fund (ETF) by assets under management that specifically "
    "tracks the S&P 500 Dividend Aristocrats index. For this ETF, provide comprehensive verification including:\n\n"
    "1. Product Details: The complete official name of the ETF, its ticker symbol, the issuing asset management company, "
    "and the primary stock exchange where it is listed.\n\n"
    "2. Asset Scale: The approximate assets under management (AUM) figure, along with confirmation that this is indeed "
    "the largest ETF focused on tracking the S&P 500 Dividend Aristocrats.\n\n"
    "3. SEC Regulatory Compliance: Verification that the ETF complies with SEC Rule 6c-11 requirements for transparent ETFs, "
    "including confirmation that it provides daily portfolio holdings disclosure on its website before market open and displays the "
    "median bid-ask spread calculated over a 30-day period as required by the rule. Provide the specific webpage location where these "
    "Rule 6c-11 disclosures can be found.\n\n"
    "4. Asset Manager Status: Confirmation that the issuing asset manager is an institutional investment manager that files "
    "quarterly Form 13F reports with the SEC (noting that the Q4 2025 filing deadline was February 17, 2026).\n\n"
    "All information must be supported with verifiable reference URLs from official sources."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ETFProductInfo(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    asset_manager: Optional[str] = None
    primary_exchange: Optional[str] = None
    product_urls: List[str] = Field(default_factory=list)
    index_tracked: Optional[str] = None  # e.g., "S&P 500 Dividend Aristocrats"


class AssetScaleInfo(BaseModel):
    aum: Optional[str] = None  # Keep as free text (e.g., "$12.3 billion", "about $10B")
    aum_source_urls: List[str] = Field(default_factory=list)
    largest_confirmation_statement: Optional[str] = None
    largest_confirmation_urls: List[str] = Field(default_factory=list)
    aum_comparison_urls: List[str] = Field(default_factory=list)


class SecComplianceInfo(BaseModel):
    rule6c11_status_statement: Optional[str] = None
    daily_holdings_url: Optional[str] = None
    median_bid_ask_url: Optional[str] = None
    website_disclosure_location_url: Optional[str] = None
    compliance_reference_urls: List[str] = Field(default_factory=list)


class AssetManagerInfo(BaseModel):
    name: Optional[str] = None
    official_website_url: Optional[str] = None
    form13f_status_statement: Optional[str] = None
    form13f_reference_urls: List[str] = Field(default_factory=list)
    q42025_deadline_reference_url: Optional[str] = None
    additional_regulatory_filing_urls: List[str] = Field(default_factory=list)


class ETFExtraction(BaseModel):
    product: Optional[ETFProductInfo] = None
    asset_scale: Optional[AssetScaleInfo] = None
    sec_compliance: Optional[SecComplianceInfo] = None
    asset_manager: Optional[AssetManagerInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_etf_info() -> str:
    return """
Extract structured details for the single ETF that the answer identifies as the largest ETF (by assets) tracking the S&P 500 Dividend Aristocrats index.
Only extract what is explicitly present in the answer text. Do not invent or infer.

Return a JSON with the following structure:

{
  "product": {
    "name": string|null,                       // Complete official ETF name as stated
    "ticker": string|null,                     // Ticker symbol
    "asset_manager": string|null,              // Issuing firm (e.g., ProShares, etc.)
    "primary_exchange": string|null,           // Primary listing exchange as stated (e.g., NYSE Arca, Cboe BZX)
    "product_urls": string[],                  // All official issuer product page URLs provided in the answer
    "index_tracked": string|null               // Index tracked if stated (e.g., "S&P 500 Dividend Aristocrats")
  },
  "asset_scale": {
    "aum": string|null,                        // Approximate AUM figure as text (include currency if present)
    "aum_source_urls": string[],               // URL(s) cited for AUM
    "largest_confirmation_statement": string|null, // Text in the answer asserting it is the largest
    "largest_confirmation_urls": string[],     // URL(s) cited for the "largest" confirmation
    "aum_comparison_urls": string[]            // URL(s) that compare AUM vs other Dividend Aristocrats ETFs, if any
  },
  "sec_compliance": {
    "rule6c11_status_statement": string|null,  // Any explicit statement referencing Rule 6c-11 or compliance
    "daily_holdings_url": string|null,         // URL for daily holdings disclosure (if separate page)
    "median_bid_ask_url": string|null,         // URL for 30-Day Median Bid/Ask Spread (if separate page)
    "website_disclosure_location_url": string|null, // A specific URL that contains 6c-11 disclosures (daily holdings and/or median bid-ask)
    "compliance_reference_urls": string[]      // Any additional URL(s) used to support 6c-11 compliance assertions
  },
  "asset_manager": {
    "name": string|null,                       // Asset manager name (should match product.asset_manager if present)
    "official_website_url": string|null,       // Official website homepage of the asset manager
    "form13f_status_statement": string|null,   // Statement that the manager files 13F as institutional investment manager
    "form13f_reference_urls": string[],        // URL(s) that show 13F filings or SEC EDGAR page for the manager
    "q42025_deadline_reference_url": string|null, // URL supporting that Q4 2025 13F deadline was Feb 17, 2026
    "additional_regulatory_filing_urls": string[] // Any other URLs that show 13F filings or institutional manager status
  }
}

Rules:
- Extract only URLs explicitly present in the answer. Include full URLs with protocol.
- If multiple ETFs are mentioned, choose the one the answer names as the largest Dividend Aristocrats S&P 500 ETF. If not clearly indicated, choose the first ETF presented as the main one.
- If any field is missing in the answer, set it to null (or [] for lists).
"""


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def _listify(*items: Optional[Any]) -> List[str]:
    """Flatten strings/lists of strings and drop falsy values."""
    out: List[str] = []
    seen = set()
    for it in items:
        if not it:
            continue
        if isinstance(it, str):
            if it not in seen:
                out.append(it)
                seen.add(it)
        elif isinstance(it, list):
            for s in it:
                if s and isinstance(s, str) and s not in seen:
                    out.append(s)
                    seen.add(s)
    return out


def _nz(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_product_identification_checks(
    evaluator: Evaluator,
    parent,
    product: ETFProductInfo,
) -> None:
    node = evaluator.add_parallel(
        id="ProductIdentification",
        desc="Accurate identification of the ETF product details",
        parent=parent,
        critical=True
    )

    product_sources = _listify(product.product_urls)

    # ETF Name
    leaf = evaluator.add_leaf(
        id="ETFName",
        desc="Provide the complete official name of the ETF",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official ETF name is '{_nz(product.name)}'.",
        node=leaf,
        sources=product_sources,
        additional_instruction="Verify on the official issuer product page. Allow minor punctuation/case differences."
    )

    # Ticker Symbol
    leaf = evaluator.add_leaf(
        id="TickerSymbol",
        desc="Provide the correct ticker symbol",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The ETF's ticker symbol is '{_nz(product.ticker)}'.",
        node=leaf,
        sources=product_sources,
        additional_instruction="Confirm on the official product page that the stated ticker matches."
    )

    # Asset Manager
    leaf = evaluator.add_leaf(
        id="AssetManager",
        desc="Identify the issuing asset management company",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The ETF is issued by '{_nz(product.asset_manager)}'.",
        node=leaf,
        sources=product_sources,
        additional_instruction="Confirm that the issuer (asset manager) on the page matches the stated firm."
    )

    # Primary Exchange
    leaf = evaluator.add_leaf(
        id="PrimaryExchange",
        desc="Specify the primary stock exchange where the ETF is listed",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The ETF (ticker: {_nz(product.ticker)}) is primarily listed on '{_nz(product.primary_exchange)}'.",
        node=leaf,
        sources=product_sources,
        additional_instruction="Accept reasonable variants (e.g., 'NYSE Arca' vs 'NYSE Arca, Inc.'; 'Cboe BZX' vs 'Cboe')."
    )

    # Product Reference URL validity
    leaf = evaluator.add_leaf(
        id="ProductReferenceURL",
        desc="Provide a verifiable URL with official ETF information",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This webpage is an official issuer product page providing authoritative information for the ETF '{_nz(product.name)}' (ticker: {_nz(product.ticker)}).",
        node=leaf,
        sources=product_sources,
        additional_instruction="The page should be on the issuer's official domain and clearly be the ETF's product page."
    )


async def build_asset_scale_checks(
    evaluator: Evaluator,
    parent,
    product: ETFProductInfo,
    scale: AssetScaleInfo,
) -> None:
    # Note: Parent is critical in rubric; to satisfy framework constraints,
    # all children under a critical parent must also be critical (even those originally marked non-critical).
    node = evaluator.add_parallel(
        id="AssetScaleVerification",
        desc="Verification that this is the largest Dividend Aristocrats-focused ETF",
        parent=parent,
        critical=True
    )

    # Assets Under Management
    leaf = evaluator.add_leaf(
        id="AssetsUnderManagement",
        desc="Provide the approximate AUM figure for the ETF",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The ETF has approximately '{_nz(scale.aum)}' in assets under management (AUM).",
        node=leaf,
        sources=_listify(scale.aum_source_urls, product.product_urls),
        additional_instruction="Allow approximate values and rounding. Confirm on the cited page(s) that this AUM matches or is reasonably close."
    )

    # Largest ETF Confirmation
    leaf = evaluator.add_leaf(
        id="LargestETFConfirmation",
        desc="Confirm this is the largest ETF tracking S&P 500 Dividend Aristocrats by assets",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Among ETFs that track the S&P 500 Dividend Aristocrats index, the ETF '{_nz(product.name)}' (ticker: {_nz(product.ticker)}) has the largest AUM.",
        node=leaf,
        sources=_listify(scale.largest_confirmation_urls, scale.aum_comparison_urls, scale.aum_source_urls),
        additional_instruction="Focus on ETFs specifically tracking the S&P 500 Dividend Aristocrats (not S&P High Yield Dividend Aristocrats). The evidence should support that this ETF has the largest AUM."
    )

    # AUM Comparison Reference (elevated to critical due to framework constraint)
    leaf = evaluator.add_leaf(
        id="AUMComparisonReference",
        desc="Provide evidence comparing AUM with other Dividend Aristocrats ETFs",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This webpage provides AUM (or fund size) information for multiple ETFs that track the S&P 500 Dividend Aristocrats (or clearly named 'Dividend Aristocrats'), enabling comparison.",
        node=leaf,
        sources=_listify(scale.aum_comparison_urls),
        additional_instruction="Pass if the page contains multiple Dividend Aristocrats ETFs with AUM or clear comparative statements."
    )

    # AUM Source URL validity
    leaf = evaluator.add_leaf(
        id="AUMSourceURL",
        desc="Provide URL source for AUM information",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This webpage explicitly provides the ETF's AUM (assets under management) figure or 'net assets'.",
        node=leaf,
        sources=_listify(scale.aum_source_urls),
        additional_instruction="The page should clearly present AUM/net assets for the ETF."
    )


async def build_sec_compliance_checks(
    evaluator: Evaluator,
    parent,
    product: ETFProductInfo,
    comp: SecComplianceInfo,
) -> None:
    node = evaluator.add_parallel(
        id="SECRegulatoryCompliance",
        desc="Verification of SEC Rule 6c-11 compliance and required disclosures",
        parent=parent,
        critical=True
    )

    # Rule 6c-11 compliance status (explicit or inferred via required disclosures)
    leaf = evaluator.add_leaf(
        id="Rule6c11ComplianceStatus",
        desc="Confirm the ETF operates under SEC Rule 6c-11 as a transparent ETF",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The ETF complies with SEC Rule 6c-11 for transparent ETFs, evidenced by explicit reference to Rule 6c-11 and/or presence of required disclosures on the website.",
        node=leaf,
        sources=_listify(comp.compliance_reference_urls, comp.website_disclosure_location_url, comp.daily_holdings_url, comp.median_bid_ask_url),
        additional_instruction="Pass if the page explicitly references 'Rule 6c-11' OR clearly shows both daily holdings disclosure and 30-Day median bid-ask spread."
    )

    # Daily holdings disclosure (before market open)
    leaf = evaluator.add_leaf(
        id="DailyHoldingsDisclosure",
        desc="Verify that daily portfolio holdings are disclosed on the ETF website before market open",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The ETF's website posts its complete portfolio holdings daily and does so before the market opens.",
        node=leaf,
        sources=_listify(comp.daily_holdings_url, comp.website_disclosure_location_url),
        additional_instruction="Look for language like 'daily holdings', 'before market open', or equivalent phrasing that implies daily pre-open disclosure."
    )

    # 30-Day Median Bid/Ask spread displayed
    leaf = evaluator.add_leaf(
        id="MedianBidAskSpread",
        desc="Confirm that median bid-ask spread over 30-day period is displayed on website",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The ETF's website displays the 30-Day Median Bid/Ask Spread.",
        node=leaf,
        sources=_listify(comp.median_bid_ask_url, comp.website_disclosure_location_url),
        additional_instruction="Look for '30-Day Median Bid/Ask Spread' or similar phrasing on the official page."
    )

    # Specific webpage URL for disclosures
    leaf = evaluator.add_leaf(
        id="WebsiteDisclosureLocation",
        desc="Provide the specific webpage URL where Rule 6c-11 disclosures are located",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This specific webpage contains the ETF's Rule 6c-11 disclosures (daily holdings and/or 30-Day median bid-ask spread).",
        node=leaf,
        sources=_listify(comp.website_disclosure_location_url),
        additional_instruction="Confirm that this particular URL shows at least one of: daily holdings and the 30-Day median bid/ask spread; ideally both."
    )

    # Compliance reference URL validity
    leaf = evaluator.add_leaf(
        id="ComplianceReferenceURL",
        desc="Provide URL documenting regulatory compliance information",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This webpage documents regulatory or compliance-related disclosures for the ETF (e.g., Rule 6c-11 statement, daily holdings, or 30-Day median bid-ask spread).",
        node=leaf,
        sources=_listify(comp.compliance_reference_urls),
        additional_instruction="The page should clearly be about disclosures/compliance for the ETF."
    )


async def build_asset_manager_checks(
    evaluator: Evaluator,
    parent,
    product: ETFProductInfo,
    mgr: AssetManagerInfo,
) -> None:
    node = evaluator.add_parallel(
        id="AssetManagerInformation",
        desc="Information about the asset manager's regulatory status",
        parent=parent,
        critical=True
    )

    # Form 13F filing status
    leaf = evaluator.add_leaf(
        id="Form13FFilingStatus",
        desc="Confirm the asset manager files Form 13F as an institutional investment manager",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_nz(mgr.name) or _nz(product.asset_manager)} files Form 13F with the SEC as an institutional investment manager.",
        node=leaf,
        sources=_listify(mgr.form13f_reference_urls, mgr.additional_regulatory_filing_urls),
        additional_instruction="Confirm via SEC EDGAR or other official sources showing Form 13F filings for the manager."
    )

    # Q4 2025 deadline note (elevated to critical due to framework constraint)
    leaf = evaluator.add_leaf(
        id="Q42025FilingDeadline",
        desc="Note that Q4 2025 Form 13F was due by February 17, 2026",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Q4 2025 Form 13F filing deadline was February 17, 2026.",
        node=leaf,
        sources=_listify(mgr.q42025_deadline_reference_url),
        additional_instruction="Verify the date from an authoritative SEC source or recognized compliance calendar."
    )

    # Official asset manager website URL
    leaf = evaluator.add_leaf(
        id="AssetManagerWebsite",
        desc="Provide the official website URL of the asset management company",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This is the official website of {_nz(mgr.name) or _nz(product.asset_manager)}.",
        node=leaf,
        sources=_listify(mgr.official_website_url),
        additional_instruction="The page should clearly be the official homepage for the asset manager."
    )

    # Regulatory filing reference URL validity
    leaf = evaluator.add_leaf(
        id="RegulatoryFilingReference",
        desc="Provide URL or reference to Form 13F filings or institutional investor status",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This webpage shows Form 13F filings or confirms institutional investment manager status for {_nz(mgr.name) or _nz(product.asset_manager)}.",
        node=leaf,
        sources=_listify(mgr.form13f_reference_urls, mgr.additional_regulatory_filing_urls),
        additional_instruction="A direct SEC EDGAR company page for 13F filings or similar authoritative source is preferred."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
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
        default_model=model,
    )

    # 1) Extract structured info from the answer
    extracted: ETFExtraction = await evaluator.extract(
        prompt=prompt_extract_etf_info(),
        template_class=ETFExtraction,
        extraction_name="etf_extraction"
    )

    # 2) Build top-level critical analysis node (parallel aggregation)
    analysis_node = evaluator.add_parallel(
        id="DividendAristocratsETFAnalysis",
        desc="Complete analysis of the largest S&P 500 Dividend Aristocrats ETF with regulatory compliance verification",
        parent=root,
        critical=True
    )

    # 3) Build subtrees (with robust None handling)
    product = extracted.product or ETFProductInfo()
    asset_scale = extracted.asset_scale or AssetScaleInfo()
    sec_comp = extracted.sec_compliance or SecComplianceInfo()
    asset_mgr = extracted.asset_manager or AssetManagerInfo()

    await build_product_identification_checks(evaluator, analysis_node, product)
    await build_asset_scale_checks(evaluator, analysis_node, product, asset_scale)
    await build_sec_compliance_checks(evaluator, analysis_node, product, sec_comp)
    await build_asset_manager_checks(evaluator, analysis_node, product, asset_mgr)

    # 4) Return scoring summary
    return evaluator.get_summary()