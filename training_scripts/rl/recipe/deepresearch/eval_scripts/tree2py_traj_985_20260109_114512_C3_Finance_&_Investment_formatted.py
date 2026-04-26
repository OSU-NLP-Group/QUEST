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
TASK_ID = "xrp_etf_lowest_fee_2025"
TASK_DESCRIPTION = """
Among the spot XRP exchange-traded funds (ETFs) that launched in the United States in 2025, identify the one with the lowest standard management fee (excluding any temporary fee waivers). For this ETF, provide the following information: (1) The ETF's official name and ticker symbol, (2) A direct URL to the ETF's Form S-1 registration statement filed with the U.S. Securities and Exchange Commission (SEC), (3) The name of the qualified custodian responsible for holding the ETF's XRP assets, as specified in the SEC filing, and (4) A reference URL that confirms the custodian information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CompetitorETF(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    standard_fee: Optional[str] = None  # e.g., "0.20%" or "0.19"
    fee_source_urls: List[str] = Field(default_factory=list)


class ETFSelection(BaseModel):
    selected_name: Optional[str] = None  # Official name of the chosen ETF
    selected_ticker: Optional[str] = None  # Ticker symbol
    standard_fee: Optional[str] = None  # Standard (non-waived) management fee as stated in the answer
    fee_source_urls: List[str] = Field(default_factory=list)  # URLs used to support the fee claim
    s1_url: Optional[str] = None  # Direct SEC EDGAR S-1 link for the selected ETF
    custodian_name: Optional[str] = None  # Qualified custodian per S-1
    custodian_source_urls: List[str] = Field(default_factory=list)  # Reference URLs confirming custodian
    us_launch_year: Optional[str] = None  # Expected "2025" if stated in answer
    market_location: Optional[str] = None  # e.g., "United States"
    asset_type: Optional[str] = None  # e.g., "spot XRP ETF"
    sponsor_domain: Optional[str] = None  # e.g., "blackrock.com" if sponsor site domain is provided in the answer
    sponsor_site_urls: List[str] = Field(default_factory=list)  # sponsor URLs cited (if any)
    # Optional: The answer may list competitors; extract if present
    competitors: List[CompetitorETF] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_selection() -> str:
    return """
    From the answer, extract the details for the single ETF the answer claims has the lowest standard (non-waived) management fee among U.S.-launched 2025 spot XRP ETFs.

    Required fields to extract (return null for any missing field, and return empty arrays where applicable):
    - selected_name: The ETF’s official name (as presented in the answer).
    - selected_ticker: The ETF’s ticker symbol.
    - standard_fee: The ETF’s standard (non-waived) management fee as a string exactly as written (e.g., "0.19%" or "0.20%"). Do NOT extract a waived or net-of-waiver fee.
    - fee_source_urls: All URLs cited to support the management fee (only those explicitly present in the answer).
    - s1_url: A direct URL to the ETF’s Form S-1 (or S-1/A) registration statement on the SEC’s EDGAR website (explicitly present in the answer).
    - custodian_name: The name of the qualified custodian holding the ETF’s XRP assets, as stated in the SEC filing, exactly as written in the answer.
    - custodian_source_urls: All URLs cited to confirm the custodian information (only those explicitly present in the answer; can include the SEC filing and/or the sponsor’s official website).
    - us_launch_year: The launch year as stated (should be "2025" for eligibility if the answer states it).
    - market_location: The market/country (e.g., "United States") as stated in the answer.
    - asset_type: The asset description as stated (e.g., "spot XRP ETF", "spot XRP trust", etc.).
    - sponsor_domain: The fund sponsor’s official website root domain if provided by the answer (e.g., "blackrock.com", "fidelity.com"); else null.
    - sponsor_site_urls: Any sponsor official website URLs cited in the answer for this ETF.

    Additionally, if the answer compares other eligible ETFs and their fees, extract them into the "competitors" array:
    - For each competitor, return:
      - name
      - ticker
      - standard_fee (non-waived, exactly as written in the answer; if waived-only values are present, return null)
      - fee_source_urls (all exactly as cited in the answer for that competitor)

    IMPORTANT RULES:
    - Only extract information explicitly present in the answer. Do not invent any URLs or fields.
    - For all URL fields, extract only valid URLs explicitly shown in the answer. If a URL is missing protocol, prepend "http://".
    - For fees, prefer the standard/gross/base “management” or “sponsor” fee, not a temporary net-of-waiver figure.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_lowest_fee_determination(
    evaluator: Evaluator,
    parent_node,
    selection: ETFSelection,
) -> None:
    """
    Build and verify the 'Lowest_Fee_Determination' subtree.
    """
    node = evaluator.add_parallel(
        id="Lowest_Fee_Determination",
        desc="Correctly determine which eligible ETF has the lowest standard (non-waived) management fee",
        parent=parent_node,
        critical=True,
    )

    # 1) Eligibility_Check (leaf)
    eligibility_leaf = evaluator.add_leaf(
        id="Eligibility_Check",
        desc="The selected fund is a spot XRP ETF launched in the United States in 2025",
        parent=node,
        critical=True,
    )
    name = selection.selected_name or "the selected ETF"
    ticker = selection.selected_ticker or ""
    claim_eligibility = (
        f"The selected fund '{name}' {f'({ticker})' if ticker else ''} is a spot XRP exchange-traded fund "
        f"launched in the United States in 2025."
    )
    # We rely on answer context primarily; the separate 'Source_Verifiability' ensures allowed sources exist.
    await evaluator.verify(
        claim=claim_eligibility,
        node=eligibility_leaf,
        sources=None,
        additional_instruction=(
            "Judge based on the answer text whether the selected fund is explicitly presented as: "
            "1) a spot (physically-backed) XRP ETF (not futures/synthetic/ETN), "
            "2) launched in 2025, and "
            "3) launched/listed in the United States. "
            "If any of these are missing or contradicted, mark Incorrect."
        ),
    )

    # 2) Standard_Fee_Stated_NonWaived (leaf)
    fee_leaf = evaluator.add_leaf(
        id="Standard_Fee_Stated_NonWaived",
        desc="Provide the ETF’s standard management fee (explicitly non-waived / excluding any temporary fee waivers) with a supporting citation",
        parent=node,
        critical=True,
    )
    fee_text = selection.standard_fee or ""
    fee_sources = _dedup_urls(_safe_urls(selection.fee_source_urls))
    claim_fee = (
        f"The ETF's standard (non-waived) management fee for '{name}' "
        f"{f'({ticker})' if ticker else ''} is {fee_text}."
    )
    await evaluator.verify(
        claim=claim_fee,
        node=fee_leaf,
        sources=fee_sources if fee_sources else None,
        additional_instruction=(
            "Verify the fee is the standard/gross/base management or sponsor fee and NOT reduced by temporary fee waivers. "
            "If the citation only states a net-of-waiver fee (temporary discount) without the standard fee, mark Incorrect. "
            "Prefer SEC EDGAR (S-1) and/or the sponsor’s official site."
        ),
    )

    # 3) Lowest_Fee_Justification (leaf)
    lowest_leaf = evaluator.add_leaf(
        id="Lowest_Fee_Justification",
        desc="Provide verifiable justification (via SEC filings and/or fund sponsor official sites) that no other eligible U.S. spot XRP ETF launched in 2025 has a lower standard (non-waived) management fee",
        parent=node,
        critical=True,
    )
    # Build a summary of competitors from the extracted data (if any)
    competitors_summary_parts: List[str] = []
    for c in selection.competitors:
        if not (c and (c.name or c.ticker)):
            continue
        parts = []
        if c.name:
            parts.append(c.name)
        if c.ticker:
            parts.append(f"({c.ticker})")
        base = " ".join(parts).strip()
        if c.standard_fee:
            base += f": {c.standard_fee}"
        if base:
            competitors_summary_parts.append(base)
    competitors_summary = "; ".join(competitors_summary_parts) if competitors_summary_parts else "no competitors listed"

    claim_lowest = (
        f"Based on the answer's comparisons, no other eligible U.S. spot XRP ETF launched in 2025 has a lower "
        f"standard (non-waived) management fee than {fee_text} for '{name}' {f'({ticker})' if ticker else ''}. "
        f"Competitors and their standard fees as provided: {competitors_summary}. "
        f"Ties (equal fees) are not lower."
    )
    await evaluator.verify(
        claim=claim_lowest,
        node=lowest_leaf,
        sources=None,
        additional_instruction=(
            "Judge strictly from the answer's provided comparisons and citations (not your own knowledge). "
            "Only consider U.S.-launched 2025 spot XRP ETFs and standard (non-waived) fees. "
            "If the answer offers no explicit comparative justification or cites irrelevant sources, mark Incorrect."
        ),
    )


async def verify_required_output(
    evaluator: Evaluator,
    parent_node,
    selection: ETFSelection,
) -> None:
    """
    Build and verify the 'Required_Output_For_Selected_ETF' subtree.
    """
    node = evaluator.add_parallel(
        id="Required_Output_For_Selected_ETF",
        desc="Provide the required identifying, filing, and custodian information for the selected ETF",
        parent=parent_node,
        critical=True,
    )

    name = selection.selected_name or ""
    ticker = selection.selected_ticker or ""
    s1_url = selection.s1_url or ""
    custodian = selection.custodian_name or ""

    # ETF_Official_Name
    leaf_name = evaluator.add_leaf(
        id="ETF_Official_Name",
        desc="Provide the ETF’s official name",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The official name of the ETF is '{name}'.",
        node=leaf_name,
        sources=s1_url if s1_url else None,
        additional_instruction=(
            "Verify the exact or reasonably equivalent official product name on the SEC S-1 (or S-1/A). "
            "Allow minor formatting variations (e.g., punctuation, capitalization). If no verifiable evidence, mark Incorrect."
        ),
    )

    # ETF_Ticker
    leaf_ticker = evaluator.add_leaf(
        id="ETF_Ticker",
        desc="Provide the ETF’s ticker symbol",
        parent=node,
        critical=True,
    )
    # Prefer S-1; if not available, include fee sources as backups
    ticker_sources = _dedup_urls(([s1_url] if s1_url else []) + _safe_urls(selection.fee_source_urls))
    await evaluator.verify(
        claim=f"The ETF’s ticker symbol is '{ticker}'.",
        node=leaf_ticker,
        sources=ticker_sources if ticker_sources else None,
        additional_instruction=(
            "Confirm the ticker symbol from the SEC filing and/or the sponsor’s official site. "
            "If the cited page does not clearly show the ticker, mark Incorrect."
        ),
    )

    # SEC_S1_Direct_URL
    leaf_s1 = evaluator.add_leaf(
        id="SEC_S1_Direct_URL",
        desc="Provide a direct URL to the ETF’s Form S-1 registration statement filed with the SEC (EDGAR)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"This URL is a direct SEC EDGAR page for the Form S-1 (or S-1/A) registration statement for '{name}' "
            f"{f'({ticker})' if ticker else ''}."
        ),
        node=leaf_s1,
        sources=s1_url if s1_url else None,
        additional_instruction=(
            "Verify that the URL points to sec.gov (EDGAR) and the page indicates Form S-1 or S-1/A for this product. "
            "If the URL is missing, not on sec.gov, or not a direct filing page, mark Incorrect."
        ),
    )

    # Custodian_From_S1
    leaf_custodian = evaluator.add_leaf(
        id="Custodian_From_S1",
        desc="Identify the qualified custodian responsible for holding the ETF’s XRP assets, as specified in the Form S-1",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The SEC Form S-1 (or S-1/A) specifies '{custodian}' as the qualified custodian responsible for holding "
            f"the ETF’s XRP assets."
        ),
        node=leaf_custodian,
        sources=s1_url if s1_url else None,
        additional_instruction=(
            "Check that the filing text explicitly names this custodian role for XRP assets. "
            "If the role is a different service (e.g., administrator, transfer agent) or unclear, mark Incorrect."
        ),
    )

    # Custodian_Reference_URL
    leaf_custodian_ref = evaluator.add_leaf(
        id="Custodian_Reference_URL",
        desc="Provide a reference URL that confirms the custodian information (SEC filing and/or fund sponsor official website)",
        parent=node,
        critical=True,
    )
    custodian_ref_urls = _dedup_urls(_safe_urls(selection.custodian_source_urls))
    await evaluator.verify(
        claim=f"The provided reference URL(s) confirm that the custodian is '{custodian}'.",
        node=leaf_custodian_ref,
        sources=custodian_ref_urls if custodian_ref_urls else None,
        additional_instruction=(
            "At least one provided URL should explicitly confirm the custodian name for this ETF. "
            "Prefer SEC EDGAR S-1 and/or the sponsor’s official website. If none of the URLs confirm it, mark Incorrect."
        ),
    )


async def verify_source_verifiability(
    evaluator: Evaluator,
    parent_node,
    selection: ETFSelection,
) -> None:
    """
    Build and verify the 'Source_Verifiability' subtree: sources must be limited to SEC EDGAR and/or sponsor official site.
    """
    node = evaluator.add_parallel(
        id="Source_Verifiability",
        desc="All key claims are supported by verifiable sources limited to official SEC filings (EDGAR) and/or the fund sponsor’s official website",
        parent=parent_node,
        critical=True,
    )

    sponsor_domain = (selection.sponsor_domain or "").strip()
    fee_sources = _dedup_urls(_safe_urls(selection.fee_source_urls))
    cust_sources = _dedup_urls(_safe_urls(selection.custodian_source_urls))

    # Fee_Source_Is_Allowed
    fee_allowed_leaf = evaluator.add_leaf(
        id="Fee_Source_Is_Allowed",
        desc="The citation supporting the standard (non-waived) management fee is from SEC EDGAR and/or the sponsor’s official website",
        parent=node,
        critical=True,
    )
    # We use a simple verification: the LLM will look at the URLs in the claim and judge allowedness.
    fee_urls_inline = ", ".join(fee_sources) if fee_sources else "none"
    await evaluator.verify(
        claim=(
            "The fee citation URLs are exclusively from SEC EDGAR (.sec.gov) and/or the sponsor’s official website "
            f"(sponsor domain stated as '{sponsor_domain}' if provided). URLs: {fee_urls_inline}"
        ),
        node=fee_allowed_leaf,
        sources=None,
        additional_instruction=(
            "Allowed sources are limited to: (a) SEC EDGAR pages on sec.gov, and/or (b) the fund sponsor’s official website "
            "(recognizable brand-domain; do not accept media/news/aggregators). If any fee citation falls outside these, mark Incorrect."
        ),
    )

    # Custodian_Source_Is_Allowed
    cust_allowed_leaf = evaluator.add_leaf(
        id="Custodian_Source_Is_Allowed",
        desc="The citation supporting the custodian information is from SEC EDGAR and/or the sponsor’s official website",
        parent=node,
        critical=True,
    )
    cust_urls_inline = ", ".join(cust_sources) if cust_sources else "none"
    await evaluator.verify(
        claim=(
            "The custodian citation URLs are exclusively from SEC EDGAR (.sec.gov) and/or the sponsor’s official website "
            f"(sponsor domain stated as '{sponsor_domain}' if provided). URLs: {cust_urls_inline}"
        ),
        node=cust_allowed_leaf,
        sources=None,
        additional_instruction=(
            "Allowed sources are limited to: (a) SEC EDGAR pages on sec.gov, and/or (b) the fund sponsor’s official website. "
            "If any custodian citation falls outside these, mark Incorrect."
        ),
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
    Evaluate an answer for the 2025 XRP ETF with lowest standard fee task.
    """
    # Initialize evaluator with a sequential root (as per rubric)
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

    # Extract selection and comparison data
    selection: ETFSelection = await evaluator.extract(
        prompt=prompt_extract_selection(),
        template_class=ETFSelection,
        extraction_name="selected_etf_and_comparisons",
    )

    # Build and verify subtrees according to rubric
    await verify_lowest_fee_determination(evaluator, root, selection)
    await verify_required_output(evaluator, root, selection)
    await verify_source_verifiability(evaluator, root, selection)

    # Return the evaluation summary
    return evaluator.get_summary()