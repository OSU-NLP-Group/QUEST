import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "nasdaq_top_tech_3t_2024"
TASK_DESCRIPTION = (
    "As of November 2024, identify the United States-based technology company that has the highest market capitalization "
    "among all companies listed on the NASDAQ stock exchange with a market capitalization exceeding $3 trillion. For this company, "
    "provide the following information:\n\n"
    "1. The official company name\n"
    "2. The NASDAQ ticker symbol\n"
    "3. The current market capitalization value (in USD)\n"
    "4. A reference URL that confirms this information\n\n"
    "Additionally, verify that:\n"
    "- The company is listed on the NASDAQ exchange\n"
    "- The company is classified in the technology sector\n"
    "- The company is domiciled in the United States\n"
    "- The company's market capitalization meets the minimum S&P 500 eligibility threshold of at least $8.2 billion"
)


class CompanySelectionExtraction(BaseModel):
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    market_cap_usd: Optional[str] = None
    market_cap_as_of_date: Optional[str] = None
    exchange_mentioned: Optional[str] = None
    sector_mentioned: Optional[str] = None
    domicile_country: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


def prompt_extract_company_info() -> str:
    return (
        "Extract the selected company's information exactly as presented in the answer. "
        "Return a JSON object with the following fields:\n"
        "1. company_name: The official company name as stated.\n"
        "2. ticker: The NASDAQ ticker symbol (e.g., AAPL, MSFT). If exchange prefix like 'NASDAQ:' appears, extract just the symbol in 'ticker'.\n"
        "3. market_cap_usd: The stated market capitalization value in USD, exactly as written in the answer "
        "(e.g., '$3.1T', '3.10 trillion USD', '3,100,000,000,000 USD').\n"
        "4. market_cap_as_of_date: The explicit 'as of' date tied to the market cap or highest-market-cap comparison, "
        "preferably in a recognizable string (e.g., 'Nov 12, 2024', 'November 2024', '2024-11-12'). If absent, return null.\n"
        "5. exchange_mentioned: The exchange as stated in the answer (e.g., 'NASDAQ'). If not stated, return null.\n"
        "6. sector_mentioned: The sector as stated (e.g., 'Technology', 'Information Technology'). If not stated, return null.\n"
        "7. domicile_country: The domicile country stated (e.g., 'United States', 'USA', 'U.S.'). If not stated, return null.\n"
        "8. reference_urls: An array of all URLs explicitly provided in the answer that serve as references for any of the above claims. "
        "Include all valid URLs, including markdown links and plain URLs. If none provided, return an empty array.\n"
        "Important rules:\n"
        "- Do not invent or infer information not present in the answer.\n"
        "- Keep text values exactly as in the answer (case-preserving), but it's okay to trim leading/trailing spaces.\n"
        "- For ticker, remove any exchange prefix like 'NASDAQ:' and return only the symbol.\n"
        "- Only URLs explicitly present in the answer should be included in reference_urls."
    )


async def verify_company_selection_and_eligibility(
    evaluator: Evaluator,
    parent_node,
    info: CompanySelectionExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="company_selection_and_eligibility",
        desc="Chosen company satisfies all required selection constraints (including being the highest market-cap qualifier) for a date in Nov 2024.",
        parent=parent_node,
        critical=True,
    )

    # Helper variables for readable claims
    company_name = info.company_name or "the chosen company"
    ticker = info.ticker or "(ticker unspecified)"
    sources = info.reference_urls if info.reference_urls else None

    # Timeframe within November 2024
    timeframe_leaf = evaluator.add_leaf(
        id="timeframe_nov_2024",
        desc="Market-cap figure and the highest-market-cap comparison are explicitly tied to a date within November 2024.",
        parent=node,
        critical=True,
    )
    timeframe_claim = (
        "The market capitalization figure and the 'highest market cap' comparison are explicitly tied to a date in November 2024."
    )
    await evaluator.verify(
        claim=timeframe_claim,
        node=timeframe_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the references show an 'as of' or date label clearly within November 2024 (Nov 1–Nov 30, 2024). "
            "If no date is shown or it is outside November 2024, mark as incorrect."
        ),
    )

    # NASDAQ listing
    nasdaq_leaf = evaluator.add_leaf(
        id="nasdaq_listing",
        desc="Company is listed on the NASDAQ exchange.",
        parent=node,
        critical=True,
    )
    nasdaq_claim = f"{company_name} is listed on the NASDAQ exchange (ticker '{ticker}')."
    await evaluator.verify(
        claim=nasdaq_claim,
        node=nasdaq_leaf,
        sources=sources,
        additional_instruction=(
            "Rely on the reference page(s) to confirm that the company is listed on NASDAQ. "
            "Accept reasonable variations like 'NASDAQ Global Select Market'."
        ),
    )

    # Technology sector
    sector_leaf = evaluator.add_leaf(
        id="technology_sector",
        desc="Company is classified in the technology sector.",
        parent=node,
        critical=True,
    )
    sector_claim = f"{company_name} is classified in the Technology sector."
    await evaluator.verify(
        claim=sector_claim,
        node=sector_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm via the references that the company's sector is technology-related. "
            "Allow synonyms like 'Information Technology'."
        ),
    )

    # US domicile
    domicile_leaf = evaluator.add_leaf(
        id="us_domicile",
        desc="Company is domiciled in the United States.",
        parent=node,
        critical=True,
    )
    domicile_claim = f"{company_name} is domiciled in the United States."
    await evaluator.verify(
        claim=domicile_claim,
        node=domicile_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the company is U.S.-domiciled (e.g., incorporated or headquartered in the United States). "
            "Accept 'USA' or 'U.S.' as equivalent."
        ),
    )

    # Market cap exceeds $3T
    cap3t_leaf = evaluator.add_leaf(
        id="market_cap_exceeds_3t",
        desc="Company market capitalization exceeds $3 trillion.",
        parent=node,
        critical=True,
    )
    cap3t_claim = f"The market capitalization of {company_name} exceeds $3 trillion USD."
    await evaluator.verify(
        claim=cap3t_claim,
        node=cap3t_leaf,
        sources=sources,
        additional_instruction=(
            "Use the reference page(s) to confirm the market cap is strictly greater than $3,000,000,000,000 (USD). "
            "Allow minor rounding differences (e.g., $3.01T)."
        ),
    )

    # Highest among qualifiers
    highest_leaf = evaluator.add_leaf(
        id="highest_among_qualifiers",
        desc="Company has the highest market capitalization among NASDAQ-listed, U.S.-domiciled, technology-sector companies with market cap exceeding $3 trillion.",
        parent=node,
        critical=True,
    )
    highest_claim = (
        f"As of November 2024, among NASDAQ-listed, U.S.-domiciled technology-sector companies with market cap exceeding $3 trillion, "
        f"{company_name} has the highest market capitalization."
    )
    await evaluator.verify(
        claim=highest_claim,
        node=highest_leaf,
        sources=sources,
        additional_instruction=(
            "Prefer ranking or comparison pages from reliable sources (e.g., market data aggregators). "
            "If sources do not present a clear comparison or show another company with higher market cap, mark as incorrect."
        ),
    )

    # Meets S&P 500 minimum threshold
    sp500_leaf = evaluator.add_leaf(
        id="meets_sp500_minimum",
        desc="Company market capitalization is at least $8.2 billion (S&P 500 minimum eligibility threshold).",
        parent=node,
        critical=True,
    )
    sp500_claim = f"The market capitalization of {company_name} is at least $8.2 billion USD."
    await evaluator.verify(
        claim=sp500_claim,
        node=sp500_leaf,
        sources=sources,
        additional_instruction=(
            "This is a trivial check given trillion-scale market caps; confirm the references report a market cap far above $8.2B."
        ),
    )


async def verify_required_output_fields_with_citations(
    evaluator: Evaluator,
    parent_node,
    info: CompanySelectionExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="required_output_fields_with_citations",
        desc="Provide all requested output fields and reference URL(s) that corroborate them.",
        parent=parent_node,
        critical=True,
    )

    # Official company name present and corroborated in answer
    name_leaf = evaluator.add_leaf(
        id="official_company_name",
        desc="Provide the official company name.",
        parent=node,
        critical=True,
    )
    name_claim = (
        f"The answer provides the official company name: '{info.company_name or ''}'."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        additional_instruction=(
            "Judge 'Correct' only if a non-empty official company name is present in the answer. "
            "You do not need to verify correctness against sources here; this check is for presence in the answer."
        ),
    )

    # NASDAQ ticker symbol present in answer
    ticker_leaf = evaluator.add_leaf(
        id="nasdaq_ticker_symbol",
        desc="Provide the NASDAQ ticker symbol.",
        parent=node,
        critical=True,
    )
    ticker_claim = (
        f"The answer provides a NASDAQ ticker symbol: '{info.ticker or ''}'."
    )
    await evaluator.verify(
        claim=ticker_claim,
        node=ticker_leaf,
        additional_instruction=(
            "Judge 'Correct' only if a non-empty ticker symbol appears in the answer. "
            "Exchange prefix like 'NASDAQ:' is acceptable if the symbol was extracted."
        ),
    )

    # Market cap value (USD) present in answer
    mcap_value_leaf = evaluator.add_leaf(
        id="market_cap_value_usd",
        desc="Provide the market capitalization value in USD.",
        parent=node,
        critical=True,
    )
    mcap_value_claim = (
        f"The answer provides a market capitalization value in USD: '{info.market_cap_usd or ''}'."
    )
    await evaluator.verify(
        claim=mcap_value_claim,
        node=mcap_value_leaf,
        additional_instruction=(
            "Judge 'Correct' only if a non-empty USD-denominated market cap value is present in the answer. "
            "Allow formats like '$3.1T', '3.10 trillion USD', 'USD 3,100,000,000,000'."
        ),
    )

    # Reference URLs corroborate identity/ticker and market cap/date
    refs_leaf = evaluator.add_leaf(
        id="reference_urls_corroborate_claims",
        desc="Provide reference URL(s) from reliable sources that corroborate the provided company identity/ticker and the market-cap figure/date used.",
        parent=node,
        critical=True,
    )
    refs_claim = (
        f"The provided reference URLs corroborate {info.company_name or 'the company'}'s identity, NASDAQ ticker '{info.ticker or ''}', "
        f"and the stated market capitalization '{info.market_cap_usd or ''}' as of a date in November 2024."
    )
    await evaluator.verify(
        claim=refs_claim,
        node=refs_leaf,
        sources=info.reference_urls if info.reference_urls else None,
        additional_instruction=(
            "Pass if at least one reference URL clearly supports the company identity/name and ticker symbol, "
            "and also provides the market cap figure tied to a date in November 2024. "
            "If URLs are missing or do not corroborate these claims, mark as incorrect."
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_company_info(),
        template_class=CompanySelectionExtraction,
        extraction_name="company_selection_extraction",
    )

    company_selection_node = evaluator.add_parallel(
        id="company_selection_and_eligibility_root",
        desc="Chosen company satisfies constraints and eligibility (Nov 2024).",
        parent=root,
        critical=True,
    )
    await verify_company_selection_and_eligibility(evaluator, company_selection_node, extracted)

    required_fields_node = evaluator.add_parallel(
        id="required_output_fields_with_citations_root",
        desc="All required output fields provided with corroborating references.",
        parent=root,
        critical=True,
    )
    await verify_required_output_fields_with_citations(evaluator, required_fields_node, extracted)

    return evaluator.get_summary()