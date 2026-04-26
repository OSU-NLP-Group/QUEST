import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.verification_tree import AggregationStrategy as VT_AggregationStrategy  # for type reference if needed

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "reit_ca_90pct"
TASK_DESCRIPTION = """
Identify a publicly traded Real Estate Investment Trust (REIT) that is headquartered in California and meets the IRS requirement of distributing at least 90% of its taxable income to shareholders. Provide the company name, stock ticker symbol, headquarters location, and confirmation of its dividend distribution requirement compliance.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class REITInfo(BaseModel):
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    exchange: Optional[str] = None
    headquarters: Optional[str] = None
    compliance_confirmation_text: Optional[str] = None
    hq_urls: List[str] = Field(default_factory=list)
    listing_urls: List[str] = Field(default_factory=list)
    compliance_urls: List[str] = Field(default_factory=list)


class REITExtraction(BaseModel):
    reit: Optional[REITInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_reit_info() -> str:
    return """
    Extract the details of the single REIT used to answer the question. If multiple REITs are mentioned, pick the first one actually used to answer the task.
    Return a JSON object with a single field 'reit' containing:
      - company_name: The specific REIT company name mentioned in the answer.
      - ticker: The stock ticker symbol provided in the answer (letters only; do not include exchange prefix).
      - exchange: The stock exchange mentioned (e.g., "NYSE", "Nasdaq", "NYSE American", "Nasdaq Global Select").
      - headquarters: The headquarters location string as presented in the answer (e.g., "San Diego, California" or "San Diego, CA").
      - compliance_confirmation_text: The exact sentence/phrase from the answer that confirms the 90% taxable income distribution requirement compliance (or the requirement itself for this company). If not present, set to null.
      - hq_urls: A list of all URLs in the answer that support or state the headquarters location.
      - listing_urls: A list of all URLs in the answer that support that the company is publicly traded on NYSE or Nasdaq and show the ticker.
      - compliance_urls: A list of all URLs in the answer that support the statement that this company (as a REIT) meets or is required to meet the IRS 90% taxable income distribution requirement.
    
    Rules:
    - Only extract URLs explicitly present in the answer (including markdown links). Do not invent or infer URLs.
    - If a field is not present in the answer, set it to null (for strings) or an empty list (for URL lists).
    - If the answer mentions an exchange variant (e.g., "Nasdaq Global Select", "NYSE American"), keep it as-is in 'exchange'.
    - Preserve the headquarters string exactly as shown in the answer.
    """.strip()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: List[str]) -> List[str]:
    """Merge and deduplicate URL lists while preserving order."""
    seen = set()
    merged = []
    for lst in lists:
        for url in lst:
            if url and (url not in seen):
                seen.add(url)
                merged.append(url)
    return merged


def _normalize_exchange(exchange: Optional[str]) -> Optional[str]:
    if not exchange:
        return None
    up = exchange.strip().upper()
    if "NASDAQ" in up:
        return "Nasdaq"
    if "NYSE" in up:
        return "NYSE"
    return exchange.strip()


# --------------------------------------------------------------------------- #
# Tree construction and verifications                                         #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, extracted: REITExtraction) -> None:
    """
    Build the verification tree according to the rubric and perform checks.
    """
    # Create the rubric root under evaluator.root because evaluator root is always non-critical
    rubric_root = evaluator.add_parallel(
        id="REIT_Identification_and_Details",
        desc="Answer identifies a publicly traded REIT headquartered in California that satisfies the IRS 90% taxable income distribution requirement, and provides the requested identifying details.",
        parent=evaluator.root,
        critical=True
    )

    reit = extracted.reit or REITInfo()
    name = (reit.company_name or "").strip()
    ticker = (reit.ticker or "").strip()
    exchange_raw = (reit.exchange or "").strip()
    exchange_norm = _normalize_exchange(exchange_raw)
    headquarters = (reit.headquarters or "").strip()
    compliance_text = (reit.compliance_confirmation_text or "").strip()

    urls_all = _merge_sources(reit.hq_urls, reit.listing_urls, reit.compliance_urls)

    # 1) Named_Specific_REIT_Company (critical parent) -> split into two leaves: name provided + is REIT
    node_company = evaluator.add_parallel(
        id="Named_Specific_REIT_Company",
        desc="Response provides the company name of a specific Real Estate Investment Trust (REIT).",
        parent=rubric_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(name),
        id="Company_Name_Provided",
        desc="Company name is provided.",
        parent=node_company,
        critical=True
    )

    leaf_is_reit = evaluator.add_leaf(
        id="Company_Is_REIT",
        desc="The named company is a Real Estate Investment Trust (REIT).",
        parent=node_company,
        critical=True
    )
    claim_is_reit = f"The company named '{name}' is a Real Estate Investment Trust (REIT)."
    await evaluator.verify(
        claim=claim_is_reit,
        node=leaf_is_reit,
        sources=urls_all,
        additional_instruction="Accept if the provided page(s) explicitly describe the company as a REIT. Variants like 'real estate investment trust' or 'REIT' are acceptable."
    )

    # 2) Headquarters_Location_Provided_and_In_California (critical parent) -> provided + in California
    node_hq = evaluator.add_parallel(
        id="Headquarters_Location_Provided_and_In_California",
        desc="Response provides the REIT's headquarters location, and that location is in California.",
        parent=rubric_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(headquarters),
        id="Headquarters_Provided",
        desc="Headquarters location is provided.",
        parent=node_hq,
        critical=True
    )

    leaf_hq_ca = evaluator.add_leaf(
        id="HQ_In_California_Supported",
        desc="Headquarters is in California.",
        parent=node_hq,
        critical=True
    )
    claim_hq = f"The headquarters of {name or 'the company'} is in California. The answer states: '{headquarters}'."
    await evaluator.verify(
        claim=claim_hq,
        node=leaf_hq_ca,
        sources=reit.hq_urls if reit.hq_urls else urls_all,
        additional_instruction="Confirm that the company's headquarters is located within the state of California. Treat 'CA' as equivalent to 'California'."
    )

    # 3) Publicly_Traded_on_NYSE_or_Nasdaq_with_Ticker (critical parent) -> ticker provided + listed on NYSE/Nasdaq with ticker
    node_listing = evaluator.add_parallel(
        id="Publicly_Traded_on_NYSE_or_Nasdaq_with_Ticker",
        desc="Response includes an identifiable stock ticker symbol and indicates the REIT is listed on NYSE or Nasdaq.",
        parent=rubric_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(ticker),
        id="Ticker_Provided",
        desc="Ticker symbol is provided.",
        parent=node_listing,
        critical=True
    )

    leaf_listing = evaluator.add_leaf(
        id="Listed_on_NYSE_or_Nasdaq_with_Ticker_Supported",
        desc="Company is listed on NYSE or Nasdaq with the provided ticker.",
        parent=node_listing,
        critical=True
    )
    if exchange_norm in ("NYSE", "Nasdaq"):
        listing_claim = f"{name or 'The company'} is listed on {exchange_norm} with ticker symbol '{ticker}'."
    else:
        listing_claim = f"{name or 'The company'} is listed on NYSE or Nasdaq with ticker symbol '{ticker}'."
    await evaluator.verify(
        claim=listing_claim,
        node=leaf_listing,
        sources=reit.listing_urls if reit.listing_urls else urls_all,
        additional_instruction="Accept 'NYSE', 'NYSE American', or 'NYSE Arca' as NYSE-family, and 'Nasdaq', 'Nasdaq Global Select/Market/Capital' as Nasdaq-family. The page should show the company and the ticker."
    )

    # 4) Confirms_90pct_Distribution_Requirement_Compliance (critical parent) -> text provided + supported
    node_compliance = evaluator.add_parallel(
        id="Confirms_90pct_Distribution_Requirement_Compliance",
        desc="Response confirms the REIT meets the IRS requirement to distribute at least 90% of taxable income to shareholders annually (to maintain REIT status).",
        parent=rubric_root,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(compliance_text),
        id="Compliance_Text_Provided",
        desc="A statement confirming 90% distribution requirement compliance is provided.",
        parent=node_compliance,
        critical=True
    )

    leaf_compliance = evaluator.add_leaf(
        id="Compliance_Supported",
        desc="90% taxable income distribution requirement compliance is supported by sources.",
        parent=node_compliance,
        critical=True
    )
    compliance_claim = f"As a REIT, {name or 'the company'} meets the IRS requirement to distribute at least 90% of its taxable income to shareholders annually."
    await evaluator.verify(
        claim=compliance_claim,
        node=leaf_compliance,
        sources=reit.compliance_urls if reit.compliance_urls else urls_all,
        additional_instruction="Pass if the source explicitly states that the company, as a REIT, must distribute or distributes at least 90% of its taxable income to shareholders (e.g., in 10-K, investor relations, or other authoritative materials). A general statement about REITs is acceptable only if the page clearly refers to this specific company as a REIT on the same page."
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
    Evaluate an answer for the California-headquartered REIT with 90% distribution requirement task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_reit_info(),
        template_class=REITExtraction,
        extraction_name="reit_extraction"
    )

    await build_and_verify(evaluator, extracted)

    return evaluator.get_summary()