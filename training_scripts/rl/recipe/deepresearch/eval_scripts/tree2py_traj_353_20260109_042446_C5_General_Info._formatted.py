import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_tech_companies_4_constraints"
TASK_DESCRIPTION = """
Identify four publicly traded technology companies in the United States that meet the following criteria. For each company, provide the complete headquarters address (including street address, city, state, and ZIP code), the current CEO's full name, the stock ticker symbol, the founding year, and at least one reference URL from an official corporate website or reputable financial source to verify the information.

The four companies must satisfy these specific requirements:

1. Company 1: A technology company that was founded in the 1970s (between 1970 and 1979, inclusive) and is currently headquartered in Texas.

2. Company 2: A technology company that was founded in the 1990s (between 1990 and 1999, inclusive) and is currently headquartered in California.

3. Company 3: A technology company that was founded in the 1990s (between 1990 and 1999, inclusive) and is currently headquartered in Washington state.

4. Company 4: A technology company that was founded in the 1970s (between 1970 and 1979, inclusive) and is currently headquartered in California.

All four companies must be publicly traded on major US stock exchanges (NASDAQ or NYSE).
"""


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class CompanyAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    full: Optional[str] = None


class CompanyExtract(BaseModel):
    name: Optional[str] = None
    headquarters: Optional[CompanyAddress] = None
    ceo: Optional[str] = None
    ticker: Optional[str] = None
    founding_year: Optional[str] = None
    # Optional extracted label if present in answer (not required but helpful context)
    sector_or_industry: Optional[str] = None
    # All URLs explicitly mentioned in the answer for this company
    reference_urls: List[str] = Field(default_factory=list)


class CompaniesExtraction(BaseModel):
    companies: List[CompanyExtract] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_companies() -> str:
    return """
Extract exactly four companies from the answer, mapping them in order to the four categories required by the task:
- Company 1: Founded in the 1970s (1970–1979), headquartered in Texas.
- Company 2: Founded in the 1990s (1990–1999), headquartered in California.
- Company 3: Founded in the 1990s (1990–1999), headquartered in Washington.
- Company 4: Founded in the 1970s (1970–1979), headquartered in California.

For each company, extract:
- name: The company's full name as given in the answer (string).
- headquarters: The complete physical headquarters address, broken into:
  - street: Street address (e.g., "1 Microsoft Way").
  - city: City (e.g., "Redmond").
  - state: US state (full name or two-letter abbreviation, e.g., "WA" or "Washington").
  - zip: Zip or ZIP+4 (e.g., "98052" or "98052-6399").
  - full: The full address string as presented in the answer (if available).
- ceo: The current CEO's full name as provided in the answer (string).
- ticker: The stock ticker symbol (string).
- founding_year: The founding year as provided in the answer (string; four digits if available).
- sector_or_industry: Any sector or industry label present in the answer (string if present; otherwise null).
- reference_urls: An array of all URLs explicitly provided in the answer that are suitable to verify the information. These can include an official corporate website (e.g., investor relations, contact, leadership pages) or reputable financial sources (e.g., NASDAQ, NYSE, SEC filings, Yahoo Finance, Bloomberg). Extract only actual URLs that appear in the answer.

Rules:
- Do not invent any information. If a field is missing in the answer, set it to null (or empty array for reference_urls).
- Ensure the address is split into the required fields (street, city, state, zip) when possible. If the answer only provides a single-line address, put that into 'full' and parse fields when possible.
- Return a JSON object with the following shape:
{
  "companies": [
    {
      "name": null or string,
      "headquarters": {
        "street": null or string,
        "city": null or string,
        "state": null or string,
        "zip": null or string,
        "full": null or string
      },
      "ceo": null or string,
      "ticker": null or string,
      "founding_year": null or string,
      "sector_or_industry": null or string,
      "reference_urls": [ ... zero or more URLs ... ]
    },
    ... (total 4 items, corresponding to categories #1 to #4 as listed above)
  ]
}
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def requirement_for_index(idx: int) -> Dict[str, Any]:
    """
    Return the required state and founding-year window for company index 0..3.
    """
    mapping = {
        0: {"state_name": "Texas", "start_year": 1970, "end_year": 1979},
        1: {"state_name": "California", "start_year": 1990, "end_year": 1999},
        2: {"state_name": "Washington", "start_year": 1990, "end_year": 1999},
        3: {"state_name": "California", "start_year": 1970, "end_year": 1979},
    }
    return mapping[idx]


def company_desc(idx: int) -> str:
    req = requirement_for_index(idx)
    period = f"{req['start_year']}-{req['end_year']}"
    return (
        f"A publicly traded US technology company founded in the {period} "
        f"and headquartered in {req['state_name']}"
    )


def safe_len(xs: Optional[List[str]]) -> int:
    return 0 if not xs else len(xs)


def non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification logic per company                                              #
# --------------------------------------------------------------------------- #
async def verify_company(
    evaluator: Evaluator,
    parent_node,
    company: CompanyExtract,
    idx: int,
) -> None:
    req = requirement_for_index(idx)
    state_required = req["state_name"]
    start_year = req["start_year"]
    end_year = req["end_year"]

    # Top-level company node (non-critical to allow partial credit across companies)
    company_node = evaluator.add_parallel(
        id=f"company_{idx+1}",
        desc=company_desc(idx),
        parent=parent_node,
        critical=False,
    )

    # Reference URL presence (critical). This gates verification that depends on URLs.
    has_refs = safe_len(company.reference_urls) > 0
    evaluator.add_custom_node(
        result=has_refs,
        id=f"company_{idx+1}_reference",
        desc="At least one reference URL from an official corporate website or reputable financial source is provided to verify the company information",
        parent=company_node,
        critical=True,
    )

    # Technology sector check (critical)
    tech_leaf = evaluator.add_leaf(
        id=f"company_{idx+1}_technology_sector",
        desc="The company operates in the technology sector",
        parent=company_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The company operates in the technology sector (accept synonyms such as Information Technology, Software, Semiconductors, Technology Hardware & Equipment, IT Services, or Internet technology).",
        node=tech_leaf,
        sources=company.reference_urls if has_refs else None,
        additional_instruction=(
            "Check the provided page(s) for sector/industry labels or explicit statements that the company is a technology company. "
            "Accept reasonable synonyms and related industry classifications commonly grouped under the Technology sector."
        ),
    )

    # Founding year range check (critical) – logical check using the provided founding year
    fy_leaf = evaluator.add_leaf(
        id=f"company_{idx+1}_founding_year",
        desc=f"The company was founded between {start_year} and {end_year} (inclusive)",
        parent=company_node,
        critical=True,
    )
    year_txt = company.founding_year if company.founding_year else "unknown"
    await evaluator.verify(
        claim=f"The founding year '{year_txt}' is between {start_year} and {end_year}, inclusive.",
        node=fy_leaf,
        additional_instruction=(
            "If the year is approximate (e.g., 'c. 1975') or embedded in text, still judge whether it clearly falls within the inclusive range. "
            "If no four-digit year is provided, consider this incorrect."
        ),
    )

    # Headquarters checks
    hq_node = evaluator.add_parallel(
        id=f"company_{idx+1}_headquarters_main",
        desc="Headquarters requirement breakdown",
        parent=company_node,
        critical=True,  # The original rubric treats HQ as critical
    )

    # 1) Complete address presence (critical custom check)
    hq = company.headquarters or CompanyAddress()
    complete_address = (
        non_empty(hq.street) and non_empty(hq.city) and non_empty(hq.state) and non_empty(hq.zip)
    )
    evaluator.add_custom_node(
        result=complete_address,
        id=f"company_{idx+1}_address_complete",
        desc="The company's headquarters address in the answer includes street, city, state, and ZIP code",
        parent=hq_node,
        critical=True,
    )

    # 2) State match supported by sources (critical)
    hq_state_leaf = evaluator.add_leaf(
        id=f"company_{idx+1}_headquarters_state_match",
        desc=f"The company's headquarters is located in {state_required}",
        parent=hq_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The company's headquarters is located in the state of {state_required}.",
        node=hq_state_leaf,
        sources=company.reference_urls if has_refs else None,
        additional_instruction=(
            "Verify via 'Headquarters' or official address on the page. Contact pages, investor relations, or company overview pages are acceptable. "
            "Accept either the state name or postal abbreviation matching the required state."
        ),
    )

    # Publicly traded checks
    pub_node = evaluator.add_parallel(
        id=f"company_{idx+1}_publicly_traded_main",
        desc="Public listing requirement breakdown",
        parent=company_node,
        critical=True,  # The original rubric treats this as critical
    )

    # 1) Ticker provided (critical custom check)
    ticker_provided = non_empty(company.ticker)
    evaluator.add_custom_node(
        result=ticker_provided,
        id=f"company_{idx+1}_ticker_provided",
        desc="A valid stock ticker symbol is provided",
        parent=pub_node,
        critical=True,
    )

    # 2) Exchange verification via sources (critical)
    exchange_leaf = evaluator.add_leaf(
        id=f"company_{idx+1}_publicly_traded",
        desc="The company is publicly traded on NASDAQ or NYSE with the provided ticker symbol",
        parent=pub_node,
        critical=True,
    )
    ticker_upper = (company.ticker or "").strip().upper()
    await evaluator.verify(
        claim=(
            f"The company is listed on a major US stock exchange (NASDAQ or NYSE) under the ticker '{ticker_upper}'. "
            "Accept exchange labels such as NASDAQ, NasdaqGS, Nasdaq Global Select Market, NYSE, or NYSE American."
        ),
        node=exchange_leaf,
        sources=company.reference_urls if has_refs else None,
        additional_instruction=(
            "Confirm that the referenced page shows both the ticker and that it is associated with NASDAQ or NYSE. "
            "If the reference shows a different exchange or lacks exchange information, consider this incorrect."
        ),
    )

    # CEO checks
    ceo_node = evaluator.add_parallel(
        id=f"company_{idx+1}_ceo_main",
        desc="CEO requirement breakdown",
        parent=company_node,
        critical=True,  # The original rubric treats CEO as critical
    )

    # 1) CEO provided (critical custom check)
    ceo_provided = non_empty(company.ceo) and (" " in (company.ceo or ""))
    evaluator.add_custom_node(
        result=ceo_provided,
        id=f"company_{idx+1}_ceo_provided",
        desc="The current CEO's full name is provided",
        parent=ceo_node,
        critical=True,
    )

    # 2) CEO supported by sources (critical)
    ceo_leaf = evaluator.add_leaf(
        id=f"company_{idx+1}_ceo",
        desc="The current CEO's full name is correct per the references",
        parent=ceo_node,
        critical=True,
    )
    company_name_for_claim = company.name or "the company"
    ceo_name = company.ceo or "unknown"
    await evaluator.verify(
        claim=f"The current CEO of {company_name_for_claim} is {ceo_name}.",
        node=ceo_leaf,
        sources=company.reference_urls if has_refs else None,
        additional_instruction=(
            "Check official leadership pages, investor relations pages, press releases, or reputable financial sources. "
            "Accept titles like 'Chief Executive Officer' or 'CEO & President' as valid confirmation of the CEO role."
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
    Evaluate an answer for the 4 constrained US technology companies task.
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
        default_model=model,
    )

    # Extract structured company info
    extracted = await evaluator.extract(
        prompt=prompt_extract_companies(),
        template_class=CompaniesExtraction,
        extraction_name="companies_extraction",
    )

    # Normalize to exactly 4 companies (pad with empty if fewer)
    companies = (extracted.companies or [])[:4]
    while len(companies) < 4:
        companies.append(CompanyExtract())

    # Build verification tree for each company (parallel at root)
    # Each company node uses its own critical children
    for idx in range(4):
        await verify_company(evaluator, root, companies[idx], idx)

    # Optional: record ground truth-like expectations to help interpretation
    evaluator.add_ground_truth({
        "requirements": [
            {"company_index": 1, "founded_in": "1970-1979", "headquarters_state": "Texas", "exchange": "NASDAQ/NYSE"},
            {"company_index": 2, "founded_in": "1990-1999", "headquarters_state": "California", "exchange": "NASDAQ/NYSE"},
            {"company_index": 3, "founded_in": "1990-1999", "headquarters_state": "Washington", "exchange": "NASDAQ/NYSE"},
            {"company_index": 4, "founded_in": "1970-1979", "headquarters_state": "California", "exchange": "NASDAQ/NYSE"},
        ],
        "notes": "All four must be US technology companies, publicly traded on NASDAQ or NYSE, with complete HQ address, CEO, ticker, founding year, and at least one reference URL."
    }, gt_type="expected_constraints")

    return evaluator.get_summary()