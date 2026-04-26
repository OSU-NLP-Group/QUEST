import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broker_ohio_ira"
TASK_DESCRIPTION = """Identify three major brokerage firms that meet all of the following requirements for retirement investors in Ohio:

1. Branch Locations: The firm must have physical branch offices in all three of these Ohio cities: Columbus, Cleveland, and Cincinnati.

2. IRA Account Types: The firm must offer both Traditional IRA and Roth IRA accounts.

3. Account Minimum: The firm must allow investors to open an IRA account with a $0 minimum initial deposit (no minimum investment required).

4. Account Fees: The firm must charge no annual account maintenance fee for IRA accounts.

5. Investment Options: The firm must provide access to index mutual funds or index ETFs with expense ratios at or below 0.10%.

For each of the three firms you identify, provide:
- The firm's name
- The physical address of one branch location in each city (Columbus, Cleveland, Cincinnati)
- Confirmation that both Traditional and Roth IRAs are offered
- Confirmation of the $0 account minimum policy
- Confirmation of no annual IRA account fees
- At least one example of an index fund available through the firm with an expense ratio ≤ 0.10%
- Direct URLs to the firm's official website pages that verify these features
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BranchInfo(BaseModel):
    address: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class IRAInfo(BaseModel):
    confirmation_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PolicyInfo(BaseModel):
    statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class FundExample(BaseModel):
    name: Optional[str] = None
    expense_ratio: Optional[str] = None  # keep as string to be flexible (e.g., "0.03%")
    is_index_text: Optional[str] = None  # e.g., "index fund" or "ETF tracking S&P 500"
    urls: List[str] = Field(default_factory=list)


class FirmItem(BaseModel):
    name: Optional[str] = None

    columbus: Optional[BranchInfo] = None
    cleveland: Optional[BranchInfo] = None
    cincinnati: Optional[BranchInfo] = None

    ira_accounts: Optional[IRAInfo] = None
    zero_minimum: Optional[PolicyInfo] = None
    no_annual_fee: Optional[PolicyInfo] = None

    index_fund: Optional[FundExample] = None

    official_urls: List[str] = Field(default_factory=list)  # any extra direct official URLs mentioned


class FirmsExtraction(BaseModel):
    firms: List[FirmItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_firms() -> str:
    return """
Extract up to three brokerage firms from the answer that claim to meet ALL of the following requirements for retirement investors in Ohio:
- Branch locations in Columbus, Cleveland, Cincinnati (with specific street addresses).
- Offers both Traditional IRA and Roth IRA accounts.
- $0 minimum initial deposit to open an IRA (no minimum investment required to open).
- No annual account maintenance fee for IRA accounts.
- Access to at least one index mutual fund or index ETF with expense ratio ≤ 0.10%.

For each firm, return a JSON object with the fields below. Extract ONLY what is explicitly stated in the answer text and the URLs that are explicitly provided. For URLs, extract direct official pages from the firm's own website whenever present in the answer; do NOT invent or infer any URLs.

Structure per firm:
- name: Firm name as written in the answer.
- columbus:
  - address: The specific street address provided for a Columbus, OH branch.
  - urls: Direct official firm URLs in the answer that show/confirm this Columbus branch/address (e.g., a branch page or locator result for Columbus).
- cleveland:
  - address: The specific street address provided for a Cleveland, OH branch.
  - urls: Direct official firm URLs in the answer that show/confirm this Cleveland branch/address.
- cincinnati:
  - address: The specific street address provided for a Cincinnati, OH branch.
  - urls: Direct official firm URLs in the answer that show/confirm this Cincinnati branch/address.
- ira_accounts:
  - confirmation_text: Text snippet from the answer confirming both Traditional IRA and Roth IRA are offered.
  - urls: Direct official firm URLs in the answer that confirm both Traditional and Roth IRAs are offered (e.g., IRA product page).
- zero_minimum:
  - statement: Text snippet from the answer confirming $0 minimum initial deposit to open the IRA account.
  - urls: Direct official firm URLs in the answer that confirm $0 minimum to open (focus on account opening minimum; ignore trading/fund purchase minimums).
- no_annual_fee:
  - statement: Text snippet from the answer confirming no annual IRA account maintenance fee.
  - urls: Direct official firm URLs in the answer that confirm $0 annual IRA account fee (fee schedule, pricing page, or product page).
- index_fund:
  - name: The specific index mutual fund or index ETF name/ticker from the answer.
  - expense_ratio: The expense ratio string as written (e.g., "0.03%").
  - is_index_text: Text from the answer that indicates this is an index fund/ETF.
  - urls: Direct official URLs in the answer that substantiate either the fund’s index nature and/or its expense ratio (e.g., the firm’s fund detail page or the fund provider’s official page).
- official_urls: Any other direct official firm URLs cited in the answer relevant to verification (optional).

Rules:
- Extract AT MOST three firms in an array 'firms'.
- If a field is missing in the answer, set it to null (for a string) or [] (for URLs).
- Do not invent any data or URLs. Exactly reflect what the answer provides.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Filter obviously invalid URLs
    out = []
    for u in urls:
        if isinstance(u, str) and (u.startswith("http://") or u.startswith("https://")):
            out.append(u.strip())
    return out


def _domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def _slug(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


COMMON_STOP_WORDS = {
    "inc", "llc", "l.l.c", "co", "corp", "corporation", "company", "group", "bank",
    "banking", "advisors", "adviser", "advisory", "financial", "investments",
    "investment", "brokerage", "securities", "wealth", "management", "services",
    "plc", "lp", "ltd", "limited", "partners", "partner"
}


def _brand_tokens(name: Optional[str]) -> List[str]:
    if not name:
        return []
    # split on non-alphanumeric
    toks = re.split(r"[^a-zA-Z0-9]+", name.lower())
    toks = [t for t in toks if t and t not in COMMON_STOP_WORDS and len(t) >= 3]
    # add a compact slug form (e.g., "etrade" for "E*TRADE")
    compact = _slug(name)
    if compact and compact not in toks:
        toks.append(compact)
    return toks


def is_official_url(url: str, firm_name: Optional[str]) -> bool:
    """
    Heuristic: URL is considered 'official' if the firm's brand tokens (including compact slug)
    appear in the domain. This is a best-effort check to encourage official pages.
    """
    if not url or not firm_name:
        return False
    d = _domain(url)
    if not d:
        return False
    tokens = _brand_tokens(firm_name)
    return any(t in d for t in tokens)


def collect_all_urls(firm: FirmItem) -> List[str]:
    urls: List[str] = []
    for br in [firm.columbus, firm.cleveland, firm.cincinnati]:
        if br:
            urls.extend(_safe_list(br.urls))
    if firm.ira_accounts:
        urls.extend(_safe_list(firm.ira_accounts.urls))
    if firm.zero_minimum:
        urls.extend(_safe_list(firm.zero_minimum.urls))
    if firm.no_annual_fee:
        urls.extend(_safe_list(firm.no_annual_fee.urls))
    if firm.index_fund:
        urls.extend(_safe_list(firm.index_fund.urls))
    urls.extend(_safe_list(firm.official_urls))
    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification for a single firm                                              #
# --------------------------------------------------------------------------- #
async def verify_firm(
    evaluator: Evaluator,
    parent_node,
    firm: FirmItem,
    idx: int,
) -> None:
    ordinal = idx + 1
    firm_desc = f"{ordinal}{'st' if ordinal == 1 else ('nd' if ordinal == 2 else ('rd' if ordinal == 3 else 'th'))} brokerage firm meets all requirements"

    firm_node = evaluator.add_parallel(
        id=f"firm_{ordinal}",
        desc=firm_desc,
        parent=parent_node,
        critical=False  # Each firm is non-critical relative to the root; allows partial credit across firms
    )

    # 1) Firm name provided (existence check)
    evaluator.add_custom_node(
        result=bool(firm and firm.name and firm.name.strip()),
        id=f"firm_{ordinal}_name_provided",
        desc="The firm's name is provided",
        parent=firm_node,
        critical=True
    )

    # Helper to add branch verification
    async def add_branch_check(city_key: str, city_label: str, branch: Optional[BranchInfo]):
        node_id = f"firm_{ordinal}_{city_key}_branch_address"
        node_desc = f"Firm has a physical branch in {city_label}, Ohio, and the specific address is provided"
        # If missing address or no URLs -> immediate fail (no source-grounded evidence)
        if not branch or not (branch.address and branch.address.strip()) or not _safe_list(branch.urls):
            evaluator.add_custom_node(
                result=False,
                id=node_id,
                desc=node_desc,
                parent=firm_node,
                critical=True
            )
            return
        # Otherwise, verify with URLs
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=node_desc,
            parent=firm_node,
            critical=True
        )
        claim = f"The firm '{firm.name}' has a physical branch office in {city_label}, Ohio at the address '{branch.address}'."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=_safe_list(branch.urls),
            additional_instruction=(
                "Verify that the cited official firm page(s) show a branch located in the specified city with the provided street address. "
                "Minor formatting differences (e.g., Street vs St.) are acceptable if clearly the same address."
            ),
        )

    # 2) Branch locations (Columbus, Cleveland, Cincinnati)
    await add_branch_check("columbus", "Columbus", firm.columbus)
    await add_branch_check("cleveland", "Cleveland", firm.cleveland)
    await add_branch_check("cincinnati", "Cincinnati", firm.cincinnati)

    # 3) IRA account types: both Traditional and Roth
    ira_node_id = f"firm_{ordinal}_ira_accounts"
    ira_node_desc = "Firm offers both Traditional IRA and Roth IRA accounts, and this is confirmed in the solution"
    ira_urls = _safe_list(firm.ira_accounts.urls) if firm and firm.ira_accounts else []
    if not ira_urls:
        evaluator.add_custom_node(
            result=False,
            id=ira_node_id,
            desc=ira_node_desc,
            parent=firm_node,
            critical=True
        )
    else:
        ira_leaf = evaluator.add_leaf(
            id=ira_node_id,
            desc=ira_node_desc,
            parent=firm_node,
            critical=True
        )
        claim = f"The firm '{firm.name}' offers both Traditional IRA and Roth IRA accounts."
        await evaluator.verify(
            claim=claim,
            node=ira_leaf,
            sources=ira_urls,
            additional_instruction=(
                "Confirm that BOTH Traditional IRA and Roth IRA are offered. "
                "Synonyms (e.g., Individual Retirement Account) are acceptable if clearly indicating Traditional and Roth variants."
            ),
        )

    # 4) $0 minimum to open IRA
    zero_min_node_id = f"firm_{ordinal}_zero_minimum"
    zero_min_node_desc = "Firm allows opening an IRA with $0 minimum initial deposit, and this is confirmed in the solution"
    zm_urls = _safe_list(firm.zero_minimum.urls) if firm and firm.zero_minimum else []
    if not zm_urls:
        evaluator.add_custom_node(
            result=False,
            id=zero_min_node_id,
            desc=zero_min_node_desc,
            parent=firm_node,
            critical=True
        )
    else:
        zm_leaf = evaluator.add_leaf(
            id=zero_min_node_id,
            desc=zero_min_node_desc,
            parent=firm_node,
            critical=True
        )
        claim = f"Opening an IRA account at '{firm.name}' requires a $0 minimum initial deposit (no minimum to open the account)."
        await evaluator.verify(
            claim=claim,
            node=zm_leaf,
            sources=zm_urls,
            additional_instruction=(
                "Focus on the minimum to OPEN the IRA account (account-opening deposit). "
                "Do NOT confuse with minimums to place trades or fund-specific purchase minimums."
            ),
        )

    # 5) No annual IRA account maintenance fee
    no_fee_node_id = f"firm_{ordinal}_no_account_fee"
    no_fee_node_desc = "Firm charges no annual account maintenance fee for IRA accounts, and this is confirmed in the solution"
    nf_urls = _safe_list(firm.no_annual_fee.urls) if firm and firm.no_annual_fee else []
    if not nf_urls:
        evaluator.add_custom_node(
            result=False,
            id=no_fee_node_id,
            desc=no_fee_node_desc,
            parent=firm_node,
            critical=True
        )
    else:
        nf_leaf = evaluator.add_leaf(
            id=no_fee_node_id,
            desc=no_fee_node_desc,
            parent=firm_node,
            critical=True
        )
        claim = f"There is no annual account maintenance fee for IRA accounts at '{firm.name}'."
        await evaluator.verify(
            claim=claim,
            node=nf_leaf,
            sources=nf_urls,
            additional_instruction=(
                "Confirm that the annual maintenance/custodial fee for IRAs is $0. "
                "Waived fees count as $0 if explicitly stated as no annual IRA maintenance fee."
            ),
        )

    # 6) Index fund example with expense ratio <= 0.10%
    fund_node_id = f"firm_{ordinal}_index_fund_example"
    fund_node_desc = "At least one specific index fund with expense ratio at or below 0.10% is provided, including the fund name and expense ratio"
    fund_ok = bool(
        firm and firm.index_fund and
        firm.index_fund.name and firm.index_fund.name.strip() and
        firm.index_fund.expense_ratio and firm.index_fund.expense_ratio.strip() and
        _safe_list(firm.index_fund.urls)
    )
    if not fund_ok:
        evaluator.add_custom_node(
            result=False,
            id=fund_node_id,
            desc=fund_node_desc,
            parent=firm_node,
            critical=True
        )
    else:
        fund_leaf = evaluator.add_leaf(
            id=fund_node_id,
            desc=fund_node_desc,
            parent=firm_node,
            critical=True
        )
        er = firm.index_fund.expense_ratio
        claim = (
            f"The example fund '{firm.index_fund.name}' is an index mutual fund or index ETF and its expense ratio is {er}, "
            f"which is at or below 0.10%."
        )
        await evaluator.verify(
            claim=claim,
            node=fund_leaf,
            sources=_safe_list(firm.index_fund.urls),
            additional_instruction=(
                "Confirm both that it is an index fund/ETF and that its expense ratio is ≤ 0.10%. "
                "Allow small formatting differences (e.g., 0.03% vs 0.030%)."
            ),
        )

    # 7) URL Reference: at least one direct official firm page is provided overall
    all_urls = collect_all_urls(firm)
    official_found = any(is_official_url(u, firm.name) for u in all_urls)
    evaluator.add_custom_node(
        result=official_found,
        id=f"firm_{ordinal}_url_reference",
        desc="Direct URLs to the firm's official website pages are provided to verify the features",
        parent=firm_node,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the brokerage firms in Ohio IRA requirements task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # 3 firms evaluated independently
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

    # Extract structured info
    extraction: FirmsExtraction = await evaluator.extract(
        prompt=prompt_extract_firms(),
        template_class=FirmsExtraction,
        extraction_name="firms_extraction",
    )

    # Ensure exactly three slots (pad with empty firms if needed; truncate if more)
    firms: List[FirmItem] = list(extraction.firms[:3])
    while len(firms) < 3:
        firms.append(FirmItem())

    # Build verification subtrees for each firm
    for i in range(3):
        await verify_firm(evaluator, root, firms[i], i)

    # Return standardized summary
    return evaluator.get_summary()