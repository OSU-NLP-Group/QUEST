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
TASK_ID = "ria_selection_2025"
TASK_DESCRIPTION = """
I am seeking to hire a financial advisor and want to evaluate fee-only registered investment advisory (RIA) firms that meet high professional standards. Please identify three SEC-registered investment advisory (RIA) firms that meet ALL of the following criteria:

1. The firm must be registered with the SEC as a Registered Investment Advisor (RIA)
2. The firm must manage at least $500 million in assets under management (AUM)
3. The firm must appear on at least one major industry ranking for 2025: Barron's Top RIA Firms, Forbes Top RIA Firms, or CNBC Financial Advisor 100
4. The firm must operate exclusively on a fee-only basis (compensated only by client fees, not commissions)
5. The firm must be headquartered in one of these states: California, Texas, Florida, or New York
6. At least one of the firm's key advisors or principals must hold a CFP (Certified Financial Planner) or CFA (Chartered Financial Analyst) designation
7. The firm must explicitly state on its website or in public materials that it operates as a fiduciary

For each of the three firms, provide:
- Firm name and headquarters location (city and state)
- Total assets under management (AUM)
- Which industry ranking list(s) the firm appears on for 2025
- Confirmation of fee-only compensation structure
- Name and professional certification (CFP or CFA) of at least one key advisor
- Link to the firm's official website
- Link to the firm's Form ADV filing on the SEC's Investment Adviser Public Disclosure website
"""

ALLOWED_STATES = ["California", "Texas", "Florida", "New York"]
ALLOWED_STATE_ABBREVS = ["CA", "TX", "FL", "NY"]
ALLOWED_RANKING_LISTS = ["Barron's", "Forbes", "CNBC"]
REQUIRED_RANKING_YEAR = "2025"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RankingEntry(BaseModel):
    list_name: Optional[str] = None  # Expected one of "Barron's", "Forbes", "CNBC"
    year: Optional[str] = None       # Prefer "2025"
    url: Optional[str] = None


class AdvisorCert(BaseModel):
    name: Optional[str] = None
    certification: Optional[str] = None  # "CFP" or "CFA"
    source_url: Optional[str] = None


class FirmItem(BaseModel):
    name: Optional[str] = None
    headquarters_city: Optional[str] = None
    headquarters_state: Optional[str] = None
    aum: Optional[str] = None
    website_url: Optional[str] = None
    form_adv_url: Optional[str] = None
    sec_crd_number: Optional[str] = None
    sec_file_number: Optional[str] = None
    fee_only_statement: Optional[str] = None
    fee_only_source_url: Optional[str] = None
    fiduciary_statement: Optional[str] = None
    fiduciary_source_url: Optional[str] = None
    services_description: Optional[str] = None
    services_source_url: Optional[str] = None
    rankings: List[RankingEntry] = Field(default_factory=list)
    advisor: Optional[AdvisorCert] = None
    additional_source_urls: List[str] = Field(default_factory=list)


class FirmsExtraction(BaseModel):
    firms: List[FirmItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_firms() -> str:
    return """
    Extract up to three SEC‑registered RIA firms described in the answer, along with the specific fields and URLs explicitly provided in the answer text.

    For each firm, return a JSON object with the following fields:

    1. name: Firm legal name as stated
    2. headquarters_city: City of the firm's headquarters (or principal office)
    3. headquarters_state: State of the headquarters (prefer full state name; two-letter abbreviation allowed)
    4. aum: Assets under management as stated (extract the exact wording or number, e.g., "$2.3B", ">$500 million")
    5. website_url: Official firm website URL (full URL)
    6. form_adv_url: Direct link to the firm's Form ADV page on the SEC Investment Adviser Public Disclosure site (adviserinfo.sec.gov)
    7. sec_crd_number: CRD number if explicitly provided in the answer (otherwise null)
    8. sec_file_number: SEC file number if explicitly provided in the answer (otherwise null)
    9. fee_only_statement: The text snippet in the answer that confirms fee‑only compensation (extract verbatim)
    10. fee_only_source_url: URL where the fee‑only claim is stated (e.g., firm site page or credible source)
    11. fiduciary_statement: The text snippet asserting fiduciary status (extract verbatim)
    12. fiduciary_source_url: URL where fiduciary status is stated
    13. services_description: A short snippet describing comprehensive wealth management or financial planning services (extract verbatim)
    14. services_source_url: URL where services description appears
    15. advisor: Object with:
        - name: Name of at least one key advisor/principal
        - certification: The professional designation "CFP" or "CFA" (verbatim)
        - source_url: URL where the designation is shown (e.g., team bio page)
    16. rankings: Array of ranking entries. For each entry include:
        - list_name: One of "Barron's", "Forbes", or "CNBC" ONLY
        - year: The year of the ranking (must be "2025" if provided in the answer)
        - url: Direct URL to the ranking page that lists the firm
        Only include entries the answer explicitly mentions; if the year is not 2025, still include it but set year accordingly. Do not invent URLs.
    17. additional_source_urls: Any other supporting URLs explicitly included in the answer (e.g., press releases, third‑party profiles)

    RULES:
    - Extract ONLY information explicitly present in the answer. Do not invent or infer anything.
    - For any missing field, use null (or empty array for lists).
    - For URLs, extract full valid URLs. If a protocol is missing, prepend http://
    - If the answer includes more than three firms, keep ONLY the first three mentioned, in order.

    Return a JSON object with a single field "firms" which is an array of up to three firm objects.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    return bool(url and url.strip() and url.strip().lower().startswith(("http://", "https://")))


def collect_valid_urls(*urls: Optional[str]) -> List[str]:
    out = []
    for u in urls:
        if is_valid_url(u):
            out.append(u.strip())
    return out


def unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def ranking_urls(firm: FirmItem) -> List[str]:
    urls = []
    for r in (firm.rankings or []):
        if is_valid_url(r.url):
            urls.append(r.url.strip())
    return unique_urls(urls)


def all_firm_related_urls(firm: FirmItem) -> List[str]:
    base = collect_valid_urls(firm.website_url, firm.form_adv_url)
    base.extend(ranking_urls(firm))
    base.extend([u for u in firm.additional_source_urls if is_valid_url(u)])
    # advisor/source urls
    if firm.advisor and is_valid_url(firm.advisor.source_url):
        base.append(firm.advisor.source_url.strip())
    if is_valid_url(firm.fee_only_source_url):
        base.append(firm.fee_only_source_url.strip())
    if is_valid_url(firm.fiduciary_source_url):
        base.append(firm.fiduciary_source_url.strip())
    if is_valid_url(firm.services_source_url):
        base.append(firm.services_source_url.strip())
    return unique_urls(base)


# --------------------------------------------------------------------------- #
# Verification per firm                                                       #
# --------------------------------------------------------------------------- #
async def verify_single_firm(
    evaluator: Evaluator,
    parent_node,
    firm: FirmItem,
    firm_index: int,
) -> None:
    """
    Build verification sub-tree for one firm and run verifications.
    The firm's node is a parallel aggregator with 10 critical leaf checks.
    """
    firm_node = evaluator.add_parallel(
        id=f"firm_{firm_index+1}",
        desc=f"{['First','Second','Third'][firm_index]} qualifying RIA firm",
        parent=parent_node,
        critical=False
    )

    # Precompute useful URL groups
    website_only = collect_valid_urls(firm.website_url)
    adv_only = collect_valid_urls(firm.form_adv_url)
    website_or_adv = unique_urls(website_only + adv_only)
    rankings_only = ranking_urls(firm)
    fee_only_urls = unique_urls(collect_valid_urls(firm.fee_only_source_url) + website_only)
    fiduciary_urls = unique_urls(collect_valid_urls(firm.fiduciary_source_url) + website_only)
    services_urls = unique_urls(collect_valid_urls(firm.services_source_url) + website_only)
    advisor_urls = unique_urls(collect_valid_urls(firm.advisor.source_url if firm.advisor else None) + website_only)

    # Prepare claims and target leaves
    verify_items: List[Dict[str, Any]] = []

    # 1) SEC registration
    sec_node = evaluator.add_leaf(
        id=f"firm_{firm_index+1}_sec_registration",
        desc="Firm is registered with the SEC as an RIA with a valid CRD number and SEC file number",
        parent=firm_node,
        critical=True,
        status="initialized",
        score=0.0
    )
    if website_or_adv:
        claim = (
            f"The firm '{firm.name or 'the firm'}' is registered with the SEC as a Registered Investment Adviser. "
            f"Its SEC IAPD page shows a CRD number and SEC file number."
        )
        verify_items.append({
            "claim": claim,
            "sources": website_or_adv if adv_only else website_or_adv,  # Prefer ADV; include website if needed
            "node": sec_node,
            "add_ins": "Prefer the SEC IAPD (adviserinfo.sec.gov) page to confirm registration and presence of CRD and SEC file numbers."
        })
    else:
        sec_node.status = "failed"
        sec_node.score = 0.0

    # 2) AUM threshold
    aum_node = evaluator.add_leaf(
        id=f"firm_{firm_index+1}_aum_threshold",
        desc="Firm manages at least $500 million in assets under management",
        parent=firm_node,
        critical=True,
        status="initialized",
        score=0.0
    )
    if website_or_adv:
        claim = (
            "The firm has assets under management (AUM) of at least $500 million. "
            "Use 'Regulatory Assets Under Management' on the SEC page or a credible figure on the official site."
        )
        verify_items.append({
            "claim": claim,
            "sources": website_or_adv,
            "node": aum_node,
            "add_ins": "Numbers may be abbreviated (e.g., $0.5B or $500,000,000). If multiple figures exist, prefer the SEC IAPD regulatory AUM."
        })
    else:
        aum_node.status = "failed"
        aum_node.score = 0.0

    # 3) Industry ranking (2025)
    rank_node = evaluator.add_leaf(
        id=f"firm_{firm_index+1}_industry_ranking",
        desc="Firm appears on at least one of the specified 2025 industry rankings (Barron's, Forbes, or CNBC FA 100)",
        parent=firm_node,
        critical=True,
        status="initialized",
        score=0.0
    )
    if rankings_only:
        claim = (
            f"The firm '{firm.name or 'the firm'}' appears on at least one 2025 ranking list among: "
            f"Barron's Top RIA, Forbes Top RIA, or CNBC Financial Advisor 100."
        )
        verify_items.append({
            "claim": claim,
            "sources": rankings_only,
            "node": rank_node,
            "add_ins": "Confirm the page lists the firm and that the ranking is for year 2025."
        })
    else:
        rank_node.status = "failed"
        rank_node.score = 0.0

    # 4) Fee-only structure
    fee_node = evaluator.add_leaf(
        id=f"firm_{firm_index+1}_fee_only_structure",
        desc="Firm operates exclusively on a fee-only basis, receiving compensation only from clients",
        parent=firm_node,
        critical=True,
        status="initialized",
        score=0.0
    )
    if fee_only_urls:
        claim = (
            "The firm operates exclusively on a fee-only basis, receiving compensation only from client fees and not commissions."
        )
        verify_items.append({
            "claim": claim,
            "sources": fee_only_urls,
            "node": fee_node,
            "add_ins": "Prefer explicit statements such as 'fee-only' or membership in fee-only organizations; look for 'no commissions' language."
        })
    else:
        fee_node.status = "failed"
        fee_node.score = 0.0

    # 5) Geographic location
    geo_node = evaluator.add_leaf(
        id=f"firm_{firm_index+1}_geographic_location",
        desc="Firm is headquartered in California, Texas, Florida, or New York",
        parent=firm_node,
        critical=True,
        status="initialized",
        score=0.0
    )
    if website_or_adv:
        claim = (
            "The firm's principal office/headquarters is in one of the following states: California, Texas, Florida, or New York."
        )
        verify_items.append({
            "claim": claim,
            "sources": website_or_adv,
            "node": geo_node,
            "add_ins": "Use the firm's contact/about page or SEC IAPD 'Principal Office and Place of Business' to confirm the HQ state is CA, TX, FL, or NY."
        })
    else:
        geo_node.status = "failed"
        geo_node.score = 0.0

    # 6) Advisor certification (CFP or CFA)
    cert_node = evaluator.add_leaf(
        id=f"firm_{firm_index+1}_advisor_certification",
        desc="At least one key advisor holds CFP or CFA designation",
        parent=firm_node,
        critical=True,
        status="initialized",
        score=0.0
    )
    if advisor_urls:
        claim = (
            "At least one key advisor or principal at the firm holds either the CFP (Certified Financial Planner) or CFA (Chartered Financial Analyst) designation."
        )
        verify_items.append({
            "claim": claim,
            "sources": advisor_urls,
            "node": cert_node,
            "add_ins": "Verify on the firm's team/bio page or other credible source that at least one advisor holds CFP or CFA."
        })
    else:
        cert_node.status = "failed"
        cert_node.score = 0.0

    # 7) Fiduciary status
    fid_node = evaluator.add_leaf(
        id=f"firm_{firm_index+1}_fiduciary_status",
        desc="Firm explicitly states it operates as a fiduciary",
        parent=firm_node,
        critical=True,
        status="initialized",
        score=0.0
    )
    if fiduciary_urls:
        claim = "The firm explicitly states that it operates as a fiduciary."
        verify_items.append({
            "claim": claim,
            "sources": fiduciary_urls,
            "node": fid_node,
            "add_ins": "Look for the word 'fiduciary' on the firm's official materials (e.g., About, Services, Disclosures)."
        })
    else:
        fid_node.status = "failed"
        fid_node.score = 0.0

    # 8) Comprehensive services
    svc_node = evaluator.add_leaf(
        id=f"firm_{firm_index+1}_comprehensive_services",
        desc="Firm provides comprehensive wealth management or financial planning services, not just investment management",
        parent=firm_node,
        critical=True,
        status="initialized",
        score=0.0
    )
    if services_urls:
        claim = (
            "The firm provides comprehensive wealth management or financial planning services (beyond pure investment management)."
        )
        verify_items.append({
            "claim": claim,
            "sources": services_urls,
            "node": svc_node,
            "add_ins": "Look for explicit mention of 'financial planning', 'comprehensive planning', 'wealth management', tax/estate planning on services pages."
        })
    else:
        svc_node.status = "failed"
        svc_node.score = 0.0

    # 9) Form ADV accessibility
    advacc_node = evaluator.add_leaf(
        id=f"firm_{firm_index+1}_form_adv_accessibility",
        desc="Firm's Form ADV is publicly accessible on the SEC IAPD website",
        parent=firm_node,
        critical=True,
        status="initialized",
        score=0.0
    )
    if adv_only:
        claim = (
            "This URL is the firm's Form ADV page on the SEC Investment Adviser Public Disclosure (IAPD) website and is publicly accessible."
        )
        verify_items.append({
            "claim": claim,
            "sources": adv_only,
            "node": advacc_node,
            "add_ins": "Confirm the page loads and is an SEC IAPD page showing ADV/firm summary."
        })
    else:
        advacc_node.status = "failed"
        advacc_node.score = 0.0

    # 10) Website verification
    web_node = evaluator.add_leaf(
        id=f"firm_{firm_index+1}_website_verification",
        desc="Firm has an official website with verifiable information and a link is provided",
        parent=firm_node,
        critical=True,
        status="initialized",
        score=0.0
    )
    if website_only:
        claim = (
            f"The URL provided is the official website of the firm '{firm.name or 'the firm'}' and it contains verifiable information about the firm."
        )
        verify_items.append({
            "claim": claim,
            "sources": website_only,
            "node": web_node,
            "add_ins": "Check that the site shows the firm name, services, or contact matching the firm's identity."
        })
    else:
        web_node.status = "failed"
        web_node.score = 0.0

    # Execute verifications in parallel for this firm
    if verify_items:
        await evaluator.batch_verify(
            claims_and_sources=[
                (item["claim"], item["sources"], item["node"], item["add_ins"])
                for item in verify_items
            ]
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
    Evaluate an answer for the SEC-registered fee-only RIA selection task (2025 constraints).
    """
    # Initialize evaluator with root parallel strategy
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

    # Add reference info
    evaluator.add_custom_info(
        info={
            "allowed_states": ALLOWED_STATES,
            "allowed_state_abbrevs": ALLOWED_STATE_ABBREVS,
            "required_ranking_year": REQUIRED_RANKING_YEAR,
            "allowed_ranking_lists": ALLOWED_RANKING_LISTS
        },
        info_type="constraints",
        info_name="evaluation_constraints"
    )

    # Extract firms from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_firms(),
        template_class=FirmsExtraction,
        extraction_name="firms_extraction"
    )

    # Normalize to exactly three firms (pad with empty stubs if needed; slice if too many)
    firms: List[FirmItem] = list(extracted.firms or [])
    if len(firms) > 3:
        firms = firms[:3]
    while len(firms) < 3:
        firms.append(FirmItem())

    # Build firm subtrees and verify each in parallel
    # We will launch verify_single_firm coroutines concurrently
    tasks = [
        verify_single_firm(evaluator, root, firms[i], i)
        for i in range(3)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Return structured summary
    return evaluator.get_summary()