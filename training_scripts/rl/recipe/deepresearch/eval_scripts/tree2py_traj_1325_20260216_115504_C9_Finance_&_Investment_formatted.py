import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sec_funds_sb164_xrp_formpf_eval"
TASK_DESCRIPTION = """A US-based investment advisory firm, headquartered in San Francisco, California, currently manages three private equity funds with the following structure:

Current Portfolio:
- Fund A: 3(c)(1) structure with 85 beneficial owners (all accredited investors), $95 million AUM
- Fund B: 3(c)(7) structure with 150 investors (all qualified purchasers), $180 million AUM
- Fund C: 3(c)(1) structure with 60 beneficial owners (all accredited investors), $70 million AUM

In March 2025, the firm made its first investment in a Los Angeles-based technology startup, whose founding team members completed their demographic survey in April 2025.

Proposed Changes (Target Implementation: March 2026):
1. Add 20 new investors to Fund A (all meet accredited investor requirements; only 12 meet qualified purchaser requirements)
2. Launch Fund D as a new 3(c)(7) structure targeting $185 million from 30 investors (all qualified purchasers)
3. Allocate $27 million from Fund B to XRP ETF investments, selecting the fund with the lowest annual management fee

Required Analysis:

Provide comprehensive answers addressing:

1. SEC Registration: Determine the firm's current total regulatory assets under management (RAUM), current registration status, projected post-change RAUM, and projected registration requirement. Consider the $100M-$110M buffer zone and $150M private fund adviser exemption threshold.

2. Fund Structure Compliance: Evaluate whether Fund A's proposed expansion complies with 3(c)(1) requirements (100 investor limit, accredited investor minimum). Assess if Fund D's structure as a 3(c)(7) fund is viable with 30 investors and specify the minimum investor qualification requirement.

3. California Reporting Obligations: Determine if the firm has California nexus under SB 164, identify the DFPI registration deadline, and specify the 2026 annual demographic report filing deadline.

4. XRP ETF Selection: Calculate the dollar amount for Fund B's XRP ETF allocation and identify which XRP ETF (among Franklin Templeton XRPZ, Bitwise, or 21Shares) offers the lowest annual management fee. Provide the specific ETF ticker and fee percentage.

5. Form PF Compliance: Determine if the firm must file Form PF based on its SEC registration status and private fund AUM. If required, specify the filing deadline for fiscal year 2025 (fiscal year ends December 31, 2025).

Provide all answers with specific numerical values, dates, fund names/tickers, and regulatory threshold citations.
"""

# Ground truths derived from the scenario in the task description
FUND_A_CURRENT_OWNERS = 85
FUND_A_ADDED_OWNERS = 20
FUND_A_POST_OWNERS = FUND_A_CURRENT_OWNERS + FUND_A_ADDED_OWNERS  # 105
CURRENT_RAUM_MILLIONS = 95 + 180 + 70  # 345
PROJECTED_RAUM_MILLIONS = CURRENT_RAUM_MILLIONS + 185  # Adds Fund D target => 530

# Form PF 120-day deadline for FY ending 2025-12-31
FORM_PF_EXPECTED_DEADLINE = (datetime(2025, 12, 31)).replace().strftime("%Y-%m-%d")  # anchor
# Expected date is April 30, 2026 (120 calendar days after Dec 31, 2025)
FORM_PF_EXPECTED_DEADLINE_YMD = (2026, 4, 30)


# --------------------------------------------------------------------------- #
# Utility parsing helpers                                                     #
# --------------------------------------------------------------------------- #
def _normalize_str(s: Optional[str]) -> str:
    return (s or "").strip()


def parse_number_from_string(s: Optional[str]) -> Optional[float]:
    """Extract first numeric (int/float) from a string."""
    if not s:
        return None
    match = re.search(r'[-+]?\d[\d,]*(?:\.\d+)?', s)
    if not match:
        return None
    num_str = match.group(0).replace(",", "")
    try:
        return float(num_str)
    except Exception:
        return None


def parse_money_to_millions(s: Optional[str]) -> Optional[float]:
    """
    Parse a money string (e.g., "$345M", "345 million", "345,000,000") into millions float.
    """
    if not s:
        return None
    text = s.lower().replace("$", "").replace("usd", "").replace(",", " ").strip()
    num = parse_number_from_string(text)
    if num is None:
        return None

    # Detect unit
    if "billion" in text or "bn" in text:
        return num * 1000.0
    # 'mm' sometimes used for million
    if "million" in text or re.search(r'\bmm\b', text) or re.search(r'(?<![a-zA-Z0-9])m(?![a-zA-Z0-9])', text):
        return num
    # If raw number looks like full dollars (>= 1e6), convert to millions
    if num >= 1_000_000:
        return num / 1_000_000.0
    # If looks like thousand (K) amounts (rare here), handle if indicated
    if "k" in text:
        return num / 1000.0
    # Fallback: if small number without unit, assume it's in millions (interpretation of typical reporting)
    return num


def parse_investor_count(s: Optional[str]) -> Optional[int]:
    """Parse an integer count from a string."""
    if not s:
        return None
    m = re.search(r'\d{1,6}', s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def contains_keywords(s: Optional[str], keywords: List[str]) -> bool:
    if not s:
        return False
    low = s.lower()
    return all(kw.lower() in low for kw in keywords)


def parse_date_to_tuple(s: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """
    Parse a date string into (YYYY, M, D).
    Supports formats like:
    - April 30, 2026
    - 30 April 2026
    - 2026-04-30
    - 04/30/2026
    - 2026/04/30
    """
    if not s:
        return None
    text = s.strip()

    # ISO-like: YYYY-MM-DD or YYYY/MM/DD
    m = re.match(r'^\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s*$', text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    # MM/DD/YYYY
    m = re.match(r'^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$', text)
    if m:
        return int(m.group(3)), int(m.group(1)), int(m.group(2))

    # "Month D, YYYY" or "D Month YYYY"
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    # Month D, YYYY
    m = re.match(r'^\s*([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\s*$', text)
    if m and m.group(1).lower() in months:
        return int(m.group(3)), months[m.group(1).lower()], int(m.group(2))
    # D Month YYYY
    m = re.match(r'^\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\s*$', text)
    if m and m.group(2).lower() in months:
        return int(m.group(3)), months[m.group(2).lower()], int(m.group(1))

    # Fallback: try to extract components heuristically
    nums = re.findall(r'\d+', text)
    if len(nums) >= 3:
        # Heuristic ordering: try YYYY, M, D if the first is 4-digit year
        if len(nums[0]) == 4:
            return int(nums[0]), int(nums[1]), int(nums[2])
        # Else try M, D, YYYY or D, M, YYYY
        if len(nums[2]) == 4:
            mth = int(nums[0])
            day = int(nums[1])
            yr = int(nums[2])
            if 1 <= mth <= 12 and 1 <= day <= 31:
                return yr, mth, day
            # reverse if misordered
            if 1 <= int(nums[1]) <= 12 and 1 <= int(nums[0]) <= 31:
                return yr, int(nums[1]), int(nums[0])
    return None


def equals_expected_date(extracted: Optional[str], expected_tuple: Tuple[int, int, int]) -> bool:
    dt = parse_date_to_tuple(extracted)
    return dt == expected_tuple


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SECRegistrationExtraction(BaseModel):
    current_raum: Optional[str] = None
    current_registration_status: Optional[str] = None
    projected_raum: Optional[str] = None
    projected_registration_status: Optional[str] = None
    buffer_zone_discussion: Optional[str] = None
    private_fund_exemption_discussion: Optional[str] = None
    sec_threshold_citations: List[str] = Field(default_factory=list)


class FundStructureExtraction(BaseModel):
    fund_a_post_investor_count: Optional[str] = None
    fund_a_3c1_conclusion: Optional[str] = None
    fund_a_accredited_discussion: Optional[str] = None
    fund_d_investor_count: Optional[str] = None
    fund_d_3c7_viability_conclusion: Optional[str] = None
    fund_d_min_qualification_requirement: Optional[str] = None
    fund_structure_citations: List[str] = Field(default_factory=list)


class CaliforniaSB164Extraction(BaseModel):
    nexus_determination: Optional[str] = None
    dfpi_registration_deadline: Optional[str] = None
    annual_demographic_report_deadline: Optional[str] = None
    sb164_citations: List[str] = Field(default_factory=list)


class XRPEtfSelectionExtraction(BaseModel):
    allocation_amount: Optional[str] = None
    selected_etf_name: Optional[str] = None
    selected_etf_ticker: Optional[str] = None
    selected_etf_fee: Optional[str] = None
    selected_etf_sources: List[str] = Field(default_factory=list)
    competitor_etf_sources: List[str] = Field(default_factory=list)


class FormPFExtraction(BaseModel):
    must_file_form_pf: Optional[str] = None
    fy2025_deadline: Optional[str] = None
    form_pf_citations: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_sec_registration() -> str:
    return """
Extract the SEC registration analysis elements from the answer. Return the following fields:
- current_raum: The firm's current total regulatory assets under management (RAUM) as stated in the answer (e.g., "$345M" or "345 million")
- current_registration_status: The firm's current registration status conclusion (e.g., "SEC-registered", "SEC registration required", or "state-registered")
- projected_raum: The projected post-change RAUM including Fund D (e.g., "$530M" or "530 million")
- projected_registration_status: The projected registration requirement conclusion (e.g., "SEC registration required")
- buffer_zone_discussion: The text snippet where the answer explicitly addresses or discusses the $100M–$110M buffer zone
- private_fund_exemption_discussion: The text snippet where the answer addresses the <$150M private fund adviser exemption (exempt reporting adviser) and the conclusion
- sec_threshold_citations: A list of all URLs in the answer that are offered as citations for RAUM thresholds and the $150M private fund adviser exemption
If any field is not present, set it to null (or an empty array for the citations list).
"""


def prompt_extract_fund_structure() -> str:
    return """
Extract the fund structure compliance discussion. Return:
- fund_a_post_investor_count: The post-change beneficial owner count for Fund A as stated/calculated in the answer (e.g., "105")
- fund_a_3c1_conclusion: The explicit compliance/noncompliance conclusion for Fund A under 3(c)(1) investor limit
- fund_a_accredited_discussion: The text indicating whether Fund A (including the 20 added investors) meets the accredited investor requirement
- fund_d_investor_count: The planned investor count for Fund D (e.g., "30")
- fund_d_3c7_viability_conclusion: The explicit viability/compliance conclusion for Fund D under 3(c)(7)
- fund_d_min_qualification_requirement: The minimum investor qualification requirement for 3(c)(7) as stated (e.g., "all investors must be qualified purchasers")
- fund_structure_citations: A list of all URLs cited for 3(c)(1) and 3(c)(7) investor-limit and qualification requirements
Use null for missing strings, and an empty array for missing citations.
"""


def prompt_extract_california_sb164() -> str:
    return """
Extract California SB 164 obligations content. Return:
- nexus_determination: The conclusion about whether the firm has California nexus under SB 164
- dfpi_registration_deadline: The specific date (as written) for the DFPI registration deadline
- annual_demographic_report_deadline: The specific date (as written) for the 2026 annual demographic report filing deadline
- sb164_citations: All URLs cited supporting the nexus criteria and the deadlines
Use null for any missing field and an empty array for the citations.
"""


def prompt_extract_xrp_etf_selection() -> str:
    return """
Extract XRP ETF allocation and selection info. Return:
- allocation_amount: The dollar amount allocated from Fund B to XRP ETF investments (as stated in the answer)
- selected_etf_name: The name of the ETF selected as the lowest-fee option among Franklin Templeton XRPZ, Bitwise, and 21Shares
- selected_etf_ticker: The ticker of the selected ETF
- selected_etf_fee: The selected ETF's annual management fee percentage (e.g., "0.19%")
- selected_etf_sources: URLs the answer cites for the selected ETF's fee information
- competitor_etf_sources: URLs the answer cites for the other ETFs' fee information (Franklin/Bitwise/21Shares) or any comparative fee sources
If any field is missing, use null or empty arrays accordingly.
"""


def prompt_extract_form_pf() -> str:
    return """
Extract Form PF compliance elements. Return:
- must_file_form_pf: The conclusion on whether the firm must file Form PF (e.g., "yes", "no", "must file", "not required")
- fy2025_deadline: The filing deadline date for the fiscal year ending December 31, 2025 as stated (e.g., "April 30, 2026" or "2026-04-30")
- form_pf_citations: All URLs cited supporting the Form PF triggering threshold and the 120-day filing deadline rule
If any field is absent, set to null or empty array.
"""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_sec_registration_verification(evaluator: Evaluator,
                                              parent,
                                              sec: SECRegistrationExtraction) -> None:
    sec_node = evaluator.add_parallel(
        id="sec_registration",
        desc="SEC registration analysis: current RAUM and registration status; projected post-change RAUM and registration status; buffer zone and $150M exemption; citations.",
        parent=parent,
        critical=True
    )

    # current_raum check against scenario-derived ground truth
    cur_raum_m = parse_money_to_millions(sec.current_raum)
    evaluator.add_custom_node(
        result=(cur_raum_m is not None and abs(cur_raum_m - CURRENT_RAUM_MILLIONS) < 0.51),
        id="current_raum",
        desc=f"States the firm's current total RAUM as {CURRENT_RAUM_MILLIONS} million.",
        parent=sec_node,
        critical=True
    )

    # current registration status should indicate SEC (>= $110M)
    status_text_cur = _normalize_str(sec.current_registration_status).lower()
    evaluator.add_custom_node(
        result=("sec" in status_text_cur and ("register" in status_text_cur or "registered" in status_text_cur)),
        id="current_registration_status",
        desc="Determines the firm's current registration status correctly (SEC, given ≥$110M).",
        parent=sec_node,
        critical=True
    )

    # projected_raum check against scenario-derived ground truth
    proj_raum_m = parse_money_to_millions(sec.projected_raum)
    evaluator.add_custom_node(
        result=(proj_raum_m is not None and abs(proj_raum_m - PROJECTED_RAUM_MILLIONS) < 0.51),
        id="projected_raum",
        desc=f"States the projected post-change RAUM as {PROJECTED_RAUM_MILLIONS} million.",
        parent=sec_node,
        critical=True
    )

    # projected registration status should indicate SEC (≥$110M)
    status_text_proj = _normalize_str(sec.projected_registration_status).lower()
    evaluator.add_custom_node(
        result=("sec" in status_text_proj and ("register" in status_text_proj or "registered" in status_text_proj)),
        id="projected_registration_status",
        desc="Determines the projected registration requirement correctly (SEC).",
        parent=sec_node,
        critical=True
    )

    # buffer zone addressed explicitly
    buf_text = _normalize_str(sec.buffer_zone_discussion)
    buffer_ok = (("100" in buf_text and "110" in buf_text) and any(w in buf_text.lower() for w in ["buffer", "zone", "range", "band", "window"]))
    evaluator.add_custom_node(
        result=buffer_ok,
        id="buffer_zone_addressed",
        desc="Explicitly addresses the $100M–$110M buffer zone and its effect.",
        parent=sec_node,
        critical=True
    )

    # 150M private fund adviser exemption addressed
    pfa_text = _normalize_str(sec.private_fund_exemption_discussion)
    pfa_ok = ("150" in pfa_text and any(w in pfa_text.lower() for w in ["exempt", "exemption", "era", "exempt reporting adviser"]))
    evaluator.add_custom_node(
        result=pfa_ok,
        id="private_fund_adviser_exemption_150m_addressed",
        desc="Addresses the <$150M private fund adviser exemption and states the conclusion.",
        parent=sec_node,
        critical=True
    )

    # citations for thresholds
    urls = sec.sec_threshold_citations or []
    cite_node = evaluator.add_leaf(
        id="sec_threshold_citations",
        desc="Provides citations supporting RAUM thresholds and $150M private fund adviser exemption threshold.",
        parent=sec_node,
        critical=True
    )
    if not urls:
        # Fail if no citations provided
        cite_node.score = 0.0
        cite_node.status = "failed"
    else:
        claim = ("These source(s) support SEC RAUM registration thresholds, including: "
                 "permitted SEC registration for RAUM $100M–$110M, mandatory SEC registration at ≥$110M, "
                 "and the <$150M private fund adviser exemption for private fund advisers (exempt reporting adviser).")
        await evaluator.verify(
            claim=claim,
            node=cite_node,
            sources=urls,
            additional_instruction="Verify the sources explicitly discuss the $100M–$110M buffer, the ≥$110M SEC registration threshold, and the <$150M private fund adviser exemption threshold."
        )


async def build_fund_structure_verification(evaluator: Evaluator,
                                            parent,
                                            fs: FundStructureExtraction) -> None:
    fs_node = evaluator.add_parallel(
        id="fund_structure_compliance",
        desc="Fund structure compliance for Fund A (3(c)(1)) and Fund D (3(c)(7)), with citations.",
        parent=parent,
        critical=True
    )

    # Fund A 3(c)(1) investor cap compliance (100)
    post_count = parse_investor_count(fs.fund_a_post_investor_count)
    conc_a = _normalize_str(fs.fund_a_3c1_conclusion).lower()
    a_limit_ok = ((post_count == FUND_A_POST_OWNERS) or ("105" in _normalize_str(fs.fund_a_post_investor_count))) and \
                 any(word in conc_a for word in ["noncompliant", "exceed", "violat", "not compliant", "not permissible", "over 100"])
    evaluator.add_custom_node(
        result=a_limit_ok,
        id="fund_a_3c1_investor_limit",
        desc="Evaluates Fund A investor cap under 3(c)(1) (should note 105 > 100 and noncompliant).",
        parent=fs_node,
        critical=True
    )

    # Fund A accredited investor requirement
    acc_text = _normalize_str(fs.fund_a_accredited_discussion).lower()
    a_accredited_ok = ("accredited" in acc_text and any(k in acc_text for k in ["all", "100%", "meet", "satisfy"]))
    evaluator.add_custom_node(
        result=a_accredited_ok,
        id="fund_a_3c1_accredited_requirement",
        desc="Evaluates whether Fund A (including 20 additions) meets accredited investor minimum.",
        parent=fs_node,
        critical=True
    )

    # Fund D 3(c)(7) investor count viability
    d_count = parse_investor_count(fs.fund_d_investor_count)
    d_conc = _normalize_str(fs.fund_d_3c7_viability_conclusion).lower()
    d_viable_ok = (d_count is None or d_count >= 0) and any(k in d_conc for k in ["viable", "compliant", "permissible", "ok"])
    evaluator.add_custom_node(
        result=d_viable_ok,
        id="fund_d_3c7_investor_count_viability",
        desc="Assesses whether Fund D’s 30-investor plan is viable under 3(c)(7).",
        parent=fs_node,
        critical=True
    )

    # Fund D minimum qualification requirement
    d_req = _normalize_str(fs.fund_d_min_qualification_requirement).lower()
    d_req_ok = ("qualified purchaser" in d_req) and any(w in d_req for w in ["all", "must", "only"])
    evaluator.add_custom_node(
        result=d_req_ok,
        id="fund_d_3c7_minimum_qualification",
        desc="States minimum qualification for 3(c)(7) (all investors must be qualified purchasers) and applies it.",
        parent=fs_node,
        critical=True
    )

    # Citations for 3(c)(1) and 3(c)(7)
    urls = fs.fund_structure_citations or []
    cite_node = evaluator.add_leaf(
        id="fund_structure_citations",
        desc="Citations support 3(c)(1) 100-owner cap and 3(c)(7) qualified purchaser requirement.",
        parent=fs_node,
        critical=True
    )
    if not urls:
        cite_node.score = 0.0
        cite_node.status = "failed"
    else:
        claim = ("These source(s) support that: (i) 3(c)(1) funds are limited to 100 beneficial owners; "
                 "(ii) 3(c)(7) funds require all investors to be qualified purchasers (and are not constrained by a 100-investor cap).")
        await evaluator.verify(
            claim=claim,
            node=cite_node,
            sources=urls,
            additional_instruction="Confirm the cited materials clearly state the 100-investor cap for 3(c)(1) and the 'all qualified purchasers' requirement for 3(c)(7)."
        )


async def build_california_sb164_verification(evaluator: Evaluator,
                                              parent,
                                              ca: CaliforniaSB164Extraction) -> None:
    ca_node = evaluator.add_parallel(
        id="california_sb164",
        desc="California SB 164 obligations: nexus determination and deadlines with citations.",
        parent=parent,
        critical=True
    )

    # Nexus: headquartered in San Francisco, CA -> nexus likely yes.
    nexus_text = _normalize_str(ca.nexus_determination).lower()
    nexus_ok = any(k in nexus_text for k in ["yes", "has nexus", "nexus exists", "california nexus", "subject to sb 164"])
    evaluator.add_custom_node(
        result=nexus_ok,
        id="nexus_determination",
        desc="Determines whether the firm has California nexus under SB 164 and states conclusion.",
        parent=ca_node,
        critical=True
    )

    # DFPI registration deadline: verify specific date against sources (if provided)
    dfpi_leaf = evaluator.add_leaf(
        id="dfpi_registration_deadline",
        desc="States the DFPI registration deadline (specific date) and supports with sources.",
        parent=ca_node,
        critical=True
    )
    if not ca.sb164_citations:
        dfpi_leaf.score = 0.0
        dfpi_leaf.status = "failed"
    else:
        claim = f"The DFPI registration deadline for SB 164 compliance is {(_normalize_str(ca.dfpi_registration_deadline))}."
        await evaluator.verify(
            claim=claim,
            node=dfpi_leaf,
            sources=ca.sb164_citations,
            additional_instruction="Verify that the cited sources explicitly mention this DFPI registration deadline date."
        )

    # Annual demographic report deadline: verify specific date against sources
    demo_leaf = evaluator.add_leaf(
        id="annual_demographic_report_deadline",
        desc="States the 2026 annual demographic report filing deadline (specific date) and supports with sources.",
        parent=ca_node,
        critical=True
    )
    if not ca.sb164_citations:
        demo_leaf.score = 0.0
        demo_leaf.status = "failed"
    else:
        claim = f"The 2026 annual demographic report filing deadline under SB 164 is {(_normalize_str(ca.annual_demographic_report_deadline))}."
        await evaluator.verify(
            claim=claim,
            node=demo_leaf,
            sources=ca.sb164_citations,
            additional_instruction="Verify that the cited sources explicitly mention this annual demographic report filing deadline for 2026."
        )

    # Citations themselves support nexus criteria and deadlines
    sb_cite = evaluator.add_leaf(
        id="sb164_citations",
        desc="Citations support the nexus criteria and the two deadlines.",
        parent=ca_node,
        critical=True
    )
    if not ca.sb164_citations:
        sb_cite.score = 0.0
        sb_cite.status = "failed"
    else:
        claim = "These sources describe SB 164 nexus criteria and provide/confirm the DFPI registration and annual demographic reporting deadlines."
        await evaluator.verify(
            claim=claim,
            node=sb_cite,
            sources=ca.sb164_citations,
            additional_instruction="Confirm that the sources explicitly set out nexus conditions and state the deadlines."
        )


async def build_xrp_etf_verification(evaluator: Evaluator,
                                     parent,
                                     xrp: XRPEtfSelectionExtraction) -> None:
    xrp_node = evaluator.add_parallel(
        id="xrp_etf_selection",
        desc="XRP ETF allocation and selection with lowest fee, including ticker and fee.",
        parent=parent,
        critical=True
    )

    # Allocation amount should be $27 million (from scenario)
    alloc_m = parse_money_to_millions(xrp.allocation_amount)
    evaluator.add_custom_node(
        result=(alloc_m is not None and abs(alloc_m - 27.0) < 0.51),
        id="allocation_amount",
        desc="States the dollar amount allocated from Fund B to XRP ETFs as $27 million.",
        parent=xrp_node,
        critical=True
    )

    # Selected lowest-fee ETF: verify with sources; construct claim using ticker & fee
    etf_urls = (xrp.selected_etf_sources or []) + (xrp.competitor_etf_sources or [])
    selected_leaf = evaluator.add_leaf(
        id="selected_lowest_fee_etf",
        desc="Identifies the lowest-fee ETF among Franklin Templeton XRPZ, Bitwise, 21Shares; provides ticker and fee.",
        parent=xrp_node,
        critical=True
    )
    if not etf_urls:
        selected_leaf.score = 0.0
        selected_leaf.status = "failed"
    else:
        ticker = _normalize_str(xrp.selected_etf_ticker)
        fee = _normalize_str(xrp.selected_etf_fee)
        name = _normalize_str(xrp.selected_etf_name)
        claim = (f"Among Franklin Templeton XRPZ, Bitwise, and 21Shares XRP ETFs, the lowest annual management fee is {fee} "
                 f"for the ETF {name} (ticker {ticker}).")
        await evaluator.verify(
            claim=claim,
            node=selected_leaf,
            sources=etf_urls,
            additional_instruction="Use the provided ETF pages or reputable sources to confirm fee percentages and compare across the three named ETFs."
        )


async def build_form_pf_verification(evaluator: Evaluator,
                                     parent,
                                     formpf: FormPFExtraction) -> None:
    fp_node = evaluator.add_sequential(
        id="form_pf",
        desc="Form PF requirement determination and FY2025 deadline (if required), with citations.",
        parent=parent,
        critical=True
    )

    # Must file determination (based on SEC registration and private fund AUM >= $150M from scenario)
    must_text = _normalize_str(formpf.must_file_form_pf).lower()
    must_ok = any(k in must_text for k in ["yes", "must", "required"])
    evaluator.add_custom_node(
        result=must_ok,
        id="must_file_form_pf_determination",
        desc="Determines whether the firm must file Form PF and states conclusion.",
        parent=fp_node,
        critical=True
    )

    # FY2025 deadline (120 calendar days after 2025-12-31 => 2026-04-30)
    deadline_ok = equals_expected_date(formpf.fy2025_deadline, FORM_PF_EXPECTED_DEADLINE_YMD)
    evaluator.add_custom_node(
        result=deadline_ok,
        id="fy2025_deadline_if_required",
        desc="States the FY2025 Form PF filing deadline as April 30, 2026 (120 days after 2025-12-31).",
        parent=fp_node,
        critical=True
    )

    # Citations supporting triggering thresholds and 120-day rule
    urls = formpf.form_pf_citations or []
    cite_leaf = evaluator.add_leaf(
        id="form_pf_citations",
        desc="Citations support Form PF triggering threshold and the 120-day filing rule.",
        parent=fp_node,
        critical=True
    )
    if not urls:
        cite_leaf.score = 0.0
        cite_leaf.status = "failed"
    else:
        claim = ("These sources state that SEC-registered advisers to private funds meeting the private fund AUM threshold must file Form PF, "
                 "and that the annual Form PF is due within 120 calendar days of fiscal year-end.")
        await evaluator.verify(
            claim=claim,
            node=cite_leaf,
            sources=urls,
            additional_instruction="Confirm the sources mention the private fund AUM threshold (e.g., $150M) and the 120-calendar-day deadline rule."
        )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
) -> Dict[str, Any]:
    """
    Evaluate the agent's answer against the rubric using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel; critical children will gate the score
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

    # Record ground truths for transparency
    evaluator.add_ground_truth({
        "expected_current_raum_millions": CURRENT_RAUM_MILLIONS,
        "expected_projected_raum_millions": PROJECTED_RAUM_MILLIONS,
        "fund_a_post_investor_count_expected": FUND_A_POST_OWNERS,
        "form_pf_expected_deadline": "2026-04-30",
        "notes": "Ground truths derive from scenario details in the task; citations must still be provided for regulatory thresholds."
    }, gt_type="ground_truth")

    # Extract sections in parallel
    sec_task = evaluator.extract(
        prompt=prompt_extract_sec_registration(),
        template_class=SECRegistrationExtraction,
        extraction_name="sec_registration_extraction"
    )
    fs_task = evaluator.extract(
        prompt=prompt_extract_fund_structure(),
        template_class=FundStructureExtraction,
        extraction_name="fund_structure_extraction"
    )
    ca_task = evaluator.extract(
        prompt=prompt_extract_california_sb164(),
        template_class=CaliforniaSB164Extraction,
        extraction_name="california_sb164_extraction"
    )
    xrp_task = evaluator.extract(
        prompt=prompt_extract_xrp_etf_selection(),
        template_class=XRPEtfSelectionExtraction,
        extraction_name="xrp_etf_selection_extraction"
    )
    fp_task = evaluator.extract(
        prompt=prompt_extract_form_pf(),
        template_class=FormPFExtraction,
        extraction_name="form_pf_extraction"
    )

    sec_ex, fs_ex, ca_ex, xrp_ex, fp_ex = await asyncio.gather(sec_task, fs_task, ca_task, xrp_task, fp_task)

    # Build verification subtrees (all children of root are critical to emulate overall critical rubric)
    # SEC Registration
    await build_sec_registration_verification(evaluator, root, sec_ex)

    # Fund Structure Compliance
    await build_fund_structure_verification(evaluator, root, fs_ex)

    # California SB 164
    await build_california_sb164_verification(evaluator, root, ca_ex)

    # XRP ETF Selection
    await build_xrp_etf_verification(evaluator, root, xrp_ex)

    # Form PF Compliance
    await build_form_pf_verification(evaluator, root, fp_ex)

    # Return summary with verification tree and scores
    return evaluator.get_summary()