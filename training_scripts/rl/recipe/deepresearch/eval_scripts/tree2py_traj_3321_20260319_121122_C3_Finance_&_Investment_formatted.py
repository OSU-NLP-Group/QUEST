import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "utility_bond_screen"
TASK_DESCRIPTION = """
Identify a corporate bond issued by a U.S. electric utility company that meets the following institutional investment criteria: 
(1) has an investment grade credit rating of at least BBB- from S&P or Fitch, or Baa3 from Moody's; 
(2) is either non-callable or has call protection of at least 10 years from the original issue date; 
(3) has a maturity date between January 1, 2032 and December 31, 2036; and 
(4) pays a fixed coupon on a semi-annual basis. 
Provide the bond's CUSIP or ISIN, the issuer's name, the exact maturity date, the credit rating, the callable status, and the coupon payment frequency, along with reference URLs documenting each of these bond characteristics.
"""

MATURITY_START = datetime(2032, 1, 1)
MATURITY_END = datetime(2036, 12, 31)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BondExtraction(BaseModel):
    # Core identifiers
    identifier: Optional[str] = None  # CUSIP or ISIN exactly as written in the answer
    identifier_type: Optional[str] = None  # "CUSIP" or "ISIN" if stated

    # Issuer
    issuer_name: Optional[str] = None
    issuer_urls: List[str] = Field(default_factory=list)  # pages evidencing US electric utility status

    # Rating
    rating: Optional[str] = None              # e.g., "BBB+", "Baa2"
    rating_agency: Optional[str] = None       # e.g., "S&P", "Moody's", "Fitch"
    rating_urls: List[str] = Field(default_factory=list)  # pages evidencing rating

    # Callable / call protection
    callable_status: Optional[str] = None     # free text, e.g., "Non-callable", "Callable NC10", "Make-whole"
    call_protection_years: Optional[str] = None  # free text number if directly stated, e.g., "10", "10+"
    issue_date: Optional[str] = None          # original issue date, if available
    first_call_date: Optional[str] = None     # first call date, if available
    callable_urls: List[str] = Field(default_factory=list)  # pages evidencing callable/call-protection

    # Maturity
    maturity_date: Optional[str] = None
    maturity_urls: List[str] = Field(default_factory=list)  # pages evidencing maturity

    # Coupon
    coupon_type: Optional[str] = None         # "Fixed", "Floating", etc.
    coupon_frequency: Optional[str] = None    # "Semi-annual", "Semiannual", "2x per year", etc.
    coupon_urls: List[str] = Field(default_factory=list)    # pages evidencing coupon structure

    # Optional general references (e.g., prospectus, factsheet)
    general_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_bond() -> str:
    return """
Extract exactly one target bond from the answer (the first one that is presented as satisfying the task). 
Return a JSON object with the following fields (return null for any missing single-value field, and [] for any URL list not present in the answer):

- identifier: The CUSIP or ISIN exactly as shown in the answer (do not invent one).
- identifier_type: "CUSIP" or "ISIN" if explicitly stated; else null.
- issuer_name: The issuer's full name as written.
- issuer_urls: Array of URL(s) that the answer cites to document the issuer's business sector (electric utility) and U.S. status.
- rating: The specific credit rating string quoted in the answer (e.g., "BBB", "BBB+", "Baa2", "A3").
- rating_agency: The agency name from which the rating is claimed (e.g., "S&P", "Moody's", "Fitch").
- rating_urls: Array of URL(s) that the answer cites to document this rating.
- callable_status: Free-text description of callable status as written (e.g., "Non-callable", "NC10", "Make-whole").
- call_protection_years: If the answer explicitly states years of call protection (e.g., "10" or "10 years"), put only the numeric part as a string; otherwise null.
- issue_date: Original issue date if present in the answer (e.g., "January 15, 2024" or "2024-01-15"); else null.
- first_call_date: First call date if present (e.g., "January 15, 2035"); else null.
- callable_urls: Array of URL(s) that the answer cites to document callable status and/or call protection.
- maturity_date: Exact maturity date as written in the answer (do not normalize).
- maturity_urls: Array of URL(s) that the answer cites to document the maturity date.
- coupon_type: The coupon structure as written (e.g., "Fixed", "Floating").
- coupon_frequency: The payment frequency as written (e.g., "Semi-annual", "Semiannual", "2x per year").
- coupon_urls: Array of URL(s) that the answer cites to document the coupon type/frequency.
- general_urls: Any other reference URL(s) cited for this bond (e.g., prospectus, fact sheet) not already covered above.

Special rules:
- Only extract URLs that are explicitly present in the answer text. If none are present for a given category, return an empty list for that category.
- Preserve the capitalization and punctuation of values exactly as written in the answer where reasonable.
- Do not infer or fabricate missing information.
"""


# --------------------------------------------------------------------------- #
# Helper parsing and normalization                                            #
# --------------------------------------------------------------------------- #
_DATE_PATTERNS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
]


def try_parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    s = date_str.strip()
    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    # Try to extract a month name + year like "Jan 2035" -> assume day 15
    m = re.search(r"(?i)\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
                  r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\b", s)
    if m:
        mon_name = m.group(1)
        year = int(m.group(2))
        try:
            return datetime.strptime(f"{mon_name} 15, {year}", "%B %d, %Y")
        except Exception:
            try:
                return datetime.strptime(f"{mon_name} 15, {year}", "%b %d, %Y")
            except Exception:
                pass
    # Fallback: year only
    y = re.search(r"\b(20\d{2})\b", s)
    if y:
        year = int(y.group(1))
        # Pick mid-year to avoid boundary bias when only the year is known
        return datetime(year, 6, 30)
    return None


def date_in_range(date_str: Optional[str], start: datetime, end: datetime) -> bool:
    dt = try_parse_date(date_str)
    if not dt:
        return False
    return start <= dt <= end


def parse_years_from_text(text: str) -> Optional[float]:
    """Extract a numeric years value from free text like '10', '10 years', '10+'."""
    if not text:
        return None
    # NC10 pattern means 10 years non-call
    m = re.search(r"(?i)\bNC\s?(\d{1,2})\b", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # General '10 years' / '10+' / 'at least 10'
    m = re.search(r"(?i)\b(at least\s*)?(\d{1,2})(\+)?\s*(years|yrs|y|yr)?\b", text)
    if m:
        try:
            return float(m.group(2))
        except Exception:
            pass
    return None


def looks_non_callable(status: Optional[str]) -> bool:
    if not status:
        return False
    s = status.lower()
    return any(kw in s for kw in ["non-callable", "non callable", "no call", "noncallable", "nc life", "nc to maturity", "no-callable"])


def compute_years_between(d1: Optional[str], d2: Optional[str]) -> Optional[float]:
    """Compute approximate year difference between two date strings."""
    dt1 = try_parse_date(d1)
    dt2 = try_parse_date(d2)
    if not dt1 or not dt2:
        return None
    delta_days = abs((dt2 - dt1).days)
    return delta_days / 365.25


def validate_identifier_format(identifier: Optional[str], id_type: Optional[str]) -> bool:
    if not identifier or not identifier.strip():
        return False
    s = identifier.strip().replace(" ", "")
    if id_type:
        t = id_type.strip().lower()
        if "cusip" in t:
            # CUSIP is 9 characters (alphanumeric)
            return bool(re.fullmatch(r"[A-Za-z0-9]{9}", s))
        if "isin" in t:
            # ISIN is 12 characters (alphanumeric)
            return bool(re.fullmatch(r"[A-Za-z0-9]{12}", s))
    # If type unknown, accept non-empty identifier
    return True


def has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0 and any(isinstance(u, str) and u.strip() for u in urls))


def fail_leaf_due_to_missing_sources(node) -> None:
    node.score = 0.0
    node.status = "failed"


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root, data: BondExtraction) -> None:
    # Top-level critical node
    bond_node = evaluator.add_parallel(
        id="Bond_Identification",
        desc="Identify a U.S. corporate bond that satisfies all institutional investment criteria and provide all required information",
        parent=root,
        critical=True
    )

    # 1) Bond Identifier Provision (critical)
    evaluator.add_custom_node(
        result=validate_identifier_format(data.identifier, data.identifier_type),
        id="Bond_Identifier_Provision",
        desc="Provide the bond's CUSIP or ISIN identifier",
        parent=bond_node,
        critical=True
    )

    # 2) Issuer Information (critical group)
    issuer_node = evaluator.add_parallel(
        id="Issuer_Information",
        desc="Provide the issuer's name and verify it is a U.S. electric utility company",
        parent=bond_node,
        critical=True
    )

    # 2.1) Issuer Name Provision
    evaluator.add_custom_node(
        result=bool(data.issuer_name and data.issuer_name.strip()),
        id="Issuer_Name_Provision",
        desc="Provide the complete name of the bond issuer",
        parent=issuer_node,
        critical=True
    )

    # 2.2) Utility Sector Verification (URL-supported)
    util_sec_leaf = evaluator.add_leaf(
        id="Utility_Sector_Verification",
        desc="Verify the issuer is a U.S. electric utility company",
        parent=issuer_node,
        critical=True
    )
    if has_any_url(data.issuer_urls):
        issuer_claim = f"The issuer '{data.issuer_name or 'the issuer'}' is a U.S. electric utility company (engaged in electric generation, transmission, or distribution in the United States)."
        await evaluator.verify(
            claim=issuer_claim,
            node=util_sec_leaf,
            sources=data.issuer_urls,
            additional_instruction="Look for explicit statements that the company is a U.S. electric utility (e.g., investor-owned utility, regulated electric utility). Accept reasonable variants like 'electric utility holding company' if it clearly indicates U.S.-based electric utility operations."
        )
    else:
        fail_leaf_due_to_missing_sources(util_sec_leaf)

    # 2.3) Issuer Reference URL (URL-supported)
    issuer_ref_leaf = evaluator.add_leaf(
        id="Issuer_Reference_URL",
        desc="Provide reference URL documenting the issuer's business sector",
        parent=issuer_node,
        critical=True
    )
    if has_any_url(data.issuer_urls):
        issuer_ref_claim = f"The provided source(s) document that '{data.issuer_name or 'the issuer'}' operates as a U.S. electric utility."
        await evaluator.verify(
            claim=issuer_ref_claim,
            node=issuer_ref_leaf,
            sources=data.issuer_urls,
            additional_instruction="The page should clearly indicate the issuer operates as a U.S. electric utility. References like investor relations, company profile, or authoritative listings (e.g., regulator, industry association) are acceptable."
        )
    else:
        fail_leaf_due_to_missing_sources(issuer_ref_leaf)

    # 3) Credit Rating Information (critical group)
    rating_node = evaluator.add_parallel(
        id="Credit_Rating_Information",
        desc="Provide the credit rating and verify it meets investment grade requirements",
        parent=bond_node,
        critical=True
    )

    # 3.1) Credit Rating Provision
    evaluator.add_custom_node(
        result=bool(data.rating and data.rating.strip() and data.rating_agency and data.rating_agency.strip()),
        id="Credit_Rating_Provision",
        desc="Provide the specific credit rating from a major rating agency (S&P, Moody's, or Fitch)",
        parent=rating_node,
        critical=True
    )

    # 3.2) Investment Grade Verification (simple logical check)
    inv_grade_leaf = evaluator.add_leaf(
        id="Investment_Grade_Verification",
        desc="Verify the credit rating is at least BBB- (S&P/Fitch) or Baa3 (Moody's)",
        parent=rating_node,
        critical=True
    )
    agency = (data.rating_agency or "").strip()
    rating_str = (data.rating or "").strip()
    inv_grade_claim = (
        f"The rating '{rating_str}' from '{agency}' is investment grade and meets or exceeds the minimum threshold: "
        f"BBB- for S&P/Fitch or Baa3 for Moody's."
    )
    await evaluator.verify(
        claim=inv_grade_claim,
        node=inv_grade_leaf,
        additional_instruction=(
            "Judge by standard agency scales:\n"
            "- S&P/Fitch: AAA, AA(+/-), A(+/-), BBB(+/-) are investment grade; BBB- is the lowest IG.\n"
            "- Moody's: Aaa, Aa1-3, A1-3, Baa1-3 are investment grade; Baa3 is the lowest IG.\n"
            "Decide solely based on the agency specified in the claim."
        )
    )

    # 3.3) Rating Reference URL (URL-supported)
    rating_ref_leaf = evaluator.add_leaf(
        id="Rating_Reference_URL",
        desc="Provide reference URL documenting the credit rating",
        parent=rating_node,
        critical=True
    )
    if has_any_url(data.rating_urls):
        rating_ref_claim = f"The provided source(s) state a rating of '{rating_str}' from '{agency}' for the bond or its issuer."
        await evaluator.verify(
            claim=rating_ref_claim,
            node=rating_ref_leaf,
            sources=data.rating_urls,
            additional_instruction="Issuer-level ratings are acceptable if an explicit bond-level rating is not available, as long as the page clearly shows the rating and the agency."
        )
    else:
        fail_leaf_due_to_missing_sources(rating_ref_leaf)

    # 4) Callable Status Information (critical group)
    callable_node = evaluator.add_parallel(
        id="Callable_Status_Information",
        desc="Provide callable status and verify call protection requirements",
        parent=bond_node,
        critical=True
    )

    # 4.1) Callable Status Provision
    callable_info_exists = any([
        data.callable_status and data.callable_status.strip(),
        data.call_protection_years and data.call_protection_years.strip(),
        data.first_call_date and data.first_call_date.strip(),
        data.issue_date and data.issue_date.strip()
    ])
    evaluator.add_custom_node(
        result=callable_info_exists,
        id="Callable_Status_Provision",
        desc="Provide the callable status (non-callable or callable with call protection details)",
        parent=callable_node,
        critical=True
    )

    # 4.2) Call Protection Verification (URL-supported)
    call_protect_leaf = evaluator.add_leaf(
        id="Call_Protection_Verification",
        desc="Verify the bond is either non-callable or has call protection of at least 10 years from issue date",
        parent=callable_node,
        critical=True
    )

    if has_any_url(data.callable_urls):
        # Craft a precise claim depending on available info
        claim_text: str
        if looks_non_callable(data.callable_status):
            claim_text = "This bond is non-callable."
        else:
            yrs_text = parse_years_from_text(data.call_protection_years or "") or parse_years_from_text(data.callable_status or "")
            if yrs_text and yrs_text >= 10.0:
                claim_text = "This bond is callable but has call protection of at least 10 years from the original issue date."
            else:
                # Try to compute from dates if available
                yrs = None
                if data.issue_date and data.first_call_date:
                    yrs = compute_years_between(data.issue_date, data.first_call_date)
                if yrs and yrs >= 9.5:  # allow small rounding
                    claim_text = f"The bond's first call date ({data.first_call_date}) is at least 10 years after the original issue date ({data.issue_date})."
                else:
                    # Fallback generic claim; the verifier should check the schedule (e.g., 'NC10', 'No call until YYYY', etc.)
                    claim_text = "The bond has call protection of at least 10 years from the original issue date."
        await evaluator.verify(
            claim=claim_text,
            node=call_protect_leaf,
            sources=data.callable_urls,
            additional_instruction=(
                "Support the claim using the provided pages. Look for explicit indicators like 'Non-callable', 'NC10', "
                "'No call until [date]', 'First call date [date]'. "
                "Make-whole at any time (MWC) does NOT count as call protection. "
                "If both issue date and first call date appear, ensure the gap is at least 10 years."
            )
        )
    else:
        fail_leaf_due_to_missing_sources(call_protect_leaf)

    # 4.3) Callable Reference URL (URL-supported)
    callable_ref_leaf = evaluator.add_leaf(
        id="Callable_Reference_URL",
        desc="Provide reference URL documenting the callable status and call protection",
        parent=callable_node,
        critical=True
    )
    if has_any_url(data.callable_urls):
        callable_ref_claim = "The provided source(s) document the bond's callable status and the details of any call protection."
        await evaluator.verify(
            claim=callable_ref_claim,
            node=callable_ref_leaf,
            sources=data.callable_urls,
            additional_instruction="Accept pages like term sheets, prospectuses, offering circulars, or trusted market data that specify callable status and call schedule."
        )
    else:
        fail_leaf_due_to_missing_sources(callable_ref_leaf)

    # 5) Maturity Date Information (critical group)
    maturity_node = evaluator.add_parallel(
        id="Maturity_Date_Information",
        desc="Provide maturity date and verify it falls within required range",
        parent=bond_node,
        critical=True
    )

    # 5.1) Maturity Date Provision
    evaluator.add_custom_node(
        result=bool(data.maturity_date and data.maturity_date.strip()),
        id="Maturity_Date_Provision",
        desc="Provide the exact maturity date of the bond",
        parent=maturity_node,
        critical=True
    )

    # 5.2) Maturity Range Verification (computed)
    evaluator.add_custom_node(
        result=date_in_range(data.maturity_date, MATURITY_START, MATURITY_END),
        id="Maturity_Range_Verification",
        desc="Verify the maturity date is between January 1, 2032 and December 31, 2036 (inclusive)",
        parent=maturity_node,
        critical=True
    )

    # 5.3) Maturity Reference URL (URL-supported)
    maturity_ref_leaf = evaluator.add_leaf(
        id="Maturity_Reference_URL",
        desc="Provide reference URL documenting the maturity date",
        parent=maturity_node,
        critical=True
    )
    if has_any_url(data.maturity_urls):
        maturity_ref_claim = f"The bond's maturity date is {data.maturity_date}."
        await evaluator.verify(
            claim=maturity_ref_claim,
            node=maturity_ref_leaf,
            sources=data.maturity_urls,
            additional_instruction="The page should clearly state the exact maturity date of the bond."
        )
    else:
        fail_leaf_due_to_missing_sources(maturity_ref_leaf)

    # 6) Coupon Information (critical group)
    coupon_node = evaluator.add_parallel(
        id="Coupon_Information",
        desc="Provide coupon payment frequency and verify it meets requirements",
        parent=bond_node,
        critical=True
    )

    # 6.1) Coupon Frequency Provision
    coupon_info_exists = bool(data.coupon_type and data.coupon_type.strip() and data.coupon_frequency and data.coupon_frequency.strip())
    evaluator.add_custom_node(
        result=coupon_info_exists,
        id="Coupon_Frequency_Provision",
        desc="Provide the coupon payment frequency",
        parent=coupon_node,
        critical=True
    )

    # 6.2) Coupon Structure Verification (URL-supported)
    coupon_ver_leaf = evaluator.add_leaf(
        id="Coupon_Structure_Verification",
        desc="Verify the bond pays a fixed coupon on a semi-annual basis",
        parent=coupon_node,
        critical=True
    )
    if has_any_url(data.coupon_urls):
        coupon_claim = "This bond pays a fixed coupon on a semi-annual basis (i.e., twice per year)."
        await evaluator.verify(
            claim=coupon_claim,
            node=coupon_ver_leaf,
            sources=data.coupon_urls,
            additional_instruction="Look for 'Fixed' or equivalent for coupon type, and 'Semi-annual', 'Semiannual', or '2x per year' for frequency."
        )
    else:
        fail_leaf_due_to_missing_sources(coupon_ver_leaf)

    # 6.3) Coupon Reference URL (URL-supported)
    coupon_ref_leaf = evaluator.add_leaf(
        id="Coupon_Reference_URL",
        desc="Provide reference URL documenting the coupon payment structure",
        parent=coupon_node,
        critical=True
    )
    if has_any_url(data.coupon_urls):
        coupon_ref_claim = "The provided source(s) document that the coupon type is fixed and the payment frequency is semi-annual."
        await evaluator.verify(
            claim=coupon_ref_claim,
            node=coupon_ref_leaf,
            sources=data.coupon_urls,
            additional_instruction="Accept authoritative pages (prospectus, term sheet, trusted market data) that clearly state both fixed coupon and semi-annual payments."
        )
    else:
        fail_leaf_due_to_missing_sources(coupon_ref_leaf)


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
    Evaluate an answer for the U.S. electric utility corporate bond screening task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # independent checks under Bond_Identification node
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

    # Extract structured bond information from the agent's answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_bond(),
        template_class=BondExtraction,
        extraction_name="bond_extraction"
    )

    # Add criteria info to summary for transparency
    evaluator.add_custom_info(
        info={
            "maturity_range_start": MATURITY_START.strftime("%Y-%m-%d"),
            "maturity_range_end": MATURITY_END.strftime("%Y-%m-%d"),
            "investment_grade_thresholds": {
                "S&P/Fitch": "BBB- or better",
                "Moody's": "Baa3 or better"
            },
            "call_protection_requirement": "Non-callable OR ≥ 10 years from original issue date"
        },
        info_type="criteria",
        info_name="screening_criteria"
    )

    # Build tree and run verifications
    await build_verification_tree(evaluator, root, extraction)

    # Return standardized summary
    return evaluator.get_summary()