import asyncio
import logging
import re
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ibr_federal_funding_2025"
TASK_DESCRIPTION = (
    "According to official Interstate Bridge Replacement Program sources, what is the total amount of committed "
    "federal funding that has been awarded to the program as of 2025? Your answer must identify the total amount and "
    "break down each individual federal grant that comprises this total, including the specific grant program name, "
    "award amount, and a reference URL from an official government or IBR Program source for each grant. Additionally, "
    "verify that the sum of the individual grants equals the stated total."
)

# Ground truth reference (for context/informational recording only)
GROUND_TRUTH = {
    "total_committed_funding": "$2.1 billion (rounded)",
    "grants": [
        {
            "program": "FHWA Bridge Investment Program (BIP)",
            "components": [
                {"amount": "$1.499 billion", "year": "2024", "type": "construction"},
                {"amount": "$1 million", "year": "2022", "type": "planning"},
            ],
            "total": "$1.5 billion (approx)"
        },
        {
            "program": "USDOT National Infrastructure Project Assistance (MEGA) program",
            "amount": "$600 million"
        },
        {
            "program": "USDOT Reconnecting Communities (RCN/RCP) program",
            "amount": "$30 million",
            "year": "2025"
        }
    ]
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BIPDetail(BaseModel):
    program_name: Optional[str] = None
    total_amount: Optional[str] = None
    construction_amount: Optional[str] = None
    construction_award_year: Optional[str] = None
    planning_amount: Optional[str] = None
    planning_award_year: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class SimpleGrant(BaseModel):
    program_name: Optional[str] = None
    award_amount: Optional[str] = None
    award_year: Optional[str] = None  # required for Reconnecting Communities check (should be 2025)
    urls: List[str] = Field(default_factory=list)


class FundingExtraction(BaseModel):
    total_amount: Optional[str] = None
    total_sources: List[str] = Field(default_factory=list)
    bip: Optional[BIPDetail] = None
    mega: Optional[SimpleGrant] = None
    reconnecting: Optional[SimpleGrant] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_funding() -> str:
    return """
    Extract, from the provided answer text, the Interstate Bridge Replacement Program's committed federal funding as of 2025 and the details for each component federal grant, exactly as stated in the answer.

    You must extract the following fields (return null for any missing field; extract only URLs explicitly present in the answer):

    1) total_amount: The total committed federal funding amount stated in the answer (e.g., "$2.1 billion").
    2) total_sources: An array of URLs cited in the answer that support the total amount. Only include URLs explicitly present in the answer.

    3) bip: Object describing the FHWA Bridge Investment Program (BIP) grants:
       - program_name: The name of the program as written in the answer (e.g., "FHWA Bridge Investment Program (BIP)").
       - total_amount: The total BIP funding amount as written in the answer (e.g., "$1.5 billion"), if present.
       - construction_amount: The construction grant amount as written in the answer (e.g., "$1.499 billion"), if present.
       - construction_award_year: The construction grant award year (e.g., "2024"), if present.
       - planning_amount: The planning grant amount as written in the answer (e.g., "$1 million"), if present.
       - planning_award_year: The planning grant award year (e.g., "2022"), if present.
       - urls: Array of URLs cited in the answer for BIP grants (include all relevant BIP-related URLs mentioned).

    4) mega: Object describing the USDOT MEGA grant (National Infrastructure Project Assistance program):
       - program_name: The name as written in the answer (e.g., "USDOT MEGA grant" or "National Infrastructure Project Assistance (MEGA) program").
       - award_amount: The award amount as written in the answer (e.g., "$600 million").
       - award_year: The award year if mentioned (optional).
       - urls: Array of URLs cited in the answer for the MEGA grant.

    5) reconnecting: Object describing the USDOT Reconnecting Communities grant:
       - program_name: The name as written in the answer (e.g., "Reconnecting Communities grant" or "Reconnecting Communities and Neighborhoods").
       - award_amount: The award amount as written in the answer (e.g., "$30 million").
       - award_year: The year (e.g., "2025") as written in the answer, if present.
       - urls: Array of URLs cited in the answer for the Reconnecting Communities grant.

    RULES:
    - Extract only what appears in the answer text; do not invent any values or URLs.
    - URLs may appear in plain form or markdown link form; extract actual URLs.
    - If multiple URLs are present for a section, include all of them.
    - If any field is not present in the answer, set it to null (for strings) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _parse_amount_to_millions(amount_str: Optional[str]) -> Optional[float]:
    """
    Parse a textual USD amount into millions (float).
    Supports formats like "$1.5 billion", "1.499B", "$600 million", "600M", "1,499,000,000", etc.
    Returns None if parsing fails.
    """
    if not amount_str:
        return None

    s = amount_str.strip().lower()
    s = s.replace(",", "")
    s = s.replace("usd", "").replace("$", "").strip()

    # Detect absolute amounts like 1499000000
    m_abs = re.findall(r"\d+(?:\.\d+)?", s)
    if not m_abs:
        return None
    try:
        num = float(m_abs[0])
    except Exception:
        return None

    # Unit detection
    # Cases: "billion", "bn", "b", "million", "mm", "m"
    is_billion = ("billion" in s) or ("bn" in s) or re.search(r"\b\d+(?:\.\d+)?\s*b\b", s) is not None
    is_million = ("million" in s) or ("mm" in s) or re.search(r"\b\d+(?:\.\d+)?\s*m\b", s) is not None

    # If absolute dollar amount indicated by many digits
    if not is_billion and not is_million:
        # Assume raw dollars if value > 10,000,000; convert to millions
        if num > 10_000_000:
            return num / 1_000_000.0
        # Otherwise ambiguous small numbers; assume million units if plausible
        # For values like 600 with no unit, assume millions (common)
        if num >= 1.0:
            return num  # assume it's already in millions
        return None

    if is_billion:
        return num * 1000.0
    if is_million:
        return num
    return None


def _has_official_source(urls: List[str]) -> bool:
    """
    Return True if at least one URL appears to be an official government source (.gov) or official IBR program site.
    """
    if not urls:
        return False
    allowed_patterns = [
        r"\.gov($|/)",          # any .gov domain
        r"(^|//)([^/]*\.)?transportation\.gov",  # USDOT
        r"(^|//)([^/]*\.)?fhwa\.dot\.gov",       # FHWA
        r"(^|//)([^/]*\.)?usdot\.gov",           # USDOT alt
        r"(^|//)([^/]*\.)?wsdot\.wa\.gov",       # Washington State DOT
        r"(^|//)([^/]*\.)?oregon\.gov",          # Oregon government
        r"(^|//)([^/]*\.)?interstatebridge\.org" # IBR Program
    ]
    for u in urls:
        if not isinstance(u, str):
            continue
        url = u.strip().lower()
        for pat in allowed_patterns:
            if re.search(pat, url):
                return True
    return False


def _all_official(urls: List[str]) -> bool:
    """Return True if every URL is official (optional stricter check)."""
    if not urls:
        return False
    allowed_patterns = [
        r"\.gov($|/)",
        r"(^|//)([^/]*\.)?transportation\.gov",
        r"(^|//)([^/]*\.)?fhwa\.dot\.gov",
        r"(^|//)([^/]*\.)?usdot\.gov",
        r"(^|//)([^/]*\.)?wsdot\.wa\.gov",
        r"(^|//)([^/]*\.)?oregon\.gov",
        r"(^|//)([^/]*\.)?interstatebridge\.org"
    ]
    for u in urls:
        url = (u or "").strip().lower()
        ok = any(re.search(pat, url) for pat in allowed_patterns)
        if not ok:
            return False
    return True


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_total(evaluator: Evaluator, parent_node, extracted: FundingExtraction) -> None:
    total_node = evaluator.add_parallel(
        id="Total_Federal_Funding_Amount",
        desc="States the total committed federal funding as of 2025 as documented in official IBR Program sources ($2.1 billion).",
        parent=parent_node,
        critical=True
    )

    total_exists = evaluator.add_custom_node(
        result=(bool(extracted.total_amount) and bool(extracted.total_sources)),
        id="total_exists",
        desc="Total amount and at least one source URL are provided in the answer",
        parent=total_node,
        critical=True
    )

    total_supported = evaluator.add_leaf(
        id="total_supported_by_sources",
        desc="Total committed federal funding amount is supported by cited official sources",
        parent=total_node,
        critical=True
    )
    claim_total = f"As of 2025, the total committed federal funding awarded to the Interstate Bridge Replacement Program is {extracted.total_amount or ''}."
    await evaluator.verify(
        claim=claim_total,
        node=total_supported,
        sources=extracted.total_sources,
        additional_instruction=(
            "Verify that official IBR Program or government webpages explicitly state the total committed federal "
            "funding amount for the Interstate Bridge Replacement Program. Allow minor rounding differences, e.g., "
            "$2.13B reported versus an answer stating $2.1B as a rounded total."
        )
    )

    official_source_check = evaluator.add_custom_node(
        result=_has_official_source(extracted.total_sources),
        id="total_has_official_source",
        desc="Total amount is backed by at least one official government or IBR Program URL",
        parent=total_node,
        critical=True
    )


async def verify_bip(evaluator: Evaluator, parent_node, bip: Optional[BIPDetail]) -> None:
    bip_node = evaluator.add_parallel(
        id="FHWA_BIP_Grant_Details",
        desc="Identifies the FHWA Bridge Investment Program (BIP) grant and provides details with an official URL.",
        parent=parent_node,
        critical=True
    )

    bip_exists = evaluator.add_custom_node(
        result=(
            bip is not None and
            bool(bip.program_name) and
            bool(bip.urls) and
            (
                bool(bip.total_amount) or
                (bool(bip.construction_amount) and bool(bip.planning_amount))
            )
        ),
        id="bip_exists",
        desc="BIP grant info exists with required fields (program name, URLs, and total or breakdown amounts)",
        parent=bip_node,
        critical=True
    )

    bip_program_match = evaluator.add_leaf(
        id="bip_program_name_match",
        desc="Program name corresponds to the FHWA Bridge Investment Program (BIP)",
        parent=bip_node,
        critical=True
    )
    claim_prog = "The grant program is the FHWA Bridge Investment Program (BIP) for the Interstate Bridge Replacement Program."
    await evaluator.verify(
        claim=claim_prog,
        node=bip_program_match,
        sources=bip.urls if bip else [],
        additional_instruction="Allow variants: 'Bridge Investment Program', 'FHWA BIP'. Confirm it is the FHWA program."
    )

    # Construction grant check (2024, $1.499B)
    bip_construction_supported = evaluator.add_leaf(
        id="bip_construction_supported",
        desc="BIP construction grant (2024) amount is supported by cited official sources",
        parent=bip_node,
        critical=True
    )
    claim_constr = (
        f"In 2024, the FHWA Bridge Investment Program awarded a construction grant of "
        f"{bip.construction_amount or ''} to the Interstate Bridge Replacement Program."
    )
    await evaluator.verify(
        claim=claim_constr,
        node=bip_construction_supported,
        sources=bip.urls if bip else [],
        additional_instruction=(
            "Verify the page states the BIP construction grant amount awarded in 2024 to the Interstate Bridge Replacement Program. "
            "Allow rounding equivalence for '$1.499B'."
        )
    )

    # Planning grant check (2022, $1M)
    bip_planning_supported = evaluator.add_leaf(
        id="bip_planning_supported",
        desc="BIP planning grant (2022) amount is supported by cited official sources",
        parent=bip_node,
        critical=True
    )
    claim_plan = (
        f"In 2022, the FHWA Bridge Investment Program awarded a planning grant of "
        f"{bip.planning_amount or ''} to the Interstate Bridge Replacement Program."
    )
    await evaluator.verify(
        claim=claim_plan,
        node=bip_planning_supported,
        sources=bip.urls if bip else [],
        additional_instruction=(
            "Verify the page states the BIP planning grant amount awarded in 2022 to the Interstate Bridge Replacement Program."
        )
    )

    # Optional: total amount check if present (approx $1.5B)
    bip_total_present = evaluator.add_custom_node(
        result=bool(bip and bip.total_amount),
        id="bip_total_present",
        desc="BIP total amount is explicitly provided in the answer",
        parent=bip_node,
        critical=True
    )
    bip_total_supported = evaluator.add_leaf(
        id="bip_total_supported",
        desc="Total BIP funding (approx $1.5B) is supported by cited official sources",
        parent=bip_node,
        critical=True
    )
    claim_bip_total = (
        f"The Interstate Bridge Replacement Program received approximately {bip.total_amount or ''} from the FHWA Bridge Investment Program, "
        f"consisting of a construction grant (2024) and a planning grant (2022)."
    )
    # Make the total verification depend on the 'bip_total_present' precondition so it skips if total not provided
    await evaluator.verify(
        claim=claim_bip_total,
        node=bip_total_supported,
        sources=bip.urls if bip else [],
        additional_instruction=(
            "Confirm the page(s) indicate the total BIP funding for the Interstate Bridge Replacement Program "
            "is approximately $1.5B. Allow rounding."
        ),
        extra_prerequisites=[bip_total_present]
    )

    bip_official = evaluator.add_custom_node(
        result=_has_official_source(bip.urls if bip else []),
        id="bip_has_official_source",
        desc="BIP grant has at least one official government or IBR Program URL",
        parent=bip_node,
        critical=True
    )

    # Internal consistency: breakdown sums should approximately equal total, if both are present
    bip_breakdown_consistency = evaluator.add_custom_node(
        result=(
            (bip is not None) and
            (bip.construction_amount is not None) and
            (bip.planning_amount is not None) and
            (bip.total_amount is not None) and
            (
                abs(
                    (_parse_amount_to_millions(bip.construction_amount) or -1) +
                    (_parse_amount_to_millions(bip.planning_amount) or -1) -
                    (_parse_amount_to_millions(bip.total_amount) or -1)
                ) <= 10  # allow up to $10M tolerance inside BIP total
            )
        ),
        id="bip_breakdown_matches_total",
        desc="BIP breakdown (construction + planning) approximately equals BIP total",
        parent=bip_node,
        critical=True
    )


async def verify_mega(evaluator: Evaluator, parent_node, mega: Optional[SimpleGrant]) -> None:
    mega_node = evaluator.add_parallel(
        id="USDOT_Mega_Grant_Details",
        desc="Identifies the USDOT MEGA grant with program name, amount ($600M), and official URL.",
        parent=parent_node,
        critical=True
    )

    mega_exists = evaluator.add_custom_node(
        result=(mega is not None and bool(mega.program_name) and bool(mega.award_amount) and bool(mega.urls)),
        id="mega_exists",
        desc="MEGA grant info exists with required fields (program name, amount, URLs)",
        parent=mega_node,
        critical=True
    )

    mega_program_match = evaluator.add_leaf(
        id="mega_program_name_match",
        desc="Program name corresponds to USDOT MEGA program",
        parent=mega_node,
        critical=True
    )
    claim_prog = (
        "The grant program is the USDOT National Infrastructure Project Assistance (MEGA) program for the Interstate Bridge Replacement Program."
    )
    await evaluator.verify(
        claim=claim_prog,
        node=mega_program_match,
        sources=mega.urls if mega else [],
        additional_instruction=(
            "Allow variants such as 'MEGA grant', 'National Infrastructure Project Assistance'. Confirm it is the USDOT MEGA program."
        )
    )

    mega_amount_supported = evaluator.add_leaf(
        id="mega_amount_supported",
        desc="MEGA grant amount ($600M) is supported by cited official sources",
        parent=mega_node,
        critical=True
    )
    claim_amt = f"The USDOT MEGA grant award amount for the Interstate Bridge Replacement Program is {mega.award_amount or ''}."
    await evaluator.verify(
        claim=claim_amt,
        node=mega_amount_supported,
        sources=mega.urls if mega else [],
        additional_instruction="Confirm the page states a $600 million MEGA grant for the Interstate Bridge Replacement Program."
    )

    mega_official = evaluator.add_custom_node(
        result=_has_official_source(mega.urls if mega else []),
        id="mega_has_official_source",
        desc="MEGA grant has at least one official government or IBR Program URL",
        parent=mega_node,
        critical=True
    )


async def verify_reconnecting(evaluator: Evaluator, parent_node, reconnecting: Optional[SimpleGrant]) -> None:
    rc_node = evaluator.add_parallel(
        id="USDOT_Reconnecting_Communities_Grant_Details",
        desc="Identifies the USDOT Reconnecting Communities grant with program name, amount ($30M), 2025 award year, and official URL.",
        parent=parent_node,
        critical=True
    )

    rc_exists = evaluator.add_custom_node(
        result=(reconnecting is not None and bool(reconnecting.program_name) and bool(reconnecting.award_amount) and bool(reconnecting.urls)),
        id="rc_exists",
        desc="Reconnecting Communities grant info exists with required fields (program name, amount, URLs)",
        parent=rc_node,
        critical=True
    )

    rc_program_match = evaluator.add_leaf(
        id="rc_program_name_match",
        desc="Program name corresponds to USDOT Reconnecting Communities program (RCN/RCP)",
        parent=rc_node,
        critical=True
    )
    claim_prog = "The grant is part of USDOT's Reconnecting Communities program (RCN or RCP) for the Interstate Bridge Replacement Program."
    await evaluator.verify(
        claim=claim_prog,
        node=rc_program_match,
        sources=reconnecting.urls if reconnecting else [],
        additional_instruction=(
            "Allow naming variants such as 'Reconnecting Communities', 'Reconnecting Communities and Neighborhoods (RCN)', "
            "or 'Reconnecting Communities Pilot Program (RCP)'. Confirm it is a USDOT Reconnecting Communities grant."
        )
    )

    rc_amount_supported = evaluator.add_leaf(
        id="rc_amount_supported",
        desc="Reconnecting Communities grant amount ($30M) is supported by cited official sources",
        parent=rc_node,
        critical=True
    )
    claim_amt = f"The USDOT Reconnecting Communities grant award amount for the Interstate Bridge Replacement Program is {reconnecting.award_amount or ''}."
    await evaluator.verify(
        claim=claim_amt,
        node=rc_amount_supported,
        sources=reconnecting.urls if reconnecting else [],
        additional_instruction="Confirm the page states a $30 million Reconnecting Communities grant for the Interstate Bridge Replacement Program."
    )

    rc_award_year_supported = evaluator.add_leaf(
        id="rc_award_year_supported",
        desc="Reconnecting Communities grant was awarded in 2025, supported by official sources",
        parent=rc_node,
        critical=True
    )
    claim_year = "The Reconnecting Communities grant for the Interstate Bridge Replacement Program was awarded in 2025."
    await evaluator.verify(
        claim=claim_year,
        node=rc_award_year_supported,
        sources=reconnecting.urls if reconnecting else [],
        additional_instruction="Confirm the page indicates the grant award occurred in 2025."
    )

    rc_official = evaluator.add_custom_node(
        result=_has_official_source(reconnecting.urls if reconnecting else []),
        id="rc_has_official_source",
        desc="Reconnecting Communities grant has at least one official government or IBR Program URL",
        parent=rc_node,
        critical=True
    )


async def verify_arithmetic(evaluator: Evaluator, parent_node, extracted: FundingExtraction) -> None:
    """
    Verify that the sum of the individual grants equals the stated total, allowing rounding variance.
    Tolerance: up to $50M difference allowed (to accommodate rounding to $2.1B vs $2.13B).
    """
    # Compute amounts in millions
    total_m = _parse_amount_to_millions(extracted.total_amount)
    bip_total_m = None
    if extracted.bip:
        if extracted.bip.total_amount:
            bip_total_m = _parse_amount_to_millions(extracted.bip.total_amount)
        else:
            ca = _parse_amount_to_millions(extracted.bip.construction_amount)
            pa = _parse_amount_to_millions(extracted.bip.planning_amount)
            if ca is not None and pa is not None:
                bip_total_m = ca + pa

    mega_m = _parse_amount_to_millions(extracted.mega.award_amount) if extracted.mega else None
    rc_m = _parse_amount_to_millions(extracted.reconnecting.award_amount) if extracted.reconnecting else None

    all_present = (total_m is not None and bip_total_m is not None and mega_m is not None and rc_m is not None)
    sum_m = None
    if all_present:
        sum_m = bip_total_m + mega_m + rc_m

    tolerance_millions = 50.0
    arithmetic_ok = bool(all_present and abs(sum_m - total_m) <= tolerance_millions)

    evaluator.add_custom_node(
        result=arithmetic_ok,
        id="Arithmetic_Verification_With_Rounding",
        desc="Sum of individual grants is consistent with stated total committed federal funding (allow rounding up to $50M).",
        parent=parent_node,
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for Interstate Bridge Replacement Program federal funding as of 2025.
    """
    # Initialize evaluator with PARALLEL aggregation at root
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
        default_model=model
    )

    # Record ground truth info (for context)
    evaluator.add_ground_truth(GROUND_TRUTH, gt_type="reference_info")

    # Extract structured data from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_funding(),
        template_class=FundingExtraction,
        extraction_name="ibr_funding_extraction"
    )

    # Build verification tree according to rubric
    # All nodes under root are critical to emulate rubric "critical" requirement
    await verify_total(evaluator, root, extracted)
    await verify_bip(evaluator, root, extracted.bip)
    await verify_mega(evaluator, root, extracted.mega)
    await verify_reconnecting(evaluator, root, extracted.reconnecting)
    await verify_arithmetic(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()