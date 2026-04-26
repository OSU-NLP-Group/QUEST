import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "multi_city_business_registration_requirements"
TASK_DESCRIPTION = """
A retail business owner is planning to expand operations from their current location to open new stores in both San Francisco and Los Angeles, California. To ensure compliance with all local business registration requirements, they need to determine:

1. Within how many days must a new business register in San Francisco after commencing operations?
2. By what date each year must San Francisco business registrations be renewed?
3. What is the basis (e.g., gross receipts, flat fee, payroll) used to calculate San Francisco's business registration fees?
4. What is the minimum business registration fee amount in San Francisco (excluding state fees) for businesses with gross receipts between $0 and $100,000?
5. By what date must Los Angeles business taxes be paid each year to avoid becoming delinquent?
6. What is the amount of the mandatory state fee that must be included with all California business licenses?
7. Where must the Business Registration Certificate be displayed in San Francisco?
8. If operating in both San Francisco and Los Angeles, does the business need to obtain separate licenses for each city, or does one California license cover both locations?

Provide specific answers for each requirement with supporting references to official government sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LADueDelinquency(BaseModel):
    due_date_text: Optional[str] = None  # e.g., "January 1 each year"
    delinquent_after_text: Optional[str] = None  # e.g., "last day of February"
    sources: List[str] = Field(default_factory=list)


class SFFeeRangePeriod(BaseModel):
    period: Optional[str] = None  # e.g., "2025–26" or "2025-26"
    min_fee: Optional[str] = None  # e.g., "$41"
    max_fee: Optional[str] = None  # e.g., "$45,000"
    sources: List[str] = Field(default_factory=list)


class BusinessRegExtraction(BaseModel):
    # 1. SF new registration deadline (days after commencing operations)
    sf_registration_deadline: FieldWithSources = Field(default_factory=FieldWithSources)

    # 2. SF annual renewal deadline
    sf_annual_renewal: FieldWithSources = Field(default_factory=FieldWithSources)

    # 3. SF fee structure basis
    sf_fee_basis: FieldWithSources = Field(default_factory=FieldWithSources)

    # 4. SF business registration fee range for specific period (e.g., 2025–26)
    sf_fee_range_2025_26: SFFeeRangePeriod = Field(default_factory=SFFeeRangePeriod)

    # 5. LA tax due and delinquent rule
    la_tax_due_rule: LADueDelinquency = Field(default_factory=LADueDelinquency)

    # 6. CA state CASp fee amount
    ca_state_casp_fee: FieldWithSources = Field(default_factory=FieldWithSources)

    # 7. SF certificate display requirement
    sf_certificate_display: FieldWithSources = Field(default_factory=FieldWithSources)

    # 8. Multi-city license requirement (separate vs single)
    multi_city_license_requirement: FieldWithSources = Field(default_factory=FieldWithSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_business_requirements() -> str:
    return """
    Extract the specific answers and corresponding sources (URLs) provided in the answer for each of the following 8 items. Use the exact phrasing or concise numeric/temporal phrases that appear in the answer for each value. For each item, also extract the set of URLs the answer cites as support for that specific item. If none are provided, return an empty list for sources.

    Items to extract (return fields exactly as specified):

    1) sf_registration_deadline:
       - value: The timeframe stated for when a new business must register in San Francisco after commencing operations (e.g., "30 days", "within 15 days", etc.).
       - sources: All URLs the answer cites that support this SF registration deadline.

    2) sf_annual_renewal:
       - value: The annual renewal deadline for San Francisco business registration (e.g., "last day of February", "February 28/29", etc.).
       - sources: All URLs the answer cites that support the SF annual renewal deadline.

    3) sf_fee_basis:
       - value: The basis used to calculate San Francisco business registration fees (e.g., "San Francisco gross receipts from the prior calendar year", "flat fee", "payroll expense", etc.).
       - sources: All URLs the answer cites that support the SF fee basis.

    4) sf_fee_range_2025_26:
       - period: The period stated in the answer for the fee range (e.g., "2025–26").
       - min_fee: The minimum SF business registration fee stated for that period, excluding any state fee (e.g., "$41").
       - max_fee: The maximum SF business registration fee stated for that period, excluding any state fee (e.g., "$45,000").
       - sources: All URLs the answer cites that support the 2025–26 fee range.

    5) la_tax_due_rule:
       - due_date_text: The date when Los Angeles business taxes are due each year (e.g., "January 1").
       - delinquent_after_text: The date after which they become delinquent (e.g., "last day of February").
       - sources: All URLs the answer cites that support the LA due/delinquency rule.

    6) ca_state_casp_fee:
       - value: The amount of the mandatory state CASp fee to be included with all California business licenses (e.g., "$4").
       - sources: All URLs the answer cites that support the CASp fee amount.

    7) sf_certificate_display:
       - value: The requirement for where/how the San Francisco Business Registration Certificate must be displayed (e.g., "conspicuously displayed at the place of business").
       - sources: All URLs the answer cites that support the display requirement.

    8) multi_city_license_requirement:
       - value: The statement regarding whether a business operating in both San Francisco and Los Angeles must obtain separate licenses for each city, or if one California license covers both (e.g., "separate licenses required for each city", or "one California license covers both").
       - sources: All URLs the answer cites that support this multi-city licensing statement.

    URI/Sources extraction rules:
    - Extract only URLs explicitly present in the answer.
    - Accept plain URLs or markdown links; always return the actual URL strings.
    - If the answer provides a general sources section instead of per-item sources, assign each URL to the relevant item(s) it supports; if unclear, include in multiple items as appropriate.

    If any textual value is truly missing in the answer, set that value to null. If sources are missing for an item, return an empty list for its sources.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_official_government_domain(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        if not netloc:
            return False
        # General .gov and .ca.gov
        if netloc.endswith(".gov") or netloc.endswith(".ca.gov"):
            return True
        # City official domains that may not end with .gov
        official_extras = (
            "sf.gov", "sfgov.org", "sftreasurer.org",
            "lacity.org", "finance.lacity.org", "latax.lacity.org",
            "leginfo.legislature.ca.gov",
        )
        return any(netloc == d or netloc.endswith("." + d) for d in official_extras)
    except Exception:
        return False


def _has_at_least_one_official(urls: List[str]) -> bool:
    return any(_is_official_government_domain(u) for u in urls if isinstance(u, str))


def _clean_sources(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip()


def _interpret_multi_city_requirement(text: Optional[str]) -> Optional[bool]:
    """
    Interpret whether the extracted statement asserts that separate city licenses are required.
    Returns:
        True  -> separate licenses/registrations are required for each city
        False -> one/single/state license covers both; separate not required
        None  -> cannot determine
    """
    if not text:
        return None
    t = text.lower()
    # Signals for separate requirements
    if ("separate" in t and ("license" in t or "registration" in t)) or \
       ("each city" in t) or \
       ("both cities" in t and ("license" in t or "registration" in t)) or \
       ("must register" in t and "both" in t):
        return True
    # Signals for single coverage (likely incorrect)
    if ("one license" in t) or ("single license" in t) or ("covers both" in t) or ("state license covers" in t):
        return False
    return None


async def _maybe_add_verify_task(
    evaluator: Evaluator,
    tasks: List,
    *,
    node_id: str,
    desc: str,
    parent: VerificationNode,
    critical: bool,
    claim: Optional[str],
    sources: Optional[List[str]],
    additional_instruction: str
) -> None:
    """
    Create a leaf node and, if claim and sources are valid, schedule a verify task.
    Otherwise, mark node as failed immediately.
    """
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )

    value_missing = claim is None or _normalize_text(claim) == ""
    srcs = _clean_sources(sources)

    if value_missing or len(srcs) == 0:
        # Missing content or missing sources => fail this critical leaf
        node.score = 0.0
        node.status = "failed"
        return

    # Schedule verification
    tasks.append((claim, srcs, node, additional_instruction))


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: BusinessRegExtraction,
    root: VerificationNode
) -> None:
    """
    Build the rubric tree and perform verifications using the extracted information.
    """
    # Top-level critical parallel node
    top = evaluator.add_parallel(
        id="Multi-City_Business_Registration_Requirements",
        desc="Answer provides all required SF/LA/CA business registration facts for the scenario and includes supporting references to official government sources.",
        parent=root,
        critical=True
    )

    verify_tasks: List[tuple] = []

    # 1) SF new business registration deadline
    sf_reg_value = _normalize_text(extracted.sf_registration_deadline.value)
    sf_reg_sources = _clean_sources(extracted.sf_registration_deadline.sources)
    sf_reg_claim = None if not sf_reg_value else f"San Francisco requires new businesses to register within {sf_reg_value} after commencing business operations in the City and County of San Francisco."
    await _maybe_add_verify_task(
        evaluator, verify_tasks,
        node_id="SF_New_Business_Registration_Deadline",
        desc="States that San Francisco requires new businesses to register within the stated timeframe after commencing business operations.",
        parent=top,
        critical=True,
        claim=sf_reg_claim,
        sources=sf_reg_sources,
        additional_instruction="Verify on official San Francisco government websites (e.g., Office of the Treasurer & Tax Collector) that the specified timeframe is required. Allow minor paraphrasing such as 'within X days of starting business.'"
    )

    # 2) SF annual renewal deadline
    sf_renew_value = _normalize_text(extracted.sf_annual_renewal.value)
    sf_renew_sources = _clean_sources(extracted.sf_annual_renewal.sources)
    sf_renew_claim = None if not sf_renew_value else f"San Francisco business registrations must be renewed annually by {sf_renew_value}."
    await _maybe_add_verify_task(
        evaluator, verify_tasks,
        node_id="SF_Annual_Renewal_Deadline",
        desc="States that San Francisco business registrations must be renewed annually by the specified deadline.",
        parent=top,
        critical=True,
        claim=sf_renew_claim,
        sources=sf_renew_sources,
        additional_instruction="Confirm that the renewal deadline matches official SF guidance. If the answer says 'last day of February,' treat Feb 28 or 29 as appropriate."
    )

    # 3) SF fee structure basis
    sf_basis_value = _normalize_text(extracted.sf_fee_basis.value)
    sf_basis_sources = _clean_sources(extracted.sf_fee_basis.sources)
    sf_basis_claim = None if not sf_basis_value else f"San Francisco calculates business registration fees based on {sf_basis_value}."
    await _maybe_add_verify_task(
        evaluator, verify_tasks,
        node_id="SF_Fee_Structure_Basis",
        desc="States the basis used to calculate San Francisco's business registration fees.",
        parent=top,
        critical=True,
        claim=sf_basis_claim,
        sources=sf_basis_sources,
        additional_instruction="Verify that the fee calculation basis is correctly described (e.g., SF gross receipts from the prior calendar year) on official SF government webpages."
    )

    # 4) SF business registration fee range for specified period (e.g., 2025–26)
    period = _normalize_text(extracted.sf_fee_range_2025_26.period)
    min_fee = _normalize_text(extracted.sf_fee_range_2025_26.min_fee)
    max_fee = _normalize_text(extracted.sf_fee_range_2025_26.max_fee)
    sf_fee_sources = _clean_sources(extracted.sf_fee_range_2025_26.sources)

    if period and min_fee and max_fee:
        sf_fee_claim = f"For the {period} period (excluding any state fee), San Francisco business registration fees range from a minimum of {min_fee} to a maximum of {max_fee}."
    else:
        sf_fee_claim = None

    await _maybe_add_verify_task(
        evaluator, verify_tasks,
        node_id="SF_Business_Registration_Fee_Range_2025_26",
        desc="States the San Francisco business registration fee range for the specified period (excluding the state fee).",
        parent=top,
        critical=True,
        claim=sf_fee_claim,
        sources=sf_fee_sources,
        additional_instruction="Verify the minimum and maximum fees for the specified period (e.g., 2025–26) on official SF government sources (e.g., fee schedules). Minor formatting differences (commas, currency symbols) are acceptable."
    )

    # 5) LA tax due and delinquency rule
    la_due = _normalize_text(extracted.la_tax_due_rule.due_date_text)
    la_delinq = _normalize_text(extracted.la_tax_due_rule.delinquent_after_text)
    la_sources = _clean_sources(extracted.la_tax_due_rule.sources)
    if la_due and la_delinq:
        la_claim = f"In Los Angeles, business taxes are due {la_due} each year and become delinquent if not paid by {la_delinq}."
    else:
        la_claim = None

    await _maybe_add_verify_task(
        evaluator, verify_tasks,
        node_id="LA_Tax_Due_And_Delinquency_Rule",
        desc="States when Los Angeles business taxes are due and when they become delinquent.",
        parent=top,
        critical=True,
        claim=la_claim,
        sources=la_sources,
        additional_instruction="Verify on official Los Angeles Office of Finance pages that the due date and delinquency cutoff are correctly stated. Allow paraphrasing like 'must be paid by the last day of February to avoid delinquency.'"
    )

    # 6) State CASp fee requirement
    casp_value = _normalize_text(extracted.ca_state_casp_fee.value)
    casp_sources = _clean_sources(extracted.ca_state_casp_fee.sources)
    casp_claim = None if not casp_value else f"A mandatory state CASp (Certified Access Specialist Program) fee of {casp_value} must be included with all California business licenses."
    await _maybe_add_verify_task(
        evaluator, verify_tasks,
        node_id="State_CASp_Fee_Requirement",
        desc="States the amount of the mandatory state CASp fee that must be included with all California business licenses.",
        parent=top,
        critical=True,
        claim=casp_claim,
        sources=casp_sources,
        additional_instruction="Confirm via official California state or city government sources that the stated CASp fee amount applies to all local business licenses in California."
    )

    # 7) SF certificate display requirement
    sf_disp_value = _normalize_text(extracted.sf_certificate_display.value)
    sf_disp_sources = _clean_sources(extracted.sf_certificate_display.sources)
    sf_disp_claim = None if not sf_disp_value else f"San Francisco requires the Business Registration Certificate to be {sf_disp_value}."
    await _maybe_add_verify_task(
        evaluator, verify_tasks,
        node_id="SF_Certificate_Display_Requirement",
        desc="States where/how the San Francisco Business Registration Certificate must be displayed.",
        parent=top,
        critical=True,
        claim=sf_disp_claim,
        sources=sf_disp_sources,
        additional_instruction="Verify on official SF government pages that the Business Registration Certificate display requirement matches (e.g., 'conspicuously displayed at the place of business')."
    )

    # 8) Multi-city license requirement (separate vs single)
    mc_value = _normalize_text(extracted.multi_city_license_requirement.value)
    mc_sources = _clean_sources(extracted.multi_city_license_requirement.sources)
    separate_required = _interpret_multi_city_requirement(mc_value)
    if separate_required is True:
        mc_claim = "A business operating in both San Francisco and Los Angeles must obtain separate city licenses/registrations for each city; a single California license does not cover both locations."
    elif separate_required is False:
        mc_claim = "A single California license covers both San Francisco and Los Angeles, so separate city licenses/registrations are not required."
    else:
        mc_claim = None

    await _maybe_add_verify_task(
        evaluator, verify_tasks,
        node_id="Multi_City_License_Requirement",
        desc="States whether separate city licenses/registrations are required for SF and LA.",
        parent=top,
        critical=True,
        claim=mc_claim,
        sources=mc_sources,
        additional_instruction="Use official SF and LA government sources to determine whether a business must register/license separately in each city. The correct rule is that each city requires its own registration/license; a statewide license does not replace local requirements."
    )

    # 9) Official government source references check (programmatic)
    # For each major requirement above, ensure at least one official government URL is cited.
    per_item_sources: Dict[str, List[str]] = {
        "SF_New_Business_Registration_Deadline": sf_reg_sources,
        "SF_Annual_Renewal_Deadline": sf_renew_sources,
        "SF_Fee_Structure_Basis": sf_basis_sources,
        "SF_Business_Registration_Fee_Range_2025_26": sf_fee_sources,
        "LA_Tax_Due_And_Delinquency_Rule": la_sources,
        "State_CASp_Fee_Requirement": casp_sources,
        "SF_Certificate_Display_Requirement": sf_disp_sources,
        "Multi_City_License_Requirement": mc_sources,
    }

    official_ok = True
    official_details: Dict[str, Dict[str, Any]] = {}
    for key, srcs in per_item_sources.items():
        has_official = _has_at_least_one_official(srcs)
        official_ok = official_ok and has_official
        official_details[key] = {
            "sources": srcs,
            "has_official_government_source": has_official
        }

    evaluator.add_custom_info(official_details, info_type="official_source_audit")

    evaluator.add_custom_node(
        result=official_ok,
        id="Official_Government_Source_References",
        desc="Provides supporting references/URLs from official government sources for the above requirements.",
        parent=top,
        critical=True
    )

    # Execute all scheduled verifications in parallel
    if verify_tasks:
        await evaluator.batch_verify(verify_tasks)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for multi-city (SF/LA/CA) business registration requirements.
    """
    # Initialize evaluator
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_business_requirements(),
        template_class=BusinessRegExtraction,
        extraction_name="business_registration_extraction"
    )

    # Add rubric-descriptive "ground truth expectations" as informational context (not enforced)
    evaluator.add_ground_truth({
        "expected_items": [
            "SF new business registration deadline after commencing operations",
            "SF annual renewal deadline",
            "SF fee structure basis",
            "SF fee range for specified period (min/max, excluding state fee)",
            "LA tax due date and delinquency cutoff",
            "CA state CASp fee amount",
            "SF Business Registration Certificate display requirement",
            "Multi-city (SF & LA) separate license requirement"
        ]
    }, gt_type="rubric_items")

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted, root)

    # Return summary
    return evaluator.get_summary()