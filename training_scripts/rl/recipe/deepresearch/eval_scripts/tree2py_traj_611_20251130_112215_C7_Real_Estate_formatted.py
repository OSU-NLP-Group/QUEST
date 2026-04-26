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
TASK_ID = "state_broker_criteria"
TASK_DESCRIPTION = """
Identify a U.S. state that meets ALL of the following real estate broker licensing and regulatory criteria:

1. The state's real estate broker pre-licensing education requirement is less than 100 hours
2. The minimum age requirement for obtaining a broker license is 18 years old
3. The state does NOT accept a bachelor's degree as an alternative to the required education hours for broker licensing
4. The state requires active real estate salesperson license experience before an individual can obtain a broker license
5. The required salesperson experience period is less than 3 years
6. The state has a documented specific assessment rate (percentage of market value) for commercial property tax purposes
7. The state has a documented regular cycle for real estate property tax reassessment
8. The state requires fingerprints or a criminal background check as part of the licensing process
9. The state has continuing education requirements for license renewal
10. The state has documented requirements regarding property management licensing (whether a separate license is needed or if a real estate license is required)

Provide the name of the state and at least one authoritative reference URL that supports the licensing requirements.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StateCriteriaExtraction(BaseModel):
    """
    Structured extraction of the identified state and any URLs the answer provides.
    For textual fields, only extract if explicitly stated in the answer; otherwise return null.
    """
    state_name: Optional[str] = None

    # Optional explicit statements in the answer (if present)
    broker_prelicensing_hours: Optional[str] = None
    min_age_broker: Optional[str] = None
    bachelor_degree_alternative_policy: Optional[str] = None
    requires_salesperson_experience: Optional[str] = None
    salesperson_experience_period: Optional[str] = None
    commercial_property_assessment_rate: Optional[str] = None
    property_tax_reassessment_cycle: Optional[str] = None
    background_check_requirement: Optional[str] = None
    continuing_education_requirements: Optional[str] = None
    property_management_license_requirement: Optional[str] = None

    # Source URLs (explicitly present in the answer)
    licensing_urls: List[str] = Field(default_factory=list)
    tax_urls: List[str] = Field(default_factory=list)
    property_management_urls: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)  # catch-all list of any URLs mentioned


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_state_criteria() -> str:
    return """
    Extract structured information from the answer as follows:

    1) state_name: The specific U.S. state identified by name.

    2) If the answer explicitly states any of the following details, extract them verbatim; otherwise return null:
       - broker_prelicensing_hours: The broker pre-licensing education hours requirement as stated (e.g., "90 hours").
       - min_age_broker: The minimum age requirement to obtain a broker license (e.g., "18").
       - bachelor_degree_alternative_policy: Whether a bachelor's degree is accepted as an alternative to broker pre-licensing education hours. Extract the exact statement if given (e.g., "Bachelor's degree is not accepted as a substitute for broker education hours"), otherwise null.
       - requires_salesperson_experience: Whether salesperson license experience is required before getting a broker license. Extract the statement if given; otherwise null.
       - salesperson_experience_period: The stated length of required salesperson experience (e.g., "2 years", "24 months"). If not stated, null.
       - commercial_property_assessment_rate: The stated assessment rate/ratio for commercial property (e.g., "25% of market value"). If not stated, null.
       - property_tax_reassessment_cycle: The stated reassessment cycle (e.g., "annual", "every 2 years"). If not stated, null.
       - background_check_requirement: The stated requirement for fingerprints or criminal background check. If not stated, null.
       - continuing_education_requirements: The stated requirement and cadence for continuing education to renew broker licenses. If not stated, null.
       - property_management_license_requirement: The stated rule about property management licensing (e.g., "property managers must have a real estate license" or "a separate property manager license is required"). If not stated, null.

    3) Extract URLs explicitly present in the answer and categorize them if possible:
       - licensing_urls: URLs that appear to be related to real estate broker licensing requirements/prerequisites.
       - tax_urls: URLs that appear to relate to property tax assessment or reassessment cycles.
       - property_management_urls: URLs that appear to relate to property management licensing requirements.
       - reference_urls: A complete list of all URLs mentioned in the answer (include all categories above, as long as they are present in the answer text).

    IMPORTANT:
    - Only extract information explicitly present in the answer. Do not infer missing information.
    - Extract only valid URLs that are explicitly mentioned; if a source is only described but not provided as a URL, do not include it.
    - If a URL is missing a protocol, prepend "http://".
    - If categorization is unclear, include the URLs in 'reference_urls' and leave the specific category lists empty.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    """Deduplicate while preserving order."""
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _union_nonempty(*url_lists: List[str]) -> List[str]:
    """Union and dedup of multiple lists; return empty list if all empty."""
    combined = []
    for lst in url_lists:
        combined.extend(lst or [])
    return _dedup_urls(combined)


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_state_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: StateCriteriaExtraction,
) -> None:
    """
    Build verification tree and run checks for all criteria.
    All children under this node are critical; any failure should fail the overall result.
    """
    # Create a critical parallel node to host all checks
    main_node = evaluator.add_parallel(
        id="all_criteria",
        desc="Identification and verification of a U.S. state that meets all specified real estate broker licensing criteria",
        parent=parent_node,
        critical=True,
    )

    # Prepare grouped source sets
    licensing_sources = _union_nonempty(extracted.licensing_urls, extracted.reference_urls)
    tax_sources = _union_nonempty(extracted.tax_urls, extracted.reference_urls)
    pm_sources = _union_nonempty(extracted.property_management_urls, extracted.reference_urls)
    any_sources = _union_nonempty(extracted.reference_urls)

    state = extracted.state_name or ""

    # 1. State identified
    evaluator.add_custom_node(
        result=bool(extracted.state_name and extracted.state_name.strip()),
        id="state_identified",
        desc="A specific U.S. state is clearly identified by name",
        parent=main_node,
        critical=True,
    )

    # 2. Broker license available
    node_broker_license = evaluator.add_leaf(
        id="broker_license_available",
        desc="The identified state offers real estate broker licensing",
        parent=main_node,
        critical=True,
    )
    claim_broker_license = f"The state of {state} offers a real estate broker license credential (i.e., a 'Real Estate Broker' license)."
    await evaluator.verify(
        claim=claim_broker_license,
        node=node_broker_license,
        sources=licensing_sources if licensing_sources else None,
        additional_instruction="Check official licensing authority or state statute pages for a Real Estate Broker license program. Synonyms like 'Broker License' or 'Real Estate Broker' are acceptable.",
    )

    # 3. Education hours below 100
    node_hours = evaluator.add_leaf(
        id="education_hours_below_100",
        desc="The state's broker pre-licensing education requirement is less than 100 hours",
        parent=main_node,
        critical=True,
    )
    claim_hours = f"In {state}, the broker pre-licensing education requirement is less than 100 hours."
    await evaluator.verify(
        claim=claim_hours,
        node=node_hours,
        sources=licensing_sources if licensing_sources else None,
        additional_instruction="Confirm the broker pre-licensing education hour requirement on the licensing page. If the page shows a number like 60, 72, 90 hours, that is less than 100. Treat 'credit hours' as hours unless explicitly defined otherwise.",
    )

    # 4. Minimum age 18
    node_age = evaluator.add_leaf(
        id="minimum_age_18",
        desc="The state's minimum age requirement for broker license is 18 years old",
        parent=main_node,
        critical=True,
    )
    claim_age = f"In {state}, the minimum age to obtain a real estate broker license is 18 years old."
    await evaluator.verify(
        claim=claim_age,
        node=node_age,
        sources=licensing_sources if licensing_sources else None,
        additional_instruction="Look for prerequisites or eligibility on the licensing page. Phrases like 'must be at least 18 years old' are acceptable.",
    )

    # 5. No bachelor's degree alternative
    node_no_degree_alt = evaluator.add_leaf(
        id="no_bachelor_degree_alternative",
        desc="The state does NOT accept a bachelor's degree as an alternative to education hours",
        parent=main_node,
        critical=True,
    )
    claim_no_degree_alt = f"In {state}, a bachelor's degree is NOT accepted as an alternative to the broker pre-licensing education hour requirement."
    await evaluator.verify(
        claim=claim_no_degree_alt,
        node=node_no_degree_alt,
        sources=licensing_sources if licensing_sources else None,
        additional_instruction="Check whether the licensing page allows substituting a bachelor's degree for the broker pre-licensing hours. Consider the claim supported only if the page explicitly requires specified hours with no allowance for degree substitution or explicitly states that degrees are not accepted as substitutes.",
    )

    # 6. Requires salesperson experience
    node_sales_exp = evaluator.add_leaf(
        id="requires_salesperson_experience",
        desc="The state requires active salesperson license experience before obtaining broker license",
        parent=main_node,
        critical=True,
    )
    claim_sales_exp = f"In {state}, active real estate salesperson license experience is required before one can obtain a broker license."
    await evaluator.verify(
        claim=claim_sales_exp,
        node=node_sales_exp,
        sources=licensing_sources if licensing_sources else None,
        additional_instruction="Look for terms like 'experience as a licensed salesperson', 'actively engaged as a real estate salesperson', or similar prerequisites.",
    )

    # 7. Experience under 3 years
    node_exp_under3 = evaluator.add_leaf(
        id="experience_under_3_years",
        desc="The required salesperson experience is less than 3 years",
        parent=main_node,
        critical=True,
    )
    claim_exp_under3 = f"In {state}, the required real estate salesperson experience period before obtaining a broker license is less than 3 years."
    await evaluator.verify(
        claim=claim_exp_under3,
        node=node_exp_under3,
        sources=licensing_sources if licensing_sources else None,
        additional_instruction="Check the page for the required salesperson experience period. If it states '2 years', '24 months', or any period clearly under 3 years, consider the claim supported.",
    )

    # 8. Commercial property tax assessment rate documented
    node_tax_rate = evaluator.add_leaf(
        id="commercial_property_tax_rate",
        desc="The state has a specific commercial property tax assessment rate documented",
        parent=main_node,
        critical=True,
    )
    claim_tax_rate = f"In {state}, there is a documented assessment rate (percentage of market value) specifically for commercial property taxation."
    await evaluator.verify(
        claim=claim_tax_rate,
        node=node_tax_rate,
        sources=tax_sources if tax_sources else None,
        additional_instruction="Look for property tax resources indicating an 'assessment ratio' or percentage of market value for commercial property. This may be statewide or set by statute; county-level pages are acceptable if they document a standard rate applicable within the state.",
    )

    # 9. Property tax reassessment cycle documented
    node_reassess = evaluator.add_leaf(
        id="property_tax_reassessment_cycle",
        desc="The state has a documented regular reassessment cycle for real estate property",
        parent=main_node,
        critical=True,
    )
    claim_reassess = f"In {state}, there is a documented regular reassessment cycle for real estate property taxes."
    await evaluator.verify(
        claim=claim_reassess,
        node=node_reassess,
        sources=tax_sources if tax_sources else None,
        additional_instruction="Look for 'annual', 'biennial', or another specified cycle for reassessing property values for taxation purposes.",
    )

    # 10. Background check required
    node_bg = evaluator.add_leaf(
        id="background_check_required",
        desc="The state requires fingerprints or background check for licensing",
        parent=main_node,
        critical=True,
    )
    claim_bg = f"In {state}, fingerprints or a criminal background check are required as part of the real estate broker licensing process."
    await evaluator.verify(
        claim=claim_bg,
        node=node_bg,
        sources=licensing_sources if licensing_sources else None,
        additional_instruction="Check for 'fingerprints', 'criminal background check', or similar screening requirements on the broker licensing page.",
    )

    # 11. Continuing education exists
    node_ce = evaluator.add_leaf(
        id="continuing_education_exists",
        desc="The state has continuing education requirements for license renewal",
        parent=main_node,
        critical=True,
    )
    claim_ce = f"In {state}, continuing education is required for real estate broker license renewal."
    await evaluator.verify(
        claim=claim_ce,
        node=node_ce,
        sources=licensing_sources if licensing_sources else None,
        additional_instruction="Look for renewal requirements specifying 'continuing education' hours or courses for brokers.",
    )

    # 12. Property management licensing requirement documented
    node_pm = evaluator.add_leaf(
        id="property_management_license_requirement",
        desc="The state's property management licensing requirement is documented",
        parent=main_node,
        critical=True,
    )
    claim_pm = f"In {state}, property management licensing requirements are documented (e.g., a separate property manager license is required or property management requires a real estate license)."
    await evaluator.verify(
        claim=claim_pm,
        node=node_pm,
        sources=pm_sources if pm_sources else None,
        additional_instruction="Check for official guidance on whether property managers need a separate license or must hold a real estate license to perform property management services.",
    )

    # 13. Reference URLs provided (at least one)
    evaluator.add_custom_node(
        result=bool(extracted.reference_urls and len(extracted.reference_urls) > 0),
        id="reference_urls_provided",
        desc="At least one authoritative reference URL is provided supporting the state's requirements",
        parent=main_node,
        critical=True,
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
    Evaluate an answer for the state broker licensing criteria task.
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
        default_model=model,
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_state_criteria(),
        template_class=StateCriteriaExtraction,
        extraction_name="state_criteria_extraction",
    )

    # Optional: record a small custom info summary
    evaluator.add_custom_info(
        info={
            "extracted_state": extracted.state_name,
            "licensing_urls_count": len(extracted.licensing_urls),
            "tax_urls_count": len(extracted.tax_urls),
            "pm_urls_count": len(extracted.property_management_urls),
            "all_reference_urls_count": len(extracted.reference_urls),
        },
        info_type="extraction_summary",
        info_name="extraction_summary",
    )

    # Build verification tree and run checks
    await verify_state_requirements(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()