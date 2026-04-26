import asyncio
import logging
from typing import Any, List, Optional, Dict
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "senior_lifetime_pass_evaluation"
TASK_DESCRIPTION = (
    "For a U.S. citizen who is 62 years old and wants lifetime access to federal recreation lands managed by the "
    "National Park Service and other federal agencies, identify the appropriate America the Beautiful pass type and "
    "provide comprehensive information including: (1) the exact base cost of the pass, (2) the total cost when purchasing "
    "online or by mail through the USGS Store including any processing fees, (3) the total cost when purchasing in person "
    "at a Federal recreation site, (4) the specific discount benefits provided by this pass including the discount percentage "
    "on amenity fees and any limitations on what is NOT covered, and (5) the documentation requirements to prove eligibility "
    "for this pass. Include reference URLs from official USGS or National Park Service websites to support your information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SeniorPassExtraction(BaseModel):
    # Identification
    pass_type: Optional[str] = None

    # Costs
    base_cost: Optional[str] = None
    online_or_mail_total_cost: Optional[str] = None
    in_person_total_cost: Optional[str] = None

    # Eligibility and documentation
    eligibility_requirements: Optional[str] = None
    documentation_requirements: Optional[str] = None

    # Validity and rules
    lifetime_validity: Optional[str] = None
    discount_benefits_and_fee_limitations: Optional[str] = None
    fee_coverage_scope: Optional[str] = None
    admission_rules: Optional[str] = None
    physical_card_only: Optional[str] = None

    # Official references
    official_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_senior_pass() -> str:
    return """
    Extract all relevant information the answer provides about the appropriate America the Beautiful pass for a 62-year-old U.S. citizen seeking lifetime access.

    Required fields:
    1. pass_type: The name of the pass stated in the answer (e.g., "Senior Lifetime Pass", "Lifetime Senior Pass", "Senior Pass (Lifetime)").
    2. base_cost: The exact base cost of the pass (exclude processing fees).
    3. online_or_mail_total_cost: The total cost when purchasing online or by mail via the USGS Store (include processing/document fee if stated).
    4. in_person_total_cost: The total cost when purchasing in person at a federal recreation site (note if there is no processing fee).
    5. eligibility_requirements: Eligibility statement (should include U.S. citizen or permanent resident AND age requirement 62+ at time of purchase).
    6. documentation_requirements: Documentation required to prove eligibility (proof of age and proof of U.S. citizenship or permanent residency; include examples if present such as driver's license, U.S. passport, Permanent Resident Card/Green Card).
    7. lifetime_validity: Statement about validity (e.g., valid for lifetime; does not expire).
    8. discount_benefits_and_fee_limitations: Description of discount benefits (e.g., 50% on some expanded amenity fees) and limitations (e.g., does not cover special recreation permit fees or concessioner fees).
    9. fee_coverage_scope: Statement that the pass covers entrance fees and standard amenity (day-use) fees at sites managed by six federal agencies: NPS, FWS, USFS, BLM, BOR, USACE.
    10. admission_rules: Admission rules (e.g., per-vehicle areas admit pass owner and passengers in non-commercial vehicle; per-person areas admit pass owner + up to 3 adults; children under 16 free).
    11. physical_card_only: Statement that the pass must be a physical card; receipts, photos, confirmation emails, or photocopies are not valid for entry.
    12. official_reference_urls: A list of official URLs cited in the answer that support the above information. IMPORTANT: Include only official USGS or National Park Service URLs if present. Valid domains include 'usgs.gov' (including 'store.usgs.gov') and 'nps.gov'. Ignore third-party or non-official sites.

    Extraction rules:
    - Return each field as a string exactly as stated in the answer, or null if not mentioned.
    - For official_reference_urls, extract all URLs from the answer text that belong to usgs.gov (including store.usgs.gov) or nps.gov domains. Return an empty list if none are present.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_official_domain(url: str) -> bool:
    """Check whether a URL belongs to USGS or NPS domains."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        host = url.lower()
    return ("nps.gov" in host) or ("usgs.gov" in host)


def filter_official_urls(urls: List[str]) -> List[str]:
    """Return only official USGS/NPS urls."""
    return [u for u in urls if is_official_domain(u)]


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_identify_pass(
    evaluator: Evaluator,
    parent_node,
    extracted: SeniorPassExtraction,
) -> None:
    """
    Node group: identify_pass
    Leaf: pass_type (critical) - ensure the pass is named as Senior Lifetime Pass / Lifetime Senior Pass.
    """
    identify_node = evaluator.add_parallel(
        id="identify_pass",
        desc="Correctly identifies the appropriate pass type for a 62-year-old U.S. citizen seeking lifetime access.",
        parent=parent_node,
        critical=True,
    )

    pass_type_leaf = evaluator.add_leaf(
        id="pass_type",
        desc="Names the pass as the Senior Lifetime Pass (a.k.a. Lifetime Senior Pass).",
        parent=identify_node,
        critical=True,
    )

    provided_name = extracted.pass_type or ""
    claim = (
        f"The pass identified in the answer ('{provided_name}') refers to the Senior Lifetime Pass "
        f"(also known as the Lifetime Senior Pass), which is the correct America the Beautiful pass for a 62-year-old "
        f"U.S. citizen seeking lifetime access."
    )
    await evaluator.verify(
        claim=claim,
        node=pass_type_leaf,
        additional_instruction=(
            "Treat synonyms like 'Senior Pass (Lifetime)', 'America the Beautiful Senior Lifetime Pass', and "
            "'Lifetime Senior Pass' as equivalent. The correct choice is the lifetime Senior pass, not the annual Senior pass."
        ),
    )


async def verify_required_information(
    evaluator: Evaluator,
    parent_node,
    extracted: SeniorPassExtraction,
) -> None:
    """
    Node group: provide_required_information - parallel, all critical leaves.
    """
    required_node = evaluator.add_parallel(
        id="provide_required_information",
        desc="Provides all required details (costs, eligibility/documentation, validity, benefits/limitations, scope) and supports them with official USGS/NPS URLs.",
        parent=parent_node,
        critical=True,
    )

    # Prepare sources: filter to official USGS/NPS only
    all_official_urls = filter_official_urls(extracted.official_reference_urls or [])

    # Leaf: official_reference_urls presence (critical)
    official_urls_leaf = evaluator.add_custom_node(
        result=len(all_official_urls) > 0,
        id="official_reference_urls",
        desc="Includes official reference URL(s) from USGS Store and/or National Park Service that support the provided cost, eligibility/documentation, and benefits/limitations information.",
        parent=required_node,
        critical=True,
    )

    # Leaf: base_cost
    base_cost_leaf = evaluator.add_leaf(
        id="base_cost",
        desc="States the exact base cost of the Senior Lifetime Pass is $80.00.",
        parent=required_node,
        critical=True,
    )
    base_cost_value = extracted.base_cost or ""
    base_cost_claim = (
        f"The Senior Lifetime Pass base cost is {base_cost_value}. The base price excludes any online/by-mail processing fees."
    )
    await evaluator.verify(
        claim=base_cost_claim,
        node=base_cost_leaf,
        sources=all_official_urls,
        additional_instruction=(
            "Confirm the base price is exactly $80.00 (pass price itself). Do not include USGS Store processing/document fees in the base cost."
        ),
    )

    # Leaf: online_or_mail_total_cost
    online_mail_leaf = evaluator.add_leaf(
        id="online_or_mail_total_cost",
        desc="States the total cost when purchasing online/by mail through the USGS Store is $90.00, including the $10.00 processing/document fee.",
        parent=required_node,
        critical=True,
    )
    online_mail_value = extracted.online_or_mail_total_cost or ""
    online_mail_claim = (
        f"When purchasing through the USGS Store online or by mail, the total cost is {online_mail_value}, which includes a $10.00 processing/document fee (the pass itself costs $80.00)."
    )
    await evaluator.verify(
        claim=online_mail_claim,
        node=online_mail_leaf,
        sources=all_official_urls,
        additional_instruction=(
            "Verify the total online/by-mail cost equals $90.00 due to the $10 processing/document fee added to the $80 pass price."
        ),
    )

    # Leaf: in_person_total_cost
    in_person_leaf = evaluator.add_leaf(
        id="in_person_total_cost",
        desc="States the total cost when purchasing in person at a Federal recreation site is $80.00 with no processing fee.",
        parent=required_node,
        critical=True,
    )
    in_person_value = extracted.in_person_total_cost or ""
    in_person_claim = (
        f"When purchasing in person at a federal recreation site, the total cost is {in_person_value}, with no processing/document fee added."
    )
    await evaluator.verify(
        claim=in_person_claim,
        node=in_person_leaf,
        sources=all_official_urls,
        additional_instruction=(
            "Confirm in-person purchases at federal recreation sites cost $80.00 and do not add the USGS Store's $10 processing/document fee."
        ),
    )

    # Leaf: eligibility_requirements
    eligibility_leaf = evaluator.add_leaf(
        id="eligibility_requirements",
        desc="States eligibility: available only to U.S. citizens or permanent residents who are age 62+ and must have turned 62 before purchase.",
        parent=required_node,
        critical=True,
    )
    eligibility_text = extracted.eligibility_requirements or ""
    eligibility_claim = (
        f"Eligibility for the Senior Lifetime Pass requires being a U.S. citizen or permanent resident aged 62 or older; "
        f"you must have turned 62 before purchasing the pass. As stated: {eligibility_text}"
    )
    await evaluator.verify(
        claim=eligibility_claim,
        node=eligibility_leaf,
        sources=all_official_urls,
        additional_instruction=(
            "Verify that the Senior pass is restricted to U.S. citizens or permanent residents and requires age 62+ at time of purchase."
        ),
    )

    # Leaf: documentation_requirements
    documentation_leaf = evaluator.add_leaf(
        id="documentation_requirements",
        desc="States documentation requirements to prove eligibility (proof of age and U.S. citizenship or permanent residency), with acceptable examples (e.g., U.S. driver's license, Green Card, U.S. passport).",
        parent=required_node,
        critical=True,
    )
    documentation_text = extracted.documentation_requirements or ""
    documentation_claim = (
        f"Documentation to obtain the Senior Lifetime Pass requires proof of age and proof of U.S. citizenship or permanent residency. "
        f"Acceptable examples include a U.S. driver's license or U.S. passport (for age/citizenship) and a Permanent Resident Card/Green Card (for residency). "
        f"As stated: {documentation_text}"
    )
    await evaluator.verify(
        claim=documentation_claim,
        node=documentation_leaf,
        sources=all_official_urls,
        additional_instruction=(
            "Confirm that proof of age AND proof of U.S. citizenship or permanent residency are required; acceptable examples include driver's license, "
            "U.S. passport, and Permanent Resident Card (Green Card)."
        ),
    )

    # Leaf: lifetime_validity
    lifetime_leaf = evaluator.add_leaf(
        id="lifetime_validity",
        desc="States the pass is valid for the lifetime of the pass holder (does not expire).",
        parent=required_node,
        critical=True,
    )
    lifetime_text = extracted.lifetime_validity or ""
    lifetime_claim = (
        f"The Senior Lifetime Pass is valid for the lifetime of the pass holder and does not expire. As stated: {lifetime_text}"
    )
    await evaluator.verify(
        claim=lifetime_claim,
        node=lifetime_leaf,
        sources=all_official_urls,
        additional_instruction="Verify that the Senior Lifetime Pass does not expire and is valid for the lifetime of the holder.",
    )

    # Leaf: discount_benefits_and_fee_limitations
    discount_leaf = evaluator.add_leaf(
        id="discount_benefits_and_fee_limitations",
        desc="Describes that the pass may provide a 50% discount on some expanded amenity fees (e.g., camping, swimming, boat launching, guided tours) AND notes key limitations (generally does not cover special recreation permit fees or fees charged by concessioners).",
        parent=required_node,
        critical=True,
    )
    discount_text = extracted.discount_benefits_and_fee_limitations or ""
    discount_claim = (
        f"The Senior Lifetime Pass provides a 50% discount on some expanded amenity fees (such as camping, swimming, boat launching, and guided tours), "
        f"and generally does not cover special recreation permit fees or fees charged by concessioners. As stated: {discount_text}"
    )
    await evaluator.verify(
        claim=discount_claim,
        node=discount_leaf,
        sources=all_official_urls,
        additional_instruction=(
            "Confirm both parts: (1) 50% discount applies to some expanded amenity fees (examples listed), and (2) exclusions include special recreation permit fees and concessioner fees."
        ),
    )

    # Leaf: fee_coverage_scope
    scope_leaf = evaluator.add_leaf(
        id="fee_coverage_scope",
        desc="States the pass covers entrance fees and standard amenity (day-use) fees at sites managed by the six specified federal agencies (NPS, FWS, USFS, BLM, BOR, USACE).",
        parent=required_node,
        critical=True,
    )
    scope_text = extracted.fee_coverage_scope or ""
    scope_claim = (
        f"The pass covers entrance fees and standard amenity (day-use) fees at sites managed by these agencies: "
        f"National Park Service (NPS), U.S. Fish and Wildlife Service (FWS), U.S. Forest Service (USFS), Bureau of Land Management (BLM), "
        f"Bureau of Reclamation (BOR), and U.S. Army Corps of Engineers (USACE). As stated: {scope_text}"
    )
    await evaluator.verify(
        claim=scope_claim,
        node=scope_leaf,
        sources=all_official_urls,
        additional_instruction=(
            "Verify that the coverage applies across entrance fees and standard amenity day-use fees and explicitly includes NPS, FWS, USFS, BLM, BOR, and USACE."
        ),
    )

    # Leaf: admission_rules
    admission_leaf = evaluator.add_leaf(
        id="admission_rules",
        desc="States admission rules: admits pass owner and passengers in a non-commercial vehicle at per-vehicle areas OR pass owner plus up to 3 additional adults (max 4 adults total) at per-person areas; children under 16 admitted free.",
        parent=required_node,
        critical=True,
    )
    admission_text = extracted.admission_rules or ""
    admission_claim = (
        f"Admission rules: At per-vehicle fee areas, the pass admits the owner and passengers in a non-commercial vehicle. "
        f"At per-person areas, it admits the pass owner plus up to 3 additional adults (maximum 4 adults total); children under 16 are admitted free. "
        f"As stated: {admission_text}"
    )
    await evaluator.verify(
        claim=admission_claim,
        node=admission_leaf,
        sources=all_official_urls,
        additional_instruction=(
            "Confirm both modes: per-vehicle (owner + passengers in non-commercial vehicle) and per-person (owner + up to 3 other adults, children under 16 admitted free)."
        ),
    )

    # Leaf: physical_card_only
    physical_leaf = evaluator.add_leaf(
        id="physical_card_only",
        desc="States the pass is a physical card only; receipts, photos, confirmation emails, or copies are not valid for entry.",
        parent=required_node,
        critical=True,
    )
    physical_text = extracted.physical_card_only or ""
    physical_claim = (
        f"The pass must be presented as a physical card for entry; receipts, photos, confirmation emails, or photocopies are not valid. "
        f"As stated: {physical_text}"
    )
    await evaluator.verify(
        claim=physical_claim,
        node=physical_leaf,
        sources=all_official_urls,
        additional_instruction=(
            "Confirm that a physical pass card is required for entry and that receipts, images, confirmations, or copies are not accepted as valid passes."
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
    Evaluate an answer for the Senior Lifetime Pass task and return a structured summary.
    """
    # Initialize evaluator (root is non-critical by design; children can be set critical)
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
        prompt=prompt_extract_senior_pass(),
        template_class=SeniorPassExtraction,
        extraction_name="senior_pass_extraction",
    )

    # Build verification tree per rubric
    # 1) Identify the correct pass type
    await verify_identify_pass(evaluator, root, extracted)

    # 2) Provide all required information + verify with official URLs
    await verify_required_information(evaluator, root, extracted)

    # Optional: record filtered official URLs for transparency
    filtered_urls = filter_official_urls(extracted.official_reference_urls or [])
    evaluator.add_custom_info(
        info={"provided_official_urls": filtered_urls},
        info_type="official_urls",
        info_name="official_urls_used_for_verification",
    )

    # Return evaluation summary
    return evaluator.get_summary()