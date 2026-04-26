import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pet_relocation_cdc_aug2024"
TASK_DESCRIPTION = (
    "Identify a pet relocation company that provides services for transporting dogs from Canada to California, United States. "
    "The company must offer comprehensive support for complying with CDC dog import regulations that became effective on August 1, 2024. "
    "Specifically, verify that the identified company provides or facilitates all of the following required services: "
    "(1) Transportation service coverage from Canada to California, "
    "(2) Assistance with completing the CDC Dog Import Form (required for all dogs entering the U.S.), "
    "(3) Microchip verification to ensure dogs have chips detectable by universal scanners, "
    "(4) Coordination with USDA-accredited veterinarians for required certifications, "
    "(5) Assistance with obtaining health certificates (Certificate of Veterinary Inspection), "
    "(6) Verification that dogs meet the minimum age requirement of 6 months, and "
    "(7) Management of rabies vaccination documentation, ensuring the 28-day minimum waiting period before travel is observed. "
    "Provide the company name, website URL, and specific evidence demonstrating how the company meets each of these seven critical requirements."
)


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class CompanyComplianceExtraction(BaseModel):
    # Company identification
    company_name: Optional[str] = None
    company_website: Optional[str] = None

    # Evidence URLs explicitly cited in the answer for each requirement
    route_service_urls: List[str] = Field(default_factory=list)
    cdc_form_urls: List[str] = Field(default_factory=list)
    cdc_form_timing_urls: List[str] = Field(default_factory=list)

    microchip_presence_urls: List[str] = Field(default_factory=list)
    microchip_iso_urls: List[str] = Field(default_factory=list)
    microchip_timing_urls: List[str] = Field(default_factory=list)

    usda_vet_urls: List[str] = Field(default_factory=list)

    health_cert_urls: List[str] = Field(default_factory=list)
    health_cert_timing_urls: List[str] = Field(default_factory=list)

    age_requirement_urls: List[str] = Field(default_factory=list)

    rabies_docs_urls: List[str] = Field(default_factory=list)
    rabies_waiting_urls: List[str] = Field(default_factory=list)

    # Additional non-critical compliance
    high_risk_urls: List[str] = Field(default_factory=list)
    interstate_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_company_and_evidence() -> str:
    return """
Extract the pet relocation company and all evidence URLs explicitly cited in the answer that support the company's compliance with CDC dog import rules (effective August 1, 2024). If multiple companies are mentioned, select the primary or first company that is actually proposed to handle a Canada-to-California relocation.

Return:
- company_name: The company's name (string).
- company_website: The main website URL for the company (string). If multiple websites are present, choose the primary/homepage.
- route_service_urls: URLs that show service coverage for transporting dogs from Canada to the United States (and/or explicitly to California or nationwide US coverage).
- cdc_form_urls: URLs that show the company assists with completing or submitting the CDC Dog Import Form (for dogs entering the U.S.).
- cdc_form_timing_urls: URLs that show the company provides guidance on when to submit the CDC Dog Import Form (submission window relative to travel date).
- microchip_presence_urls: URLs that show the company verifies or requires dogs to have a microchip for U.S. entry.
- microchip_iso_urls: URLs that show the company verifies the microchip is ISO-compatible and/or readable by a universal scanner.
- microchip_timing_urls: URLs that show the company verifies the microchip is implanted before the most recent rabies vaccination.
- usda_vet_urls: URLs that show the company coordinates with or provides access to USDA-accredited veterinarians for certifications/endorsements.
- health_cert_urls: URLs that show the company assists with obtaining a health certificate (Certificate of Veterinary Inspection / CVI).
- health_cert_timing_urls: URLs that show the company manages the timing for health certificates (e.g., typically within 10 days of travel per airline/import requirements).
- age_requirement_urls: URLs that show the company verifies dogs meet the minimum 6-month age requirement for U.S. entry under CDC rules effective Aug 1, 2024.
- rabies_docs_urls: URLs that show the company manages/handles documentation of rabies vaccination certificates.
- rabies_waiting_urls: URLs that show the company ensures the minimum 28-day waiting period after rabies vaccination before travel.
- high_risk_urls: URLs (if any) that show the company can facilitate high-risk country requirements (e.g., 'Certification of U.S.-Issued Rabies Vaccination' and Ministry of Agriculture endorsement).
- interstate_urls: URLs (if any) that show the company ensures compliance with interstate transport requirements (CVI for movement from port of entry to California).

Important:
- Extract only URLs that are explicitly present in the answer (plain or markdown links). Do not invent or infer URLs.
- If a particular category has no URLs cited in the answer, return an empty list for that category.
- Normalize URLs to include http/https if missing.
"""


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
def _sources_or_fallback(urls: List[str], fallback: Optional[str]) -> List[str]:
    """Use provided URLs if present; otherwise fall back to the main company website if available."""
    if urls and len(urls) > 0:
        # Deduplicate while preserving order
        seen = set()
        out: List[str] = []
        for u in urls:
            if u and u not in seen:
                out.append(u)
                seen.add(u)
        return out
    return [fallback] if (fallback and fallback.strip()) else []


def _add_leaf_and_prepare(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    sources: List[str],
    additional_instruction: str,
    critical: bool = True,
):
    """Create a leaf and return tuple for batch verification."""
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )
    return (claim, sources, leaf, additional_instruction)


# --------------------------------------------------------------------------- #
# Verification Logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_core_compliance(
    evaluator: Evaluator,
    root,
    data: CompanyComplianceExtraction,
) -> None:
    """
    Build the core compliance verification subtree (critical) and run verifications.
    """
    company = data.company_name or "the company"
    base_url = data.company_website or None

    # Critical aggregator for all required/expanded checks
    core_node = evaluator.add_parallel(
        id="core_regulatory_compliance",
        desc="All critical CDC dog import compliance requirements (effective Aug 1, 2024) are met by the identified company",
        parent=root,
        critical=True
    )

    batch_items: List[tuple] = []

    # 1) Geographic Service Coverage: Canada to California/US
    batch_items.append(_add_leaf_and_prepare(
        evaluator=evaluator,
        parent_node=core_node,
        node_id="service_coverage_geographic",
        desc="The company explicitly offers pet relocation services from Canada to California or serves this specific international-to-US route",
        claim=f"{company} offers international pet relocation services from Canada to the United States and can transport dogs to California (or to any U.S. state including California).",
        sources=_sources_or_fallback(data.route_service_urls, base_url),
        additional_instruction="Accept language like 'Canada to the U.S.' or 'to anywhere in the United States' or 'nationwide US coverage'. The evidence must be on the company's own site or a directly cited page."
    ))

    # 2) Assistance with completing the CDC Dog Import Form
    batch_items.append(_add_leaf_and_prepare(
        evaluator, core_node, "cdc_import_form_completion",
        "The company provides assistance with completing the CDC Dog Import Form online, which is required for all dogs entering the United States",
        claim=f"{company} assists clients with completing or submitting the CDC Dog Import Form required for dogs entering the United States.",
        sources=_sources_or_fallback(data.cdc_form_urls, base_url),
        additional_instruction="Allow synonyms such as 'CDC dog entry form', 'CDC dog import portal/form', or 'CDC import paperwork assistance'. The page should indicate help with the CDC dog import form."
    ))

    # 3) CDC Dog Import Form timing guidance
    batch_items.append(_add_leaf_and_prepare(
        evaluator, core_node, "cdc_import_form_timing",
        "The company guides on proper submission timing of the CDC Dog Import Form (2-10 days before travel for re-entry, up to 6 months in advance)",
        claim=f"{company} provides guidance on when to submit the CDC Dog Import Form, specifying a submission window relative to the travel date (e.g., a few days before travel or up to several months in advance).",
        sources=_sources_or_fallback(data.cdc_form_timing_urls, base_url),
        additional_instruction="Accept reasonable timing guidance; wording may vary (e.g., 'submit several days before travel', 'up to 6 months before'). The check is about presence of timing guidance, not exact numbers."
    ))

    # 4) Microchip presence verification
    batch_items.append(_add_leaf_and_prepare(
        evaluator, core_node, "microchip_presence_verification",
        "The company verifies that dogs have microchips implanted as required for US entry",
        claim=f"{company} verifies or requires that dogs are microchipped for U.S. entry.",
        sources=_sources_or_fallback(data.microchip_presence_urls, base_url),
        additional_instruction="The page should mention microchip requirement or verification as part of the service or pre-travel checklist."
    ))

    # 5) Microchip ISO compatibility (universal scanner)
    batch_items.append(_add_leaf_and_prepare(
        evaluator, core_node, "microchip_iso_compatibility",
        "The company verifies that the microchip is ISO-compatible and detectable by universal scanners as mandated by CDC regulations",
        claim=f"{company} verifies that the dog's microchip is ISO-compatible (e.g., ISO 11784/11785) or readable by a universal scanner.",
        sources=_sources_or_fallback(data.microchip_iso_urls, base_url),
        additional_instruction="Accept explicit mention of 'ISO-compatible', 'ISO 11784/11785', or 'readable by universal scanner'."
    ))

    # 6) Microchip timing (before rabies vaccination)
    batch_items.append(_add_leaf_and_prepare(
        evaluator, core_node, "microchip_timing_verification",
        "The company verifies that the microchip was implanted before the dog's most recent rabies vaccination, as required by regulations",
        claim=f"{company} verifies that the microchip must be implanted before the dog's most recent rabies vaccination.",
        sources=_sources_or_fallback(data.microchip_timing_urls, base_url),
        additional_instruction="Look for language requiring the microchip to be implanted prior to rabies vaccination."
    ))

    # 7) USDA-accredited veterinarian coordination
    batch_items.append(_add_leaf_and_prepare(
        evaluator, core_node, "usda_veterinarian_access",
        "The company coordinates appointments with or provides access to USDA-accredited veterinarians for required certifications",
        claim=f"{company} coordinates with or provides access to USDA-accredited veterinarians for required documentation or endorsements.",
        sources=_sources_or_fallback(data.usda_vet_urls, base_url),
        additional_instruction="Accept references to 'USDA-accredited vet', 'APHIS endorsement assistance', 'USDA endorsement', or similar."
    ))

    # 8) Health certificate (CVI) assistance
    batch_items.append(_add_leaf_and_prepare(
        evaluator, core_node, "health_certificate_obtaining",
        "The company assists with obtaining health certificates (Certificate of Veterinary Inspection) from licensed veterinarians",
        claim=f"{company} assists with obtaining a health certificate (Certificate of Veterinary Inspection/CVI) from a licensed veterinarian.",
        sources=_sources_or_fallback(data.health_cert_urls, base_url),
        additional_instruction="Accept 'health certificate', 'CVI', 'certificate of veterinary inspection', or similar phrasing."
    ))

    # 9) Health certificate timing (e.g., within ~10 days)
    batch_items.append(_add_leaf_and_prepare(
        evaluator, core_node, "health_certificate_timing",
        "The company manages the timing to ensure health certificates are completed within the required timeframe (typically within 10 days of travel for airline requirements)",
        claim=f"{company} provides guidance or management to ensure the health certificate is completed within the required timeframe around the travel date (commonly within about 10 days).",
        sources=_sources_or_fallback(data.health_cert_timing_urls, base_url),
        additional_instruction="Accept mentions of airline or import timing windows (e.g., 10 days) for validity of CVI/health certificate."
    ))

    # 10) Age requirement (>= 6 months)
    batch_items.append(_add_leaf_and_prepare(
        evaluator, core_node, "age_requirement_verification",
        "The company verifies that dogs meet the minimum age requirement of 6 months as mandated by CDC regulations effective August 1, 2024",
        claim=f"{company} verifies that dogs must be at least 6 months old to enter the United States under CDC rules effective August 1, 2024.",
        sources=_sources_or_fallback(data.age_requirement_urls, base_url),
        additional_instruction="Accept explicit mention of 'at least 6 months old' or equivalent wording."
    ))

    # 11) Rabies documentation management
    batch_items.append(_add_leaf_and_prepare(
        evaluator, core_node, "rabies_vaccination_documentation",
        "The company handles documentation and verification of rabies vaccination certificates",
        claim=f"{company} manages, verifies, or assists with documentation of rabies vaccination certificates.",
        sources=_sources_or_fallback(data.rabies_docs_urls, base_url),
        additional_instruction="Accept mentions of 'rabies certificate', 'proof of rabies vaccination', or similar."
    ))

    # 12) Rabies waiting period (>= 28 days)
    batch_items.append(_add_leaf_and_prepare(
        evaluator, core_node, "rabies_waiting_period_compliance",
        "The company ensures compliance with the 28-day minimum waiting period between rabies vaccination and travel (30 days before titer test for high-risk countries)",
        claim=f"{company} ensures compliance with a minimum waiting period of at least 28 days after rabies vaccination before travel.",
        sources=_sources_or_fallback(data.rabies_waiting_urls, base_url),
        additional_instruction="Accept wording such as '28 days', 'four weeks', or 'approximately 30 days'; the key is a minimum waiting period after vaccination before travel."
    ))

    # Execute batch verification for all core critical claims
    await evaluator.batch_verify(batch_items)


async def build_and_verify_additional(
    evaluator: Evaluator,
    root,
    data: CompanyComplianceExtraction,
) -> None:
    """
    Build the additional (non-critical) compliance subtree and run verifications.
    """
    company = data.company_name or "the company"
    base_url = data.company_website or None

    additional_node = evaluator.add_parallel(
        id="additional_compliance",
        desc="Additional non-critical compliance assurances (nice-to-have)",
        parent=root,
        critical=False
    )

    batch_items: List[tuple] = []

    # A) High-risk country compliance facilitation
    batch_items.append(_add_leaf_and_prepare(
        evaluator, additional_node, "high_risk_country_compliance",
        "For dogs from high-risk rabies countries, the company facilitates obtaining the 'Certification of U.S.-Issued Rabies Vaccination' form and Ministry of Agriculture endorsement when applicable",
        claim=f"For dogs from CDC-designated high-risk rabies countries, {company} can facilitate additional documentation such as the 'Certification of U.S.-Issued Rabies Vaccination' and Ministry of Agriculture endorsement (when applicable).",
        sources=_sources_or_fallback(data.high_risk_urls, base_url),
        additional_instruction="Accept language indicating capability to handle high-risk CDC requirements/forms and MOA endorsements."
    ))

    # B) Interstate transport compliance to California
    batch_items.append(_add_leaf_and_prepare(
        evaluator, additional_node, "interstate_transport_compliance",
        "The company ensures compliance with interstate transport requirements, including proper CVI documentation for crossing state lines from port of entry to California",
        claim=f"{company} ensures compliance with interstate transport requirements in the U.S., including obtaining a CVI for movement from the port of entry to California.",
        sources=_sources_or_fallback(data.interstate_urls, base_url),
        additional_instruction="Accept mentions of 'interstate CVI', 'interstate health certificate', or equivalent; look for movement between states compliance."
    ))

    await evaluator.batch_verify(batch_items)


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
    Evaluate an answer for the pet relocation CDC Aug 2024 compliance task and return a structured result dictionary.
    """
    # Initialize evaluator (root node kept non-critical to allow partial credit; critical gating handled in core subtree)
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

    # Extract structured company info and evidence URLs from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_company_and_evidence(),
        template_class=CompanyComplianceExtraction,
        extraction_name="company_and_evidence"
    )

    # Build and verify core (critical) requirements
    await build_and_verify_core_compliance(evaluator, root, extraction)

    # Build and verify additional (non-critical) compliance
    await build_and_verify_additional(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()