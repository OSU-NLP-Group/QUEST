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
TASK_ID = "ca_wildlife_rehab_facility_compliance"
TASK_DESCRIPTION = """
Identify a wildlife rehabilitation facility in California that meets all of the following requirements:

1. The facility must be located in California and hold a valid California Department of Fish and Wildlife rehabilitation permit (Section 679).

2. The facility must hold (or have staff who hold) a valid federal migratory bird rehabilitation permit issued under 50 CFR 21.76 by the U.S. Fish and Wildlife Service.

3. The facility must operate as a nonprofit organization with 501(c)(3) tax-exempt status.

4. The facility must employ at least one licensed veterinarian (DVM) on staff.

5. The facility must comply with California's continuing education requirement of 8 hours annually as mandated by California Code of Regulations Title 14, Section 679.4(3).

6. The facility must maintain enclosures that meet minimum size requirements for the species housed, following NWRA/IWRC Minimum Standards for Wildlife Rehabilitation or California's Section 679.4 facility standards.

7. The facility must make Form 990 financial documents available to the public as required by law.

8. The facility must rehabilitate native California wildlife including both birds (such as raptors or migratory birds) and mammals.

Provide the name of the facility, its location in California, and URL references that verify each of these requirements.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityExtraction(BaseModel):
    facility_name: Optional[str] = None
    facility_location: Optional[str] = None
    facility_website_url: Optional[str] = None
    veterinarian_name: Optional[str] = None

    # Dedicated URL lists for each verification requirement
    location_urls: List[str] = Field(default_factory=list)
    state_permit_urls: List[str] = Field(default_factory=list)
    ce_urls: List[str] = Field(default_factory=list)
    federal_bird_permit_urls: List[str] = Field(default_factory=list)
    nonprofit_501c3_urls: List[str] = Field(default_factory=list)
    form_990_urls: List[str] = Field(default_factory=list)
    vet_staff_urls: List[str] = Field(default_factory=list)
    vet_license_urls: List[str] = Field(default_factory=list)
    enclosure_standard_urls: List[str] = Field(default_factory=list)
    birds_urls: List[str] = Field(default_factory=list)
    mammals_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_info() -> str:
    return """
    Extract the single facility proposed in the answer and all URLs that the answer cites to support each required verification item.
    If multiple facilities are mentioned, extract only the first one that is used to satisfy the requirements.

    Return a JSON object with the following fields:
    - facility_name: The full name of the wildlife rehabilitation facility.
    - facility_location: The stated location (city and/or address and state) of the facility as written in the answer.
    - facility_website_url: The primary website URL of the facility, if given.
    - veterinarian_name: The full name of a staff veterinarian (DVM) if explicitly mentioned; otherwise null.

    For each of the following, extract ALL URLs explicitly included in the answer text that support the item. If none are provided for an item, return an empty list for that item.
    - location_urls: URLs that show the facility's location/address in California (e.g., contact/about page).
    - state_permit_urls: URLs evidencing a valid California Department of Fish and Wildlife rehabilitation permit (Title 14 CCR §679).
    - ce_urls: URLs evidencing compliance with California's 8-hour annual continuing education requirement (CCR Title 14 §679.4(3)).
    - federal_bird_permit_urls: URLs evidencing a valid federal migratory bird rehabilitation permit (50 CFR 21.76), either facility-level or individual staff.
    - nonprofit_501c3_urls: URLs evidencing 501(c)(3) tax-exempt status (e.g., IRS, GuideStar/Candid, Charity Navigator, ProPublica Nonprofit Explorer, or official statement).
    - form_990_urls: URLs that provide public access to IRS Form 990 filings or explicitly state that Form 990s are publicly available.
    - vet_staff_urls: URLs that identify at least one licensed veterinarian (DVM) on staff at the facility (e.g., staff/bio page).
    - vet_license_urls: URLs for verifying the California veterinary license of a staff DVM (e.g., California VMB license lookup page for the named vet).
    - enclosure_standard_urls: URLs showing that enclosures meet NWRA/IWRC Minimum Standards or California CCR §679.4 facility standards (species-specific minimum sizes).
    - birds_urls: URLs that demonstrate the facility rehabilitates native California birds (e.g., raptors, songbirds, migratory birds).
    - mammals_urls: URLs that demonstrate the facility rehabilitates native California mammals.

    Rules:
    - Extract only URLs explicitly present in the answer; do not invent or infer URLs.
    - If a URL is in markdown format ([text](url)), extract the actual URL.
    - If a URL is missing a protocol, prepend http:// to make it a valid URL.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def _has_urls(urls: Optional[List[str]]) -> bool:
    return len(_clean_urls(urls)) > 0


# --------------------------------------------------------------------------- #
# Generic builders for sequential requirement checks                          #
# --------------------------------------------------------------------------- #
async def add_presence_then_verify_seq(
    evaluator: Evaluator,
    parent,
    base_id: str,
    requirement_desc: str,
    presence_desc: str,
    evidence_desc: str,
    claim: str,
    urls: List[str],
    *,
    critical: bool,
    additional_instruction: str,
) -> None:
    """
    Build a sequential requirement node:
    1) presence (custom existence check for URLs)
    2) evidence verification via provided URLs

    If 'critical' is True, the parent and all children will be critical to satisfy framework constraints.
    """
    seq_node = evaluator.add_sequential(
        id=base_id,
        desc=requirement_desc,
        parent=parent,
        critical=critical,
    )

    # 1) Presence of supporting URL(s)
    evaluator.add_custom_node(
        result=_has_urls(urls),
        id=f"{base_id}_reference_url",
        desc=presence_desc,
        parent=seq_node,
        critical=True if critical else True  # child can be critical; parent may be non-critical
    )

    # 2) Evidence verification (will auto-skip if presence failed due to sequential precondition)
    evidence_node = evaluator.add_leaf(
        id=f"{base_id}_evidence",
        desc=evidence_desc,
        parent=seq_node,
        critical=True if critical else True
    )
    await evaluator.verify(
        claim=claim,
        node=evidence_node,
        sources=_clean_urls(urls),
        additional_instruction=additional_instruction
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
    Evaluate an answer for the California wildlife rehabilitation facility compliance task.
    """
    # Initialize evaluator (root is non-critical parallel aggregator by design)
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facility_info(),
        template_class=FacilityExtraction,
        extraction_name="facility_extraction",
    )

    # Add ground truth expectations (rubric checklist)
    evaluator.add_ground_truth({
        "required_items": [
            "Located in California",
            "Valid CA wildlife rehabilitation permit (Title 14 CCR §679)",
            "Federal migratory bird rehabilitation permit (50 CFR 21.76) – facility or staff",
            "501(c)(3) nonprofit tax-exempt status",
            "At least one licensed veterinarian (DVM) on staff",
            "8 hours annual continuing education (CCR Title 14 §679.4(3))",
            "Enclosures meet NWRA/IWRC or CCR §679.4 standards (minimum sizes)",
            "Form 990s publicly available",
            "Rehabilitates birds (e.g., raptors/migratory birds) and mammals",
        ]
    })

    # ------------------------------------------------------------------ #
    # 1) Location in California (critical)
    # ------------------------------------------------------------------ #
    await add_presence_then_verify_seq(
        evaluator=evaluator,
        parent=root,
        base_id="location_verification",
        requirement_desc="Facility is physically located in California",
        presence_desc="URL reference(s) present for California location verification",
        evidence_desc="Evidence confirms the facility is located in California",
        claim=(
            f"This page shows that the facility"
            f"{f' named {extracted.facility_name}' if extracted.facility_name else ''}"
            f" is located in California (CA)."
        ),
        urls=extracted.location_urls,
        critical=True,
        additional_instruction=(
            "Accept if the page shows an address or location in California (e.g., 'CA' or 'California' with a city/ZIP). "
            "Pages like 'Contact' or 'About' on the official website are acceptable."
        ),
    )

    # ------------------------------------------------------------------ #
    # 2) California state compliance (critical group)
    #     - CA permit (Section 679)
    #     - Continuing education (8 hours annually, CCR 679.4(3))
    # ------------------------------------------------------------------ #
    state_compliance = evaluator.add_parallel(
        id="state_compliance",
        desc="Facility meets California state regulatory requirements",
        parent=root,
        critical=True,
    )

    # 2.1) CA rehabilitation permit (Section 679) – critical
    await add_presence_then_verify_seq(
        evaluator=evaluator,
        parent=state_compliance,
        base_id="state_permit",
        requirement_desc="Facility holds valid California wildlife rehabilitation permit under Section 679",
        presence_desc="URL reference for state permit verification is provided",
        evidence_desc="Evidence confirms active California rehabilitation permit (Title 14 CCR §679)",
        claim=(
            f"This page confirms that"
            f" {extracted.facility_name} holds an active California Department of Fish and Wildlife wildlife "
            f"rehabilitation permit under Title 14 CCR Section 679."
            if extracted.facility_name
            else "This page confirms an active California Department of Fish and Wildlife wildlife rehabilitation permit under Title 14 CCR Section 679 for the facility."
        ),
        urls=extracted.state_permit_urls,
        critical=True,
        additional_instruction=(
            "Accept if the facility (or its legal name/DBA) appears on an official CDFW permittee list or in an official permit/letter. "
            "Synonyms such as 'Wildlife Rehabilitation Permit' are acceptable. Screenshots or PDFs from .gov domains are strong evidence."
        ),
    )

    # 2.2) Continuing education (8 hours/year) – critical
    await add_presence_then_verify_seq(
        evaluator=evaluator,
        parent=state_compliance,
        base_id="continuing_education",
        requirement_desc="Facility demonstrates compliance with California's 8-hour annual continuing education requirement",
        presence_desc="URL reference for continuing education compliance is provided",
        evidence_desc="Evidence indicates compliance with CCR Title 14, Section 679.4(3) (8 hours annually)",
        claim=(
            "This page provides evidence that facility personnel complete at least 8 hours of continuing education annually, "
            "as required by CCR Title 14, Section 679.4(3)."
        ),
        urls=extracted.ce_urls,
        critical=True,
        additional_instruction=(
            "Look for explicit mention of '8 hours' of CE annually, training policy statements, or official documentation "
            "that staff meet the continuing education requirement. It can be a policy page, handbook, or regulatory compliance statement."
        ),
    )

    # ------------------------------------------------------------------ #
    # 3) Federal compliance – Migratory Bird Rehab Permit (critical)
    # ------------------------------------------------------------------ #
    federal_compliance = evaluator.add_parallel(
        id="federal_compliance",
        desc="Facility meets federal regulatory requirements",
        parent=root,
        critical=True,
    )

    await add_presence_then_verify_seq(
        evaluator=evaluator,
        parent=federal_compliance,
        base_id="migratory_bird_permit",
        requirement_desc="Facility/staff holds a valid federal migratory bird rehabilitation permit",
        presence_desc="URL reference for federal permit verification is provided",
        evidence_desc="Evidence confirms a federal Migratory Bird Rehabilitation permit under 50 CFR 21.76",
        claim=(
            f"This page confirms that the facility or its staff holds a federal Migratory Bird Rehabilitation permit "
            f"under 50 CFR 21.76."
        ),
        urls=extracted.federal_bird_permit_urls,
        critical=True,
        additional_instruction=(
            "Accept if the facility or named staff is listed as holding a migratory bird rehabilitation permit on a USFWS or authoritative list. "
            "Mentions of '50 CFR 21.76' or recognized permit numbers/names strengthen the evidence."
        ),
    )

    # ------------------------------------------------------------------ #
    # 4) Organizational status (critical)
    #     - 501(c)(3) status
    #     - Form 990 public availability
    # ------------------------------------------------------------------ #
    organizational_status = evaluator.add_parallel(
        id="organizational_status",
        desc="Facility operates as a qualified nonprofit organization",
        parent=root,
        critical=True,
    )

    # 4.1) Nonprofit 501(c)(3) – critical
    await add_presence_then_verify_seq(
        evaluator=evaluator,
        parent=organizational_status,
        base_id="nonprofit_status",
        requirement_desc="Facility operates as 501(c)(3) tax-exempt nonprofit",
        presence_desc="URL reference for nonprofit status verification is provided",
        evidence_desc="Evidence confirms 501(c)(3) status",
        claim=(
            f"This page confirms that {extracted.facility_name} is a 501(c)(3) tax-exempt nonprofit organization."
            if extracted.facility_name
            else "This page confirms that the facility is a 501(c)(3) tax-exempt nonprofit organization."
        ),
        urls=extracted.nonprofit_501c3_urls,
        critical=True,
        additional_instruction=(
            "Accept sources such as IRS Pub 78 listings, IRS determination letters, GuideStar/Candid, Charity Navigator, "
            "ProPublica Nonprofit Explorer, or an official statement on the facility website clearly indicating 501(c)(3) status."
        ),
    )

    # 4.2) Form 990 public availability – critical
    await add_presence_then_verify_seq(
        evaluator=evaluator,
        parent=organizational_status,
        base_id="financial_transparency",
        requirement_desc="Facility provides required financial transparency",
        presence_desc="URL reference to Form 990 (or explicit public availability statement) is provided",
        evidence_desc="Evidence confirms that IRS Form 990 filings are publicly available",
        claim=(
            "This page provides public access to the facility's IRS Form 990 filings or explicitly states that Form 990s are available to the public."
        ),
        urls=extracted.form_990_urls,
        critical=True,
        additional_instruction=(
            "Accept if actual Form 990 PDFs or listings are accessible (e.g., ProPublica Nonprofit Explorer, GuideStar/Candid), "
            "or an explicit statement that Form 990s are publicly available upon request."
        ),
    )

    # ------------------------------------------------------------------ #
    # 5) Professional staffing (critical + optional)
    #     - At least one staff veterinarian (critical)
    #     - Vet license verification (optional / non-critical)
    # ------------------------------------------------------------------ #
    # 5.1) Staff veterinarian (DVM) – critical
    await add_presence_then_verify_seq(
        evaluator=evaluator,
        parent=root,
        base_id="veterinarian_employment",
        requirement_desc="Facility employs at least one licensed veterinarian (DVM)",
        presence_desc="URL reference for veterinarian employment verification is provided",
        evidence_desc="Evidence identifies at least one DVM as a staff member",
        claim=(
            f"This page identifies at least one licensed veterinarian (DVM){f' (e.g., Dr. {extracted.veterinarian_name})' if extracted.veterinarian_name else ''} on staff at the facility."
        ),
        urls=extracted.vet_staff_urls,
        critical=True,
        additional_instruction=(
            "Accept if a staff page, biography, or organizational listing shows a DVM employed on staff. "
            "Titles like 'Veterinarian', 'Medical Director (DVM)', or 'Staff Veterinarian' are acceptable."
        ),
    )

    # 5.2) Veterinarian license verification – non-critical (bonus)
    # Build a 3-step sequential node: vet name -> license URL presence -> license verification
    vet_license_seq = evaluator.add_sequential(
        id="veterinarian_licensing",
        desc="Employed veterinarian holds valid California veterinary license (optional bonus)",
        parent=root,
        critical=False,
    )
    # Step A: Ensure we have a vet name to check
    evaluator.add_custom_node(
        result=bool(extracted.veterinarian_name and extracted.veterinarian_name.strip()),
        id="license_vet_name_present",
        desc="Veterinarian name is available for license verification",
        parent=vet_license_seq,
        critical=True
    )
    # Step B: License URL(s) provided
    evaluator.add_custom_node(
        result=_has_urls(extracted.vet_license_urls),
        id="license_reference_url",
        desc="URL reference for veterinary license verification is provided",
        parent=vet_license_seq,
        critical=True
    )
    # Step C: Verify active CA license for the named DVM
    license_check_node = evaluator.add_leaf(
        id="license_verification",
        desc="Evidence of valid California DVM license",
        parent=vet_license_seq,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"This page shows that {extracted.veterinarian_name} holds an active California veterinary (DVM) license."
            if extracted.veterinarian_name
            else "This page shows that the named veterinarian holds an active California veterinary (DVM) license."
        ),
        node=license_check_node,
        sources=_clean_urls(extracted.vet_license_urls),
        additional_instruction=(
            "Prefer official California Veterinary Medical Board (VMB) license lookup or similarly authoritative sources. "
            "Accept if the status is 'active' or equivalent. If the page clearly indicates lapsed/inactive, it should fail."
        ),
    )

    # ------------------------------------------------------------------ #
    # 6) Facility standards (critical)
    #     - Enclosure compliance
    #     - Species scope (birds AND mammals)
    # ------------------------------------------------------------------ #
    facility_standards = evaluator.add_parallel(
        id="facility_standards",
        desc="Physical facility meets regulatory standards",
        parent=root,
        critical=True,
    )

    # 6.1) Enclosure compliance – critical
    await add_presence_then_verify_seq(
        evaluator=evaluator,
        parent=facility_standards,
        base_id="enclosure_compliance",
        requirement_desc="Facility maintains enclosures meeting species-specific minimum size requirements",
        presence_desc="URL reference for enclosure standards compliance is provided",
        evidence_desc="Evidence indicates compliance with NWRA/IWRC Minimum Standards or CCR §679.4",
        claim=(
            "This page indicates that the facility's enclosures meet species-specific minimum size requirements in compliance "
            "with NWRA/IWRC Minimum Standards for Wildlife Rehabilitation or California CCR Title 14 §679.4."
        ),
        urls=extracted.enclosure_standard_urls,
        critical=True,
        additional_instruction=(
            "Accept explicit statements that the facility follows NWRA/IWRC Minimum Standards, CCR §679.4, or equivalent documented "
            "standards for species-specific enclosure sizes. Policies, accreditation criteria, or facility standards pages are acceptable."
        ),
    )

    # 6.2) Species scope – birds and mammals (both critical)
    species_scope = evaluator.add_parallel(
        id="species_scope",
        desc="Facility rehabilitates both birds and mammals",
        parent=facility_standards,
        critical=True,
    )

    # Birds – critical
    await add_presence_then_verify_seq(
        evaluator=evaluator,
        parent=species_scope,
        base_id="birds_rehabilitation",
        requirement_desc="Facility rehabilitates native California birds (e.g., raptors or migratory birds)",
        presence_desc="URL reference for bird rehabilitation is provided",
        evidence_desc="Evidence confirms bird rehabilitation services",
        claim=(
            "This page shows that the facility rehabilitates native California birds, such as raptors, songbirds, or other migratory birds."
        ),
        urls=extracted.birds_urls,
        critical=True,
        additional_instruction=(
            "Accept service/species lists, patient intake pages, or program descriptions that include birds (e.g., raptors, hawks, owls, songbirds, seabirds)."
        ),
    )

    # Mammals – critical
    await add_presence_then_verify_seq(
        evaluator=evaluator,
        parent=species_scope,
        base_id="mammals_rehabilitation",
        requirement_desc="Facility rehabilitates native California land mammals",
        presence_desc="URL reference for mammal rehabilitation is provided",
        evidence_desc="Evidence confirms mammal rehabilitation services",
        claim=(
            "This page shows that the facility rehabilitates native California mammals (e.g., raccoons, squirrels, foxes, coyotes, opossums, bats, etc.)."
        ),
        urls=extracted.mammals_urls,
        critical=True,
        additional_instruction=(
            "Accept service/species lists, patient intake pages, or program descriptions that include mammals."
        ),
    )

    # ------------------------------------------------------------------ #
    # Return summary                                                      #
    # ------------------------------------------------------------------ #
    return evaluator.get_summary()