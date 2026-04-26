import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "ca_hospital_multi_criteria"
TASK_DESCRIPTION = """
Identify three hospitals located in California that simultaneously meet all of the following criteria:

1. Trauma Center Designation: The hospital must be verified as a Level I or Level II trauma center by the American College of Surgeons (ACS) or designated by the State of California.

2. Stroke Center Certification: The hospital must hold either Primary Stroke Center or Comprehensive Stroke Center certification from the Joint Commission, DNV, or another equivalent recognized accrediting body.

3. Teaching Hospital Status: The hospital must have at least one ACGME-accredited residency program, demonstrating its role as a teaching hospital.

4. Magnet Recognition: The hospital must currently hold ANCC Magnet Recognition for nursing excellence.

5. NICU Level: The hospital must have a Level III or Level IV Neonatal Intensive Care Unit (NICU).

6. Hospital Size: The hospital must have at least 200 licensed inpatient beds.

7. CMS Quality Rating: The hospital must have a CMS Overall Hospital Quality Star Rating of 3 stars or higher.

8. Current Accreditation: The hospital must maintain current Joint Commission or DNV hospital accreditation.

For each of the three hospitals, provide:
- The official hospital name
- Complete physical address (including street, city, and state)
- Official hospital website URL
- For each of the eight criteria listed above, provide:
  - Verification that the hospital meets the requirement
  - A reference URL from an official or authoritative source that confirms this designation/certification

All information must be verifiable through official hospital websites, accrediting body websites, government databases, or other authoritative public sources.
"""


# ---------------------------- Data Models ---------------------------------- #
class HospitalInfo(BaseModel):
    # Basic info
    name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    website_url: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    # Criteria-specific fields and source urls
    trauma_level: Optional[str] = None  # e.g., "Level I", "Level II"
    trauma_urls: List[str] = Field(default_factory=list)

    stroke_certification: Optional[str] = None  # e.g., "Primary Stroke Center", "Comprehensive Stroke Center"
    stroke_accreditor: Optional[str] = None  # e.g., "The Joint Commission", "DNV"
    stroke_urls: List[str] = Field(default_factory=list)

    acgme_programs: List[str] = Field(default_factory=list)  # program names if listed
    teaching_urls: List[str] = Field(default_factory=list)  # ACGME directory, sponsoring inst, official

    magnet_status: Optional[str] = None  # e.g., "ANCC Magnet Recognized"
    magnet_urls: List[str] = Field(default_factory=list)

    nicu_level: Optional[str] = None  # e.g., "Level III", "Level IV"
    nicu_urls: List[str] = Field(default_factory=list)

    beds_text: Optional[str] = None  # e.g., "450 licensed beds"
    bed_urls: List[str] = Field(default_factory=list)

    cms_stars: Optional[str] = None  # e.g., "4 stars"
    cms_urls: List[str] = Field(default_factory=list)

    accreditation_body: Optional[str] = None  # e.g., "Joint Commission", "DNV"
    accreditation_urls: List[str] = Field(default_factory=list)


class HospitalsExtraction(BaseModel):
    hospitals: List[HospitalInfo] = Field(default_factory=list)


# -------------------------- Extraction Prompt ------------------------------ #
def prompt_extract_hospitals() -> str:
    return """
    Extract up to three hospitals listed in the answer that are located in California (CA). For each hospital, return the following fields exactly as presented in the answer:

    BASIC IDENTIFICATION (required if present):
    - name: Official hospital name
    - street_address: Street address line
    - city: City name
    - state: State (e.g., "California" or "CA")
    - website_url: Official hospital website URL
    - location_urls: A list of URLs that the answer cites as evidence for the hospital's location/address (often the hospital website's contact/location page). If none are given, return an empty list.

    CRITERIA EVIDENCE (return any mentioned; if not mentioned, leave null or empty):
    - trauma_level: The trauma level claimed (e.g., "Level I", "Level II")
    - trauma_urls: URLs that support ACS verification or California state designation for trauma level
    - stroke_certification: "Primary Stroke Center" or "Comprehensive Stroke Center" (or equivalent phrasing)
    - stroke_accreditor: Accrediting body (e.g., "The Joint Commission", "DNV", or equivalent recognized body)
    - stroke_urls: URLs that support stroke certification (accreditor directory or official hospital source)
    - acgme_programs: List of program names if mentioned (or any indication of ACGME-accredited residency)
    - teaching_urls: URLs that support ACGME-accredited program existence (ACGME directory or sponsoring institution/hospital GME page)
    - magnet_status: e.g., "ANCC Magnet Recognition"
    - magnet_urls: URLs that support Magnet recognition (ANCC directory or official hospital page)
    - nicu_level: "Level III" or "Level IV"
    - nicu_urls: URLs that support NICU level (official hospital or authoritative source)
    - beds_text: Textual beds count (e.g., "350 licensed beds")
    - bed_urls: URLs that support licensed bed count (official hospital or authoritative source)
    - cms_stars: e.g., "3 stars", "4 stars", "5 stars"
    - cms_urls: URLs to CMS/Medicare Care Compare or authoritative listing that shows the star rating
    - accreditation_body: "Joint Commission" or "DNV" (or equivalent if clearly stated)
    - accreditation_urls: URLs that support current hospital accreditation (Joint Commission QualityCheck, DNV accreditation listing, or official hospital source)

    IMPORTANT:
    - Only include URLs explicitly present in the answer. Do not invent or infer any URLs.
    - If a field isn’t mentioned, return null for single-value fields or an empty list for array fields.
    - Preserve the original phrasing for text fields found in the answer.
    - Return the hospitals in the order they appear in the answer.
    """


# ------------------------------- Helpers ----------------------------------- #
def _nonempty_urls(*url_lists: List[str], maybe_single: Optional[str] = None) -> List[str]:
    urls: List[str] = []
    for lst in url_lists:
        for u in lst:
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())
    if isinstance(maybe_single, str) and maybe_single.strip():
        urls.append(maybe_single.strip())
    # Deduplicate while preserving order
    seen = set()
    unique_urls: List[str] = []
    for u in urls:
        if u not in seen:
            unique_urls.append(u)
            seen.add(u)
    return unique_urls


# --------------------------- Verification Logic ---------------------------- #
async def verify_hospital(
    evaluator: Evaluator,
    parent_node,
    hospital: HospitalInfo,
    hospital_index: int,
) -> None:
    """
    Build verification subtree for a single hospital (basic info + 8 criteria).
    """
    hosp_name = hospital.name or f"Hospital #{hospital_index}"

    # Hospital node (non-critical to allow partial credit across hospitals)
    hospital_node = evaluator.add_parallel(
        id=f"hospital_{hospital_index}",
        desc=f"Hospital #{hospital_index} (one of three) — info + verification of all criteria",
        parent=parent_node,
        critical=False,
    )

    # ---------------- Basic Info (critical under hospital) ---------------- #
    basic_info = evaluator.add_parallel(
        id=f"hospital_{hospital_index}_basic_info",
        desc="Provide required basic identifying information for the hospital",
        parent=hospital_node,
        critical=True,
    )

    # Name provided
    evaluator.add_custom_node(
        result=bool(hospital.name and hospital.name.strip()),
        id=f"hospital_{hospital_index}_name",
        desc="Official hospital name provided",
        parent=basic_info,
        critical=True,
    )

    # Address provided (street, city, state)
    address_ok = all([
        hospital.street_address and hospital.street_address.strip(),
        hospital.city and hospital.city.strip(),
        hospital.state and hospital.state.strip()
    ])
    address_node = evaluator.add_custom_node(
        result=address_ok,
        id=f"hospital_{hospital_index}_address",
        desc="Complete physical address provided (street, city, state)",
        parent=basic_info,
        critical=True,
    )

    # Website provided
    website_ok = bool(hospital.website_url and hospital.website_url.strip())
    website_node = evaluator.add_custom_node(
        result=website_ok,
        id=f"hospital_{hospital_index}_website",
        desc="Official hospital website URL provided",
        parent=basic_info,
        critical=True,
    )

    # California location verification (uses sources like hospital website/location page)
    ca_loc_leaf = evaluator.add_leaf(
        id=f"hospital_{hospital_index}_california_location",
        desc="Hospital is located in California",
        parent=basic_info,
        critical=True,
    )
    ca_sources = _nonempty_urls(hospital.location_urls, maybe_single=hospital.website_url)
    ca_claim = f"The hospital '{hosp_name}' is located in the state of California."
    await evaluator.verify(
        claim=ca_claim,
        node=ca_loc_leaf,
        sources=ca_sources,
        additional_instruction=(
            "Verify the hospital's address is in California. Accept 'California' or 'CA' on the address/website. "
            "Use official or authoritative location pages. If the address clearly shows a California city and 'CA', "
            "consider it supported."
        ),
        extra_prerequisites=[address_node, website_node],
    )

    # ---------------- Criteria (critical under hospital) ------------------ #
    criteria_node = evaluator.add_parallel(
        id=f"hospital_{hospital_index}_criteria",
        desc="Hospital meets all 8 criteria; each criterion includes verification + authoritative reference URL",
        parent=hospital_node,
        critical=True,
    )

    # 1) Trauma center
    trauma_node = evaluator.add_parallel(
        id=f"hospital_{hospital_index}_trauma_center",
        desc="Trauma center requirement satisfied and sourced",
        parent=criteria_node,
        critical=True,
    )
    trauma_ref = evaluator.add_custom_node(
        result=len(hospital.trauma_urls) > 0,
        id=f"hospital_{hospital_index}_trauma_reference_url",
        desc="Authoritative reference URL provided for trauma designation (official hospital / accreditor / government source)",
        parent=trauma_node,
        critical=True,
    )
    trauma_leaf = evaluator.add_leaf(
        id=f"hospital_{hospital_index}_trauma_meets_requirement",
        desc="Hospital is a Level I or Level II trauma center verified by ACS or designated by the State of California",
        parent=trauma_node,
        critical=True,
    )
    trauma_level_text = hospital.trauma_level or "Level I or Level II"
    trauma_claim = (
        f"The hospital '{hosp_name}' is a {trauma_level_text} trauma center, verified by the ACS or designated by the State of California."
    )
    await evaluator.verify(
        claim=trauma_claim,
        node=trauma_leaf,
        sources=hospital.trauma_urls,
        additional_instruction=(
            "Confirm that the source explicitly shows ACS verification or California state designation of Level I/II trauma status. "
            "Accept ACS 'Verified Trauma Center' listings or California EMS/state designation pages, or the hospital's page clearly stating "
            "Level I/II status."
        ),
        extra_prerequisites=[trauma_ref],
    )

    # 2) Stroke center
    stroke_node = evaluator.add_parallel(
        id=f"hospital_{hospital_index}_stroke_center",
        desc="Stroke center requirement satisfied and sourced",
        parent=criteria_node,
        critical=True,
    )
    stroke_ref = evaluator.add_custom_node(
        result=len(hospital.stroke_urls) > 0,
        id=f"hospital_{hospital_index}_stroke_reference_url",
        desc="Authoritative reference URL provided for stroke certification (official hospital / accreditor / government source)",
        parent=stroke_node,
        critical=True,
    )
    stroke_leaf = evaluator.add_leaf(
        id=f"hospital_{hospital_index}_stroke_meets_requirement",
        desc="Hospital holds Primary Stroke Center or Comprehensive Stroke Center certification from Joint Commission, DNV, or equivalent recognized accrediting body",
        parent=stroke_node,
        critical=True,
    )
    stroke_cert_text = hospital.stroke_certification or "Primary or Comprehensive Stroke Center"
    stroke_accr_text = hospital.stroke_accreditor or "a recognized accrediting body (The Joint Commission or DNV)"
    stroke_claim = (
        f"The hospital '{hosp_name}' holds {stroke_cert_text} certification from {stroke_accr_text}."
    )
    await evaluator.verify(
        claim=stroke_claim,
        node=stroke_leaf,
        sources=hospital.stroke_urls,
        additional_instruction=(
            "Verify stroke center certification via accreditor directories (e.g., The Joint Commission or DNV) or official hospital pages. "
            "Accept 'Primary Stroke Center' or 'Comprehensive Stroke Center' (PSC/CSC) designations."
        ),
        extra_prerequisites=[stroke_ref],
    )

    # 3) Teaching status (ACGME-accredited program)
    teaching_node = evaluator.add_parallel(
        id=f"hospital_{hospital_index}_teaching_status",
        desc="Teaching hospital requirement satisfied and sourced",
        parent=criteria_node,
        critical=True,
    )
    teaching_ref = evaluator.add_custom_node(
        result=len(hospital.teaching_urls) > 0,
        id=f"hospital_{hospital_index}_teaching_reference_url",
        desc="Authoritative reference URL provided for ACGME residency verification (ACGME / sponsoring institution / official source)",
        parent=teaching_node,
        critical=True,
    )
    teaching_leaf = evaluator.add_leaf(
        id=f"hospital_{hospital_index}_teaching_meets_requirement",
        desc="Hospital has at least one ACGME-accredited residency program",
        parent=teaching_node,
        critical=True,
    )
    acgme_example = ", ".join(hospital.acgme_programs) if hospital.acgme_programs else "at least one program"
    teaching_claim = (
        f"The hospital '{hosp_name}' has at least one ACGME-accredited residency program ({acgme_example})."
    )
    await evaluator.verify(
        claim=teaching_claim,
        node=teaching_leaf,
        sources=hospital.teaching_urls,
        additional_instruction=(
            "Confirm via the ACGME public directory or official sponsoring institution/hospital GME pages that the hospital has an ACGME-accredited residency program."
        ),
        extra_prerequisites=[teaching_ref],
    )

    # 4) Magnet recognition (ANCC)
    magnet_node = evaluator.add_parallel(
        id=f"hospital_{hospital_index}_magnet_recognition",
        desc="Magnet recognition requirement satisfied and sourced",
        parent=criteria_node,
        critical=True,
    )
    magnet_ref = evaluator.add_custom_node(
        result=len(hospital.magnet_urls) > 0,
        id=f"hospital_{hospital_index}_magnet_reference_url",
        desc="Authoritative reference URL provided for Magnet recognition (ANCC / official hospital source)",
        parent=magnet_node,
        critical=True,
    )
    magnet_leaf = evaluator.add_leaf(
        id=f"hospital_{hospital_index}_magnet_meets_requirement",
        desc="Hospital currently holds ANCC Magnet Recognition",
        parent=magnet_node,
        critical=True,
    )
    magnet_status_text = hospital.magnet_status or "ANCC Magnet Recognition"
    magnet_claim = f"The hospital '{hosp_name}' currently holds {magnet_status_text}."
    await evaluator.verify(
        claim=magnet_claim,
        node=magnet_leaf,
        sources=hospital.magnet_urls,
        additional_instruction=(
            "Verify Magnet status via the ANCC Magnet Recognition Program directory or official hospital announcements. "
            "Status must be current."
        ),
        extra_prerequisites=[magnet_ref],
    )

    # 5) NICU level (III or IV)
    nicu_node = evaluator.add_parallel(
        id=f"hospital_{hospital_index}_nicu_level",
        desc="NICU level requirement satisfied and sourced",
        parent=criteria_node,
        critical=True,
    )
    nicu_ref = evaluator.add_custom_node(
        result=len(hospital.nicu_urls) > 0,
        id=f"hospital_{hospital_index}_nicu_reference_url",
        desc="Authoritative reference URL provided for NICU level (official hospital / government / authoritative source)",
        parent=nicu_node,
        critical=True,
    )
    nicu_leaf = evaluator.add_leaf(
        id=f"hospital_{hospital_index}_nicu_meets_requirement",
        desc="Hospital has a Level III or Level IV NICU",
        parent=nicu_node,
        critical=True,
    )
    nicu_level_text = hospital.nicu_level or "Level III or Level IV"
    nicu_claim = f"The hospital '{hosp_name}' has a {nicu_level_text} Neonatal Intensive Care Unit (NICU)."
    await evaluator.verify(
        claim=nicu_claim,
        node=nicu_leaf,
        sources=hospital.nicu_urls,
        additional_instruction=(
            "Confirm NICU level via official hospital pages or authoritative perinatal care level listings (Level III/IV). "
            "Accept clear statements like 'Level III NICU' or 'Level IV NICU'."
        ),
        extra_prerequisites=[nicu_ref],
    )

    # 6) Bed count (>=200 licensed)
    beds_node = evaluator.add_parallel(
        id=f"hospital_{hospital_index}_bed_count",
        desc="Bed count requirement satisfied and sourced",
        parent=criteria_node,
        critical=True,
    )
    beds_ref = evaluator.add_custom_node(
        result=len(hospital.bed_urls) > 0,
        id=f"hospital_{hospital_index}_beds_reference_url",
        desc="Authoritative reference URL provided for licensed bed count (official hospital / government / authoritative source)",
        parent=beds_node,
        critical=True,
    )
    beds_leaf = evaluator.add_leaf(
        id=f"hospital_{hospital_index}_beds_meets_requirement",
        desc="Hospital has at least 200 licensed inpatient beds",
        parent=beds_node,
        critical=True,
    )
    beds_text = hospital.beds_text or "at least 200 licensed inpatient beds"
    beds_claim = f"The hospital '{hosp_name}' has {beds_text}, which is at least 200 licensed inpatient beds."
    await evaluator.verify(
        claim=beds_claim,
        node=beds_leaf,
        sources=hospital.bed_urls,
        additional_instruction=(
            "Verify bed count from hospital fact sheets, annual reports, or authoritative databases. "
            "The statement must support that licensed inpatient beds are >= 200. Minor phrasing differences are acceptable."
        ),
        extra_prerequisites=[beds_ref],
    )

    # 7) CMS star rating (>=3)
    cms_node = evaluator.add_parallel(
        id=f"hospital_{hospital_index}_cms_star_rating",
        desc="CMS star rating requirement satisfied and sourced",
        parent=criteria_node,
        critical=True,
    )
    cms_ref = evaluator.add_custom_node(
        result=len(hospital.cms_urls) > 0,
        id=f"hospital_{hospital_index}_cms_reference_url",
        desc="Authoritative reference URL provided for CMS star rating (CMS/Medicare database)",
        parent=cms_node,
        critical=True,
    )
    cms_leaf = evaluator.add_leaf(
        id=f"hospital_{hospital_index}_cms_meets_requirement",
        desc="Hospital has a CMS Overall Hospital Quality Star Rating of 3 stars or higher",
        parent=cms_node,
        critical=True,
    )
    cms_text = hospital.cms_stars or "3 or more stars"
    cms_claim = f"The hospital '{hosp_name}' has a CMS Overall Hospital Quality Star Rating of {cms_text}, which is 3 stars or higher."
    await evaluator.verify(
        claim=cms_claim,
        node=cms_leaf,
        sources=hospital.cms_urls,
        additional_instruction=(
            "Check Medicare Care Compare (CMS) pages or authoritative sources showing the hospital's overall star rating. "
            "Accept if the rating is 3, 4, or 5 stars."
        ),
        extra_prerequisites=[cms_ref],
    )

    # 8) Current accreditation (Joint Commission or DNV)
    accred_node = evaluator.add_parallel(
        id=f"hospital_{hospital_index}_current_accreditation",
        desc="Current hospital accreditation requirement satisfied and sourced",
        parent=criteria_node,
        critical=True,
    )
    accred_ref = evaluator.add_custom_node(
        result=len(hospital.accreditation_urls) > 0,
        id=f"hospital_{hospital_index}_accreditation_reference_url",
        desc="Authoritative reference URL provided for accreditation (Joint Commission / DNV / official hospital source)",
        parent=accred_node,
        critical=True,
    )
    accred_leaf = evaluator.add_leaf(
        id=f"hospital_{hospital_index}_accreditation_meets_requirement",
        desc="Hospital maintains current Joint Commission or DNV hospital accreditation",
        parent=accred_node,
        critical=True,
    )
    accred_body_text = hospital.accreditation_body or "Joint Commission or DNV"
    accred_claim = f"The hospital '{hosp_name}' maintains current hospital accreditation from {accred_body_text}."
    await evaluator.verify(
        claim=accred_claim,
        node=accred_leaf,
        sources=hospital.accreditation_urls,
        additional_instruction=(
            "Confirm via Joint Commission QualityCheck, DNV accredited organizations listings, or official hospital sources "
            "that accreditation is current."
        ),
        extra_prerequisites=[accred_ref],
    )


# ---------------------------- Main Entry Point ----------------------------- #
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
    Evaluate an answer for the California hospitals multi-criteria verification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates hospitals independently
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

    # Extract hospitals info
    extracted = await evaluator.extract(
        prompt=prompt_extract_hospitals(),
        template_class=HospitalsExtraction,
        extraction_name="hospitals_extraction",
    )

    # Select up to 3 hospitals; pad with empty placeholders if fewer
    hospitals = list(extracted.hospitals[:3])
    while len(hospitals) < 3:
        hospitals.append(HospitalInfo())

    # Build verification for each hospital
    for idx, hosp in enumerate(hospitals, start=1):
        await verify_hospital(evaluator, root, hosp, idx)

    # Return summary with verification tree
    return evaluator.get_summary()