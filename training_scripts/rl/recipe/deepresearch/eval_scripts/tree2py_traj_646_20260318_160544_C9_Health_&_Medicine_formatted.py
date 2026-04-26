import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_major_teaching_hospitals_4"
TASK_DESCRIPTION = """
Identify four distinct major teaching hospitals in the United States that satisfy ALL of the following criteria:

1. Teaching Hospital Status:
   - Has a resident-to-bed ratio above 0.25, qualifying it as a major teaching hospital under Medicare classifications
   - Has at least 400 beds in service
   - Receives Medicare Graduate Medical Education (GME) funding, including both Direct GME (DGME) and Indirect Medical Education (IME) payments

2. Academic Medical Center Integration:
   - Is organizationally and administratively integrated with an accredited medical school
   - Has ACGME-accredited residency programs

3. Research Funding:
   - Is ranked among the top 50 NIH-funded institutions in 2025 (as documented in BRIMR rankings, NIH official reports, or similar authoritative sources)

4. CMS Quality Programs:
   - Received a CMS Overall Hospital Quality Star Rating in the 2026 rating cycle
   - Participates in the Hospital Value-Based Purchasing (VBP) Program
   - Is subject to the Hospital Readmissions Reduction Program (HRRP)

For each of the four hospitals you identify, provide:
- The hospital's full official name
- Documentation/verification for each criterion through reference URLs from authoritative sources (CMS databases, NIH funding reports, hospital websites, ACGME listings, etc.).
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class HospitalEvidence(BaseModel):
    # Identity
    name: Optional[str] = None

    # Teaching hospital status
    resident_to_bed_ratio_value: Optional[str] = None
    ratio_urls: List[str] = Field(default_factory=list)

    bed_count_value: Optional[str] = None
    bed_count_urls: List[str] = Field(default_factory=list)

    gme_notes: Optional[str] = None
    gme_urls: List[str] = Field(default_factory=list)  # Should support DGME & IME documentation

    # Academic integration
    affiliated_medical_school: Optional[str] = None
    affiliation_urls: List[str] = Field(default_factory=list)

    acgme_notes: Optional[str] = None
    acgme_urls: List[str] = Field(default_factory=list)

    # Research funding
    nih_top50_2025_notes: Optional[str] = None
    nih_top50_2025_urls: List[str] = Field(default_factory=list)

    # CMS quality programs (2026 cycle star rating)
    cms_star_2026_notes: Optional[str] = None
    cms_star_2026_urls: List[str] = Field(default_factory=list)

    # CMS program participation
    vbp_notes: Optional[str] = None
    vbp_urls: List[str] = Field(default_factory=list)

    hrrp_notes: Optional[str] = None
    hrrp_urls: List[str] = Field(default_factory=list)


class HospitalsExtraction(BaseModel):
    hospitals: List[HospitalEvidence] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hospitals() -> str:
    return """
    Extract up to four distinct U.S. hospitals from the answer. For each hospital, return a JSON object containing:
      - name: Full official hospital name
      - resident_to_bed_ratio_value: The value stated or implied in the answer (string; keep as written)
      - ratio_urls: URLs that directly document the resident-to-bed ratio or explicitly confirm "major teaching hospital" classification based on ratio > 0.25
      - bed_count_value: The bed count stated or implied (string; keep as written)
      - bed_count_urls: URLs that document the hospital's bed count
      - gme_notes: Any note about GME funding (DGME/IME)
      - gme_urls: URLs that document the hospital receiving Medicare GME funding, including both DGME (Direct GME) and IME (Indirect Medical Education) payments
      - affiliated_medical_school: The affiliated accredited medical school (string if stated)
      - affiliation_urls: URLs that document the formal affiliation with an accredited medical school (hospital or school site, AAMC, LCME, etc.)
      - acgme_notes: Any note about ACGME-accredited residency programs
      - acgme_urls: URLs that document ACGME-accredited residency programs at the hospital (ACGME directories, GME office pages)
      - nih_top50_2025_notes: Any note about NIH top 50 status in 2025
      - nih_top50_2025_urls: URLs from authoritative sources (BRIMR, NIH awards database, official rankings) showing a top-50 NIH-funded position in 2025 (hospital, associated academic medical center, or parent system if applicable)
      - cms_star_2026_notes: Any note about CMS Overall Star Rating in the 2026 cycle
      - cms_star_2026_urls: URLs from CMS Care Compare or Provider Data Catalog that show the hospital's 2026 star rating (any star value counts as having a rating)
      - vbp_notes: Any note about participation in the Hospital Value-Based Purchasing (VBP) Program
      - vbp_urls: URLs that document VBP participation (CMS datasets, facility profiles, or official publications)
      - hrrp_notes: Any note about being subject to the Hospital Readmissions Reduction Program (HRRP)
      - hrrp_urls: URLs that document HRRP status (CMS datasets, facility profiles, or official publications)

    IMPORTANT:
    - Extract only URLs explicitly present in the answer. If a criterion lacks URLs, return an empty list for that field.
    - Do not invent or infer URLs.
    - Keep numbers as strings (e.g., "0.29", "1,050", "≈500") exactly as written in the answer.
    - Return a single object with a field "hospitals": an array of up to four such hospital objects, in the order presented in the answer.
    - If fewer than four hospitals are provided, include as many as present. Missing fields should be null or empty arrays as appropriate.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def _verify_with_sources(
    evaluator: Evaluator,
    parent,
    *,
    leaf_id: str,
    desc: str,
    claim: str,
    sources: List[str],
    additional_instruction: str,
    critical: bool = True
):
    """
    Create a leaf node and verify a claim using provided sources.
    """
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=additional_instruction
    )


def _urls_exist(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and len(u.strip()) > 0 for u in urls)


# --------------------------------------------------------------------------- #
# Hospital verification tree builder                                          #
# --------------------------------------------------------------------------- #
async def verify_single_hospital(
    evaluator: Evaluator,
    root_parent,
    hospital: HospitalEvidence,
    idx: int
) -> None:
    """
    Build verification subtree for a single hospital and run all checks per rubric.
    """
    display_idx = idx + 1
    hosp_name = hospital.name or f"Hospital #{display_idx} (name missing)"

    # Top node for this hospital (non-critical to allow partial credit across hospitals)
    hospital_node = evaluator.add_parallel(
        id=f"hospital_{display_idx}",
        desc=f"{['First','Second','Third','Fourth'][idx]} identified hospital meets all required criteria",
        parent=root_parent,
        critical=False
    )

    # 0) Basic presence of a hospital name (critical for this hospital)
    evaluator.add_custom_node(
        result=(hospital.name is not None and hospital.name.strip() != ""),
        id=f"h{display_idx}_name_present",
        desc="Hospital name is provided",
        parent=hospital_node,
        critical=True
    )

    # 1) Teaching hospital status (parallel, all critical subcriteria)
    ths_node = evaluator.add_parallel(
        id=f"h{display_idx}_teaching_hospital_status",
        desc="Hospital qualifies as a major teaching hospital based on Medicare and standard classification criteria",
        parent=hospital_node,
        critical=True
    )

    # 1.a) Resident-to-bed ratio > 0.25
    ratio_node = evaluator.add_parallel(
        id=f"h{display_idx}_resident_to_bed_ratio",
        desc="Hospital has a resident-to-bed ratio above 0.25, qualifying it as a major teaching hospital under Medicare classifications",
        parent=ths_node,
        critical=True
    )
    # Existence of documentation URL(s)
    evaluator.add_custom_node(
        result=_urls_exist(hospital.ratio_urls),
        id=f"h{display_idx}_ratio_verification_url",
        desc="Provides a reference URL documenting the hospital's resident-to-bed ratio or teaching hospital classification",
        parent=ratio_node,
        critical=True
    )
    # Verify threshold claim using the provided URLs
    await _verify_with_sources(
        evaluator,
        ratio_node,
        leaf_id=f"h{display_idx}_resident_to_bed_ratio_check",
        desc="Resident-to-bed ratio above 0.25 is supported by the cited sources",
        claim=f"{hosp_name} has a resident-to-bed ratio above 0.25 (qualifying it as a 'major teaching hospital' under Medicare's ratio criterion).",
        sources=hospital.ratio_urls,
        additional_instruction="Accept if the source explicitly lists the resident-to-bed ratio above 0.25, or explicitly classifies the hospital as a 'major teaching hospital' based on this ratio. Allow reasonable synonyms ('teaching intensity', 'resident-to-bed index').",
        critical=True
    )

    # 1.b) Minimum bed count >= 400
    bed_node = evaluator.add_parallel(
        id=f"h{display_idx}_minimum_bed_count",
        desc="Hospital has at least 400 beds in service, meeting the size requirement for major teaching hospital designation",
        parent=ths_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_exist(hospital.bed_count_urls),
        id=f"h{display_idx}_bed_count_verification_url",
        desc="Provides a reference URL documenting the hospital's bed count",
        parent=bed_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        bed_node,
        leaf_id=f"h{display_idx}_minimum_bed_count_check",
        desc="At least 400 beds claim is supported by the cited sources",
        claim=f"{hosp_name} has at least 400 beds in service.",
        sources=hospital.bed_count_urls,
        additional_instruction="Confirm that the bed count is ≥ 400. Accept reasonable authoritative sources (hospital fact sheets, AHA, CMS, state profiles). Allow small formatting variations (e.g., 'licensed beds', 'staffed beds'); if multiple figures are present, prefer the one explicitly tied to current operational or staffed beds.",
        critical=True
    )

    # 1.c) GME funding includes both DGME and IME
    gme_node = evaluator.add_parallel(
        id=f"h{display_idx}_gme_funding_status",
        desc="Hospital receives Medicare Graduate Medical Education (GME) funding including both Direct GME and Indirect Medical Education payments",
        parent=ths_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_exist(hospital.gme_urls),
        id=f"h{display_idx}_gme_verification_url",
        desc="Provides a reference URL documenting the hospital's GME funding status",
        parent=gme_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        gme_node,
        leaf_id=f"h{display_idx}_gme_funding_check",
        desc="GME funding (DGME and IME) is supported by the cited sources",
        claim=f"{hosp_name} receives Medicare GME funding that includes both Direct GME (DGME) and Indirect Medical Education (IME) payments.",
        sources=hospital.gme_urls,
        additional_instruction="Verify that both DGME and IME (or equivalent wording) are indicated for the hospital. Accept CMS sources, cost reports, or official documentation that clearly indicate eligibility/receipt of both components.",
        critical=True
    )

    # 2) Academic medical center integration
    ami_node = evaluator.add_parallel(
        id=f"h{display_idx}_academic_integration",
        desc="Hospital is organizationally and administratively integrated with a medical school as an academic medical center",
        parent=hospital_node,
        critical=True
    )

    # 2.a) Medical school affiliation
    aff_node = evaluator.add_parallel(
        id=f"h{display_idx}_medical_school_affiliation",
        desc="Hospital is formally affiliated with an accredited medical school",
        parent=ami_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_exist(hospital.affiliation_urls),
        id=f"h{display_idx}_affiliation_verification_url",
        desc="Provides a reference URL documenting the hospital's medical school affiliation",
        parent=aff_node,
        critical=True
    )
    school_label = hospital.affiliated_medical_school or "an accredited medical school"
    await _verify_with_sources(
        evaluator,
        aff_node,
        leaf_id=f"h{display_idx}_medical_school_affiliation_check",
        desc="Medical school affiliation is supported by the cited sources",
        claim=f"{hosp_name} is formally affiliated with {school_label}.",
        sources=hospital.affiliation_urls,
        additional_instruction="Confirm formal affiliation with an accredited medical school (e.g., LCME accredited). Accept hospital or university sites, AAMC/LCME/AMA or similarly authoritative pages.",
        critical=True
    )

    # 2.b) ACGME-accredited programs
    acgme_node = evaluator.add_parallel(
        id=f"h{display_idx}_acgme_accredited_programs",
        desc="Hospital has ACGME-accredited residency programs",
        parent=ami_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_exist(hospital.acgme_urls),
        id=f"h{display_idx}_acgme_verification_url",
        desc="Provides a reference URL documenting the hospital's ACGME-accredited programs",
        parent=acgme_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        acgme_node,
        leaf_id=f"h{display_idx}_acgme_programs_check",
        desc="Presence of ACGME-accredited residency programs is supported by the cited sources",
        claim=f"{hosp_name} has ACGME-accredited residency program(s).",
        sources=hospital.acgme_urls,
        additional_instruction="Accept ACGME directory listings, sponsoring institution pages, or the hospital's GME office pages explicitly stating ACGME accreditation.",
        critical=True
    )

    # 3) Research funding (top 50 NIH-funded institutions in 2025)
    rf_node = evaluator.add_parallel(
        id=f"h{display_idx}_research_funding",
        desc="Hospital is ranked among the top 50 NIH-funded institutions in 2025",
        parent=hospital_node,
        critical=True
    )
    top50_node = evaluator.add_parallel(
        id=f"h{display_idx}_top_50_nih_ranking",
        desc="Hospital appears in the top 50 of NIH funding rankings for 2025 (as published in BRIMR, NIH reports, or similar official sources)",
        parent=rf_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_exist(hospital.nih_top50_2025_urls),
        id=f"h{display_idx}_nih_ranking_verification_url",
        desc="Provides a reference URL from an official NIH funding ranking source (such as BRIMR, NIH Awards database, or published rankings) showing the hospital's top 50 status",
        parent=top50_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        top50_node,
        leaf_id=f"h{display_idx}_nih_top50_2025_check",
        desc="Top-50 NIH funding status in 2025 is supported by the cited sources",
        claim=f"{hosp_name} (or its directly integrated academic medical center or parent system) is among the top 50 NIH-funded institutions in 2025.",
        sources=hospital.nih_top50_2025_urls,
        additional_instruction="Accept rankings or datasets (e.g., BRIMR, NIH awards) that rank hospitals, academic medical centers, or directly integrated parent systems. Allow reasonable name variants (e.g., health system vs. flagship hospital) if the affiliation is clear.",
        critical=True
    )

    # 4) CMS Quality Programs
    qr_node = evaluator.add_parallel(
        id=f"h{display_idx}_quality_ratings",
        desc="Hospital has received quality ratings and participates in CMS quality programs",
        parent=hospital_node,
        critical=True
    )

    # 4.a) CMS Star Rating (2026)
    star_node = evaluator.add_parallel(
        id=f"h{display_idx}_cms_star_rating_2026",
        desc="Hospital received a CMS Overall Hospital Quality Star Rating in the 2026 rating cycle",
        parent=qr_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_exist(hospital.cms_star_2026_urls),
        id=f"h{display_idx}_star_rating_verification_url",
        desc="Provides a reference URL from CMS Care Compare or Provider Data Catalog showing the hospital's 2026 star rating",
        parent=star_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        star_node,
        leaf_id=f"h{display_idx}_cms_star_2026_check",
        desc="Receipt of a CMS Overall Star Rating in 2026 is supported by the cited sources",
        claim=f"{hosp_name} received a CMS Overall Hospital Quality Star Rating in the 2026 rating cycle.",
        sources=hospital.cms_star_2026_urls,
        additional_instruction="Any star value (1–5) counts as having a rating; confirm the 2026 cycle. Prefer CMS Care Compare or Provider Data Catalog pages.",
        critical=True
    )

    # 4.b) VBP participation
    vbp_node = evaluator.add_parallel(
        id=f"h{display_idx}_vbp_participation",
        desc="Hospital participates in the Hospital Value-Based Purchasing (VBP) Program",
        parent=qr_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_exist(hospital.vbp_urls),
        id=f"h{display_idx}_vbp_verification_url",
        desc="Provides a reference URL documenting the hospital's VBP program participation",
        parent=vbp_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        vbp_node,
        leaf_id=f"h{display_idx}_vbp_participation_check",
        desc="VBP participation is supported by the cited sources",
        claim=f"{hosp_name} participates in the Hospital Value-Based Purchasing (VBP) Program.",
        sources=hospital.vbp_urls,
        additional_instruction="Confirm that the hospital is included in CMS VBP program participation lists/profiles. Accept CMS documentation or other authoritative federal references.",
        critical=True
    )

    # 4.c) HRRP subject
    hrrp_node = evaluator.add_parallel(
        id=f"h{display_idx}_hrrp_subject",
        desc="Hospital is subject to the Hospital Readmissions Reduction Program (HRRP)",
        parent=qr_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_urls_exist(hospital.hrrp_urls),
        id=f"h{display_idx}_hrrp_verification_url",
        desc="Provides a reference URL documenting the hospital's HRRP status or readmission measures",
        parent=hrrp_node,
        critical=True
    )
    await _verify_with_sources(
        evaluator,
        hrrp_node,
        leaf_id=f"h{display_idx}_hrrp_subject_check",
        desc="HRRP subject status is supported by the cited sources",
        claim=f"{hosp_name} is subject to the Hospital Readmissions Reduction Program (HRRP).",
        sources=hospital.hrrp_urls,
        additional_instruction="Confirm inclusion in HRRP-related CMS datasets, profiles, or official publications indicating the hospital is subject to HRRP measures/penalties.",
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
    Evaluate an answer for identifying four distinct U.S. major teaching hospitals
    meeting all specified criteria with authoritative URL documentation.
    """
    # Initialize evaluator (root should be non-critical to allow partial credit across hospitals)
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

    # Extract structured hospital info
    extracted = await evaluator.extract(
        prompt=prompt_extract_hospitals(),
        template_class=HospitalsExtraction,
        extraction_name="hospitals_extraction"
    )

    # Prepare exactly four hospital entries (pad with empty if fewer)
    hospitals: List[HospitalEvidence] = list(extracted.hospitals or [])
    while len(hospitals) < 4:
        hospitals.append(HospitalEvidence())
    hospitals = hospitals[:4]

    # Add a uniqueness check for hospital names (critical to enforce "distinct" requirement)
    names = [h.name.strip() for h in hospitals if h.name]
    unique_count = len(set(n.lower() for n in names))
    evaluator.add_custom_node(
        result=(len(names) == 4 and unique_count == 4),
        id="distinct_hospital_names",
        desc="All four identified hospitals are distinct (no duplicates or missing names).",
        parent=root,
        critical=True
    )

    # Build verification subtrees per hospital
    for i in range(4):
        await verify_single_hospital(evaluator, root, hospitals[i], i)

    # Return structured summary
    return evaluator.get_summary()