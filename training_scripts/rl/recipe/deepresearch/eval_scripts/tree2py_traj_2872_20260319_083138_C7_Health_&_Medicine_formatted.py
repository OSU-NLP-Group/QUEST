import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "ca_hospital_all_criteria"
TASK_DESCRIPTION = """
Find one hospital in California that meets ALL of the following requirements:

1. Has a CMS Overall Hospital Quality Star Rating of 4 or 5 stars
2. Is designated as a Level I Trauma Center
3. Has Comprehensive Stroke Center certification
4. Is accredited by The Joint Commission
5. Has a Level III or Level IV Neonatal Intensive Care Unit (NICU)
6. Has Level III or Level IV Maternal Care designation
7. Has an accredited cardiac surgery program or cardiac care certification
8. Has Commission on Cancer (CoC) accreditation
9. Has an MBSAQIP-accredited bariatric surgery program
10. Has a 24/7 emergency department
11. Is a teaching hospital with residency programs accredited by ACGME
12. Has at least 300 licensed beds

For your answer, provide:
- The hospital's name
- The city where it is located
- The hospital's official website URL
- For each of the 12 requirements above, provide either a direct link to a page that confirms the requirement, or a specific reference explaining where/how the requirement can be verified
""".strip()


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class HospitalExtraction(BaseModel):
    # Core fields
    hospital_name: Optional[str] = None
    city: Optional[str] = None
    website_url: Optional[str] = None

    # Evidence URLs (extract only explicit URLs from the answer)
    cms_star_rating_urls: List[str] = Field(default_factory=list)
    trauma_level_urls: List[str] = Field(default_factory=list)
    stroke_center_urls: List[str] = Field(default_factory=list)
    jc_accreditation_urls: List[str] = Field(default_factory=list)
    nicu_level_urls: List[str] = Field(default_factory=list)
    maternal_care_urls: List[str] = Field(default_factory=list)
    cardiac_program_urls: List[str] = Field(default_factory=list)
    coc_accreditation_urls: List[str] = Field(default_factory=list)
    mbsaqip_bariatric_urls: List[str] = Field(default_factory=list)
    emergency_dept_urls: List[str] = Field(default_factory=list)
    acgme_teaching_urls: List[str] = Field(default_factory=list)
    bed_capacity_urls: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_hospital() -> str:
    return """
Extract structured information for a SINGLE hospital proposed in the answer (the primary one if multiple are mentioned). Extract ONLY what is explicitly present in the answer text. Do not invent or infer.

Return a JSON object with the following fields:
- hospital_name: string | null
- city: string | null  (city in which the hospital is located)
- website_url: string | null  (the hospital's official website URL)

For each of the 12 requirements, extract an array of all explicit URLs in the answer that support that requirement. If the answer provides no explicit URL for a requirement, return an empty array for that field.

- cms_star_rating_urls: URL(s) that confirm the CMS Overall Hospital Quality Star Rating (ideally Medicare Care Compare or official hospital page citing CMS)
- trauma_level_urls: URL(s) confirming designation as a Level I Trauma Center (state/ACS/official hospital)
- stroke_center_urls: URL(s) confirming Comprehensive Stroke Center certification (The Joint Commission, DNV, HFAP, or official hospital page citing it)
- jc_accreditation_urls: URL(s) confirming The Joint Commission hospital accreditation
- nicu_level_urls: URL(s) confirming Level III or Level IV NICU
- maternal_care_urls: URL(s) confirming Level III or Level IV Maternal Care
- cardiac_program_urls: URL(s) confirming accredited cardiac surgery program OR cardiac care certification (ACC, TJC, DNV, etc.)
- coc_accreditation_urls: URL(s) confirming Commission on Cancer (CoC) accreditation
- mbsaqip_bariatric_urls: URL(s) confirming MBSAQIP-accredited bariatric surgery program
- emergency_dept_urls: URL(s) confirming a 24/7 emergency department
- acgme_teaching_urls: URL(s) confirming ACGME-accredited residency programs or teaching status tied to the hospital
- bed_capacity_urls: URL(s) confirming the hospital has at least 300 licensed beds

Important rules:
- Only extract URLs explicitly present in the answer (including within markdown links). Do not infer new URLs.
- Keep arrays even if a single URL is present.
- If a field is missing in the answer, set it to null (for strings) or [] (for URL arrays).
    """.strip()


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _dedup(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if isinstance(u, str):
            uu = u.strip()
            if uu and uu not in seen:
                out.append(uu)
                seen.add(uu)
    return out


def _merge_sources(*url_lists: List[str], singletons: Optional[List[Optional[str]]] = None) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        merged.extend(lst or [])
    if singletons:
        for s in singletons:
            if isinstance(s, str) and s.strip():
                merged.append(s.strip())
    return _dedup(merged)


# -----------------------------------------------------------------------------
# Verification builder
# -----------------------------------------------------------------------------
async def build_and_verify_hospital_criteria(evaluator: Evaluator, root, ex: HospitalExtraction) -> None:
    # Top-level critical node that represents the rubric root
    criteria_root = evaluator.add_parallel(
        id="Hospital_Meeting_All_Criteria",
        desc="A hospital in California that meets all specified certification and quality requirements",
        parent=root,
        critical=True,  # All children must be critical
    )

    # 0) Core info existence (answer-format requirements: name and city must be provided)
    core_info_group = evaluator.add_parallel(
        id="Core_Info",
        desc="Core info (hospital name and city are provided in the answer)",
        parent=criteria_root,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ex.hospital_name and ex.hospital_name.strip()),
        id="hospital_name_present",
        desc="Hospital name is provided in the answer",
        parent=core_info_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ex.city and ex.city.strip()),
        id="city_present",
        desc="Hospital city is provided in the answer",
        parent=core_info_group,
        critical=True
    )

    # 1) Website and evidence presence checks (gates all other verifications)
    web_evidence_group = evaluator.add_parallel(
        id="Website_and_Evidence",
        desc="Official hospital website URL is provided along with links or references for each certification",
        parent=criteria_root,
        critical=True
    )

    # 1.a) Official website URL validity (verify by URL that it is the hospital's official website)
    website_leaf = evaluator.add_leaf(
        id="website_official_valid",
        desc="The provided URL is the hospital's official website (or the hospital's official page within a health system)",
        parent=web_evidence_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage is the official website (or an official hospital page within its health system) for {ex.hospital_name or 'the referenced hospital'}.",
        node=website_leaf,
        sources=ex.website_url,
        additional_instruction="Accept if the page clearly appears to be the hospital's official site (branding, contact info, About pages)."
    )

    # 1.b) Evidence URLs exist for each of the 12 required criteria
    all_requirements_have_links = all([
        len(ex.cms_star_rating_urls) > 0,
        len(ex.trauma_level_urls) > 0,
        len(ex.stroke_center_urls) > 0,
        len(ex.jc_accreditation_urls) > 0,
        len(ex.nicu_level_urls) > 0,
        len(ex.maternal_care_urls) > 0,
        len(ex.cardiac_program_urls) > 0,
        len(ex.coc_accreditation_urls) > 0,
        len(ex.mbsaqip_bariatric_urls) > 0,
        len(ex.emergency_dept_urls) > 0,
        len(ex.acgme_teaching_urls) > 0,
        len(ex.bed_capacity_urls) > 0,
    ])
    evaluator.add_custom_node(
        result=all_requirements_have_links,
        id="evidence_links_provided_for_all_requirements",
        desc="All 12 requirements have at least one explicit evidence URL in the answer",
        parent=web_evidence_group,
        critical=True
    )

    # 2) Individual requirement verifications (all critical)
    # 2.1 Geographic location (California)
    location_leaf = evaluator.add_leaf(
        id="Geographic_Location",
        desc="Hospital is located in California",
        parent=criteria_root,
        critical=True
    )
    # Allow multiple possible sources (official website or any supporting link that states CA)
    loc_sources = _merge_sources(
        ex.cms_star_rating_urls, ex.trauma_level_urls, ex.stroke_center_urls, ex.jc_accreditation_urls,
        ex.nicu_level_urls, ex.maternal_care_urls, ex.cardiac_program_urls, ex.coc_accreditation_urls,
        ex.mbsaqip_bariatric_urls, ex.emergency_dept_urls, ex.acgme_teaching_urls, ex.bed_capacity_urls,
        singletons=[ex.website_url],
    )
    await evaluator.verify(
        claim=f"{ex.hospital_name or 'The hospital'} is located in California.",
        node=location_leaf,
        sources=loc_sources,
        additional_instruction="Confirm that the page clearly indicates the hospital is in California (state mentioned explicitly; city can help but state must be CA)."
    )

    # 2.2 CMS star rating 4 or 5
    cms_leaf = evaluator.add_leaf(
        id="CMS_Star_Rating",
        desc="Hospital has CMS Overall Hospital Quality Star Rating of 4 or 5 stars",
        parent=criteria_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"According to the cited source(s), {ex.hospital_name or 'the hospital'} has a CMS Overall Hospital Quality Star Rating of either 4 stars or 5 stars (out of 5).",
        node=cms_leaf,
        sources=ex.cms_star_rating_urls,
        additional_instruction="Prefer Medicare Care Compare (medicare.gov/care-compare). Accept if the page explicitly states Overall Hospital Quality Star Rating is 4 or 5."
    )

    # 2.3 Level I Trauma Center
    trauma_leaf = evaluator.add_leaf(
        id="Level_I_Trauma_Center",
        desc="Hospital is designated as a Level I Trauma Center",
        parent=criteria_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.hospital_name or 'The hospital'} is designated as a Level I Trauma Center.",
        node=trauma_leaf,
        sources=ex.trauma_level_urls,
        additional_instruction="Accept 'Level I' for adult and/or pediatric trauma. Accept ACS verification or state designation explicitly showing Level I."
    )

    # 2.4 Comprehensive Stroke Center
    stroke_leaf = evaluator.add_leaf(
        id="Comprehensive_Stroke_Center",
        desc="Hospital has Comprehensive Stroke Center certification from Joint Commission, DNV, or HFAP",
        parent=criteria_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.hospital_name or 'The hospital'} has a Comprehensive Stroke Center certification.",
        node=stroke_leaf,
        sources=ex.stroke_center_urls,
        additional_instruction="Accept certifications from The Joint Commission, DNV, or HFAP that explicitly say 'Comprehensive Stroke Center'."
    )

    # 2.5 Joint Commission (TJC) Accreditation
    jc_leaf = evaluator.add_leaf(
        id="Joint_Commission_Accreditation",
        desc="Hospital is accredited by The Joint Commission",
        parent=criteria_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.hospital_name or 'The hospital'} is accredited by The Joint Commission (TJC).",
        node=jc_leaf,
        sources=ex.jc_accreditation_urls,
        additional_instruction="Look for explicit language such as 'accredited by The Joint Commission' on TJC pages or the hospital site."
    )

    # 2.6 Level III/IV NICU
    nicu_leaf = evaluator.add_leaf(
        id="Advanced_NICU",
        desc="Hospital has a Level III or Level IV Neonatal Intensive Care Unit (NICU)",
        parent=criteria_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.hospital_name or 'The hospital'} has a Level III or Level IV Neonatal Intensive Care Unit (NICU).",
        node=nicu_leaf,
        sources=ex.nicu_level_urls,
        additional_instruction="Accept if the page explicitly states 'Level III NICU' or 'Level IV NICU' (or equivalent phrasing)."
    )

    # 2.7 Level III/IV Maternal Care
    maternal_leaf = evaluator.add_leaf(
        id="Advanced_Maternal_Care",
        desc="Hospital has Level III or Level IV Maternal Care designation",
        parent=criteria_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.hospital_name or 'The hospital'} has a Level III or Level IV Maternal Care designation.",
        node=maternal_leaf,
        sources=ex.maternal_care_urls,
        additional_instruction="Accept recognized designations per state/regulator/recognized body. Must explicitly indicate Level III or Level IV Maternal Care."
    )

    # 2.8 Cardiac surgery/care accreditation or certification
    cardiac_leaf = evaluator.add_leaf(
        id="Cardiac_Program",
        desc="Hospital has an accredited cardiac surgery program or cardiac care certification",
        parent=criteria_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.hospital_name or 'The hospital'} holds an accredited cardiac surgery program or a recognized cardiac care certification.",
        node=cardiac_leaf,
        sources=ex.cardiac_program_urls,
        additional_instruction="Accept certifications/accreditations such as ACC (Chest Pain Center), TJC, DNV, or similar recognized bodies that are specific to cardiac care/surgery."
    )

    # 2.9 Commission on Cancer (CoC) accreditation
    coc_leaf = evaluator.add_leaf(
        id="Cancer_Center_Accreditation",
        desc="Hospital has Commission on Cancer (CoC) accreditation",
        parent=criteria_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.hospital_name or 'The hospital'} is accredited by the Commission on Cancer (CoC).",
        node=coc_leaf,
        sources=ex.coc_accreditation_urls,
        additional_instruction="Accept if the page explicitly indicates CoC accreditation from the American College of Surgeons Commission on Cancer."
    )

    # 2.10 MBSAQIP-accredited bariatric surgery program
    bari_leaf = evaluator.add_leaf(
        id="Bariatric_Surgery_Program",
        desc="Hospital has MBSAQIP-accredited bariatric surgery program",
        parent=criteria_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.hospital_name or 'The hospital'} has an MBSAQIP-accredited bariatric surgery program.",
        node=bari_leaf,
        sources=ex.mbsaqip_bariatric_urls,
        additional_instruction="Prefer MBSAQIP official directory/pages; accept official hospital page that explicitly states MBSAQIP accreditation."
    )

    # 2.11 24/7 Emergency Department
    ed_leaf = evaluator.add_leaf(
        id="Emergency_Department",
        desc="Hospital has a 24/7 emergency department",
        parent=criteria_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.hospital_name or 'The hospital'} has an emergency department that operates 24/7 (24 hours a day, 7 days a week).",
        node=ed_leaf,
        sources=ex.emergency_dept_urls,
        additional_instruction="Accept phrasing such as 'open 24/7', '24 hours a day', 'around the clock', or equivalent."
    )

    # 2.12 Teaching hospital with ACGME-accredited programs
    teach_leaf = evaluator.add_leaf(
        id="Teaching_Hospital_Status",
        desc="Hospital is a teaching hospital with residency programs accredited by ACGME",
        parent=criteria_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.hospital_name or 'The hospital'} is a teaching hospital that hosts residency programs accredited by ACGME.",
        node=teach_leaf,
        sources=ex.acgme_teaching_urls,
        additional_instruction="Accept if ACGME public directory or hospital/graduate medical education page explicitly shows ACGME-accredited residency programs at this hospital."
    )

    # 2.13 ≥300 licensed beds
    beds_leaf = evaluator.add_leaf(
        id="Bed_Capacity",
        desc="Hospital has at least 300 licensed beds",
        parent=criteria_root,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ex.hospital_name or 'The hospital'} has at least 300 licensed beds.",
        node=beds_leaf,
        sources=ex.bed_capacity_urls,
        additional_instruction="Accept if the page states ≥300 beds. Prefer 'licensed beds', but if only 'beds' are stated and clearly represent hospital capacity, accept as sufficient."
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    # Initialize evaluator (root is a non-critical container per framework design)
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

    # Extract structured fields from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_hospital(),
        template_class=HospitalExtraction,
        extraction_name="hospital_extraction",
    )

    # Optional: Record task requirement summary for debugging/traceability
    evaluator.add_custom_info(
        info={
            "requirement_count": 12,
            "must_be_in_state": "California",
            "must_meet_all": True,
        },
        info_type="task_requirements",
        info_name="requirements_summary"
    )

    # Build verification tree and run checks
    await build_and_verify_hospital_criteria(evaluator, root, extraction)

    # Return standard evaluation summary
    return evaluator.get_summary()