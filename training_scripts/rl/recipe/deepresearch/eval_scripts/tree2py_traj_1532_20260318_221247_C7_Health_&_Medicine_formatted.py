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
TASK_ID = "pa_hospital_leapfrog_2025"
TASK_DESCRIPTION = """
Identify at least one general acute care hospital in Pennsylvania that meets ALL of the following criteria:

1. The hospital received a Leapfrog Hospital Safety Grade for Fall 2025 (released November 13, 2025)
2. The hospital received an "A" grade in the Fall 2025 Leapfrog Hospital Safety Grade assessment
3. The hospital has sustained an "A" Leapfrog Safety Grade for at least five consecutive grading periods
4. The hospital operates an emergency department providing 24-hour emergency care services
5. The hospital provides the 2025-2026 COVID-19 vaccine (JN.1-lineage formula) to eligible patients
6. The hospital offers RSV vaccination services for adults ages 50 and older in accordance with CDC recommendations
7. The hospital provides childhood vaccination services following the CDC 2025 childhood immunization schedule for children ages 0-18 years
8. The hospital accepts Medicare insurance and provides services to Medicare beneficiaries
9. The hospital offers telehealth services to patients, including services covered under Medicare telehealth flexibilities
10. The hospital accepts health insurance plans from the Health Insurance Marketplace established under the Affordable Care Act
11. The hospital is classified as a general acute care hospital (not a specialty, long-term acute care, or rehabilitation facility)

For your answer, provide:
- The full official name of the hospital
- Confirmation of its location in Pennsylvania
- Reference URL(s) from official sources (such as the hospital's website, Leapfrog's hospitalsafetygrade.org, or Pennsylvania health authorities) that verify the hospital meets these criteria
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HospitalCandidate(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    classification: Optional[str] = None  # e.g., "general acute care hospital"


class HospitalExtraction(BaseModel):
    # Primary identified hospital (pick the first/primary one in the answer)
    hospital: Optional[HospitalCandidate] = None

    # Criterion-specific sources (URLs explicitly present in the answer)
    location_sources: List[str] = Field(default_factory=list)
    classification_sources: List[str] = Field(default_factory=list)

    leapfrog_fall_2025_url: Optional[str] = None
    leapfrog_grade_fall_2025: Optional[str] = None  # e.g., "A"
    leapfrog_sources: List[str] = Field(default_factory=list)

    sustained_a_sources: List[str] = Field(default_factory=list)

    emergency_dept_sources: List[str] = Field(default_factory=list)

    covid_vax_sources: List[str] = Field(default_factory=list)          # 2025-2026 JN.1 lineage
    rsv_vax_sources: List[str] = Field(default_factory=list)            # Adults 50+
    pediatric_vax_sources: List[str] = Field(default_factory=list)      # CDC 2025 schedule 0-18

    medicare_sources: List[str] = Field(default_factory=list)
    telehealth_sources: List[str] = Field(default_factory=list)
    marketplace_sources: List[str] = Field(default_factory=list)

    # General references section, if present in the answer
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hospital() -> str:
    return """
    From the provided answer, extract structured information for a single hospital in Pennsylvania that the answer claims satisfies the criteria. If multiple hospitals are mentioned, choose the first one presented as meeting the criteria.

    Return a JSON object with the following fields:
    - hospital:
        - name: Full official hospital name as written in the answer
        - city: City if provided
        - state: State abbreviation or full state name if provided
        - classification: The facility type/classification text as written (e.g., "general acute care hospital")
    - location_sources: List of URLs explicitly cited in the answer that confirm the hospital's Pennsylvania location
    - classification_sources: List of URLs explicitly cited that confirm the hospital is a general acute care hospital (not specialty/LTACH/rehab)
    - leapfrog_fall_2025_url: The specific hospitalsafetygrade.org page URL for the hospital (if provided)
    - leapfrog_grade_fall_2025: The letter grade the answer claims for Fall 2025 (e.g., "A"), exactly as written
    - leapfrog_sources: Additional Leapfrog-related URLs (if any) from the answer
    - sustained_a_sources: URLs the answer cites to support at least five consecutive "A" grades
    - emergency_dept_sources: URLs that confirm the hospital operates a 24/7 emergency department
    - covid_vax_sources: URLs that confirm the hospital provides the 2025–2026 COVID-19 vaccine (JN.1-lineage formula)
    - rsv_vax_sources: URLs that confirm the hospital offers RSV vaccination for adults ages 50+ in accordance with CDC recommendations
    - pediatric_vax_sources: URLs that confirm the hospital provides childhood vaccinations per the CDC 2025 schedule for ages 0–18
    - medicare_sources: URLs that confirm the hospital accepts Medicare
    - telehealth_sources: URLs that confirm the hospital offers telehealth services (preferably indicating Medicare-covered telehealth)
    - marketplace_sources: URLs that confirm acceptance of insurance plans from the ACA Health Insurance Marketplace
    - reference_urls: All other reference URLs explicitly listed in the answer (hospital website, hospitalsafetygrade.org, PA health authorities, etc.)

    IMPORTANT:
    - Only include URLs explicitly present in the answer. Do not invent or infer any new URLs.
    - If a field is missing in the answer, set it to null (for single values) or an empty list (for lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(urls: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for u in urls:
        if not u:
            continue
        if isinstance(u, str):
            uu = u.strip()
            if uu and uu not in seen:
                cleaned.append(uu)
                seen.add(uu)
    return cleaned


def combine_urls(*groups: Optional[List[str] | str]) -> List[str]:
    acc: List[str] = []
    for g in groups:
        if not g:
            continue
        if isinstance(g, list):
            acc.extend(g)
        elif isinstance(g, str):
            acc.append(g)
    return _unique_urls(acc)


def safe_name(ex: HospitalExtraction) -> str:
    return (ex.hospital.name if ex.hospital and ex.hospital.name else "the hospital")


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def _build_verification_tree(evaluator: Evaluator, ex: HospitalExtraction) -> None:
    # Top-level critical node under root
    top = evaluator.add_parallel(
        id="Hospital_Identification_and_Compliance",
        desc="Identify at least one general acute care hospital in Pennsylvania that meets all specified healthcare service and safety criteria",
        parent=evaluator.root,
        critical=True,
    )

    # 1) Hospital Name & Location
    name_loc = evaluator.add_parallel(
        id="Hospital_Name_and_Location",
        desc="Provide the full official name of the hospital and confirm it is located in Pennsylvania",
        parent=top,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(ex.hospital and ex.hospital.name and ex.hospital.name.strip()),
        id="hospital_name_provided",
        desc="Hospital name is provided in the answer",
        parent=name_loc,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(ex.location_sources) > 0,
        id="location_sources_provided",
        desc="Location confirmation sources are provided",
        parent=name_loc,
        critical=True,
    )
    node_loc = evaluator.add_leaf(
        id="hospital_located_in_pennsylvania",
        desc="The identified hospital is located in Pennsylvania",
        parent=name_loc,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hospital named '{safe_name(ex)}' is located in Pennsylvania (PA).",
        node=node_loc,
        sources=combine_urls(ex.location_sources),
        additional_instruction="Confirm that the hospital is in Pennsylvania. Accept variations like 'PA'. City/state pages or official profiles are acceptable."
    )

    # 2) General Acute Care classification
    gac = evaluator.add_parallel(
        id="General_Acute_Care_Facility",
        desc="Verify the hospital is classified as a general acute care hospital, not a specialty, long-term acute care, or rehabilitation facility",
        parent=top,
        critical=True,
    )
    gac_sources = combine_urls(ex.classification_sources, ex.location_sources, ex.reference_urls)
    evaluator.add_custom_node(
        result=len(gac_sources) > 0,
        id="gac_sources_provided",
        desc="Sources provided for facility classification",
        parent=gac,
        critical=True,
    )
    node_gac = evaluator.add_leaf(
        id="gac_is_general_acute",
        desc="Hospital is a general acute care hospital (not specialty/LTACH/rehab)",
        parent=gac,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{safe_name(ex)}' is a general acute care hospital (short-term acute care), not a specialty, long-term acute care (LTACH), or rehabilitation facility.",
        node=node_gac,
        sources=gac_sources,
        additional_instruction="Allow synonyms such as 'acute care hospital', 'short-term acute care hospital'. Reject if specialty-only, LTACH, or rehab-only."
    )

    # 3) Leapfrog Fall 2025 Safety Grade (presence)
    leapfrog_presence = evaluator.add_parallel(
        id="Leapfrog_Fall_2025_Safety_Grade",
        desc="Verify the hospital received a Leapfrog Hospital Safety Grade for Fall 2025 (released November 13, 2025)",
        parent=top,
        critical=True,
    )
    lf_presence_sources = combine_urls([ex.leapfrog_fall_2025_url] if ex.leapfrog_fall_2025_url else [], ex.leapfrog_sources, ex.reference_urls)
    evaluator.add_custom_node(
        result=len(lf_presence_sources) > 0,
        id="leapfrog_presence_sources_provided",
        desc="Leapfrog sources provided",
        parent=leapfrog_presence,
        critical=True,
    )
    node_lf_presence = evaluator.add_leaf(
        id="leapfrog_fall_2025_present",
        desc="Hospital has a Leapfrog Safety Grade entry for Fall 2025",
        parent=leapfrog_presence,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hospital '{safe_name(ex)}' has a Leapfrog Hospital Safety Grade entry specifically for Fall 2025.",
        node=node_lf_presence,
        sources=lf_presence_sources,
        additional_instruction="Check the hospitalsafetygrade.org page shows 'Fall 2025' for the hospital. The release date is Nov 13, 2025; focus on confirming the Fall 2025 cycle is present."
    )

    # 4) Safety Grade 'A' in Fall 2025
    leapfrog_A = evaluator.add_parallel(
        id="Safety_Grade_A_Achievement",
        desc="Confirm the hospital received an 'A' grade in the Fall 2025 Leapfrog Hospital Safety Grade assessment",
        parent=top,
        critical=True,
    )
    lf_grade_sources = lf_presence_sources
    evaluator.add_custom_node(
        result=len(lf_grade_sources) > 0,
        id="leapfrog_grade_sources_provided",
        desc="Leapfrog grade sources provided",
        parent=leapfrog_A,
        critical=True,
    )
    node_gradeA = evaluator.add_leaf(
        id="leapfrog_fall_2025_grade_A",
        desc="Hospital received an 'A' grade in Fall 2025",
        parent=leapfrog_A,
        critical=True,
    )
    claimed_grade = ex.leapfrog_grade_fall_2025 or "A"
    await evaluator.verify(
        claim=f"In Fall 2025, the Leapfrog Hospital Safety Grade for '{safe_name(ex)}' is '{claimed_grade}', and this grade is an 'A'.",
        node=node_gradeA,
        sources=lf_grade_sources,
        additional_instruction="Confirm that the Fall 2025 grade shown is 'A'. If the answer states the grade, verify it matches the page."
    )

    # 5) Sustained 'A' for >=5 consecutive grading periods
    sustained = evaluator.add_parallel(
        id="Sustained_A_Grade_History",
        desc="Verify the hospital has sustained an 'A' Leapfrog Safety Grade for at least five consecutive grading periods",
        parent=top,
        critical=True,
    )
    sustained_sources = combine_urls(ex.sustained_a_sources, ex.leapfrog_sources, ex.reference_urls)
    evaluator.add_custom_node(
        result=len(sustained_sources) > 0,
        id="sustained_A_sources_provided",
        desc="Sources provided for sustained 'A' history",
        parent=sustained,
        critical=True,
    )
    node_sustained = evaluator.add_leaf(
        id="sustained_A_5_consecutive",
        desc="Hospital has at least five consecutive 'A' Leapfrog grades",
        parent=sustained,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{safe_name(ex)}' has received an 'A' Leapfrog Hospital Safety Grade for at least five consecutive cycles.",
        node=node_sustained,
        sources=sustained_sources,
        additional_instruction="Look for a grade history timeline on hospitalsafetygrade.org (e.g., Spring/Fall cycles) or official announcements showing ≥5 consecutive 'A' grades."
    )

    # 6) Emergency department 24/7
    ed = evaluator.add_parallel(
        id="Emergency_Department_Presence",
        desc="Confirm the hospital operates an emergency department providing 24-hour emergency care services",
        parent=top,
        critical=True,
    )
    ed_sources = combine_urls(ex.emergency_dept_sources, ex.reference_urls)
    evaluator.add_custom_node(
        result=len(ed_sources) > 0,
        id="ed_sources_provided",
        desc="Sources provided for 24/7 emergency department",
        parent=ed,
        critical=True,
    )
    node_ed = evaluator.add_leaf(
        id="ed_24_7",
        desc="Hospital operates a 24/7 emergency department",
        parent=ed,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{safe_name(ex)}' operates an emergency department that provides 24-hour emergency care services.",
        node=node_ed,
        sources=ed_sources,
        additional_instruction="Confirm explicit mention of emergency department and 24/7/24-hour availability."
    )

    # 7) COVID-19 vaccine 2025–2026 JN.1-lineage
    covid = evaluator.add_parallel(
        id="COVID_19_Vaccine_Availability",
        desc="Verify the hospital provides the 2025-2026 COVID-19 vaccine (JN.1-lineage formula) to eligible patients",
        parent=top,
        critical=True,
    )
    covid_sources = combine_urls(ex.covid_vax_sources, ex.reference_urls)
    evaluator.add_custom_node(
        result=len(covid_sources) > 0,
        id="covid_sources_provided",
        desc="Sources provided for 2025–2026 COVID-19 (JN.1) vaccine availability",
        parent=covid,
        critical=True,
    )
    node_covid = evaluator.add_leaf(
        id="covid_2025_2026_jn1_available",
        desc="Hospital provides the 2025–2026 JN.1-lineage COVID-19 vaccine",
        parent=covid,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{safe_name(ex)}' provides the 2025–2026 COVID-19 vaccine based on the JN.1-lineage formula to eligible patients.",
        node=node_covid,
        sources=covid_sources,
        additional_instruction="Accept mentions like '2025-2026 COVID-19 vaccine', 'JN.1-lineage', or equivalent wording indicating the current (2025–26) updated formulation."
    )

    # 8) RSV vaccine services for adults 50+
    rsv = evaluator.add_parallel(
        id="RSV_Vaccine_Services",
        desc="Confirm the hospital offers RSV vaccination services for adults ages 50 and older in accordance with CDC recommendations",
        parent=top,
        critical=True,
    )
    rsv_sources = combine_urls(ex.rsv_vax_sources, ex.reference_urls)
    evaluator.add_custom_node(
        result=len(rsv_sources) > 0,
        id="rsv_sources_provided",
        desc="Sources provided for RSV vaccination for adults 50+",
        parent=rsv,
        critical=True,
    )
    node_rsv = evaluator.add_leaf(
        id="rsv_50_plus",
        desc="Hospital offers RSV vaccination for adults 50+",
        parent=rsv,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{safe_name(ex)}' offers RSV vaccination services for adults ages 50 and older consistent with CDC recommendations.",
        node=node_rsv,
        sources=rsv_sources,
        additional_instruction="Confirm a service page or notice indicating RSV vaccines available to adults 50+ (or broader age ranges that include 50+)."
    )

    # 9) Pediatric vaccination per CDC 2025 schedule (0–18)
    ped = evaluator.add_parallel(
        id="Pediatric_Vaccination_Services",
        desc="Verify the hospital provides childhood vaccination services following the CDC 2025 childhood immunization schedule for children ages 0-18 years",
        parent=top,
        critical=True,
    )
    ped_sources = combine_urls(ex.pediatric_vax_sources, ex.reference_urls)
    evaluator.add_custom_node(
        result=len(ped_sources) > 0,
        id="pediatric_sources_provided",
        desc="Sources provided for pediatric vaccinations per CDC 2025 schedule",
        parent=ped,
        critical=True,
    )
    node_ped = evaluator.add_leaf(
        id="pediatric_vaccinations_cdc_2025",
        desc="Hospital provides childhood vaccinations per CDC 2025 schedule (0–18)",
        parent=ped,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{safe_name(ex)}' provides childhood vaccination services consistent with the CDC 2025 immunization schedule for ages 0–18.",
        node=node_ped,
        sources=ped_sources,
        additional_instruction="Accept explicit mention of following CDC childhood schedule or specific well-child immunization services aligned with the current 2025 schedule."
    )

    # 10) Medicare acceptance
    medicare = evaluator.add_parallel(
        id="Medicare_Acceptance",
        desc="Confirm the hospital accepts Medicare insurance and provides services to Medicare beneficiaries",
        parent=top,
        critical=True,
    )
    medicare_sources = combine_urls(ex.medicare_sources, ex.reference_urls)
    evaluator.add_custom_node(
        result=len(medicare_sources) > 0,
        id="medicare_sources_provided",
        desc="Sources provided for Medicare acceptance",
        parent=medicare,
        critical=True,
    )
    node_medicare = evaluator.add_leaf(
        id="accepts_medicare",
        desc="Hospital accepts Medicare",
        parent=medicare,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{safe_name(ex)}' accepts Medicare and provides services to Medicare beneficiaries.",
        node=node_medicare,
        sources=medicare_sources,
        additional_instruction="Confirm explicit acceptance of Medicare; insurance lists or billing pages are acceptable."
    )

    # 11) Telehealth services (including those covered under Medicare telehealth flexibilities)
    tele = evaluator.add_parallel(
        id="Telehealth_Services",
        desc="Verify the hospital offers telehealth services to patients, including services covered under Medicare telehealth flexibilities",
        parent=top,
        critical=True,
    )
    tele_sources = combine_urls(ex.telehealth_sources, ex.reference_urls)
    evaluator.add_custom_node(
        result=len(tele_sources) > 0,
        id="telehealth_sources_provided",
        desc="Sources provided for telehealth services",
        parent=tele,
        critical=True,
    )
    node_tele = evaluator.add_leaf(
        id="telehealth_including_medicare",
        desc="Hospital offers telehealth, including Medicare-covered services",
        parent=tele,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{safe_name(ex)}' offers telehealth services to patients, including services covered under Medicare telehealth flexibilities.",
        node=node_tele,
        sources=tele_sources,
        additional_instruction="Accept pages that (a) describe telehealth services, and (b) either mention Medicare coverage for telehealth or indicate Medicare is accepted for clinical services including telehealth."
    )

    # 12) ACA Marketplace insurance participation
    mkt = evaluator.add_parallel(
        id="Marketplace_Insurance_Participation",
        desc="Confirm the hospital accepts health insurance plans from the Health Insurance Marketplace established under the Affordable Care Act",
        parent=top,
        critical=True,
    )
    mkt_sources = combine_urls(ex.marketplace_sources, ex.reference_urls)
    evaluator.add_custom_node(
        result=len(mkt_sources) > 0,
        id="marketplace_sources_provided",
        desc="Sources provided for ACA Marketplace participation",
        parent=mkt,
        critical=True,
    )
    node_mkt = evaluator.add_leaf(
        id="accepts_aca_marketplace_plans",
        desc="Hospital accepts insurance from the ACA Marketplace",
        parent=mkt,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{safe_name(ex)}' accepts health insurance plans from the Health Insurance Marketplace established under the Affordable Care Act (ACA).",
        node=node_mkt,
        sources=mkt_sources,
        additional_instruction="Accept terms like 'Health Insurance Marketplace', 'ACA Marketplace', or 'Exchange plans' if clearly referring to ACA marketplace plans."
    )

    # 13) Reference URL verification (official sources present)
    refs = evaluator.add_parallel(
        id="Reference_URL_Verification",
        desc="Provide valid reference URL(s) from official sources (hospital website, Leapfrog, or Pennsylvania health authorities) supporting the hospital identification and criteria compliance",
        parent=top,
        critical=True,
    )
    # Build a union of all URLs present in extraction
    all_urls = combine_urls(
        ex.location_sources, ex.classification_sources,
        [ex.leapfrog_fall_2025_url] if ex.leapfrog_fall_2025_url else [],
        ex.leapfrog_sources, ex.sustained_a_sources,
        ex.emergency_dept_sources, ex.covid_vax_sources,
        ex.rsv_vax_sources, ex.pediatric_vax_sources,
        ex.medicare_sources, ex.telehealth_sources,
        ex.marketplace_sources, ex.reference_urls
    )
    evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id="reference_urls_provided",
        desc="At least one reference URL is provided in the answer",
        parent=refs,
        critical=True,
    )
    node_refs_official = evaluator.add_leaf(
        id="references_from_official_sources",
        desc="References include official sources (hospital website, hospitalsafetygrade.org, or Pennsylvania health authorities)",
        parent=refs,
        critical=True,
    )
    await evaluator.verify(
        claim="This page is an official source for the hospital information, such as a hospital-owned website, hospitalsafetygrade.org, or a Pennsylvania government/health authority domain.",
        node=node_refs_official,
        sources=all_urls,
        additional_instruction="Judge a page as 'official' if it is one of: (1) hospital's own domain (e.g., the hospital or health system site), (2) hospitalsafetygrade.org (Leapfrog), or (3) a Pennsylvania government/health authority domain (e.g., *.pa.gov or official PA health sites)."
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
    # Initialize evaluator (root is always non-critical; we add a critical child as the task root)
    evaluator = Evaluator()
    evaluator.initialize(
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
        prompt=prompt_extract_hospital(),
        template_class=HospitalExtraction,
        extraction_name="hospital_extraction",
    )

    # Build verification tree and run checks
    await _build_verification_tree(evaluator, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()