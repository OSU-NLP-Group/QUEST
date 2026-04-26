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
TASK_ID = "ohio_hospital_six_criteria"
TASK_DESCRIPTION = """
Identify a hospital in Ohio that meets all of the following six criteria:

1. The hospital must be located in the state of Ohio.
2. The hospital must hold a Level I trauma center designation, verified by either the Ohio Department of Health Emergency Medical Services or the American College of Surgeons Committee on Trauma (ACS-COT).
3. The hospital must be part of or house an NCI-designated comprehensive cancer center, as listed on the National Cancer Institute's official website.
4. The hospital must have received a 4-star or 5-star overall hospital quality rating from the Centers for Medicare & Medicaid Services (CMS) in their most recent annual rating update (2024 or 2025).
5. The hospital must be a teaching hospital with a formal affiliation to an accredited medical school for clinical education and residency training.
6. The hospital must have a licensed bed capacity of at least 400 staffed beds.

Provide the full official name of the hospital that satisfies all six requirements, along with supporting evidence for each criterion.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HospitalSelection(BaseModel):
    # Core identification
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    official_website: Optional[str] = None

    # Criterion 1: Ohio Location
    location_urls: List[str] = Field(default_factory=list)

    # Criterion 2: Level I Trauma Center (verified by ACS-COT or ODH EMS)
    trauma_level: Optional[str] = None  # e.g., "Level I", "Adult Level I", "Level 1 Adult & Pediatric"
    trauma_agency: Optional[str] = None  # e.g., "ACS-COT", "Ohio Department of Health EMS"
    trauma_urls: List[str] = Field(default_factory=list)

    # Criterion 3: NCI-designated comprehensive cancer center
    nci_center_name: Optional[str] = None
    nci_designation: Optional[str] = None  # e.g., "Comprehensive Cancer Center"
    nci_urls: List[str] = Field(default_factory=list)

    # Criterion 4: CMS overall hospital quality rating (4 or 5 stars; year 2024 or 2025)
    cms_star_rating: Optional[str] = None  # e.g., "4", "5", "5-star", "*****"
    cms_rating_year: Optional[str] = None  # e.g., "2024", "2025"
    cms_urls: List[str] = Field(default_factory=list)

    # Criterion 5: Teaching hospital with formal affiliation to accredited medical school
    teaching_status: Optional[str] = None  # e.g., "teaching hospital", "academic medical center"
    med_school_affiliation: Optional[str] = None  # e.g., "The Ohio State University College of Medicine"
    teaching_urls: List[str] = Field(default_factory=list)

    # Criterion 6: Licensed bed capacity >= 400 staffed beds
    bed_capacity: Optional[str] = None  # keep as string to be lenient (e.g., "1,000", "456 staffed beds")
    bed_capacity_type: Optional[str] = None  # e.g., "licensed beds", "staffed beds", "bed capacity"
    beds_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hospital_selection() -> str:
    return """
    From the provided answer text, extract the single primary hospital that the answer claims satisfies ALL SIX criteria. 
    If multiple hospitals are mentioned, choose the one the answer ultimately presents as the final choice; otherwise choose the first one mentioned as meeting all six requirements.

    Extract the following fields exactly from the answer:
    1) name: Full official hospital name.
    2) city: City of the hospital.
    3) state: State of the hospital (prefer two-letter 'OH' or 'Ohio').
    4) official_website: The hospital's official website URL, if explicitly provided in the answer.
    5) location_urls: All URLs provided that support the hospital being located in Ohio.
    6) trauma_level: The trauma level claimed in the answer (e.g., "Level I", "Adult Level I", "Level 1 Adult & Pediatric").
    7) trauma_agency: The named verifying body for trauma designation, if mentioned (e.g., "ACS-COT", "Ohio Department of Health EMS").
    8) trauma_urls: All URLs provided that support the Level I trauma center designation.
    9) nci_center_name: The name of the NCI-designated comprehensive cancer center the hospital houses/partners with, if stated.
    10) nci_designation: The designation type as stated (should include "Comprehensive" if claimed).
    11) nci_urls: All URLs provided that support the NCI comprehensive designation (must be from NCI official site to be valid, but still extract any URLs explicitly given).
    12) cms_star_rating: The overall CMS hospital quality star rating claimed (e.g., "4", "5", "5-star", "*****").
    13) cms_rating_year: The rating year claimed (should be "2024" or "2025" if stated).
    14) cms_urls: All URLs provided that support the CMS rating.
    15) teaching_status: The claimed teaching/academic status (e.g., "teaching hospital", "academic medical center").
    16) med_school_affiliation: The named medical school(s) to which the hospital has a formal affiliation, if provided.
    17) teaching_urls: All URLs provided that support the teaching hospital status and medical school affiliation.
    18) bed_capacity: The hospital's bed count as claimed (e.g., "456", "1,050", "450 staffed beds").
    19) bed_capacity_type: The phrase used (e.g., "licensed beds", "staffed beds", "bed capacity").
    20) beds_urls: All URLs provided that support the bed capacity claim.

    IMPORTANT EXTRACTION RULES:
    - Extract ONLY what is explicitly present in the answer text. Do not infer or invent values or URLs.
    - For all *_urls fields, include ONLY actual URLs that appear in the answer (plain URLs or markdown links). Do not create or guess URLs.
    - If a requested field is not present in the answer, set it to null; for URL lists, return an empty list.
    - Be robust to formatting variations; keep numbers and units exactly as written in the answer (do not normalize or round).

    Return a single JSON object matching the schema of the provided template class.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Criterion verification functions                                            #
# --------------------------------------------------------------------------- #
async def verify_location(evaluator: Evaluator, parent, data: HospitalSelection) -> None:
    node = evaluator.add_sequential(
        id="Ohio_Location",
        desc="The hospital must be located in the state of Ohio.",
        parent=parent,
        critical=True
    )

    urls = _non_empty_urls(data.location_urls)

    evaluator.add_custom_node(
        result=(bool(data.name) and len(urls) > 0),
        id="ohio_location_sources_present",
        desc="Location evidence sources are provided in the answer (and hospital name is present).",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="ohio_location_supported",
        desc=f"Confirm that '{data.name}' is located in Ohio (OH).",
        parent=node,
        critical=True
    )
    claim = f"The hospital named '{data.name}' is located in the state of Ohio (OH)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Treat this as supported only if the page clearly indicates that the specific hospital/campus is in Ohio. "
            "Accept indicators like 'Cleveland, OH', 'Columbus, Ohio', etc. "
            "Ensure the page is about the named hospital, not just a parent system spanning multiple states."
        )
    )


async def verify_trauma(evaluator: Evaluator, parent, data: HospitalSelection) -> None:
    node = evaluator.add_sequential(
        id="Level_I_Trauma_Center",
        desc="The hospital must hold a Level I trauma center designation, verified by ODH EMS or ACS-COT.",
        parent=parent,
        critical=True
    )

    urls = _non_empty_urls(data.trauma_urls)

    evaluator.add_custom_node(
        result=(bool(data.name) and len(urls) > 0),
        id="trauma_sources_present",
        desc="Trauma designation evidence sources are provided in the answer (and hospital name is present).",
        parent=node,
        critical=True
    )

    checks = evaluator.add_parallel(
        id="trauma_checks",
        desc="Trauma designation checks",
        parent=node,
        critical=True
    )

    # Check Level I designation itself
    leaf_level = evaluator.add_leaf(
        id="trauma_is_level_1",
        desc=f"Verify '{data.name}' is designated as a Level I trauma center (adult and/or pediatric).",
        parent=checks,
        critical=True
    )
    claim_level = (
        f"'{data.name}' is designated as a Level I trauma center (adult and/or pediatric). "
        f"If trauma level is provided in the answer, it is: '{data.trauma_level}'."
    )
    await evaluator.verify(
        claim=claim_level,
        node=leaf_level,
        sources=urls,
        additional_instruction=(
            "Confirm explicitly that the hospital is Level I (or 'Level 1') trauma center. "
            "Minor formatting variants like 'Adult Level I' or 'Level I Adult & Pediatric' count as Level I."
        )
    )

    # Check authoritative verification source (ACS-COT or ODH EMS)
    leaf_auth = evaluator.add_leaf(
        id="trauma_verified_by_authority",
        desc="Verification is from ACS-COT (facs.org) or Ohio Department of Health EMS official listing.",
        parent=checks,
        critical=True
    )
    claim_auth = (
        "This evidence comes from an official ACS Committee on Trauma (ACS-COT) directory/listing page or from the "
        "Ohio Department of Health EMS official trauma center listing, and it lists the hospital at Level I."
    )
    await evaluator.verify(
        claim=claim_auth,
        node=leaf_auth,
        sources=urls,
        additional_instruction=(
            "Only accept if the verifying page is clearly an official ACS-COT page (on facs.org) or an official Ohio "
            "Department of Health EMS page (on ohio.gov/ems or equivalent official ODH EMS domain). "
            "Third-party or hospital self-claims without ACS/ODH listing should be considered unsupported for this requirement."
        )
    )


async def verify_nci(evaluator: Evaluator, parent, data: HospitalSelection) -> None:
    node = evaluator.add_sequential(
        id="NCI_Comprehensive_Cancer_Center",
        desc="The hospital must be part of or house an NCI-designated comprehensive cancer center.",
        parent=parent,
        critical=True
    )

    urls = _non_empty_urls(data.nci_urls)

    evaluator.add_custom_node(
        result=(bool(data.name) and len(urls) > 0),
        id="nci_sources_present",
        desc="NCI designation evidence sources are provided in the answer (and hospital name is present).",
        parent=node,
        critical=True
    )

    checks = evaluator.add_parallel(
        id="nci_checks",
        desc="NCI designation checks",
        parent=node,
        critical=True
    )

    # Check that at least one URL is on official NCI domain
    leaf_official = evaluator.add_leaf(
        id="nci_source_is_official",
        desc="Evidence page is on the official NCI website (cancer.gov).",
        parent=checks,
        critical=True
    )
    claim_official = "This evidence page is hosted on the National Cancer Institute's official domain (cancer.gov)."
    await evaluator.verify(
        claim=claim_official,
        node=leaf_official,
        sources=urls,
        additional_instruction=(
            "Only consider pages on cancer.gov as official NCI pages. The URL shown to you should clearly be on cancer.gov."
        )
    )

    # Check that designation is comprehensive and that hospital is part of or houses it
    leaf_comp = evaluator.add_leaf(
        id="nci_is_comprehensive",
        desc=f"'{data.name}' is part of or houses an NCI-designated Comprehensive Cancer Center.",
        parent=checks,
        critical=True
    )
    claim_comp = (
        f"The NCI page shows that the hospital '{data.name}' either houses or is formally part of an "
        f"NCI-designated Comprehensive Cancer Center (not just 'Clinical' or 'Basic Laboratory'). "
        f"If a center name was provided: '{data.nci_center_name}'. "
        f"If a designation text was provided: '{data.nci_designation}'."
    )
    await evaluator.verify(
        claim=claim_comp,
        node=leaf_comp,
        sources=urls,
        additional_instruction=(
            "Look for the 'Comprehensive Cancer Center' designation on the NCI page. "
            "It must clearly indicate 'Comprehensive'. Affiliations/consortium structures that include the hospital "
            "are acceptable if the hospital is explicitly listed as part of the NCI Comprehensive Cancer Center."
        )
    )


async def verify_cms(evaluator: Evaluator, parent, data: HospitalSelection) -> None:
    node = evaluator.add_sequential(
        id="CMS_High_Quality_Rating",
        desc="The hospital must have a 4-star or 5-star overall CMS hospital quality rating (2024 or 2025).",
        parent=parent,
        critical=True
    )

    urls = _non_empty_urls(data.cms_urls)

    evaluator.add_custom_node(
        result=(bool(data.name) and len(urls) > 0),
        id="cms_sources_present",
        desc="CMS rating evidence sources are provided in the answer (and hospital name is present).",
        parent=node,
        critical=True
    )

    # First, ensure source is official CMS/Medicare
    leaf_official = evaluator.add_leaf(
        id="cms_source_is_official",
        desc="Evidence page is an official CMS/Medicare page (cms.gov, data.cms.gov, or medicare.gov Care Compare).",
        parent=node,
        critical=True
    )
    claim_official = (
        "This evidence page is an official source from CMS or Medicare (domains like cms.gov, data.cms.gov, or "
        "medicare.gov Care Compare)."
    )
    await evaluator.verify(
        claim=claim_official,
        node=leaf_official,
        sources=urls,
        additional_instruction=(
            "Only accept pages from official CMS/Medicare domains (cms.gov, data.cms.gov, medicare.gov). "
            "Organizational press releases or third-party aggregators are not acceptable for this criterion."
        )
    )

    # Then check stars and year in parallel (both must pass)
    checks = evaluator.add_parallel(
        id="cms_checks",
        desc="CMS star rating and year checks",
        parent=node,
        critical=True
    )

    leaf_stars = evaluator.add_leaf(
        id="cms_stars_4_or_5",
        desc=f"'{data.name}' has overall CMS hospital quality rating of 4 or 5 stars.",
        parent=checks,
        critical=True
    )
    claim_stars = (
        f"The official CMS/Medicare source shows that '{data.name}' has an overall hospital quality star rating of "
        f"either 4 or 5 stars (not the patient survey star rating). "
        f"If a star value was provided in the answer: '{data.cms_star_rating}'."
    )
    await evaluator.verify(
        claim=claim_stars,
        node=leaf_stars,
        sources=urls,
        additional_instruction=(
            "Confirm the 'Overall hospital quality star rating' is 4 or 5. "
            "Do not confuse with HCAHPS 'patient survey star rating'."
        )
    )

    leaf_year = evaluator.add_leaf(
        id="cms_year_2024_or_2025",
        desc="The CMS overall rating cited corresponds to year 2024 or 2025.",
        parent=checks,
        critical=True
    )
    claim_year = (
        "The CMS/Medicare page indicates that the overall hospital quality star rating shown is from the most recent "
        "annual update in 2024 or 2025."
    )
    await evaluator.verify(
        claim=claim_year,
        node=leaf_year,
        sources=urls,
        additional_instruction=(
            "Look for the dataset year, 'last updated' year, or explicit labeling indicating the rating year is 2024 or 2025. "
            "If the page clearly states a different year or no year, this should fail."
        )
    )


async def verify_teaching(evaluator: Evaluator, parent, data: HospitalSelection) -> None:
    node = evaluator.add_sequential(
        id="Teaching_Hospital_Affiliation",
        desc="The hospital must be a teaching hospital with formal affiliation to an accredited medical school.",
        parent=parent,
        critical=True
    )

    urls = _non_empty_urls(data.teaching_urls)

    evaluator.add_custom_node(
        result=(bool(data.name) and len(urls) > 0),
        id="teaching_sources_present",
        desc="Teaching/affiliation evidence sources are provided in the answer (and hospital name is present).",
        parent=node,
        critical=True
    )

    checks = evaluator.add_parallel(
        id="teaching_checks",
        desc="Teaching status and medical school affiliation checks",
        parent=node,
        critical=True
    )

    leaf_teaching = evaluator.add_leaf(
        id="is_teaching_hospital",
        desc=f"'{data.name}' is a teaching hospital / academic medical center.",
        parent=checks,
        critical=True
    )
    claim_teaching = (
        f"The evidence shows that '{data.name}' is a teaching hospital (or academic medical center) that participates "
        f"in clinical education and residency training."
    )
    await evaluator.verify(
        claim=claim_teaching,
        node=leaf_teaching,
        sources=urls,
        additional_instruction=(
            "Accept sources like the hospital's GME/education pages, university/medical school sites, ACGME program listings, "
            "AAMC pages, or similarly credible sources that clearly state teaching status."
        )
    )

    leaf_affil = evaluator.add_leaf(
        id="formal_affiliation_with_accredited_med_school",
        desc="The hospital has a formal affiliation with an accredited medical school.",
        parent=checks,
        critical=True
    )
    claim_affil = (
        "The evidence explicitly shows a formal affiliation between the hospital and a named medical school for clinical "
        "education and residency training, and the school is an accredited MD (LCME) or DO (COCA) medical school."
    )
    await evaluator.verify(
        claim=claim_affil,
        node=leaf_affil,
        sources=urls,
        additional_instruction=(
            "Look for explicit statements of affiliation with a medical school (e.g., 'teaching hospital for X College of Medicine'). "
            "Accept medical school accreditation implicitly if it is a well-known accredited US medical school; "
            "you do not need a separate accreditation page if the school's status is unambiguous."
        )
    )


async def verify_beds(evaluator: Evaluator, parent, data: HospitalSelection) -> None:
    node = evaluator.add_sequential(
        id="Bed_Capacity_Minimum",
        desc="The hospital must have a licensed bed capacity of at least 400 staffed beds.",
        parent=parent,
        critical=True
    )

    urls = _non_empty_urls(data.beds_urls)

    evaluator.add_custom_node(
        result=(bool(data.name) and len(urls) > 0),
        id="beds_sources_present",
        desc="Bed capacity evidence sources are provided in the answer (and hospital name is present).",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="beds_at_least_400",
        desc=f"'{data.name}' has at least 400 staffed or licensed beds.",
        parent=node,
        critical=True
    )
    claim = (
        f"The evidence shows that '{data.name}' has a licensed or staffed bed capacity of at least 400. "
        f"If a value was provided in the answer: '{data.bed_capacity}' ({data.bed_capacity_type})."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=(
            "Accept phrases like 'licensed beds', 'staffed beds', or 'bed capacity'. "
            "If the page clearly shows a number >= 400 for staffed/licensed beds, pass; otherwise fail."
        )
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
    Evaluate an answer for the Ohio hospital six-criteria task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: allow independent aggregation at top-level
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_hospital_selection(),
        template_class=HospitalSelection,
        extraction_name="hospital_selection"
    )

    # Build rubric root (critical) to mirror JSON semantics
    hospital_node = evaluator.add_parallel(
        id="Hospital_Identification",
        desc="Identify a hospital in Ohio that meets all six specified criteria for trauma care, cancer treatment, quality rating, academic affiliation, and facility size.",
        parent=root,
        critical=True
    )

    # Add six critical criteria under Hospital_Identification
    await verify_location(evaluator, hospital_node, extracted)
    await verify_trauma(evaluator, hospital_node, extracted)
    await verify_nci(evaluator, hospital_node, extracted)
    await verify_cms(evaluator, hospital_node, extracted)
    await verify_teaching(evaluator, hospital_node, extracted)
    await verify_beds(evaluator, hospital_node, extracted)

    return evaluator.get_summary()