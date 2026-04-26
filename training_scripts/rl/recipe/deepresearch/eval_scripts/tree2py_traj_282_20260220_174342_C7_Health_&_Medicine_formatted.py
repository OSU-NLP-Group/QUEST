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
TASK_ID = "ohio_hospital_qual_cert_directory"
TASK_DESCRIPTION = """
I am developing a healthcare provider directory for complex medical cases in Ohio. Identify one hospital in Ohio that meets ALL of the following criteria:

1. Located in the state of Ohio
2. Accredited by The Joint Commission
3. Medicare-certified acute care hospital
4. CMS Overall Hospital Quality Star Rating of 4 or 5 stars
5. Provides 24/7 emergency department services
6. Teaching hospital with medical school affiliation
7. Certified as either a Primary Stroke Center, Thrombectomy-Capable Stroke Center, or Comprehensive Stroke Center
8. Designated trauma center (any level: I, II, III, or IV)
9. Offers cardiac surgery services
10. Offers neurosurgery services
11. Has inpatient capacity of at least 300 beds
12. Has a CMS patient experience (HCAHPS) summary star rating of at least 3 stars

For the identified hospital, provide:
- Official hospital name
- Complete physical address
- Direct URL to the hospital's listing on CMS Hospital Compare showing overall star rating
- Direct URL confirming Joint Commission accreditation status
- Description and URL reference for the hospital's stroke center certification
- Description and URL reference for the hospital's trauma center designation
- Evidence of teaching hospital status and medical school affiliation with URL
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StrokeCertification(BaseModel):
    certification_type: Optional[str] = None  # e.g., "Primary Stroke Center", "Comprehensive Stroke Center"
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TraumaDesignation(BaseModel):
    level: Optional[str] = None  # e.g., "Level I", "Level II", ...
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TeachingStatus(BaseModel):
    affiliation: Optional[str] = None  # e.g., affiliated medical school
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Services(BaseModel):
    emergency_24_7: Optional[bool] = None
    emergency_urls: List[str] = Field(default_factory=list)
    cardiac_surgery: Optional[bool] = None
    cardiac_urls: List[str] = Field(default_factory=list)
    neurosurgery: Optional[bool] = None
    neurosurgery_urls: List[str] = Field(default_factory=list)


class Capacity(BaseModel):
    inpatient_beds: Optional[str] = None  # keep as string; may be "300+", "over 1000", etc.
    urls: List[str] = Field(default_factory=list)


class CMSRatings(BaseModel):
    cms_compare_url: Optional[str] = None
    overall_star_rating: Optional[str] = None  # keep as string to allow "4", "4 stars", etc.
    hcahps_star_rating: Optional[str] = None


class Accreditation(BaseModel):
    joint_commission_url: Optional[str] = None
    medicare_certification_urls: List[str] = Field(default_factory=list)


class HospitalExtraction(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None

    accreditations: Optional[Accreditation] = None
    cms_ratings: Optional[CMSRatings] = None
    stroke: Optional[StrokeCertification] = None
    trauma: Optional[TraumaDesignation] = None
    teaching: Optional[TeachingStatus] = None
    services: Optional[Services] = None
    capacity: Optional[Capacity] = None

    other_supporting_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hospital() -> str:
    return """
Extract details for exactly one Ohio hospital described in the answer that aims to satisfy all listed criteria. Return all fields even if some are missing (use null or [] accordingly). Use the exact strings from the answer without adding new information.

Return a JSON object with these fields:

- name: Official hospital name as stated in the answer
- address: Complete physical address as stated in the answer (street, city, state, zip where available)
- city: City (if present)
- state: State abbreviation or name (e.g., "OH" or "Ohio") if present
- zip_code: ZIP code (if present)

- cms_ratings: {
    cms_compare_url: Direct URL to the hospital’s CMS Hospital Compare page showing overall rating (if provided)
    overall_star_rating: Overall quality star rating value mentioned in the answer, if stated (e.g., "4", "5", "4 stars")
    hcahps_star_rating: HCAHPS patient experience star rating mentioned in the answer, if stated
  }

- accreditations: {
    joint_commission_url: Direct URL confirming The Joint Commission accreditation (if provided)
    medicare_certification_urls: [List of URLs that the answer uses to support that the hospital is Medicare-certified acute care, if any. Include the CMS Compare URL if used for this.]
  }

- stroke: {
    certification_type: The stroke center level/type stated (e.g., "Primary Stroke Center", "Thrombectomy-Capable Stroke Center", "Comprehensive Stroke Center")
    description: The short description from the answer about the stroke certification
    urls: [All URLs cited to support the stroke certification]
  }

- trauma: {
    level: The trauma center level stated (e.g., "Level I", "Level II", "Level III", "Level IV") or generic "Trauma center"
    description: The short description from the answer about the trauma designation
    urls: [All URLs cited to support the trauma designation]
  }

- teaching: {
    affiliation: Medical school affiliation named in the answer (e.g., university name), if given
    description: The short description from the answer about teaching status
    urls: [All URLs cited to support teaching-hospital status or affiliation]
  }

- services: {
    emergency_24_7: true/false if the answer explicitly claims 24/7 emergency department services
    emergency_urls: [All URLs cited to support ED 24/7 claim; if none, return []]
    cardiac_surgery: true/false if the answer explicitly claims cardiac surgery services
    cardiac_urls: [All URLs cited to support cardiac surgery; if none, return []]
    neurosurgery: true/false if the answer explicitly claims neurosurgery services
    neurosurgery_urls: [All URLs cited to support neurosurgery; if none, return []]
  }

- capacity: {
    inpatient_beds: The bed count string stated (e.g., "300", "more than 300", "1,041") if provided
    urls: [All URLs cited to support inpatient bed count; if none, return []]
  }

- other_supporting_urls: [Any additional URLs the answer cites that may support the hospital meeting the criteria; exclude duplicates of the above]

Special rules:
- Only extract URLs that are explicitly present in the answer (plain or markdown links).
- Keep numbers as strings if ambiguous (e.g., "300+", "1,041").
- If a sub-object is not mentioned, return its fields as null or [] as appropriate.
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def valid_url(u: Optional[str]) -> bool:
    if not u:
        return False
    s = u.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def merge_urls(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            u = url.strip()
            if valid_url(u) and u not in seen:
                seen.add(u)
                out.append(u)
    return out


def stringify_or_empty(x: Optional[str]) -> str:
    return (x or "").strip()


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_hospital(evaluator: Evaluator, parent_node, ex: HospitalExtraction) -> None:
    """
    Build verification tree and run checks for a single hospital against all criteria.
    """

    main = evaluator.add_parallel(
        id="Hospital_Identification",
        desc="Identify one hospital in Ohio meeting all comprehensive quality and certification criteria",
        parent=parent_node,
        critical=False
    )

    # Pull frequently used fields safely
    name = stringify_or_empty(ex.name)
    address = stringify_or_empty(ex.address)
    cms_url = ex.cms_ratings.cms_compare_url if (ex.cms_ratings and ex.cms_ratings.cms_compare_url) else None
    jc_url = ex.accreditations.joint_commission_url if (ex.accreditations and ex.accreditations.joint_commission_url) else None

    # 0) Existence of key URLs (treat as separate critical leaves to gate dependent verifications)
    cms_url_node = evaluator.add_custom_node(
        result=valid_url(cms_url),
        id="CMS_Compare_URL",
        desc="Provide direct URL to hospital's CMS Hospital Compare page showing overall star rating",
        parent=main,
        critical=True
    )

    jc_url_node = evaluator.add_custom_node(
        result=valid_url(jc_url),
        id="Joint_Commission_URL",
        desc="Provide direct URL confirming Joint Commission accreditation status",
        parent=main,
        critical=True
    )

    # Stroke and trauma documentation existence
    stroke_urls = ex.stroke.urls if (ex.stroke and ex.stroke.urls) else []
    trauma_urls = ex.trauma.urls if (ex.trauma and ex.trauma.urls) else []
    teaching_urls = ex.teaching.urls if (ex.teaching and ex.teaching.urls) else []

    stroke_doc_node = evaluator.add_custom_node(
        result=len(stroke_urls) > 0,
        id="Stroke_Certification_Documentation",
        desc="Provide description and URL reference for stroke center certification",
        parent=main,
        critical=True
    )

    trauma_doc_node = evaluator.add_custom_node(
        result=len(trauma_urls) > 0,
        id="Trauma_Designation_Documentation",
        desc="Provide description and URL reference for trauma center designation",
        parent=main,
        critical=True
    )

    teaching_doc_node = evaluator.add_custom_node(
        result=len(teaching_urls) > 0,
        id="Teaching_Hospital_Documentation",
        desc="Provide evidence and URL for teaching hospital status and medical school affiliation",
        parent=main,
        critical=True
    )

    # 1) Hospital name verification
    name_node = evaluator.add_leaf(
        id="Hospital_Name",
        desc="Provide the official hospital name",
        parent=main,
        critical=True
    )
    name_claim = f"The hospital's official name is '{name}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=cms_url if valid_url(cms_url) else None,
        additional_instruction="Verify the hospital name exactly or with minor variations (punctuation/casing) as shown on the CMS Hospital Compare page."
    )

    # 2) Address verification
    address_node = evaluator.add_leaf(
        id="Physical_Address",
        desc="Provide the complete physical address of the hospital",
        parent=main,
        critical=True
    )
    addr_claim = f"The hospital's physical address is '{address}'."
    await evaluator.verify(
        claim=addr_claim,
        node=address_node,
        sources=cms_url if valid_url(cms_url) else None,
        additional_instruction="Confirm the street, city, state, and ZIP on the CMS Hospital Compare page; allow minor formatting/abbreviation differences."
    )

    # 3) Ohio location verification
    ohio_node = evaluator.add_leaf(
        id="Ohio_Location",
        desc="Hospital is physically located in the state of Ohio",
        parent=main,
        critical=True
    )
    ohio_claim = "This hospital is located in Ohio (OH)."
    await evaluator.verify(
        claim=ohio_claim,
        node=ohio_node,
        sources=cms_url if valid_url(cms_url) else None,
        additional_instruction="Check the location/state on the page; allow for 'OH' abbreviation or 'Ohio'."
    )

    # 4) Joint Commission accreditation
    jc_accr_node = evaluator.add_leaf(
        id="Joint_Commission_Accreditation",
        desc="Hospital is currently accredited by The Joint Commission",
        parent=main,
        critical=True
    )
    jc_claim = "The hospital is accredited by The Joint Commission."
    await evaluator.verify(
        claim=jc_claim,
        node=jc_accr_node,
        sources=jc_url if valid_url(jc_url) else None,
        additional_instruction="Verify accreditation status on an official Joint Commission page (e.g., Quality Check)."
    )

    # 5) Medicare-certified acute care hospital
    medicare_urls = []
    if ex.accreditations and ex.accreditations.medicare_certification_urls:
        medicare_urls.extend(ex.accreditations.medicare_certification_urls)
    if valid_url(cms_url):
        medicare_urls.append(cms_url)
    medicare_urls = merge_urls(medicare_urls)

    medicare_node = evaluator.add_leaf(
        id="Medicare_Certification",
        desc="Hospital is Medicare-certified as an acute care hospital",
        parent=main,
        critical=True
    )
    medicare_claim = "On CMS Hospital Compare, the hospital type is 'Acute Care Hospital' and participation indicates Medicare certification."
    await evaluator.verify(
        claim=medicare_claim,
        node=medicare_node,
        sources=medicare_urls if medicare_urls else None,
        additional_instruction="Accept if the CMS page lists 'Acute Care Hospital' as type. Presence on CMS Hospital Compare generally implies Medicare certification."
    )

    # 6) CMS Overall star rating (4 or 5 stars)
    overall_node = evaluator.add_leaf(
        id="CMS_Overall_Star_Rating",
        desc="Hospital has a CMS Overall Hospital Quality Star Rating of 4 or 5 stars",
        parent=main,
        critical=True
    )
    overall_claim = "On the CMS Hospital Compare page, the hospital's Overall Hospital Quality star rating is 4 or 5 stars."
    await evaluator.verify(
        claim=overall_claim,
        node=overall_node,
        sources=cms_url if valid_url(cms_url) else None,
        additional_instruction="Look for the 'Overall Rating' or 'Overall Hospital Quality Star Rating'; accept 4 or 5 as passing."
    )

    # 7) Emergency department 24/7
    emergency_sources = merge_urls(
        ex.services.emergency_urls if (ex.services and ex.services.emergency_urls) else [],
        [cms_url] if valid_url(cms_url) else []
    )
    emergency_node = evaluator.add_leaf(
        id="Emergency_Services",
        desc="Hospital provides 24/7 emergency department services",
        parent=main,
        critical=True
    )
    emergency_claim = "The hospital provides 24/7 emergency department services (open 24 hours a day)."
    await evaluator.verify(
        claim=emergency_claim,
        node=emergency_node,
        sources=emergency_sources if emergency_sources else None,
        additional_instruction="Accept statements like '24/7 emergency department', 'emergency department open 24 hours', or equivalent on official hospital/CMS pages."
    )

    # 8) Teaching hospital with medical school affiliation
    teaching_affil = stringify_or_empty(ex.teaching.affiliation if ex and ex.teaching else None)
    teaching_node = evaluator.add_leaf(
        id="Teaching_Hospital_Status",
        desc="Hospital is a teaching hospital with formal affiliation to a medical school",
        parent=main,
        critical=True
    )
    if teaching_affil:
        teaching_claim = f"The hospital is a teaching hospital affiliated with {teaching_affil}."
    else:
        teaching_claim = "The hospital is a teaching hospital with a formal affiliation to a medical school."
    await evaluator.verify(
        claim=teaching_claim,
        node=teaching_node,
        sources=teaching_urls if teaching_urls else None,
        additional_instruction="Accept evidence from official hospital or university/medical school pages or recognized bodies (e.g., AAMC/COTH)."
    )

    # 9) Stroke center certification
    stroke_type = stringify_or_empty(ex.stroke.certification_type if ex and ex.stroke else None)
    stroke_node = evaluator.add_leaf(
        id="Stroke_Center_Certification",
        desc="Hospital is certified as a Primary Stroke Center, Thrombectomy-Capable Stroke Center, or Comprehensive Stroke Center",
        parent=main,
        critical=True
    )
    if stroke_type:
        stroke_claim = f"The hospital is certified as a {stroke_type}."
    else:
        stroke_claim = "The hospital holds a recognized stroke center certification (Primary, Thrombectomy-Capable, or Comprehensive)."
    await evaluator.verify(
        claim=stroke_claim,
        node=stroke_node,
        sources=stroke_urls if stroke_urls else None,
        additional_instruction="Accept certifications by The Joint Commission (TJC), DNV, HFAP, or state-recognized programs; the page should clearly state the stroke center level/type."
    )

    # 10) Trauma center designation (any level)
    trauma_level = stringify_or_empty(ex.trauma.level if ex and ex.trauma else None)
    trauma_node = evaluator.add_leaf(
        id="Trauma_Center_Designation",
        desc="Hospital is designated as a trauma center at any level (I, II, III, or IV)",
        parent=main,
        critical=True
    )
    if trauma_level:
        trauma_claim = f"The hospital is designated as a {trauma_level} trauma center."
    else:
        trauma_claim = "The hospital is designated as a trauma center (any level)."
    await evaluator.verify(
        claim=trauma_claim,
        node=trauma_node,
        sources=trauma_urls if trauma_urls else None,
        additional_instruction="Accept verification from state health department (e.g., Ohio Department of Health), the American College of Surgeons, or official hospital pages."
    )

    # 11) Cardiac surgery services
    cardiac_sources = merge_urls(ex.services.cardiac_urls if (ex and ex.services and ex.services.cardiac_urls) else [])
    cardiac_node = evaluator.add_leaf(
        id="Cardiac_Surgery_Services",
        desc="Hospital offers cardiac surgery services",
        parent=main,
        critical=True
    )
    cardiac_claim = "The hospital offers cardiac surgery services (e.g., cardiothoracic or cardiac surgery program)."
    await evaluator.verify(
        claim=cardiac_claim,
        node=cardiac_node,
        sources=cardiac_sources if cardiac_sources else None,
        additional_instruction="Verify from the hospital's official service pages or authoritative clinical program pages indicating cardiac/cardiothoracic surgery."
    )

    # 12) Neurosurgery services
    neuro_sources = merge_urls(ex.services.neurosurgery_urls if (ex and ex.services and ex.services.neurosurgery_urls) else [])
    neuro_node = evaluator.add_leaf(
        id="Neurosurgery_Services",
        desc="Hospital offers neurosurgery services",
        parent=main,
        critical=True
    )
    neuro_claim = "The hospital offers neurosurgery services."
    await evaluator.verify(
        claim=neuro_claim,
        node=neuro_node,
        sources=neuro_sources if neuro_sources else None,
        additional_instruction="Verify from the hospital's official neurosurgery/neurosciences clinical program pages."
    )

    # 13) Inpatient bed capacity >= 300
    bed_sources = merge_urls(ex.capacity.urls if (ex and ex.capacity and ex.capacity.urls) else [], [cms_url] if valid_url(cms_url) else [])
    bed_node = evaluator.add_leaf(
        id="Inpatient_Bed_Capacity",
        desc="Hospital has at least 300 licensed inpatient beds",
        parent=main,
        critical=True
    )
    bed_claim = "The hospital has at least 300 inpatient beds (licensed or staffed)."
    await evaluator.verify(
        claim=bed_claim,
        node=bed_node,
        sources=bed_sources if bed_sources else None,
        additional_instruction="Accept hospital-reported bed counts on official pages, AHA/ACS listings, or CMS/HCRIS data if clearly shown."
    )

    # 14) HCAHPS patient experience >= 3 stars
    hcahps_node = evaluator.add_leaf(
        id="Patient_Experience_Rating",
        desc="Hospital has a CMS HCAHPS summary star rating of at least 3 stars",
        parent=main,
        critical=True
    )
    hcahps_claim = "On the CMS Hospital Compare page, the Patient experience (HCAHPS) summary star rating is at least 3 stars."
    await evaluator.verify(
        claim=hcahps_claim,
        node=hcahps_node,
        sources=cms_url if valid_url(cms_url) else None,
        additional_instruction="Look specifically for the HCAHPS Patient experience summary star rating; accept 3, 4, or 5."
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
    Evaluate an answer for the Ohio hospital quality/certification directory task.
    """
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

    # Extract the hospital details from the answer
    hospital_extraction: HospitalExtraction = await evaluator.extract(
        prompt=prompt_extract_hospital(),
        template_class=HospitalExtraction,
        extraction_name="hospital_extraction"
    )

    # Build and run verification against all criteria
    await verify_hospital(evaluator, root, hospital_extraction)

    # Return the evaluation summary
    return evaluator.get_summary()