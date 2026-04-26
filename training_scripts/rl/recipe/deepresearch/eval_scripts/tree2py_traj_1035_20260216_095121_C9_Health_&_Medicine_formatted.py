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
TASK_ID = "ny_hospitals_multi_designation"
TASK_DESCRIPTION = (
    "Identify three hospitals in New York State that simultaneously meet all of the following criteria:\n\n"
    "1. Level I Adult Trauma Center: The hospital must be designated as a Level I Adult Trauma Center by the "
    "New York State Department of Health.\n"
    "2. Comprehensive Stroke Center: The hospital must be designated as a Comprehensive Stroke Center by the "
    "New York State Department of Health.\n"
    "3. Level IV NICU (Regional Perinatal Center): The hospital must operate a Level IV Neonatal Intensive Care Unit, "
    "designated by New York State as a Regional Perinatal Center.\n"
    "4. Teaching Hospital with Residency Programs: The hospital must be a teaching hospital that offers "
    "ACGME-accredited medical residency training programs.\n"
    "5. CMS Quality Rating: The hospital must have a CMS Overall Hospital Quality Star Rating of 3 stars or higher.\n\n"
    "For each of the three hospitals, provide: official hospital name; complete address; one verification URL for "
    "each of the five criteria (Trauma, Stroke, NICU/RPC, Teaching/Residency, CMS Star Rating)."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HospitalItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    trauma_urls: List[str] = Field(default_factory=list)
    stroke_urls: List[str] = Field(default_factory=list)
    nicu_urls: List[str] = Field(default_factory=list)
    teaching_urls: List[str] = Field(default_factory=list)
    cms_urls: List[str] = Field(default_factory=list)


class HospitalsExtraction(BaseModel):
    hospitals: List[HospitalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hospitals() -> str:
    return """
    Extract up to three hospitals from the answer that are claimed to meet ALL of these New York State requirements:
    - Level I Adult Trauma Center
    - Comprehensive Stroke Center
    - Level IV NICU / Regional Perinatal Center
    - Teaching hospital with ACGME-accredited residency programs
    - CMS Overall Hospital Quality Star Rating 3+ (≥3 stars)

    For each hospital, return:
    - name: Official hospital name as a single string.
    - address: Complete street address, including street, city, state, and ZIP code as one line string. If presented across multiple lines, combine into one.
    - trauma_urls: Array of URL(s) in the answer that verify Level I Adult Trauma Center designation (prefer NYS Department of Health pages or explicit hospital statements referencing NYS DOH).
    - stroke_urls: Array of URL(s) that verify Comprehensive Stroke Center designation (prefer NYS DOH pages).
    - nicu_urls: Array of URL(s) that verify Level IV NICU or Regional Perinatal Center (NYS) designation.
    - teaching_urls: Array of URL(s) that verify teaching hospital status or list ACGME-accredited residency programs.
    - cms_urls: Array of URL(s) that show the hospital's CMS Overall Hospital Quality Star Rating (prefer Medicare Care Compare).

    IMPORTANT:
    - Only include URLs explicitly present in the answer (plain or markdown links). Do not invent URLs.
    - Normalize URLs to include protocol. If missing, prepend "http://".
    - If more than 3 hospitals are listed in the answer, include only the first three in answer order.
    - If a specific URL category is missing, return an empty array for that field; do not fabricate.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _is_nonempty_url_list(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0 and any(u and isinstance(u, str) for u in urls))


# --------------------------------------------------------------------------- #
# Verification for a single hospital                                          #
# --------------------------------------------------------------------------- #
async def verify_hospital(
    evaluator: Evaluator,
    parent_node,
    hospital: HospitalItem,
    idx: int
) -> None:
    """
    Build the verification sub-tree and run verifications for a single hospital.
    """
    i = idx + 1  # For 1-based labels
    hosp_node = evaluator.add_parallel(
        id=f"Hospital_{i}",
        desc=f"{['First', 'Second', 'Third'][idx]} hospital meeting all certification requirements",
        parent=parent_node,
        critical=False
    )

    # Basic info existence checks
    name_ok = _non_empty_str(hospital.name)
    addr_ok = _non_empty_str(hospital.address)

    evaluator.add_custom_node(
        result=name_ok,
        id=f"Hospital_Name_H{i}",
        desc="Official hospital name is provided",
        parent=hosp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=addr_ok,
        id=f"Complete_Address_H{i}",
        desc="Complete address including street address, city, state, and ZIP code is provided",
        parent=hosp_node,
        critical=True
    )

    # NY State location check (simple verification without external sources)
    ny_loc_leaf = evaluator.add_leaf(
        id=f"NY_State_Location_H{i}",
        desc="Hospital is located in New York State (verified through address)",
        parent=hosp_node,
        critical=True
    )
    address_text = hospital.address or ""
    ny_claim = f'The following address is located in New York State: "{address_text}".'
    await evaluator.verify(
        claim=ny_claim,
        node=ny_loc_leaf,
        additional_instruction="Determine whether the address string clearly indicates a location in New York State (e.g., contains 'NY' or 'New York' as the state)."
    )

    # --- Level I Adult Trauma Center cluster ---
    trauma_cluster = evaluator.add_parallel(
        id=f"Level_I_Trauma_Center_H{i}",
        desc="Hospital has Level I Adult Trauma Center designation from NYS Department of Health",
        parent=hosp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty_url_list(hospital.trauma_urls),
        id=f"Trauma_URL_Reference_H{i}",
        desc="URL reference provided that verifies Level I Adult Trauma Center designation",
        parent=trauma_cluster,
        critical=True
    )

    trauma_verify_leaf = evaluator.add_leaf(
        id=f"Trauma_Designation_Confirmed_H{i}",
        desc="Designation is specifically for Level I Adult Trauma Center (not Level II or Pediatric only)",
        parent=trauma_cluster,
        critical=True
    )
    trauma_claim = (
        f"{hospital.name or 'The hospital'} is designated as a Level I Adult Trauma Center by the New York State "
        f"Department of Health (or the provided source explicitly states Level I adult trauma center status in New York)."
    )
    await evaluator.verify(
        claim=trauma_claim,
        node=trauma_verify_leaf,
        sources=hospital.trauma_urls,
        additional_instruction=(
            "Confirm the page(s) state Level I Adult Trauma Center (accept minor variants like 'Level 1'). "
            "It must be for ADULT trauma (not pediatric only) and applicable to New York State. "
            "Accept NYS DOH official lists or explicit hospital pages that clearly reference the NYS designation."
        )
    )

    # --- Comprehensive Stroke Center cluster ---
    stroke_cluster = evaluator.add_parallel(
        id=f"Comprehensive_Stroke_Center_H{i}",
        desc="Hospital has Comprehensive Stroke Center designation from NYS Department of Health",
        parent=hosp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty_url_list(hospital.stroke_urls),
        id=f"Stroke_URL_Reference_H{i}",
        desc="URL reference provided that verifies Comprehensive Stroke Center designation",
        parent=stroke_cluster,
        critical=True
    )

    stroke_verify_leaf = evaluator.add_leaf(
        id=f"Stroke_Level_Confirmed_H{i}",
        desc="Designation is specifically for Comprehensive Stroke Center (not Primary or Thrombectomy-Capable only)",
        parent=stroke_cluster,
        critical=True
    )
    stroke_claim = (
        f"{hospital.name or 'The hospital'} is designated as a Comprehensive Stroke Center by the New York State "
        f"Department of Health (or the provided source explicitly states 'Comprehensive Stroke Center' in New York)."
    )
    await evaluator.verify(
        claim=stroke_claim,
        node=stroke_verify_leaf,
        sources=hospital.stroke_urls,
        additional_instruction=(
            "Confirm the page(s) state 'Comprehensive Stroke Center' (accept 'CSC'). "
            "Do NOT accept Primary or Thrombectomy-Capable only. Accept NYS DOH lists or explicit hospital pages that "
            "clearly reference NYS CSC designation."
        )
    )

    # --- Level IV NICU / Regional Perinatal Center cluster ---
    nicu_cluster = evaluator.add_parallel(
        id=f"Level_IV_NICU_H{i}",
        desc="Hospital operates a Level IV NICU designated as a Regional Perinatal Center by NYS",
        parent=hosp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty_url_list(hospital.nicu_urls),
        id=f"NICU_URL_Reference_H{i}",
        desc="URL reference provided that verifies Level IV NICU or Regional Perinatal Center designation",
        parent=nicu_cluster,
        critical=True
    )

    nicu_verify_leaf = evaluator.add_leaf(
        id=f"NICU_Level_Confirmed_H{i}",
        desc="Designation is specifically for Level IV NICU / Regional Perinatal Center (not Level II or III)",
        parent=nicu_cluster,
        critical=True
    )
    nicu_claim = (
        f"{hospital.name or 'The hospital'} operates a Level IV Neonatal Intensive Care Unit and is designated a "
        f"Regional Perinatal Center in New York State."
    )
    await evaluator.verify(
        claim=nicu_claim,
        node=nicu_verify_leaf,
        sources=hospital.nicu_urls,
        additional_instruction=(
            "Confirm the page(s) state 'Level IV NICU' and/or 'Regional Perinatal Center (RPC)' in New York State. "
            "Accept NYS DOH lists or explicit hospital pages that clearly reference the NYS Level IV/RPC designation."
        )
    )

    # --- Teaching hospital / residency programs cluster ---
    teaching_cluster = evaluator.add_parallel(
        id=f"Teaching_Hospital_Status_H{i}",
        desc="Hospital is a teaching hospital with ACGME-accredited residency programs",
        parent=hosp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty_url_list(hospital.teaching_urls),
        id=f"Teaching_URL_Reference_H{i}",
        desc="URL reference provided that verifies teaching hospital status or lists residency programs",
        parent=teaching_cluster,
        critical=True
    )

    teaching_verify_leaf = evaluator.add_leaf(
        id=f"Residency_Programs_Confirmed_H{i}",
        desc="Evidence of ACGME-accredited residency programs at the hospital",
        parent=teaching_cluster,
        critical=True
    )
    teaching_claim = (
        f"{hospital.name or 'The hospital'} is a teaching hospital that offers ACGME-accredited residency programs "
        f"(e.g., GME/Residency pages or ACGME listings)."
    )
    await evaluator.verify(
        claim=teaching_claim,
        node=teaching_verify_leaf,
        sources=hospital.teaching_urls,
        additional_instruction=(
            "Look for hospital Graduate Medical Education (GME) residency program pages that explicitly indicate "
            "ACGME accreditation, or ACGME/AAMC/CMS listings confirming residency programs. "
            "Evidence should clearly support ACGME-accredited residency presence."
        )
    )

    # --- CMS Overall Hospital Quality Star Rating cluster ---
    cms_cluster = evaluator.add_parallel(
        id=f"CMS_Star_Rating_H{i}",
        desc="Hospital has CMS Overall Hospital Quality Star Rating of 3 stars or higher",
        parent=hosp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_nonempty_url_list(hospital.cms_urls),
        id=f"CMS_URL_Reference_H{i}",
        desc="URL reference provided that shows the hospital's CMS star rating",
        parent=cms_cluster,
        critical=True
    )

    cms_verify_leaf = evaluator.add_leaf(
        id=f"Star_Rating_Meets_Threshold_H{i}",
        desc="CMS Overall Hospital Quality Star Rating is 3, 4, or 5 stars",
        parent=cms_cluster,
        critical=True
    )
    cms_claim = (
        f"{hospital.name or 'The hospital'} has a CMS Overall Hospital Quality Star Rating of at least 3 stars "
        f"(i.e., 3, 4, or 5 stars)."
    )
    await evaluator.verify(
        claim=cms_claim,
        node=cms_verify_leaf,
        sources=hospital.cms_urls,
        additional_instruction=(
            "Use Medicare Care Compare or other CMS-official pages if provided. "
            "Confirm that the overall hospital quality star rating shown is 3 or higher (3-5 stars). "
            "If multiple ratings appear, focus on the overall quality star rating."
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
    Evaluate an answer for the New York hospitals multi-designation task.
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
        default_model=model
    )

    # Record a note: Adjusted criticality to satisfy framework constraints
    evaluator.add_custom_info(
        info={
            "note": "The original rubric marks 'Task_Completion' as critical while hospital groups are non-critical. "
                    "The framework enforces that a critical parent must have all critical children. "
                    "We set Task_Completion as non-critical to preserve hospital-level partial credit while "
                    "keeping each requirement within a hospital critical."
        },
        info_type="design_decision",
        info_name="criticality_adjustment"
    )

    # Add a top-level task node (parallel aggregation across hospitals)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Successfully identify at least 3 hospitals in New York State that meet all specified certification requirements",
        parent=root,
        critical=False
    )

    # Extract hospitals from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hospitals(),
        template_class=HospitalsExtraction,
        extraction_name="hospitals_extraction"
    )

    # Prepare up to 3 hospitals (pad with empty items if fewer)
    hospitals = list(extracted.hospitals[:3])
    while len(hospitals) < 3:
        hospitals.append(HospitalItem())

    # Build verification subtrees for the three hospitals
    for idx in range(3):
        await verify_hospital(
            evaluator=evaluator,
            parent_node=task_node,
            hospital=hospitals[idx],
            idx=idx
        )

    # Return summary
    return evaluator.get_summary()