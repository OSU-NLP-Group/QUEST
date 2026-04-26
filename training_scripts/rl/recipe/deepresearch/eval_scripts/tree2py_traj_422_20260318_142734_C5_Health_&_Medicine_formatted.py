import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ohio_hospitals_multi_criteria"
TASK_DESCRIPTION = """Identify three hospitals in Ohio that simultaneously meet all of the following criteria:

1. The hospital must be located in the state of Ohio.
2. The hospital must be designated as a Level I Adult Trauma Center, either state-designated by Ohio or verified by the American College of Surgeons (ACS).
3. The hospital must hold Comprehensive Stroke Center certification from the Joint Commission or an equivalent nationally recognized accrediting organization.
4. The hospital must be an Academic Medical Center, defined as a tertiary care hospital that is organizationally and administratively integrated with a medical school.
5. The hospital must have received EITHER a Leapfrog Hospital Safety Grade of 'A' in the most recent biannual grading period (Fall 2025 or later) OR a CMS Overall Hospital Quality Star Rating of 5 stars in the most recent annual rating update (2025 or later). At least one of these two quality ratings is required.

For each of the three hospitals, provide:
- The official name of the hospital
- The city where the hospital is located
- A direct URL to the hospital's official website
- A URL that verifies at least one of the hospital's key designations or quality ratings (such as from the Ohio Department of Health trauma center list, Joint Commission database, Leapfrog Hospital Safety Grade website, or CMS Care Compare website)"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HospitalItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    official_url: Optional[str] = None
    verification_urls: List[str] = Field(default_factory=list)


class HospitalsExtraction(BaseModel):
    hospitals: List[HospitalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hospitals() -> str:
    return """
    Extract up to three hospitals listed in the answer that the user claims meet the specified Ohio criteria.
    For each hospital, extract the following fields exactly as presented:
    - name: Official hospital name (string)
    - city: City where the hospital is located (string)
    - official_url: Direct URL to the hospital's official website (string URL). If multiple are listed, pick the main homepage.
    - verification_urls: An array of URLs that purportedly verify at least one of the key designations/ratings
      (e.g., Ohio Dept. of Health trauma list, ACS verification, Joint Commission or DNV for stroke, Leapfrog, or CMS Care Compare).
      Include all such URLs mentioned for that hospital, preserving order.

    Important rules:
    - Return a JSON object with a single key "hospitals" which is an array of hospital objects.
    - If more than three hospitals are present in the answer, include only the first three in order of appearance.
    - If fewer than three are present, include what is available; for missing hospitals do not fabricate data.
    - If a field is missing, set it to null (for strings) or an empty array (for verification_urls).
    - Only extract URLs that are explicitly present in the answer text (including markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    if not (u.startswith("http://") or u.startswith("https://")):
        return False
    return "." in u and " " not in u


def filter_valid_urls(urls: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for u in urls:
        if is_valid_url(u) and u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def ordinal(n: int) -> str:
    return ["First", "Second", "Third"][n] if 0 <= n <= 2 else f"#{n+1}"


# --------------------------------------------------------------------------- #
# Verification logic per hospital                                             #
# --------------------------------------------------------------------------- #
async def verify_one_hospital(
    evaluator: Evaluator,
    parent_node,
    hospital: HospitalItem,
    idx: int,
) -> None:
    # Create parent node for this hospital (parallel aggregation, non-critical to allow partial credit across hospitals)
    hosp_node = evaluator.add_parallel(
        id=f"hospital_{idx+1}",
        desc=f"{ordinal(idx)} hospital meeting all criteria",
        parent=parent_node,
        critical=False,
    )

    name = (hospital.name or "").strip()
    city = (hospital.city or "").strip()
    official = hospital.official_url or ""
    verif_urls = filter_valid_urls(hospital.verification_urls or [])
    sources_for_all = filter_valid_urls(([official] if is_valid_url(official) else []) + verif_urls)

    # Existence/format checks (critical)
    evaluator.add_custom_node(
        result=bool(name),
        id=f"hospital_{idx+1}_name_provided",
        desc="The official name of the hospital is provided",
        parent=hosp_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(city),
        id=f"hospital_{idx+1}_city_provided",
        desc="The city where the hospital is located is provided",
        parent=hosp_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=is_valid_url(official),
        id=f"hospital_{idx+1}_official_url_valid",
        desc="A valid URL to the hospital's official website is provided",
        parent=hosp_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(verif_urls) > 0,
        id=f"hospital_{idx+1}_verification_url_present",
        desc="A valid URL is provided that verifies at least one of the hospital's designations or quality ratings",
        parent=hosp_node,
        critical=True,
    )

    # Create verification leaves (all critical under this hospital)
    # 1) Located in Ohio
    located_leaf = evaluator.add_leaf(
        id=f"hospital_{idx+1}_located_in_ohio",
        desc="The hospital is located within the state of Ohio",
        parent=hosp_node,
        critical=True,
    )
    located_claim = (
        f"The hospital '{name}' is located in the state of Ohio."
        if name else "This hospital is located in the state of Ohio."
    )
    located_ins = (
        "Use the provided webpages to confirm the hospital is in Ohio (OH). "
        "Look for city/state/address or explicit mentions of Ohio. "
        "If the evidence is missing from all provided URLs, mark as NOT SUPPORTED."
    )
    located_sources = sources_for_all if sources_for_all else None

    # 2) Level I Adult Trauma Center
    trauma_leaf = evaluator.add_leaf(
        id=f"hospital_{idx+1}_level1_adult_trauma",
        desc="The hospital is designated as a Level I Adult Trauma Center (state-designated or ACS-verified)",
        parent=hosp_node,
        critical=True,
    )
    trauma_claim = (
        f"The hospital '{name}' is designated as a Level I adult trauma center, either Ohio state-designated or ACS-verified."
        if name else "This hospital is designated as a Level I adult trauma center, either Ohio state-designated or ACS-verified."
    )
    trauma_ins = (
        "Accept only adult Level I. If the source shows pediatric Level I without adult Level I, or adult Level II/III, then NOT SUPPORTED. "
        "Valid sources include Ohio Department of Health trauma center lists, ACS Verification Program listings, or the hospital site explicitly stating 'Adult Level I Trauma Center'. "
        "If no relevant evidence is present in the provided URLs, mark as NOT SUPPORTED."
    )
    trauma_sources = sources_for_all if sources_for_all else None

    # 3) Comprehensive Stroke Center
    stroke_leaf = evaluator.add_leaf(
        id=f"hospital_{idx+1}_comprehensive_stroke_center",
        desc="The hospital is certified as a Comprehensive Stroke Center by Joint Commission or equivalent",
        parent=hosp_node,
        critical=True,
    )
    stroke_claim = (
        f"The hospital '{name}' holds Comprehensive Stroke Center (CSC) certification from the Joint Commission or an equivalent nationally recognized accrediting organization (e.g., DNV)."
        if name else "This hospital holds Comprehensive Stroke Center (CSC) certification from the Joint Commission or an equivalent nationally recognized accrediting organization (e.g., DNV)."
    )
    stroke_ins = (
        "Accept 'Comprehensive Stroke Center' (CSC) certifications by the Joint Commission or an equivalent national accreditor such as DNV. "
        "Do NOT accept lower tiers like Primary/Thrombectomy/Acute Stroke Ready. "
        "If no evidence in provided URLs, NOT SUPPORTED."
    )
    stroke_sources = sources_for_all if sources_for_all else None

    # 4) Academic Medical Center status
    amc_leaf = evaluator.add_leaf(
        id=f"hospital_{idx+1}_academic_med_center",
        desc="The hospital is an Academic Medical Center integrated with a medical school",
        parent=hosp_node,
        critical=True,
    )
    amc_claim = (
        f"The hospital '{name}' is an academic medical center integrated with a medical school (organizationally and administratively)."
        if name else "This hospital is an academic medical center integrated with a medical school (organizationally and administratively)."
    )
    amc_ins = (
        "Accept if the hospital is explicitly described as an academic medical center or clearly integrated with a university medical school/college of medicine. "
        "Evidence may include being part of a university health system or explicit statements of integration/teaching with a medical school. "
        "If such integration is unclear or absent in the provided URLs, NOT SUPPORTED."
    )
    amc_sources = sources_for_all if sources_for_all else None

    # 5) Quality rating requirement (Leapfrog A Fall 2025+ OR CMS 5-star 2025+)
    quality_leaf = evaluator.add_leaf(
        id=f"hospital_{idx+1}_quality_requirement",
        desc="The hospital has received either Leapfrog Safety Grade A or CMS 5-star rating (at least one required)",
        parent=hosp_node,
        critical=True,
    )
    quality_claim = (
        f"The hospital '{name}' meets at least one of the following: "
        f"(a) Leapfrog Hospital Safety Grade of 'A' in Fall 2025 or any later term (e.g., Spring 2026), OR "
        f"(b) CMS Overall Hospital Quality Star Rating of 5 stars in 2025 or later."
        if name else
        "This hospital meets at least one of the following: (a) Leapfrog Hospital Safety Grade of 'A' in Fall 2025 or later, "
        "(b) CMS Overall Hospital Quality Star Rating of 5 stars in 2025 or later."
    )
    quality_ins = (
        "Evaluate ONLY the provided URLs. Mark as SUPPORTED if EITHER of the following is evidenced: "
        "• Leapfrog Hospital Safety Grade = 'A' with a term of Fall 2025 or later (e.g., Spring 2026, Fall 2026). "
        "• CMS Overall Hospital Quality Star Rating = 5 stars with rating year 2025 or later. "
        "If only earlier periods (pre-Fall 2025 for Leapfrog or pre-2025 for CMS) are shown, NOT SUPPORTED. "
        "If both sources exist, only one needs to meet the threshold."
    )
    quality_sources = sources_for_all if sources_for_all else None

    # Prepare verifications (parallel)
    verify_items = [
        (located_claim, located_sources, located_leaf, located_ins),
        (trauma_claim, trauma_sources, trauma_leaf, trauma_ins),
        (stroke_claim, stroke_sources, stroke_leaf, stroke_ins),
        (amc_claim, amc_sources, amc_leaf, amc_ins),
        (quality_claim, quality_sources, quality_leaf, quality_ins),
    ]

    # If no sources for a claim, steer the verifier to mark NOT SUPPORTED (policy)
    adjusted_items: List[tuple] = []
    for claim, srcs, node, add_ins in verify_items:
        if not srcs:
            adjusted_items.append((
                claim,
                None,
                node,
                add_ins + " No URLs were provided for this claim; per policy, treat as NOT SUPPORTED."
            ))
        else:
            adjusted_items.append((claim, srcs, node, add_ins))

    await evaluator.batch_verify(adjusted_items)


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
    # Initialize evaluator (root: parallel across 3 hospitals)
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

    # Create top-level task node mirroring rubric
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Evaluate whether 3 hospitals in Ohio meeting all specified criteria have been correctly identified",
        parent=root,
        critical=False,
    )

    # Extract structured hospitals from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hospitals(),
        template_class=HospitalsExtraction,
        extraction_name="hospitals_extracted",
    )

    # Normalize to exactly 3 slots
    hospitals = list(extracted.hospitals[:3])
    while len(hospitals) < 3:
        hospitals.append(HospitalItem())

    # Build child nodes for Hospital_1..3 (to align with rubric names)
    hosp_nodes: List[Any] = []
    for i in range(3):
        # Create container nodes to mirror rubric labels; actual checks are attached within verify_one_hospital
        label = f"Hospital_{i+1}"
        node = evaluator.add_parallel(
            id=label,
            desc=f"{ordinal(i)} hospital meeting all criteria",
            parent=task_node,
            critical=False,
        )
        hosp_nodes.append(node)

    # Verify each hospital under its label node
    # For alignment with rubric hierarchy, we pass the labeled node as the parent
    tasks = []
    for i in range(3):
        tasks.append(verify_one_hospital(evaluator, hosp_nodes[i], hospitals[i], i))
    await asyncio.gather(*tasks)

    # Return full summary
    return evaluator.get_summary()