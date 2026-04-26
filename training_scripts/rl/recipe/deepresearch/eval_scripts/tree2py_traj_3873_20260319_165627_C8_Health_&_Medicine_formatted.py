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
TASK_ID = "oh_pa_teaching_hospitals"
TASK_DESCRIPTION = """I am researching high-quality teaching hospitals in the Ohio-Pennsylvania region for a healthcare analysis project. Please identify three teaching hospitals located in either Ohio or Pennsylvania that meet all of the following criteria:

1. The hospital must be located in Ohio or Pennsylvania
2. The hospital must have an Overall Hospital Quality Star Rating of 4 or 5 stars on Medicare's Care Compare website
3. The hospital must be designated as a teaching hospital (affiliated with a medical school or listed on the CMS Teaching Hospital List)
4. The hospital must be nationally ranked or recognized in at least one medical specialty by U.S. News & World Report for 2025-2026
5. The hospital must be a medium or large facility with at least 100 beds
6. The hospital must have publicly available safety of care measures on Medicare Care Compare
7. The hospital must have HCAHPS (patient experience) survey data publicly reported on Medicare Care Compare

For each of the three hospitals, provide:
- Hospital name
- City and state
- Overall star rating from Medicare Care Compare
- Evidence of teaching hospital status (medical school affiliation or CMS designation)
- At least one medical specialty in which the hospital is nationally ranked by U.S. News & World Report 2025-2026
- Approximate bed count or size classification (small/medium/large)
- Direct link to the hospital's profile page on Medicare Care Compare (www.medicare.gov/care-compare)
- HCAHPS patient survey star rating or response information
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HospitalItem(BaseModel):
    # Basic identification
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Prefer the two-letter code ("OH" / "PA") or full name
    # Medicare Care Compare
    care_compare_url: Optional[str] = None
    star_rating: Optional[str] = None  # e.g., "4", "4/5", "4 out of 5", "Five-Star"
    safety_measures_note: Optional[str] = None  # Any mention in answer
    hcahps_note: Optional[str] = None  # Any mention in answer
    # Teaching status
    teaching_status_note: Optional[str] = None  # e.g., "Affiliated with X school", "CMS Teaching Hospital"
    teaching_sources: List[str] = Field(default_factory=list)  # URLs supporting teaching status
    # US News specialty recognition
    usnews_specialties: List[str] = Field(default_factory=list)  # At least one specialty named in answer
    usnews_url: Optional[str] = None  # Direct US News 2025-2026 hospital profile URL if cited
    # Size / beds
    beds: Optional[str] = None  # numeric or free text like "950", "Approx. 800", "Large"
    size_classification: Optional[str] = None  # small/medium/large if stated
    beds_sources: List[str] = Field(default_factory=list)  # URLs that support bed count/size
    # Extra sources that might support various claims
    extra_sources: List[str] = Field(default_factory=list)


class HospitalsExtraction(BaseModel):
    hospitals: List[HospitalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hospitals() -> str:
    return """
Extract up to five candidate teaching hospitals that the answer claims meet the specified criteria. For each hospital mentioned, return a JSON object with the following fields:

- name: The hospital name, exactly as written in the answer (string or null)
- city: The city (string or null)
- state: The state (string or null). Prefer the two-letter code (e.g., "OH", "PA") when available; otherwise use full name.
- care_compare_url: A direct URL to www.medicare.gov/care-compare hospital profile page if provided (string or null)
- star_rating: The Overall Hospital Quality Star Rating from Care Compare as mentioned (string or null; keep the original phrasing such as "4", "4/5", "4 out of 5", or "Five-Star")
- teaching_status_note: Any text in the answer that indicates teaching hospital status (string or null)
- teaching_sources: A list of URLs explicitly cited in the answer that support teaching status (e.g., CMS teaching list, medical school affiliation, US News page, hospital site) (array of strings, may be empty)
- usnews_specialties: A list of at least one specialty for which the hospital is nationally ranked or recognized per the answer for 2025-2026 (array of strings, may be empty)
- usnews_url: The specific U.S. News & World Report 2025–2026 hospital profile URL if cited (string or null)
- beds: The bed count as mentioned (keep as string; e.g., "100", "about 300", "licensed beds 250") (string or null)
- size_classification: small/medium/large as stated if present (string or null)
- beds_sources: A list of URLs explicitly cited in the answer that support the bed count or size (array of strings, may be empty)
- safety_measures_note: Any text in the answer that mentions 'Safety of care' measures presence on Care Compare (string or null)
- hcahps_note: Any text in the answer that mentions HCAHPS patient experience data (string or null)
- extra_sources: Any additional URLs cited for this hospital that may help verify claims (array of strings, may be empty)

IMPORTANT:
- Only extract URLs that explicitly appear in the answer (including in markdown links). Do not invent or infer URLs.
- Preserve the original strings; do not normalize numbers or names.
- If some fields are not present in the answer, set them to null (or empty arrays for lists).

Return a single JSON object with a top-level key "hospitals" which is an array of these hospital objects.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _prefix_for_index(idx: int) -> str:
    return f"H{idx + 1}"


def _hospital_node_desc(idx: int) -> str:
    return ["First hospital meeting all criteria",
            "Second hospital meeting all criteria",
            "Third hospital meeting all criteria"][idx]


def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def _is_oh_or_pa_text(state_text: Optional[str]) -> bool:
    if not state_text:
        return False
    st = state_text.strip().lower()
    return st in {"oh", "ohio", "pa", "pennsylvania"}


# --------------------------------------------------------------------------- #
# Per-hospital verification                                                   #
# --------------------------------------------------------------------------- #
async def verify_one_hospital(
    evaluator: Evaluator,
    parent_node,
    hospital: HospitalItem,
    idx: int,
):
    """
    Build the verification subtree for one hospital with strictly binary leaf checks per rubric.
    Child leaves are created in a robust order (Care Compare URL first) so that dependent checks
    have evidence available.
    """
    prefix = _prefix_for_index(idx)

    hospital_node = evaluator.add_parallel(
        id=f"Hospital_{idx + 1}",
        desc=_hospital_node_desc(idx),
        parent=parent_node,
        critical=False,  # Non-critical at the hospital level; all-or-nothing via critical children
    )

    # ---------- 1) Care Compare URL presence & accessibility (critical) ----------
    cc_url_node_desc = "Direct link to hospital's Medicare Care Compare profile is provided and accessible"
    if hospital.care_compare_url:
        leaf_cc = evaluator.add_leaf(
            id=f"{prefix}_Care_Compare_URL",
            desc=cc_url_node_desc,
            parent=hospital_node,
            critical=True,
        )
        if hospital.name:
            cc_claim = (
                f"This webpage is the Medicare Care Compare profile for the hospital named '{hospital.name}'. "
                f"The page is accessible, and it belongs to the Medicare Care Compare site."
            )
        else:
            cc_claim = (
                "This webpage is a valid and accessible Medicare Care Compare hospital profile page."
            )
        await evaluator.verify(
            claim=cc_claim,
            node=leaf_cc,
            sources=hospital.care_compare_url,
            additional_instruction=(
                "Verify that the URL loads and belongs to 'www.medicare.gov/care-compare', "
                "and that it clearly shows a hospital profile (not a nursing home or other facility). "
                "If the page is inaccessible or not a hospital profile on Medicare Care Compare, mark as incorrect."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{prefix}_Care_Compare_URL",
            desc=cc_url_node_desc,
            parent=hospital_node,
            critical=True,
        )

    # ---------- 2) Geographic location in OH/PA (critical) ----------
    geo_leaf = evaluator.add_leaf(
        id=f"{prefix}_Geographic_Location",
        desc="Hospital is located in Ohio or Pennsylvania",
        parent=hospital_node,
        critical=True,
    )
    if hospital.care_compare_url:
        # Prefer verifying from Care Compare page
        geo_claim = (
            "According to this Medicare Care Compare hospital profile, the hospital's address shows it is located "
            "in the state of Ohio (OH) or Pennsylvania (PA)."
        )
        await evaluator.verify(
            claim=geo_claim,
            node=geo_leaf,
            sources=hospital.care_compare_url,
            additional_instruction=(
                "Confirm the state on the page is Ohio (OH) or Pennsylvania (PA). "
                "Minor formatting differences (e.g., 'OH' vs 'Ohio') are acceptable. "
                "If the state is not OH/PA, mark as incorrect."
            ),
        )
    else:
        # No URL evidence available; fail to respect source-grounding requirement
        geo_leaf.status = "failed"
        geo_leaf.score = 0.0

    # ---------- 3) Star Rating 4 or 5 on Care Compare (critical) ----------
    star_leaf = evaluator.add_leaf(
        id=f"{prefix}_Star_Rating",
        desc="Hospital has Overall Hospital Quality Star Rating of 4 or 5 stars on Medicare Care Compare",
        parent=hospital_node,
        critical=True,
    )
    if hospital.care_compare_url:
        if hospital.star_rating:
            star_claim = (
                f"The Medicare Care Compare profile shows the 'Overall hospital quality star rating' for this hospital "
                f"is '{hospital.star_rating}', which is 4 or 5 stars."
            )
        else:
            star_claim = (
                "The Medicare Care Compare profile shows the 'Overall hospital quality star rating' is 4 or 5 stars."
            )
        await evaluator.verify(
            claim=star_claim,
            node=star_leaf,
            sources=hospital.care_compare_url,
            additional_instruction=(
                "Locate the 'Overall hospital quality star rating' on the page. "
                "Pass only if it is 4 or 5 stars. If it is 3 or fewer, or if not reported, mark as incorrect."
            ),
        )
    else:
        star_leaf.status = "failed"
        star_leaf.score = 0.0

    # ---------- 4) Teaching Hospital status (critical) ----------
    teach_leaf = evaluator.add_leaf(
        id=f"{prefix}_Teaching_Status",
        desc="Hospital is designated as a teaching hospital with verifiable medical school affiliation or CMS teaching hospital list designation",
        parent=hospital_node,
        critical=True,
    )
    teaching_sources = _dedup_urls(
        (hospital.teaching_sources or []) + ([hospital.usnews_url] if hospital.usnews_url else []) + (hospital.extra_sources or [])
    )
    if not teaching_sources and hospital.care_compare_url:
        # As a weak fallback, include Care Compare (may or may not show teaching status)
        teaching_sources = [hospital.care_compare_url]
    if teaching_sources:
        teach_claim = (
            "The provided source page(s) explicitly indicate that this hospital is a teaching hospital "
            "(e.g., affiliated with a medical school or listed on a CMS/official teaching hospital list)."
        )
        await evaluator.verify(
            claim=teach_claim,
            node=teach_leaf,
            sources=teaching_sources,
            additional_instruction=(
                "Accept evidence such as: 'Teaching hospital? Yes' on U.S. News; stated affiliation with a medical school "
                "(e.g., residents/fellowships, university affiliation) on official pages; or inclusion on an official "
                "CMS/AAMC teaching hospital list. Reject vague mentions without explicit confirmation."
            ),
        )
    else:
        teach_leaf.status = "failed"
        teach_leaf.score = 0.0

    # ---------- 5) US News 2025-2026 specialty recognition (critical) ----------
    spec_leaf = evaluator.add_leaf(
        id=f"{prefix}_Specialty_Recognition",
        desc="Hospital is nationally ranked or recognized in at least one medical specialty by U.S. News & World Report 2025-2026",
        parent=hospital_node,
        critical=True,
    )
    usnews_sources = _dedup_urls(([hospital.usnews_url] if hospital.usnews_url else []) + (hospital.extra_sources or []))
    if usnews_sources:
        if hospital.usnews_specialties:
            spec_list_str = "; ".join(hospital.usnews_specialties)
            spec_claim = (
                f"For the 2025–2026 U.S. News & World Report rankings, the hospital is nationally ranked or recognized "
                f"in at least one medical specialty (e.g., {spec_list_str})."
            )
        else:
            spec_claim = (
                "For the 2025–2026 U.S. News & World Report rankings, the hospital is nationally ranked or recognized "
                "in at least one medical specialty."
            )
        await evaluator.verify(
            claim=spec_claim,
            node=spec_leaf,
            sources=usnews_sources,
            additional_instruction=(
                "Use the U.S. News 2025–2026 hospital profile. Consider 'Nationally Ranked' in a specialty as sufficient. "
                "Also accept clear specialty-level 'High Performing' recognitions where U.S. News treats them as recognition "
                "in that specialty (not just procedure/condition). Do not accept only 2024–2025 results."
            ),
        )
    else:
        spec_leaf.status = "failed"
        spec_leaf.score = 0.0

    # ---------- 6) Hospital size: >= 100 beds (critical) ----------
    size_leaf = evaluator.add_leaf(
        id=f"{prefix}_Hospital_Size",
        desc="Hospital is classified as medium or large facility with at least 100 beds",
        parent=hospital_node,
        critical=True,
    )
    size_sources = _dedup_urls((hospital.beds_sources or []) + ([hospital.care_compare_url] if hospital.care_compare_url else []) + (hospital.extra_sources or []))
    if size_sources:
        if hospital.beds and any(ch.isdigit() for ch in hospital.beds):
            size_claim = (
                f"The hospital has at least 100 beds (reported/indicated bed count is '{hospital.beds}')."
            )
        elif hospital.size_classification:
            size_claim = (
                f"The hospital is a medium or large facility (size classification reported as '{hospital.size_classification}'), "
                f"which indicates at least 100 beds."
            )
        else:
            size_claim = (
                "The hospital has at least 100 beds (i.e., medium or large facility)."
            )
        await evaluator.verify(
            claim=size_claim,
            node=size_leaf,
            sources=size_sources,
            additional_instruction=(
                "Check for 'beds', 'licensed beds', 'staffed beds', 'bed capacity', or credible size classification. "
                "Pass only if the evidence supports >= 100 beds or clearly indicates medium/large facility meeting that threshold."
            ),
        )
    else:
        size_leaf.status = "failed"
        size_leaf.score = 0.0

    # ---------- 7) Safety of care measures present on Care Compare (critical) ----------
    safety_leaf = evaluator.add_leaf(
        id=f"{prefix}_Safety_Measures",
        desc="Hospital has publicly available safety of care measures on Medicare Care Compare",
        parent=hospital_node,
        critical=True,
    )
    if hospital.care_compare_url:
        safety_claim = (
            "The Medicare Care Compare hospital profile displays 'Safety of care' measures for this hospital."
        )
        await evaluator.verify(
            claim=safety_claim,
            node=safety_leaf,
            sources=hospital.care_compare_url,
            additional_instruction=(
                "Look for a 'Safety of care' section with measures or ratings. If the section is missing entirely or all "
                "metrics are not reported, mark as incorrect."
            ),
        )
    else:
        safety_leaf.status = "failed"
        safety_leaf.score = 0.0

    # ---------- 8) HCAHPS (patient experience) publicly reported (critical) ----------
    px_leaf = evaluator.add_leaf(
        id=f"{prefix}_Patient_Experience",
        desc="Hospital has HCAHPS patient experience survey data publicly reported on Medicare Care Compare",
        parent=hospital_node,
        critical=True,
    )
    if hospital.care_compare_url:
        px_claim = (
            "The Medicare Care Compare hospital profile displays HCAHPS (patient experience) survey results for this hospital."
        )
        await evaluator.verify(
            claim=px_claim,
            node=px_leaf,
            sources=hospital.care_compare_url,
            additional_instruction=(
                "Look for 'Patient experience' or 'HCAHPS' results or star ratings. If not reported at all, mark as incorrect."
            ),
        )
    else:
        px_leaf.status = "failed"
        px_leaf.score = 0.0


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Ohio/Pennsylvania teaching hospitals task using the Mind2Web2 framework.
    """
    # Initialize evaluator with parallel aggregation at root
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

    # Extract hospital candidates from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hospitals(),
        template_class=HospitalsExtraction,
        extraction_name="hospitals_extraction",
    )

    # Prepare exactly three hospitals (pad with empty placeholders if fewer)
    hospitals: List[HospitalItem] = list(extracted.hospitals[:3])
    while len(hospitals) < 3:
        hospitals.append(HospitalItem())

    # Build verification for each of the three hospitals
    for i in range(3):
        await verify_one_hospital(evaluator, root, hospitals[i], i)

    # Return the final evaluation summary
    return evaluator.get_summary()